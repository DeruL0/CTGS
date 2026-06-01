from __future__ import annotations

import math

import torch

def _freeze_ct_feature_params(gaussians):
    if gaussians._features_dc.numel() > 0:
        gaussians._features_dc.requires_grad_(False)
    if gaussians._features_rest.numel() > 0:
        gaussians._features_rest.requires_grad_(False)
    if gaussians.optimizer is not None:
        for group in gaussians.optimizer.param_groups:
            if group["name"] in {"f_dc", "f_rest"}:
                group["lr"] = 0.0
                if group["params"]:
                    group["params"][0].requires_grad_(False)


def _sanitize_xyz_parameter(gaussians):
    xyz = gaussians.get_xyz
    if xyz.numel() == 0:
        return 0

    finite_mask = torch.isfinite(xyz).all(dim=1)
    if torch.all(finite_mask):
        return 0

    bad_count = int((~finite_mask).sum().item())
    with torch.no_grad():
        replacement = xyz[finite_mask].mean(dim=0, keepdim=True) if torch.any(finite_mask) else torch.zeros((1, 3), dtype=xyz.dtype, device=xyz.device)
        xyz[~finite_mask] = replacement
    return bad_count


def _attenuation_only_training_enabled(args) -> bool:
    return bool(getattr(args, "ct_train_bulk_atten_only", False))


def _attenuation_only_bulk_gate_training_enabled(args) -> bool:
    return _attenuation_only_training_enabled(args) and bool(getattr(args, "ct_atten_only_train_bulk_opacity", True))


def _attenuation_only_preview_early_stop_enabled(args) -> bool:
    return _attenuation_only_training_enabled(args) and bool(getattr(args, "ct_atten_only_early_stop", True))


def _bulk_atten_only_lr_scale(args, iteration: int | None, max_iter: int | None) -> float:
    if not _attenuation_only_training_enabled(args):
        return 1.0
    final_scale = min(max(float(getattr(args, "ct_atten_only_lr_final_scale", 1.0)), 1e-8), 1.0)
    if iteration is None or max_iter is None or int(max_iter) <= 1:
        return 1.0
    progress = min(max((float(int(iteration)) - 1.0) / float(int(max_iter) - 1), 0.0), 1.0)
    return float(math.exp(math.log(final_scale) * progress))


def _apply_bulk_atten_only_optimizer_mode(gaussians, args, *, iteration: int | None = None, max_iter: int | None = None) -> None:
    if not _attenuation_only_training_enabled(args) or gaussians.optimizer is None:
        return
    original_lrs = getattr(args, "_ct_bulk_atten_only_original_lrs", None)
    if not isinstance(original_lrs, dict):
        original_lrs = {str(group.get("name", "")): float(group.get("lr", 0.0)) for group in gaussians.optimizer.param_groups}
        setattr(args, "_ct_bulk_atten_only_original_lrs", original_lrs)
    atten_scale = _bulk_atten_only_lr_scale(args, iteration, max_iter)
    for group in gaussians.optimizer.param_groups:
        name = str(group.get("name", ""))
        param = group["params"][0]
        if name == "attenuation":
            base_lr = float(original_lrs.get(name, group.get("lr", 0.0)))
            group["lr"] = float(base_lr * atten_scale)
            param.requires_grad_(True)
        elif name == "opacity" and _attenuation_only_bulk_gate_training_enabled(args):
            base_lr = float(original_lrs.get("attenuation", original_lrs.get(name, group.get("lr", 0.0))))
            group["lr"] = float(base_lr * atten_scale)
            param.requires_grad_(True)
        else:
            group["lr"] = 0.0
            param.requires_grad_(False)


def _restore_best_bulk_attenuation(
    gaussians,
    atten_logit: torch.Tensor | None,
    opacity_logit: torch.Tensor | None = None,
) -> bool:
    if atten_logit is None or gaussians.optimizer is None:
        return False
    gaussians._assign_parameter("_atten_logit", atten_logit.detach(), optimizer_name="attenuation", requires_grad=True)
    if opacity_logit is not None:
        gaussians._assign_parameter("_opacity", opacity_logit.detach(), optimizer_name="opacity", requires_grad=True)
    return True


def _bulk_attenuation_grad_stats(gaussians) -> dict[str, float]:
    grad = getattr(getattr(gaussians, "_atten_logit", None), "grad", None)
    region_type = getattr(gaussians, "get_region_type", None)
    if grad is None or region_type is None:
        return {}
    stats = {}
    bulk_mask = gaussians.get_region_type.reshape(-1) == 1
    if torch.any(bulk_mask):
        grad_values = grad.reshape(-1)[bulk_mask]
        if grad_values.numel() > 0:
            abs_grad = grad_values.detach().abs().to(dtype=torch.float32)
            stats.update(
                {
                    "atten_grad_mean": float(abs_grad.mean().item()),
                    "atten_grad_p50": float(torch.quantile(abs_grad, 0.50).item()),
                    "atten_grad_p90": float(torch.quantile(abs_grad, 0.90).item()),
                }
            )
    surface_mask = gaussians.get_region_type.reshape(-1) == 0
    if torch.any(surface_mask):
        grad_values = grad.reshape(-1)[surface_mask]
        if grad_values.numel() > 0:
            abs_grad = grad_values.detach().abs().to(dtype=torch.float32)
            stats.update(
                {
                    "surface_atten_grad_mean": float(abs_grad.mean().item()),
                    "surface_atten_grad_p90": float(torch.quantile(abs_grad, 0.90).item()),
                }
            )
    return stats


def maybe_apply_stage1_freeze(gaussians, args, iteration: int) -> bool:
    """Stage 1 (v5.2.1): freeze sigma/mu/rotation by zeroing their optimizer LRs.

    The atten_logit / features groups stay live so the coverage-first basis fits
    intensity without prematurely moving / shrinking. xyz scheduler value is
    overridden when in stage1 and left alone afterward (so the existing decay
    plan resumes naturally once stage1 ends).
    """
    stage1_until = int(getattr(args, "ct_stage1_freeze_until_iter", 0))
    if stage1_until <= 0 or gaussians.optimizer is None:
        return False
    if not hasattr(args, "_stage1_original_lrs"):
        cache = {}
        for group in gaussians.optimizer.param_groups:
            if group["name"] in {"scaling", "rotation"}:
                cache[group["name"]] = float(group["lr"])
        args._stage1_original_lrs = cache
    in_stage1 = int(iteration) <= int(stage1_until)
    for group in gaussians.optimizer.param_groups:
        name = group["name"]
        if name in {"scaling", "rotation"}:
            original_lr = float(args._stage1_original_lrs.get(name, group["lr"]))
            group["lr"] = 0.0 if in_stage1 else original_lr
        elif name == "xyz" and in_stage1:
            group["lr"] = 0.0
    return in_stage1
