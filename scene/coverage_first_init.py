"""Coverage-first bulk initialization (v5.2.1 Stage 0).

Reverses v5.2's "underfill -> reseed补洞" strategy to "材料域过覆盖 -> 收缩压实".

Invariants enforced here:
  1. sigma <= min(curvature_cap, c * |D(mu)|, sigma_global_max)
  2. boundary bulk centers nudged to D ~ -ratio * sigma  (not D=0)
  3. attenuation initialized coverage-normalized via KNN overlap estimate

Apply after training_setup() so the optimizer is ready for replace_tensor_to_optimizer.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from scene.gaussian_model_optimizer import replace_tensor_to_optimizer
from ct_pipeline.training.losses import sample_sdf_normals, sample_volume_field


def _to_5d(volume_t: torch.Tensor) -> torch.Tensor:
    if volume_t.ndim == 5:
        return volume_t
    if volume_t.ndim == 3:
        return volume_t.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"unexpected volume rank: {volume_t.ndim}")


def _sample_field_at(volume_5d: torch.Tensor, points_xyz: torch.Tensor, spacing_zyx) -> torch.Tensor:
    return sample_volume_field(volume_5d, points_xyz, spacing_zyx).reshape(-1).to(dtype=torch.float32)


def _sample_normals_at(sdf_volume_5d: torch.Tensor, points_xyz: torch.Tensor, spacing_zyx) -> torch.Tensor:
    n = sample_sdf_normals(sdf_volume_5d, points_xyz, spacing_zyx).to(dtype=torch.float32)
    return F.normalize(n, dim=-1, eps=1e-8)


def _knn_overlap_factor(xyz_np: np.ndarray, sigma_np: np.ndarray, k: int) -> np.ndarray:
    """E[K(mu_i)] approx 1 + sum_j G_j(mu_i) for j in KNN(i)."""
    if xyz_np.shape[0] == 0 or k <= 0:
        return np.ones((xyz_np.shape[0],), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return np.ones((xyz_np.shape[0],), dtype=np.float32)
    n = int(xyz_np.shape[0])
    k_eff = min(int(k), max(n - 1, 0))
    if k_eff <= 0:
        return np.ones((n,), dtype=np.float32)
    tree = cKDTree(xyz_np)
    dists, idxs = tree.query(xyz_np, k=k_eff + 1)
    dists = np.asarray(dists, dtype=np.float64)[:, 1:]
    idxs = np.asarray(idxs, dtype=np.int64)[:, 1:]
    nbr_sigma = sigma_np[idxs]
    G = np.exp(-0.5 * (dists ** 2) / np.maximum(nbr_sigma ** 2, 1e-8))
    return (1.0 + G.sum(axis=1)).astype(np.float32)


def _coverage_at_points(
    points_xyz: torch.Tensor,
    bulk_xyz: torch.Tensor,
    bulk_sigma: torch.Tensor,
    bulk_sdf: torch.Tensor,
    tau: float,
    chunk: int = 4096,
) -> torch.Tensor:
    """C(x) = sum_i G_i(x) * g_i(x) with linearized half-space gate.

    Coverage-only diagnostic (does not multiply by a_i). Computed in chunks to
    stay within memory.
    """
    if points_xyz.numel() == 0 or bulk_xyz.numel() == 0:
        return torch.zeros((int(points_xyz.shape[0]),), dtype=torch.float32, device=points_xyz.device)
    device = bulk_xyz.device
    p = points_xyz.to(device=device, dtype=torch.float32)
    out = torch.zeros((int(p.shape[0]),), dtype=torch.float32, device=device)
    sigma_sq = bulk_sigma.to(dtype=torch.float32).clamp_min(1e-6).square()
    for start in range(0, int(p.shape[0]), int(chunk)):
        end = min(start + int(chunk), int(p.shape[0]))
        q = p[start:end]
        diff = q[:, None, :] - bulk_xyz[None, :, :]
        d2 = (diff * diff).sum(dim=-1)
        G = torch.exp(-0.5 * d2 / sigma_sq[None, :])
        # linearized SDF along outward normal: l(x) = D(mu) + n . (x - mu) ; approximate gate via
        # bulk-center SDF only (good enough for coverage diagnostic at init time).
        # We use D(mu_i) since normals are not strictly needed for the coverage proxy.
        gate = torch.sigmoid(-bulk_sdf[None, :].to(dtype=torch.float32) / max(float(tau), 1e-6))
        K = G * gate
        out[start:end] = K.sum(dim=1)
    return out


def _quantile_safe(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.to(dtype=torch.float32), float(q)).item())


def _write_init_report(model_path: str, payload: dict) -> None:
    if not model_path:
        return
    try:
        os.makedirs(model_path, exist_ok=True)
    except OSError:
        return
    path = os.path.join(model_path, "coverage_init_report.txt")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            for key, value in payload.items():
                if isinstance(value, float):
                    fh.write(f"{key} = {value:.6f}\n")
                else:
                    fh.write(f"{key} = {value}\n")
    except OSError:
        pass


def apply_coverage_first_init(
    gaussians,
    volume_cuda: torch.Tensor,
    spacing_zyx,
    signed_distance_field: dict,
    args,
    *,
    model_path: str | None = None,
) -> dict:
    """v5.2.1 coverage-first bulk re-initialization (in-place via optimizer).

    Strategy controlled by args.ct_init_strategy:
      - "volume_sampled": no-op (legacy init already applied).
      - "coverage_first": partition bulk by D, set sigma <= min(c|D|, cap),
        nudge centers inward to D ~ -ratio*sigma, set a_i coverage-normalized.

    Ablation knobs:
      ct_init_coverage_normalized_atten  -> invariant 3 toggle
      ct_init_boundary_inward_nudge      -> invariant 2 toggle
      ct_init_boundary_sigma_c           -> invariant 1: sigma <= c * |D|

    Returns a dict of coverage stats (also written to init_report.txt).
    """
    strategy = str(getattr(args, "ct_init_strategy", "volume_sampled"))
    if strategy != "coverage_first":
        return {"strategy": strategy, "applied": False}

    region_type = gaussians._region_type.reshape(-1) if gaussians._region_type.numel() > 0 else None
    if region_type is None:
        return {"strategy": strategy, "applied": False, "reason": "no_region_type"}
    bulk_mask = (region_type == 1)
    n_bulk = int(bulk_mask.sum().item())
    if n_bulk == 0:
        return {"strategy": strategy, "applied": False, "reason": "no_bulk_gaussians"}

    device = gaussians._xyz.device
    bulk_indices = torch.nonzero(bulk_mask, as_tuple=False).reshape(-1).to(device=device)
    bulk_xyz = gaussians._xyz.detach()[bulk_indices].to(dtype=torch.float32)

    sdf_volume_5d = signed_distance_field["signed_distance"]  # already (1,1,D,H,W)
    if sdf_volume_5d.ndim != 5:
        sdf_volume_5d = _to_5d(sdf_volume_5d)
    sdf_spacing = signed_distance_field.get("spacing_zyx", spacing_zyx)
    sdf_at_centers = _sample_field_at(sdf_volume_5d, bulk_xyz, sdf_spacing)
    normals_at_centers = _sample_normals_at(sdf_volume_5d, bulk_xyz, sdf_spacing)

    voxel = float(min(float(v) for v in spacing_zyx))
    deep_depth = float(getattr(args, "ct_init_deep_depth_voxel", 2.0)) * voxel
    deep_sigma = float(getattr(args, "ct_init_deep_sigma_voxel", 1.5)) * voxel
    sigma_floor = float(getattr(args, "ct_bulk_scale_floor", 0.05))
    sigma_global_max = float(getattr(args, "ct_bulk_scale_global_max", 1.5))
    c_factor = float(getattr(args, "ct_init_boundary_sigma_c", 0.7))
    nudge_ratio = float(getattr(args, "ct_init_inward_nudge_sigma_ratio", 0.7))
    do_nudge = bool(getattr(args, "ct_init_boundary_inward_nudge", True))
    do_norm = bool(getattr(args, "ct_init_coverage_normalized_atten", True))

    # --- Sigma target per bulk gaussian ---
    abs_d = sdf_at_centers.abs()
    is_deep = sdf_at_centers <= -deep_depth
    # boundary cap: c * |D|, with at least the sigma_floor
    boundary_sigma = (c_factor * abs_d).clamp_min(sigma_floor)
    sigma_target = torch.where(is_deep, torch.full_like(sdf_at_centers, deep_sigma), boundary_sigma)
    sigma_target = sigma_target.clamp(min=sigma_floor, max=sigma_global_max)
    # Air-side (D > 0): defensively shrink to floor and skip nudge (rare for phase1 bulk).
    sigma_target = torch.where(sdf_at_centers > 0.0, torch.full_like(sigma_target, sigma_floor), sigma_target)

    # --- Inward nudge: only for centers shallower than -ratio*sigma_target ---
    new_xyz_bulk = bulk_xyz.clone()
    if do_nudge:
        target_depth = -nudge_ratio * sigma_target
        # only move centers that are currently too shallow (sdf > target_depth) AND inside material region
        # NOTE: we still nudge centers with sdf > 0 to bring them across the boundary
        need_nudge = sdf_at_centers > target_depth
        if torch.any(need_nudge):
            # move along outward normal toward inside, i.e. subtract (sdf - target_depth) * normal
            move = (sdf_at_centers - target_depth) * 1.0
            delta = move[:, None] * normals_at_centers
            # Outward normal is gradient of D, points OUT of material. Inside has D<0.
            # To go deeper inside (more negative D), step along -normal.
            candidate_xyz = bulk_xyz - delta
            new_xyz_bulk = torch.where(need_nudge[:, None], candidate_xyz, bulk_xyz)
        # re-sample sdf at the new centers
        sdf_at_centers = _sample_field_at(sdf_volume_5d, new_xyz_bulk, sdf_spacing)

    # --- Sample intensity at (possibly nudged) centers ---
    volume_5d = _to_5d(volume_cuda)
    intensity = _sample_field_at(volume_5d, new_xyz_bulk, spacing_zyx).clamp(1e-4, 1.0 - 1e-4)

    # --- Attenuation init ---
    if do_norm:
        k_nn = int(getattr(args, "ct_init_coverage_knn_k", 16))
        E_K = torch.from_numpy(
            _knn_overlap_factor(new_xyz_bulk.detach().cpu().numpy(), sigma_target.detach().cpu().numpy(), k_nn)
        ).to(device=device, dtype=torch.float32)
        a_target = (intensity / E_K.clamp_min(1e-3)).clamp(1e-4, 4.0)
    else:
        a_target = intensity.clamp(1e-4, 1.0)
    # softplus is monotonic; inverse on a positive value
    atten_logit_bulk = gaussians._inverse_softplus(a_target.reshape(-1, 1))

    # --- Build full-tensor replacements: bulk slice updated, surface slice untouched ---
    current_scaling = gaussians._scaling.detach().to(dtype=torch.float32)
    new_scaling = current_scaling.clone()
    new_scaling[bulk_indices] = torch.log(sigma_target.clamp_min(1e-6))[:, None].expand(-1, 3)

    current_xyz = gaussians._xyz.detach().to(dtype=torch.float32)
    new_xyz_full = current_xyz.clone()
    if do_nudge:
        new_xyz_full[bulk_indices] = new_xyz_bulk

    current_atten = gaussians._atten_logit.detach().to(dtype=torch.float32)
    new_atten_full = current_atten.clone()
    new_atten_full[bulk_indices] = atten_logit_bulk.to(dtype=current_atten.dtype)

    # --- Swap parameters through the optimizer to keep state consistent ---
    swapped = replace_tensor_to_optimizer(gaussians, new_scaling.to(dtype=gaussians._scaling.dtype), "scaling")
    if "scaling" in swapped:
        gaussians._scaling = swapped["scaling"]
    if do_nudge:
        swapped = replace_tensor_to_optimizer(gaussians, new_xyz_full.to(dtype=gaussians._xyz.dtype), "xyz")
        if "xyz" in swapped:
            gaussians._xyz = swapped["xyz"]
    swapped = replace_tensor_to_optimizer(gaussians, new_atten_full.to(dtype=gaussians._atten_logit.dtype), "attenuation")
    if "attenuation" in swapped:
        gaussians._atten_logit = swapped["attenuation"]

    # --- Coverage diagnostics: sample points in material/shell/void, compute C(x) ---
    stats = {"strategy": strategy, "applied": True, "n_bulk": n_bulk}
    try:
        # sample diagnostic points uniformly from voxel grid via a coarse random sampling around bulk centers
        sample_count = min(8192, int(new_xyz_full.shape[0]) * 4)
        if sample_count > 0:
            # random uniform sample within the bounding box of bulk centers
            mins = new_xyz_bulk.min(dim=0).values
            maxs = new_xyz_bulk.max(dim=0).values
            jitter = torch.rand((sample_count, 3), device=device) * (maxs - mins) + mins
            sdf_jitter = _sample_field_at(sdf_volume_5d, jitter, sdf_spacing)
            tau_init = float(getattr(args, "ct_bulk_halfspace_tau_init", 0.5))
            new_sigma_bulk = sigma_target  # using updated sigma
            new_sdf_bulk = sdf_at_centers   # already updated
            C_vals = _coverage_at_points(jitter, new_xyz_bulk, new_sigma_bulk, new_sdf_bulk, tau_init)
            delta = float(getattr(args, "ct_loss_boundary_band_delta", 0.5))
            material_mask = sdf_jitter <= -delta
            shell_mask = (sdf_jitter > -delta) & (sdf_jitter <= 0.0)
            void_mask = sdf_jitter > 0.0
            stats["coverage_p10_material"] = _quantile_safe(C_vals[material_mask], 0.10)
            stats["coverage_p50_material"] = _quantile_safe(C_vals[material_mask], 0.50)
            stats["coverage_p10_shell"] = _quantile_safe(C_vals[shell_mask], 0.10)
            stats["coverage_p50_shell"] = _quantile_safe(C_vals[shell_mask], 0.50)
            stats["coverage_p95_void"] = _quantile_safe(C_vals[void_mask], 0.95)
            stats["coverage_p50_void"] = _quantile_safe(C_vals[void_mask], 0.50)
            stats["material_p10_gate_pass"] = bool(
                stats["coverage_p10_material"] >= float(getattr(args, "ct_init_coverage_c_min_material", 0.5))
            )
            stats["shell_p10_gate_pass"] = bool(
                stats["coverage_p10_shell"] >= float(getattr(args, "ct_init_coverage_c_min_shell", 0.3))
            )
            stats["void_p95_gate_pass"] = bool(
                stats["coverage_p95_void"] <= float(getattr(args, "ct_init_coverage_void_epsilon", 0.05))
            )
    except Exception as exc:  # diagnostics should never block init
        stats["coverage_diagnostics_error"] = str(exc)

    # Sigma / atten statistics (cheap, always emit)
    stats["sigma_p10"] = _quantile_safe(sigma_target, 0.10)
    stats["sigma_p50"] = _quantile_safe(sigma_target, 0.50)
    stats["sigma_p90"] = _quantile_safe(sigma_target, 0.90)
    stats["atten_p10"] = _quantile_safe(a_target, 0.10)
    stats["atten_p50"] = _quantile_safe(a_target, 0.50)
    stats["atten_p90"] = _quantile_safe(a_target, 0.90)
    stats["n_deep"] = int(is_deep.sum().item())
    stats["n_boundary"] = int(((~is_deep) & (sdf_at_centers <= 0.0)).sum().item())
    stats["coverage_normalized_atten"] = bool(do_norm)
    stats["boundary_inward_nudge"] = bool(do_nudge)
    stats["boundary_sigma_c"] = float(c_factor)
    stats["deep_sigma_voxel"] = float(deep_sigma / voxel) if voxel > 0 else float(deep_sigma)

    if bool(getattr(args, "ct_init_coverage_report", True)):
        _write_init_report(model_path, stats)

    return stats
