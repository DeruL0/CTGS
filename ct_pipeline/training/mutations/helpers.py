from __future__ import annotations

import torch

from ct_pipeline.geometry.coordinates import world_xyz_to_voxel_indices_floor_torch
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.utils import as_device_tensor


def _bulk_mask_tensor(gaussians) -> torch.Tensor:
    return gaussians.get_region_type.reshape(-1) == 1


def _sample_binary_mask_nearest(mask_volume, points_xyz: torch.Tensor, spacing_zyx) -> torch.Tensor:
    points_xyz = torch.as_tensor(points_xyz)
    if points_xyz.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device=points_xyz.device)

    mask_t = as_device_tensor(mask_volume, device=points_xyz.device, dtype=torch.bool)
    if mask_t.ndim != 3:
        raise ValueError("mask_volume must have shape (D, H, W).")

    depth, height, width = [int(value) for value in mask_t.shape]
    z_idx, y_idx, x_idx = world_xyz_to_voxel_indices_floor_torch(points_xyz, spacing_zyx, shape_dhw=(depth, height, width))
    return mask_t[z_idx, y_idx, x_idx]


def _apply_surface_scale_hard_projection(gaussians, spacing_zyx, max_scale: float):
    surface_mask = gaussians.get_region_type.reshape(-1) == 0
    if not torch.any(surface_mask):
        return 0
    del spacing_zyx
    max_scale = float(max_scale)
    with torch.no_grad():
        surface_scales = torch.exp(gaussians._scaling[surface_mask])
        clipped = torch.clamp(surface_scales, min=1e-8, max=max_scale)
        changed = torch.any(torch.abs(clipped - surface_scales) > 1e-8, dim=1)
        gaussians._scaling[surface_mask] = torch.log(clipped)
    return int(changed.sum().item())


def _log_air_shell_diagnostics(field_pools):
    ratio = float(field_pools.get("air_shell_band_ratio", 1.0) or 0.0)
    air_shell = field_pools.get("air_shell")
    shell_count = int(air_shell.shape[0]) if isinstance(air_shell, torch.Tensor) else int(len(air_shell))
    print(
        "Air-shell diagnostics: near-boundary fraction within configured band = {0:.3f} (air_shell_count={1})".format(
            ratio,
            shell_count,
        )
    )
    return ratio


def _freeze_bulk_xyz_gradients(gaussians):
    if not getattr(gaussians, "is_initialized", lambda: False)():
        return 0
    xyz_grad = getattr(gaussians, "_xyz", None)
    if not isinstance(xyz_grad, torch.Tensor) or xyz_grad.grad is None:
        return 0
    region_type = gaussians.get_region_type.reshape(-1)
    bulk_mask = region_type == 1
    if not torch.any(bulk_mask):
        return 0
    with torch.no_grad():
        xyz_grad.grad[bulk_mask] = 0
    return int(bulk_mask.sum().item())


def _ct_topk_indices(candidate_mask: torch.Tensor, scores: torch.Tensor, max_count: int) -> torch.Tensor:
    candidate_indices = torch.nonzero(candidate_mask.reshape(-1), as_tuple=False).reshape(-1)
    if candidate_indices.numel() == 0 or int(max_count) <= 0:
        return candidate_indices[:0]
    candidate_scores = as_device_tensor(scores, device=candidate_indices.device).reshape(-1)[candidate_indices]
    valid = torch.isfinite(candidate_scores)
    candidate_indices = candidate_indices[valid]
    candidate_scores = candidate_scores[valid]
    if candidate_indices.numel() == 0:
        return candidate_indices
    if candidate_indices.numel() <= int(max_count):
        return candidate_indices
    _, order = torch.topk(candidate_scores, k=int(max_count), largest=True, sorted=False)
    return candidate_indices[order]


def _ct_support_volume_from_analysis(analysis, device, dtype):
    support_mask = analysis.get("material_mask")
    if support_mask is None:
        return None
    support_tensor = as_device_tensor(support_mask, device=device, dtype=dtype)
    return support_tensor.reshape(1, 1, *tuple(int(value) for value in support_tensor.shape[-3:]))


def _sample_sdf_normals_for_reseed(points_xyz: torch.Tensor, signed_distance_field: dict) -> torch.Tensor:
    normal_volume = signed_distance_field.get("sdf_normal")
    if normal_volume is not None:
        normals = sample_volume_field(
            normal_volume,
            points_xyz,
            signed_distance_field.get("spacing_zyx"),
        ).to(dtype=points_xyz.dtype)
    else:
        eps = 0.5 * float(min(signed_distance_field["spacing_zyx"]))
        offsets = torch.eye(3, dtype=points_xyz.dtype, device=points_xyz.device) * eps
        sdf = signed_distance_field["signed_distance"]
        spacing = signed_distance_field["spacing_zyx"]
        normals = torch.stack(
            (
                (sample_volume_field(sdf, points_xyz + offsets[0], spacing).reshape(-1) - sample_volume_field(sdf, points_xyz - offsets[0], spacing).reshape(-1)) / (2.0 * eps),
                (sample_volume_field(sdf, points_xyz + offsets[1], spacing).reshape(-1) - sample_volume_field(sdf, points_xyz - offsets[1], spacing).reshape(-1)) / (2.0 * eps),
                (sample_volume_field(sdf, points_xyz + offsets[2], spacing).reshape(-1) - sample_volume_field(sdf, points_xyz - offsets[2], spacing).reshape(-1)) / (2.0 * eps),
            ),
            dim=1,
        )
    normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.norm(normals, dim=1, keepdim=True)
    fallback = torch.tensor((0.0, 0.0, 1.0), dtype=points_xyz.dtype, device=points_xyz.device).reshape(1, 3).expand_as(normals)
    return torch.where(norm > 1e-6, normals / norm.clamp_min(1e-6), fallback)


def _frames_from_surface_normals(normals: torch.Tensor) -> torch.Tensor:
    normal = torch.nn.functional.normalize(normals, dim=-1, eps=1e-8)
    x_axis = torch.tensor((1.0, 0.0, 0.0), dtype=normal.dtype, device=normal.device).reshape(1, 3).expand_as(normal)
    y_axis = torch.tensor((0.0, 1.0, 0.0), dtype=normal.dtype, device=normal.device).reshape(1, 3).expand_as(normal)
    reference = torch.where(torch.abs(normal[:, :1]) < 0.9, x_axis, y_axis)
    tangent_u = reference - torch.sum(reference * normal, dim=1, keepdim=True) * normal
    tangent_u = torch.nn.functional.normalize(tangent_u, dim=-1, eps=1e-8)
    tangent_v = torch.cross(normal, tangent_u, dim=1)
    tangent_v = torch.nn.functional.normalize(tangent_v, dim=-1, eps=1e-8)
    return torch.stack((tangent_u, tangent_v, normal), dim=2)


def _ensure_gap_seed_birth_iter(gaussians) -> torch.Tensor:
    count = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    birth = getattr(gaussians, "_ct_gap_seed_birth_iter", None)
    if not isinstance(birth, torch.Tensor) or birth.shape[0] != count:
        birth = torch.full((count,), -1, dtype=torch.long, device=device)
        setattr(gaussians, "_ct_gap_seed_birth_iter", birth)
    return birth.to(device=device, dtype=torch.long)


def _append_gap_seed_birth_iter(gaussians, previous_birth: torch.Tensor, new_count: int, iteration: int, *, gap_seed: bool) -> None:
    device = gaussians.get_xyz.device
    fill_value = int(iteration) if gap_seed else -1
    new_birth = torch.full((int(new_count),), fill_value, dtype=torch.long, device=device)
    setattr(
        gaussians,
        "_ct_gap_seed_birth_iter",
        torch.cat((previous_birth.to(device=device, dtype=torch.long), new_birth), dim=0),
    )


def _project_xyz_to_sdf_zero(
    xyz: torch.Tensor,
    signed_distance_field: dict,
    max_iter: int = 3,
    step_factor: float = 0.8,
) -> torch.Tensor:
    if xyz.numel() == 0 or signed_distance_field is None:
        return xyz

    sdf_volume = signed_distance_field["signed_distance"]
    sdf_spacing = signed_distance_field["spacing_zyx"]
    eps = float(min(sdf_spacing)) * 0.5

    current = xyz.clone()
    offsets = torch.eye(3, dtype=current.dtype, device=current.device) * eps
    for _ in range(int(max_iter)):
        d_center = sample_volume_field(sdf_volume, current, sdf_spacing).reshape(-1)
        d_plus_x = sample_volume_field(sdf_volume, current + offsets[0], sdf_spacing).reshape(-1)
        d_minus_x = sample_volume_field(sdf_volume, current - offsets[0], sdf_spacing).reshape(-1)
        d_plus_y = sample_volume_field(sdf_volume, current + offsets[1], sdf_spacing).reshape(-1)
        d_minus_y = sample_volume_field(sdf_volume, current - offsets[1], sdf_spacing).reshape(-1)
        d_plus_z = sample_volume_field(sdf_volume, current + offsets[2], sdf_spacing).reshape(-1)
        d_minus_z = sample_volume_field(sdf_volume, current - offsets[2], sdf_spacing).reshape(-1)
        grad = torch.stack(
            (
                (d_plus_x - d_minus_x) / (2.0 * eps),
                (d_plus_y - d_minus_y) / (2.0 * eps),
                (d_plus_z - d_minus_z) / (2.0 * eps),
            ),
            dim=1,
        )
        grad_norm_sq = (grad * grad).sum(dim=1).clamp_min(1e-8)
        update = (d_center / grad_norm_sq).unsqueeze(1) * grad
        current = current - float(step_factor) * update
    return current
