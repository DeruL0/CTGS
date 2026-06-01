from __future__ import annotations

import math

import numpy as np
import torch

from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.mutations.helpers import (
    _bulk_mask_tensor,
    _sample_sdf_normals_for_reseed,
)
from utils.rotation_utils import quaternion_to_matrix


def _bulk_reseed_stats(gaussians):
    return {
        "added": 0,
        "candidates": 0,
        "low_coverage_ratio": 0.0,
        "sigma_init_mean": float("nan"),
        "atten_init_mean": float("nan"),
        "bulk_grown_count": 0,
        "num_uncovered_components": 0,
        "max_uncovered_component_voxels": 0,
        "repair_residual_mass": 0.0,
        "repair_stretched_count": 0,
        "repair_skipped_small_components": 0,
        "repair_skipped_exclusion": 0,
        "repair_skipped_low_gain": 0,
        "repair_skipped_containment": 0,
        "repair_skipped_overfill": 0,
        "repair_skipped_no_clearance_headroom": 0,
        "repair_clearance_limited": 0,
        "repair_components_considered": 0,
        "completion_pass": 0,
        "count_before": int(gaussians.get_xyz.shape[0]),
        "count_after": int(gaussians.get_xyz.shape[0]),
    }


def _apply_material_limited_bulk_growth(gaussians, args, signed_distance_field) -> int:
    factor = float(getattr(args, "ct_gap_bulk_growth_factor", 1.0))
    if factor <= 1.0 or signed_distance_field is None:
        return 0
    bulk_mask = _bulk_mask_tensor(gaussians)
    if not torch.any(bulk_mask):
        return 0
    with torch.no_grad():
        bulk_xyz = gaussians.get_xyz.detach()[bulk_mask]
        current_scales = torch.exp(gaussians._scaling.detach()[bulk_mask]).to(dtype=torch.float32)
        proposed_scales = current_scales * factor
        rotations = quaternion_to_matrix(gaussians.get_rotation.detach()[bulk_mask]).to(device=bulk_xyz.device, dtype=torch.float32)
        can_grow_parts = []
        sqrt_q = 2.0
        diag_factor = math.sqrt(2.0)
        chunk = 32768
        for start in range(0, int(bulk_xyz.shape[0]), chunk):
            end = min(start + chunk, int(bulk_xyz.shape[0]))
            xyz = bulk_xyz[start:end].to(dtype=torch.float32)
            scales = proposed_scales[start:end].to(device=xyz.device, dtype=torch.float32)
            rotation = rotations[start:end]
            offsets = [torch.zeros_like(xyz)]
            for axis in range(3):
                offset = rotation[:, :, axis] * scales[:, axis : axis + 1] * sqrt_q
                offsets.extend((offset, -offset))
            diag_01 = (
                rotation[:, :, 0] * scales[:, 0:1]
                + rotation[:, :, 1] * scales[:, 1:2]
            ) * diag_factor
            diag_0m1 = (
                rotation[:, :, 0] * scales[:, 0:1]
                - rotation[:, :, 1] * scales[:, 1:2]
            ) * diag_factor
            offsets.extend((diag_01, -diag_01, diag_0m1, -diag_0m1))
            probes = torch.stack([xyz + offset for offset in offsets], dim=1).reshape(-1, 3)
            sdf = sample_volume_field(
                signed_distance_field["signed_distance"],
                probes,
                signed_distance_field["spacing_zyx"],
            ).reshape(end - start, -1)
            contained = torch.all(torch.isfinite(sdf) & (sdf < 0.0), dim=1)
            can_grow_parts.append(contained)
        can_grow = torch.cat(can_grow_parts, dim=0) if can_grow_parts else torch.empty((0,), dtype=torch.bool, device=bulk_xyz.device)
        if not torch.any(can_grow):
            return 0
        bulk_indices = torch.nonzero(bulk_mask, as_tuple=False).reshape(-1)
        grow_indices = bulk_indices[can_grow]
        gaussians._scaling[grow_indices] = gaussians._scaling[grow_indices] + math.log(factor)
        return int(grow_indices.numel())


def _enforce_bulk_sdf_containment(
    xyz: torch.Tensor,
    scales: torch.Tensor,
    rotation_mats: torch.Tensor,
    signed_distance_field: dict,
    spacing_zyx,
    *,
    margin: float = 0.0,
    min_scale: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if xyz.numel() == 0 or signed_distance_field is None:
        keep = torch.ones((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
        return scales, keep
    signed_distance = sample_volume_field(
        signed_distance_field["signed_distance"],
        xyz,
        signed_distance_field["spacing_zyx"],
    ).reshape(-1).to(device=xyz.device, dtype=torch.float32)
    normals = _sample_sdf_normals_for_reseed(xyz, signed_distance_field).to(device=xyz.device, dtype=xyz.dtype)
    local_normals = torch.einsum("nij,nj->ni", rotation_mats.transpose(1, 2), normals)
    radius = torch.sqrt(torch.sum((local_normals * scales) ** 2, dim=-1).clamp_min(1e-8)).to(dtype=torch.float32)
    allowed = (-signed_distance - float(margin)).to(dtype=torch.float32)
    keep = torch.isfinite(allowed) & (allowed > float(min_scale)) & torch.isfinite(radius)
    factor = torch.ones_like(radius)
    shrink = keep & (radius > allowed)
    factor[shrink] = (allowed[shrink] / radius[shrink].clamp_min(1e-8)).clamp(0.0, 1.0)
    contained_scales = scales * factor.to(device=scales.device, dtype=scales.dtype).unsqueeze(1)
    keep = keep & torch.all(contained_scales >= float(min_scale), dim=1)
    return contained_scales.clamp_min(float(min_scale)), keep


def _mask_to_numpy_bool(mask) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy().astype(bool, copy=False)
    return np.asarray(mask, dtype=bool)


def _sdf_to_numpy(signed_distance_field, analysis, shape) -> np.ndarray | None:
    sdf = None
    if signed_distance_field is not None:
        sdf = signed_distance_field.get("signed_distance_native", signed_distance_field.get("signed_distance"))
    if sdf is None and isinstance(analysis, dict):
        sdf = analysis.get("material_signed_distance")
    if sdf is None:
        return None
    if isinstance(sdf, torch.Tensor):
        sdf_np = sdf.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        sdf_np = np.asarray(sdf, dtype=np.float32)
    try:
        sdf_np = np.reshape(sdf_np, tuple(int(v) for v in shape))
    except ValueError:
        return None
    return sdf_np


def _voxel_indices_to_world(indices_zyx: np.ndarray, spacing_zyx, device, dtype) -> torch.Tensor:
    if indices_zyx.size == 0:
        return torch.empty((0, 3), dtype=dtype, device=device)
    spacing_z, spacing_y, spacing_x = [float(v) for v in spacing_zyx]
    points_np = np.stack(
        (
            (indices_zyx[:, 2].astype(np.float32) + 0.5) * spacing_x,
            (indices_zyx[:, 1].astype(np.float32) + 0.5) * spacing_y,
            (indices_zyx[:, 0].astype(np.float32) + 0.5) * spacing_z,
        ),
        axis=1,
    )
    return torch.as_tensor(points_np, dtype=dtype, device=device)
