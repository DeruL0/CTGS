from __future__ import annotations

import torch
import torch.nn.functional as F

from ct_pipeline.training.bootstrap import CTTrainingBootstrap
from ct_pipeline.training.sampling import (
    _cached_or_filter_candidates,
    _candidate_count,
    _ct_empty_points,
    _sample_occupancy_points,
    _sample_signed_distance,
)
from ct_pipeline.training.utils import as_device_tensor


def _phase_mask_from_analysis(analysis: dict):
    if not isinstance(analysis, dict):
        return None
    return analysis.get("material_mask")


def _sample_support_membership(support_mask, points_xyz: torch.Tensor, spacing_zyx) -> torch.Tensor | None:
    if support_mask is None or points_xyz.numel() == 0:
        return None

    mask = torch.as_tensor(support_mask, device=points_xyz.device)
    if mask.ndim < 3:
        return None
    mask = mask.reshape(*tuple(int(value) for value in mask.shape[-3:]))

    spacing_z, spacing_y, spacing_x = [max(float(value), 1e-8) for value in spacing_zyx]
    depth, height, width = [int(value) for value in mask.shape]
    x_index = torch.floor(points_xyz[:, 0] / spacing_x).to(dtype=torch.long).clamp(0, max(width - 1, 0))
    y_index = torch.floor(points_xyz[:, 1] / spacing_y).to(dtype=torch.long).clamp(0, max(height - 1, 0))
    z_index = torch.floor(points_xyz[:, 2] / spacing_z).to(dtype=torch.long).clamp(0, max(depth - 1, 0))
    return mask[z_index, y_index, x_index].to(dtype=torch.float32).reshape(-1)


def _split_phase_occupancy_sample_counts(args, sample_count: int) -> tuple[int, int, int]:
    sample_count = max(1, int(sample_count))
    boundary_ratio = float(getattr(args, "ct_occ_boundary_sample_ratio", 0.50))
    material_ratio = float(getattr(args, "ct_occ_deep_material_sample_ratio", 0.35))
    boundary_ratio = min(max(boundary_ratio, 0.0), 1.0)
    material_ratio = min(max(material_ratio, 0.0), 1.0 - boundary_ratio)
    boundary_count = int(round(float(sample_count) * boundary_ratio))
    material_count = int(round(float(sample_count) * material_ratio))
    air_count = sample_count - boundary_count - material_count
    if sample_count >= 3:
        boundary_count = max(1, boundary_count)
        material_count = max(1, material_count)
        air_count = max(1, sample_count - boundary_count - material_count)
        overflow = boundary_count + material_count + air_count - sample_count
        if overflow > 0:
            boundary_count = max(1, boundary_count - overflow)
    return boundary_count, material_count, air_count


def _sample_air_points_with_void_bias(context: CTTrainingBootstrap, air_count: int, boundary_band_distance: float, device) -> torch.Tensor:
    if int(air_count) <= 0:
        return _ct_empty_points(device=device)

    void_candidates = context.field_pools.get("void_air_pool", context.field_pools.get("void_air"))
    exterior_candidates = context.field_pools.get("exterior_air_near_band")
    if _candidate_count(exterior_candidates) == 0:
        exterior_candidates = context.field_pools.get("exterior_air_pool")
    if _candidate_count(exterior_candidates) == 0:
        exterior_candidates = _cached_or_filter_candidates(
            context.field_pools,
            "exterior_air",
            context.signed_distance_field,
            boundary_band_distance,
            keep_boundary=False,
        )
    if _candidate_count(exterior_candidates) == 0:
        exterior_candidates = _cached_or_filter_candidates(
            context.field_pools,
            "air",
            context.signed_distance_field,
            boundary_band_distance,
            keep_boundary=False,
        )

    void_available = _candidate_count(void_candidates)
    exterior_available = _candidate_count(exterior_candidates)
    if void_available <= 0 and exterior_available <= 0:
        return _ct_empty_points(device=device)

    if void_available <= 0:
        void_count = 0
        exterior_count = int(air_count)
    elif exterior_available <= 0:
        void_count = int(air_count)
        exterior_count = 0
    else:
        natural_void_ratio = float(void_available) / float(void_available + exterior_available)
        void_share = min(1.0, max(0.5, natural_void_ratio))
        max_exterior_ratio = min(max(float(getattr(context, "exterior_air_sample_ratio", 0.5)), 0.0), 1.0)
        void_share = max(void_share, 1.0 - max_exterior_ratio)
        void_count = int(round(float(air_count) * void_share))
        void_count = max(1, min(int(air_count) - 1, void_count))
        exterior_count = int(air_count) - void_count

    parts = []
    if void_count > 0:
        parts.append(_sample_occupancy_points(void_candidates, void_count, context.spacing_zyx, device=device))
    if exterior_count > 0:
        parts.append(_sample_occupancy_points(exterior_candidates, exterior_count, context.spacing_zyx, device=device))
    if not parts:
        return _ct_empty_points(device=device)
    return torch.cat(parts, dim=0)[: int(air_count)]


def _phase_occupancy_sdf_weights(
    signed_distance: torch.Tensor,
    boundary_band_distance: float,
    boundary_weight: float,
) -> torch.Tensor:
    signed_distance = signed_distance.to(dtype=torch.float32).reshape(-1)
    band = max(float(boundary_band_distance), 1e-6)
    max_weight = max(float(boundary_weight), 1.0)
    proximity = (1.0 - torch.abs(signed_distance) / band).clamp(0.0, 1.0)
    return 1.0 + (max_weight - 1.0) * proximity


def _sample_filtered_semantic_points(
    candidates,
    sample_count: int,
    context: CTTrainingBootstrap,
    *,
    device,
    signed_distance_predicate,
    oversample: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(sample_count) <= 0 or _candidate_count(candidates) <= 0:
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    draw_count = max(int(sample_count), int(sample_count) * max(1, int(oversample)))
    points = _sample_occupancy_points(candidates, draw_count, context.spacing_zyx, device=device)
    if points.numel() == 0:
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    signed_distance = _sample_signed_distance(context.signed_distance_field, points).to(dtype=torch.float32)
    keep = signed_distance_predicate(signed_distance) & torch.isfinite(signed_distance)
    if not torch.any(keep):
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    points = points[keep][: int(sample_count)]
    signed_distance = signed_distance[keep][: int(sample_count)]
    return points, signed_distance


def _sample_filtered_from_candidate_sets(
    candidate_sets,
    sample_count: int,
    context: CTTrainingBootstrap,
    *,
    device,
    signed_distance_predicate,
    oversample: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    remaining = int(sample_count)
    if remaining <= 0:
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    point_parts = []
    sdf_parts = []
    for candidates in candidate_sets:
        if remaining <= 0:
            break
        if _candidate_count(candidates) <= 0:
            continue
        points, signed_distance = _sample_filtered_semantic_points(
            candidates,
            remaining,
            context,
            device=device,
            signed_distance_predicate=signed_distance_predicate,
            oversample=oversample,
        )
        if points.numel() == 0:
            continue
        point_parts.append(points)
        sdf_parts.append(signed_distance)
        remaining -= int(points.shape[0])
    if not point_parts:
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    return torch.cat(point_parts, dim=0)[: int(sample_count)], torch.cat(sdf_parts, dim=0)[: int(sample_count)]


def _sample_boundary_offset_shell_points(
    context: CTTrainingBootstrap,
    sample_count: int,
    *,
    offset_min: float,
    offset_max: float,
    device,
    signed_distance_predicate,
    oversample: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(sample_count) <= 0:
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    if float(offset_max) <= float(offset_min):
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    anchors = context.analysis.get("boundary_points")
    normals = context.analysis.get("boundary_normals", context.analysis.get("boundary_normal"))
    if anchors is None or normals is None:
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    anchors = as_device_tensor(anchors, device=device, dtype=torch.float32, reshape=(-1, 3))
    normals = as_device_tensor(normals, device=device, dtype=torch.float32, reshape=(-1, 3))
    if anchors.numel() == 0 or normals.shape[0] != anchors.shape[0]:
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    request_count = max(int(sample_count), int(sample_count) * max(1, int(oversample)))
    indices = torch.randint(0, int(anchors.shape[0]), (request_count,), device=device)
    chosen_anchors = anchors.index_select(0, indices)
    chosen_normals = F.normalize(normals.index_select(0, indices), dim=-1, eps=1e-6)
    min_spacing = min(float(value) for value in context.spacing_zyx)
    offsets = torch.empty((request_count, 1), dtype=torch.float32, device=device).uniform_(
        float(offset_min) * float(min_spacing),
        float(offset_max) * float(min_spacing),
    )
    points = chosen_anchors + chosen_normals * offsets
    depth, height, width = [int(value) for value in context.volume_shape]
    spacing_z, spacing_y, spacing_x = [float(value) for value in context.spacing_zyx]
    lower = torch.zeros((3,), dtype=torch.float32, device=device)
    upper = torch.tensor(
        [
            max(0.0, (float(width) - 1e-3) * spacing_x),
            max(0.0, (float(height) - 1e-3) * spacing_y),
            max(0.0, (float(depth) - 1e-3) * spacing_z),
        ],
        dtype=torch.float32,
        device=device,
    )
    points = torch.minimum(torch.maximum(points, lower.unsqueeze(0)), upper.unsqueeze(0))
    signed_distance = _sample_signed_distance(context.signed_distance_field, points).to(dtype=torch.float32)
    keep = signed_distance_predicate(signed_distance) & torch.isfinite(signed_distance)
    if not torch.any(keep):
        return _ct_empty_points(device=device), torch.empty((0,), dtype=torch.float32, device=device)
    return points[keep][: int(sample_count)], signed_distance[keep][: int(sample_count)]
