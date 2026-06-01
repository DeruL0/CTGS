from __future__ import annotations

import math

import torch

from ct_pipeline.rendering.fields import density_to_occupancy, query_ct_density_from_state_by_region
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.mutations.helpers import (
    _frames_from_surface_normals,
    _project_xyz_to_sdf_zero,
    _sample_sdf_normals_for_reseed,
)
from ct_pipeline.training.utils import as_device_tensor
from utils.rotation_utils import matrix_to_quaternion


def _surface_reseed_stats(gaussians):
    count = int(gaussians.get_xyz.shape[0]) if getattr(gaussians, "is_initialized", lambda: False)() else 0
    return {
        "candidates": 0,
        "coverage_gap_ratio": 0.0,
        "bulk_owned_ratio": 0.0,
        "added": 0,
        "count_before": count,
        "count_after": count,
    }


def _sample_boundary_reseed_anchors(analysis, args, spacing_zyx, sample_count: int, curvature_field=None, device="cuda") -> torch.Tensor:
    boundary_points = analysis.get("boundary_points") if isinstance(analysis, dict) else None
    if boundary_points is None or int(sample_count) <= 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    points = as_device_tensor(boundary_points, device=device, dtype=torch.float32).reshape(-1, 3)
    if points.numel() == 0:
        return points

    count = int(sample_count)
    point_count = int(points.shape[0])
    uniform_ratio = 0.4
    uniform_count = min(count, int(round(count * uniform_ratio)))
    weighted_count = count - uniform_count
    selected_parts = []

    if uniform_count > 0:
        if uniform_count > point_count:
            indices = torch.randint(0, point_count, (uniform_count,), device=device)
        else:
            indices = torch.randperm(point_count, device=device)[:uniform_count]
        selected_parts.append(points.index_select(0, indices))

    if weighted_count > 0:
        weights = None
        if curvature_field is not None:
            curvature = sample_volume_field(
                curvature_field["curvature"],
                points,
                curvature_field["spacing_zyx"],
            ).reshape(-1).to(device=device, dtype=torch.float32)
            curvature = torch.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            weights = 1.0 + curvature
        elif "boundary_strength" in analysis:
            strength = as_device_tensor(analysis["boundary_strength"], device=device, dtype=torch.float32).reshape(-1)
            if strength.shape[0] == point_count:
                weights = torch.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        if weights is None or weights.numel() != point_count or float(weights.sum().item()) <= 0.0:
            if weighted_count > point_count:
                indices = torch.randint(0, point_count, (weighted_count,), device=device)
            else:
                indices = torch.randperm(point_count, device=device)[:weighted_count]
        else:
            indices = torch.multinomial(weights, weighted_count, replacement=weighted_count > point_count)
        selected_parts.append(points.index_select(0, indices))

    if not selected_parts:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    anchors = torch.cat(selected_parts, dim=0)
    if anchors.shape[0] > count:
        anchors = anchors[:count]
    return anchors


def _surface_reseed_material_ids(analysis, points_xyz: torch.Tensor, spacing_zyx) -> torch.Tensor:
    material_volume = analysis.get("material_label_volume") if isinstance(analysis, dict) else None
    if material_volume is None:
        return torch.zeros((points_xyz.shape[0], 1), dtype=torch.long, device=points_xyz.device)
    material_tensor = as_device_tensor(material_volume, device=points_xyz.device, dtype=torch.float32)
    material_tensor = material_tensor.reshape(1, 1, *tuple(int(value) for value in material_tensor.shape[-3:]))
    sampled = sample_volume_field(material_tensor, points_xyz, spacing_zyx).reshape(-1)
    labels = torch.round(sampled).to(dtype=torch.long) - 1
    labels = torch.clamp(labels, min=0)
    return labels.reshape(-1, 1)


def _apply_surface_reseeding(
    gaussians,
    args,
    training_state,
    spacing_zyx,
    analysis,
    initial_gaussian_count: int,
    signed_distance_field,
    curvature_field=None,
    volume_field=None,
):
    stats = _surface_reseed_stats(gaussians)
    if not bool(getattr(args, "ct_enable_surface_reseeding", False)):
        return stats
    if gaussians.optimizer is None or not getattr(gaussians, "is_initialized", lambda: False)():
        return stats

    with torch.no_grad():
        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        count_before = int(gaussians.get_xyz.shape[0])
        stats["count_before"] = count_before
        max_count = int(math.floor(max(1, int(initial_gaussian_count)) * float(args.ct_surface_reseed_max_gaussian_ratio)))
        remaining_budget = max(0, max_count - count_before)
        max_new = min(remaining_budget, int(args.ct_surface_reseed_max_new_per_iter))
        if max_new <= 0:
            return stats

        anchors = _sample_boundary_reseed_anchors(
            analysis,
            args,
            spacing_zyx,
            int(args.ct_surface_reseed_sample_count),
            curvature_field=curvature_field,
            device=device,
        ).to(device=device, dtype=dtype)
        if anchors.numel() == 0:
            return stats
        anchors = _project_xyz_to_sdf_zero(anchors, signed_distance_field)

        bulk_occ = density_to_occupancy(
            query_ct_density_from_state_by_region(training_state, anchors, region="bulk", detach=True)
        ).to(dtype=torch.float32)
        surface_occ = density_to_occupancy(
            query_ct_density_from_state_by_region(training_state, anchors, region="surface", detach=True)
        ).to(dtype=torch.float32)
        coverage_gap = surface_occ <= float(args.ct_surface_reseed_max_surface_occupancy)
        bulk_owned = coverage_gap & (bulk_occ >= float(args.ct_surface_reseed_min_bulk_occupancy))
        stats["candidates"] = int(anchors.shape[0])
        stats["coverage_gap_ratio"] = float(coverage_gap.float().mean().item()) if coverage_gap.numel() > 0 else 0.0
        stats["bulk_owned_ratio"] = float(bulk_owned.float().mean().item()) if bulk_owned.numel() > 0 else 0.0
        if not torch.any(bulk_owned):
            return stats

        candidate_indices = torch.nonzero(bulk_owned, as_tuple=False).reshape(-1)
        candidate_score = bulk_occ[candidate_indices] - surface_occ[candidate_indices]
        if curvature_field is not None:
            curvature = sample_volume_field(
                curvature_field["curvature"],
                anchors[candidate_indices],
                curvature_field["spacing_zyx"],
            ).reshape(-1).to(device=device, dtype=torch.float32).clamp_min(0.0)
            candidate_score = candidate_score * (1.0 + curvature)
        if candidate_indices.numel() > max_new:
            _, order = torch.topk(candidate_score, k=max_new, largest=True, sorted=False)
            candidate_indices = candidate_indices[order]

        new_xyz = anchors[candidate_indices].to(dtype=dtype)
        new_count = int(new_xyz.shape[0])
        if new_count <= 0:
            return stats

        normals = _sample_sdf_normals_for_reseed(new_xyz, signed_distance_field).to(dtype=dtype)
        rotations = matrix_to_quaternion(_frames_from_surface_normals(normals)).to(dtype=dtype)

        tangent_scale = max(
            float(getattr(args, "ct_surface_reseed_coverage_radius", args.ct_surface_reseed_coverage_radius_voxels)),
            float(args.ct_surface_sigma_t_min),
            1e-8,
        )
        normal_scale = max(float(args.ct_surface_sigma_n_max), 1e-8)
        new_scaling = torch.log(
            torch.tensor((tangent_scale, tangent_scale, normal_scale), dtype=dtype, device=device).reshape(1, 3).expand(new_count, 3)
        )
        new_opacity = gaussians.inverse_opacity_activation(
            torch.full((new_count, 1), 0.5, dtype=dtype, device=device)
        )
        feature_dc_shape = (new_count,) + tuple(int(value) for value in gaussians._features_dc.shape[1:])
        new_features_dc = torch.full(feature_dc_shape, 0.5, dtype=gaussians._features_dc.dtype, device=device)
        feature_rest_shape = (new_count,) + tuple(int(value) for value in gaussians._features_rest.shape[1:])
        new_features_rest = torch.zeros(feature_rest_shape, dtype=gaussians._features_rest.dtype, device=device)
        primitive_value = float(getattr(gaussians, "nonplanar_logit_value", -8.0))
        new_primitive_type = torch.full((new_count, 1), primitive_value, dtype=dtype, device=device)
        new_material_id = _surface_reseed_material_ids(analysis, new_xyz, spacing_zyx)
        new_planarity = torch.zeros((new_count, 1), dtype=torch.float32, device=device)
        new_region_type = torch.zeros((new_count, 1), dtype=torch.long, device=device)

        # Initialize ct_value_logit from sampled CT volume at anchor position (per user spec).
        new_ct_value_logit = None
        new_atten_logit = None
        if gaussians._ct_value_logit.numel() > 0:
            if volume_field is not None:
                sampled = sample_volume_field(volume_field, new_xyz, spacing_zyx).reshape(-1).to(dtype=torch.float32)
                sampled = torch.clamp(sampled, 1e-4, 1.0 - 1e-4)
                new_ct_value_logit = torch.log(sampled / (1.0 - sampled)).reshape(new_count, 1).to(
                    dtype=gaussians._ct_value_logit.dtype, device=device
                )
                if gaussians._atten_logit.numel() > 0:
                    new_atten_logit = gaussians._inverse_softplus(sampled.reshape(new_count, 1).clamp_min(1e-6)).to(
                        dtype=gaussians._atten_logit.dtype, device=device
                    )
            else:
                new_ct_value_logit = torch.zeros((new_count, 1), dtype=gaussians._ct_value_logit.dtype, device=device)
                if gaussians._atten_logit.numel() > 0:
                    new_atten_logit = torch.zeros((new_count, 1), dtype=gaussians._atten_logit.dtype, device=device)

        gaussians.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            rotations,
            new_primitive_type,
            normals,
            new_material_id,
            new_planarity,
            new_region_type,
            new_ct_value_logit=new_ct_value_logit,
            new_atten_logit=new_atten_logit,
        )
        stats["added"] = new_count
        stats["count_after"] = int(gaussians.get_xyz.shape[0])
    return stats
