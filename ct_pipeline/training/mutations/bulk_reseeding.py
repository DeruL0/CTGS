from __future__ import annotations

import math

import numpy as np
import torch

from ct_pipeline.rendering.fields import (
    density_to_occupancy,
    is_bulk_intensity_field_mode,
    query_ct_density_from_state_by_region,
    query_ct_fields_unified,
)
from ct_pipeline.training.mutations.bulk_completion import (
    _append_bulk_completion_seeds,
    _apply_material_coverage_completion,
    _material_completion_components,
)
from ct_pipeline.training.mutations.bulk_repair import (
    _append_budgeted_repair_seeds,
    _apply_budgeted_component_bulk_repair,
    _budgeted_repair_components,
    _clearance_cap_candidate_scales,
    _component_world_points,
    _directional_clearance_one_side_torch,
    _gaussian_density_np,
    _probe_candidate_inside_material,
    _sdf_inside_np,
)
from ct_pipeline.training.mutations.bulk_reseed_common import (
    _apply_material_limited_bulk_growth,
    _bulk_reseed_stats,
    _enforce_bulk_sdf_containment,
    _mask_to_numpy_bool,
    _sdf_to_numpy,
    _voxel_indices_to_world,
)
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.mutations.helpers import (
    _append_gap_seed_birth_iter,
    _bulk_mask_tensor,
    _ensure_gap_seed_birth_iter,
    _frames_from_surface_normals,
    _sample_binary_mask_nearest,
    _sample_sdf_normals_for_reseed,
)
from ct_pipeline.training.sampling import _candidate_count, _sample_occupancy_points
from ct_pipeline.training.utils import as_device_tensor
from utils.rotation_utils import matrix_to_quaternion


def _apply_profile_integrated_bulk_reseeding(
    gaussians,
    args,
    training_state,
    spacing_zyx,
    analysis,
    initial_gaussian_count: int,
    volume_field=None,
    signed_distance_field=None,
):
    stats = _bulk_reseed_stats(gaussians)
    if gaussians.optimizer is None or not getattr(gaussians, "is_initialized", lambda: False)():
        return stats
    if signed_distance_field is None or volume_field is None:
        return stats

    with torch.no_grad():
        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        count_before = int(gaussians.get_xyz.shape[0])
        stats["count_before"] = count_before

        max_ratio = float(getattr(args, "ct_bulk_reseed_max_gaussian_ratio", 2.0))
        max_count = int(math.floor(max(1, initial_gaussian_count) * max_ratio))
        remaining_budget = max(0, max_count - count_before)
        bulk_count = int((gaussians.get_region_type.reshape(-1) == 1).sum().item())
        per_iter_limit = max(1, int(math.ceil(max(1, bulk_count) * max(float(getattr(args, "ct_reseed_max_new_fraction", 0.05)), 0.0))))
        max_new = min(remaining_budget, per_iter_limit)
        if max_new <= 0:
            return stats

        anchors = analysis.get("boundary_points")
        normals = analysis.get("boundary_normals", analysis.get("boundary_normal"))
        if anchors is None or normals is None:
            return stats
        anchors = as_device_tensor(anchors, device=device, dtype=torch.float32, reshape=(-1, 3))
        normals = as_device_tensor(normals, device=device, dtype=torch.float32, reshape=(-1, 3))
        if anchors.numel() == 0 or normals.shape[0] != anchors.shape[0]:
            return stats

        sample_count = min(int(getattr(args, "ct_bulk_reseed_sample_count", 8192)), int(anchors.shape[0]))
        if sample_count <= 0:
            return stats
        indices = torch.randint(0, int(anchors.shape[0]), (sample_count,), device=device)
        anchors = anchors.index_select(0, indices)
        normals = torch.nn.functional.normalize(normals.index_select(0, indices), dim=-1, eps=1e-6)

        min_spacing = min(float(value) for value in spacing_zyx)
        probe_length = max(float(getattr(args, "ct_reseed_probe_length", 3.0)), 0.5) * float(min_spacing)
        probe_steps = max(4, int(math.ceil(float(getattr(args, "ct_reseed_probe_length", 3.0)) * 2.0)))
        offsets = torch.linspace(0.25 * float(min_spacing), probe_length, probe_steps, device=device, dtype=torch.float32)
        line_points = anchors[:, None, :] - normals[:, None, :] * offsets[None, :, None]

        depth, height, width = [int(value) for value in analysis.get("material_mask").shape]
        spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
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
        line_points = torch.minimum(torch.maximum(line_points, lower.view(1, 1, 3)), upper.view(1, 1, 3))

        flat_points = line_points.reshape(-1, 3)
        flat_sdf = sample_volume_field(
            signed_distance_field["signed_distance"],
            flat_points,
            signed_distance_field["spacing_zyx"],
        ).reshape(-1).to(dtype=torch.float32)
        fields = query_ct_fields_unified(
            flat_points,
            training_state,
            signed_distance=flat_sdf,
            config=args,
            intensity_air=0.0,
            include_surface=False,
            train_ct_value=False,
        )
        pred = fields["I_pred"].reshape(sample_count, probe_steps).to(dtype=torch.float32)
        target = sample_volume_field(volume_field, flat_points, spacing_zyx).reshape(sample_count, probe_steps).to(dtype=torch.float32)
        sdf = flat_sdf.reshape(sample_count, probe_steps)
        deficit = torch.relu(target - pred) * (sdf <= 0.0).to(dtype=torch.float32)
        deficit_integral = deficit.sum(dim=1) * (probe_length / float(max(probe_steps - 1, 1)))
        stats["candidates"] = int(sample_count)
        stats["low_coverage_ratio"] = float((deficit_integral > float(getattr(args, "ct_reseed_band_deficit_thr", 0.15))).float().mean().item())

        valid = deficit_integral > float(getattr(args, "ct_reseed_band_deficit_thr", 0.15))
        if not torch.any(valid):
            return stats

        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if valid_indices.numel() > max_new:
            _, order = torch.topk(deficit_integral[valid_indices], k=max_new, largest=True, sorted=False)
            valid_indices = valid_indices[order]

        seed_offset = max(float(getattr(args, "ct_reseed_seed_offset_sigma", 1.0)) * float(min_spacing), 0.0)
        seed_points = []
        for idx in valid_indices.tolist():
            row_sdf = sdf[idx]
            row_deficit = deficit[idx]
            preferred = torch.nonzero(row_sdf <= -float(seed_offset), as_tuple=False).reshape(-1)
            if preferred.numel() > 0:
                best_local = preferred[torch.argmax(row_deficit.index_select(0, preferred))]
            else:
                material_steps = torch.nonzero(row_sdf < 0.0, as_tuple=False).reshape(-1)
                if material_steps.numel() == 0:
                    continue
                best_local = material_steps[torch.argmax(row_deficit.index_select(0, material_steps))]
            seed_points.append(line_points[idx, int(best_local.item())])

        if not seed_points:
            return stats

        new_xyz = torch.stack(seed_points, dim=0).to(device=device, dtype=dtype)
        new_count = int(new_xyz.shape[0])
        normals = _sample_sdf_normals_for_reseed(new_xyz, signed_distance_field).to(dtype=dtype)
        new_rotation_mats = _frames_from_surface_normals(normals)
        new_rotations = matrix_to_quaternion(new_rotation_mats).to(dtype=dtype)
        signed_distance = sample_volume_field(
            signed_distance_field["signed_distance"],
            new_xyz,
            signed_distance_field["spacing_zyx"],
        ).reshape(-1).to(device=device, dtype=torch.float32)
        inside_distance = torch.relu(-signed_distance)
        sigma_init = torch.clamp(
            inside_distance * float(getattr(args, "ct_reseed_sigma_init_factor", 1.0)),
            min=max(float(min_spacing) * float(getattr(args, "ct_reseed_sigma_init_floor_ratio", 0.5)), 1e-6),
            max=float(getattr(args, "ct_bulk_scale_global_max", getattr(args, "ct_bulk_max_scale", 1.5))),
        )
        isotropic_scales = sigma_init.unsqueeze(1).expand(new_count, 3).to(dtype=dtype)
        contained_scales, contained_keep = _enforce_bulk_sdf_containment(
            new_xyz,
            isotropic_scales,
            new_rotation_mats.to(dtype=dtype),
            signed_distance_field,
            spacing_zyx,
            margin=float(getattr(args, "ct_bulk_sdf_containment_margin", 0.0)),
            min_scale=max(float(min_spacing) * 0.05, 1e-6),
        )
        if not torch.any(contained_keep):
            return stats
        new_xyz = new_xyz[contained_keep]
        new_rotations = new_rotations[contained_keep]
        new_scaling = torch.log(contained_scales[contained_keep].to(dtype=dtype).clamp_min(1e-8))
        new_normals = normals[contained_keep]
        kept_sigma_init = sigma_init[contained_keep]
        new_count = int(new_xyz.shape[0])
        if new_count <= 0:
            return stats

        new_opacity = gaussians.inverse_opacity_activation(
            torch.full(
                (new_count, 1),
                min(max(float(getattr(args, "ct_bulk_reseed_initial_opacity", 0.65)), 1e-4), 1.0 - 1e-4),
                dtype=dtype,
                device=device,
            )
        )
        feature_dc_shape = (new_count,) + tuple(int(v) for v in gaussians._features_dc.shape[1:])
        new_features_dc = torch.full(feature_dc_shape, 0.5, dtype=gaussians._features_dc.dtype, device=device)
        feature_rest_shape = (new_count,) + tuple(int(v) for v in gaussians._features_rest.shape[1:])
        new_features_rest = torch.zeros(feature_rest_shape, dtype=gaussians._features_rest.dtype, device=device)
        primitive_value = float(getattr(gaussians, "nonplanar_logit_value", -8.0))
        new_primitive_type = torch.full((new_count, 1), primitive_value, dtype=dtype, device=device)
        new_material_id = torch.ones((new_count, 1), dtype=torch.long, device=device)
        new_planarity = torch.zeros((new_count, 1), dtype=torch.float32, device=device)
        new_region_type = torch.ones((new_count, 1), dtype=torch.long, device=device)

        sampled = sample_volume_field(volume_field, new_xyz, spacing_zyx).reshape(-1).to(dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
        new_ct_value_logit = None
        if gaussians._ct_value_logit.numel() > 0:
            new_ct_value_logit = torch.log(sampled / (1.0 - sampled)).reshape(new_count, 1).to(
                dtype=gaussians._ct_value_logit.dtype,
                device=device,
            )
        new_atten_logit = None
        sampled_boosted = None
        if gaussians._atten_logit.numel() > 0:
            boost = float(getattr(args, "ct_reseed_atten_init_boost", 2.0))
            sampled_boosted = (sampled * boost).clamp(1e-6, 50.0)
            new_atten_logit = gaussians._inverse_softplus(sampled_boosted.reshape(new_count, 1)).to(
                dtype=gaussians._atten_logit.dtype,
                device=device,
            )

        gaussians.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotations,
            new_primitive_type,
            new_normals,
            new_material_id,
            new_planarity,
            new_region_type,
            new_ct_value_logit=new_ct_value_logit,
            new_atten_logit=new_atten_logit,
        )
        stats["added"] = new_count
        stats["sigma_init_mean"] = float(kept_sigma_init.mean().item()) if kept_sigma_init.numel() > 0 else float("nan")
        if sampled_boosted is not None and sampled_boosted.numel() > 0:
            stats["atten_init_mean"] = float(sampled_boosted.mean().item())
        stats["count_after"] = int(gaussians.get_xyz.shape[0])
    return stats


def _sample_gap_reseed_candidates(
    support_mask_t: torch.Tensor,
    field_pools: dict | None,
    sample_count: int,
    gap_ratio: float,
    spacing_zyx,
    device,
    *,
    use_boundary_subvoxel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_count = int(sample_count)
    if sample_count <= 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device), torch.empty((0,), dtype=torch.float32, device=device)
    gap_count = int(round(sample_count * min(max(float(gap_ratio), 0.0), 1.0)))
    normal_count = max(0, sample_count - gap_count)
    cavity_count = gap_count // 2
    boundary_count = gap_count - cavity_count
    points = []
    scores = []

    support_indices = torch.nonzero(support_mask_t.reshape(-1), as_tuple=False).reshape(-1)
    if support_indices.numel() > 0 and normal_count > 0:
        chosen = support_indices[torch.randint(0, int(support_indices.numel()), (normal_count,), device=device)]
        vol_shape = support_mask_t.shape
        iz = chosen // (vol_shape[1] * vol_shape[2])
        iy = (chosen % (vol_shape[1] * vol_shape[2])) // vol_shape[2]
        ix = chosen % vol_shape[2]
        sp = torch.as_tensor(spacing_zyx, device=device, dtype=torch.float32)
        normal_points = torch.stack(
            (
                (ix.to(torch.float32) + 0.5) * sp[2],
                (iy.to(torch.float32) + 0.5) * sp[1],
                (iz.to(torch.float32) + 0.5) * sp[0],
            ),
            dim=1,
        )
        points.append(normal_points)
        scores.append(torch.zeros((normal_points.shape[0],), dtype=torch.float32, device=device))

    field_pools = field_pools or {}
    cavity_candidates = field_pools.get("cavity_material_shell")
    if cavity_count > 0 and _candidate_count(cavity_candidates) > 0:
        cavity_points = _sample_occupancy_points(cavity_candidates, cavity_count, spacing_zyx, device=device)
        points.append(cavity_points)
        scores.append(torch.full((cavity_points.shape[0],), 0.75, dtype=torch.float32, device=device))

    boundary_points = None
    if use_boundary_subvoxel:
        boundary_points = field_pools.get("boundary_material_subvoxel_points")
    if _candidate_count(boundary_points) > 0 and boundary_count > 0:
        boundary_points = torch.as_tensor(boundary_points, dtype=torch.float32, device=device).reshape(-1, 3)
        selected = torch.randint(0, int(boundary_points.shape[0]), (boundary_count,), device=device)
        selected_points = boundary_points.index_select(0, selected)
        points.append(selected_points)
        scores.append(torch.full((selected_points.shape[0],), 1.0, dtype=torch.float32, device=device))
    else:
        boundary_candidates = field_pools.get("support_boundary", field_pools.get("boundary_pool"))
        if boundary_count > 0 and _candidate_count(boundary_candidates) > 0:
            selected_points = _sample_occupancy_points(boundary_candidates, boundary_count, spacing_zyx, device=device)
            points.append(selected_points)
            scores.append(torch.full((selected_points.shape[0],), 0.5, dtype=torch.float32, device=device))

    if not points:
        return torch.empty((0, 3), dtype=torch.float32, device=device), torch.empty((0,), dtype=torch.float32, device=device)
    return torch.cat(points, dim=0), torch.cat(scores, dim=0)


def _apply_gap_aware_bulk_reseeding(
    gaussians,
    args,
    training_state,
    spacing_zyx,
    analysis,
    initial_gaussian_count: int,
    *,
    volume_field,
    signed_distance_field,
    field_pools=None,
    iteration: int = 0,
):
    if bool(getattr(args, "ct_budgeted_component_repair", True)):
        return _apply_budgeted_component_bulk_repair(
            gaussians,
            args,
            training_state,
            spacing_zyx,
            analysis,
            initial_gaussian_count,
            volume_field=volume_field,
            signed_distance_field=signed_distance_field,
            iteration=iteration,
        )
    stats = _bulk_reseed_stats(gaussians)
    if gaussians.optimizer is None or not getattr(gaussians, "is_initialized", lambda: False)():
        return stats
    if signed_distance_field is None or volume_field is None:
        return stats

    with torch.no_grad():
        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        count_before = int(gaussians.get_xyz.shape[0])
        stats["count_before"] = count_before
        stats["bulk_grown_count"] = _apply_material_limited_bulk_growth(gaussians, args, signed_distance_field)

        max_ratio = float(getattr(args, "ct_bulk_reseed_max_gaussian_ratio", 2.5))
        max_count = int(math.floor(max(1, initial_gaussian_count) * max_ratio))
        remaining_budget = max(0, max_count - int(gaussians.get_xyz.shape[0]))
        configured_max_new = int(getattr(args, "ct_gap_reseed_max_per_iter", 0))
        if configured_max_new > 0:
            per_iter_limit = configured_max_new
        else:
            configured_bulk_max = int(getattr(args, "ct_bulk_reseed_max_per_iter", 0))
            if configured_bulk_max > 0:
                per_iter_limit = configured_bulk_max
            else:
                bulk_count = int(_bulk_mask_tensor(gaussians).sum().item())
                per_iter_limit = max(1, int(math.ceil(max(1, bulk_count) * max(float(getattr(args, "ct_bulk_reseed_max_new_fraction", 0.03)), 0.0))))
        max_new = min(remaining_budget, per_iter_limit)
        if max_new <= 0:
            return stats

        support_mask = analysis.get("material_mask")
        if support_mask is None:
            return stats
        support_mask_t = as_device_tensor(support_mask, device=device, dtype=torch.bool)
        if not torch.any(support_mask_t):
            return stats

        candidate_xyz, priority = _sample_gap_reseed_candidates(
            support_mask_t,
            field_pools,
            int(getattr(args, "ct_bulk_reseed_sample_count", 8192)),
            float(getattr(args, "ct_gap_reseed_sample_ratio", 0.6)),
            spacing_zyx,
            device,
            use_boundary_subvoxel=bool(getattr(args, "ct_gap_reseed_boundary_subvoxel", False)),
        )
        if candidate_xyz.numel() == 0:
            return stats

        candidate_xyz = candidate_xyz.to(device=device, dtype=dtype)
        center_inside = _sample_binary_mask_nearest(support_mask_t, candidate_xyz, spacing_zyx)
        signed_distance = sample_volume_field(
            signed_distance_field["signed_distance"],
            candidate_xyz,
            signed_distance_field["spacing_zyx"],
        ).reshape(-1).to(device=device, dtype=torch.float32)
        inside_distance = torch.relu(-signed_distance)
        bulk_occ = density_to_occupancy(
            query_ct_density_from_state_by_region(training_state, candidate_xyz, region="bulk", detach=True)
        ).to(dtype=torch.float32)
        den_target = float(getattr(args, "ct_gap_reseed_den_target", 0.9))
        low_coverage = bulk_occ < den_target
        target = sample_volume_field(volume_field, candidate_xyz, spacing_zyx).reshape(-1).to(dtype=torch.float32)
        valid = center_inside & torch.isfinite(signed_distance) & (signed_distance < 0.0) & low_coverage & torch.isfinite(target)
        stats["candidates"] = int(candidate_xyz.shape[0])
        stats["low_coverage_ratio"] = float(low_coverage.float().mean().item()) if bulk_occ.numel() > 0 else 0.0
        if not torch.any(valid):
            return stats

        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        score = (den_target - bulk_occ[valid_indices]).clamp_min(0.0) + priority[valid_indices].to(dtype=torch.float32)
        if valid_indices.numel() > max_new:
            _, order = torch.topk(score, k=max_new, largest=True, sorted=False)
            valid_indices = valid_indices[order]

        new_xyz = candidate_xyz[valid_indices].to(dtype=dtype)
        kept_inside_distance = inside_distance[valid_indices]
        seed_radius = float(getattr(args, "ct_gap_reseed_radius_vox", 1.0)) * float(min(spacing_zyx))
        allowed_radius = 0.85 * kept_inside_distance
        sigma_init = torch.minimum(torch.full_like(allowed_radius, seed_radius), allowed_radius)
        min_scale = max(float(min(spacing_zyx)) * 0.05, 1e-6)
        keep = torch.isfinite(sigma_init) & (sigma_init >= min_scale)
        if not torch.any(keep):
            return stats
        new_xyz = new_xyz[keep]
        sigma_init = sigma_init[keep]
        new_count = int(new_xyz.shape[0])
        normals = _sample_sdf_normals_for_reseed(new_xyz, signed_distance_field).to(dtype=dtype)
        rotation_mats = _frames_from_surface_normals(normals)
        new_rotations = matrix_to_quaternion(rotation_mats).to(dtype=dtype)
        new_scaling = torch.log(sigma_init.reshape(-1, 1).expand(new_count, 3).to(dtype=dtype).clamp_min(1e-8))
        new_opacity = gaussians.inverse_opacity_activation(
            torch.full(
                (new_count, 1),
                min(max(float(getattr(args, "ct_bulk_reseed_initial_opacity", 0.65)), 1e-4), 1.0 - 1e-4),
                dtype=dtype,
                device=device,
            )
        )
        feature_dc_shape = (new_count,) + tuple(int(v) for v in gaussians._features_dc.shape[1:])
        new_features_dc = torch.full(feature_dc_shape, 0.5, dtype=gaussians._features_dc.dtype, device=device)
        feature_rest_shape = (new_count,) + tuple(int(v) for v in gaussians._features_rest.shape[1:])
        new_features_rest = torch.zeros(feature_rest_shape, dtype=gaussians._features_rest.dtype, device=device)
        primitive_value = float(getattr(gaussians, "nonplanar_logit_value", -8.0))
        new_primitive_type = torch.full((new_count, 1), primitive_value, dtype=dtype, device=device)
        new_material_id = torch.ones((new_count, 1), dtype=torch.long, device=device)
        new_planarity = torch.zeros((new_count, 1), dtype=torch.float32, device=device)
        new_region_type = torch.ones((new_count, 1), dtype=torch.long, device=device)
        sampled = sample_volume_field(volume_field, new_xyz, spacing_zyx).reshape(-1).to(dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
        new_ct_value_logit = None
        if gaussians._ct_value_logit.numel() > 0:
            new_ct_value_logit = torch.log(sampled / (1.0 - sampled)).reshape(new_count, 1).to(
                dtype=gaussians._ct_value_logit.dtype,
                device=device,
            )
        new_atten_logit = None
        if gaussians._atten_logit.numel() > 0:
            new_atten_logit = gaussians._inverse_softplus(sampled.reshape(new_count, 1).clamp_min(1e-6)).to(
                dtype=gaussians._atten_logit.dtype,
                device=device,
            )
        previous_birth = _ensure_gap_seed_birth_iter(gaussians).clone()
        gaussians.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotations,
            new_primitive_type,
            normals,
            new_material_id,
            new_planarity,
            new_region_type,
            new_ct_value_logit=new_ct_value_logit,
            new_atten_logit=new_atten_logit,
        )
        _append_gap_seed_birth_iter(gaussians, previous_birth, new_count, int(iteration), gap_seed=True)
        stats["added"] = new_count
        stats["sigma_init_mean"] = float(sigma_init.mean().item()) if sigma_init.numel() > 0 else float("nan")
        stats["atten_init_mean"] = float(sampled.mean().item()) if sampled.numel() > 0 else float("nan")
        stats["count_after"] = int(gaussians.get_xyz.shape[0])
    return stats


def _apply_bulk_coverage_reseeding(
    gaussians,
    args,
    training_state,
    spacing_zyx,
    analysis,
    support_distance_field,
    initial_gaussian_count: int,
    volume_field=None,
    signed_distance_field=None,
    field_pools=None,
    iteration: int = 0,
):
    if bool(getattr(args, "ct_material_coverage_completion", False)):
        if not bool(getattr(args, "ct_completion_repair", True)):
            return _bulk_reseed_stats(gaussians)
        return _apply_material_coverage_completion(
            gaussians,
            args,
            training_state,
            spacing_zyx,
            analysis,
            volume_field=volume_field,
            signed_distance_field=signed_distance_field,
            initial_gaussian_count=initial_gaussian_count,
            iteration=iteration,
        )
    if bool(getattr(args, "ct_gap_aware_reseed", False)):
        return _apply_gap_aware_bulk_reseeding(
            gaussians,
            args,
            training_state,
            spacing_zyx,
            analysis,
            initial_gaussian_count,
            volume_field=volume_field,
            signed_distance_field=signed_distance_field,
            field_pools=field_pools,
            iteration=iteration,
        )
    if is_bulk_intensity_field_mode(getattr(args, "ct_bulk_field_mode", "bulk_intensity_field")):
        return _apply_profile_integrated_bulk_reseeding(
            gaussians,
            args,
            training_state,
            spacing_zyx,
            analysis,
            initial_gaussian_count,
            volume_field=volume_field,
            signed_distance_field=signed_distance_field,
        )
    stats = _bulk_reseed_stats(gaussians)
    if gaussians.optimizer is None or not getattr(gaussians, "is_initialized", lambda: False)():
        return stats

    with torch.no_grad():
        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        count_before = int(gaussians.get_xyz.shape[0])
        stats["count_before"] = count_before

        max_ratio = float(getattr(args, "ct_bulk_reseed_max_gaussian_ratio", 2.0))
        max_count = int(math.floor(max(1, initial_gaussian_count) * max_ratio))
        remaining_budget = max(0, max_count - count_before)
        configured_max_new = int(getattr(args, "ct_bulk_reseed_max_per_iter", 0))
        if configured_max_new > 0:
            per_iter_limit = configured_max_new
        else:
            fraction = float(getattr(args, "ct_bulk_reseed_max_new_fraction", 0.03))
            bulk_count = int((gaussians.get_region_type.reshape(-1) == 1).sum().item())
            per_iter_limit = max(1, int(math.ceil(max(1, bulk_count) * max(fraction, 0.0))))
        max_new = min(remaining_budget, per_iter_limit)
        if max_new <= 0:
            return stats

        support_mask = analysis.get("material_mask")
        if support_mask is None:
            return stats
        if isinstance(support_mask, torch.Tensor):
            support_mask_t = support_mask.to(device=device, dtype=torch.bool)
        else:
            import numpy as np
            support_mask_t = torch.as_tensor(np.asarray(support_mask, dtype=bool), device=device)

        sample_count = int(getattr(args, "ct_bulk_reseed_sample_count", 8192))
        interior_indices = torch.nonzero(support_mask_t.reshape(-1), as_tuple=False).reshape(-1)
        if interior_indices.numel() == 0:
            return stats
        chosen = interior_indices[torch.randint(0, int(interior_indices.numel()), (sample_count,), device=device)]

        vol_shape = support_mask_t.shape
        iz = chosen // (vol_shape[1] * vol_shape[2])
        iy = (chosen % (vol_shape[1] * vol_shape[2])) // vol_shape[2]
        ix = chosen % vol_shape[2]
        sp = torch.as_tensor(spacing_zyx, device=device, dtype=dtype)
        candidate_xyz = torch.stack([
            ix.to(dtype) * sp[2],
            iy.to(dtype) * sp[1],
            iz.to(dtype) * sp[0],
        ], dim=1)

        edt_dist = sample_volume_field(
            support_distance_field["support_distance"],
            candidate_xyz,
            support_distance_field["spacing_zyx"],
        ).reshape(-1)
        material_side = torch.isfinite(edt_dist) & (edt_dist >= 0.0)
        if signed_distance_field is not None:
            candidate_sdf = sample_volume_field(
                signed_distance_field["signed_distance"],
                candidate_xyz,
                signed_distance_field["spacing_zyx"],
            ).reshape(-1).to(device=device, dtype=torch.float32)
            min_sdf_margin = min(float(getattr(args, "ct_bulk_reseed_min_sdf_margin", -0.2)), -1e-6)
            material_side = material_side & torch.isfinite(candidate_sdf) & (candidate_sdf < float(min_sdf_margin))

        bulk_occ = density_to_occupancy(
            query_ct_density_from_state_by_region(training_state, candidate_xyz, region="bulk", detach=True)
        ).to(dtype=torch.float32)

        occ_threshold = float(getattr(args, "ct_bulk_reseed_occupancy_threshold", 0.6))
        low_coverage = bulk_occ < occ_threshold

        valid = material_side & low_coverage
        stats["candidates"] = int(sample_count)
        stats["low_coverage_ratio"] = float(low_coverage.float().mean().item())

        if not torch.any(valid):
            return stats

        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        score = occ_threshold - bulk_occ[valid_indices]
        if valid_indices.numel() > max_new:
            _, order = torch.topk(score, k=max_new, largest=True, sorted=False)
            valid_indices = valid_indices[order]

        new_xyz = candidate_xyz[valid_indices].to(dtype=dtype)
        new_count = int(new_xyz.shape[0])
        if new_count <= 0:
            return stats

        bulk_max_scale = float(getattr(args, "ct_bulk_max_scale", 1.5))
        reseed_scale = min(bulk_max_scale, max(float(max(spacing_zyx)) * 0.75, float(min(spacing_zyx)) * 0.05))
        if signed_distance_field is not None:
            normals = _sample_sdf_normals_for_reseed(new_xyz, signed_distance_field).to(dtype=dtype)
            new_rotation_mats = _frames_from_surface_normals(normals)
            new_rotations = matrix_to_quaternion(new_rotation_mats).to(dtype=dtype)
            signed_distance = sample_volume_field(
                signed_distance_field["signed_distance"],
                new_xyz,
                signed_distance_field["spacing_zyx"],
            ).reshape(-1).to(device=device, dtype=torch.float32)
            inside_distance = torch.relu(-signed_distance)
            depth_scale = torch.clamp(
                inside_distance * 0.5,
                min=max(float(min(spacing_zyx)) * 0.05, 1e-6),
                max=float(reseed_scale),
            )
            normal_scale = torch.minimum(
                torch.full_like(inside_distance, float(reseed_scale)),
                depth_scale,
            )
            tangent_scale = torch.minimum(torch.full_like(normal_scale, float(reseed_scale)), depth_scale)
            new_scaling = torch.log(
                torch.stack((tangent_scale, tangent_scale, normal_scale), dim=1).to(dtype=dtype).clamp_min(1e-8)
            )
            contained_scales, contained_keep = _enforce_bulk_sdf_containment(
                new_xyz,
                torch.exp(new_scaling),
                new_rotation_mats.to(dtype=dtype),
                signed_distance_field,
                spacing_zyx,
                margin=float(getattr(args, "ct_bulk_sdf_containment_margin", 0.0)),
                min_scale=max(float(min(spacing_zyx)) * 0.05, 1e-6),
            )
            if not torch.any(contained_keep):
                return stats
            new_xyz = new_xyz[contained_keep]
            normals = normals[contained_keep]
            new_rotations = new_rotations[contained_keep]
            new_scaling = torch.log(contained_scales[contained_keep].to(dtype=dtype).clamp_min(1e-8))
            new_count = int(new_xyz.shape[0])
            new_normals = normals
        else:
            scale_per_gaussian = torch.full(
                (new_count,),
                max(reseed_scale, 1e-6),
                dtype=torch.float32,
                device=device,
            )
            new_scaling = torch.log(
                scale_per_gaussian.unsqueeze(-1).expand(new_count, 3).to(dtype=dtype)
            )
            new_rotations = torch.zeros((new_count, 4), dtype=dtype, device=device)
            new_rotations[:, 0] = 1.0
            new_normals = torch.zeros((new_count, 3), dtype=dtype, device=device)
            new_normals[:, 0] = 1.0

        new_opacity = gaussians.inverse_opacity_activation(
            torch.full(
                (new_count, 1),
                min(max(float(getattr(args, "ct_bulk_reseed_initial_opacity", 0.65)), 1e-4), 1.0 - 1e-4),
                dtype=dtype,
                device=device,
            )
        )

        feature_dc_shape = (new_count,) + tuple(int(v) for v in gaussians._features_dc.shape[1:])
        new_features_dc = torch.full(feature_dc_shape, 0.5, dtype=gaussians._features_dc.dtype, device=device)
        feature_rest_shape = (new_count,) + tuple(int(v) for v in gaussians._features_rest.shape[1:])
        new_features_rest = torch.zeros(feature_rest_shape, dtype=gaussians._features_rest.dtype, device=device)

        primitive_value = float(getattr(gaussians, "nonplanar_logit_value", -8.0))
        new_primitive_type = torch.full((new_count, 1), primitive_value, dtype=dtype, device=device)
        new_material_id = torch.ones((new_count, 1), dtype=torch.long, device=device)
        new_planarity = torch.zeros((new_count, 1), dtype=torch.float32, device=device)
        new_region_type = torch.ones((new_count, 1), dtype=torch.long, device=device)

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
            new_rotations,
            new_primitive_type,
            new_normals,
            new_material_id,
            new_planarity,
            new_region_type,
            new_ct_value_logit=new_ct_value_logit,
            new_atten_logit=new_atten_logit,
        )
        stats["added"] = new_count
        stats["count_after"] = int(gaussians.get_xyz.shape[0])
    return stats


__all__ = [
    "_append_budgeted_repair_seeds",
    "_append_bulk_completion_seeds",
    "_apply_budgeted_component_bulk_repair",
    "_apply_bulk_coverage_reseeding",
    "_apply_gap_aware_bulk_reseeding",
    "_apply_material_coverage_completion",
    "_apply_material_limited_bulk_growth",
    "_apply_profile_integrated_bulk_reseeding",
    "_budgeted_repair_components",
    "_bulk_reseed_stats",
    "_clearance_cap_candidate_scales",
    "_component_world_points",
    "_directional_clearance_one_side_torch",
    "_enforce_bulk_sdf_containment",
    "_gaussian_density_np",
    "_mask_to_numpy_bool",
    "_material_completion_components",
    "_probe_candidate_inside_material",
    "_sample_gap_reseed_candidates",
    "_sdf_inside_np",
    "_sdf_to_numpy",
    "_voxel_indices_to_world",
]
