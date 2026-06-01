from __future__ import annotations

import math

import torch

from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.mutations.helpers import (
    _ct_support_volume_from_analysis,
    _ct_topk_indices,
    _project_xyz_to_sdf_zero,
)
from ct_pipeline.training.utils import as_device_tensor
from utils.rotation_utils import quaternion_to_matrix


def _apply_ct_densification(
    gaussians,
    args,
    xyz_grad_norm,
    support_distance_field,
    spacing_zyx,
    analysis,
    initial_gaussian_count: int,
    signed_distance_field=None,
    curvature_field=None,
):
    del support_distance_field
    stats = {
        "surface_split": 0,
        "bulk_split": 0,
        "parents_pruned": 0,
        "children_added": 0,
        "net_added": 0,
        "count_before": int(gaussians.get_xyz.shape[0]),
        "count_after": int(gaussians.get_xyz.shape[0]),
    }
    if not bool(getattr(args, "ct_enable_densification", False)):
        return stats
    if gaussians.optimizer is None or not getattr(gaussians, "is_initialized", lambda: False)():
        return stats

    with torch.no_grad():
        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        count_before = int(gaussians.get_xyz.shape[0])
        max_count = int(math.floor(max(1, int(initial_gaussian_count)) * float(args.ct_densify_max_gaussian_ratio)))
        remaining_budget = max(0, max_count - count_before)
        if remaining_budget <= 0:
            return stats

        xyz = gaussians.get_xyz.detach()
        region_type = gaussians.get_region_type.reshape(-1).detach()
        raw_scaling = gaussians.get_raw_scaling.detach()
        raw_scale_values = torch.exp(raw_scaling).clamp_min(1e-8)
        scales = gaussians.get_scaling.detach().clamp_min(1e-8)
        opacity = gaussians.get_opacity.detach().reshape(-1).clamp(1e-6, 1.0 - 1e-6)
        rotations = quaternion_to_matrix(gaussians.get_rotation.detach())
        if xyz_grad_norm is None:
            grad_norm = torch.zeros((count_before,), dtype=dtype, device=device)
        else:
            grad_norm = torch.zeros((count_before,), dtype=dtype, device=device)
            source_grad = as_device_tensor(xyz_grad_norm, device=device, dtype=dtype).reshape(-1)
            grad_norm[: min(count_before, source_grad.shape[0])] = source_grad[:count_before]

        new_xyz_parts = []
        new_scaling_parts = []
        new_opacity_parts = []
        source_index_parts = []
        parent_index_parts = []

        surface_mask = region_type == 0
        surface_count = int(surface_mask.sum().item())
        surface_budget = min(int(math.ceil(surface_count * float(args.ct_densify_surface_percent))), remaining_budget)
        surface_split_count = 0
        if surface_budget > 0 and surface_count > 0:
            tangent_scales = scales[:, :2]
            tangential_rms = torch.sqrt(torch.clamp(0.5 * torch.sum(tangent_scales * tangent_scales, dim=1), min=1e-8))
            tangent_threshold = max(float(args.ct_densify_surface_tangent_ratio) * float(max(spacing_zyx)), 1e-8)
            surface_scores = grad_norm * (1.0 + tangential_rms / tangent_threshold)
            surface_candidates = (
                surface_mask
                & (opacity >= float(args.ct_densify_min_opacity))
                & (tangential_rms >= tangent_threshold)
                & (grad_norm > 0.0)
            )
            surface_indices = _ct_topk_indices(surface_candidates, surface_scores, surface_budget)
            surface_split_count = int(surface_indices.numel())
            if surface_split_count > 0:
                selected_rotations = rotations[surface_indices]
                selected_tangent_scales = tangent_scales[surface_indices]
                dominant_axis = torch.argmax(selected_tangent_scales, dim=1)
                tangent_dirs = selected_rotations.gather(2, dominant_axis.reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(2)
                max_tangent_scale = selected_tangent_scales.max(dim=1).values
                offset = tangent_dirs * (0.35 * max_tangent_scale).unsqueeze(1)
                plus_xyz = xyz[surface_indices] + offset
                minus_xyz = xyz[surface_indices] - offset
                if signed_distance_field is not None:
                    plus_xyz = _project_xyz_to_sdf_zero(plus_xyz, signed_distance_field)
                    minus_xyz = _project_xyz_to_sdf_zero(minus_xyz, signed_distance_field)
                new_xyz_parts.append(torch.cat((plus_xyz, minus_xyz), dim=0))

                child_scales = raw_scale_values[surface_indices].clone()
                row_ids = torch.arange(surface_split_count, device=device)
                child_scales[row_ids, dominant_axis] *= 0.6
                child_scales[row_ids, 1 - dominant_axis] *= 0.9
                child_scales = child_scales.clamp_min(1e-8)
                new_scaling_parts.append(torch.log(torch.cat((child_scales, child_scales), dim=0)))

                child_opacity = (opacity[surface_indices] * 0.5).clamp(1e-6, 1.0 - 1e-6).reshape(-1, 1)
                child_opacity = gaussians.inverse_opacity_activation(child_opacity)
                new_opacity_parts.append(torch.cat((child_opacity, child_opacity), dim=0))
                source_index_parts.append(torch.cat((surface_indices, surface_indices), dim=0))
                parent_index_parts.append(surface_indices)
                remaining_budget -= surface_split_count

        bulk_mask = region_type == 1
        bulk_count = int(bulk_mask.sum().item())
        bulk_budget = min(int(math.ceil(bulk_count * float(args.ct_densify_bulk_percent))), remaining_budget)
        bulk_split_count = 0
        if bulk_budget > 0 and bulk_count > 0:
            max_scale = scales.max(dim=1).values
            bulk_split_threshold = max(float(args.ct_densify_bulk_scale_ratio) * float(max(spacing_zyx)), 1e-8)
            bulk_ratio = max_scale / bulk_split_threshold
            bulk_candidates = (
                bulk_mask
                & (opacity >= float(args.ct_densify_min_opacity))
                & (max_scale >= bulk_split_threshold)
            )
            bulk_indices = _ct_topk_indices(bulk_candidates, bulk_ratio, bulk_budget)
            if bulk_indices.numel() > 0:
                selected_scales = raw_scale_values[bulk_indices]
                dominant_axis = torch.argmax(selected_scales, dim=1)
                selected_rotations = rotations[bulk_indices]
                split_dirs = selected_rotations.gather(2, dominant_axis.reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(2)
                max_bulk_scale = selected_scales.max(dim=1).values
                offset = split_dirs * (0.35 * max_bulk_scale).unsqueeze(1)
                bulk_plus = xyz[bulk_indices] + offset
                bulk_minus = xyz[bulk_indices] - offset
                support_volume = _ct_support_volume_from_analysis(analysis, device=device, dtype=dtype)
                if support_volume is not None:
                    support_values = sample_volume_field(
                        support_volume,
                        torch.stack((bulk_plus, bulk_minus), dim=1).reshape(-1, 3),
                        spacing_zyx,
                    ).reshape(-1, 2)
                    valid_bulk = torch.all(support_values >= 0.5, dim=1)
                    bulk_indices = bulk_indices[valid_bulk]
                    bulk_plus = bulk_plus[valid_bulk]
                    bulk_minus = bulk_minus[valid_bulk]
                    selected_scales = selected_scales[valid_bulk]
                bulk_split_count = int(bulk_indices.numel())
                if bulk_split_count > 0:
                    new_xyz_parts.append(torch.cat((bulk_plus, bulk_minus), dim=0))
                    child_scales = (selected_scales * 0.7).clamp_min(1e-8)
                    new_scaling_parts.append(torch.log(torch.cat((child_scales, child_scales), dim=0)))
                    child_opacity = (opacity[bulk_indices] * 0.5).clamp(1e-6, 1.0 - 1e-6).reshape(-1, 1)
                    child_opacity = gaussians.inverse_opacity_activation(child_opacity)
                    new_opacity_parts.append(torch.cat((child_opacity, child_opacity), dim=0))
                    source_index_parts.append(torch.cat((bulk_indices, bulk_indices), dim=0))
                    parent_index_parts.append(bulk_indices)

        if not source_index_parts:
            return stats

        child_source_indices = torch.cat(source_index_parts, dim=0)
        parent_indices = torch.cat(parent_index_parts, dim=0).unique()
        new_xyz = torch.cat(new_xyz_parts, dim=0)
        new_scaling = torch.cat(new_scaling_parts, dim=0)
        new_opacity = torch.cat(new_opacity_parts, dim=0)
        new_features_dc = gaussians._features_dc.detach()[child_source_indices].clone()
        new_features_rest = gaussians._features_rest.detach()[child_source_indices].clone()
        new_rotation = gaussians._rotation.detach()[child_source_indices].clone()
        new_primitive_type = gaussians._primitive_type.detach()[child_source_indices].clone()
        new_normal = gaussians._normal.detach()[child_source_indices].clone()
        new_material_id = gaussians._material_id.detach()[child_source_indices].clone()
        new_planarity = gaussians._planarity.detach()[child_source_indices].clone()
        new_region_type = gaussians._region_type.detach()[child_source_indices].clone()
        # Children inherit parent's ct_value_logit (per user spec).
        new_ct_value_logit = (
            gaussians._ct_value_logit.detach()[child_source_indices].clone()
            if gaussians._ct_value_logit.numel() > 0
            else None
        )
        new_atten_logit = (
            gaussians._atten_logit.detach()[child_source_indices].clone()
            if gaussians._atten_logit.numel() > 0
            else None
        )

        gaussians.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_primitive_type,
            new_normal,
            new_material_id,
            new_planarity,
            new_region_type,
            new_ct_value_logit=new_ct_value_logit,
            new_atten_logit=new_atten_logit,
        )

        prune_mask = torch.zeros((count_before + new_xyz.shape[0],), dtype=torch.bool, device=device)
        prune_mask[parent_indices] = True
        gaussians.prune_points(prune_mask)
        count_after = int(gaussians.get_xyz.shape[0])
        stats.update(
            {
                "surface_split": surface_split_count,
                "bulk_split": bulk_split_count,
                "parents_pruned": int(parent_indices.numel()),
                "children_added": int(new_xyz.shape[0]),
                "net_added": int(count_after - count_before),
                "count_after": count_after,
            }
        )
    return stats
