from __future__ import annotations

from ct_pipeline.training.bootstrap import (
    CTTrainingBootstrap,
    _central_difference_axis_np,
    _ct_spatial_extent,
    _empty_support_distance_field,
    _ensure_intensity_driven_analysis,
    _load_ct_analysis_bundle,
    _prepare_curvature_proxy_field,
    _prepare_intensity_calibration,
    _prepare_signed_distance_field,
    _prepare_support_distance_field,
    _require_active_boundary_bundle,
    _run_ct_init_preflight,
    _sample_coarse_sdf_normals,
    _to_cuda_analysis,
    prepare_ct_training_bootstrap,
)
from ct_pipeline.training.control import (
    _apply_bulk_atten_only_optimizer_mode,
    _attenuation_only_bulk_gate_training_enabled,
    _attenuation_only_preview_early_stop_enabled,
    _attenuation_only_training_enabled,
    _bulk_atten_only_lr_scale,
    _bulk_attenuation_grad_stats,
    _freeze_ct_feature_params,
    _restore_best_bulk_attenuation,
    _sanitize_xyz_parameter,
    _save_ct_gaussians,
    maybe_apply_stage1_freeze,
)
from ct_pipeline.training.objectives import CTLossTerms
from ct_pipeline.training.objectives.modes import (
    _boundary_band_distance,
    _bulk_halfspace_tau_current,
    _bulk_intensity_sample_mode,
    _bulk_intensity_training_enabled,
    _containment_ramp_weight,
    _dual_separated_training_enabled,
    _dual_surface_air_band,
    _dual_surface_inner_band,
    _intensity_geometry_flags,
    _surface_material_render_band,
    _use_unified_compositor,
)

from ct_pipeline.training.objectives.prediction import (
    _bulk_intensity_from_fields,
    _bulk_intensity_prediction,
    compute_raw_combined_ct_occupancy,
    compute_role_separated_ct_prediction,
    compute_surface_bounded_bulk_volume_prediction,
)
from ct_pipeline.training.objectives.sampling import (
    _phase_mask_from_analysis,
    _phase_occupancy_sdf_weights,
    _sample_air_points_with_void_bias,
    _sample_boundary_offset_shell_points,
    _sample_filtered_from_candidate_sets,
    _sample_filtered_semantic_points,
    _sample_support_membership,
    _split_phase_occupancy_sample_counts,
)

import math

import numpy as np
import torch
import torch.nn.functional as F

from ct_pipeline.rendering.slices import _build_query_points_from_base, sample_gt_slice_patch
from ct_pipeline.rendering.fields import (
    _query_ct_density_native_chunked,
    compose_signed_overlap_occupancy,
    density_to_occupancy,
    query_ct_fields_unified,
    query_ct_density_from_state_by_region,
)
from ct_pipeline.training.sampling import (
    _cached_or_filter_candidates,
    _candidate_count,
    _ct_empty_points,
    _sample_occupancy_points,
    _sample_far_phase_points,
    _sample_signed_distance,
    _split_role_sample_counts,
    sample_bulk_volume_points_excluding_boundary,
    sample_surface_boundary_points,
)
from ct_pipeline.training.utils import as_device_tensor
from ct_pipeline.training.losses import (
    asymmetric_binary_focal_loss,
    eagle_patch_loss,
    sample_volume_field,
    surface_sdf_thickness_loss,
)


def unified_phase_occupancy_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    boundary_band_distance: float,
) -> torch.Tensor:
    device = getattr(getattr(training_state, "xyz", None), "device", None)
    if device is None:
        signed_distance = context.signed_distance_field.get("signed_distance")
        device = signed_distance.device if torch.is_tensor(signed_distance) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_count = max(1, int(getattr(args, "ct_occupancy_sample_count", args.ct_volume_sample_count)))
    boundary_count, material_count, air_count = _split_phase_occupancy_sample_counts(args, sample_count)

    parts = []
    if boundary_count > 0:
        boundary_candidates = context.field_pools.get("boundary_pool")
        if _candidate_count(boundary_candidates) == 0:
            boundary_candidates = _cached_or_filter_candidates(
                context.field_pools,
                "roi",
                context.signed_distance_field,
                boundary_band_distance,
                keep_boundary=True,
            )
        if _candidate_count(boundary_candidates) > 0:
            parts.append(_sample_occupancy_points(boundary_candidates, boundary_count, context.spacing_zyx, device=device))

    if material_count > 0:
        material_candidates = context.field_pools.get("material_deep_pool")
        if _candidate_count(material_candidates) == 0:
            material_candidates = _cached_or_filter_candidates(
                context.field_pools,
                "support",
                context.signed_distance_field,
                boundary_band_distance,
                keep_boundary=False,
            )
        if _candidate_count(material_candidates) > 0:
            parts.append(_sample_occupancy_points(material_candidates, material_count, context.spacing_zyx, device=device))

    air_points = _sample_air_points_with_void_bias(context, air_count, boundary_band_distance, device)
    if air_points.numel() > 0:
        parts.append(air_points)

    if not parts:
        return torch.zeros((), dtype=torch.float32, device=device)

    points = torch.cat(parts, dim=0)[:sample_count]
    phase_mask = _phase_mask_from_analysis(context.analysis_gpu)
    targets = _sample_support_membership(phase_mask, points, context.spacing_zyx)
    if targets is None:
        return torch.zeros((), dtype=torch.float32, device=device)

    prediction = compute_raw_combined_ct_occupancy(
        training_state,
        points,
        detach_bulk=False,
        detach_surface=False,
    )
    per_sample_loss = F.smooth_l1_loss(
        prediction,
        targets.to(device=prediction.device, dtype=prediction.dtype),
        beta=float(args.ct_huber_beta),
        reduction="none",
    )
    signed_distance = _sample_signed_distance(context.signed_distance_field, points)
    weights = _phase_occupancy_sdf_weights(
        signed_distance,
        boundary_band_distance,
        boundary_weight=float(getattr(args, "ct_occ_boundary_weight", 3.0)),
    ).to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def _dual_separated_volume_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    iteration: int,
    *,
    boundary_band_distance: float,
    bulk_sample_count: int,
    boundary_sample_count: int,
    volume_field: torch.Tensor,
    intensity_flags: dict,
) -> torch.Tensor:
    device = getattr(training_state.xyz, "device", torch.device("cuda"))
    zero = torch.zeros((), dtype=torch.float32, device=device)
    air_band = _dual_surface_air_band(args)
    material_band = _surface_material_render_band(args, boundary_band_distance)
    surface_material_weight = max(float(getattr(args, "ct_surface_material_render_weight", 0.0)), 0.0)
    air_ratio = min(max(float(getattr(args, "ct_dual_surface_air_sample_ratio", 0.25)), 0.0), 1.0)
    surface_air_count = int(round(float(boundary_sample_count) * air_ratio))
    surface_air_count = max(0, min(int(boundary_sample_count), surface_air_count))
    surface_material_count = int(boundary_sample_count) - surface_air_count if surface_material_weight > 0.0 else 0

    support_candidates = (
        context.field_pools.get("material_deep_pool"),
        context.field_pools.get("support"),
        context.field_pools.get("cavity_material_shell"),
    )
    air_candidates = (
        context.field_pools.get("cavity_air_shell"),
        context.field_pools.get("void_air"),
        context.field_pools.get("near_material_air"),
        context.field_pools.get("air_shell"),
        context.preferred_air_candidates,
    )

    bulk_points, bulk_sdf = _sample_filtered_from_candidate_sets(
        support_candidates,
        int(bulk_sample_count),
        context,
        device=device,
        signed_distance_predicate=lambda sdf: sdf < 0.0,
    )
    if surface_material_count > 0:
        surface_material_points, surface_material_sdf = _sample_boundary_offset_shell_points(
            context,
            surface_material_count,
            offset_min=-float(material_band),
            offset_max=0.0,
            device=device,
            signed_distance_predicate=lambda sdf: (sdf >= -float(material_band)) & (sdf <= 0.0),
        )
        if int(surface_material_points.shape[0]) < surface_material_count:
            extra_points, extra_sdf = _sample_filtered_from_candidate_sets(
                support_candidates,
                surface_material_count - int(surface_material_points.shape[0]),
                context,
                device=device,
                signed_distance_predicate=lambda sdf: (sdf >= -float(material_band)) & (sdf <= 0.0),
            )
            if extra_points.numel() > 0:
                surface_material_points = torch.cat((surface_material_points, extra_points), dim=0)
                surface_material_sdf = torch.cat((surface_material_sdf, extra_sdf), dim=0)
    else:
        surface_material_points = _ct_empty_points(device=device)
        surface_material_sdf = torch.zeros((0,), device=device, dtype=torch.float32)
    surface_air_points, surface_air_sdf = _sample_boundary_offset_shell_points(
        context,
        surface_air_count,
        offset_min=1e-3,
        offset_max=float(air_band),
        device=device,
        signed_distance_predicate=lambda sdf: (sdf > 0.0) & (sdf <= float(air_band)),
        oversample=8,
    )
    if int(surface_air_points.shape[0]) < surface_air_count:
        extra_points, extra_sdf = _sample_filtered_from_candidate_sets(
            air_candidates,
            surface_air_count - int(surface_air_points.shape[0]),
            context,
            device=device,
            signed_distance_predicate=lambda sdf: (sdf > 0.0) & (sdf <= float(air_band)),
            oversample=8,
        )
        if extra_points.numel() > 0:
            surface_air_points = torch.cat((surface_air_points, extra_points), dim=0)
            surface_air_sdf = torch.cat((surface_air_sdf, extra_sdf), dim=0)

    if surface_material_count > 0 and surface_material_points.numel() == 0:
        surface_material_points, surface_material_sdf = _sample_filtered_from_candidate_sets(
            support_candidates,
            surface_material_count,
            context,
            device=device,
            signed_distance_predicate=lambda sdf: (sdf >= -float(material_band)) & (sdf <= 0.0),
        )
    if surface_air_points.numel() == 0:
        surface_air_points, surface_air_sdf = _sample_filtered_from_candidate_sets(
            air_candidates,
            surface_air_count,
            context,
            device=device,
            signed_distance_predicate=lambda sdf: (sdf > 0.0) & (sdf <= float(air_band)),
            oversample=8,
        )

    # Avoid carrying extra samples if both anchor-offset and voxel pools produced enough points.
    surface_material_points = surface_material_points[:surface_material_count]
    surface_material_sdf = surface_material_sdf[:surface_material_count]
    surface_air_points = surface_air_points[:surface_air_count]
    surface_air_sdf = surface_air_sdf[:surface_air_count]

    # Bulk now owns the full material side up to the boundary.
    bulk_points, bulk_sdf = bulk_points[: int(bulk_sample_count)], bulk_sdf[: int(bulk_sample_count)]

    loss_sum = zero
    weight_sum = 0
    intensity_air = float(context.intensity_air)

    def _add_render_loss(points: torch.Tensor, signed_distance: torch.Tensor, field_key: str, target, loss_weight: float = 1.0) -> None:
        nonlocal loss_sum, weight_sum
        if points.numel() == 0:
            return
        fields = query_ct_fields_unified(
            points,
            training_state,
            signed_distance=signed_distance,
            config=args,
            intensity_air=intensity_air,
            include_surface=(field_key == "I_s"),
            bulk_train_opacity=bool(field_key == "I_b" and intensity_flags["bulk_train_opacity"]),
            bulk_train_scale=bool(field_key == "I_b" and intensity_flags["bulk_train_scale"]),
            bulk_scale_grad=float(intensity_flags["bulk_scale_grad"]),
            surface_train_opacity=bool(field_key == "I_s" and intensity_flags["surface_train_opacity"]),
            train_ct_value=True,
            detach_value_geometry=True,
        )
        predicted = fields[field_key].to(dtype=torch.float32)
        if isinstance(target, torch.Tensor):
            target_intensity = target.to(device=predicted.device, dtype=predicted.dtype).reshape(-1)
        else:
            target_intensity = torch.full_like(predicted, float(target))
        losses = F.smooth_l1_loss(
            predicted,
            target_intensity,
            beta=float(args.ct_huber_beta),
            reduction="none",
        )
        loss_sum = loss_sum + float(loss_weight) * losses.sum()
        weight_sum += int(losses.numel())

    if bulk_points.numel() > 0:
        bulk_target = sample_volume_field(volume_field, bulk_points, context.spacing_zyx).reshape(-1).to(dtype=torch.float32)
        _add_render_loss(bulk_points, bulk_sdf, "I_b", bulk_target)
    if surface_material_weight > 0.0 and surface_material_points.numel() > 0:
        surface_target = sample_volume_field(
            volume_field,
            surface_material_points,
            context.spacing_zyx,
        ).reshape(-1).to(dtype=torch.float32)
        _add_render_loss(
            surface_material_points,
            surface_material_sdf,
            "I_s",
            surface_target,
            loss_weight=surface_material_weight,
        )
    _add_render_loss(surface_air_points, surface_air_sdf, "I_s", intensity_air)

    if weight_sum <= 0:
        return zero
    return loss_sum / float(weight_sum)


def _estimate_boundary_miss_rate(
    context: CTTrainingBootstrap,
    args,
    training_state,
    volume_field: torch.Tensor,
) -> float:
    delta = max(float(getattr(args, "ct_loss_boundary_band_delta", 0.5)), 1e-6)
    device = getattr(training_state.xyz, "device", torch.device("cuda"))
    support_candidates = (
        context.field_pools.get("support"),
        context.field_pools.get("cavity_material_shell"),
        context.field_pools.get("material_deep_pool"),
    )
    points, signed_distance = _sample_filtered_from_candidate_sets(
        support_candidates,
        2048,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: (sdf >= -float(delta)) & (sdf <= 0.0),
        oversample=6,
    )
    if points.numel() == 0:
        return float("inf")
    fields = query_ct_fields_unified(
        points,
        training_state,
        signed_distance=signed_distance,
        config=args,
        intensity_air=float(context.intensity_air),
        include_surface=False,
        train_ct_value=False,
    )
    pred = _bulk_intensity_prediction(fields, signed_distance, float(context.intensity_air), args)
    target = sample_volume_field(volume_field, points, context.spacing_zyx).reshape(-1).to(dtype=torch.float32)
    margin = 0.5 * float(getattr(args, "ct_huber_beta", 0.1))
    miss = pred < (target - margin)
    return float(miss.float().mean().item()) if miss.numel() > 0 else float("inf")


def _maybe_update_halfspace_tau(
    context: CTTrainingBootstrap,
    args,
    training_state,
    iteration: int,
    volume_field: torch.Tensor,
) -> None:
    if not _bulk_intensity_training_enabled(args):
        return
    if not bool(getattr(args, "ct_bulk_halfspace_enable", True)):
        return
    interval = max(int(getattr(args, "ct_bulk_halfspace_tau_update_interval", 250)), 1)
    if int(iteration) > 1 and int(iteration) % interval != 0:
        return
    current = _bulk_halfspace_tau_current(args)
    final = float(getattr(args, "ct_bulk_halfspace_tau_final", 0.2))
    if current <= final + 1e-8:
        setattr(args, "ct_bulk_halfspace_tau_current", final)
        return
    miss_rate = _estimate_boundary_miss_rate(context, args, training_state, volume_field)
    setattr(args, "ct_boundary_miss_rate_last", float(miss_rate))
    threshold = float(getattr(args, "ct_bulk_halfspace_tau_gate_threshold", 0.10))
    if np.isfinite(miss_rate) and miss_rate <= threshold:
        step = max(float(getattr(args, "ct_bulk_halfspace_tau_step", 0.05)), 0.0)
        setattr(args, "ct_bulk_halfspace_tau_current", max(final, current - step))


def _bulk_intensity_volume_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    *,
    boundary_band_distance: float,
    bulk_sample_count: int,
    boundary_sample_count: int,
    volume_field: torch.Tensor,
    intensity_flags: dict,
) -> torch.Tensor:
    device = getattr(training_state.xyz, "device", torch.device("cuda"))
    zero = torch.zeros((), dtype=torch.float32, device=device)
    train_bulk_geometry = not _attenuation_only_training_enabled(args)
    train_bulk_gate = _attenuation_only_bulk_gate_training_enabled(args)
    delta = max(float(getattr(args, "ct_loss_boundary_band_delta", 0.5)), 1e-6)
    support_candidates = (
        context.field_pools.get("material_deep_pool"),
        context.field_pools.get("support"),
        context.field_pools.get("cavity_material_shell"),
    )
    air_candidates = (
        context.field_pools.get("cavity_air_shell"),
        context.field_pools.get("void_air"),
        context.field_pools.get("near_material_air"),
        context.field_pools.get("air_shell"),
        context.preferred_air_candidates,
    )
    sample_mode = _bulk_intensity_sample_mode(args)
    if sample_mode == "material_interior_only":
        interior_margin = max(float(getattr(args, "ct_intensity_erode_margin_vox", 0.5)), 0.0)
        support_candidates = (context.field_pools.get("support"),)
        interior_points, interior_sdf = _sample_filtered_from_candidate_sets(
            support_candidates,
            int(bulk_sample_count),
            context,
            device=device,
            signed_distance_predicate=lambda sdf: sdf < -float(interior_margin),
            oversample=6,
        )
        material_band_points = _ct_empty_points(device=device)
        material_band_sdf = torch.empty((0,), dtype=torch.float32, device=device)
        void_band_points = _ct_empty_points(device=device)
        void_band_sdf = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        interior_points, interior_sdf = _sample_filtered_from_candidate_sets(
            support_candidates,
            int(bulk_sample_count),
            context,
            device=device,
            signed_distance_predicate=lambda sdf: sdf < -float(delta),
        )
        material_band_points, material_band_sdf = _sample_filtered_from_candidate_sets(
            support_candidates,
            int(boundary_sample_count),
            context,
            device=device,
            signed_distance_predicate=lambda sdf: (sdf >= -float(delta)) & (sdf <= 0.0),
            oversample=6,
        )
        void_band_points, void_band_sdf = _sample_filtered_from_candidate_sets(
            air_candidates,
            int(boundary_sample_count),
            context,
            device=device,
            signed_distance_predicate=lambda sdf: (sdf >= 0.0) & (sdf <= float(delta)),
            oversample=8,
        )

    loss_sum = zero
    weight_sum = 0
    support_loss = zero
    beta = float(getattr(args, "ct_huber_beta", 0.1))
    band_margin = 0.5 * beta
    den_min = max(float(getattr(args, "ct_intensity_den_min", 0.0)), 0.0) if sample_mode == "material_interior_only" else 0.0

    if interior_points.numel() > 0:
        fields = query_ct_fields_unified(
            interior_points,
            training_state,
            signed_distance=interior_sdf,
            config=args,
            intensity_air=float(context.intensity_air),
            include_surface=False,
            bulk_train_xyz=train_bulk_geometry,
            bulk_train_opacity=train_bulk_gate,
            bulk_train_scale=bool(train_bulk_geometry and intensity_flags.get("bulk_train_scale", False)),
            bulk_scale_grad=float(intensity_flags.get("bulk_scale_grad", 1.0)),
            train_ct_value=False,
        )
        pred = _bulk_intensity_prediction(fields, interior_sdf, float(context.intensity_air), args)
        target = sample_volume_field(volume_field, interior_points, context.spacing_zyx).reshape(-1).to(dtype=torch.float32)
        keep = torch.isfinite(pred) & torch.isfinite(target)
        if den_min > 0.0:
            keep = keep & (fields["den_b"].to(dtype=torch.float32) > float(den_min))
        if torch.any(keep):
            loss_sum = loss_sum + F.smooth_l1_loss(pred[keep], target[keep], beta=beta, reduction="sum")
            weight_sum += int(keep.sum().item())

    if material_band_points.numel() > 0:
        fields = query_ct_fields_unified(
            material_band_points,
            training_state,
            signed_distance=material_band_sdf,
            config=args,
            intensity_air=float(context.intensity_air),
            include_surface=False,
            bulk_train_xyz=train_bulk_geometry,
            bulk_train_opacity=train_bulk_gate,
            bulk_train_scale=bool(train_bulk_geometry and intensity_flags.get("bulk_train_scale", False)),
            bulk_scale_grad=float(intensity_flags.get("bulk_scale_grad", 1.0)),
            train_ct_value=False,
        )
        pred = _bulk_intensity_prediction(fields, material_band_sdf, float(context.intensity_air), args)
        target = sample_volume_field(volume_field, material_band_points, context.spacing_zyx).reshape(-1).to(dtype=torch.float32)
        target_lo = (target - float(band_margin)).clamp_min(0.0)
        loss_sum = loss_sum + float(getattr(args, "ct_loss_band_in_weight", 0.25)) * torch.relu(target_lo - pred).square().sum()
        weight_sum += int(pred.numel())

    if void_band_points.numel() > 0:
        fields = query_ct_fields_unified(
            void_band_points,
            training_state,
            signed_distance=void_band_sdf,
            config=args,
            intensity_air=float(context.intensity_air),
            include_surface=False,
            bulk_train_xyz=train_bulk_geometry,
            bulk_train_opacity=train_bulk_gate,
            bulk_train_scale=bool(train_bulk_geometry and intensity_flags.get("bulk_train_scale", False)),
            bulk_scale_grad=float(intensity_flags.get("bulk_scale_grad", 1.0)),
            train_ct_value=False,
        )
        if train_bulk_gate:
            support_mass = fields["W_b"].to(dtype=torch.float32)
            support_loss = support_loss + float(getattr(args, "ct_bulk_air_exclusion_weight", 1.0)) * support_mass.mean()
        else:
            pred = _bulk_intensity_prediction(fields, void_band_sdf, float(context.intensity_air), args)
            target = sample_volume_field(volume_field, void_band_points, context.spacing_zyx).reshape(-1).to(dtype=torch.float32)
            target_hi = (target + float(band_margin)).clamp_max(1.0)
            loss_sum = loss_sum + float(getattr(args, "ct_loss_band_out_weight", 0.25)) * torch.relu(pred - target_hi).square().sum()
            weight_sum += int(pred.numel())

    if train_bulk_gate:
        air_sample_count = int(getattr(args, "ct_bulk_air_exclusion_sample_count", 0))
        air_weight = float(getattr(args, "ct_bulk_air_exclusion_weight", 0.0))
        if air_weight > 0.0 and air_sample_count > 0:
            air_points, air_sdf = _sample_filtered_from_candidate_sets(
                air_candidates,
                int(air_sample_count),
                context,
                device=device,
                signed_distance_predicate=lambda sdf: sdf > 0.0,
                oversample=8,
            )
            if air_points.numel() > 0:
                air_fields = query_ct_fields_unified(
                    air_points,
                    training_state,
                    signed_distance=air_sdf,
                    config=args,
                    intensity_air=float(context.intensity_air),
                    include_surface=False,
                    bulk_train_xyz=train_bulk_geometry,
                    bulk_train_opacity=True,
                    bulk_train_scale=bool(train_bulk_geometry and intensity_flags.get("bulk_train_scale", False)),
                    bulk_scale_grad=float(intensity_flags.get("bulk_scale_grad", 1.0)),
                    train_ct_value=False,
                )
                support_loss = support_loss + float(air_weight) * air_fields["W_b"].to(dtype=torch.float32).mean()

        sparse_weight = float(getattr(args, "ct_bulk_opacity_sparse_weight", 0.0))
        if sparse_weight > 0.0:
            support_loss = support_loss + float(sparse_weight) * training_state.bulk_opacity.to(dtype=torch.float32).mean()

    if weight_sum <= 0:
        return support_loss
    return loss_sum / float(weight_sum) + support_loss


def _surface_intensity_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    *,
    boundary_band_distance: float,
    volume_field: torch.Tensor,
) -> torch.Tensor:
    weight = float(getattr(args, "ct_surface_intensity_weight", 0.0))
    device = getattr(training_state.xyz, "device", torch.device("cuda"))
    zero = torch.zeros((), dtype=torch.float32, device=device)
    if weight <= 0.0 or training_state.surface_xyz.numel() == 0:
        return zero
    if getattr(training_state, "surface_attenuation", None) is None and getattr(training_state, "surface_ct_value", None) is None:
        return zero

    sample_count = int(getattr(args, "ct_surface_intensity_sample_count", 2048))
    if sample_count <= 0:
        return zero
    material_ratio = min(max(float(getattr(args, "ct_surface_intensity_material_ratio", 0.75)), 0.0), 1.0)
    material_count = int(round(float(sample_count) * material_ratio))
    air_count = max(0, int(sample_count) - material_count)
    delta = max(float(boundary_band_distance), 1e-6)

    support_candidates = (
        context.field_pools.get("support"),
        context.field_pools.get("cavity_material_shell"),
        context.field_pools.get("material_deep_pool"),
    )
    air_candidates = (
        context.field_pools.get("cavity_air_shell"),
        context.field_pools.get("void_air"),
        context.field_pools.get("near_material_air"),
        context.field_pools.get("air_shell"),
        context.preferred_air_candidates,
    )

    material_points, material_sdf = _sample_filtered_from_candidate_sets(
        support_candidates,
        material_count,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: (sdf >= -float(delta)) & (sdf <= 0.0),
        oversample=8,
    )
    air_points, air_sdf = _sample_filtered_from_candidate_sets(
        air_candidates,
        air_count,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: (sdf > 0.0) & (sdf <= float(delta)),
        oversample=8,
    )

    loss_sum = zero
    item_count = 0
    beta = float(getattr(args, "ct_huber_beta", 0.1))

    def _add(points: torch.Tensor, sdf: torch.Tensor, target) -> None:
        nonlocal loss_sum, item_count
        if points.numel() == 0:
            return
        fields = query_ct_fields_unified(
            points,
            training_state,
            signed_distance=sdf,
            config=args,
            intensity_air=float(context.intensity_air),
            include_surface=True,
            train_ct_value=True,
            detach_value_geometry=True,
        )
        predicted = fields["I_s"].to(dtype=torch.float32)
        if isinstance(target, torch.Tensor):
            target_values = target.to(device=predicted.device, dtype=predicted.dtype).reshape(-1)
        else:
            target_values = torch.full_like(predicted, float(target))
        keep = torch.isfinite(predicted) & torch.isfinite(target_values)
        if not torch.any(keep):
            return
        loss_sum = loss_sum + F.smooth_l1_loss(predicted[keep], target_values[keep], beta=beta, reduction="sum")
        item_count += int(keep.sum().item())

    if material_points.numel() > 0:
        material_target = sample_volume_field(
            volume_field,
            material_points,
            context.spacing_zyx,
        ).reshape(-1).to(dtype=torch.float32)
        _add(material_points, material_sdf, material_target)
    _add(air_points, air_sdf, float(context.intensity_air))

    if item_count <= 0:
        return zero
    return weight * loss_sum / float(item_count)


def _material_membership_at(context: CTTrainingBootstrap, points_xyz: torch.Tensor):
    """Hard material-mask membership (1 inside / 0 outside) sampled at world points.

    Replaces the SDF half-space gate for inside/outside ownership. Returns None
    when the mask is unavailable.
    """
    mask = context.analysis_gpu.get("material_mask")
    if mask is None or points_xyz.numel() == 0:
        return None
    cache = context.intensity_field_cache
    field = cache.get("material_membership_volume")
    if field is None:
        m = mask if torch.is_tensor(mask) else torch.as_tensor(np.asarray(mask, dtype=np.float32))
        field = m.to(device=points_xyz.device, dtype=torch.float32).reshape(
            1, 1, *tuple(int(v) for v in context.volume_shape)
        )
        cache["material_membership_volume"] = field
    return (
        sample_volume_field(field, points_xyz, context.spacing_zyx)
        .reshape(-1)
        .to(dtype=torch.float32)
        .clamp(0.0, 1.0)
    )


def bulk_void_leak_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
) -> torch.Tensor:
    """Anti-leak penalty on UNGATED bulk density over confident void.

    Because display/eval render ungated raw A_b, growing sigma could push tails
    into air/cavities. This penalizes the ungated den_b at mask-void points so the
    ungated field stays clean by construction (no gate hiding the leak)::

        L = mean( den_b(x) )   over D > margin (void), evaluated with apply_bulk_gate=False

    Gradient flows to bulk scale (shrinks leaking tails). Opt-in via weight.
    """
    device = training_state.xyz.device
    zero = torch.zeros((), dtype=torch.float32, device=device)
    weight = float(getattr(args, "ct_bulk_void_leak_weight", 0.0))
    if weight <= 0.0:
        return zero
    bulk_xyz = getattr(training_state, "bulk_xyz", None)
    if bulk_xyz is None or bulk_xyz.numel() == 0:
        return zero
    sample_count = int(getattr(args, "ct_bulk_void_leak_sample_count", 4096))
    if sample_count <= 0:
        return zero
    margin = max(float(getattr(args, "ct_bulk_void_leak_margin_vox", 0.5)), 0.0) * float(min(context.spacing_zyx))
    air_candidates = (
        context.field_pools.get("cavity_air_shell"),
        context.field_pools.get("void_air"),
        context.field_pools.get("near_material_air"),
        context.field_pools.get("air_shell"),
        context.preferred_air_candidates,
    )
    pts, sdf_vals = _sample_filtered_from_candidate_sets(
        air_candidates,
        sample_count,
        context,
        device=device,
        signed_distance_predicate=lambda s: s > float(margin),
        oversample=8,
    )
    if pts.numel() == 0:
        return zero
    fields = query_ct_fields_unified(
        pts,
        training_state,
        signed_distance=sdf_vals,
        config=args,
        intensity_air=float(context.intensity_air),
        include_surface=False,
        bulk_train_xyz=False,
        bulk_train_scale=True,
        bulk_scale_grad=float(getattr(args, "ct_bulk_coverage_growth_scale_grad", 1.0)),
        train_ct_value=False,
        apply_bulk_gate=False,  # ungated: see the true leak so we can penalize it
    )
    den_b = fields["den_b"].to(dtype=torch.float32)
    return weight * den_b.clamp_min(0.0).mean()


def bulk_coverage_growth_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
) -> torch.Tensor:
    """Coverage-driven bulk sigma growth over the confident material interior.

    Pushes bulk sigma (gradient to scale only) UP wherever the material-interior
    occupancy ``occ_b = 1 - exp(-den_b)`` falls short of ``ct_bulk_coverage_target``::

        L = mean( relu(target_occ - occ_b)^2 )   over samples with D < -margin

    The half-space gate keeps den_b ~ 0 on the void side, so growing sigma cannot
    leak coverage into air / cavities. Intensity (a_i) and centers stay frozen;
    this is a pure geometry-growth term. Opt-in via ct_bulk_coverage_growth_weight.
    """
    device = training_state.xyz.device
    zero = torch.zeros((), dtype=torch.float32, device=device)
    weight = float(getattr(args, "ct_bulk_coverage_growth_weight", 0.0))
    if weight <= 0.0:
        return zero
    bulk_xyz = getattr(training_state, "bulk_xyz", None)
    if bulk_xyz is None or bulk_xyz.numel() == 0:
        return zero
    sample_count = int(getattr(args, "ct_bulk_coverage_growth_sample_count", 4096))
    if sample_count <= 0:
        return zero
    margin = max(float(getattr(args, "ct_bulk_coverage_growth_margin_vox", 0.5)), 0.0) * float(
        min(context.spacing_zyx)
    )
    support_candidates = (
        context.field_pools.get("material_deep_pool"),
        context.field_pools.get("support"),
        context.field_pools.get("cavity_material_shell"),
    )
    pts, sdf_vals = _sample_filtered_from_candidate_sets(
        support_candidates,
        sample_count,
        context,
        device=device,
        signed_distance_predicate=lambda s: s < -float(margin),
        oversample=6,
    )
    if pts.numel() == 0:
        return zero
    fields = query_ct_fields_unified(
        pts,
        training_state,
        signed_distance=sdf_vals,
        config=args,
        intensity_air=float(context.intensity_air),
        include_surface=False,
        bulk_train_xyz=False,
        bulk_train_scale=True,
        bulk_scale_grad=float(getattr(args, "ct_bulk_coverage_growth_scale_grad", 1.0)),
        bulk_train_opacity=bool(getattr(args, "ct_bulk_coverage_growth_train_opacity", False)),
        train_ct_value=False,
        material_membership=_material_membership_at(context, pts),
    )
    occ_b = fields["occ_b"].to(dtype=torch.float32)
    target_occ = min(max(float(getattr(args, "ct_bulk_coverage_target", 0.9)), 0.0), 1.0 - 1e-4)
    gap = torch.relu(target_occ - occ_b)
    if gap.numel() == 0:
        return zero
    return weight * gap.square().mean()


def _bulk_scale_adaptive_cap_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
) -> torch.Tensor:
    if not bool(getattr(args, "ct_bulk_scale_adaptive_cap", True)):
        return torch.zeros((), dtype=torch.float32, device=training_state.xyz.device)
    sigma = getattr(training_state, "bulk_sigma", None)
    center_sdf = getattr(training_state, "bulk_center_sdf", None)
    curvature = getattr(training_state, "bulk_center_curvature", None)
    if sigma is None or center_sdf is None or curvature is None or sigma.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=training_state.xyz.device)
    boundary_mask = center_sdf.to(dtype=torch.float32) > -float(getattr(args, "ct_bulk_halfspace_skip_depth", 2.0))
    if not torch.any(boundary_mask):
        return torch.zeros((), dtype=torch.float32, device=training_state.xyz.device)
    voxel = min(float(value) for value in context.spacing_zyx)
    curv_abs = torch.abs(curvature.to(dtype=torch.float32)).clamp_min(1e-6)
    radius_curv = 1.0 / curv_abs
    sigma_max = torch.sqrt(2.0 * float(voxel) * radius_curv).clamp(
        min=float(getattr(args, "ct_bulk_scale_floor", 0.05)),
        max=float(getattr(args, "ct_bulk_scale_global_max", 1.5)),
    )
    gap = torch.relu(sigma.to(dtype=torch.float32)[boundary_mask] - sigma_max[boundary_mask])
    if gap.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=training_state.xyz.device)
    return gap.square().mean()


def bulk_semantic_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    boundary_band_distance: float,
) -> torch.Tensor:
    device = getattr(getattr(training_state, "xyz", None), "device", None)
    if device is None:
        signed_distance = context.signed_distance_field.get("signed_distance")
        device = signed_distance.device if torch.is_tensor(signed_distance) else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_count = int(getattr(args, "ct_bulk_semantic_sample_count", 0))
    if sample_count <= 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    epsilon = float(getattr(args, "ct_bulk_semantic_epsilon", 0.5))
    inner_band = _dual_surface_inner_band(args) if _dual_separated_training_enabled(args) else max(float(boundary_band_distance), epsilon)

    def _count_split(total: int, ratios: tuple[float, ...]) -> list[int]:
        weights = torch.tensor([max(float(value), 0.0) for value in ratios], dtype=torch.float64)
        if total <= 0 or float(weights.sum().item()) <= 0.0:
            return [0 for _ in ratios]
        raw = weights / weights.sum() * int(total)
        counts = torch.floor(raw).to(dtype=torch.long)
        remainder = int(total) - int(counts.sum().item())
        if remainder > 0:
            order = torch.argsort(raw - counts.to(dtype=raw.dtype), descending=True)
            counts[order[:remainder]] += 1
        return [int(value) for value in counts.tolist()]

    positive_ratio = min(max(float(getattr(args, "ct_bulk_semantic_pos_ratio", 0.70)), 0.0), 1.0)
    positive_count = int(round(float(sample_count) * positive_ratio))
    negative_count = sample_count - positive_count
    deep_count, boundary_count, cavity_material_count = _count_split(
        positive_count,
        (
            float(getattr(args, "ct_bulk_semantic_deep_ratio", 0.45)),
            float(getattr(args, "ct_bulk_semantic_boundary_ratio", 0.20)),
            float(getattr(args, "ct_bulk_semantic_cavity_material_ratio", 0.15)),
        ),
    )
    void_air_count, cavity_air_count, exterior_air_count = _count_split(
        negative_count,
        (
            float(getattr(args, "ct_bulk_semantic_void_air_ratio", 0.10)),
            float(getattr(args, "ct_bulk_semantic_cavity_air_ratio", 0.10)),
            float(getattr(args, "ct_bulk_semantic_exterior_air_ratio", 0.10)),
        ),
    )

    boundary_band = max(float(boundary_band_distance), epsilon)

    deep_candidates = context.field_pools.get("material_deep_pool")
    if _candidate_count(deep_candidates) == 0:
        deep_candidates = context.field_pools.get("support")
    boundary_candidates = context.field_pools.get("support")
    cavity_material_candidates = context.field_pools.get("cavity_material_shell")
    if _candidate_count(cavity_material_candidates) == 0:
        cavity_material_candidates = context.field_pools.get("support")

    void_air_candidates = context.field_pools.get("void_air")
    if _candidate_count(void_air_candidates) == 0:
        void_air_candidates = context.field_pools.get("near_material_air")
    cavity_air_candidates = context.field_pools.get("cavity_air_shell")
    if _candidate_count(cavity_air_candidates) == 0:
        cavity_air_candidates = context.field_pools.get("void_air")
    if _candidate_count(cavity_air_candidates) == 0:
        cavity_air_candidates = context.field_pools.get("near_material_air")
    exterior_air_candidates = context.field_pools.get("exterior_air_near_band")
    if _candidate_count(exterior_air_candidates) == 0:
        exterior_air_candidates = context.field_pools.get("exterior_air")
    if _candidate_count(exterior_air_candidates) == 0:
        exterior_air_candidates = context.field_pools.get("near_material_air")
    if _candidate_count(exterior_air_candidates) == 0:
        exterior_air_candidates = context.field_pools.get("air_shell")

    deep_points, deep_sdf = _sample_filtered_semantic_points(
        deep_candidates,
        deep_count,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: sdf < -float(epsilon),
    )
    boundary_points, boundary_sdf = _sample_filtered_semantic_points(
        boundary_candidates,
        boundary_count,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: (sdf < -1e-6) & (sdf >= -float(boundary_band)),
    )
    cavity_material_points, cavity_material_sdf = _sample_filtered_semantic_points(
        cavity_material_candidates,
        cavity_material_count,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: sdf < -float(epsilon),
    )
    void_air_points, void_air_sdf = _sample_filtered_semantic_points(
        void_air_candidates,
        void_air_count,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: sdf > float(epsilon),
    )
    cavity_air_points, cavity_air_sdf = _sample_filtered_semantic_points(
        cavity_air_candidates,
        cavity_air_count,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: sdf > float(epsilon),
    )
    exterior_air_points, exterior_air_sdf = _sample_filtered_semantic_points(
        exterior_air_candidates,
        exterior_air_count,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: sdf > float(epsilon),
    )

    parts = []
    targets = []
    signed_parts = []
    floor_targets = []

    def _append(points, sdf, target_value: float, floor_value: float | None = None) -> None:
        if points.numel() == 0:
            return
        parts.append(points)
        targets.append(torch.full((points.shape[0],), float(target_value), dtype=torch.float32, device=device))
        signed_parts.append(sdf)
        if floor_value is None:
            floor_targets.append(torch.full((points.shape[0],), -1.0, dtype=torch.float32, device=device))
        else:
            floor_targets.append(torch.full((points.shape[0],), float(floor_value), dtype=torch.float32, device=device))

    _append(deep_points, deep_sdf, 1.0, float(getattr(args, "ct_bulk_floor_deep_target", 0.85)))
    _append(
        boundary_points,
        boundary_sdf,
        float(getattr(args, "ct_bulk_semantic_boundary_target", 0.75)),
        float(getattr(args, "ct_bulk_floor_shell_target", 0.65)),
    )
    _append(
        cavity_material_points,
        cavity_material_sdf,
        float(getattr(args, "ct_bulk_semantic_cavity_target", 0.90)),
        float(getattr(args, "ct_bulk_floor_shell_target", 0.65)),
    )
    _append(void_air_points, void_air_sdf, 0.0)
    _append(cavity_air_points, cavity_air_sdf, 0.0)
    _append(exterior_air_points, exterior_air_sdf, 0.0)
    if not parts:
        return torch.zeros((), dtype=torch.float32, device=device)

    points = torch.cat(parts, dim=0)
    target = torch.cat(targets, dim=0)
    signed_distance = torch.cat(signed_parts, dim=0)
    floor_target = torch.cat(floor_targets, dim=0)
    fields = query_ct_fields_unified(
        points,
        training_state,
        signed_distance=signed_distance,
        config=args,
        intensity_air=float(context.intensity_air),
        include_surface=False,
        bulk_train_xyz=True,
        bulk_train_rotation=True,
        bulk_train_scale=True,
        bulk_train_opacity=True,
        train_ct_value=False,
    )
    occ = fields["occ_b_raw"].to(dtype=torch.float32)
    semantic = asymmetric_binary_focal_loss(
        occ,
        target,
        gamma_pos=float(getattr(args, "ct_focal_gamma_pos", 0.0)),
        gamma_neg=float(getattr(args, "ct_focal_gamma_neg", getattr(args, "ct_bulk_semantic_focal_gamma", 2.0))),
        alpha_pos=float(getattr(args, "ct_focal_alpha_pos", 0.70)),
    )
    floor_weight = float(getattr(args, "ct_bulk_floor_weight", 0.15))
    total_loss = semantic
    if floor_weight > 0.0:
        floor_mask = floor_target >= 0.0
        if torch.any(floor_mask):
            floor_gap = torch.relu(floor_target[floor_mask].to(device=occ.device, dtype=occ.dtype) - occ[floor_mask])
            total_loss = total_loss + float(floor_weight) * floor_gap.square().mean()
    void_weight = float(getattr(args, "ct_bulk_void_weight", 2.0))
    void_mask = target <= 0.0
    if void_weight > 0.0 and torch.any(void_mask):
        void_occ = occ[void_mask].clamp(1e-6, 1.0 - 1e-6)
        void_loss = F.binary_cross_entropy(void_occ, torch.zeros_like(void_occ))
        total_loss = total_loss + float(void_weight) * void_loss
    return total_loss


def masked_bulk_coverage_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    boundary_band_distance: float,
) -> torch.Tensor:
    device = getattr(getattr(training_state, "xyz", None), "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    active_terms = []

    material_sample_count = int(getattr(args, "ct_bulk_coverage_sample_count", 0))
    material_candidates = context.field_pools.get("support")
    if _candidate_count(material_candidates) == 0:
        material_candidates = context.field_pools.get("material_deep_pool")
    if material_sample_count > 0 and _candidate_count(material_candidates) > 0:
        points = _sample_occupancy_points(material_candidates, material_sample_count, context.spacing_zyx, device=device)
        if points.numel() > 0:
            bulk_density = query_ct_density_from_state_by_region(
                training_state,
                points,
                region="bulk",
                detach=False,
            ).to(dtype=torch.float32)
            bulk_occupancy = density_to_occupancy(bulk_density)
            target = float(getattr(args, "ct_bulk_coverage_target", 0.85))
            coverage_gap = torch.relu(torch.full_like(bulk_occupancy, target) - bulk_occupancy)
            active_terms.append(
                F.smooth_l1_loss(
                    coverage_gap,
                    torch.zeros_like(coverage_gap),
                    beta=float(args.ct_huber_beta),
                )
            )

    cavity_sample_count = int(getattr(args, "ct_false_hole_sample_count", 0))
    cavity_candidates = context.field_pools.get("cavity_material_shell")
    if cavity_sample_count > 0 and _candidate_count(cavity_candidates) > 0:
        points = _sample_occupancy_points(cavity_candidates, cavity_sample_count, context.spacing_zyx, device=device)
        if points.numel() > 0:
            volume_field = context.volume_cuda.reshape(1, 1, *tuple(int(value) for value in context.volume_shape))
            target_intensity = sample_volume_field(volume_field, points, context.spacing_zyx).reshape(-1).to(dtype=torch.float32)
            intensity_air = float(context.intensity_air)
            intensity_mat = float(context.intensity_mat)
            intensity_range = max(abs(intensity_mat - intensity_air), 1e-6)
            material_threshold = intensity_air + float(getattr(args, "ct_false_hole_material_threshold", 0.65)) * (
                intensity_mat - intensity_air
            )
            with torch.no_grad():
                predicted_intensity = _compute_detached_bounded_intensity_prediction(
                    training_state,
                    points,
                    signed_distance=_sample_signed_distance(context.signed_distance_field, points),
                    intensity_air=intensity_air,
                    boundary_band_distance=boundary_band_distance,
                    surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                    material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
                    config=args,
                    use_unified=_use_unified_compositor(args),
                ).reshape(-1).to(dtype=torch.float32)
                dark_margin = float(getattr(args, "ct_false_hole_dark_margin", 0.15)) * intensity_range
                dark_gap = target_intensity - predicted_intensity - float(dark_margin)
                material_gt = target_intensity >= float(material_threshold)
                false_hole_boost = 1.0 + float(getattr(args, "ct_cavity_false_hole_boost", 2.0)) * (
                    dark_gap / intensity_range
                ).clamp(0.0, 1.0)
                weights = torch.where(material_gt, false_hole_boost, torch.zeros_like(false_hole_boost))

            if torch.any(weights > 0.0):
                bulk_density = query_ct_density_from_state_by_region(
                    training_state,
                    points,
                    region="bulk",
                    detach=False,
                ).to(dtype=torch.float32)
                bulk_occupancy = density_to_occupancy(bulk_density)
                target = float(getattr(args, "ct_false_hole_target_occupancy", getattr(args, "ct_bulk_coverage_target", 0.90)))
                coverage_gap = torch.relu(torch.full_like(bulk_occupancy, target) - bulk_occupancy)
                per_sample_loss = F.smooth_l1_loss(
                    coverage_gap,
                    torch.zeros_like(coverage_gap),
                    beta=float(args.ct_huber_beta),
                    reduction="none",
                )
                weights = weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype).detach()
                cavity_term = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)
                active_terms.append(float(getattr(args, "ct_cavity_material_coverage_weight", 3.0)) * cavity_term)

    air_sample_count = int(getattr(args, "ct_bulk_air_exclusion_sample_count", 0))
    air_weight = float(getattr(args, "ct_bulk_air_exclusion_weight", 0.0))
    if air_weight > 0.0 and air_sample_count > 0:
        air_candidates = context.field_pools.get("void_air")
        exterior_candidates = context.field_pools.get("exterior_air_near_band")
        parts = []
        void_count = air_sample_count // 2
        exterior_count = air_sample_count - void_count
        if void_count > 0 and _candidate_count(air_candidates) > 0:
            parts.append(_sample_occupancy_points(air_candidates, void_count, context.spacing_zyx, device=device))
        if exterior_count > 0 and _candidate_count(exterior_candidates) > 0:
            parts.append(_sample_occupancy_points(exterior_candidates, exterior_count, context.spacing_zyx, device=device))
        if not parts:
            fallback_air = context.field_pools.get("air")
            if _candidate_count(fallback_air) > 0:
                parts.append(_sample_occupancy_points(fallback_air, air_sample_count, context.spacing_zyx, device=device))
        if parts:
            air_points = torch.cat([part for part in parts if part.numel() > 0], dim=0)[:air_sample_count]
            if air_points.numel() > 0:
                bulk_density = query_ct_density_from_state_by_region(
                    training_state,
                    air_points,
                    region="bulk",
                    detach=False,
                ).to(dtype=torch.float32)
                bulk_occupancy = density_to_occupancy(bulk_density)
                target = float(getattr(args, "ct_bulk_air_exclusion_target", 0.05))
                overflow = torch.relu(bulk_occupancy - target)
                active_terms.append(
                    air_weight
                    * F.smooth_l1_loss(
                        overflow,
                        torch.zeros_like(overflow),
                        beta=float(args.ct_huber_beta),
                    )
                )

    if not active_terms:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(active_terms).sum()


def _false_hole_zero_metrics() -> dict:
    return {
        "false_hole_candidate_count": 0,
        "false_hole_active_ratio": 0.0,
        "false_hole_pred_intensity_p10": float("nan"),
        "false_hole_pred_intensity_p50": float("nan"),
        "false_hole_target_intensity_p50": float("nan"),
        "false_hole_bulk_occ_p10": float("nan"),
        "false_hole_bulk_occ_p50": float("nan"),
    }


def _tensor_quantile_or_nan(values: torch.Tensor, q: float) -> float:
    values = values.detach().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.to(dtype=torch.float32), float(q)).item())


def _query_detached_region_density_and_value(training_state, points_xyz: torch.Tensor, region: str) -> tuple[torch.Tensor, torch.Tensor]:
    if region == "bulk":
        means = training_state.bulk_xyz
        rotations = training_state.bulk_rotation_mats
        scales = training_state.bulk_scales
        opacity = training_state.bulk_opacity
        ct_value = training_state.bulk_ct_value
        spatial_grid = getattr(training_state, "bulk_spatial_grid", None)
        support_extent = getattr(training_state, "bulk_support_extent", None)
    elif region == "surface":
        means = training_state.surface_xyz
        rotations = training_state.surface_rotation_mats
        scales = training_state.surface_scales
        opacity = training_state.surface_opacity
        ct_value = training_state.surface_ct_value
        spatial_grid = getattr(training_state, "surface_spatial_grid", None)
        support_extent = getattr(training_state, "surface_support_extent", None)
    else:
        raise ValueError("region must be 'bulk' or 'surface'.")

    zeros = torch.zeros((points_xyz.shape[0],), dtype=points_xyz.dtype, device=points_xyz.device)
    if means.numel() == 0 or ct_value is None or ct_value.numel() == 0:
        return zeros, zeros

    means = means.detach()
    rotations = rotations.detach()
    scales = scales.detach().clamp_min(1e-6)
    opacity = opacity.detach()
    ct_value = ct_value.detach().to(device=opacity.device, dtype=opacity.dtype)
    support_extent = None if support_extent is None else support_extent.detach()

    density = _query_ct_density_native_chunked(
        means,
        rotations,
        scales,
        opacity,
        points_xyz,
        spatial_grid=spatial_grid,
        support_extent=support_extent,
    )
    weighted_density = _query_ct_density_native_chunked(
        means,
        rotations,
        scales,
        opacity * ct_value,
        points_xyz,
        spatial_grid=spatial_grid,
        support_extent=support_extent,
    )
    return density.to(dtype=torch.float32), weighted_density.to(dtype=torch.float32)


def _compute_detached_bounded_intensity_prediction(
    training_state,
    points: torch.Tensor,
    signed_distance: torch.Tensor,
    *,
    intensity_air: float,
    boundary_band_distance: float,
    surface_material_gate_sigma=None,
    material_compose_mode: str = "bulk_first_material",
    config=None,
    use_unified: bool | None = None,
) -> torch.Tensor:
    if use_unified is None:
        use_unified = _use_unified_compositor(config) if config is not None else False
    if use_unified:
        fields = query_ct_fields_unified(
            points,
            training_state,
            signed_distance=signed_distance,
            config=config,
            intensity_air=float(intensity_air),
            include_surface=True,
            train_ct_value=False,
            detach_value_geometry=True,
        )
        return fields["I_pred"].detach()
    bulk_density, bulk_weighted_density = _query_detached_region_density_and_value(training_state, points, "bulk")
    surface_density, surface_weighted_density = _query_detached_region_density_and_value(training_state, points, "surface")
    combined_occupancy, bulk_weight, surface_weight = compose_signed_overlap_occupancy(
        bulk_density,
        surface_density,
        signed_distance,
        boundary_band_distance,
        surface_material_gate_sigma=surface_material_gate_sigma,
        material_compose_mode=material_compose_mode,
    )
    bulk_value = bulk_weighted_density / bulk_density.clamp_min(1e-6)
    surface_value = surface_weighted_density / surface_density.clamp_min(1e-6)
    if surface_weight is None:
        local_intensity = bulk_value
    else:
        local_intensity = (
            bulk_weight.detach() * bulk_value + surface_weight.detach() * surface_value
        ) / (bulk_weight + surface_weight).detach().clamp_min(1e-6)
    return float(intensity_air) + (local_intensity - float(intensity_air)) * combined_occupancy.detach()


def false_hole_geometry_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    boundary_band_distance: float,
    *,
    return_metrics: bool = False,
):
    """Diagnostic wrapper for cavity material false-hole coverage.

    Training now uses masked_bulk_coverage_loss so false-hole correction is part
    of the bulk mask coverage term instead of a separate loss branch.
    """
    device = getattr(getattr(training_state, "xyz", None), "device", None)
    if device is None:
        signed_distance = context.signed_distance_field.get("signed_distance")
        device = signed_distance.device if torch.is_tensor(signed_distance) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zero = torch.zeros((), dtype=torch.float32, device=device)
    metrics = _false_hole_zero_metrics() if return_metrics else {}

    sample_count = int(getattr(args, "ct_false_hole_sample_count", 0))
    if sample_count <= 0:
        return (zero, metrics) if return_metrics else zero

    candidates = context.field_pools.get("cavity_material_shell")
    if _candidate_count(candidates) == 0:
        return (zero, metrics) if return_metrics else zero

    points = _sample_occupancy_points(candidates, sample_count, context.spacing_zyx, device=device)
    if points.numel() == 0:
        return (zero, metrics) if return_metrics else zero

    volume_field = context.volume_cuda.reshape(1, 1, *tuple(int(value) for value in context.volume_shape))
    target_intensity = sample_volume_field(volume_field, points, context.spacing_zyx).reshape(-1).to(dtype=torch.float32)

    intensity_air = float(context.intensity_air)
    intensity_mat = float(context.intensity_mat)
    intensity_range = max(abs(intensity_mat - intensity_air), 1e-6)
    material_threshold = intensity_air + float(getattr(args, "ct_false_hole_material_threshold", 0.65)) * (
        intensity_mat - intensity_air
    )
    dark_margin = float(getattr(args, "ct_false_hole_dark_margin", 0.15)) * intensity_range

    with torch.no_grad():
        predicted_intensity = _compute_detached_bounded_intensity_prediction(
            training_state,
            points,
            signed_distance=_sample_signed_distance(context.signed_distance_field, points),
            intensity_air=intensity_air,
            boundary_band_distance=boundary_band_distance,
            surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
            material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
            config=args,
            use_unified=_use_unified_compositor(args),
        ).reshape(-1).to(dtype=torch.float32)

        material_gt = target_intensity >= float(material_threshold)
        dark_gap = target_intensity - predicted_intensity - float(dark_margin)
        predicted_dark = dark_gap > 0.0
        finite = torch.isfinite(target_intensity) & torch.isfinite(predicted_intensity)
        active = material_gt & predicted_dark & finite
        darkness_weight = (dark_gap / intensity_range).clamp(0.0, 1.0)

    if return_metrics:
        candidate_count = int(points.shape[0])
        metrics["false_hole_candidate_count"] = candidate_count
        metrics["false_hole_active_ratio"] = 0.0 if candidate_count == 0 else float(active.float().mean().item())
        metrics["false_hole_pred_intensity_p10"] = _tensor_quantile_or_nan(predicted_intensity[active], 0.10)
        metrics["false_hole_pred_intensity_p50"] = _tensor_quantile_or_nan(predicted_intensity[active], 0.50)
        metrics["false_hole_target_intensity_p50"] = _tensor_quantile_or_nan(target_intensity[active], 0.50)

    if not torch.any(active):
        return (zero, metrics) if return_metrics else zero

    active_points = points[active]
    bulk_density = query_ct_density_from_state_by_region(
        training_state,
        active_points,
        region="bulk",
        detach=False,
    ).to(dtype=torch.float32)
    bulk_occupancy = density_to_occupancy(bulk_density)
    target_occ = float(getattr(args, "ct_false_hole_target_occupancy", 0.90))
    coverage_gap = torch.relu(torch.full_like(bulk_occupancy, target_occ) - bulk_occupancy)
    per_sample_loss = F.smooth_l1_loss(
        coverage_gap,
        torch.zeros_like(coverage_gap),
        beta=float(args.ct_huber_beta),
        reduction="none",
    )
    weights = darkness_weight[active].to(device=per_sample_loss.device, dtype=per_sample_loss.dtype).detach().clamp_min(1e-3)
    loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)

    if return_metrics:
        metrics["false_hole_bulk_occ_p10"] = _tensor_quantile_or_nan(bulk_occupancy, 0.10)
        metrics["false_hole_bulk_occ_p50"] = _tensor_quantile_or_nan(bulk_occupancy, 0.50)
    return (loss, metrics) if return_metrics else loss


def bulk_sdf_containment_loss(context: CTTrainingBootstrap, args, training_state) -> torch.Tensor:
    """Soft bulk overflow penalty using an isotropic center-plus-radius footprint."""
    bulk_xyz = training_state.bulk_xyz
    if bulk_xyz.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=training_state.xyz.device)

    bulk_sigma = getattr(training_state, "bulk_sigma", None)
    bulk_scales = getattr(training_state, "bulk_scales", None)
    bulk_rotation_mats = getattr(training_state, "bulk_rotation_mats", None)
    sample_count = int(getattr(args, "ct_bulk_sdf_containment_sample_count", 0))
    if sample_count > 0 and bulk_xyz.shape[0] > sample_count:
        sample_indices = torch.randint(
            0,
            int(bulk_xyz.shape[0]),
            (sample_count,),
            device=bulk_xyz.device,
        )
        bulk_xyz = bulk_xyz.index_select(0, sample_indices)
        if bulk_sigma is not None and bulk_sigma.numel() > 0:
            bulk_sigma = bulk_sigma.index_select(0, sample_indices)
        if bulk_scales is not None and bulk_scales.numel() > 0:
            bulk_scales = bulk_scales.index_select(0, sample_indices)
        if bulk_rotation_mats is not None and bulk_rotation_mats.numel() > 0:
            bulk_rotation_mats = bulk_rotation_mats.index_select(0, sample_indices)

    signed_distance_volume = context.signed_distance_field["signed_distance"]
    signed_distance = sample_volume_field(
        signed_distance_volume,
        bulk_xyz,
        context.signed_distance_field["spacing_zyx"],
    ).reshape(-1).to(dtype=torch.float32)
    margin = float(getattr(args, "ct_bulk_sdf_containment_margin", 0.0))
    if bulk_sigma is not None and bulk_sigma.numel() > 0:
        footprint_extent = bulk_sigma.to(device=bulk_xyz.device, dtype=torch.float32).clamp_min(1e-6)
    else:
        sdf_normals = _sample_coarse_sdf_normals(
            context.signed_distance_field,
            bulk_xyz,
            context.spacing_zyx,
        ).to(device=bulk_xyz.device, dtype=bulk_xyz.dtype)
        local_sdf_normals = torch.einsum("nij,nj->ni", bulk_rotation_mats.transpose(1, 2), sdf_normals)
        footprint_extent = torch.sqrt(
            torch.sum((local_sdf_normals * bulk_scales) ** 2, dim=-1).clamp_min(1e-8)
        ).to(dtype=torch.float32)
    combined = torch.relu(signed_distance + footprint_extent + margin).square()
    valid = torch.isfinite(combined)
    if not torch.any(valid):
        return torch.zeros((), dtype=torch.float32, device=bulk_xyz.device)
    return combined[valid].mean()


def _far_phase_occupancy_loss(context: CTTrainingBootstrap, args, training_state, boundary_band_distance: float):
    phase_count = max(1, int(round(float(args.ct_volume_sample_count) * (1.0 - float(getattr(args, "ct_surface_boundary_sample_ratio", 0.25))))))
    points, targets = _sample_far_phase_points(
        context.field_pools,
        phase_count,
        context.spacing_zyx,
        context.volume_shape,
        args.ct_volume_jitter,
        context.signed_distance_field,
        boundary_band_distance,
        support_sample_ratio=getattr(args, "ct_bulk_volume_support_sample_ratio", 0.5),
        exterior_air_sample_ratio=getattr(args, "ct_exterior_air_sample_ratio", 0.5),
        device="cuda",
    )
    if points.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device="cuda")
    signed_distance = _sample_signed_distance(context.signed_distance_field, points)
    occupancy = compute_role_separated_ct_prediction(
        training_state,
        points,
        signed_distance,
        boundary_band_distance,
        include_surface=True,
        detach_bulk=False,
        detach_surface=True,
        surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
        material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
    )
    return F.smooth_l1_loss(
        occupancy,
        targets.to(device=occupancy.device, dtype=occupancy.dtype),
        beta=float(args.ct_huber_beta),
    )


def surface_regularizer_loss(context: CTTrainingBootstrap, args, training_state, iteration: int = 0):
    if training_state.surface_xyz.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device="cuda")

    surface_xyz = training_state.surface_xyz
    surface_scales = training_state.surface_scales
    surface_rotation_mats = training_state.surface_rotation_mats
    surface_normals = training_state.surface_normals
    sample_count = int(getattr(args, "ct_surface_regularizer_sample_count", 8192))
    if sample_count > 0 and surface_xyz.shape[0] > sample_count:
        sample_indices = torch.randint(
            0,
            int(surface_xyz.shape[0]),
            (sample_count,),
            device=surface_xyz.device,
        )
        surface_xyz = surface_xyz.index_select(0, sample_indices)
        surface_scales = surface_scales.index_select(0, sample_indices)
        surface_rotation_mats = surface_rotation_mats.index_select(0, sample_indices)
        surface_normals = surface_normals.index_select(0, sample_indices)

    max_normal_thickness = float(args.ct_surface_sigma_n_max)
    signed_distance_volume = context.signed_distance_field["signed_distance"]
    signed_distance_spacing = context.signed_distance_field["spacing_zyx"]
    chunk_size = max(1, int(getattr(args, "ct_surface_regularizer_chunk_size", 2048)))
    total = torch.zeros((), dtype=torch.float32, device=surface_xyz.device)
    total_count = 0

    for start in range(0, int(surface_xyz.shape[0]), int(chunk_size)):
        xyz_chunk = surface_xyz[start : start + int(chunk_size)]
        scales_chunk = surface_scales[start : start + int(chunk_size)]
        rotations_chunk = surface_rotation_mats[start : start + int(chunk_size)]
        normals_chunk = surface_normals[start : start + int(chunk_size)]

        signed_distance = sample_volume_field(
            signed_distance_volume,
            xyz_chunk,
            signed_distance_spacing,
        ).reshape(-1).to(dtype=torch.float32)
        sampled_sdf_normals = _sample_coarse_sdf_normals(
            context.signed_distance_field,
            xyz_chunk,
            context.spacing_zyx,
        ).to(device=xyz_chunk.device, dtype=xyz_chunk.dtype)
        sampled_sdf_normals = F.normalize(sampled_sdf_normals, dim=-1, eps=1e-8)
        normals_chunk = F.normalize(normals_chunk.to(device=xyz_chunk.device, dtype=xyz_chunk.dtype), dim=-1, eps=1e-8)
        normal_alignment = 1.0 - torch.sum(normals_chunk * sampled_sdf_normals, dim=-1).clamp(-1.0, 1.0)
        local_sdf_normals = torch.einsum("nij,nj->ni", rotations_chunk.transpose(1, 2), sampled_sdf_normals)
        normal_thickness = torch.sqrt(torch.sum((local_sdf_normals * scales_chunk) ** 2, dim=-1).clamp_min(1e-8))
        thickness_term = torch.relu(normal_thickness - float(max_normal_thickness)).square()
        combined = (
            torch.abs(signed_distance)
            + float(getattr(args, "ct_surface_normal_weight", 1.0)) * normal_alignment.to(dtype=torch.float32)
            + float(getattr(args, "ct_surface_thickness_weight", 0.25)) * thickness_term.to(dtype=torch.float32)
        )
        valid = torch.isfinite(combined)
        if torch.any(valid):
            total = total + combined[valid].sum()
            total_count += int(valid.sum().item())

    if total_count > 0:
        surface_term = total / float(total_count)
    else:
        surface_term = torch.zeros((), dtype=torch.float32, device=surface_xyz.device)

    coverage_weight = float(getattr(args, "ct_surface_coverage_weight", 0.0))
    coverage_count = int(getattr(args, "ct_surface_coverage_sample_count", 0))
    coverage_until = int(getattr(args, "ct_surface_coverage_until_iter", 2500))
    if coverage_weight <= 0.0 or coverage_count <= 0 or int(iteration) > coverage_until:
        return surface_term

    boundary_points = context.analysis_gpu.get("boundary_points")
    if boundary_points is None:
        return surface_term
    anchors = as_device_tensor(boundary_points, device=surface_xyz.device, dtype=surface_xyz.dtype).reshape(-1, 3)
    if anchors.numel() == 0:
        return surface_term

    if anchors.shape[0] > coverage_count:
        sample_indices = torch.randint(
            0,
            int(anchors.shape[0]),
            (coverage_count,),
            device=surface_xyz.device,
        )
        anchors = anchors.index_select(0, sample_indices)

    surface_density = query_ct_density_from_state_by_region(
        training_state,
        anchors,
        region="surface",
        detach=False,
    ).to(dtype=torch.float32)
    surface_occupancy = density_to_occupancy(surface_density)
    target = torch.full_like(
        surface_occupancy,
        float(getattr(args, "ct_surface_coverage_target", 0.75)),
    )
    coverage_term = F.smooth_l1_loss(
        surface_occupancy,
        target,
        beta=float(args.ct_huber_beta),
    )
    return surface_term + coverage_weight * coverage_term


# ---------------------------------------------------------------------------
# Method-C role-separated losses
# ---------------------------------------------------------------------------

def surface_phase_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    *,
    mat_conf_pts: torch.Tensor | None = None,
    air_conf_pts: torch.Tensor | None = None,
    sample_count: int = 4096,
) -> torch.Tensor:
    """Phase loss for surface primitives using confidence-map supervision.

    Pushes surface signed-distance D_s:
      - below -margin at confident-material points
      - above +margin at confident-air points

    Uses softplus so loss is zero when the constraint is already satisfied.
    Gradient flows only to surface parameters (p_i, n_i).
    Bulk parameters are not touched.
    """
    from ct_pipeline.rendering.fields import query_surface_signed_distance

    if training_state.surface_xyz.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device="cuda")

    min_sp = min(float(v) for v in context.spacing_zyx)
    margin = float(getattr(args, "ct_surface_phase_margin_vox", 0.5)) * min_sp
    temp = max(float(getattr(args, "ct_surface_phase_temp_vox", 0.1)) * min_sp, 1e-6)
    device = training_state.surface_xyz.device

    terms = []

    def _phase_term(pts: torch.Tensor, sign: float) -> torch.Tensor:
        """sign=+1 inside (D_s < -margin); sign=-1 outside (D_s > +margin)."""
        if pts is None or pts.numel() == 0:
            return None
        pts_d = pts.to(device=device, dtype=torch.float32)
        D_s = query_surface_signed_distance(training_state, pts_d)
        # violation: sign*D_s > -margin  鈫? sign*D_s + margin > 0
        violation = sign * D_s + margin
        return F.softplus(violation / temp).mean()

    # fall back to phase1 SDF-based sampling if no confidence pts given
    if mat_conf_pts is None and context.field_pools.get("support") is not None:
        mat_conf_pts, _ = _sample_filtered_from_candidate_sets(
            (context.field_pools.get("material_deep_pool"), context.field_pools.get("support")),
            sample_count // 2,
            context,
            device=device,
            signed_distance_predicate=lambda sdf: sdf < -0.5,
        )
    if air_conf_pts is None:
        air_conf_pts, _ = _sample_filtered_from_candidate_sets(
            (context.field_pools.get("void_air"), context.field_pools.get("exterior_air_near_band")),
            sample_count // 2,
            context,
            device=device,
            signed_distance_predicate=lambda sdf: sdf > 0.5,
        )

    t_mat = _phase_term(mat_conf_pts, +1.0)  # inside: D_s < -margin
    t_air = _phase_term(air_conf_pts, -1.0)  # outside: D_s > +margin
    for t in (t_mat, t_air):
        if t is not None and torch.isfinite(t):
            terms.append(t)

    if not terms:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(terms).mean()


def role_separated_intensity_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    volume_field: torch.Tensor,
    intensity_flags: dict,
) -> torch.Tensor:
    """Bulk-only intensity loss for CTGS-vFinal.

    Implements::

        A_b(x) = raw_b(x) / den_b(x)
        pred(x) = A_b(x)

    Surface parameters and SDF masks do not participate in the CT intensity
    readout. SDF is used only for selecting confident material samples.
    """
    device = training_state.xyz.device
    zero = torch.zeros((), dtype=torch.float32, device=device)

    # sample interior material points
    # use a smaller count than generic volume sampling 鈥?D_s query over 40K+ surface
    # Gaussians is expensive; 4096 pts gives good coverage with acceptable cost.
    support_candidates = (
        context.field_pools.get("material_deep_pool"),
        context.field_pools.get("support"),
        context.field_pools.get("cavity_material_shell"),
    )
    sample_count = int(getattr(args, "ct_role_sep_intensity_sample_count",
                               min(4096, int(getattr(args, "ct_volume_sample_count", 16384)))))
    pts, sdf_vals = _sample_filtered_from_candidate_sets(
        support_candidates,
        sample_count,
        context,
        device=device,
        signed_distance_predicate=lambda sdf: sdf < 0.0,
    )
    if pts.numel() == 0:
        return zero

    # query bulk fields (no surface, gradients to bulk a_i only)
    fields = query_ct_fields_unified(
        pts,
        training_state,
        signed_distance=sdf_vals,
        config=args,
        intensity_air=float(context.intensity_air),
        include_surface=False,
        bulk_train_xyz=False,
        bulk_train_scale=bool(intensity_flags.get("bulk_train_scale", False)),
        bulk_scale_grad=float(intensity_flags.get("bulk_scale_grad", 1.0)),
        train_ct_value=False,
        material_membership=_material_membership_at(context, pts),
    )

    # CTGS-vFinal: raw A_b is the only CT intensity readout.
    mu_pred = fields["A_b"].to(dtype=torch.float32)

    target = sample_volume_field(volume_field, pts, context.spacing_zyx).reshape(-1).to(dtype=torch.float32)
    keep = torch.isfinite(mu_pred) & torch.isfinite(target)
    den_min = max(float(getattr(args, "ct_intensity_den_min", 0.0)), 0.0)
    if den_min > 0.0:
        keep = keep & (fields["den_b"].to(dtype=torch.float32) > den_min)
    if not torch.any(keep):
        return zero
    beta = float(getattr(args, "ct_huber_beta", 0.1))
    return F.smooth_l1_loss(mu_pred[keep], target[keep], beta=beta)


def bulk_adaptive_anchor_losses(
    context: CTTrainingBootstrap,
    args,
    training_state,
) -> torch.Tensor:
    """Scale and position anchor regularization for adaptive bulk mode.

    Prevents bulk scale / center from drifting too far from their initial values:
      L_scale_anchor = mean(||log sigma - log sigma0||^2) over bulk Gaussians
      L_pos_anchor   = mean(||delta_p||^2)                 over bulk Gaussians

    Both losses are very small (lambda <= 1e-3) and purely act as priors.
    """
    adaptive_mode = str(getattr(args, "ct_bulk_adaptive_mode", "fixed"))
    if adaptive_mode == "fixed":
        return torch.zeros((), dtype=torch.float32, device=training_state.xyz.device)

    device = training_state.xyz.device
    zero = torch.zeros((), dtype=torch.float32, device=device)
    total = zero

    # --- scale anchor ---
    scale_w = float(getattr(args, "ct_bulk_scale_anchor_weight", 1e-3))
    if scale_w > 0.0 and training_state.bulk_scales.numel() > 0:
        # log sigma - log sigma0 where sigma0 is the initial sigma stored at first iter
        sigma = training_state.bulk_scales.mean(dim=1).clamp_min(1e-6)  # (M,)
        sigma0 = getattr(training_state, "bulk_sigma_init", sigma.detach())
        log_diff = torch.log(sigma) - torch.log(sigma0.clamp_min(1e-6))
        total = total + scale_w * log_diff.square().mean()

    # --- position anchor ---
    pos_w = float(getattr(args, "ct_bulk_pos_anchor_weight", 1e-3))
    if pos_w > 0.0 and adaptive_mode == "scale_offset":
        gaussians = context.gaussians
        bulk_offset = getattr(gaussians, "_bulk_offset", None)
        if bulk_offset is not None and bulk_offset.numel() > 0:
            total = total + pos_w * bulk_offset.square().mean()

    return total


def _eagle_middle_patch_loss(context: CTTrainingBootstrap, args, training_state, iteration: int) -> torch.Tensor:
    if float(getattr(args, "ct_eagle_loss_weight", 0.0)) <= 0.0:
        return torch.zeros((), dtype=torch.float32, device="cuda")
    patch_size_value = int(getattr(args, "ct_eagle_patch_size", 64))
    if patch_size_value <= 0:
        return torch.zeros((), dtype=torch.float32, device="cuda")

    volume_shape = tuple(int(value) for value in context.volume_shape)
    if len(volume_shape) != 3:
        return torch.zeros((), dtype=torch.float32, device="cuda")
    patch_h = min(patch_size_value, volume_shape[1])
    patch_w = min(patch_size_value, volume_shape[2])
    origin = ((volume_shape[1] - patch_h) // 2, (volume_shape[2] - patch_w) // 2)
    slice_idx = int(volume_shape[0] // 2)
    gt_patch = sample_gt_slice_patch(context.volume_cuda, "z", slice_idx, origin, (patch_h, patch_w)).to(dtype=torch.float32)
    rr, cc = torch.meshgrid(
        torch.arange(patch_h, dtype=torch.float32, device=gt_patch.device),
        torch.arange(patch_w, dtype=torch.float32, device=gt_patch.device),
        indexing="ij",
    )
    points = _build_query_points_from_base(rr, cc, 0, slice_idx, origin, context.spacing_zyx)
    signed_distance = _sample_signed_distance(context.signed_distance_field, points)
    flags = _intensity_geometry_flags(args, iteration)
    pred = query_ct_fields_unified(
        points,
        training_state,
        signed_distance=signed_distance,
        config=args,
        intensity_air=float(context.intensity_air),
        include_surface=True,
        bulk_train_opacity=flags["bulk_train_opacity"],
        bulk_train_scale=flags["bulk_train_scale"],
        bulk_scale_grad=flags["bulk_scale_grad"],
        surface_train_opacity=flags["surface_train_opacity"],
        train_ct_value=True,
        detach_value_geometry=True,
    )["I_pred"].reshape(patch_h, patch_w)
    return eagle_patch_loss(
        pred,
        gt_patch,
        block_size=int(getattr(args, "ct_eagle_block_size", 16)),
    )


def compute_ct_loss_terms(context: CTTrainingBootstrap, args, training_state, iteration: int = 0) -> CTLossTerms:
    zero_loss = torch.zeros((), dtype=torch.float32, device="cuda")

    boundary_band_distance = _boundary_band_distance(args)
    bulk_intensity_training = _bulk_intensity_training_enabled(args)
    bulk_sample_count, boundary_sample_count = _split_role_sample_counts(
        args.ct_volume_sample_count,
        getattr(args, "ct_surface_boundary_sample_ratio", 0.25),
    )
    has_ct_value = getattr(training_state, "ct_value", None) is not None
    dual_volume_training = bool((not bulk_intensity_training) and has_ct_value and _dual_separated_training_enabled(args))
    if dual_volume_training:
        bulk_points = _ct_empty_points(device="cuda")
        boundary_points = _ct_empty_points(device="cuda")
    else:
        bulk_points = sample_bulk_volume_points_excluding_boundary(
            context.field_pools,
            bulk_sample_count,
            context.spacing_zyx,
            context.volume_shape,
            args.ct_volume_jitter,
            context.signed_distance_field,
            boundary_band_distance,
            device="cuda",
            preferred_air_candidates=context.preferred_air_candidates,
            support_sample_ratio=getattr(args, "ct_bulk_volume_support_sample_ratio", 0.5),
        )
        boundary_points = sample_surface_boundary_points(
            context.field_pools,
            boundary_sample_count,
            context.spacing_zyx,
            context.volume_shape,
            args.ct_volume_jitter,
            context.signed_distance_field,
            boundary_band_distance,
            device="cuda",
            preferred_air_candidates=context.preferred_air_candidates,
        )

    volume_loss = zero_loss
    volume_loss_sum = zero_loss
    volume_weight_sum = 0
    volume_field = context.volume_cuda.reshape(1, 1, *tuple(int(value) for value in context.volume_shape))

    intensity_air = float(context.intensity_air)
    intensity_flags = _intensity_geometry_flags(args, iteration)
    if bulk_intensity_training:
        _maybe_update_halfspace_tau(context, args, training_state, iteration, volume_field)

    training_mode = str(getattr(args, "ct_training_mode", "default"))
    if training_mode == "role_separated_joint" and bulk_intensity_training:
        volume_loss = role_separated_intensity_loss(
            context,
            args,
            training_state,
            volume_field=volume_field,
            intensity_flags=intensity_flags,
        )
        anchor = bulk_adaptive_anchor_losses(context, args, training_state)
        if torch.isfinite(anchor) and float(anchor) != 0.0:
            volume_loss = volume_loss + anchor
    elif bulk_intensity_training:
        volume_loss = _bulk_intensity_volume_loss(
            context,
            args,
            training_state,
            boundary_band_distance=boundary_band_distance,
            bulk_sample_count=bulk_sample_count,
            boundary_sample_count=boundary_sample_count,
            volume_field=volume_field,
            intensity_flags=intensity_flags,
        )
        surface_intensity = _surface_intensity_loss(
            context,
            args,
            training_state,
            boundary_band_distance=boundary_band_distance,
            volume_field=volume_field,
        )
        if torch.isfinite(surface_intensity) and float(surface_intensity) != 0.0:
            volume_loss = volume_loss + surface_intensity
    elif dual_volume_training:
        volume_loss = _dual_separated_volume_loss(
            context,
            args,
            training_state,
            iteration,
            boundary_band_distance=boundary_band_distance,
            bulk_sample_count=bulk_sample_count,
            boundary_sample_count=boundary_sample_count,
            volume_field=volume_field,
            intensity_flags=intensity_flags,
        )
    elif bulk_points.numel() > 0:
        bulk_signed_distance = _sample_signed_distance(context.signed_distance_field, bulk_points)
        if has_ct_value:
            predicted_intensity = compute_surface_bounded_bulk_volume_prediction(
                training_state,
                bulk_points,
                signed_distance=bulk_signed_distance,
                intensity_air=intensity_air,
                boundary_band_distance=boundary_band_distance,
                surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
                config=args,
                use_unified=_use_unified_compositor(args),
                bulk_train_opacity=intensity_flags["bulk_train_opacity"],
                bulk_train_scale=intensity_flags["bulk_train_scale"],
                bulk_scale_grad=intensity_flags["bulk_scale_grad"],
                surface_train_opacity=intensity_flags["surface_train_opacity"],
            )
        else:
            bulk_occupancy = compute_role_separated_ct_prediction(
                training_state,
                bulk_points,
                bulk_signed_distance,
                boundary_band_distance,
                include_surface=False,
                detach_bulk=False,
                surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
            )
            predicted_intensity = intensity_air + (float(context.intensity_mat) - intensity_air) * bulk_occupancy
        target_intensity = sample_volume_field(
            volume_field,
            bulk_points,
            context.spacing_zyx,
        ).reshape(-1).to(dtype=torch.float32)
        bulk_losses = F.smooth_l1_loss(
            predicted_intensity,
            target_intensity,
            beta=float(args.ct_huber_beta),
            reduction="none",
        )
        volume_loss_sum = volume_loss_sum + bulk_losses.sum()
        volume_weight_sum += int(bulk_losses.numel())

    if boundary_points.numel() > 0:
        boundary_signed_distance = _sample_signed_distance(context.signed_distance_field, boundary_points)
        if has_ct_value:
            predicted_intensity = compute_surface_bounded_bulk_volume_prediction(
                training_state,
                boundary_points,
                signed_distance=boundary_signed_distance,
                intensity_air=intensity_air,
                boundary_band_distance=boundary_band_distance,
                surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
                config=args,
                use_unified=_use_unified_compositor(args),
                bulk_train_opacity=intensity_flags["bulk_train_opacity"],
                bulk_train_scale=intensity_flags["bulk_train_scale"],
                bulk_scale_grad=intensity_flags["bulk_scale_grad"],
                surface_train_opacity=intensity_flags["surface_train_opacity"],
            )
        else:
            boundary_occupancy = compute_role_separated_ct_prediction(
                training_state,
                boundary_points,
                boundary_signed_distance,
                boundary_band_distance,
                include_surface=True,
                detach_bulk=True,
                surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
            )
            predicted_intensity = intensity_air + (float(context.intensity_mat) - intensity_air) * boundary_occupancy
        target_intensity = sample_volume_field(
            volume_field,
            boundary_points,
            context.spacing_zyx,
        ).reshape(-1).to(dtype=torch.float32)
        boundary_losses = F.smooth_l1_loss(
            predicted_intensity,
            target_intensity,
            beta=float(args.ct_huber_beta),
            reduction="none",
        )
        volume_loss_sum = volume_loss_sum + boundary_losses.sum()
        volume_weight_sum += int(boundary_losses.numel())

    if volume_weight_sum > 0:
        volume_loss = volume_loss_sum / float(volume_weight_sum)
    eagle_weight = float(getattr(args, "ct_eagle_loss_weight", 0.0))
    if eagle_weight > 0.0 and _use_unified_compositor(args) and not dual_volume_training:
        volume_loss = volume_loss + eagle_weight * _eagle_middle_patch_loss(context, args, training_state, iteration)

    occupancy_term = zero_loss
    false_hole_term = zero_loss
    false_hole_metrics = _false_hole_zero_metrics()
    if args.ct_lambda_occupancy != 0.0:
        if bulk_intensity_training:
            occupancy_term = occupancy_term + _bulk_scale_adaptive_cap_loss(context, args, training_state)
            occupancy_term = occupancy_term + bulk_coverage_growth_loss(context, args, training_state)
            occupancy_term = occupancy_term + bulk_void_leak_loss(context, args, training_state)
        elif _use_unified_compositor(args):
            bulk_semantic_weight = float(getattr(args, "ct_bulk_semantic_weight", 1.0))
            if bulk_semantic_weight != 0.0:
                occupancy_term = occupancy_term + bulk_semantic_weight * bulk_semantic_loss(
                    context,
                    args,
                    training_state,
                    boundary_band_distance,
                )
        else:
            occupancy_term = unified_phase_occupancy_loss(
                context,
                args,
                training_state,
                boundary_band_distance,
            )
            bulk_coverage_weight = float(getattr(args, "ct_bulk_coverage_weight", 0.0))
            if bulk_coverage_weight != 0.0:
                occupancy_term = occupancy_term + bulk_coverage_weight * masked_bulk_coverage_loss(
                    context,
                    args,
                    training_state,
                    boundary_band_distance,
                )
        bulk_containment_weight = _containment_ramp_weight(args, iteration) if _use_unified_compositor(args) else float(getattr(args, "ct_bulk_sdf_containment_weight", 0.0))
        if bulk_containment_weight != 0.0 and training_state.bulk_xyz.shape[0] > 0:
            occupancy_term = occupancy_term + bulk_containment_weight * bulk_sdf_containment_loss(
                context,
                args,
                training_state,
            )
        metrics_interval = int(getattr(args, "ct_false_hole_metrics_interval", 100))
        collect_metrics = metrics_interval > 0 and (int(iteration) <= 1 or int(iteration) % metrics_interval == 0)
        if collect_metrics:
            false_hole_term, false_hole_metrics = false_hole_geometry_loss(
                context,
                args,
                training_state,
                boundary_band_distance,
                return_metrics=True,
            )

    surface_term = zero_loss
    if training_state.surface_xyz.shape[0] > 0:
        if training_mode == "role_separated_joint":
            # Method C: surface only gets phase loss + geometry regularization
            # intensity loss does NOT reach surface (gradient isolation)
            phase_weight = float(getattr(args, "ct_surface_phase_loss_weight", 1.0))
            norm_weight = float(getattr(args, "ct_surface_normal_smooth_weight", 0.05))
            anchor_weight = float(getattr(args, "ct_surface_anchor_weight", 0.01))
            phase_term = phase_weight * surface_phase_loss(context, args, training_state)
            reg_term = zero_loss
            if args.ct_surface_regularizer_weight != 0.0:
                reg_term = args.ct_surface_regularizer_weight * surface_regularizer_loss(
                    context, args, training_state, iteration=iteration
                )
            surface_term = phase_term + reg_term
        elif args.ct_surface_regularizer_weight != 0.0:
            surface_term = surface_regularizer_loss(context, args, training_state, iteration=iteration)

    return CTLossTerms(
        volume=volume_loss,
        occupancy=occupancy_term,
        surface=surface_term,
        false_hole=false_hole_term,
        false_hole_metrics=false_hole_metrics,
    )
