from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ct_pipeline.rendering.fields import query_ct_fields_unified
from ct_pipeline.rendering.slices import _build_query_points_from_base, sample_gt_slice_patch
from ct_pipeline.training.bootstrap import CTTrainingBootstrap
from ct_pipeline.training.control import (
    _attenuation_only_bulk_gate_training_enabled,
    _attenuation_only_training_enabled,
)
from ct_pipeline.training.losses import eagle_patch_loss, sample_volume_field
from ct_pipeline.training.objectives.modes import (
    _bulk_halfspace_tau_current,
    _bulk_intensity_sample_mode,
    _bulk_intensity_training_enabled,
    _dual_surface_air_band,
    _intensity_geometry_flags,
    _surface_material_render_band,
)
from ct_pipeline.training.objectives.prediction import (
    _bulk_intensity_prediction,
    compute_raw_combined_ct_occupancy,
)
from ct_pipeline.training.objectives.sampling import (
    _phase_mask_from_analysis,
    _phase_occupancy_sdf_weights,
    _sample_air_points_with_void_bias,
    _sample_boundary_offset_shell_points,
    _sample_filtered_from_candidate_sets,
    _sample_support_membership,
    _split_phase_occupancy_sample_counts,
)
from ct_pipeline.training.sampling import (
    _cached_or_filter_candidates,
    _candidate_count,
    _ct_empty_points,
    _sample_occupancy_points,
    _sample_signed_distance,
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
