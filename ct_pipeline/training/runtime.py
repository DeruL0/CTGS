from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ct_pipeline.training.bootstrap.context import CTTrainingBootstrap
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.objectives.bulk_losses import (
    _bulk_scale_adaptive_cap_loss,
    bulk_coverage_growth_loss,
    bulk_sdf_containment_loss,
    bulk_semantic_loss,
    bulk_void_leak_loss,
    masked_bulk_coverage_loss,
)
from ct_pipeline.training.objectives.modes import (
    _boundary_band_distance,
    _bulk_intensity_training_enabled,
    _containment_ramp_weight,
    _dual_separated_training_enabled,
    _intensity_geometry_flags,
    _use_unified_compositor,
)
from ct_pipeline.training.objectives.prediction import (
    compute_role_separated_ct_prediction,
    compute_surface_bounded_bulk_volume_prediction,
)
from ct_pipeline.training.objectives.surface_losses import (
    bulk_adaptive_anchor_losses,
    role_separated_intensity_loss,
    surface_phase_loss,
    surface_regularizer_loss,
)
from ct_pipeline.training.objectives.volume_losses import (
    _bulk_intensity_volume_loss,
    _dual_separated_volume_loss,
    _eagle_middle_patch_loss,
    _maybe_update_halfspace_tau,
    _surface_intensity_loss,
    unified_phase_occupancy_loss,
)
from ct_pipeline.training.sampling import (
    _ct_empty_points,
    _sample_signed_distance,
    _split_role_sample_counts,
    sample_bulk_volume_points_excluding_boundary,
    sample_surface_boundary_points,
)


@dataclass
class CTLossTerms:
    volume: torch.Tensor
    occupancy: torch.Tensor
    surface: torch.Tensor

    @property
    def render(self) -> torch.Tensor:
        return self.volume


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
    surface_term = zero_loss
    if training_state.surface_xyz.shape[0] > 0:
        if training_mode == "role_separated_joint":
            # Method C: surface only gets phase loss + geometry regularization
            # intensity loss does NOT reach surface (gradient isolation)
            phase_weight = float(getattr(args, "ct_surface_phase_loss_weight", 1.0))
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
    )
