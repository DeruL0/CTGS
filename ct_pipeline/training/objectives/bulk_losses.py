from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ct_pipeline.rendering.fields import (
    _query_ct_density_native_chunked,
    compose_signed_overlap_occupancy,
    density_to_occupancy,
    query_ct_density_from_state_by_region,
    query_ct_fields_unified,
)
from ct_pipeline.training.bootstrap import CTTrainingBootstrap, _sample_coarse_sdf_normals
from ct_pipeline.training.losses import asymmetric_binary_focal_loss, sample_volume_field
from ct_pipeline.training.objectives.modes import (
    _dual_separated_training_enabled,
    _dual_surface_inner_band,
    _use_unified_compositor,
)
from ct_pipeline.training.objectives.sampling import (
    _sample_filtered_from_candidate_sets,
    _sample_filtered_semantic_points,
)
from ct_pipeline.training.sampling import _candidate_count, _sample_occupancy_points, _sample_signed_distance


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
