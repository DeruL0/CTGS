from __future__ import annotations

import torch
import torch.nn.functional as F

from ct_pipeline.rendering.fields import density_to_occupancy, query_ct_density_from_state_by_region, query_ct_fields_unified
from ct_pipeline.training.bootstrap.analysis import _sample_coarse_sdf_normals
from ct_pipeline.training.bootstrap.context import CTTrainingBootstrap
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.objectives.bulk_losses import _material_membership_at
from ct_pipeline.training.objectives.sampling import _sample_filtered_from_candidate_sets
from ct_pipeline.training.utils import as_device_tensor


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

def surface_phase_loss(
    context: CTTrainingBootstrap,
    args,
    training_state,
    *,
    material_phase_pts: torch.Tensor | None = None,
    air_phase_pts: torch.Tensor | None = None,
    sample_count: int = 4096,
) -> torch.Tensor:
    """Phase loss for surface primitives using Phase 1 material/air samples.

    Pushes surface signed-distance D_s:
      - below -margin at material-side points
      - above +margin at air-side points

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

    if material_phase_pts is None and context.field_pools.get("support") is not None:
        material_phase_pts, _ = _sample_filtered_from_candidate_sets(
            (context.field_pools.get("material_deep_pool"), context.field_pools.get("support")),
            sample_count // 2,
            context,
            device=device,
            signed_distance_predicate=lambda sdf: sdf < -0.5,
        )
    if air_phase_pts is None:
        air_phase_pts, _ = _sample_filtered_from_candidate_sets(
            (context.field_pools.get("void_air"), context.field_pools.get("exterior_air_near_band")),
            sample_count // 2,
            context,
            device=device,
            signed_distance_predicate=lambda sdf: sdf > 0.5,
        )

    t_mat = _phase_term(material_phase_pts, +1.0)  # inside: D_s < -margin
    t_air = _phase_term(air_phase_pts, -1.0)  # outside: D_s > +margin
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
