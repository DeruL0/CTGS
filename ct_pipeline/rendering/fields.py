from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ct_pipeline.rendering.bulk_support import resolve_bulk_containment_q
from utils.rotation_utils import quaternion_to_matrix


def is_bulk_intensity_field_mode(mode) -> bool:
    return str(mode) == "bulk_intensity_field"


def _as_query_points(points_xyz, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not isinstance(points_xyz, torch.Tensor):
        points_xyz = torch.as_tensor(points_xyz, dtype=dtype, device=device)
    else:
        points_xyz = points_xyz.to(device=device, dtype=dtype)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3).")
    return points_xyz


def query_ct_density_python(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    points_xyz,
    return_material_volume: bool = False,
    material_ids: torch.Tensor | None = None,
    chunk_size: int = 32768,
):
    points_xyz = _as_query_points(points_xyz, means.dtype, means.device)

    if points_xyz.numel() == 0:
        empty = torch.zeros((0,), dtype=means.dtype, device=means.device)
        if not return_material_volume:
            return empty
        return empty, empty.reshape(0, 0), np.zeros((0,), dtype=np.int32)

    if means.numel() == 0:
        empty = torch.zeros((points_xyz.shape[0],), dtype=points_xyz.dtype, device=points_xyz.device)
        if not return_material_volume:
            return empty
        return empty, empty.reshape(points_xyz.shape[0], 0), np.zeros((0,), dtype=np.int32)

    total_density = torch.zeros((points_xyz.shape[0],), dtype=means.dtype, device=means.device)

    material_volume = None
    material_labels = np.zeros((0,), dtype=np.int32)
    if return_material_volume:
        if material_ids is None:
            material_ids = torch.zeros((means.shape[0],), dtype=torch.long, device=means.device)
        else:
            material_ids = torch.as_tensor(material_ids, device=means.device, dtype=torch.long).reshape(-1)
        valid_materials = material_ids[material_ids >= 0]
        if valid_materials.numel() == 0:
            material_labels = np.asarray([0], dtype=np.int32)
            material_masks = [torch.ones_like(material_ids, dtype=torch.bool)]
        else:
            unique_materials = torch.unique(valid_materials)
            material_labels = unique_materials.detach().cpu().numpy().astype(np.int32)
            material_masks = [(material_ids == material).reshape(-1) for material in unique_materials]
        material_volume = torch.zeros((points_xyz.shape[0], len(material_masks)), dtype=means.dtype, device=means.device)
    else:
        material_masks = []

    gaussians_per_chunk = max(1, int(chunk_size // max(1, points_xyz.shape[0])))
    for start in range(0, means.shape[0], gaussians_per_chunk):
        end = min(start + gaussians_per_chunk, means.shape[0])
        mean_chunk = means[start:end]
        rotation_chunk = rotations[start:end]
        scale_chunk = scales[start:end]
        opacity_chunk = opacity[start:end]

        diff = points_xyz.unsqueeze(1) - mean_chunk.unsqueeze(0)
        local = torch.einsum("qci,cij->qcj", diff, rotation_chunk)
        normalized = local / scale_chunk.unsqueeze(0)
        exponent = -0.5 * torch.sum(normalized * normalized, dim=-1)
        chunk_density = torch.exp(exponent) * opacity_chunk.unsqueeze(0)
        total_density += chunk_density.sum(dim=1)

        if return_material_volume:
            global_indices = torch.arange(start, end, device=means.device)
            for material_index, material_mask in enumerate(material_masks):
                local_mask = material_mask[global_indices]
                if torch.any(local_mask):
                    material_volume[:, material_index] += chunk_density[:, local_mask].sum(dim=1)

    if not return_material_volume:
        return total_density
    return total_density, material_volume, material_labels


def query_ct_density(
    model,
    points_xyz,
    return_material_volume: bool = False,
    chunk_size: int = 32768,
):
    means = model.get_xyz
    points_xyz = _as_query_points(points_xyz, means.dtype, means.device)

    if return_material_volume:
        return query_ct_density_python(
            means,
            quaternion_to_matrix(model.get_rotation),
            model.get_scaling.clamp_min(1e-6),
            model.get_opacity.squeeze(-1),
            points_xyz,
            return_material_volume=True,
            material_ids=model.get_material_id.reshape(-1),
            chunk_size=chunk_size,
        )

    return query_ct_density_python(
        means,
        quaternion_to_matrix(model.get_rotation),
        model.get_scaling.clamp_min(1e-6),
        model.get_opacity.squeeze(-1),
        points_xyz,
        return_material_volume=False,
        chunk_size=chunk_size,
    )


def _query_ct_density_native_chunked(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    points_xyz: torch.Tensor,
    spatial_grid=None,
    support_extent=None,
    chunk_points: int = 2048,
) -> torch.Tensor:
    from ct_pipeline.backend.query import query_ct_density_native
    from torch.utils.checkpoint import checkpoint

    def _query_chunk(
        means_arg: torch.Tensor,
        rotations_arg: torch.Tensor,
        scales_arg: torch.Tensor,
        opacity_arg: torch.Tensor,
        points_arg: torch.Tensor,
    ) -> torch.Tensor:
        return query_ct_density_native(
            means_arg,
            rotations_arg,
            scales_arg,
            opacity_arg,
            points_arg,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
        )

    def _should_checkpoint() -> bool:
        return torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (means, rotations, scales, opacity)
        )

    if points_xyz.shape[0] <= int(chunk_points):
        if _should_checkpoint():
            return checkpoint(_query_chunk, means, rotations, scales, opacity, points_xyz, use_reentrant=False)
        return _query_chunk(means, rotations, scales, opacity, points_xyz)

    chunks = []
    for start in range(0, points_xyz.shape[0], int(chunk_points)):
        chunk = points_xyz[start : start + int(chunk_points)]
        if _should_checkpoint():
            chunks.append(checkpoint(_query_chunk, means, rotations, scales, opacity, chunk, use_reentrant=False))
        else:
            chunks.append(_query_chunk(means, rotations, scales, opacity, chunk))
    return torch.cat(chunks, dim=0)


def query_ct_density_from_state_by_region(
    training_state,
    points_xyz,
    region,
    detach: bool = False,
):
    if region in (0, "surface"):
        means = getattr(training_state, "surface_xyz")
        rotations = getattr(training_state, "surface_rotation_mats")
        scales = getattr(training_state, "surface_scales")
        opacity = getattr(training_state, "surface_opacity")
        spatial_grid = getattr(training_state, "surface_spatial_grid", None)
        support_extent = getattr(training_state, "surface_support_extent", None)
    elif region in (1, "bulk"):
        means = getattr(training_state, "bulk_xyz")
        rotations = getattr(training_state, "bulk_rotation_mats")
        scales = getattr(training_state, "bulk_scales")
        opacity = getattr(training_state, "bulk_opacity")
        spatial_grid = getattr(training_state, "bulk_spatial_grid", None)
        support_extent = getattr(training_state, "bulk_support_extent", None)
        q_cut = getattr(training_state, "bulk_query_q_support", None)
    else:
        raise ValueError("region must be 0/'surface' or 1/'bulk'.")

    points_xyz = _as_query_points(points_xyz, training_state.xyz.dtype, training_state.xyz.device)
    if means.numel() == 0 or points_xyz.numel() == 0:
        return torch.zeros((points_xyz.shape[0],), dtype=points_xyz.dtype, device=points_xyz.device)

    if detach:
        means = means.detach()
        rotations = rotations.detach()
        scales = scales.detach()
        opacity = opacity.detach()
        support_extent = None if support_extent is None else support_extent.detach()

    if region in (1, "bulk") and q_cut is not None:
        return _query_density_with_q_cutoff(
            means,
            rotations,
            scales.clamp_min(1e-6),
            opacity,
            points_xyz,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
            q_cut=float(q_cut),
        )

    return _query_ct_density_native_chunked(
        means,
        rotations,
        scales.clamp_min(1e-6),
        opacity,
        points_xyz,
        spatial_grid=spatial_grid,
        support_extent=support_extent,
    )


def density_to_occupancy(density: torch.Tensor) -> torch.Tensor:
    density = torch.as_tensor(density)
    return 1.0 - torch.exp(-density.clamp_min(0.0))


def bulk_intensity_readout(raw_bulk: torch.Tensor, den_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    raw_bulk = torch.as_tensor(raw_bulk)
    den_b = torch.as_tensor(den_b, device=raw_bulk.device, dtype=raw_bulk.dtype)
    return torch.where(
        den_b > float(eps),
        raw_bulk / den_b.clamp_min(float(eps)),
        torch.zeros_like(raw_bulk),
    )


def sdf_soft_bulk_display(
    A_b: torch.Tensor,
    signed_distance: torch.Tensor | None,
    intensity_air: float,
    *,
    tau: float = 0.5,
) -> torch.Tensor:
    """Geometry-conditioned display only: m_sdf * A_b + (1-m_sdf) * air."""
    if signed_distance is None:
        return A_b
    sdf = torch.as_tensor(signed_distance, device=A_b.device, dtype=A_b.dtype).reshape(-1)
    mask = torch.sigmoid(-sdf / max(float(tau), 1e-6))
    return mask * A_b + (1.0 - mask) * float(intensity_air)


def hard_masked_bulk_diagnostic(
    A_b: torch.Tensor,
    signed_distance: torch.Tensor | None,
    intensity_air: float,
) -> torch.Tensor:
    """Hard geometry mask diagnostic; not a bulk-capability score."""
    if signed_distance is None:
        return A_b
    sdf = torch.as_tensor(signed_distance, device=A_b.device, dtype=A_b.dtype).reshape(-1)
    return torch.where(sdf < 0.0, A_b, torch.full_like(A_b, float(intensity_air)))


def query_surface_signed_distance(
    training_state,
    points_xyz: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Differentiable signed-distance query from 2DGS-like surface primitives.

    For each query point x, computes a soft signed distance D_s(x) as the
    opacity-weighted average of per-primitive signed distances::

        d_i(x) = dot(x - p_i, n_i)           # signed distance to primitive plane
        t_i(x) = ||x - p_i - d_i*n_i||^2     # tangential distance squared
        w_i(x) = exp(-0.5 * t_i / sigma_t_i^2) * alpha_i
        D_s(x)  = sum_i(w_i * d_i) / (sum_i(w_i) + eps)

    D_s < 0 鈫?inside material;  D_s > 0 鈫?outside.

    Gradients flow through p_i (surface centres) and n_i (normals) so surface
    geometry is updated by any loss that uses D_s.

    Returns
    -------
    (N,) float32 tensor of signed distances at each query point.
    """
    points_xyz = _as_query_points(points_xyz, training_state.surface_xyz.dtype, training_state.surface_xyz.device)
    P = points_xyz.shape[0]
    device = points_xyz.device
    dtype = torch.float32

    means = training_state.surface_xyz        # (M, 3)   鈥?differentiable
    normals = training_state.surface_normals  # (M, 3)   鈥?differentiable (explicit)
    scales = training_state.surface_scales    # (M, 3)   tangential sigma
    opacity = training_state.surface_opacity  # (M,)
    M = means.shape[0]

    if M == 0:
        return torch.zeros((P,), dtype=dtype, device=device)

    # Subsample surface Gaussians to bound memory: O(P * M_sub * 3 * 4 bytes)
    # With M_sub=512 and P=4096: 4096*512*3*4 = 24 MB 鈥?safe.
    # Gradients still flow through the sampled subset.
    max_m = max(1, min(M, chunk_size))   # chunk_size repurposed as max_m_per_query
    if M > max_m:
        sub_idx = torch.randperm(M, device=device)[:max_m]
        means = means[sub_idx]
        normals_raw = normals[sub_idx]
        scales = scales[sub_idx]
        opacity = opacity[sub_idx]
    else:
        normals_raw = normals
    M_sub = means.shape[0]

    normals_f = F.normalize(normals_raw.to(dtype=dtype), dim=-1, eps=1e-8)
    sigma_t = scales.to(dtype=dtype).mean(dim=1).clamp_min(1e-6)  # (M_sub,)
    opacity_f = opacity.to(dtype=dtype)

    # Chunk over query points: keep (Cp * M_sub * 3 * 4) 鈮?32 MB
    max_bytes = 32 * 1024 * 1024
    q_chunk = max(1, max_bytes // (M_sub * 12 + 1))

    num_out = torch.zeros(P, dtype=dtype, device=device)
    den_out = torch.zeros(P, dtype=dtype, device=device)

    for p_start in range(0, P, q_chunk):
        p_end = min(p_start + q_chunk, P)
        q = points_xyz[p_start:p_end]                                 # (Cp, 3)

        diff = q.unsqueeze(1) - means.unsqueeze(0)                    # (Cp, M_sub, 3)
        d_i = (diff * normals_f.unsqueeze(0)).sum(-1)                 # (Cp, M_sub)
        tang = diff - d_i.unsqueeze(-1) * normals_f.unsqueeze(0)      # (Cp, M_sub, 3)
        t2 = (tang * tang).sum(-1)                                    # (Cp, M_sub)

        w_i = torch.exp(-0.5 * t2 / sigma_t.unsqueeze(0).square().clamp_min(1e-8))
        w_i = w_i * opacity_f.unsqueeze(0)

        num_out[p_start:p_end] = (w_i * d_i).sum(dim=1)
        den_out[p_start:p_end] = w_i.sum(dim=1)

    return num_out / den_out.clamp_min(float(eps))


def _config_float(config, name: str, default: float) -> float:
    if config is None:
        return float(default)
    return float(getattr(config, name, default))


def _config_optional_float(config, name: str, default):
    if config is None:
        return default
    value = getattr(config, name, default)
    return None if value is None else float(value)


def _config_str(config, name: str, default: str) -> str:
    if config is None:
        return str(default)
    return str(getattr(config, name, default))


def _use_bulk_intensity_field(training_state, config) -> bool:
    mode = _config_str(config, "ct_bulk_field_mode", "bulk_intensity_field")
    if not is_bulk_intensity_field_mode(mode):
        return False
    return (
        getattr(training_state, "bulk_attenuation", None) is not None
        and getattr(training_state, "bulk_center_sdf", None) is not None
        and getattr(training_state, "bulk_center_normals", None) is not None
    )


def _bulk_halfspace_tau(config) -> float:
    current = _config_optional_float(config, "ct_bulk_halfspace_tau_current", None)
    if current is not None:
        return max(float(current), 1e-6)
    return max(_config_float(config, "ct_bulk_halfspace_tau_init", 0.5), 1e-6)


def _query_local_candidate_pairs(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    query_points: torch.Tensor,
    *,
    spatial_grid=None,
    support_extent=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = query_points.device
    if means.numel() == 0 or query_points.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    if spatial_grid is None or support_extent is None or not query_points.is_cuda:
        query_ids = torch.arange(query_points.shape[0], device=device, dtype=torch.long).repeat_interleave(int(means.shape[0]))
        gaussian_ids = torch.arange(means.shape[0], device=device, dtype=torch.long).repeat(int(query_points.shape[0]))
        return query_ids, gaussian_ids

    from ct_pipeline.backend.core import get_ct_native_extension

    native = get_ct_native_extension()
    with torch.no_grad():
        dummy_weight = torch.ones((means.shape[0],), dtype=means.dtype, device=means.device)
        _density, query_offsets, query_gaussian_ids = native.query_density_local_forward(
            means.detach().contiguous(),
            rotations.detach().contiguous(),
            scales.detach().contiguous(),
            dummy_weight.contiguous(),
            support_extent.detach().contiguous(),
            query_points.detach().contiguous(),
            spatial_grid.world_min.contiguous(),
            spatial_grid.grid_dims.contiguous(),
            float(spatial_grid.cell_size),
            spatial_grid.cell_offsets.contiguous(),
            spatial_grid.cell_gaussian_ids.contiguous(),
        )
    counts = (query_offsets[1:] - query_offsets[:-1]).to(dtype=torch.long)
    if counts.numel() == 0 or int(counts.sum().item()) == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty
    query_ids = torch.repeat_interleave(
        torch.arange(query_points.shape[0], device=device, dtype=torch.long),
        counts,
    )
    return query_ids, query_gaussian_ids.to(device=device, dtype=torch.long)


def _query_density_with_q_cutoff(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    query_points: torch.Tensor,
    *,
    spatial_grid=None,
    support_extent=None,
    q_cut: float,
) -> torch.Tensor:
    if spatial_grid is not None and support_extent is not None and query_points.is_cuda:
        from ct_pipeline.backend.query import has_ct_native_qcut_density_query, query_ct_density_qcut_native

        if has_ct_native_qcut_density_query():
            return query_ct_density_qcut_native(
                means,
                rotations,
                scales,
                opacity,
                query_points,
                spatial_grid=spatial_grid,
                support_extent=support_extent,
                q_cut=float(q_cut),
            ).to(dtype=torch.float32)

    zeros = torch.zeros((query_points.shape[0],), dtype=torch.float32, device=query_points.device)
    query_ids, gaussian_ids = _query_local_candidate_pairs(
        means,
        rotations,
        scales,
        query_points,
        spatial_grid=spatial_grid,
        support_extent=support_extent,
    )
    if gaussian_ids.numel() == 0:
        return zeros
    query_points_sel = query_points.index_select(0, query_ids)
    means_sel = means.index_select(0, gaussian_ids)
    rotations_sel = rotations.index_select(0, gaussian_ids)
    scales_sel = scales.index_select(0, gaussian_ids).clamp_min(1e-6)
    diff = query_points_sel - means_sel
    local = torch.einsum("ni,nij->nj", diff, rotations_sel)
    q = torch.sum(torch.square(local / scales_sel), dim=-1)
    gaussian = torch.where(q <= float(q_cut), torch.exp(-0.5 * q), torch.zeros_like(q))
    weighted = gaussian * opacity.index_select(0, gaussian_ids).to(dtype=gaussian.dtype)
    return zeros.index_add(0, query_ids, weighted.to(dtype=zeros.dtype))


def _query_bulk_intensity_field(
    training_state,
    points_xyz: torch.Tensor,
    config,
    *,
    train_xyz: bool = False,
    train_scale: bool = False,
    scale_grad: float = 1.0,
    train_opacity: bool = False,
    opacity_grad: float = 1.0,
    train_attenuation: bool = True,
    material_membership: torch.Tensor | None = None,
    apply_gate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bulk intensity field.

    Inside/outside ownership is decided per query point:
      * ``apply_gate=False``           -> ungated (honest raw bulk; for display/eval)
      * ``material_membership`` given  -> hard mask membership gate (1 inside / 0 outside)
      * otherwise                      -> center half-space fallback for direct queries

    The mask membership replaces the SDF half-space gate so inside/outside is
    unambiguous ("鏈夊氨鏄湁"), avoiding the linearized-gradient mislabeling that
    leaks bulk into cavities. Display passes ``apply_gate=False`` so leaks show.
    """
    means = getattr(training_state, "bulk_xyz", None)
    zeros = torch.zeros((points_xyz.shape[0],), dtype=torch.float32, device=points_xyz.device)
    if means is None or means.numel() == 0 or points_xyz.numel() == 0:
        return zeros, zeros

    means = _grad_mixed_tensor(means, bool(train_xyz))
    bulk_scales = _grad_mixed_tensor(getattr(training_state, "bulk_scales"), bool(train_scale), float(scale_grad)).clamp_min(1e-6)
    # Clamp scale when adaptive bulk mode is active.
    # ct_bulk_sigma_min_mm / ct_bulk_sigma_max_mm are in world units (mm).
    # Default: no clamp (0 / inf).
    sigma_min_mm = _config_float(config, "ct_bulk_sigma_min_mm", 0.0)
    sigma_max_mm = _config_float(config, "ct_bulk_sigma_max_mm", 1e9)
    if sigma_min_mm > 0.0 or sigma_max_mm < 1e8:
        bulk_scales = bulk_scales.clamp(min=max(sigma_min_mm, 1e-6), max=sigma_max_mm)
    bulk_opacity = getattr(training_state, "bulk_opacity", None)
    if bulk_opacity is None or bulk_opacity.numel() == 0:
        bulk_opacity = torch.ones((means.shape[0],), dtype=means.dtype, device=means.device)
    else:
        bulk_opacity = bulk_opacity.to(device=means.device, dtype=means.dtype).reshape(-1)
        bulk_opacity = _grad_mixed_tensor(bulk_opacity, bool(train_opacity), float(opacity_grad)).clamp(0.0, 1.0)
    attenuation = getattr(training_state, "bulk_attenuation")
    if attenuation is None or attenuation.numel() == 0:
        return zeros, zeros
    attenuation = attenuation.to(device=means.device, dtype=means.dtype).reshape(-1)
    if not train_attenuation:
        attenuation = attenuation.detach()
    center_sdf = getattr(training_state, "bulk_center_sdf")
    center_normals = getattr(training_state, "bulk_center_normals")
    if center_sdf is None or center_normals is None:
        return zeros, zeros
    center_sdf = center_sdf.to(device=means.device, dtype=means.dtype).reshape(-1)
    center_normals = F.normalize(center_normals.to(device=means.device, dtype=means.dtype).reshape(-1, 3), dim=-1, eps=1e-6)
    rotations = getattr(training_state, "bulk_rotation_mats")
    spatial_grid = getattr(training_state, "bulk_spatial_grid", None)
    support_extent = getattr(training_state, "bulk_support_extent", None)
    tau = _bulk_halfspace_tau(config)
    q_cut = resolve_bulk_containment_q(config)
    skip_depth = max(_config_float(config, "ct_bulk_halfspace_skip_depth", 2.0), 0.0)
    chunk_points = max(1, int(_config_float(config, "ct_bulk_intensity_query_chunk_points", 512)))

    def _query_chunk(
        points_chunk: torch.Tensor,
        membership_chunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial_grid is not None and support_extent is not None and points_chunk.is_cuda:
            from ct_pipeline.backend.query import has_ct_native_bulk_intensity_query, query_bulk_intensity_native

            if has_ct_native_bulk_intensity_query():
                return query_bulk_intensity_native(
                    means,
                    rotations,
                    bulk_scales,
                    bulk_opacity,
                    attenuation,
                    center_sdf,
                    center_normals,
                    points_chunk,
                    spatial_grid=spatial_grid,
                    support_extent=support_extent,
                    q_cut=float(q_cut),
                    tau=float(tau),
                    skip_depth=float(skip_depth),
                    material_membership=membership_chunk,
                    apply_gate=bool(apply_gate),
                )

        zeros_chunk = torch.zeros((points_chunk.shape[0],), dtype=torch.float32, device=points_chunk.device)
        query_ids, gaussian_ids = _query_local_candidate_pairs(
            means,
            rotations,
            bulk_scales,
            points_chunk,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
        )
        if gaussian_ids.numel() == 0:
            return zeros_chunk, zeros_chunk

        query_points = points_chunk.index_select(0, query_ids)
        means_sel = means.index_select(0, gaussian_ids)
        rotations_sel = rotations.index_select(0, gaussian_ids)
        scales_sel = bulk_scales.index_select(0, gaussian_ids).clamp_min(1e-6)
        diff = query_points - means_sel
        local = torch.einsum("ni,nij->nj", diff, rotations_sel)
        q = torch.sum(torch.square(local / scales_sel), dim=-1)
        gaussian = torch.where(q <= float(q_cut), torch.exp(-0.5 * q), torch.zeros_like(q))

        if not apply_gate:
            # ungated: honest raw bulk (display / void-leak diagnostic)
            gate = torch.ones_like(gaussian)
        elif membership_chunk is not None:
            # hard mask membership: gate is the per-query material indicator (1 in / 0 out)
            gate = membership_chunk.index_select(0, query_ids).to(dtype=gaussian.dtype).clamp(0.0, 1.0)
        else:
            # Direct-query fallback when no material mask membership was supplied.
            center_sdf_sel = center_sdf.index_select(0, gaussian_ids)
            linearized_sdf = center_sdf_sel + torch.sum(
                diff * center_normals.index_select(0, gaussian_ids),
                dim=-1,
            )
            deep_mask = center_sdf_sel <= -float(skip_depth)
            gate = torch.sigmoid(-linearized_sdf / float(tau))
            if torch.any(deep_mask):
                gate = torch.where(deep_mask, torch.ones_like(gate), gate)

        kernel = gaussian * gate

        gated_kernel = kernel * bulk_opacity.index_select(0, gaussian_ids)
        contribution = gated_kernel * attenuation.index_select(0, gaussian_ids)
        mu_chunk = zeros_chunk.index_add(0, query_ids, contribution.to(dtype=zeros_chunk.dtype))
        kernel_chunk = zeros_chunk.index_add(0, query_ids, gated_kernel.to(dtype=zeros_chunk.dtype))
        return mu_chunk, kernel_chunk

    membership = None
    if apply_gate and material_membership is not None:
        membership = material_membership.to(device=points_xyz.device, dtype=torch.float32).reshape(-1)
        if membership.shape[0] != points_xyz.shape[0]:
            raise ValueError("material_membership must match points_xyz length.")

    if points_xyz.shape[0] <= chunk_points:
        return _query_chunk(points_xyz, membership)

    mu_parts = []
    kernel_parts = []
    for start in range(0, int(points_xyz.shape[0]), int(chunk_points)):
        chunk = points_xyz[start : start + int(chunk_points)]
        membership_chunk = None if membership is None else membership[start : start + int(chunk_points)]
        mu_chunk, kernel_chunk = _query_chunk(chunk, membership_chunk)
        mu_parts.append(mu_chunk)
        kernel_parts.append(kernel_chunk)
    return torch.cat(mu_parts, dim=0), torch.cat(kernel_parts, dim=0)


def query_bulk_anisotropic_density(
    points_xyz,
    training_state,
    config=None,
    *,
    apply_gate: bool = False,
    material_membership: torch.Tensor | None = None,
    return_raw: bool = False,
    chunk_points: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Query the same anisotropic bulk kernel used by bulk A_b readout."""
    points_xyz = _as_query_points(points_xyz, training_state.xyz.dtype, training_state.xyz.device)
    if not bool(apply_gate) and material_membership is None and not bool(return_raw):
        means = getattr(training_state, "bulk_xyz", None)
        zeros = torch.zeros((points_xyz.shape[0],), dtype=torch.float32, device=points_xyz.device)
        if means is None or means.numel() == 0 or points_xyz.numel() == 0:
            return zeros
        opacity = getattr(training_state, "bulk_opacity", None)
        if opacity is None or opacity.numel() == 0:
            opacity = torch.ones((means.shape[0],), dtype=means.dtype, device=means.device)
        else:
            opacity = opacity.to(device=means.device, dtype=means.dtype).reshape(-1).clamp(0.0, 1.0)

        rotations = getattr(training_state, "bulk_rotation_mats")
        scales = getattr(training_state, "bulk_scales").clamp_min(1e-6)
        spatial_grid = getattr(training_state, "bulk_spatial_grid", None)
        support_extent = getattr(training_state, "bulk_support_extent", None)
        q_cut = resolve_bulk_containment_q(config)
        native_available = False
        if spatial_grid is not None and support_extent is not None and points_xyz.is_cuda:
            from ct_pipeline.backend.query import has_ct_native_qcut_density_query

            native_available = has_ct_native_qcut_density_query()

        if chunk_points is None:
            chunk_points = 65536 if native_available else int(_config_float(config, "ct_bulk_intensity_query_chunk_points", 512))
        chunk_points = max(1, int(chunk_points))

        def _query_density_chunk(points_chunk: torch.Tensor) -> torch.Tensor:
            return _query_density_with_q_cutoff(
                means,
                rotations,
                scales,
                opacity,
                points_chunk,
                spatial_grid=spatial_grid,
                support_extent=support_extent,
                q_cut=float(q_cut),
            ).to(dtype=torch.float32)

        if points_xyz.shape[0] <= chunk_points:
            return _query_density_chunk(points_xyz)
        return torch.cat(
            [_query_density_chunk(points_xyz[start : start + chunk_points]) for start in range(0, int(points_xyz.shape[0]), chunk_points)],
            dim=0,
        )

    raw, den = _query_bulk_intensity_field(
        training_state,
        points_xyz,
        config,
        train_xyz=False,
        train_scale=False,
        train_opacity=False,
        train_attenuation=False,
        material_membership=material_membership,
        apply_gate=bool(apply_gate),
    )
    if bool(return_raw):
        return raw, den
    return den


def _region_state_terms(training_state, region: str):
    if region == "surface":
        return (
            training_state.surface_xyz,
            training_state.surface_rotation_mats,
            training_state.surface_scales,
            training_state.surface_opacity,
            getattr(training_state, "surface_ct_value", None),
            getattr(training_state, "surface_spatial_grid", None),
            getattr(training_state, "surface_support_extent", None),
        )
    if region == "bulk":
        return (
            training_state.bulk_xyz,
            training_state.bulk_rotation_mats,
            training_state.bulk_scales,
            training_state.bulk_opacity,
            getattr(training_state, "bulk_ct_value", None),
            getattr(training_state, "bulk_spatial_grid", None),
            getattr(training_state, "bulk_support_extent", None),
        )
    raise ValueError("region must be 'surface' or 'bulk'.")


def _grad_mixed_tensor(tensor: torch.Tensor, allow_grad: bool, grad_scale: float = 1.0) -> torch.Tensor:
    if allow_grad:
        if float(grad_scale) == 1.0:
            return tensor
        return tensor.detach() + float(grad_scale) * (tensor - tensor.detach())
    return tensor.detach()


def _query_region_density_for_unified(
    training_state,
    points_xyz: torch.Tensor,
    region: str,
    *,
    train_xyz: bool = False,
    train_rotation: bool = False,
    train_scale: bool = False,
    train_opacity: bool = False,
    scale_grad: float = 1.0,
    opacity_grad: float = 1.0,
) -> torch.Tensor:
    means, rotations, scales, opacity, _ct_value, spatial_grid, support_extent = _region_state_terms(training_state, region)
    zeros = torch.zeros((points_xyz.shape[0],), dtype=points_xyz.dtype, device=points_xyz.device)
    if means.numel() == 0 or points_xyz.numel() == 0:
        return zeros
    means = _grad_mixed_tensor(means, bool(train_xyz))
    rotations = _grad_mixed_tensor(rotations, bool(train_rotation))
    scales = _grad_mixed_tensor(scales, bool(train_scale), float(scale_grad)).clamp_min(1e-6)
    opacity = _grad_mixed_tensor(opacity, bool(train_opacity), float(opacity_grad))
    support_extent = None if support_extent is None else support_extent.detach()
    q_cut = getattr(training_state, "bulk_query_q_support", None) if region == "bulk" else None
    if q_cut is not None:
        return _query_density_with_q_cutoff(
            means,
            rotations,
            scales,
            opacity,
            points_xyz,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
            q_cut=float(q_cut),
        ).to(dtype=torch.float32)
    return _query_ct_density_native_chunked(
        means,
        rotations,
        scales,
        opacity,
        points_xyz,
        spatial_grid=spatial_grid,
        support_extent=support_extent,
    ).to(dtype=torch.float32)


def _query_region_value_for_unified(
    training_state,
    points_xyz: torch.Tensor,
    region: str,
    *,
    train_ct_value: bool = True,
    detach_geometry: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    means, rotations, scales, opacity, ct_value, spatial_grid, support_extent = _region_state_terms(training_state, region)
    zeros = torch.zeros((points_xyz.shape[0],), dtype=points_xyz.dtype, device=points_xyz.device)
    if means.numel() == 0 or ct_value is None or ct_value.numel() == 0 or points_xyz.numel() == 0:
        return zeros, zeros
    if detach_geometry:
        means = means.detach()
        rotations = rotations.detach()
        scales = scales.detach()
        opacity = opacity.detach()
        support_extent = None if support_extent is None else support_extent.detach()
    ct_value = ct_value.to(device=opacity.device, dtype=opacity.dtype)
    if not train_ct_value:
        ct_value = ct_value.detach()
    q_cut = getattr(training_state, "bulk_query_q_support", None) if region == "bulk" else None
    if q_cut is not None:
        denominator = _query_density_with_q_cutoff(
            means,
            rotations,
            scales.clamp_min(1e-6),
            opacity,
            points_xyz,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
            q_cut=float(q_cut),
        ).to(dtype=torch.float32)
        numerator = _query_density_with_q_cutoff(
            means,
            rotations,
            scales.clamp_min(1e-6),
            opacity * ct_value,
            points_xyz,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
            q_cut=float(q_cut),
        ).to(dtype=torch.float32)
    else:
        denominator = _query_ct_density_native_chunked(
            means,
            rotations,
            scales.clamp_min(1e-6),
            opacity,
            points_xyz,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
        ).to(dtype=torch.float32)
        numerator = _query_ct_density_native_chunked(
            means,
            rotations,
            scales.clamp_min(1e-6),
            opacity * ct_value,
            points_xyz,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
        ).to(dtype=torch.float32)
    return numerator / denominator.detach().clamp_min(float(eps)), denominator


def _query_surface_intensity_field(
    training_state,
    points_xyz: torch.Tensor,
    *,
    train_intensity: bool = True,
    detach_geometry: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    means = getattr(training_state, "surface_xyz")
    rotations = getattr(training_state, "surface_rotation_mats")
    scales = getattr(training_state, "surface_scales")
    opacity = getattr(training_state, "surface_opacity")
    intensity = getattr(training_state, "surface_attenuation", None)
    if intensity is None or intensity.numel() == 0:
        intensity = getattr(training_state, "surface_ct_value", None)

    zeros = torch.zeros((points_xyz.shape[0],), dtype=torch.float32, device=points_xyz.device)
    if means.numel() == 0 or intensity is None or intensity.numel() == 0:
        return zeros, zeros

    if detach_geometry:
        means = means.detach()
        rotations = rotations.detach()
        scales = scales.detach()
        opacity = opacity.detach()
        support_extent = getattr(training_state, "surface_support_extent", None)
        support_extent = None if support_extent is None else support_extent.detach()
    else:
        support_extent = getattr(training_state, "surface_support_extent", None)

    intensity = intensity.to(device=opacity.device, dtype=opacity.dtype).reshape(-1)
    if not train_intensity:
        intensity = intensity.detach()

    denominator = _query_ct_density_native_chunked(
        means,
        rotations,
        scales.clamp_min(1e-6),
        opacity,
        points_xyz,
        spatial_grid=getattr(training_state, "surface_spatial_grid", None),
        support_extent=support_extent,
    ).to(dtype=torch.float32)
    numerator = _query_ct_density_native_chunked(
        means,
        rotations,
        scales.clamp_min(1e-6),
        opacity * intensity,
        points_xyz,
        spatial_grid=getattr(training_state, "surface_spatial_grid", None),
        support_extent=support_extent,
    ).to(dtype=torch.float32)
    return numerator / denominator.detach().clamp_min(float(eps)), denominator


def query_ct_fields_unified(
    points_xyz,
    training_state,
    signed_distance: torch.Tensor | None = None,
    config=None,
    *,
    intensity_air: float = 0.0,
    include_surface: bool = True,
    bulk_train_xyz: bool = False,
    bulk_train_rotation: bool = False,
    bulk_train_scale: bool = False,
    bulk_train_opacity: bool = False,
    bulk_scale_grad: float = 1.0,
    surface_train_xyz: bool = False,
    surface_train_rotation: bool = False,
    surface_train_scale: bool = False,
    surface_train_opacity: bool = False,
    surface_scale_grad: float = 1.0,
    train_ct_value: bool = True,
    detach_value_geometry: bool = True,
    material_membership: torch.Tensor | None = None,
    apply_bulk_gate: bool = True,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Unified CTGS-v2 compositor.

    Bulk occupancy remains intentionally raw: ``occ_b == occ_b_raw``. In
    bulk-intensity mode, bulk intensity is the partition-of-unity readout
    ``A_b = 危 a_i K_i / 危 K_i``. The SDF gate only defines one-sided surface
    material contribution, so rendering and semantic air/void checks cannot
    hide overflowing bulk behind an SDF mask.
    """
    points_xyz = _as_query_points(points_xyz, training_state.xyz.dtype, training_state.xyz.device)
    device = points_xyz.device
    dtype = torch.float32
    zeros = torch.zeros((points_xyz.shape[0],), dtype=dtype, device=device)

    bulk_intensity_mode = _use_bulk_intensity_field(training_state, config)
    if bulk_intensity_mode:
        raw_bulk_b, kernel_b = _query_bulk_intensity_field(
            training_state,
            points_xyz,
            config,
            train_xyz=bulk_train_xyz,
            train_scale=bulk_train_scale,
            scale_grad=bulk_scale_grad,
            train_opacity=bulk_train_opacity,
            train_attenuation=True,
            material_membership=material_membership,
            apply_gate=apply_bulk_gate,
        )
        rho_b = kernel_b.to(dtype=dtype)
        occ_b_raw = density_to_occupancy(rho_b).to(dtype=dtype)
        occ_b = occ_b_raw
        den_b = kernel_b.to(dtype=dtype)
        c_b = bulk_intensity_readout(raw_bulk_b.to(dtype=dtype), den_b, eps=eps).to(dtype=dtype)
    else:
        rho_b = _query_region_density_for_unified(
            training_state,
            points_xyz,
            "bulk",
            train_xyz=bulk_train_xyz,
            train_rotation=bulk_train_rotation,
            train_scale=bulk_train_scale,
            train_opacity=bulk_train_opacity,
            scale_grad=bulk_scale_grad,
        )
        occ_b_raw = density_to_occupancy(rho_b).to(dtype=dtype)
        occ_b = occ_b_raw

    if include_surface:
        rho_s = _query_region_density_for_unified(
            training_state,
            points_xyz,
            "surface",
            train_xyz=surface_train_xyz,
            train_rotation=surface_train_rotation,
            train_scale=surface_train_scale,
            train_opacity=surface_train_opacity,
            scale_grad=surface_scale_grad,
        )
        occ_s_raw = density_to_occupancy(rho_s).to(dtype=dtype)
    else:
        rho_s = zeros
        occ_s_raw = zeros

    sdf = None
    if signed_distance is None:
        g_s_geom = torch.ones_like(occ_b_raw)
        g_s_mat = torch.ones_like(occ_b_raw) if include_surface else torch.zeros_like(occ_b_raw)
    else:
        sdf = torch.as_tensor(signed_distance, device=device, dtype=dtype).reshape(-1)
        if sdf.shape[0] != points_xyz.shape[0]:
            raise ValueError("signed_distance must match points_xyz length.")
        band = _config_float(config, "ct_boundary_band", _config_float(config, "ct_boundary_band_voxels", 1.5))
        sigma_default = max(0.5, 0.5 * float(band))
        sigma_s = _config_optional_float(config, "ct_surface_gate_sigma", None)
        if sigma_s is None:
            sigma_s = _config_optional_float(config, "ct_surface_material_gate_sigma", None)
        if sigma_s is None:
            sigma_s = sigma_default
        sigma_s = max(float(sigma_s), 1e-6)
        delta = _config_float(config, "ct_surface_material_delta", 0.25)
        tau = max(_config_float(config, "ct_surface_material_tau", 0.25), 1e-6)
        g_s_geom = torch.exp(-0.5 * torch.square(sdf / sigma_s))
        g_s_mat = g_s_geom * torch.sigmoid((-sdf + float(delta)) / float(tau))
        if not include_surface:
            g_s_mat = torch.zeros_like(g_s_mat)

    occ_s = (g_s_mat * occ_s_raw).clamp(0.0, 1.0)
    occ = (occ_b + (1.0 - occ_b).clamp(0.0, 1.0) * occ_s).clamp(0.0, 1.0)

    if bulk_intensity_mode:
        if include_surface:
            c_s, den_s = _query_surface_intensity_field(
                training_state,
                points_xyz,
                train_intensity=train_ct_value,
                detach_geometry=detach_value_geometry,
                eps=eps,
            )
        else:
            c_s = zeros
            den_s = zeros
        # Surface has its own diagnostic/color intensity, but it still does not
        # participate in the bulk A_b path.
        c = c_b
        i_pred = c_b
    else:
        c_b, den_b = _query_region_value_for_unified(
            training_state,
            points_xyz,
            "bulk",
            train_ct_value=train_ct_value,
            detach_geometry=detach_value_geometry,
            eps=eps,
        )
        if include_surface:
            c_s, den_s = _query_region_value_for_unified(
                training_state,
                points_xyz,
                "surface",
                train_ct_value=train_ct_value,
                detach_geometry=detach_value_geometry,
                eps=eps,
            )
        else:
            c_s = zeros
            den_s = zeros

        value_weight_b = occ_b.detach()
        value_weight_s = occ_s.detach()
        c = (value_weight_b * c_b + value_weight_s * c_s) / (value_weight_b + value_weight_s).clamp_min(float(eps))
        i_pred = float(intensity_air) + (c - float(intensity_air)) * occ

    surface_mask = g_s_mat.to(dtype=dtype)
    if bulk_intensity_mode:
        bulk_mask = torch.ones_like(occ_b_raw, dtype=dtype, device=device)
        i_raw = i_pred.to(dtype=dtype)
        i_bulk = i_raw
        i_surface = (float(intensity_air) + occ_s * (c_s - float(intensity_air))).to(dtype=dtype)
        tau_pv = _config_float(config, "ct_sdf_display_tau", _config_float(config, "ct_dual_soft_sigma", 0.5))
        i_sdf_soft = sdf_soft_bulk_display(i_raw, sdf, float(intensity_air), tau=tau_pv).to(dtype=dtype)
        i_hard = hard_masked_bulk_diagnostic(i_raw, sdf, float(intensity_air)).to(dtype=dtype)
        # Compatibility keys are aliases; no surface CT value is blended here.
        i_hybrid_hard = i_hard
        i_hybrid_soft = i_sdf_soft
    else:
        if sdf is None:
            bulk_mask = torch.ones_like(occ_b_raw, dtype=dtype, device=device)
            i_hybrid_hard = i_pred
            i_hybrid_soft = i_pred
        else:
            bulk_epsilon = _config_float(config, "ct_bulk_renderer_epsilon", 0.5)
            bulk_tau = max(_config_float(config, "ct_bulk_renderer_tau", 0.25), 1e-6)
            bulk_mask = torch.sigmoid((-sdf - float(bulk_epsilon)) / float(bulk_tau)).to(dtype=dtype)
            surface_air_band = max(_config_float(config, "ct_dual_surface_air_band", 0.25), 0.0)
            air_value = torch.full_like(i_pred, float(intensity_air))
            i_surface = float(intensity_air) + surface_mask * (c_s - float(intensity_air))
            i_bulk = float(intensity_air) + bulk_mask * (c_b - float(intensity_air))
            surface_region = (sdf >= 0.0) & (sdf <= float(surface_air_band))
            bulk_region = sdf < 0.0
            i_hybrid_hard = torch.where(
                surface_region,
                i_surface,
                torch.where(bulk_region, i_bulk, air_value),
            )
            soft_sigma = max(_config_float(config, "ct_dual_soft_sigma", max(0.25, float(surface_air_band))), 1e-6)
            surface_weight = torch.exp(-0.5 * torch.square(sdf / float(soft_sigma))).to(dtype=dtype)
            soft_band = surface_weight * i_surface + (1.0 - surface_weight) * i_bulk
            i_hybrid_soft = torch.where(
                surface_region,
                soft_band,
                torch.where(bulk_region, i_bulk, air_value),
            )
        i_surface = float(intensity_air) + surface_mask * (c_s - float(intensity_air))
        i_bulk = float(intensity_air) + bulk_mask * (c_b - float(intensity_air))
        i_raw = i_pred.to(dtype=dtype)
        i_sdf_soft = i_hybrid_soft
        i_hard = i_hybrid_hard

    return {
        "rho_s": rho_s,
        "rho_b": rho_b,
        "W_s": rho_s,
        "W_b": rho_b,
        "occ_s_raw": occ_s_raw,
        "occ_b_raw": occ_b_raw,
        "g_s_geom": g_s_geom.to(dtype=dtype),
        "g_s_mat": g_s_mat.to(dtype=dtype),
        "M_s": surface_mask.to(dtype=dtype),
        "M_b": bulk_mask.to(dtype=dtype),
        "occ_s": occ_s.to(dtype=dtype),
        "occ_b": occ_b.to(dtype=dtype),
        "occ": occ.to(dtype=dtype),
        "c_s": c_s.to(dtype=dtype),
        "c_b": c_b.to(dtype=dtype),
        "A_s": c_s.to(dtype=dtype),
        "A_b": c_b.to(dtype=dtype),
        "den_s": den_s.to(dtype=dtype),
        "den_b": den_b.to(dtype=dtype),
        "c": c.to(dtype=dtype),
        "I_pred": i_pred.to(dtype=dtype),
        "I_s": i_surface.to(dtype=dtype),
        "I_b": i_bulk.to(dtype=dtype),
        "I_raw": i_raw.to(dtype=dtype),
        "I_sdf_soft": i_sdf_soft.to(dtype=dtype),
        "I_hard": i_hard.to(dtype=dtype),
        "I_b_raw": raw_bulk_b.to(dtype=dtype) if bulk_intensity_mode else rho_b.to(dtype=dtype),
        "I_hybrid_hard": i_hybrid_hard.to(dtype=dtype),
        "I_hybrid_soft": i_hybrid_soft.to(dtype=dtype),
    }


def signed_overlap_sdf_gates(
    signed_distance: torch.Tensor | None,
    boundary_band_distance: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    surface_material_gate_sigma: float | None = None,
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
    """Ownership weights for boundary-shell surface detail.

    Bulk is no longer clipped by SDF. Material/air membership is supervised by
    mask-based bulk coverage and air-exclusion losses instead. SDF remains only
    as a soft surface ownership hint so thin surface detail does not dominate
    deep material intensity.
    """
    if signed_distance is None:
        return 1.0, 1.0
    sdf = signed_distance.to(device=device, dtype=dtype).reshape(-1)
    band = max(float(boundary_band_distance), 1e-6)
    if surface_material_gate_sigma is None:
        surface_sigma_material = max(0.5, 0.5 * band)
    else:
        surface_sigma_material = max(float(surface_material_gate_sigma), 1e-6)
    material_side = sdf <= 0
    bulk_gate = torch.ones_like(sdf, dtype=dtype, device=device)
    surface_gate = torch.where(
        material_side,
        torch.exp(-0.5 * torch.square(sdf / surface_sigma_material)),
        torch.zeros_like(sdf),
    )
    return bulk_gate, surface_gate


def compose_signed_overlap_occupancy(
    bulk_density: torch.Tensor,
    surface_density: torch.Tensor | None,
    signed_distance: torch.Tensor | None,
    boundary_band_distance: float,
    surface_material_gate_sigma: float | None = None,
    material_compose_mode: str = "bulk_first_material",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compose bulk and surface occupancy without SDF clipping bulk.

    Bulk and surface first become independent occupancies. On material-side
    samples, bulk can take first claim so a thin surface shell does not erase
    material intensity. Air/void suppression is handled by mask-supervised
    occupancy losses, not by SDF ownership.
    """
    bulk_occupancy = density_to_occupancy(bulk_density).to(dtype=torch.float32)
    bulk_gate, surface_gate = signed_overlap_sdf_gates(
        signed_distance,
        boundary_band_distance,
        dtype=bulk_occupancy.dtype,
        device=bulk_occupancy.device,
        surface_material_gate_sigma=surface_material_gate_sigma,
    )
    bulk_shell = bulk_gate * bulk_occupancy
    if surface_density is None:
        return bulk_shell.to(dtype=torch.float32), bulk_shell.to(dtype=torch.float32), None

    surface_occupancy = density_to_occupancy(surface_density).to(dtype=torch.float32)
    surface_raw = (surface_gate * surface_occupancy).clamp(0.0, 1.0)
    if material_compose_mode == "bulk_first_material" and torch.is_tensor(bulk_gate):
        material_side = bulk_gate > 0.5
        bulk_first_bulk = bulk_shell
        bulk_first_surface = (1.0 - bulk_shell).clamp(0.0, 1.0) * surface_raw
        surface_first_bulk = (1.0 - surface_raw).clamp(0.0, 1.0) * bulk_shell
        surface_contribution = torch.where(material_side, bulk_first_surface, surface_raw)
        bulk_contribution = torch.where(material_side, bulk_first_bulk, surface_first_bulk)
    else:
        surface_contribution = surface_raw
        bulk_contribution = (1.0 - surface_contribution).clamp(0.0, 1.0) * bulk_shell
    total = (surface_contribution + bulk_contribution).clamp(0.0, 1.0)
    return total.to(dtype=torch.float32), bulk_contribution.to(dtype=torch.float32), surface_contribution.to(dtype=torch.float32)


def query_ct_local_intensity_from_state(
    training_state,
    points_xyz,
    detach_geometry: bool = True,
    include_surface: bool = True,
    signed_distance: torch.Tensor | None = None,
    boundary_band_distance: float | None = None,
    surface_material_gate_sigma: float | None = None,
    material_compose_mode: str = "bulk_first_material",
    eps: float = 1e-6,
):
    """Locally weighted CT intensity using region-local native density queries."""
    points_xyz = _as_query_points(points_xyz, training_state.xyz.dtype, training_state.xyz.device)
    if points_xyz.numel() == 0 or training_state.xyz.numel() == 0:
        return torch.zeros((points_xyz.shape[0],), dtype=points_xyz.dtype, device=points_xyz.device)

    if getattr(training_state, "ct_value", None) is None or training_state.ct_value.numel() == 0:
        return torch.zeros((points_xyz.shape[0],), dtype=points_xyz.dtype, device=points_xyz.device)

    def _region_terms(region: str):
        if region == "surface":
            means = training_state.surface_xyz
            rotations = training_state.surface_rotation_mats
            scales = training_state.surface_scales
            opacity = training_state.surface_opacity
            ct_value = training_state.surface_ct_value
            spatial_grid = getattr(training_state, "surface_spatial_grid", None)
            support_extent = getattr(training_state, "surface_support_extent", None)
        elif region == "bulk":
            means = training_state.bulk_xyz
            rotations = training_state.bulk_rotation_mats
            scales = training_state.bulk_scales
            opacity = training_state.bulk_opacity
            ct_value = training_state.bulk_ct_value
            spatial_grid = getattr(training_state, "bulk_spatial_grid", None)
            support_extent = getattr(training_state, "bulk_support_extent", None)
        else:
            raise ValueError("region must be 'surface' or 'bulk'.")

        zeros = torch.zeros((points_xyz.shape[0],), dtype=points_xyz.dtype, device=points_xyz.device)
        if means.numel() == 0 or ct_value is None or ct_value.numel() == 0:
            return zeros, zeros

        if detach_geometry:
            means = means.detach()
            rotations = rotations.detach()
            scales = scales.detach()
            opacity = opacity.detach()
            support_extent = None if support_extent is None else support_extent.detach()

        weighted_opacity = opacity * ct_value.to(dtype=opacity.dtype, device=opacity.device)
        numerator = _query_ct_density_native_chunked(
            means,
            rotations,
            scales.clamp_min(1e-6),
            weighted_opacity,
            points_xyz,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
        )
        denominator = _query_ct_density_native_chunked(
            means,
            rotations,
            scales.clamp_min(1e-6),
            opacity,
            points_xyz,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
        ).detach()
        return numerator, denominator

    bulk_num, bulk_den = _region_terms("bulk")
    bulk_value = bulk_num / bulk_den.clamp_min(eps)
    if not include_surface:
        return bulk_value

    surface_num, surface_den = _region_terms("surface")
    surface_value = surface_num / surface_den.clamp_min(eps)
    _, bulk_weight, surface_weight = compose_signed_overlap_occupancy(
        bulk_den.detach(),
        surface_den.detach(),
        signed_distance,
        1.5 if boundary_band_distance is None else float(boundary_band_distance),
        surface_material_gate_sigma=surface_material_gate_sigma,
        material_compose_mode=material_compose_mode,
    )
    if surface_weight is None:
        return bulk_value
    numerator = bulk_weight.detach() * bulk_value + surface_weight.detach() * surface_value
    denominator = (bulk_weight + surface_weight).detach()
    return numerator / denominator.clamp_min(eps)
