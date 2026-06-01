import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree

from ct_pipeline.rendering.bulk_support import ellipsoid_probe_directions, resolve_bulk_containment_q
from ct_pipeline.rendering.fields import (
    bulk_intensity_readout,
    density_to_occupancy,
    query_ct_density_from_state_by_region,
    query_ct_fields_unified,
)
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.utils import as_device_tensor

from .common import (
    _mask_to_np,
    _quantile_metrics,
    _roi_window_from_analysis,
    _sample_mask_points,
    _voxel_indices_to_world,
)


def _record_uncovered_component_metrics(
    metrics,
    training_state,
    analysis,
    spacing_zyx,
    device,
    dtype,
    *,
    config=None,
):
    material_mask = _mask_to_np(analysis.get("material_mask", analysis.get("coarse_support_mask")))
    if material_mask is None or not np.any(material_mask):
        return
    check_mask = np.asarray(material_mask, dtype=bool).copy()
    # Final diagnostics stay dense even when the training repair loop uses a
    # coarser check grid for runtime.
    stride = 1
    if stride > 1:
        grid = np.indices(check_mask.shape)
        check_mask &= (grid[0] % stride == 0) & (grid[1] % stride == 0) & (grid[2] % stride == 0)
    indices = np.argwhere(check_mask)
    if indices.shape[0] == 0:
        metrics["num_uncovered_components"] = 0
        metrics["max_uncovered_component_voxels"] = 0
        return
    target = float(getattr(config, "ct_completion_den_target", 0.9)) if config is not None else 0.9
    uncovered = np.zeros_like(check_mask, dtype=bool)
    chunk_size = 65536
    low_parts = []
    for start in range(0, int(indices.shape[0]), chunk_size):
        part = indices[start : start + chunk_size]
        points = _voxel_indices_to_world(part, spacing_zyx, device=device, dtype=dtype)
        occ = density_to_occupancy(
            query_ct_density_from_state_by_region(training_state, points, region="bulk", detach=True)
        ).to(dtype=torch.float32)
        low_parts.append((occ < target).detach().cpu().numpy())
    low = np.concatenate(low_parts, axis=0) if low_parts else np.zeros((0,), dtype=bool)
    uncovered[indices[:, 0], indices[:, 1], indices[:, 2]] = low
    labels, component_count = ndimage.label(uncovered, structure=ndimage.generate_binary_structure(3, 1))
    if int(component_count) <= 0:
        metrics["num_uncovered_components"] = 0
        metrics["max_uncovered_component_voxels"] = 0
        return
    sizes = np.bincount(labels.reshape(-1), minlength=int(component_count) + 1)
    metrics["num_uncovered_components"] = int(component_count)
    metrics["max_uncovered_component_voxels"] = int(sizes[1:].max()) if sizes.shape[0] > 1 else 0

def _sample_material_voxel_points(
    analysis,
    spacing_zyx,
    device,
    dtype,
    *,
    deep_only: bool,
    boundary_band: float,
    max_count: int = 4096,
    signed_distance_field=None,
):
    material_mask = analysis.get("material_mask", analysis.get("coarse_support_mask"))
    if material_mask is None:
        return None, None
    if isinstance(material_mask, torch.Tensor):
        material_np = material_mask.detach().cpu().numpy().astype(bool)
    else:
        material_np = np.asarray(material_mask, dtype=bool)

    signed_distance = analysis.get("material_signed_distance")
    if signed_distance is None and signed_distance_field is not None:
        signed_distance = signed_distance_field.get("signed_distance_native")
        if signed_distance is None:
            signed_distance = signed_distance_field.get("signed_distance")
            if isinstance(signed_distance, torch.Tensor):
                signed_distance = signed_distance.reshape(*tuple(int(value) for value in signed_distance.shape[-3:]))
    if signed_distance is not None:
        if isinstance(signed_distance, torch.Tensor):
            sdf_np = signed_distance.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            sdf_np = np.asarray(signed_distance, dtype=np.float32)
        if sdf_np.shape == material_np.shape:
            candidate_mask = material_np & (sdf_np <= -float(boundary_band) if deep_only else np.ones_like(material_np, dtype=bool))
        else:
            sdf_np = None
            candidate_mask = material_np
    else:
        sdf_np = None
        candidate_mask = material_np

    indices = np.argwhere(candidate_mask)
    if indices.shape[0] == 0:
        return None, None
    if indices.shape[0] > int(max_count):
        sample_ids = np.linspace(0, indices.shape[0] - 1, int(max_count)).astype(np.int64)
        indices = indices[sample_ids]

    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
    points_np = np.stack(
        (
            indices[:, 2].astype(np.float32) * spacing_x,
            indices[:, 1].astype(np.float32) * spacing_y,
            indices[:, 0].astype(np.float32) * spacing_z,
        ),
        axis=1,
    )
    points = as_device_tensor(points_np, device=device, dtype=dtype, reshape=(-1, 3))
    if sdf_np is None:
        depth = torch.full((points.shape[0],), float("nan"), device=device, dtype=dtype)
    else:
        depth_np = -sdf_np[indices[:, 0], indices[:, 1], indices[:, 2]].astype(np.float32)
        depth = as_device_tensor(depth_np, device=device, dtype=dtype, reshape=(-1,))
    return points, depth

def _sample_material_sdf_band_points(
    analysis,
    spacing_zyx,
    device,
    dtype,
    *,
    signed_distance_field,
    sdf_min: float,
    sdf_max: float,
    max_count: int = 4096,
):
    material_mask = analysis.get("material_mask", analysis.get("coarse_support_mask"))
    if material_mask is None or signed_distance_field is None:
        return None
    material_np = _mask_to_np(material_mask)
    signed_distance = signed_distance_field.get("signed_distance_native", signed_distance_field.get("signed_distance"))
    if signed_distance is None:
        return None
    if isinstance(signed_distance, torch.Tensor):
        sdf_np = signed_distance.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        sdf_np = np.asarray(signed_distance, dtype=np.float32)
    sdf_np = np.reshape(sdf_np, tuple(int(value) for value in material_np.shape))
    if sdf_np.shape != material_np.shape:
        return None
    mask = material_np & np.isfinite(sdf_np) & (sdf_np >= float(sdf_min)) & (sdf_np < float(sdf_max))
    return _sample_mask_points(mask, spacing_zyx, device, dtype, int(max_count))

def _bulk_coverage_gap_threshold(config=None) -> float:
    return float(getattr(config, "ct_bulk_coverage_gap_threshold", 0.50))

def _record_bulk_coverage_metrics(metrics, prefix, training_state, points, depth=None, bulk_xyz=None, config=None):
    if points is None or points.numel() == 0:
        return
    bulk_den = query_ct_density_from_state_by_region(training_state, points, region="bulk", detach=True).to(
        dtype=torch.float32
    )
    bulk_occ = density_to_occupancy(bulk_den).to(dtype=torch.float32)
    gap_threshold = _bulk_coverage_gap_threshold(config)
    gap_mask = bulk_den < gap_threshold
    metrics[f"{prefix}_coverage_gap_threshold"] = gap_threshold
    metrics[f"{prefix}_coverage_gap_ratio"] = float(gap_mask.float().mean().item())
    metrics[f"{prefix}_sample_count"] = int(points.shape[0])
    for suffix, value in _quantile_metrics(bulk_den.detach().cpu().numpy(), ("p10", "p50")).items():
        metrics[f"{prefix}_den_b_{suffix}"] = value
    for suffix, value in _quantile_metrics(bulk_occ.detach().cpu().numpy(), ("p10", "p50")).items():
        metrics[f"{prefix}_occupancy_{suffix}"] = value

    if depth is not None and depth.numel() == bulk_occ.numel():
        depth_f = depth.to(dtype=torch.float32)
        for label, lower, upper in (
            ("sdf_1p5_3", 1.5, 3.0),
            ("sdf_3_6", 3.0, 6.0),
            ("sdf_6_plus", 6.0, float("inf")),
        ):
            mask = depth_f >= float(lower)
            if np.isfinite(upper):
                mask = mask & (depth_f < float(upper))
            if torch.any(mask):
                metrics[f"{prefix}_{label}_gap_ratio"] = float(gap_mask[mask].float().mean().item())
                metrics[f"{prefix}_{label}_sample_count"] = int(mask.sum().item())
        if torch.any(gap_mask):
            for suffix, value in _quantile_metrics(depth_f[gap_mask].detach().cpu().numpy(), ("p50", "p90")).items():
                metrics[f"{prefix}_low_occ_sdf_depth_{suffix}"] = value

    if bulk_xyz is not None and bulk_xyz.numel() > 0 and torch.any(gap_mask):
        low_points_np = points[gap_mask].detach().cpu().numpy()
        bulk_np = bulk_xyz.detach().cpu().numpy()
        if low_points_np.shape[0] > 0 and bulk_np.shape[0] > 0:
            distances, _ = cKDTree(bulk_np).query(low_points_np, k=1)
            for suffix, value in _quantile_metrics(distances, ("p50", "p90")).items():
                metrics[f"{prefix}_low_occ_nearest_bulk_distance_{suffix}"] = value

def _record_bulk_occ_quantiles(metrics, prefix, training_state, points, names=("p10", "p50")):
    if points is None or points.numel() == 0:
        return
    bulk_occ = density_to_occupancy(
        query_ct_density_from_state_by_region(training_state, points, region="bulk", detach=True)
    ).to(dtype=torch.float32)
    metrics[f"{prefix}_sample_count"] = int(points.shape[0])
    for suffix, value in _quantile_metrics(bulk_occ.detach().cpu().numpy(), names).items():
        metrics[f"{prefix}_occ_b_raw_{suffix}"] = value

def _record_surface_owned_bulk_field_metrics(metrics, training_state, points, signed_distance_field, config, intensity_air: float):
    """Record W_b (危K_i) and 渭_raw (危a_i K_i) at surface-owned material positions (boundary shell)."""
    if points is None or points.numel() == 0 or signed_distance_field is None:
        return
    with torch.no_grad():
        sdf_vals = sample_volume_field(
            signed_distance_field["signed_distance"],
            points,
            signed_distance_field["spacing_zyx"],
        ).reshape(-1).to(dtype=torch.float32)
        fields = query_ct_fields_unified(
            points,
            training_state,
            signed_distance=sdf_vals,
            config=config,
            intensity_air=float(intensity_air),
            include_surface=False,
            train_ct_value=False,
            apply_bulk_gate=False,
        )
        # In bulk-intensity mode: den_b = 危K_i, I_b_raw = 危a_i K_i, A_b = I_b_raw / den_b.
        W_b = fields["den_b"].to(dtype=torch.float32)
        mu_raw = fields.get("I_b_raw", fields["I_b"]).to(dtype=torch.float32)
        a_b = fields.get("A_b", bulk_intensity_readout(mu_raw, W_b)).to(dtype=torch.float32)
        for suffix, val in _quantile_metrics(W_b.detach().cpu().numpy(), ("p10", "p50")).items():
            metrics[f"surface_owned_bulk_W_{suffix}"] = val
        for suffix, val in _quantile_metrics(mu_raw.detach().cpu().numpy(), ("p10", "p50")).items():
            metrics[f"surface_owned_bulk_mu_raw_{suffix}"] = val
        for suffix, val in _quantile_metrics(a_b.detach().cpu().numpy(), ("p10", "p50")).items():
            metrics[f"surface_owned_bulk_A_b_{suffix}"] = val

def _record_bulk_intensity_quantiles(
    metrics,
    prefix,
    training_state,
    points,
    signed_distance_field,
    config,
    intensity_air: float,
    names=("p10", "p50", "p90"),
):
    if points is None or points.numel() == 0 or signed_distance_field is None:
        return
    signed_distance = sample_volume_field(
        signed_distance_field["signed_distance"],
        points,
        signed_distance_field["spacing_zyx"],
    ).reshape(-1).to(dtype=torch.float32)
    fields = query_ct_fields_unified(
        points,
        training_state,
        signed_distance=signed_distance,
        config=config,
        intensity_air=float(intensity_air),
        include_surface=False,
        train_ct_value=False,
        apply_bulk_gate=False,
    )
    a_b = fields.get("A_b")
    if a_b is None:
        a_b = bulk_intensity_readout(
            fields.get("I_b_raw", fields["I_b"]).to(dtype=torch.float32),
            fields["den_b"].to(dtype=torch.float32),
        )
    else:
        a_b = a_b.to(dtype=torch.float32)
    metrics[f"{prefix}_sample_count"] = int(points.shape[0])
    for suffix, value in _quantile_metrics(a_b.detach().cpu().numpy(), names).items():
        metrics[f"{prefix}_A_b_{suffix}"] = value

def _record_combined_occ_quantiles(metrics, prefix, training_state, points, signed_distance_field, config, intensity_air: float, names=("p10", "p50")):
    if points is None or points.numel() == 0 or signed_distance_field is None:
        return
    signed_distance = sample_volume_field(
        signed_distance_field["signed_distance"],
        points,
        signed_distance_field["spacing_zyx"],
    ).reshape(-1).to(dtype=torch.float32)
    fields = query_ct_fields_unified(
        points,
        training_state,
        signed_distance=signed_distance,
        config=config,
        intensity_air=float(intensity_air),
        include_surface=True,
        train_ct_value=False,
        apply_bulk_gate=False,
    )
    occ_b = fields["occ_b_raw"].to(dtype=torch.float32)
    occ_s = fields["occ_s_raw"].to(dtype=torch.float32)
    combined = (1.0 - (1.0 - occ_b).clamp(0.0, 1.0) * (1.0 - occ_s).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    gap_threshold = _bulk_coverage_gap_threshold(config)
    metrics[f"{prefix}_sample_count"] = int(points.shape[0])
    metrics[f"{prefix}_coverage_gap_threshold"] = gap_threshold
    metrics[f"{prefix}_coverage_gap_ratio"] = float((combined < gap_threshold).float().mean().item())
    for suffix, value in _quantile_metrics(combined.detach().cpu().numpy(), names).items():
        metrics[f"{prefix}_occ_union_raw_{suffix}"] = value

def _record_bulk_surface_gap_distance_metrics(
    metrics,
    training_state,
    analysis,
    spacing_zyx,
    device,
    dtype,
    *,
    signed_distance_field,
    config,
    intensity_air: float,
    intensity_mat: float,
    max_count: int = 512,
):
    if signed_distance_field is None:
        return
    boundary_points = analysis.get("boundary_points")
    boundary_normals = analysis.get("boundary_normals")
    if boundary_points is None or boundary_normals is None:
        return
    anchors = as_device_tensor(boundary_points, device=device, dtype=dtype, reshape=(-1, 3))
    normals = as_device_tensor(boundary_normals, device=device, dtype=dtype, reshape=(-1, 3))
    if anchors.numel() == 0 or normals.numel() == 0 or anchors.shape[0] != normals.shape[0]:
        return
    if anchors.shape[0] > int(max_count):
        ids = torch.linspace(0, anchors.shape[0] - 1, int(max_count), device=device).to(dtype=torch.long)
        anchors = anchors.index_select(0, ids)
        normals = normals.index_select(0, ids)
    normals = F.normalize(normals, dim=-1, eps=1e-8)
    band = float(getattr(config, "ct_boundary_band", 1.5)) if config is not None else 1.5
    probe_steps = 16
    probe_t = torch.linspace(0.0, max(2.0 * band, float(min(spacing_zyx)) * 2.0), probe_steps, device=device, dtype=dtype)
    line_points = anchors[:, None, :] - probe_t[None, :, None] * normals[:, None, :]
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
        config=config,
        intensity_air=float(intensity_air),
        include_surface=False,
        train_ct_value=False,
        apply_bulk_gate=False,
    )
    pred = fields["I_pred"].reshape(anchors.shape[0], probe_steps).to(dtype=torch.float32)
    threshold = float(intensity_air) + 0.5 * float(intensity_mat - intensity_air)
    hit = pred >= threshold
    any_hit = torch.any(hit, dim=1)
    if not torch.any(any_hit):
        return
    first_hit = torch.argmax(hit.to(dtype=torch.int32), dim=1)
    distances = probe_t.index_select(0, first_hit[any_hit]).detach().cpu().numpy()
    for suffix, value in _quantile_metrics(distances, ("p50", "p90")).items():
        metrics[f"bulk_surface_gap_distance_{suffix}"] = value

def _record_bulk_containment_metrics(metrics, training_state, signed_distance_field, margin: float, config=None):
    bulk_xyz = as_device_tensor(getattr(training_state, "bulk_xyz", torch.empty((0, 3))), reshape=(-1, 3))
    if bulk_xyz.numel() == 0 or signed_distance_field is None:
        return
    device = bulk_xyz.device
    dtype = bulk_xyz.dtype
    bulk_scales = as_device_tensor(
        getattr(training_state, "bulk_scales", torch.empty((0, 3), device=device)),
        device=device,
        dtype=dtype,
        reshape=(-1, 3),
    )
    bulk_rotation_mats = as_device_tensor(
        getattr(training_state, "bulk_rotation_mats", torch.empty((0, 3, 3), device=device)),
        device=device,
        dtype=dtype,
        reshape=(-1, 3, 3),
    )
    if bulk_scales.shape[0] != bulk_xyz.shape[0] or bulk_rotation_mats.shape[0] != bulk_xyz.shape[0]:
        return
    sample_count = min(int(bulk_xyz.shape[0]), 8192)
    if bulk_xyz.shape[0] > sample_count:
        indices = torch.linspace(0, bulk_xyz.shape[0] - 1, sample_count, device=device).to(dtype=torch.long)
        bulk_xyz = bulk_xyz.index_select(0, indices)
        bulk_scales = bulk_scales.index_select(0, indices)
        bulk_rotation_mats = bulk_rotation_mats.index_select(0, indices)
    q_support = resolve_bulk_containment_q(config)
    sqrt_q = math.sqrt(q_support)
    offsets = []
    for direction in ellipsoid_probe_directions():
        local_dir = torch.as_tensor(direction, dtype=dtype, device=device).reshape(1, 3)
        local_offset = sqrt_q * bulk_scales * local_dir
        offset = torch.einsum("nij,nj->ni", bulk_rotation_mats, local_offset)
        offsets.extend([offset, -offset])
    probes = torch.cat([bulk_xyz + offset for offset in offsets], dim=0)
    probe_sdf = sample_volume_field(
        signed_distance_field["signed_distance"],
        probes,
        signed_distance_field["spacing_zyx"],
    ).reshape(len(offsets), -1).to(dtype=torch.float32)
    violation = probe_sdf > -float(margin)
    valid = torch.isfinite(probe_sdf).all(dim=0)
    if torch.any(valid):
        metrics["bulk_containment_violation_ratio"] = float(torch.any(violation[:, valid], dim=0).float().mean().item())

def _record_dual_surface_shell_metrics(
    metrics,
    training_state,
    analysis,
    spacing_zyx,
    device,
    dtype,
    *,
    signed_distance_field,
    volume_cuda,
    intensity_air: float,
    config=None,
):
    if signed_distance_field is None or volume_cuda is None:
        return
    boundary_band = float(getattr(config, "ct_boundary_band", 1.5)) if config is not None else 1.5
    material_band = max(min(boundary_band, 0.5), 1e-6)
    air_band = max(float(getattr(config, "ct_dual_surface_air_band", 0.25)) if config is not None else 0.25, 0.0)
    volume_field = volume_cuda.reshape(1, 1, *tuple(int(value) for value in volume_cuda.shape[-3:]))

    material_points = _sample_material_sdf_band_points(
        analysis,
        spacing_zyx,
        device,
        dtype,
        signed_distance_field=signed_distance_field,
        sdf_min=-float(material_band),
        sdf_max=0.0,
        max_count=4096,
    )
    if material_points is not None and material_points.numel() > 0:
        signed_distance = sample_volume_field(
            signed_distance_field["signed_distance"],
            material_points,
            signed_distance_field["spacing_zyx"],
        ).reshape(-1).to(dtype=torch.float32)
        fields = query_ct_fields_unified(
            material_points,
            training_state,
            signed_distance=signed_distance,
            config=config,
            intensity_air=float(intensity_air),
            include_surface=True,
            train_ct_value=False,
            apply_bulk_gate=False,
        )
        target = sample_volume_field(volume_field, material_points, spacing_zyx).reshape(-1).to(dtype=torch.float32)
        pred = fields["I_s"].to(dtype=torch.float32)
        metrics["surface_mat_shell_sample_count"] = int(material_points.shape[0])
        metrics["surface_mat_shell_mae"] = float(torch.mean(torch.abs(pred - target)).item())

    material_mask = _mask_to_np(analysis.get("material_mask", analysis.get("coarse_support_mask")))
    if material_mask is None or air_band <= 0.0:
        return
    signed_distance = signed_distance_field.get("signed_distance_native", signed_distance_field.get("signed_distance"))
    if signed_distance is None:
        return
    if isinstance(signed_distance, torch.Tensor):
        sdf_np = signed_distance.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        sdf_np = np.asarray(signed_distance, dtype=np.float32)
    sdf_np = np.reshape(sdf_np, tuple(int(value) for value in material_mask.shape))
    roi_window = _roi_window_from_analysis(analysis, material_mask.shape)
    air_mask = np.logical_and(roi_window, np.logical_not(material_mask))
    air_shell = air_mask & np.isfinite(sdf_np) & (sdf_np > 0.0) & (sdf_np <= float(air_band))
    air_points = _sample_mask_points(air_shell, spacing_zyx, device, dtype, 4096)
    if air_points is None or air_points.numel() == 0:
        anchors = analysis.get("boundary_points")
        normals = analysis.get("boundary_normals", analysis.get("boundary_normal"))
        if anchors is not None and normals is not None and air_band > 0.0:
            anchors = as_device_tensor(anchors, device=device, dtype=dtype, reshape=(-1, 3))
            normals = as_device_tensor(normals, device=device, dtype=dtype, reshape=(-1, 3))
            if anchors.numel() > 0 and normals.shape[0] == anchors.shape[0]:
                sample_count = min(int(anchors.shape[0]), 4096)
                if anchors.shape[0] > sample_count:
                    indices = torch.linspace(0, anchors.shape[0] - 1, sample_count, device=device).to(dtype=torch.long)
                    anchors = anchors.index_select(0, indices)
                    normals = normals.index_select(0, indices)
                offset = 0.5 * float(air_band) * min(float(value) for value in spacing_zyx)
                air_points = anchors + F.normalize(normals, dim=-1, eps=1e-6) * float(offset)
                depth, height, width = [int(value) for value in material_mask.shape]
                spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
                lower = torch.zeros((3,), dtype=dtype, device=device)
                upper = torch.tensor(
                    [
                        max(0.0, (float(width) - 1e-3) * spacing_x),
                        max(0.0, (float(height) - 1e-3) * spacing_y),
                        max(0.0, (float(depth) - 1e-3) * spacing_z),
                    ],
                    dtype=dtype,
                    device=device,
                )
                air_points = torch.minimum(torch.maximum(air_points, lower.unsqueeze(0)), upper.unsqueeze(0))
                sampled_sdf = sample_volume_field(
                    signed_distance_field["signed_distance"],
                    air_points,
                    signed_distance_field["spacing_zyx"],
                ).reshape(-1).to(dtype=torch.float32)
                keep = torch.isfinite(sampled_sdf) & (sampled_sdf > 0.0) & (sampled_sdf <= float(air_band))
                air_points = air_points[keep]
    if air_points is None or air_points.numel() == 0:
        return
    signed_distance = sample_volume_field(
        signed_distance_field["signed_distance"],
        air_points,
        signed_distance_field["spacing_zyx"],
    ).reshape(-1).to(dtype=torch.float32)
    fields = query_ct_fields_unified(
        air_points,
        training_state,
        signed_distance=signed_distance,
        config=config,
        intensity_air=float(intensity_air),
        include_surface=True,
        train_ct_value=False,
        apply_bulk_gate=False,
    )
    pred = fields["I_s"].to(dtype=torch.float32)
    metrics["surface_air_shell_sample_count"] = int(air_points.shape[0])
    metrics["surface_air_shell_leak"] = float(torch.mean(torch.abs(pred - float(intensity_air))).item())
