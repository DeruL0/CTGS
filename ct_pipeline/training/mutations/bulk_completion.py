from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch

from ct_pipeline.rendering.fields import density_to_occupancy, query_ct_density_from_state_by_region
from ct_pipeline.training.mutations.bulk_reseed_common import (
    _apply_material_limited_bulk_growth,
    _bulk_reseed_stats,
    _mask_to_numpy_bool,
    _sdf_to_numpy,
    _voxel_indices_to_world,
)
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.mutations.helpers import (
    _append_gap_seed_birth_iter,
    _ensure_gap_seed_birth_iter,
)


def _material_completion_components(
    training_state,
    material_mask_np: np.ndarray,
    spacing_zyx,
    device,
    dtype,
    *,
    den_target: float,
    check_stride: int,
    max_check_points: int = 0,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    check_mask = np.asarray(material_mask_np, dtype=bool).copy()
    stride = max(int(check_stride), 1)
    if stride > 1:
        grid = np.indices(check_mask.shape)
        stride_mask = (grid[0] % stride == 0) & (grid[1] % stride == 0) & (grid[2] % stride == 0)
        check_mask &= stride_mask
    indices = np.argwhere(check_mask)
    uncovered = np.zeros_like(check_mask, dtype=bool)
    if indices.shape[0] == 0:
        return uncovered, 0, 0, np.zeros_like(check_mask, dtype=np.int32)
    max_check_points = int(max_check_points)
    if max_check_points > 0 and indices.shape[0] > max_check_points:
        sample_ids = np.linspace(0, indices.shape[0] - 1, max_check_points).astype(np.int64)
        indices = indices[sample_ids]

    bulk_xyz = getattr(training_state, "bulk_xyz", torch.empty((0, 3), device=device))
    bulk_scales = getattr(training_state, "bulk_scales", torch.empty((0, 3), device=device))
    bulk_opacity = getattr(training_state, "bulk_opacity", torch.empty((0,), device=device))
    if bulk_xyz.numel() > 0 and bulk_scales.shape[0] == bulk_xyz.shape[0]:
        spacing_z, spacing_y, spacing_x = [float(v) for v in spacing_zyx]
        points_np = np.stack(
            (
                (indices[:, 2].astype(np.float32) + 0.5) * spacing_x,
                (indices[:, 1].astype(np.float32) + 0.5) * spacing_y,
                (indices[:, 0].astype(np.float32) + 0.5) * spacing_z,
            ),
            axis=1,
        )
        bulk_np = bulk_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        scales_np = bulk_scales.detach().cpu().numpy().astype(np.float32, copy=False).max(axis=1).clip(min=1e-6)
        opacity_np = bulk_opacity.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if opacity_np.shape[0] != bulk_np.shape[0]:
            opacity_np = np.ones((bulk_np.shape[0],), dtype=np.float32)
        tree = cKDTree(bulk_np)
        k = min(16, int(bulk_np.shape[0]))
        distances, neighbor_ids = tree.query(points_np, k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            neighbor_ids = neighbor_ids[:, None]
        neighbor_scales = scales_np[neighbor_ids]
        neighbor_opacity = opacity_np[neighbor_ids]
        density = np.sum(np.exp(-0.5 * (distances / np.maximum(neighbor_scales, 1e-6)) ** 2) * neighbor_opacity, axis=1)
        occ = 1.0 - np.exp(-density)
        low = occ < float(den_target)
    else:
        low_parts = []
        for start in range(0, int(indices.shape[0]), int(chunk_size)):
            part = indices[start : start + int(chunk_size)]
            points = _voxel_indices_to_world(part, spacing_zyx, device=device, dtype=dtype)
            occ = density_to_occupancy(
                query_ct_density_from_state_by_region(training_state, points, region="bulk", detach=True)
            ).to(dtype=torch.float32)
            low_parts.append((occ < float(den_target)).detach().cpu().numpy())
        low = np.concatenate(low_parts, axis=0) if low_parts else np.zeros((0,), dtype=bool)
    uncovered[indices[:, 0], indices[:, 1], indices[:, 2]] = low

    labels, component_count = ndimage.label(uncovered, structure=ndimage.generate_binary_structure(3, 1))
    if component_count <= 0:
        return uncovered, 0, 0, labels.astype(np.int32, copy=False)
    sizes = np.bincount(labels.reshape(-1), minlength=component_count + 1)
    max_size = int(sizes[1:].max()) if sizes.shape[0] > 1 else 0
    return uncovered, int(component_count), max_size, labels.astype(np.int32, copy=False)


def _append_bulk_completion_seeds(
    gaussians,
    args,
    spacing_zyx,
    component_labels: np.ndarray,
    component_count: int,
    component_sizes: np.ndarray,
    dist_to_air_or_void: np.ndarray,
    volume_field,
    *,
    iteration: int,
) -> tuple[int, float, float]:
    if component_count <= 0:
        return 0, float("nan"), float("nan")
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    min_component_voxels = max(int(getattr(args, "ct_completion_min_component_voxels", 1)), 1)
    max_new = int(getattr(args, "ct_completion_max_new_per_pass", 0))
    if max_new <= 0:
        return 0, float("nan"), float("nan")

    component_ids = np.arange(1, int(component_count) + 1, dtype=np.int32)
    component_ids = component_ids[component_sizes[component_ids] >= min_component_voxels]
    if component_ids.size == 0:
        return 0, float("nan"), float("nan")
    order = np.argsort(component_sizes[component_ids])[::-1]
    component_ids = component_ids[order[:max_new]]

    seed_indices = []
    for component_id in component_ids.tolist():
        voxels = np.argwhere(component_labels == int(component_id))
        if voxels.shape[0] == 0:
            continue
        distances = dist_to_air_or_void[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
        valid = np.isfinite(distances) & (distances > 0.0)
        if not np.any(valid):
            continue
        voxels = voxels[valid]
        distances = distances[valid]
        seed_indices.append(voxels[int(np.argmax(distances))])
    if not seed_indices:
        return 0, float("nan"), float("nan")

    seed_indices_np = np.asarray(seed_indices, dtype=np.int64)
    new_xyz = _voxel_indices_to_world(seed_indices_np, spacing_zyx, device=device, dtype=dtype)
    distances_np = dist_to_air_or_void[seed_indices_np[:, 0], seed_indices_np[:, 1], seed_indices_np[:, 2]]
    seed_radius = float(getattr(args, "ct_completion_radius_vox", 1.0)) * float(min(spacing_zyx))
    sigma_init_np = np.minimum(float(seed_radius), 0.8 * distances_np.astype(np.float32))
    min_scale = max(float(min(spacing_zyx)) * 0.05, 1e-6)
    keep_np = np.isfinite(sigma_init_np) & (sigma_init_np >= min_scale)
    if not np.any(keep_np):
        return 0, float("nan"), float("nan")

    new_xyz = new_xyz[torch.as_tensor(keep_np, dtype=torch.bool, device=device)]
    sigma_init = torch.as_tensor(sigma_init_np[keep_np], dtype=dtype, device=device)
    new_count = int(new_xyz.shape[0])
    if new_count <= 0:
        return 0, float("nan"), float("nan")

    new_rotations = torch.zeros((new_count, 4), dtype=dtype, device=device)
    new_rotations[:, 0] = 1.0
    new_normals = torch.zeros((new_count, 3), dtype=dtype, device=device)
    new_normals[:, 2] = 1.0
    new_scaling = torch.log(sigma_init.reshape(-1, 1).expand(new_count, 3).clamp_min(1e-8))
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
        new_normals,
        new_material_id,
        new_planarity,
        new_region_type,
        new_ct_value_logit=new_ct_value_logit,
        new_atten_logit=new_atten_logit,
    )
    _append_gap_seed_birth_iter(gaussians, previous_birth, new_count, int(iteration), gap_seed=True)
    return new_count, float(sigma_init.mean().item()), float(sampled.mean().item())


def _apply_material_coverage_completion(
    gaussians,
    args,
    training_state,
    spacing_zyx,
    analysis,
    *,
    volume_field,
    signed_distance_field,
    initial_gaussian_count: int,
    iteration: int = 0,
    pass_index: int = 0,
):
    stats = _bulk_reseed_stats(gaussians)
    stats["completion_pass"] = int(pass_index)
    if gaussians.optimizer is None or not getattr(gaussians, "is_initialized", lambda: False)():
        return stats
    if volume_field is None:
        return stats
    material_mask_np = _mask_to_numpy_bool(analysis.get("material_mask", analysis.get("coarse_support_mask")))
    if material_mask_np is None or not np.any(material_mask_np):
        return stats
    sdf_np = _sdf_to_numpy(signed_distance_field, analysis, material_mask_np.shape)
    if sdf_np is None:
        dist_to_air_or_void = ndimage.distance_transform_edt(material_mask_np, sampling=spacing_zyx).astype(np.float32)
    else:
        dist_to_air_or_void = np.maximum(-sdf_np.astype(np.float32, copy=False), 0.0)

    with torch.no_grad():
        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        stats["count_before"] = int(gaussians.get_xyz.shape[0])
        stats["bulk_grown_count"] = _apply_material_limited_bulk_growth(gaussians, args, signed_distance_field)

        max_ratio = float(getattr(args, "ct_bulk_reseed_max_gaussian_ratio", 2.5))
        max_count = int(math.floor(max(1, int(initial_gaussian_count)) * max_ratio))
        remaining_budget = max(0, max_count - int(gaussians.get_xyz.shape[0]))
        if remaining_budget <= 0:
            return stats

        uncovered, component_count, max_component_size, labels = _material_completion_components(
            training_state,
            material_mask_np,
            spacing_zyx,
            device,
            dtype,
            den_target=float(getattr(args, "ct_completion_den_target", getattr(args, "ct_gap_reseed_den_target", 0.9))),
            check_stride=int(getattr(args, "ct_completion_check_stride", 1)),
            max_check_points=int(getattr(args, "ct_completion_max_check_points", 0)),
        )
        del uncovered
        stats["candidates"] = int(material_mask_np.sum())
        stats["num_uncovered_components"] = int(component_count)
        stats["max_uncovered_component_voxels"] = int(max_component_size)
        stats["low_coverage_ratio"] = float((labels > 0).sum() / max(1, int(material_mask_np.sum())))
        if component_count <= 0:
            return stats

        sizes = np.bincount(labels.reshape(-1), minlength=int(component_count) + 1)
        original_max_new = int(getattr(args, "ct_completion_max_new_per_pass", 0))
        if original_max_new <= 0 or original_max_new > remaining_budget:
            setattr(args, "ct_completion_max_new_per_pass", int(remaining_budget))
        added, sigma_mean, atten_mean = _append_bulk_completion_seeds(
            gaussians,
            args,
            spacing_zyx,
            labels,
            component_count,
            sizes,
            dist_to_air_or_void,
            volume_field,
            iteration=iteration,
        )
        setattr(args, "ct_completion_max_new_per_pass", original_max_new)
        stats["added"] = int(added)
        stats["sigma_init_mean"] = sigma_mean
        stats["atten_init_mean"] = atten_mean
        stats["count_after"] = int(gaussians.get_xyz.shape[0])
    return stats
