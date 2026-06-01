from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch

from ct_pipeline.rendering.bulk_support import (
    DEFAULT_BULK_CONTAINMENT_Q_SUPPORT,
    ellipsoid_probe_directions,
    resolve_bulk_containment_q,
)
from ct_pipeline.rendering.fields import query_bulk_anisotropic_density
from ct_pipeline.training.mutations.bulk_reseed_common import (
    _bulk_reseed_stats,
    _mask_to_numpy_bool,
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
from ct_pipeline.training.utils import as_device_tensor
from utils.rotation_utils import matrix_to_quaternion, quaternion_to_matrix


def _budgeted_repair_components(
    training_state,
    material_mask_np: np.ndarray,
    spacing_zyx,
    *,
    config=None,
    add_threshold: float,
    stop_threshold: float,
    check_stride: int,
    max_check_points: int,
):
    stride = max(int(check_stride), 1)
    material_mask_np = np.asarray(material_mask_np, dtype=bool)
    while True:
        check_mask = material_mask_np[::stride, ::stride, ::stride]
        if int(max_check_points) <= 0 or int(check_mask.sum()) <= int(max_check_points):
            break
        stride += 1

    local_indices = np.argwhere(check_mask)
    density_grid = np.zeros(check_mask.shape, dtype=np.float32)
    labels = np.zeros(check_mask.shape, dtype=np.int32)
    if local_indices.shape[0] == 0:
        return labels, density_grid, 0, 0, stride

    source_indices = local_indices * int(stride)
    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
    points_np = np.stack(
        (
            (source_indices[:, 2].astype(np.float32) + 0.5) * spacing_x,
            (source_indices[:, 1].astype(np.float32) + 0.5) * spacing_y,
            (source_indices[:, 0].astype(np.float32) + 0.5) * spacing_z,
        ),
        axis=1,
    )
    bulk_xyz = getattr(training_state, "bulk_xyz", torch.empty((0, 3)))
    if bulk_xyz.numel() > 0:
        device = bulk_xyz.device
        dtype = bulk_xyz.dtype if torch.is_floating_point(bulk_xyz) else torch.float32
        points_t = as_device_tensor(points_np, device=device, dtype=dtype, reshape=(-1, 3))
        density_parts = []
        chunk_size = 65536
        for start in range(0, int(points_t.shape[0]), chunk_size):
            den = query_bulk_anisotropic_density(
                points_t[start : start + chunk_size],
                training_state,
                config,
                apply_gate=False,
            )
            density_parts.append(den.detach().cpu().to(dtype=torch.float32).numpy())
        sampled_density = np.concatenate(density_parts, axis=0) if density_parts else np.zeros((0,), dtype=np.float32)
    else:
        sampled_density = np.zeros((local_indices.shape[0],), dtype=np.float32)
    density_grid[local_indices[:, 0], local_indices[:, 1], local_indices[:, 2]] = sampled_density.astype(np.float32)

    true_holes = check_mask & (density_grid < float(add_threshold))
    labels, component_count = ndimage.label(true_holes, structure=ndimage.generate_binary_structure(3, 1))
    if int(component_count) <= 0:
        return labels.astype(np.int32, copy=False), density_grid, 0, 0, stride
    sizes = np.bincount(labels.reshape(-1), minlength=int(component_count) + 1)
    max_size = int(sizes[1:].max()) if sizes.shape[0] > 1 else 0
    return labels.astype(np.int32, copy=False), density_grid, int(component_count), max_size, stride


def _component_world_points(local_indices: np.ndarray, stride: int, spacing_zyx) -> np.ndarray:
    source = np.asarray(local_indices, dtype=np.int64) * int(stride)
    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
    return np.stack(
        (
            (source[:, 2].astype(np.float32) + 0.5) * spacing_x,
            (source[:, 1].astype(np.float32) + 0.5) * spacing_y,
            (source[:, 0].astype(np.float32) + 0.5) * spacing_z,
        ),
        axis=1,
    ).astype(np.float32)


def _gaussian_density_np(points_xyz: np.ndarray, center_xyz: np.ndarray, scales: np.ndarray, rotation: np.ndarray, opacity: float):
    delta = np.asarray(points_xyz, dtype=np.float32) - np.asarray(center_xyz, dtype=np.float32).reshape(1, 3)
    local = delta @ np.asarray(rotation, dtype=np.float32)
    exponent = -0.5 * np.sum((local / np.maximum(np.asarray(scales, dtype=np.float32), 1e-6)) ** 2, axis=1)
    return float(opacity) * np.exp(exponent)


def _probe_candidate_inside_material(
    center,
    scales,
    rotation,
    support_mask_t,
    spacing_zyx,
    *,
    normal_axis: int = 2,
    tangent_factor: float = 1.0,
    support_radius: float | None = None,
    signed_distance_field: dict | None = None,
    q_support: float = DEFAULT_BULK_CONTAINMENT_Q_SUPPORT,
):
    center = torch.as_tensor(center, dtype=torch.float32, device=support_mask_t.device).reshape(1, 3)
    scales = torch.as_tensor(scales, dtype=torch.float32, device=support_mask_t.device).reshape(3)
    rotation = torch.as_tensor(rotation, dtype=torch.float32, device=support_mask_t.device).reshape(3, 3)
    sqrt_q = math.sqrt(max(float(q_support), 1e-8))
    offsets = []
    for direction in ellipsoid_probe_directions():
        local_dir = torch.as_tensor(direction, dtype=torch.float32, device=support_mask_t.device)
        offset = rotation @ (sqrt_q * scales * local_dir)
        offsets.extend((offset, -offset))
    if support_radius is not None:
        radius = max(float(support_radius), 1e-8)
        offsets = [
            offset * min(1.0, radius / max(float(torch.linalg.norm(offset).item()), 1e-8))
            for offset in offsets
        ]
    probes = torch.cat((center, center + torch.stack(offsets, dim=0)), dim=0)
    if signed_distance_field is not None:
        sdf = sample_volume_field(
            signed_distance_field["signed_distance"],
            probes,
            signed_distance_field["spacing_zyx"],
        ).reshape(-1)
        return bool(torch.all(torch.isfinite(sdf) & (sdf < 0.0)).item())
    return bool(torch.all(_sample_binary_mask_nearest(support_mask_t, probes, spacing_zyx)).item())


def _sdf_inside_np(points_xyz: np.ndarray, signed_distance_field: dict, device) -> np.ndarray:
    points = torch.as_tensor(np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3), device=device)
    sdf = sample_volume_field(
        signed_distance_field["signed_distance"],
        points,
        signed_distance_field["spacing_zyx"],
    ).reshape(-1)
    inside = torch.isfinite(sdf) & (sdf < 0.0)
    return inside.detach().cpu().numpy().astype(bool, copy=False)


def _directional_clearance_one_side_torch(
    center,
    direction,
    signed_distance_field: dict,
    device,
    max_distance: float,
    step_distance: float,
) -> float:
    max_distance = float(max(max_distance, 0.0))
    if max_distance <= 0.0:
        return 0.0
    center_np = np.asarray(center, dtype=np.float32).reshape(3)
    direction_np = np.asarray(direction, dtype=np.float32).reshape(3)
    direction_norm = float(np.linalg.norm(direction_np))
    if direction_norm <= 1e-8:
        return 0.0
    direction_np = direction_np / direction_norm
    step_distance = max(float(step_distance), 1e-6)
    steps = max(1, int(math.ceil(max_distance / step_distance)))
    distances = np.linspace(step_distance, max_distance, steps, dtype=np.float32)
    probes = center_np.reshape(1, 3) + distances.reshape(-1, 1) * direction_np.reshape(1, 3)
    inside = _sdf_inside_np(probes, signed_distance_field, device)
    outside = np.nonzero(~inside)[0]
    if outside.size == 0:
        return max_distance
    high = float(distances[int(outside[0])])
    low = 0.0 if int(outside[0]) == 0 else float(distances[int(outside[0]) - 1])
    for _ in range(5):
        mid = 0.5 * (low + high)
        mid_point = center_np + mid * direction_np
        if bool(_sdf_inside_np(mid_point.reshape(1, 3), signed_distance_field, device)[0]):
            low = mid
        else:
            high = mid
    return low


def _clearance_cap_candidate_scales(
    center,
    desired_scales,
    rotation,
    signed_distance_field: dict | None,
    device,
    min_spacing: float,
    *,
    q_cont: float = DEFAULT_BULK_CONTAINMENT_Q_SUPPORT,
    safety: float = 0.85,
) -> tuple[np.ndarray, bool]:
    desired = np.asarray(desired_scales, dtype=np.float32).reshape(3).copy()
    if signed_distance_field is None:
        return desired, False
    center_np = np.asarray(center, dtype=np.float32).reshape(3)
    rotation_np = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    if not bool(_sdf_inside_np(center_np.reshape(1, 3), signed_distance_field, device)[0]):
        return np.zeros((3,), dtype=np.float32), True
    sqrt_q = math.sqrt(max(float(q_cont), 1e-8))
    safety = max(float(safety), 1e-6)
    step_distance = max(0.35 * float(min_spacing), 1e-6)
    capped = desired.copy()
    limited = False
    for axis in range(3):
        scale = float(desired[axis])
        if scale <= 0.0 or not np.isfinite(scale):
            capped[axis] = 0.0
            limited = True
            continue
        max_distance = scale * sqrt_q / safety
        c_pos = _directional_clearance_one_side_torch(
            center_np,
            rotation_np[:, axis],
            signed_distance_field,
            device,
            max_distance,
            step_distance,
        )
        c_neg = _directional_clearance_one_side_torch(
            center_np,
            -rotation_np[:, axis],
            signed_distance_field,
            device,
            max_distance,
            step_distance,
        )
        safe_scale = safety * min(c_pos, c_neg) / sqrt_q
        capped_axis = min(scale, max(float(safe_scale), 0.0))
        if capped_axis < 0.999 * scale:
            limited = True
        capped[axis] = capped_axis
    return capped.astype(np.float32, copy=False), limited


def _append_budgeted_repair_seeds(
    gaussians,
    spacing_zyx,
    volume_field,
    new_xyz: torch.Tensor,
    new_scales: torch.Tensor,
    new_rotation_mats: torch.Tensor,
    *,
    iteration: int,
):
    new_count = int(new_xyz.shape[0])
    if new_count <= 0:
        return 0, float("nan"), float("nan")
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    new_xyz = new_xyz.to(device=device, dtype=dtype)
    new_scales = new_scales.to(device=device, dtype=dtype)
    new_rotation_mats = new_rotation_mats.to(device=device, dtype=dtype)
    new_rotations = matrix_to_quaternion(new_rotation_mats).to(dtype=dtype)
    new_normals = new_rotation_mats[:, :, 2]
    new_scaling = torch.log(new_scales.clamp_min(1e-8))
    new_opacity = gaussians.inverse_opacity_activation(torch.full((new_count, 1), 0.65, dtype=dtype, device=device))
    new_features_dc = torch.full(
        (new_count,) + tuple(int(v) for v in gaussians._features_dc.shape[1:]),
        0.5,
        dtype=gaussians._features_dc.dtype,
        device=device,
    )
    new_features_rest = torch.zeros(
        (new_count,) + tuple(int(v) for v in gaussians._features_rest.shape[1:]),
        dtype=gaussians._features_rest.dtype,
        device=device,
    )
    new_primitive_type = torch.full(
        (new_count, 1),
        float(getattr(gaussians, "nonplanar_logit_value", -8.0)),
        dtype=dtype,
        device=device,
    )
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
    return new_count, float(new_scales.mean().item()), float(sampled.mean().item())


def _apply_budgeted_component_bulk_repair(
    gaussians,
    args,
    training_state,
    spacing_zyx,
    analysis,
    initial_gaussian_count: int,
    *,
    volume_field,
    signed_distance_field,
    iteration: int,
):
    stats = _bulk_reseed_stats(gaussians)
    if gaussians.optimizer is None or not getattr(gaussians, "is_initialized", lambda: False)():
        return stats
    if signed_distance_field is None or volume_field is None:
        return stats
    material_mask_np = _mask_to_numpy_bool(analysis.get("material_mask", analysis.get("coarse_support_mask")))
    if material_mask_np is None or not np.any(material_mask_np):
        return stats

    with torch.no_grad():
        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        min_spacing = float(min(spacing_zyx))
        stats["count_before"] = int(gaussians.get_xyz.shape[0])
        labels, density_grid, component_count, max_component_size, stride = _budgeted_repair_components(
            training_state,
            material_mask_np,
            spacing_zyx,
            config=args,
            add_threshold=float(getattr(args, "ct_repair_add_threshold", 0.50)),
            stop_threshold=float(getattr(args, "ct_repair_stop_threshold", 0.85)),
            check_stride=int(getattr(args, "ct_repair_check_stride", 2)),
            max_check_points=int(getattr(args, "ct_repair_max_check_points", 200000)),
        )
        stats["candidates"] = int(np.asarray(material_mask_np, dtype=bool).sum())
        stats["num_uncovered_components"] = int(component_count)
        stats["max_uncovered_component_voxels"] = int(max_component_size * (stride ** 3))
        stats["low_coverage_ratio"] = float((labels > 0).sum() / max(1, int(labels.size)))
        if component_count <= 0:
            return stats

        stop_threshold = float(getattr(args, "ct_repair_stop_threshold", 0.85))
        sizes = np.bincount(labels.reshape(-1), minlength=int(component_count) + 1)
        residual_mass = np.zeros((int(component_count) + 1,), dtype=np.float64)
        for component_id in range(1, int(component_count) + 1):
            component_density = density_grid[labels == component_id]
            residual_mass[component_id] = float(np.maximum(0.0, stop_threshold - component_density).sum())
        stats["repair_residual_mass"] = float(residual_mass[1:].sum())

        min_points = max(int(getattr(args, "ct_repair_min_component_points", 16)), 1)
        component_ids = np.arange(1, int(component_count) + 1, dtype=np.int32)
        keep_components = component_ids[sizes[component_ids] >= min_points]
        stats["repair_skipped_small_components"] = int(component_ids.size - keep_components.size)
        if keep_components.size == 0:
            return stats
        component_ids = keep_components[np.argsort(residual_mass[keep_components])[::-1]]
        top_components = max(int(getattr(args, "ct_repair_top_components", 256)), 1)
        component_ids = component_ids[:top_components]
        stats["repair_components_considered"] = int(component_ids.size)

        bulk_mask = _bulk_mask_tensor(gaussians)
        bulk_indices = torch.nonzero(bulk_mask, as_tuple=False).reshape(-1)
        if bulk_indices.numel() == 0:
            return stats
        bulk_xyz = gaussians.get_xyz.detach()[bulk_mask]
        bulk_scales = torch.exp(gaussians._scaling.detach()[bulk_mask]).to(dtype=torch.float32)
        bulk_rotations = quaternion_to_matrix(gaussians.get_rotation.detach()[bulk_mask]).to(dtype=torch.float32)
        bulk_opacity = gaussians.get_opacity.detach()[bulk_mask].reshape(-1).to(dtype=torch.float32)
        bulk_xyz_np = bulk_xyz.cpu().numpy().astype(np.float32, copy=False)
        bulk_tree = cKDTree(bulk_xyz_np)
        support_mask_t = as_device_tensor(material_mask_np, device=device, dtype=torch.bool)

        max_ratio = float(getattr(args, "ct_bulk_reseed_max_gaussian_ratio", 2.5))
        max_count = int(math.floor(max(1, int(initial_gaussian_count)) * max_ratio))
        remaining_budget = max(0, max_count - int(gaussians.get_xyz.shape[0]))
        bulk_count = int(bulk_indices.numel())
        configured_cap = int(getattr(args, "ct_repair_max_new_per_pass", 1000))
        fraction_cap = int(math.floor(float(getattr(args, "ct_repair_max_new_fraction", 0.005)) * max(1, bulk_count)))
        pass_cap = min(remaining_budget, configured_cap, max(1, fraction_cap))

        gain_ratio_min = float(getattr(args, "ct_repair_gain_ratio_min", 0.15))
        exclusion_radius = float(getattr(args, "ct_repair_exclusion_radius_vox", 0.75)) * min_spacing
        stretch_factor = float(getattr(args, "ct_repair_stretch_growth_factor", 1.15))
        stretch_secondary_factor = float(getattr(args, "ct_repair_stretch_secondary_factor", 1.10))
        stretch_max_ratio = float(getattr(args, "ct_repair_stretch_max_ratio", 4.0))
        overfill_threshold = float(getattr(args, "ct_repair_overfill_threshold", 1.25))
        nearby_candidates = max(int(getattr(args, "ct_repair_nearby_candidates", 8)), 1)
        stretch_first = bool(getattr(args, "ct_repair_stretch_first", True))
        probe_tangent_factor = float(getattr(args, "ct_repair_probe_tangent_factor", 1.5))
        probe_shrink = float(getattr(args, "ct_repair_probe_shrink", 0.75))
        probe_iters = int(getattr(args, "ct_repair_max_probe_shrink_iters", 4))
        q_cont = resolve_bulk_containment_q(args)
        clearance_safety = 0.85
        seed_radius = float(getattr(args, "ct_gap_reseed_radius_vox", 1.0)) * min_spacing
        new_xyz_parts = []
        new_scale_parts = []
        new_rotation_parts = []
        accepted_centers = []
        stretched = 0
        skipped_exclusion = 0
        skipped_gain = 0
        skipped_containment = 0
        skipped_overfill = 0
        skipped_headroom = 0
        clearance_limited = 0
        expanded_bulk_ids = set()
        for component_id in component_ids.tolist():
            local_indices = np.argwhere(labels == int(component_id))
            if local_indices.shape[0] == 0:
                continue
            if local_indices.shape[0] > 2048:
                ids = np.linspace(0, local_indices.shape[0] - 1, 2048).astype(np.int64)
                local_indices = local_indices[ids]
            points_np = _component_world_points(local_indices, stride, spacing_zyx)
            local_density = density_grid[local_indices[:, 0], local_indices[:, 1], local_indices[:, 2]]
            residual = np.maximum(0.0, stop_threshold - local_density).astype(np.float32)
            center_np = points_np[int(np.argmax(residual))]
            nearest_distance, nearest_bulk = bulk_tree.query(center_np, k=1)
            nearest_bulk = int(nearest_bulk)

            if stretch_first and np.isfinite(nearest_distance):
                query_k = min(max(nearby_candidates * 4, nearby_candidates), int(bulk_xyz_np.shape[0]))
                nearby_distances, nearby_ids = bulk_tree.query(center_np, k=query_k)
                nearby_distances = np.asarray(nearby_distances, dtype=np.float32).reshape(-1)
                nearby_ids = np.asarray(nearby_ids, dtype=np.int64).reshape(-1)
                opacity_np = bulk_opacity.detach().cpu().numpy().astype(np.float32, copy=False)
                sort_ids = np.lexsort((nearby_distances, -opacity_np[nearby_ids]))
                expanded = False
                for nearby_bulk in nearby_ids[sort_ids].tolist():
                    nearby_bulk = int(nearby_bulk)
                    if nearby_bulk in expanded_bulk_ids:
                        continue
                    scales_np = bulk_scales[nearby_bulk].cpu().numpy().astype(np.float32, copy=True)
                    rotation_np = bulk_rotations[nearby_bulk].cpu().numpy().astype(np.float32, copy=False)
                    bbox_distance = np.maximum(
                        np.maximum(points_np.min(axis=0) - bulk_xyz_np[nearby_bulk], bulk_xyz_np[nearby_bulk] - points_np.max(axis=0)),
                        0.0,
                    )
                    if float(np.linalg.norm(bbox_distance)) > 3.0 * float(scales_np.max()):
                        continue
                    opacity = float(opacity_np[nearby_bulk])
                    if opacity < 0.05:
                        continue
                    bulk_center = torch.as_tensor(bulk_xyz_np[nearby_bulk], dtype=dtype, device=device).reshape(1, 3)
                    sdf_normal = _sample_sdf_normals_for_reseed(bulk_center, signed_distance_field)[0].detach().cpu().numpy()
                    center_sdf = sample_volume_field(
                        signed_distance_field["signed_distance"], bulk_center, signed_distance_field["spacing_zyx"]
                    ).reshape(-1).to(dtype=torch.float32)
                    normal_axis = int(np.argmax(np.abs(rotation_np.T @ sdf_normal)))
                    tangent_axes = [axis for axis in range(3) if axis != normal_axis]
                    local_direction = rotation_np.T @ (center_np - bulk_xyz_np[nearby_bulk])
                    tangent_axes.sort(key=lambda axis: abs(float(local_direction[axis])), reverse=True)
                    stretched_scales = scales_np.copy()
                    scale_cap = max(float(scales_np.max()), float(scales_np.min()) * stretch_max_ratio)
                    stretched_scales[tangent_axes[0]] = min(stretched_scales[tangent_axes[0]] * stretch_factor, scale_cap)
                    stretched_scales[tangent_axes[1]] = min(stretched_scales[tangent_axes[1]] * stretch_secondary_factor, scale_cap)
                    stretched_scales, limited_by_clearance = _clearance_cap_candidate_scales(
                        bulk_xyz_np[nearby_bulk],
                        stretched_scales,
                        rotation_np,
                        signed_distance_field,
                        device,
                        min_spacing,
                        q_cont=q_cont,
                        safety=clearance_safety,
                    )
                    clearance_limited += int(limited_by_clearance)
                    if not np.all(np.isfinite(stretched_scales)) or np.any(stretched_scales <= 1e-6):
                        skipped_containment += 1
                        continue
                    if np.all(stretched_scales <= scales_np * (1.0 + 1e-4)):
                        skipped_headroom += 1
                        continue
                    for _ in range(max(probe_iters, 0) + 1):
                        if _probe_candidate_inside_material(
                            bulk_xyz_np[nearby_bulk],
                            stretched_scales,
                            rotation_np,
                            support_mask_t,
                            spacing_zyx,
                            normal_axis=normal_axis,
                            tangent_factor=probe_tangent_factor,
                            signed_distance_field=signed_distance_field,
                            q_support=q_cont,
                        ):
                            break
                        stretched_scales[tangent_axes] = np.maximum(
                            scales_np[tangent_axes],
                            stretched_scales[tangent_axes] * probe_shrink,
                        )
                    else:
                        skipped_containment += 1
                        continue
                    if np.allclose(stretched_scales, scales_np):
                        skipped_containment += 1
                        continue
                    local_radius = max(3.0 * float(stretched_scales.max()), 2.0 * min_spacing)
                    gain_mask = np.linalg.norm(points_np - bulk_xyz_np[nearby_bulk].reshape(1, 3), axis=1) <= local_radius
                    if not np.any(gain_mask):
                        skipped_gain += 1
                        continue
                    old_density = _gaussian_density_np(
                        points_np[gain_mask], bulk_xyz_np[nearby_bulk], scales_np, rotation_np, opacity
                    )
                    new_density = _gaussian_density_np(
                        points_np[gain_mask], bulk_xyz_np[nearby_bulk], stretched_scales, rotation_np, opacity
                    )
                    density_delta = np.maximum(0.0, new_density - old_density)
                    residual_local = residual[gain_mask]
                    residual_local_sum = max(float(residual_local.sum()), 1e-8)
                    den_new = local_density[gain_mask] + density_delta
                    stretch_gain = float(np.minimum(residual_local, density_delta).sum())
                    if stretch_gain / residual_local_sum < gain_ratio_min:
                        skipped_gain += 1
                        continue
                    if float(np.quantile(den_new, 0.95)) > overfill_threshold:
                        skipped_overfill += 1
                        continue
                    gaussians._scaling[bulk_indices[nearby_bulk]] = torch.log(
                        torch.as_tensor(stretched_scales, dtype=gaussians._scaling.dtype, device=device).clamp_min(1e-8)
                    )
                    bulk_scales[nearby_bulk] = torch.as_tensor(stretched_scales, dtype=bulk_scales.dtype, device=device)
                    density_grid[
                        local_indices[gain_mask, 0],
                        local_indices[gain_mask, 1],
                        local_indices[gain_mask, 2],
                    ] = den_new
                    expanded_bulk_ids.add(nearby_bulk)
                    stretched += 1
                    expanded = True
                    break
                if expanded:
                    continue

            if len(new_xyz_parts) >= pass_cap:
                continue
            if np.isfinite(nearest_distance) and float(nearest_distance) < exclusion_radius:
                skipped_exclusion += 1
                continue
            if accepted_centers:
                accepted_distance = np.linalg.norm(np.asarray(accepted_centers, dtype=np.float32) - center_np.reshape(1, 3), axis=1)
                if np.any(accepted_distance < exclusion_radius):
                    skipped_exclusion += 1
                    continue

            center = torch.as_tensor(center_np, dtype=dtype, device=device).reshape(1, 3)
            sdf = sample_volume_field(
                signed_distance_field["signed_distance"], center, signed_distance_field["spacing_zyx"]
            ).reshape(-1).to(dtype=torch.float32)
            inside_distance = float(torch.relu(-sdf)[0].item())
            if inside_distance <= 0.0:
                continue
            normals = _sample_sdf_normals_for_reseed(center, signed_distance_field).to(dtype=dtype)
            rotation = _frames_from_surface_normals(normals)[0]
            tangent_sigma = min(seed_radius, max(0.10 * min_spacing, 0.80 * inside_distance))
            normal_sigma = min(0.70 * seed_radius, max(0.10 * min_spacing, 0.75 * inside_distance))
            scales = torch.tensor((tangent_sigma, tangent_sigma, normal_sigma), dtype=dtype, device=device)
            rotation_np = rotation.cpu().numpy().astype(np.float32, copy=False)
            scale_np, limited_by_clearance = _clearance_cap_candidate_scales(
                center_np,
                scales.detach().cpu().numpy().astype(np.float32, copy=False),
                rotation_np,
                signed_distance_field,
                device,
                min_spacing,
                q_cont=q_cont,
                safety=clearance_safety,
            )
            clearance_limited += int(limited_by_clearance)
            if not np.all(np.isfinite(scale_np)) or np.any(scale_np <= 1e-6):
                skipped_containment += 1
                continue
            scales = torch.as_tensor(scale_np, dtype=dtype, device=device)
            for _ in range(max(probe_iters, 0) + 1):
                if _probe_candidate_inside_material(
                    center[0],
                    scales,
                    rotation,
                    support_mask_t,
                    spacing_zyx,
                    normal_axis=2,
                    tangent_factor=probe_tangent_factor,
                    signed_distance_field=signed_distance_field,
                    q_support=q_cont,
                ):
                    break
                scales[:2] *= probe_shrink
            else:
                skipped_containment += 1
                continue
            scale_np = scales.cpu().numpy().astype(np.float32, copy=False)
            local_radius = max(3.0 * float(scale_np.max()), 2.0 * min_spacing)
            gain_mask = np.linalg.norm(points_np - center_np.reshape(1, 3), axis=1) <= local_radius
            if not np.any(gain_mask):
                skipped_gain += 1
                continue
            seed_density = _gaussian_density_np(points_np[gain_mask], center_np, scale_np, rotation_np, 0.65)
            residual_local = residual[gain_mask]
            residual_local_sum = max(float(residual_local.sum()), 1e-8)
            den_new = local_density[gain_mask] + seed_density
            gain = float(np.minimum(residual_local, seed_density).sum())
            if gain / residual_local_sum < gain_ratio_min:
                skipped_gain += 1
                continue
            if float(np.quantile(den_new, 0.95)) > overfill_threshold:
                skipped_overfill += 1
                continue
            new_xyz_parts.append(center[0])
            new_scale_parts.append(scales)
            new_rotation_parts.append(rotation)
            accepted_centers.append(center_np)

        stats["repair_stretched_count"] = int(stretched)
        stats["repair_skipped_exclusion"] = int(skipped_exclusion)
        stats["repair_skipped_low_gain"] = int(skipped_gain)
        stats["repair_skipped_containment"] = int(skipped_containment)
        stats["repair_skipped_overfill"] = int(skipped_overfill)
        stats["repair_skipped_no_clearance_headroom"] = int(skipped_headroom)
        stats["repair_clearance_limited"] = int(clearance_limited)
        stats["bulk_grown_count"] = int(stretched)
        if not new_xyz_parts:
            stats["count_after"] = int(gaussians.get_xyz.shape[0])
            return stats
        added, sigma_mean, atten_mean = _append_budgeted_repair_seeds(
            gaussians,
            spacing_zyx,
            volume_field,
            torch.stack(new_xyz_parts, dim=0),
            torch.stack(new_scale_parts, dim=0),
            torch.stack(new_rotation_parts, dim=0),
            iteration=iteration,
        )
        stats["added"] = int(added)
        stats["sigma_init_mean"] = sigma_mean
        stats["atten_init_mean"] = atten_mean
        stats["count_after"] = int(gaussians.get_xyz.shape[0])
    return stats
