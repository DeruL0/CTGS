from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ct_pipeline.rendering.bulk_support import (
    DEFAULT_BULK_CLEARANCE_SAFETY,
    DEFAULT_BULK_CONTAINMENT_Q_SUPPORT,
    ellipsoid_probe_directions,
)
from ct_pipeline.geometry.coordinates import world_xyz_to_voxel_float_numpy, world_xyz_to_voxel_indices_floor_numpy

CT_DENSE_INIT_SURFACE_THICKNESS_RATIO = 0.4
CT_DENSE_INIT_SURFACE_TANGENT_RATIO = 0.7
CT_DENSE_INIT_SURFACE_MIN_SCALE_RATIO = 0.35
CT_DENSE_INIT_BULK_RADIUS_RATIO = 0.5
CT_DENSE_INIT_BULK_MIN_SCALE_RATIO = 0.3
CT_DENSE_INIT_BULK_MAX_SCALE_RATIO = 1.25
CT_DENSE_INIT_BULK_CONTAINMENT_SIGMA = 4.0
CT_DENSE_INIT_BULK_TANGENT_RATIO = 0.7
CT_DENSE_INIT_BULK_GRADIENT_THRESHOLD = 0.05  # min |grad SDF| to trust normal direction
CT_DENSE_INIT_SURFACE_POISSON_BASE_RATIO = 1.0  # base min-distance in min_spacing units
CT_DENSE_INIT_SURFACE_POISSON_CURVATURE_ALPHA = 2.0  # high curvature -> smaller radius

# contained lattice init
CT_BULK_LATTICE_SPACING_VOX = 2.0    # lattice spacing in voxels (~400K pts for bunny)
CT_BULK_LATTICE_SIGMA_VOX = 2.0      # isotropic sigma in voxels (= spacing → W_b continuous)
CT_BULK_LATTICE_MARGIN_VOX = 0.05    # SDF margin: only place if D < -margin*vox (thin to cover boundary band)
CT_BULK_LATTICE_ATTEN_INIT = 0.75    # default attenuation init value
CT_BULK_LATTICE_ANISOTROPIC = False
CT_BULK_LATTICE_SIGMA_T_VOX = 2.0
CT_BULK_LATTICE_SIGMA_N_VOX = 0.8
CT_FEATURE_ADAPTIVE_R_SHELL_VOX = 3.0
CT_FEATURE_ADAPTIVE_BLUR_SIGMA_VOX = 0.75
CT_FEATURE_ADAPTIVE_JITTER = True
CT_FEATURE_ADAPTIVE_JITTER_FRACTION = 0.35
CT_FEATURE_ADAPTIVE_SEED = 17
CT_FEATURE_ADAPTIVE_SPACING_HIGH_VOX = 2
CT_FEATURE_ADAPTIVE_SPACING_MID_VOX = 6
CT_FEATURE_ADAPTIVE_SPACING_LOW_VOX = 10
CT_FEATURE_ADAPTIVE_DIRECTIONAL_CLEARANCE = True
CT_FEATURE_ADAPTIVE_PROBE_CONTAINMENT = True
CT_FEATURE_ADAPTIVE_PROBE_NORMAL_SHRINK = 0.70
CT_FEATURE_ADAPTIVE_PROBE_TANGENT_SHRINK = 0.75
CT_FEATURE_ADAPTIVE_PROBE_ITERS = 4
CT_FEATURE_ADAPTIVE_CLEARANCE_Q_CONT = DEFAULT_BULK_CONTAINMENT_Q_SUPPORT
CT_FEATURE_ADAPTIVE_CLEARANCE_SAFETY = DEFAULT_BULK_CLEARANCE_SAFETY


def _build_contained_lattice_points(
    signed_distance_volume,
    spacing_zyx,
    material_mask_volume=None,
    spacing_vox: float = CT_BULK_LATTICE_SPACING_VOX,
    margin_vox: float = CT_BULK_LATTICE_MARGIN_VOX,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (points_xyz [N,3], sdf_values [N,]) for a regular lattice inside material ownership.

    spacing_vox: lattice step in units of min(spacing_zyx)
    margin_vox:  optional interior distance margin in units of min(spacing_zyx)
    """
    mask = None if material_mask_volume is None else np.asarray(material_mask_volume, dtype=bool)
    sdf = None if signed_distance_volume is None else np.asarray(signed_distance_volume, dtype=np.float32)
    if mask is None and sdf is None:
        raise ValueError("contained lattice requires either material_mask_volume or signed_distance_volume.")
    sz, sy, sx = [float(v) for v in spacing_zyx]
    min_sp = min(sz, sy, sx)
    step_z = max(spacing_vox * sz, min_sp * 0.1)
    step_y = max(spacing_vox * sy, min_sp * 0.1)
    step_x = max(spacing_vox * sx, min_sp * 0.1)
    margin = margin_vox * min_sp

    shape_source = mask if mask is not None else sdf
    D, H, W = shape_source.shape
    z_coords = np.arange(step_z * 0.5, D * sz, step_z, dtype=np.float32)
    y_coords = np.arange(step_y * 0.5, H * sy, step_y, dtype=np.float32)
    x_coords = np.arange(step_x * 0.5, W * sx, step_x, dtype=np.float32)

    gz, gy, gx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    pts_zyx = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], axis=-1)  # (M, 3)

    # sample SDF at lattice candidates
    # Lattice candidates are placed at cell centers (0.5, 1.5, ... voxel units),
    # so ownership should map them back to the containing voxel cell, not the nearest neighbor.
    zi = np.clip(np.floor(pts_zyx[:, 0] / max(sz, 1e-8)).astype(np.int64), 0, D - 1)
    yi = np.clip(np.floor(pts_zyx[:, 1] / max(sy, 1e-8)).astype(np.int64), 0, H - 1)
    xi = np.clip(np.floor(pts_zyx[:, 2] / max(sx, 1e-8)).astype(np.int64), 0, W - 1)

    if sdf is not None:
        sdf_vals = sdf[zi, yi, xi]
    else:
        sdf_vals = np.zeros((zi.shape[0],), dtype=np.float32)

    if mask is not None:
        inside = mask[zi, yi, xi]
        if margin > 0.0:
            inside_distance = ndimage.distance_transform_edt(mask, sampling=np.asarray(spacing_zyx, dtype=np.float32))
            inside &= inside_distance[zi, yi, xi] >= margin
    else:
        inside = sdf_vals < -margin

    pts_zyx = pts_zyx[inside]
    sdf_vals = sdf_vals[inside]

    # convert (z, y, x) world → (x, y, z) world (CTGS convention)
    pts_xyz = pts_zyx[:, [2, 1, 0]].astype(np.float32)
    return pts_xyz, sdf_vals.astype(np.float32)


def _compute_feature_priority_maps(
    material_mask_volume,
    signed_distance_volume,
    intensity_volume,
    spacing_zyx,
    *,
    r_shell_vox: float = CT_FEATURE_ADAPTIVE_R_SHELL_VOX,
    blur_sigma_vox: float = CT_FEATURE_ADAPTIVE_BLUR_SIGMA_VOX,
):
    mask = np.asarray(material_mask_volume, dtype=bool)
    spacing = tuple(float(v) for v in spacing_zyx)
    min_spacing = max(min(spacing), 1e-8)

    if signed_distance_volume is not None:
        sdf = np.asarray(signed_distance_volume, dtype=np.float32)
        inside_vox = np.maximum(-sdf, 0.0) / float(min_spacing)
    else:
        sdf = None
        inside_vox = ndimage.distance_transform_edt(mask).astype(np.float32)

    geo_score = np.clip((float(r_shell_vox) - inside_vox) / max(float(r_shell_vox), 1e-6), 0.0, 1.0)
    geo_score = np.where(mask, geo_score, 0.0).astype(np.float32)

    grad_score = np.zeros(mask.shape, dtype=np.float32)
    grad_xyz = np.zeros(mask.shape + (3,), dtype=np.float32)
    if intensity_volume is not None:
        volume = np.asarray(intensity_volume, dtype=np.float32)
        if volume.shape == mask.shape:
            smoothed = ndimage.gaussian_filter(volume, sigma=max(float(blur_sigma_vox), 0.0)).astype(np.float32)
            grad_z, grad_y, grad_x = np.gradient(smoothed)
            grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).astype(np.float32)
            material_grad = grad_mag[mask & np.isfinite(grad_mag)]
            if material_grad.size > 0:
                q75 = float(np.quantile(material_grad, 0.75))
                q95 = float(np.quantile(material_grad, 0.95))
                if q95 > q75:
                    grad_score = np.clip((grad_mag - q75) / max(q95 - q75, 1e-8), 0.0, 1.0).astype(np.float32)
                    grad_score = np.where(mask, grad_score, 0.0).astype(np.float32)
            grad_xyz[..., 0] = grad_x
            grad_xyz[..., 1] = grad_y
            grad_xyz[..., 2] = grad_z

    priority = np.maximum(geo_score, grad_score).astype(np.float32)

    sdf_grad_xyz = np.zeros(mask.shape + (3,), dtype=np.float32)
    if sdf is not None:
        spacing_z, spacing_y, spacing_x = spacing
        sdf_grad_z, sdf_grad_y, sdf_grad_x = np.gradient(sdf, spacing_z, spacing_y, spacing_x)
        sdf_grad_xyz[..., 0] = sdf_grad_x
        sdf_grad_xyz[..., 1] = sdf_grad_y
        sdf_grad_xyz[..., 2] = sdf_grad_z

    return {
        "inside_vox": inside_vox.astype(np.float32),
        "geo_score": geo_score,
        "grad_score": grad_score,
        "priority": priority,
        "sdf_grad_xyz": sdf_grad_xyz,
        "intensity_grad_xyz": grad_xyz,
    }


def _select_adaptive_cells(tier_mask, priority, spacing_vox: int):
    indices = np.argwhere(np.asarray(tier_mask, dtype=bool))
    if indices.shape[0] == 0:
        return indices, indices
    h = max(int(spacing_vox), 1)
    if h == 1:
        return indices.astype(np.int64), indices.astype(np.int64)
    keys = (indices // h).astype(np.int64)
    grid_shape = tuple(int(np.ceil(float(s) / float(h))) for s in tier_mask.shape)
    linear_keys = np.ravel_multi_index((keys[:, 0], keys[:, 1], keys[:, 2]), grid_shape)
    pri = np.asarray(priority, dtype=np.float32)[indices[:, 0], indices[:, 1], indices[:, 2]]
    order = np.lexsort((-pri, linear_keys))
    sorted_linear = linear_keys[order]
    first = np.concatenate(([True], sorted_linear[1:] != sorted_linear[:-1]))
    picked = order[first]
    return keys[picked].astype(np.int64), indices[picked].astype(np.int64)


def _build_feature_adaptive_bulk_points(
    material_mask_volume,
    signed_distance_volume,
    intensity_volume,
    spacing_zyx,
    *,
    r_shell_vox: float = CT_FEATURE_ADAPTIVE_R_SHELL_VOX,
    blur_sigma_vox: float = CT_FEATURE_ADAPTIVE_BLUR_SIGMA_VOX,
    jitter: bool = CT_FEATURE_ADAPTIVE_JITTER,
    jitter_fraction: float = CT_FEATURE_ADAPTIVE_JITTER_FRACTION,
    seed: int = CT_FEATURE_ADAPTIVE_SEED,
    spacing_high_vox: int = CT_FEATURE_ADAPTIVE_SPACING_HIGH_VOX,
    spacing_mid_vox: int = CT_FEATURE_ADAPTIVE_SPACING_MID_VOX,
    spacing_low_vox: int = CT_FEATURE_ADAPTIVE_SPACING_LOW_VOX,
):
    mask = np.asarray(material_mask_volume, dtype=bool)
    maps = _compute_feature_priority_maps(
        mask,
        signed_distance_volume,
        intensity_volume,
        spacing_zyx,
        r_shell_vox=r_shell_vox,
        blur_sigma_vox=blur_sigma_vox,
    )
    priority = maps["priority"]
    tier_specs = (
        (max(int(spacing_high_vox), 1), mask & (priority >= 0.75)),
        (max(int(spacing_mid_vox), 1), mask & (priority >= 0.35) & (priority < 0.75)),
        (max(int(spacing_low_vox), 1), mask & (priority < 0.35)),
    )
    rng = np.random.default_rng(int(seed))
    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]

    points = []
    selected_indices = []
    local_spacing = []
    for h, tier_mask in tier_specs:
        keys, picked = _select_adaptive_cells(tier_mask, priority, h)
        if picked.shape[0] == 0:
            continue
        if h == 1:
            coords_zyx = picked.astype(np.float32) + 0.5
        else:
            coords_zyx = keys.astype(np.float32) * float(h) + 0.5 * float(h)
        if bool(jitter):
            coords_zyx = coords_zyx + rng.uniform(
                -float(jitter_fraction) * float(h),
                float(jitter_fraction) * float(h),
                size=coords_zyx.shape,
            ).astype(np.float32)
        floor_zyx = np.floor(coords_zyx).astype(np.int64)
        floor_zyx[:, 0] = np.clip(floor_zyx[:, 0], 0, mask.shape[0] - 1)
        floor_zyx[:, 1] = np.clip(floor_zyx[:, 1], 0, mask.shape[1] - 1)
        floor_zyx[:, 2] = np.clip(floor_zyx[:, 2], 0, mask.shape[2] - 1)
        valid = mask[floor_zyx[:, 0], floor_zyx[:, 1], floor_zyx[:, 2]]
        coords_zyx[~valid] = picked[~valid].astype(np.float32) + 0.5
        floor_zyx[~valid] = picked[~valid]
        points.append(
            np.stack(
                (
                    coords_zyx[:, 2] * spacing_x,
                    coords_zyx[:, 1] * spacing_y,
                    coords_zyx[:, 0] * spacing_z,
                ),
                axis=1,
            ).astype(np.float32)
        )
        selected_indices.append(floor_zyx.astype(np.int64))
        local_spacing.append(np.full((floor_zyx.shape[0],), float(h), dtype=np.float32))

    if not points:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_meta = {
            "spacing_vox": np.empty((0,), dtype=np.float32),
            "inside_vox": np.empty((0,), dtype=np.float32),
            "geo_score": np.empty((0,), dtype=np.float32),
            "grad_score": np.empty((0,), dtype=np.float32),
            "sdf_grad_xyz": np.empty((0, 3), dtype=np.float32),
            "intensity_grad_xyz": np.empty((0, 3), dtype=np.float32),
        }
        return empty_points, empty_meta

    points_xyz = np.concatenate(points, axis=0)
    selected = np.concatenate(selected_indices, axis=0)
    spacing_arr = np.concatenate(local_spacing, axis=0)
    z, y, x = selected[:, 0], selected[:, 1], selected[:, 2]
    meta = {
        "spacing_vox": spacing_arr.astype(np.float32),
        "inside_vox": maps["inside_vox"][z, y, x].astype(np.float32),
        "geo_score": maps["geo_score"][z, y, x].astype(np.float32),
        "grad_score": maps["grad_score"][z, y, x].astype(np.float32),
        "sdf_grad_xyz": maps["sdf_grad_xyz"][z, y, x].astype(np.float32),
        "intensity_grad_xyz": maps["intensity_grad_xyz"][z, y, x].astype(np.float32),
    }
    return points_xyz, meta


def _build_feature_adaptive_bulk_attributes(
    adaptive_meta,
    min_spacing,
    *,
    anisotropic: bool,
    interior_points=None,
    signed_distance_volume=None,
    spacing_zyx=None,
):
    spacing_vox = np.asarray(adaptive_meta.get("spacing_vox"), dtype=np.float32).reshape(-1)
    count = int(spacing_vox.shape[0])
    if count == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )
    base_sigma = np.maximum(0.65 * spacing_vox * float(min_spacing), 0.35 * float(min_spacing))
    scales = np.repeat(base_sigma[:, np.newaxis], 3, axis=1).astype(np.float32)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[np.newaxis, :, :], count, axis=0)
    normals = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), count, axis=0)
    if not bool(anisotropic):
        return scales, rotations, normals

    geo_score = np.asarray(adaptive_meta.get("geo_score"), dtype=np.float32).reshape(-1)
    grad_score = np.asarray(adaptive_meta.get("grad_score"), dtype=np.float32).reshape(-1)
    inside_vox = np.asarray(adaptive_meta.get("inside_vox"), dtype=np.float32).reshape(-1)
    sdf_grad = np.asarray(adaptive_meta.get("sdf_grad_xyz"), dtype=np.float32).reshape(-1, 3)
    intensity_grad = np.asarray(adaptive_meta.get("intensity_grad_xyz"), dtype=np.float32).reshape(-1, 3)
    if interior_points is not None and signed_distance_volume is not None and spacing_zyx is not None:
        sdf_values, sdf_grad = _sample_sdf_and_gradient_at_points(interior_points, signed_distance_volume, spacing_zyx)
        inside_vox = np.maximum(-sdf_values, 0.0).astype(np.float32) / max(float(min_spacing), 1e-8)
    tangent_sigma = np.clip(
        0.80 * spacing_vox * float(min_spacing),
        1.6 * float(min_spacing),
        2.0 * float(min_spacing),
    ).astype(np.float32)
    geom_normal_sigma = 0.6 * float(min_spacing)
    grad_normal_sigma = 0.7 * float(min_spacing)
    min_normal = 0.30 * float(min_spacing)
    for index in range(count):
        if geo_score[index] >= 0.35 and geo_score[index] >= grad_score[index]:
            normal = sdf_grad[index]
            normal_sigma = min(geom_normal_sigma, max(0.95 * float(inside_vox[index]) * float(min_spacing), min_normal))
        elif grad_score[index] >= 0.75:
            normal = intensity_grad[index]
            normal_sigma = grad_normal_sigma
        else:
            continue
        if np.linalg.norm(normal) <= 1e-8:
            continue
        tangent_u, tangent_v, normal = _build_frame_from_normal(normal)
        rotations[index] = np.stack((tangent_u, tangent_v, normal), axis=1)
        normals[index] = normal
        scales[index] = np.array([tangent_sigma[index], tangent_sigma[index], normal_sigma], dtype=np.float32)
    return scales.astype(np.float32), rotations.astype(np.float32), normals.astype(np.float32)


def _points_inside_mask(points_xyz, material_mask_volume, spacing_zyx):
    mask = np.asarray(material_mask_volume, dtype=bool)
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    spacing_z, spacing_y, spacing_x = [float(v) for v in spacing_zyx]
    x = np.floor(points[:, 0] / max(spacing_x, 1e-8)).astype(np.int64)
    y = np.floor(points[:, 1] / max(spacing_y, 1e-8)).astype(np.int64)
    z = np.floor(points[:, 2] / max(spacing_z, 1e-8)).astype(np.int64)
    in_bounds = (
        (z >= 0)
        & (z < mask.shape[0])
        & (y >= 0)
        & (y < mask.shape[1])
        & (x >= 0)
        & (x < mask.shape[2])
    )
    inside = np.zeros((points.shape[0],), dtype=bool)
    valid = np.nonzero(in_bounds)[0]
    if valid.size > 0:
        inside[valid] = mask[z[valid], y[valid], x[valid]]
    return inside


def _sample_sdf_trilinear_numpy(points_xyz, signed_distance_volume, spacing_zyx):
    sdf = np.asarray(signed_distance_volume, dtype=np.float32)
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    x_idx, y_idx, z_idx = world_xyz_to_voxel_float_numpy(points, spacing_zyx)
    coords = np.stack((z_idx, y_idx, x_idx), axis=0)
    return ndimage.map_coordinates(sdf, coords, order=1, mode="constant", cval=1e6).astype(np.float32)


def _points_inside_domain(points_xyz, material_mask_volume, spacing_zyx, signed_distance_volume=None):
    if signed_distance_volume is not None:
        return _sample_sdf_trilinear_numpy(points_xyz, signed_distance_volume, spacing_zyx) < 0.0
    return _points_inside_mask(points_xyz, material_mask_volume, spacing_zyx)


def _directional_clearance_one_side(
    center,
    direction,
    material_mask_volume,
    spacing_zyx,
    max_distance,
    step_distance,
    signed_distance_volume=None,
):
    max_distance = float(max(max_distance, 0.0))
    if max_distance <= 0.0:
        return 0.0
    center = np.asarray(center, dtype=np.float32).reshape(3)
    direction = np.asarray(direction, dtype=np.float32).reshape(3)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-8:
        return 0.0
    direction = direction / direction_norm
    step_distance = max(float(step_distance), 1e-6)
    steps = max(1, int(np.ceil(max_distance / step_distance)))
    distances = np.linspace(step_distance, max_distance, steps, dtype=np.float32)
    probes = center.reshape(1, 3) + distances.reshape(-1, 1) * direction.reshape(1, 3)
    inside = _points_inside_domain(probes, material_mask_volume, spacing_zyx, signed_distance_volume=signed_distance_volume)
    outside = np.nonzero(~inside)[0]
    if outside.size == 0:
        return max_distance
    high = float(distances[int(outside[0])])
    low = 0.0 if int(outside[0]) == 0 else float(distances[int(outside[0]) - 1])
    for _ in range(5):
        mid = 0.5 * (low + high)
        mid_point = center + mid * direction
        if bool(
            _points_inside_domain(
                mid_point.reshape(1, 3),
                material_mask_volume,
                spacing_zyx,
                signed_distance_volume=signed_distance_volume,
            )[0]
        ):
            low = mid
        else:
            high = mid
    return low


def _apply_directional_clearance_scales(
    interior_points,
    bulk_scales,
    bulk_rotations,
    material_mask_volume,
    spacing_zyx,
    min_spacing,
    *,
    signed_distance_volume=None,
    q_cont: float = CT_FEATURE_ADAPTIVE_CLEARANCE_Q_CONT,
    safety: float = CT_FEATURE_ADAPTIVE_CLEARANCE_SAFETY,
):
    points = np.asarray(interior_points, dtype=np.float32).reshape(-1, 3)
    scales = np.asarray(bulk_scales, dtype=np.float32).reshape(-1, 3).copy()
    rotations = np.asarray(bulk_rotations, dtype=np.float32).reshape(-1, 3, 3)
    stats = {
        "num_init_candidates_clearance_limited": 0,
        "init_clearance_scale_ratio_p10": float("nan"),
        "init_clearance_scale_ratio_p50": float("nan"),
    }
    if points.shape[0] == 0 or (material_mask_volume is None and signed_distance_volume is None):
        return scales.astype(np.float32), stats
    sqrt_q = math.sqrt(max(float(q_cont), 1e-8))
    safety = max(float(safety), 1e-6)
    step_distance = max(0.35 * float(min_spacing), 1e-6)
    ratios = []
    limited = np.zeros((points.shape[0],), dtype=bool)
    for index in range(points.shape[0]):
        if not bool(
            _points_inside_domain(
                points[index].reshape(1, 3),
                material_mask_volume,
                spacing_zyx,
                signed_distance_volume=signed_distance_volume,
            )[0]
        ):
            scales[index] = 0.0
            limited[index] = True
            ratios.extend([0.0, 0.0, 0.0])
            continue
        for axis in range(3):
            desired = float(scales[index, axis])
            if desired <= 0.0:
                ratios.append(1.0)
                continue
            max_distance = desired * sqrt_q / safety
            direction = rotations[index, :, axis]
            c_pos = _directional_clearance_one_side(
                points[index],
                direction,
                material_mask_volume,
                spacing_zyx,
                max_distance,
                step_distance,
                signed_distance_volume=signed_distance_volume,
            )
            c_neg = _directional_clearance_one_side(
                points[index],
                -direction,
                material_mask_volume,
                spacing_zyx,
                max_distance,
                step_distance,
                signed_distance_volume=signed_distance_volume,
            )
            safe_scale = safety * min(c_pos, c_neg) / sqrt_q
            new_scale = min(desired, max(float(safe_scale), 0.0))
            if new_scale < 0.999 * desired:
                limited[index] = True
            scales[index, axis] = new_scale
            ratios.append(new_scale / max(desired, 1e-8))
    stats["num_init_candidates_clearance_limited"] = int(limited.sum())
    finite_ratios = np.asarray(ratios, dtype=np.float32)
    finite_ratios = finite_ratios[np.isfinite(finite_ratios)]
    if finite_ratios.size > 0:
        stats["init_clearance_scale_ratio_p10"] = float(np.quantile(finite_ratios, 0.10))
        stats["init_clearance_scale_ratio_p50"] = float(np.quantile(finite_ratios, 0.50))
    return scales.astype(np.float32), stats


def _probe_candidate_axes(
    center,
    scales,
    rotation,
    material_mask_volume,
    spacing_zyx,
    *,
    signed_distance_volume=None,
    q_support: float = CT_FEATURE_ADAPTIVE_CLEARANCE_Q_CONT,
):
    center = np.asarray(center, dtype=np.float32).reshape(3)
    scales = np.asarray(scales, dtype=np.float32).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    sqrt_q = math.sqrt(max(float(q_support), 1e-8))
    probes = [center]
    direction_axes = []
    for direction in ellipsoid_probe_directions():
        start = len(probes)
        local_dir = np.asarray(direction, dtype=np.float32)
        offset = rotation @ (sqrt_q * scales * local_dir)
        probes.append(center + offset)
        probes.append(center - offset)
        direction_axes.append((start, start + 2, np.abs(local_dir) > 1e-5))
    inside = _points_inside_domain(
        np.asarray(probes, dtype=np.float32),
        material_mask_volume,
        spacing_zyx,
        signed_distance_volume=signed_distance_volume,
    )
    axis_bad = np.zeros((3,), dtype=bool)
    for start, end, affected_axes in direction_axes:
        if not bool(np.all(inside[start:end])):
            axis_bad |= affected_axes
    legal = bool(inside[0] and not np.any(axis_bad))
    return legal, axis_bad


def _probe_correct_feature_adaptive_bulk_attributes(
    interior_points,
    bulk_scales,
    bulk_rotations,
    bulk_normals,
    signed_distance_volume,
    material_mask_volume,
    spacing_zyx,
    min_spacing,
    *,
    normal_shrink: float = CT_FEATURE_ADAPTIVE_PROBE_NORMAL_SHRINK,
    tangent_shrink: float = CT_FEATURE_ADAPTIVE_PROBE_TANGENT_SHRINK,
    max_iters: int = CT_FEATURE_ADAPTIVE_PROBE_ITERS,
    q_support: float = CT_FEATURE_ADAPTIVE_CLEARANCE_Q_CONT,
):
    points = np.asarray(interior_points, dtype=np.float32).reshape(-1, 3)
    scales = np.asarray(bulk_scales, dtype=np.float32).reshape(-1, 3).copy()
    rotations = np.asarray(bulk_rotations, dtype=np.float32).reshape(-1, 3, 3).copy()
    normals = np.asarray(bulk_normals, dtype=np.float32).reshape(-1, 3).copy()
    stats = {
        "num_init_candidates_total": int(points.shape[0]),
        "num_init_candidates_shrunk": 0,
        "num_init_candidates_downgraded": 0,
        "num_init_candidates_rejected": 0,
    }
    if points.shape[0] == 0 or (material_mask_volume is None and signed_distance_volume is None):
        return scales, rotations, normals, np.ones((points.shape[0],), dtype=bool), stats

    sdf_values, _ = _sample_sdf_and_gradient_at_points(points, signed_distance_volume, spacing_zyx)
    keep = np.ones((points.shape[0],), dtype=bool)
    min_scale = max(0.08 * float(min_spacing), 1e-8)
    for index in range(points.shape[0]):
        legal, axis_bad = _probe_candidate_axes(
            points[index],
            scales[index],
            rotations[index],
            material_mask_volume,
            spacing_zyx,
            signed_distance_volume=signed_distance_volume,
            q_support=q_support,
        )
        changed = False
        if not legal:
            for _ in range(max(int(max_iters), 0)):
                if axis_bad.shape[0] > 2 and axis_bad[2]:
                    scales[index, 2] = max(scales[index, 2] * float(normal_shrink), min_scale)
                    changed = True
                if axis_bad.shape[0] > 0 and axis_bad[0]:
                    scales[index, 0] = max(scales[index, 0] * float(tangent_shrink), min_scale)
                    changed = True
                if axis_bad.shape[0] > 1 and axis_bad[1]:
                    scales[index, 1] = max(scales[index, 1] * float(tangent_shrink), min_scale)
                    changed = True
                if axis_bad.shape[0] > 3 and np.any(axis_bad[3:]):
                    scales[index, 0] = max(scales[index, 0] * float(tangent_shrink), min_scale)
                    scales[index, 1] = max(scales[index, 1] * float(tangent_shrink), min_scale)
                    changed = True
                legal, axis_bad = _probe_candidate_axes(
                    points[index],
                    scales[index],
                    rotations[index],
                    material_mask_volume,
                    spacing_zyx,
                    signed_distance_volume=signed_distance_volume,
                    q_support=q_support,
                )
                if legal:
                    break
        if legal:
            if changed:
                stats["num_init_candidates_shrunk"] += 1
            continue

        inside_distance = max(float(-sdf_values[index]), 0.0)
        radius = min(0.60 * float(min_spacing), 0.70 * inside_distance)
        if radius > min_scale:
            fallback_scale = np.full((3,), max(radius, min_scale), dtype=np.float32)
            fallback_rotation = np.eye(3, dtype=np.float32)
            fallback_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            legal, _ = _probe_candidate_axes(
                points[index],
                fallback_scale,
                fallback_rotation,
                material_mask_volume,
                spacing_zyx,
                signed_distance_volume=signed_distance_volume,
                q_support=q_support,
            )
            if legal:
                scales[index] = fallback_scale
                rotations[index] = fallback_rotation
                normals[index] = fallback_normal
                stats["num_init_candidates_downgraded"] += 1
                continue

        keep[index] = False
        stats["num_init_candidates_rejected"] += 1
    return scales.astype(np.float32), rotations.astype(np.float32), normals.astype(np.float32), keep, stats


def _normalize_np(vector):
    norm = np.linalg.norm(vector)
    if norm <= 1e-8:
        return None
    return vector / norm


def _orthogonal_hint(normal):
    hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(hint, normal)) > 0.9:
        hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent = hint - np.dot(hint, normal) * normal
    tangent = _normalize_np(tangent)
    if tangent is None:
        tangent = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        tangent = tangent - np.dot(tangent, normal) * normal
        tangent = _normalize_np(tangent)
    return tangent


def _sample_sdf_and_gradient_at_points(points_world, sdf_volume, spacing_zyx):
    """Sample SDF value and gradient at world-space points via nearest-voxel finite differences.

    Returns (sdf_values, sdf_gradients), both float32. Gradients are in world-space (x, y, z).
    """
    sdf = np.asarray(sdf_volume, dtype=np.float32)
    spacing_z, spacing_y, spacing_x = np.asarray(spacing_zyx, dtype=np.float32)
    points_world = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    z, y, x = world_xyz_to_voxel_indices_floor_numpy(points_world, spacing_zyx, shape_dhw=sdf.shape)

    sdf_values = sdf[z, y, x]

    z_plus = np.clip(z + 1, 0, sdf.shape[0] - 1)
    z_minus = np.clip(z - 1, 0, sdf.shape[0] - 1)
    y_plus = np.clip(y + 1, 0, sdf.shape[1] - 1)
    y_minus = np.clip(y - 1, 0, sdf.shape[1] - 1)
    x_plus = np.clip(x + 1, 0, sdf.shape[2] - 1)
    x_minus = np.clip(x - 1, 0, sdf.shape[2] - 1)

    grad_z = (sdf[z_plus, y, x] - sdf[z_minus, y, x]) / (2.0 * max(float(spacing_z), 1e-8))
    grad_y = (sdf[z, y_plus, x] - sdf[z, y_minus, x]) / (2.0 * max(float(spacing_y), 1e-8))
    grad_x = (sdf[z, y, x_plus] - sdf[z, y, x_minus]) / (2.0 * max(float(spacing_x), 1e-8))
    sdf_gradients = np.stack([grad_x, grad_y, grad_z], axis=1).astype(np.float32)
    return sdf_values.astype(np.float32), sdf_gradients


def _augment_interior_points(
    interior_points,
    density_seed,
    material_id,
    material_mask,
    spacing_zyx,
    target_count,
):
    """Augment interior_points up to target_count by sampling material voxels.

    Uses the same (x*sx, y*sy, z*sz) coord format Phase1 uses. New density_seed entries
    default to 0.5 (will be re-fit by training). New material_id entries default to 0.
    """
    interior_points = np.asarray(interior_points, dtype=np.float32).reshape(-1, 3)
    density_seed = np.asarray(density_seed, dtype=np.float32).reshape(-1, 1)
    material_id = np.asarray(material_id, dtype=np.int64).reshape(-1, 1)
    current = int(interior_points.shape[0])
    target = int(target_count)
    if current >= target or material_mask is None:
        return interior_points, density_seed, material_id

    mask = np.asarray(material_mask, dtype=bool)
    candidate_indices = np.argwhere(mask)
    if candidate_indices.shape[0] == 0:
        return interior_points, density_seed, material_id

    needed = target - current
    replace = candidate_indices.shape[0] < needed
    selected = np.random.choice(candidate_indices.shape[0], size=needed, replace=replace)
    voxel_indices = candidate_indices[selected]
    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
    jitter = np.random.random((needed, 3)).astype(np.float32)
    new_points = np.stack(
        (
            (voxel_indices[:, 2].astype(np.float32) + jitter[:, 0]) * spacing_x,
            (voxel_indices[:, 1].astype(np.float32) + jitter[:, 1]) * spacing_y,
            (voxel_indices[:, 0].astype(np.float32) + jitter[:, 2]) * spacing_z,
        ),
        axis=1,
    ).astype(np.float32)
    new_density_seed = np.full((needed, 1), 0.5, dtype=np.float32)
    new_material_id = np.zeros((needed, 1), dtype=np.int64)
    return (
        np.concatenate([interior_points, new_points], axis=0),
        np.concatenate([density_seed, new_density_seed], axis=0),
        np.concatenate([material_id, new_material_id], axis=0),
    )


def _build_sdf_aligned_bulk_attributes(
    interior_points,
    signed_distance_volume,
    spacing_zyx,
    nn_distance,
    min_spacing,
):
    """Compute SDF-aligned bulk ellipsoids.

    Deep material stays close to isotropic. Near the material boundary, the SDF
    normal axis starts thinner while tangent axes keep enough support to cover
    narrow material shells without forcing all bulk to shrink like spheres.
    """
    bulk_count = int(interior_points.shape[0])
    if bulk_count == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    sdf_values, sdf_gradients = _sample_sdf_and_gradient_at_points(
        interior_points, signed_distance_volume, spacing_zyx
    )
    edt_distance = np.abs(sdf_values).astype(np.float32)

    min_scale = CT_DENSE_INIT_BULK_MIN_SCALE_RATIO * min_spacing
    max_scale = CT_DENSE_INIT_BULK_MAX_SCALE_RATIO * min_spacing
    nn_radius = np.asarray(nn_distance, dtype=np.float32).reshape(-1) * 0.5
    containment_radius = (edt_distance / float(CT_DENSE_INIT_BULK_CONTAINMENT_SIGMA)).astype(np.float32)
    tangent_radius = np.clip(
        nn_radius,
        a_min=min_scale,
        a_max=max_scale,
    ).astype(np.float32)
    tangent_radius = np.minimum(tangent_radius, containment_radius).astype(np.float32)
    normal_radius = np.clip(
        np.minimum(tangent_radius, containment_radius),
        a_min=min_scale,
        a_max=max_scale,
    ).astype(np.float32)

    fallback_radius = np.minimum(tangent_radius, normal_radius)
    bulk_scales = np.repeat(fallback_radius[:, np.newaxis], 3, axis=1).astype(np.float32)
    bulk_rotations = np.repeat(np.eye(3, dtype=np.float32)[np.newaxis, :, :], bulk_count, axis=0)
    bulk_normals = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), bulk_count, axis=0)
    gradient_norm = np.linalg.norm(sdf_gradients, axis=1)
    trusted = gradient_norm > float(CT_DENSE_INIT_BULK_GRADIENT_THRESHOLD)
    for index in np.nonzero(trusted)[0]:
        normal = sdf_gradients[index] / max(float(gradient_norm[index]), 1e-8)
        tangent_u, tangent_v, normal = _build_frame_from_normal(normal)
        bulk_rotations[index] = np.stack((tangent_u, tangent_v, normal), axis=1)
        bulk_normals[index] = normal
        bulk_scales[index] = np.array(
            [tangent_radius[index], tangent_radius[index], normal_radius[index]],
            dtype=np.float32,
        )
    return bulk_scales, bulk_rotations, bulk_normals


def _contain_initial_bulk_attributes(
    interior_points,
    bulk_scales,
    bulk_rotations,
    bulk_normals,
    signed_distance_volume,
    spacing_zyx,
    min_spacing,
):
    """Shrink/reject initial bulk ellipsoids whose truncated footprint reaches air."""
    if signed_distance_volume is None or int(np.asarray(interior_points).reshape(-1, 3).shape[0]) == 0:
        keep = np.ones((np.asarray(interior_points).reshape(-1, 3).shape[0],), dtype=bool)
        return bulk_scales, bulk_rotations, bulk_normals, keep
    points = np.asarray(interior_points, dtype=np.float32).reshape(-1, 3)
    scales = np.asarray(bulk_scales, dtype=np.float32).reshape(-1, 3).copy()
    rotations = np.asarray(bulk_rotations, dtype=np.float32).reshape(-1, 3, 3)
    normals = np.asarray(bulk_normals, dtype=np.float32).reshape(-1, 3)
    sdf_values, sdf_gradients = _sample_sdf_and_gradient_at_points(points, signed_distance_volume, spacing_zyx)
    grad_norm = np.linalg.norm(sdf_gradients, axis=1, keepdims=True)
    sdf_normals = np.where(grad_norm > 1e-8, sdf_gradients / np.maximum(grad_norm, 1e-8), normals).astype(np.float32)
    local_normals = np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), sdf_normals)
    radius = np.sqrt(np.sum((local_normals * scales) ** 2, axis=1)).astype(np.float32)
    allowed = ((-sdf_values) / float(CT_DENSE_INIT_BULK_CONTAINMENT_SIGMA)).astype(np.float32)
    min_allowed = max(float(min_spacing) * 0.05, 1e-6)
    keep = np.isfinite(allowed) & np.isfinite(radius) & (allowed > min_allowed)
    shrink = keep & (radius > allowed)
    factor = np.ones_like(radius, dtype=np.float32)
    factor[shrink] = np.clip(allowed[shrink] / np.maximum(radius[shrink], 1e-8), 0.0, 1.0)
    scales *= factor[:, np.newaxis]
    keep &= np.all(scales >= min_allowed, axis=1)
    return scales.astype(np.float32), rotations.astype(np.float32), normals.astype(np.float32), keep


def _poisson_disk_filter_boundary(
    boundary_points,
    boundary_strength,
    base_spacing,
    curvature_alpha=CT_DENSE_INIT_SURFACE_POISSON_CURVATURE_ALPHA,
):
    """Greedy Poisson-disk filter with curvature-adaptive radius.

    Uses ``boundary_strength`` as a curvature proxy. High-strength anchors get a smaller
    min-distance, so they survive the filter at higher density. Returns indices into
    the input arrays that pass the filter.
    """
    points = np.asarray(boundary_points, dtype=np.float32).reshape(-1, 3)
    n = points.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)

    strength = np.asarray(boundary_strength, dtype=np.float32).reshape(-1)
    if strength.shape[0] != n:
        strength = np.full((n,), 0.5, dtype=np.float32)
    finite = np.isfinite(strength)
    if not np.any(finite):
        normalized = np.zeros((n,), dtype=np.float32)
    else:
        s_min = float(strength[finite].min())
        s_max = float(strength[finite].max())
        if s_max > s_min:
            normalized = np.clip((strength - s_min) / max(s_max - s_min, 1e-8), 0.0, 1.0).astype(np.float32)
        else:
            normalized = np.zeros((n,), dtype=np.float32)
        normalized = np.where(finite, normalized, np.float32(0.0))

    target_radius = (float(base_spacing) / (1.0 + normalized * float(curvature_alpha))).astype(np.float32)
    priority = np.argsort(-normalized, kind="stable")

    tree = cKDTree(points)
    accepted = np.zeros((n,), dtype=bool)
    for index in priority:
        radius = float(target_radius[index])
        if radius <= 0.0:
            accepted[index] = True
            continue
        neighbors = tree.query_ball_point(points[index], r=radius)
        conflict = False
        for nbr in neighbors:
            if nbr == int(index):
                continue
            if accepted[nbr]:
                conflict = True
                break
        if not conflict:
            accepted[index] = True
    return np.nonzero(accepted)[0].astype(np.int64)


def _build_frame_from_normal(normal, tangent_hint=None):
    normal = _normalize_np(np.asarray(normal, dtype=np.float32))
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    if tangent_hint is None:
        tangent_u = _orthogonal_hint(normal)
    else:
        tangent_u = np.asarray(tangent_hint, dtype=np.float32)
        tangent_u = tangent_u - np.dot(tangent_u, normal) * normal
        tangent_u = _normalize_np(tangent_u)
        if tangent_u is None:
            tangent_u = _orthogonal_hint(normal)

    tangent_v = _normalize_np(np.cross(normal, tangent_u))
    if tangent_v is None:
        tangent_u = _orthogonal_hint(normal)
        tangent_v = _normalize_np(np.cross(normal, tangent_u))
    tangent_u = _normalize_np(np.cross(tangent_v, normal))
    return tangent_u.astype(np.float32), tangent_v.astype(np.float32), normal.astype(np.float32)
