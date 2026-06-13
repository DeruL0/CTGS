import random

import numpy as np
import torch
from scipy import ndimage

from ct_pipeline.geometry.coordinates import voxel_center_world_bounds_torch, world_xyz_to_voxel_indices_floor_torch
from ct_pipeline.training.losses import sample_volume_field


CT_MAX_FIELD_POOL_INDICES = 65_536


def _default_roi_bbox(volume_shape_dhw):
    depth, height, width = [int(value) for value in volume_shape_dhw]
    return np.asarray([[0, depth], [0, height], [0, width]], dtype=np.int32)


def _flat_indices_to_zyx(flat_indices: np.ndarray, shape_dhw) -> np.ndarray:
    flat = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    if flat.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    depth, height, width = [int(value) for value in shape_dhw]
    plane = max(1, height * width)
    z = flat // plane
    rem = flat - z * plane
    y = rem // max(1, width)
    x = rem - y * max(1, width)
    indices = np.empty((flat.size, 3), dtype=np.int32)
    indices[:, 0] = z.astype(np.int32, copy=False)
    indices[:, 1] = y.astype(np.int32, copy=False)
    indices[:, 2] = x.astype(np.int32, copy=False)
    return indices


def _mask_to_candidate_indices(mask: np.ndarray, max_count: int = CT_MAX_FIELD_POOL_INDICES) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    true_count = int(np.count_nonzero(mask))
    if true_count == 0:
        return np.empty((0, 3), dtype=np.int32)

    max_count = max(1, int(max_count))
    flat_mask = mask.reshape(-1)
    if true_count <= max_count:
        return _flat_indices_to_zyx(np.flatnonzero(flat_mask), mask.shape)

    total_count = int(flat_mask.size)
    density = max(float(true_count) / max(float(total_count), 1.0), 1e-6)
    draw_count = min(total_count, max(max_count * 2, int(np.ceil(float(max_count) / density * 1.25))))
    accepted_parts = []
    accepted_count = 0
    for _ in range(16):
        sampled = np.random.randint(0, total_count, size=draw_count, dtype=np.int64)
        accepted = sampled[flat_mask[sampled]]
        if accepted.size > 0:
            accepted_parts.append(accepted)
            accepted_count += int(accepted.size)
            if accepted_count >= max_count:
                break
    if not accepted_parts:
        return _flat_indices_to_zyx(np.flatnonzero(flat_mask)[:max_count], mask.shape)
    flat = np.concatenate(accepted_parts, axis=0)[:max_count]
    return _flat_indices_to_zyx(flat, mask.shape)


def _component_balanced_mask_to_candidate_indices(mask: np.ndarray, max_count: int = CT_MAX_FIELD_POOL_INDICES) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    true_count = int(np.count_nonzero(mask))
    if true_count == 0:
        return np.empty((0, 3), dtype=np.int32)
    if true_count <= int(max_count):
        return _flat_indices_to_zyx(np.flatnonzero(mask.reshape(-1)), mask.shape)

    labels, component_count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if component_count <= 1:
        return _mask_to_candidate_indices(mask, max_count=max_count)

    flat_labels = labels.reshape(-1)
    nonzero_flat = np.flatnonzero(flat_labels)
    if nonzero_flat.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    nonzero_labels = flat_labels[nonzero_flat].astype(np.int32, copy=False)
    order = np.argsort(nonzero_labels, kind="stable")
    sorted_flat = nonzero_flat[order]
    sorted_labels = nonzero_labels[order]
    component_ids, component_starts, component_sizes = np.unique(
        sorted_labels,
        return_index=True,
        return_counts=True,
    )

    budget_per_component = max(1, int(max_count) // int(component_count))
    selected_parts = []
    selected_count = 0
    for start, size in zip(component_starts, component_sizes):
        component_flat = sorted_flat[start : start + size]
        if component_flat.size == 0:
            continue
        take = min(component_flat.size, budget_per_component)
        if take < component_flat.size:
            component_flat = np.random.choice(component_flat, size=take, replace=False)
        selected_parts.append(component_flat.astype(np.int64, copy=False))
        selected_count += int(take)

    if selected_count < int(max_count) and selected_parts:
        selected_flat = np.concatenate(selected_parts, axis=0)
        remaining_flat = nonzero_flat[
            np.isin(nonzero_flat, selected_flat, assume_unique=False, invert=True)
        ]
        remaining_budget = int(max_count) - selected_count
        if remaining_flat.size > 0 and remaining_budget > 0:
            take = min(remaining_flat.size, remaining_budget)
            if take < remaining_flat.size:
                remaining_flat = np.random.choice(remaining_flat, size=take, replace=False)
            selected_parts.append(remaining_flat.astype(np.int64, copy=False))

    if not selected_parts:
        return np.empty((0, 3), dtype=np.int32)
    flat = np.concatenate(selected_parts, axis=0)[: int(max_count)]
    return _flat_indices_to_zyx(flat, mask.shape)


def _concat_candidate_indices(*candidate_sets, max_count: int = CT_MAX_FIELD_POOL_INDICES):
    non_empty = [candidates for candidates in candidate_sets if _candidate_count(candidates) > 0]
    if not non_empty:
        return None
    first = non_empty[0]
    if isinstance(first, torch.Tensor):
        combined = torch.cat([candidates.to(device=first.device, dtype=torch.int32) for candidates in non_empty], dim=0)
        if combined.shape[0] > int(max_count):
            selected = torch.randperm(int(combined.shape[0]), device=combined.device)[: int(max_count)]
            combined = combined.index_select(0, selected)
        return combined
    combined = np.concatenate([np.asarray(candidates, dtype=np.int32).reshape(-1, 3) for candidates in non_empty], axis=0)
    if combined.shape[0] > int(max_count):
        selected = np.random.choice(combined.shape[0], size=int(max_count), replace=False)
        combined = combined[selected]
    return combined.astype(np.int32, copy=False)


def _boundary_material_subvoxel_points(analysis, material_mask, spacing_zyx, device=None):
    boundary_points = analysis.get("boundary_points")
    boundary_normals = analysis.get("boundary_normals", analysis.get("boundary_normal"))
    if boundary_points is None or boundary_normals is None:
        if device is not None:
            return torch.empty((0, 3), dtype=torch.float32, device=device)
        return np.empty((0, 3), dtype=np.float32)

    offsets_vox = (0.25, 0.5, 0.75, 1.0)
    min_spacing = float(min(float(value) for value in spacing_zyx))
    if isinstance(material_mask, torch.Tensor):
        tensor_device = material_mask.device if device is None else torch.device(device)
        points = torch.as_tensor(boundary_points, dtype=torch.float32, device=tensor_device).reshape(-1, 3)
        normals = torch.as_tensor(boundary_normals, dtype=torch.float32, device=tensor_device).reshape(-1, 3)
        if points.numel() == 0 or normals.shape[0] != points.shape[0]:
            return torch.empty((0, 3), dtype=torch.float32, device=tensor_device)
        normals = torch.nn.functional.normalize(normals, dim=1, eps=1e-6)
        deltas = torch.as_tensor(offsets_vox, dtype=torch.float32, device=tensor_device).reshape(1, -1, 1) * min_spacing
        candidates = (points[:, None, :] - normals[:, None, :] * deltas).reshape(-1, 3)
        z_idx, y_idx, x_idx = world_xyz_to_voxel_indices_floor_torch(
            candidates,
            spacing_zyx,
            shape_dhw=tuple(int(value) for value in material_mask.shape),
        )
        keep = material_mask.to(device=tensor_device, dtype=torch.bool)[z_idx, y_idx, x_idx]
        return candidates[keep]

    material_np = np.asarray(material_mask, dtype=bool)
    points_np = np.asarray(boundary_points, dtype=np.float32).reshape(-1, 3)
    normals_np = np.asarray(boundary_normals, dtype=np.float32).reshape(-1, 3)
    if points_np.size == 0 or normals_np.shape[0] != points_np.shape[0]:
        return np.empty((0, 3), dtype=np.float32)
    norm = np.linalg.norm(normals_np, axis=1, keepdims=True)
    normals_np = np.divide(normals_np, np.maximum(norm, 1e-6), out=np.zeros_like(normals_np), where=norm > 0)
    deltas_np = np.asarray(offsets_vox, dtype=np.float32).reshape(1, -1, 1) * np.float32(min_spacing)
    candidates_np = (points_np[:, None, :] - normals_np[:, None, :] * deltas_np).reshape(-1, 3)
    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
    z_idx = np.clip(np.floor(candidates_np[:, 2] / max(spacing_z, 1e-8)).astype(np.int64), 0, material_np.shape[0] - 1)
    y_idx = np.clip(np.floor(candidates_np[:, 1] / max(spacing_y, 1e-8)).astype(np.int64), 0, material_np.shape[1] - 1)
    x_idx = np.clip(np.floor(candidates_np[:, 0] / max(spacing_x, 1e-8)).astype(np.int64), 0, material_np.shape[2] - 1)
    keep_np = material_np[z_idx, y_idx, x_idx]
    return candidates_np[keep_np].astype(np.float32, copy=False)


def _torch_flat_indices_to_zyx(flat_indices: torch.Tensor, shape_dhw) -> torch.Tensor:
    flat = flat_indices.to(dtype=torch.long).reshape(-1)
    if flat.numel() == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=flat.device)
    _, height, width = [int(value) for value in shape_dhw]
    plane = max(1, height * width)
    z = torch.div(flat, plane, rounding_mode="floor")
    rem = flat - z * plane
    y = torch.div(rem, max(1, width), rounding_mode="floor")
    x = rem - y * max(1, width)
    return torch.stack((z, y, x), dim=1).to(dtype=torch.int32)


def _torch_mask_to_candidate_indices(mask: torch.Tensor, max_count: int = CT_MAX_FIELD_POOL_INDICES) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    true_count = int(torch.count_nonzero(mask).item())
    if true_count == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=mask.device)

    max_count = max(1, int(max_count))
    flat_mask = mask.reshape(-1)
    if true_count <= max_count:
        return torch.nonzero(mask, as_tuple=False).to(dtype=torch.int32)

    total_count = int(flat_mask.numel())
    density = max(float(true_count) / max(float(total_count), 1.0), 1e-6)
    draw_count = min(total_count, max(max_count * 2, int(np.ceil(float(max_count) / density * 1.25))))
    chunk_count = min(draw_count, max(8192, max_count // 4))
    accepted_parts = []
    accepted_count = 0
    for _ in range(64):
        sampled = torch.randint(0, total_count, (chunk_count,), dtype=torch.long, device=mask.device)
        accepted = sampled[flat_mask[sampled]]
        if accepted.numel() == 0:
            continue
        accepted_parts.append(accepted)
        accepted_count += int(accepted.numel())
        if accepted_count >= max_count:
            break
    if not accepted_parts:
        nonzero = torch.nonzero(mask, as_tuple=False)
        return nonzero[:max_count].to(dtype=torch.int32)
    flat = torch.cat(accepted_parts, dim=0)[:max_count]
    return _torch_flat_indices_to_zyx(flat, mask.shape)


def _interior_void_air_mask_np(
    air_mask: np.ndarray,
    roi_window: np.ndarray | None = None,
    *,
    roi_bbox: np.ndarray | None = None,
) -> np.ndarray:
    air_mask = np.asarray(air_mask, dtype=bool)
    result = np.zeros_like(air_mask, dtype=bool)

    if roi_bbox is None:
        if roi_window is None:
            roi_bbox = _default_roi_bbox(air_mask.shape)
        else:
            roi_window = np.asarray(roi_window, dtype=bool)
            if roi_window.shape != air_mask.shape or not np.any(roi_window):
                return result
            z_ids = np.flatnonzero(np.any(roi_window, axis=(1, 2)))
            y_ids = np.flatnonzero(np.any(roi_window, axis=(0, 2)))
            x_ids = np.flatnonzero(np.any(roi_window, axis=(0, 1)))
            if z_ids.size == 0 or y_ids.size == 0 or x_ids.size == 0:
                return result
            roi_bbox = np.asarray(
                [
                    [z_ids[0], z_ids[-1] + 1],
                    [y_ids[0], y_ids[-1] + 1],
                    [x_ids[0], x_ids[-1] + 1],
                ],
                dtype=np.int32,
            )
    else:
        roi_bbox = np.asarray(roi_bbox, dtype=np.int32)

    lower = np.maximum(roi_bbox[:, 0], 0).astype(np.int32, copy=False)
    upper = np.minimum(roi_bbox[:, 1], np.asarray(air_mask.shape, dtype=np.int32)).astype(np.int32, copy=False)
    if np.any(upper <= lower):
        return result

    roi_slices = (
        slice(int(lower[0]), int(upper[0])),
        slice(int(lower[1]), int(upper[1])),
        slice(int(lower[2]), int(upper[2])),
    )
    air_roi = air_mask[roi_slices]
    if roi_window is None:
        roi_roi = np.ones_like(air_roi, dtype=bool)
    else:
        roi_roi = np.asarray(roi_window[roi_slices], dtype=bool)
    air_roi = np.logical_and(air_roi, roi_roi)
    if not np.any(air_roi):
        return result

    structure = ndimage.generate_binary_structure(3, 1)
    labels, _ = ndimage.label(air_roi, structure=structure)
    if not np.any(labels):
        return result

    border = np.zeros_like(air_roi, dtype=bool)
    border[0, :, :] |= roi_roi[0, :, :]
    border[-1, :, :] |= roi_roi[-1, :, :]
    border[:, 0, :] |= roi_roi[:, 0, :]
    border[:, -1, :] |= roi_roi[:, -1, :]
    border[:, :, 0] |= roi_roi[:, :, 0]
    border[:, :, -1] |= roi_roi[:, :, -1]

    exterior_labels = np.unique(labels[np.logical_and(border, labels > 0)])
    if exterior_labels.size == 0:
        result[roi_slices] = labels > 0
        return result
    exterior_lookup = np.zeros(int(labels.max()) + 1, dtype=bool)
    exterior_lookup[exterior_labels.astype(np.int64)] = True
    result[roi_slices] = np.logical_and(labels > 0, ~exterior_lookup[labels])
    return result


def _prepare_field_sample_pools(
    analysis,
    volume_shape_dhw,
    boundary_margin_voxels,
    device=None,
    spacing_zyx=(1.0, 1.0, 1.0),
):
    if isinstance(analysis.get("material_mask"), torch.Tensor):
        tensor_device = analysis["material_mask"].device if device is None else torch.device(device)
        support_mask = analysis["material_mask"].to(device=tensor_device, dtype=torch.bool)
        roi_bbox = analysis["roi_bbox"].to(device=tensor_device, dtype=torch.long)
        padding = max(4, int(boundary_margin_voxels) * 2)
        lower = torch.clamp(roi_bbox[:, 0] - padding, min=0)
        upper = torch.minimum(roi_bbox[:, 1] + padding, torch.as_tensor(volume_shape_dhw, dtype=torch.long, device=tensor_device))
        roi_window = torch.zeros_like(support_mask)
        roi_window[
            int(lower[0].item()) : int(upper[0].item()),
            int(lower[1].item()) : int(upper[1].item()),
            int(lower[2].item()) : int(upper[2].item()),
        ] = True
        air_mask = torch.logical_and(roi_window, torch.logical_not(support_mask))
        if not torch.any(air_mask):
            air_mask = torch.logical_not(support_mask)

        air_cpu = air_mask.detach().cpu().numpy().astype(bool, copy=False)
        roi_bbox_np = torch.stack((lower, upper), dim=1).detach().cpu().numpy().astype(np.int32, copy=False)
        interior_void_air = _interior_void_air_mask_np(air_cpu, roi_bbox=roi_bbox_np)
        exterior_air_np = np.logical_and(air_cpu, np.logical_not(interior_void_air))
        if not np.any(exterior_air_np):
            exterior_air_np = air_cpu

        near_support = torch.zeros_like(support_mask)
        near_support[1:, :, :] |= support_mask[:-1, :, :]
        near_support[:-1, :, :] |= support_mask[1:, :, :]
        near_support[:, 1:, :] |= support_mask[:, :-1, :]
        near_support[:, :-1, :] |= support_mask[:, 1:, :]
        near_support[:, :, 1:] |= support_mask[:, :, :-1]
        near_support[:, :, :-1] |= support_mask[:, :, 1:]
        air_shell = torch.logical_and(air_mask, near_support)
        if not torch.any(air_shell):
            air_shell = air_mask
        cavity_air_shell = np.logical_and(
            interior_void_air,
            air_shell.detach().cpu().numpy().astype(bool, copy=False),
        )
        K_CAVITY_MATERIAL_BAND = 3
        structure_np = ndimage.generate_binary_structure(3, 1)
        support_np = support_mask.detach().cpu().numpy().astype(bool, copy=False)
        cavity_material_shell = np.logical_and(
            support_np,
            ndimage.binary_dilation(interior_void_air, structure=structure_np, iterations=K_CAVITY_MATERIAL_BAND),
        )
        boundary_material_subvoxel = _boundary_material_subvoxel_points(
            analysis,
            support_mask,
            spacing_zyx=spacing_zyx,
            device=tensor_device,
        )

        # K-voxel band of air near material. This is the CTGS-v2 negative
        # semantic pool: one air concept covering cavity walls and exterior skin
        # while excluding far-away easy air.
        K_EXTERIOR_BAND = 3
        dilated = support_mask.clone()
        for _ in range(K_EXTERIOR_BAND):
            shifted = dilated.clone()
            shifted[1:, :, :] |= dilated[:-1, :, :]
            shifted[:-1, :, :] |= dilated[1:, :, :]
            shifted[:, 1:, :] |= dilated[:, :-1, :]
            shifted[:, :-1, :] |= dilated[:, 1:, :]
            shifted[:, :, 1:] |= dilated[:, :, :-1]
            shifted[:, :, :-1] |= dilated[:, :, 1:]
            dilated = shifted
        dilated_np = dilated.detach().cpu().numpy().astype(bool, copy=False)
        exterior_air_near_band_np = np.logical_and(exterior_air_np, dilated_np)
        if not np.any(exterior_air_near_band_np):
            exterior_air_near_band_np = exterior_air_np
        near_material_air_np = np.logical_and(air_cpu, dilated_np)
        if not np.any(near_material_air_np):
            near_material_air_np = air_cpu

        return {
            "roi": _torch_mask_to_candidate_indices(roi_window),
            "support": _torch_mask_to_candidate_indices(support_mask),
            "air_shell": _torch_mask_to_candidate_indices(air_shell),
            "cavity_air_shell": torch.as_tensor(
                _component_balanced_mask_to_candidate_indices(cavity_air_shell),
                dtype=torch.int32,
                device=tensor_device,
            ),
            "cavity_material_shell": torch.as_tensor(
                _component_balanced_mask_to_candidate_indices(cavity_material_shell),
                dtype=torch.int32,
                device=tensor_device,
            ),
            "boundary_material_subvoxel_points": boundary_material_subvoxel,
            "air_shell_band": _torch_mask_to_candidate_indices(air_shell),
            "air_shell_band_ratio": 1.0 if torch.any(air_shell) else 0.0,
            "air": _torch_mask_to_candidate_indices(air_mask),
            "void_air": torch.as_tensor(
                _component_balanced_mask_to_candidate_indices(interior_void_air),
                dtype=torch.int32,
                device=tensor_device,
            ),
            "exterior_air": torch.as_tensor(
                _mask_to_candidate_indices(exterior_air_np),
                dtype=torch.int32,
                device=tensor_device,
            ),
            "exterior_air_near_band": torch.as_tensor(
                _mask_to_candidate_indices(exterior_air_near_band_np),
                dtype=torch.int32,
                device=tensor_device,
            ),
            "near_material_air": torch.as_tensor(
                _component_balanced_mask_to_candidate_indices(near_material_air_np),
                dtype=torch.int32,
                device=tensor_device,
            ),
        }

    support_mask = np.asarray(analysis["material_mask"], dtype=bool)
    roi_bbox = np.asarray(analysis["roi_bbox"] if "roi_bbox" in analysis else _default_roi_bbox(volume_shape_dhw), dtype=np.int32)
    padding = max(4, int(boundary_margin_voxels) * 2)
    lower = np.maximum(roi_bbox[:, 0] - padding, 0)
    upper = np.minimum(roi_bbox[:, 1] + padding, np.asarray(volume_shape_dhw, dtype=np.int32))
    roi_window = np.zeros_like(support_mask, dtype=bool)
    roi_window[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]] = True
    air_mask = np.logical_and(roi_window, np.logical_not(support_mask))
    if not np.any(air_mask):
        air_mask = np.logical_not(support_mask)
    interior_void_air = _interior_void_air_mask_np(
        air_mask,
        roi_bbox=np.stack((lower, upper), axis=1).astype(np.int32, copy=False),
    )
    exterior_air = np.logical_and(air_mask, np.logical_not(interior_void_air))
    if not np.any(exterior_air):
        exterior_air = air_mask
    structure = ndimage.generate_binary_structure(3, 1)
    near_support = ndimage.binary_dilation(support_mask, structure=structure, iterations=1)
    air_shell = np.logical_and(air_mask, near_support)
    if not np.any(air_shell):
        air_shell = air_mask
    cavity_air_shell = np.logical_and(interior_void_air, air_shell)
    K_CAVITY_MATERIAL_BAND = 3
    cavity_material_shell = np.logical_and(
        support_mask,
        ndimage.binary_dilation(interior_void_air, structure=structure, iterations=K_CAVITY_MATERIAL_BAND),
    )
    boundary_material_subvoxel = _boundary_material_subvoxel_points(
        analysis,
        support_mask,
        spacing_zyx=spacing_zyx,
    )
    K_EXTERIOR_BAND = 3
    dilated_support = ndimage.binary_dilation(support_mask, structure=structure, iterations=K_EXTERIOR_BAND)
    exterior_air_near_band = np.logical_and(exterior_air, dilated_support)
    if not np.any(exterior_air_near_band):
        exterior_air_near_band = exterior_air
    near_material_air = np.logical_and(air_mask, dilated_support)
    if not np.any(near_material_air):
        near_material_air = air_mask
    pools = {
        "roi": _mask_to_candidate_indices(roi_window),
        "support": _mask_to_candidate_indices(support_mask),
        "air_shell": _mask_to_candidate_indices(air_shell),
        "cavity_air_shell": _component_balanced_mask_to_candidate_indices(cavity_air_shell),
        "cavity_material_shell": _component_balanced_mask_to_candidate_indices(cavity_material_shell),
        "boundary_material_subvoxel_points": boundary_material_subvoxel,
        "air_shell_band": _mask_to_candidate_indices(air_shell),
        "air_shell_band_ratio": 1.0 if np.any(air_shell) else 0.0,
        "air": _mask_to_candidate_indices(air_mask),
        "void_air": _component_balanced_mask_to_candidate_indices(interior_void_air),
        "exterior_air": _mask_to_candidate_indices(exterior_air),
        "exterior_air_near_band": _mask_to_candidate_indices(exterior_air_near_band),
        "near_material_air": _component_balanced_mask_to_candidate_indices(near_material_air),
    }
    if device is None:
        return pools
    return {
        key: (
            torch.as_tensor(value, dtype=torch.int32, device=device)
            if isinstance(value, np.ndarray)
            else float(value)
        )
        for key, value in pools.items()
    }


def _resolve_field_sample_counts(args, total_gaussians: int):
    total_gaussians = max(1, int(total_gaussians))
    auto_scaled_count = max(1, total_gaussians // 8)
    support_count = auto_scaled_count if getattr(args, "ct_support_sample_count", None) is None else int(args.ct_support_sample_count)
    air_count = auto_scaled_count if getattr(args, "ct_air_sample_count", None) is None else int(args.ct_air_sample_count)

    if getattr(args, "_ct_support_sample_count_auto", False):
        support_count = max(support_count, auto_scaled_count)
    if getattr(args, "_ct_air_sample_count_auto", False):
        air_count = max(air_count, auto_scaled_count)

    return support_count, air_count


def _resolve_air_sampling_candidates(field_pools, near_boundary_ratio_threshold: float = 0.7):
    ratio = float(field_pools.get("air_shell_band_ratio", 1.0) or 0.0)
    cavity_air_shell = field_pools.get("cavity_air_shell")
    air_shell = field_pools.get("air_shell")
    if _candidate_count(cavity_air_shell) > 0 and _candidate_count(air_shell) > 0:
        focused = _concat_candidate_indices(cavity_air_shell, air_shell, max_count=CT_MAX_FIELD_POOL_INDICES)
        return focused, ratio, True
    air_shell_band = field_pools.get("air_shell_band")
    if ratio < float(near_boundary_ratio_threshold) and air_shell_band is not None:
        if isinstance(air_shell_band, torch.Tensor):
            if air_shell_band.shape[0] > 0:
                return air_shell_band, ratio, True
        elif len(air_shell_band) > 0:
            return air_shell_band, ratio, True
    return air_shell, ratio, False


def _sample_occupancy_points(candidate_indices, sample_count, spacing_zyx, device):
    if candidate_indices.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    count = int(sample_count)
    if isinstance(candidate_indices, torch.Tensor):
        candidate_indices = candidate_indices.to(device=device)
        candidate_count = int(candidate_indices.shape[0])
        replace = count > candidate_count
        large_candidate_pool = candidate_count > max(count * 8, 65536)
        if replace or large_candidate_pool:
            selected = torch.randint(0, candidate_count, (count,), device=device)
        else:
            selected = torch.randperm(candidate_count, device=device)[:count]
        voxel_indices = candidate_indices.index_select(0, selected).to(dtype=torch.float32)
        points_xyz = torch.stack(
            (
                (voxel_indices[:, 2] + 0.5) * float(spacing_zyx[2]),
                (voxel_indices[:, 1] + 0.5) * float(spacing_zyx[1]),
                (voxel_indices[:, 0] + 0.5) * float(spacing_zyx[0]),
            ),
            dim=1,
        )
        return points_xyz

    if count <= candidate_indices.shape[0]:
        selected = np.asarray(random.sample(range(candidate_indices.shape[0]), count), dtype=np.int64)
    else:
        selected = np.random.choice(candidate_indices.shape[0], size=count, replace=True)
    voxel_indices = candidate_indices[selected]
    points_xyz = np.stack(
        (
            (voxel_indices[:, 2].astype(np.float32) + 0.5) * float(spacing_zyx[2]),
            (voxel_indices[:, 1].astype(np.float32) + 0.5) * float(spacing_zyx[1]),
            (voxel_indices[:, 0].astype(np.float32) + 0.5) * float(spacing_zyx[0]),
        ),
        axis=1,
    )
    return torch.as_tensor(points_xyz, dtype=torch.float32, device=device)


def _candidate_count(candidate_indices) -> int:
    return int(candidate_indices.shape[0]) if candidate_indices is not None else 0


def _first_non_empty_candidate(*candidate_sets):
    for candidate_indices in candidate_sets:
        if _candidate_count(candidate_indices) > 0:
            return candidate_indices
    return None


def _ct_empty_points(device="cuda") -> torch.Tensor:
    return torch.empty((0, 3), dtype=torch.float32, device=device)


def _jitter_and_clamp_volume_points(points_xyz: torch.Tensor, spacing_zyx, volume_shape_dhw, jitter_distance: float) -> torch.Tensor:
    if points_xyz.numel() == 0:
        return points_xyz

    lower_xyz, upper_xyz = voxel_center_world_bounds_torch(volume_shape_dhw, spacing_zyx, points_xyz.device, points_xyz.dtype)

    jitter_width = float(jitter_distance)
    if jitter_width > 0.0:
        jitter = (torch.rand_like(points_xyz) - 0.5) * (2.0 * jitter_width)
        points_xyz = points_xyz + jitter
    return torch.minimum(torch.maximum(points_xyz, lower_xyz.unsqueeze(0)), upper_xyz.unsqueeze(0))


def _sample_ct_volume_points(
    field_pools: dict,
    sample_count: int,
    spacing_zyx,
    volume_shape_dhw,
    jitter_voxels: float,
    device="cuda",
    preferred_air_candidates=None,
    support_sample_ratio: float = 0.5,
) -> torch.Tensor:
    sample_count = max(1, int(sample_count))
    support_candidates = field_pools.get("support")
    shell_candidates = preferred_air_candidates if _candidate_count(preferred_air_candidates) > 0 else field_pools.get("air_shell")
    air_candidates = field_pools.get("air")
    fallback = _first_non_empty_candidate(support_candidates, shell_candidates, air_candidates)
    if fallback is None:
        return torch.empty((0, 3), dtype=torch.float32, device=device)

    support_candidates = support_candidates if _candidate_count(support_candidates) > 0 else fallback
    shell_candidates = shell_candidates if _candidate_count(shell_candidates) > 0 else fallback
    air_candidates = air_candidates if _candidate_count(air_candidates) > 0 else fallback

    support_ratio = min(max(float(support_sample_ratio), 0.0), 1.0)
    support_count = int(round(float(sample_count) * support_ratio))
    support_count = max(0, min(sample_count, support_count))
    remaining = sample_count - support_count
    shell_count = remaining // 2
    air_count = sample_count - support_count - shell_count
    sampled_parts = [
        _sample_occupancy_points(support_candidates, support_count, spacing_zyx, device=device),
        _sample_occupancy_points(shell_candidates, shell_count, spacing_zyx, device=device),
        _sample_occupancy_points(air_candidates, air_count, spacing_zyx, device=device),
    ]
    points_xyz = torch.cat([part for part in sampled_parts if part.numel() > 0], dim=0)
    return _jitter_and_clamp_volume_points(points_xyz, spacing_zyx, volume_shape_dhw, float(jitter_voxels))


def _sample_signed_distance(signed_distance_field: dict, points_xyz: torch.Tensor) -> torch.Tensor:
    if points_xyz.numel() == 0:
        return torch.empty((0,), dtype=points_xyz.dtype, device=points_xyz.device)
    return sample_volume_field(
        signed_distance_field["signed_distance"],
        points_xyz,
        signed_distance_field["spacing_zyx"],
    ).reshape(-1).to(dtype=points_xyz.dtype)


def _split_role_sample_counts(sample_count: int, boundary_ratio: float) -> tuple[int, int]:
    sample_count = max(1, int(sample_count))
    boundary_ratio = min(max(float(boundary_ratio), 0.0), 1.0)
    boundary_count = int(round(float(sample_count) * boundary_ratio))
    boundary_count = max(0, min(sample_count, boundary_count))
    bulk_count = sample_count - boundary_count
    if bulk_count == 0 and sample_count > 1:
        bulk_count = 1
        boundary_count = sample_count - bulk_count
    return bulk_count, boundary_count


def _filter_candidate_indices_by_sdf(candidate_indices, signed_distance_field: dict, boundary_band_distance: float, keep_boundary: bool):
    if candidate_indices is None or _candidate_count(candidate_indices) == 0:
        return candidate_indices

    signed_distance_native = signed_distance_field.get("signed_distance_native")
    if signed_distance_native is None:
        signed_distance_native = signed_distance_field["signed_distance"].reshape(
            *tuple(int(value) for value in signed_distance_field["signed_distance"].shape[-3:])
        )

    if isinstance(candidate_indices, torch.Tensor):
        indices = candidate_indices.to(dtype=torch.long)
        sdf_volume = torch.as_tensor(signed_distance_native, device=indices.device)
        sdf_values = sdf_volume[indices[:, 0], indices[:, 1], indices[:, 2]]
        mask = torch.abs(sdf_values) <= float(boundary_band_distance)
        if not keep_boundary:
            mask = torch.logical_not(mask)
        return candidate_indices[mask]

    indices = np.asarray(candidate_indices, dtype=np.int64)
    sdf_volume = np.asarray(signed_distance_native)
    sdf_values = sdf_volume[indices[:, 0], indices[:, 1], indices[:, 2]]
    mask = np.abs(sdf_values) <= float(boundary_band_distance)
    if not keep_boundary:
        mask = np.logical_not(mask)
    return candidate_indices[mask]


def precompute_sdf_filtered_field_pools(
    field_pools: dict,
    signed_distance_field: dict,
    boundary_band_distance: float,
    preferred_air_candidates=None,
) -> dict:
    """Cache SDF boundary/far candidate pools once instead of filtering every iteration."""
    enriched = dict(field_pools)
    for key in ("roi", "support", "air_shell", "cavity_air_shell", "cavity_material_shell", "air", "void_air", "exterior_air", "near_material_air"):
        if key not in field_pools:
            continue
        enriched[f"{key}_boundary"] = _filter_candidate_indices_by_sdf(
            field_pools.get(key),
            signed_distance_field,
            boundary_band_distance,
            keep_boundary=True,
        )
        enriched[f"{key}_far"] = _filter_candidate_indices_by_sdf(
            field_pools.get(key),
            signed_distance_field,
            boundary_band_distance,
            keep_boundary=False,
        )
    if "roi" in field_pools:
        enriched["boundary_pool"] = enriched.get("roi_boundary")
    if "support" in field_pools:
        enriched["material_deep_pool"] = enriched.get("support_far")
    if "exterior_air" in field_pools:
        enriched["exterior_air_pool"] = enriched.get("exterior_air_far")
    elif "air" in field_pools:
        enriched["exterior_air_pool"] = enriched.get("air_far")
    enriched["void_air_pool"] = field_pools.get("void_air")
    if preferred_air_candidates is not None:
        enriched["_preferred_air_boundary"] = _filter_candidate_indices_by_sdf(
            preferred_air_candidates,
            signed_distance_field,
            boundary_band_distance,
            keep_boundary=True,
        )
        enriched["_preferred_air_far"] = _filter_candidate_indices_by_sdf(
            preferred_air_candidates,
            signed_distance_field,
            boundary_band_distance,
            keep_boundary=False,
        )
    return enriched


def _cached_or_filter_candidates(
    field_pools: dict,
    key: str,
    signed_distance_field: dict,
    boundary_band_distance: float,
    keep_boundary: bool,
):
    suffix = "boundary" if keep_boundary else "far"
    cached = field_pools.get(f"{key}_{suffix}")
    if cached is not None:
        return cached
    return _filter_candidate_indices_by_sdf(
        field_pools.get(key),
        signed_distance_field,
        boundary_band_distance,
        keep_boundary,
    )


def _sample_points_from_boundary_role_pools(
    role_pools: dict,
    sample_count: int,
    spacing_zyx,
    volume_shape_dhw,
    jitter_voxels: float,
    signed_distance_field: dict,
    boundary_band_distance: float,
    keep_boundary: bool,
    device="cuda",
    preferred_air_candidates=None,
    support_sample_ratio: float = 0.5,
) -> torch.Tensor:
    sample_count = int(sample_count)
    if sample_count <= 0:
        return _ct_empty_points(device=device)

    selected = []
    remaining = sample_count
    oversample = 8 if keep_boundary else 3

    for _ in range(8):
        if remaining <= 0:
            break
        request_count = max(sample_count, remaining * oversample)
        candidates = _sample_ct_volume_points(
            role_pools,
            request_count,
            spacing_zyx,
            volume_shape_dhw,
            jitter_voxels,
            device=device,
            preferred_air_candidates=preferred_air_candidates,
            support_sample_ratio=support_sample_ratio,
        )
        if candidates.numel() == 0:
            break
        signed_distance = _sample_signed_distance(signed_distance_field, candidates)
        distance_mask = torch.abs(signed_distance) <= float(boundary_band_distance)
        if not keep_boundary:
            distance_mask = torch.logical_not(distance_mask)
        accepted = candidates[distance_mask]
        if accepted.numel() == 0:
            continue
        accepted = accepted[:remaining]
        selected.append(accepted)
        remaining -= int(accepted.shape[0])

    if remaining > 0:
        fallback = _sample_ct_volume_points(
            role_pools,
            remaining,
            spacing_zyx,
            volume_shape_dhw,
            jitter_voxels,
            device=device,
            preferred_air_candidates=preferred_air_candidates,
            support_sample_ratio=support_sample_ratio,
        )
        if fallback.numel() > 0:
            selected.append(fallback[:remaining])

    if not selected:
        return _ct_empty_points(device=device)
    return torch.cat(selected, dim=0)[:sample_count]


def _sample_points_by_boundary_mask(
    field_pools: dict,
    sample_count: int,
    spacing_zyx,
    volume_shape_dhw,
    jitter_voxels: float,
    signed_distance_field: dict,
    boundary_band_distance: float,
    keep_boundary: bool,
    device="cuda",
    preferred_air_candidates=None,
    support_sample_ratio: float = 0.5,
) -> torch.Tensor:
    sample_count = int(sample_count)
    if sample_count <= 0:
        return _ct_empty_points(device=device)

    role_pools = {
        "support": _cached_or_filter_candidates(field_pools, "support", signed_distance_field, boundary_band_distance, keep_boundary),
        "air_shell": _cached_or_filter_candidates(field_pools, "air_shell", signed_distance_field, boundary_band_distance, keep_boundary),
        "air": _cached_or_filter_candidates(field_pools, "air", signed_distance_field, boundary_band_distance, keep_boundary),
    }
    role_preferred_air = field_pools.get("_preferred_air_boundary" if keep_boundary else "_preferred_air_far")
    if role_preferred_air is None:
        role_preferred_air = _filter_candidate_indices_by_sdf(
            preferred_air_candidates,
            signed_distance_field,
            boundary_band_distance,
            keep_boundary,
        )
    if _first_non_empty_candidate(role_pools.get("support"), role_pools.get("air_shell"), role_pools.get("air")) is None:
        role_pools = field_pools
        role_preferred_air = preferred_air_candidates

    return _sample_points_from_boundary_role_pools(
        role_pools,
        sample_count,
        spacing_zyx,
        volume_shape_dhw,
        jitter_voxels,
        signed_distance_field,
        boundary_band_distance,
        keep_boundary,
        device=device,
        preferred_air_candidates=role_preferred_air,
        support_sample_ratio=support_sample_ratio,
    )


def sample_bulk_volume_points_excluding_boundary(
    field_pools: dict,
    sample_count: int,
    spacing_zyx,
    volume_shape_dhw,
    jitter_voxels: float,
    signed_distance_field: dict,
    boundary_band_distance: float,
    device="cuda",
    preferred_air_candidates=None,
    support_sample_ratio: float = 0.5,
) -> torch.Tensor:
    return _sample_points_by_boundary_mask(
        field_pools,
        sample_count,
        spacing_zyx,
        volume_shape_dhw,
        jitter_voxels,
        signed_distance_field,
        boundary_band_distance,
        keep_boundary=False,
        device=device,
        preferred_air_candidates=preferred_air_candidates,
        support_sample_ratio=support_sample_ratio,
    )


def sample_surface_boundary_points(
    field_pools: dict,
    sample_count: int,
    spacing_zyx,
    volume_shape_dhw,
    jitter_voxels: float,
    signed_distance_field: dict,
    boundary_band_distance: float,
    device="cuda",
    preferred_air_candidates=None,
) -> torch.Tensor:
    return _sample_points_by_boundary_mask(
        field_pools,
        sample_count,
        spacing_zyx,
        volume_shape_dhw,
        jitter_voxels,
        signed_distance_field,
        boundary_band_distance,
        keep_boundary=True,
        device=device,
        preferred_air_candidates=preferred_air_candidates,
    )
