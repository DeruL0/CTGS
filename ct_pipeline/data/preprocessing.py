from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_multiotsu, threshold_otsu


def _boundary_mask_from_support(support_mask: np.ndarray) -> np.ndarray:
    support_mask = np.asarray(support_mask, dtype=bool)
    if support_mask.ndim != 3:
        raise ValueError("support_mask must have shape (D, H, W).")
    boundary = np.zeros_like(support_mask, dtype=bool)
    for axis in range(3):
        slicer_a = [slice(None)] * 3
        slicer_b = [slice(None)] * 3
        slicer_a[axis] = slice(1, None)
        slicer_b[axis] = slice(None, -1)
        current = support_mask[tuple(slicer_a)]
        previous = support_mask[tuple(slicer_b)]
        change = current != previous
        boundary[tuple(slicer_a)] |= change
        boundary[tuple(slicer_b)] |= change
    return boundary


def build_support_signed_distance(
    support_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    boundary_mode: str = "interface",
) -> np.ndarray:
    support_mask = np.asarray(support_mask, dtype=bool)
    if support_mask.ndim != 3:
        raise ValueError("support_mask must have shape (D, H, W).")
    if not np.any(support_mask) or np.all(support_mask):
        return np.zeros_like(support_mask, dtype=np.float32)

    spacing = tuple(float(value) for value in spacing_zyx)
    outside_distance = ndimage.distance_transform_edt(~support_mask, sampling=spacing).astype(np.float32)
    inside_distance = ndimage.distance_transform_edt(support_mask, sampling=spacing).astype(np.float32)

    # The binary mask represents samples at voxel centers. The physical
    # material/air interface lies halfway between adjacent opposite-class
    # samples, so boundary voxel centers should not be assigned D=0.
    half_voxel = 0.5 * float(min(spacing))
    signed_distance = np.where(
        support_mask,
        -(inside_distance - half_voxel),
        outside_distance - half_voxel,
    ).astype(np.float32)
    mode = str(boundary_mode).strip().lower()
    if mode in {"material_zero", "material-side-zero", "material_side_zero"}:
        material_boundary = _boundary_mask_from_support(support_mask) & support_mask
        signed_distance[material_boundary] = 0.0
    elif mode != "interface":
        raise ValueError("boundary_mode must be one of {'interface', 'material_zero'}.")
    return signed_distance


class CTPreprocessor:
    def segment_coarse_support(
        self,
        volume,
        threshold_mode: str = "otsu",
        min_component_voxels: int = 0,
        min_component_fraction: float = 1.0,
    ) -> dict:
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError("volume must have shape (D, H, W).")

        finite_values = volume[np.isfinite(volume)]
        if finite_values.size == 0:
            raise ValueError("volume does not contain finite values.")

        mode = str(threshold_mode).lower()
        if mode not in {"otsu", "multi_otsu"}:
            raise ValueError("threshold_mode must be one of {'otsu', 'multi_otsu'}.")

        if mode == "multi_otsu" and finite_values.size >= 8:
            try:
                thresholds = threshold_multiotsu(finite_values, classes=3)
                support_threshold = float(thresholds[0])
            except ValueError:
                support_threshold = float(threshold_otsu(finite_values))
        else:
            support_threshold = float(threshold_otsu(finite_values))

        support_mask = volume > support_threshold
        support_mask = self._filtered_connected_components(
            support_mask,
            min_component_voxels=min_component_voxels,
            min_component_fraction=min_component_fraction,
        )
        roi_bbox = self._compute_roi_bbox(support_mask)
        roi_mask = self._bbox_mask(support_mask.shape, roi_bbox)
        air_mask = np.logical_and(roi_mask, np.logical_not(support_mask))
        return {
            "support_mask": support_mask.astype(bool),
            "roi_bbox": roi_bbox.astype(np.int32),
            "roi_mask": roi_mask.astype(bool),
            "air_mask": air_mask.astype(bool),
            "support_threshold": support_threshold,
        }

    def extract_intensity_surface_points(
        self,
        volume,
        support_mask,
        spacing,
    ):
        volume = np.asarray(volume, dtype=np.float32)
        support_mask = np.asarray(support_mask, dtype=bool)
        if volume.shape != support_mask.shape:
            raise ValueError("volume and support_mask must have the same shape.")
        if not np.any(support_mask):
            return np.empty((0, 3), dtype=np.float32)

        # Boundary existence is geometric. CT gradients may be useful as confidence
        # signals later, but they must not remove surface ownership anchors.
        support_label_volume = support_mask.astype(np.int32)
        boundary_mask = self._material_boundary_mask(support_label_volume)
        if not np.any(boundary_mask):
            return np.empty((0, 3), dtype=np.float32)

        filtered_indices = np.argwhere(boundary_mask)
        points_xyz = np.stack(
            (
                (filtered_indices[:, 2].astype(np.float32) + 0.5) * float(spacing[2]),
                (filtered_indices[:, 1].astype(np.float32) + 0.5) * float(spacing[1]),
                (filtered_indices[:, 0].astype(np.float32) + 0.5) * float(spacing[0]),
            ),
            axis=1,
        )
        return points_xyz.astype(np.float32)

    def subsample_surface_points_by_voxel_grid(
        self,
        points_xyz,
        spacing,
        spacing_voxels: float = 3.0,
        return_indices: bool = False,
    ):
        points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        if points_xyz.shape[0] == 0:
            if return_indices:
                return points_xyz, np.empty((0,), dtype=np.int64)
            return points_xyz
        spacing_voxels = float(spacing_voxels)
        if spacing_voxels <= 1.0:
            if return_indices:
                return points_xyz.copy(), np.arange(points_xyz.shape[0], dtype=np.int64)
            return points_xyz.copy()

        spacing_z, spacing_y, spacing_x = [max(float(value), 1e-8) for value in spacing]
        coords_zyx = np.stack(
            (
                points_xyz[:, 2] / spacing_z,
                points_xyz[:, 1] / spacing_y,
                points_xyz[:, 0] / spacing_x,
            ),
            axis=1,
        )
        keys = np.floor(coords_zyx / spacing_voxels).astype(np.int64)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        cell_centers = (unique_keys[inverse].astype(np.float32) + 0.5) * spacing_voxels
        distance2 = np.sum((coords_zyx - cell_centers) ** 2, axis=1)
        order = np.lexsort((distance2, inverse))
        sorted_inverse = inverse[order]
        first = np.concatenate(([True], sorted_inverse[1:] != sorted_inverse[:-1]))
        selected = np.sort(order[first])
        selected_points = points_xyz[selected].astype(np.float32, copy=False)
        if return_indices:
            return selected_points, selected.astype(np.int64, copy=False)
        return selected_points

    def _compute_roi_bbox(self, mask) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 3:
            raise ValueError("mask must have shape (D, H, W).")
        if not np.any(mask):
            shape = np.asarray(mask.shape, dtype=np.int32)
            return np.stack((np.zeros(3, dtype=np.int32), shape), axis=1)

        coords = np.argwhere(mask)
        lower = coords.min(axis=0).astype(np.int32)
        upper = (coords.max(axis=0) + 1).astype(np.int32)
        return np.stack((lower, upper), axis=1)

    def sample_interior_points(
        self,
        material_mask,
        volume,
        spacing,
        target_count,
        boundary_margin_voxels: int = 1,
    ):
        material_mask = np.asarray(material_mask, dtype=bool)
        volume = np.asarray(volume, dtype=np.float32)
        if material_mask.shape != volume.shape:
            raise ValueError("material_mask and volume must have the same shape.")
        if target_count < 1 or not np.any(material_mask):
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.int64),
            )

        distance = ndimage.distance_transform_edt(material_mask)
        margin = max(int(boundary_margin_voxels), 0)
        candidate_mask = np.logical_and(material_mask, distance >= float(margin))
        if not np.any(candidate_mask):
            candidate_mask = material_mask

        candidate_indices = np.argwhere(candidate_mask)
        if candidate_indices.shape[0] == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.int64),
            )

        count = int(target_count)
        if candidate_indices.shape[0] >= count:
            selected = np.random.choice(candidate_indices.shape[0], size=count, replace=False)
        else:
            selected = np.random.choice(candidate_indices.shape[0], size=count, replace=True)
        voxel_indices = candidate_indices[selected]

        points_xyz = np.stack(
            (
                (voxel_indices[:, 2].astype(np.float32) + 0.5) * float(spacing[2]),
                (voxel_indices[:, 1].astype(np.float32) + 0.5) * float(spacing[1]),
                (voxel_indices[:, 0].astype(np.float32) + 0.5) * float(spacing[0]),
            ),
            axis=1,
        )
        density_seed = volume[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]].reshape(-1, 1).astype(np.float32)
        material_ids = np.zeros((voxel_indices.shape[0], 1), dtype=np.int64)
        return points_xyz.astype(np.float32), density_seed, material_ids

    def _filtered_connected_components(
        self,
        mask,
        min_component_voxels: int = 0,
        min_component_fraction: float = 1.0,
    ) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        labeled, count = ndimage.label(mask)
        if count <= 1:
            return mask
        component_sizes = np.bincount(labeled.reshape(-1))
        component_sizes[0] = 0
        largest_size = int(component_sizes.max())
        if largest_size <= 0:
            return np.zeros_like(mask, dtype=bool)
        min_voxels = max(int(min_component_voxels), 0)
        min_fraction = max(float(min_component_fraction), 0.0)
        keep_threshold = max(min_voxels, int(np.ceil(float(largest_size) * min_fraction)))
        keep_labels = np.flatnonzero(component_sizes >= keep_threshold)
        if keep_labels.size == 0:
            keep_labels = np.asarray([int(np.argmax(component_sizes))], dtype=np.int64)
        return np.isin(labeled, keep_labels)

    def _bbox_mask(self, shape, roi_bbox) -> np.ndarray:
        bbox_mask = np.zeros(shape, dtype=bool)
        bbox_mask[
            int(roi_bbox[0, 0]) : int(roi_bbox[0, 1]),
            int(roi_bbox[1, 0]) : int(roi_bbox[1, 1]),
            int(roi_bbox[2, 0]) : int(roi_bbox[2, 1]),
        ] = True
        return bbox_mask

    def _material_boundary_mask(self, material_label_volume: np.ndarray) -> np.ndarray:
        material_label_volume = np.asarray(material_label_volume, dtype=np.int32)
        material_mask = material_label_volume > 0
        if not np.any(material_mask):
            return np.zeros_like(material_mask, dtype=bool)

        boundary_mask = np.zeros_like(material_mask, dtype=bool)
        for axis in range(3):
            slicer_a = [slice(None)] * 3
            slicer_b = [slice(None)] * 3
            slicer_a[axis] = slice(1, None)
            slicer_b[axis] = slice(None, -1)
            current = material_label_volume[tuple(slicer_a)]
            previous = material_label_volume[tuple(slicer_b)]
            change = current != previous
            current_material = current > 0
            previous_material = previous > 0
            boundary_mask[tuple(slicer_a)] |= change & current_material
            boundary_mask[tuple(slicer_b)] |= change & previous_material
        return np.logical_and(boundary_mask, material_mask)
