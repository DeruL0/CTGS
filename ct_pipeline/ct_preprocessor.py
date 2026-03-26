from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.measure import marching_cubes


@dataclass
class CTSegmentationResult:
    material_mask: np.ndarray
    void_mask: np.ndarray
    foreground_mask: np.ndarray
    material_label_volume: np.ndarray
    roi_bbox: np.ndarray


class CTPreprocessor:
    def segment_material_void(self, volume, method: str = "multi_otsu", max_material_classes: int = 3) -> dict:
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError("volume must have shape (D, H, W).")

        finite_values = volume[np.isfinite(volume)]
        if finite_values.size == 0:
            raise ValueError("volume does not contain finite values.")

        air_threshold = float(threshold_otsu(finite_values))
        material_candidate = volume > air_threshold
        material_candidate = self._largest_connected_component(material_candidate)
        if not np.any(material_candidate):
            return {
                "material_mask": np.zeros_like(material_candidate, dtype=bool),
                "void_mask": np.zeros_like(material_candidate, dtype=bool),
                "foreground_mask": np.zeros_like(material_candidate, dtype=bool),
                "material_label_volume": np.zeros_like(volume, dtype=np.int32),
                "roi_bbox": self.compute_roi_bbox(material_candidate),
                "material_class_count": 0,
            }

        material_label_volume, material_class_count = self._split_material_classes(
            volume,
            material_candidate,
            method=method,
            max_material_classes=max_material_classes,
        )
        material_mask = material_label_volume > 0
        roi_bbox = self.compute_roi_bbox(material_mask)
        foreground_mask = self._bbox_mask(material_mask.shape, roi_bbox)
        void_mask = np.logical_and(foreground_mask, np.logical_not(material_mask))
        return {
            "material_mask": material_mask.astype(bool),
            "void_mask": void_mask.astype(bool),
            "foreground_mask": foreground_mask.astype(bool),
            "material_label_volume": material_label_volume.astype(np.int32),
            "roi_bbox": roi_bbox.astype(np.int32),
            "material_class_count": int(material_class_count),
        }

    def segment_foreground(self, volume, method: str = "otsu", max_material_classes: int = 3) -> np.ndarray:
        del method
        segmentation = self.segment_material_void(volume, method="multi_otsu", max_material_classes=max_material_classes)
        return segmentation["foreground_mask"]

    def dilate_boundary(self, mask, width_voxels: int = 3) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if width_voxels <= 0:
            return np.zeros_like(mask, dtype=bool)
        dilated = ndimage.binary_dilation(mask, iterations=int(width_voxels))
        return np.logical_and(dilated, np.logical_not(mask))

    def compute_roi_bbox(self, mask) -> np.ndarray:
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

    def extract_surface_points(self, mask, spacing) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return np.empty((0, 3), dtype=np.float32)

        verts_zyx, _, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=tuple(float(v) for v in spacing))
        verts_xyz = np.stack((verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]), axis=1)
        return verts_xyz.astype(np.float32)

    def extract_material_surface_points(self, material_label_volume, spacing):
        material_label_volume = np.asarray(material_label_volume, dtype=np.int32)
        if material_label_volume.ndim != 3:
            raise ValueError("material_label_volume must have shape (D, H, W).")

        all_points = []
        all_material_ids = []
        for label in np.unique(material_label_volume):
            if int(label) <= 0:
                continue
            mask = material_label_volume == int(label)
            if np.count_nonzero(mask) < 4:
                continue
            points = self.extract_surface_points(mask, spacing)
            if points.shape[0] == 0:
                continue
            all_points.append(points)
            all_material_ids.append(np.full((points.shape[0], 1), int(label) - 1, dtype=np.int64))

        if not all_points:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 1), dtype=np.int64)
        return (
            np.concatenate(all_points, axis=0).astype(np.float32),
            np.concatenate(all_material_ids, axis=0).astype(np.int64),
        )

    def sample_interior_points(
        self,
        material_mask,
        volume,
        spacing,
        target_count,
        boundary_margin_voxels: int = 2,
        material_label_volume=None,
    ):
        material_mask = np.asarray(material_mask, dtype=bool)
        volume = np.asarray(volume, dtype=np.float32)
        if material_mask.shape != volume.shape:
            raise ValueError("material_mask and volume must have the same shape.")
        if material_label_volume is not None and np.asarray(material_label_volume).shape != volume.shape:
            raise ValueError("material_label_volume must have the same shape as volume.")
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
        if material_label_volume is None:
            material_ids = np.zeros((voxel_indices.shape[0], 1), dtype=np.int64)
        else:
            labels = np.asarray(material_label_volume, dtype=np.int32)[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
            material_ids = np.maximum(labels - 1, 0).reshape(-1, 1).astype(np.int64)
        return points_xyz.astype(np.float32), density_seed, material_ids

    def _largest_connected_component(self, mask) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        labeled, count = ndimage.label(mask)
        if count <= 1:
            return mask
        component_sizes = np.bincount(labeled.reshape(-1))
        component_sizes[0] = 0
        largest = int(np.argmax(component_sizes))
        return labeled == largest

    def _bbox_mask(self, shape, roi_bbox) -> np.ndarray:
        bbox_mask = np.zeros(shape, dtype=bool)
        bbox_mask[
            int(roi_bbox[0, 0]) : int(roi_bbox[0, 1]),
            int(roi_bbox[1, 0]) : int(roi_bbox[1, 1]),
            int(roi_bbox[2, 0]) : int(roi_bbox[2, 1]),
        ] = True
        return bbox_mask

    def _split_material_classes(self, volume, material_candidate, method: str, max_material_classes: int):
        del method
        material_values = volume[material_candidate]
        if material_values.size == 0:
            return np.zeros_like(volume, dtype=np.int32), 0
        if np.allclose(material_values, material_values[0]):
            material_label_volume = np.zeros_like(volume, dtype=np.int32)
            material_label_volume[material_candidate] = 1
            return material_label_volume, 1

        thresholds = None
        class_count = 1
        max_material_classes = max(1, int(max_material_classes))
        for candidate_classes in range(max_material_classes, 1, -1):
            try:
                candidate_thresholds = threshold_multiotsu(material_values, classes=candidate_classes)
            except ValueError:
                continue
            material_bins = np.digitize(material_values, bins=candidate_thresholds)
            counts = np.bincount(material_bins, minlength=candidate_classes)
            min_count = max(32, int(0.01 * material_values.size))
            if np.any(counts < min_count):
                continue
            thresholds = candidate_thresholds
            class_count = candidate_classes
            break

        if thresholds is None:
            material_label_volume = np.zeros_like(volume, dtype=np.int32)
            material_label_volume[material_candidate] = 1
            return material_label_volume, 1

        labels = np.digitize(material_values, bins=thresholds).astype(np.int32) + 1
        material_label_volume = np.zeros_like(volume, dtype=np.int32)
        material_label_volume[material_candidate] = labels
        return material_label_volume, class_count
