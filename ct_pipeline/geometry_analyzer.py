from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


class GeometryAnalyzer:
    """Estimate surface normals, classify local surface geometry, and fit local planes."""

    def __init__(
        self,
        sigma: float = 1.0,
        planar_ratio_threshold: float = 0.15,
        planar_similarity_threshold: float = 0.05,
        edge_ratio_threshold: float = 0.5,
        edge_tail_threshold: float = 0.15,
        planar_residual_threshold: float = 0.01,
    ) -> None:
        self.sigma = float(sigma)
        self.planar_ratio_threshold = float(planar_ratio_threshold)
        self.planar_similarity_threshold = float(planar_similarity_threshold)
        self.edge_ratio_threshold = float(edge_ratio_threshold)
        self.edge_tail_threshold = float(edge_tail_threshold)
        self.planar_residual_threshold = float(planar_residual_threshold)

    def compute_structure_tensor(self, volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        gradients = self._compute_gradients(volume, sigma)
        gx, gy, gz = gradients["gx"], gradients["gy"], gradients["gz"]

        tensor = np.empty(volume.shape + (3, 3), dtype=np.float32)
        tensor[..., 0, 0] = ndimage.gaussian_filter(gx * gx, sigma=sigma).astype(np.float32)
        tensor[..., 0, 1] = ndimage.gaussian_filter(gx * gy, sigma=sigma).astype(np.float32)
        tensor[..., 0, 2] = ndimage.gaussian_filter(gx * gz, sigma=sigma).astype(np.float32)
        tensor[..., 1, 0] = tensor[..., 0, 1]
        tensor[..., 1, 1] = ndimage.gaussian_filter(gy * gy, sigma=sigma).astype(np.float32)
        tensor[..., 1, 2] = ndimage.gaussian_filter(gy * gz, sigma=sigma).astype(np.float32)
        tensor[..., 2, 0] = tensor[..., 0, 2]
        tensor[..., 2, 1] = tensor[..., 1, 2]
        tensor[..., 2, 2] = ndimage.gaussian_filter(gz * gz, sigma=sigma).astype(np.float32)
        return tensor

    def estimate_surface_normals(self, points: np.ndarray, volume: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
        if points.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        gradients = self._compute_gradients(volume, self.sigma)
        coords_zyx = self._points_xyz_to_zyx(points, spacing)

        sampled_gx = self._interpolate_volume(gradients["gx"], coords_zyx)
        sampled_gy = self._interpolate_volume(gradients["gy"], coords_zyx)
        sampled_gz = self._interpolate_volume(gradients["gz"], coords_zyx)
        gradients_xyz = np.stack((sampled_gx, sampled_gy, sampled_gz), axis=1)

        normals = -gradients_xyz
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-8
        normals[valid] = normals[valid] / norms[valid]
        normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return normals.astype(np.float32)

    def classify_regions(
        self,
        points: np.ndarray,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be shaped as (N, 3).")
        if points.shape[0] == 0:
            empty = np.zeros((0,), dtype=bool)
            return empty, empty, empty

        tensor = self.compute_structure_tensor(volume, sigma=self.sigma)
        sampled_tensors = self._sample_structure_tensor(tensor, points, spacing)
        eigenvalues = np.linalg.eigvalsh(sampled_tensors)
        eigenvalues = eigenvalues[:, ::-1]
        lambda1 = eigenvalues[:, 0]
        lambda2 = eigenvalues[:, 1]
        lambda3 = eigenvalues[:, 2]
        eps = 1e-8

        ratio21 = lambda2 / (lambda1 + eps)
        ratio32 = lambda3 / (lambda2 + eps)
        planar_similarity = np.abs(lambda2 - lambda3) / (lambda1 + eps)
        local_residuals = self._local_surface_residuals(points)

        edge_mask = (ratio21 >= self.edge_ratio_threshold) & (ratio32 < self.edge_tail_threshold)
        # Sobel responses on discretized binary surfaces tend to underestimate lambda2/lambda1 at sharp edges.
        # A relaxed fallback keeps edge strips from collapsing into the curved bucket while leaving smooth
        # curved regions untouched because their lambda2/lambda1 stays low and lambda3/lambda2 stays high.
        relaxed_edge_mask = (ratio21 >= 0.18) & (ratio32 < 0.15) & (local_residuals > self.planar_residual_threshold)
        edge_mask = edge_mask | relaxed_edge_mask
        planar_mask = (
            (ratio21 < self.planar_ratio_threshold)
            & (planar_similarity < self.planar_similarity_threshold)
            & (local_residuals < self.planar_residual_threshold)
            & np.logical_not(edge_mask)
        )
        curved_mask = np.logical_not(planar_mask | edge_mask)
        return planar_mask, edge_mask, curved_mask

    def fit_local_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k_neighbors: int = 20,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be shaped as (N, 3).")
        if normals.shape != points.shape:
            raise ValueError("normals must match points shape.")

        num_points = points.shape[0]
        plane_normals = np.full((num_points, 3), np.nan, dtype=np.float32)
        tangent_u = np.full((num_points, 3), np.nan, dtype=np.float32)
        tangent_v = np.full((num_points, 3), np.nan, dtype=np.float32)
        offsets = np.full((num_points,), np.nan, dtype=np.float32)
        residuals = np.full((num_points,), np.inf, dtype=np.float32)

        if num_points < 3:
            return {
                "normal": plane_normals,
                "tangent_u": tangent_u,
                "tangent_v": tangent_v,
                "offset": offsets,
            }, residuals

        k = min(max(int(k_neighbors), 3), num_points)
        tree = cKDTree(points)
        _, neighbor_indices = tree.query(points, k=k)
        if neighbor_indices.ndim == 1:
            neighbor_indices = neighbor_indices[:, np.newaxis]

        for point_index in range(num_points):
            indices = np.unique(neighbor_indices[point_index])
            neighborhood = points[indices]
            if neighborhood.shape[0] < 3:
                continue

            centroid = neighborhood.mean(axis=0)
            centered = neighborhood - centroid
            covariance = np.dot(centered.T, centered) / max(neighborhood.shape[0] - 1, 1)
            if np.linalg.matrix_rank(covariance, tol=1e-8) < 2:
                continue

            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            smallest_vector = eigenvectors[:, 0]
            largest_vector = eigenvectors[:, 2]

            reference_normal = normals[point_index]
            if np.all(np.isfinite(reference_normal)) and np.dot(smallest_vector, reference_normal) < 0:
                smallest_vector = -smallest_vector

            tangent_vector = largest_vector - np.dot(largest_vector, smallest_vector) * smallest_vector
            tangent_norm = np.linalg.norm(tangent_vector)
            if tangent_norm <= 1e-8:
                continue
            tangent_vector = tangent_vector / tangent_norm
            binormal_vector = np.cross(smallest_vector, tangent_vector)
            binormal_norm = np.linalg.norm(binormal_vector)
            if binormal_norm <= 1e-8:
                continue
            binormal_vector = binormal_vector / binormal_norm

            plane_distance = centered.dot(smallest_vector)
            residual = float(np.sqrt(np.mean(np.square(plane_distance))))

            plane_normals[point_index] = smallest_vector.astype(np.float32)
            tangent_u[point_index] = tangent_vector.astype(np.float32)
            tangent_v[point_index] = binormal_vector.astype(np.float32)
            offsets[point_index] = float(-np.dot(smallest_vector, centroid))
            residuals[point_index] = residual

        return {
            "normal": plane_normals,
            "tangent_u": tangent_u,
            "tangent_v": tangent_v,
            "offset": offsets,
        }, residuals

    def _compute_gradients(self, volume: np.ndarray, sigma: float) -> Dict[str, np.ndarray]:
        smoothed = ndimage.gaussian_filter(volume.astype(np.float32), sigma=sigma)
        gz = ndimage.sobel(smoothed, axis=0, mode="nearest").astype(np.float32)
        gy = ndimage.sobel(smoothed, axis=1, mode="nearest").astype(np.float32)
        gx = ndimage.sobel(smoothed, axis=2, mode="nearest").astype(np.float32)
        return {"gx": gx, "gy": gy, "gz": gz}

    def _points_xyz_to_zyx(self, points: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
        sz, sy, sx = [float(value) for value in spacing]
        coords_zyx = np.column_stack(
            (
                points[:, 2] / max(sz, 1e-8),
                points[:, 1] / max(sy, 1e-8),
                points[:, 0] / max(sx, 1e-8),
            )
        )
        return coords_zyx.astype(np.float64)

    def _interpolate_volume(self, volume: np.ndarray, coords_zyx: np.ndarray) -> np.ndarray:
        grid = [np.arange(size, dtype=np.float64) for size in volume.shape]
        interpolator = RegularGridInterpolator(
            grid,
            volume,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return interpolator(coords_zyx)

    def _sample_structure_tensor(
        self,
        tensor: np.ndarray,
        points: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> np.ndarray:
        coords_zyx = self._points_xyz_to_zyx(points, spacing)
        sampled_components = []
        for row in range(3):
            for col in range(3):
                sampled_components.append(self._interpolate_volume(tensor[..., row, col], coords_zyx))
        stacked = np.stack(sampled_components, axis=1).reshape(points.shape[0], 3, 3)
        return stacked.astype(np.float32)

    def _local_surface_residuals(self, points: np.ndarray, k_neighbors: int = 20) -> np.ndarray:
        if points.shape[0] < 3:
            return np.full((points.shape[0],), np.inf, dtype=np.float32)

        k = min(max(int(k_neighbors), 3), points.shape[0])
        tree = cKDTree(points)
        _, neighbor_indices = tree.query(points, k=k)
        if neighbor_indices.ndim == 1:
            neighbor_indices = neighbor_indices[:, np.newaxis]

        residuals = np.full((points.shape[0],), np.inf, dtype=np.float32)
        for point_index in range(points.shape[0]):
            neighborhood = points[np.unique(neighbor_indices[point_index])]
            if neighborhood.shape[0] < 3:
                continue

            centered = neighborhood - neighborhood.mean(axis=0)
            covariance = np.dot(centered.T, centered) / max(neighborhood.shape[0] - 1, 1)
            eigenvalues = np.linalg.eigvalsh(covariance)
            if np.sum(eigenvalues) <= 1e-8:
                continue
            residuals[point_index] = float(eigenvalues[0] / (np.sum(eigenvalues) + 1e-8))

        return residuals
