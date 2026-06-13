from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator


class GeometryAnalyzer:
    """Estimate CT boundary normals, tangent frames, and gradient strength."""

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = float(sigma)

    def _compute_structure_tensor(self, volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
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

    def estimate_boundary_geometry(
        self,
        points: np.ndarray,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if points.size == 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty, empty, np.zeros((0, 1), dtype=np.float32)

        gradients = self._compute_gradients(volume, self.sigma)
        gradient_normals, gradient_strength = self._sample_boundary_gradients(points, spacing, gradients)
        sampled_tensors = self._sample_structure_tensor(self._compute_structure_tensor(volume, sigma=self.sigma), points, spacing)

        tangent_u = np.full_like(gradient_normals, np.nan, dtype=np.float32)
        tangent_v = np.full_like(gradient_normals, np.nan, dtype=np.float32)
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for index in range(points.shape[0]):
            normal = gradient_normals[index]
            if not np.all(np.isfinite(normal)):
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            eigenvalues, eigenvectors = np.linalg.eigh(sampled_tensors[index])
            order = np.argsort(eigenvalues)[::-1]
            tangent_hint = eigenvectors[:, order[1]].astype(np.float32)
            tangent_hint = tangent_hint - np.dot(tangent_hint, normal) * normal
            if np.linalg.norm(tangent_hint) <= 1e-6:
                tangent_hint = fallback.copy()
                if abs(np.dot(tangent_hint, normal)) > 0.9:
                    tangent_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                tangent_hint = tangent_hint - np.dot(tangent_hint, normal) * normal
            tangent_hint /= max(np.linalg.norm(tangent_hint), 1e-6)
            tangent_binormal = np.cross(normal, tangent_hint)
            tangent_binormal /= max(np.linalg.norm(tangent_binormal), 1e-6)
            tangent_hint = np.cross(tangent_binormal, normal)
            tangent_hint /= max(np.linalg.norm(tangent_hint), 1e-6)
            tangent_u[index] = tangent_hint
            tangent_v[index] = tangent_binormal

        return (
            gradient_normals.astype(np.float32),
            tangent_u.astype(np.float32),
            tangent_v.astype(np.float32),
            gradient_strength.reshape(-1, 1).astype(np.float32),
        )

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

    def _sample_boundary_gradients(self, points: np.ndarray, spacing: Tuple[float, float, float], gradients: Dict[str, np.ndarray]):
        coords_zyx = self._points_xyz_to_zyx(points, spacing)
        sampled_gx = self._interpolate_volume(gradients["gx"], coords_zyx)
        sampled_gy = self._interpolate_volume(gradients["gy"], coords_zyx)
        sampled_gz = self._interpolate_volume(gradients["gz"], coords_zyx)
        gradient_xyz = np.stack((-sampled_gx, -sampled_gy, -sampled_gz), axis=1).astype(np.float32)
        strength = np.linalg.norm(gradient_xyz, axis=1)
        normalizer = float(np.percentile(strength, 99.0)) if strength.size > 0 else 0.0
        if normalizer <= 1e-8:
            normalizer = float(np.max(strength)) if strength.size > 0 else 0.0
        if normalizer > 1e-8:
            normalized_strength = np.clip(strength / normalizer, 0.0, 1.0)
        else:
            normalized_strength = np.zeros_like(strength, dtype=np.float32)

        norms = np.linalg.norm(gradient_xyz, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-8
        gradient_xyz[valid] = gradient_xyz[valid] / norms[valid]
        gradient_xyz[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return gradient_xyz.astype(np.float32), normalized_strength.astype(np.float32)
