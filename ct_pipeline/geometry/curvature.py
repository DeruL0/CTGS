from __future__ import annotations

import numpy as np
from scipy import ndimage


def compute_curvature_proxy_np(
    volume: np.ndarray,
    spacing_zyx,
    sigma: float = 1.0,
    normalizer_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a normalized Hessian/gradient curvature proxy for a CT volume."""

    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("volume must have shape (D, H, W).")

    sigma = float(max(sigma, 0.0))
    smoothed = ndimage.gaussian_filter(volume, sigma=sigma, mode="nearest").astype(np.float32, copy=False) if sigma > 0.0 else volume

    del spacing_zyx
    grad_x = ndimage.sobel(smoothed, axis=2, mode="nearest").astype(np.float32)
    grad_y = ndimage.sobel(smoothed, axis=1, mode="nearest").astype(np.float32)
    grad_z = ndimage.sobel(smoothed, axis=0, mode="nearest").astype(np.float32)
    gradient_magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).astype(np.float32)

    hsq = np.zeros_like(grad_x)
    h = ndimage.sobel(grad_x, axis=2, mode="nearest").astype(np.float32)
    hsq += h * h
    h = ndimage.sobel(grad_x, axis=1, mode="nearest").astype(np.float32)
    hsq += 2.0 * (h * h)
    h = ndimage.sobel(grad_x, axis=0, mode="nearest").astype(np.float32)
    hsq += 2.0 * (h * h)
    del grad_x
    h = ndimage.sobel(grad_y, axis=1, mode="nearest").astype(np.float32)
    hsq += h * h
    h = ndimage.sobel(grad_y, axis=0, mode="nearest").astype(np.float32)
    hsq += 2.0 * (h * h)
    del grad_y
    h = ndimage.sobel(grad_z, axis=0, mode="nearest").astype(np.float32)
    hsq += h * h
    del grad_z, h
    hessian_frob = np.sqrt(hsq).astype(np.float32)
    del hsq
    proxy = hessian_frob / np.maximum(gradient_magnitude, 1e-6)
    proxy[~np.isfinite(proxy)] = 0.0

    if normalizer_mask is not None:
        mask = np.asarray(normalizer_mask, dtype=bool)
        if mask.shape != proxy.shape:
            raise ValueError("normalizer_mask must match volume shape.")
        values = proxy[mask]
        values = values[np.isfinite(values) & (values > 0.0)]
    else:
        values = proxy[np.isfinite(proxy) & (proxy > 0.0)]

    if values.size == 0:
        return np.zeros_like(proxy, dtype=np.float32)

    normalizer = float(np.quantile(values, 0.95))
    if not np.isfinite(normalizer) or normalizer <= 1e-6:
        normalizer = float(np.max(values))
    if not np.isfinite(normalizer) or normalizer <= 1e-6:
        return np.zeros_like(proxy, dtype=np.float32)
    return np.clip(proxy / normalizer, 0.0, 1.0).astype(np.float32, copy=False)
