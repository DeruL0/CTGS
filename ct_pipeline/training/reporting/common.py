import numpy as np
import torch

from ct_pipeline.training.utils import as_device_tensor


def _quantile_metrics(values, names):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {name: float("nan") for name in names}
    quantiles = [float(name[1:]) / 100.0 for name in names]
    measured = np.quantile(values, quantiles)
    return {name: float(value) for name, value in zip(names, measured)}

def _voxel_indices_to_world(indices_zyx: np.ndarray, spacing_zyx, device, dtype) -> torch.Tensor:
    if indices_zyx.size == 0:
        return torch.empty((0, 3), dtype=dtype, device=device)
    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
    points_np = np.stack(
        (
            (indices_zyx[:, 2].astype(np.float32) + 0.5) * spacing_x,
            (indices_zyx[:, 1].astype(np.float32) + 0.5) * spacing_y,
            (indices_zyx[:, 0].astype(np.float32) + 0.5) * spacing_z,
        ),
        axis=1,
    )
    return as_device_tensor(points_np, device=device, dtype=dtype, reshape=(-1, 3))

def _mask_to_np(mask, *, fallback_shape=None):
    if mask is None:
        if fallback_shape is None:
            return None
        return np.zeros(tuple(int(value) for value in fallback_shape), dtype=bool)
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy().astype(bool)
    return np.asarray(mask, dtype=bool)

def _roi_window_from_analysis(analysis, shape):
    roi_bbox = analysis.get("roi_bbox")
    if roi_bbox is None:
        return np.ones(tuple(int(value) for value in shape), dtype=bool)
    if isinstance(roi_bbox, torch.Tensor):
        roi_bbox = roi_bbox.detach().cpu().numpy()
    roi_bbox = np.asarray(roi_bbox, dtype=np.int64)
    window = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    lower = np.maximum(roi_bbox[:, 0], 0)
    upper = np.minimum(roi_bbox[:, 1], np.asarray(shape, dtype=np.int64))
    window[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]] = True
    return window

def _roi_bbox_from_analysis(analysis, shape):
    roi_bbox = analysis.get("roi_bbox")
    if roi_bbox is None:
        return np.asarray([[0, int(shape[0])], [0, int(shape[1])], [0, int(shape[2])]], dtype=np.int32)
    if isinstance(roi_bbox, torch.Tensor):
        roi_bbox = roi_bbox.detach().cpu().numpy()
    roi_bbox = np.asarray(roi_bbox, dtype=np.int64)
    lower = np.maximum(roi_bbox[:, 0], 0)
    upper = np.minimum(roi_bbox[:, 1], np.asarray(shape, dtype=np.int64))
    return np.stack((lower, upper), axis=1).astype(np.int32, copy=False)

def _sample_mask_points(mask, spacing_zyx, device, dtype, max_count):
    indices = np.argwhere(np.asarray(mask, dtype=bool))
    if indices.shape[0] == 0:
        return None
    if indices.shape[0] > int(max_count):
        sample_ids = np.linspace(0, indices.shape[0] - 1, int(max_count)).astype(np.int64)
        indices = indices[sample_ids]
    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
    points_np = np.stack(
        (
            (indices[:, 2].astype(np.float32) + 0.5) * spacing_x,
            (indices[:, 1].astype(np.float32) + 0.5) * spacing_y,
            (indices[:, 0].astype(np.float32) + 0.5) * spacing_z,
        ),
        axis=1,
    )
    return as_device_tensor(points_np, device=device, dtype=dtype, reshape=(-1, 3))
