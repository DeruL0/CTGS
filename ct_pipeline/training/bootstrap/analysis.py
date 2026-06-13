from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from ct_pipeline.data.preprocessing import build_support_signed_distance
from ct_pipeline.geometry.curvature import compute_curvature_proxy_np
from ct_pipeline.training.losses import sample_sdf_normals, sample_volume_field
from ct_pipeline.training.utils import as_device_tensor

def _ct_spatial_extent(volume_shape_dhw, spacing_zyx):
    extents = [float(dim) * float(spacing) for dim, spacing in zip(volume_shape_dhw, spacing_zyx)]
    return max(extents)


def _to_cuda_analysis(analysis):
    analysis_cuda = {}
    for key, value in analysis.items():
        if isinstance(value, torch.Tensor):
            analysis_cuda[key] = value.to(device="cuda")
        elif isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                analysis_cuda[key] = as_device_tensor(value, device="cuda", dtype=torch.bool)
            elif np.issubdtype(value.dtype, np.integer):
                analysis_cuda[key] = as_device_tensor(value, device="cuda", dtype=torch.long)
            else:
                analysis_cuda[key] = as_device_tensor(value, device="cuda", dtype=torch.float32)
        else:
            analysis_cuda[key] = value
    return analysis_cuda


def _central_difference_axis_np(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    out = np.empty_like(field, dtype=np.float32)
    scale = np.float32(0.5 / max(float(spacing), 1e-8))
    out.fill(0.0)
    if field.shape[axis] <= 1:
        return out

    center = [slice(None)] * field.ndim
    before = [slice(None)] * field.ndim
    after = [slice(None)] * field.ndim
    center[axis] = slice(1, -1)
    before[axis] = slice(None, -2)
    after[axis] = slice(2, None)
    out[tuple(center)] = (field[tuple(after)] - field[tuple(before)]) * scale

    first = [slice(None)] * field.ndim
    first_next = [slice(None)] * field.ndim
    first[axis] = 0
    first_next[axis] = 1
    out[tuple(first)] = (field[tuple(first_next)] - field[tuple(first)]) * scale

    last = [slice(None)] * field.ndim
    last_prev = [slice(None)] * field.ndim
    last[axis] = -1
    last_prev[axis] = -2
    out[tuple(last)] = (field[tuple(last)] - field[tuple(last_prev)]) * scale
    return out


def _load_ct_analysis_bundle(ct_phase1_dir):
    bundle_dir = Path(ct_phase1_dir)
    analysis_path = bundle_dir / "analysis.npz"
    metadata_path = bundle_dir / "metadata.json"
    if not analysis_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("CT Phase 1 bundle must contain analysis.npz and metadata.json.")
    with np.load(str(analysis_path)) as analysis_npz:
        analysis = {key: analysis_npz[key] for key in analysis_npz.files}
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return analysis, metadata, str(analysis_path), str(metadata_path)


def _ensure_intensity_driven_analysis(analysis):
    upgraded = dict(analysis)
    if "coarse_support_mask" not in upgraded and "material_mask" in upgraded:
        upgraded["coarse_support_mask"] = upgraded["material_mask"]
    required_keys = (
        "coarse_support_mask",
        "roi_bbox",
        "material_mask",
        "foreground_mask",
        "void_mask",
        "boundary_points",
        "boundary_normals",
        "boundary_tangent_u",
        "boundary_tangent_v",
        "boundary_strength",
        "boundary_material_id",
        "interior_points",
        "interior_density_seed",
        "interior_material_id",
    )
    missing = [key for key in required_keys if key not in upgraded]
    if missing:
        raise RuntimeError(
            "The active CTGS training path only supports canonical Phase 1 bundles. "
            f"Missing keys: {', '.join(missing)}"
        )
    return upgraded


def _prepare_curvature_proxy_field(volume_np, spacing_zyx, sigma: float = 1.0, device="cuda"):
    proxy = compute_curvature_proxy_np(volume_np, spacing_zyx, sigma=float(sigma))
    curvature_tensor = as_device_tensor(proxy, device=device, dtype=torch.float32)
    return {
        "curvature": curvature_tensor.unsqueeze(0).unsqueeze(0).contiguous(),
        "spacing_zyx": tuple(float(value) for value in spacing_zyx),
    }


def _prepare_support_distance_field(analysis, spacing_zyx, device=None):
    if isinstance(analysis.get("material_mask"), torch.Tensor):
        support_mask = analysis["material_mask"].detach().cpu().numpy().astype(bool)
    else:
        support_mask = np.asarray(analysis["material_mask"], dtype=bool)
    try:
        support_distance = ndimage.distance_transform_edt(support_mask, sampling=spacing_zyx).astype(np.float32)
    except MemoryError:
        print("CT support EDT allocation failed; falling back to chamfer distance for bulk scale limits.")
        try:
            support_distance = ndimage.distance_transform_cdt(support_mask, metric="chessboard").astype(np.float32)
            support_distance *= float(min(spacing_zyx))
        except MemoryError:
            print("CT support chamfer distance allocation failed; using conservative one-voxel bulk scale limits.")
            support_distance = support_mask.astype(np.float32) * float(min(spacing_zyx))
    if device is None:
        return {
            "support_distance_native": support_distance,
            "support_distance": support_distance[None, None, ...],
            "spacing_zyx": tuple(float(value) for value in spacing_zyx),
        }
    support_distance_tensor = as_device_tensor(support_distance, device=device, dtype=torch.float32)
    return {
        "support_distance_native": support_distance_tensor.contiguous(),
        "support_distance": support_distance_tensor.unsqueeze(0).unsqueeze(0).contiguous(),
        "spacing_zyx": tuple(float(value) for value in spacing_zyx),
    }


def _empty_support_distance_field(spacing_zyx, device=None):
    if device is None:
        support_distance = np.zeros((1, 1, 1), dtype=np.float32)
        return {
            "support_distance_native": support_distance,
            "support_distance": support_distance[None, None, ...],
            "spacing_zyx": tuple(float(value) for value in spacing_zyx),
        }
    support_distance_tensor = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
    return {
        "support_distance_native": support_distance_tensor,
        "support_distance": support_distance_tensor.unsqueeze(0).unsqueeze(0).contiguous(),
        "spacing_zyx": tuple(float(value) for value in spacing_zyx),
    }


def _prepare_signed_distance_field(analysis, spacing_zyx, device=None, boundary_mode: str = "interface"):
    if isinstance(analysis.get("material_mask"), torch.Tensor):
        support_mask = analysis["material_mask"].detach().cpu().numpy().astype(bool)
    else:
        support_mask = np.asarray(analysis["material_mask"], dtype=bool)

    saved_signed_distance = analysis.get("material_signed_distance")
    if saved_signed_distance is not None:
        if isinstance(saved_signed_distance, torch.Tensor):
            signed_distance = saved_signed_distance.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            signed_distance = np.asarray(saved_signed_distance, dtype=np.float32)
        if signed_distance.shape != support_mask.shape:
            raise ValueError("material_signed_distance and material_mask must have the same shape.")
    else:
        signed_distance = build_support_signed_distance(
            support_mask,
            tuple(float(value) for value in spacing_zyx),
            boundary_mode=boundary_mode,
        )

    precompute_normals = int(signed_distance.size) <= 8_000_000
    sdf_normal_channels = None
    if precompute_normals:
        spacing_z, spacing_y, spacing_x = [max(float(value), 1e-8) for value in spacing_zyx]
        grad_z = _central_difference_axis_np(signed_distance, axis=0, spacing=spacing_z)
        grad_y = _central_difference_axis_np(signed_distance, axis=1, spacing=spacing_y)
        grad_x = _central_difference_axis_np(signed_distance, axis=2, spacing=spacing_x)
        sdf_normal_channels = np.stack((grad_x, grad_y, grad_z), axis=0).astype(np.float32, copy=False)
        sdf_normal_norm = np.linalg.norm(sdf_normal_channels, axis=0, keepdims=True)
        valid_normal = sdf_normal_norm > 1e-8
        sdf_normal_channels = np.where(valid_normal, sdf_normal_channels / np.maximum(sdf_normal_norm, 1e-8), 0.0).astype(
            np.float32,
            copy=False,
        )
    elif device is not None:
        print("CT SDF normal volume skipped for large field; normals will be sampled from SDF on demand.")

    if device is None:
        result = {
            "signed_distance_native": signed_distance,
            "signed_distance": signed_distance[None, None, ...],
            "spacing_zyx": tuple(float(value) for value in spacing_zyx),
        }
        if sdf_normal_channels is not None:
            result["sdf_normal_native"] = np.moveaxis(sdf_normal_channels, 0, -1)
            result["sdf_normal"] = sdf_normal_channels[None, ...]
        return result
    signed_distance_tensor = as_device_tensor(signed_distance, device=device, dtype=torch.float32)
    result = {
        "signed_distance_native": signed_distance,
        "signed_distance": signed_distance_tensor.unsqueeze(0).unsqueeze(0).contiguous(),
        "spacing_zyx": tuple(float(value) for value in spacing_zyx),
    }
    if sdf_normal_channels is not None:
        sdf_normal_tensor = as_device_tensor(sdf_normal_channels, device=device, dtype=torch.float32)
        result["sdf_normal_native"] = torch.movedim(sdf_normal_tensor, 0, -1).contiguous()
        result["sdf_normal"] = sdf_normal_tensor.unsqueeze(0).contiguous()
    return result


def _prepare_intensity_calibration(analysis, volume_np):
    volume = np.asarray(volume_np, dtype=np.float32)
    if isinstance(analysis.get("material_mask"), torch.Tensor):
        support_mask = analysis["material_mask"].detach().cpu().numpy().astype(bool)
    else:
        support_mask = np.asarray(analysis["material_mask"], dtype=bool)

    if "air_mask" in analysis:
        air_mask = np.asarray(analysis["air_mask"], dtype=bool)
    else:
        void_mask = np.asarray(analysis.get("void_mask", np.zeros_like(support_mask, dtype=bool)), dtype=bool)
        air_mask = np.logical_or(~support_mask, void_mask)

    finite_volume = volume[np.isfinite(volume)]
    if finite_volume.size == 0:
        return 0.0, 1.0

    material_values = volume[np.logical_and(support_mask, np.isfinite(volume))]
    air_values = volume[np.logical_and(air_mask, np.isfinite(volume))]
    intensity_air = float(np.quantile(air_values, 0.5)) if air_values.size > 0 else float(np.quantile(finite_volume, 0.05))
    intensity_mat = float(np.quantile(material_values, 0.5)) if material_values.size > 0 else float(np.quantile(finite_volume, 0.95))
    if not np.isfinite(intensity_air):
        intensity_air = float(np.quantile(finite_volume, 0.05))
    if not np.isfinite(intensity_mat):
        intensity_mat = float(np.quantile(finite_volume, 0.95))
    if abs(intensity_mat - intensity_air) <= 1e-6:
        fallback_mat = float(np.quantile(material_values, 0.95)) if material_values.size > 0 else float(np.quantile(finite_volume, 0.95))
        intensity_mat = fallback_mat if abs(fallback_mat - intensity_air) > 1e-6 else intensity_air + 1.0
    return intensity_air, intensity_mat


def _sample_coarse_sdf_normals(signed_distance_field: dict, points_xyz: torch.Tensor, spacing_zyx):
    normal_volume = signed_distance_field.get("sdf_normal")
    field_spacing = signed_distance_field.get("spacing_zyx", spacing_zyx)
    if normal_volume is not None:
        return sample_volume_field(normal_volume, points_xyz, field_spacing)
    return sample_sdf_normals(signed_distance_field["signed_distance"], points_xyz, field_spacing)


def _require_active_boundary_bundle(analysis):
    required_keys = (
        "coarse_support_mask",
        "roi_bbox",
        "material_mask",
        "foreground_mask",
        "void_mask",
        "boundary_points",
        "boundary_normals",
        "boundary_tangent_u",
        "boundary_tangent_v",
        "boundary_strength",
        "boundary_material_id",
        "interior_points",
        "interior_density_seed",
        "interior_material_id",
    )
    missing = [key for key in required_keys if key not in analysis]
    if missing:
        raise RuntimeError(
            "The active intensity-driven CT training path requires a coarse-support Phase 1 bundle. "
            f"Missing keys: {', '.join(missing)}"
        )
