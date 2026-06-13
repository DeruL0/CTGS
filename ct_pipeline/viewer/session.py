from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ct_pipeline.rendering.slices import (
    CTPatchGridCache,
    CTRenderState,
    _WORLD_AXIS_RENDER_CONFIG,
    _build_world_query_points_from_base,
    _normalize_world_axis,
    prepare_ct_render_state,
    render_ct_slice_world_patch,
)
from ct_pipeline.rendering.bulk_support import DEFAULT_BULK_QUERY_TRUNCATION_SIGMA
from scene.ct_gaussian_model import CTGaussianModel


BUFFER_FIELDS = {
    "position": {"offset": 0, "size": 3},
    "scale": {"offset": 3, "size": 3},
    "rotation": {"offset": 6, "size": 4},
    "normal": {"offset": 10, "size": 3},
    "opacity": {"offset": 13, "size": 1},
    "region_type": {"offset": 14, "size": 1},
    "material_id": {"offset": 15, "size": 1},
    "support_radius": {"offset": 16, "size": 1},
    "attenuation": {"offset": 17, "size": 1},
}
BUFFER_STRIDE_FLOATS = 18
BUFFER_STRIDE_BYTES = BUFFER_STRIDE_FLOATS * 4


def _subset_render_state(render_state: CTRenderState, mask: torch.Tensor) -> CTRenderState:
    return CTRenderState(
        means=render_state.means[mask],
        rotations=render_state.rotations[mask],
        scales=render_state.scales[mask],
        opacity=render_state.opacity[mask],
        radius=render_state.radius[mask],
    )


def _model_bbox(model, bulk_mask: torch.Tensor, bulk_support_radius: torch.Tensor, padding_factor: float = 3.0):
    xyz = model.get_xyz.detach()
    scales = model.get_scaling.detach()
    if xyz.numel() == 0:
        raise ValueError("Cannot build a viewer session from an empty CTGS model.")
    radius = padding_factor * scales.max(dim=1).values.unsqueeze(1)
    if bulk_support_radius.numel() > 0:
        radius[bulk_mask] = bulk_support_radius.reshape(-1, 1)
    lower = (xyz - radius).min(dim=0).values
    upper = (xyz + radius).max(dim=0).values
    return torch.stack((lower, upper), dim=0)


def _layer_bounds_from_bbox(axis: str, bbox_min, bbox_max):
    if axis == "x":
        return ((float(bbox_min[2]), float(bbox_max[2])), (float(bbox_min[1]), float(bbox_max[1])))
    if axis == "y":
        return ((float(bbox_min[2]), float(bbox_max[2])), (float(bbox_min[0]), float(bbox_max[0])))
    if axis == "z":
        return ((float(bbox_min[1]), float(bbox_max[1])), (float(bbox_min[0]), float(bbox_max[0])))
    raise ValueError("axis must be one of {'x', 'y', 'z'}.")


def _normalize_axis(axis: str) -> str:
    normalized = str(axis).lower()
    if normalized not in {"x", "y", "z"}:
        raise ValueError("axis must be one of {'x', 'y', 'z'}.")
    return normalized


def _normalize_layer(layer: str) -> str:
    normalized = str(layer).lower()
    if normalized not in {"all", "surface", "bulk"}:
        raise ValueError("layer must be one of {'all', 'surface', 'bulk'}.")
    return normalized


def _normalize_t(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _load_bulk_support_radius(bulk_scales: torch.Tensor) -> torch.Tensor:
    sigma = bulk_scales.max(dim=1).values.clamp_min(1e-6)
    return float(DEFAULT_BULK_QUERY_TRUNCATION_SIGMA) * sigma


def _bulk_intensity_stats(bulk_attenuation: torch.Tensor) -> Dict[str, float]:
    values = bulk_attenuation.detach().reshape(-1)
    if values.numel() == 0:
        return {
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 1.0,
            "p99": 1.0,
            "max": 1.0,
        }
    values_np = values.float().clamp(0.0, 1.0).cpu().numpy()
    return {
        "min": float(np.min(values_np)),
        "p01": float(np.quantile(values_np, 0.01)),
        "p05": float(np.quantile(values_np, 0.05)),
        "p50": float(np.quantile(values_np, 0.50)),
        "p95": float(np.quantile(values_np, 0.95)),
        "p99": float(np.quantile(values_np, 0.99)),
        "max": float(np.max(values_np)),
    }


def _normalize_slice_preview_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"raw", "clipped", "mask"}:
        raise ValueError("preview_mode must be one of {'raw', 'clipped', 'mask'}.")
    return normalized


def _apply_intensity_preview(
    image: np.ndarray,
    *,
    intensity_min: float,
    intensity_max: float,
    intensity_clip: bool,
    preview_mode: str,
) -> np.ndarray:
    mode = _normalize_slice_preview_mode(preview_mode)
    source = np.asarray(image, dtype=np.float32)
    if mode == "raw" or not bool(intensity_clip):
        return np.clip(source, 0.0, 1.0).astype(np.float32)

    lo = float(np.clip(min(float(intensity_min), float(intensity_max)), 0.0, 1.0))
    hi = float(np.clip(max(float(intensity_min), float(intensity_max)), 0.0, 1.0))
    in_range = np.logical_and(source > 1e-8, np.logical_and(source >= lo, source <= hi))
    if mode == "mask":
        return in_range.astype(np.float32)
    return np.where(in_range, source, 0.0).astype(np.float32)


def _pack_gaussian_buffer(model, bulk_mask: torch.Tensor, bulk_support_radius: torch.Tensor) -> tuple[bytes, Dict[str, object]]:
    xyz = model.get_xyz.detach().cpu().numpy().astype(np.float32)
    scales = model.get_scaling.detach().cpu().numpy().astype(np.float32)
    rotation = model.get_rotation.detach().cpu().numpy().astype(np.float32)
    normals = F.normalize(model._normal.detach(), dim=1).cpu().numpy().astype(np.float32)
    opacity = model.get_opacity.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
    region_type = model.get_region_type.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
    material_id = model.get_material_id.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
    attenuation = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    model_attenuation = model.get_attenuation.detach()
    if model_attenuation.numel() == xyz.shape[0]:
        attenuation = model_attenuation.cpu().numpy().reshape(-1, 1).astype(np.float32)

    packed = np.zeros((xyz.shape[0], BUFFER_STRIDE_FLOATS), dtype=np.float32)
    packed[:, 0:3] = xyz
    packed[:, 3:6] = scales
    packed[:, 6:10] = rotation
    packed[:, 10:13] = normals
    packed[:, 13:14] = opacity
    packed[:, 14:15] = region_type
    packed[:, 15:16] = material_id
    if bulk_support_radius.numel() > 0:
        packed[bulk_mask.detach().cpu().numpy(), 16] = bulk_support_radius.detach().cpu().numpy()
    packed[:, 17:18] = attenuation

    meta = {
        "count": int(packed.shape[0]),
        "stride_floats": BUFFER_STRIDE_FLOATS,
        "stride_bytes": BUFFER_STRIDE_BYTES,
        "dtype": "float32",
        "fields": BUFFER_FIELDS,
    }
    return packed.tobytes(), meta


@dataclass
class ViewerSession:
    ply_path: Path
    model: CTGaussianModel
    render_state_all: CTRenderState
    render_state_surface: CTRenderState
    render_state_bulk: CTRenderState
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    gaussian_buffer: bytes
    gaussian_buffer_meta: Dict[str, object]
    surface_count: int
    bulk_count: int
    bulk_support_radius: torch.Tensor
    surface_attenuation: torch.Tensor
    bulk_attenuation: torch.Tensor
    device: str
    frontend_dist: Path | None
    patch_grid_cache: CTPatchGridCache

    def session_payload(self) -> Dict[str, object]:
        bbox_size = self.bbox_max - self.bbox_min
        center = 0.5 * (self.bbox_min + self.bbox_max)
        max_extent = float(np.max(bbox_size))
        return {
            "ply_path": str(self.ply_path),
            "gaussian_count": int(self.surface_count + self.bulk_count),
            "surface_count": int(self.surface_count),
            "bulk_count": int(self.bulk_count),
            "device": self.device,
            "bbox": {
                "min": self.bbox_min.tolist(),
                "max": self.bbox_max.tolist(),
                "size": bbox_size.tolist(),
                "center": center.tolist(),
            },
            "available_axes": ["x", "y", "z"],
            "render_modes": ["composite", "surface-lit", "surface-normal", "region", "bulk-only", "intensity"],
            "slice_layers": ["all", "surface", "bulk"],
            "slice_preview_modes": ["raw", "clipped", "mask"],
            "surface_intensity": _bulk_intensity_stats(self.surface_attenuation),
            "bulk_intensity": _bulk_intensity_stats(self.bulk_attenuation),
            "defaults": {
                "renderMode": "composite",
                "axis": "z",
                "sliceT": 0.5,
                "sliceLayer": "bulk",
                "surfaceAlpha": 0.92,
                "bulkAlpha": 0.28,
                "sliceFadeWidthMm": max(1e-3, max_extent * 0.08),
                "clipSoftnessMm": max(1e-3, max_extent * 0.03),
                "clipEnabled": False,
                "sliceSize": 512,
                "fpsLimit": 60,
                "splatRadiusScale": 1.0,
                "intensityClipEnabled": False,
                "intensityMin": 0.0,
                "intensityMax": 1.0,
                "slicePreviewMode": "raw",
            },
        }

    def gaussian_meta_payload(self) -> Dict[str, object]:
        payload = dict(self.gaussian_buffer_meta)
        payload["surface_count"] = int(self.surface_count)
        payload["bulk_count"] = int(self.bulk_count)
        return payload

    def render_slice(
        self,
        axis: str,
        t: float,
        layer: str = "all",
        size: int = 256,
        intensity_min: float = 0.0,
        intensity_max: float = 1.0,
        intensity_clip: bool = False,
        preview_mode: str = "raw",
    ) -> np.ndarray:
        normalized_axis = _normalize_axis(axis)
        normalized_layer = _normalize_layer(layer)
        normalized_t = _normalize_t(t)
        _normalize_slice_preview_mode(preview_mode)
        patch_size = max(64, min(int(size), 1024))

        axis_index = {"x": 0, "y": 1, "z": 2}[normalized_axis]
        slice_coord = float(self.bbox_min[axis_index] + normalized_t * (self.bbox_max[axis_index] - self.bbox_min[axis_index]))
        bounds_hw = _layer_bounds_from_bbox(normalized_axis, self.bbox_min, self.bbox_max)

        if normalized_layer != "surface":
            raw_patch = self._render_bulk_raw_a_b(normalized_axis, slice_coord, bounds_hw, patch_size)
            preview_patch = _apply_intensity_preview(
                raw_patch,
                intensity_min=float(intensity_min),
                intensity_max=float(intensity_max),
                intensity_clip=bool(intensity_clip),
                preview_mode=preview_mode,
            )
            return np.flipud(preview_patch)

        with torch.no_grad():
            density_patch = render_ct_slice_world_patch(
                self.render_state_surface,
                axis=normalized_axis,
                slice_coord=slice_coord,
                bounds_hw=bounds_hw,
                patch_size_hw=(patch_size, patch_size),
                patch_grid_cache=self.patch_grid_cache,
                output_mode="density",
            )
            occupancy_patch = (1.0 - torch.exp(-density_patch.clamp_min(0.0))).clamp(0.0, 1.0)
            patch_np = occupancy_patch.detach().cpu().numpy().astype(np.float32)
        return np.flipud(patch_np)

    def _render_bulk_raw_a_b(self, axis: str, slice_coord: float, bounds_hw, patch_size: int) -> np.ndarray:
        """Render the ungated anisotropic raw bulk A_b used by previews."""
        render_state = self.render_state_bulk
        if render_state.means.numel() == 0:
            return np.zeros((patch_size, patch_size), dtype=np.float32)
        device = render_state.device
        dtype = render_state.dtype
        axis_index = _normalize_world_axis(axis)
        axis_cfg = _WORLD_AXIS_RENDER_CONFIG[axis_index]
        rr, cc = self.patch_grid_cache.get((patch_size, patch_size), device, dtype)
        query_points = _build_world_query_points_from_base(
            rr, cc, axis_index, float(slice_coord), bounds_hw, (patch_size, patch_size)
        )

        means = render_state.means
        scales = render_state.scales.clamp_min(1e-6)
        rotations = render_state.rotations
        opacity = render_state.opacity.reshape(-1).clamp(0.0, 1.0)
        radius = self.bulk_support_radius.reshape(-1).clamp_min(1e-6)
        attenuation = self.bulk_attenuation.reshape(-1)

        plane_axis = int(axis_cfg["plane_axis"])
        dim_h = int(axis_cfg["dim_h"])
        dim_w = int(axis_cfg["dim_w"])
        (h_min, h_max), (w_min, w_max) = bounds_hw
        visible = (means[:, plane_axis] - float(slice_coord)).abs() <= radius
        visible &= means[:, dim_h] >= float(h_min) - radius
        visible &= means[:, dim_h] <= float(h_max) + radius
        visible &= means[:, dim_w] >= float(w_min) - radius
        visible &= means[:, dim_w] <= float(w_max) + radius
        if not torch.any(visible):
            return np.zeros((patch_size, patch_size), dtype=np.float32)

        means = means[visible]
        scales = scales[visible]
        rotations = rotations[visible]
        opacity = opacity[visible]
        radius = radius[visible]
        attenuation = attenuation[visible]
        mu = torch.zeros((query_points.shape[0],), dtype=torch.float32, device=device)
        den = torch.zeros_like(mu)
        chunk_size = max(1, 4_194_304 // max(1, int(query_points.shape[0])))
        with torch.no_grad():
            for start in range(0, means.shape[0], chunk_size):
                end = min(start + chunk_size, means.shape[0])
                diff = query_points[:, None, :] - means[None, start:end, :]
                local = torch.einsum("qni,nij->qnj", diff, rotations[start:end])
                q = torch.sum(torch.square(local / scales[None, start:end]), dim=-1)
                kernel = torch.exp(-0.5 * q)
                weighted_kernel = kernel * opacity[None, start:end]
                den += weighted_kernel.sum(dim=1).to(dtype=den.dtype)
                mu += (weighted_kernel * attenuation[None, start:end]).sum(dim=1).to(dtype=mu.dtype)
        result = torch.where(den > 1e-8, mu / den, torch.zeros_like(mu))
        return result.clamp(0.0, 1.0).reshape(patch_size, patch_size).detach().cpu().numpy().astype(np.float32)

def load_viewer_session(ply_path: Path, *, device=None) -> ViewerSession:
    resolved_ply = Path(ply_path).expanduser().resolve()
    if not resolved_ply.exists():
        raise FileNotFoundError(f"PLY file does not exist: {resolved_ply}")

    model = CTGaussianModel(0, device=device)
    model.load_ply(str(resolved_ply))
    render_state = prepare_ct_render_state(model)
    region_type = model.get_region_type.reshape(-1)
    surface_mask = region_type == 0
    bulk_mask = region_type == 1
    render_state_surface = _subset_render_state(render_state, surface_mask)
    render_state_bulk = _subset_render_state(render_state, bulk_mask)
    bulk_support_radius = _load_bulk_support_radius(render_state_bulk.scales)
    attenuation = model.get_attenuation.detach().reshape(-1)
    surface_attenuation = attenuation[surface_mask] if attenuation.numel() > 0 else torch.empty((0,), device=model.get_xyz.device)
    bulk_attenuation = attenuation[bulk_mask] if attenuation.numel() > 0 else torch.empty((0,), device=model.get_xyz.device)
    bbox = _model_bbox(model, bulk_mask, bulk_support_radius).detach().cpu().numpy().astype(np.float32)
    gaussian_buffer, gaussian_buffer_meta = _pack_gaussian_buffer(model, bulk_mask, bulk_support_radius)
    frontend_dist = Path(__file__).resolve().parents[2] / "viewer" / "dist"

    return ViewerSession(
        ply_path=resolved_ply,
        model=model,
        render_state_all=render_state,
        render_state_surface=render_state_surface,
        render_state_bulk=render_state_bulk,
        bbox_min=bbox[0],
        bbox_max=bbox[1],
        gaussian_buffer=gaussian_buffer,
        gaussian_buffer_meta=gaussian_buffer_meta,
        surface_count=int(surface_mask.sum().item()),
        bulk_count=int(bulk_mask.sum().item()),
        bulk_support_radius=bulk_support_radius,
        surface_attenuation=surface_attenuation,
        bulk_attenuation=bulk_attenuation,
        device=str(model.get_xyz.device),
        frontend_dist=frontend_dist,
        patch_grid_cache=CTPatchGridCache(),
    )
