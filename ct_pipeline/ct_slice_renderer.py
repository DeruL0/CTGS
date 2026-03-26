from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from utils.rotation_utils import quaternion_to_matrix


_AXIS_MAP = {"z": 0, "y": 1, "x": 2, 0: 0, 1: 1, 2: 2}
_AXIS_RENDER_CONFIG = {
    0: {"plane_axis": 2, "dim_h": 1, "dim_w": 0},
    1: {"plane_axis": 1, "dim_h": 2, "dim_w": 0},
    2: {"plane_axis": 0, "dim_h": 2, "dim_w": 1},
}


@dataclass
class CTRenderState:
    means: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    opacity: torch.Tensor
    radius: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.means.dtype


class CTPatchGridCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, patch_size_hw, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(patch_size_hw, int):
            patch_height = patch_width = int(patch_size_hw)
        else:
            patch_height, patch_width = [int(value) for value in patch_size_hw]
        key = (patch_height, patch_width, str(device), str(dtype))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        rows = torch.arange(patch_height, device=device, dtype=dtype)
        cols = torch.arange(patch_width, device=device, dtype=dtype)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        self._cache[key] = (rr, cc)
        return rr, cc


def _normalize_axis(axis) -> int:
    if axis not in _AXIS_MAP:
        raise ValueError("axis must be one of {'z', 'y', 'x', 0, 1, 2}.")
    return _AXIS_MAP[axis]


def _slice_shape(volume_shape_dhw: Tuple[int, int, int], axis_index: int) -> Tuple[int, int]:
    depth, height, width = [int(value) for value in volume_shape_dhw]
    if axis_index == 0:
        return height, width
    if axis_index == 1:
        return depth, width
    return depth, height


def _normalize_patch_parameters(patch_origin_hw, patch_size_hw, slice_shape_hw):
    slice_height, slice_width = [int(value) for value in slice_shape_hw]
    if isinstance(patch_size_hw, int):
        patch_height = patch_width = int(patch_size_hw)
    else:
        patch_height, patch_width = [int(value) for value in patch_size_hw]

    patch_height = max(1, min(patch_height, slice_height))
    patch_width = max(1, min(patch_width, slice_width))

    origin_h, origin_w = [int(value) for value in patch_origin_hw]
    origin_h = max(0, min(origin_h, slice_height - patch_height))
    origin_w = max(0, min(origin_w, slice_width - patch_width))
    return (origin_h, origin_w), (patch_height, patch_width)


def prepare_ct_render_state(gaussians) -> CTRenderState:
    means = gaussians.get_xyz
    if means.numel() == 0:
        empty = torch.empty((0,), dtype=means.dtype, device=means.device)
        return CTRenderState(
            means=means,
            rotations=torch.empty((0, 3, 3), dtype=means.dtype, device=means.device),
            scales=torch.empty((0, 3), dtype=means.dtype, device=means.device),
            opacity=empty,
            radius=empty,
        )

    rotations = quaternion_to_matrix(gaussians.get_rotation)
    scales = gaussians.get_scaling.clamp_min(1e-6)
    opacity = gaussians.get_opacity.squeeze(-1)
    radius = 4.0 * scales.max(dim=1).values
    return CTRenderState(
        means=means,
        rotations=rotations,
        scales=scales,
        opacity=opacity,
        radius=radius,
    )


def _build_query_points_from_base(
    rr: torch.Tensor,
    cc: torch.Tensor,
    axis_index: int,
    slice_idx: int,
    patch_origin_hw,
    spacing_zyx,
) -> torch.Tensor:
    origin_h, origin_w = patch_origin_hw
    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]

    rr = rr + float(origin_h)
    cc = cc + float(origin_w)

    if axis_index == 0:
        x = cc * spacing_x
        y = rr * spacing_y
        z = torch.full_like(rr, float(slice_idx) * spacing_z)
    elif axis_index == 1:
        x = cc * spacing_x
        y = torch.full_like(rr, float(slice_idx) * spacing_y)
        z = rr * spacing_z
    else:
        x = torch.full_like(rr, float(slice_idx) * spacing_x)
        y = cc * spacing_y
        z = rr * spacing_z

    return torch.stack((x, y, z), dim=-1).reshape(-1, 3)


def _render_ct_slice_patch_impl(
    query_points: torch.Tensor,
    slice_coord: torch.Tensor,
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    radius: torch.Tensor,
    plane_axis: int,
    dim_h: int,
    dim_w: int,
    spacing_axis: float,
    gaussians_per_chunk: int,
) -> torch.Tensor:
    if means.numel() == 0:
        return torch.zeros((query_points.shape[0],), dtype=query_points.dtype, device=query_points.device)

    plane_mask = torch.abs(means[:, plane_axis] - slice_coord) <= (radius + float(spacing_axis))

    min_h = torch.min(query_points[:, dim_h])
    max_h = torch.max(query_points[:, dim_h])
    min_w = torch.min(query_points[:, dim_w])
    max_w = torch.max(query_points[:, dim_w])
    plane_mask = plane_mask & (means[:, dim_h] >= (min_h - radius)) & (means[:, dim_h] <= (max_h + radius))
    plane_mask = plane_mask & (means[:, dim_w] >= (min_w - radius)) & (means[:, dim_w] <= (max_w + radius))

    if not torch.any(plane_mask):
        return torch.zeros((query_points.shape[0],), dtype=query_points.dtype, device=query_points.device)

    visible_means = means[plane_mask]
    visible_rotations = rotations[plane_mask]
    visible_scales = scales[plane_mask]
    visible_opacity = opacity[plane_mask]

    patch_values = torch.zeros((query_points.shape[0],), dtype=query_points.dtype, device=query_points.device)
    chunk_size = max(1, int(gaussians_per_chunk))
    for start in range(0, visible_means.shape[0], chunk_size):
        end = min(start + chunk_size, visible_means.shape[0])
        mean_chunk = visible_means[start:end]
        rotation_chunk = visible_rotations[start:end]
        scale_chunk = visible_scales[start:end]
        opacity_chunk = visible_opacity[start:end]

        diff = query_points.unsqueeze(1) - mean_chunk.unsqueeze(0)
        local = torch.einsum("qci,cij->qcj", diff, rotation_chunk)
        normalized = local / scale_chunk.unsqueeze(0)
        exponent = -0.5 * torch.sum(normalized * normalized, dim=-1)
        patch_values += (torch.exp(exponent) * opacity_chunk.unsqueeze(0)).sum(dim=1)

    return patch_values


def _resolve_render_state(render_source) -> CTRenderState:
    if isinstance(render_source, CTRenderState):
        return render_source
    return prepare_ct_render_state(render_source)


def _is_compile_fallback_error(exc: Exception) -> bool:
    exc_type_name = type(exc).__name__.lower()
    exc_message = str(exc).lower()
    return any(
        token in exc_type_name or token in exc_message
        for token in (
            "tritonmissing",
            "backendcompilerfailed",
            "inductor",
            "triton",
        )
    )


def _has_triton_support() -> bool:
    return importlib.util.find_spec("triton") is not None


def render_ct_slice_patch(
    render_source,
    axis,
    slice_idx,
    patch_origin_hw,
    patch_size_hw,
    spacing_zyx,
    volume_shape_dhw,
    gaussians_per_chunk: int = 2048,
    patch_grid_cache: Optional[CTPatchGridCache] = None,
    render_impl: Optional[Callable[..., torch.Tensor]] = None,
):
    axis_index = _normalize_axis(axis)
    slice_shape_hw = _slice_shape(volume_shape_dhw, axis_index)
    patch_origin_hw, patch_size_hw = _normalize_patch_parameters(patch_origin_hw, patch_size_hw, slice_shape_hw)
    patch_height, patch_width = patch_size_hw

    render_state = _resolve_render_state(render_source)
    if render_state.means.numel() == 0:
        return torch.zeros((patch_height, patch_width), dtype=torch.float32, device=render_state.device)

    patch_grid_cache = patch_grid_cache if patch_grid_cache is not None else CTPatchGridCache()
    rr, cc = patch_grid_cache.get(patch_size_hw, render_state.device, render_state.dtype)
    query_points = _build_query_points_from_base(rr, cc, axis_index, slice_idx, patch_origin_hw, spacing_zyx)
    axis_cfg = _AXIS_RENDER_CONFIG[axis_index]
    slice_coord = query_points[0, axis_cfg["plane_axis"]]

    render_impl = render_impl or _render_ct_slice_patch_impl
    patch_values = render_impl(
        query_points,
        slice_coord,
        render_state.means,
        render_state.rotations,
        render_state.scales,
        render_state.opacity,
        render_state.radius,
        axis_cfg["plane_axis"],
        axis_cfg["dim_h"],
        axis_cfg["dim_w"],
        float(spacing_zyx[axis_index]),
        int(gaussians_per_chunk),
    )

    return patch_values.reshape(patch_height, patch_width).clamp_(0.0, 1.0)


def build_ct_patch_renderer(compile_renderer: bool = False) -> Callable[..., torch.Tensor]:
    eager_impl: Callable[..., torch.Tensor] = _render_ct_slice_patch_impl
    compiled_impl: Optional[Callable[..., torch.Tensor]] = None
    if compile_renderer and hasattr(torch, "compile") and _has_triton_support():
        try:
            compiled_impl = torch.compile(_render_ct_slice_patch_impl, dynamic=True, fullgraph=False)
        except Exception:
            compiled_impl = None

    compile_enabled = {"value": compiled_impl is not None}

    def _renderer(*args, **kwargs):
        if compile_enabled["value"]:
            try:
                kwargs.setdefault("render_impl", compiled_impl)
                return render_ct_slice_patch(*args, **kwargs)
            except Exception as exc:
                if not _is_compile_fallback_error(exc):
                    raise
                compile_enabled["value"] = False
                kwargs.pop("render_impl", None)

        kwargs.setdefault("render_impl", eager_impl)
        return render_ct_slice_patch(*args, **kwargs)

    return _renderer


def sample_gt_slice_patch(volume, axis, slice_idx, patch_origin_hw, patch_size_hw):
    axis_index = _normalize_axis(axis)
    slice_shape_hw = _slice_shape(volume.shape, axis_index)
    patch_origin_hw, patch_size_hw = _normalize_patch_parameters(patch_origin_hw, patch_size_hw, slice_shape_hw)
    origin_h, origin_w = patch_origin_hw
    patch_height, patch_width = patch_size_hw

    if axis_index == 0:
        patch = volume[int(slice_idx), origin_h : origin_h + patch_height, origin_w : origin_w + patch_width]
    elif axis_index == 1:
        patch = volume[origin_h : origin_h + patch_height, int(slice_idx), origin_w : origin_w + patch_width]
    else:
        patch = volume[origin_h : origin_h + patch_height, origin_w : origin_w + patch_width, int(slice_idx)]

    if isinstance(volume, torch.Tensor):
        return patch if torch.is_floating_point(patch) else patch.float()
    return patch.astype("float32", copy=False)
