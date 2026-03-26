from __future__ import annotations

import warnings
from typing import Callable

import torch

from ct_pipeline.ct_slice_renderer import (
    CTRenderState,
    _AXIS_MAP,
    _AXIS_RENDER_CONFIG,
    _build_query_points_from_base,
    _normalize_patch_parameters,
    _resolve_render_state,
    _render_ct_slice_patch_impl,
    _slice_shape,
    build_ct_patch_renderer,
)
from utils.ct_losses import (
    PointToPlaneCache,
    point_to_plane_loss_from_cache,
    prepare_point_to_plane_cache,
)

try:
    from ct_native_backend import _C as _CT_NATIVE_C
except Exception as import_error:  # pragma: no cover
    _CT_NATIVE_C = None
    _CT_NATIVE_IMPORT_ERROR = import_error
else:  # pragma: no cover
    _CT_NATIVE_IMPORT_ERROR = None


_BACKEND_WARNING_EMITTED = False


def _normalize_axis(axis) -> int:
    if axis not in _AXIS_MAP:
        raise ValueError("axis must be one of {'z', 'y', 'x', 0, 1, 2}.")
    return _AXIS_MAP[axis]


def _primitive_mask(primitive_type, count: int, device: torch.device) -> torch.Tensor:
    if primitive_type is None:
        return torch.ones((count,), dtype=torch.bool, device=device)
    primitive_type = torch.as_tensor(primitive_type, device=device).reshape(count, -1)
    values = primitive_type[:, 0]
    if values.dtype == torch.bool:
        return values
    if values.is_floating_point():
        if torch.any((values < 0.0) | (values > 1.0)):
            return torch.sigmoid(values) >= 0.5
        return values >= 0.5
    return values != 0


def has_ct_native_backend() -> bool:
    return _CT_NATIVE_C is not None


def get_ct_native_backend_error() -> Exception | None:
    return _CT_NATIVE_IMPORT_ERROR


def resolve_ct_backend(backend: str) -> str:
    global _BACKEND_WARNING_EMITTED

    normalized = str(backend).lower()
    if normalized not in {"auto", "cuda", "python"}:
        raise ValueError("--ct_backend must be one of {'auto', 'cuda', 'python'}.")

    if normalized == "python":
        return "python"

    cuda_ready = torch.cuda.is_available()
    native_ready = has_ct_native_backend()
    if normalized == "cuda":
        if not cuda_ready:
            raise RuntimeError("CT native CUDA backend requested, but CUDA is unavailable.")
        if not native_ready:
            raise RuntimeError(f"CT native CUDA backend requested, but ct_native_backend is unavailable: {_CT_NATIVE_IMPORT_ERROR!r}")
        return "cuda"

    if cuda_ready and native_ready:
        return "cuda"

    if not _BACKEND_WARNING_EMITTED:
        message = "Falling back to Python CT backend."
        if not cuda_ready:
            message += " CUDA is unavailable."
        elif _CT_NATIVE_IMPORT_ERROR is not None:
            message += f" Native extension import failed: {_CT_NATIVE_IMPORT_ERROR!r}."
        else:
            message += " Native extension is unavailable."
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _BACKEND_WARNING_EMITTED = True
    return "python"


def build_ct_backend_patch_renderer(backend: str, compile_renderer: bool = False) -> Callable[..., torch.Tensor]:
    if backend == "cuda" and has_ct_native_backend():
        return render_ct_slice_patch_native
    return build_ct_patch_renderer(compile_renderer=compile_renderer)


class _NativeDensityQueryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        if _CT_NATIVE_C is None:
            raise RuntimeError("ct_native_backend is unavailable.")
        output = _CT_NATIVE_C.query_density_forward(
            means.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacity.contiguous(),
            query_points.contiguous(),
        )
        ctx.save_for_backward(means, rotations, scales, opacity, query_points)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        means, rotations, scales, opacity, query_points = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[:4]
        grad_output = grad_output.contiguous().to(dtype=means.dtype, device=means.device)
        grad_means, grad_rotations, grad_scales, grad_opacity = _CT_NATIVE_C.query_density_backward(
            means.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacity.contiguous(),
            query_points.contiguous(),
            grad_output,
        )
        return (
            grad_means if needs_grad[0] else None,
            grad_rotations if needs_grad[1] else None,
            grad_scales if needs_grad[2] else None,
            grad_opacity if needs_grad[3] else None,
            None,
        )


def query_ct_density_native(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    query_points: torch.Tensor,
) -> torch.Tensor:
    if (
        not has_ct_native_backend()
        or not means.is_cuda
        or not query_points.is_cuda
    ):
        raise RuntimeError("Native CT density query requires a CUDA tensor input and an available extension.")
    return _NativeDensityQueryFunction.apply(
        means,
        rotations,
        scales,
        opacity,
        query_points,
    )


def build_neighbor_index_native(xyz, k: int, tile_size: int = 2048) -> torch.Tensor:
    xyz = torch.as_tensor(xyz)
    if (
        not has_ct_native_backend()
        or not xyz.is_cuda
    ):
        raise RuntimeError("Native CT neighbor index build requires a CUDA tensor input and an available extension.")
    return _CT_NATIVE_C.build_neighbor_index_cuda(
        xyz.contiguous(),
        int(k),
        int(tile_size),
    )


def _render_ct_slice_patch_python_unclamped(
    render_state: CTRenderState,
    axis_index: int,
    slice_idx: int,
    patch_origin_hw,
    patch_size_hw,
    spacing_zyx,
    volume_shape_dhw,
    gaussians_per_chunk: int,
) -> torch.Tensor:
    slice_shape_hw = _slice_shape(volume_shape_dhw, axis_index)
    patch_origin_hw, patch_size_hw = _normalize_patch_parameters(patch_origin_hw, patch_size_hw, slice_shape_hw)
    patch_height, patch_width = patch_size_hw
    if render_state.means.numel() == 0:
        return torch.zeros((patch_height, patch_width), dtype=torch.float32, device=render_state.device)

    rr = torch.arange(patch_height, device=render_state.device, dtype=render_state.dtype)
    cc = torch.arange(patch_width, device=render_state.device, dtype=render_state.dtype)
    rr, cc = torch.meshgrid(rr, cc, indexing="ij")
    query_points = _build_query_points_from_base(rr, cc, axis_index, slice_idx, patch_origin_hw, spacing_zyx)
    axis_cfg = _AXIS_RENDER_CONFIG[axis_index]
    slice_coord = query_points[0, axis_cfg["plane_axis"]]
    patch_values = _render_ct_slice_patch_impl(
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
    return patch_values.reshape(patch_height, patch_width)


class _NativeSlicePatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        axis_index: int,
        slice_idx: int,
        origin_h: int,
        origin_w: int,
        patch_h: int,
        patch_w: int,
        spacing_zyx,
        volume_shape_dhw,
        gaussians_per_chunk: int,
    ) -> torch.Tensor:
        if _CT_NATIVE_C is None:
            raise RuntimeError("ct_native_backend is unavailable.")

        output = _CT_NATIVE_C.render_slice_patch_forward(
            means.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacity.contiguous(),
            int(axis_index),
            int(slice_idx),
            int(origin_h),
            int(origin_w),
            int(patch_h),
            int(patch_w),
            float(spacing_zyx[0]),
            float(spacing_zyx[1]),
            float(spacing_zyx[2]),
        )

        ctx.save_for_backward(means, rotations, scales, opacity)
        ctx.axis_index = int(axis_index)
        ctx.slice_idx = int(slice_idx)
        ctx.origin_h = int(origin_h)
        ctx.origin_w = int(origin_w)
        ctx.patch_h = int(patch_h)
        ctx.patch_w = int(patch_w)
        ctx.spacing_zyx = tuple(float(value) for value in spacing_zyx)
        ctx.volume_shape_dhw = tuple(int(value) for value in volume_shape_dhw)
        ctx.gaussians_per_chunk = int(gaussians_per_chunk)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        means, rotations, scales, opacity = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[:4]
        grad_output = grad_output.contiguous().to(dtype=means.dtype, device=means.device)
        grad_means, grad_rotations, grad_scales, grad_opacity = _CT_NATIVE_C.render_slice_patch_backward(
            means.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacity.contiguous(),
            grad_output,
            ctx.axis_index,
            ctx.slice_idx,
            ctx.origin_h,
            ctx.origin_w,
            ctx.patch_h,
            ctx.patch_w,
            float(ctx.spacing_zyx[0]),
            float(ctx.spacing_zyx[1]),
            float(ctx.spacing_zyx[2]),
        )

        return (
            grad_means if needs_grad[0] else None,
            grad_rotations if needs_grad[1] else None,
            grad_scales if needs_grad[2] else None,
            grad_opacity if needs_grad[3] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def render_ct_slice_patch_native(
    render_source,
    axis,
    slice_idx,
    patch_origin_hw,
    patch_size_hw,
    spacing_zyx,
    volume_shape_dhw,
    gaussians_per_chunk: int = 2048,
    patch_grid_cache=None,
):
    del patch_grid_cache
    axis_index = _normalize_axis(axis)
    slice_shape_hw = _slice_shape(volume_shape_dhw, axis_index)
    patch_origin_hw, patch_size_hw = _normalize_patch_parameters(patch_origin_hw, patch_size_hw, slice_shape_hw)
    patch_height, patch_width = patch_size_hw

    render_state = _resolve_render_state(render_source)
    if render_state.means.numel() == 0:
        return torch.zeros((patch_height, patch_width), dtype=torch.float32, device=render_state.device)
    if not has_ct_native_backend() or not render_state.means.is_cuda:
        return _render_ct_slice_patch_python_unclamped(
            render_state,
            axis_index,
            slice_idx,
            patch_origin_hw,
            patch_size_hw,
            spacing_zyx,
            volume_shape_dhw,
            gaussians_per_chunk,
        ).clamp_(0.0, 1.0)

    patch = _NativeSlicePatchFunction.apply(
        render_state.means,
        render_state.rotations,
        render_state.scales,
        render_state.opacity,
        axis_index,
        int(slice_idx),
        int(patch_origin_hw[0]),
        int(patch_origin_hw[1]),
        int(patch_height),
        int(patch_width),
        tuple(float(value) for value in spacing_zyx),
        tuple(int(value) for value in volume_shape_dhw),
        int(gaussians_per_chunk),
    )
    return patch.clamp_(0.0, 1.0)


def prepare_point_to_plane_cache_native(xyz, normals, planarity, material_ids=None, primitive_type=None, neighbor_index=None, k: int = 8) -> PointToPlaneCache:
    xyz = torch.as_tensor(xyz)
    if (
        not has_ct_native_backend()
        or not xyz.is_cuda
        or neighbor_index is None
    ):
        return prepare_point_to_plane_cache(
            xyz,
            normals,
            planarity,
            material_ids=material_ids,
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
            k=k,
        )

    normals = torch.as_tensor(normals, device=xyz.device, dtype=xyz.dtype)
    planarity = torch.as_tensor(planarity, device=xyz.device, dtype=xyz.dtype).reshape(-1)
    planar_mask = _primitive_mask(primitive_type, xyz.shape[0], xyz.device)
    material_tensor = (
        torch.full((xyz.shape[0],), -1, dtype=torch.long, device=xyz.device)
        if material_ids is None
        else torch.as_tensor(material_ids, device=xyz.device, dtype=torch.long).reshape(-1)
    )
    neighbor_tensor = torch.as_tensor(neighbor_index, device=xyz.device, dtype=torch.long)
    active_indices, centroids, fitted_normals, weights = _CT_NATIVE_C.build_plane_targets_cuda(
        xyz.detach().contiguous(),
        normals.detach().contiguous(),
        planarity.detach().contiguous(),
        material_tensor.detach().contiguous(),
        planar_mask.detach().contiguous(),
        neighbor_tensor.detach().contiguous(),
    )
    return PointToPlaneCache(
        active_indices=active_indices,
        centroids=centroids,
        fitted_normals=fitted_normals,
        weights=weights,
    )


def prepare_point_to_plane_cache_backend(backend: str, xyz, normals, planarity, material_ids=None, primitive_type=None, neighbor_index=None, k: int = 8) -> PointToPlaneCache:
    if backend == "cuda":
        return prepare_point_to_plane_cache_native(
            xyz,
            normals,
            planarity,
            material_ids=material_ids,
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
            k=k,
        )
    return prepare_point_to_plane_cache(
        xyz,
        normals,
        planarity,
        material_ids=material_ids,
        primitive_type=primitive_type,
        neighbor_index=neighbor_index,
        k=k,
    )


def point_to_plane_loss_native(xyz, cache: PointToPlaneCache):
    xyz = torch.as_tensor(xyz)
    if cache.is_empty:
        return torch.zeros((), dtype=xyz.dtype if xyz.numel() > 0 else torch.float32, device=xyz.device)
    if not has_ct_native_backend() or not xyz.is_cuda:
        return point_to_plane_loss_from_cache(xyz, cache)
    return _CT_NATIVE_C.point_to_plane_loss_cuda(
        xyz,
        cache.active_indices,
        cache.centroids,
        cache.fitted_normals,
        cache.weights,
    )


def point_to_plane_loss_backend(backend: str, xyz, cache: PointToPlaneCache):
    if backend == "cuda":
        return point_to_plane_loss_native(xyz, cache)
    return point_to_plane_loss_from_cache(xyz, cache)


def query_ct_density_backend(
    backend: str,
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    query_points: torch.Tensor,
):
    if backend == "cuda":
        return query_ct_density_native(means, rotations, scales, opacity, query_points)
    from ct_pipeline.field_query import query_ct_density_python

    return query_ct_density_python(
        means,
        rotations,
        scales,
        opacity,
        query_points,
    )


def build_neighbor_index_backend(backend: str, xyz, k: int, tile_size: int = 2048):
    if backend == "cuda":
        return build_neighbor_index_native(xyz, k=k, tile_size=tile_size)
    return None
