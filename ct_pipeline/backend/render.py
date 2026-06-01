from __future__ import annotations

import torch

from ct_pipeline.rendering.slices import (
    CTRenderState,
    _normalize_patch_parameters,
    _normalize_axis,
    _resolve_render_state,
    _slice_shape,
)
from ct_pipeline.backend.core import get_ct_native_extension, require_ct_native_backend


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
    ) -> torch.Tensor:
        native = get_ct_native_extension()
        output = native.render_slice_patch_forward(
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
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        means, rotations, scales, opacity = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[:4]
        grad_output = grad_output.contiguous().to(dtype=means.dtype, device=means.device)
        native = get_ct_native_extension()
        grad_means, grad_rotations, grad_scales, grad_opacity = native.render_slice_patch_backward(
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
        )


class _NativeLocalSlicePatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        support_extent: torch.Tensor,
        grid_world_min: torch.Tensor,
        grid_dims: torch.Tensor,
        cell_size: float,
        cell_offsets: torch.Tensor,
        cell_gaussian_ids: torch.Tensor,
        axis_index: int,
        slice_idx: int,
        origin_h: int,
        origin_w: int,
        patch_h: int,
        patch_w: int,
        spacing_zyx,
        tile_size: int,
    ) -> torch.Tensor:
        native = get_ct_native_extension()
        output, tile_offsets, tile_gaussian_ids = native.render_slice_patch_local_forward(
            means.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacity.contiguous(),
            support_extent.contiguous(),
            grid_world_min.contiguous(),
            grid_dims.contiguous(),
            float(cell_size),
            cell_offsets.contiguous(),
            cell_gaussian_ids.contiguous(),
            int(axis_index),
            int(slice_idx),
            int(origin_h),
            int(origin_w),
            int(patch_h),
            int(patch_w),
            float(spacing_zyx[0]),
            float(spacing_zyx[1]),
            float(spacing_zyx[2]),
            int(tile_size),
        )
        ctx.save_for_backward(means, rotations, scales, opacity, tile_offsets, tile_gaussian_ids)
        ctx.axis_index = int(axis_index)
        ctx.slice_idx = int(slice_idx)
        ctx.origin_h = int(origin_h)
        ctx.origin_w = int(origin_w)
        ctx.patch_h = int(patch_h)
        ctx.patch_w = int(patch_w)
        ctx.spacing_zyx = tuple(float(value) for value in spacing_zyx)
        ctx.tile_size = int(tile_size)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        means, rotations, scales, opacity, tile_offsets, tile_gaussian_ids = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[:4]
        grad_output = grad_output.contiguous().to(dtype=means.dtype, device=means.device)
        native = get_ct_native_extension()
        grad_means, grad_rotations, grad_scales, grad_opacity = native.render_slice_patch_local_backward(
            means.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacity.contiguous(),
            tile_offsets.contiguous(),
            tile_gaussian_ids.contiguous(),
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
            int(ctx.tile_size),
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
    slice_tile_size: int = 8,
):
    """Render occupancy/density slices only.

    CTGS-vFinal intensity previews must use query_ct_fields_unified and the
    bulk A_b readout. This native path intentionally remains a geometry
    renderer, not a second CT intensity formula.
    """
    axis_index = _normalize_axis(axis)
    slice_shape_hw = _slice_shape(volume_shape_dhw, axis_index)
    patch_origin_hw, patch_size_hw = _normalize_patch_parameters(patch_origin_hw, patch_size_hw, slice_shape_hw)
    patch_height, patch_width = patch_size_hw

    render_state = _resolve_render_state(render_source)
    require_ct_native_backend()
    if not render_state.means.is_cuda:
        raise RuntimeError("Native CT slice rendering requires CUDA tensor inputs.")
    if render_state.means.numel() == 0:
        return torch.zeros((patch_height, patch_width), dtype=torch.float32, device=render_state.device)

    if render_state.spatial_grid is not None and render_state.support_extent is not None:
        patch = _NativeLocalSlicePatchFunction.apply(
            render_state.means,
            render_state.rotations,
            render_state.scales,
            render_state.opacity,
            render_state.support_extent,
            render_state.spatial_grid.world_min,
            render_state.spatial_grid.grid_dims,
            float(render_state.spatial_grid.cell_size),
            render_state.spatial_grid.cell_offsets,
            render_state.spatial_grid.cell_gaussian_ids,
            axis_index,
            int(slice_idx),
            int(patch_origin_hw[0]),
            int(patch_origin_hw[1]),
            int(patch_height),
            int(patch_width),
            tuple(float(value) for value in spacing_zyx),
            int(slice_tile_size),
        )
        return patch.clamp_(0.0, 1.0)

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
    )
    return patch.clamp_(0.0, 1.0)
