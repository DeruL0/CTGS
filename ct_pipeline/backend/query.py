from __future__ import annotations

import torch

from ct_pipeline.backend.core import (
    CTSpatialGrid,
    get_ct_native_extension,
    has_ct_native_backend,
    require_cuda_tensors,
)


def _native_volume_sampling_points(query_points: torch.Tensor, spacing_zyx) -> torch.Tensor:
    """Shift voxel-center world coordinates into the native extension's legacy frame."""

    spacing = torch.tensor(
        [
            0.5 * float(spacing_zyx[2]),
            0.5 * float(spacing_zyx[1]),
            0.5 * float(spacing_zyx[0]),
        ],
        dtype=query_points.dtype,
        device=query_points.device,
    )
    return query_points - spacing


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
        native = get_ct_native_extension()
        output = native.query_density_forward(
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
        native = get_ct_native_extension()
        grad_means, grad_rotations, grad_scales, grad_opacity = native.query_density_backward(
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


class _NativeLocalDensityQueryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        support_extent: torch.Tensor,
        query_points: torch.Tensor,
        grid_world_min: torch.Tensor,
        grid_dims: torch.Tensor,
        cell_size: float,
        cell_offsets: torch.Tensor,
        cell_gaussian_ids: torch.Tensor,
    ) -> torch.Tensor:
        native = get_ct_native_extension()
        density, query_offsets, query_gaussian_ids = native.query_density_local_forward(
            means.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacity.contiguous(),
            support_extent.contiguous(),
            query_points.contiguous(),
            grid_world_min.contiguous(),
            grid_dims.contiguous(),
            float(cell_size),
            cell_offsets.contiguous(),
            cell_gaussian_ids.contiguous(),
        )
        ctx.save_for_backward(
            means,
            rotations,
            scales,
            opacity,
            query_points,
            query_offsets,
            query_gaussian_ids,
        )
        return density

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        means, rotations, scales, opacity, query_points, query_offsets, query_gaussian_ids = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[:4]
        grad_output = grad_output.contiguous().to(dtype=means.dtype, device=means.device)
        native = get_ct_native_extension()
        grad_means, grad_rotations, grad_scales, grad_opacity = native.query_density_local_backward(
            means.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacity.contiguous(),
            query_points.contiguous(),
            query_offsets.contiguous(),
            query_gaussian_ids.contiguous(),
            grad_output,
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


def query_ct_density_native(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    query_points: torch.Tensor,
    spatial_grid: CTSpatialGrid | None = None,
    support_extent: torch.Tensor | None = None,
) -> torch.Tensor:
    require_cuda_tensors(
        "Native CT density query requires a CUDA tensor input and an available extension.",
        means,
        query_points,
    )
    if not has_ct_native_backend():
        raise RuntimeError("Native CT density query requires a CUDA tensor input and an available extension.")
    if spatial_grid is not None and support_extent is not None:
        return _NativeLocalDensityQueryFunction.apply(
            means,
            rotations,
            scales,
            opacity,
            support_extent,
            query_points,
            spatial_grid.world_min,
            spatial_grid.grid_dims,
            float(spatial_grid.cell_size),
            spatial_grid.cell_offsets,
            spatial_grid.cell_gaussian_ids,
        )
    return _NativeDensityQueryFunction.apply(
        means,
        rotations,
        scales,
        opacity,
        query_points,
    )


def build_signed_field_native(material_mask: torch.Tensor, band_voxels: int) -> torch.Tensor:
    require_cuda_tensors(
        "Native signed-field construction requires a CUDA mask tensor and an available extension.",
        material_mask,
    )
    native = get_ct_native_extension()
    return native.build_signed_field_cuda(material_mask.contiguous(), int(band_voxels))


def _normalize_boundary_volumes(strength_volume: torch.Tensor, normal_volume: torch.Tensor):
    if strength_volume.ndim == 5:
        strength_native = strength_volume.squeeze(0).squeeze(0)
    else:
        strength_native = strength_volume
    if normal_volume.ndim == 5:
        normal_native = normal_volume.squeeze(0).permute(1, 2, 3, 0).contiguous()
    elif normal_volume.ndim == 4 and normal_volume.shape[-1] == 3:
        normal_native = normal_volume
    elif normal_volume.ndim == 4 and normal_volume.shape[0] == 3:
        normal_native = normal_volume.permute(1, 2, 3, 0).contiguous()
    else:
        raise ValueError("normal_volume must have shape (1, 3, D, H, W), (3, D, H, W), or (D, H, W, 3).")
    return strength_native.contiguous(), normal_native.contiguous()


class _NativeSampleBoundaryFieldFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        strength_volume: torch.Tensor,
        normal_volume: torch.Tensor,
        query_points: torch.Tensor,
        spacing_zyx,
    ):
        native = get_ct_native_extension()
        native_query_points = _native_volume_sampling_points(query_points, spacing_zyx)
        sampled_strength, sampled_normals = native.sample_boundary_field_forward(
            strength_volume.contiguous(),
            normal_volume.contiguous(),
            native_query_points.contiguous(),
            float(spacing_zyx[0]),
            float(spacing_zyx[1]),
            float(spacing_zyx[2]),
        )
        ctx.save_for_backward(strength_volume, normal_volume, native_query_points)
        ctx.spacing_zyx = tuple(float(value) for value in spacing_zyx)
        return sampled_strength, sampled_normals

    @staticmethod
    def backward(ctx, grad_strength: torch.Tensor, grad_normals: torch.Tensor):
        strength_volume, normal_volume, query_points = ctx.saved_tensors
        native = get_ct_native_extension()
        grad_points = native.sample_boundary_field_backward(
            strength_volume.contiguous(),
            normal_volume.contiguous(),
            query_points.contiguous(),
            grad_strength.contiguous().to(dtype=query_points.dtype, device=query_points.device),
            grad_normals.contiguous().to(dtype=query_points.dtype, device=query_points.device),
            float(ctx.spacing_zyx[0]),
            float(ctx.spacing_zyx[1]),
            float(ctx.spacing_zyx[2]),
        )
        return None, None, grad_points, None


def sample_boundary_field_native(
    strength_volume: torch.Tensor,
    normal_volume: torch.Tensor,
    query_points: torch.Tensor,
    spacing_zyx,
):
    strength_native, normal_native = _normalize_boundary_volumes(strength_volume, normal_volume)
    require_cuda_tensors(
        "Native CT boundary field sampling requires CUDA tensor inputs and an available extension.",
        query_points,
        strength_native,
        normal_native,
    )
    if not has_ct_native_backend():
        raise RuntimeError("Native CT boundary field sampling requires CUDA tensor inputs and an available extension.")
    if query_points.numel() == 0:
        dtype = query_points.dtype if query_points.numel() > 0 else strength_native.dtype
        device = query_points.device
        return (
            torch.empty((0,), dtype=dtype, device=device),
            torch.empty((0, 3), dtype=dtype, device=device),
        )
    return _NativeSampleBoundaryFieldFunction.apply(
        strength_native,
        normal_native,
        query_points,
        tuple(float(value) for value in spacing_zyx),
    )


class _NativeSurfaceThicknessLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_scaling, rotation_mats, normals, max_thickness: float):
        native = get_ct_native_extension()
        output = native.surface_thickness_loss_forward(
            raw_scaling.contiguous(),
            rotation_mats.contiguous(),
            normals.contiguous(),
            float(max_thickness),
        )
        ctx.save_for_backward(raw_scaling, rotation_mats, normals)
        ctx.max_thickness = float(max_thickness)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raw_scaling, rotation_mats, normals = ctx.saved_tensors
        native = get_ct_native_extension()
        grad_raw_scaling, grad_rotation_mats, grad_normals = native.surface_thickness_loss_backward(
            raw_scaling.contiguous(),
            rotation_mats.contiguous(),
            normals.contiguous(),
            float(ctx.max_thickness),
            grad_output.contiguous().to(dtype=raw_scaling.dtype, device=raw_scaling.device),
        )
        return grad_raw_scaling, grad_rotation_mats, grad_normals, None


def surface_thickness_loss_native(
    raw_scaling: torch.Tensor,
    rotation_mats: torch.Tensor,
    normals: torch.Tensor,
    max_thickness: float,
):
    require_cuda_tensors(
        "Native CT surface thickness loss requires CUDA tensor inputs and an available extension.",
        raw_scaling,
        rotation_mats,
        normals,
    )
    if not has_ct_native_backend():
        raise RuntimeError("Native CT surface thickness loss requires CUDA tensor inputs and an available extension.")
    return _NativeSurfaceThicknessLossFunction.apply(
        raw_scaling,
        rotation_mats,
        normals,
        float(max_thickness),
    )
