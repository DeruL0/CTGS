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


def _has_native_symbols(*names: str) -> bool:
    if not has_ct_native_backend():
        return False
    try:
        native = get_ct_native_extension()
    except RuntimeError:
        return False
    return all(hasattr(native, name) for name in names)


def has_ct_native_qcut_density_query() -> bool:
    return _has_native_symbols(
        "query_density_qcut_local_forward",
        "query_density_qcut_local_backward",
    )


def has_ct_native_bulk_intensity_query() -> bool:
    return _has_native_symbols(
        "query_bulk_intensity_local_forward",
        "query_bulk_intensity_local_backward",
    )


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


class _NativeLocalQCutDensityQueryFunction(torch.autograd.Function):
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
        q_cut: float,
    ) -> torch.Tensor:
        native = get_ct_native_extension()
        density, query_offsets, query_gaussian_ids = native.query_density_qcut_local_forward(
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
            float(q_cut),
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
        grad_means, grad_rotations, grad_scales, grad_opacity = native.query_density_qcut_local_backward(
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
            None,
        )


class _NativeBulkIntensityQueryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        attenuation: torch.Tensor,
        center_sdf: torch.Tensor,
        center_normals: torch.Tensor,
        material_membership: torch.Tensor,
        support_extent: torch.Tensor,
        query_points: torch.Tensor,
        grid_world_min: torch.Tensor,
        grid_dims: torch.Tensor,
        cell_size: float,
        cell_offsets: torch.Tensor,
        cell_gaussian_ids: torch.Tensor,
        q_cut: float,
        tau: float,
        skip_depth: float,
        apply_gate: bool,
        has_membership: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        native = get_ct_native_extension()
        raw_bulk, density, query_offsets, query_gaussian_ids = native.query_bulk_intensity_local_forward(
            means.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacity.contiguous(),
            attenuation.contiguous(),
            center_sdf.contiguous(),
            center_normals.contiguous(),
            material_membership.contiguous(),
            support_extent.contiguous(),
            query_points.contiguous(),
            grid_world_min.contiguous(),
            grid_dims.contiguous(),
            float(cell_size),
            cell_offsets.contiguous(),
            cell_gaussian_ids.contiguous(),
            float(q_cut),
            float(tau),
            float(skip_depth),
            bool(apply_gate),
            bool(has_membership),
        )
        ctx.save_for_backward(
            means,
            rotations,
            scales,
            opacity,
            attenuation,
            center_sdf,
            center_normals,
            material_membership,
            query_points,
            query_offsets,
            query_gaussian_ids,
        )
        ctx.q_cut = float(q_cut)
        ctx.tau = float(tau)
        ctx.skip_depth = float(skip_depth)
        ctx.apply_gate = bool(apply_gate)
        ctx.has_membership = bool(has_membership)
        return raw_bulk, density

    @staticmethod
    def backward(ctx, grad_raw: torch.Tensor, grad_den: torch.Tensor):
        (
            means,
            rotations,
            scales,
            opacity,
            attenuation,
            center_sdf,
            center_normals,
            material_membership,
            query_points,
            query_offsets,
            query_gaussian_ids,
        ) = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[:5]
        native = get_ct_native_extension()
        grad_means, grad_rotations, grad_scales, grad_opacity, grad_attenuation = (
            native.query_bulk_intensity_local_backward(
                means.contiguous(),
                rotations.contiguous(),
                scales.contiguous(),
                opacity.contiguous(),
                attenuation.contiguous(),
                center_sdf.contiguous(),
                center_normals.contiguous(),
                material_membership.contiguous(),
                query_points.contiguous(),
                query_offsets.contiguous(),
                query_gaussian_ids.contiguous(),
                grad_raw.contiguous().to(dtype=means.dtype, device=means.device),
                grad_den.contiguous().to(dtype=means.dtype, device=means.device),
                float(ctx.q_cut),
                float(ctx.tau),
                float(ctx.skip_depth),
                bool(ctx.apply_gate),
                bool(ctx.has_membership),
            )
        )
        return (
            grad_means if needs_grad[0] else None,
            grad_rotations if needs_grad[1] else None,
            grad_scales if needs_grad[2] else None,
            grad_opacity if needs_grad[3] else None,
            grad_attenuation if needs_grad[4] else None,
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


def query_ct_density_qcut_native(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    query_points: torch.Tensor,
    *,
    spatial_grid: CTSpatialGrid,
    support_extent: torch.Tensor,
    q_cut: float,
) -> torch.Tensor:
    require_cuda_tensors(
        "Native CT q-cut density query requires CUDA tensor inputs and an available extension.",
        means,
        query_points,
        support_extent,
    )
    if not has_ct_native_qcut_density_query():
        raise RuntimeError("Native CT q-cut density query is unavailable in the loaded extension.")
    return _NativeLocalQCutDensityQueryFunction.apply(
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
        float(q_cut),
    )


def query_bulk_intensity_native(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    attenuation: torch.Tensor,
    center_sdf: torch.Tensor,
    center_normals: torch.Tensor,
    query_points: torch.Tensor,
    *,
    spatial_grid: CTSpatialGrid,
    support_extent: torch.Tensor,
    q_cut: float,
    tau: float,
    skip_depth: float,
    material_membership: torch.Tensor | None = None,
    apply_gate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    require_cuda_tensors(
        "Native CT bulk intensity query requires CUDA tensor inputs and an available extension.",
        means,
        rotations,
        scales,
        opacity,
        attenuation,
        center_sdf,
        center_normals,
        query_points,
        support_extent,
    )
    if not has_ct_native_bulk_intensity_query():
        raise RuntimeError("Native CT bulk intensity query is unavailable in the loaded extension.")
    has_membership = material_membership is not None
    if material_membership is None:
        material_membership = torch.empty((0,), dtype=means.dtype, device=means.device)
    else:
        material_membership = material_membership.to(device=means.device, dtype=means.dtype).reshape(-1)
    return _NativeBulkIntensityQueryFunction.apply(
        means,
        rotations,
        scales,
        opacity,
        attenuation,
        center_sdf,
        center_normals,
        material_membership,
        support_extent,
        query_points,
        spatial_grid.world_min,
        spatial_grid.grid_dims,
        float(spatial_grid.cell_size),
        spatial_grid.cell_offsets,
        spatial_grid.cell_gaussian_ids,
        float(q_cut),
        float(tau),
        float(skip_depth),
        bool(apply_gate),
        bool(has_membership),
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
