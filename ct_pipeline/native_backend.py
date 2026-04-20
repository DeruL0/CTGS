from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

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
    prepare_ct_render_state,
)
from utils.rotation_utils import quaternion_to_matrix
from utils.ct_losses import (
    PointToPlaneCache,
    material_boundary_loss,
    point_to_plane_loss_from_cache,
    prepare_point_to_plane_cache,
    sample_volume_field,
)

try:
    from ct_native_backend import _C as _CT_NATIVE_C
except Exception as import_error:  # pragma: no cover
    _CT_NATIVE_C = None
    _CT_NATIVE_IMPORT_ERROR = import_error
else:  # pragma: no cover
    _CT_NATIVE_IMPORT_ERROR = None


_BACKEND_WARNING_EMITTED = False


@dataclass
class CTTrainingState:
    xyz: torch.Tensor
    rotation_quat: torch.Tensor
    rotation_mats: torch.Tensor
    raw_scaling: torch.Tensor
    scales: torch.Tensor
    opacity: torch.Tensor
    normals: torch.Tensor
    material_id: torch.Tensor
    region_type: torch.Tensor
    surface_mask: torch.Tensor
    surface_xyz: torch.Tensor
    surface_rotation_quat: torch.Tensor
    surface_rotation_mats: torch.Tensor
    surface_raw_scaling: torch.Tensor
    surface_opacity: torch.Tensor
    surface_normals: torch.Tensor
    surface_material_id: torch.Tensor
    support_extent: torch.Tensor | None
    spatial_grid: "CTSpatialGrid | None"
    render_state: CTRenderState


@dataclass
class CTSpatialGrid:
    world_min: torch.Tensor
    grid_dims: torch.Tensor
    cell_size: float
    cell_offsets: torch.Tensor
    cell_gaussian_ids: torch.Tensor
    support_extent: torch.Tensor
    truncation_sigma: float


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


def build_signed_field_native(material_mask: torch.Tensor, band_voxels: int) -> torch.Tensor:
    if not has_ct_native_backend() or not material_mask.is_cuda:
        raise RuntimeError("Native signed-field construction requires a CUDA mask tensor and an available extension.")
    return _CT_NATIVE_C.build_signed_field_cuda(
        material_mask.contiguous(),
        int(band_voxels),
    )


def build_signed_field_backend(backend: str, material_mask: torch.Tensor, band_voxels: int) -> torch.Tensor:
    if backend == "cuda":
        return build_signed_field_native(material_mask, band_voxels)
    raise RuntimeError("The active signed-field CT training path requires the CUDA backend.")


def _compute_support_extent(rotations: torch.Tensor, scales: torch.Tensor, truncation_sigma: float) -> torch.Tensor:
    return torch.sum(rotations.abs() * scales.unsqueeze(1), dim=2) * float(truncation_sigma)


def build_uniform_grid_native(cell_min: torch.Tensor, cell_max: torch.Tensor, grid_dims: torch.Tensor):
    if (
        not has_ct_native_backend()
        or not cell_min.is_cuda
        or not cell_max.is_cuda
        or not grid_dims.is_cuda
    ):
        raise RuntimeError("Native CT uniform grid build requires CUDA tensor inputs and an available extension.")
    return _CT_NATIVE_C.build_uniform_grid_cuda(
        cell_min.contiguous(),
        cell_max.contiguous(),
        grid_dims.contiguous(),
    )


def build_ct_spatial_grid(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    spacing_zyx,
    truncation_sigma: float = 4.0,
    grid_cell_voxels: int = 8,
) -> CTSpatialGrid | None:
    means = torch.as_tensor(means)
    if means.numel() == 0:
        return None
    if not means.is_cuda:
        raise RuntimeError("The active CT spatial grid path requires CUDA tensors.")

    cell_size = max(float(min(spacing_zyx)) * float(grid_cell_voxels), 1e-6)
    support_extent = _compute_support_extent(rotations, scales, truncation_sigma)
    world_min = torch.amin(means - support_extent, dim=0)
    world_max = torch.amax(means + support_extent, dim=0)
    grid_min = torch.floor(world_min / cell_size) * cell_size
    grid_max = torch.ceil(world_max / cell_size) * cell_size
    grid_dims = torch.ceil((grid_max - grid_min) / cell_size).to(dtype=torch.int32) + 1
    grid_dims = torch.clamp(grid_dims, min=1)

    cell_min = torch.floor((means - support_extent - grid_min) / cell_size).to(dtype=torch.int32)
    cell_max = torch.floor((means + support_extent - grid_min) / cell_size).to(dtype=torch.int32)
    upper = (grid_dims - 1).reshape(1, 3)
    cell_min = torch.minimum(torch.maximum(cell_min, torch.zeros_like(cell_min)), upper)
    cell_max = torch.minimum(torch.maximum(cell_max, torch.zeros_like(cell_max)), upper)

    cell_offsets, cell_gaussian_ids = build_uniform_grid_native(cell_min, cell_max, grid_dims)
    return CTSpatialGrid(
        world_min=grid_min.to(dtype=means.dtype),
        grid_dims=grid_dims,
        cell_size=float(cell_size),
        cell_offsets=cell_offsets,
        cell_gaussian_ids=cell_gaussian_ids,
        support_extent=support_extent,
        truncation_sigma=float(truncation_sigma),
    )


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


def prepare_ct_training_state(
    gaussians,
    spacing_zyx=None,
    truncation_sigma: float = 4.0,
    grid_cell_voxels: int = 8,
) -> CTTrainingState:
    render_state = prepare_ct_render_state(gaussians)
    xyz = gaussians.get_xyz
    rotation_quat = gaussians.get_rotation
    rotation_mats = render_state.rotations if render_state.rotations.shape[0] == xyz.shape[0] else quaternion_to_matrix(rotation_quat)
    raw_scaling = gaussians.get_raw_scaling
    scales = render_state.scales if render_state.scales.shape[0] == xyz.shape[0] else gaussians.get_scaling.clamp_min(1e-6)
    opacity = render_state.opacity if render_state.opacity.shape[0] == xyz.shape[0] else gaussians.get_opacity.squeeze(-1)
    normals = gaussians.get_normals()
    material_id = gaussians.get_material_id.reshape(-1)
    region_type = gaussians.get_region_type.reshape(-1)
    surface_mask = region_type == 0
    support_extent = None
    spatial_grid = None
    if spacing_zyx is not None and xyz.is_cuda and xyz.numel() > 0:
        spatial_grid = build_ct_spatial_grid(
            xyz,
            rotation_mats,
            scales,
            spacing_zyx=spacing_zyx,
            truncation_sigma=truncation_sigma,
            grid_cell_voxels=grid_cell_voxels,
        )
        if spatial_grid is not None:
            support_extent = spatial_grid.support_extent
            render_state.support_extent = support_extent
            render_state.spatial_grid = spatial_grid
            render_state.truncation_sigma = float(truncation_sigma)

    return CTTrainingState(
        xyz=xyz,
        rotation_quat=rotation_quat,
        rotation_mats=rotation_mats,
        raw_scaling=raw_scaling,
        scales=scales,
        opacity=opacity,
        normals=normals,
        material_id=material_id,
        region_type=region_type,
        surface_mask=surface_mask,
        surface_xyz=xyz[surface_mask],
        surface_rotation_quat=rotation_quat[surface_mask],
        surface_rotation_mats=rotation_mats[surface_mask],
        surface_raw_scaling=raw_scaling[surface_mask],
        surface_opacity=opacity[surface_mask],
        surface_normals=normals[surface_mask],
        surface_material_id=material_id[surface_mask],
        support_extent=support_extent,
        spatial_grid=spatial_grid,
        render_state=render_state,
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
        if _CT_NATIVE_C is None:
            raise RuntimeError("ct_native_backend is unavailable.")
        density, query_offsets, query_gaussian_ids = _CT_NATIVE_C.query_density_local_forward(
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
        grad_means, grad_rotations, grad_scales, grad_opacity = _CT_NATIVE_C.query_density_local_backward(
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
    if (
        not has_ct_native_backend()
        or not means.is_cuda
        or not query_points.is_cuda
    ):
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
        if _CT_NATIVE_C is None:
            raise RuntimeError("ct_native_backend is unavailable.")
        sampled_strength, sampled_normals = _CT_NATIVE_C.sample_boundary_field_forward(
            strength_volume.contiguous(),
            normal_volume.contiguous(),
            query_points.contiguous(),
            float(spacing_zyx[0]),
            float(spacing_zyx[1]),
            float(spacing_zyx[2]),
        )
        ctx.save_for_backward(strength_volume, normal_volume, query_points)
        ctx.spacing_zyx = tuple(float(value) for value in spacing_zyx)
        return sampled_strength, sampled_normals

    @staticmethod
    def backward(ctx, grad_strength: torch.Tensor, grad_normals: torch.Tensor):
        strength_volume, normal_volume, query_points = ctx.saved_tensors
        grad_points = _CT_NATIVE_C.sample_boundary_field_backward(
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
    if (
        not has_ct_native_backend()
        or not query_points.is_cuda
        or not strength_native.is_cuda
        or not normal_native.is_cuda
    ):
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


def _surface_thickness_loss_python_surface(
    raw_scaling: torch.Tensor,
    rotation_mats: torch.Tensor,
    normals: torch.Tensor,
    max_thickness: float,
):
    if raw_scaling.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=raw_scaling.device)
    scales = torch.exp(raw_scaling)
    work_normals = F.normalize(normals, dim=-1)
    local_normals = torch.einsum("nij,nj->ni", rotation_mats.transpose(1, 2), work_normals)
    variance_along_normal = torch.sum((local_normals * scales) ** 2, dim=-1)
    thickness = torch.sqrt(variance_along_normal.clamp_min(1e-8))
    return torch.relu(thickness - float(max_thickness)).mean()


class _NativeSurfaceThicknessLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_scaling, rotation_mats, normals, max_thickness: float):
        if _CT_NATIVE_C is None:
            raise RuntimeError("ct_native_backend is unavailable.")
        output = _CT_NATIVE_C.surface_thickness_loss_forward(
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
        grad_raw_scaling, grad_rotation_mats, grad_normals = _CT_NATIVE_C.surface_thickness_loss_backward(
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
    if (
        not has_ct_native_backend()
        or not raw_scaling.is_cuda
        or not rotation_mats.is_cuda
        or not normals.is_cuda
    ):
        raise RuntimeError("Native CT surface thickness loss requires CUDA tensor inputs and an available extension.")
    return _NativeSurfaceThicknessLossFunction.apply(
        raw_scaling,
        rotation_mats,
        normals,
        float(max_thickness),
    )


class _NativeMaterialBoundaryLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, material_ids, opacity, neighbor_index, target_opacity: float):
        if _CT_NATIVE_C is None:
            raise RuntimeError("ct_native_backend is unavailable.")
        output = _CT_NATIVE_C.material_boundary_loss_forward(
            xyz.contiguous(),
            material_ids.contiguous(),
            opacity.contiguous(),
            neighbor_index.contiguous(),
            float(target_opacity),
        )
        ctx.save_for_backward(xyz, material_ids, opacity, neighbor_index)
        ctx.target_opacity = float(target_opacity)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        xyz, material_ids, opacity, neighbor_index = ctx.saved_tensors
        grad_xyz, grad_opacity = _CT_NATIVE_C.material_boundary_loss_backward(
            xyz.contiguous(),
            material_ids.contiguous(),
            opacity.contiguous(),
            neighbor_index.contiguous(),
            float(ctx.target_opacity),
            grad_output.contiguous().to(dtype=xyz.dtype, device=xyz.device),
        )
        return grad_xyz, None, grad_opacity, None, None


def material_boundary_loss_native(
    xyz: torch.Tensor,
    material_ids: torch.Tensor,
    opacity: torch.Tensor,
    neighbor_index: torch.Tensor,
    target_opacity: float = 0.5,
):
    if (
        not has_ct_native_backend()
        or not xyz.is_cuda
        or not material_ids.is_cuda
        or not opacity.is_cuda
        or not neighbor_index.is_cuda
    ):
        raise RuntimeError("Native CT material boundary loss requires CUDA tensor inputs and an available extension.")
    return _NativeMaterialBoundaryLossFunction.apply(
        xyz,
        material_ids.reshape(-1).to(dtype=torch.long),
        opacity.reshape(-1),
        neighbor_index.to(dtype=torch.long),
        float(target_opacity),
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
        if _CT_NATIVE_C is None:
            raise RuntimeError("ct_native_backend is unavailable.")
        output, tile_offsets, tile_gaussian_ids = _CT_NATIVE_C.render_slice_patch_local_forward(
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
        grad_means, grad_rotations, grad_scales, grad_opacity = _CT_NATIVE_C.render_slice_patch_local_backward(
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
    gaussians_per_chunk: int = 2048,
    patch_grid_cache=None,
    slice_tile_size: int = 8,
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


def sample_boundary_field_backend(
    backend: str,
    strength_volume: torch.Tensor,
    normal_volume: torch.Tensor,
    query_points: torch.Tensor,
    spacing_zyx,
):
    if backend == "cuda":
        return sample_boundary_field_native(
            strength_volume,
            normal_volume,
            query_points,
            spacing_zyx,
        )
    sampled_strength = sample_volume_field(
        strength_volume,
        query_points,
        spacing_zyx,
    ).reshape(-1)
    sampled_normals = sample_volume_field(
        normal_volume,
        query_points,
        spacing_zyx,
    )
    return sampled_strength, sampled_normals


def sample_signed_field_backend(
    backend: str,
    signed_field_volume: torch.Tensor,
    signed_gradient_volume: torch.Tensor,
    query_points: torch.Tensor,
    spacing_zyx,
):
    return sample_boundary_field_backend(
        backend,
        signed_field_volume,
        signed_gradient_volume,
        query_points,
        spacing_zyx,
    )


def surface_thickness_loss_backend(
    backend: str,
    raw_scaling: torch.Tensor,
    rotation_mats: torch.Tensor,
    normals: torch.Tensor,
    max_thickness: float,
):
    if backend == "cuda":
        return surface_thickness_loss_native(
            raw_scaling,
            rotation_mats,
            normals,
            max_thickness=max_thickness,
        )
    return _surface_thickness_loss_python_surface(
        raw_scaling,
        rotation_mats,
        normals,
        max_thickness=max_thickness,
    )


def material_boundary_loss_backend(
    backend: str,
    xyz: torch.Tensor,
    material_ids: torch.Tensor,
    opacity: torch.Tensor,
    neighbor_index: torch.Tensor | None = None,
    target_opacity: float = 0.5,
):
    if neighbor_index is None:
        return torch.zeros((), dtype=xyz.dtype if xyz.numel() > 0 else torch.float32, device=xyz.device)
    if backend == "cuda":
        return material_boundary_loss_native(
            xyz,
            material_ids,
            opacity,
            neighbor_index=neighbor_index,
            target_opacity=target_opacity,
        )
    return material_boundary_loss(
        xyz,
        material_ids.reshape(-1, 1),
        opacity.reshape(-1, 1),
        neighbor_index=neighbor_index,
        target_opacity=target_opacity,
    )


def query_ct_density_backend(
    backend: str,
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    query_points: torch.Tensor,
    spatial_grid: CTSpatialGrid | None = None,
    support_extent: torch.Tensor | None = None,
):
    if backend == "cuda":
        return query_ct_density_native(
            means,
            rotations,
            scales,
            opacity,
            query_points,
            spatial_grid=spatial_grid,
            support_extent=support_extent,
        )
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
