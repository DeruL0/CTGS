from __future__ import annotations

from dataclasses import dataclass

import torch

from ct_pipeline.rendering.slices import CTRenderState, prepare_ct_render_state
from utils.rotation_utils import quaternion_to_matrix

try:
    from ct_native_backend import _C as _CT_NATIVE_C
except Exception as import_error:  # pragma: no cover
    _CT_NATIVE_C = None
    _CT_NATIVE_IMPORT_ERROR = import_error
else:  # pragma: no cover
    _CT_NATIVE_IMPORT_ERROR = None

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
    bulk_mask: torch.Tensor
    surface_xyz: torch.Tensor
    surface_rotation_quat: torch.Tensor
    surface_rotation_mats: torch.Tensor
    surface_raw_scaling: torch.Tensor
    surface_scales: torch.Tensor
    surface_opacity: torch.Tensor
    surface_attenuation: torch.Tensor | None
    surface_normals: torch.Tensor
    surface_material_id: torch.Tensor
    bulk_xyz: torch.Tensor
    bulk_rotation_mats: torch.Tensor
    bulk_scales: torch.Tensor
    bulk_opacity: torch.Tensor
    bulk_attenuation: torch.Tensor | None
    bulk_sigma: torch.Tensor | None
    bulk_center_sdf: torch.Tensor | None
    bulk_center_normals: torch.Tensor | None
    bulk_center_curvature: torch.Tensor | None
    support_extent: torch.Tensor | None
    spatial_grid: "CTSpatialGrid | None"
    surface_support_extent: torch.Tensor | None
    surface_spatial_grid: "CTSpatialGrid | None"
    bulk_support_extent: torch.Tensor | None
    bulk_spatial_grid: "CTSpatialGrid | None"
    bulk_query_truncation_sigma: float
    bulk_query_q_support: float
    render_state: CTRenderState
    ct_value: torch.Tensor | None = None
    surface_ct_value: torch.Tensor | None = None
    bulk_ct_value: torch.Tensor | None = None


@dataclass
class CTSpatialGrid:
    world_min: torch.Tensor
    grid_dims: torch.Tensor
    cell_size: float
    cell_offsets: torch.Tensor
    cell_gaussian_ids: torch.Tensor
    support_extent: torch.Tensor
    truncation_sigma: float
def has_ct_native_backend() -> bool:
    return _CT_NATIVE_C is not None


def get_ct_native_backend_error() -> Exception | None:
    return _CT_NATIVE_IMPORT_ERROR


def get_ct_native_extension():
    if _CT_NATIVE_C is None:
        raise RuntimeError("ct_native_backend is unavailable.")
    return _CT_NATIVE_C


def require_ct_native_backend() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CT training requires CUDA in the current implementation.")
    if not has_ct_native_backend():
        raise RuntimeError(f"CT native CUDA backend is unavailable: {_CT_NATIVE_IMPORT_ERROR!r}")


def require_cuda_tensors(message: str, *tensors) -> None:
    for tensor in tensors:
        if tensor is None:
            continue
        if not torch.is_tensor(tensor) or not tensor.is_cuda:
            raise RuntimeError(message)


def prepare_ct_training_state(
    gaussians,
    spacing_zyx=None,
    truncation_sigma: float = 4.0,
    bulk_truncation_sigma: float | None = None,
    grid_cell_voxels: int = 8,
    build_full_grid: bool = True,
    build_region_grids: bool = True,
    signed_distance_field=None,
    curvature_field=None,
) -> CTTrainingState:
    from ct_pipeline.training.losses import sample_volume_field, sample_sdf_normals

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
    bulk_mask = region_type == 1
    ct_value = gaussians.get_ct_value.reshape(-1) if gaussians.get_ct_value.numel() > 0 else None
    bulk_attenuation = gaussians.get_attenuation.reshape(-1) if gaussians.get_attenuation.numel() > 0 else None
    # Apply bounded bulk offset if present (adaptive bulk mode)
    bulk_offset = getattr(gaussians, "_bulk_offset", None)
    if bulk_offset is not None and bulk_offset.numel() > 0 and bulk_mask.sum() > 0:
        max_offset = float(getattr(gaussians, "_bulk_max_offset", 1e9))
        bounded = max_offset * torch.tanh(bulk_offset.to(device=xyz.device, dtype=xyz.dtype))
        xyz = xyz.clone()
        xyz[bulk_mask] = xyz[bulk_mask] + bounded
    bulk_sigma = None
    bulk_center_sdf = None
    bulk_center_normals = None
    bulk_center_curvature = None
    support_extent = None
    spatial_grid = None
    surface_support_extent = None
    surface_spatial_grid = None
    bulk_support_extent = None
    bulk_spatial_grid = None
    if spacing_zyx is not None and xyz.is_cuda and xyz.numel() > 0:
        from ct_pipeline.backend.grid import build_ct_spatial_grid

        bulk_truncation_sigma = float(truncation_sigma if bulk_truncation_sigma is None else bulk_truncation_sigma)
        if build_full_grid:
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
        if build_region_grids and torch.any(surface_mask):
            surface_spatial_grid = build_ct_spatial_grid(
                xyz[surface_mask],
                rotation_mats[surface_mask],
                scales[surface_mask],
                spacing_zyx=spacing_zyx,
                truncation_sigma=truncation_sigma,
                grid_cell_voxels=grid_cell_voxels,
            )
            if surface_spatial_grid is not None:
                surface_support_extent = surface_spatial_grid.support_extent
        if build_region_grids and torch.any(bulk_mask):
            bulk_spatial_grid = build_ct_spatial_grid(
                xyz[bulk_mask],
                rotation_mats[bulk_mask],
                scales[bulk_mask],
                spacing_zyx=spacing_zyx,
                truncation_sigma=bulk_truncation_sigma,
                grid_cell_voxels=grid_cell_voxels,
            )
            if bulk_spatial_grid is not None:
                bulk_support_extent = bulk_spatial_grid.support_extent

    if torch.any(bulk_mask):
        bulk_scale_values = scales[bulk_mask]
        bulk_sigma = bulk_scale_values.mean(dim=1).clamp_min(1e-6)
        if signed_distance_field is not None:
            bulk_xyz = xyz[bulk_mask]
            bulk_center_sdf = sample_volume_field(
                signed_distance_field["signed_distance"],
                bulk_xyz,
                signed_distance_field["spacing_zyx"],
            ).reshape(-1).to(device=bulk_xyz.device, dtype=torch.float32)
            normal_volume = None if signed_distance_field is None else signed_distance_field.get("sdf_normal")
            if normal_volume is not None:
                bulk_center_normals = sample_volume_field(
                    normal_volume,
                    bulk_xyz,
                    signed_distance_field["spacing_zyx"],
                ).to(device=bulk_xyz.device, dtype=bulk_xyz.dtype)
            else:
                bulk_center_normals = sample_sdf_normals(
                    signed_distance_field["signed_distance"],
                    bulk_xyz,
                    signed_distance_field["spacing_zyx"],
                ).to(device=bulk_xyz.device, dtype=bulk_xyz.dtype)
            bulk_center_normals = torch.nn.functional.normalize(bulk_center_normals, dim=-1, eps=1e-6)
        if curvature_field is not None:
            curvature_volume = curvature_field.get("curvature") if isinstance(curvature_field, dict) else curvature_field
            curvature_spacing = curvature_field.get("spacing_zyx", spacing_zyx) if isinstance(curvature_field, dict) else spacing_zyx
            bulk_center_curvature = sample_volume_field(
                curvature_volume,
                xyz[bulk_mask],
                curvature_spacing,
            ).reshape(-1).to(device=xyz.device, dtype=torch.float32)

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
        bulk_mask=bulk_mask,
        surface_xyz=xyz[surface_mask],
        surface_rotation_quat=rotation_quat[surface_mask],
        surface_rotation_mats=rotation_mats[surface_mask],
        surface_raw_scaling=raw_scaling[surface_mask],
        surface_scales=scales[surface_mask],
        surface_opacity=opacity[surface_mask],
        surface_attenuation=bulk_attenuation[surface_mask] if bulk_attenuation is not None else None,
        surface_normals=normals[surface_mask],
        surface_material_id=material_id[surface_mask],
        bulk_xyz=xyz[bulk_mask],
        bulk_rotation_mats=rotation_mats[bulk_mask],
        bulk_scales=scales[bulk_mask],
        bulk_opacity=opacity[bulk_mask],
        bulk_attenuation=bulk_attenuation[bulk_mask] if bulk_attenuation is not None else None,
        bulk_sigma=bulk_sigma,
        bulk_center_sdf=bulk_center_sdf,
        bulk_center_normals=bulk_center_normals,
        bulk_center_curvature=bulk_center_curvature,
        support_extent=support_extent,
        spatial_grid=spatial_grid,
        surface_support_extent=surface_support_extent,
        surface_spatial_grid=surface_spatial_grid,
        bulk_support_extent=bulk_support_extent,
        bulk_spatial_grid=bulk_spatial_grid,
        bulk_query_truncation_sigma=float(bulk_truncation_sigma if bulk_truncation_sigma is not None else truncation_sigma),
        bulk_query_q_support=float(bulk_truncation_sigma if bulk_truncation_sigma is not None else truncation_sigma) ** 2,
        render_state=render_state,
        ct_value=ct_value,
        surface_ct_value=ct_value[surface_mask] if ct_value is not None else None,
        bulk_ct_value=ct_value[bulk_mask] if ct_value is not None else None,
    )
