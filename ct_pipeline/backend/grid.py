from __future__ import annotations

import torch

from ct_pipeline.backend.core import CTSpatialGrid, get_ct_native_extension, require_cuda_tensors

DEFAULT_MAX_CELL_GAUSSIAN_PAIRS = 20_000_000
MAX_EFFECTIVE_GRID_CELL_VOXELS = 512


def _compute_support_extent(rotations: torch.Tensor, scales: torch.Tensor, truncation_sigma: float) -> torch.Tensor:
    return torch.sum(rotations.abs() * scales.unsqueeze(1), dim=2) * float(truncation_sigma)


def _resolve_cell_size(spacing_zyx, grid_cell_voxels: int) -> float:
    return max(float(min(spacing_zyx)) * float(grid_cell_voxels), 1e-6)


def _compute_grid_layout(means: torch.Tensor, support_extent: torch.Tensor, cell_size: float):
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

    cell_span = (cell_max - cell_min + 1).clamp_min(1).to(dtype=torch.int64)
    pair_count = torch.sum(cell_span[:, 0] * cell_span[:, 1] * cell_span[:, 2])
    return grid_min, grid_dims, cell_min, cell_max, pair_count


def build_uniform_grid_native(cell_min: torch.Tensor, cell_max: torch.Tensor, grid_dims: torch.Tensor):
    require_cuda_tensors(
        "Native CT uniform grid build requires CUDA tensor inputs and an available extension.",
        cell_min,
        cell_max,
        grid_dims,
    )
    native = get_ct_native_extension()
    return native.build_uniform_grid_cuda(
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
    max_cell_gaussian_pairs: int | None = DEFAULT_MAX_CELL_GAUSSIAN_PAIRS,
) -> CTSpatialGrid | None:
    means = torch.as_tensor(means)
    if means.numel() == 0:
        return None
    require_cuda_tensors("The active CT spatial grid path requires CUDA tensors.", means)

    support_extent = _compute_support_extent(rotations, scales, truncation_sigma)
    base_grid_cell_voxels = max(1, int(grid_cell_voxels))
    max_grid_cell_voxels = max(base_grid_cell_voxels, MAX_EFFECTIVE_GRID_CELL_VOXELS)
    effective_grid_cell_voxels = base_grid_cell_voxels
    selected_layout = None
    selected_pair_count = None
    while effective_grid_cell_voxels <= max_grid_cell_voxels:
        cell_size = _resolve_cell_size(spacing_zyx, effective_grid_cell_voxels)
        layout = _compute_grid_layout(means, support_extent, cell_size)
        pair_count = int(layout[-1].item())
        selected_layout = (cell_size, *layout)
        selected_pair_count = pair_count
        if max_cell_gaussian_pairs is None or pair_count <= int(max_cell_gaussian_pairs):
            break
        effective_grid_cell_voxels *= 2

    if (
        max_cell_gaussian_pairs is not None
        and selected_pair_count is not None
        and selected_pair_count > int(max_cell_gaussian_pairs)
    ):
        return None

    cell_size, grid_min, grid_dims, cell_min, cell_max, _ = selected_layout
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
