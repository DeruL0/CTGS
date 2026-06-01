from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ct_pipeline.rendering.bulk_support import resolve_bulk_query_truncation_sigma
from ct_pipeline.backend.core import CTSpatialGrid, CTTrainingState
from ct_pipeline.backend.grid import _compute_support_extent, build_uniform_grid_native


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


def _build_ct_spatial_grid_with_support_extent(
    means: torch.Tensor,
    support_extent: torch.Tensor,
    spacing_zyx,
    truncation_sigma: float,
    grid_cell_voxels: int,
    max_cell_gaussian_pairs: int | None = None,
) -> CTSpatialGrid | None:
    if means.numel() == 0:
        return None

    base_grid_cell_voxels = max(1, int(grid_cell_voxels))
    max_grid_cell_voxels = max(base_grid_cell_voxels, 512)
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

    should_release_cache = (
        max_cell_gaussian_pairs is not None
        and selected_pair_count is not None
        and selected_pair_count > int(max_cell_gaussian_pairs) * 0.75
    )
    if should_release_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
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


@dataclass
class CTRegionGridCache:
    name: str
    rebuild_interval: int
    inflation_margin: float
    drift_check: bool = True
    max_cell_gaussian_pairs: int = 20_000_000
    grid: CTSpatialGrid | None = None
    current_exact_support_extent: torch.Tensor | None = None
    last_build_iter: int = -1
    last_build_count: int = 0
    last_build_indices: torch.Tensor | None = None
    last_build_xyz: torch.Tensor | None = None
    last_exact_support_extent: torch.Tensor | None = None
    rebuild_count: int = 0
    last_rebuild_reason: str = "never"

    def invalidate(self) -> None:
        self.grid = None
        self.current_exact_support_extent = None
        self.last_build_iter = -1
        self.last_build_count = 0
        self.last_build_indices = None
        self.last_build_xyz = None
        self.last_exact_support_extent = None
        self.last_rebuild_reason = "invalidated"

    def _is_stale(
        self,
        iteration: int,
        xyz: torch.Tensor,
        exact_support_extent: torch.Tensor,
        region_indices: torch.Tensor,
        cell_size: float,
        hard_invalidate: bool,
    ) -> tuple[bool, str]:
        count = int(xyz.shape[0])
        if hard_invalidate:
            return True, "hard_invalidate"
        if self.grid is None:
            skipped_recently = (
                self.last_rebuild_reason == "too_many_grid_pairs"
                and count == self.last_build_count
                and self.last_build_indices is not None
                and torch.equal(region_indices, self.last_build_indices)
                and int(iteration) - int(self.last_build_iter) < int(self.rebuild_interval)
            )
            if skipped_recently:
                return False, "fresh"
            return True, "first_build"
        if count != self.last_build_count:
            return True, "count_changed"
        if self.last_build_indices is None or not torch.equal(region_indices, self.last_build_indices):
            return True, "membership_changed"
        if int(iteration) - int(self.last_build_iter) >= int(self.rebuild_interval):
            return True, "interval"
        if not self.drift_check:
            return False, "fresh"
        if self.last_build_xyz is None or self.last_exact_support_extent is None:
            return True, "missing_snapshot"

        max_xyz_drift = torch.max(torch.abs(xyz - self.last_build_xyz)) if count > 0 else xyz.new_tensor(0.0)
        if float(max_xyz_drift.item()) > float(self.inflation_margin) * float(cell_size):
            return True, "xyz_drift"

        support_ratio = exact_support_extent / self.last_exact_support_extent.clamp_min(1e-8)
        max_support_ratio = torch.max(support_ratio) if support_ratio.numel() > 0 else exact_support_extent.new_tensor(1.0)
        if float(max_support_ratio.item()) > 1.0 + float(self.inflation_margin):
            return True, "support_growth"
        return False, "fresh"

    def refresh(
        self,
        *,
        iteration: int,
        xyz: torch.Tensor,
        rotation_mats: torch.Tensor,
        scales: torch.Tensor,
        region_indices: torch.Tensor,
        spacing_zyx,
        truncation_sigma: float,
        grid_cell_voxels: int,
        hard_invalidate: bool = False,
    ) -> None:
        with torch.no_grad():
            xyz_snapshot = xyz.detach()
            rotation_snapshot = rotation_mats.detach()
            scales_snapshot = scales.detach().clamp_min(1e-6)
            region_indices_snapshot = region_indices.detach().clone()
            if xyz_snapshot.numel() == 0:
                self.grid = None
                self.current_exact_support_extent = None
                self.last_build_iter = int(iteration)
                self.last_build_count = 0
                self.last_build_indices = region_indices_snapshot
                self.last_build_xyz = xyz_snapshot.clone()
                self.last_exact_support_extent = None
                self.last_rebuild_reason = "empty"
                return

            exact_support_extent = _compute_support_extent(rotation_snapshot, scales_snapshot, truncation_sigma).detach()
            self.current_exact_support_extent = exact_support_extent
            if self.grid is not None:
                cell_size = float(self.grid.cell_size)
            else:
                cell_size = _resolve_cell_size(spacing_zyx, grid_cell_voxels)
            stale, reason = self._is_stale(
                int(iteration),
                xyz_snapshot,
                exact_support_extent,
                region_indices_snapshot,
                cell_size,
                hard_invalidate,
            )
            if not stale:
                return

            inflated_support_extent = exact_support_extent * (1.0 + float(self.inflation_margin))
            inflated_support_extent = inflated_support_extent + float(self.inflation_margin) * float(cell_size)
            self.grid = _build_ct_spatial_grid_with_support_extent(
                xyz_snapshot,
                inflated_support_extent,
                spacing_zyx=spacing_zyx,
                truncation_sigma=truncation_sigma,
                grid_cell_voxels=grid_cell_voxels,
                max_cell_gaussian_pairs=self.max_cell_gaussian_pairs,
            )
            self.last_build_iter = int(iteration)
            self.last_build_count = int(xyz_snapshot.shape[0])
            self.last_build_indices = region_indices_snapshot
            self.last_build_xyz = xyz_snapshot.clone()
            self.last_exact_support_extent = exact_support_extent.clone()
            self.rebuild_count += 1
            self.last_rebuild_reason = "too_many_grid_pairs" if self.grid is None else reason


@dataclass
class CTGridCacheManager:
    enabled: bool
    spacing_zyx: tuple[float, float, float]
    truncation_sigma: float
    bulk_truncation_sigma: float
    grid_cell_voxels: int
    surface: CTRegionGridCache = field(default_factory=lambda: CTRegionGridCache("surface", 10, 0.25))
    bulk: CTRegionGridCache = field(default_factory=lambda: CTRegionGridCache("bulk", 50, 0.25))

    @classmethod
    def from_args(cls, args, spacing_zyx) -> "CTGridCacheManager":
        margin = float(getattr(args, "ct_grid_cache_inflation_margin", 0.25))
        drift_check = bool(getattr(args, "ct_grid_cache_drift_check", True))
        max_pairs = int(getattr(args, "ct_grid_cache_max_cell_gaussian_pairs", 20_000_000))
        return cls(
            enabled=bool(getattr(args, "ct_grid_cache", True)),
            spacing_zyx=tuple(float(value) for value in spacing_zyx),
            truncation_sigma=float(getattr(args, "ct_gaussian_truncation_sigma", 4.0)),
            bulk_truncation_sigma=resolve_bulk_query_truncation_sigma(args),
            grid_cell_voxels=int(getattr(args, "ct_grid_cell_voxels", 8)),
            surface=CTRegionGridCache(
                name="surface",
                rebuild_interval=int(getattr(args, "ct_surface_grid_rebuild_interval", 10)),
                inflation_margin=margin,
                drift_check=drift_check,
                max_cell_gaussian_pairs=max_pairs,
            ),
            bulk=CTRegionGridCache(
                name="bulk",
                rebuild_interval=int(getattr(args, "ct_bulk_grid_rebuild_interval", 50)),
                inflation_margin=margin,
                drift_check=drift_check,
                max_cell_gaussian_pairs=max_pairs,
            ),
        )

    def refresh(self, training_state: CTTrainingState, iteration: int, hard_invalidate: bool = False) -> None:
        if not self.enabled:
            return

        surface_indices = torch.nonzero(training_state.surface_mask, as_tuple=False).reshape(-1)
        bulk_indices = torch.nonzero(training_state.bulk_mask, as_tuple=False).reshape(-1)
        self.surface.refresh(
            iteration=iteration,
            xyz=training_state.surface_xyz,
            rotation_mats=training_state.surface_rotation_mats,
            scales=training_state.surface_scales,
            region_indices=surface_indices,
            spacing_zyx=self.spacing_zyx,
            truncation_sigma=self.truncation_sigma,
            grid_cell_voxels=self.grid_cell_voxels,
            hard_invalidate=hard_invalidate,
        )
        self.bulk.refresh(
            iteration=iteration,
            xyz=training_state.bulk_xyz,
            rotation_mats=training_state.bulk_rotation_mats,
            scales=training_state.bulk_scales,
            region_indices=bulk_indices,
            spacing_zyx=self.spacing_zyx,
            truncation_sigma=self.bulk_truncation_sigma,
            grid_cell_voxels=self.grid_cell_voxels,
            hard_invalidate=hard_invalidate,
        )

    def attach(self, training_state: CTTrainingState) -> None:
        if not self.enabled:
            return
        training_state.surface_spatial_grid = self.surface.grid
        training_state.surface_support_extent = self.surface.current_exact_support_extent
        training_state.bulk_spatial_grid = self.bulk.grid
        training_state.bulk_support_extent = self.bulk.current_exact_support_extent

    def invalidate_all(self) -> None:
        self.surface.invalidate()
        self.bulk.invalidate()
