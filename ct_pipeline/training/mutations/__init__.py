"""Compatibility facade for CT training mutation mechanics.

The implementation lives in focused modules: schedules, shared helpers,
surface reseeding, bulk reseeding/repair, bulk pruning, and densification.
Keep this module as the stable import surface for older training code and scripts.
"""

from __future__ import annotations

from ct_pipeline.training.mutations.bulk_pruning import (
    _bulk_prune_stats,
    _apply_bulk_pruning,
)

from ct_pipeline.training.mutations.bulk_reseeding import (
    _bulk_reseed_stats,
    _apply_material_limited_bulk_growth,
    _enforce_bulk_sdf_containment,
    _mask_to_numpy_bool,
    _sdf_to_numpy,
    _voxel_indices_to_world,
    _material_completion_components,
    _budgeted_repair_components,
    _component_world_points,
    _gaussian_density_np,
    _probe_candidate_inside_material,
    _sdf_inside_np,
    _directional_clearance_one_side_torch,
    _clearance_cap_candidate_scales,
    _append_budgeted_repair_seeds,
    _apply_budgeted_component_bulk_repair,
    _append_bulk_completion_seeds,
    _apply_material_coverage_completion,
    _apply_profile_integrated_bulk_reseeding,
    _sample_gap_reseed_candidates,
    _apply_gap_aware_bulk_reseeding,
    _apply_bulk_coverage_reseeding,
)

from ct_pipeline.training.mutations.densification import (
    _apply_ct_densification,
)

from ct_pipeline.training.mutations.helpers import (
    _bulk_mask_tensor,
    _sample_binary_mask_nearest,
    _apply_surface_scale_hard_projection,
    _log_air_shell_diagnostics,
    _freeze_bulk_xyz_gradients,
    _ct_topk_indices,
    _ct_support_volume_from_analysis,
    _sample_sdf_normals_for_reseed,
    _frames_from_surface_normals,
    _ensure_gap_seed_birth_iter,
    _append_gap_seed_birth_iter,
    _project_xyz_to_sdf_zero,
)

from ct_pipeline.training.mutations.schedules import (
    _ct_densification_active,
    _ct_should_densify,
    _ct_surface_reseeding_active,
    _ct_should_reseed_surface,
    _ct_effective_loss_weights,
    _ct_should_reseed_bulk,
    _ct_should_prune_bulk,
)

from ct_pipeline.training.mutations.surface_reseeding import (
    _surface_reseed_stats,
    _sample_boundary_reseed_anchors,
    _surface_reseed_material_ids,
    _apply_surface_reseeding,
)

__all__ = [
    "_append_budgeted_repair_seeds",
    "_append_bulk_completion_seeds",
    "_append_gap_seed_birth_iter",
    "_apply_budgeted_component_bulk_repair",
    "_apply_bulk_coverage_reseeding",
    "_apply_bulk_pruning",
    "_apply_ct_densification",
    "_apply_gap_aware_bulk_reseeding",
    "_apply_material_coverage_completion",
    "_apply_material_limited_bulk_growth",
    "_apply_profile_integrated_bulk_reseeding",
    "_apply_surface_reseeding",
    "_apply_surface_scale_hard_projection",
    "_budgeted_repair_components",
    "_bulk_mask_tensor",
    "_bulk_prune_stats",
    "_bulk_reseed_stats",
    "_clearance_cap_candidate_scales",
    "_component_world_points",
    "_ct_densification_active",
    "_ct_effective_loss_weights",
    "_ct_should_densify",
    "_ct_should_prune_bulk",
    "_ct_should_reseed_bulk",
    "_ct_should_reseed_surface",
    "_ct_support_volume_from_analysis",
    "_ct_surface_reseeding_active",
    "_ct_topk_indices",
    "_directional_clearance_one_side_torch",
    "_enforce_bulk_sdf_containment",
    "_ensure_gap_seed_birth_iter",
    "_frames_from_surface_normals",
    "_freeze_bulk_xyz_gradients",
    "_gaussian_density_np",
    "_log_air_shell_diagnostics",
    "_mask_to_numpy_bool",
    "_material_completion_components",
    "_probe_candidate_inside_material",
    "_project_xyz_to_sdf_zero",
    "_sample_binary_mask_nearest",
    "_sample_boundary_reseed_anchors",
    "_sample_gap_reseed_candidates",
    "_sample_sdf_normals_for_reseed",
    "_sdf_inside_np",
    "_sdf_to_numpy",
    "_surface_reseed_material_ids",
    "_surface_reseed_stats",
    "_voxel_indices_to_world",
]
