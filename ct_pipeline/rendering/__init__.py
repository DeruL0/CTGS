"""CT field queries and slice rendering."""

from .fields import (
    density_to_occupancy,
    query_bulk_fields,
    query_ct_density,
    query_ct_density_from_state,
    query_ct_density_from_state_by_region,
    query_ct_density_python,
    query_ct_fields_unified,
    query_surface_fields,
    render_ct_hybrid,
)
from .slices import (
    CTPatchGridCache,
    CTRenderState,
    build_ct_patch_renderer,
    prepare_ct_render_state,
    render_ct_slice_patch,
    render_ct_slice_world_patch,
    sample_gt_slice_patch,
)

__all__ = [
    "CTPatchGridCache",
    "CTRenderState",
    "build_ct_patch_renderer",
    "density_to_occupancy",
    "prepare_ct_render_state",
    "query_bulk_fields",
    "query_ct_density",
    "query_ct_density_from_state",
    "query_ct_density_from_state_by_region",
    "query_ct_density_python",
    "query_ct_fields_unified",
    "query_surface_fields",
    "render_ct_hybrid",
    "render_ct_slice_patch",
    "render_ct_slice_world_patch",
    "sample_gt_slice_patch",
]
