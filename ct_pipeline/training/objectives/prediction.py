from __future__ import annotations

import torch

from ct_pipeline.rendering.fields import (
    bulk_intensity_readout,
    compose_signed_overlap_occupancy,
    density_to_occupancy,
    query_ct_fields_unified,
    query_ct_density_from_state_by_region,
    query_ct_local_intensity_from_state,
)
from ct_pipeline.training.objectives.modes import _use_unified_compositor


def _bulk_intensity_from_fields(fields: dict[str, torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    if "A_b" in fields:
        return fields["A_b"].to(dtype=torch.float32)
    raw_bulk = fields.get("I_b_raw", fields["I_b"]).to(dtype=torch.float32)
    den_b = fields["den_b"].to(dtype=torch.float32)
    return bulk_intensity_readout(raw_bulk, den_b, eps=eps).to(dtype=torch.float32)


def _bulk_intensity_prediction(
    fields: dict[str, torch.Tensor],
    signed_distance: torch.Tensor | None,
    intensity_air: float,
    config=None,
) -> torch.Tensor:
    del signed_distance, intensity_air, config
    return _bulk_intensity_from_fields(fields).to(dtype=torch.float32)


def compute_role_separated_ct_prediction(
    training_state,
    points_xyz: torch.Tensor,
    signed_distance: torch.Tensor,
    boundary_band_distance: float,
    include_surface: bool = True,
    detach_bulk: bool = False,
    detach_surface: bool = False,
    surface_material_gate_sigma: float | None = None,
    material_compose_mode: str = "bulk_first_material",
) -> torch.Tensor:
    if points_xyz.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=points_xyz.device)

    bulk_density = query_ct_density_from_state_by_region(
        training_state,
        points_xyz,
        region="bulk",
        detach=detach_bulk,
    ).to(dtype=torch.float32)
    if not include_surface:
        occupancy, _, _ = compose_signed_overlap_occupancy(
            bulk_density,
            None,
            signed_distance,
            boundary_band_distance,
            surface_material_gate_sigma=surface_material_gate_sigma,
            material_compose_mode=material_compose_mode,
        )
        return occupancy

    surface_density = query_ct_density_from_state_by_region(
        training_state,
        points_xyz,
        region="surface",
        detach=detach_surface,
    ).to(dtype=torch.float32)
    occupancy, _, _ = compose_signed_overlap_occupancy(
        bulk_density,
        surface_density,
        signed_distance,
        boundary_band_distance,
        surface_material_gate_sigma=surface_material_gate_sigma,
        material_compose_mode=material_compose_mode,
    )
    return occupancy.to(dtype=torch.float32)


def compute_raw_combined_ct_occupancy(
    training_state,
    points_xyz: torch.Tensor,
    detach_bulk: bool = False,
    detach_surface: bool = False,
) -> torch.Tensor:
    if points_xyz.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=points_xyz.device)

    bulk_density = query_ct_density_from_state_by_region(
        training_state,
        points_xyz,
        region="bulk",
        detach=detach_bulk,
    ).to(dtype=torch.float32)
    surface_density = query_ct_density_from_state_by_region(
        training_state,
        points_xyz,
        region="surface",
        detach=detach_surface,
    ).to(dtype=torch.float32)
    o_bulk = density_to_occupancy(bulk_density)
    o_surface = density_to_occupancy(surface_density)
    return (o_surface + (1.0 - o_surface) * o_bulk).clamp(0.0, 1.0).to(dtype=torch.float32)


def compute_surface_bounded_bulk_volume_prediction(
    training_state,
    points_xyz: torch.Tensor,
    signed_distance: torch.Tensor,
    intensity_air: float,
    boundary_band_distance: float,
    surface_material_gate_sigma: float | None = None,
    material_compose_mode: str = "bulk_first_material",
    config=None,
    use_unified: bool | None = None,
    bulk_train_opacity: bool = False,
    bulk_train_scale: bool = False,
    bulk_scale_grad: float = 1.0,
    surface_train_opacity: bool = False,
) -> torch.Tensor:
    if points_xyz.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=points_xyz.device)
    if use_unified is None:
        use_unified = _use_unified_compositor(config) if config is not None else False
    if use_unified:
        fields = query_ct_fields_unified(
            points_xyz,
            training_state,
            signed_distance=signed_distance,
            config=config,
            intensity_air=float(intensity_air),
            include_surface=True,
            bulk_train_opacity=bulk_train_opacity,
            bulk_train_scale=bulk_train_scale,
            bulk_scale_grad=bulk_scale_grad,
            surface_train_opacity=surface_train_opacity,
            train_ct_value=True,
            detach_value_geometry=True,
        )
        return fields["I_pred"]
    bulk_density = query_ct_density_from_state_by_region(
        training_state,
        points_xyz,
        region="bulk",
        detach=True,
    ).to(dtype=torch.float32)
    surface_density = query_ct_density_from_state_by_region(
        training_state,
        points_xyz,
        region="surface",
        detach=True,
    ).to(dtype=torch.float32)
    combined_occupancy, _, _ = compose_signed_overlap_occupancy(
        bulk_density,
        surface_density,
        signed_distance,
        boundary_band_distance,
        surface_material_gate_sigma=surface_material_gate_sigma,
        material_compose_mode=material_compose_mode,
    )
    combined_occupancy = combined_occupancy.to(device=points_xyz.device, dtype=torch.float32)
    local_intensity = query_ct_local_intensity_from_state(
        training_state,
        points_xyz,
        detach_geometry=True,
        include_surface=True,
        signed_distance=signed_distance,
        boundary_band_distance=boundary_band_distance,
        surface_material_gate_sigma=surface_material_gate_sigma,
        material_compose_mode=material_compose_mode,
    ).to(device=points_xyz.device, dtype=torch.float32)
    return float(intensity_air) + (local_intensity - float(intensity_air)) * combined_occupancy
