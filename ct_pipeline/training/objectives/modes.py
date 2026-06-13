from __future__ import annotations

from ct_pipeline.rendering.fields import is_bulk_intensity_field_mode
from ct_pipeline.training.control.optimizer import (
    _attenuation_only_bulk_gate_training_enabled,
    _attenuation_only_training_enabled,
)

def _boundary_band_distance(args) -> float:
    return max(float(getattr(args, "ct_boundary_band", getattr(args, "ct_boundary_band_voxels", 1.5))), 1e-6)


def _use_unified_compositor(args) -> bool:
    return bool(getattr(args, "ct_use_unified_compositor", True))


def _dual_separated_training_enabled(args) -> bool:
    return _use_unified_compositor(args) and bool(getattr(args, "ct_dual_separated_training", True))


def _bulk_intensity_training_enabled(args) -> bool:
    return _use_unified_compositor(args) and is_bulk_intensity_field_mode(
        getattr(args, "ct_bulk_field_mode", "bulk_intensity_field")
    )


def _dual_surface_inner_band(args) -> float:
    return max(float(getattr(args, "ct_dual_surface_inner_band", 0.0)), 0.0)


def _dual_surface_air_band(args) -> float:
    return max(float(getattr(args, "ct_dual_surface_air_band", 0.25)), 0.0)


def _surface_material_render_band(args, boundary_band_distance: float) -> float:
    boundary_band = max(float(boundary_band_distance), 0.0)
    return max(_dual_surface_air_band(args), min(boundary_band, 0.5))


def _bulk_halfspace_tau_current(args) -> float:
    current = getattr(args, "ct_bulk_halfspace_tau_current", None)
    if current is None:
        current = float(getattr(args, "ct_bulk_halfspace_tau_init", 0.5))
        setattr(args, "ct_bulk_halfspace_tau_current", current)
    return float(current)


def _intensity_geometry_flags(args, iteration: int):
    adaptive_mode = str(getattr(args, "ct_bulk_adaptive_mode", "fixed"))
    if _attenuation_only_training_enabled(args) and adaptive_mode == "fixed":
        return {
            "bulk_train_opacity": _attenuation_only_bulk_gate_training_enabled(args),
            "bulk_train_scale": False,
            "bulk_scale_grad": 0.0,
            "surface_train_opacity": False,
        }
    # adaptive bulk: scale can train from iter 0 (with slower LR via bulk_scale_grad)
    if adaptive_mode in ("scale", "scale_offset"):
        scale_grad = float(getattr(args, "ct_intensity_bulk_scale_grad", 0.1))
        return {
            "bulk_train_opacity": False,
            "bulk_train_scale": True,
            "bulk_scale_grad": scale_grad,
            "surface_train_opacity": False,
        }
    max_iter = max(1, int(getattr(args, "iterations", 1)))
    phase2_start = int(round(float(getattr(args, "ct_intensity_phase2_start_ratio", 0.6)) * float(max_iter)))
    phase3_start = int(round(float(getattr(args, "ct_intensity_phase3_start_ratio", 0.85)) * float(max_iter)))
    in_phase2 = int(iteration) >= phase2_start
    in_phase3 = int(iteration) >= phase3_start
    return {
        "bulk_train_opacity": bool(in_phase2),
        "bulk_train_scale": bool(in_phase3 and getattr(args, "ct_intensity_train_bulk_scale", False)),
        "bulk_scale_grad": float(getattr(args, "ct_intensity_bulk_scale_grad", 0.1)),
        "surface_train_opacity": bool(in_phase3 and getattr(args, "ct_intensity_train_surface_opacity", False)),
    }


def _bulk_intensity_sample_mode(args) -> str:
    return str(getattr(args, "ct_intensity_sample_mode", "full_band"))


def _containment_ramp_weight(args, iteration: int) -> float:
    base = float(getattr(args, "ct_containment_weight", getattr(args, "ct_bulk_sdf_containment_weight", 0.0)))
    if base <= 0.0:
        return 0.0
    iteration = max(0, int(iteration))
    i1 = int(getattr(args, "ct_containment_ramp_iter_1", 1000))
    w1 = float(getattr(args, "ct_containment_ramp_weight_1", 0.25))
    i2 = int(getattr(args, "ct_containment_ramp_iter_2", 3000))
    w2 = float(getattr(args, "ct_containment_ramp_weight_2", 0.5))
    i3 = int(getattr(args, "ct_containment_ramp_iter_3", 6000))
    w3 = float(getattr(args, "ct_containment_ramp_weight_3", 1.0))
    if i1 <= 0 or iteration <= i1:
        factor = w1 if i1 <= 0 else w1 * float(iteration) / float(max(i1, 1))
    elif iteration <= i2:
        factor = w1 + (w2 - w1) * float(iteration - i1) / float(max(i2 - i1, 1))
    elif iteration <= i3:
        factor = w2 + (w3 - w2) * float(iteration - i2) / float(max(i3 - i2, 1))
    else:
        factor = w3
    return base * max(0.0, factor)
