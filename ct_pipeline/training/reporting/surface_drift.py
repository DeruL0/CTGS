from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from ct_pipeline.rendering.fields import density_to_occupancy, query_ct_density_from_state_by_region
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.sampling import _interior_void_air_mask_np
from ct_pipeline.training.utils import as_device_tensor, write_key_value_report

from .bulk_metrics import (
    _bulk_coverage_gap_threshold,
    _record_bulk_containment_metrics,
    _record_bulk_coverage_metrics,
    _record_bulk_intensity_quantiles,
    _record_bulk_occ_quantiles,
    _record_bulk_surface_gap_distance_metrics,
    _record_combined_occ_quantiles,
    _record_dual_surface_shell_metrics,
    _record_surface_owned_bulk_field_metrics,
    _record_uncovered_component_metrics,
    _sample_material_sdf_band_points,
    _sample_material_voxel_points,
)
from .common import _mask_to_np, _quantile_metrics, _roi_bbox_from_analysis, _roi_window_from_analysis, _sample_mask_points
from .volume_metrics import (
    _record_false_hole_diagnostics,
    _record_high_gradient_material_metrics,
    _record_volume_material_intensity_metrics,
)


def _compute_surface_drift_diagnostics(
    training_state,
    analysis,
    spacing_zyx,
    curvature_field=None,
    surface_reseed_added: int = 0,
    bulk_reseed_stats=None,
    bulk_prune_stats=None,
    densify_stats=None,
    boundary_band_distance: float = 1.5,
    signed_distance_field=None,
    volume_cuda=None,
    intensity_air: float = 0.0,
    intensity_mat: float = 1.0,
    false_hole_sample_count: int = 4096,
    false_hole_boundary_band: float = 2.0,
    false_hole_material_threshold: float = 0.65,
    false_hole_dark_margin: float = 0.15,
    surface_material_gate_sigma=None,
    material_compose_mode: str = "bulk_first_material",
    config=None,
    use_unified_compositor: bool = True,
):
    surface_xyz = as_device_tensor(getattr(training_state, "surface_xyz", torch.empty((0, 3))))
    bulk_xyz = as_device_tensor(getattr(training_state, "bulk_xyz", torch.empty((0, 3))))
    total_xyz = getattr(training_state, "xyz", torch.empty((0, 3)))
    model_xyz = getattr(training_state, "xyz", None)
    if isinstance(model_xyz, torch.Tensor):
        device = model_xyz.device
    else:
        device = surface_xyz.device
    dtype = surface_xyz.dtype if torch.is_floating_point(surface_xyz) else torch.float32
    surface_xyz = as_device_tensor(surface_xyz, device=device, dtype=dtype, reshape=(-1, 3))
    bulk_xyz = as_device_tensor(bulk_xyz, device=device, dtype=dtype, reshape=(-1, 3))
    total_xyz = as_device_tensor(total_xyz, device=device, dtype=dtype, reshape=(-1, 3))
    surface_count = int(surface_xyz.shape[0])
    bulk_count = int(bulk_xyz.shape[0])
    total_count = int(total_xyz.shape[0])
    bulk_reseed_stats = bulk_reseed_stats or {}
    bulk_prune_stats = bulk_prune_stats or {}
    densify_stats = densify_stats or {}

    metrics = {
        "total_count": total_count,
        "bulk_count": bulk_count,
        "surface_count": surface_count,
        "surface_outside_support_ratio": 0.0 if surface_count == 0 else float("nan"),
        "surface_to_phase1_boundary_distance_p50": float("nan"),
        "surface_to_phase1_boundary_distance_p90": float("nan"),
        "surface_to_phase1_boundary_distance_p99": float("nan"),
        "surface_max_scale_p90": float("nan"),
        "surface_max_scale_p99": float("nan"),
        "surface_opacity_p50": float("nan"),
        "surface_opacity_p90": float("nan"),
        "surface_coverage_gap_ratio": float("nan"),
        "bulk_interior_coverage_gap_ratio": float("nan"),
        "bulk_interior_occupancy_p10": float("nan"),
        "bulk_interior_occupancy_p50": float("nan"),
        "bulk_material_coverage_gap_ratio": float("nan"),
        "bulk_material_occupancy_p10": float("nan"),
        "bulk_material_occupancy_p50": float("nan"),
        "bulk_material_sample_count": 0,
        "combined_material_coverage_gap_ratio": float("nan"),
        "combined_material_occ_union_raw_p10": float("nan"),
        "combined_material_occ_union_raw_p50": float("nan"),
        "combined_material_sample_count": 0,
        "bulk_deep_coverage_gap_ratio": float("nan"),
        "bulk_deep_occupancy_p10": float("nan"),
        "bulk_deep_occupancy_p50": float("nan"),
        "bulk_deep_sample_count": 0,
        "bulk_deep_sdf_1p5_3_gap_ratio": float("nan"),
        "bulk_deep_sdf_1p5_3_sample_count": 0,
        "bulk_deep_sdf_3_6_gap_ratio": float("nan"),
        "bulk_deep_sdf_3_6_sample_count": 0,
        "bulk_deep_sdf_6_plus_gap_ratio": float("nan"),
        "bulk_deep_sdf_6_plus_sample_count": 0,
        "bulk_deep_low_occ_sdf_depth_p50": float("nan"),
        "bulk_deep_low_occ_sdf_depth_p90": float("nan"),
        "bulk_deep_low_occ_nearest_bulk_distance_p50": float("nan"),
        "bulk_deep_low_occ_nearest_bulk_distance_p90": float("nan"),
        "material_deep_occ_b_raw_p10": float("nan"),
        "material_deep_occ_b_raw_p50": float("nan"),
        "material_boundary_shell_occ_b_raw_p10": float("nan"),
        "material_boundary_shell_occ_b_raw_p50": float("nan"),
        "cavity_material_shell_occ_b_raw_p10": float("nan"),
        "cavity_material_shell_occ_b_raw_p50": float("nan"),
        "void_air_occ_b_raw_p95": float("nan"),
        "exterior_air_near_occ_b_raw_p95": float("nan"),
        "boundary_miss_rate": float("nan"),
        "tau_current": float("nan"),
        "bulk_surface_gap_distance_p50": float("nan"),
        "bulk_surface_gap_distance_p90": float("nan"),
        "bulk_containment_violation_ratio": float("nan"),
        "surface_mat_shell_mae": float("nan"),
        "surface_mat_shell_sample_count": 0,
        "surface_air_shell_leak": float("nan"),
        "surface_air_shell_sample_count": 0,
        "false_hole_candidate_count": 0,
        "false_hole_active_ratio": float("nan"),
        "false_hole_pred_intensity_p10": float("nan"),
        "false_hole_pred_intensity_p50": float("nan"),
        "false_hole_target_intensity_p50": float("nan"),
        "false_hole_bulk_occ_p10": float("nan"),
        "false_hole_bulk_occ_p50": float("nan"),
        "cavity_void_false_fill_ratio": float("nan"),
        "surface_owned_bulk_W_p10": float("nan"),
        "surface_owned_bulk_W_p50": float("nan"),
        "surface_owned_bulk_mu_raw_p10": float("nan"),
        "surface_owned_bulk_mu_raw_p50": float("nan"),
        "surface_owned_bulk_A_b_p10": float("nan"),
        "surface_owned_bulk_A_b_p50": float("nan"),
        "bulk_material_A_b_p10": float("nan"),
        "bulk_material_A_b_p50": float("nan"),
        "bulk_material_A_b_p90": float("nan"),
        "bulk_material_sample_count": 0,
        "material_volume_MAE": float("nan"),
        "material_volume_RMSE": float("nan"),
        "material_volume_corr": float("nan"),
        "material_volume_A_b_p10": float("nan"),
        "material_volume_A_b_p50": float("nan"),
        "material_volume_A_b_p90": float("nan"),
        "material_volume_sample_count": 0,
        "high_gradient_material_MAE": float("nan"),
        "high_gradient_material_sample_count": 0,
        "num_uncovered_components": int(bulk_reseed_stats.get("num_uncovered_components", 0)),
        "max_uncovered_component_voxels": int(bulk_reseed_stats.get("max_uncovered_component_voxels", 0)),
        "void_air_A_b_p95": float("nan"),
        "void_air_sample_count": 0,
        "exterior_air_near_A_b_p95": float("nan"),
        "exterior_air_near_sample_count": 0,
        "surface_owned_false_hole_ratio": float("nan"),
        "high_curvature_bulk_owned_ratio": float("nan"),
        "surface_reseed_added": int(surface_reseed_added),
        "bulk_reseed_added": int(bulk_reseed_stats.get("added", 0)),
        "bulk_reseed_candidates": int(bulk_reseed_stats.get("candidates", 0)),
        "bulk_reseed_low_coverage_ratio": float(bulk_reseed_stats.get("low_coverage_ratio", 0.0)),
        "bulk_reseed_sigma_init_mean": float(bulk_reseed_stats.get("sigma_init_mean", float("nan"))),
        "bulk_reseed_atten_init_mean": float(bulk_reseed_stats.get("atten_init_mean", float("nan"))),
        "bulk_gap_grown_count": int(bulk_reseed_stats.get("bulk_grown_count", 0)),
        "bulk_repair_residual_mass": float(bulk_reseed_stats.get("repair_residual_mass", 0.0)),
        "bulk_repair_stretched_count": int(bulk_reseed_stats.get("repair_stretched_count", 0)),
        "bulk_repair_skipped_small_components": int(bulk_reseed_stats.get("repair_skipped_small_components", 0)),
        "bulk_repair_skipped_exclusion": int(bulk_reseed_stats.get("repair_skipped_exclusion", 0)),
        "bulk_repair_skipped_low_gain": int(bulk_reseed_stats.get("repair_skipped_low_gain", 0)),
        "bulk_repair_skipped_containment": int(bulk_reseed_stats.get("repair_skipped_containment", 0)),
        "bulk_repair_skipped_overfill": int(bulk_reseed_stats.get("repair_skipped_overfill", 0)),
        "bulk_repair_skipped_no_clearance_headroom": int(
            bulk_reseed_stats.get("repair_skipped_no_clearance_headroom", 0)
        ),
        "bulk_repair_clearance_limited": int(bulk_reseed_stats.get("repair_clearance_limited", 0)),
        "bulk_repair_components_considered": int(bulk_reseed_stats.get("repair_components_considered", 0)),
        "bulk_pruned_count": int(bulk_prune_stats.get("pruned", 0)),
        "bulk_prune_protected_gap_seed": int(bulk_prune_stats.get("protected_gap_seed", 0)),
        "densify_surface_split": int(densify_stats.get("surface_split", 0)),
        "densify_bulk_split": int(densify_stats.get("bulk_split", 0)),
        "densify_children_added": int(densify_stats.get("children_added", 0)),
        "densify_net_added": int(densify_stats.get("net_added", 0)),
    }
    if config is not None:
        metrics["boundary_miss_rate"] = float(getattr(config, "ct_boundary_miss_rate_last", float("nan")))
        metrics["tau_current"] = float(getattr(config, "ct_bulk_halfspace_tau_current", getattr(config, "ct_bulk_halfspace_tau_init", float("nan"))))
    _record_uncovered_component_metrics(metrics, training_state, analysis, spacing_zyx, device, dtype, config=config)

    interior_points = analysis.get("interior_points", analysis.get("material_points"))
    if interior_points is not None and getattr(training_state, "xyz", torch.empty((0, 3))).numel() > 0:
        with torch.no_grad():
            interior = as_device_tensor(interior_points, device=device, dtype=dtype).reshape(-1, 3)
            if interior.shape[0] > 4096:
                indices = torch.linspace(0, interior.shape[0] - 1, 4096, device=device).to(dtype=torch.long)
                interior = interior.index_select(0, indices)
            if interior.numel() > 0:
                bulk_den = query_ct_density_from_state_by_region(
                    training_state, interior, region="bulk", detach=True
                ).to(dtype=torch.float32)
                bulk_occ = density_to_occupancy(bulk_den).to(dtype=torch.float32)
                gap_threshold = _bulk_coverage_gap_threshold(config)
                metrics["bulk_interior_coverage_gap_threshold"] = gap_threshold
                metrics["bulk_interior_coverage_gap_ratio"] = float((bulk_den < gap_threshold).float().mean().item())
                for suffix, value in _quantile_metrics(bulk_den.detach().cpu().numpy(), ("p10", "p50")).items():
                    metrics[f"bulk_interior_den_b_{suffix}"] = value
                for suffix, value in _quantile_metrics(bulk_occ.detach().cpu().numpy(), ("p10", "p50")).items():
                    metrics[f"bulk_interior_occupancy_{suffix}"] = value

            boundary_band = float(boundary_band_distance)
            material_points, material_depth = _sample_material_voxel_points(
                analysis,
                spacing_zyx,
                device,
                dtype,
                deep_only=False,
                boundary_band=boundary_band,
                signed_distance_field=signed_distance_field,
            )
            _record_bulk_coverage_metrics(
                metrics,
                "bulk_material",
                training_state,
                material_points,
                depth=material_depth,
                bulk_xyz=bulk_xyz,
                config=config,
            )
            _record_bulk_intensity_quantiles(
                metrics,
                "bulk_material",
                training_state,
                material_points,
                signed_distance_field,
                config,
                intensity_air,
            )
            _record_combined_occ_quantiles(
                metrics,
                "combined_material",
                training_state,
                material_points,
                signed_distance_field,
                config,
                intensity_air,
            )
            _record_high_gradient_material_metrics(
                metrics,
                training_state,
                analysis,
                spacing_zyx,
                device,
                dtype,
                signed_distance_field=signed_distance_field,
                volume_cuda=volume_cuda,
                config=config,
                intensity_air=float(intensity_air),
            )
            _record_volume_material_intensity_metrics(
                metrics,
                training_state,
                analysis,
                spacing_zyx,
                device,
                dtype,
                signed_distance_field=signed_distance_field,
                volume_cuda=volume_cuda,
                config=config,
                intensity_air=float(intensity_air),
            )
            deep_points, deep_depth = _sample_material_voxel_points(
                analysis,
                spacing_zyx,
                device,
                dtype,
                deep_only=True,
                boundary_band=boundary_band,
                signed_distance_field=signed_distance_field,
            )
            _record_bulk_coverage_metrics(
                metrics,
                "bulk_deep",
                training_state,
                deep_points,
                depth=deep_depth,
                bulk_xyz=bulk_xyz,
                config=config,
            )
            _record_bulk_occ_quantiles(metrics, "material_deep", training_state, deep_points)

            boundary_shell_points = _sample_material_sdf_band_points(
                analysis,
                spacing_zyx,
                device,
                dtype,
                signed_distance_field=signed_distance_field,
                sdf_min=-float(boundary_band),
                sdf_max=-0.2,
            )
            _record_bulk_occ_quantiles(metrics, "material_boundary_shell", training_state, boundary_shell_points)
            _record_surface_owned_bulk_field_metrics(
                metrics, training_state, boundary_shell_points, signed_distance_field, config, float(intensity_air)
            )

            material_mask = _mask_to_np(analysis.get("material_mask", analysis.get("coarse_support_mask")))
            void_mask = _mask_to_np(analysis.get("void_mask"), fallback_shape=material_mask.shape if material_mask is not None else None)
            if material_mask is not None and void_mask is not None:
                air_mask = np.logical_or(~material_mask, void_mask)
                roi_window = _roi_window_from_analysis(analysis, material_mask.shape)
                roi_bbox = _roi_bbox_from_analysis(analysis, material_mask.shape)
                interior_void_air = _interior_void_air_mask_np(air_mask, roi_bbox=roi_bbox)
                structure = ndimage.generate_binary_structure(3, 1)
                cavity_material = np.logical_and(
                    material_mask,
                    ndimage.binary_dilation(interior_void_air, structure=structure, iterations=4),
                )
                cavity_material_points = _sample_mask_points(cavity_material, spacing_zyx, device, dtype, 4096)
                _record_bulk_occ_quantiles(metrics, "cavity_material_shell", training_state, cavity_material_points)

                void_air_points = _sample_mask_points(interior_void_air, spacing_zyx, device, dtype, 4096)
                _record_bulk_occ_quantiles(metrics, "void_air", training_state, void_air_points, names=("p95",))
                _record_bulk_intensity_quantiles(
                    metrics,
                    "void_air",
                    training_state,
                    void_air_points,
                    signed_distance_field,
                    config,
                    intensity_air,
                    names=("p95",),
                )

                exterior_air = np.logical_and(air_mask, np.logical_not(interior_void_air))
                exterior_near = np.logical_and(
                    exterior_air,
                    ndimage.binary_dilation(material_mask, structure=structure, iterations=3),
                )
                exterior_near_points = _sample_mask_points(exterior_near, spacing_zyx, device, dtype, 4096)
                _record_bulk_occ_quantiles(metrics, "exterior_air_near", training_state, exterior_near_points, names=("p95",))
                _record_bulk_intensity_quantiles(
                    metrics,
                    "exterior_air_near",
                    training_state,
                    exterior_near_points,
                    signed_distance_field,
                    config,
                    intensity_air,
                    names=("p95",),
                )

            _record_bulk_containment_metrics(
                metrics,
                training_state,
                signed_distance_field,
                margin=float(getattr(config, "ct_bulk_sdf_containment_margin", 0.0)) if config is not None else 0.0,
                config=config,
            )

            _record_dual_surface_shell_metrics(
                metrics,
                training_state,
                analysis,
                spacing_zyx,
                device,
                dtype,
                signed_distance_field=signed_distance_field,
                volume_cuda=volume_cuda,
                intensity_air=float(intensity_air),
                config=config,
            )
            _record_bulk_surface_gap_distance_metrics(
                metrics,
                training_state,
                analysis,
                spacing_zyx,
                device,
                dtype,
                signed_distance_field=signed_distance_field,
                config=config,
                intensity_air=float(intensity_air),
                intensity_mat=float(intensity_mat),
            )

    _record_false_hole_diagnostics(
        metrics,
        training_state,
        analysis,
        spacing_zyx,
        device,
        dtype,
        signed_distance_field=signed_distance_field,
        volume_cuda=volume_cuda,
        intensity_air=float(intensity_air),
        intensity_mat=float(intensity_mat),
        boundary_band_distance=float(boundary_band_distance),
        false_hole_sample_count=int(false_hole_sample_count),
        false_hole_boundary_band=float(false_hole_boundary_band),
        false_hole_material_threshold=float(false_hole_material_threshold),
        false_hole_dark_margin=float(false_hole_dark_margin),
        surface_material_gate_sigma=surface_material_gate_sigma,
        material_compose_mode=material_compose_mode,
        config=config,
        use_unified_compositor=use_unified_compositor,
    )

    if surface_count == 0:
        return metrics

    support_mask = analysis.get("coarse_support_mask", analysis.get("material_mask"))
    if support_mask is not None:
        support_tensor = as_device_tensor(support_mask, device=device, dtype=dtype)
        support_volume = support_tensor.reshape(1, 1, *tuple(int(value) for value in support_tensor.shape[-3:]))
        sampled_support = sample_volume_field(support_volume, surface_xyz, spacing_zyx).reshape(-1)
        metrics["surface_outside_support_ratio"] = float((sampled_support < 0.5).float().mean().item())

    boundary_points = analysis.get("boundary_points")
    if boundary_points is not None:
        if isinstance(boundary_points, torch.Tensor):
            boundary_np = boundary_points.detach().cpu().numpy()
        else:
            boundary_np = np.asarray(boundary_points)
        boundary_np = np.asarray(boundary_np, dtype=np.float32).reshape(-1, 3)
        if boundary_np.shape[0] > 0:
            surface_np = surface_xyz.detach().cpu().numpy()
            distances, _ = cKDTree(boundary_np).query(surface_np, k=1)
            for suffix, value in _quantile_metrics(distances, ("p50", "p90", "p99")).items():
                metrics[f"surface_to_phase1_boundary_distance_{suffix}"] = value

    raw_scaling = as_device_tensor(
        getattr(training_state, "surface_raw_scaling", torch.empty((0, 3), device=device)),
        device=device,
        dtype=dtype,
        reshape=(-1, 3),
    )
    if raw_scaling.shape[0] == surface_count:
        surface_max_scale = torch.exp(raw_scaling).max(dim=1).values.detach().cpu().numpy()
        for suffix, value in _quantile_metrics(surface_max_scale, ("p90", "p99")).items():
            metrics[f"surface_max_scale_{suffix}"] = value

    surface_opacity = as_device_tensor(
        getattr(training_state, "surface_opacity", torch.empty((0,), device=device)),
        device=device,
        dtype=dtype,
        reshape=(-1,),
    )
    if surface_opacity.shape[0] == surface_count:
        for suffix, value in _quantile_metrics(surface_opacity.detach().cpu().numpy(), ("p50", "p90")).items():
            metrics[f"surface_opacity_{suffix}"] = value

    boundary_points_for_coverage = analysis.get("boundary_points")
    if boundary_points_for_coverage is not None and getattr(training_state, "xyz", torch.empty((0, 3))).numel() > 0:
        with torch.no_grad():
            anchors = as_device_tensor(boundary_points_for_coverage, device=device, dtype=dtype).reshape(-1, 3)
            if anchors.shape[0] > 4096:
                indices = torch.linspace(0, anchors.shape[0] - 1, 4096, device=device).to(dtype=torch.long)
                anchors = anchors.index_select(0, indices)
            if anchors.numel() > 0:
                bulk_occ = density_to_occupancy(
                    query_ct_density_from_state_by_region(training_state, anchors, region="bulk", detach=True)
                ).to(dtype=torch.float32)
                surface_occ = density_to_occupancy(
                    query_ct_density_from_state_by_region(training_state, anchors, region="surface", detach=True)
                ).to(dtype=torch.float32)
                gap_mask = surface_occ <= 0.15
                metrics["surface_coverage_gap_ratio"] = float(gap_mask.float().mean().item())

                high_mask = torch.ones_like(gap_mask, dtype=torch.bool)
                if curvature_field is not None:
                    curvature = sample_volume_field(
                        curvature_field["curvature"],
                        anchors,
                        curvature_field["spacing_zyx"],
                    ).reshape(-1).to(dtype=torch.float32)
                    curvature = torch.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
                    if curvature.numel() > 0 and bool(torch.any(curvature > 0.0).item()):
                        high_mask = curvature >= torch.quantile(curvature, 0.75)
                elif "boundary_strength" in analysis:
                    strength = as_device_tensor(analysis["boundary_strength"], device=device, dtype=torch.float32).reshape(-1)
                    if strength.shape[0] > 4096:
                        strength = strength.index_select(0, torch.linspace(0, strength.shape[0] - 1, 4096, device=device).to(dtype=torch.long))
                    if strength.shape[0] == anchors.shape[0] and bool(torch.any(strength > 0.0).item()):
                        high_mask = strength >= torch.quantile(strength, 0.75)
                if torch.any(high_mask):
                    bulk_owned = (bulk_occ >= 0.35) & gap_mask
                    metrics["high_curvature_bulk_owned_ratio"] = float(bulk_owned[high_mask].float().mean().item())

    return metrics

def _save_surface_drift_diagnostics(
    training_state,
    analysis,
    spacing_zyx,
    model_path,
    iteration,
    tb_writer=None,
    curvature_field=None,
    surface_reseed_added: int = 0,
    bulk_reseed_stats=None,
    bulk_prune_stats=None,
    densify_stats=None,
    boundary_band_distance: float = 1.5,
    signed_distance_field=None,
    volume_cuda=None,
    intensity_air: float = 0.0,
    intensity_mat: float = 1.0,
    false_hole_sample_count: int = 4096,
    false_hole_boundary_band: float = 2.0,
    false_hole_material_threshold: float = 0.65,
    false_hole_dark_margin: float = 0.15,
    surface_material_gate_sigma=None,
    material_compose_mode: str = "bulk_first_material",
    config=None,
    use_unified_compositor: bool = True,
):
    metrics = _compute_surface_drift_diagnostics(
        training_state,
        analysis,
        spacing_zyx,
        curvature_field=curvature_field,
        surface_reseed_added=surface_reseed_added,
        bulk_reseed_stats=bulk_reseed_stats,
        bulk_prune_stats=bulk_prune_stats,
        densify_stats=densify_stats,
        boundary_band_distance=boundary_band_distance,
        signed_distance_field=signed_distance_field,
        volume_cuda=volume_cuda,
        intensity_air=intensity_air,
        intensity_mat=intensity_mat,
        false_hole_sample_count=false_hole_sample_count,
        false_hole_boundary_band=false_hole_boundary_band,
        false_hole_material_threshold=false_hole_material_threshold,
        false_hole_dark_margin=false_hole_dark_margin,
        surface_material_gate_sigma=surface_material_gate_sigma,
        material_compose_mode=material_compose_mode,
        config=config,
        use_unified_compositor=use_unified_compositor,
    )
    output_dir = Path(model_path).resolve() / "diagnostics"
    metrics_path = output_dir / f"drift_iter_{int(iteration):06d}.txt"

    ordered_keys = (
        "total_count",
        "bulk_count",
        "surface_count",
        "surface_outside_support_ratio",
        "surface_to_phase1_boundary_distance_p50",
        "surface_to_phase1_boundary_distance_p90",
        "surface_to_phase1_boundary_distance_p99",
        "surface_max_scale_p90",
        "surface_max_scale_p99",
        "surface_opacity_p50",
        "surface_opacity_p90",
        "surface_coverage_gap_ratio",
        "bulk_interior_coverage_gap_ratio",
        "bulk_interior_occupancy_p10",
        "bulk_interior_occupancy_p50",
        "bulk_material_coverage_gap_ratio",
        "bulk_material_occupancy_p10",
        "bulk_material_occupancy_p50",
        "bulk_material_sample_count",
        "num_uncovered_components",
        "max_uncovered_component_voxels",
        "bulk_material_A_b_p10",
        "bulk_material_A_b_p50",
        "bulk_material_A_b_p90",
        "material_volume_MAE",
        "material_volume_RMSE",
        "material_volume_corr",
        "material_volume_A_b_p10",
        "material_volume_A_b_p50",
        "material_volume_A_b_p90",
        "material_volume_sample_count",
        "high_gradient_material_MAE",
        "high_gradient_material_sample_count",
        "combined_material_coverage_gap_ratio",
        "combined_material_occ_union_raw_p10",
        "combined_material_occ_union_raw_p50",
        "combined_material_sample_count",
        "bulk_deep_coverage_gap_ratio",
        "bulk_deep_occupancy_p10",
        "bulk_deep_occupancy_p50",
        "bulk_deep_sample_count",
        "bulk_deep_sdf_1p5_3_gap_ratio",
        "bulk_deep_sdf_1p5_3_sample_count",
        "bulk_deep_sdf_3_6_gap_ratio",
        "bulk_deep_sdf_3_6_sample_count",
        "bulk_deep_sdf_6_plus_gap_ratio",
        "bulk_deep_sdf_6_plus_sample_count",
        "bulk_deep_low_occ_sdf_depth_p50",
        "bulk_deep_low_occ_sdf_depth_p90",
        "bulk_deep_low_occ_nearest_bulk_distance_p50",
        "bulk_deep_low_occ_nearest_bulk_distance_p90",
        "material_deep_occ_b_raw_p10",
        "material_deep_occ_b_raw_p50",
        "material_boundary_shell_occ_b_raw_p10",
        "material_boundary_shell_occ_b_raw_p50",
        "cavity_material_shell_occ_b_raw_p10",
        "cavity_material_shell_occ_b_raw_p50",
        "void_air_occ_b_raw_p95",
        "void_air_A_b_p95",
        "void_air_sample_count",
        "exterior_air_near_occ_b_raw_p95",
        "exterior_air_near_A_b_p95",
        "exterior_air_near_sample_count",
        "boundary_miss_rate",
        "tau_current",
        "bulk_surface_gap_distance_p50",
        "bulk_surface_gap_distance_p90",
        "bulk_containment_violation_ratio",
        "surface_mat_shell_mae",
        "surface_mat_shell_sample_count",
        "surface_air_shell_leak",
        "surface_air_shell_sample_count",
        "false_hole_candidate_count",
        "false_hole_active_ratio",
        "false_hole_pred_intensity_p10",
        "false_hole_pred_intensity_p50",
        "false_hole_target_intensity_p50",
        "false_hole_bulk_occ_p10",
        "false_hole_bulk_occ_p50",
        "cavity_void_false_fill_ratio",
        "surface_owned_bulk_W_p10",
        "surface_owned_bulk_W_p50",
        "surface_owned_bulk_mu_raw_p10",
        "surface_owned_bulk_mu_raw_p50",
        "surface_owned_bulk_A_b_p10",
        "surface_owned_bulk_A_b_p50",
        "surface_owned_false_hole_ratio",
        "high_curvature_bulk_owned_ratio",
        "surface_reseed_added",
        "bulk_reseed_added",
        "bulk_reseed_candidates",
        "bulk_reseed_low_coverage_ratio",
        "bulk_reseed_sigma_init_mean",
        "bulk_reseed_atten_init_mean",
        "bulk_gap_grown_count",
        "bulk_repair_residual_mass",
        "bulk_repair_stretched_count",
        "bulk_repair_skipped_small_components",
        "bulk_repair_skipped_exclusion",
        "bulk_repair_skipped_low_gain",
        "bulk_repair_skipped_containment",
        "bulk_repair_skipped_overfill",
        "bulk_repair_skipped_no_clearance_headroom",
        "bulk_repair_clearance_limited",
        "bulk_repair_components_considered",
        "bulk_pruned_count",
        "bulk_prune_protected_gap_seed",
        "densify_surface_split",
        "densify_bulk_split",
        "densify_children_added",
        "densify_net_added",
    )
    write_key_value_report(
        metrics_path,
        [("iteration", int(iteration))] + [(key, metrics[key]) for key in ordered_keys],
    )

    print(
        "[ITER {0}] Drift diagnostics: outside={1:.4f} boundary_d(p50/p90/p99)=({2:.4f}, {3:.4f}, {4:.4f}) "
        "scale(p90/p99)=({5:.4f}, {6:.4f}) opacity(p50/p90)=({7:.4f}, {8:.4f})".format(
            int(iteration),
            float(metrics["surface_outside_support_ratio"]),
            float(metrics["surface_to_phase1_boundary_distance_p50"]),
            float(metrics["surface_to_phase1_boundary_distance_p90"]),
            float(metrics["surface_to_phase1_boundary_distance_p99"]),
            float(metrics["surface_max_scale_p90"]),
            float(metrics["surface_max_scale_p99"]),
            float(metrics["surface_opacity_p50"]),
            float(metrics["surface_opacity_p90"]),
        )
    )
    if tb_writer:
        for key, value in metrics.items():
            if key == "surface_count":
                continue
            if isinstance(value, float) and np.isfinite(value):
                tb_writer.add_scalar(f"ct_drift/{key}", value, int(iteration))
    return {"drift_diagnostics": str(metrics_path), **metrics}
