from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ct_pipeline.backend.core import prepare_ct_training_state
from ct_pipeline.data.loader import CTVolumeLoader
from ct_pipeline.rendering.bulk_support import resolve_bulk_query_truncation_sigma
from ct_pipeline.training.bootstrap.analysis import (
    _ct_spatial_extent,
    _empty_support_distance_field,
    _ensure_intensity_driven_analysis,
    _load_ct_analysis_bundle,
    _prepare_curvature_proxy_field,
    _prepare_intensity_calibration,
    _prepare_signed_distance_field,
    _prepare_support_distance_field,
    _require_active_boundary_bundle,
    _to_cuda_analysis,
)
from ct_pipeline.training.bootstrap.context import CTTrainingBootstrap
from ct_pipeline.training.control.optimizer import _freeze_ct_feature_params
from ct_pipeline.training.objectives.modes import _boundary_band_distance, _bulk_intensity_training_enabled
from ct_pipeline.training.sampling import (
    _prepare_field_sample_pools,
    _resolve_air_sampling_candidates,
    _resolve_field_sample_counts,
    precompute_sdf_filtered_field_pools,
)
from ct_pipeline.training.session import build_renderer_autocast_kwargs, prepare_output_and_logger
from ct_pipeline.training.utils import as_device_tensor, write_key_value_report
from scene.ct_gaussian_model import CTGaussianModel

def _run_ct_init_preflight(
    gaussians,
    analysis,
    spacing_zyx,
    signed_distance_field,
    volume_cuda,
    intensity_air: float,
    intensity_mat: float,
    args,
):
    training_state = prepare_ct_training_state(
        gaussians,
        spacing_zyx=spacing_zyx,
        truncation_sigma=float(getattr(args, "ct_gaussian_truncation_sigma", 4.0)),
        bulk_truncation_sigma=resolve_bulk_query_truncation_sigma(args),
        grid_cell_voxels=int(getattr(args, "ct_grid_cell_voxels", 8)),
        signed_distance_field=signed_distance_field,
    )
    from ct_pipeline.training.reporting.surface_drift import _compute_surface_drift_diagnostics

    metrics = _compute_surface_drift_diagnostics(
        training_state,
        analysis,
        spacing_zyx,
        signed_distance_field=signed_distance_field,
        volume_cuda=volume_cuda,
        intensity_air=float(intensity_air),
        intensity_mat=float(intensity_mat),
        false_hole_sample_count=1024,
        boundary_band_distance=float(getattr(args, "ct_boundary_band", 1.5)),
        config=args,
        use_unified_compositor=True,
    )
    init_stats = getattr(gaussians, "ct_feature_adaptive_init_stats", {}) or {}
    preflight = {
        "initial_bulk_count": int(getattr(training_state, "bulk_xyz", torch.empty((0, 3))).shape[0]),
        "initial_containment_violation_ratio": float(metrics.get("bulk_containment_violation_ratio", float("nan"))),
        "initial_material_volume_A_b_p10": float(metrics.get("material_volume_A_b_p10", float("nan"))),
        "initial_material_coverage_gap": float(metrics.get("bulk_material_coverage_gap_ratio", float("nan"))),
        "initial_num_uncovered_components": int(metrics.get("num_uncovered_components", 0)),
        "initial_max_uncovered_component_voxels": int(metrics.get("max_uncovered_component_voxels", 0)),
        "num_init_candidates_shrunk": int(init_stats.get("num_init_candidates_shrunk", 0)),
        "num_init_candidates_downgraded": int(init_stats.get("num_init_candidates_downgraded", 0)),
        "num_init_candidates_rejected": int(init_stats.get("num_init_candidates_rejected", 0)),
    }
    print(
        "[CT init preflight] containment={0:.6f} material_A_b_p10={1:.6f} "
        "material_gap={2:.6f} components={3} max_component={4} "
        "shrunk={5} downgraded={6} rejected={7}".format(
            preflight["initial_containment_violation_ratio"],
            preflight["initial_material_volume_A_b_p10"],
            preflight["initial_material_coverage_gap"],
            preflight["initial_num_uncovered_components"],
            preflight["initial_max_uncovered_component_voxels"],
            preflight["num_init_candidates_shrunk"],
            preflight["num_init_candidates_downgraded"],
            preflight["num_init_candidates_rejected"],
        )
    )
    diagnostics_dir = Path(args.model_path) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    max_violation = float(getattr(args, "ct_init_preflight_max_containment_violation", 0.01))
    min_material_a_b = float(getattr(args, "ct_init_preflight_min_material_a_b_p10", 0.70))
    max_material_gap = float(getattr(args, "ct_init_preflight_max_material_coverage_gap", 0.001))
    gate_checks = (
        (
            "initial_containment_violation_ratio",
            preflight["initial_containment_violation_ratio"],
            "<=",
            max_violation,
        ),
        (
            "initial_material_volume_A_b_p10",
            preflight["initial_material_volume_A_b_p10"],
            ">=",
            min_material_a_b,
        ),
        (
            "initial_material_coverage_gap",
            preflight["initial_material_coverage_gap"],
            "<=",
            max_material_gap,
        ),
    )
    failed_checks = []
    for key, value, op, threshold in gate_checks:
        passed = (
            math.isfinite(value)
            and ((value <= threshold) if op == "<=" else (value >= threshold))
        )
        preflight[f"{key}_gate_pass"] = bool(passed)
        if not passed:
            failed_checks.append(f"{key}={value:.6f} {op} {threshold:.6f}")
    preflight["initial_preflight_gate_pass"] = not failed_checks
    if failed_checks:
        preflight["initial_preflight_gate_failure"] = "; ".join(failed_checks)
    rows = list(preflight.items())
    rows.extend((f"raw_{key}", value) for key, value in metrics.items())
    rows.extend((f"init_{key}", value) for key, value in init_stats.items())
    write_key_value_report(diagnostics_dir / "init_preflight.txt", rows)
    if bool(getattr(args, "ct_init_preflight_abort", False)):
        if failed_checks:
            raise RuntimeError(
                "CT init preflight failed: " + "; ".join(failed_checks)
            )
    return preflight


def prepare_ct_training_bootstrap(dataset, opt, args, checkpoint) -> CTTrainingBootstrap:
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    loader = CTVolumeLoader()
    volume_np = loader.load(args.ct_volume_path, fmt=args.ct_volume_format, raw_meta_path=args.ct_raw_meta)
    spacing_zyx = loader.get_voxel_spacing()
    volume_cuda = as_device_tensor(volume_np, device="cuda", dtype=torch.float32)
    volume_shape = tuple(int(value) for value in volume_np.shape)
    analysis, metadata, analysis_path, metadata_path = _load_ct_analysis_bundle(args.ct_phase1_dir)
    analysis = _ensure_intensity_driven_analysis(analysis)
    _require_active_boundary_bundle(analysis)
    analysis_gpu = _to_cuda_analysis(analysis)

    spatial_lr_scale = _ct_spatial_extent(volume_np.shape, spacing_zyx)
    gaussians = CTGaussianModel(dataset.sh_degree)
    gaussians.create_from_phase1_bundle(
        analysis_path,
        metadata_path,
        spatial_lr_scale=spatial_lr_scale,
        surface_thickness_max=args.ct_thickness_max,
        bulk_continuous_init=args.ct_bulk_continuous_init,
        bulk_augment_factor=float(getattr(args, "ct_bulk_augment_factor", 2.0)),
        bulk_init_mode=str(getattr(args, "ct_bulk_init_mode", "sparse_reseed")),
        bulk_lattice_spacing_vox=float(getattr(args, "ct_bulk_lattice_spacing_vox", 0.5)),
        bulk_lattice_sigma_vox=float(getattr(args, "ct_bulk_lattice_sigma_vox", 0.5)),
        bulk_lattice_margin_vox=float(getattr(args, "ct_bulk_lattice_margin_vox", 0.25)),
        bulk_lattice_atten_init=float(getattr(args, "ct_bulk_lattice_atten_init", 0.75)),
        bulk_lattice_anisotropic=bool(getattr(args, "ct_bulk_lattice_anisotropic", False)),
        bulk_lattice_sigma_t_vox=float(getattr(args, "ct_bulk_lattice_sigma_t_vox", 2.0)),
        bulk_lattice_sigma_n_vox=float(getattr(args, "ct_bulk_lattice_sigma_n_vox", 0.8)),
        intensity_volume=volume_np,
        feature_adaptive_jitter=bool(getattr(args, "ct_feature_adaptive_jitter", True)),
        feature_adaptive_seed=int(getattr(args, "ct_feature_adaptive_seed", 17)),
        feature_adaptive_r_shell_vox=float(getattr(args, "ct_feature_adaptive_r_shell_vox", 3.0)),
        feature_adaptive_blur_sigma_vox=float(getattr(args, "ct_feature_adaptive_blur_sigma_vox", 0.75)),
        feature_adaptive_spacing_high_vox=int(getattr(args, "ct_feature_adaptive_spacing_high_vox", 2)),
        feature_adaptive_spacing_mid_vox=int(getattr(args, "ct_feature_adaptive_spacing_mid_vox", 6)),
        feature_adaptive_spacing_low_vox=int(getattr(args, "ct_feature_adaptive_spacing_low_vox", 10)),
        feature_adaptive_directional_clearance=bool(
            getattr(args, "ct_feature_adaptive_directional_clearance", True)
        ),
        feature_adaptive_probe_containment=bool(getattr(args, "ct_feature_adaptive_probe_containment", True)),
    )
    gaussians.initialize_ct_value_from_volume(volume_np, spacing_zyx)

    gaussians.spatial_lr_scale = spatial_lr_scale
    opt.ct_freeze_primitive_type = args.ct_freeze_primitive_type
    opt.surface_thickness_max = args.ct_thickness_max
    opt.planar_thickness_max = args.ct_thickness_max
    gaussians.training_setup(opt)
    _freeze_ct_feature_params(gaussians)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
        _freeze_ct_feature_params(gaussians)

    renderer_autocast_kwargs = build_renderer_autocast_kwargs()
    field_pools = _prepare_field_sample_pools(
        analysis_gpu,
        volume_shape,
        args.ct_bulk_boundary_margin_voxels,
        device="cuda",
        spacing_zyx=spacing_zyx,
    )
    support_sample_count, air_sample_count = _resolve_field_sample_counts(args, gaussians.get_xyz.shape[0])
    print(
        "CT field sampling budget: support={0} air={1} total_gaussians={2}".format(
            support_sample_count,
            air_sample_count,
            int(gaussians.get_xyz.shape[0]),
        )
    )
    print(
        "CT sampling mix: exterior_air_max_ratio={0:.2f} bulk_volume_support_ratio={1:.2f} "
        "surface_coverage_weight={2:.2f} target={3:.2f} bulk_coverage_weight={4:.2f} target={5:.2f}".format(
            float(getattr(args, "ct_exterior_air_sample_ratio", 0.5)),
            float(getattr(args, "ct_bulk_volume_support_sample_ratio", 0.5)),
            float(getattr(args, "ct_surface_coverage_weight", 0.0)),
            float(getattr(args, "ct_surface_coverage_target", 0.75)),
            float(getattr(args, "ct_bulk_coverage_weight", 0.0)),
            float(getattr(args, "ct_bulk_coverage_target", 0.85)),
        )
    )
    air_focus_source = field_pools["air_shell"] if field_pools["air_shell"].shape[0] > 0 else field_pools["air"]
    if air_focus_source.shape[0] > 0:
        max_focus_indices = min(int(air_focus_source.shape[0]), 65536)
        if air_focus_source.shape[0] > max_focus_indices:
            if isinstance(air_focus_source, torch.Tensor):
                chosen = torch.randperm(int(air_focus_source.shape[0]), device=air_focus_source.device)[:max_focus_indices]
                air_focus_source = air_focus_source.index_select(0, chosen)
            else:
                chosen = np.random.choice(int(air_focus_source.shape[0]), size=max_focus_indices, replace=False)
                air_focus_source = air_focus_source[chosen]
        analysis_gpu["_air_focus_indices"] = torch.as_tensor(air_focus_source, dtype=torch.long, device="cuda")
    else:
        analysis_gpu["_air_focus_indices"] = torch.empty((0, 3), dtype=torch.long, device="cuda")
    signed_distance_field = _prepare_signed_distance_field(
        analysis,
        spacing_zyx,
        device="cuda",
        boundary_mode=str(getattr(args, "ct_sdf_boundary_mode", "interface")),
    )
    if _bulk_intensity_training_enabled(args):
        support_distance_field = _empty_support_distance_field(spacing_zyx, device="cuda")
        print("CT support EDT skipped for bulk intensity mode.")
    else:
        support_distance_field = _prepare_support_distance_field(analysis, spacing_zyx, device="cuda")
    intensity_field_cache = {}
    reseed_uses_curvature = bool(getattr(args, "ct_enable_surface_reseeding", False))
    if reseed_uses_curvature:
        try:
            curvature_proxy = _prepare_curvature_proxy_field(volume_np, spacing_zyx, sigma=1.0, device="cuda")
        except (MemoryError, RuntimeError, np.core._exceptions._ArrayMemoryError) as exc:
            curvature_proxy = None
            print(
                "CT curvature proxy field unavailable ({0}); surface reseeding will use uniform boundary anchors.".format(
                    exc.__class__.__name__,
                )
            )
        if curvature_proxy is not None:
            intensity_field_cache["curvature_proxy"] = curvature_proxy
            print("CT curvature proxy field prepared for surface reseeding diagnostics.")
    else:
        print("CT surface geometry uses coarse/Phase1 SDF.")
    intensity_air, intensity_mat = _prepare_intensity_calibration(analysis, volume_np)
    print(
        "CT intensity calibration: I_air={0:.6f} I_mat={1:.6f}".format(
            intensity_air,
            intensity_mat,
        )
    )
    _run_ct_init_preflight(
        gaussians,
        analysis,
        spacing_zyx,
        signed_distance_field,
        volume_cuda,
        intensity_air,
        intensity_mat,
        args,
    )
    initial_gaussian_count = int(gaussians.get_xyz.shape[0])

    preferred_air_candidates, air_shell_ratio, preferred_air_is_band = _resolve_air_sampling_candidates(field_pools)
    if preferred_air_is_band:
        print(
            "Air sampling: using near-boundary shell subset because configured-band fraction is low ({0:.3f}).".format(
                air_shell_ratio,
            )
        )
    else:
        print(
            "Air sampling: keeping current shell sampling because configured-band fraction is already high ({0:.3f}).".format(
                air_shell_ratio,
            )
        )
    field_pools = precompute_sdf_filtered_field_pools(
        field_pools,
        signed_distance_field,
        _boundary_band_distance(args),
        preferred_air_candidates=preferred_air_candidates,
    )

    if str(getattr(args, "ct_init_strategy", "volume_sampled")) == "coverage_first":
        from scene.coverage_first_init import apply_coverage_first_init
        init_stats = apply_coverage_first_init(
            gaussians,
            volume_cuda,
            spacing_zyx,
            signed_distance_field,
            args,
            model_path=getattr(dataset, "model_path", None),
        )
        if init_stats.get("applied"):
            print(
                "[CT v5.2.1] coverage-first init applied: n_bulk={0} n_deep={1} n_boundary={2} "
                "C_mat_p10={3:.4f} C_shell_p10={4:.4f} C_void_p95={5:.4f} sigma_p50={6:.4f} atten_p50={7:.4f}".format(
                    init_stats.get("n_bulk", 0),
                    init_stats.get("n_deep", 0),
                    init_stats.get("n_boundary", 0),
                    init_stats.get("coverage_p10_material", float("nan")),
                    init_stats.get("coverage_p10_shell", float("nan")),
                    init_stats.get("coverage_p95_void", float("nan")),
                    init_stats.get("sigma_p50", float("nan")),
                    init_stats.get("atten_p50", float("nan")),
                )
            )
        else:
            print("[CT v5.2.1] coverage-first init skipped: {0}".format(init_stats.get("reason", "unknown")))

    del metadata
    return CTTrainingBootstrap(
        first_iter=first_iter,
        tb_writer=tb_writer,
        loader=loader,
        volume_np=volume_np,
        volume_cuda=volume_cuda,
        volume_shape=volume_shape,
        spacing_zyx=spacing_zyx,
        analysis=analysis,
        analysis_gpu=analysis_gpu,
        analysis_path=analysis_path,
        metadata_path=metadata_path,
        gaussians=gaussians,
        renderer_autocast_kwargs=renderer_autocast_kwargs,
        field_pools=field_pools,
        support_sample_count=support_sample_count,
        air_sample_count=air_sample_count,
        support_distance_field=support_distance_field,
        signed_distance_field=signed_distance_field,
        intensity_field_cache=intensity_field_cache,
        intensity_air=intensity_air,
        intensity_mat=intensity_mat,
        initial_gaussian_count=initial_gaussian_count,
        preferred_air_candidates=preferred_air_candidates,
        exterior_air_sample_ratio=float(getattr(args, "ct_exterior_air_sample_ratio", 0.5)),
    )
