from .defaults import *
from .normalization import _normalize_ct_training_args


def _validate_path_and_io_args(args) -> None:
    if not args.ct_phase1_dir:
        raise ValueError("--ct_phase1_dir is required for train_ct.py.")
    if not args.ct_volume_path:
        raise ValueError("--ct_volume_path is required for train_ct.py.")
    if args.ct_preview_interval < 0:
        raise ValueError("--ct_preview_interval must be >= 0.")
    if args.ct_volume_format == "raw" and not args.ct_raw_meta:
        raise ValueError("--ct_raw_meta is required for RAW CT volumes.")
    if args.load_ply:
        raise ValueError("--load_ply is not supported in CT training mode. Use --start_checkpoint to resume.")

def _validate_core_training_args(args) -> None:
    if args.ct_bulk_boundary_margin_voxels < 0:
        raise ValueError("--ct_bulk_boundary_margin_voxels must be >= 0.")
    if args.ct_support_sample_count is not None and args.ct_support_sample_count < 1:
        raise ValueError("--ct_support_sample_count must be >= 1 when provided.")
    if args.ct_air_sample_count is not None and args.ct_air_sample_count < 1:
        raise ValueError("--ct_air_sample_count must be >= 1 when provided.")
    if args.ct_volume_sample_count < 1:
        raise ValueError("--ct_volume_sample_count must be >= 1.")
    if args.ct_volume_jitter < 0.0:
        raise ValueError("--ct_volume_jitter must be >= 0.")
    if args.ct_boundary_band <= 0.0:
        raise ValueError("--ct_boundary_band must be > 0.")
    if args.ct_surface_boundary_sample_ratio < 0.0 or args.ct_surface_boundary_sample_ratio > 1.0:
        raise ValueError("--ct_surface_boundary_sample_ratio must be in [0, 1].")
    if (
        args.ct_lambda_volume < 0.0
        or args.ct_lambda_occupancy < 0.0
        or args.ct_surface_regularizer_weight < 0.0
    ):
        raise ValueError("CT loss weights must be >= 0.")
    if args.ct_bulk_field_mode != CT_BULK_FIELD_MODE:
        raise ValueError("bulk field mode is fixed to 'bulk_intensity_field'.")
    if args.ct_bulk_halfspace_tau_init <= 0.0 or args.ct_bulk_halfspace_tau_final <= 0.0:
        raise ValueError("CT bulk half-space tau values must be > 0.")
    if args.ct_bulk_halfspace_tau_final > args.ct_bulk_halfspace_tau_init:
        raise ValueError("--ct_bulk_halfspace_tau_final must be <= --ct_bulk_halfspace_tau_init.")
    if args.ct_bulk_halfspace_tau_gate_threshold < 0.0:
        raise ValueError("--ct_bulk_halfspace_tau_gate_threshold must be >= 0.")
    if args.ct_bulk_halfspace_tau_step < 0.0:
        raise ValueError("--ct_bulk_halfspace_tau_step must be >= 0.")
    if args.ct_bulk_halfspace_tau_update_interval < 1:
        raise ValueError("--ct_bulk_halfspace_tau_update_interval must be >= 1.")
    if args.ct_bulk_halfspace_skip_depth < 0.0:
        raise ValueError("--ct_bulk_halfspace_skip_depth must be >= 0.")
    if args.ct_occupancy_sample_count < 1:
        raise ValueError("--ct_occupancy_sample_count must be >= 1.")
    if args.ct_occ_boundary_sample_ratio < 0.0 or args.ct_occ_boundary_sample_ratio > 1.0:
        raise ValueError("--ct_occ_boundary_sample_ratio must be in [0, 1].")
    if args.ct_occ_deep_material_sample_ratio < 0.0 or args.ct_occ_deep_material_sample_ratio > 1.0:
        raise ValueError("--ct_occ_deep_material_sample_ratio must be in [0, 1].")
    if args.ct_occ_boundary_weight < 1.0:
        raise ValueError("--ct_occ_boundary_weight must be >= 1.")
    if args.ct_exterior_air_sample_ratio < 0.0 or args.ct_exterior_air_sample_ratio > 1.0:
        raise ValueError("--ct_exterior_air_sample_ratio must be in [0, 1].")
    if args.ct_bulk_volume_support_sample_ratio < 0.0 or args.ct_bulk_volume_support_sample_ratio > 1.0:
        raise ValueError("--ct_bulk_volume_support_sample_ratio must be in [0, 1].")
    if float(getattr(args, "ct_value_lr", 1e-3)) < 0.0:
        raise ValueError("--ct_value_lr must be >= 0.")
    if float(getattr(args, "ct_attenuation_lr", getattr(args, "ct_value_lr", 1e-3))) < 0.0:
        raise ValueError("--ct_attenuation_lr must be >= 0.")

def _validate_surface_and_bulk_args(args) -> None:
    if args.ct_bulk_max_scale <= 0.0:
        raise ValueError("--ct_bulk_max_scale must be > 0.")
    if args.ct_bulk_augment_factor <= 0.0:
        raise ValueError("--ct_bulk_augment_factor must be > 0.")
    if args.ct_surface_sigma_n_max <= 0.0:
        raise ValueError("--ct_surface_sigma_n_max must be > 0.")
    if args.ct_surface_sigma_t_min < 0.0:
        raise ValueError("--ct_surface_sigma_t_min must be >= 0.")
    if args.ct_surface_max_scale <= 0.0:
        raise ValueError("--ct_surface_max_scale must be > 0.")
    if args.ct_bulk_scale_global_max <= 0.0:
        raise ValueError("--ct_bulk_scale_global_max must be > 0.")
    if args.ct_bulk_scale_floor <= 0.0:
        raise ValueError("--ct_bulk_scale_floor must be > 0.")
    if args.ct_surface_normal_weight < 0.0:
        raise ValueError("--ct_surface_normal_weight must be >= 0.")
    if args.ct_surface_thickness_weight < 0.0:
        raise ValueError("--ct_surface_thickness_weight must be >= 0.")
    if args.ct_surface_coverage_weight < 0.0:
        raise ValueError("--ct_surface_coverage_weight must be >= 0.")
    if args.ct_surface_coverage_sample_count < 0:
        raise ValueError("--ct_surface_coverage_sample_count must be >= 0.")
    if args.ct_surface_coverage_target < 0.0 or args.ct_surface_coverage_target > 1.0:
        raise ValueError("--ct_surface_coverage_target must be in [0, 1].")
    if args.ct_surface_coverage_until_iter < 0:
        raise ValueError("--ct_surface_coverage_until_iter must be >= 0.")
    if args.ct_surface_material_gate_sigma is not None and args.ct_surface_material_gate_sigma <= 0.0:
        raise ValueError("--ct_surface_material_gate_sigma must be > 0 when provided.")
    if args.ct_surface_material_tau <= 0.0:
        raise ValueError("--ct_surface_material_tau must be > 0.")
    if args.ct_surface_material_render_weight < 0.0:
        raise ValueError("--ct_surface_material_render_weight must be >= 0.")
    if args.ct_surface_intensity_weight < 0.0:
        raise ValueError("--ct_surface_intensity_weight must be >= 0.")
    if args.ct_surface_intensity_sample_count < 0:
        raise ValueError("--ct_surface_intensity_sample_count must be >= 0.")
    if args.ct_surface_intensity_material_ratio < 0.0 or args.ct_surface_intensity_material_ratio > 1.0:
        raise ValueError("--ct_surface_intensity_material_ratio must be in [0, 1].")
    if args.ct_dual_surface_inner_band < 0.0:
        raise ValueError("--ct_dual_surface_inner_band must be >= 0.")
    if args.ct_dual_surface_air_band < 0.0:
        raise ValueError("--ct_dual_surface_air_band must be >= 0.")
    if args.ct_dual_soft_sigma <= 0.0:
        raise ValueError("--ct_dual_soft_sigma must be > 0.")
    if args.ct_dual_surface_air_sample_ratio < 0.0 or args.ct_dual_surface_air_sample_ratio > 1.0:
        raise ValueError("--ct_dual_surface_air_sample_ratio must be in [0, 1].")
    if args.ct_bulk_renderer_epsilon < 0.0:
        raise ValueError("--ct_bulk_renderer_epsilon must be >= 0.")
    if args.ct_bulk_renderer_tau <= 0.0:
        raise ValueError("--ct_bulk_renderer_tau must be > 0.")
    if args.ct_material_compose_mode not in {"surface_first", "bulk_first_material"}:
        raise ValueError("--ct_material_compose_mode must be one of {'surface_first', 'bulk_first_material'}.")
    if args.ct_bulk_semantic_weight < 0.0:
        raise ValueError("--ct_bulk_semantic_weight must be >= 0.")
    if args.ct_bulk_semantic_sample_count < 0:
        raise ValueError("--ct_bulk_semantic_sample_count must be >= 0.")
    if args.ct_bulk_semantic_focal_gamma < 0.0:
        raise ValueError("--ct_bulk_semantic_focal_gamma must be >= 0.")
    if args.ct_bulk_semantic_epsilon < 0.0:
        raise ValueError("--ct_bulk_semantic_epsilon must be >= 0.")
    ratio_attrs = (
        "ct_bulk_semantic_pos_ratio",
        "ct_bulk_semantic_deep_ratio",
        "ct_bulk_semantic_boundary_ratio",
        "ct_bulk_semantic_cavity_material_ratio",
        "ct_bulk_semantic_void_air_ratio",
        "ct_bulk_semantic_cavity_air_ratio",
        "ct_bulk_semantic_exterior_air_ratio",
    )
    for attr in ratio_attrs:
        if getattr(args, attr) < 0.0:
            raise ValueError(f"--{attr} must be >= 0.")
    if args.ct_bulk_semantic_pos_ratio > 1.0:
        raise ValueError("--ct_bulk_semantic_pos_ratio must be <= 1.")
    if args.ct_focal_gamma_pos < 0.0 or args.ct_focal_gamma_neg < 0.0:
        raise ValueError("CT focal gamma values must be >= 0.")
    if args.ct_focal_alpha_pos < 0.0 or args.ct_focal_alpha_pos > 1.0:
        raise ValueError("--ct_focal_alpha_pos must be in [0, 1].")
    for attr in (
        "ct_bulk_semantic_boundary_target",
        "ct_bulk_semantic_cavity_target",
        "ct_bulk_floor_deep_target",
        "ct_bulk_floor_shell_target",
    ):
        if getattr(args, attr) < 0.0 or getattr(args, attr) > 1.0:
            raise ValueError(f"--{attr} must be in [0, 1].")
    if args.ct_bulk_floor_weight < 0.0:
        raise ValueError("--ct_bulk_floor_weight must be >= 0.")
    if args.ct_bulk_void_weight < 0.0:
        raise ValueError("--ct_bulk_void_weight must be >= 0.")
    if args.ct_loss_boundary_band_delta <= 0.0:
        raise ValueError("--ct_loss_boundary_band_delta must be > 0.")
    if args.ct_loss_band_in_weight < 0.0 or args.ct_loss_band_out_weight < 0.0:
        raise ValueError("CT boundary band weights must be >= 0.")
    if args.ct_bulk_coverage_weight < 0.0:
        raise ValueError("--ct_bulk_coverage_weight must be >= 0.")
    if args.ct_bulk_coverage_sample_count < 0:
        raise ValueError("--ct_bulk_coverage_sample_count must be >= 0.")
    if args.ct_bulk_coverage_target < 0.0 or args.ct_bulk_coverage_target > 1.0:
        raise ValueError("--ct_bulk_coverage_target must be in [0, 1].")
    if args.ct_bulk_air_exclusion_weight < 0.0:
        raise ValueError("--ct_bulk_air_exclusion_weight must be >= 0.")
    if args.ct_bulk_air_exclusion_sample_count < 0:
        raise ValueError("--ct_bulk_air_exclusion_sample_count must be >= 0.")
    if args.ct_bulk_air_exclusion_target < 0.0 or args.ct_bulk_air_exclusion_target > 1.0:
        raise ValueError("--ct_bulk_air_exclusion_target must be in [0, 1].")
    if args.ct_cavity_material_coverage_weight < 0.0:
        raise ValueError("--ct_cavity_material_coverage_weight must be >= 0.")
    if args.ct_cavity_false_hole_boost < 0.0:
        raise ValueError("--ct_cavity_false_hole_boost must be >= 0.")
    if args.ct_bulk_sdf_containment_weight < 0.0:
        raise ValueError("--ct_bulk_sdf_containment_weight must be >= 0.")
    if args.ct_containment_weight < 0.0:
        raise ValueError("--ct_containment_weight must be >= 0.")
    if args.ct_bulk_sdf_containment_sample_count < 0:
        raise ValueError("--ct_bulk_sdf_containment_sample_count must be >= 0.")
    if args.ct_bulk_sdf_containment_margin < 0.0:
        raise ValueError("--ct_bulk_sdf_containment_margin must be >= 0.")
    if not (0 <= args.ct_containment_ramp_iter_1 <= args.ct_containment_ramp_iter_2 <= args.ct_containment_ramp_iter_3):
        raise ValueError("CT containment ramp iterations must be ordered and >= 0.")
    if args.ct_init_strategy not in ("volume_sampled", "coverage_first"):
        raise ValueError("--ct_init_strategy must be 'volume_sampled' or 'coverage_first'.")
    if args.ct_init_boundary_sigma_c <= 0.0:
        raise ValueError("--ct_init_boundary_sigma_c must be > 0.")
    if args.ct_init_deep_sigma_voxel <= 0.0:
        raise ValueError("--ct_init_deep_sigma_voxel must be > 0.")
    if args.ct_init_deep_depth_voxel < 0.0:
        raise ValueError("--ct_init_deep_depth_voxel must be >= 0.")
    if args.ct_init_inward_nudge_sigma_ratio < 0.0:
        raise ValueError("--ct_init_inward_nudge_sigma_ratio must be >= 0.")
    if args.ct_init_coverage_knn_k < 0:
        raise ValueError("--ct_init_coverage_knn_k must be >= 0.")
    if args.ct_stage1_freeze_until_iter < 0:
        raise ValueError("--ct_stage1_freeze_until_iter must be >= 0.")
    for attr in ("ct_containment_ramp_weight_1", "ct_containment_ramp_weight_2", "ct_containment_ramp_weight_3"):
        if getattr(args, attr) < 0.0:
            raise ValueError("CT containment ramp weights must be >= 0.")
    if args.ct_false_hole_sample_count < 0:
        raise ValueError("--ct_false_hole_sample_count must be >= 0.")
    if args.ct_false_hole_boundary_band <= 0.0:
        raise ValueError("--ct_false_hole_boundary_band must be > 0.")
    if args.ct_false_hole_material_threshold < 0.0 or args.ct_false_hole_material_threshold > 1.0:
        raise ValueError("--ct_false_hole_material_threshold must be in [0, 1].")
    if args.ct_false_hole_dark_margin < 0.0:
        raise ValueError("--ct_false_hole_dark_margin must be >= 0.")
    if args.ct_false_hole_target_occupancy < 0.0 or args.ct_false_hole_target_occupancy > 1.0:
        raise ValueError("--ct_false_hole_target_occupancy must be in [0, 1].")
    if args.ct_false_hole_metrics_interval < 0:
        raise ValueError("--ct_false_hole_metrics_interval must be >= 0.")
    if args.ct_densify_from_iter < 0 or args.ct_densify_until_iter < 0:
        raise ValueError("CT densify iteration bounds must be >= 0.")
    if args.ct_densify_interval <= 0:
        raise ValueError("--ct_densify_interval must be > 0.")
    if args.ct_densify_until_iter < args.ct_densify_from_iter:
        raise ValueError("--ct_densify_until_iter must be >= --ct_densify_from_iter.")
    if args.ct_densify_surface_percent < 0.0 or args.ct_densify_bulk_percent < 0.0:
        raise ValueError("CT densify percents must be >= 0.")
    if args.ct_densify_max_gaussian_ratio < 1.0:
        raise ValueError("--ct_densify_max_gaussian_ratio must be >= 1.")
    if args.ct_densify_min_opacity < 0.0 or args.ct_densify_min_opacity > 1.0:
        raise ValueError("--ct_densify_min_opacity must be in [0, 1].")
    if args.ct_densify_surface_tangent_ratio < 0.0 or args.ct_densify_bulk_scale_ratio < 0.0:
        raise ValueError("CT densify scale ratios must be >= 0.")
    if args.ct_bulk_reseed_from_iter < 0 or args.ct_bulk_reseed_until_iter < 0:
        raise ValueError("CT bulk reseed iteration bounds must be >= 0.")
    if args.ct_bulk_reseed_interval <= 0:
        raise ValueError("--ct_bulk_reseed_interval must be > 0.")
    if args.ct_bulk_reseed_until_iter < args.ct_bulk_reseed_from_iter:
        raise ValueError("--ct_bulk_reseed_until_iter must be >= --ct_bulk_reseed_from_iter.")
    if args.ct_bulk_reseed_sample_count < 0:
        raise ValueError("--ct_bulk_reseed_sample_count must be >= 0.")
    if args.ct_bulk_reseed_max_per_iter < 0:
        raise ValueError("--ct_bulk_reseed_max_per_iter must be >= 0.")
    if args.ct_bulk_reseed_max_new_fraction < 0.0:
        raise ValueError("--ct_bulk_reseed_max_new_fraction must be >= 0.")
    if args.ct_bulk_reseed_max_gaussian_ratio < 1.0:
        raise ValueError("--ct_bulk_reseed_max_gaussian_ratio must be >= 1.")
    if args.ct_bulk_reseed_occupancy_threshold < 0.0 or args.ct_bulk_reseed_occupancy_threshold > 1.0:
        raise ValueError("--ct_bulk_reseed_occupancy_threshold must be in [0, 1].")
    if args.ct_bulk_reseed_initial_opacity < 0.0 or args.ct_bulk_reseed_initial_opacity > 1.0:
        raise ValueError("--ct_bulk_reseed_initial_opacity must be in [0, 1].")
    if args.ct_gap_reseed_den_target < 0.0 or args.ct_gap_reseed_den_target > 1.0:
        raise ValueError("--ct_gap_reseed_den_target must be in [0, 1].")
    if args.ct_gap_reseed_sample_ratio < 0.0 or args.ct_gap_reseed_sample_ratio > 1.0:
        raise ValueError("--ct_gap_reseed_sample_ratio must be in [0, 1].")
    if args.ct_gap_reseed_radius_vox <= 0.0:
        raise ValueError("--ct_gap_reseed_radius_vox must be > 0.")
    if args.ct_gap_reseed_max_per_iter < 0:
        raise ValueError("--ct_gap_reseed_max_per_iter must be >= 0.")
    if args.ct_gap_reseed_protect_iters < 0:
        raise ValueError("--ct_gap_reseed_protect_iters must be >= 0.")
    if args.ct_gap_bulk_growth_factor < 1.0:
        raise ValueError("--ct_gap_bulk_growth_factor must be >= 1.")
    if not 0.0 <= args.ct_repair_add_threshold < args.ct_repair_stop_threshold <= 1.0:
        raise ValueError("CT repair thresholds must satisfy 0 <= add < stop <= 1.")
    if args.ct_repair_min_component_points < 1:
        raise ValueError("--ct_repair_min_component_points must be >= 1.")
    if args.ct_repair_max_new_per_pass < 0:
        raise ValueError("--ct_repair_max_new_per_pass must be >= 0.")
    if args.ct_repair_max_new_fraction < 0.0:
        raise ValueError("--ct_repair_max_new_fraction must be >= 0.")
    if not 0.0 <= args.ct_repair_gain_ratio_min <= 1.0:
        raise ValueError("--ct_repair_gain_ratio_min must be in [0, 1].")
    if args.ct_repair_exclusion_radius_vox < 0.0:
        raise ValueError("--ct_repair_exclusion_radius_vox must be >= 0.")
    if args.ct_repair_stretch_growth_factor < 1.0:
        raise ValueError("--ct_repair_stretch_growth_factor must be >= 1.")
    if args.ct_repair_stretch_secondary_factor < 1.0:
        raise ValueError("--ct_repair_stretch_secondary_factor must be >= 1.")
    if args.ct_repair_stretch_max_ratio < 1.0:
        raise ValueError("--ct_repair_stretch_max_ratio must be >= 1.")
    if args.ct_repair_overfill_threshold <= 0.0:
        raise ValueError("--ct_repair_overfill_threshold must be > 0.")
    if args.ct_repair_top_components < 1:
        raise ValueError("--ct_repair_top_components must be >= 1.")
    if args.ct_repair_nearby_candidates < 1:
        raise ValueError("--ct_repair_nearby_candidates must be >= 1.")
    if args.ct_repair_probe_tangent_factor < 1.0:
        raise ValueError("--ct_repair_probe_tangent_factor must be >= 1.")
    if not 0.0 < args.ct_repair_probe_shrink <= 1.0:
        raise ValueError("--ct_repair_probe_shrink must be in (0, 1].")
    if args.ct_repair_max_probe_shrink_iters < 0:
        raise ValueError("--ct_repair_max_probe_shrink_iters must be >= 0.")
    if args.ct_repair_check_stride < 1:
        raise ValueError("--ct_repair_check_stride must be >= 1.")
    if args.ct_repair_max_check_points < 0:
        raise ValueError("--ct_repair_max_check_points must be >= 0.")
    if args.ct_completion_den_target < 0.0 or args.ct_completion_den_target > 1.0:
        raise ValueError("--ct_completion_den_target must be in [0, 1].")
    if args.ct_completion_radius_vox <= 0.0:
        raise ValueError("--ct_completion_radius_vox must be > 0.")
    if args.ct_completion_max_init_passes < 0:
        raise ValueError("--ct_completion_max_init_passes must be >= 0.")
    if args.ct_completion_max_new_per_pass < 0:
        raise ValueError("--ct_completion_max_new_per_pass must be >= 0.")
    if args.ct_completion_min_component_voxels < 1:
        raise ValueError("--ct_completion_min_component_voxels must be >= 1.")
    if args.ct_completion_check_stride < 1:
        raise ValueError("--ct_completion_check_stride must be >= 1.")
    if args.ct_completion_max_check_points < 0:
        raise ValueError("--ct_completion_max_check_points must be >= 0.")
    if args.ct_sdf_boundary_mode not in {"interface", "material_zero"}:
        raise ValueError("--ct_sdf_boundary_mode must be one of: interface, material_zero.")
    if args.ct_feature_adaptive_r_shell_vox <= 0.0:
        raise ValueError("--ct_feature_adaptive_r_shell_vox must be > 0.")
    if args.ct_feature_adaptive_blur_sigma_vox < 0.0:
        raise ValueError("--ct_feature_adaptive_blur_sigma_vox must be >= 0.")
    if args.ct_feature_adaptive_spacing_high_vox < 1:
        raise ValueError("--ct_feature_adaptive_spacing_high_vox must be >= 1.")
    if args.ct_feature_adaptive_spacing_mid_vox < 1:
        raise ValueError("--ct_feature_adaptive_spacing_mid_vox must be >= 1.")
    if args.ct_feature_adaptive_spacing_low_vox < 1:
        raise ValueError("--ct_feature_adaptive_spacing_low_vox must be >= 1.")
    if args.ct_bulk_containment_q_support <= 0.0:
        raise ValueError("--ct_bulk_containment_q_support must be > 0.")
    if args.ct_bulk_query_truncation_sigma <= 0.0:
        raise ValueError("--ct_bulk_query_truncation_sigma must be > 0.")
    if args.ct_init_preflight_max_containment_violation < 0.0:
        raise ValueError("--ct_init_preflight_max_containment_violation must be >= 0.")
    if args.ct_init_preflight_min_material_a_b_p10 < 0.0:
        raise ValueError("--ct_init_preflight_min_material_a_b_p10 must be >= 0.")
    if args.ct_init_preflight_max_material_coverage_gap < 0.0:
        raise ValueError("--ct_init_preflight_max_material_coverage_gap must be >= 0.")
    if args.ct_bulk_coverage_gap_threshold < 0.0:
        raise ValueError("--ct_bulk_coverage_gap_threshold must be >= 0.")
    if args.ct_reseed_probe_length <= 0.0:
        raise ValueError("--ct_reseed_probe_length must be > 0.")
    if args.ct_reseed_band_deficit_thr < 0.0:
        raise ValueError("--ct_reseed_band_deficit_thr must be >= 0.")
    if args.ct_reseed_seed_offset_sigma < 0.0:
        raise ValueError("--ct_reseed_seed_offset_sigma must be >= 0.")
    if args.ct_reseed_max_new_fraction < 0.0:
        raise ValueError("--ct_reseed_max_new_fraction must be >= 0.")
    if args.ct_reseed_sigma_init_factor < 0.0:
        raise ValueError("--ct_reseed_sigma_init_factor must be >= 0.")
    if args.ct_reseed_sigma_init_floor_ratio < 0.0:
        raise ValueError("--ct_reseed_sigma_init_floor_ratio must be >= 0.")
    if args.ct_reseed_atten_init_boost <= 0.0:
        raise ValueError("--ct_reseed_atten_init_boost must be > 0.")
    if args.ct_bulk_prune_interval < 0 or args.ct_bulk_prune_warmup < 0:
        raise ValueError("CT bulk prune schedule values must be >= 0.")
    if args.ct_bulk_prune_min_opacity < 0.0 or args.ct_bulk_prune_min_opacity > 1.0:
        raise ValueError("--ct_bulk_prune_min_opacity must be in [0, 1].")
    if args.ct_bulk_prune_air_sdf_epsilon < 0.0:
        raise ValueError("--ct_bulk_prune_air_sdf_epsilon must be >= 0.")
    if args.ct_bulk_prune_raw_air_threshold < 0.0 or args.ct_bulk_prune_raw_air_threshold > 1.0:
        raise ValueError("--ct_bulk_prune_raw_air_threshold must be in [0, 1].")
    if args.ct_bulk_prune_sample_count < 0:
        raise ValueError("--ct_bulk_prune_sample_count must be >= 0.")
    if args.ct_bulk_prune_neighbor_radius <= 0.0:
        raise ValueError("--ct_bulk_prune_neighbor_radius must be > 0.")
    if args.ct_bulk_prune_min_neighbors < 0:
        raise ValueError("--ct_bulk_prune_min_neighbors must be >= 0.")
    if args.ct_intensity_phase2_start_ratio < 0.0 or args.ct_intensity_phase2_start_ratio > 1.0:
        raise ValueError("--ct_intensity_phase2_start_ratio must be in [0, 1].")
    if args.ct_intensity_phase3_start_ratio < 0.0 or args.ct_intensity_phase3_start_ratio > 1.0:
        raise ValueError("--ct_intensity_phase3_start_ratio must be in [0, 1].")
    if args.ct_intensity_phase3_start_ratio < args.ct_intensity_phase2_start_ratio:
        raise ValueError("--ct_intensity_phase3_start_ratio must be >= --ct_intensity_phase2_start_ratio.")
    if args.ct_intensity_bulk_scale_grad < 0.0:
        raise ValueError("--ct_intensity_bulk_scale_grad must be >= 0.")
    if args.ct_intensity_sample_mode not in {"full_band", "material_interior_only"}:
        raise ValueError("--ct_intensity_sample_mode must be one of {'full_band', 'material_interior_only'}.")
    if args.ct_intensity_erode_margin_vox < 0.0:
        raise ValueError("--ct_intensity_erode_margin_vox must be >= 0.")
    if args.ct_intensity_den_min < 0.0:
        raise ValueError("--ct_intensity_den_min must be >= 0.")
    if args.ct_bulk_opacity_sparse_weight < 0.0:
        raise ValueError("--ct_bulk_opacity_sparse_weight must be >= 0.")
    if args.ct_atten_only_lr_final_scale <= 0.0 or args.ct_atten_only_lr_final_scale > 1.0:
        raise ValueError("--ct_atten_only_lr_final_scale must be in (0, 1].")
    if args.ct_atten_only_early_stop_warmup_iters < 0:
        raise ValueError("--ct_atten_only_early_stop_warmup_iters must be >= 0.")
    if args.ct_atten_only_early_stop_eval_interval < 1:
        raise ValueError("--ct_atten_only_early_stop_eval_interval must be >= 1.")
    if args.ct_atten_only_early_stop_patience < 1:
        raise ValueError("--ct_atten_only_early_stop_patience must be >= 1.")
    if args.ct_atten_only_early_stop_min_delta < 0.0:
        raise ValueError("--ct_atten_only_early_stop_min_delta must be >= 0.")
    if args.ct_eagle_loss_weight < 0.0:
        raise ValueError("--ct_eagle_loss_weight must be >= 0.")
    if args.ct_eagle_patch_size < 1:
        raise ValueError("--ct_eagle_patch_size must be >= 1.")
    if args.ct_eagle_block_size < 1:
        raise ValueError("--ct_eagle_block_size must be >= 1.")

def validate_ct_training_args(args):
    _normalize_ct_training_args(args)
    _validate_path_and_io_args(args)
    _validate_core_training_args(args)
    _validate_surface_and_bulk_args(args)
