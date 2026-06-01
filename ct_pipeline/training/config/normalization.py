from ct_pipeline.rendering.bulk_support import resolve_bulk_query_truncation_sigma

from .defaults import *


def _normalize_path_and_io_args(args) -> None:
    args.ct_auto_preview = bool(getattr(args, "ct_auto_preview", True))
    args.ct_preview_interval = int(getattr(args, "ct_preview_interval", CT_PREVIEW_INTERVAL))
    if getattr(args, "ct_raw_meta", None) is not None:
        args.ct_raw_meta = str(args.ct_raw_meta)
    args.ct_thickness_max = None

def _normalize_core_training_args(args) -> None:
    args._ct_support_sample_count_auto = getattr(args, "ct_support_sample_count", None) is None
    args._ct_air_sample_count_auto = getattr(args, "ct_air_sample_count", None) is None
    if getattr(args, "ct_support_sample_count", None) is not None:
        args.ct_support_sample_count = int(args.ct_support_sample_count)
    if getattr(args, "ct_air_sample_count", None) is not None:
        args.ct_air_sample_count = int(args.ct_air_sample_count)

    args.ct_bulk_boundary_margin_voxels = int(getattr(args, "ct_bulk_boundary_margin_voxels", 2))
    args.ct_volume_sample_count = int(getattr(args, "ct_volume_sample_count", 8192))
    args.ct_volume_jitter = float(getattr(args, "ct_volume_jitter", CT_VOLUME_JITTER))
    args.ct_boundary_band = float(getattr(args, "ct_boundary_band", CT_BOUNDARY_BAND))
    args.ct_surface_boundary_sample_ratio = float(getattr(args, "ct_surface_boundary_sample_ratio", 0.25))
    args.ct_lambda_volume = float(getattr(args, "ct_lambda_volume", 1.0))
    args.ct_training_mode = str(getattr(args, "ct_training_mode", CT_TRAINING_MODE))
    args.ct_lambda_occupancy = float(getattr(args, "ct_lambda_occupancy", 1.0))
    args.ct_surface_regularizer_weight = float(getattr(args, "ct_surface_regularizer_weight", 0.5))
    args.ct_use_unified_compositor = bool(getattr(args, "ct_use_unified_compositor", CT_USE_UNIFIED_COMPOSITOR))
    args.ct_dual_separated_training = bool(
        getattr(args, "ct_dual_separated_training", CT_DUAL_SEPARATED_TRAINING)
    )
    args.ct_bulk_field_mode = str(getattr(args, "ct_bulk_field_mode", CT_BULK_FIELD_MODE))
    args.ct_bulk_halfspace_enable = bool(
        getattr(args, "ct_bulk_halfspace_enable", CT_BULK_HALFSPACE_ENABLE)
    )
    args.ct_bulk_halfspace_tau_init = float(
        getattr(args, "ct_bulk_halfspace_tau_init", CT_BULK_HALFSPACE_TAU_INIT)
    )
    args.ct_bulk_halfspace_tau_final = float(
        getattr(args, "ct_bulk_halfspace_tau_final", CT_BULK_HALFSPACE_TAU_FINAL)
    )
    args.ct_bulk_halfspace_tau_gate_threshold = float(
        getattr(args, "ct_bulk_halfspace_tau_gate_threshold", CT_BULK_HALFSPACE_TAU_GATE_THRESHOLD)
    )
    args.ct_bulk_halfspace_tau_step = float(
        getattr(args, "ct_bulk_halfspace_tau_step", CT_BULK_HALFSPACE_TAU_STEP)
    )
    args.ct_bulk_halfspace_tau_update_interval = int(
        getattr(args, "ct_bulk_halfspace_tau_update_interval", CT_BULK_HALFSPACE_TAU_UPDATE_INTERVAL)
    )
    args.ct_bulk_halfspace_skip_depth = float(
        getattr(args, "ct_bulk_halfspace_skip_depth", CT_BULK_HALFSPACE_SKIP_DEPTH)
    )
    args.ct_occupancy_sample_count = int(getattr(args, "ct_occupancy_sample_count", 16384))
    args.ct_occ_boundary_sample_ratio = float(getattr(args, "ct_occ_boundary_sample_ratio", 0.50))
    args.ct_occ_deep_material_sample_ratio = float(getattr(args, "ct_occ_deep_material_sample_ratio", 0.35))
    args.ct_occ_boundary_weight = float(getattr(args, "ct_occ_boundary_weight", 3.0))
    args.ct_exterior_air_sample_ratio = float(getattr(args, "ct_exterior_air_sample_ratio", CT_EXTERIOR_AIR_SAMPLE_RATIO))
    args.ct_bulk_volume_support_sample_ratio = float(
        getattr(args, "ct_bulk_volume_support_sample_ratio", CT_BULK_VOLUME_SUPPORT_SAMPLE_RATIO)
    )
    args.ct_huber_beta = CT_HUBER_BETA

def _normalize_internal_geometry_args(args) -> None:
    args.ct_gaussian_truncation_sigma = CT_GAUSSIAN_TRUNCATION_SIGMA
    args.ct_slice_tile_size = CT_SLICE_TILE_SIZE
    args.ct_grid_cell_voxels = CT_GRID_CELL_VOXELS
    args.ct_grid_cache = CT_GRID_CACHE_ENABLED
    args.ct_surface_grid_rebuild_interval = CT_SURFACE_GRID_REBUILD_INTERVAL
    args.ct_bulk_grid_rebuild_interval = CT_BULK_GRID_REBUILD_INTERVAL
    args.ct_grid_cache_inflation_margin = CT_GRID_CACHE_INFLATION_MARGIN
    args.ct_grid_cache_drift_check = CT_GRID_CACHE_DRIFT_CHECK
    args.ct_grid_cache_max_cell_gaussian_pairs = CT_GRID_CACHE_MAX_CELL_GAUSSIAN_PAIRS
    args.ct_surface_sigma_n_max = float(getattr(args, "ct_surface_sigma_n_max", CT_SURFACE_SIGMA_N_MAX))
    args.ct_surface_sigma_t_min = float(getattr(args, "ct_surface_sigma_t_min", CT_SURFACE_SIGMA_T_MIN))
    args.ct_surface_max_scale = float(getattr(args, "ct_surface_max_scale", CT_SURFACE_MAX_SCALE))
    args.ct_bulk_scale_adaptive_cap = bool(
        getattr(args, "ct_bulk_scale_adaptive_cap", CT_BULK_SCALE_ADAPTIVE_CAP)
    )
    args.ct_bulk_scale_global_max = float(
        getattr(args, "ct_bulk_scale_global_max", CT_BULK_SCALE_GLOBAL_MAX)
    )
    args.ct_bulk_scale_floor = float(getattr(args, "ct_bulk_scale_floor", CT_BULK_SCALE_FLOOR))
    args.ct_surface_normal_weight = float(getattr(args, "ct_surface_normal_weight", CT_SURFACE_NORMAL_WEIGHT))
    args.ct_surface_thickness_weight = float(getattr(args, "ct_surface_thickness_weight", CT_SURFACE_THICKNESS_WEIGHT))
    args.ct_surface_regularizer_sample_count = CT_SURFACE_REGULARIZER_SAMPLE_COUNT
    args.ct_surface_coverage_weight = float(getattr(args, "ct_surface_coverage_weight", CT_SURFACE_COVERAGE_WEIGHT))
    args.ct_surface_coverage_sample_count = int(
        getattr(args, "ct_surface_coverage_sample_count", CT_SURFACE_COVERAGE_SAMPLE_COUNT)
    )
    args.ct_surface_coverage_target = float(getattr(args, "ct_surface_coverage_target", CT_SURFACE_COVERAGE_TARGET))
    args.ct_surface_coverage_until_iter = int(
        getattr(args, "ct_surface_coverage_until_iter", CT_SURFACE_COVERAGE_UNTIL_ITER)
    )
    surface_material_gate_sigma = getattr(args, "ct_surface_material_gate_sigma", CT_SURFACE_MATERIAL_GATE_SIGMA)
    args.ct_surface_material_gate_sigma = (
        None if surface_material_gate_sigma is None else float(surface_material_gate_sigma)
    )
    args.ct_surface_material_delta = float(getattr(args, "ct_surface_material_delta", CT_SURFACE_MATERIAL_DELTA))
    args.ct_surface_material_tau = float(getattr(args, "ct_surface_material_tau", CT_SURFACE_MATERIAL_TAU))
    args.ct_surface_material_render_weight = float(
        getattr(args, "ct_surface_material_render_weight", CT_SURFACE_MATERIAL_RENDER_WEIGHT)
    )
    args.ct_surface_intensity_weight = float(
        getattr(args, "ct_surface_intensity_weight", CT_SURFACE_INTENSITY_WEIGHT)
    )
    args.ct_surface_intensity_sample_count = int(
        getattr(args, "ct_surface_intensity_sample_count", CT_SURFACE_INTENSITY_SAMPLE_COUNT)
    )
    args.ct_surface_intensity_material_ratio = float(
        getattr(args, "ct_surface_intensity_material_ratio", CT_SURFACE_INTENSITY_MATERIAL_RATIO)
    )
    args.ct_dual_surface_inner_band = float(
        getattr(args, "ct_dual_surface_inner_band", CT_DUAL_SURFACE_INNER_BAND)
    )
    args.ct_dual_surface_air_band = float(getattr(args, "ct_dual_surface_air_band", CT_DUAL_SURFACE_AIR_BAND))
    args.ct_dual_soft_sigma = float(getattr(args, "ct_dual_soft_sigma", CT_DUAL_SOFT_SIGMA))
    args.ct_dual_surface_air_sample_ratio = float(
        getattr(args, "ct_dual_surface_air_sample_ratio", CT_DUAL_SURFACE_AIR_SAMPLE_RATIO)
    )
    args.ct_bulk_renderer_epsilon = float(getattr(args, "ct_bulk_renderer_epsilon", CT_BULK_RENDERER_EPSILON))
    args.ct_bulk_renderer_tau = float(getattr(args, "ct_bulk_renderer_tau", CT_BULK_RENDERER_TAU))
    args.ct_material_compose_mode = str(getattr(args, "ct_material_compose_mode", CT_MATERIAL_COMPOSE_MODE))
    args.ct_bulk_semantic_weight = float(getattr(args, "ct_bulk_semantic_weight", CT_BULK_SEMANTIC_WEIGHT))
    args.ct_bulk_semantic_sample_count = int(
        getattr(args, "ct_bulk_semantic_sample_count", CT_BULK_SEMANTIC_SAMPLE_COUNT)
    )
    args.ct_bulk_semantic_focal_gamma = float(
        getattr(args, "ct_bulk_semantic_focal_gamma", CT_BULK_SEMANTIC_FOCAL_GAMMA)
    )
    args.ct_bulk_semantic_epsilon = float(getattr(args, "ct_bulk_semantic_epsilon", CT_BULK_SEMANTIC_EPSILON))
    args.ct_bulk_semantic_pos_ratio = float(
        getattr(args, "ct_bulk_semantic_pos_ratio", CT_BULK_SEMANTIC_POS_RATIO)
    )
    args.ct_bulk_semantic_deep_ratio = float(
        getattr(args, "ct_bulk_semantic_deep_ratio", CT_BULK_SEMANTIC_DEEP_RATIO)
    )
    args.ct_bulk_semantic_boundary_ratio = float(
        getattr(args, "ct_bulk_semantic_boundary_ratio", CT_BULK_SEMANTIC_BOUNDARY_RATIO)
    )
    args.ct_bulk_semantic_cavity_material_ratio = float(
        getattr(args, "ct_bulk_semantic_cavity_material_ratio", CT_BULK_SEMANTIC_CAVITY_MATERIAL_RATIO)
    )
    args.ct_bulk_semantic_void_air_ratio = float(
        getattr(args, "ct_bulk_semantic_void_air_ratio", CT_BULK_SEMANTIC_VOID_AIR_RATIO)
    )
    args.ct_bulk_semantic_cavity_air_ratio = float(
        getattr(args, "ct_bulk_semantic_cavity_air_ratio", CT_BULK_SEMANTIC_CAVITY_AIR_RATIO)
    )
    args.ct_bulk_semantic_exterior_air_ratio = float(
        getattr(args, "ct_bulk_semantic_exterior_air_ratio", CT_BULK_SEMANTIC_EXTERIOR_AIR_RATIO)
    )
    args.ct_focal_gamma_pos = float(getattr(args, "ct_focal_gamma_pos", CT_FOCAL_GAMMA_POS))
    focal_gamma_neg = getattr(args, "ct_focal_gamma_neg", None)
    legacy_focal_gamma = float(getattr(args, "ct_bulk_semantic_focal_gamma", CT_BULK_SEMANTIC_FOCAL_GAMMA))
    if focal_gamma_neg is None:
        focal_gamma_neg = legacy_focal_gamma
    elif float(focal_gamma_neg) == CT_FOCAL_GAMMA_NEG and legacy_focal_gamma != CT_BULK_SEMANTIC_FOCAL_GAMMA:
        focal_gamma_neg = legacy_focal_gamma
    args.ct_focal_gamma_neg = float(focal_gamma_neg)
    args.ct_bulk_semantic_focal_gamma = args.ct_focal_gamma_neg
    args.ct_focal_alpha_pos = float(getattr(args, "ct_focal_alpha_pos", CT_FOCAL_ALPHA_POS))
    args.ct_bulk_semantic_boundary_target = float(
        getattr(args, "ct_bulk_semantic_boundary_target", CT_BULK_SEMANTIC_BOUNDARY_TARGET)
    )
    args.ct_bulk_semantic_cavity_target = float(
        getattr(args, "ct_bulk_semantic_cavity_target", CT_BULK_SEMANTIC_CAVITY_TARGET)
    )
    args.ct_bulk_floor_weight = float(getattr(args, "ct_bulk_floor_weight", CT_BULK_FLOOR_WEIGHT))
    args.ct_bulk_floor_deep_target = float(
        getattr(args, "ct_bulk_floor_deep_target", CT_BULK_FLOOR_DEEP_TARGET)
    )
    args.ct_bulk_floor_shell_target = float(
        getattr(args, "ct_bulk_floor_shell_target", CT_BULK_FLOOR_SHELL_TARGET)
    )
    args.ct_bulk_void_weight = float(getattr(args, "ct_bulk_void_weight", CT_BULK_VOID_WEIGHT))
    args.ct_loss_boundary_band_delta = float(
        getattr(args, "ct_loss_boundary_band_delta", CT_LOSS_BOUNDARY_BAND_DELTA)
    )
    args.ct_loss_band_in_weight = float(
        getattr(args, "ct_loss_band_in_weight", CT_LOSS_BAND_IN_WEIGHT)
    )
    args.ct_loss_band_out_weight = float(
        getattr(args, "ct_loss_band_out_weight", CT_LOSS_BAND_OUT_WEIGHT)
    )
    args.ct_bulk_coverage_weight = float(getattr(args, "ct_bulk_coverage_weight", CT_BULK_COVERAGE_WEIGHT))
    args.ct_bulk_coverage_sample_count = int(
        getattr(args, "ct_bulk_coverage_sample_count", CT_BULK_COVERAGE_SAMPLE_COUNT)
    )
    args.ct_bulk_coverage_target = float(getattr(args, "ct_bulk_coverage_target", CT_BULK_COVERAGE_TARGET))
    args.ct_bulk_air_exclusion_weight = float(
        getattr(args, "ct_bulk_air_exclusion_weight", CT_BULK_AIR_EXCLUSION_WEIGHT)
    )
    args.ct_bulk_air_exclusion_sample_count = int(
        getattr(args, "ct_bulk_air_exclusion_sample_count", CT_BULK_AIR_EXCLUSION_SAMPLE_COUNT)
    )
    args.ct_bulk_air_exclusion_target = float(
        getattr(args, "ct_bulk_air_exclusion_target", CT_BULK_AIR_EXCLUSION_TARGET)
    )
    args.ct_cavity_material_coverage_weight = float(
        getattr(args, "ct_cavity_material_coverage_weight", CT_CAVITY_MATERIAL_COVERAGE_WEIGHT)
    )
    args.ct_cavity_false_hole_boost = float(getattr(args, "ct_cavity_false_hole_boost", CT_CAVITY_FALSE_HOLE_BOOST))
    args.ct_bulk_sdf_containment_weight = float(
        getattr(args, "ct_bulk_sdf_containment_weight", CT_BULK_SDF_CONTAINMENT_WEIGHT)
    )
    args.ct_containment_weight = float(getattr(args, "ct_containment_weight", args.ct_bulk_sdf_containment_weight))
    args.ct_bulk_sdf_containment_sample_count = int(
        getattr(args, "ct_bulk_sdf_containment_sample_count", CT_BULK_SDF_CONTAINMENT_SAMPLE_COUNT)
    )
    args.ct_bulk_sdf_containment_margin = float(
        getattr(args, "ct_bulk_sdf_containment_margin", CT_BULK_SDF_CONTAINMENT_MARGIN)
    )
    args.ct_init_strategy = str(getattr(args, "ct_init_strategy", CT_INIT_STRATEGY))
    args.ct_init_coverage_normalized_atten = bool(
        getattr(args, "ct_init_coverage_normalized_atten", CT_INIT_COVERAGE_NORMALIZED_ATTEN)
    )
    args.ct_init_boundary_inward_nudge = bool(
        getattr(args, "ct_init_boundary_inward_nudge", CT_INIT_BOUNDARY_INWARD_NUDGE)
    )
    args.ct_init_boundary_sigma_c = float(
        getattr(args, "ct_init_boundary_sigma_c", CT_INIT_BOUNDARY_SIGMA_C)
    )
    args.ct_init_deep_sigma_voxel = float(
        getattr(args, "ct_init_deep_sigma_voxel", CT_INIT_DEEP_SIGMA_VOXEL)
    )
    args.ct_init_deep_depth_voxel = float(
        getattr(args, "ct_init_deep_depth_voxel", CT_INIT_DEEP_DEPTH_VOXEL)
    )
    args.ct_init_inward_nudge_sigma_ratio = float(
        getattr(args, "ct_init_inward_nudge_sigma_ratio", CT_INIT_INWARD_NUDGE_SIGMA_RATIO)
    )
    args.ct_init_coverage_knn_k = int(
        getattr(args, "ct_init_coverage_knn_k", CT_INIT_COVERAGE_KNN_K)
    )
    args.ct_init_coverage_report = bool(
        getattr(args, "ct_init_coverage_report", CT_INIT_COVERAGE_REPORT)
    )
    args.ct_init_coverage_c_min_material = float(
        getattr(args, "ct_init_coverage_c_min_material", CT_INIT_COVERAGE_C_MIN_MATERIAL)
    )
    args.ct_init_coverage_c_min_shell = float(
        getattr(args, "ct_init_coverage_c_min_shell", CT_INIT_COVERAGE_C_MIN_SHELL)
    )
    args.ct_init_coverage_void_epsilon = float(
        getattr(args, "ct_init_coverage_void_epsilon", CT_INIT_COVERAGE_VOID_EPSILON)
    )
    args.ct_stage1_freeze_until_iter = int(
        getattr(args, "ct_stage1_freeze_until_iter", CT_STAGE1_FREEZE_UNTIL_ITER)
    )
    args.ct_containment_ramp_iter_1 = int(getattr(args, "ct_containment_ramp_iter_1", CT_CONTAINMENT_RAMP_ITER_1))
    args.ct_containment_ramp_weight_1 = float(
        getattr(args, "ct_containment_ramp_weight_1", CT_CONTAINMENT_RAMP_WEIGHT_1)
    )
    args.ct_containment_ramp_iter_2 = int(getattr(args, "ct_containment_ramp_iter_2", CT_CONTAINMENT_RAMP_ITER_2))
    args.ct_containment_ramp_weight_2 = float(
        getattr(args, "ct_containment_ramp_weight_2", CT_CONTAINMENT_RAMP_WEIGHT_2)
    )
    args.ct_containment_ramp_iter_3 = int(getattr(args, "ct_containment_ramp_iter_3", CT_CONTAINMENT_RAMP_ITER_3))
    args.ct_containment_ramp_weight_3 = float(
        getattr(args, "ct_containment_ramp_weight_3", CT_CONTAINMENT_RAMP_WEIGHT_3)
    )
    args.ct_false_hole_sample_count = int(getattr(args, "ct_false_hole_sample_count", CT_FALSE_HOLE_SAMPLE_COUNT))
    args.ct_false_hole_boundary_band = float(getattr(args, "ct_false_hole_boundary_band", CT_FALSE_HOLE_BOUNDARY_BAND))
    args.ct_false_hole_material_threshold = float(
        getattr(args, "ct_false_hole_material_threshold", CT_FALSE_HOLE_MATERIAL_THRESHOLD)
    )
    args.ct_false_hole_dark_margin = float(getattr(args, "ct_false_hole_dark_margin", CT_FALSE_HOLE_DARK_MARGIN))
    args.ct_false_hole_target_occupancy = float(
        getattr(args, "ct_false_hole_target_occupancy", CT_FALSE_HOLE_TARGET_OCCUPANCY)
    )
    args.ct_false_hole_metrics_interval = int(
        getattr(args, "ct_false_hole_metrics_interval", CT_FALSE_HOLE_METRICS_INTERVAL)
    )
    args.ct_freeze_bulk_xyz = bool(getattr(args, "ct_freeze_bulk_xyz", CT_FREEZE_BULK_XYZ))
    args.ct_bulk_continuous_init = bool(getattr(args, "ct_bulk_continuous_init", CT_BULK_CONTINUOUS_INIT))
    args.ct_freeze_primitive_type = CT_FREEZE_PRIMITIVE_TYPE

def _normalize_surface_and_bulk_args(args) -> None:
    args.ct_bulk_max_scale = float(getattr(args, "ct_bulk_max_scale", 2.0))
    args.ct_bulk_augment_factor = float(getattr(args, "ct_bulk_augment_factor", 2.0))

def _normalize_densification_args(args) -> None:
    args.ct_enable_densification = bool(getattr(args, "ct_enable_densification", True))
    args.ct_densify_from_iter = int(getattr(args, "ct_densify_from_iter", CT_DENSIFY_FROM_ITER))
    args.ct_densify_until_iter = int(getattr(args, "ct_densify_until_iter", CT_DENSIFY_UNTIL_ITER))
    args.ct_densify_interval = int(getattr(args, "ct_densify_interval", CT_DENSIFY_INTERVAL))
    args.ct_densify_surface_percent = float(getattr(args, "ct_densify_surface_percent", CT_DENSIFY_SURFACE_PERCENT))
    args.ct_densify_bulk_percent = float(getattr(args, "ct_densify_bulk_percent", CT_DENSIFY_BULK_PERCENT))
    args.ct_densify_max_gaussian_ratio = float(getattr(args, "ct_densify_max_gaussian_ratio", CT_DENSIFY_MAX_GAUSSIAN_RATIO))
    args.ct_densify_min_opacity = float(getattr(args, "ct_densify_min_opacity", CT_DENSIFY_MIN_OPACITY))
    args.ct_densify_surface_tangent_ratio = float(getattr(args, "ct_densify_surface_tangent_ratio", CT_DENSIFY_SURFACE_TANGENT_RATIO))
    args.ct_densify_bulk_scale_ratio = float(getattr(args, "ct_densify_bulk_scale_ratio", CT_DENSIFY_BULK_SCALE_RATIO))
    args.ct_enable_surface_reseeding = bool(getattr(args, "ct_enable_surface_reseeding", True))
    args.ct_surface_reseed_from_iter = CT_SURFACE_RESEED_FROM_ITER
    args.ct_surface_reseed_until_iter = CT_SURFACE_RESEED_UNTIL_ITER
    args.ct_surface_reseed_interval = CT_SURFACE_RESEED_INTERVAL
    args.ct_surface_reseed_sample_count = CT_SURFACE_RESEED_SAMPLE_COUNT
    args.ct_surface_reseed_max_new_per_iter = CT_SURFACE_RESEED_MAX_NEW_PER_ITER
    args.ct_surface_reseed_max_gaussian_ratio = CT_SURFACE_RESEED_MAX_GAUSSIAN_RATIO
    args.ct_surface_reseed_min_bulk_occupancy = CT_SURFACE_RESEED_MIN_BULK_OCCUPANCY
    args.ct_surface_reseed_max_surface_occupancy = CT_SURFACE_RESEED_MAX_SURFACE_OCCUPANCY
    args.ct_surface_reseed_coverage_radius = float(
        getattr(args, "ct_surface_reseed_coverage_radius", getattr(args, "ct_surface_reseed_coverage_radius_voxels", CT_SURFACE_RESEED_COVERAGE_RADIUS_VOXELS))
    )
    args.ct_surface_reseed_coverage_radius_voxels = args.ct_surface_reseed_coverage_radius

    args.ct_enable_bulk_reseeding = bool(getattr(args, "ct_enable_bulk_reseeding", True))
    args.ct_bulk_reseed_from_iter = int(getattr(args, "ct_bulk_reseed_from_iter", 500))
    args.ct_bulk_reseed_until_iter = int(getattr(args, "ct_bulk_reseed_until_iter", 4000))
    args.ct_bulk_reseed_interval = int(getattr(args, "ct_bulk_reseed_interval", 500))
    args.ct_bulk_reseed_sample_count = int(getattr(args, "ct_bulk_reseed_sample_count", 8192))
    args.ct_bulk_reseed_max_per_iter = int(getattr(args, "ct_bulk_reseed_max_per_iter", 0))
    args.ct_bulk_reseed_max_new_fraction = float(getattr(args, "ct_bulk_reseed_max_new_fraction", 0.03))
    args.ct_bulk_reseed_max_gaussian_ratio = float(getattr(args, "ct_bulk_reseed_max_gaussian_ratio", 2.5))
    args.ct_bulk_reseed_occupancy_threshold = float(getattr(args, "ct_bulk_reseed_occupancy_threshold", 0.85))
    args.ct_bulk_reseed_min_sdf_margin = float(getattr(args, "ct_bulk_reseed_min_sdf_margin", -0.2))
    args.ct_bulk_reseed_initial_opacity = float(getattr(args, "ct_bulk_reseed_initial_opacity", 0.65))
    args.ct_gap_aware_reseed = bool(getattr(args, "ct_gap_aware_reseed", CT_GAP_AWARE_RESEED))
    args.ct_gap_reseed_den_target = float(getattr(args, "ct_gap_reseed_den_target", CT_GAP_RESEED_DEN_TARGET))
    args.ct_gap_reseed_sample_ratio = float(getattr(args, "ct_gap_reseed_sample_ratio", CT_GAP_RESEED_SAMPLE_RATIO))
    args.ct_gap_reseed_radius_vox = float(getattr(args, "ct_gap_reseed_radius_vox", CT_GAP_RESEED_RADIUS_VOX))
    args.ct_gap_reseed_max_per_iter = int(getattr(args, "ct_gap_reseed_max_per_iter", CT_GAP_RESEED_MAX_PER_ITER))
    args.ct_gap_reseed_boundary_subvoxel = bool(
        getattr(args, "ct_gap_reseed_boundary_subvoxel", CT_GAP_RESEED_BOUNDARY_SUBVOXEL)
    )
    args.ct_gap_reseed_protect_prune = bool(
        getattr(args, "ct_gap_reseed_protect_prune", CT_GAP_RESEED_PROTECT_PRUNE)
    )
    args.ct_gap_reseed_protect_iters = int(
        getattr(args, "ct_gap_reseed_protect_iters", CT_GAP_RESEED_PROTECT_ITERS)
    )
    args.ct_gap_bulk_growth_factor = float(getattr(args, "ct_gap_bulk_growth_factor", CT_GAP_BULK_GROWTH_FACTOR))
    args.ct_budgeted_component_repair = bool(
        getattr(args, "ct_budgeted_component_repair", CT_BUDGETED_COMPONENT_REPAIR)
    )
    args.ct_repair_add_threshold = float(getattr(args, "ct_repair_add_threshold", CT_REPAIR_ADD_THRESHOLD))
    args.ct_repair_stop_threshold = float(getattr(args, "ct_repair_stop_threshold", CT_REPAIR_STOP_THRESHOLD))
    args.ct_repair_min_component_points = int(
        getattr(args, "ct_repair_min_component_points", CT_REPAIR_MIN_COMPONENT_POINTS)
    )
    args.ct_repair_max_new_per_pass = int(getattr(args, "ct_repair_max_new_per_pass", CT_REPAIR_MAX_NEW_PER_PASS))
    args.ct_repair_max_new_fraction = float(getattr(args, "ct_repair_max_new_fraction", CT_REPAIR_MAX_NEW_FRACTION))
    args.ct_repair_gain_ratio_min = float(getattr(args, "ct_repair_gain_ratio_min", CT_REPAIR_GAIN_RATIO_MIN))
    args.ct_repair_exclusion_radius_vox = float(
        getattr(args, "ct_repair_exclusion_radius_vox", CT_REPAIR_EXCLUSION_RADIUS_VOX)
    )
    args.ct_repair_stretch_first = bool(getattr(args, "ct_repair_stretch_first", CT_REPAIR_STRETCH_FIRST))
    args.ct_repair_stretch_growth_factor = float(
        getattr(args, "ct_repair_stretch_growth_factor", CT_REPAIR_STRETCH_GROWTH_FACTOR)
    )
    args.ct_repair_stretch_secondary_factor = float(
        getattr(args, "ct_repair_stretch_secondary_factor", CT_REPAIR_STRETCH_SECONDARY_FACTOR)
    )
    args.ct_repair_stretch_max_ratio = float(getattr(args, "ct_repair_stretch_max_ratio", CT_REPAIR_STRETCH_MAX_RATIO))
    args.ct_repair_overfill_threshold = float(
        getattr(args, "ct_repair_overfill_threshold", CT_REPAIR_OVERFILL_THRESHOLD)
    )
    args.ct_repair_top_components = int(getattr(args, "ct_repair_top_components", CT_REPAIR_TOP_COMPONENTS))
    args.ct_repair_nearby_candidates = int(getattr(args, "ct_repair_nearby_candidates", CT_REPAIR_NEARBY_CANDIDATES))
    args.ct_repair_probe_tangent_factor = float(
        getattr(args, "ct_repair_probe_tangent_factor", CT_REPAIR_PROBE_TANGENT_FACTOR)
    )
    args.ct_repair_probe_shrink = float(getattr(args, "ct_repair_probe_shrink", CT_REPAIR_PROBE_SHRINK))
    args.ct_repair_max_probe_shrink_iters = int(
        getattr(args, "ct_repair_max_probe_shrink_iters", CT_REPAIR_MAX_PROBE_SHRINK_ITERS)
    )
    args.ct_repair_check_stride = int(getattr(args, "ct_repair_check_stride", CT_REPAIR_CHECK_STRIDE))
    args.ct_repair_max_check_points = int(getattr(args, "ct_repair_max_check_points", CT_REPAIR_MAX_CHECK_POINTS))
    args.ct_material_coverage_completion = bool(
        getattr(args, "ct_material_coverage_completion", CT_MATERIAL_COVERAGE_COMPLETION)
    )
    args.ct_completion_init = bool(getattr(args, "ct_completion_init", CT_COMPLETION_INIT))
    args.ct_completion_repair = bool(getattr(args, "ct_completion_repair", CT_COMPLETION_REPAIR))
    args.ct_completion_den_target = float(getattr(args, "ct_completion_den_target", CT_COMPLETION_DEN_TARGET))
    args.ct_completion_radius_vox = float(getattr(args, "ct_completion_radius_vox", CT_COMPLETION_RADIUS_VOX))
    args.ct_completion_max_init_passes = int(
        getattr(args, "ct_completion_max_init_passes", CT_COMPLETION_MAX_INIT_PASSES)
    )
    args.ct_completion_max_new_per_pass = int(
        getattr(args, "ct_completion_max_new_per_pass", CT_COMPLETION_MAX_NEW_PER_PASS)
    )
    args.ct_completion_min_component_voxels = int(
        getattr(args, "ct_completion_min_component_voxels", CT_COMPLETION_MIN_COMPONENT_VOXELS)
    )
    args.ct_completion_check_stride = int(getattr(args, "ct_completion_check_stride", CT_COMPLETION_CHECK_STRIDE))
    args.ct_completion_max_check_points = int(
        getattr(args, "ct_completion_max_check_points", CT_COMPLETION_MAX_CHECK_POINTS)
    )
    args.ct_sdf_boundary_mode = str(getattr(args, "ct_sdf_boundary_mode", CT_SDF_BOUNDARY_MODE)).strip().lower()
    args.ct_feature_adaptive_jitter = bool(
        getattr(args, "ct_feature_adaptive_jitter", CT_FEATURE_ADAPTIVE_JITTER)
    )
    args.ct_feature_adaptive_seed = int(getattr(args, "ct_feature_adaptive_seed", CT_FEATURE_ADAPTIVE_SEED))
    args.ct_feature_adaptive_r_shell_vox = float(
        getattr(args, "ct_feature_adaptive_r_shell_vox", CT_FEATURE_ADAPTIVE_R_SHELL_VOX)
    )
    args.ct_feature_adaptive_blur_sigma_vox = float(
        getattr(args, "ct_feature_adaptive_blur_sigma_vox", CT_FEATURE_ADAPTIVE_BLUR_SIGMA_VOX)
    )
    args.ct_feature_adaptive_spacing_high_vox = int(
        getattr(args, "ct_feature_adaptive_spacing_high_vox", CT_FEATURE_ADAPTIVE_SPACING_HIGH_VOX)
    )
    args.ct_feature_adaptive_spacing_mid_vox = int(
        getattr(args, "ct_feature_adaptive_spacing_mid_vox", CT_FEATURE_ADAPTIVE_SPACING_MID_VOX)
    )
    args.ct_feature_adaptive_spacing_low_vox = int(
        getattr(args, "ct_feature_adaptive_spacing_low_vox", CT_FEATURE_ADAPTIVE_SPACING_LOW_VOX)
    )
    args.ct_feature_adaptive_directional_clearance = bool(
        getattr(
            args,
            "ct_feature_adaptive_directional_clearance",
            CT_FEATURE_ADAPTIVE_DIRECTIONAL_CLEARANCE,
        )
    )
    args.ct_feature_adaptive_probe_containment = bool(
        getattr(args, "ct_feature_adaptive_probe_containment", CT_FEATURE_ADAPTIVE_PROBE_CONTAINMENT)
    )
    args.ct_bulk_containment_q_support = float(
        getattr(args, "ct_bulk_containment_q_support", CT_BULK_CONTAINMENT_Q_SUPPORT)
    )
    args.ct_bulk_query_truncation_sigma = resolve_bulk_query_truncation_sigma(args)
    args.ct_init_preflight_abort = bool(getattr(args, "ct_init_preflight_abort", CT_INIT_PREFLIGHT_ABORT))
    args.ct_init_preflight_max_containment_violation = float(
        getattr(
            args,
            "ct_init_preflight_max_containment_violation",
            CT_INIT_PREFLIGHT_MAX_CONTAINMENT_VIOLATION,
        )
    )
    args.ct_init_preflight_min_material_a_b_p10 = float(
        getattr(
            args,
            "ct_init_preflight_min_material_a_b_p10",
            CT_INIT_PREFLIGHT_MIN_MATERIAL_A_B_P10,
        )
    )
    args.ct_init_preflight_max_material_coverage_gap = float(
        getattr(
            args,
            "ct_init_preflight_max_material_coverage_gap",
            CT_INIT_PREFLIGHT_MAX_MATERIAL_COVERAGE_GAP,
        )
    )
    args.ct_bulk_coverage_gap_threshold = float(
        getattr(args, "ct_bulk_coverage_gap_threshold", CT_BULK_COVERAGE_GAP_THRESHOLD)
    )
    args.ct_reseed_probe_length = float(getattr(args, "ct_reseed_probe_length", CT_RESEED_PROBE_LENGTH))
    args.ct_reseed_band_deficit_thr = float(
        getattr(args, "ct_reseed_band_deficit_thr", CT_RESEED_BAND_DEFICIT_THR)
    )
    args.ct_reseed_seed_offset_sigma = float(
        getattr(args, "ct_reseed_seed_offset_sigma", CT_RESEED_SEED_OFFSET_SIGMA)
    )
    args.ct_reseed_max_new_fraction = float(
        getattr(args, "ct_reseed_max_new_fraction", CT_RESEED_MAX_NEW_FRACTION)
    )
    args.ct_reseed_sigma_init_factor = float(
        getattr(args, "ct_reseed_sigma_init_factor", CT_RESEED_SIGMA_INIT_FACTOR)
    )
    args.ct_reseed_sigma_init_floor_ratio = float(
        getattr(args, "ct_reseed_sigma_init_floor_ratio", CT_RESEED_SIGMA_INIT_FLOOR_RATIO)
    )
    args.ct_reseed_atten_init_boost = float(
        getattr(args, "ct_reseed_atten_init_boost", CT_RESEED_ATTEN_INIT_BOOST)
    )
    args.ct_bulk_prune_interval = int(getattr(args, "ct_bulk_prune_interval", CT_BULK_PRUNE_INTERVAL))
    args.ct_bulk_prune_warmup = int(getattr(args, "ct_bulk_prune_warmup", CT_BULK_PRUNE_WARMUP))
    args.ct_bulk_prune_min_opacity = float(getattr(args, "ct_bulk_prune_min_opacity", CT_BULK_PRUNE_MIN_OPACITY))
    args.ct_bulk_prune_air_sdf_epsilon = float(
        getattr(args, "ct_bulk_prune_air_sdf_epsilon", CT_BULK_PRUNE_AIR_SDF_EPSILON)
    )
    args.ct_bulk_prune_raw_air_threshold = float(
        getattr(args, "ct_bulk_prune_raw_air_threshold", CT_BULK_PRUNE_RAW_AIR_THRESHOLD)
    )
    args.ct_bulk_prune_sample_count = int(getattr(args, "ct_bulk_prune_sample_count", CT_BULK_PRUNE_SAMPLE_COUNT))
    args.ct_bulk_prune_neighbor_radius = float(
        getattr(args, "ct_bulk_prune_neighbor_radius", CT_BULK_PRUNE_NEIGHBOR_RADIUS)
    )
    args.ct_bulk_prune_min_neighbors = int(getattr(args, "ct_bulk_prune_min_neighbors", CT_BULK_PRUNE_MIN_NEIGHBORS))
    args.ct_intensity_phase2_start_ratio = float(
        getattr(args, "ct_intensity_phase2_start_ratio", CT_INTENSITY_PHASE2_START_RATIO)
    )
    args.ct_intensity_phase3_start_ratio = float(
        getattr(args, "ct_intensity_phase3_start_ratio", CT_INTENSITY_PHASE3_START_RATIO)
    )
    args.ct_intensity_train_bulk_scale = bool(
        getattr(args, "ct_intensity_train_bulk_scale", CT_INTENSITY_TRAIN_BULK_SCALE)
    )
    args.ct_intensity_train_surface_opacity = bool(
        getattr(args, "ct_intensity_train_surface_opacity", CT_INTENSITY_TRAIN_SURFACE_OPACITY)
    )
    args.ct_intensity_bulk_scale_grad = float(
        getattr(args, "ct_intensity_bulk_scale_grad", CT_INTENSITY_BULK_SCALE_GRAD)
    )
    args.ct_intensity_sample_mode = str(
        getattr(args, "ct_intensity_sample_mode", CT_INTENSITY_SAMPLE_MODE)
    )
    args.ct_intensity_erode_margin_vox = float(
        getattr(args, "ct_intensity_erode_margin_vox", CT_INTENSITY_ERODE_MARGIN_VOX)
    )
    args.ct_intensity_den_min = float(
        getattr(args, "ct_intensity_den_min", CT_INTENSITY_DEN_MIN)
    )
    args.ct_atten_only_train_bulk_opacity = bool(
        getattr(args, "ct_atten_only_train_bulk_opacity", CT_ATTEN_ONLY_TRAIN_BULK_OPACITY)
    )
    args.ct_bulk_opacity_sparse_weight = float(
        getattr(args, "ct_bulk_opacity_sparse_weight", CT_BULK_OPACITY_SPARSE_WEIGHT)
    )
    args.ct_atten_only_lr_final_scale = float(
        getattr(args, "ct_atten_only_lr_final_scale", CT_ATTEN_ONLY_LR_FINAL_SCALE)
    )
    args.ct_atten_only_early_stop = bool(
        getattr(args, "ct_atten_only_early_stop", CT_ATTEN_ONLY_EARLY_STOP)
    )
    args.ct_atten_only_early_stop_warmup_iters = int(
        getattr(args, "ct_atten_only_early_stop_warmup_iters", CT_ATTEN_ONLY_EARLY_STOP_WARMUP_ITERS)
    )
    args.ct_atten_only_early_stop_eval_interval = int(
        getattr(args, "ct_atten_only_early_stop_eval_interval", CT_ATTEN_ONLY_EARLY_STOP_EVAL_INTERVAL)
    )
    args.ct_atten_only_early_stop_patience = int(
        getattr(args, "ct_atten_only_early_stop_patience", CT_ATTEN_ONLY_EARLY_STOP_PATIENCE)
    )
    args.ct_atten_only_early_stop_min_delta = float(
        getattr(args, "ct_atten_only_early_stop_min_delta", CT_ATTEN_ONLY_EARLY_STOP_MIN_DELTA)
    )
    args.ct_eagle_loss_weight = float(getattr(args, "ct_eagle_loss_weight", CT_EAGLE_LOSS_WEIGHT))
    args.ct_eagle_patch_size = int(getattr(args, "ct_eagle_patch_size", CT_EAGLE_PATCH_SIZE))
    args.ct_eagle_block_size = int(getattr(args, "ct_eagle_block_size", CT_EAGLE_BLOCK_SIZE))

def _normalize_bulk_atten_only_args(args) -> None:
    args.ct_freeze_surface = bool(getattr(args, "ct_freeze_surface", CT_FREEZE_SURFACE))
    args.ct_freeze_bulk_geometry = bool(getattr(args, "ct_freeze_bulk_geometry", CT_FREEZE_BULK_GEOMETRY))
    args.ct_train_bulk_atten_only = bool(getattr(args, "ct_train_bulk_atten_only", CT_TRAIN_BULK_ATTEN_ONLY))
    if not args.ct_train_bulk_atten_only:
        return
    args.ct_freeze_surface = True
    args.ct_freeze_bulk_geometry = True
    args.ct_enable_densification = False
    args.ct_enable_surface_reseeding = False
    args.ct_enable_bulk_reseeding = bool(
        getattr(args, "ct_gap_aware_reseed", False) or getattr(args, "ct_material_coverage_completion", False)
    )
    args.ct_lambda_occupancy = 0.0
    args.ct_surface_regularizer_weight = 0.0
    args.ct_intensity_train_bulk_scale = False
    args.ct_intensity_train_surface_opacity = False

def _normalize_ct_training_args(args):
    _normalize_path_and_io_args(args)
    _normalize_core_training_args(args)
    _normalize_internal_geometry_args(args)
    _normalize_surface_and_bulk_args(args)
    _normalize_densification_args(args)
    _normalize_bulk_atten_only_args(args)
