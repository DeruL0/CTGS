    from argparse import ArgumentParser, BooleanOptionalAction

from ct_pipeline.rendering.bulk_support import (
    DEFAULT_BULK_CONTAINMENT_Q_SUPPORT,
    DEFAULT_BULK_QUERY_TRUNCATION_SIGMA,
    resolve_bulk_query_truncation_sigma,
)
from ct_pipeline.config import add_ct_model_args, add_ct_optimization_args
from ct_pipeline.training.presets import CTTrainingArgumentParser, available_ct_training_presets


CT_TRAINING_MODE = "default"           # "default" | "role_separated_joint"
CT_BULK_ADAPTIVE_MODE = "fixed"        # "fixed" | "scale" | "scale_offset"
CT_BULK_SIGMA_MIN_MM = 0.0             # min bulk sigma in world mm (0 = no lower clamp)
CT_BULK_SIGMA_MAX_MM = 1e9             # max bulk sigma in world mm (large = no upper clamp)
CT_BULK_MAX_OFFSET_VOX = 0.25         # max center offset magnitude in voxels
CT_BULK_SCALE_ANCHOR_WEIGHT = 1e-3    # L_scale_anchor = ||log 蟽 - log 蟽0||虏
CT_BULK_POS_ANCHOR_WEIGHT = 1e-3      # L_pos_anchor = ||螖p||虏
CT_GAUSSIAN_TRUNCATION_SIGMA = 4.0
CT_SLICE_TILE_SIZE = 8
CT_GRID_CELL_VOXELS = 8
CT_GRID_CACHE_ENABLED = True
CT_SURFACE_GRID_REBUILD_INTERVAL = 10
CT_BULK_GRID_REBUILD_INTERVAL = 50
CT_GRID_CACHE_INFLATION_MARGIN = 0.25
CT_GRID_CACHE_DRIFT_CHECK = True
CT_GRID_CACHE_MAX_CELL_GAUSSIAN_PAIRS = 20_000_000
CT_HUBER_BETA = 0.1
CT_OCC_TAU_VOXELS = 1.0
CT_USE_UNIFIED_COMPOSITOR = True
CT_DUAL_SEPARATED_TRAINING = False
CT_BULK_FIELD_MODE = "bulk_intensity_field"
CT_BULK_HALFSPACE_ENABLE = True
CT_BULK_HALFSPACE_TAU_INIT = 0.5
CT_BULK_HALFSPACE_TAU_FINAL = 0.2
CT_BULK_HALFSPACE_TAU_GATE_THRESHOLD = 0.10
CT_BULK_HALFSPACE_TAU_STEP = 0.05
CT_BULK_HALFSPACE_TAU_UPDATE_INTERVAL = 250
CT_BULK_HALFSPACE_SKIP_DEPTH = 2.0
CT_BULK_SCALE_ADAPTIVE_CAP = True
CT_BULK_SCALE_GLOBAL_MAX = 1.5
CT_BULK_SCALE_FLOOR = 0.05
CT_EXTERIOR_AIR_SAMPLE_RATIO = 0.15
CT_BULK_VOLUME_SUPPORT_SAMPLE_RATIO = 0.75
CT_VOLUME_JITTER = 0.5
CT_BOUNDARY_BAND = 1.5
CT_OCC_TAU = 1.0
CT_SURFACE_SIGMA_N_MAX = 0.45
CT_SURFACE_SIGMA_T_MIN = 0.2
CT_SURFACE_MAX_SCALE = 1.4
CT_SURFACE_NORMAL_WEIGHT = 1.0
CT_SURFACE_THICKNESS_WEIGHT = 0.25
CT_SURFACE_THICKNESS_BETA = 0.25
CT_SURFACE_SPREAD_BETA = 0.2
CT_SURFACE_OUTSIDE_BETA = 16.0
CT_SURFACE_REGULARIZER_SAMPLE_COUNT = 8192
CT_SURFACE_COVERAGE_WEIGHT = 0.08
CT_SURFACE_COVERAGE_SAMPLE_COUNT = 4096
CT_SURFACE_COVERAGE_TARGET = 0.65
CT_SURFACE_COVERAGE_UNTIL_ITER = 4000
CT_SURFACE_MATERIAL_GATE_SIGMA = None
CT_SURFACE_MATERIAL_DELTA = 0.25
CT_SURFACE_MATERIAL_TAU = 0.25
CT_SURFACE_MATERIAL_RENDER_WEIGHT = 0.0
CT_SURFACE_INTENSITY_WEIGHT = 0.05
CT_SURFACE_INTENSITY_SAMPLE_COUNT = 2048
CT_SURFACE_INTENSITY_MATERIAL_RATIO = 0.75
CT_DUAL_SURFACE_INNER_BAND = 0.0
CT_DUAL_SURFACE_AIR_BAND = 0.25
CT_DUAL_SOFT_SIGMA = 0.5
CT_DUAL_SURFACE_AIR_SAMPLE_RATIO = 0.25
CT_BULK_RENDERER_EPSILON = 0.0
CT_BULK_RENDERER_TAU = 0.25
CT_MATERIAL_COMPOSE_MODE = "bulk_first_material"
CT_BULK_COVERAGE_WEIGHT = 0.5
CT_BULK_COVERAGE_SAMPLE_COUNT = 8192
CT_BULK_COVERAGE_TARGET = 0.85
CT_BULK_SEMANTIC_WEIGHT = 0.0
CT_BULK_SEMANTIC_SAMPLE_COUNT = 8192
CT_BULK_SEMANTIC_FOCAL_GAMMA = 2.0
CT_BULK_SEMANTIC_EPSILON = 0.5
CT_BULK_SEMANTIC_POS_RATIO = 0.55
CT_BULK_SEMANTIC_DEEP_RATIO = 1.0
CT_BULK_SEMANTIC_BOUNDARY_RATIO = 0.0
CT_BULK_SEMANTIC_CAVITY_MATERIAL_RATIO = 0.0
CT_BULK_SEMANTIC_VOID_AIR_RATIO = 0.45
CT_BULK_SEMANTIC_CAVITY_AIR_RATIO = 0.35
CT_BULK_SEMANTIC_EXTERIOR_AIR_RATIO = 0.20
CT_FOCAL_GAMMA_POS = 0.0
CT_FOCAL_GAMMA_NEG = 2.0
CT_FOCAL_ALPHA_POS = 0.70
CT_BULK_SEMANTIC_BOUNDARY_TARGET = 1.0
CT_BULK_SEMANTIC_CAVITY_TARGET = 1.0
CT_BULK_FLOOR_WEIGHT = 0.15
CT_BULK_FLOOR_DEEP_TARGET = 1.0
CT_BULK_FLOOR_SHELL_TARGET = 1.0
CT_BULK_VOID_WEIGHT = 1.0
CT_BULK_AIR_EXCLUSION_WEIGHT = 1.0
CT_BULK_AIR_EXCLUSION_SAMPLE_COUNT = 4096
CT_BULK_AIR_EXCLUSION_TARGET = 0.05
CT_CAVITY_MATERIAL_COVERAGE_WEIGHT = 3.0
CT_CAVITY_FALSE_HOLE_BOOST = 2.0
CT_BULK_SDF_CONTAINMENT_WEIGHT = 1.0
CT_BULK_SDF_CONTAINMENT_SAMPLE_COUNT = 8192
CT_BULK_SDF_CONTAINMENT_MARGIN = 0.0
CT_CONTAINMENT_RAMP_ITER_1 = 1000
CT_CONTAINMENT_RAMP_WEIGHT_1 = 0.05
CT_CONTAINMENT_RAMP_ITER_2 = 5000
CT_CONTAINMENT_RAMP_WEIGHT_2 = 0.5
CT_CONTAINMENT_RAMP_ITER_3 = 8000
CT_CONTAINMENT_RAMP_WEIGHT_3 = 1.0
CT_RESEED_PROBE_LENGTH = 3.0
CT_RESEED_BAND_DEFICIT_THR = 0.15
CT_RESEED_SEED_OFFSET_SIGMA = 1.0
CT_RESEED_MAX_NEW_FRACTION = 0.05
CT_RESEED_SIGMA_INIT_FACTOR = 1.0
CT_RESEED_SIGMA_INIT_FLOOR_RATIO = 0.5
CT_RESEED_ATTEN_INIT_BOOST = 2.0
CT_GAP_AWARE_RESEED = False
CT_GAP_RESEED_DEN_TARGET = 0.90
CT_GAP_RESEED_SAMPLE_RATIO = 0.60
CT_GAP_RESEED_RADIUS_VOX = 1.0
CT_GAP_RESEED_MAX_PER_ITER = 0
CT_GAP_RESEED_BOUNDARY_SUBVOXEL = False
CT_GAP_RESEED_PROTECT_PRUNE = False
CT_GAP_RESEED_PROTECT_ITERS = 300
CT_GAP_BULK_GROWTH_FACTOR = 1.03
CT_BUDGETED_COMPONENT_REPAIR = True
CT_REPAIR_ADD_THRESHOLD = 0.50
CT_REPAIR_STOP_THRESHOLD = 0.85
CT_REPAIR_MIN_COMPONENT_POINTS = 16
CT_REPAIR_MAX_NEW_PER_PASS = 1000
CT_REPAIR_MAX_NEW_FRACTION = 0.005
CT_REPAIR_GAIN_RATIO_MIN = 0.15
CT_REPAIR_EXCLUSION_RADIUS_VOX = 0.75
CT_REPAIR_STRETCH_FIRST = True
CT_REPAIR_STRETCH_GROWTH_FACTOR = 1.15
CT_REPAIR_STRETCH_SECONDARY_FACTOR = 1.10
CT_REPAIR_STRETCH_MAX_RATIO = 4.0
CT_REPAIR_OVERFILL_THRESHOLD = 1.25
CT_REPAIR_TOP_COMPONENTS = 256
CT_REPAIR_NEARBY_CANDIDATES = 8
CT_REPAIR_PROBE_TANGENT_FACTOR = 1.5
CT_REPAIR_PROBE_SHRINK = 0.75
CT_REPAIR_MAX_PROBE_SHRINK_ITERS = 3
CT_REPAIR_CHECK_STRIDE = 2
CT_REPAIR_MAX_CHECK_POINTS = 200000
CT_MATERIAL_COVERAGE_COMPLETION = False
CT_COMPLETION_INIT = True
CT_COMPLETION_REPAIR = True
CT_COMPLETION_DEN_TARGET = 0.90
CT_COMPLETION_RADIUS_VOX = 0.75
CT_COMPLETION_MAX_INIT_PASSES = 1
CT_COMPLETION_MAX_NEW_PER_PASS = 1500
CT_COMPLETION_MIN_COMPONENT_VOXELS = 32
CT_COMPLETION_CHECK_STRIDE = 2
CT_COMPLETION_MAX_CHECK_POINTS = 20000
CT_SDF_BOUNDARY_MODE = "interface"
CT_FEATURE_ADAPTIVE_JITTER = True
CT_FEATURE_ADAPTIVE_SEED = 17
CT_FEATURE_ADAPTIVE_R_SHELL_VOX = 3.0
CT_FEATURE_ADAPTIVE_BLUR_SIGMA_VOX = 0.75
CT_FEATURE_ADAPTIVE_SPACING_HIGH_VOX = 2
CT_FEATURE_ADAPTIVE_SPACING_MID_VOX = 6
CT_FEATURE_ADAPTIVE_SPACING_LOW_VOX = 10
CT_FEATURE_ADAPTIVE_DIRECTIONAL_CLEARANCE = True
CT_FEATURE_ADAPTIVE_PROBE_CONTAINMENT = True
CT_BULK_CONTAINMENT_Q_SUPPORT = DEFAULT_BULK_CONTAINMENT_Q_SUPPORT
CT_BULK_QUERY_TRUNCATION_SIGMA = DEFAULT_BULK_QUERY_TRUNCATION_SIGMA
CT_INIT_PREFLIGHT_ABORT = True
CT_INIT_PREFLIGHT_MAX_CONTAINMENT_VIOLATION = 0.01
CT_INIT_PREFLIGHT_MIN_MATERIAL_A_B_P10 = 0.70
CT_INIT_PREFLIGHT_MAX_MATERIAL_COVERAGE_GAP = 0.001
CT_BULK_COVERAGE_GAP_THRESHOLD = 0.50
CT_LOSS_BOUNDARY_BAND_DELTA = 0.5
CT_LOSS_BAND_IN_WEIGHT = 1.0
CT_LOSS_BAND_OUT_WEIGHT = 0.25
# ---- v5.2.1 coverage-first init / stage warmup ----
CT_INIT_STRATEGY = "volume_sampled"          # legacy default; "coverage_first" enables v5.2.1
CT_INIT_COVERAGE_NORMALIZED_ATTEN = True     # invariant 3
CT_INIT_BOUNDARY_INWARD_NUDGE = True         # invariant 2 (seed at D 鈮?-蟽_safe, not D=0)
CT_INIT_BOUNDARY_SIGMA_C = 0.7               # invariant 1: 蟽 鈮?c路|D|; large value disables
CT_INIT_DEEP_SIGMA_VOXEL = 1.5               # 蟽 assigned to deep bulk gaussians (voxel units)
CT_INIT_DEEP_DEPTH_VOXEL = 2.0               # |D| > this counts as "deep"
CT_INIT_INWARD_NUDGE_SIGMA_RATIO = 0.7       # nudge target D = -ratio 路 蟽_target
CT_INIT_COVERAGE_KNN_K = 16                  # KNN size for coverage-normalized atten
CT_INIT_COVERAGE_REPORT = True               # emit init_report.txt with C(x) stats
CT_INIT_COVERAGE_C_MIN_MATERIAL = 0.5        # validation gate: material p10 鈮?this
CT_INIT_COVERAGE_C_MIN_SHELL = 0.3           # validation gate: shell p10 鈮?this
CT_INIT_COVERAGE_VOID_EPSILON = 0.05         # validation gate: void p95 鈮?this
CT_STAGE1_FREEZE_UNTIL_ITER = 0              # 0 = disabled; e.g. 1000 = Stage 1 蟽/渭 frozen warmup
CT_FALSE_HOLE_SAMPLE_COUNT = 4096
CT_FALSE_HOLE_BOUNDARY_BAND = 2.0
CT_FALSE_HOLE_MATERIAL_THRESHOLD = 0.65
CT_FALSE_HOLE_DARK_MARGIN = 0.15
CT_FALSE_HOLE_TARGET_OCCUPANCY = 0.90
CT_FALSE_HOLE_METRICS_INTERVAL = 100
CT_FREEZE_BULK_XYZ = True
CT_FREEZE_SURFACE = False
CT_FREEZE_BULK_GEOMETRY = False
CT_TRAIN_BULK_ATTEN_ONLY = False
CT_BULK_CONTINUOUS_INIT = True
CT_FREEZE_PRIMITIVE_TYPE = True
CT_DENSIFY_FROM_ITER = 1000
CT_DENSIFY_UNTIL_ITER = 15000
CT_DENSIFY_INTERVAL = 500
CT_DENSIFY_SURFACE_PERCENT = 0.0
CT_DENSIFY_BULK_PERCENT = 0.0
CT_DENSIFY_MAX_GAUSSIAN_RATIO = 1.5
CT_DENSIFY_MIN_OPACITY = 0.35
CT_DENSIFY_SURFACE_TANGENT_RATIO = 0.8
CT_DENSIFY_BULK_SCALE_RATIO = 0.8
CT_SURFACE_RESEED_FROM_ITER = 500
CT_SURFACE_RESEED_UNTIL_ITER = 2500
CT_SURFACE_RESEED_INTERVAL = 500
CT_SURFACE_RESEED_SAMPLE_COUNT = 4096
CT_SURFACE_RESEED_MAX_NEW_PER_ITER = 2048
CT_SURFACE_RESEED_MAX_GAUSSIAN_RATIO = 1.5
CT_SURFACE_RESEED_MIN_BULK_OCCUPANCY = 0.35
CT_SURFACE_RESEED_MAX_SURFACE_OCCUPANCY = 0.15
CT_SURFACE_RESEED_COVERAGE_RADIUS_VOXELS = 0.3
CT_BULK_PRUNE_INTERVAL = 1000
CT_BULK_PRUNE_WARMUP = 1000
CT_BULK_PRUNE_MIN_OPACITY = 0.05
CT_BULK_PRUNE_AIR_SDF_EPSILON = 0.5
CT_BULK_PRUNE_RAW_AIR_THRESHOLD = 0.35
CT_BULK_PRUNE_SAMPLE_COUNT = 4096
CT_BULK_PRUNE_NEIGHBOR_RADIUS = 2.0
CT_BULK_PRUNE_MIN_NEIGHBORS = 1
CT_INTENSITY_PHASE2_START_RATIO = 0.6
CT_INTENSITY_PHASE3_START_RATIO = 0.85
CT_INTENSITY_TRAIN_BULK_SCALE = False
CT_INTENSITY_TRAIN_SURFACE_OPACITY = False
CT_INTENSITY_BULK_SCALE_GRAD = 0.1
CT_INTENSITY_SAMPLE_MODE = "full_band"
CT_INTENSITY_ERODE_MARGIN_VOX = 0.5
CT_INTENSITY_DEN_MIN = 0.0
CT_ATTEN_ONLY_TRAIN_BULK_OPACITY = True
CT_BULK_OPACITY_SPARSE_WEIGHT = 1e-3
CT_ATTEN_ONLY_LR_FINAL_SCALE = 0.25
CT_ATTEN_ONLY_EARLY_STOP = True
CT_ATTEN_ONLY_EARLY_STOP_WARMUP_ITERS = 100
CT_ATTEN_ONLY_EARLY_STOP_EVAL_INTERVAL = 100
CT_ATTEN_ONLY_EARLY_STOP_PATIENCE = 2
CT_ATTEN_ONLY_EARLY_STOP_MIN_DELTA = 1e-4
CT_EAGLE_LOSS_WEIGHT = 0.0
CT_EAGLE_PATCH_SIZE = 64
CT_EAGLE_BLOCK_SIZE = 16
CT_CAVITY_PATCH_BIAS = 0.6
CT_PREVIEW_INTERVAL = 0


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
    args.ct_volume_jitter = float(getattr(args, "ct_volume_jitter", getattr(args, "ct_volume_jitter_voxels", CT_VOLUME_JITTER)))
    args.ct_volume_jitter_voxels = args.ct_volume_jitter
    args.ct_boundary_band = float(getattr(args, "ct_boundary_band", getattr(args, "ct_boundary_band_voxels", CT_BOUNDARY_BAND)))
    args.ct_boundary_band_voxels = args.ct_boundary_band
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
    args.ct_occ_sample_count = args.ct_occupancy_sample_count
    args.ct_occ_boundary_sample_ratio = float(getattr(args, "ct_occ_boundary_sample_ratio", 0.50))
    args.ct_occ_deep_material_sample_ratio = float(getattr(args, "ct_occ_deep_material_sample_ratio", 0.35))
    args.ct_occ_boundary_weight = float(getattr(args, "ct_occ_boundary_weight", 3.0))
    args.ct_exterior_air_sample_ratio = float(getattr(args, "ct_exterior_air_sample_ratio", CT_EXTERIOR_AIR_SAMPLE_RATIO))
    args.ct_bulk_volume_support_sample_ratio = float(
        getattr(args, "ct_bulk_volume_support_sample_ratio", CT_BULK_VOLUME_SUPPORT_SAMPLE_RATIO)
    )
    args.ct_huber_beta = CT_HUBER_BETA
    args.ct_occ_tau = float(getattr(args, "ct_occ_tau", getattr(args, "ct_occ_tau_voxels", CT_OCC_TAU)))
    args.ct_occ_tau_voxels = args.ct_occ_tau


def _normalize_internal_geometry_args(args) -> None:
    args.ct_cavity_patch_bias = CT_CAVITY_PATCH_BIAS
    args.ct_gaussian_truncation_sigma = CT_GAUSSIAN_TRUNCATION_SIGMA
    args.ct_slice_tile_size = CT_SLICE_TILE_SIZE
    args.ct_grid_cell_voxels = CT_GRID_CELL_VOXELS
    args.ct_grid_cache = CT_GRID_CACHE_ENABLED
    args.ct_surface_grid_rebuild_interval = CT_SURFACE_GRID_REBUILD_INTERVAL
    args.ct_bulk_grid_rebuild_interval = CT_BULK_GRID_REBUILD_INTERVAL
    args.ct_grid_cache_inflation_margin = CT_GRID_CACHE_INFLATION_MARGIN
    args.ct_grid_cache_drift_check = CT_GRID_CACHE_DRIFT_CHECK
    args.ct_grid_cache_max_cell_gaussian_pairs = CT_GRID_CACHE_MAX_CELL_GAUSSIAN_PAIRS
    args.ct_surface_sigma_n_max = float(
        getattr(args, "ct_surface_sigma_n_max", getattr(args, "ct_surface_sigma_n_max_voxels", CT_SURFACE_SIGMA_N_MAX))
    )
    args.ct_surface_sigma_n_max_voxels = args.ct_surface_sigma_n_max
    args.ct_surface_sigma_t_min = float(
        getattr(args, "ct_surface_sigma_t_min", getattr(args, "ct_surface_sigma_t_min_voxels", CT_SURFACE_SIGMA_T_MIN))
    )
    args.ct_surface_sigma_t_min_voxels = args.ct_surface_sigma_t_min
    args.ct_surface_max_scale = float(
        getattr(args, "ct_surface_max_scale", getattr(args, "ct_surface_max_scale_voxels", CT_SURFACE_MAX_SCALE))
    )
    args.ct_surface_max_scale_voxels = args.ct_surface_max_scale
    args.ct_bulk_scale_adaptive_cap = bool(
        getattr(args, "ct_bulk_scale_adaptive_cap", CT_BULK_SCALE_ADAPTIVE_CAP)
    )
    args.ct_bulk_scale_global_max = float(
        getattr(args, "ct_bulk_scale_global_max", CT_BULK_SCALE_GLOBAL_MAX)
    )
    args.ct_bulk_scale_floor = float(getattr(args, "ct_bulk_scale_floor", CT_BULK_SCALE_FLOOR))
    args.ct_surface_normal_weight = float(getattr(args, "ct_surface_normal_weight", CT_SURFACE_NORMAL_WEIGHT))
    args.ct_surface_thickness_weight = float(getattr(args, "ct_surface_thickness_weight", CT_SURFACE_THICKNESS_WEIGHT))
    args.ct_surface_thickness_beta = CT_SURFACE_THICKNESS_BETA
    args.ct_surface_spread_beta = CT_SURFACE_SPREAD_BETA
    args.ct_surface_outside_beta = CT_SURFACE_OUTSIDE_BETA
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
    if args.ct_occ_tau <= 0.0:
        raise ValueError("--ct_occ_tau must be > 0.")
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


def build_parser():
    parser = CTTrainingArgumentParser(description="CT training script parameters")
    add_ct_model_args(parser)
    add_ct_optimization_args(parser)
    parser.add_argument(
        "--ct_preset",
        choices=available_ct_training_presets(),
        default=None,
        help="Apply a named CT training recipe before explicit CLI overrides.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--load_ply", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--ct_phase1_dir", type=str, default=None)
    parser.add_argument("--ct_volume_path", type=str, default=None)
    parser.add_argument("--ct_volume_format", type=str, default="auto", choices=["auto", "dicom", "raw", "tiff"])
    parser.add_argument("--ct_raw_meta", type=str, default=None)
    parser.add_argument("--ct_bulk_boundary_margin_voxels", type=int, default=1)
    parser.add_argument("--ct_support_sample_count", type=int, default=None)
    parser.add_argument("--ct_air_sample_count", type=int, default=None)
    parser.add_argument("--ct_volume_sample_count", type=int, default=4096)
    parser.add_argument("--ct_volume_jitter", "--ct_volume_jitter_voxels", dest="ct_volume_jitter", type=float, default=CT_VOLUME_JITTER)
    parser.add_argument("--ct_boundary_band", "--ct_boundary_band_voxels", dest="ct_boundary_band", type=float, default=CT_BOUNDARY_BAND)
    parser.add_argument("--ct_surface_boundary_sample_ratio", type=float, default=0.5)
    parser.add_argument("--ct_lambda_volume", type=float, default=1.0)
    parser.add_argument("--ct_lambda_occupancy", type=float, default=1.0)
    parser.add_argument("--ct_surface_regularizer_weight", type=float, default=0.5)
    parser.add_argument("--ct_training_mode", type=str, default=CT_TRAINING_MODE,
                        choices=["default", "role_separated_joint"],
                        help="Training mode: default (current) or role_separated_joint (Method C)")
    parser.add_argument("--ct_bulk_adaptive_mode", type=str, default=CT_BULK_ADAPTIVE_MODE,
                        choices=["fixed", "scale", "scale_offset"],
                        help="Bulk geometry adaptation: fixed (geometry frozen) | scale (trainable 蟽) | scale_offset (蟽 + bounded 螖p)")
    parser.add_argument("--ct_bulk_sigma_min_mm", type=float, default=CT_BULK_SIGMA_MIN_MM,
                        help="Min bulk sigma in world mm (0 = no clamp). E.g. 0.10 for bunny (spacing=0.39mm)")
    parser.add_argument("--ct_bulk_sigma_max_mm", type=float, default=CT_BULK_SIGMA_MAX_MM,
                        help="Max bulk sigma in world mm (large = no clamp). E.g. 1.5 for bunny")
    parser.add_argument("--ct_bulk_max_offset_vox", type=float, default=CT_BULK_MAX_OFFSET_VOX)
    parser.add_argument("--ct_bulk_scale_anchor_weight", type=float, default=CT_BULK_SCALE_ANCHOR_WEIGHT)
    parser.add_argument("--ct_bulk_pos_anchor_weight", type=float, default=CT_BULK_POS_ANCHOR_WEIGHT)
    # role_separated_joint config
    parser.add_argument("--ct_surface_phase_loss_weight", type=float, default=1.0)
    parser.add_argument("--ct_surface_phase_margin_vox", type=float, default=0.5,
                        help="Margin in voxels for surface phase loss (inside/outside pushes)")
    parser.add_argument("--ct_surface_phase_temp_vox", type=float, default=0.1,
                        help="Softplus temperature for phase loss")
    parser.add_argument("--ct_surface_normal_smooth_weight", type=float, default=0.05)
    parser.add_argument("--ct_surface_anchor_weight", type=float, default=0.01)
    # confidence map config
    parser.add_argument("--ct_confidence_mode", type=str, default="auto_percentile",
                        choices=["auto_percentile", "static_threshold"])
    parser.add_argument("--ct_conf_material_threshold", type=float, default=None)
    parser.add_argument("--ct_conf_air_threshold", type=float, default=None)
    parser.add_argument("--ct_conf_unknown_band_width", type=float, default=0.0)
    parser.add_argument("--ct_conf_min_component_size", type=int, default=500)
    parser.add_argument("--ct_use_unified_compositor", action=BooleanOptionalAction, default=CT_USE_UNIFIED_COMPOSITOR)
    parser.add_argument("--ct_dual_separated_training", action=BooleanOptionalAction, default=CT_DUAL_SEPARATED_TRAINING)
    parser.add_argument("--ct_bulk_halfspace_enable", action=BooleanOptionalAction, default=CT_BULK_HALFSPACE_ENABLE)
    parser.add_argument("--ct_bulk_halfspace_tau_init", type=float, default=CT_BULK_HALFSPACE_TAU_INIT)
    parser.add_argument("--ct_bulk_halfspace_tau_final", type=float, default=CT_BULK_HALFSPACE_TAU_FINAL)
    parser.add_argument("--ct_bulk_halfspace_tau_gate_threshold", type=float, default=CT_BULK_HALFSPACE_TAU_GATE_THRESHOLD)
    parser.add_argument("--ct_bulk_halfspace_tau_step", type=float, default=CT_BULK_HALFSPACE_TAU_STEP)
    parser.add_argument("--ct_bulk_halfspace_tau_update_interval", type=int, default=CT_BULK_HALFSPACE_TAU_UPDATE_INTERVAL)
    parser.add_argument("--ct_bulk_halfspace_skip_depth", type=float, default=CT_BULK_HALFSPACE_SKIP_DEPTH)
    parser.add_argument("--ct_occ_tau", "--ct_occ_tau_voxels", dest="ct_occ_tau", type=float, default=CT_OCC_TAU)
    parser.add_argument("--ct_exterior_air_sample_ratio", type=float, default=CT_EXTERIOR_AIR_SAMPLE_RATIO)
    parser.add_argument("--ct_bulk_volume_support_sample_ratio", type=float, default=CT_BULK_VOLUME_SUPPORT_SAMPLE_RATIO)
    parser.add_argument("--ct_surface_sigma_n_max", "--ct_surface_sigma_n_max_voxels", dest="ct_surface_sigma_n_max", type=float, default=CT_SURFACE_SIGMA_N_MAX)
    parser.add_argument("--ct_surface_sigma_t_min", "--ct_surface_sigma_t_min_voxels", dest="ct_surface_sigma_t_min", type=float, default=CT_SURFACE_SIGMA_T_MIN)
    parser.add_argument("--ct_surface_max_scale", "--ct_surface_max_scale_voxels", dest="ct_surface_max_scale", type=float, default=CT_SURFACE_MAX_SCALE)
    parser.add_argument("--ct_bulk_scale_adaptive_cap", action=BooleanOptionalAction, default=CT_BULK_SCALE_ADAPTIVE_CAP)
    parser.add_argument("--ct_bulk_scale_global_max", type=float, default=CT_BULK_SCALE_GLOBAL_MAX)
    parser.add_argument("--ct_bulk_scale_floor", type=float, default=CT_BULK_SCALE_FLOOR)
    # coverage-driven bulk sigma growth (geometry-grow phase): grow sigma to fill
    # material interior coverage; mask membership keeps it out of air/cavities.
    parser.add_argument("--ct_bulk_coverage_growth_weight", type=float, default=0.0)
    parser.add_argument("--ct_bulk_coverage_growth_sample_count", type=int, default=4096)
    parser.add_argument("--ct_bulk_coverage_growth_margin_vox", type=float, default=0.5)
    parser.add_argument("--ct_bulk_coverage_growth_scale_grad", type=float, default=1.0)
    parser.add_argument("--ct_bulk_coverage_growth_train_opacity", action=BooleanOptionalAction, default=False)
    # anti-leak: penalize UNGATED bulk density on void so the ungated display stays clean.
    parser.add_argument("--ct_bulk_void_leak_weight", type=float, default=0.0)
    parser.add_argument("--ct_bulk_void_leak_sample_count", type=int, default=4096)
    parser.add_argument("--ct_bulk_void_leak_margin_vox", type=float, default=0.5)
    parser.add_argument("--ct_surface_normal_weight", type=float, default=CT_SURFACE_NORMAL_WEIGHT)
    parser.add_argument("--ct_surface_thickness_weight", type=float, default=CT_SURFACE_THICKNESS_WEIGHT)
    parser.add_argument("--ct_surface_coverage_weight", type=float, default=CT_SURFACE_COVERAGE_WEIGHT)
    parser.add_argument("--ct_surface_coverage_sample_count", type=int, default=CT_SURFACE_COVERAGE_SAMPLE_COUNT)
    parser.add_argument("--ct_surface_coverage_target", type=float, default=CT_SURFACE_COVERAGE_TARGET)
    parser.add_argument("--ct_surface_coverage_until_iter", type=int, default=CT_SURFACE_COVERAGE_UNTIL_ITER)
    parser.add_argument("--ct_surface_material_gate_sigma", type=float, default=CT_SURFACE_MATERIAL_GATE_SIGMA)
    parser.add_argument("--ct_surface_material_delta", type=float, default=CT_SURFACE_MATERIAL_DELTA)
    parser.add_argument("--ct_surface_material_tau", type=float, default=CT_SURFACE_MATERIAL_TAU)
    parser.add_argument("--ct_surface_material_render_weight", type=float, default=CT_SURFACE_MATERIAL_RENDER_WEIGHT)
    parser.add_argument("--ct_surface_intensity_weight", type=float, default=CT_SURFACE_INTENSITY_WEIGHT)
    parser.add_argument("--ct_surface_intensity_sample_count", type=int, default=CT_SURFACE_INTENSITY_SAMPLE_COUNT)
    parser.add_argument("--ct_surface_intensity_material_ratio", type=float, default=CT_SURFACE_INTENSITY_MATERIAL_RATIO)
    parser.add_argument("--ct_dual_surface_inner_band", type=float, default=CT_DUAL_SURFACE_INNER_BAND)
    parser.add_argument("--ct_dual_surface_air_band", type=float, default=CT_DUAL_SURFACE_AIR_BAND)
    parser.add_argument("--ct_dual_soft_sigma", type=float, default=CT_DUAL_SOFT_SIGMA)
    parser.add_argument("--ct_dual_surface_air_sample_ratio", type=float, default=CT_DUAL_SURFACE_AIR_SAMPLE_RATIO)
    parser.add_argument("--ct_bulk_renderer_epsilon", type=float, default=CT_BULK_RENDERER_EPSILON)
    parser.add_argument("--ct_bulk_renderer_tau", type=float, default=CT_BULK_RENDERER_TAU)
    parser.add_argument("--ct_material_compose_mode", choices=["surface_first", "bulk_first_material"], default=CT_MATERIAL_COMPOSE_MODE)
    parser.add_argument("--ct_bulk_semantic_weight", type=float, default=CT_BULK_SEMANTIC_WEIGHT)
    parser.add_argument("--ct_bulk_semantic_sample_count", type=int, default=CT_BULK_SEMANTIC_SAMPLE_COUNT)
    parser.add_argument("--ct_bulk_semantic_focal_gamma", type=float, default=CT_BULK_SEMANTIC_FOCAL_GAMMA)
    parser.add_argument("--ct_bulk_semantic_epsilon", type=float, default=CT_BULK_SEMANTIC_EPSILON)
    parser.add_argument("--ct_bulk_semantic_pos_ratio", type=float, default=CT_BULK_SEMANTIC_POS_RATIO)
    parser.add_argument("--ct_bulk_semantic_deep_ratio", type=float, default=CT_BULK_SEMANTIC_DEEP_RATIO)
    parser.add_argument("--ct_bulk_semantic_boundary_ratio", type=float, default=CT_BULK_SEMANTIC_BOUNDARY_RATIO)
    parser.add_argument("--ct_bulk_semantic_cavity_material_ratio", type=float, default=CT_BULK_SEMANTIC_CAVITY_MATERIAL_RATIO)
    parser.add_argument("--ct_bulk_semantic_void_air_ratio", type=float, default=CT_BULK_SEMANTIC_VOID_AIR_RATIO)
    parser.add_argument("--ct_bulk_semantic_cavity_air_ratio", type=float, default=CT_BULK_SEMANTIC_CAVITY_AIR_RATIO)
    parser.add_argument("--ct_bulk_semantic_exterior_air_ratio", type=float, default=CT_BULK_SEMANTIC_EXTERIOR_AIR_RATIO)
    parser.add_argument("--ct_focal_gamma_pos", type=float, default=CT_FOCAL_GAMMA_POS)
    parser.add_argument("--ct_focal_gamma_neg", type=float, default=CT_FOCAL_GAMMA_NEG)
    parser.add_argument("--ct_focal_alpha_pos", type=float, default=CT_FOCAL_ALPHA_POS)
    parser.add_argument("--ct_bulk_semantic_boundary_target", type=float, default=CT_BULK_SEMANTIC_BOUNDARY_TARGET)
    parser.add_argument("--ct_bulk_semantic_cavity_target", type=float, default=CT_BULK_SEMANTIC_CAVITY_TARGET)
    parser.add_argument("--ct_bulk_floor_weight", type=float, default=CT_BULK_FLOOR_WEIGHT)
    parser.add_argument("--ct_bulk_floor_deep_target", type=float, default=CT_BULK_FLOOR_DEEP_TARGET)
    parser.add_argument("--ct_bulk_floor_shell_target", type=float, default=CT_BULK_FLOOR_SHELL_TARGET)
    parser.add_argument("--ct_bulk_void_weight", type=float, default=CT_BULK_VOID_WEIGHT)
    parser.add_argument("--ct_loss_boundary_band_delta", type=float, default=CT_LOSS_BOUNDARY_BAND_DELTA)
    parser.add_argument("--ct_loss_band_in_weight", type=float, default=CT_LOSS_BAND_IN_WEIGHT)
    parser.add_argument("--ct_loss_band_out_weight", type=float, default=CT_LOSS_BAND_OUT_WEIGHT)
    parser.add_argument("--ct_bulk_coverage_weight", type=float, default=CT_BULK_COVERAGE_WEIGHT)
    parser.add_argument("--ct_bulk_coverage_sample_count", type=int, default=CT_BULK_COVERAGE_SAMPLE_COUNT)
    parser.add_argument("--ct_bulk_coverage_target", type=float, default=CT_BULK_COVERAGE_TARGET)
    parser.add_argument("--ct_bulk_air_exclusion_weight", type=float, default=CT_BULK_AIR_EXCLUSION_WEIGHT)
    parser.add_argument("--ct_bulk_air_exclusion_sample_count", type=int, default=CT_BULK_AIR_EXCLUSION_SAMPLE_COUNT)
    parser.add_argument("--ct_bulk_air_exclusion_target", type=float, default=CT_BULK_AIR_EXCLUSION_TARGET)
    parser.add_argument("--ct_cavity_material_coverage_weight", type=float, default=CT_CAVITY_MATERIAL_COVERAGE_WEIGHT)
    parser.add_argument("--ct_cavity_false_hole_boost", type=float, default=CT_CAVITY_FALSE_HOLE_BOOST)
    parser.add_argument("--ct_containment_weight", type=float, default=CT_BULK_SDF_CONTAINMENT_WEIGHT)
    parser.add_argument("--ct_bulk_sdf_containment_sample_count", type=int, default=CT_BULK_SDF_CONTAINMENT_SAMPLE_COUNT)
    parser.add_argument("--ct_bulk_sdf_containment_margin", type=float, default=CT_BULK_SDF_CONTAINMENT_MARGIN)
    parser.add_argument("--ct_init_strategy", type=str, default=CT_INIT_STRATEGY, choices=["volume_sampled", "coverage_first"])
    parser.add_argument("--ct_init_coverage_normalized_atten", action=BooleanOptionalAction, default=CT_INIT_COVERAGE_NORMALIZED_ATTEN)
    parser.add_argument("--ct_init_boundary_inward_nudge", action=BooleanOptionalAction, default=CT_INIT_BOUNDARY_INWARD_NUDGE)
    parser.add_argument("--ct_init_boundary_sigma_c", type=float, default=CT_INIT_BOUNDARY_SIGMA_C)
    parser.add_argument("--ct_init_deep_sigma_voxel", type=float, default=CT_INIT_DEEP_SIGMA_VOXEL)
    parser.add_argument("--ct_init_deep_depth_voxel", type=float, default=CT_INIT_DEEP_DEPTH_VOXEL)
    parser.add_argument("--ct_init_inward_nudge_sigma_ratio", type=float, default=CT_INIT_INWARD_NUDGE_SIGMA_RATIO)
    parser.add_argument("--ct_init_coverage_knn_k", type=int, default=CT_INIT_COVERAGE_KNN_K)
    parser.add_argument("--ct_init_coverage_report", action=BooleanOptionalAction, default=CT_INIT_COVERAGE_REPORT)
    parser.add_argument("--ct_init_coverage_c_min_material", type=float, default=CT_INIT_COVERAGE_C_MIN_MATERIAL)
    parser.add_argument("--ct_init_coverage_c_min_shell", type=float, default=CT_INIT_COVERAGE_C_MIN_SHELL)
    parser.add_argument("--ct_init_coverage_void_epsilon", type=float, default=CT_INIT_COVERAGE_VOID_EPSILON)
    parser.add_argument("--ct_stage1_freeze_until_iter", type=int, default=CT_STAGE1_FREEZE_UNTIL_ITER)
    parser.add_argument("--ct_containment_ramp_iter_1", type=int, default=CT_CONTAINMENT_RAMP_ITER_1)
    parser.add_argument("--ct_containment_ramp_weight_1", type=float, default=CT_CONTAINMENT_RAMP_WEIGHT_1)
    parser.add_argument("--ct_containment_ramp_iter_2", type=int, default=CT_CONTAINMENT_RAMP_ITER_2)
    parser.add_argument("--ct_containment_ramp_weight_2", type=float, default=CT_CONTAINMENT_RAMP_WEIGHT_2)
    parser.add_argument("--ct_containment_ramp_iter_3", type=int, default=CT_CONTAINMENT_RAMP_ITER_3)
    parser.add_argument("--ct_containment_ramp_weight_3", type=float, default=CT_CONTAINMENT_RAMP_WEIGHT_3)
    parser.add_argument("--ct_false_hole_sample_count", type=int, default=CT_FALSE_HOLE_SAMPLE_COUNT)
    parser.add_argument("--ct_false_hole_boundary_band", type=float, default=CT_FALSE_HOLE_BOUNDARY_BAND)
    parser.add_argument("--ct_false_hole_material_threshold", type=float, default=CT_FALSE_HOLE_MATERIAL_THRESHOLD)
    parser.add_argument("--ct_false_hole_dark_margin", type=float, default=CT_FALSE_HOLE_DARK_MARGIN)
    parser.add_argument("--ct_false_hole_target_occupancy", type=float, default=CT_FALSE_HOLE_TARGET_OCCUPANCY)
    parser.add_argument("--ct_false_hole_metrics_interval", type=int, default=CT_FALSE_HOLE_METRICS_INTERVAL)
    parser.add_argument("--ct_bulk_max_scale", type=float, default=2.0)
    parser.add_argument("--ct_bulk_augment_factor", type=float, default=2.0)
    parser.add_argument("--ct_bulk_continuous_init", action=BooleanOptionalAction, default=CT_BULK_CONTINUOUS_INIT)
    parser.add_argument("--ct_bulk_init_mode", type=str, default="sparse_reseed",
                        choices=["sparse_reseed", "contained_lattice", "conservative_envelope", "feature_adaptive", "fasj"],
                        help="bulk initialization mode")
    parser.add_argument("--ct_bulk_lattice_spacing_vox", type=float, default=2.0)
    parser.add_argument("--ct_bulk_lattice_sigma_vox", type=float, default=2.0)
    parser.add_argument("--ct_bulk_lattice_margin_vox", type=float, default=0.05)
    parser.add_argument("--ct_bulk_lattice_atten_init", type=float, default=0.75)
    parser.add_argument("--ct_bulk_lattice_anisotropic", action=BooleanOptionalAction, default=False)
    parser.add_argument("--ct_bulk_lattice_sigma_t_vox", type=float, default=2.0)
    parser.add_argument("--ct_bulk_lattice_sigma_n_vox", type=float, default=0.8)
    parser.add_argument("--ct_feature_adaptive_jitter", action=BooleanOptionalAction, default=CT_FEATURE_ADAPTIVE_JITTER)
    parser.add_argument("--ct_feature_adaptive_seed", type=int, default=CT_FEATURE_ADAPTIVE_SEED)
    parser.add_argument("--ct_feature_adaptive_r_shell_vox", type=float, default=CT_FEATURE_ADAPTIVE_R_SHELL_VOX)
    parser.add_argument("--ct_feature_adaptive_blur_sigma_vox", type=float, default=CT_FEATURE_ADAPTIVE_BLUR_SIGMA_VOX)
    parser.add_argument("--ct_feature_adaptive_spacing_high_vox", type=int, default=CT_FEATURE_ADAPTIVE_SPACING_HIGH_VOX)
    parser.add_argument("--ct_feature_adaptive_spacing_mid_vox", type=int, default=CT_FEATURE_ADAPTIVE_SPACING_MID_VOX)
    parser.add_argument("--ct_feature_adaptive_spacing_low_vox", type=int, default=CT_FEATURE_ADAPTIVE_SPACING_LOW_VOX)
    parser.add_argument(
        "--ct_feature_adaptive_directional_clearance",
        action=BooleanOptionalAction,
        default=CT_FEATURE_ADAPTIVE_DIRECTIONAL_CLEARANCE,
    )
    parser.add_argument("--ct_feature_adaptive_probe_containment", action=BooleanOptionalAction, default=CT_FEATURE_ADAPTIVE_PROBE_CONTAINMENT)
    parser.add_argument("--ct_bulk_containment_q_support", type=float, default=CT_BULK_CONTAINMENT_Q_SUPPORT)
    parser.add_argument("--ct_bulk_query_truncation_sigma", type=float, default=None)
    parser.add_argument("--ct_init_preflight_abort", action=BooleanOptionalAction, default=CT_INIT_PREFLIGHT_ABORT)
    parser.add_argument("--ct_init_preflight_max_containment_violation", type=float, default=CT_INIT_PREFLIGHT_MAX_CONTAINMENT_VIOLATION)
    parser.add_argument("--ct_init_preflight_min_material_a_b_p10", type=float, default=CT_INIT_PREFLIGHT_MIN_MATERIAL_A_B_P10)
    parser.add_argument("--ct_init_preflight_max_material_coverage_gap", type=float, default=CT_INIT_PREFLIGHT_MAX_MATERIAL_COVERAGE_GAP)
    parser.add_argument("--ct_bulk_coverage_gap_threshold", type=float, default=CT_BULK_COVERAGE_GAP_THRESHOLD)
    parser.add_argument("--ct_freeze_bulk_xyz", action=BooleanOptionalAction, default=CT_FREEZE_BULK_XYZ)
    parser.add_argument("--ct_freeze_surface", action=BooleanOptionalAction, default=CT_FREEZE_SURFACE)
    parser.add_argument("--ct_freeze_bulk_geometry", action=BooleanOptionalAction, default=CT_FREEZE_BULK_GEOMETRY)
    parser.add_argument("--ct_train_bulk_atten_only", action=BooleanOptionalAction, default=CT_TRAIN_BULK_ATTEN_ONLY)
    parser.add_argument("--ct_enable_surface_reseeding", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ct_enable_densification", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ct_densify_from_iter", type=int, default=CT_DENSIFY_FROM_ITER)
    parser.add_argument("--ct_densify_until_iter", type=int, default=CT_DENSIFY_UNTIL_ITER)
    parser.add_argument("--ct_densify_interval", type=int, default=CT_DENSIFY_INTERVAL)
    parser.add_argument("--ct_densify_surface_percent", type=float, default=CT_DENSIFY_SURFACE_PERCENT)
    parser.add_argument("--ct_densify_bulk_percent", type=float, default=CT_DENSIFY_BULK_PERCENT)
    parser.add_argument("--ct_densify_max_gaussian_ratio", type=float, default=CT_DENSIFY_MAX_GAUSSIAN_RATIO)
    parser.add_argument("--ct_densify_min_opacity", type=float, default=CT_DENSIFY_MIN_OPACITY)
    parser.add_argument("--ct_densify_surface_tangent_ratio", type=float, default=CT_DENSIFY_SURFACE_TANGENT_RATIO)
    parser.add_argument("--ct_densify_bulk_scale_ratio", type=float, default=CT_DENSIFY_BULK_SCALE_RATIO)
    parser.add_argument("--ct_enable_bulk_reseeding", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ct_bulk_reseed_from_iter", type=int, default=500)
    parser.add_argument("--ct_bulk_reseed_until_iter", type=int, default=4000)
    parser.add_argument("--ct_bulk_reseed_interval", type=int, default=500)
    parser.add_argument("--ct_bulk_reseed_sample_count", type=int, default=8192)
    parser.add_argument("--ct_bulk_reseed_max_per_iter", type=int, default=0)
    parser.add_argument("--ct_bulk_reseed_max_new_fraction", type=float, default=0.03)
    parser.add_argument("--ct_bulk_reseed_max_gaussian_ratio", type=float, default=2.5)
    parser.add_argument("--ct_bulk_reseed_occupancy_threshold", type=float, default=0.85)
    parser.add_argument("--ct_bulk_reseed_min_sdf_margin", type=float, default=-0.2)
    parser.add_argument("--ct_bulk_reseed_initial_opacity", type=float, default=0.65)
    parser.add_argument("--ct_gap_aware_reseed", action=BooleanOptionalAction, default=CT_GAP_AWARE_RESEED)
    parser.add_argument("--ct_gap_reseed_den_target", type=float, default=CT_GAP_RESEED_DEN_TARGET)
    parser.add_argument("--ct_gap_reseed_sample_ratio", type=float, default=CT_GAP_RESEED_SAMPLE_RATIO)
    parser.add_argument("--ct_gap_reseed_radius_vox", type=float, default=CT_GAP_RESEED_RADIUS_VOX)
    parser.add_argument("--ct_gap_reseed_max_per_iter", type=int, default=CT_GAP_RESEED_MAX_PER_ITER)
    parser.add_argument("--ct_gap_reseed_boundary_subvoxel", action=BooleanOptionalAction, default=CT_GAP_RESEED_BOUNDARY_SUBVOXEL)
    parser.add_argument("--ct_gap_reseed_protect_prune", action=BooleanOptionalAction, default=CT_GAP_RESEED_PROTECT_PRUNE)
    parser.add_argument("--ct_gap_reseed_protect_iters", type=int, default=CT_GAP_RESEED_PROTECT_ITERS)
    parser.add_argument("--ct_gap_bulk_growth_factor", type=float, default=CT_GAP_BULK_GROWTH_FACTOR)
    parser.add_argument("--ct_budgeted_component_repair", action=BooleanOptionalAction, default=CT_BUDGETED_COMPONENT_REPAIR)
    parser.add_argument("--ct_repair_add_threshold", type=float, default=CT_REPAIR_ADD_THRESHOLD)
    parser.add_argument("--ct_repair_stop_threshold", type=float, default=CT_REPAIR_STOP_THRESHOLD)
    parser.add_argument("--ct_repair_min_component_points", type=int, default=CT_REPAIR_MIN_COMPONENT_POINTS)
    parser.add_argument("--ct_repair_max_new_per_pass", type=int, default=CT_REPAIR_MAX_NEW_PER_PASS)
    parser.add_argument("--ct_repair_max_new_fraction", type=float, default=CT_REPAIR_MAX_NEW_FRACTION)
    parser.add_argument("--ct_repair_gain_ratio_min", type=float, default=CT_REPAIR_GAIN_RATIO_MIN)
    parser.add_argument("--ct_repair_exclusion_radius_vox", type=float, default=CT_REPAIR_EXCLUSION_RADIUS_VOX)
    parser.add_argument("--ct_repair_stretch_first", action=BooleanOptionalAction, default=CT_REPAIR_STRETCH_FIRST)
    parser.add_argument("--ct_repair_stretch_growth_factor", type=float, default=CT_REPAIR_STRETCH_GROWTH_FACTOR)
    parser.add_argument("--ct_repair_stretch_secondary_factor", type=float, default=CT_REPAIR_STRETCH_SECONDARY_FACTOR)
    parser.add_argument("--ct_repair_stretch_max_ratio", type=float, default=CT_REPAIR_STRETCH_MAX_RATIO)
    parser.add_argument("--ct_repair_overfill_threshold", type=float, default=CT_REPAIR_OVERFILL_THRESHOLD)
    parser.add_argument("--ct_repair_top_components", type=int, default=CT_REPAIR_TOP_COMPONENTS)
    parser.add_argument("--ct_repair_nearby_candidates", type=int, default=CT_REPAIR_NEARBY_CANDIDATES)
    parser.add_argument("--ct_repair_probe_tangent_factor", type=float, default=CT_REPAIR_PROBE_TANGENT_FACTOR)
    parser.add_argument("--ct_repair_probe_shrink", type=float, default=CT_REPAIR_PROBE_SHRINK)
    parser.add_argument("--ct_repair_max_probe_shrink_iters", type=int, default=CT_REPAIR_MAX_PROBE_SHRINK_ITERS)
    parser.add_argument("--ct_repair_check_stride", type=int, default=CT_REPAIR_CHECK_STRIDE)
    parser.add_argument("--ct_repair_max_check_points", type=int, default=CT_REPAIR_MAX_CHECK_POINTS)
    parser.add_argument("--ct_material_coverage_completion", action=BooleanOptionalAction, default=CT_MATERIAL_COVERAGE_COMPLETION)
    parser.add_argument("--ct_completion_init", action=BooleanOptionalAction, default=CT_COMPLETION_INIT)
    parser.add_argument("--ct_completion_repair", action=BooleanOptionalAction, default=CT_COMPLETION_REPAIR)
    parser.add_argument("--ct_completion_den_target", type=float, default=CT_COMPLETION_DEN_TARGET)
    parser.add_argument("--ct_completion_radius_vox", type=float, default=CT_COMPLETION_RADIUS_VOX)
    parser.add_argument("--ct_completion_max_init_passes", type=int, default=CT_COMPLETION_MAX_INIT_PASSES)
    parser.add_argument("--ct_completion_max_new_per_pass", type=int, default=CT_COMPLETION_MAX_NEW_PER_PASS)
    parser.add_argument("--ct_completion_min_component_voxels", type=int, default=CT_COMPLETION_MIN_COMPONENT_VOXELS)
    parser.add_argument("--ct_completion_check_stride", type=int, default=CT_COMPLETION_CHECK_STRIDE)
    parser.add_argument("--ct_completion_max_check_points", type=int, default=CT_COMPLETION_MAX_CHECK_POINTS)
    parser.add_argument("--ct_sdf_boundary_mode", type=str, default=CT_SDF_BOUNDARY_MODE)
    parser.add_argument("--ct_reseed_probe_length", type=float, default=CT_RESEED_PROBE_LENGTH)
    parser.add_argument("--ct_reseed_band_deficit_thr", type=float, default=CT_RESEED_BAND_DEFICIT_THR)
    parser.add_argument("--ct_reseed_seed_offset_sigma", type=float, default=CT_RESEED_SEED_OFFSET_SIGMA)
    parser.add_argument("--ct_reseed_max_new_fraction", type=float, default=CT_RESEED_MAX_NEW_FRACTION)
    parser.add_argument("--ct_reseed_sigma_init_factor", type=float, default=CT_RESEED_SIGMA_INIT_FACTOR)
    parser.add_argument("--ct_reseed_sigma_init_floor_ratio", type=float, default=CT_RESEED_SIGMA_INIT_FLOOR_RATIO)
    parser.add_argument("--ct_reseed_atten_init_boost", type=float, default=CT_RESEED_ATTEN_INIT_BOOST)
    parser.add_argument("--ct_bulk_prune_interval", type=int, default=CT_BULK_PRUNE_INTERVAL)
    parser.add_argument("--ct_bulk_prune_warmup", type=int, default=CT_BULK_PRUNE_WARMUP)
    parser.add_argument("--ct_bulk_prune_min_opacity", type=float, default=CT_BULK_PRUNE_MIN_OPACITY)
    parser.add_argument("--ct_bulk_prune_air_sdf_epsilon", type=float, default=CT_BULK_PRUNE_AIR_SDF_EPSILON)
    parser.add_argument("--ct_bulk_prune_raw_air_threshold", type=float, default=CT_BULK_PRUNE_RAW_AIR_THRESHOLD)
    parser.add_argument("--ct_bulk_prune_sample_count", type=int, default=CT_BULK_PRUNE_SAMPLE_COUNT)
    parser.add_argument("--ct_bulk_prune_neighbor_radius", type=float, default=CT_BULK_PRUNE_NEIGHBOR_RADIUS)
    parser.add_argument("--ct_bulk_prune_min_neighbors", type=int, default=CT_BULK_PRUNE_MIN_NEIGHBORS)
    parser.add_argument("--ct_intensity_phase2_start_ratio", type=float, default=CT_INTENSITY_PHASE2_START_RATIO)
    parser.add_argument("--ct_intensity_phase3_start_ratio", type=float, default=CT_INTENSITY_PHASE3_START_RATIO)
    parser.add_argument("--ct_intensity_train_bulk_scale", action=BooleanOptionalAction, default=CT_INTENSITY_TRAIN_BULK_SCALE)
    parser.add_argument("--ct_intensity_train_surface_opacity", action=BooleanOptionalAction, default=CT_INTENSITY_TRAIN_SURFACE_OPACITY)
    parser.add_argument("--ct_intensity_bulk_scale_grad", type=float, default=CT_INTENSITY_BULK_SCALE_GRAD)
    parser.add_argument("--ct_intensity_sample_mode", type=str, default=CT_INTENSITY_SAMPLE_MODE, choices=["full_band", "material_interior_only"])
    parser.add_argument("--ct_intensity_erode_margin_vox", type=float, default=CT_INTENSITY_ERODE_MARGIN_VOX)
    parser.add_argument("--ct_intensity_den_min", type=float, default=CT_INTENSITY_DEN_MIN)
    parser.add_argument("--ct_atten_only_train_bulk_opacity", action=BooleanOptionalAction, default=CT_ATTEN_ONLY_TRAIN_BULK_OPACITY)
    parser.add_argument("--ct_bulk_opacity_sparse_weight", type=float, default=CT_BULK_OPACITY_SPARSE_WEIGHT)
    parser.add_argument("--ct_atten_only_lr_final_scale", type=float, default=CT_ATTEN_ONLY_LR_FINAL_SCALE)
    parser.add_argument("--ct_atten_only_early_stop", action=BooleanOptionalAction, default=CT_ATTEN_ONLY_EARLY_STOP)
    parser.add_argument("--ct_atten_only_early_stop_warmup_iters", type=int, default=CT_ATTEN_ONLY_EARLY_STOP_WARMUP_ITERS)
    parser.add_argument("--ct_atten_only_early_stop_eval_interval", type=int, default=CT_ATTEN_ONLY_EARLY_STOP_EVAL_INTERVAL)
    parser.add_argument("--ct_atten_only_early_stop_patience", type=int, default=CT_ATTEN_ONLY_EARLY_STOP_PATIENCE)
    parser.add_argument("--ct_atten_only_early_stop_min_delta", type=float, default=CT_ATTEN_ONLY_EARLY_STOP_MIN_DELTA)
    parser.add_argument("--ct_eagle_loss_weight", type=float, default=CT_EAGLE_LOSS_WEIGHT)
    parser.add_argument("--ct_eagle_patch_size", type=int, default=CT_EAGLE_PATCH_SIZE)
    parser.add_argument("--ct_eagle_block_size", type=int, default=CT_EAGLE_BLOCK_SIZE)
    parser.add_argument("--ct_auto_preview", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ct_preview_first_iter", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ct_preview_interval", type=int, default=CT_PREVIEW_INTERVAL)
    parser.add_argument("--output_gs", type=str, default=None)
    parser.add_argument("--output_mesh", type=str, default=None)
    parser.add_argument("--output_sdf", type=str, default=None)
    parser.add_argument("--export_mesh_resolution", type=float, default=0.05)
    parser.add_argument("--export_sdf_resolution", type=int, default=256)
    parser.add_argument("--skip_export_mesh", action="store_true", default=False)
    parser.add_argument("--skip_export_sdf", action="store_true", default=False)
    return parser
