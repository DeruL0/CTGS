import os
import random
import sys
import time
import uuid
import warnings
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from tqdm import tqdm

from ct_pipeline.ct_args import (
    add_ct_model_args,
    add_ct_optimization_args,
    extract_ct_model_args,
    extract_ct_optimization_args,
)
from ct_pipeline.ct_exporter import CTExporter
from ct_pipeline.field_query import density_to_occupancy, query_ct_density_backend
from ct_pipeline.ct_loader import CTVolumeLoader
from ct_pipeline.ct_preprocessor import CTPreprocessor
from ct_pipeline.native_backend import (
    build_neighbor_index_backend,
    build_ct_backend_patch_renderer,
    point_to_plane_loss_backend,
    prepare_point_to_plane_cache_backend,
    resolve_ct_backend,
)
from ct_pipeline.ct_slice_renderer import (
    CTPatchGridCache,
    prepare_ct_render_state,
    sample_gt_slice_patch,
)
from scene import CTGaussianModel
from utils.ct_losses import (
    material_boundary_loss,
    normal_alignment_loss,
    occupancy_loss,
    thickness_penalty,
    volume_rendering_loss,
)
from utils.general_utils import safe_state

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    import nvidia_smi
except ImportError:  # pragma: no cover
    try:
        import pynvml as nvidia_smi
    except ImportError:  # pragma: no cover
        nvidia_smi = None

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:  # pragma: no cover
    TENSORBOARD_FOUND = False


_NVML_INITIALIZED = False


def initialize_nvml():
    global _NVML_INITIALIZED
    if nvidia_smi is None:
        return False
    if _NVML_INITIALIZED:
        return True
    try:
        nvidia_smi.nvmlInit()
    except Exception:
        return False
    _NVML_INITIALIZED = True
    return True


def shutdown_nvml():
    global _NVML_INITIALIZED
    if not _NVML_INITIALIZED:
        return
    try:
        nvidia_smi.nvmlShutdown()
    except Exception:
        pass
    _NVML_INITIALIZED = False


def save_command(path):
    command = " ".join(sys.argv)
    with open(os.path.join(path, "command.txt"), "a", encoding="utf-8") as handle:
        handle.write(command + "\n")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: not logging progress")
    return None


def validate_ct_training_args(args):
    _normalize_ct_query_args(args)
    if not args.ct_phase1_dir:
        raise ValueError("--ct_phase1_dir is required for train_ct.py.")
    if not args.ct_volume_path:
        raise ValueError("--ct_volume_path is required for train_ct.py.")
    if args.ct_volume_format == "raw" and not args.ct_raw_meta:
        raise ValueError("--ct_raw_meta is required for RAW CT volumes.")
    if args.load_ply:
        raise ValueError("--load_ply is not supported in CT training mode. Use --start_checkpoint to resume.")
    if args.ct_patch_size < 1:
        raise ValueError("--ct_patch_size must be >= 1.")
    if args.ct_slice_batch_size < 1:
        raise ValueError("--ct_slice_batch_size must be >= 1.")
    if args.ct_render_chunk_gaussians < 1:
        raise ValueError("--ct_render_chunk_gaussians must be >= 1.")
    if args.ct_bulk_points_ratio <= 0.0:
        raise ValueError("--ct_bulk_points_ratio must be > 0.")
    if args.ct_bulk_boundary_margin_voxels < 0:
        raise ValueError("--ct_bulk_boundary_margin_voxels must be >= 0.")
    if args.ct_material_query_count < 1 or args.ct_void_query_count < 1 or args.ct_exterior_query_count < 1:
        raise ValueError("--ct_material_query_count, --ct_void_query_count, and --ct_exterior_query_count must be >= 1.")
    if args.ct_max_material_classes < 1:
        raise ValueError("--ct_max_material_classes must be >= 1.")
    if args.ct_void_negative_weight <= 0.0:
        raise ValueError("--ct_void_negative_weight must be > 0.")
    if args.ct_density_query_tile_points < 1:
        raise ValueError("--ct_density_query_tile_points must be >= 1.")
    if args.ct_knn_tile_size < 1:
        raise ValueError("--ct_knn_tile_size must be >= 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CT training requires CUDA in the current implementation.")
    if args.ct_backend not in {"auto", "cuda", "python"}:
        raise ValueError("--ct_backend must be one of {'auto', 'cuda', 'python'}.")


def _normalize_ct_query_args(args):
    legacy_interior = getattr(args, "ct_interior_query_count", None)
    material_count = getattr(args, "ct_material_query_count", None)
    if material_count is None:
        material_count = legacy_interior if legacy_interior is not None else 4096
    args.ct_material_query_count = int(material_count)
    args.ct_interior_query_count = int(args.ct_material_query_count)
    if getattr(args, "ct_void_query_count", None) is None:
        args.ct_void_query_count = 4096
    else:
        args.ct_void_query_count = int(args.ct_void_query_count)
    if getattr(args, "ct_max_material_classes", None) is None:
        args.ct_max_material_classes = 3
    else:
        args.ct_max_material_classes = int(args.ct_max_material_classes)
    if getattr(args, "ct_void_negative_weight", None) is None:
        args.ct_void_negative_weight = 2.0
    else:
        args.ct_void_negative_weight = float(args.ct_void_negative_weight)


def _ct_spatial_extent(volume_shape_dhw, spacing_zyx):
    extents = [float(dim) * float(spacing) for dim, spacing in zip(volume_shape_dhw, spacing_zyx)]
    return max(extents)


def _freeze_ct_feature_params(gaussians):
    for param in (gaussians._features_dc, gaussians._features_rest):
        param.requires_grad_(False)
    if gaussians.optimizer is None:
        return
    for group in gaussians.optimizer.param_groups:
        if group["name"] in {"f_dc", "f_rest"}:
            group["lr"] = 0.0
            for parameter in group["params"]:
                parameter.requires_grad_(False)


def _save_ct_gaussians(gaussians, model_path, iteration):
    point_cloud_path = os.path.join(model_path, "point_cloud", "iteration_{0}".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


def _load_ct_analysis_bundle(ct_phase1_dir):
    bundle_dir = Path(ct_phase1_dir)
    analysis_path = bundle_dir / "analysis.npz"
    metadata_path = bundle_dir / "metadata.json"
    if not analysis_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("CT Phase 1 bundle must contain analysis.npz and metadata.json.")
    with np.load(str(analysis_path)) as analysis_npz:
        analysis = {key: analysis_npz[key] for key in analysis_npz.files}
    return analysis, str(analysis_path), str(metadata_path)


def _default_roi_bbox(volume_shape_dhw):
    depth, height, width = [int(value) for value in volume_shape_dhw]
    return np.asarray([[0, depth], [0, height], [0, width]], dtype=np.int32)


def _foreground_slice_bbox(foreground_mask, axis_index, slice_idx):
    if foreground_mask is None:
        return None
    if axis_index == 0:
        mask_2d = foreground_mask[int(slice_idx)]
    elif axis_index == 1:
        mask_2d = foreground_mask[:, int(slice_idx), :]
    else:
        mask_2d = foreground_mask[:, :, int(slice_idx)]
    if mask_2d.size == 0 or not np.any(mask_2d):
        return None
    coords = np.argwhere(mask_2d)
    lower = coords.min(axis=0)
    upper = coords.max(axis=0) + 1
    return lower, upper


def _choose_patch_origin(lower, upper, full_size, patch_size):
    patch_size = min(int(patch_size), int(full_size))
    lower = int(lower)
    upper = int(upper)
    min_origin = max(0, upper - patch_size)
    max_origin = min(lower, full_size - patch_size)
    if max_origin >= min_origin:
        return int(np.random.randint(min_origin, max_origin + 1))
    centered = int(round((lower + upper - patch_size) * 0.5))
    return int(np.clip(centered, 0, full_size - patch_size))


def _sample_ct_patch_spec(analysis, volume_shape_dhw, requested_patch_size):
    axis_index = int(np.random.randint(0, 3))
    axis_name = ["z", "y", "x"][axis_index]
    roi_bbox = analysis["roi_bbox"] if "roi_bbox" in analysis else _default_roi_bbox(volume_shape_dhw)
    foreground_mask = analysis["material_mask"] if "material_mask" in analysis else analysis["foreground_mask"] if "foreground_mask" in analysis else None

    slice_lower, slice_upper = [int(value) for value in roi_bbox[axis_index]]
    slice_upper = max(slice_lower + 1, min(slice_upper, int(volume_shape_dhw[axis_index])))
    slice_idx = int(np.random.randint(slice_lower, slice_upper))

    if axis_index == 0:
        full_height, full_width = int(volume_shape_dhw[1]), int(volume_shape_dhw[2])
    elif axis_index == 1:
        full_height, full_width = int(volume_shape_dhw[0]), int(volume_shape_dhw[2])
    else:
        full_height, full_width = int(volume_shape_dhw[0]), int(volume_shape_dhw[1])

    patch_height = min(int(requested_patch_size), full_height)
    patch_width = min(int(requested_patch_size), full_width)

    foreground_bbox = _foreground_slice_bbox(foreground_mask, axis_index, slice_idx)
    if foreground_bbox is None:
        origin_h = int(np.random.randint(0, max(1, full_height - patch_height + 1)))
        origin_w = int(np.random.randint(0, max(1, full_width - patch_width + 1)))
    else:
        lower, upper = foreground_bbox
        origin_h = _choose_patch_origin(lower[0], upper[0], full_height, patch_height)
        origin_w = _choose_patch_origin(lower[1], upper[1], full_width, patch_width)

    return axis_name, slice_idx, (origin_h, origin_w), (patch_height, patch_width)


def _build_neighbor_index_python(xyz, k):
    if xyz.shape[0] <= 1 or k <= 0:
        return torch.empty((xyz.shape[0], 0), dtype=torch.long, device=xyz.device)
    k = min(int(k), xyz.shape[0] - 1)
    xyz_np = xyz.detach().float().cpu().numpy()
    finite_mask = np.isfinite(xyz_np).all(axis=1)
    if not np.all(finite_mask):
        if np.any(finite_mask):
            replacement = xyz_np[finite_mask].mean(axis=0)
        else:
            replacement = np.zeros((3,), dtype=np.float32)
        xyz_np = xyz_np.copy()
        xyz_np[~finite_mask] = replacement
    tree = cKDTree(xyz_np)
    _, neighbor_index = tree.query(xyz_np, k=k + 1)
    neighbor_index = np.asarray(neighbor_index)
    if neighbor_index.ndim == 1:
        neighbor_index = neighbor_index[:, np.newaxis]
    neighbor_index = neighbor_index[:, 1:]
    return torch.as_tensor(neighbor_index, dtype=torch.long, device=xyz.device)


def _build_neighbor_index(xyz, k, backend: str, tile_size: int = 2048):
    if xyz.shape[0] <= 1 or k <= 0:
        return torch.empty((xyz.shape[0], 0), dtype=torch.long, device=xyz.device)
    if backend == "cuda":
        return build_neighbor_index_backend(backend, xyz, k=int(k), tile_size=int(tile_size))
    return _build_neighbor_index_python(xyz, k)


def _ensure_void_aware_analysis(analysis, volume_np, max_material_classes):
    if "material_mask" in analysis and "void_mask" in analysis and "material_label_volume" in analysis:
        return analysis

    preprocessor = CTPreprocessor()
    segmentation = preprocessor.segment_material_void(
        volume_np,
        method="multi_otsu",
        max_material_classes=max_material_classes,
    )
    upgraded = dict(analysis)
    upgraded["material_mask"] = segmentation["material_mask"]
    upgraded["void_mask"] = segmentation["void_mask"]
    upgraded["foreground_mask"] = segmentation["foreground_mask"]
    upgraded["material_label_volume"] = segmentation["material_label_volume"]
    upgraded["roi_bbox"] = segmentation["roi_bbox"]
    warnings.warn(
        "Phase 1 bundle is legacy foreground-only data; material_mask/void_mask/material_label_volume were recomputed at train-time.",
        RuntimeWarning,
        stacklevel=2,
    )
    return upgraded


def _prepare_occupancy_voxel_pools(analysis, volume_shape_dhw, boundary_margin_voxels):
    foreground_mask = np.asarray(analysis["foreground_mask"], dtype=bool)
    roi_bbox = np.asarray(analysis["roi_bbox"] if "roi_bbox" in analysis else _default_roi_bbox(volume_shape_dhw), dtype=np.int32)
    if "material_mask" in analysis:
        material_mask = np.asarray(analysis["material_mask"], dtype=bool)
        void_mask = np.asarray(analysis["void_mask"], dtype=bool) if "void_mask" in analysis else np.logical_and(
            foreground_mask,
            np.logical_not(material_mask),
        )
    else:
        warnings.warn(
            "Occupancy pools are falling back to legacy foreground-only sampling. Re-run Phase 1 for void-aware training.",
            RuntimeWarning,
            stacklevel=2,
        )
        material_mask = foreground_mask
        void_mask = np.zeros_like(foreground_mask, dtype=bool)

    distance = ndimage.distance_transform_edt(material_mask)
    material_interior = np.logical_and(material_mask, distance >= float(max(int(boundary_margin_voxels), 0)))
    if not np.any(material_interior):
        material_interior = material_mask

    padding = max(4, int(boundary_margin_voxels) * 2)
    lower = np.maximum(roi_bbox[:, 0] - padding, 0)
    upper = np.minimum(roi_bbox[:, 1] + padding, np.asarray(volume_shape_dhw, dtype=np.int32))
    exterior_window = np.zeros_like(foreground_mask, dtype=bool)
    exterior_window[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]] = True
    exterior_mask = np.logical_and(exterior_window, np.logical_not(foreground_mask))
    if not np.any(exterior_mask):
        exterior_mask = np.logical_not(foreground_mask)

    return {
        "material": np.argwhere(material_interior).astype(np.int32),
        "void": np.argwhere(void_mask).astype(np.int32),
        "exterior": np.argwhere(exterior_mask).astype(np.int32),
    }


def _sample_occupancy_points(candidate_indices, sample_count, spacing_zyx, device):
    if candidate_indices.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    count = int(sample_count)
    if count <= candidate_indices.shape[0]:
        selected = np.asarray(random.sample(range(candidate_indices.shape[0]), count), dtype=np.int64)
    else:
        selected = np.random.choice(candidate_indices.shape[0], size=count, replace=True)
    voxel_indices = candidate_indices[selected]
    points_xyz = np.stack(
        (
            (voxel_indices[:, 2].astype(np.float32) + 0.5) * float(spacing_zyx[2]),
            (voxel_indices[:, 1].astype(np.float32) + 0.5) * float(spacing_zyx[1]),
            (voxel_indices[:, 0].astype(np.float32) + 0.5) * float(spacing_zyx[0]),
        ),
        axis=1,
    )
    return torch.as_tensor(points_xyz, dtype=torch.float32, device=device)


def _sanitize_xyz_parameter(gaussians):
    xyz = gaussians.get_xyz
    if xyz.numel() == 0:
        return 0

    finite_mask = torch.isfinite(xyz).all(dim=1)
    if torch.all(finite_mask):
        return 0

    bad_count = int((~finite_mask).sum().item())
    with torch.no_grad():
        if torch.any(finite_mask):
            replacement = xyz[finite_mask].mean(dim=0, keepdim=True)
        else:
            replacement = torch.zeros((1, 3), dtype=xyz.dtype, device=xyz.device)
        xyz[~finite_mask] = replacement
    return bad_count


def _compute_ct_normal_loss(gaussians, neighbor_index):
    if neighbor_index.numel() == 0:
        return torch.zeros((), dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)

    normals = gaussians.get_normals()
    planarity = gaussians.get_planarity.reshape(-1)
    planar_mask = gaussians.get_is_planar.reshape(-1)
    material_id = gaussians.get_material_id.reshape(-1)

    clamped_neighbors = neighbor_index.clamp_min(0)
    neighbor_normals = normals[clamped_neighbors]
    neighbor_planarity = planarity[clamped_neighbors]
    pair_weights = torch.sqrt(planarity.unsqueeze(1).clamp_min(0.0) * neighbor_planarity.clamp_min(0.0))
    pair_weights = pair_weights * planar_mask.unsqueeze(1).float() * planar_mask[clamped_neighbors].float()
    if material_id.numel() > 0:
        same_material = material_id.unsqueeze(1) == material_id[clamped_neighbors]
        pair_weights = pair_weights * same_material.float()
    return normal_alignment_loss(normals, neighbor_normals, pair_weights)


def _ct_log_gpu_memory():
    if not initialize_nvml():
        return None
    try:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    except Exception:
        return None
    return info.used / (1024 ** 2 * 1000)


def _build_renderer_autocast_kwargs() -> dict:
    return {
        "device_type": "cuda",
        "dtype": torch.bfloat16,
        "enabled": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    }


def _export_ct_outputs(gaussians, args):
    exporter = CTExporter()
    outputs = {}
    if args.output_gs:
        outputs["output_gs"] = str(exporter.export_display_gs(gaussians, args.output_gs, compress=True))
    if not args.skip_export_mesh and args.output_mesh:
        outputs["output_mesh"] = str(exporter.export_metrology_mesh(gaussians, args.output_mesh, resolution=args.export_mesh_resolution))
    if not args.skip_export_sdf and args.output_sdf:
        sdf_path, sdf_meta = exporter.export_sdf(gaussians, args.output_sdf, grid_resolution=args.export_sdf_resolution)
        outputs["output_sdf"] = str(sdf_path)
        outputs["output_sdf_metadata"] = str(sdf_meta)
    return outputs


def training_ct(dataset, opt, saving_iterations, checkpoint_iterations, checkpoint, args):
    validate_ct_training_args(args)

    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    loader = CTVolumeLoader()
    volume_np = loader.load(args.ct_volume_path, fmt=args.ct_volume_format, raw_meta_path=args.ct_raw_meta)
    spacing_zyx = loader.get_voxel_spacing()
    analysis, analysis_path, metadata_path = _load_ct_analysis_bundle(args.ct_phase1_dir)
    analysis = _ensure_void_aware_analysis(analysis, volume_np, args.ct_max_material_classes)

    gaussians = CTGaussianModel(dataset.sh_degree)
    gaussians.create_from_phase1_bundle(
        analysis_path,
        metadata_path,
        spatial_lr_scale=_ct_spatial_extent(volume_np.shape, spacing_zyx),
        planar_thickness_max=args.planar_thickness_max,
        volume=volume_np,
        bulk_points_ratio=args.ct_bulk_points_ratio,
        bulk_boundary_margin_voxels=args.ct_bulk_boundary_margin_voxels,
    )

    gaussians.spatial_lr_scale = _ct_spatial_extent(volume_np.shape, spacing_zyx)
    opt.primitive_harden_iter = args.primitive_harden_iter
    opt.planar_thickness_max = args.planar_thickness_max
    gaussians.training_setup(opt)
    _freeze_ct_feature_params(gaussians)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
        _freeze_ct_feature_params(gaussians)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="CT training progress")
    refresh_interval = max(1, int(args.ct_neighbor_refresh_interval))
    backend = resolve_ct_backend(args.ct_backend)
    neighbor_index = None
    plane_cache = None
    total_computing_time = 0.0
    patch_grid_cache = CTPatchGridCache()
    ct_patch_renderer = build_ct_backend_patch_renderer(backend, compile_renderer=bool(args.ct_compile_renderer))
    renderer_autocast_kwargs = _build_renderer_autocast_kwargs()
    occupancy_pools = _prepare_occupancy_voxel_pools(analysis, volume_np.shape, args.ct_bulk_boundary_margin_voxels)

    for iteration in range(first_iter + 1, opt.iterations + 1):
        tic = time.time()
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        bad_xyz_count = _sanitize_xyz_parameter(gaussians)
        if bad_xyz_count > 0:
            print(f"[WARN] Replaced {bad_xyz_count} non-finite Gaussian centers before neighbor refresh.")

        if neighbor_index is None or iteration == first_iter + 1 or iteration % refresh_interval == 0:
            neighbor_index = _build_neighbor_index(
                gaussians.get_xyz,
                args.ct_neighbor_k,
                backend=backend,
                tile_size=args.ct_knn_tile_size,
            )
            if args.ct_lambda_plane != 0.0:
                plane_cache = prepare_point_to_plane_cache_backend(
                    backend,
                    gaussians.get_xyz,
                    gaussians.get_normals(),
                    gaussians.get_planarity,
                    material_ids=gaussians.get_material_id,
                    primitive_type=gaussians.get_primitive_type_prob,
                    neighbor_index=neighbor_index,
                    k=args.ct_neighbor_k,
                )

        slice_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        render_state = prepare_ct_render_state(gaussians)
        for _ in range(int(args.ct_slice_batch_size)):
            axis, slice_idx, patch_origin, patch_size = _sample_ct_patch_spec(analysis, volume_np.shape, args.ct_patch_size)
            gt_patch = torch.as_tensor(
                sample_gt_slice_patch(volume_np, axis, slice_idx, patch_origin, patch_size),
                dtype=torch.float32,
                device="cuda",
            )
            with torch.autocast(**renderer_autocast_kwargs):
                rendered_patch = ct_patch_renderer(
                    render_state,
                    axis,
                    slice_idx,
                    patch_origin,
                    patch_size,
                    spacing_zyx,
                    volume_np.shape,
                    gaussians_per_chunk=args.ct_render_chunk_gaussians,
                    patch_grid_cache=patch_grid_cache,
                )
                slice_loss = slice_loss + volume_rendering_loss(rendered_patch, gt_patch)
        slice_loss = slice_loss / float(args.ct_slice_batch_size)

        zero_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        occupancy_term = zero_loss
        if args.ct_lambda_occupancy != 0.0:
            material_points = _sample_occupancy_points(
                occupancy_pools["material"],
                args.ct_material_query_count,
                spacing_zyx,
                device="cuda",
            )
            void_count = int(args.ct_void_query_count)
            exterior_count = int(args.ct_exterior_query_count)
            if occupancy_pools["void"].shape[0] == 0:
                exterior_count += void_count
                void_count = 0
            elif occupancy_pools["exterior"].shape[0] == 0:
                void_count += exterior_count
                exterior_count = 0
            void_points = _sample_occupancy_points(
                occupancy_pools["void"],
                void_count,
                spacing_zyx,
                device="cuda",
            )
            exterior_points = _sample_occupancy_points(
                occupancy_pools["exterior"],
                exterior_count,
                spacing_zyx,
                device="cuda",
            )
            occupancy_points = torch.cat((material_points, void_points, exterior_points), dim=0)
            occupancy_target = torch.cat(
                (
                    torch.ones((material_points.shape[0],), dtype=torch.float32, device="cuda"),
                    torch.zeros((void_points.shape[0],), dtype=torch.float32, device="cuda"),
                    torch.zeros((exterior_points.shape[0],), dtype=torch.float32, device="cuda"),
                ),
                dim=0,
            )
            occupancy_weights = torch.cat(
                (
                    torch.ones((material_points.shape[0],), dtype=torch.float32, device="cuda"),
                    torch.full((void_points.shape[0],), float(args.ct_void_negative_weight), dtype=torch.float32, device="cuda"),
                    torch.ones((exterior_points.shape[0],), dtype=torch.float32, device="cuda"),
                ),
                dim=0,
            )
            occupancy_pred = density_to_occupancy(
                query_ct_density_backend(
                    backend,
                    gaussians,
                    occupancy_points,
                    chunk_size=args.ct_density_query_tile_points,
                )
            )
            occupancy_term = occupancy_loss(occupancy_pred, occupancy_target, sample_weights=occupancy_weights)
        plane_loss = zero_loss
        if args.ct_lambda_plane != 0.0 and plane_cache is not None:
            plane_loss = point_to_plane_loss_backend(backend, gaussians.get_xyz, plane_cache)
        normal_loss = zero_loss
        if args.ct_lambda_normal != 0.0:
            normal_loss = _compute_ct_normal_loss(gaussians, neighbor_index)
        thickness_loss = zero_loss
        if args.ct_lambda_thickness != 0.0:
            thickness_loss = thickness_penalty(
                gaussians.get_raw_scaling,
                gaussians.get_rotation,
                gaussians.get_normals(),
                gaussians.get_primitive_type_prob,
                gaussians.planar_thickness_max if gaussians.planar_thickness_max is not None else 1.0,
            )
        material_loss = zero_loss
        if args.ct_lambda_material != 0.0:
            material_loss = material_boundary_loss(
                gaussians.get_xyz,
                gaussians.get_material_id,
                gaussians.get_opacity,
                neighbor_index=neighbor_index,
            )

        loss = (
            args.ct_lambda_slice * slice_loss
            + args.ct_lambda_occupancy * occupancy_term
            + args.ct_lambda_plane * plane_loss
            + args.ct_lambda_normal * normal_loss
            + args.ct_lambda_thickness * thickness_loss
            + args.ct_lambda_material * material_loss
        )
        loss.backward()

        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            elapsed = iter_start.elapsed_time(iter_end)
            total_computing_time += time.time() - tic
            if tb_writer:
                tb_writer.add_scalar("ct_loss/slice", slice_loss.item(), iteration)
                tb_writer.add_scalar("ct_loss/occupancy", occupancy_term.item(), iteration)
                tb_writer.add_scalar("ct_loss/plane", plane_loss.item(), iteration)
                tb_writer.add_scalar("ct_loss/normal", normal_loss.item(), iteration)
                tb_writer.add_scalar("ct_loss/thickness", thickness_loss.item(), iteration)
                tb_writer.add_scalar("ct_loss/material", material_loss.item(), iteration)
                tb_writer.add_scalar("ct_loss/total", loss.item(), iteration)
                tb_writer.add_scalar("iter_time", elapsed, iteration)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                _save_ct_gaussians(gaussians, dataset.model_path, iteration)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.post_optimizer_step(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), dataset.model_path + "/chkpnt" + str(iteration) + ".pth")

        if args.wandb and wandb is not None:
            wandb_logs = {
                "loss": loss.item(),
                "slice_loss": slice_loss.item(),
                "occupancy_loss": occupancy_term.item(),
                "plane_loss": plane_loss.item(),
                "normal_loss": normal_loss.item(),
                "thickness_loss": thickness_loss.item(),
                "material_loss": material_loss.item(),
                "t": total_computing_time,
                "num_gaussian": len(gaussians.get_xyz),
            }
            gpu_memory = _ct_log_gpu_memory()
            if gpu_memory is not None:
                wandb_logs["gpu"] = gpu_memory
            wandb.log(wandb_logs, commit=True)

    exports = _export_ct_outputs(gaussians, args)
    return {"branch": "ct", "backend": backend, "iterations": opt.iterations, **exports}


def build_parser():
    parser = ArgumentParser(description="CT training script parameters")
    add_ct_model_args(parser)
    add_ct_optimization_args(parser)
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
    parser.add_argument("--ct_patch_size", type=int, default=128)
    parser.add_argument("--ct_slice_batch_size", type=int, default=1)
    parser.add_argument("--ct_neighbor_k", type=int, default=8)
    parser.add_argument("--ct_neighbor_refresh_interval", type=int, default=20)
    parser.add_argument("--ct_backend", type=str, default="auto", choices=["auto", "cuda", "python"])
    parser.add_argument("--ct_render_chunk_gaussians", type=int, default=2048)
    parser.add_argument("--ct_compile_renderer", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ct_bulk_points_ratio", type=float, default=1.0)
    parser.add_argument("--ct_bulk_boundary_margin_voxels", type=int, default=2)
    parser.add_argument("--ct_material_query_count", type=int, default=None)
    parser.add_argument("--ct_void_query_count", type=int, default=4096)
    parser.add_argument("--ct_interior_query_count", type=int, default=4096)
    parser.add_argument("--ct_exterior_query_count", type=int, default=4096)
    parser.add_argument("--ct_max_material_classes", type=int, default=3)
    parser.add_argument("--ct_void_negative_weight", type=float, default=2.0)
    parser.add_argument("--ct_density_query_tile_points", type=int, default=8192)
    parser.add_argument("--ct_knn_tile_size", type=int, default=2048)
    parser.add_argument("--ct_lambda_slice", type=float, default=1.0)
    parser.add_argument("--ct_lambda_occupancy", type=float, default=1.0)
    parser.add_argument("--ct_lambda_plane", type=float, default=0.2)
    parser.add_argument("--ct_lambda_normal", type=float, default=0.1)
    parser.add_argument("--ct_lambda_thickness", type=float, default=0.05)
    parser.add_argument("--ct_lambda_material", type=float, default=0.1)
    parser.add_argument("--output_gs", type=str, default=None)
    parser.add_argument("--output_mesh", type=str, default=None)
    parser.add_argument("--output_sdf", type=str, default=None)
    parser.add_argument("--export_mesh_resolution", type=float, default=0.05)
    parser.add_argument("--export_sdf_resolution", type=int, default=256)
    parser.add_argument("--skip_export_mesh", action="store_true", default=False)
    parser.add_argument("--skip_export_sdf", action="store_true", default=False)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[:10])

    print("Optimizing " + args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    save_command(args.model_path)
    safe_state(args.quiet)

    if args.wandb and wandb is not None:
        wandb.login()
        wandb.init(
            project="gaussian_splatting",
            name=os.path.basename(args.model_path.rstrip("/\\")),
            config={},
            save_code=True,
            notes="",
            mode="online",
        )

    initialize_nvml()
    torch.autograd.set_detect_anomaly(False)
    result = training_ct(
        extract_ct_model_args(args),
        extract_ct_optimization_args(args),
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args,
    )
    shutdown_nvml()
    print("\nTraining complete.")
    return result


if __name__ == "__main__":
    main()
