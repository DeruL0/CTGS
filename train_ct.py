import os
import random
import sys
import time
import uuid
import warnings
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import json
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
from ct_pipeline.field_query import density_to_occupancy, query_ct_density_from_state
from ct_pipeline.ct_loader import CTVolumeLoader
from ct_pipeline.ct_preprocessor import CTPreprocessor
from ct_pipeline.geometry_analyzer import GeometryAnalyzer
from ct_pipeline.native_backend import (
    build_neighbor_index_backend,
    build_ct_backend_patch_renderer,
    prepare_ct_training_state,
    resolve_ct_backend,
)
from ct_pipeline.ct_slice_renderer import (
    CTPatchGridCache,
    sample_gt_slice_patch,
)
from scene import CTGaussianModel
from utils.ct_losses import (
    bulk_regularization_loss,
    material_boundary_compact_loss,
    occupancy_loss,
    sample_volume_field,
    surface_shape_loss,
    volume_rendering_loss,
)
from utils.general_utils import safe_state
from utils.loss_utils import ssim

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
_LPIPS_MODEL_CACHE = {}


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


def _get_lpips_model(device):
    device = torch.device(device)
    cache_key = str(device)
    if cache_key in _LPIPS_MODEL_CACHE:
        return _LPIPS_MODEL_CACHE[cache_key]
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"LPIPS metric disabled during CT preview generation: {exc!r}", RuntimeWarning, stacklevel=2)
        model = None
    _LPIPS_MODEL_CACHE[cache_key] = model
    return model


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
    if args.ct_support_sample_count < 1 or args.ct_air_sample_count < 1:
        raise ValueError("--ct_support_sample_count and --ct_air_sample_count must be >= 1.")
    if args.ct_max_material_classes < 1:
        raise ValueError("--ct_max_material_classes must be >= 1.")
    if args.ct_void_negative_weight <= 0.0:
        raise ValueError("--ct_void_negative_weight must be > 0.")
    if args.ct_cavity_patch_bias < 0.0 or args.ct_cavity_patch_bias > 1.0:
        raise ValueError("--ct_cavity_patch_bias must be in [0, 1].")
    if args.ct_void_boundary_offset_scale <= 0.0:
        raise ValueError("--ct_void_boundary_offset_scale must be > 0.")
    if args.ct_void_boundary_margin < 0.0:
        raise ValueError("--ct_void_boundary_margin must be >= 0.")
    if args.ct_signed_field_band_voxels < 1:
        raise ValueError("--ct_signed_field_band_voxels must be >= 1.")
    if args.ct_density_query_tile_points < 1:
        raise ValueError("--ct_density_query_tile_points must be >= 1.")
    if args.ct_knn_tile_size < 1:
        raise ValueError("--ct_knn_tile_size must be >= 1.")
    if args.ct_gaussian_truncation_sigma <= 0.0:
        raise ValueError("--ct_gaussian_truncation_sigma must be > 0.")
    if args.ct_slice_tile_size < 1:
        raise ValueError("--ct_slice_tile_size must be >= 1.")
    if args.ct_grid_cell_voxels < 1:
        raise ValueError("--ct_grid_cell_voxels must be >= 1.")
    if args.ct_gradient_sigma_voxels < 0.0:
        raise ValueError("--ct_gradient_sigma_voxels must be >= 0.")
    if args.ct_bulk_edt_alpha <= 0.0:
        raise ValueError("--ct_bulk_edt_alpha must be > 0.")
    if args.ct_thickness_max is not None and args.ct_thickness_max <= 0.0:
        raise ValueError("--ct_thickness_max must be > 0 when provided.")
    if args.ct_tangential_max_scale <= 0.0:
        raise ValueError("--ct_tangential_max_scale must be > 0.")
    if args.ct_surface_tangential_max_scale <= 0.0:
        raise ValueError("--ct_surface_tangential_max_scale must be > 0.")
    if args.ct_bulk_max_scale <= 0.0:
        raise ValueError("--ct_bulk_max_scale must be > 0.")
    if args.ct_bulk_density_cap <= 0.0:
        raise ValueError("--ct_bulk_density_cap must be > 0.")
    if args.ct_material_k < 1:
        raise ValueError("--ct_material_k must be >= 1.")
    if args.ct_bulk_k < 1:
        raise ValueError("--ct_bulk_k must be >= 1.")
    if args.ct_bulk_overlap_k < 1:
        raise ValueError("--ct_bulk_overlap_k must be >= 1.")
    if args.ct_surface_min_opacity <= 0.0 or args.ct_surface_min_opacity >= 1.0:
        raise ValueError("--ct_surface_min_opacity must be in (0, 1).")
    if args.ct_surface_target_opacity <= 0.0 or args.ct_surface_target_opacity >= 1.0:
        raise ValueError("--ct_surface_target_opacity must be in (0, 1).")
    if args.ct_support_threshold_mode not in {"otsu", "multi_otsu"}:
        raise ValueError("--ct_support_threshold_mode must be one of {'otsu', 'multi_otsu'}.")
    if (
        args.ct_lambda_render < 0.0
        or args.ct_lambda_occupancy < 0.0
        or args.ct_lambda_shape < 0.0
        or args.ct_lambda_material < 0.0
        or args.ct_lambda_bulk < 0.0
        or args.ct_align_weight < 0.0
        or args.ct_cross_floor_weight < 0.0
        or args.ct_lambda_field_recon < 0.0
        or args.ct_lambda_boundary_ridge < 0.0
        or args.ct_lambda_surface_thickness < 0.0
        or args.ct_lambda_surface_tangential_scale < 0.0
        or args.ct_lambda_surface_opacity < 0.0
        or args.ct_lambda_bulk_scale < 0.0
        or args.ct_lambda_bulk_overlap < 0.0
    ):
        raise ValueError(
            "CT lambda and compact-loss weights must be >= 0."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CT training requires CUDA in the current implementation.")
    if args.ct_backend not in {"auto", "cuda", "python"}:
        raise ValueError("--ct_backend must be one of {'auto', 'cuda', 'python'}.")
    if args.ct_backend == "python":
        raise ValueError("The active intensity-driven CT training path requires the CUDA/native backend; --ct_backend=python is unsupported.")


def _normalize_ct_query_args(args):
    args._ct_support_sample_count_auto = getattr(args, "ct_support_sample_count", None) is None
    args._ct_air_sample_count_auto = getattr(args, "ct_air_sample_count", None) is None
    legacy_interior = getattr(args, "ct_interior_query_count", None)
    material_count = getattr(args, "ct_material_query_count", None)
    if material_count is None:
        material_count = legacy_interior if legacy_interior is not None else 4096
    args.ct_material_query_count = int(material_count)
    args.ct_interior_query_count = int(args.ct_material_query_count)
    if getattr(args, "ct_support_sample_count", None) is None:
        args.ct_support_sample_count = int(args.ct_material_query_count)
    else:
        args.ct_support_sample_count = int(args.ct_support_sample_count)
    if getattr(args, "ct_air_sample_count", None) is None:
        args.ct_air_sample_count = int(
            max(
                int(getattr(args, "ct_void_query_count", 4096) or 0),
                int(getattr(args, "ct_exterior_query_count", 4096) or 0),
            )
        )
    else:
        args.ct_air_sample_count = int(args.ct_air_sample_count)
    if getattr(args, "ct_void_query_count", None) is None:
        args.ct_void_query_count = 4096
    else:
        args.ct_void_query_count = int(args.ct_void_query_count)
    if getattr(args, "ct_max_material_classes", None) is None:
        args.ct_max_material_classes = 3
    else:
        args.ct_max_material_classes = int(args.ct_max_material_classes)
    if getattr(args, "ct_void_negative_weight", None) is None:
        args.ct_void_negative_weight = 4.0
    else:
        args.ct_void_negative_weight = float(args.ct_void_negative_weight)
    if getattr(args, "ct_cavity_patch_bias", None) is None:
        args.ct_cavity_patch_bias = 0.6
    else:
        args.ct_cavity_patch_bias = float(args.ct_cavity_patch_bias)
    if getattr(args, "ct_void_boundary_offset_scale", None) is None:
        args.ct_void_boundary_offset_scale = 0.75
    else:
        args.ct_void_boundary_offset_scale = float(args.ct_void_boundary_offset_scale)
    if getattr(args, "ct_void_boundary_margin", None) is None:
        args.ct_void_boundary_margin = 0.25
    else:
        args.ct_void_boundary_margin = float(args.ct_void_boundary_margin)
    render_weight = getattr(args, "ct_lambda_render", None)
    if render_weight is None:
        render_weight = getattr(args, "ct_lambda_slice", 1.0)
    args.ct_lambda_render = float(render_weight)
    args.ct_lambda_slice = float(args.ct_lambda_render)

    occupancy_weight = getattr(args, "ct_lambda_occupancy", None)
    legacy_field_recon = getattr(args, "ct_lambda_field_recon", None)
    if legacy_field_recon is not None:
        occupancy_weight = legacy_field_recon
    if occupancy_weight is None:
        occupancy_weight = 0.5
    args.ct_lambda_occupancy = float(occupancy_weight)
    args.ct_lambda_field_recon = float(args.ct_lambda_occupancy)

    legacy_shape_weights = [
        value
        for value in (
            getattr(args, "ct_lambda_surface_thickness", None),
            getattr(args, "ct_lambda_surface_tangential_scale", None),
            getattr(args, "ct_lambda_surface_opacity", None),
            getattr(args, "ct_lambda_thickness", None),
        )
        if value is not None
    ]
    shape_weight = getattr(args, "ct_lambda_shape", None)
    if shape_weight is None:
        shape_weight = max(float(value) for value in legacy_shape_weights) if legacy_shape_weights else 0.02
    args.ct_lambda_shape = float(shape_weight)
    args.ct_lambda_surface_thickness = float(getattr(args, "ct_lambda_surface_thickness", 0.0) or 0.0)
    args.ct_lambda_surface_tangential_scale = float(getattr(args, "ct_lambda_surface_tangential_scale", 0.0) or 0.0)
    args.ct_lambda_surface_opacity = float(getattr(args, "ct_lambda_surface_opacity", 0.0) or 0.0)
    args.ct_lambda_thickness = float(getattr(args, "ct_lambda_thickness", 0.0) or 0.0)

    legacy_bulk_weights = [
        value
        for value in (
            getattr(args, "ct_lambda_bulk_scale", None),
            getattr(args, "ct_lambda_bulk_overlap", None),
        )
        if value is not None
    ]
    bulk_weight = getattr(args, "ct_lambda_bulk", None)
    if bulk_weight is None:
        bulk_weight = max(float(value) for value in legacy_bulk_weights) if legacy_bulk_weights else 0.01
    args.ct_lambda_bulk = float(bulk_weight)
    args.ct_lambda_bulk_scale = float(getattr(args, "ct_lambda_bulk_scale", 0.0) or 0.0)
    args.ct_lambda_bulk_overlap = float(getattr(args, "ct_lambda_bulk_overlap", 0.0) or 0.0)

    args.ct_lambda_boundary_ridge = float(getattr(args, "ct_lambda_boundary_ridge", 0.0) or 0.0)
    args.ct_lambda_boundary_center = float(getattr(args, "ct_lambda_boundary_center", 0.0) or 0.0)
    args.ct_lambda_boundary_normal = float(getattr(args, "ct_lambda_boundary_normal", 0.0) or 0.0)
    args.ct_lambda_signed_surface = float(getattr(args, "ct_lambda_signed_surface", 0.0) or 0.0)
    args.ct_lambda_void_boundary = float(getattr(args, "ct_lambda_void_boundary", 0.0) or 0.0)
    args.ct_lambda_plane = float(getattr(args, "ct_lambda_plane", 0.0) or 0.0)
    args.ct_lambda_normal = float(getattr(args, "ct_lambda_normal", 0.0) or 0.0)
    if getattr(args, "ct_signed_field_band_voxels", None) is None:
        args.ct_signed_field_band_voxels = 3
    else:
        args.ct_signed_field_band_voxels = int(args.ct_signed_field_band_voxels)
    if getattr(args, "ct_gradient_sigma_voxels", None) is None:
        args.ct_gradient_sigma_voxels = 1.0
    else:
        args.ct_gradient_sigma_voxels = float(args.ct_gradient_sigma_voxels)
    if getattr(args, "ct_support_threshold_mode", None) is None:
        args.ct_support_threshold_mode = "otsu"
    else:
        args.ct_support_threshold_mode = str(args.ct_support_threshold_mode)
    if getattr(args, "ct_density_query_tile_points", None) is None:
        args.ct_density_query_tile_points = 32
    else:
        args.ct_density_query_tile_points = int(args.ct_density_query_tile_points)
    if getattr(args, "ct_knn_tile_size", None) is None:
        args.ct_knn_tile_size = 16
    else:
        args.ct_knn_tile_size = int(args.ct_knn_tile_size)
    if getattr(args, "ct_gaussian_truncation_sigma", None) is None:
        args.ct_gaussian_truncation_sigma = 4.0
    else:
        args.ct_gaussian_truncation_sigma = float(args.ct_gaussian_truncation_sigma)
    if getattr(args, "ct_slice_tile_size", None) is None:
        args.ct_slice_tile_size = 8
    else:
        args.ct_slice_tile_size = int(args.ct_slice_tile_size)
    if getattr(args, "ct_grid_cell_voxels", None) is None:
        args.ct_grid_cell_voxels = 8
    else:
        args.ct_grid_cell_voxels = int(args.ct_grid_cell_voxels)
    if getattr(args, "ct_freeze_bulk_xyz", None) is None:
        args.ct_freeze_bulk_xyz = True
    else:
        args.ct_freeze_bulk_xyz = bool(args.ct_freeze_bulk_xyz)
    if getattr(args, "ct_bulk_edt_alpha", None) is None:
        args.ct_bulk_edt_alpha = 1.0
    else:
        args.ct_bulk_edt_alpha = float(args.ct_bulk_edt_alpha)
    if getattr(args, "ct_surface_target_opacity", None) is None:
        args.ct_surface_target_opacity = 0.9
    else:
        args.ct_surface_target_opacity = float(args.ct_surface_target_opacity)
    if getattr(args, "ct_surface_min_opacity", None) is None:
        args.ct_surface_min_opacity = 0.8
    else:
        args.ct_surface_min_opacity = float(args.ct_surface_min_opacity)
    args.ct_lambda_material = float(getattr(args, "ct_lambda_material", 0.1) if getattr(args, "ct_lambda_material", None) is not None else 0.1)
    args.ct_align_weight = float(getattr(args, "ct_align_weight", 0.3) if getattr(args, "ct_align_weight", None) is not None else 0.3)
    args.ct_cross_floor_weight = float(
        getattr(args, "ct_cross_floor_weight", 0.2) if getattr(args, "ct_cross_floor_weight", None) is not None else 0.2
    )
    surface_thickness_max = getattr(args, "surface_thickness_max", None)
    if surface_thickness_max is None:
        surface_thickness_max = getattr(args, "planar_thickness_max", None)
    if surface_thickness_max is None:
        surface_thickness_max = getattr(args, "ct_thickness_max", None)
    args.surface_thickness_max = None if surface_thickness_max is None else float(surface_thickness_max)
    args.planar_thickness_max = args.surface_thickness_max
    ct_thickness_max = getattr(args, "ct_thickness_max", None)
    if ct_thickness_max is None:
        ct_thickness_max = args.surface_thickness_max
    args.ct_thickness_max = None if ct_thickness_max is None else float(ct_thickness_max)
    if getattr(args, "ct_tangential_max_scale", None) is None:
        args.ct_tangential_max_scale = float(getattr(args, "ct_surface_tangential_max_scale", 4.0) or 4.0)
    else:
        args.ct_tangential_max_scale = float(args.ct_tangential_max_scale)
    args.ct_surface_tangential_max_scale = float(args.ct_tangential_max_scale)
    if getattr(args, "ct_bulk_max_scale", None) is None:
        args.ct_bulk_max_scale = 4.0
    else:
        args.ct_bulk_max_scale = float(args.ct_bulk_max_scale)
    if getattr(args, "ct_bulk_density_cap", None) is None:
        args.ct_bulk_density_cap = 3.0
    else:
        args.ct_bulk_density_cap = float(args.ct_bulk_density_cap)
    if getattr(args, "ct_material_k", None) is None:
        args.ct_material_k = int(getattr(args, "ct_neighbor_k", 8) or 8)
    else:
        args.ct_material_k = int(args.ct_material_k)
    args.ct_neighbor_k = int(args.ct_material_k)
    if getattr(args, "ct_bulk_k", None) is None:
        legacy_bulk_k = getattr(args, "ct_bulk_overlap_k", None)
        args.ct_bulk_k = 8 if legacy_bulk_k is None else int(legacy_bulk_k)
    else:
        args.ct_bulk_k = int(args.ct_bulk_k)
    if getattr(args, "ct_bulk_overlap_k", None) is None:
        args.ct_bulk_overlap_k = int(args.ct_bulk_k)
    else:
        args.ct_bulk_overlap_k = int(args.ct_bulk_overlap_k)


def _ct_spatial_extent(volume_shape_dhw, spacing_zyx):
    extents = [float(dim) * float(spacing) for dim, spacing in zip(volume_shape_dhw, spacing_zyx)]
    return max(extents)


def _to_cuda_analysis(analysis):
    analysis_cuda = {}
    for key, value in analysis.items():
        if isinstance(value, torch.Tensor):
            analysis_cuda[key] = value.to(device="cuda")
            continue
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                tensor = torch.as_tensor(value, dtype=torch.bool, device="cuda")
            elif np.issubdtype(value.dtype, np.integer):
                tensor = torch.as_tensor(value, dtype=torch.long, device="cuda")
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32, device="cuda")
            analysis_cuda[key] = tensor
            continue
        analysis_cuda[key] = value
    return analysis_cuda


def _build_gaussian_kernel1d(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if sigma <= 0.0:
        return torch.ones((1,), dtype=dtype, device=device)
    radius = max(1, int(round(3.0 * float(sigma))))
    positions = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(positions * positions) / (2.0 * float(sigma) * float(sigma)))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def _smooth_volume_3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return volume
    if volume.ndim != 3:
        raise ValueError("volume must have shape (D, H, W).")
    dtype = volume.dtype
    device = volume.device
    kernel = _build_gaussian_kernel1d(float(sigma), dtype=dtype, device=device)
    radius = int((kernel.numel() - 1) // 2)
    work = volume.unsqueeze(0).unsqueeze(0)

    kernel_z = kernel.view(1, 1, -1, 1, 1)
    kernel_y = kernel.view(1, 1, 1, -1, 1)
    kernel_x = kernel.view(1, 1, 1, 1, -1)

    work = F.conv3d(F.pad(work, (0, 0, 0, 0, radius, radius), mode="replicate"), kernel_z)
    work = F.conv3d(F.pad(work, (0, 0, radius, radius, 0, 0), mode="replicate"), kernel_y)
    work = F.conv3d(F.pad(work, (radius, radius, 0, 0, 0, 0), mode="replicate"), kernel_x)
    return work.squeeze(0).squeeze(0)


def _central_difference_3d(volume: torch.Tensor):
    if volume.ndim != 3:
        raise ValueError("volume must have shape (D, H, W).")
    padded = F.pad(volume.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="replicate")
    gz = 0.5 * (padded[:, :, 2:, 1:-1, 1:-1] - padded[:, :, :-2, 1:-1, 1:-1]).squeeze(0).squeeze(0)
    gy = 0.5 * (padded[:, :, 1:-1, 2:, 1:-1] - padded[:, :, 1:-1, :-2, 1:-1]).squeeze(0).squeeze(0)
    gx = 0.5 * (padded[:, :, 1:-1, 1:-1, 2:] - padded[:, :, 1:-1, 1:-1, :-2]).squeeze(0).squeeze(0)
    return gx, gy, gz


def _material_boundary_mask_torch(material_label_volume: torch.Tensor) -> torch.Tensor:
    material_label_volume = torch.as_tensor(material_label_volume, device=material_label_volume.device, dtype=torch.long)
    material_mask = material_label_volume > 0
    if not torch.any(material_mask):
        return torch.zeros_like(material_mask, dtype=torch.bool)

    boundary_mask = torch.zeros_like(material_mask, dtype=torch.bool)
    for axis in range(3):
        slicer_a = [slice(None)] * 3
        slicer_b = [slice(None)] * 3
        slicer_a[axis] = slice(1, None)
        slicer_b[axis] = slice(None, -1)
        current = material_label_volume[tuple(slicer_a)]
        previous = material_label_volume[tuple(slicer_b)]
        change = current != previous
        current_material = current > 0
        previous_material = previous > 0
        boundary_mask[tuple(slicer_a)] |= change & current_material
        boundary_mask[tuple(slicer_b)] |= change & previous_material
    return boundary_mask & material_mask


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
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return analysis, metadata, str(analysis_path), str(metadata_path)


def _default_roi_bbox(volume_shape_dhw):
    depth, height, width = [int(value) for value in volume_shape_dhw]
    return np.asarray([[0, depth], [0, height], [0, width]], dtype=np.int32)


def _foreground_slice_bbox(foreground_mask, axis_index, slice_idx):
    if foreground_mask is None:
        return None
    if isinstance(foreground_mask, torch.Tensor):
        if axis_index == 0:
            mask_2d = foreground_mask[int(slice_idx)]
        elif axis_index == 1:
            mask_2d = foreground_mask[:, int(slice_idx), :]
        else:
            mask_2d = foreground_mask[:, :, int(slice_idx)]
        if mask_2d.numel() == 0 or not bool(torch.any(mask_2d).item()):
            return None
        coords = torch.nonzero(mask_2d, as_tuple=False)
        lower = coords.min(dim=0).values.detach().cpu().numpy()
        upper = (coords.max(dim=0).values + 1).detach().cpu().numpy()
        return lower, upper
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


def _get_air_focus_indices(analysis):
    cached = analysis.get("_air_focus_indices")
    if cached is not None:
        return cached
    if isinstance(analysis.get("air_mask"), torch.Tensor) or isinstance(analysis.get("void_mask"), torch.Tensor):
        air_mask = analysis.get("air_mask", analysis.get("void_mask")).to(dtype=torch.bool)
        if not torch.any(air_mask):
            empty = torch.empty((0, 3), dtype=torch.long, device=air_mask.device)
            analysis["_air_focus_indices"] = empty
            return empty
        support_mask = analysis.get("coarse_support_mask", analysis.get("material_mask"))
        if support_mask is not None and torch.any(support_mask):
            near_support = torch.zeros_like(air_mask)
            near_support[1:, :, :] |= support_mask[:-1, :, :]
            near_support[:-1, :, :] |= support_mask[1:, :, :]
            near_support[:, 1:, :] |= support_mask[:, :-1, :]
            near_support[:, :-1, :] |= support_mask[:, 1:, :]
            near_support[:, :, 1:] |= support_mask[:, :, :-1]
            near_support[:, :, :-1] |= support_mask[:, :, 1:]
            air_focus_mask = torch.logical_and(air_mask, near_support)
            if not torch.any(air_focus_mask):
                air_focus_mask = air_mask
        else:
            air_focus_mask = air_mask

        indices = torch.nonzero(air_focus_mask, as_tuple=False).to(dtype=torch.long)
        analysis["_air_focus_indices"] = indices
        return indices
    air_mask = None
    if "air_mask" in analysis:
        air_mask = np.asarray(analysis["air_mask"], dtype=bool)
    elif "void_mask" in analysis:
        air_mask = np.asarray(analysis["void_mask"], dtype=bool)
    if air_mask is None or not np.any(air_mask):
        empty = np.empty((0, 3), dtype=np.int32)
        analysis["_air_focus_indices"] = empty
        return empty

    support_mask = np.asarray(analysis["coarse_support_mask"], dtype=bool) if "coarse_support_mask" in analysis else np.asarray(analysis["material_mask"], dtype=bool) if "material_mask" in analysis else None
    if support_mask is not None and np.any(support_mask):
        structure = ndimage.generate_binary_structure(3, 1)
        near_support = ndimage.binary_dilation(support_mask, structure=structure, iterations=1)
        air_focus_mask = np.logical_and(air_mask, near_support)
        if not np.any(air_focus_mask):
            air_focus_mask = air_mask
    else:
        air_focus_mask = air_mask

    indices = np.argwhere(air_focus_mask).astype(np.int32)
    analysis["_air_focus_indices"] = indices
    return indices


def _sample_ct_patch_spec(analysis, volume_shape_dhw, requested_patch_size, spacing_zyx, cavity_patch_bias=0.0):
    if isinstance(analysis.get("roi_bbox"), torch.Tensor):
        device = analysis["roi_bbox"].device

        def _randint(low: int, high: int) -> int:
            return int(torch.randint(int(low), int(high), (1,), device=device).item())

        def _randfloat() -> float:
            return float(torch.rand((), device=device).item())

        axis_index = _randint(0, 3)
        axis_name = ["z", "y", "x"][axis_index]
        roi_bbox = analysis["roi_bbox"]
        foreground_mask = analysis["coarse_support_mask"] if "coarse_support_mask" in analysis else analysis["material_mask"] if "material_mask" in analysis else analysis["foreground_mask"] if "foreground_mask" in analysis else None
        boundary_points = analysis["boundary_points"] if "boundary_points" in analysis else torch.empty((0, 3), dtype=torch.float32, device=device)
        air_focus_indices = _get_air_focus_indices(analysis)

        slice_lower = int(roi_bbox[axis_index, 0].item())
        slice_upper = int(max(slice_lower + 1, min(int(roi_bbox[axis_index, 1].item()), int(volume_shape_dhw[axis_index]))))

        if axis_index == 0:
            full_height, full_width = int(volume_shape_dhw[1]), int(volume_shape_dhw[2])
        elif axis_index == 1:
            full_height, full_width = int(volume_shape_dhw[0]), int(volume_shape_dhw[2])
        else:
            full_height, full_width = int(volume_shape_dhw[0]), int(volume_shape_dhw[1])

        patch_height = min(int(requested_patch_size), full_height)
        patch_width = min(int(requested_patch_size), full_width)

        if air_focus_indices.shape[0] > 0 and float(cavity_patch_bias) > 0.0 and _randfloat() < float(cavity_patch_bias):
            center_index = air_focus_indices[_randint(0, int(air_focus_indices.shape[0]))].to(dtype=torch.float32)
            if axis_index == 0:
                slice_idx = int(torch.clamp(center_index[0], min=slice_lower, max=slice_upper - 1).item())
                center_h = float(center_index[1].item()) + 0.5
                center_w = float(center_index[2].item()) + 0.5
            elif axis_index == 1:
                slice_idx = int(torch.clamp(center_index[1], min=slice_lower, max=slice_upper - 1).item())
                center_h = float(center_index[0].item()) + 0.5
                center_w = float(center_index[2].item()) + 0.5
            else:
                slice_idx = int(torch.clamp(center_index[2], min=slice_lower, max=slice_upper - 1).item())
                center_h = float(center_index[0].item()) + 0.5
                center_w = float(center_index[1].item()) + 0.5
            jitter_h = _randint(-max(1, patch_height // 5), max(2, patch_height // 5 + 1))
            jitter_w = _randint(-max(1, patch_width // 5), max(2, patch_width // 5 + 1))
            origin_h = int(np.clip(round(center_h - patch_height * 0.5 + jitter_h), 0, max(0, full_height - patch_height)))
            origin_w = int(np.clip(round(center_w - patch_width * 0.5 + jitter_w), 0, max(0, full_width - patch_width)))
            return axis_name, slice_idx, (origin_h, origin_w), (patch_height, patch_width)

        if boundary_points.shape[0] > 0:
            center_point = boundary_points[_randint(0, int(boundary_points.shape[0]))]
            if axis_index == 0:
                slice_idx = int(np.clip(np.floor(float(center_point[2].item()) / max(float(spacing_zyx[0]), 1e-8)), slice_lower, slice_upper - 1))
                center_h = float(center_point[1].item()) / max(float(spacing_zyx[1]), 1e-8)
                center_w = float(center_point[0].item()) / max(float(spacing_zyx[2]), 1e-8)
            elif axis_index == 1:
                slice_idx = int(np.clip(np.floor(float(center_point[1].item()) / max(float(spacing_zyx[1]), 1e-8)), slice_lower, slice_upper - 1))
                center_h = float(center_point[2].item()) / max(float(spacing_zyx[0]), 1e-8)
                center_w = float(center_point[0].item()) / max(float(spacing_zyx[2]), 1e-8)
            else:
                slice_idx = int(np.clip(np.floor(float(center_point[0].item()) / max(float(spacing_zyx[2]), 1e-8)), slice_lower, slice_upper - 1))
                center_h = float(center_point[2].item()) / max(float(spacing_zyx[0]), 1e-8)
                center_w = float(center_point[1].item()) / max(float(spacing_zyx[1]), 1e-8)
            jitter_h = _randint(-max(1, patch_height // 4), max(2, patch_height // 4 + 1))
            jitter_w = _randint(-max(1, patch_width // 4), max(2, patch_width // 4 + 1))
            origin_h = int(np.clip(round(center_h - patch_height * 0.5 + jitter_h), 0, max(0, full_height - patch_height)))
            origin_w = int(np.clip(round(center_w - patch_width * 0.5 + jitter_w), 0, max(0, full_width - patch_width)))
            return axis_name, slice_idx, (origin_h, origin_w), (patch_height, patch_width)

        slice_idx = _randint(slice_lower, slice_upper)
        foreground_bbox = _foreground_slice_bbox(foreground_mask, axis_index, slice_idx)
        if foreground_bbox is None:
            origin_h = _randint(0, max(1, full_height - patch_height + 1))
            origin_w = _randint(0, max(1, full_width - patch_width + 1))
        else:
            lower, upper = foreground_bbox
            origin_h = _choose_patch_origin(lower[0], upper[0], full_height, patch_height)
            origin_w = _choose_patch_origin(lower[1], upper[1], full_width, patch_width)

        return axis_name, slice_idx, (origin_h, origin_w), (patch_height, patch_width)

    axis_index = int(np.random.randint(0, 3))
    axis_name = ["z", "y", "x"][axis_index]
    roi_bbox = analysis["roi_bbox"] if "roi_bbox" in analysis else _default_roi_bbox(volume_shape_dhw)
    foreground_mask = analysis["coarse_support_mask"] if "coarse_support_mask" in analysis else analysis["material_mask"] if "material_mask" in analysis else analysis["foreground_mask"] if "foreground_mask" in analysis else None
    boundary_points = np.asarray(analysis["boundary_points"], dtype=np.float32) if "boundary_points" in analysis else np.empty((0, 3), dtype=np.float32)
    air_focus_indices = _get_air_focus_indices(analysis)

    slice_lower, slice_upper = [int(value) for value in roi_bbox[axis_index]]
    slice_upper = max(slice_lower + 1, min(slice_upper, int(volume_shape_dhw[axis_index])))

    if axis_index == 0:
        full_height, full_width = int(volume_shape_dhw[1]), int(volume_shape_dhw[2])
    elif axis_index == 1:
        full_height, full_width = int(volume_shape_dhw[0]), int(volume_shape_dhw[2])
    else:
        full_height, full_width = int(volume_shape_dhw[0]), int(volume_shape_dhw[1])

    patch_height = min(int(requested_patch_size), full_height)
    patch_width = min(int(requested_patch_size), full_width)

    if air_focus_indices.shape[0] > 0 and float(cavity_patch_bias) > 0.0 and np.random.random() < float(cavity_patch_bias):
        center_index = air_focus_indices[int(np.random.randint(0, air_focus_indices.shape[0]))]
        if axis_index == 0:
            slice_idx = int(np.clip(center_index[0], slice_lower, slice_upper - 1))
            center_h = float(center_index[1]) + 0.5
            center_w = float(center_index[2]) + 0.5
        elif axis_index == 1:
            slice_idx = int(np.clip(center_index[1], slice_lower, slice_upper - 1))
            center_h = float(center_index[0]) + 0.5
            center_w = float(center_index[2]) + 0.5
        else:
            slice_idx = int(np.clip(center_index[2], slice_lower, slice_upper - 1))
            center_h = float(center_index[0]) + 0.5
            center_w = float(center_index[1]) + 0.5
        jitter_h = int(np.random.randint(-max(1, patch_height // 5), max(2, patch_height // 5 + 1)))
        jitter_w = int(np.random.randint(-max(1, patch_width // 5), max(2, patch_width // 5 + 1)))
        origin_h = int(np.clip(round(center_h - patch_height * 0.5 + jitter_h), 0, max(0, full_height - patch_height)))
        origin_w = int(np.clip(round(center_w - patch_width * 0.5 + jitter_w), 0, max(0, full_width - patch_width)))
        return axis_name, slice_idx, (origin_h, origin_w), (patch_height, patch_width)

    if boundary_points.shape[0] > 0:
        center_point = boundary_points[int(np.random.randint(0, boundary_points.shape[0]))]
        if axis_index == 0:
            slice_idx = int(np.clip(np.floor(center_point[2] / max(float(spacing_zyx[0]), 1e-8)), slice_lower, slice_upper - 1))
            center_h = center_point[1] / max(float(spacing_zyx[1]), 1e-8)
            center_w = center_point[0] / max(float(spacing_zyx[2]), 1e-8)
        elif axis_index == 1:
            slice_idx = int(np.clip(np.floor(center_point[1] / max(float(spacing_zyx[1]), 1e-8)), slice_lower, slice_upper - 1))
            center_h = center_point[2] / max(float(spacing_zyx[0]), 1e-8)
            center_w = center_point[0] / max(float(spacing_zyx[2]), 1e-8)
        else:
            slice_idx = int(np.clip(np.floor(center_point[0] / max(float(spacing_zyx[2]), 1e-8)), slice_lower, slice_upper - 1))
            center_h = center_point[2] / max(float(spacing_zyx[0]), 1e-8)
            center_w = center_point[1] / max(float(spacing_zyx[1]), 1e-8)
        jitter_h = int(np.random.randint(-max(1, patch_height // 4), max(2, patch_height // 4 + 1)))
        jitter_w = int(np.random.randint(-max(1, patch_width // 4), max(2, patch_width // 4 + 1)))
        origin_h = int(np.clip(round(center_h - patch_height * 0.5 + jitter_h), 0, max(0, full_height - patch_height)))
        origin_w = int(np.clip(round(center_w - patch_width * 0.5 + jitter_w), 0, max(0, full_width - patch_width)))
        return axis_name, slice_idx, (origin_h, origin_w), (patch_height, patch_width)

    slice_idx = int(np.random.randint(slice_lower, slice_upper))
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


def _make_void_boundary_query_state(training_state):
    return SimpleNamespace(
        xyz=training_state.xyz,
        rotation_mats=training_state.rotation_mats.detach(),
        scales=training_state.scales.detach(),
        opacity=training_state.opacity.detach(),
    )


def _ensure_intensity_driven_analysis(analysis, volume_np, spacing_zyx, sigma, threshold_mode):
    upgraded = dict(analysis)
    preprocessor = CTPreprocessor()

    has_support = all(key in upgraded for key in ("coarse_support_mask", "roi_bbox"))
    if not has_support:
        if "material_mask" in upgraded and "roi_bbox" in upgraded:
            support = {
                "support_mask": np.asarray(upgraded["material_mask"], dtype=bool),
                "roi_bbox": np.asarray(upgraded["roi_bbox"], dtype=np.int32),
                "roi_mask": np.asarray(upgraded["foreground_mask"], dtype=bool) if "foreground_mask" in upgraded else np.asarray(upgraded["material_mask"], dtype=bool),
                "air_mask": np.asarray(upgraded["void_mask"], dtype=bool) if "void_mask" in upgraded else np.logical_not(np.asarray(upgraded["material_mask"], dtype=bool)),
                "support_threshold": float(np.asarray(volume_np, dtype=np.float32)[np.asarray(upgraded["material_mask"], dtype=bool)].min()) if np.any(upgraded["material_mask"]) else 0.5,
            }
        else:
            support = preprocessor.segment_coarse_support(
                np.asarray(volume_np, dtype=np.float32),
                threshold_mode=threshold_mode,
            )
        upgraded["coarse_support_mask"] = np.asarray(support["support_mask"], dtype=bool)
        upgraded["material_mask"] = np.asarray(support["support_mask"], dtype=bool)
        upgraded["foreground_mask"] = np.asarray(support["roi_mask"], dtype=bool)
        upgraded["void_mask"] = np.asarray(support["air_mask"], dtype=bool)
        upgraded["air_mask"] = np.asarray(support["air_mask"], dtype=bool)
        upgraded["material_label_volume"] = np.asarray(support["support_mask"], dtype=np.int32)
        upgraded["roi_bbox"] = np.asarray(support["roi_bbox"], dtype=np.int32)
        upgraded["support_threshold"] = np.asarray([float(support["support_threshold"])], dtype=np.float32)
        warnings.warn(
            "Phase 1 bundle was upgraded to the intensity-driven coarse-support schema at train-time.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif "air_mask" not in upgraded:
        foreground_mask = np.asarray(upgraded.get("foreground_mask", upgraded["coarse_support_mask"]), dtype=bool)
        support_mask = np.asarray(upgraded["coarse_support_mask"], dtype=bool)
        upgraded["air_mask"] = np.logical_and(foreground_mask, np.logical_not(support_mask))

    has_boundary = all(
        key in upgraded
        for key in (
            "boundary_points",
            "boundary_normals",
            "boundary_tangent_u",
            "boundary_tangent_v",
            "boundary_strength",
            "boundary_material_id",
        )
    )
    if not has_boundary:
        analyzer = GeometryAnalyzer(sigma=sigma)
        boundary_points = preprocessor.extract_intensity_surface_points(
            np.asarray(volume_np, dtype=np.float32),
            np.asarray(upgraded["coarse_support_mask"], dtype=bool),
            spacing_zyx,
            sigma=sigma,
            gradient_percentile=60.0,
        )
        boundary_material_id = np.zeros((boundary_points.shape[0], 1), dtype=np.int64)
        boundary_normals, boundary_tangent_u, boundary_tangent_v, boundary_strength = analyzer.estimate_boundary_geometry(
            boundary_points,
            volume_np,
            spacing_zyx,
        )
        upgraded["boundary_points"] = boundary_points
        upgraded["boundary_normals"] = boundary_normals
        upgraded["boundary_tangent_u"] = boundary_tangent_u
        upgraded["boundary_tangent_v"] = boundary_tangent_v
        upgraded["boundary_strength"] = boundary_strength
        upgraded["boundary_material_id"] = boundary_material_id
        warnings.warn(
            "Phase 1 bundle is legacy surface-first data; boundary samples were recomputed at train-time.",
            RuntimeWarning,
            stacklevel=2,
        )

    has_bulk = all(key in upgraded for key in ("interior_points", "interior_density_seed", "interior_material_id"))
    if not has_bulk:
        target_count = max(1, int(np.asarray(upgraded["boundary_points"]).shape[0]))
        interior_points, interior_density_seed, interior_material_id = preprocessor.sample_support_points(
            np.asarray(upgraded["coarse_support_mask"], dtype=bool),
            np.asarray(volume_np, dtype=np.float32),
            spacing_zyx,
            target_count=target_count,
            boundary_margin_voxels=0,
        )
        upgraded["interior_points"] = interior_points
        upgraded["interior_density_seed"] = interior_density_seed
        upgraded["interior_material_id"] = interior_material_id
    return upgraded


def _prepare_intensity_field_cache(analysis, volume_np, spacing_zyx, sigma, device):
    if not isinstance(volume_np, torch.Tensor):
        volume = torch.as_tensor(np.asarray(volume_np, dtype=np.float32), device=device, dtype=torch.float32)
    else:
        volume = volume_np.to(device=device, dtype=torch.float32)

    smoothed_volume = _smooth_volume_3d(volume, sigma)
    gx, gy, gz = _central_difference_3d(smoothed_volume)
    gradient_vectors = torch.stack((-gx, -gy, -gz), dim=-1)
    gradient_magnitude = torch.linalg.norm(gradient_vectors, dim=-1)
    gradient_norm = gradient_magnitude.unsqueeze(-1)
    unit_normals = torch.zeros_like(gradient_vectors)
    valid = gradient_norm[..., 0] > 1e-6
    unit_normals[valid] = gradient_vectors[valid] / gradient_norm[valid]

    dgx, dgy, dgz = _central_difference_3d(gradient_magnitude)
    grad_mag_gradient = torch.stack((-dgx, -dgy, -dgz), dim=-1)
    directional_derivative = torch.sum(grad_mag_gradient * unit_normals, dim=-1)

    support_mask = torch.as_tensor(
        analysis.get("coarse_support_mask", analysis.get("material_mask")),
        device=device,
        dtype=torch.bool,
    )
    if "air_mask" in analysis:
        air_mask = torch.as_tensor(analysis["air_mask"], device=device, dtype=torch.bool)
    else:
        foreground_mask = torch.as_tensor(analysis.get("foreground_mask", support_mask), device=device, dtype=torch.bool)
        air_mask = torch.logical_and(foreground_mask, torch.logical_not(support_mask))

    threshold_tensor = analysis.get("support_threshold")
    if threshold_tensor is None:
        support_values = volume[support_mask]
        threshold_value = float(torch.quantile(support_values, 0.05).item()) if support_values.numel() > 0 else 0.5
    else:
        threshold_value = float(torch.as_tensor(threshold_tensor, device=device, dtype=torch.float32).reshape(-1)[0].item())
    scale = max(threshold_value, 1.0 - threshold_value, 1e-4)
    target_field_native = ((smoothed_volume - threshold_value) / scale).clamp(-1.0, 1.0)

    boundary_mask = torch.zeros_like(support_mask)
    boundary_mask[1:, :, :] |= support_mask[:-1, :, :] & (~support_mask[1:, :, :])
    boundary_mask[:-1, :, :] |= support_mask[1:, :, :] & (~support_mask[:-1, :, :])
    boundary_mask[:, 1:, :] |= support_mask[:, :-1, :] & (~support_mask[:, 1:, :])
    boundary_mask[:, :-1, :] |= support_mask[:, 1:, :] & (~support_mask[:, :-1, :])
    boundary_mask[:, :, 1:] |= support_mask[:, :, :-1] & (~support_mask[:, :, 1:])
    boundary_mask[:, :, :-1] |= support_mask[:, :, 1:] & (~support_mask[:, :, :-1])
    boundary_values = gradient_magnitude[boundary_mask]
    if boundary_values.numel() > 0:
        normalizer = torch.quantile(boundary_values, 0.99)
        if not torch.isfinite(normalizer) or float(normalizer.item()) <= 1e-6:
            normalizer = boundary_values.max()
    else:
        normalizer = gradient_magnitude.max()
    if not torch.isfinite(normalizer) or float(normalizer.item()) <= 1e-6:
        strength_native = torch.zeros_like(gradient_magnitude)
    else:
        strength_native = (gradient_magnitude / normalizer).clamp(0.0, 1.0)

    return {
        "smoothed_volume": smoothed_volume.contiguous(),
        "target_field_native": target_field_native.contiguous(),
        "target_field": target_field_native.unsqueeze(0).unsqueeze(0).contiguous(),
        "gradient_strength_native": strength_native.contiguous(),
        "gradient_strength": strength_native.unsqueeze(0).unsqueeze(0).contiguous(),
        "gradient_normal_native": unit_normals.contiguous(),
        "gradient_normal": unit_normals.permute(3, 0, 1, 2).unsqueeze(0).contiguous(),
        "directional_derivative_native": directional_derivative.contiguous(),
        "directional_derivative": directional_derivative.unsqueeze(0).unsqueeze(0).contiguous(),
        "support_mask": support_mask.contiguous(),
        "air_mask": air_mask.contiguous(),
        "spacing_zyx": tuple(float(value) for value in spacing_zyx),
        "support_threshold": float(threshold_value),
    }


def _prepare_field_sample_pools(analysis, volume_shape_dhw, boundary_margin_voxels, device=None):
    if isinstance(analysis.get("coarse_support_mask"), torch.Tensor) or isinstance(analysis.get("material_mask"), torch.Tensor):
        tensor_device = analysis.get("coarse_support_mask", analysis.get("material_mask")).device if device is None else torch.device(device)
        support_mask = analysis.get("coarse_support_mask", analysis.get("material_mask")).to(device=tensor_device, dtype=torch.bool)
        roi_bbox = analysis["roi_bbox"].to(device=tensor_device, dtype=torch.long)
        padding = max(4, int(boundary_margin_voxels) * 2)
        lower = torch.clamp(roi_bbox[:, 0] - padding, min=0)
        upper = torch.minimum(roi_bbox[:, 1] + padding, torch.as_tensor(volume_shape_dhw, dtype=torch.long, device=tensor_device))
        roi_window = torch.zeros_like(support_mask)
        roi_window[
            int(lower[0].item()) : int(upper[0].item()),
            int(lower[1].item()) : int(upper[1].item()),
            int(lower[2].item()) : int(upper[2].item()),
        ] = True
        air_mask = torch.logical_and(roi_window, torch.logical_not(support_mask))
        if not torch.any(air_mask):
            air_mask = torch.logical_not(support_mask)

        near_support = torch.zeros_like(support_mask)
        near_support[1:, :, :] |= support_mask[:-1, :, :]
        near_support[:-1, :, :] |= support_mask[1:, :, :]
        near_support[:, 1:, :] |= support_mask[:, :-1, :]
        near_support[:, :-1, :] |= support_mask[:, 1:, :]
        near_support[:, :, 1:] |= support_mask[:, :, :-1]
        near_support[:, :, :-1] |= support_mask[:, :, 1:]
        air_shell = torch.logical_and(air_mask, near_support)
        if not torch.any(air_shell):
            air_shell = air_mask

        return {
            "support": torch.nonzero(support_mask, as_tuple=False).to(dtype=torch.int32),
            "air_shell": torch.nonzero(air_shell, as_tuple=False).to(dtype=torch.int32),
            "air_shell_band": torch.nonzero(air_shell, as_tuple=False).to(dtype=torch.int32),
            "air_shell_band_ratio": 1.0 if torch.any(air_shell) else 0.0,
            "air": torch.nonzero(air_mask, as_tuple=False).to(dtype=torch.int32),
        }

    support_mask = np.asarray(analysis.get("coarse_support_mask", analysis.get("material_mask")), dtype=bool)
    roi_bbox = np.asarray(analysis["roi_bbox"] if "roi_bbox" in analysis else _default_roi_bbox(volume_shape_dhw), dtype=np.int32)
    padding = max(4, int(boundary_margin_voxels) * 2)
    lower = np.maximum(roi_bbox[:, 0] - padding, 0)
    upper = np.minimum(roi_bbox[:, 1] + padding, np.asarray(volume_shape_dhw, dtype=np.int32))
    roi_window = np.zeros_like(support_mask, dtype=bool)
    roi_window[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]] = True
    air_mask = np.logical_and(roi_window, np.logical_not(support_mask))
    if not np.any(air_mask):
        air_mask = np.logical_not(support_mask)
    structure = ndimage.generate_binary_structure(3, 1)
    near_support = ndimage.binary_dilation(support_mask, structure=structure, iterations=1)
    air_shell = np.logical_and(air_mask, near_support)
    if not np.any(air_shell):
        air_shell = air_mask
    pools = {
        "support": np.argwhere(support_mask).astype(np.int32),
        "air_shell": np.argwhere(air_shell).astype(np.int32),
        "air_shell_band": np.argwhere(air_shell).astype(np.int32),
        "air_shell_band_ratio": 1.0 if np.any(air_shell) else 0.0,
        "air": np.argwhere(air_mask).astype(np.int32),
    }
    if device is None:
        return pools
    return {
        key: (
            torch.as_tensor(value, dtype=torch.int32, device=device)
            if isinstance(value, np.ndarray)
            else float(value)
        )
        for key, value in pools.items()
    }


def _prepare_support_distance_field(analysis, spacing_zyx, device=None):
    if isinstance(analysis.get("coarse_support_mask"), torch.Tensor) or isinstance(analysis.get("material_mask"), torch.Tensor):
        support_mask = analysis.get("coarse_support_mask", analysis.get("material_mask")).detach().cpu().numpy().astype(bool)
    else:
        support_mask = np.asarray(analysis.get("coarse_support_mask", analysis.get("material_mask")), dtype=bool)
    support_distance = ndimage.distance_transform_edt(support_mask, sampling=spacing_zyx).astype(np.float32)
    if device is None:
        return {
            "support_distance_native": support_distance,
            "support_distance": support_distance[None, None, ...],
            "spacing_zyx": tuple(float(value) for value in spacing_zyx),
        }
    support_distance_tensor = torch.as_tensor(support_distance, dtype=torch.float32, device=device)
    return {
        "support_distance_native": support_distance_tensor.contiguous(),
        "support_distance": support_distance_tensor.unsqueeze(0).unsqueeze(0).contiguous(),
        "spacing_zyx": tuple(float(value) for value in spacing_zyx),
    }


def _resolve_field_sample_counts(args, total_gaussians: int):
    support_count = int(args.ct_support_sample_count)
    air_count = int(args.ct_air_sample_count)
    total_gaussians = max(1, int(total_gaussians))
    auto_scaled_count = max(1, total_gaussians // 8)

    if getattr(args, "_ct_support_sample_count_auto", False):
        support_count = max(support_count, auto_scaled_count)
    if getattr(args, "_ct_air_sample_count_auto", False):
        air_count = max(air_count, auto_scaled_count)

    return support_count, air_count


def _resolve_air_sampling_candidates(field_pools, near_boundary_ratio_threshold: float = 0.7):
    ratio = float(field_pools.get("air_shell_band_ratio", 1.0) or 0.0)
    air_shell_band = field_pools.get("air_shell_band")
    if ratio < float(near_boundary_ratio_threshold) and air_shell_band is not None:
        if isinstance(air_shell_band, torch.Tensor):
            if air_shell_band.shape[0] > 0:
                return air_shell_band, ratio, True
        elif len(air_shell_band) > 0:
            return air_shell_band, ratio, True
    return field_pools["air_shell"], ratio, False


def _bulk_mask_tensor(gaussians) -> torch.Tensor:
    return gaussians.get_region_type.reshape(-1) == 1


def _bulk_scale_limits_from_distance(distance_values, spacing_zyx, edt_alpha: float, bulk_max_scale: float):
    min_limit = float(min(spacing_zyx))
    return torch.clamp(
        torch.as_tensor(distance_values, dtype=torch.float32, device=distance_values.device) * float(edt_alpha),
        min=min_limit,
        max=float(bulk_max_scale),
    )


def _log_bulk_barrier_diagnostics(gaussians, support_distance_field, spacing_zyx, edt_alpha: float, bulk_max_scale: float):
    bulk_mask = _bulk_mask_tensor(gaussians)
    if not torch.any(bulk_mask):
        return

    bulk_xyz = gaussians.get_xyz[bulk_mask]
    bulk_distance = sample_volume_field(
        support_distance_field["support_distance"],
        bulk_xyz,
        support_distance_field["spacing_zyx"],
    ).reshape(-1)
    bulk_limits = _bulk_scale_limits_from_distance(
        bulk_distance,
        spacing_zyx=spacing_zyx,
        edt_alpha=edt_alpha,
        bulk_max_scale=bulk_max_scale,
    )
    bulk_scales = torch.exp(gaussians.get_raw_scaling[bulk_mask])
    bulk_max_axis = torch.max(bulk_scales, dim=-1).values

    def _quantiles(tensor: torch.Tensor):
        if tensor.numel() == 0:
            return (0.0, 0.0, 0.0)
        q = torch.quantile(tensor.detach().float(), torch.tensor([0.1, 0.5, 0.9], device=tensor.device))
        return tuple(float(value.item()) for value in q)

    limit_q = _quantiles(bulk_limits)
    scale_q = _quantiles(bulk_max_axis)
    print(
        "Bulk barrier diagnostics: limit(p10/p50/p90)=({0:.4f}, {1:.4f}, {2:.4f}) "
        "initial_bulk_max_axis(p10/p50/p90)=({3:.4f}, {4:.4f}, {5:.4f})".format(
            limit_q[0],
            limit_q[1],
            limit_q[2],
            scale_q[0],
            scale_q[1],
            scale_q[2],
        )
    )
    if bulk_limits.numel() > 0 and bulk_max_axis.numel() > 0:
        median_limit = float(torch.quantile(bulk_limits, 0.5).item())
        median_scale = float(torch.quantile(bulk_max_axis, 0.5).item())
        if median_limit < median_scale:
            warnings.warn(
                "Bulk barrier cap is tighter than the median initial bulk scale; most bulk Gaussians will be projected from the first step.",
                RuntimeWarning,
                stacklevel=2,
            )


def _clear_bulk_scaling_optimizer_momentum(gaussians):
    if gaussians.optimizer is None:
        return 0
    bulk_mask = _bulk_mask_tensor(gaussians)
    if not torch.any(bulk_mask):
        return 0
    state = gaussians.optimizer.state.get(gaussians._scaling)
    if not state:
        return 0
    with torch.no_grad():
        if "exp_avg" in state and isinstance(state["exp_avg"], torch.Tensor):
            state["exp_avg"][bulk_mask] = 0
        if "exp_avg_sq" in state and isinstance(state["exp_avg_sq"], torch.Tensor):
            state["exp_avg_sq"][bulk_mask] = 0
    return int(bulk_mask.sum().item())


def _apply_bulk_scale_hard_projection(gaussians, support_distance_field, spacing_zyx, edt_alpha: float, bulk_max_scale: float):
    bulk_mask = _bulk_mask_tensor(gaussians)
    if not torch.any(bulk_mask):
        return 0

    bulk_xyz = gaussians.get_xyz[bulk_mask]
    bulk_distance = sample_volume_field(
        support_distance_field["support_distance"],
        bulk_xyz,
        support_distance_field["spacing_zyx"],
    ).reshape(-1)
    bulk_limits = _bulk_scale_limits_from_distance(
        bulk_distance,
        spacing_zyx=spacing_zyx,
        edt_alpha=edt_alpha,
        bulk_max_scale=bulk_max_scale,
    )
    with torch.no_grad():
        bulk_scales = torch.exp(gaussians._scaling[bulk_mask])
        clipped_scales = torch.minimum(bulk_scales, bulk_limits.unsqueeze(-1))
        clipped_scales = torch.clamp(clipped_scales, min=1e-8)
        changed = torch.any(torch.abs(clipped_scales - bulk_scales) > 1e-8, dim=1)
        gaussians._scaling[bulk_mask] = torch.log(clipped_scales)
    return int(changed.sum().item())


def _log_air_shell_diagnostics(field_pools):
    ratio = float(field_pools.get("air_shell_band_ratio", 1.0) or 0.0)
    air_shell = field_pools.get("air_shell")
    shell_count = int(air_shell.shape[0]) if isinstance(air_shell, torch.Tensor) else int(len(air_shell))
    print(
        "Air-shell diagnostics: near-boundary fraction within configured band = {0:.3f} (air_shell_count={1})".format(
            ratio,
            shell_count,
        )
    )
    return ratio


def _freeze_bulk_xyz_gradients(gaussians):
    if not getattr(gaussians, "is_initialized", lambda: False)():
        return 0
    xyz_grad = getattr(gaussians, "_xyz", None)
    if not isinstance(xyz_grad, torch.Tensor) or xyz_grad.grad is None:
        return 0
    region_type = gaussians.get_region_type.reshape(-1)
    bulk_mask = region_type == 1
    if not torch.any(bulk_mask):
        return 0
    with torch.no_grad():
        xyz_grad.grad[bulk_mask] = 0
    return int(bulk_mask.sum().item())


def _require_active_boundary_bundle(analysis):
    required_keys = (
        "coarse_support_mask",
        "roi_bbox",
        "boundary_points",
        "boundary_normals",
        "boundary_tangent_u",
        "boundary_tangent_v",
        "boundary_strength",
        "boundary_material_id",
        "interior_points",
        "interior_density_seed",
        "interior_material_id",
    )
    missing = [key for key in required_keys if key not in analysis]
    if missing:
        raise RuntimeError(
            "The active intensity-driven CT training path requires a coarse-support Phase 1 bundle. "
            f"Missing keys: {', '.join(missing)}"
        )


def _sample_occupancy_points(candidate_indices, sample_count, spacing_zyx, device):
    if candidate_indices.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    count = int(sample_count)
    if isinstance(candidate_indices, torch.Tensor):
        candidate_indices = candidate_indices.to(device=device)
        candidate_count = int(candidate_indices.shape[0])
        replace = count > candidate_count
        selected = torch.randint(
            0,
            candidate_count,
            (count,),
            device=device,
        ) if replace else torch.randperm(candidate_count, device=device)[:count]
        voxel_indices = candidate_indices.index_select(0, selected).to(dtype=torch.float32)
        points_xyz = torch.stack(
            (
                (voxel_indices[:, 2] + 0.5) * float(spacing_zyx[2]),
                (voxel_indices[:, 1] + 0.5) * float(spacing_zyx[1]),
                (voxel_indices[:, 0] + 0.5) * float(spacing_zyx[0]),
            ),
            dim=1,
        )
        return points_xyz

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


def _density_to_signed_scalar(density: torch.Tensor) -> torch.Tensor:
    occupancy = density_to_occupancy(density)
    return occupancy.mul(2.0).sub(1.0)


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


def _save_ct_middle_slice_preview(
    gaussians,
    volume_cuda,
    spacing_zyx,
    model_path,
    backend,
    render_chunk_gaussians,
    slice_tile_size,
    compile_renderer,
    truncation_sigma,
    grid_cell_voxels,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Skipping CT middle-slice preview generation: {exc!r}", RuntimeWarning, stacklevel=2)
        return {}

    volume_shape = tuple(int(value) for value in volume_cuda.shape)
    if len(volume_shape) != 3 or volume_shape[0] < 1:
        return {}

    preview_axis = "z"
    slice_idx = int(volume_shape[0] // 2)
    patch_origin = (0, 0)
    patch_size = (int(volume_shape[1]), int(volume_shape[2]))
    training_state = prepare_ct_training_state(
        gaussians,
        spacing_zyx=spacing_zyx,
        truncation_sigma=truncation_sigma,
        grid_cell_voxels=grid_cell_voxels,
    )
    patch_renderer = build_ct_backend_patch_renderer(backend, compile_renderer=bool(compile_renderer))
    gt_patch = sample_gt_slice_patch(volume_cuda, preview_axis, slice_idx, patch_origin, patch_size).to(dtype=torch.float32)
    patch_grid_cache = CTPatchGridCache()

    with torch.no_grad():
        with torch.autocast(**_build_renderer_autocast_kwargs()):
            rendered_patch = patch_renderer(
                training_state.render_state,
                preview_axis,
                slice_idx,
                patch_origin,
                patch_size,
                spacing_zyx,
                volume_shape,
                gaussians_per_chunk=render_chunk_gaussians,
                patch_grid_cache=patch_grid_cache,
                slice_tile_size=slice_tile_size,
            )
        rendered_patch = rendered_patch.to(dtype=torch.float32).clamp(0.0, 1.0)
        abs_error = torch.abs(gt_patch - rendered_patch)
        gt_patch_bchw = gt_patch.unsqueeze(0).unsqueeze(0)
        rendered_patch_bchw = rendered_patch.unsqueeze(0).unsqueeze(0)
        mse = torch.mean((gt_patch_bchw - rendered_patch_bchw) ** 2)
        mae = float(abs_error.mean().item())
        rmse = float(torch.sqrt(mse).item())
        psnr = float((-10.0 * torch.log10(mse.clamp_min(1e-10))).item())
        ssim_value = float(ssim(rendered_patch_bchw, gt_patch_bchw).item())
        lpips_value = None
        lpips_model = _get_lpips_model(gt_patch.device)
        if lpips_model is not None:
            gt_patch_lpips = gt_patch_bchw.repeat(1, 3, 1, 1) * 2.0 - 1.0
            rendered_patch_lpips = rendered_patch_bchw.repeat(1, 3, 1, 1) * 2.0 - 1.0
            lpips_value = float(lpips_model(rendered_patch_lpips, gt_patch_lpips).mean().item())

    output_dir = Path(model_path).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"preview_{Path(model_path).name}"
    image_path = output_dir / f"{stem}.png"
    metrics_path = output_dir / f"{stem}.txt"

    gt_np = gt_patch.detach().cpu().numpy()
    rendered_np = rendered_patch.detach().cpu().numpy()
    error_np = abs_error.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(gt_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"GT z={slice_idx}")
    axes[0].axis("off")
    axes[1].imshow(rendered_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("CTGS Render")
    axes[1].axis("off")
    image = axes[2].imshow(error_np, cmap="magma", vmin=0.0, vmax=max(1e-6, float(error_np.max())))
    metric_lines = [f"MAE={mae:.4f} RMSE={rmse:.4f}", f"PSNR={psnr:.2f} SSIM={ssim_value:.4f}"]
    if lpips_value is not None:
        metric_lines.append(f"LPIPS={lpips_value:.4f}")
    axes[2].set_title("Abs Error\n" + "\n".join(metric_lines))
    axes[2].axis("off")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    metrics_path.write_text(
        "\n".join(
            [
                f"axis = {preview_axis}",
                f"slice_idx = {slice_idx}",
                f"mae = {mae:.4f}",
                f"rmse = {rmse:.4f}",
                f"psnr = {psnr:.4f}",
                f"ssim = {ssim_value:.6f}",
                f"lpips = {lpips_value:.6f}" if lpips_value is not None else "lpips = unavailable",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "preview_image": str(image_path),
        "preview_metrics": str(metrics_path),
    }


def training_ct(dataset, opt, saving_iterations, checkpoint_iterations, checkpoint, args):
    validate_ct_training_args(args)
    backend = resolve_ct_backend(args.ct_backend)
    if backend != "cuda":
        raise RuntimeError(
            f"The active intensity-driven CT training path requires the CUDA/native backend; resolved backend was '{backend}'."
        )

    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    loader = CTVolumeLoader()
    volume_np = loader.load(args.ct_volume_path, fmt=args.ct_volume_format, raw_meta_path=args.ct_raw_meta)
    spacing_zyx = loader.get_voxel_spacing()
    volume_cuda = torch.as_tensor(volume_np, dtype=torch.float32, device="cuda")
    volume_shape = tuple(int(value) for value in volume_np.shape)
    analysis, metadata, analysis_path, metadata_path = _load_ct_analysis_bundle(args.ct_phase1_dir)
    sigma = float(args.ct_gradient_sigma_voxels)
    analysis = _ensure_intensity_driven_analysis(
        analysis,
        volume_np,
        spacing_zyx,
        sigma=sigma,
        threshold_mode=args.ct_support_threshold_mode,
    )
    _require_active_boundary_bundle(analysis)
    analysis_gpu = _to_cuda_analysis(analysis)

    gaussians = CTGaussianModel(dataset.sh_degree)
    gaussians.create_from_phase1_bundle(
        analysis_path,
        metadata_path,
        spatial_lr_scale=_ct_spatial_extent(volume_np.shape, spacing_zyx),
        surface_thickness_max=args.surface_thickness_max,
        planar_thickness_max=args.planar_thickness_max,
        volume=volume_np,
        bulk_points_ratio=args.ct_bulk_points_ratio,
        bulk_boundary_margin_voxels=args.ct_bulk_boundary_margin_voxels,
    )

    gaussians.spatial_lr_scale = _ct_spatial_extent(volume_np.shape, spacing_zyx)
    opt.primitive_harden_iter = args.primitive_harden_iter
    opt.surface_thickness_max = args.surface_thickness_max
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
    neighbor_index = None
    bulk_neighbor_index = None
    total_computing_time = 0.0
    patch_grid_cache = CTPatchGridCache()
    ct_patch_renderer = build_ct_backend_patch_renderer(backend, compile_renderer=bool(args.ct_compile_renderer))
    renderer_autocast_kwargs = _build_renderer_autocast_kwargs()
    field_pools = _prepare_field_sample_pools(
        analysis_gpu,
        volume_shape,
        args.ct_bulk_boundary_margin_voxels,
        device="cuda",
    )
    support_sample_count, air_sample_count = _resolve_field_sample_counts(args, gaussians.get_xyz.shape[0])
    print(
        "CT field sampling budget: support={0} air={1} total_gaussians={2}".format(
            support_sample_count,
            air_sample_count,
            int(gaussians.get_xyz.shape[0]),
        )
    )
    analysis_gpu["_air_focus_indices"] = field_pools["air_shell"] if field_pools["air_shell"].shape[0] > 0 else field_pools["air"]
    support_distance_field = _prepare_support_distance_field(analysis, spacing_zyx, device="cuda")
    shape_thickness_max = args.ct_thickness_max
    if shape_thickness_max is None:
        shape_thickness_max = 2.0 * float(np.mean(np.asarray(spacing_zyx, dtype=np.float32)))
    _log_bulk_barrier_diagnostics(
        gaussians,
        support_distance_field,
        spacing_zyx=spacing_zyx,
        edt_alpha=args.ct_bulk_edt_alpha,
        bulk_max_scale=args.ct_bulk_max_scale,
    )
    _log_air_shell_diagnostics(field_pools)
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

    for iteration in range(first_iter + 1, opt.iterations + 1):
        tic = time.time()
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        bad_xyz_count = _sanitize_xyz_parameter(gaussians)
        if bad_xyz_count > 0:
            print(f"[WARN] Replaced {bad_xyz_count} non-finite Gaussian centers before neighbor refresh.")

        training_state = prepare_ct_training_state(
            gaussians,
            spacing_zyx=spacing_zyx,
            truncation_sigma=args.ct_gaussian_truncation_sigma,
            grid_cell_voxels=args.ct_grid_cell_voxels,
        )
        surface_xyz = training_state.surface_xyz
        bulk_mask = training_state.region_type == 1
        bulk_xyz = training_state.xyz[bulk_mask]
        if (
            args.ct_lambda_material != 0.0
            and (neighbor_index is None or iteration == first_iter + 1 or iteration % refresh_interval == 0)
        ):
            neighbor_index = _build_neighbor_index(
                surface_xyz,
                args.ct_material_k,
                backend=backend,
                tile_size=args.ct_knn_tile_size,
            )
        elif args.ct_lambda_material == 0.0:
            neighbor_index = None

        if (
            args.ct_lambda_bulk != 0.0
            and bulk_xyz.shape[0] > 1
            and (bulk_neighbor_index is None or iteration == first_iter + 1 or iteration % refresh_interval == 0)
        ):
            bulk_neighbor_index = _build_neighbor_index(
                bulk_xyz,
                args.ct_bulk_k,
                backend=backend,
                tile_size=args.ct_knn_tile_size,
            )
        elif args.ct_lambda_bulk == 0.0 or bulk_xyz.shape[0] <= 1:
            bulk_neighbor_index = None

        render_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        for _ in range(int(args.ct_slice_batch_size)):
            axis, slice_idx, patch_origin, patch_size = _sample_ct_patch_spec(
                analysis_gpu,
                volume_shape,
                args.ct_patch_size,
                spacing_zyx,
                cavity_patch_bias=args.ct_cavity_patch_bias,
            )
            gt_patch = sample_gt_slice_patch(volume_cuda, axis, slice_idx, patch_origin, patch_size).to(dtype=torch.float32)
            with torch.autocast(**renderer_autocast_kwargs):
                rendered_patch = ct_patch_renderer(
                    training_state.render_state,
                    axis,
                    slice_idx,
                    patch_origin,
                    patch_size,
                    spacing_zyx,
                    volume_shape,
                    gaussians_per_chunk=args.ct_render_chunk_gaussians,
                    patch_grid_cache=patch_grid_cache,
                    slice_tile_size=args.ct_slice_tile_size,
                )
                render_loss = render_loss + volume_rendering_loss(rendered_patch, gt_patch)
        render_loss = render_loss / float(args.ct_slice_batch_size)

        zero_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        occupancy_term = zero_loss
        if args.ct_lambda_occupancy != 0.0:
            support_points = _sample_occupancy_points(
                field_pools["support"],
                support_sample_count,
                spacing_zyx,
                device="cuda",
            )
            air_candidates = preferred_air_candidates if preferred_air_candidates.shape[0] > 0 else field_pools["air_shell"]
            if air_candidates.shape[0] == 0:
                air_candidates = field_pools["air"]
            air_points = _sample_occupancy_points(
                air_candidates,
                air_sample_count,
                spacing_zyx,
                device="cuda",
            )
            occupancy_points = torch.cat((support_points, air_points), dim=0)
            occupancy_target = torch.cat(
                (
                    torch.ones((support_points.shape[0],), dtype=torch.float32, device="cuda"),
                    torch.zeros((air_points.shape[0],), dtype=torch.float32, device="cuda"),
                ),
                dim=0,
            )
            field_weights = torch.cat(
                (
                    torch.ones((support_points.shape[0],), dtype=torch.float32, device="cuda"),
                    torch.ones((air_points.shape[0],), dtype=torch.float32, device="cuda"),
                ),
                dim=0,
            )
            occupancy_density = query_ct_density_from_state(
                backend,
                training_state,
                occupancy_points,
                chunk_size=args.ct_density_query_tile_points,
            )
            occupancy_pred = density_to_occupancy(occupancy_density)
            occupancy_term = occupancy_loss(occupancy_pred, occupancy_target, sample_weights=field_weights)

        shape_term = zero_loss
        if args.ct_lambda_shape != 0.0 and training_state.surface_xyz.shape[0] > 0:
            shape_term = surface_shape_loss(
                training_state.surface_raw_scaling,
                training_state.surface_rotation_mats,
                training_state.surface_normals,
                training_state.surface_opacity,
                max_thickness=shape_thickness_max,
                max_tangential_scale=args.ct_tangential_max_scale,
                min_opacity=args.ct_surface_min_opacity,
            )

        bulk_term = zero_loss
        if args.ct_lambda_bulk != 0.0 and torch.any(bulk_mask):
            bulk_scale_limit = float(args.ct_bulk_max_scale)
            sampled_bulk_distance = sample_volume_field(
                support_distance_field["support_distance"],
                bulk_xyz,
                support_distance_field["spacing_zyx"],
            ).reshape(-1)
            bulk_scale_limit = torch.clamp(
                sampled_bulk_distance * float(args.ct_bulk_edt_alpha),
                min=min(spacing_zyx),
                max=float(args.ct_bulk_max_scale),
            )
            bulk_term = bulk_regularization_loss(
                bulk_xyz,
                training_state.raw_scaling[bulk_mask],
                training_state.rotation_mats[bulk_mask],
                training_state.opacity[bulk_mask],
                neighbor_index=bulk_neighbor_index,
                max_bulk_scale=bulk_scale_limit,
                density_cap=args.ct_bulk_density_cap,
                k=args.ct_bulk_k,
            )

        material_loss = zero_loss
        if args.ct_lambda_material != 0.0 and training_state.surface_xyz.shape[0] > 0:
            material_loss = material_boundary_compact_loss(
                training_state.surface_xyz,
                training_state.surface_normals,
                training_state.surface_material_id,
                training_state.surface_opacity,
                neighbor_index=neighbor_index,
                k=args.ct_material_k,
                min_opacity=args.ct_surface_min_opacity,
                align_weight=args.ct_align_weight,
                cross_floor_weight=args.ct_cross_floor_weight,
            )

        loss = (
            args.ct_lambda_render * render_loss
            + args.ct_lambda_occupancy * occupancy_term
            + args.ct_lambda_shape * shape_term
            + args.ct_lambda_material * material_loss
            + args.ct_lambda_bulk * bulk_term
        )
        loss.backward()
        if args.ct_freeze_bulk_xyz:
            _freeze_bulk_xyz_gradients(gaussians)

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
                tb_writer.add_scalar("ct_loss/render", render_loss.item(), iteration)
                tb_writer.add_scalar("ct_loss/occupancy", occupancy_term.item(), iteration)
                tb_writer.add_scalar("ct_loss/shape", shape_term.item(), iteration)
                tb_writer.add_scalar("ct_loss/material", material_loss.item(), iteration)
                tb_writer.add_scalar("ct_loss/bulk", bulk_term.item(), iteration)
                tb_writer.add_scalar("ct_loss/total", loss.item(), iteration)
                tb_writer.add_scalar("iter_time", elapsed, iteration)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                _save_ct_gaussians(gaussians, dataset.model_path, iteration)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.post_optimizer_step(iteration)
                _apply_bulk_scale_hard_projection(
                    gaussians,
                    support_distance_field,
                    spacing_zyx=spacing_zyx,
                    edt_alpha=args.ct_bulk_edt_alpha,
                    bulk_max_scale=args.ct_bulk_max_scale,
                )
                _clear_bulk_scaling_optimizer_momentum(gaussians)
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), dataset.model_path + "/chkpnt" + str(iteration) + ".pth")

        if args.wandb and wandb is not None:
            wandb_logs = {
                "loss": loss.item(),
                "render_loss": render_loss.item(),
                "occupancy_loss": occupancy_term.item(),
                "shape_loss": shape_term.item(),
                "material_loss": material_loss.item(),
                "bulk_loss": bulk_term.item(),
                "t": total_computing_time,
                "num_gaussian": len(gaussians.get_xyz),
            }
            gpu_memory = _ct_log_gpu_memory()
            if gpu_memory is not None:
                wandb_logs["gpu"] = gpu_memory
            wandb.log(wandb_logs, commit=True)

    exports = _export_ct_outputs(gaussians, args)
    if getattr(args, "ct_auto_preview", True):
        exports.update(
            _save_ct_middle_slice_preview(
                gaussians,
                volume_cuda,
                spacing_zyx,
                dataset.model_path,
                backend=backend,
                render_chunk_gaussians=args.ct_render_chunk_gaussians,
                slice_tile_size=args.ct_slice_tile_size,
                compile_renderer=args.ct_compile_renderer,
                truncation_sigma=args.ct_gaussian_truncation_sigma,
                grid_cell_voxels=args.ct_grid_cell_voxels,
            )
        )
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
    parser.add_argument("--ct_material_k", type=int, default=None)
    parser.add_argument("--ct_neighbor_refresh_interval", type=int, default=100)
    parser.add_argument("--ct_backend", type=str, default="auto", choices=["auto", "cuda", "python"])
    parser.add_argument("--ct_render_chunk_gaussians", type=int, default=2048)
    parser.add_argument("--ct_compile_renderer", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ct_bulk_points_ratio", type=float, default=1.0)
    parser.add_argument("--ct_bulk_boundary_margin_voxels", type=int, default=2)
    parser.add_argument("--ct_support_threshold_mode", type=str, default="otsu", choices=["otsu", "multi_otsu"])
    parser.add_argument("--ct_support_sample_count", type=int, default=None)
    parser.add_argument("--ct_air_sample_count", type=int, default=None)
    parser.add_argument("--ct_gradient_sigma_voxels", type=float, default=1.0)
    parser.add_argument("--ct_material_query_count", type=int, default=None)
    parser.add_argument("--ct_void_query_count", type=int, default=4096)
    parser.add_argument("--ct_interior_query_count", type=int, default=4096)
    parser.add_argument("--ct_exterior_query_count", type=int, default=4096)
    parser.add_argument("--ct_max_material_classes", type=int, default=3)
    parser.add_argument("--ct_void_negative_weight", type=float, default=4.0)
    parser.add_argument("--ct_cavity_patch_bias", type=float, default=0.6)
    parser.add_argument("--ct_void_boundary_offset_scale", type=float, default=0.75)
    parser.add_argument("--ct_void_boundary_margin", type=float, default=0.25)
    parser.add_argument("--ct_signed_field_band_voxels", type=int, default=3)
    parser.add_argument("--ct_density_query_tile_points", type=int, default=8192)
    parser.add_argument("--ct_knn_tile_size", type=int, default=2048)
    parser.add_argument("--ct_gaussian_truncation_sigma", type=float, default=4.0)
    parser.add_argument("--ct_slice_tile_size", type=int, default=8)
    parser.add_argument("--ct_grid_cell_voxels", type=int, default=8)
    parser.add_argument("--ct_freeze_bulk_xyz", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ct_bulk_edt_alpha", type=float, default=1.0)
    parser.add_argument("--ct_auto_preview", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ct_lambda_render", type=float, default=None)
    parser.add_argument("--ct_lambda_slice", type=float, default=1.0)
    parser.add_argument("--ct_lambda_field_recon", type=float, default=None)
    parser.add_argument("--ct_lambda_boundary_ridge", type=float, default=None)
    parser.add_argument("--ct_lambda_occupancy", type=float, default=0.5)
    parser.add_argument("--ct_lambda_shape", type=float, default=None)
    parser.add_argument("--ct_lambda_bulk", type=float, default=None)
    parser.add_argument("--ct_lambda_boundary_center", type=float, default=None)
    parser.add_argument("--ct_lambda_boundary_normal", type=float, default=None)
    parser.add_argument("--ct_lambda_signed_surface", type=float, default=None)
    parser.add_argument("--ct_lambda_void_boundary", type=float, default=0.15)
    parser.add_argument("--ct_lambda_surface_thickness", type=float, default=None)
    parser.add_argument("--ct_lambda_surface_tangential_scale", type=float, default=None)
    parser.add_argument("--ct_lambda_surface_opacity", type=float, default=None)
    parser.add_argument("--ct_lambda_bulk_scale", type=float, default=None)
    parser.add_argument("--ct_lambda_bulk_overlap", type=float, default=None)
    parser.add_argument("--ct_lambda_plane", type=float, default=0.2)
    parser.add_argument("--ct_lambda_normal", type=float, default=0.1)
    parser.add_argument("--ct_lambda_thickness", type=float, default=None)
    parser.add_argument("--ct_lambda_material", type=float, default=0.1)
    parser.add_argument("--ct_align_weight", type=float, default=0.3)
    parser.add_argument("--ct_cross_floor_weight", type=float, default=0.2)
    parser.add_argument("--ct_thickness_max", type=float, default=None)
    parser.add_argument("--ct_tangential_max_scale", type=float, default=4.0)
    parser.add_argument("--ct_surface_tangential_max_scale", type=float, default=4.0)
    parser.add_argument("--ct_surface_min_opacity", type=float, default=0.8)
    parser.add_argument("--ct_surface_target_opacity", type=float, default=0.9)
    parser.add_argument("--ct_bulk_max_scale", type=float, default=4.0)
    parser.add_argument("--ct_bulk_density_cap", type=float, default=3.0)
    parser.add_argument("--ct_bulk_k", type=int, default=None)
    parser.add_argument("--ct_bulk_overlap_k", type=int, default=None)
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
