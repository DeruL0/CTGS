from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


def _as_numpy_3vector(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(3)
    return array


def _clone_tensor(value):
    if not isinstance(value, torch.Tensor):
        return value
    cloned = value.detach().clone()
    if isinstance(value, nn.Parameter):
        return nn.Parameter(cloned.requires_grad_(value.requires_grad))
    return cloned


def _instantiate_like(model):
    model_type = type(model)
    try:
        if hasattr(model, "max_sh_degree"):
            return model_type(model.max_sh_degree)
        return model_type()
    except TypeError:
        clone = model_type.__new__(model_type)
        if hasattr(clone, "setup_functions"):
            clone.setup_functions()
        return clone


def clone_gaussian_like(model):
    clone = _instantiate_like(model)
    tensor_attrs = [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "_primitive_type",
        "_normal",
        "_material_id",
        "_planarity",
        "_region_type",
        "max_radii2D",
        "xyz_gradient_accum",
        "denom",
    ]
    scalar_attrs = [
        "active_sh_degree",
        "max_sh_degree",
        "percent_dense",
        "spatial_lr_scale",
        "pose_lr_joint",
        "primitive_harden_iter",
        "primitive_types_hardened",
        "planar_thickness_max",
        "planar_logit_value",
        "nonplanar_logit_value",
        "single_material_fallback",
    ]

    for attr in tensor_attrs:
        if hasattr(model, attr):
            setattr(clone, attr, _clone_tensor(getattr(model, attr)))
    for attr in scalar_attrs:
        if hasattr(model, attr):
            setattr(clone, attr, getattr(model, attr))

    if hasattr(clone, "optimizer"):
        clone.optimizer = None
    return clone


def _prune_gaussian_like(model, keep_mask: torch.Tensor):
    keep_mask = torch.as_tensor(keep_mask, dtype=torch.bool)
    clone = clone_gaussian_like(model)

    for attr in [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "_primitive_type",
        "_normal",
        "_material_id",
        "_planarity",
        "_region_type",
        "max_radii2D",
        "xyz_gradient_accum",
        "denom",
    ]:
        if not hasattr(clone, attr):
            continue
        tensor = getattr(clone, attr)
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 0 or tensor.shape[0] != keep_mask.shape[0]:
            continue
        sliced = tensor[keep_mask.to(device=tensor.device)]
        if isinstance(tensor, nn.Parameter):
            sliced = nn.Parameter(sliced.requires_grad_(tensor.requires_grad))
        setattr(clone, attr, sliced)

    return clone


def _weighted_quaternion_average(quaternions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    reference = quaternions[0:1]
    aligned = quaternions.clone()
    signs = torch.sign(torch.sum(aligned * reference, dim=1, keepdim=True))
    signs[signs == 0] = 1.0
    aligned = aligned * signs
    averaged = torch.sum(aligned * weights[:, None], dim=0)
    norm = torch.linalg.norm(averaged)
    if norm <= 1e-8:
        averaged = reference[0]
    else:
        averaged = averaged / norm
    return averaged


def _mode_long(values: torch.Tensor) -> torch.Tensor:
    unique, counts = torch.unique(values, return_counts=True)
    return unique[counts.argmax()]


def _merge_gaussian_like(model, merge_radius: float):
    xyz = model.get_xyz.detach()
    if xyz.shape[0] <= 1 or merge_radius <= 0:
        return clone_gaussian_like(model)

    device = xyz.device
    coords = torch.floor((xyz - xyz.min(dim=0).values) / float(merge_radius)).to(dtype=torch.long)
    _, inverse = torch.unique(coords, dim=0, return_inverse=True)
    num_clusters = int(inverse.max().item()) + 1

    clone = _instantiate_like(model)
    raw_opacity = getattr(model, "_opacity").detach()
    weights_base = torch.sigmoid(raw_opacity).reshape(-1).clamp_min(1e-6)

    xyz_list = []
    fdc_list = []
    frest_list = []
    scaling_list = []
    rotation_list = []
    opacity_list = []
    primitive_list = []
    normal_list = []
    material_list = []
    planarity_list = []
    region_list = []

    for cluster_index in range(num_clusters):
        cluster_mask = inverse == cluster_index
        cluster_weights = weights_base[cluster_mask]
        normalized_weights = cluster_weights / cluster_weights.sum().clamp_min(1e-8)

        xyz_list.append(torch.sum(getattr(model, "_xyz").detach()[cluster_mask] * normalized_weights[:, None], dim=0))
        fdc_list.append(torch.sum(getattr(model, "_features_dc").detach()[cluster_mask] * normalized_weights[:, None, None], dim=0))
        frest_list.append(torch.sum(getattr(model, "_features_rest").detach()[cluster_mask] * normalized_weights[:, None, None], dim=0))
        scaling_list.append(torch.sum(getattr(model, "_scaling").detach()[cluster_mask] * normalized_weights[:, None], dim=0))
        opacity_list.append(torch.sum(getattr(model, "_opacity").detach()[cluster_mask] * normalized_weights[:, None], dim=0))
        primitive_list.append(torch.sum(getattr(model, "_primitive_type").detach()[cluster_mask] * normalized_weights[:, None], dim=0))

        normals = getattr(model, "_normal").detach()[cluster_mask]
        averaged_normal = torch.sum(normals * normalized_weights[:, None], dim=0)
        normal_norm = torch.linalg.norm(averaged_normal).clamp_min(1e-8)
        normal_list.append(averaged_normal / normal_norm)

        rotations = getattr(model, "_rotation").detach()[cluster_mask]
        rotation_list.append(_weighted_quaternion_average(rotations, normalized_weights))

        material_values = getattr(model, "_material_id").detach()[cluster_mask, 0]
        material_list.append(_mode_long(material_values))
        planarity_list.append(torch.sum(getattr(model, "_planarity").detach()[cluster_mask, 0] * normalized_weights, dim=0))
        region_values = getattr(model, "_region_type").detach()[cluster_mask, 0]
        region_list.append(_mode_long(region_values))

    def _stack_parameter(values):
        tensor = torch.stack(values, dim=0).to(device=device)
        return nn.Parameter(tensor.requires_grad_(True))

    clone._xyz = _stack_parameter(xyz_list)
    clone._features_dc = _stack_parameter(fdc_list)
    clone._features_rest = _stack_parameter(frest_list)
    clone._scaling = _stack_parameter(scaling_list)
    clone._rotation = _stack_parameter(rotation_list)
    clone._opacity = _stack_parameter(opacity_list)
    clone._primitive_type = _stack_parameter(primitive_list)
    clone._normal = _stack_parameter(normal_list)
    clone._material_id = torch.stack(material_list, dim=0).reshape(-1, 1).to(device=device, dtype=torch.long)
    clone._planarity = torch.stack(planarity_list, dim=0).reshape(-1, 1).to(device=device, dtype=torch.float32)
    clone._region_type = torch.stack(region_list, dim=0).reshape(-1, 1).to(device=device, dtype=torch.long)

    if hasattr(model, "max_sh_degree"):
        clone.max_sh_degree = model.max_sh_degree
    if hasattr(model, "active_sh_degree"):
        clone.active_sh_degree = model.active_sh_degree
    if hasattr(model, "primitive_harden_iter"):
        clone.primitive_harden_iter = model.primitive_harden_iter
    if hasattr(model, "primitive_types_hardened"):
        clone.primitive_types_hardened = model.primitive_types_hardened
    if hasattr(model, "planar_thickness_max"):
        clone.planar_thickness_max = model.planar_thickness_max
    if hasattr(model, "single_material_fallback"):
        clone.single_material_fallback = getattr(model, "single_material_fallback")

    clone.max_radii2D = torch.zeros((clone._xyz.shape[0],), dtype=torch.float32, device=device)
    clone.xyz_gradient_accum = torch.zeros((clone._xyz.shape[0], 1), dtype=torch.float32, device=device)
    clone.denom = torch.zeros((clone._xyz.shape[0], 1), dtype=torch.float32, device=device)
    clone.optimizer = None
    return clone


@dataclass
class OccupancyHitResult:
    ray_hit_mask: np.ndarray
    ray_entry_t: np.ndarray


class OccupancyGrid:
    """Macroblock-level occupancy for fast empty-space skipping."""

    def __init__(self, bbox, block_size=8):
        bbox = np.asarray(bbox, dtype=np.float32)
        if bbox.shape == (2, 3):
            self.bbox_min = bbox[0]
            self.bbox_max = bbox[1]
        else:
            raise ValueError("bbox must have shape (2, 3).")

        if np.any(self.bbox_max <= self.bbox_min):
            raise ValueError("bbox max must be greater than bbox min on all axes.")

        block_size = np.asarray(block_size, dtype=np.float32)
        if block_size.ndim == 0:
            block_size = np.repeat(block_size, 3)
        if block_size.shape != (3,) or np.any(block_size <= 0):
            raise ValueError("block_size must be a positive scalar or a length-3 sequence.")

        self.block_size = block_size
        self.grid_shape = np.ceil((self.bbox_max - self.bbox_min) / self.block_size).astype(np.int32)
        self.occupancy = np.zeros(tuple(self.grid_shape.tolist()), dtype=bool)

    def update(self, gaussians):
        self.occupancy.fill(False)
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        if xyz.size == 0:
            return self.occupancy

        scales = gaussians.get_scaling.detach().cpu().numpy()
        radius = np.max(scales, axis=1, keepdims=True)
        lower = np.floor((xyz - radius - self.bbox_min) / self.block_size).astype(np.int32)
        upper = np.floor((xyz + radius - self.bbox_min) / self.block_size).astype(np.int32)
        lower = np.clip(lower, 0, self.grid_shape - 1)
        upper = np.clip(upper, 0, self.grid_shape - 1)

        for lo, hi in zip(lower, upper):
            self.occupancy[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = True
        return self.occupancy

    def query(self, ray_origins, ray_dirs):
        origin_is_tensor = isinstance(ray_origins, torch.Tensor)
        device = ray_origins.device if origin_is_tensor else None
        origins = np.asarray(ray_origins.detach().cpu().numpy() if origin_is_tensor else ray_origins, dtype=np.float32)
        directions = np.asarray(ray_dirs.detach().cpu().numpy() if isinstance(ray_dirs, torch.Tensor) else ray_dirs, dtype=np.float32)
        if origins.ndim == 1:
            origins = origins[None, :]
        if directions.ndim == 1:
            directions = directions[None, :]
        if origins.shape != directions.shape or origins.shape[1] != 3:
            raise ValueError("ray_origins and ray_dirs must both have shape (N, 3).")

        hit_mask = np.zeros((origins.shape[0],), dtype=bool)
        eps = 1e-8
        grid_max = self.bbox_min + self.grid_shape * self.block_size

        for index, (origin, direction) in enumerate(zip(origins, directions)):
            direction = direction.copy()
            direction[np.abs(direction) < eps] = 0.0

            tmin = -np.inf
            tmax = np.inf
            valid_ray = True
            for axis in range(3):
                if abs(direction[axis]) < eps:
                    if origin[axis] < self.bbox_min[axis] or origin[axis] > grid_max[axis]:
                        valid_ray = False
                        break
                    continue
                inv_dir = 1.0 / direction[axis]
                axis_t0 = (self.bbox_min[axis] - origin[axis]) * inv_dir
                axis_t1 = (grid_max[axis] - origin[axis]) * inv_dir
                axis_tmin = min(axis_t0, axis_t1)
                axis_tmax = max(axis_t0, axis_t1)
                tmin = max(tmin, axis_tmin)
                tmax = min(tmax, axis_tmax)
            if (not valid_ray) or (not np.isfinite(tmax)) or tmax < max(tmin, 0.0):
                continue

            t = max(tmin, 0.0)
            point = origin + t * direction
            cell = np.floor((point - self.bbox_min) / self.block_size).astype(np.int32)
            cell = np.clip(cell, 0, self.grid_shape - 1)

            step = np.sign(direction).astype(np.int32)
            next_boundary = self.bbox_min + (cell + (step > 0)) * self.block_size
            t_max = np.full((3,), np.inf, dtype=np.float32)
            t_delta = np.full((3,), np.inf, dtype=np.float32)
            for axis in range(3):
                if abs(direction[axis]) >= eps:
                    t_max[axis] = (next_boundary[axis] - origin[axis]) / direction[axis]
                    t_delta[axis] = self.block_size[axis] / abs(direction[axis])

            while np.all(cell >= 0) and np.all(cell < self.grid_shape) and t <= tmax:
                if self.occupancy[cell[0], cell[1], cell[2]]:
                    hit_mask[index] = True
                    break
                axis = int(np.argmin(t_max))
                cell[axis] += step[axis]
                t = t_max[axis]
                t_max[axis] += t_delta[axis]

        if origin_is_tensor:
            return torch.as_tensor(hit_mask, dtype=torch.bool, device=device)
        return hit_mask


class ClipPlaneManager:
    """Interactive clip plane for CT slice inspection."""

    def __init__(self):
        self.planes: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def add_plane(self, normal, offset):
        normal_tensor = torch.as_tensor(normal, dtype=torch.float32).reshape(3)
        normal_norm = torch.linalg.norm(normal_tensor).clamp_min(1e-8)
        normal_tensor = normal_tensor / normal_norm
        offset_tensor = torch.as_tensor(offset, dtype=torch.float32).reshape(())
        self.planes.append((normal_tensor, offset_tensor))

    def clip_gaussians(self, gaussians):
        xyz = gaussians.get_xyz
        if xyz.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=xyz.device)

        visibility_mask = torch.ones((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
        for normal, offset in self.planes:
            signed_distance = torch.matmul(xyz, normal.to(device=xyz.device, dtype=xyz.dtype))
            visibility_mask &= signed_distance >= offset.to(device=xyz.device, dtype=xyz.dtype)
        return visibility_mask


class LODManager:
    """Multi-level LOD based on camera distance."""

    def __init__(self, levels=3):
        if int(levels) < 1:
            raise ValueError("levels must be >= 1.")
        self.levels = int(levels)
        self.lod_models = []
        self.distance_thresholds = []

    def build_lod(self, full_model):
        if full_model.get_xyz.shape[0] == 0:
            self.lod_models = [clone_gaussian_like(full_model)]
            self.distance_thresholds = []
            return self.lod_models

        base_model = clone_gaussian_like(full_model)
        self.lod_models = [base_model]

        xyz = full_model.get_xyz.detach()
        bbox_min = xyz.min(dim=0).values
        bbox_max = xyz.max(dim=0).values
        diagonal = torch.linalg.norm(bbox_max - bbox_min).item()

        current = base_model
        median_scale = torch.median(current.get_scaling.max(dim=1).values).item()
        for level in range(1, self.levels):
            contribution = (current.get_opacity.squeeze(-1) * current.get_scaling.max(dim=1).values).detach()
            quantile = min(0.2 + 0.2 * level, 0.8)
            threshold = torch.quantile(contribution, quantile).item()
            keep_mask = contribution >= threshold
            if int(keep_mask.sum().item()) == 0:
                keep_mask[contribution.argmax()] = True
            reduced = _prune_gaussian_like(current, keep_mask)
            merge_radius = median_scale * (0.5 + 0.5 * level)
            current = _merge_gaussian_like(reduced, merge_radius=max(merge_radius, 1e-4))
            self.lod_models.append(current)

        self.distance_thresholds = [diagonal * (1.5 + level) for level in range(max(self.levels - 1, 0))]
        return self.lod_models

    def select_lod(self, camera_distance) -> int:
        if not self.lod_models:
            raise RuntimeError("LOD models have not been built yet.")
        camera_distance = float(camera_distance)
        for index, threshold in enumerate(self.distance_thresholds):
            if camera_distance < threshold:
                return index
        return len(self.lod_models) - 1
