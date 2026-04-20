from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.loss_utils import ssim, weighted_l1_loss
from utils.rotation_utils import quaternion_to_matrix


@dataclass
class PointToPlaneCache:
    active_indices: torch.Tensor
    centroids: torch.Tensor
    fitted_normals: torch.Tensor
    weights: torch.Tensor

    @property
    def is_empty(self) -> bool:
        return self.active_indices.numel() == 0


def _as_slice_batch(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError("Slice tensors must have shape (H, W), (1, H, W), or (B, 1, H, W).")


def _primitive_mask(primitive_type, count: int, device: torch.device) -> torch.Tensor:
    if primitive_type is None:
        return torch.ones((count,), dtype=torch.bool, device=device)
    primitive_type = torch.as_tensor(primitive_type, device=device).reshape(count, -1)
    values = primitive_type[:, 0]
    if values.dtype == torch.bool:
        return values
    if values.is_floating_point():
        if torch.any((values < 0.0) | (values > 1.0)):
            return torch.sigmoid(values) >= 0.5
        return values >= 0.5
    return values != 0


def _resolve_neighbor_index(xyz: torch.Tensor, neighbor_index, k: int) -> torch.Tensor:
    count = xyz.shape[0]
    if count == 0:
        return torch.empty((0, 0), dtype=torch.long, device=xyz.device)
    if neighbor_index is not None:
        return torch.as_tensor(neighbor_index, dtype=torch.long, device=xyz.device)
    if count <= 1 or k <= 0:
        return torch.empty((count, 0), dtype=torch.long, device=xyz.device)

    k = min(k, count - 1)
    distances = torch.cdist(xyz, xyz)
    neighbor_index = torch.argsort(distances, dim=1)[:, 1 : k + 1]
    return neighbor_index


def volume_rendering_loss(rendered_slice, gt_slice, weight_map=None):
    rendered = _as_slice_batch(rendered_slice)
    target = _as_slice_batch(gt_slice).to(device=rendered.device, dtype=rendered.dtype)
    weight = None if weight_map is None else _as_slice_batch(weight_map).to(device=rendered.device, dtype=rendered.dtype)
    l1_term = weighted_l1_loss(rendered, target, weight)
    ssim_term = 1.0 - ssim(rendered, target)
    return 0.8 * l1_term + 0.2 * ssim_term


def occupancy_loss(pred_occupancy, target_occupancy, sample_weights=None):
    pred = torch.as_tensor(pred_occupancy)
    target = torch.as_tensor(target_occupancy, device=pred.device, dtype=pred.dtype)
    if pred.numel() == 0:
        return torch.zeros((), dtype=pred.dtype if pred.numel() > 0 else torch.float32, device=pred.device)

    pred = pred.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    target = target.reshape(-1)
    if sample_weights is not None:
        sample_weights = torch.as_tensor(sample_weights, device=pred.device, dtype=pred.dtype).reshape(-1)
        if sample_weights.shape[0] != pred.shape[0]:
            raise ValueError("sample_weights must match pred_occupancy length.")
        sample_weights = sample_weights.clamp_min(0.0)
    positive = target >= 0.5
    negative = ~positive

    if torch.any(positive) and torch.any(negative):
        weights = torch.zeros_like(pred)
        weights[positive] = 0.5 / positive.sum().clamp_min(1)
        weights[negative] = 0.5 / negative.sum().clamp_min(1)
        if sample_weights is not None:
            weights = weights * sample_weights
            weights = weights / weights.sum().clamp_min(1e-8)
        return F.binary_cross_entropy(pred, target, weight=weights, reduction="sum")
    if sample_weights is not None:
        weights = sample_weights / sample_weights.sum().clamp_min(1e-8)
        return F.binary_cross_entropy(pred, target, weight=weights, reduction="sum")
    return F.binary_cross_entropy(pred, target)


def continuous_field_loss(pred_field, target_field, sample_weights=None, delta: float = 0.1):
    pred = torch.as_tensor(pred_field)
    target = torch.as_tensor(target_field, device=pred.device, dtype=pred.dtype)
    if pred.numel() == 0:
        return torch.zeros((), dtype=pred.dtype if pred.numel() > 0 else torch.float32, device=pred.device)

    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.shape[0] != target.shape[0]:
        raise ValueError("pred_field and target_field must share the same number of samples.")

    losses = F.smooth_l1_loss(pred, target, beta=float(delta), reduction="none")
    if sample_weights is None:
        return losses.mean()

    weights = torch.as_tensor(sample_weights, device=pred.device, dtype=pred.dtype).reshape(-1).clamp_min(0.0)
    if weights.shape[0] != pred.shape[0]:
        raise ValueError("sample_weights must match pred_field length.")
    valid = torch.isfinite(losses) & torch.isfinite(weights) & (weights > 0.0)
    if not torch.any(valid):
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    return (losses[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-8)


def sample_volume_field(volume_field: torch.Tensor, points_xyz: torch.Tensor, spacing_zyx):
    points_xyz = torch.as_tensor(points_xyz)
    if points_xyz.numel() == 0:
        channel_count = int(volume_field.shape[1]) if volume_field.ndim == 5 else 1
        return torch.empty((0, channel_count), dtype=points_xyz.dtype if points_xyz.numel() > 0 else torch.float32, device=points_xyz.device)

    if volume_field.ndim != 5:
        raise ValueError("volume_field must have shape (1, C, D, H, W).")

    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
    depth, height, width = [int(value) for value in volume_field.shape[-3:]]
    dtype = volume_field.dtype
    device = volume_field.device
    points_xyz = points_xyz.to(device=device, dtype=dtype)

    x_idx = points_xyz[:, 0] / max(spacing_x, 1e-8)
    y_idx = points_xyz[:, 1] / max(spacing_y, 1e-8)
    z_idx = points_xyz[:, 2] / max(spacing_z, 1e-8)
    x_norm = 2.0 * (x_idx / max(width - 1, 1)) - 1.0
    y_norm = 2.0 * (y_idx / max(height - 1, 1)) - 1.0
    z_norm = 2.0 * (z_idx / max(depth - 1, 1)) - 1.0
    grid = torch.stack((x_norm, y_norm, z_norm), dim=-1).reshape(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        volume_field,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(volume_field.shape[1], -1).transpose(0, 1)


def boundary_center_loss(surface_xyz, boundary_strength_values):
    surface_xyz = torch.as_tensor(surface_xyz)
    if surface_xyz.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=surface_xyz.device)
    strength = torch.as_tensor(boundary_strength_values, device=surface_xyz.device, dtype=surface_xyz.dtype).reshape(-1)
    if strength.numel() == 0:
        return torch.zeros((), dtype=surface_xyz.dtype, device=surface_xyz.device)
    strength = strength.clamp(0.0, 1.0)
    return (1.0 - strength).mean()


def boundary_normal_loss(surface_normals, boundary_normals, boundary_strength=None):
    surface_normals = F.normalize(torch.as_tensor(surface_normals), dim=-1)
    if surface_normals.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=surface_normals.device)

    boundary_normals = torch.as_tensor(boundary_normals, device=surface_normals.device, dtype=surface_normals.dtype)
    if boundary_normals.ndim != 2 or boundary_normals.shape[1] != 3:
        raise ValueError("boundary_normals must have shape (N, 3).")

    norm = torch.linalg.norm(boundary_normals, dim=-1)
    valid = torch.isfinite(norm) & (norm > 1e-6)
    if not torch.any(valid):
        return torch.zeros((), dtype=surface_normals.dtype, device=surface_normals.device)

    boundary_normals = boundary_normals.clone()
    boundary_normals[valid] = F.normalize(boundary_normals[valid], dim=-1)
    cosine = torch.abs(torch.sum(surface_normals * boundary_normals, dim=-1))
    losses = 1.0 - cosine
    if boundary_strength is not None:
        weights = torch.as_tensor(boundary_strength, device=surface_normals.device, dtype=surface_normals.dtype).reshape(-1).clamp_min(0.0)
        valid = valid & torch.isfinite(weights) & (weights > 0.0)
        if not torch.any(valid):
            return torch.zeros((), dtype=surface_normals.dtype, device=surface_normals.device)
        return (losses[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-8)
    return losses[valid].mean()


def signed_surface_loss(
    surface_xyz,
    surface_normals,
    signed_values,
    signed_gradients,
    boundary_strength=None,
    band_width_voxels: float = 3.0,
):
    del surface_xyz
    surface_normals = torch.as_tensor(surface_normals)
    if surface_normals.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=surface_normals.device)

    device = surface_normals.device
    dtype = surface_normals.dtype
    signed_values = torch.as_tensor(signed_values, device=device, dtype=dtype).reshape(-1)
    signed_gradients = torch.as_tensor(signed_gradients, device=device, dtype=dtype).reshape(-1, 3)
    if signed_values.shape[0] != surface_normals.shape[0] or signed_gradients.shape[0] != surface_normals.shape[0]:
        raise ValueError("signed_surface_loss inputs must share the same leading dimension.")

    surface_normals = F.normalize(surface_normals, dim=-1)
    gradient_norm = torch.linalg.norm(signed_gradients, dim=-1)
    valid = torch.isfinite(signed_values) & torch.isfinite(gradient_norm) & (gradient_norm > 1e-6)
    if not torch.any(valid):
        return torch.zeros((), dtype=dtype, device=device)

    normalized_band = max(float(band_width_voxels), 1e-6)
    value_term = F.smooth_l1_loss(
        (signed_values[valid] / normalized_band).clamp(-1.0, 1.0),
        torch.zeros_like(signed_values[valid]),
        reduction="none",
    )
    gradient_unit = F.normalize(signed_gradients[valid], dim=-1)
    direction_term = 1.0 - torch.abs(torch.sum(surface_normals[valid] * gradient_unit, dim=-1))
    combined = value_term + direction_term

    if boundary_strength is not None:
        weights = torch.as_tensor(boundary_strength, device=device, dtype=dtype).reshape(-1).clamp_min(0.0)
        valid = valid & torch.isfinite(weights) & (weights > 0.0)
        if not torch.any(valid):
            return torch.zeros((), dtype=dtype, device=device)
        normalized_values = (signed_values[valid] / normalized_band).clamp(-1.0, 1.0)
        value_term = F.smooth_l1_loss(
            normalized_values,
            torch.zeros_like(normalized_values),
            reduction="none",
        )
        gradient_unit = F.normalize(signed_gradients[valid], dim=-1)
        direction_term = 1.0 - torch.abs(torch.sum(surface_normals[valid] * gradient_unit, dim=-1))
        combined = (value_term + direction_term) * weights[valid]
        return combined.sum() / weights[valid].sum().clamp_min(1e-8)

    return combined.mean()


def boundary_ridge_alignment_loss(
    surface_normals,
    ridge_directional_derivative,
    ridge_strength,
    ridge_normals,
    ridge_strength_floor: float = 0.2,
    ridge_direction_delta: float = 0.1,
):
    surface_normals = torch.as_tensor(surface_normals)
    if surface_normals.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=surface_normals.device)

    device = surface_normals.device
    dtype = surface_normals.dtype
    ridge_directional_derivative = torch.as_tensor(ridge_directional_derivative, device=device, dtype=dtype).reshape(-1)
    ridge_strength = torch.as_tensor(ridge_strength, device=device, dtype=dtype).reshape(-1)
    ridge_normals = torch.as_tensor(ridge_normals, device=device, dtype=dtype).reshape(-1, 3)

    if (
        ridge_directional_derivative.shape[0] != surface_normals.shape[0]
        or ridge_strength.shape[0] != surface_normals.shape[0]
        or ridge_normals.shape[0] != surface_normals.shape[0]
    ):
        raise ValueError("boundary_ridge_alignment_loss inputs must share the same leading dimension.")

    surface_normals = F.normalize(surface_normals, dim=-1)
    ridge_norm = torch.linalg.norm(ridge_normals, dim=-1)
    valid = (
        torch.isfinite(ridge_directional_derivative)
        & torch.isfinite(ridge_strength)
        & torch.isfinite(ridge_norm)
        & (ridge_norm > 1e-6)
    )
    if not torch.any(valid):
        return torch.zeros((), dtype=dtype, device=device)

    ridge_strength = ridge_strength.clamp(0.0, 1.0)
    ridge_normals = ridge_normals.clone()
    ridge_normals[valid] = F.normalize(ridge_normals[valid], dim=-1)

    stationarity = F.smooth_l1_loss(
        ridge_directional_derivative[valid],
        torch.zeros_like(ridge_directional_derivative[valid]),
        beta=float(ridge_direction_delta),
        reduction="none",
    )
    attraction = torch.relu(float(ridge_strength_floor) - ridge_strength[valid])
    direction = 1.0 - torch.abs(torch.sum(surface_normals[valid] * ridge_normals[valid], dim=-1))
    combined = stationarity + attraction + direction

    weights = ridge_strength[valid].clamp_min(0.1)
    return (combined * weights).sum() / weights.sum().clamp_min(1e-8)


def void_boundary_loss(
    inner_occupancy,
    outer_occupancy,
    negative_region_weight,
    boundary_strength=None,
    margin=0.25,
):
    inner_occupancy = torch.as_tensor(inner_occupancy)
    if inner_occupancy.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=inner_occupancy.device)

    device = inner_occupancy.device
    dtype = inner_occupancy.dtype
    outer_occupancy = torch.as_tensor(outer_occupancy, device=device, dtype=dtype).reshape(-1)
    inner_occupancy = inner_occupancy.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    outer_occupancy = outer_occupancy.clamp(1e-6, 1.0 - 1e-6)
    if inner_occupancy.shape != outer_occupancy.shape:
        raise ValueError("inner_occupancy and outer_occupancy must have the same shape.")

    weights = torch.as_tensor(negative_region_weight, device=device, dtype=dtype).reshape(-1).clamp_min(0.0)
    if weights.shape != inner_occupancy.shape:
        raise ValueError("negative_region_weight must match occupancy length.")
    if boundary_strength is not None:
        strength = torch.as_tensor(boundary_strength, device=device, dtype=dtype).reshape(-1).clamp_min(0.0)
        if strength.shape != inner_occupancy.shape:
            raise ValueError("boundary_strength must match occupancy length.")
        weights = weights * strength

    valid = torch.isfinite(inner_occupancy) & torch.isfinite(outer_occupancy) & torch.isfinite(weights) & (weights > 0.0)
    if not torch.any(valid):
        return torch.zeros((), dtype=dtype, device=device)

    inner = inner_occupancy[valid]
    outer = outer_occupancy[valid]
    weights = weights[valid]
    target_inner = torch.ones_like(inner)
    target_outer = torch.zeros_like(outer)
    inner_loss = F.binary_cross_entropy(inner, target_inner, reduction="none")
    outer_loss = F.binary_cross_entropy(outer, target_outer, reduction="none")
    contrast_loss = torch.relu(float(margin) - (inner - outer))
    combined = (inner_loss + outer_loss + contrast_loss) * weights
    return combined.sum() / weights.sum().clamp_min(1e-8)


def surface_thickness_loss(raw_scaling, rotation, normals, region_type, max_thickness):
    raw_scaling = torch.as_tensor(raw_scaling)
    if raw_scaling.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=raw_scaling.device)

    region_type = torch.as_tensor(region_type, device=raw_scaling.device, dtype=torch.long).reshape(-1)
    surface_mask = region_type == 0
    if not torch.any(surface_mask):
        return torch.zeros((), dtype=raw_scaling.dtype, device=raw_scaling.device)

    scales = torch.exp(raw_scaling[surface_mask])
    normals = F.normalize(torch.as_tensor(normals, device=raw_scaling.device, dtype=raw_scaling.dtype).reshape(-1, 3)[surface_mask], dim=-1)
    rotations = quaternion_to_matrix(torch.as_tensor(rotation, device=raw_scaling.device, dtype=raw_scaling.dtype).reshape(-1, 4)[surface_mask])
    local_normals = torch.einsum("nij,nj->ni", rotations.transpose(1, 2), normals)
    variance_along_normal = torch.sum((local_normals * scales) ** 2, dim=-1)
    thickness = torch.sqrt(variance_along_normal.clamp_min(1e-8))
    return torch.relu(thickness - float(max_thickness)).mean()


def surface_tangential_scale_loss(raw_scaling, rotation_mats, normals, max_tangential_scale):
    raw_scaling = torch.as_tensor(raw_scaling)
    if raw_scaling.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=raw_scaling.device)

    rotations = torch.as_tensor(rotation_mats, device=raw_scaling.device, dtype=raw_scaling.dtype)
    normals = F.normalize(
        torch.as_tensor(normals, device=raw_scaling.device, dtype=raw_scaling.dtype).reshape(-1, 3),
        dim=-1,
    )
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotation_mats must have shape (N, 3, 3).")

    scales = torch.exp(raw_scaling)
    local_normals = torch.einsum("nij,nj->ni", rotations.transpose(1, 2), normals)
    variance_along_normal = torch.sum((local_normals * scales) ** 2, dim=-1)
    total_variance = torch.sum(scales * scales, dim=-1)
    tangential_variance = (total_variance - variance_along_normal).clamp_min(1e-8)
    tangential_rms = torch.sqrt(0.5 * tangential_variance)
    return torch.relu(tangential_rms - float(max_tangential_scale)).mean()


def surface_opacity_regularizer(opacity, target_opacity=0.9):
    opacity = torch.as_tensor(opacity)
    if opacity.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=opacity.device)

    values = opacity.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    return torch.relu(values - float(target_opacity)).square().mean()


def surface_shape_loss(
    raw_scaling,
    rotation_mats,
    normals,
    opacity,
    max_thickness,
    max_tangential_scale,
    min_opacity=0.8,
):
    raw_scaling = torch.as_tensor(raw_scaling)
    if raw_scaling.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=raw_scaling.device)

    device = raw_scaling.device
    dtype = raw_scaling.dtype
    rotations = torch.as_tensor(rotation_mats, device=device, dtype=dtype)
    normals = F.normalize(torch.as_tensor(normals, device=device, dtype=dtype).reshape(-1, 3), dim=-1)
    opacity = torch.as_tensor(opacity, device=device, dtype=dtype).reshape(-1)
    if raw_scaling.ndim != 2 or raw_scaling.shape[1] != 3:
        raise ValueError("raw_scaling must have shape (N, 3).")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotation_mats must have shape (N, 3, 3).")
    if rotations.shape[0] != raw_scaling.shape[0] or normals.shape[0] != raw_scaling.shape[0] or opacity.shape[0] != raw_scaling.shape[0]:
        raise ValueError("surface_shape_loss inputs must share the same leading dimension.")

    scales = torch.exp(raw_scaling)
    local_normals = torch.einsum("nij,nj->ni", rotations.transpose(1, 2), normals)
    normal_variance = torch.sum((local_normals * scales) ** 2, dim=-1)
    normal_thickness = torch.sqrt(normal_variance.clamp_min(1e-8))
    total_variance = torch.sum(scales * scales, dim=-1)
    tangential_variance = (total_variance - normal_variance).clamp_min(1e-8)
    tangential_rms = torch.sqrt(0.5 * tangential_variance)

    aniso = torch.relu(normal_thickness - float(max_thickness)) + torch.relu(
        tangential_rms - float(max_tangential_scale)
    )
    opacity_floor = torch.relu(float(min_opacity) - opacity.clamp(1e-6, 1.0 - 1e-6))
    combined = aniso + opacity_floor
    valid = torch.isfinite(combined)
    if not torch.any(valid):
        return torch.zeros((), dtype=dtype, device=device)
    return combined[valid].mean()


def bulk_scale_regularizer(raw_scaling, region_type, max_bulk_scale):
    raw_scaling = torch.as_tensor(raw_scaling)
    if raw_scaling.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=raw_scaling.device)

    region_type = torch.as_tensor(region_type, device=raw_scaling.device, dtype=torch.long).reshape(-1)
    bulk_mask = region_type == 1
    if not torch.any(bulk_mask):
        return torch.zeros((), dtype=raw_scaling.dtype, device=raw_scaling.device)

    scales = torch.exp(raw_scaling[bulk_mask])
    bulk_extent = torch.max(scales, dim=-1).values
    max_bulk_scale_tensor = torch.as_tensor(max_bulk_scale, device=raw_scaling.device, dtype=raw_scaling.dtype).reshape(-1)
    if max_bulk_scale_tensor.numel() == 1:
        limits = max_bulk_scale_tensor.expand_as(bulk_extent)
    elif max_bulk_scale_tensor.numel() == raw_scaling.shape[0]:
        limits = max_bulk_scale_tensor[bulk_mask]
    elif max_bulk_scale_tensor.numel() == bulk_extent.shape[0]:
        limits = max_bulk_scale_tensor
    else:
        raise ValueError("max_bulk_scale must be a scalar, length-N tensor, or length-bulk tensor.")
    return torch.relu(bulk_extent - limits).square().mean()


def bulk_overlap_regularizer(xyz, raw_scaling, neighbor_index=None):
    xyz = torch.as_tensor(xyz)
    raw_scaling = torch.as_tensor(raw_scaling, device=xyz.device, dtype=xyz.dtype)
    if xyz.numel() == 0 or raw_scaling.numel() == 0 or xyz.shape[0] <= 1:
        return torch.zeros((), dtype=xyz.dtype if xyz.numel() > 0 else torch.float32, device=xyz.device)

    neighbor_index = torch.as_tensor(neighbor_index, dtype=torch.long, device=xyz.device)
    if neighbor_index.ndim != 2 or neighbor_index.shape[0] != xyz.shape[0] or neighbor_index.shape[1] == 0:
        return torch.zeros((), dtype=xyz.dtype, device=xyz.device)

    neighbor_index = neighbor_index.clamp_min(0)
    scales = torch.exp(raw_scaling)
    radius = torch.max(scales, dim=-1).values
    distances = torch.norm(xyz.unsqueeze(1) - xyz[neighbor_index], dim=-1)
    overlap = torch.relu(radius.unsqueeze(1) + radius[neighbor_index] - distances)
    return overlap.mean()


def bulk_regularization_loss(
    xyz,
    raw_scaling,
    rotation_mats,
    opacity,
    neighbor_index=None,
    max_bulk_scale=4.0,
    density_cap=3.0,
    k=8,
):
    xyz = torch.as_tensor(xyz)
    if xyz.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=xyz.device)

    device = xyz.device
    dtype = xyz.dtype
    raw_scaling = torch.as_tensor(raw_scaling, device=device, dtype=dtype)
    rotation_mats = torch.as_tensor(rotation_mats, device=device, dtype=dtype)
    opacity = torch.as_tensor(opacity, device=device, dtype=dtype).reshape(-1)
    if raw_scaling.ndim != 2 or raw_scaling.shape[1] != 3:
        raise ValueError("raw_scaling must have shape (N, 3).")
    if rotation_mats.ndim != 3 or rotation_mats.shape[-2:] != (3, 3):
        raise ValueError("rotation_mats must have shape (N, 3, 3).")
    if raw_scaling.shape[0] != xyz.shape[0] or rotation_mats.shape[0] != xyz.shape[0] or opacity.shape[0] != xyz.shape[0]:
        raise ValueError("bulk_regularization_loss inputs must share the same leading dimension.")

    scales = torch.exp(raw_scaling).clamp_min(1e-8)
    bulk_extent = torch.max(scales, dim=-1).values
    max_bulk_scale_tensor = torch.as_tensor(max_bulk_scale, device=device, dtype=dtype).reshape(-1)
    if max_bulk_scale_tensor.numel() == 1:
        limits = max_bulk_scale_tensor.expand_as(bulk_extent)
    elif max_bulk_scale_tensor.numel() == bulk_extent.shape[0]:
        limits = max_bulk_scale_tensor
    else:
        raise ValueError("max_bulk_scale must be a scalar or length-N tensor.")
    scale_term = torch.relu(bulk_extent - limits)

    density_term = torch.zeros_like(scale_term)
    neighbor_index = _resolve_neighbor_index(xyz, neighbor_index, int(k))
    if xyz.shape[0] > 1 and neighbor_index.shape[1] > 0:
        raw_neighbors = neighbor_index.to(device=device, dtype=torch.long)
        valid_neighbors = (raw_neighbors >= 0) & (raw_neighbors < xyz.shape[0])
        safe_neighbors = raw_neighbors.clamp(0, max(xyz.shape[0] - 1, 0))
        center_ids = torch.arange(xyz.shape[0], device=device).unsqueeze(1)
        valid_neighbors = valid_neighbors & (safe_neighbors != center_ids)

        neighbor_xyz = xyz[safe_neighbors]
        diff = xyz.unsqueeze(1) - neighbor_xyz
        neighbor_rotations = rotation_mats[safe_neighbors]
        local = torch.einsum("nkij,nkj->nki", neighbor_rotations.transpose(-1, -2), diff)
        neighbor_scales = scales[safe_neighbors]
        normalized = local / neighbor_scales
        kernel = torch.exp(-0.5 * torch.sum(normalized * normalized, dim=-1))
        neighbor_opacity = opacity[safe_neighbors].clamp_min(0.0)
        local_density = torch.sum(torch.where(valid_neighbors, neighbor_opacity * kernel, torch.zeros_like(kernel)), dim=1)
        density_term = torch.relu(local_density - float(density_cap))

    combined = scale_term + density_term
    valid = torch.isfinite(combined)
    if not torch.any(valid):
        return torch.zeros((), dtype=dtype, device=device)
    return combined[valid].mean()


def _fit_plane_normal(local_points: torch.Tensor) -> torch.Tensor | None:
    if local_points.shape[0] < 3:
        return None

    work_points = local_points.to(dtype=torch.float64)
    centroid = work_points.mean(dim=0)
    centered = work_points - centroid
    covariance = centered.T @ centered / max(local_points.shape[0] - 1, 1)
    trace = torch.trace(covariance).abs()
    jitter = max(float(trace.item()) * 1e-6, 1e-8)
    covariance = covariance + torch.eye(3, dtype=covariance.dtype, device=covariance.device) * jitter

    try:
        _, eigenvectors = torch.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
    except torch.linalg.LinAlgError:
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
        except torch.linalg.LinAlgError:
            return None

    if not torch.all(torch.isfinite(normal)):
        return None
    return F.normalize(normal.to(dtype=local_points.dtype), dim=0)


def _empty_point_to_plane_cache(device: torch.device, dtype: torch.dtype) -> PointToPlaneCache:
    empty_long = torch.empty((0,), dtype=torch.long, device=device)
    empty_float = torch.empty((0, 3), dtype=dtype, device=device)
    empty_weight = torch.empty((0,), dtype=dtype, device=device)
    return PointToPlaneCache(
        active_indices=empty_long,
        centroids=empty_float,
        fitted_normals=empty_float.clone(),
        weights=empty_weight,
    )


def prepare_point_to_plane_cache(xyz, normals, planarity, material_ids=None, primitive_type=None, neighbor_index=None, k=8):
    xyz = torch.as_tensor(xyz)
    if xyz.numel() == 0:
        return _empty_point_to_plane_cache(xyz.device, torch.float32)

    normals = F.normalize(torch.as_tensor(normals, device=xyz.device, dtype=xyz.dtype), dim=-1)
    planarity = torch.as_tensor(planarity, device=xyz.device, dtype=xyz.dtype).reshape(-1)
    material_ids = None if material_ids is None else torch.as_tensor(material_ids, device=xyz.device, dtype=torch.long).reshape(-1)
    neighbor_index = _resolve_neighbor_index(xyz, neighbor_index, k)
    planar_mask = _primitive_mask(primitive_type, xyz.shape[0], xyz.device)
    active_indices = torch.nonzero(planar_mask & (planarity > 0.0), as_tuple=False).reshape(-1)

    if active_indices.numel() == 0 or neighbor_index.shape[1] == 0:
        return _empty_point_to_plane_cache(xyz.device, xyz.dtype)

    cached_indices = []
    cached_centroids = []
    cached_normals = []
    cached_weights = []
    for point_index in active_indices.tolist():
        neighbors = neighbor_index[point_index]
        valid = (neighbors >= 0) & (neighbors < xyz.shape[0]) & (neighbors != point_index)
        valid = valid & planar_mask[neighbors.clamp_min(0)]
        if material_ids is not None:
            center_material = material_ids[point_index]
            if center_material >= 0:
                valid = valid & (material_ids[neighbors.clamp_min(0)] == center_material)
        if int(valid.sum().item()) < 3:
            continue

        # Use a detached local plane fit as a geometric target. Backpropagating through the
        # eigenvector solve is numerically unstable on nearly planar neighborhoods and can
        # inject NaNs into xyz gradients even when the forward loss is finite.
        local_points = xyz[neighbors[valid]].detach()
        centroid = local_points.mean(dim=0)
        fitted_normal = _fit_plane_normal(local_points)
        if fitted_normal is None:
            continue
        reference_normal = normals[point_index].detach()
        sign = torch.where(torch.dot(fitted_normal, reference_normal) < 0.0, -1.0, 1.0)
        fitted_normal = fitted_normal * sign
        cached_indices.append(int(point_index))
        cached_centroids.append(centroid.to(dtype=xyz.dtype))
        cached_normals.append(fitted_normal.to(dtype=xyz.dtype))
        cached_weights.append(planarity[point_index])

    if not cached_indices:
        return _empty_point_to_plane_cache(xyz.device, xyz.dtype)

    return PointToPlaneCache(
        active_indices=torch.as_tensor(cached_indices, dtype=torch.long, device=xyz.device),
        centroids=torch.stack(cached_centroids, dim=0),
        fitted_normals=torch.stack(cached_normals, dim=0),
        weights=torch.stack(cached_weights, dim=0).reshape(-1),
    )


def point_to_plane_loss_from_cache(xyz, cache: PointToPlaneCache):
    xyz = torch.as_tensor(xyz)
    if xyz.numel() == 0 or cache.is_empty:
        return torch.zeros((), dtype=xyz.dtype if xyz.numel() > 0 else torch.float32, device=xyz.device)

    centers = xyz.index_select(0, cache.active_indices)
    distances = torch.abs(torch.sum((centers - cache.centroids) * cache.fitted_normals, dim=-1))
    return (distances * cache.weights.to(device=xyz.device, dtype=xyz.dtype)).mean()


def point_to_plane_loss(xyz, normals, planarity, material_ids=None, primitive_type=None, neighbor_index=None, k=8):
    xyz = torch.as_tensor(xyz)
    cache = prepare_point_to_plane_cache(
        xyz,
        normals,
        planarity,
        material_ids=material_ids,
        primitive_type=primitive_type,
        neighbor_index=neighbor_index,
        k=k,
    )
    return point_to_plane_loss_from_cache(xyz, cache)


def normal_alignment_loss(normals, neighbor_normals, planarity):
    normals = F.normalize(torch.as_tensor(normals), dim=-1)
    if normals.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=normals.device)

    neighbor_normals = torch.as_tensor(neighbor_normals, device=normals.device, dtype=normals.dtype)
    if neighbor_normals.ndim == 2:
        neighbor_normals = neighbor_normals.unsqueeze(1)
    if neighbor_normals.numel() == 0:
        return torch.zeros((), dtype=normals.dtype, device=normals.device)

    neighbor_normals = F.normalize(neighbor_normals, dim=-1)
    weights = torch.as_tensor(planarity, device=normals.device, dtype=normals.dtype)
    if weights.ndim == 1:
        weights = weights.unsqueeze(1)
    if weights.ndim == 2 and weights.shape[1] == 1:
        weights = weights.expand(-1, neighbor_normals.shape[1])
    elif weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)

    cosine = torch.abs(torch.sum(normals.unsqueeze(1) * neighbor_normals, dim=-1))
    valid = torch.isfinite(cosine) & torch.isfinite(weights) & (weights > 0.0)
    if not torch.any(valid):
        return torch.zeros((), dtype=normals.dtype, device=normals.device)

    losses = (1.0 - cosine) * weights
    return losses[valid].sum() / weights[valid].sum().clamp_min(1e-8)


def thickness_penalty(scaling, rotation, normals, primitive_type, max_thickness):
    del rotation, normals
    scaling = torch.as_tensor(scaling)
    if scaling.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=scaling.device)

    planar_mask = _primitive_mask(primitive_type, scaling.shape[0], scaling.device)
    if not torch.any(planar_mask):
        return torch.zeros((), dtype=scaling.dtype, device=scaling.device)

    thickness = torch.exp(scaling[planar_mask, 2])
    return torch.relu(thickness - float(max_thickness)).mean()


def material_boundary_loss(xyz, material_ids, opacity, neighbor_index=None, target_opacity=0.5):
    xyz = torch.as_tensor(xyz)
    if xyz.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=xyz.device)

    material_ids = torch.as_tensor(material_ids, device=xyz.device, dtype=torch.long).reshape(-1)
    nonnegative_ids = material_ids[material_ids >= 0]
    if nonnegative_ids.numel() == 0 or torch.unique(nonnegative_ids).numel() <= 1:
        return torch.zeros((), dtype=xyz.dtype, device=xyz.device)

    opacity = torch.as_tensor(opacity, device=xyz.device, dtype=xyz.dtype).reshape(-1)
    neighbor_index = _resolve_neighbor_index(xyz, neighbor_index, k=8)
    if neighbor_index.shape[1] == 0:
        return torch.zeros((), dtype=xyz.dtype, device=xyz.device)

    neighbor_index = neighbor_index.clamp_min(0)
    center_material = material_ids.unsqueeze(1)
    neighbor_material = material_ids[neighbor_index]
    valid_pairs = (center_material >= 0) & (neighbor_material >= 0) & (center_material != neighbor_material)
    if not torch.any(valid_pairs):
        return torch.zeros((), dtype=xyz.dtype, device=xyz.device)

    distances = torch.norm(xyz.unsqueeze(1) - xyz[neighbor_index], dim=-1)
    mean_spacing = distances[valid_pairs].mean().clamp_min(1e-6)
    pair_opacity = torch.minimum(opacity.unsqueeze(1), opacity[neighbor_index])
    losses = torch.relu(float(target_opacity) - pair_opacity) * torch.exp(-distances / mean_spacing)
    return losses[valid_pairs].mean()


def material_boundary_compact_loss(
    xyz,
    normals,
    material_ids,
    opacity,
    neighbor_index=None,
    k=8,
    min_opacity=0.8,
    align_weight=0.3,
    cross_floor_weight=0.2,
):
    xyz = torch.as_tensor(xyz)
    if xyz.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=xyz.device)

    device = xyz.device
    dtype = xyz.dtype
    normals = F.normalize(torch.as_tensor(normals, device=device, dtype=dtype).reshape(-1, 3), dim=-1)
    material_ids = torch.as_tensor(material_ids, device=device, dtype=torch.long).reshape(-1)
    opacity = torch.as_tensor(opacity, device=device, dtype=dtype).reshape(-1)
    if normals.shape[0] != xyz.shape[0] or material_ids.shape[0] != xyz.shape[0] or opacity.shape[0] != xyz.shape[0]:
        raise ValueError("material_boundary_compact_loss inputs must share the same leading dimension.")

    neighbor_index = _resolve_neighbor_index(xyz, neighbor_index, int(k))
    if neighbor_index.shape[1] == 0:
        return torch.zeros((), dtype=dtype, device=device)

    raw_neighbors = neighbor_index.to(device=device, dtype=torch.long)
    valid_neighbors = (raw_neighbors >= 0) & (raw_neighbors < xyz.shape[0])
    safe_neighbors = raw_neighbors.clamp(0, max(xyz.shape[0] - 1, 0))
    center_ids = torch.arange(xyz.shape[0], device=device).unsqueeze(1)
    valid_neighbors = valid_neighbors & (safe_neighbors != center_ids)

    center_material = material_ids.unsqueeze(1)
    neighbor_material = material_ids[safe_neighbors]
    valid_material = (center_material >= 0) & (neighbor_material >= 0)
    same_pairs = valid_neighbors & valid_material & (center_material == neighbor_material)
    cross_pairs = valid_neighbors & valid_material & (center_material != neighbor_material)

    same_term = torch.zeros((), dtype=dtype, device=device)
    if torch.any(same_pairs):
        opacity_delta = torch.abs(opacity.unsqueeze(1) - opacity[safe_neighbors])
        same_term = opacity_delta[same_pairs].mean()

    align_term = torch.zeros((), dtype=dtype, device=device)
    cross_floor_term = torch.zeros((), dtype=dtype, device=device)
    if torch.any(cross_pairs):
        diff = xyz[safe_neighbors] - xyz.unsqueeze(1)
        distances = torch.linalg.norm(diff, dim=-1)
        mean_cross_distance = distances[cross_pairs].mean().clamp_min(1e-6)
        weights = torch.exp(-distances / mean_cross_distance)

        directions = F.normalize(diff, dim=-1)
        cosine = torch.abs(torch.sum(normals.unsqueeze(1) * directions, dim=-1))
        align_losses = 1.0 - cosine
        cross_weights = weights[cross_pairs]
        align_term = (align_losses[cross_pairs] * cross_weights).sum() / cross_weights.sum().clamp_min(1e-8)

        pair_opacity = torch.minimum(opacity.unsqueeze(1), opacity[safe_neighbors])
        floor_losses = torch.relu(float(min_opacity) - pair_opacity)
        cross_floor_term = (floor_losses[cross_pairs] * cross_weights).sum() / cross_weights.sum().clamp_min(1e-8)

    return same_term + float(align_weight) * align_term + float(cross_floor_weight) * cross_floor_term


def edge_split_criterion(grads, xyz, normals, curvature):
    del xyz, normals
    grads = torch.as_tensor(grads, dtype=torch.float32)
    curvature = torch.as_tensor(curvature, device=grads.device, dtype=grads.dtype).reshape(-1, 1)
    if grads.ndim == 1:
        grads = grads.unsqueeze(-1)
    grad_strength = torch.linalg.norm(grads, dim=-1, keepdim=True)
    grad_norm = grad_strength / grad_strength.mean().clamp_min(1e-6)
    curvature_norm = curvature / curvature.mean().clamp_min(1e-6)
    edge_score = torch.tanh(0.6 * curvature_norm + 0.4 * grad_norm)
    return torch.clamp(1.4 - 0.8 * edge_score, min=0.5, max=1.4)
