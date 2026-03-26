from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.loss_utils import ssim, weighted_l1_loss


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
