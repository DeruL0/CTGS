import torch
import torch.nn.functional as F

from ct_pipeline.geometry.coordinates import world_xyz_to_voxel_float_torch


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


def _weighted_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, weight_map, beta: float) -> torch.Tensor:
    losses = F.smooth_l1_loss(pred, target, beta=float(beta), reduction="none")
    if weight_map is None:
        return losses.mean()

    weight = _as_slice_batch(weight_map).to(device=pred.device, dtype=pred.dtype)
    if weight.shape != losses.shape:
        weight = torch.broadcast_to(weight, losses.shape)
    return (losses * weight).sum() / weight.sum().clamp_min(1e-8)


def binary_focal_loss(
    pred_prob,
    target,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
    sample_weights=None,
    reduction: str = "mean",
    eps: float = 1e-6,
):
    pred = torch.as_tensor(pred_prob)
    if pred.numel() == 0:
        return torch.zeros((), dtype=pred.dtype if torch.is_floating_point(pred) else torch.float32, device=pred.device)
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0.")
    if alpha is not None and (float(alpha) < 0.0 or float(alpha) > 1.0):
        raise ValueError("alpha must be in [0, 1] when provided.")
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")

    pred = pred.reshape(-1).clamp(float(eps), 1.0 - float(eps))
    target = torch.as_tensor(target, device=pred.device, dtype=pred.dtype).reshape(-1).clamp(0.0, 1.0)
    if target.shape[0] != pred.shape[0]:
        raise ValueError("target must match pred_prob length.")

    p_t = pred * target + (1.0 - pred) * (1.0 - target)
    loss = -torch.pow(1.0 - p_t, float(gamma)) * torch.log(p_t)
    if alpha is not None:
        alpha_t = float(alpha) * target + (1.0 - float(alpha)) * (1.0 - target)
        loss = alpha_t * loss
    if sample_weights is not None:
        weights = torch.as_tensor(sample_weights, device=pred.device, dtype=pred.dtype).reshape(-1).clamp_min(0.0)
        if weights.shape[0] != pred.shape[0]:
            raise ValueError("sample_weights must match pred_prob length.")
        loss = loss * weights
        if reduction == "mean":
            return loss.sum() / weights.sum().clamp_min(1e-8)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be one of {'none', 'sum', 'mean'}.")


def asymmetric_binary_focal_loss(
    pred_prob,
    target,
    *,
    gamma_pos: float = 0.0,
    gamma_neg: float = 2.0,
    alpha_pos: float = 0.7,
    sample_weights=None,
    reduction: str = "mean",
    eps: float = 1e-6,
):
    pred = torch.as_tensor(pred_prob)
    if pred.numel() == 0:
        return torch.zeros((), dtype=pred.dtype if torch.is_floating_point(pred) else torch.float32, device=pred.device)
    if gamma_pos < 0.0 or gamma_neg < 0.0:
        raise ValueError("focal gammas must be >= 0.")
    if float(alpha_pos) < 0.0 or float(alpha_pos) > 1.0:
        raise ValueError("alpha_pos must be in [0, 1].")
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")

    pred = pred.reshape(-1).clamp(float(eps), 1.0 - float(eps))
    target = torch.as_tensor(target, device=pred.device, dtype=pred.dtype).reshape(-1).clamp(0.0, 1.0)
    if target.shape[0] != pred.shape[0]:
        raise ValueError("target must match pred_prob length.")

    bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    positive = target >= 0.5
    gamma = torch.where(
        positive,
        torch.full_like(pred, float(gamma_pos)),
        torch.full_like(pred, float(gamma_neg)),
    )
    alpha = torch.where(
        positive,
        torch.full_like(pred, float(alpha_pos)),
        torch.full_like(pred, 1.0 - float(alpha_pos)),
    )
    p_t = pred * target + (1.0 - pred) * (1.0 - target)
    loss = alpha * torch.pow(1.0 - p_t, gamma) * bce
    if sample_weights is not None:
        weights = torch.as_tensor(sample_weights, device=pred.device, dtype=pred.dtype).reshape(-1).clamp_min(0.0)
        if weights.shape[0] != pred.shape[0]:
            raise ValueError("sample_weights must match pred_prob length.")
        loss = loss * weights
        if reduction == "mean":
            return loss.sum() / weights.sum().clamp_min(1e-8)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be one of {'none', 'sum', 'mean'}.")


def eagle_patch_loss(
    pred_slice,
    target_slice,
    *,
    block_size: int = 16,
    variance_power: float = 0.5,
    high_freq_radius: float = 0.25,
    eps: float = 1e-6,
):
    """Lightweight EAGLE-style edge-aware frequency loss for CT intensity patches.

    The hook is intentionally optional in training. It weights the reconstruction
    residual by local target-gradient variance, then compares high-frequency FFT
    magnitudes so Huber keeps intensity calibration while this term favors sharp
    CT edges.
    """
    pred = _as_slice_batch(pred_slice)
    target = _as_slice_batch(target_slice).to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        raise ValueError("pred_slice and target_slice must have the same shape.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if high_freq_radius < 0.0 or high_freq_radius > 1.0:
        raise ValueError("high_freq_radius must be in [0, 1].")
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")

    grad_x = F.pad(target[..., :, 1:] - target[..., :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(target[..., 1:, :] - target[..., :-1, :], (0, 0, 0, 1))
    grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + float(eps))
    mean = F.avg_pool2d(grad_mag, int(block_size), stride=1, padding=int(block_size) // 2)
    mean_sq = F.avg_pool2d(grad_mag.square(), int(block_size), stride=1, padding=int(block_size) // 2)
    variance = (mean_sq - mean.square()).clamp_min(0.0)
    if variance.shape[-2:] != target.shape[-2:]:
        variance = variance[..., : target.shape[-2], : target.shape[-1]]
    edge_weight = 1.0 + torch.pow(variance / variance.detach().mean().clamp_min(float(eps)), float(variance_power))

    residual = (pred - target) * edge_weight.detach()
    spectrum = torch.fft.rfft2(residual, norm="ortho").abs()
    height, width = residual.shape[-2:]
    fy = torch.fft.fftfreq(height, device=residual.device, dtype=residual.dtype).reshape(-1, 1)
    fx = torch.fft.rfftfreq(width, device=residual.device, dtype=residual.dtype).reshape(1, -1)
    radius = torch.sqrt(fx.square() + fy.square())
    high_mask = radius >= float(high_freq_radius) * float(radius.max().clamp_min(float(eps)))
    if not torch.any(high_mask):
        return spectrum.mean()
    return spectrum[..., high_mask].mean()


def calibrated_render_huber_loss(
    rendered_occupancy,
    gt_slice,
    intensity_air,
    intensity_mat,
    weight_map=None,
    huber_beta: float = 0.1,
):
    if huber_beta <= 0.0:
        raise ValueError("huber_beta must be > 0.")

    occupancy = _as_slice_batch(rendered_occupancy).clamp(0.0, 1.0)
    target = _as_slice_batch(gt_slice).to(device=occupancy.device, dtype=occupancy.dtype)
    air = torch.as_tensor(intensity_air, device=occupancy.device, dtype=occupancy.dtype)
    material = torch.as_tensor(intensity_mat, device=occupancy.device, dtype=occupancy.dtype)
    calibrated = air + (material - air) * occupancy
    weight = None if weight_map is None else _as_slice_batch(weight_map).to(device=occupancy.device, dtype=occupancy.dtype)
    return _weighted_smooth_l1_loss(calibrated, target, weight, float(huber_beta))


def soft_occupancy_loss(
    pred_occupancy,
    signed_distance,
    tau=None,
    huber_beta: float = 0.1,
    sample_weights=None,
    tau_voxels=None,
):
    pred = torch.as_tensor(pred_occupancy)
    if pred.numel() == 0:
        return torch.zeros((), dtype=pred.dtype if pred.numel() > 0 else torch.float32, device=pred.device)
    if tau is None:
        tau = tau_voxels
    if tau is None:
        raise ValueError("tau must be provided.")
    if tau <= 0.0:
        raise ValueError("tau must be > 0.")
    if huber_beta <= 0.0:
        raise ValueError("huber_beta must be > 0.")

    pred = pred.reshape(-1).clamp(0.0, 1.0)
    distance = torch.as_tensor(signed_distance, device=pred.device, dtype=pred.dtype).reshape(-1)
    if distance.shape[0] != pred.shape[0]:
        raise ValueError("signed_distance must match pred_occupancy length.")
    target = torch.sigmoid(-distance / float(tau))
    losses = F.smooth_l1_loss(pred, target, beta=float(huber_beta), reduction="none")
    if sample_weights is None:
        return losses.mean()
    weights = torch.as_tensor(sample_weights, device=pred.device, dtype=pred.dtype).reshape(-1)
    if weights.shape[0] != pred.shape[0]:
        raise ValueError("sample_weights must match pred_occupancy length.")
    weights = weights.clamp_min(0.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def sample_volume_field(volume_field: torch.Tensor, points_xyz: torch.Tensor, spacing_zyx):
    """Sample a voxel field at world xyz points under the voxel-center convention."""

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

    del spacing_z, spacing_y, spacing_x
    x_idx, y_idx, z_idx = world_xyz_to_voxel_float_torch(points_xyz, spacing_zyx)
    if width <= 1:
        x_norm = torch.zeros_like(x_idx)
    else:
        x_norm = 2.0 * (x_idx / float(width - 1)) - 1.0
    if height <= 1:
        y_norm = torch.zeros_like(y_idx)
    else:
        y_norm = 2.0 * (y_idx / float(height - 1)) - 1.0
    if depth <= 1:
        z_norm = torch.zeros_like(z_idx)
    else:
        z_norm = 2.0 * (z_idx / float(depth - 1)) - 1.0
    grid = torch.stack((x_norm, y_norm, z_norm), dim=-1).reshape(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        volume_field,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(volume_field.shape[1], -1).transpose(0, 1)


def sample_sdf_normals(signed_distance_volume: torch.Tensor, points_xyz: torch.Tensor, spacing_zyx):
    points_xyz = torch.as_tensor(points_xyz)
    if points_xyz.numel() == 0:
        return torch.empty((0, 3), dtype=points_xyz.dtype if points_xyz.numel() > 0 else torch.float32, device=points_xyz.device)

    signed_distance_volume = torch.as_tensor(signed_distance_volume)
    dtype = signed_distance_volume.dtype
    device = signed_distance_volume.device
    points_xyz = points_xyz.to(device=device, dtype=dtype)
    spacing_z, spacing_y, spacing_x = [max(float(value), 1e-8) for value in spacing_zyx]
    offsets = torch.tensor(
        [
            [spacing_x, 0.0, 0.0],
            [0.0, spacing_y, 0.0],
            [0.0, 0.0, spacing_z],
        ],
        dtype=dtype,
        device=device,
    )
    gradients = []
    for axis, step in enumerate((spacing_x, spacing_y, spacing_z)):
        plus = sample_volume_field(signed_distance_volume, points_xyz + offsets[axis], spacing_zyx).reshape(-1)
        minus = sample_volume_field(signed_distance_volume, points_xyz - offsets[axis], spacing_zyx).reshape(-1)
        gradients.append((plus - minus) / (2.0 * float(step)))
    normals = torch.stack(gradients, dim=1)
    return F.normalize(normals, dim=-1, eps=1e-8)


def surface_sdf_thickness_loss(
    xyz,
    raw_scaling,
    rotation_mats,
    signed_distance_volume,
    spacing_zyx,
    max_normal_thickness,
    min_tangent_spread: float = 0.0,
    thickness_beta: float = 0.2,
    spread_beta: float = 0.2,
    outside_beta: float = 0.0,
    sdf_normals=None,
    huber_beta: float = 0.1,
):
    xyz = torch.as_tensor(xyz)
    raw_scaling = torch.as_tensor(raw_scaling)
    if xyz.numel() == 0 or raw_scaling.numel() == 0:
        device = xyz.device if xyz.numel() > 0 else raw_scaling.device
        return torch.zeros((), dtype=torch.float32, device=device)
    if huber_beta <= 0.0:
        raise ValueError("huber_beta must be > 0.")
    if max_normal_thickness <= 0.0:
        raise ValueError("max_normal_thickness must be > 0.")
    if min_tangent_spread < 0.0:
        raise ValueError("min_tangent_spread must be >= 0.")
    if thickness_beta < 0.0:
        raise ValueError("thickness_beta must be >= 0.")
    if spread_beta < 0.0:
        raise ValueError("spread_beta must be >= 0.")
    if outside_beta < 0.0:
        raise ValueError("outside_beta must be >= 0.")

    device = xyz.device
    dtype = xyz.dtype
    raw_scaling = raw_scaling.to(device=device, dtype=dtype)
    rotations = torch.as_tensor(rotation_mats, device=device, dtype=dtype)
    signed_distance_volume = torch.as_tensor(signed_distance_volume, device=device, dtype=dtype)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3).")
    if raw_scaling.ndim != 2 or raw_scaling.shape[1] != 3:
        raise ValueError("raw_scaling must have shape (N, 3).")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotation_mats must have shape (N, 3, 3).")
    if rotations.shape[0] != raw_scaling.shape[0] or xyz.shape[0] != raw_scaling.shape[0]:
        raise ValueError("surface_sdf_thickness_loss inputs must share the same leading dimension.")

    signed_distance = sample_volume_field(signed_distance_volume, xyz, spacing_zyx).reshape(-1)
    scales = torch.exp(raw_scaling)
    if sdf_normals is None:
        surface_sdf_normals = sample_sdf_normals(signed_distance_volume, xyz, spacing_zyx)
    else:
        surface_sdf_normals = torch.as_tensor(sdf_normals, device=device, dtype=dtype).reshape(-1, 3)
        if surface_sdf_normals.shape[0] != xyz.shape[0]:
            raise ValueError("sdf_normals must match xyz length.")
        surface_sdf_normals = F.normalize(surface_sdf_normals, dim=-1, eps=1e-8)
    local_sdf_normals = torch.einsum("nij,nj->ni", rotations.transpose(1, 2), surface_sdf_normals)
    normal_variance = torch.sum((local_sdf_normals * scales) ** 2, dim=-1)
    normal_thickness = torch.sqrt(normal_variance.clamp_min(1e-8))
    thickness_term = torch.relu(normal_thickness - float(max_normal_thickness)).square()
    tangent_scales = scales[:, :2]
    spread_term = torch.relu(float(min_tangent_spread) - tangent_scales).square().mean(dim=-1)
    center_term = signed_distance.pow(2) + float(outside_beta) * torch.relu(signed_distance).square()
    combined = center_term + float(thickness_beta) * thickness_term + float(spread_beta) * spread_term
    if outside_beta > 0.0:
        footprint_extent = torch.sqrt(torch.sum((local_sdf_normals * scales) ** 2, dim=-1).clamp_min(1e-8))
        footprint_outside_term = torch.relu(signed_distance + footprint_extent - float(max_normal_thickness)).square()
        combined = combined + float(outside_beta) * footprint_outside_term
    valid = torch.isfinite(combined)
    if not torch.any(valid):
        return torch.zeros((), dtype=dtype, device=device)
    return combined[valid].mean()
