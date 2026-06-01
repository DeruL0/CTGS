from __future__ import annotations

import numpy as np
import torch


def world_xyz_to_voxel_float_torch(points_xyz: torch.Tensor, spacing_zyx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert world xyz coordinates to continuous voxel indices.

    CTGS uses a voxel-center world convention: voxel index ``(z, y, x)`` lives
    at world coordinate ``((x + 0.5) * sx, (y + 0.5) * sy, (z + 0.5) * sz)``.
    """

    points_xyz = torch.as_tensor(points_xyz)
    spacing_z, spacing_y, spacing_x = [max(float(value), 1e-8) for value in spacing_zyx]
    x_idx = points_xyz[:, 0] / spacing_x - 0.5
    y_idx = points_xyz[:, 1] / spacing_y - 0.5
    z_idx = points_xyz[:, 2] / spacing_z - 0.5
    return x_idx, y_idx, z_idx


def world_xyz_to_voxel_float_numpy(points_xyz, spacing_zyx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    spacing_z, spacing_y, spacing_x = [max(float(value), 1e-8) for value in spacing_zyx]
    x_idx = points_xyz[:, 0] / spacing_x - 0.5
    y_idx = points_xyz[:, 1] / spacing_y - 0.5
    z_idx = points_xyz[:, 2] / spacing_z - 0.5
    return x_idx.astype(np.float32), y_idx.astype(np.float32), z_idx.astype(np.float32)


def world_xyz_to_voxel_indices_floor_torch(
    points_xyz: torch.Tensor,
    spacing_zyx,
    *,
    shape_dhw: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map world xyz to the owning voxel index under the voxel-center convention."""

    points_xyz = torch.as_tensor(points_xyz)
    spacing_z, spacing_y, spacing_x = [max(float(value), 1e-8) for value in spacing_zyx]
    x_idx = torch.floor(points_xyz[:, 0] / spacing_x).to(dtype=torch.long)
    y_idx = torch.floor(points_xyz[:, 1] / spacing_y).to(dtype=torch.long)
    z_idx = torch.floor(points_xyz[:, 2] / spacing_z).to(dtype=torch.long)
    if shape_dhw is not None:
        depth, height, width = [int(value) for value in shape_dhw]
        x_idx = x_idx.clamp_(0, max(width - 1, 0))
        y_idx = y_idx.clamp_(0, max(height - 1, 0))
        z_idx = z_idx.clamp_(0, max(depth - 1, 0))
    return z_idx, y_idx, x_idx


def world_xyz_to_voxel_indices_floor_numpy(
    points_xyz,
    spacing_zyx,
    *,
    shape_dhw: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    spacing_z, spacing_y, spacing_x = [max(float(value), 1e-8) for value in spacing_zyx]
    x_idx = np.floor(points_xyz[:, 0] / spacing_x).astype(np.int64)
    y_idx = np.floor(points_xyz[:, 1] / spacing_y).astype(np.int64)
    z_idx = np.floor(points_xyz[:, 2] / spacing_z).astype(np.int64)
    if shape_dhw is not None:
        depth, height, width = [int(value) for value in shape_dhw]
        x_idx = np.clip(x_idx, 0, max(width - 1, 0))
        y_idx = np.clip(y_idx, 0, max(height - 1, 0))
        z_idx = np.clip(z_idx, 0, max(depth - 1, 0))
    return z_idx, y_idx, x_idx


def voxel_center_world_bounds_torch(volume_shape_dhw, spacing_zyx, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    depth, height, width = [int(value) for value in volume_shape_dhw]
    spacing_z, spacing_y, spacing_x = [float(value) for value in spacing_zyx]
    lower = torch.tensor(
        [
            0.5 * max(spacing_x, 1e-8),
            0.5 * max(spacing_y, 1e-8),
            0.5 * max(spacing_z, 1e-8),
        ],
        dtype=dtype,
        device=device,
    )
    upper = torch.tensor(
        [
            max(float(width) - 0.5, 0.5) * max(spacing_x, 1e-8),
            max(float(height) - 0.5, 0.5) * max(spacing_y, 1e-8),
            max(float(depth) - 0.5, 0.5) * max(spacing_z, 1e-8),
        ],
        dtype=dtype,
        device=device,
    )
    return lower, upper
