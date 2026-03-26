from __future__ import annotations

import numpy as np
import torch

from utils.rotation_utils import quaternion_to_matrix


def _as_query_points(points_xyz, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not isinstance(points_xyz, torch.Tensor):
        points_xyz = torch.as_tensor(points_xyz, dtype=dtype, device=device)
    else:
        points_xyz = points_xyz.to(device=device, dtype=dtype)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3).")
    return points_xyz


def query_ct_density_python(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    points_xyz,
    return_material_volume: bool = False,
    material_ids: torch.Tensor | None = None,
    chunk_size: int = 32768,
):
    points_xyz = _as_query_points(points_xyz, means.dtype, means.device)

    if points_xyz.numel() == 0:
        empty = torch.zeros((0,), dtype=means.dtype, device=means.device)
        if not return_material_volume:
            return empty
        return empty, empty.reshape(0, 0), np.zeros((0,), dtype=np.int32)

    if means.numel() == 0:
        empty = torch.zeros((points_xyz.shape[0],), dtype=points_xyz.dtype, device=points_xyz.device)
        if not return_material_volume:
            return empty
        return empty, empty.reshape(points_xyz.shape[0], 0), np.zeros((0,), dtype=np.int32)

    total_density = torch.zeros((points_xyz.shape[0],), dtype=means.dtype, device=means.device)

    material_volume = None
    material_labels = np.zeros((0,), dtype=np.int32)
    if return_material_volume:
        if material_ids is None:
            material_ids = torch.zeros((means.shape[0],), dtype=torch.long, device=means.device)
        else:
            material_ids = torch.as_tensor(material_ids, device=means.device, dtype=torch.long).reshape(-1)
        valid_materials = material_ids[material_ids >= 0]
        if valid_materials.numel() == 0:
            material_labels = np.asarray([0], dtype=np.int32)
            material_masks = [torch.ones_like(material_ids, dtype=torch.bool)]
        else:
            unique_materials = torch.unique(valid_materials)
            material_labels = unique_materials.detach().cpu().numpy().astype(np.int32)
            material_masks = [(material_ids == material).reshape(-1) for material in unique_materials]
        material_volume = torch.zeros((points_xyz.shape[0], len(material_masks)), dtype=means.dtype, device=means.device)
    else:
        material_masks = []

    gaussians_per_chunk = max(1, int(chunk_size // max(1, points_xyz.shape[0])))
    for start in range(0, means.shape[0], gaussians_per_chunk):
        end = min(start + gaussians_per_chunk, means.shape[0])
        mean_chunk = means[start:end]
        rotation_chunk = rotations[start:end]
        scale_chunk = scales[start:end]
        opacity_chunk = opacity[start:end]

        diff = points_xyz.unsqueeze(1) - mean_chunk.unsqueeze(0)
        local = torch.einsum("qci,cij->qcj", diff, rotation_chunk)
        normalized = local / scale_chunk.unsqueeze(0)
        exponent = -0.5 * torch.sum(normalized * normalized, dim=-1)
        chunk_density = torch.exp(exponent) * opacity_chunk.unsqueeze(0)
        total_density += chunk_density.sum(dim=1)

        if return_material_volume:
            global_indices = torch.arange(start, end, device=means.device)
            for material_index, material_mask in enumerate(material_masks):
                local_mask = material_mask[global_indices]
                if torch.any(local_mask):
                    material_volume[:, material_index] += chunk_density[:, local_mask].sum(dim=1)

    if not return_material_volume:
        return total_density
    return total_density, material_volume, material_labels


def query_ct_density(
    model,
    points_xyz,
    return_material_volume: bool = False,
    chunk_size: int = 32768,
):
    means = model.get_xyz
    points_xyz = _as_query_points(points_xyz, means.dtype, means.device)

    if return_material_volume:
        return query_ct_density_python(
            means,
            quaternion_to_matrix(model.get_rotation),
            model.get_scaling.clamp_min(1e-6),
            model.get_opacity.squeeze(-1),
            points_xyz,
            return_material_volume=True,
            material_ids=model.get_material_id.reshape(-1),
            chunk_size=chunk_size,
        )

    return query_ct_density_python(
        means,
        quaternion_to_matrix(model.get_rotation),
        model.get_scaling.clamp_min(1e-6),
        model.get_opacity.squeeze(-1),
        points_xyz,
        return_material_volume=False,
        chunk_size=chunk_size,
    )


def query_ct_density_backend(
    backend: str,
    model,
    points_xyz,
    chunk_size: int = 32768,
):
    means = model.get_xyz
    points_xyz = _as_query_points(points_xyz, means.dtype, means.device)
    rotations = quaternion_to_matrix(model.get_rotation)
    scales = model.get_scaling.clamp_min(1e-6)
    opacity = model.get_opacity.squeeze(-1)

    if backend == "cuda":
        from ct_pipeline.native_backend import query_ct_density_backend as _query_ct_density_backend_impl

        return _query_ct_density_backend_impl(
            backend,
            means,
            rotations,
            scales,
            opacity,
            points_xyz,
        )

    return query_ct_density_python(
        means,
        rotations,
        scales,
        opacity,
        points_xyz,
        return_material_volume=False,
        chunk_size=chunk_size,
    )


def density_to_occupancy(density: torch.Tensor) -> torch.Tensor:
    density = torch.as_tensor(density)
    return 1.0 - torch.exp(-density.clamp_min(0.0))
