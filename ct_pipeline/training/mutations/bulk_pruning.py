from __future__ import annotations

import torch

from ct_pipeline.rendering.fields import density_to_occupancy, query_ct_density_from_state_by_region
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.mutations.helpers import (
    _ensure_gap_seed_birth_iter,
    _sample_binary_mask_nearest,
)
from ct_pipeline.training.sampling import _candidate_count, _sample_occupancy_points


def _bulk_prune_stats(gaussians):
    count = int(gaussians.get_xyz.shape[0]) if getattr(gaussians, "is_initialized", lambda: False)() else 0
    return {
        "pruned": 0,
        "low_opacity": 0,
        "air_center": 0,
        "raw_air_owner": 0,
        "isolated": 0,
        "protected_gap_seed": 0,
        "count_before": count,
        "count_after": count,
    }


def _apply_bulk_pruning(
    gaussians,
    args,
    training_state,
    spacing_zyx,
    field_pools,
    signed_distance_field,
    material_mask=None,
    iteration: int = 0,
):
    stats = _bulk_prune_stats(gaussians)
    if gaussians.optimizer is None or not getattr(gaussians, "is_initialized", lambda: False)():
        return stats
    with torch.no_grad():
        device = gaussians.get_xyz.device
        bulk_mask = gaussians.get_region_type.reshape(-1) == 1
        bulk_indices = torch.nonzero(bulk_mask, as_tuple=False).reshape(-1)
        if bulk_indices.numel() == 0:
            return stats

        gap_birth = _ensure_gap_seed_birth_iter(gaussians)[bulk_indices]
        if bool(getattr(args, "ct_gap_reseed_protect_prune", False)):
            protected_gap_seed = (gap_birth >= 0) & (
                (int(iteration) - gap_birth) < int(getattr(args, "ct_gap_reseed_protect_iters", 300))
            )
        else:
            protected_gap_seed = torch.zeros((bulk_indices.shape[0],), dtype=torch.bool, device=device)
        bulk_opacity = gaussians.get_opacity.reshape(-1)[bulk_indices]
        low_opacity = bulk_opacity < float(getattr(args, "ct_bulk_prune_min_opacity", 0.05))
        stats["low_opacity"] = int(low_opacity.sum().item())

        bulk_xyz = gaussians.get_xyz.detach()[bulk_indices]
        if material_mask is not None:
            center_inside = _sample_binary_mask_nearest(material_mask, bulk_xyz, spacing_zyx)
            air_center = ~center_inside
        else:
            center_sdf = sample_volume_field(
                signed_distance_field["signed_distance"],
                bulk_xyz,
                signed_distance_field["spacing_zyx"],
            ).reshape(-1).to(dtype=torch.float32)
            air_center = torch.isfinite(center_sdf) & (
                center_sdf > float(getattr(args, "ct_bulk_prune_air_sdf_epsilon", 0.5))
            )
        stats["air_center"] = int(air_center.sum().item())

        sample_count = int(getattr(args, "ct_bulk_prune_sample_count", 4096))
        air_candidates = field_pools.get("near_material_air")
        if _candidate_count(air_candidates) == 0:
            air_candidates = field_pools.get("air_shell")
        if sample_count > 0 and _candidate_count(air_candidates) > 0 and bulk_xyz.numel() > 0:
            air_points = _sample_occupancy_points(air_candidates, sample_count, spacing_zyx, device=device)
            if material_mask is not None:
                valid_air = ~_sample_binary_mask_nearest(material_mask, air_points, spacing_zyx)
            else:
                air_sdf = sample_volume_field(
                    signed_distance_field["signed_distance"],
                    air_points,
                    signed_distance_field["spacing_zyx"],
                ).reshape(-1).to(dtype=torch.float32)
                valid_air = torch.isfinite(air_sdf) & (
                    air_sdf > float(getattr(args, "ct_bulk_prune_air_sdf_epsilon", 0.5))
                )
            if torch.any(valid_air):
                air_points = air_points[valid_air]
                bulk_occ = density_to_occupancy(
                    query_ct_density_from_state_by_region(training_state, air_points, region="bulk", detach=True)
                ).to(dtype=torch.float32)
                bad_air = bulk_occ > float(getattr(args, "ct_bulk_prune_raw_air_threshold", 0.35))
                if torch.any(bad_air):
                    owner_points = air_points[bad_air]
                    owner_chunks = []
                    chunk_size = 512
                    for start in range(0, owner_points.shape[0], chunk_size):
                        distances = torch.cdist(owner_points[start : start + chunk_size], bulk_xyz)
                        owner_chunks.append(torch.argmin(distances, dim=1))
                    owners = torch.cat(owner_chunks, dim=0).unique()
                    raw_air_owner = torch.zeros((bulk_indices.shape[0],), dtype=torch.bool, device=device)
                    raw_air_owner[owners] = True
                    stats["raw_air_owner"] = int(raw_air_owner.sum().item())
                else:
                    raw_air_owner = torch.zeros((bulk_indices.shape[0],), dtype=torch.bool, device=device)
            else:
                raw_air_owner = torch.zeros((bulk_indices.shape[0],), dtype=torch.bool, device=device)
        else:
            raw_air_owner = torch.zeros((bulk_indices.shape[0],), dtype=torch.bool, device=device)

        min_neighbors = int(getattr(args, "ct_bulk_prune_min_neighbors", 1))
        if min_neighbors > 0 and 1 < bulk_xyz.shape[0] <= 20000:
            radius = float(getattr(args, "ct_bulk_prune_neighbor_radius", 2.0)) * float(max(spacing_zyx))
            neighbor_counts = torch.zeros((bulk_xyz.shape[0],), dtype=torch.int32, device=device)
            chunk_size = 1024
            for start in range(0, bulk_xyz.shape[0], chunk_size):
                distances = torch.cdist(bulk_xyz[start : start + chunk_size], bulk_xyz)
                neighbor_counts[start : start + chunk_size] = ((distances <= radius) & (distances > 0.0)).sum(dim=1).to(dtype=torch.int32)
            isolated = neighbor_counts < int(min_neighbors)
            stats["isolated"] = int(isolated.sum().item())
        else:
            isolated = torch.zeros((bulk_indices.shape[0],), dtype=torch.bool, device=device)

        illegal = air_center | raw_air_owner
        useless = low_opacity | isolated
        if bool(getattr(args, "ct_gap_reseed_protect_prune", False)):
            prune_bulk = illegal | (useless & ~protected_gap_seed)
        else:
            prune_bulk = illegal | useless
        stats["protected_gap_seed"] = int((protected_gap_seed & useless & ~illegal).sum().item())

        if not torch.any(prune_bulk):
            return stats
        prune_mask = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device=device)
        prune_mask[bulk_indices[prune_bulk]] = True
        birth = _ensure_gap_seed_birth_iter(gaussians)
        setattr(gaussians, "_ct_gap_seed_birth_iter", birth[~prune_mask])
        gaussians.prune_points(prune_mask)
        stats["pruned"] = int(prune_bulk.sum().item())
        stats["count_after"] = int(gaussians.get_xyz.shape[0])
    return stats
