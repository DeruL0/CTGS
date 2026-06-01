from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import skimage
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from ct_pipeline.rendering.fields import query_ct_density
from scene import CTGaussianModel
from utils.rotation_utils import quaternion_to_matrix


def _ct_model_bbox(model, padding_factor: float = 3.0):
    xyz = model.get_xyz.detach()
    scaling = model.get_scaling.detach()
    if xyz.numel() == 0:
        raise ValueError("Cannot mesh an empty CTGaussianModel.")
    radius = padding_factor * scaling.max(dim=1).values.unsqueeze(1)
    lower = (xyz - radius).min(dim=0).values.detach().cpu().numpy()
    upper = (xyz + radius).max(dim=0).values.detach().cpu().numpy()
    return np.stack((lower, upper), axis=0).astype(np.float32)


def _axes_from_bbox_resolution(bbox, resolution: float):
    bbox = np.asarray(bbox, dtype=np.float32)
    lengths = np.maximum(bbox[1] - bbox[0], 1e-6)
    counts = np.maximum(2, np.ceil(lengths / float(resolution)).astype(np.int32) + 1)
    return [
        np.linspace(float(bbox[0, axis]), float(bbox[1, axis]), int(counts[axis]), dtype=np.float32)
        for axis in range(3)
    ]


def _grid_points_from_axes(axes):
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)


def _density_volume_from_axes(model, axes, return_material_volume: bool = False):
    points = _grid_points_from_axes(axes)
    points_tensor = torch.as_tensor(points, dtype=model.get_xyz.dtype, device=model.get_xyz.device)
    if return_material_volume:
        density, material_volume, material_labels = query_ct_density(model, points_tensor, return_material_volume=True)
        density_volume = density.detach().cpu().numpy().reshape(len(axes[0]), len(axes[1]), len(axes[2]))
        material_grid = material_volume.detach().cpu().numpy().reshape(len(axes[0]), len(axes[1]), len(axes[2]), -1)
        return density_volume, material_grid, material_labels
    density = query_ct_density(model, points_tensor, return_material_volume=False)
    return density.detach().cpu().numpy().reshape(len(axes[0]), len(axes[1]), len(axes[2]))


def _marching_cubes_from_density(density_volume, axes, threshold):
    spacing = tuple(float(axis[1] - axis[0]) if len(axis) > 1 else 1.0 for axis in axes)
    if np.max(density_volume) < threshold:
        return None
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            volume=density_volume,
            level=threshold,
            spacing=spacing,
        )
    except ValueError:
        return None
    vertices = verts + np.array([float(axes[0][0]), float(axes[1][0]), float(axes[2][0])], dtype=np.float32)
    return {
        "vertices": vertices.astype(np.float32),
        "faces": faces.astype(np.int32),
        "normals": normals.astype(np.float32),
        "values": values.astype(np.float32),
    }


def _boundary_mask_from_material_grid(material_grid, density_volume, threshold):
    if material_grid.shape[-1] <= 1:
        return np.zeros_like(density_volume, dtype=bool)

    winner = np.argmax(material_grid, axis=-1)
    strength = np.max(material_grid, axis=-1)
    sorted_density = np.sort(material_grid, axis=-1)
    top1 = sorted_density[..., -1]
    top2 = sorted_density[..., -2]
    ambiguous = (top1 > threshold * 0.25) & ((top1 - top2) < max(0.05, threshold * 0.1))

    boundary = np.zeros_like(winner, dtype=bool)
    for axis in range(3):
        slicer_a = [slice(None)] * 3
        slicer_b = [slice(None)] * 3
        slicer_a[axis] = slice(1, None)
        slicer_b[axis] = slice(None, -1)
        label_change = winner[tuple(slicer_a)] != winner[tuple(slicer_b)]
        strength_mask = (strength[tuple(slicer_a)] > threshold * 0.25) | (strength[tuple(slicer_b)] > threshold * 0.25)
        change_mask = label_change & strength_mask
        boundary[tuple(slicer_a)] |= change_mask
        boundary[tuple(slicer_b)] |= change_mask
    boundary |= ambiguous
    return boundary


def _bbox_from_index_mask(mask, axes, padding_cells: int = 1):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    lower_idx = np.maximum(coords.min(axis=0) - int(padding_cells), 0)
    upper_idx = np.minimum(coords.max(axis=0) + int(padding_cells), np.array(mask.shape) - 1)
    lower = np.array([axes[axis][lower_idx[axis]] for axis in range(3)], dtype=np.float32)
    upper = np.array([axes[axis][upper_idx[axis]] for axis in range(3)], dtype=np.float32)
    return np.stack((lower, upper), axis=0)


def _compact_mesh(vertices, faces, normals):
    if faces.size == 0:
        return vertices, faces, normals
    used = np.unique(faces.reshape(-1))
    remap = -np.ones((vertices.shape[0],), dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    return vertices[used], remap[faces], normals[used]


def estimate_mesh_vertex_material_ids(model, vertices):
    if len(vertices) == 0:
        return np.zeros((0,), dtype=np.int32)
    points_tensor = torch.as_tensor(vertices, dtype=model.get_xyz.dtype, device=model.get_xyz.device)
    _, material_volume, material_labels = query_ct_density(model, points_tensor, return_material_volume=True)
    if material_volume.shape[1] == 0:
        return np.zeros((len(vertices),), dtype=np.int32)
    winning_indices = torch.argmax(material_volume, dim=1).detach().cpu().numpy()
    return material_labels[winning_indices]


def _surface_gaussian_mask(model, opacity_threshold: float):
    opacity = model.get_opacity.reshape(-1)
    if hasattr(model, "get_region_type"):
        region_type = model.get_region_type.reshape(-1)
        surface_mask = region_type == 0
    else:
        surface_mask = torch.ones_like(opacity, dtype=torch.bool)
    return surface_mask & (opacity >= float(opacity_threshold))


def _model_normals(model, count: int, device: torch.device, dtype: torch.dtype):
    if hasattr(model, "get_normals"):
        normals = model.get_normals().detach().to(device=device, dtype=dtype)
        if normals.shape == (count, 3):
            return torch.nn.functional.normalize(normals, dim=-1, eps=1e-8)
    return None


def _fallback_tangent_from_normal(normal: torch.Tensor) -> torch.Tensor:
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=normal.dtype, device=normal.device).expand_as(normal)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=normal.dtype, device=normal.device).expand_as(normal)
    reference = torch.where(torch.abs(normal[:, :1]) < 0.9, x_axis, y_axis)
    tangent = reference - torch.sum(reference * normal, dim=-1, keepdim=True) * normal
    return torch.nn.functional.normalize(tangent, dim=-1, eps=1e-8)


def _orthogonalize_tangent(tangent: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    tangent = tangent - torch.sum(tangent * normal, dim=-1, keepdim=True) * normal
    norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    fallback = _fallback_tangent_from_normal(normal)
    tangent = torch.where(norm > 1e-8, tangent / norm.clamp_min(1e-8), fallback)
    return torch.nn.functional.normalize(tangent, dim=-1, eps=1e-8)


def _surface_points_from_aligned_gaussians(
    model,
    opacity_threshold: float = 0.01,
    tangent_scale: float = 2.0,
    samples_per_gaussian: int = 5,
    max_points: int = 500_000,
    seed: int = 0,
):
    surface_mask = _surface_gaussian_mask(model, opacity_threshold=opacity_threshold)
    if not torch.any(surface_mask):
        raise RuntimeError("SuGaR-style CT meshing found no surface Gaussians above the opacity threshold.")

    xyz = model.get_xyz.detach()[surface_mask]
    scales = model.get_scaling.detach()[surface_mask].clamp_min(1e-8)
    rotations = quaternion_to_matrix(model.get_rotation.detach()[surface_mask])
    opacity = model.get_opacity.detach().reshape(-1)[surface_mask]
    material_id = (
        model.get_material_id.detach().reshape(-1)[surface_mask].to(dtype=torch.long)
        if hasattr(model, "get_material_id")
        else torch.zeros((xyz.shape[0],), dtype=torch.long, device=xyz.device)
    )

    count = int(xyz.shape[0])
    model_normals = _model_normals(model, int(model.get_xyz.shape[0]), xyz.device, xyz.dtype)
    model_normals = model_normals[surface_mask] if model_normals is not None else None

    shortest_axis = torch.argmin(scales, dim=1)
    batch = torch.arange(count, device=xyz.device)
    shortest_normal = rotations[batch, :, shortest_axis]
    shortest_normal = torch.nn.functional.normalize(shortest_normal, dim=-1, eps=1e-8)
    if model_normals is not None:
        normal = torch.nn.functional.normalize(model_normals, dim=-1, eps=1e-8)
        valid_normal = torch.linalg.norm(model_normals, dim=-1, keepdim=True) > 1e-8
        normal = torch.where(valid_normal, normal, shortest_normal)
    else:
        normal = shortest_normal

    tangent_indices = torch.stack(
        [
            torch.where(shortest_axis == 0, torch.ones_like(shortest_axis), torch.zeros_like(shortest_axis)),
            torch.where(shortest_axis == 2, torch.ones_like(shortest_axis), torch.full_like(shortest_axis, 2)),
        ],
        dim=1,
    )
    ambiguous = shortest_axis == 1
    tangent_indices[ambiguous, 0] = 0
    tangent_indices[ambiguous, 1] = 2

    tangent_u = rotations[batch, :, tangent_indices[:, 0]]
    tangent_v = rotations[batch, :, tangent_indices[:, 1]]
    tangent_u = _orthogonalize_tangent(tangent_u, normal)
    tangent_v = torch.cross(normal, tangent_u, dim=-1)
    tangent_v = torch.nn.functional.normalize(tangent_v, dim=-1, eps=1e-8)
    radius_u = scales[batch, tangent_indices[:, 0]].unsqueeze(1) * float(tangent_scale)
    radius_v = scales[batch, tangent_indices[:, 1]].unsqueeze(1) * float(tangent_scale)

    samples = [xyz]
    sample_normals = [normal]
    sample_opacity = [opacity]
    sample_material = [material_id]

    if int(samples_per_gaussian) >= 5:
        offsets = (
            radius_u * tangent_u,
            -radius_u * tangent_u,
            radius_v * tangent_v,
            -radius_v * tangent_v,
        )
        for offset in offsets:
            samples.append(xyz + offset)
            sample_normals.append(normal)
            sample_opacity.append(opacity)
            sample_material.append(material_id)
    elif int(samples_per_gaussian) > 1:
        generator = torch.Generator(device=xyz.device)
        generator.manual_seed(int(seed))
        extra = int(samples_per_gaussian) - 1
        angles = 2.0 * torch.pi * torch.rand((count, extra), generator=generator, device=xyz.device, dtype=xyz.dtype)
        radii = torch.sqrt(torch.rand((count, extra), generator=generator, device=xyz.device, dtype=xyz.dtype))
        random_offsets = (
            torch.cos(angles).unsqueeze(-1) * radii.unsqueeze(-1) * radius_u.unsqueeze(1) * tangent_u.unsqueeze(1)
            + torch.sin(angles).unsqueeze(-1) * radii.unsqueeze(-1) * radius_v.unsqueeze(1) * tangent_v.unsqueeze(1)
        )
        samples.append((xyz.unsqueeze(1) + random_offsets).reshape(-1, 3))
        sample_normals.append(normal.unsqueeze(1).expand(-1, extra, -1).reshape(-1, 3))
        sample_opacity.append(opacity.unsqueeze(1).expand(-1, extra).reshape(-1))
        sample_material.append(material_id.unsqueeze(1).expand(-1, extra).reshape(-1))

    points = torch.cat(samples, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
    normals = torch.cat(sample_normals, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
    weights = torch.cat(sample_opacity, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
    materials = torch.cat(sample_material, dim=0).detach().cpu().numpy().astype(np.int32, copy=False)

    if points.shape[0] > int(max_points) > 0:
        rng = np.random.default_rng(int(seed))
        probabilities = np.maximum(weights, 1e-6)
        probabilities /= probabilities.sum()
        indices = rng.choice(points.shape[0], size=int(max_points), replace=False, p=probabilities)
        points = points[indices]
        normals = normals[indices]
        materials = materials[indices]

    return points, normals, materials


def _clean_open3d_mesh(o3d_mesh):
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_non_manifold_edges()
    return o3d_mesh


def _open3d_mesh_to_arrays(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float32)
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int32)
    if not o3d_mesh.has_vertex_normals():
        o3d_mesh.compute_vertex_normals()
    normals = np.asarray(o3d_mesh.vertex_normals, dtype=np.float32)
    return vertices, faces, normals


def _poisson_mesh_from_surface_points(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    poisson_depth: int = 9,
    density_quantile: float = 0.0,
    normal_consistency_k: int = 30,
    outlier_neighbors: int = 20,
    outlier_std_ratio: float = 20.0,
):
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SuGaR-style mesh extraction requires open3d. Install open3d or use --method density.") from exc

    if points.shape[0] < 50:
        raise RuntimeError("SuGaR-style mesh extraction needs at least 50 oriented surface points.")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    point_cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))

    if int(outlier_neighbors) > 0 and points.shape[0] > int(outlier_neighbors):
        _, kept_indices = point_cloud.remove_statistical_outlier(
            nb_neighbors=int(outlier_neighbors),
            std_ratio=float(outlier_std_ratio),
        )
        point_cloud = point_cloud.select_by_index(kept_indices)

    if len(point_cloud.points) < 50:
        raise RuntimeError("SuGaR-style mesh extraction removed too many points during outlier filtering.")

    if int(normal_consistency_k) > 0 and len(point_cloud.points) > int(normal_consistency_k):
        try:
            point_cloud.orient_normals_consistent_tangent_plane(int(normal_consistency_k))
        except RuntimeError:
            pass

    o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=int(poisson_depth),
    )
    densities = np.asarray(densities)
    if float(density_quantile) > 0.0 and densities.size > 0:
        vertices_to_remove = densities < np.quantile(densities, float(density_quantile))
        o3d_mesh.remove_vertices_by_mask(vertices_to_remove)
    o3d_mesh.compute_vertex_normals()
    return _clean_open3d_mesh(o3d_mesh)


def _nearest_surface_material_ids(vertices, points, material_id):
    if vertices.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    if points.shape[0] == 0 or material_id.shape[0] != points.shape[0]:
        return np.zeros((vertices.shape[0],), dtype=np.int32)
    nearest = cKDTree(points).query(vertices, k=1)[1]
    return material_id[nearest].astype(np.int32, copy=False)


def _project_mesh_vertices_to_surface_points(vertices, faces, points, normals, material_id):
    if vertices.shape[0] == 0 or points.shape[0] == 0:
        return vertices, faces, normals, np.zeros((vertices.shape[0],), dtype=np.int32)
    nearest = cKDTree(points).query(vertices, k=1)[1]
    projected_vertices = points[nearest].astype(np.float32, copy=False)
    projected_normals = normals[nearest].astype(np.float32, copy=False)
    projected_material = material_id[nearest].astype(np.int32, copy=False)
    compact_vertices, compact_faces, compact_normals = _compact_mesh(
        projected_vertices,
        faces.astype(np.int32, copy=False),
        projected_normals,
    )
    used = np.unique(faces.reshape(-1)) if faces.size > 0 else np.arange(projected_vertices.shape[0])
    compact_material = projected_material[used] if used.size > 0 else np.zeros((0,), dtype=np.int32)
    return compact_vertices, compact_faces, compact_normals, compact_material


def _filter_small_mesh_components(vertices, faces, normals, material_id, min_face_ratio: float):
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if faces.shape[0] == 0 or float(min_face_ratio) <= 0.0:
        return vertices, faces, normals, material_id

    vertex_count = int(np.asarray(vertices).shape[0])
    parent = np.arange(vertex_count, dtype=np.int32)
    rank = np.zeros((vertex_count,), dtype=np.int8)

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return int(root)

    def union(a: int, b: int) -> None:
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for face in faces:
        union(int(face[0]), int(face[1]))
        union(int(face[1]), int(face[2]))
        union(int(face[2]), int(face[0]))

    face_roots = np.asarray([find(int(face[0])) for face in faces], dtype=np.int32)
    roots, counts = np.unique(face_roots, return_counts=True)
    threshold = max(1, int(round(float(min_face_ratio) * faces.shape[0])))
    keep_roots = roots[counts >= threshold]
    if keep_roots.size == 0:
        keep_roots = roots[np.asarray([int(np.argmax(counts))], dtype=np.int64)]

    keep_faces = faces[np.isin(face_roots, keep_roots)]
    compact_vertices, compact_faces, compact_normals = _compact_mesh(vertices, keep_faces, normals)
    used = np.unique(keep_faces.reshape(-1)) if keep_faces.size > 0 else np.zeros((0,), dtype=np.int32)
    compact_material = np.asarray(material_id, dtype=np.int32).reshape(-1)[used] if used.size > 0 else np.zeros((0,), dtype=np.int32)
    return compact_vertices, compact_faces, compact_normals, compact_material


def _meshing_ct_sugar(
    model: CTGaussianModel,
    *,
    fallback_resolution: float,
    threshold: float,
    opacity_threshold: float = 0.01,
    tangent_scale: float = 2.0,
    samples_per_gaussian: int = 5,
    max_points: int = 500_000,
    poisson_depth: int = 9,
    density_quantile: float = 0.0,
    normal_consistency_k: int = 30,
    component_min_face_ratio: float = 0.005,
    project_on_surface_points: bool = False,
    seed: int = 0,
):
    points, point_normals, point_material = _surface_points_from_aligned_gaussians(
        model,
        opacity_threshold=opacity_threshold,
        tangent_scale=tangent_scale,
        samples_per_gaussian=samples_per_gaussian,
        max_points=max_points,
        seed=seed,
    )
    o3d_mesh = _poisson_mesh_from_surface_points(
        points,
        point_normals,
        poisson_depth=poisson_depth,
        density_quantile=density_quantile,
        normal_consistency_k=normal_consistency_k,
    )
    vertices, faces, normals = _open3d_mesh_to_arrays(o3d_mesh)
    if project_on_surface_points:
        vertices, faces, normals, material_id = _project_mesh_vertices_to_surface_points(
            vertices,
            faces,
            points,
            point_normals,
            point_material,
        )
    else:
        material_id = _nearest_surface_material_ids(vertices, points, point_material)
    vertices, faces, normals, material_id = _filter_small_mesh_components(
        vertices,
        faces,
        normals,
        material_id,
        min_face_ratio=component_min_face_ratio,
    )
    return {
        "vertices": vertices.astype(np.float32),
        "faces": faces.astype(np.int32),
        "normals": normals.astype(np.float32),
        "material_id": material_id.astype(np.int32),
        "threshold": float(threshold),
        "resolution": float(fallback_resolution),
        "method": "sugar",
        "surface_point_count": int(points.shape[0]),
        "poisson_depth": int(poisson_depth),
        "density_quantile": float(density_quantile),
        "normal_consistency_k": int(normal_consistency_k),
        "component_min_face_ratio": float(component_min_face_ratio),
        "boundary_refined": False,
        "coarse_face_count": int(faces.shape[0]),
    }


def _meshing_ct_density(model: CTGaussianModel, resolution, threshold):
    bbox = _ct_model_bbox(model)
    coarse_axes = _axes_from_bbox_resolution(bbox, resolution)
    coarse_density, coarse_material_grid, _ = _density_volume_from_axes(model, coarse_axes, return_material_volume=True)
    coarse_mesh = _marching_cubes_from_density(coarse_density, coarse_axes, threshold)
    if coarse_mesh is None:
        raise RuntimeError("CT meshing failed: no surface extracted from the requested threshold.")

    coarse_face_count = int(coarse_mesh["faces"].shape[0])
    boundary_mask = _boundary_mask_from_material_grid(coarse_material_grid, coarse_density, threshold)
    boundary_bbox = _bbox_from_index_mask(boundary_mask, coarse_axes, padding_cells=1)

    merged_vertices = coarse_mesh["vertices"]
    merged_faces = coarse_mesh["faces"]
    merged_normals = coarse_mesh["normals"]
    boundary_refined = False

    if boundary_bbox is not None:
        fine_axes = _axes_from_bbox_resolution(boundary_bbox, float(resolution) * 0.5)
        fine_density = _density_volume_from_axes(model, fine_axes, return_material_volume=False)
        fine_mesh = _marching_cubes_from_density(fine_density, fine_axes, threshold)
        if fine_mesh is not None and fine_mesh["faces"].size > 0:
            face_centroids = merged_vertices[merged_faces].mean(axis=1)
            inside_boundary = np.all(
                (face_centroids >= boundary_bbox[0][None, :]) & (face_centroids <= boundary_bbox[1][None, :]),
                axis=1,
            )
            kept_faces = merged_faces[~inside_boundary]
            kept_vertices, kept_faces, kept_normals = _compact_mesh(merged_vertices, kept_faces, merged_normals)
            offset = kept_vertices.shape[0]
            merged_vertices = np.concatenate((kept_vertices, fine_mesh["vertices"]), axis=0)
            merged_normals = np.concatenate((kept_normals, fine_mesh["normals"]), axis=0)
            merged_faces = np.concatenate((kept_faces, fine_mesh["faces"] + offset), axis=0)
            boundary_refined = True

    material_id = estimate_mesh_vertex_material_ids(model, merged_vertices)
    return {
        "vertices": merged_vertices.astype(np.float32),
        "faces": merged_faces.astype(np.int32),
        "normals": merged_normals.astype(np.float32),
        "material_id": material_id.astype(np.int32),
        "threshold": float(threshold),
        "resolution": float(resolution),
        "method": "density",
        "coarse_face_count": coarse_face_count,
        "boundary_refined": boundary_refined,
    }


def meshing_ct(
    dataset,
    model: CTGaussianModel,
    resolution,
    threshold,
    method: str = "sugar",
    sugar_opacity_threshold: float = 0.01,
    sugar_tangent_scale: float = 2.0,
    sugar_samples_per_gaussian: int = 5,
    sugar_max_points: int = 500_000,
    sugar_poisson_depth: int = 9,
    sugar_density_quantile: float = 0.0,
    sugar_normal_consistency_k: int = 30,
    sugar_component_min_face_ratio: float = 0.005,
    sugar_project_on_surface_points: bool = False,
    seed: int = 0,
):
    del dataset
    method = str(method).lower()
    if method in ("density", "marching_cubes", "mc"):
        return _meshing_ct_density(model, resolution=resolution, threshold=threshold)
    if method not in ("sugar", "sugar_poisson", "poisson"):
        raise ValueError("method must be 'sugar' or 'density'.")
    return _meshing_ct_sugar(
        model,
        fallback_resolution=resolution,
        threshold=threshold,
        opacity_threshold=sugar_opacity_threshold,
        tangent_scale=sugar_tangent_scale,
        samples_per_gaussian=sugar_samples_per_gaussian,
        max_points=sugar_max_points,
        poisson_depth=sugar_poisson_depth,
        density_quantile=sugar_density_quantile,
        normal_consistency_k=sugar_normal_consistency_k,
        component_min_face_ratio=sugar_component_min_face_ratio,
        project_on_surface_points=sugar_project_on_surface_points,
        seed=seed,
    )


def _write_mesh_ply(path, vertices, faces, normals, material_id):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("material_id", "i4"),
    ]
    vertex_array = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_array["x"] = vertices[:, 0]
    vertex_array["y"] = vertices[:, 1]
    vertex_array["z"] = vertices[:, 2]
    vertex_array["nx"] = normals[:, 0]
    vertex_array["ny"] = normals[:, 1]
    vertex_array["nz"] = normals[:, 2]
    vertex_array["material_id"] = material_id.reshape(-1)

    face_array = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_array["vertex_indices"] = faces
    PlyData(
        [
            PlyElement.describe(vertex_array, "vertex"),
            PlyElement.describe(face_array, "face"),
        ]
    ).write(str(path))
    return path


def _find_point_cloud_ply(model_path: Path, iteration: int) -> Path:
    point_cloud_dir = model_path / "point_cloud"
    if not point_cloud_dir.exists():
        raise FileNotFoundError(f"No point_cloud directory found under {model_path}.")
    if iteration >= 0:
        candidate = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing point cloud PLY at {candidate}.")
        return candidate
    iterations = sorted(
        int(path.name.split("_")[-1])
        for path in point_cloud_dir.glob("iteration_*")
        if path.is_dir() and path.name.split("_")[-1].isdigit()
    )
    if not iterations:
        raise FileNotFoundError(f"No iteration_* folders found under {point_cloud_dir}.")
    return point_cloud_dir / f"iteration_{iterations[-1]}" / "point_cloud.ply"


def build_parser():
    parser = argparse.ArgumentParser(description="Extract a CT mesh from a CTGaussianModel PLY or checkpoint directory.")
    parser.add_argument("--input", required=True, help="Path to a hybrid GS PLY or a CT training output directory.")
    parser.add_argument("--output", required=True, help="Output PLY mesh path.")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to read when --input is a training output directory.")
    parser.add_argument("--method", choices=("sugar", "density"), default="sugar")
    parser.add_argument("--resolution", type=float, default=0.5, help="Density fallback / marching-cubes grid spacing.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument("--sugar_opacity_threshold", type=float, default=0.01)
    parser.add_argument("--sugar_tangent_scale", type=float, default=2.0)
    parser.add_argument("--sugar_samples_per_gaussian", type=int, default=5)
    parser.add_argument("--sugar_max_points", type=int, default=500_000)
    parser.add_argument("--sugar_poisson_depth", type=int, default=9)
    parser.add_argument("--sugar_density_quantile", type=float, default=0.0)
    parser.add_argument("--sugar_normal_consistency_k", type=int, default=30)
    parser.add_argument("--sugar_component_min_face_ratio", type=float, default=0.005)
    parser.add_argument("--sugar_project_on_surface_points", action="store_true", default=False)
    parser.add_argument("--no_sugar_project_on_surface_points", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    model = CTGaussianModel(sh_degree=args.sh_degree)

    if input_path.is_dir():
        ply_path = _find_point_cloud_ply(input_path, args.iteration)
    else:
        ply_path = input_path
    model.load_ply(str(ply_path))
    mesh = meshing_ct(
        None,
        model,
        resolution=args.resolution,
        threshold=args.threshold,
        method=args.method,
        sugar_opacity_threshold=args.sugar_opacity_threshold,
        sugar_tangent_scale=args.sugar_tangent_scale,
        sugar_samples_per_gaussian=args.sugar_samples_per_gaussian,
        sugar_max_points=args.sugar_max_points,
        sugar_poisson_depth=args.sugar_poisson_depth,
        sugar_density_quantile=args.sugar_density_quantile,
        sugar_normal_consistency_k=args.sugar_normal_consistency_k,
        sugar_component_min_face_ratio=args.sugar_component_min_face_ratio,
        sugar_project_on_surface_points=bool(args.sugar_project_on_surface_points) and not bool(args.no_sugar_project_on_surface_points),
        seed=args.seed,
    )
    _write_mesh_ply(Path(args.output), mesh["vertices"], mesh["faces"], mesh["normals"], mesh["material_id"])
    print_parts = [
        "method={0}".format(mesh.get("method", args.method)),
        "vertices={0}".format(int(mesh["vertices"].shape[0])),
        "faces={0}".format(int(mesh["faces"].shape[0])),
    ]
    if "surface_point_count" in mesh:
        print_parts.append("surface_points={0}".format(int(mesh["surface_point_count"])))
    print("CT mesh extracted with " + ", ".join(print_parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
