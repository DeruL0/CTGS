from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import skimage
import torch
from plyfile import PlyData, PlyElement

from ct_pipeline.field_query import query_ct_density
from scene import CTGaussianModel


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


def meshing_ct(dataset, model: CTGaussianModel, resolution, threshold):
    del dataset
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
        "coarse_face_count": coarse_face_count,
        "boundary_refined": boundary_refined,
    }


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
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sh_degree", type=int, default=0)
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
    mesh = meshing_ct(None, model, resolution=args.resolution, threshold=args.threshold)
    _write_mesh_ply(Path(args.output), mesh["vertices"], mesh["faces"], mesh["normals"], mesh["material_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
