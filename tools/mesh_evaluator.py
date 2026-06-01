from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import skimage.measure
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.spatial import cKDTree

from ct_pipeline.data import build_support_signed_distance
from ct_pipeline.training.utils import write_key_value_report


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    material_id: np.ndarray | None = None


@dataclass
class Phase1Reference:
    phase1_dir: Path
    support_mask: np.ndarray
    spacing_zyx: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    reference_points: np.ndarray
    reference_mesh: MeshData | None
    signed_distance: np.ndarray


def _read_mesh_ply(path: Path) -> MeshData:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = ply["vertex"].data
    required = ("x", "y", "z")
    missing = [name for name in required if name not in vertex.dtype.names]
    if missing:
        raise ValueError(f"PLY vertex element is missing fields: {', '.join(missing)}")

    vertices = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float32, copy=False)
    normals = None
    if all(name in vertex.dtype.names for name in ("nx", "ny", "nz")):
        normals = np.column_stack((vertex["nx"], vertex["ny"], vertex["nz"])).astype(np.float32, copy=False)

    material_id = None
    if "material_id" in vertex.dtype.names:
        material_id = np.asarray(vertex["material_id"], dtype=np.int32).reshape(-1)

    faces = np.zeros((0, 3), dtype=np.int32)
    if "face" in ply:
        face_data = ply["face"].data
        if "vertex_indices" in face_data.dtype.names:
            faces = np.asarray([tuple(indices[:3]) for indices in face_data["vertex_indices"]], dtype=np.int32)
    return MeshData(vertices=vertices, faces=faces, normals=normals, material_id=material_id)


def _write_mesh_ply(path: Path, mesh: MeshData) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    normals = mesh.normals
    if normals is None or np.asarray(normals).shape != vertices.shape:
        normals = np.zeros_like(vertices, dtype=np.float32)
    else:
        normals = np.asarray(normals, dtype=np.float32)
    material_id = mesh.material_id
    if material_id is None or np.asarray(material_id).shape[0] != vertices.shape[0]:
        material_id = np.zeros((vertices.shape[0],), dtype=np.int32)
    else:
        material_id = np.asarray(material_id, dtype=np.int32).reshape(-1)

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("material_id", "i4"),
    ]
    vertex_array = np.empty(vertices.shape[0], dtype=vertex_dtype)
    vertex_array["x"] = vertices[:, 0]
    vertex_array["y"] = vertices[:, 1]
    vertex_array["z"] = vertices[:, 2]
    vertex_array["nx"] = normals[:, 0]
    vertex_array["ny"] = normals[:, 1]
    vertex_array["nz"] = normals[:, 2]
    vertex_array["material_id"] = material_id

    face_array = np.empty(mesh.faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    face_array["vertex_indices"] = np.asarray(mesh.faces, dtype=np.int32).reshape(-1, 3)
    PlyData([PlyElement.describe(vertex_array, "vertex"), PlyElement.describe(face_array, "face")]).write(str(path))
    return path


def _load_phase1_bundle(phase1_dir: Path) -> tuple[dict, dict]:
    phase1_dir = Path(phase1_dir)
    analysis_path = phase1_dir / "analysis.npz"
    metadata_path = phase1_dir / "metadata.json"
    if not analysis_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Phase1 bundle must contain analysis.npz and metadata.json: {phase1_dir}")
    with np.load(str(analysis_path)) as analysis_npz:
        analysis = {key: analysis_npz[key] for key in analysis_npz.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return analysis, metadata


def _support_mask_from_analysis(analysis: dict) -> np.ndarray:
    if "coarse_support_mask" in analysis:
        support = analysis["coarse_support_mask"]
    elif "material_mask" in analysis:
        support = analysis["material_mask"]
    else:
        raise ValueError("Phase1 analysis must contain coarse_support_mask or material_mask.")
    support = np.asarray(support, dtype=bool)
    if support.ndim != 3:
        raise ValueError("Phase1 support mask must have shape (D, H, W).")
    return support


def _metadata_spacing_zyx(metadata: dict) -> tuple[float, float, float]:
    spacing = metadata.get("spacing_zyx", metadata.get("spacing", (1.0, 1.0, 1.0)))
    if len(spacing) != 3:
        raise ValueError("Phase1 metadata spacing_zyx must have three values.")
    return tuple(float(value) for value in spacing)


def _metadata_origin_xyz(metadata: dict) -> tuple[float, float, float]:
    origin = metadata.get("origin_xyz", (0.0, 0.0, 0.0))
    if len(origin) != 3:
        raise ValueError("Phase1 metadata origin_xyz must have three values.")
    return tuple(float(value) for value in origin)


def mesh_from_support_mask(
    support_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MeshData:
    support_float = np.asarray(support_mask, dtype=np.float32)
    if not np.any(support_float > 0.5):
        raise ValueError("Cannot extract a reference mesh from an empty support mask.")
    if np.all(support_float > 0.5):
        raise ValueError("Cannot extract a reference mesh from a full support mask.")

    verts_zyx, faces, normals_zyx, _ = skimage.measure.marching_cubes(
        support_float,
        level=0.5,
        spacing=tuple(float(value) for value in spacing_zyx),
    )
    origin = np.asarray(origin_xyz, dtype=np.float32)
    vertices = verts_zyx[:, [2, 1, 0]].astype(np.float32, copy=False) + origin[None, :]
    normals = normals_zyx[:, [2, 1, 0]].astype(np.float32, copy=False)
    material_id = np.zeros((vertices.shape[0],), dtype=np.int32)
    return MeshData(vertices=vertices, faces=faces.astype(np.int32), normals=normals, material_id=material_id)


def _sample_mesh_surface(mesh: MeshData, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1, 3)
    sample_count = int(sample_count)
    if sample_count <= 0 or faces.shape[0] == 0:
        return vertices.copy()

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(area) & (area > 1e-12)
    if not np.any(valid):
        return vertices.copy()

    valid_triangles = triangles[valid]
    area = area[valid]
    probabilities = area / np.sum(area)
    chosen = rng.choice(valid_triangles.shape[0], size=sample_count, replace=True, p=probabilities)
    tri = valid_triangles[chosen]
    u = rng.random(sample_count, dtype=np.float32)
    v = rng.random(sample_count, dtype=np.float32)
    reflected = (u + v) > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    samples = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    return samples.astype(np.float32, copy=False)


def _downsample_points(points: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    max_count = int(max_count)
    if max_count <= 0 or points.shape[0] <= max_count:
        return points
    indices = rng.choice(points.shape[0], size=max_count, replace=False)
    return points[indices]


def load_phase1_reference(
    phase1_dir: Path,
    reference_sample_count: int,
    rng: np.random.Generator,
    prefer_boundary_points: bool = True,
) -> Phase1Reference:
    analysis, metadata = _load_phase1_bundle(Path(phase1_dir))
    support_mask = _support_mask_from_analysis(analysis)
    spacing_zyx = _metadata_spacing_zyx(metadata)
    origin_xyz = _metadata_origin_xyz(metadata)
    signed_distance = build_support_signed_distance(support_mask, spacing_zyx)

    reference_mesh = None
    reference_points = None
    if prefer_boundary_points and "boundary_points" in analysis:
        boundary_points = np.asarray(analysis["boundary_points"], dtype=np.float32).reshape(-1, 3)
        if boundary_points.shape[0] > 0:
            reference_points = _downsample_points(boundary_points, reference_sample_count, rng)

    if reference_points is None:
        reference_mesh = mesh_from_support_mask(support_mask, spacing_zyx, origin_xyz)
        reference_points = _sample_mesh_surface(reference_mesh, reference_sample_count, rng)

    return Phase1Reference(
        phase1_dir=Path(phase1_dir),
        support_mask=support_mask,
        spacing_zyx=spacing_zyx,
        origin_xyz=origin_xyz,
        reference_points=reference_points.astype(np.float32, copy=False),
        reference_mesh=reference_mesh,
        signed_distance=signed_distance,
    )


def _sample_zyx_volume(
    volume_zyx: np.ndarray,
    points_xyz: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    *,
    order: int,
    cval: float,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    sx = float(spacing_zyx[2])
    sy = float(spacing_zyx[1])
    sz = float(spacing_zyx[0])
    ox, oy, oz = [float(value) for value in origin_xyz]
    coords = np.vstack(
        (
            (points[:, 2] - oz) / max(sz, 1e-8),
            (points[:, 1] - oy) / max(sy, 1e-8),
            (points[:, 0] - ox) / max(sx, 1e-8),
        )
    )
    return ndimage.map_coordinates(
        np.asarray(volume_zyx, dtype=np.float32),
        coords,
        order=order,
        mode="constant",
        cval=float(cval),
    ).astype(np.float32, copy=False)


def _quantiles(values: np.ndarray, quantiles: Iterable[float]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{int(round(q * 100)):02d}": float("nan") for q in quantiles}
    measured = np.quantile(values, list(quantiles))
    return {f"p{int(round(q * 100)):02d}": float(value) for q, value in zip(quantiles, measured)}


def _distance_metrics(prefix: str, distances: np.ndarray) -> dict[str, float]:
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_rms": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    quantiles = _quantiles(distances, (0.50, 0.90, 0.95, 0.99))
    return {
        f"{prefix}_mean": float(np.mean(distances)),
        f"{prefix}_rms": float(np.sqrt(np.mean(distances * distances))),
        f"{prefix}_p50": quantiles["p50"],
        f"{prefix}_p90": quantiles["p90"],
        f"{prefix}_p95": quantiles["p95"],
        f"{prefix}_p99": quantiles["p99"],
        f"{prefix}_max": float(np.max(distances)),
    }


def _triangle_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if faces.shape[0] == 0:
        return 0.0
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _mesh_component_metrics(vertex_count: int, faces: np.ndarray) -> dict[str, float | int]:
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if vertex_count <= 0:
        return {"component_count": 0, "largest_component_face_ratio": 0.0, "small_component_count": 0}
    if faces.shape[0] == 0:
        return {"component_count": int(vertex_count), "largest_component_face_ratio": 0.0, "small_component_count": int(vertex_count)}

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
    _, face_counts = np.unique(face_roots, return_counts=True)
    component_count = int(face_counts.shape[0])
    largest_faces = int(face_counts.max()) if face_counts.size else 0
    small_threshold = max(1, int(round(0.01 * faces.shape[0])))
    return {
        "component_count": component_count,
        "largest_component_face_ratio": float(largest_faces / max(faces.shape[0], 1)),
        "small_component_count": int(np.sum(face_counts <= small_threshold)),
    }


def _mesh_basic_metrics(mesh: MeshData) -> dict[str, float | int]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1, 3)
    metrics: dict[str, float | int] = {
        "mesh_vertex_count": int(vertices.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
        "mesh_area": _triangle_area(vertices, faces),
    }
    if vertices.shape[0] > 0:
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        metrics.update(
            {
                "mesh_bbox_min_x": float(lower[0]),
                "mesh_bbox_min_y": float(lower[1]),
                "mesh_bbox_min_z": float(lower[2]),
                "mesh_bbox_max_x": float(upper[0]),
                "mesh_bbox_max_y": float(upper[1]),
                "mesh_bbox_max_z": float(upper[2]),
            }
        )
    metrics.update(_mesh_component_metrics(int(vertices.shape[0]), faces))
    return metrics


def evaluate_mesh_extraction(
    mesh: MeshData,
    phase1_dir: Path,
    *,
    sample_count: int = 200_000,
    reference_sample_count: int = 200_000,
    seed: int = 0,
    outside_tolerance: float = 0.0,
    boundary_band_voxels: float = 1.0,
    prefer_boundary_points: bool = True,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(int(seed))
    reference = load_phase1_reference(
        Path(phase1_dir),
        reference_sample_count=reference_sample_count,
        rng=rng,
        prefer_boundary_points=prefer_boundary_points,
    )
    mesh_samples = _sample_mesh_surface(mesh, sample_count=sample_count, rng=rng)
    ref_samples = np.asarray(reference.reference_points, dtype=np.float32).reshape(-1, 3)

    if mesh_samples.shape[0] == 0:
        raise ValueError("Predicted mesh has no sampleable surface points.")
    if ref_samples.shape[0] == 0:
        raise ValueError("Phase1 reference has no surface points.")

    ref_tree = cKDTree(ref_samples)
    pred_to_ref, _ = ref_tree.query(mesh_samples, k=1)
    pred_tree = cKDTree(mesh_samples)
    ref_to_pred, _ = pred_tree.query(ref_samples, k=1)

    max_positive_sdf = float(np.nanmax(reference.signed_distance)) if reference.signed_distance.size else 0.0
    sdf_cval = max(max_positive_sdf, 0.0) + max(reference.spacing_zyx)
    signed_sdf = _sample_zyx_volume(
        reference.signed_distance,
        mesh_samples,
        reference.spacing_zyx,
        reference.origin_xyz,
        order=1,
        cval=sdf_cval,
    )
    support_values = _sample_zyx_volume(
        reference.support_mask.astype(np.float32),
        mesh_samples,
        reference.spacing_zyx,
        reference.origin_xyz,
        order=1,
        cval=0.0,
    )
    boundary_band = max(float(boundary_band_voxels), 0.0) * float(min(reference.spacing_zyx))
    outside = signed_sdf > float(outside_tolerance)
    inside = signed_sdf < -float(outside_tolerance)
    near_boundary = np.abs(signed_sdf) <= max(boundary_band, 1e-8)
    outside_distances = np.maximum(signed_sdf, 0.0)

    metrics: dict[str, float | int | str] = {
        "phase1_dir": str(Path(phase1_dir).resolve()),
        "spacing_z": float(reference.spacing_zyx[0]),
        "spacing_y": float(reference.spacing_zyx[1]),
        "spacing_x": float(reference.spacing_zyx[2]),
        "mesh_sample_count": int(mesh_samples.shape[0]),
        "reference_sample_count": int(ref_samples.shape[0]),
        "support_voxel_count": int(np.count_nonzero(reference.support_mask)),
        "mesh_sample_outside_support_ratio": float(np.mean(outside)),
        "mesh_sample_inside_support_ratio": float(np.mean(inside)),
        "mesh_sample_near_boundary_ratio": float(np.mean(near_boundary)),
        "mesh_sample_support_value_mean": float(np.mean(support_values)),
    }
    metrics.update(_mesh_basic_metrics(mesh))
    metrics.update(_distance_metrics("pred_to_ref_distance", pred_to_ref))
    metrics.update(_distance_metrics("ref_to_pred_distance", ref_to_pred))
    metrics.update(_distance_metrics("mesh_signed_sdf", signed_sdf))
    metrics.update(_distance_metrics("mesh_outside_distance", outside_distances))
    metrics["symmetric_chamfer_l1_mean"] = float(0.5 * (np.mean(pred_to_ref) + np.mean(ref_to_pred)))
    metrics["symmetric_chamfer_l2_mean"] = float(0.5 * (np.mean(pred_to_ref * pred_to_ref) + np.mean(ref_to_pred * ref_to_pred)))
    metrics["symmetric_hausdorff_distance"] = float(max(np.max(pred_to_ref), np.max(ref_to_pred)))
    metrics["symmetric_hausdorff_p95"] = float(
        max(np.quantile(pred_to_ref, 0.95), np.quantile(ref_to_pred, 0.95))
    )
    metrics["symmetric_hausdorff_p99"] = float(
        max(np.quantile(pred_to_ref, 0.99), np.quantile(ref_to_pred, 0.99))
    )
    return metrics


def _ordered_metric_entries(metrics: dict[str, float | int | str]) -> list[tuple[str, float | int | str]]:
    preferred = [
        "phase1_dir",
        "mesh_vertex_count",
        "mesh_face_count",
        "mesh_area",
        "component_count",
        "largest_component_face_ratio",
        "small_component_count",
        "mesh_sample_count",
        "reference_sample_count",
        "mesh_sample_outside_support_ratio",
        "mesh_sample_near_boundary_ratio",
        "pred_to_ref_distance_mean",
        "pred_to_ref_distance_p90",
        "pred_to_ref_distance_p95",
        "pred_to_ref_distance_p99",
        "ref_to_pred_distance_mean",
        "ref_to_pred_distance_p90",
        "ref_to_pred_distance_p95",
        "ref_to_pred_distance_p99",
        "symmetric_chamfer_l1_mean",
        "symmetric_hausdorff_p95",
        "symmetric_hausdorff_p99",
        "symmetric_hausdorff_distance",
        "mesh_signed_sdf_p50",
        "mesh_signed_sdf_p90",
        "mesh_signed_sdf_p99",
        "mesh_outside_distance_p90",
        "mesh_outside_distance_p99",
    ]
    used = set()
    entries: list[tuple[str, float | int | str]] = []
    for key in preferred:
        if key in metrics:
            entries.append((key, metrics[key]))
            used.add(key)
    for key in sorted(metrics):
        if key not in used:
            entries.append((key, metrics[key]))
    return entries


def write_mesh_evaluation(metrics: dict[str, float | int | str], output_path: Path) -> tuple[Path, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in metrics.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output_path.with_suffix(".txt")
    write_key_value_report(report_path, _ordered_metric_entries(metrics))
    return output_path, report_path


def _find_point_cloud_ply(model_path: Path, iteration: int) -> Path:
    if not model_path.is_dir():
        return model_path
    point_cloud_dir = model_path / "point_cloud"
    if not point_cloud_dir.exists():
        raise FileNotFoundError(f"No point_cloud directory found under {model_path}.")
    if int(iteration) >= 0:
        candidate = point_cloud_dir / f"iteration_{int(iteration)}" / "point_cloud.ply"
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


def _extract_mesh_from_model(
    input_path: Path,
    iteration: int,
    resolution: float,
    threshold: float,
    sh_degree: int,
    method: str,
    sugar_poisson_depth: int,
    sugar_density_quantile: float,
    sugar_normal_consistency_k: int,
    sugar_component_min_face_ratio: float,
    sugar_project_on_surface_points: bool,
) -> MeshData:
    from mesher import meshing_ct
    from scene import CTGaussianModel

    ply_path = _find_point_cloud_ply(Path(input_path), int(iteration))
    model = CTGaussianModel(sh_degree=sh_degree)
    model.load_ply(str(ply_path))
    mesh = meshing_ct(
        None,
        model,
        resolution=float(resolution),
        threshold=float(threshold),
        method=method,
        sugar_poisson_depth=sugar_poisson_depth,
        sugar_density_quantile=sugar_density_quantile,
        sugar_normal_consistency_k=sugar_normal_consistency_k,
        sugar_component_min_face_ratio=sugar_component_min_face_ratio,
        sugar_project_on_surface_points=sugar_project_on_surface_points,
    )
    return MeshData(
        vertices=np.asarray(mesh["vertices"], dtype=np.float32),
        faces=np.asarray(mesh["faces"], dtype=np.int32),
        normals=np.asarray(mesh.get("normals"), dtype=np.float32) if mesh.get("normals") is not None else None,
        material_id=np.asarray(mesh.get("material_id"), dtype=np.int32) if mesh.get("material_id") is not None else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate extracted CTGS meshes against a Phase1 support boundary.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--mesh", type=str, help="Predicted mesh PLY to evaluate.")
    input_group.add_argument("--input", type=str, help="CTGS PLY or training output directory to mesh and evaluate.")
    parser.add_argument("--phase1", required=True, type=str, help="Phase1 bundle directory containing analysis.npz and metadata.json.")
    parser.add_argument("--output", required=True, type=str, help="Output metrics JSON path. A .txt report is written next to it.")
    parser.add_argument("--mesh-output", type=str, default=None, help="Optional path for the extracted/evaluated mesh PLY.")
    parser.add_argument("--reference-mesh-output", type=str, default=None, help="Optional Phase1 support mesh PLY output path.")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to read when --input is a training output directory.")
    parser.add_argument("--mesh-method", choices=("sugar", "density"), default="sugar", help="Mesh extraction method for --input.")
    parser.add_argument("--resolution", type=float, default=0.5, help="Density fallback / marching-cubes grid spacing for --input.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Density threshold for --input mesh extraction.")
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--sugar-poisson-depth", type=int, default=9)
    parser.add_argument("--sugar-density-quantile", type=float, default=0.0)
    parser.add_argument("--sugar-normal-consistency-k", type=int, default=30)
    parser.add_argument("--sugar-component-min-face-ratio", type=float, default=0.005)
    parser.add_argument("--sugar-project-on-surface-points", action="store_true", default=False)
    parser.add_argument("--no-sugar-project-on-surface-points", action="store_true", default=False)
    parser.add_argument("--sample-count", type=int, default=200_000)
    parser.add_argument("--reference-sample-count", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outside-tolerance", type=float, default=0.0)
    parser.add_argument("--boundary-band-voxels", type=float, default=1.0)
    parser.add_argument("--use-reference-mesh", action="store_true", default=False, help="Use a marched support mesh instead of Phase1 boundary_points.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mesh:
        mesh = _read_mesh_ply(Path(args.mesh))
    else:
        mesh = _extract_mesh_from_model(
            Path(args.input),
            iteration=args.iteration,
            resolution=args.resolution,
            threshold=args.threshold,
            sh_degree=args.sh_degree,
            method=args.mesh_method,
            sugar_poisson_depth=args.sugar_poisson_depth,
            sugar_density_quantile=args.sugar_density_quantile,
            sugar_normal_consistency_k=args.sugar_normal_consistency_k,
            sugar_component_min_face_ratio=args.sugar_component_min_face_ratio,
            sugar_project_on_surface_points=bool(args.sugar_project_on_surface_points)
            and not bool(args.no_sugar_project_on_surface_points),
        )

    if args.mesh_output:
        _write_mesh_ply(Path(args.mesh_output), mesh)

    if args.reference_mesh_output:
        rng = np.random.default_rng(int(args.seed))
        reference = load_phase1_reference(
            Path(args.phase1),
            reference_sample_count=args.reference_sample_count,
            rng=rng,
            prefer_boundary_points=False,
        )
        if reference.reference_mesh is None:
            reference.reference_mesh = mesh_from_support_mask(reference.support_mask, reference.spacing_zyx, reference.origin_xyz)
        _write_mesh_ply(Path(args.reference_mesh_output), reference.reference_mesh)

    metrics = evaluate_mesh_extraction(
        mesh,
        Path(args.phase1),
        sample_count=args.sample_count,
        reference_sample_count=args.reference_sample_count,
        seed=args.seed,
        outside_tolerance=args.outside_tolerance,
        boundary_band_voxels=args.boundary_band_voxels,
        prefer_boundary_points=not bool(args.use_reference_mesh),
    )
    json_path, txt_path = write_mesh_evaluation(metrics, Path(args.output))
    print(f"Mesh evaluation written: {json_path}")
    print(f"Mesh evaluation report: {txt_path}")
    print(
        "Mesh eval: outside={0:.4f} chamfer={1:.4f} hausdorff_p95={2:.4f} pred_p90={3:.4f} ref_p90={4:.4f}".format(
            float(metrics["mesh_sample_outside_support_ratio"]),
            float(metrics["symmetric_chamfer_l1_mean"]),
            float(metrics["symmetric_hausdorff_p95"]),
            float(metrics["pred_to_ref_distance_p90"]),
            float(metrics["ref_to_pred_distance_p90"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
