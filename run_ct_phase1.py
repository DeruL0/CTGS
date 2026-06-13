import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from ct_pipeline.data.loader import CTVolumeLoader
from ct_pipeline.data.preprocessing import CTPreprocessor, build_support_signed_distance
from ct_pipeline.geometry.analysis import GeometryAnalyzer


def _orthonormal_tangents_from_normals(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct two orthonormal tangent vectors for each unit normal via Gram-Schmidt."""
    n = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    reference = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (n.shape[0], 1))
    parallel = np.abs(np.sum(n * reference, axis=1)) > 0.9
    reference[parallel] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_u = reference - np.sum(reference * n, axis=1, keepdims=True) * n
    u_norm = np.linalg.norm(tangent_u, axis=1, keepdims=True)
    tangent_u = tangent_u / np.maximum(u_norm, 1e-6)
    tangent_v = np.cross(n, tangent_u)
    v_norm = np.linalg.norm(tangent_v, axis=1, keepdims=True)
    tangent_v = tangent_v / np.maximum(v_norm, 1e-6)
    tangent_u = np.cross(tangent_v, n)
    u_norm = np.linalg.norm(tangent_u, axis=1, keepdims=True)
    tangent_u = tangent_u / np.maximum(u_norm, 1e-6)
    return tangent_u.astype(np.float32), tangent_v.astype(np.float32)


def _sample_sdf_gradient_normals(
    support_mask: np.ndarray,
    spacing: tuple,
    points_xyz: np.ndarray,
    sdf_sigma: float = 1.0,
) -> np.ndarray:
    """Compute outward unit normals at given world points from grad(SDF_support) of the binary mask."""
    sdf = build_support_signed_distance(np.asarray(support_mask, dtype=bool), tuple(float(v) for v in spacing))
    if float(sdf_sigma) > 0.0:
        sdf = ndimage.gaussian_filter(sdf, sigma=float(sdf_sigma)).astype(np.float32)
    sz, sy, sx = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    # ndimage.sobel(axis=k) approximates partial derivative in voxel units along axis k.
    # Divide by 8 to match the Sobel kernel scaling, then by world spacing.
    gz_world = (ndimage.sobel(sdf, axis=0, mode="nearest").astype(np.float32) / 8.0) / max(sz, 1e-8)
    gy_world = (ndimage.sobel(sdf, axis=1, mode="nearest").astype(np.float32) / 8.0) / max(sy, 1e-8)
    gx_world = (ndimage.sobel(sdf, axis=2, mode="nearest").astype(np.float32) / 8.0) / max(sx, 1e-8)
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # World xyz -> voxel zyx coords for map_coordinates.
    coords = np.vstack(
        (
            points[:, 2] / max(sz, 1e-8),
            points[:, 1] / max(sy, 1e-8),
            points[:, 0] / max(sx, 1e-8),
        )
    )
    sampled_gx = ndimage.map_coordinates(gx_world, coords, order=1, mode="nearest").astype(np.float32)
    sampled_gy = ndimage.map_coordinates(gy_world, coords, order=1, mode="nearest").astype(np.float32)
    sampled_gz = ndimage.map_coordinates(gz_world, coords, order=1, mode="nearest").astype(np.float32)
    # SDF inside support is negative; gradient points OUTWARD (toward increasing SDF),
    # which is exactly the outward surface normal we want.
    normals = np.stack((sampled_gx, sampled_gy, sampled_gz), axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-6
    out = np.zeros_like(normals, dtype=np.float32)
    out[valid] = (normals[valid] / norms[valid]).astype(np.float32)
    out[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone CT Phase 1 ingestion and geometry analysis.")
    parser.add_argument("--input", required=True, help="Path to a DICOM series, RAW volume, or TIFF stack.")
    parser.add_argument("--fmt", default="auto", choices=["auto", "dicom", "raw", "tiff"])
    parser.add_argument("--output", required=True, help="Directory where analysis.npz and metadata.json are written.")
    parser.add_argument("--raw-meta", default=None, help="Optional JSON sidecar for RAW or TIFF metadata.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma for differential analysis.")
    parser.add_argument("--interior-points-ratio", type=float, default=1.0, help="Interior bulk points as a ratio of extracted surface points.")
    parser.add_argument("--interior-boundary-margin", type=int, default=1, help="Minimum distance from the foreground boundary for interior bulk sampling.")
    parser.add_argument("--support-threshold-mode", type=str, default="otsu", choices=["otsu", "multi_otsu"], help="Automatic threshold used to build the coarse solid support mask.")
    parser.add_argument(
        "--support-min-component-voxels",
        type=int,
        default=0,
        help="Keep thresholded support components with at least this many voxels. Default 0 combines with fraction mode.",
    )
    parser.add_argument(
        "--support-min-component-fraction",
        type=float,
        default=1.0,
        help="Keep thresholded support components at least this fraction of the largest component. Default 1.0 keeps only the largest component.",
    )
    parser.add_argument("--surface-spacing-voxels", type=float, default=3.0, help="Geometry-uniform spacing for support-boundary surface anchors; <=1 keeps every boundary voxel.")
    parser.add_argument("--use-sdf-normals", action="store_true", default=False, help="Replace CT-gradient boundary normals with normals derived from grad(SDF) of the binary support mask. Tangents are built as orthonormal frames.")
    parser.add_argument("--sdf-normal-sigma", type=float, default=1.0, help="Gaussian smoothing applied to the support SDF before differentiating for normals.")
    return parser


def run_phase1(args: argparse.Namespace) -> None:
    loader = CTVolumeLoader()
    volume = loader.load(args.input, fmt=args.fmt, raw_meta_path=args.raw_meta)
    spacing = loader.get_voxel_spacing()
    metadata = loader.get_metadata()

    preprocessor = CTPreprocessor()
    support = preprocessor.segment_coarse_support(
        volume,
        threshold_mode=args.support_threshold_mode,
        min_component_voxels=args.support_min_component_voxels,
        min_component_fraction=args.support_min_component_fraction,
    )
    support_mask = support["support_mask"]
    if not np.any(support_mask):
        raise RuntimeError("Coarse support extraction produced an empty mask.")

    roi_bbox = support["roi_bbox"]
    foreground_mask = support["roi_mask"]
    air_mask = support["air_mask"]
    material_mask = support_mask.copy()
    void_mask = air_mask.copy()
    material_label_volume = support_mask.astype(np.int32)
    material_signed_distance = build_support_signed_distance(
        material_mask,
        tuple(float(value) for value in spacing),
    )
    boundary_candidates = preprocessor.extract_intensity_surface_points(
        volume,
        support_mask,
        spacing,
    )
    boundary_points = preprocessor.subsample_surface_points_by_voxel_grid(
        boundary_candidates,
        spacing,
        spacing_voxels=args.surface_spacing_voxels,
    )
    boundary_material_id = np.zeros((boundary_points.shape[0], 1), dtype=np.int64)
    if boundary_points.shape[0] == 0:
        raise RuntimeError("Intensity-driven surface seeding produced zero surface points.")
    interior_target_count = max(1, int(round(boundary_points.shape[0] * float(args.interior_points_ratio))))
    interior_points, interior_density_seed, interior_material_id = preprocessor.sample_interior_points(
        support_mask,
        volume,
        spacing,
        target_count=interior_target_count,
        boundary_margin_voxels=args.interior_boundary_margin,
    )

    analyzer = GeometryAnalyzer(sigma=args.sigma)
    _, _, _, boundary_strength = analyzer.estimate_boundary_geometry(
        boundary_points,
        volume,
        spacing,
    )
    if bool(args.use_sdf_normals):
        boundary_normals = _sample_sdf_gradient_normals(
            support_mask,
            spacing,
            boundary_points,
            sdf_sigma=float(args.sdf_normal_sigma),
        )
        boundary_tangent_u, boundary_tangent_v = _orthonormal_tangents_from_normals(boundary_normals)
        print(
            "Phase 1 boundary normals: using SDF-gradient normals (sdf_sigma={0:.2f}); tangents built via Gram-Schmidt.".format(
                float(args.sdf_normal_sigma),
            )
        )
    else:
        boundary_normals, boundary_tangent_u, boundary_tangent_v, _ = analyzer.estimate_boundary_geometry(
            boundary_points,
            volume,
            spacing,
        )

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(output_dir / "analysis.npz"),
        coarse_support_mask=support_mask.astype(bool),
        support_threshold=np.asarray([support["support_threshold"]], dtype=np.float32),
        material_mask=material_mask.astype(bool),
        material_signed_distance=material_signed_distance.astype(np.float32),
        void_mask=void_mask.astype(bool),
        foreground_mask=foreground_mask.astype(bool),
        roi_bbox=roi_bbox.astype(np.int32),
        material_label_volume=material_label_volume.astype(np.int32),
        boundary_points=boundary_points.astype(np.float32),
        boundary_normals=boundary_normals.astype(np.float32),
        boundary_tangent_u=boundary_tangent_u.astype(np.float32),
        boundary_tangent_v=boundary_tangent_v.astype(np.float32),
        boundary_strength=boundary_strength.astype(np.float32),
        boundary_material_id=boundary_material_id.astype(np.int64),
        interior_points=interior_points.astype(np.float32),
        interior_density_seed=interior_density_seed.astype(np.float32),
        interior_material_id=interior_material_id.astype(np.int64),
    )

    metadata.update(
        {
            "roi_bbox_zyx": roi_bbox.tolist(),
            "boundary_candidate_count": int(boundary_candidates.shape[0]),
            "boundary_point_count": int(boundary_points.shape[0]),
            "interior_point_count": int(interior_points.shape[0]),
            "material_class_count": 1,
            "parameters": {
                "sigma": float(args.sigma),
                "interior_points_ratio": float(args.interior_points_ratio),
                "interior_boundary_margin": int(args.interior_boundary_margin),
                "support_threshold_mode": str(args.support_threshold_mode),
                "support_min_component_voxels": int(args.support_min_component_voxels),
                "support_min_component_fraction": float(args.support_min_component_fraction),
                "surface_spacing_voxels": float(args.surface_spacing_voxels),
                "use_sdf_normals": bool(args.use_sdf_normals),
                "sdf_normal_sigma": float(args.sdf_normal_sigma),
            },
            "outputs": {
                "analysis_bundle": "analysis.npz",
                "metadata": "metadata.json",
            },
        }
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_phase1(args)
    except Exception as error:
        print("CT Phase 1 failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
