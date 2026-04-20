import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ct_pipeline.ct_loader import CTVolumeLoader
from ct_pipeline.ct_preprocessor import CTPreprocessor
from ct_pipeline.geometry_analyzer import GeometryAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone CT Phase 1 ingestion and geometry analysis.")
    parser.add_argument("--input", required=True, help="Path to a DICOM series, RAW volume, or TIFF stack.")
    parser.add_argument("--fmt", default="auto", choices=["auto", "dicom", "raw", "tiff"])
    parser.add_argument("--output", required=True, help="Directory where analysis.npz and metadata.json are written.")
    parser.add_argument("--raw-meta", default=None, help="Optional JSON sidecar for RAW or TIFF metadata.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma for differential analysis.")
    parser.add_argument("--boundary-width", type=int, default=3, help="Exterior boundary dilation width in voxels.")
    parser.add_argument("--k-neighbors", type=int, default=20, help="Neighborhood size for local plane fitting.")
    parser.add_argument("--interior-points-ratio", type=float, default=1.0, help="Interior bulk points as a ratio of extracted surface points.")
    parser.add_argument("--interior-boundary-margin", type=int, default=2, help="Minimum distance from the foreground boundary for interior bulk sampling.")
    parser.add_argument("--max-material-classes", type=int, default=3, help="Maximum number of automatically separated material classes.")
    parser.add_argument("--support-threshold-mode", type=str, default="otsu", choices=["otsu", "multi_otsu"], help="Automatic threshold used to build the coarse solid support mask.")
    parser.add_argument("--surface-gradient-percentile", type=float, default=60.0, help="Keep boundary-band voxels whose gradient magnitude is above this percentile.")
    return parser


def run_phase1(args: argparse.Namespace) -> None:
    loader = CTVolumeLoader()
    volume = loader.load(args.input, fmt=args.fmt, raw_meta_path=args.raw_meta)
    spacing = loader.get_voxel_spacing()
    metadata = loader.get_metadata()

    preprocessor = CTPreprocessor()
    support = preprocessor.segment_coarse_support(volume, threshold_mode=args.support_threshold_mode)
    support_mask = support["support_mask"]
    if not np.any(support_mask):
        raise RuntimeError("Coarse support extraction produced an empty mask.")

    roi_bbox = support["roi_bbox"]
    foreground_mask = support["roi_mask"]
    air_mask = support["air_mask"]
    material_mask = support_mask.copy()
    void_mask = air_mask.copy()
    material_label_volume = support_mask.astype(np.int32)
    boundary_points = preprocessor.extract_intensity_surface_points(
        volume,
        support_mask,
        spacing,
        sigma=args.sigma,
        gradient_percentile=args.surface_gradient_percentile,
    )
    boundary_material_id = np.zeros((boundary_points.shape[0], 1), dtype=np.int64)
    if boundary_points.shape[0] == 0:
        raise RuntimeError("Intensity-driven surface seeding produced zero surface points.")
    interior_target_count = max(1, int(round(boundary_points.shape[0] * float(args.interior_points_ratio))))
    interior_points, interior_density_seed, interior_material_id = preprocessor.sample_support_points(
        support_mask,
        volume,
        spacing,
        target_count=interior_target_count,
        boundary_margin_voxels=args.interior_boundary_margin,
    )

    analyzer = GeometryAnalyzer(sigma=args.sigma)
    boundary_normals, boundary_tangent_u, boundary_tangent_v, boundary_strength = analyzer.estimate_boundary_geometry(
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
            "boundary_point_count": int(boundary_points.shape[0]),
            "interior_point_count": int(interior_points.shape[0]),
            "material_class_count": 1,
            "parameters": {
                "sigma": float(args.sigma),
                "boundary_width": int(args.boundary_width),
                "k_neighbors": int(args.k_neighbors),
                "interior_points_ratio": float(args.interior_points_ratio),
                "interior_boundary_margin": int(args.interior_boundary_margin),
                "max_material_classes": int(args.max_material_classes),
                "support_threshold_mode": str(args.support_threshold_mode),
                "surface_gradient_percentile": float(args.surface_gradient_percentile),
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
