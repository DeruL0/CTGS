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
    return parser


def run_phase1(args: argparse.Namespace) -> None:
    loader = CTVolumeLoader()
    volume = loader.load(args.input, fmt=args.fmt, raw_meta_path=args.raw_meta)
    spacing = loader.get_voxel_spacing()
    metadata = loader.get_metadata()

    preprocessor = CTPreprocessor()
    segmentation = preprocessor.segment_material_void(volume, method="multi_otsu", max_material_classes=args.max_material_classes)
    material_mask = segmentation["material_mask"]
    void_mask = segmentation["void_mask"]
    foreground_mask = segmentation["foreground_mask"]
    material_label_volume = segmentation["material_label_volume"]
    if not np.any(material_mask):
        raise RuntimeError("Material segmentation produced an empty mask.")
    boundary_band = preprocessor.dilate_boundary(foreground_mask, width_voxels=args.boundary_width)
    roi_bbox = segmentation["roi_bbox"]
    surface_points, surface_material_id = preprocessor.extract_material_surface_points(material_label_volume, spacing)
    if surface_points.shape[0] == 0:
        raise RuntimeError("Surface extraction produced zero surface points.")
    interior_target_count = max(1, int(round(surface_points.shape[0] * float(args.interior_points_ratio))))
    interior_points, interior_density_seed, interior_material_id = preprocessor.sample_interior_points(
        material_mask,
        volume,
        spacing,
        target_count=interior_target_count,
        boundary_margin_voxels=args.interior_boundary_margin,
        material_label_volume=material_label_volume,
    )

    analyzer = GeometryAnalyzer(sigma=args.sigma)
    surface_normals = analyzer.estimate_surface_normals(surface_points, volume, spacing)
    mask_planar, mask_edge, mask_curved = analyzer.classify_regions(surface_points, volume, spacing)

    num_points = surface_points.shape[0]
    plane_normals = np.full((num_points, 3), np.nan, dtype=np.float32)
    plane_tangent_u = np.full((num_points, 3), np.nan, dtype=np.float32)
    plane_tangent_v = np.full((num_points, 3), np.nan, dtype=np.float32)
    plane_offsets = np.full((num_points,), np.nan, dtype=np.float32)
    plane_residuals = np.full((num_points,), np.inf, dtype=np.float32)

    if np.any(mask_planar):
        plane_params, planar_residuals = analyzer.fit_local_planes(
            surface_points[mask_planar],
            surface_normals[mask_planar],
            k_neighbors=args.k_neighbors,
        )
        plane_normals[mask_planar] = plane_params["normal"]
        plane_tangent_u[mask_planar] = plane_params["tangent_u"]
        plane_tangent_v[mask_planar] = plane_params["tangent_v"]
        plane_offsets[mask_planar] = plane_params["offset"]
        plane_residuals[mask_planar] = planar_residuals

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(output_dir / "analysis.npz"),
        material_mask=material_mask.astype(bool),
        void_mask=void_mask.astype(bool),
        foreground_mask=foreground_mask.astype(bool),
        boundary_band=boundary_band.astype(bool),
        roi_bbox=roi_bbox.astype(np.int32),
        material_label_volume=material_label_volume.astype(np.int32),
        surface_points=surface_points.astype(np.float32),
        surface_material_id=surface_material_id.astype(np.int64),
        material_id=surface_material_id.astype(np.int64),
        surface_normals=surface_normals.astype(np.float32),
        mask_planar=mask_planar.astype(bool),
        mask_edge=mask_edge.astype(bool),
        mask_curved=mask_curved.astype(bool),
        plane_normals=plane_normals.astype(np.float32),
        plane_tangent_u=plane_tangent_u.astype(np.float32),
        plane_tangent_v=plane_tangent_v.astype(np.float32),
        plane_offsets=plane_offsets.astype(np.float32),
        plane_residuals=plane_residuals.astype(np.float32),
        interior_points=interior_points.astype(np.float32),
        interior_density_seed=interior_density_seed.astype(np.float32),
        interior_material_id=interior_material_id.astype(np.int64),
    )

    metadata.update(
        {
            "roi_bbox_zyx": roi_bbox.tolist(),
            "surface_point_count": int(num_points),
            "interior_point_count": int(interior_points.shape[0]),
            "material_class_count": int(segmentation["material_class_count"]),
            "planar_count": int(np.sum(mask_planar)),
            "edge_count": int(np.sum(mask_edge)),
            "curved_count": int(np.sum(mask_curved)),
            "parameters": {
                "sigma": float(args.sigma),
                "boundary_width": int(args.boundary_width),
                "k_neighbors": int(args.k_neighbors),
                "interior_points_ratio": float(args.interior_points_ratio),
                "interior_boundary_margin": int(args.interior_boundary_margin),
                "max_material_classes": int(args.max_material_classes),
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
