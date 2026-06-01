from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ct_pipeline.config import extract_ct_model_args, extract_ct_optimization_args
from ct_pipeline.rendering.slices import CTPatchGridCache, _build_query_points_from_base
from ct_pipeline.backend.render import render_ct_slice_patch_native
from ct_pipeline.rendering.fields import density_to_occupancy, query_ct_density_from_state_by_region
from ct_pipeline.training.config import build_parser, validate_ct_training_args
from ct_pipeline.training.bootstrap import prepare_ct_training_bootstrap
from ct_pipeline.backend import prepare_ct_training_state, require_ct_native_backend

DEFAULT_MODEL_DIR = REPO_ROOT / "outputs" / "bulk_diagnostic_compare_sdfgate_20260529_023846" / "iter1000" / "variant_base"
DEFAULT_PHASE1_DIR = REPO_ROOT / "outputs" / "bunny_smoke" / "phase1_surface_complete_grid3_margin1_20260522"
DEFAULT_VOLUME_PATH = REPO_ROOT / "assets" / "bunny"


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _summarize_mask(mask: np.ndarray, arrays: dict[str, np.ndarray], total_error_mass: float) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    out: dict[str, float] = {"count": count, "pixel_ratio": float(mask.mean())}
    if count == 0:
        for key in arrays:
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_p10"] = float("nan")
            out[f"{key}_p50"] = float("nan")
            out[f"{key}_p90"] = float("nan")
        out["error_mass_ratio"] = 0.0
        return out
    for key, array in arrays.items():
        values = np.asarray(array)[mask]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_p10"] = _percentile(values, 10)
        out[f"{key}_p50"] = _percentile(values, 50)
        out[f"{key}_p90"] = _percentile(values, 90)
    error_mass = float(np.asarray(arrays["abs_error"])[mask].sum())
    out["error_mass_ratio"] = error_mass / max(float(total_error_mass), 1e-12)
    return out


def _basic_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    if values.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "p10": _percentile(values, 10),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
    }


def _edge_distance_summary(source_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    source = np.asarray(source_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    source_edge = np.logical_xor(source, ndimage.binary_erosion(source))
    target_edge = np.logical_xor(target, ndimage.binary_erosion(target))
    if not np.any(source_edge) or not np.any(target_edge):
        return {"count": int(source_edge.sum()), "p50": float("nan"), "p90": float("nan"), "mean": float("nan")}
    distance_to_target = ndimage.distance_transform_edt(~target_edge)
    distances = distance_to_target[source_edge]
    return {
        "count": int(source_edge.sum()),
        "mean": float(np.mean(distances)),
        "p50": _percentile(distances, 50),
        "p90": _percentile(distances, 90),
    }


def _query_region_occupancy(training_state, query_points: torch.Tensor, region: str, chunk: int = 8192) -> torch.Tensor:
    parts = []
    for start in range(0, int(query_points.shape[0]), int(chunk)):
        points = query_points[start : start + int(chunk)]
        density = query_ct_density_from_state_by_region(training_state, points, region=region, detach=True)
        parts.append(density_to_occupancy(density.to(dtype=torch.float32)))
    return torch.cat(parts, dim=0)


def _save_figures(output_dir: Path, arrays: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    gt = arrays["gt"]
    pred = arrays["pred"]
    err = arrays["abs_error"]
    sdf = arrays["sdf"]
    mismatch = masks["phase_gt_mismatch"].astype(np.float32)
    high_error = masks["high_error"].astype(np.float32)
    phase_material = masks["phase_material"]
    gt_material = masks["gt_material"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [
        (gt, "GT", "gray", 0.0, 1.0),
        (pred, "CTGS preview intensity", "gray", 0.0, 1.0),
        (err, "abs error", "magma", 0.0, max(1e-6, float(err.max()))),
        (sdf, "phase1 signed distance", "coolwarm", -3.0, 3.0),
        (mismatch, "GT/phase1 mask mismatch", "magma", 0.0, 1.0),
        (high_error, "top 10% error pixels", "magma", 0.0, 1.0),
    ]
    for ax, (image, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        try:
            ax.contour(phase_material.astype(np.float32), levels=[0.5], colors=["cyan"], linewidths=0.6)
            ax.contour(gt_material.astype(np.float32), levels=[0.5], colors=["yellow"], linewidths=0.5)
        except Exception:
            pass
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(output_dir / "boundary_root_cause_panels.png", dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    model_dir = Path(args.model_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else model_dir / "chkpnt1000.pth"
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "boundary_root_cause"
    output_dir.mkdir(parents=True, exist_ok=True)

    require_ct_native_backend()
    train_parser = build_parser()
    train_argv = [
            "--ct_phase1_dir",
            str(args.phase1_dir),
            "--ct_volume_path",
            str(args.volume_path),
            "--ct_volume_format",
            args.volume_format,
            "--iterations",
            "1000",
            "--model_path",
            str(output_dir / "_bootstrap"),
            "--skip_export_mesh",
            "--skip_export_sdf",
            "--quiet",
    ]
    if args.surface_material_gate_sigma is not None:
        train_argv.extend(["--ct_surface_material_gate_sigma", str(args.surface_material_gate_sigma)])
    if args.material_compose_mode is not None:
        train_argv.extend(["--ct_material_compose_mode", str(args.material_compose_mode)])
    train_args = train_parser.parse_args(train_argv)
    validate_ct_training_args(train_args)
    dataset = extract_ct_model_args(train_args)
    opt = extract_ct_optimization_args(train_args)
    context = prepare_ct_training_bootstrap(dataset, opt, train_args, str(checkpoint))
    training_state = prepare_ct_training_state(
        context.gaussians,
        spacing_zyx=context.spacing_zyx,
        truncation_sigma=train_args.ct_gaussian_truncation_sigma,
        grid_cell_voxels=train_args.ct_grid_cell_voxels,
        build_full_grid=True,
        build_region_grids=True,
    )

    volume_shape = tuple(int(value) for value in context.volume_shape)
    slice_idx = int(args.slice_idx if args.slice_idx is not None else volume_shape[0] // 2)
    height, width = volume_shape[1], volume_shape[2]
    device = training_state.xyz.device
    dtype = training_state.xyz.dtype
    rr, cc = CTPatchGridCache().get((height, width), device, dtype)
    query_points = _build_query_points_from_base(rr, cc, 0, slice_idx, (0, 0), context.spacing_zyx)

    with torch.no_grad():
        preview_occupancy = render_ct_slice_patch_native(
            training_state.render_state,
            "z",
            slice_idx,
            (0, 0),
            (height, width),
            context.spacing_zyx,
            volume_shape,
            slice_tile_size=train_args.ct_slice_tile_size,
        ).to(dtype=torch.float32)
        bulk_occ = _query_region_occupancy(training_state, query_points, "bulk").reshape(height, width)
        surface_occ = _query_region_occupancy(training_state, query_points, "surface").reshape(height, width)
        bulk_density = -torch.log1p(-bulk_occ.clamp(0.0, 1.0 - 1e-6))
        surface_density = -torch.log1p(-surface_occ.clamp(0.0, 1.0 - 1e-6))
        manual_occupancy = (1.0 - torch.exp(-(bulk_density + surface_density).clamp_min(0.0))).clamp(0.0, 1.0)
        pred = float(context.intensity_air) + (
            float(context.intensity_mat) - float(context.intensity_air)
        ) * preview_occupancy
        gt = context.volume_cuda[slice_idx].to(dtype=torch.float32)

    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    bulk_np = bulk_occ.detach().cpu().numpy()
    surface_np = surface_occ.detach().cpu().numpy()
    preview_occ_np = preview_occupancy.detach().cpu().numpy()
    manual_occ_np = manual_occupancy.detach().cpu().numpy()
    abs_error = np.abs(gt_np - pred_np)

    phase_material = np.asarray(context.analysis["material_mask"], dtype=bool)[slice_idx]
    signed_distance = np.asarray(context.signed_distance_field["signed_distance_native"], dtype=np.float32)[slice_idx]
    threshold = float(context.intensity_air + 0.5 * (context.intensity_mat - context.intensity_air))
    gt_material = gt_np >= threshold
    pred_material = pred_np >= threshold
    band = max(float(train_args.ct_boundary_band), 1e-6)
    if train_args.ct_surface_material_gate_sigma is None:
        surface_sigma_material = max(0.5, 0.5 * band)
    else:
        surface_sigma_material = max(float(train_args.ct_surface_material_gate_sigma), 1e-6)
    material_side = signed_distance <= 0.0
    surface_gate = np.where(
        material_side,
        np.exp(-0.5 * np.square(signed_distance / surface_sigma_material)),
        0.0,
    ).astype(np.float32)
    bulk_gate = material_side.astype(np.float32)
    bulk_shell = bulk_gate * bulk_np
    surface_contribution = np.clip(surface_gate * surface_np, 0.0, 1.0)
    if train_args.ct_material_compose_mode == "bulk_first_material":
        surface_contribution = np.where(
            material_side,
            np.clip(1.0 - bulk_shell, 0.0, 1.0) * surface_contribution,
            surface_contribution,
        )
        bulk_contribution = np.where(
            material_side,
            bulk_shell,
            np.clip(1.0 - surface_contribution, 0.0, 1.0) * bulk_shell,
        )
    else:
        bulk_contribution = np.clip(1.0 - surface_contribution, 0.0, 1.0) * bulk_shell
    composed_query_occ = np.clip(surface_contribution + bulk_contribution, 0.0, 1.0)
    surface_suppressed_bulk = np.clip(bulk_shell - bulk_contribution, 0.0, 1.0)

    arrays = {
        "gt": gt_np,
        "pred": pred_np,
        "bulk_occ": bulk_np,
        "surface_occ": surface_np,
        "preview_occ": preview_occ_np,
        "manual_occ": manual_occ_np,
        "surface_gate": surface_gate,
        "surface_contribution": surface_contribution,
        "bulk_contribution": bulk_contribution,
        "surface_suppressed_bulk": surface_suppressed_bulk,
        "composed_query_occ": composed_query_occ,
        "abs_error": abs_error,
        "sdf": signed_distance,
    }
    total_error_mass = float(abs_error.sum())
    bins = {
        "mat_sdf_3plus": signed_distance <= -3.0,
        "mat_sdf_1p5_3": np.logical_and(signed_distance > -3.0, signed_distance <= -1.5),
        "mat_sdf_0p5_1p5": np.logical_and(signed_distance > -1.5, signed_distance <= -0.5),
        "mat_sdf_0_0p5": np.logical_and(signed_distance > -0.5, signed_distance <= 0.0),
        "air_sdf_0_0p5": np.logical_and(signed_distance > 0.0, signed_distance <= 0.5),
        "air_sdf_0p5_1p5": np.logical_and(signed_distance > 0.5, signed_distance <= 1.5),
        "air_sdf_1p5plus": signed_distance > 1.5,
    }
    bin_summary = {name: _summarize_mask(mask, arrays, total_error_mass) for name, mask in bins.items()}

    mismatch = phase_material != gt_material
    high_error = abs_error >= float(np.quantile(abs_error, 0.90))
    bulk_gap = np.logical_and(phase_material, bulk_np < 0.85)
    total_gap = np.logical_and(gt_material, pred_np < threshold)
    false_positive = np.logical_and(~gt_material, pred_material)
    false_negative = np.logical_and(gt_material, ~pred_material)

    air_mask = ~gt_material
    air_labels, air_component_count = ndimage.label(air_mask, structure=np.ones((3, 3), dtype=np.uint8))
    border_labels = np.unique(
        np.concatenate(
            [
                air_labels[0, :],
                air_labels[-1, :],
                air_labels[:, 0],
                air_labels[:, -1],
            ]
        )
    )
    border_labels = border_labels[border_labels != 0]
    exterior_air = air_mask & np.isin(air_labels, border_labels)
    internal_air = air_mask & ~exterior_air
    internal_air_labels = np.unique(air_labels[internal_air])
    internal_air_labels = internal_air_labels[internal_air_labels != 0]
    if np.any(internal_air):
        distance_to_internal_air = ndimage.distance_transform_edt(~internal_air)
    else:
        distance_to_internal_air = np.full(gt_material.shape, np.inf, dtype=np.float32)
    if np.any(exterior_air):
        distance_to_exterior_air = ndimage.distance_transform_edt(~exterior_air)
    else:
        distance_to_exterior_air = np.full(gt_material.shape, np.inf, dtype=np.float32)
    sdf_boundary_band = np.abs(signed_distance) <= 1.5
    internal_void_boundary = sdf_boundary_band & (distance_to_internal_air <= 3.0)
    exterior_boundary = sdf_boundary_band & (distance_to_exterior_air <= 3.0)
    material_near_internal_void = gt_material & (distance_to_internal_air <= 3.0)
    material_near_exterior = gt_material & (distance_to_exterior_air <= 3.0)

    mask_summary = {
        "phase_material": _summarize_mask(phase_material, arrays, total_error_mass),
        "gt_material": _summarize_mask(gt_material, arrays, total_error_mass),
        "phase_gt_mismatch": _summarize_mask(mismatch, arrays, total_error_mass),
        "high_error_top10": _summarize_mask(high_error, arrays, total_error_mass),
        "bulk_gap_phase_material": _summarize_mask(bulk_gap, arrays, total_error_mass),
        "gt_total_false_negative": _summarize_mask(total_gap, arrays, total_error_mass),
        "pred_false_positive_air": _summarize_mask(false_positive, arrays, total_error_mass),
        "pred_false_negative_material": _summarize_mask(false_negative, arrays, total_error_mass),
        "internal_void_boundary_band": _summarize_mask(internal_void_boundary, arrays, total_error_mass),
        "exterior_boundary_band": _summarize_mask(exterior_boundary, arrays, total_error_mass),
        "material_near_internal_void": _summarize_mask(material_near_internal_void, arrays, total_error_mass),
        "material_near_exterior": _summarize_mask(material_near_exterior, arrays, total_error_mass),
        "air_component_count": int(air_component_count),
        "internal_air_component_count": int(internal_air_labels.shape[0]),
        "high_error_and_mismatch_ratio": float(np.logical_and(high_error, mismatch).sum() / max(int(high_error.sum()), 1)),
        "high_error_and_boundary_band_ratio": float(np.logical_and(high_error, np.abs(signed_distance) <= 1.5).sum() / max(int(high_error.sum()), 1)),
        "high_error_and_internal_void_boundary_ratio": float(np.logical_and(high_error, internal_void_boundary).sum() / max(int(high_error.sum()), 1)),
        "high_error_and_exterior_boundary_ratio": float(np.logical_and(high_error, exterior_boundary).sum() / max(int(high_error.sum()), 1)),
        "bulk_gap_and_internal_void_boundary_ratio": float(np.logical_and(bulk_gap, internal_void_boundary).sum() / max(int(bulk_gap.sum()), 1)),
        "bulk_gap_and_exterior_boundary_ratio": float(np.logical_and(bulk_gap, exterior_boundary).sum() / max(int(bulk_gap.sum()), 1)),
        "bulk_gap_but_total_ok_ratio": float(np.logical_and(bulk_gap, pred_np >= threshold).sum() / max(int(bulk_gap.sum()), 1)),
    }
    compose_probe_masks = {
        "high_error_top10": high_error,
        "bulk_gap_phase_material": bulk_gap,
        "internal_void_boundary_band": internal_void_boundary,
        "material_near_internal_void": material_near_internal_void,
        "material_near_exterior": material_near_exterior,
        "mat_sdf_0_0p5": bins["mat_sdf_0_0p5"],
        "mat_sdf_0p5_1p5": bins["mat_sdf_0p5_1p5"],
        "mat_sdf_1p5_3": bins["mat_sdf_1p5_3"],
    }
    compose_probe = {}
    for name, mask in compose_probe_masks.items():
        mask = np.asarray(mask, dtype=bool)
        compose_probe[name] = {
            "count": int(mask.sum()),
            "surface_gate": _basic_stats(surface_gate[mask]),
            "surface_occ": _basic_stats(surface_np[mask]),
            "surface_contribution": _basic_stats(surface_contribution[mask]),
            "bulk_occ": _basic_stats(bulk_np[mask]),
            "bulk_contribution": _basic_stats(bulk_contribution[mask]),
            "surface_suppressed_bulk": _basic_stats(surface_suppressed_bulk[mask]),
            "query_composed_occ": _basic_stats(composed_query_occ[mask]),
            "native_preview_occ": _basic_stats(preview_occ_np[mask]),
        }
    edge_summary = {
        "gt_edge_to_phase_edge": _edge_distance_summary(gt_material, phase_material),
        "phase_edge_to_gt_edge": _edge_distance_summary(phase_material, gt_material),
    }
    summary = {
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "slice_idx": slice_idx,
        "volume_shape": list(volume_shape),
        "threshold": threshold,
        "intensity_air": float(context.intensity_air),
        "intensity_mat": float(context.intensity_mat),
        "mae": float(abs_error.mean()),
        "psnr": float(-10.0 * np.log10(max(float(np.mean((gt_np - pred_np) ** 2)), 1e-10))),
        "manual_to_native_occupancy_abs_diff_mean": float(np.abs(manual_occ_np - preview_occ_np).mean()),
        "manual_to_native_occupancy_abs_diff_p99": _percentile(np.abs(manual_occ_np - preview_occ_np), 99),
        "surface_sigma_material": float(surface_sigma_material),
        "material_compose_mode": str(train_args.ct_material_compose_mode),
        "bin_summary": bin_summary,
        "mask_summary": mask_summary,
        "compose_probe": compose_probe,
        "edge_summary_voxels": edge_summary,
    }
    (output_dir / "boundary_root_cause_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"model_dir = {model_dir}",
        f"slice_idx = {slice_idx}",
        f"threshold = {threshold:.6f}",
        f"mae = {summary['mae']:.6f}",
        f"psnr = {summary['psnr']:.6f}",
        f"manual_to_native_occupancy_abs_diff_mean = {summary['manual_to_native_occupancy_abs_diff_mean']:.6f}",
        f"manual_to_native_occupancy_abs_diff_p99 = {summary['manual_to_native_occupancy_abs_diff_p99']:.6f}",
        "",
        "[mask_summary]",
    ]
    for name, values in mask_summary.items():
        if isinstance(values, dict):
            lines.append(
                "{0}: count={1} pixel_ratio={2:.4f} error_mass={3:.4f} "
                "gt_mean={4:.4f} pred_mean={5:.4f} bulk_p50={6:.4f} surface_p50={7:.4f}".format(
                    name,
                    int(values["count"]),
                    float(values["pixel_ratio"]),
                    float(values["error_mass_ratio"]),
                    float(values.get("gt_mean", float("nan"))),
                    float(values.get("pred_mean", float("nan"))),
                    float(values.get("bulk_occ_p50", float("nan"))),
                    float(values.get("surface_occ_p50", float("nan"))),
                )
            )
        else:
            lines.append(f"{name}: {values:.6f}")
    lines.append("")
    lines.append("[sdf_bins]")
    for name, values in bin_summary.items():
        lines.append(
            "{0}: count={1} pixel_ratio={2:.4f} error_mass={3:.4f} "
            "gt_mean={4:.4f} pred_mean={5:.4f} bulk_p50={6:.4f} surface_p50={7:.4f}".format(
                name,
                int(values["count"]),
                float(values["pixel_ratio"]),
                float(values["error_mass_ratio"]),
                float(values.get("gt_mean", float("nan"))),
                float(values.get("pred_mean", float("nan"))),
                float(values.get("bulk_occ_p50", float("nan"))),
                float(values.get("surface_occ_p50", float("nan"))),
            )
        )
    lines.append("")
    lines.append("[compose_probe]")
    for name, values in compose_probe.items():
        lines.append(
            "{0}: count={1} surface_gate_p50={2:.4f} surface_occ_p50={3:.4f} "
            "surface_contrib_p50={4:.4f} bulk_occ_p50={5:.4f} bulk_contrib_p50={6:.4f} "
            "surface_suppressed_bulk_p50={7:.4f} query_occ_p50={8:.4f} native_occ_p50={9:.4f}".format(
                name,
                int(values["count"]),
                float(values["surface_gate"]["p50"]),
                float(values["surface_occ"]["p50"]),
                float(values["surface_contribution"]["p50"]),
                float(values["bulk_occ"]["p50"]),
                float(values["bulk_contribution"]["p50"]),
                float(values["surface_suppressed_bulk"]["p50"]),
                float(values["query_composed_occ"]["p50"]),
                float(values["native_preview_occ"]["p50"]),
            )
        )
    lines.append("")
    lines.append(f"high_error_and_mismatch_ratio = {mask_summary['high_error_and_mismatch_ratio']:.6f}")
    lines.append(f"high_error_and_boundary_band_ratio = {mask_summary['high_error_and_boundary_band_ratio']:.6f}")
    lines.append(f"high_error_and_internal_void_boundary_ratio = {mask_summary['high_error_and_internal_void_boundary_ratio']:.6f}")
    lines.append(f"high_error_and_exterior_boundary_ratio = {mask_summary['high_error_and_exterior_boundary_ratio']:.6f}")
    lines.append(f"bulk_gap_and_internal_void_boundary_ratio = {mask_summary['bulk_gap_and_internal_void_boundary_ratio']:.6f}")
    lines.append(f"bulk_gap_and_exterior_boundary_ratio = {mask_summary['bulk_gap_and_exterior_boundary_ratio']:.6f}")
    lines.append(f"bulk_gap_but_total_ok_ratio = {mask_summary['bulk_gap_but_total_ok_ratio']:.6f}")
    lines.append(f"air_component_count = {mask_summary['air_component_count']}")
    lines.append(f"internal_air_component_count = {mask_summary['internal_air_component_count']}")
    lines.append(f"gt_edge_to_phase_edge_p90 = {edge_summary['gt_edge_to_phase_edge']['p90']:.6f}")
    lines.append(f"phase_edge_to_gt_edge_p90 = {edge_summary['phase_edge_to_gt_edge']['p90']:.6f}")
    (output_dir / "boundary_root_cause_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _save_figures(output_dir, arrays, {
        "phase_material": phase_material,
        "gt_material": gt_material,
        "phase_gt_mismatch": mismatch,
        "high_error": high_error,
    })
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CT boundary residual root cause on one slice.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--phase1-dir", default=str(DEFAULT_PHASE1_DIR))
    parser.add_argument("--volume-path", default=str(DEFAULT_VOLUME_PATH))
    parser.add_argument("--volume-format", default="dicom")
    parser.add_argument("--slice-idx", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--surface-material-gate-sigma", type=float, default=None)
    parser.add_argument("--material-compose-mode", choices=["surface_first", "bulk_first_material"], default=None)
    args = parser.parse_args()
    output_dir = run(args)
    print(output_dir)


if __name__ == "__main__":
    main()
