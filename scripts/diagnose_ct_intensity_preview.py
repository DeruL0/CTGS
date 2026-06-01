from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ct_pipeline import extract_ct_model_args, extract_ct_optimization_args
from ct_pipeline.backend import prepare_ct_training_state
from ct_pipeline.rendering.slices import _build_query_points_from_base, sample_gt_slice_patch
from ct_pipeline.rendering.fields import bulk_intensity_readout, query_ct_fields_unified
from ct_pipeline.training.config import build_parser, validate_ct_training_args
from ct_pipeline.training.bootstrap import prepare_ct_training_bootstrap
from ct_pipeline.training.reporting import _record_volume_material_intensity_metrics
from ct_pipeline.training.sampling import _sample_signed_distance
from ct_pipeline.training.utils import write_key_value_report
from utils.loss_utils import ssim

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose CTGS bulk intensity preview from a saved checkpoint.")
    parser.add_argument("--command-file", type=str, required=True, help="Path to the original command.txt.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint .pth file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for diagnostic outputs.")
    parser.add_argument("--slice-idx", type=int, default=None, help="Z slice index. Defaults to the middle slice.")
    parser.add_argument("--volume-sample-count", type=int, default=16384, help="Material-side 3D sample count.")
    return parser.parse_args()


def _load_training_args(command_file: Path, checkpoint: Path, bootstrap_dir: Path) -> argparse.Namespace:
    command_text = command_file.read_text(encoding="utf-8").strip()
    tokens = shlex.split(command_text, posix=False)
    if tokens and tokens[0].lower().endswith("train_ct.py"):
        tokens = tokens[1:]
    parser = build_parser()
    args = parser.parse_args(tokens)
    args.start_checkpoint = str(checkpoint)
    args.model_path = str(bootstrap_dir)
    args.quiet = True
    args.wandb = False
    args.load_ply = False
    validate_ct_training_args(args)
    return args


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _finite_quantiles(values: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return [float("nan") for _ in quantiles]
    measured = np.quantile(flat, np.asarray(quantiles, dtype=np.float64))
    return [float(value) for value in measured]


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return float("nan")
    x = x[valid]
    y = y[valid]
    if np.allclose(x, x.mean()) or np.allclose(y, y.mean()):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return 1.0, 0.0
    design = np.stack([x[valid], np.ones(int(valid.sum()), dtype=np.float64)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(design, y[valid], rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _full_slice_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    pred = pred.to(dtype=torch.float32)
    gt = gt.to(dtype=torch.float32)
    pred_bchw = pred.unsqueeze(0).unsqueeze(0)
    gt_bchw = gt.unsqueeze(0).unsqueeze(0)
    mse = torch.mean((pred_bchw - gt_bchw) ** 2)
    return {
        "mae": float(torch.mean(torch.abs(pred - gt)).item()),
        "rmse": float(torch.sqrt(mse).item()),
        "psnr": float((-10.0 * torch.log10(mse.clamp_min(1e-10))).item()),
        "ssim": float(ssim(pred_bchw, gt_bchw).item()),
    }


def _append_metrics(rows: list[tuple[str, object]], prefix: str, metrics: dict[str, float]) -> None:
    for key in ("mae", "rmse", "psnr", "ssim"):
        rows.append((f"{prefix}_{key}", metrics[key]))


def main() -> int:
    cli_args = _parse_args()
    checkpoint_path = Path(cli_args.checkpoint).resolve()
    command_file = Path(cli_args.command_file).resolve()
    output_dir = Path(cli_args.output_dir).resolve()
    bootstrap_dir = output_dir / "_bootstrap_runtime"
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    args = _load_training_args(command_file, checkpoint_path, bootstrap_dir)
    dataset = extract_ct_model_args(args)
    opt = extract_ct_optimization_args(args)
    context = prepare_ct_training_bootstrap(dataset, opt, args, args.start_checkpoint)
    if context.tb_writer is not None:
        context.tb_writer.close()

    slice_idx = int(cli_args.slice_idx) if cli_args.slice_idx is not None else int(context.volume_shape[0] // 2)
    patch_size = (int(context.volume_shape[1]), int(context.volume_shape[2]))
    origin = (0, 0)

    training_state = prepare_ct_training_state(
        context.gaussians,
        spacing_zyx=context.spacing_zyx,
        truncation_sigma=float(args.ct_gaussian_truncation_sigma),
        grid_cell_voxels=int(args.ct_grid_cell_voxels),
        signed_distance_field=context.signed_distance_field,
        curvature_field=context.intensity_field_cache.get("curvature_proxy"),
    )
    volume_metrics: dict[str, float | int] = {}
    model_xyz = getattr(training_state, "xyz", None)
    device = model_xyz.device if isinstance(model_xyz, torch.Tensor) else context.volume_cuda.device
    dtype = model_xyz.dtype if isinstance(model_xyz, torch.Tensor) and torch.is_floating_point(model_xyz) else torch.float32
    _record_volume_material_intensity_metrics(
        volume_metrics,
        training_state,
        context.analysis_gpu,
        context.spacing_zyx,
        device,
        dtype,
        signed_distance_field=context.signed_distance_field,
        volume_cuda=context.volume_cuda,
        config=args,
        intensity_air=float(context.intensity_air),
        max_count=max(1, int(cli_args.volume_sample_count)),
    )

    with torch.no_grad():
        gt = sample_gt_slice_patch(context.volume_cuda, "z", slice_idx, origin, patch_size).to(dtype=torch.float32)
        rr, cc = torch.meshgrid(
            torch.arange(patch_size[0], dtype=torch.float32, device=gt.device),
            torch.arange(patch_size[1], dtype=torch.float32, device=gt.device),
            indexing="ij",
        )
        points = _build_query_points_from_base(rr, cc, 0, slice_idx, origin, context.spacing_zyx)
        signed_distance = _sample_signed_distance(context.signed_distance_field, points).to(dtype=torch.float32)
        fields = query_ct_fields_unified(
            points,
            training_state,
            signed_distance=signed_distance,
            config=args,
            intensity_air=float(context.intensity_air),
            include_surface=True,
            train_ct_value=False,
            detach_value_geometry=True,
            apply_bulk_gate=False,
        )

    raw_i_b = fields.get("I_b_raw", fields["I_b"]).reshape(patch_size).to(dtype=torch.float32)
    den_b = fields["den_b"].reshape(patch_size).to(dtype=torch.float32)
    eps = 1e-6
    avg_atten = bulk_intensity_readout(raw_i_b, den_b, eps=eps).to(dtype=torch.float32)
    material_mask = (signed_distance.reshape(patch_size) <= 0.0)
    air_value = torch.full_like(raw_i_b, float(context.intensity_air))

    raw_preview = raw_i_b.clamp(0.0, 1.0)
    avg_preview = avg_atten.clamp(0.0, 1.0)
    avg_material = torch.where(material_mask, avg_preview, air_value)

    material_np = _to_numpy(material_mask)
    gt_np = _to_numpy(gt)
    avg_np = _to_numpy(avg_preview)
    slope, intercept = _fit_affine(avg_np[material_np], gt_np[material_np])
    affine_material = torch.where(material_mask, (slope * avg_preview + intercept).clamp(0.0, 1.0), air_value)

    raw_metrics = _full_slice_metrics(raw_preview, gt)
    avg_metrics = _full_slice_metrics(avg_preview, gt)
    avg_material_metrics = _full_slice_metrics(avg_material, gt)
    affine_metrics = _full_slice_metrics(affine_material, gt)

    raw_np = _to_numpy(raw_i_b)
    den_np = _to_numpy(den_b)
    raw_preview_np = _to_numpy(raw_preview)
    avg_material_np = _to_numpy(avg_material)
    affine_np = _to_numpy(affine_material)
    avg_preview_np = _to_numpy(avg_preview)
    avg_error_np = np.abs(avg_preview_np - gt_np)

    material_values_raw = raw_np[material_np]
    material_values_den = den_np[material_np]
    material_values_avg = avg_np[material_np]
    material_values_gt = gt_np[material_np]
    air_values_raw = raw_np[~material_np]

    bulk_attenuation = getattr(training_state, "bulk_attenuation", None)
    bulk_ct_value = getattr(training_state, "bulk_ct_value", None)
    bulk_attenuation_np = _to_numpy(bulk_attenuation) if bulk_attenuation is not None and bulk_attenuation.numel() > 0 else np.array([], dtype=np.float32)
    bulk_ct_value_np = _to_numpy(bulk_ct_value) if bulk_ct_value is not None and bulk_ct_value.numel() > 0 else np.array([], dtype=np.float32)

    report_rows: list[tuple[str, object]] = [
        ("checkpoint", str(checkpoint_path)),
        ("command_file", str(command_file)),
        ("slice_idx", slice_idx),
        ("material_pixel_count", int(material_np.sum())),
        ("air_pixel_count", int((~material_np).sum())),
        ("affine_avg_material_slope", slope),
        ("affine_avg_material_intercept", intercept),
        ("material_raw_i_b_sat_hi_ratio", float(np.mean(material_values_raw >= 1.0))),
        ("material_raw_i_b_sat_low_ratio", float(np.mean(material_values_raw <= 0.05))),
        ("material_den_b_gt1_ratio", float(np.mean(material_values_den >= 1.0))),
        ("material_corr_raw_i_b_vs_gt", _pearson_corr(material_values_raw, material_values_gt)),
        ("material_corr_avg_attn_vs_gt", _pearson_corr(material_values_avg, material_values_gt)),
    ]
    for key in (
        "material_volume_MAE",
        "material_volume_RMSE",
        "material_volume_corr",
        "material_volume_A_b_p10",
        "material_volume_A_b_p50",
        "material_volume_A_b_p90",
        "material_volume_sample_count",
    ):
        report_rows.append((key, volume_metrics.get(key, float("nan"))))

    for prefix, array in (
        ("gt_material", material_values_gt),
        ("raw_i_b_material", material_values_raw),
        ("den_b_material", material_values_den),
        ("avg_attn_material", material_values_avg),
        ("raw_i_b_air", air_values_raw),
        ("bulk_attenuation_gaussian", bulk_attenuation_np),
        ("bulk_ct_value_gaussian", bulk_ct_value_np),
    ):
        p10, p50, p90 = _finite_quantiles(array, (0.10, 0.50, 0.90))
        report_rows.extend(
            [
                (f"{prefix}_p10", p10),
                (f"{prefix}_p50", p50),
                (f"{prefix}_p90", p90),
            ]
        )

    _append_metrics(report_rows, "raw_preview", raw_metrics)
    _append_metrics(report_rows, "avg_preview", avg_metrics)
    _append_metrics(report_rows, "avg_material", avg_material_metrics)
    _append_metrics(report_rows, "affine_avg_material", affine_metrics)

    metrics_path = output_dir / f"intensity_diag_slice_z{slice_idx:03d}.txt"
    write_key_value_report(metrics_path, report_rows)

    figure_path = output_dir / f"intensity_diag_slice_z{slice_idx:03d}.png"
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=False)
    panels = [
        ("GT", gt_np, None),
        (f"Raw I_b clamp\nMAE={raw_metrics['mae']:.4f} SSIM={raw_metrics['ssim']:.4f}", raw_preview_np, None),
        ("Kernel mass den_b", den_np, "viridis"),
        (f"I_b / den_b\nMAE={avg_metrics['mae']:.4f} SSIM={avg_metrics['ssim']:.4f}", _to_numpy(avg_preview), None),
        ("|I_b / den_b - GT|", avg_error_np, "magma"),
        ("Raw I_b in air/void", np.where(material_np, 0.0, raw_preview_np), None),
    ]
    for ax, (title, image, cmap) in zip(axes.reshape(-1), panels):
        if cmap is None:
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(image, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.savefig(figure_path, dpi=120)
    plt.close(fig)

    ref_path = output_dir / f"intensity_diag_slice_z{slice_idx:03d}_raw_ref.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=False)
    for ax, (title, image, metrics) in zip(
        axes,
        (
            ("GT", gt_np, None),
            ("Raw A_b", avg_preview_np, avg_metrics),
            ("Raw I_b air/void", np.where(material_np, 0.0, raw_preview_np), None),
        ),
    ):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        if metrics is not None:
            title = f"{title}\nMAE={metrics['mae']:.4f} SSIM={metrics['ssim']:.4f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.savefig(ref_path, dpi=120)
    plt.close(fig)

    print(f"wrote_metrics={metrics_path}")
    print(f"wrote_figure={figure_path}")
    print(f"wrote_raw_ref={ref_path}")
    print(
        "summary "
        f"raw_mae={raw_metrics['mae']:.6f} raw_ssim={raw_metrics['ssim']:.6f} "
        f"avg_mat_mae={avg_material_metrics['mae']:.6f} avg_mat_ssim={avg_material_metrics['ssim']:.6f} "
        f"affine_mae={affine_metrics['mae']:.6f} affine_ssim={affine_metrics['ssim']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
