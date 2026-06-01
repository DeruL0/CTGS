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
from ct_pipeline.training.objectives.sampling import _sample_filtered_from_candidate_sets
from ct_pipeline.training.sampling import _candidate_count, _sample_occupancy_points, _sample_signed_distance
from ct_pipeline.training.utils import write_key_value_report
from utils.loss_utils import ssim

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep post-hoc CT readouts with a bulk volume-fraction factor.")
    parser.add_argument("--command-file", type=str, required=True, help="Path to the original command.txt.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint .pth file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for diagnostic outputs.")
    parser.add_argument("--slice-idx", type=int, default=None, help="Z slice index. Defaults to the middle slice.")
    parser.add_argument("--interior-margin-vox", type=float, default=0.5, help="Signed-distance margin for confident material interior.")
    parser.add_argument("--interior-sample-count", type=int, default=8192, help="Number of interior points used to estimate W_ref quantiles.")
    parser.add_argument("--pool-sample-count", type=int, default=4096, help="Number of points sampled per diagnostic pool.")
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


def _finite_quantile(values: np.ndarray, quantile: float) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan")
    return float(np.quantile(flat, float(quantile)))


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


def _volume_fraction_readout(
    a_b: torch.Tensor,
    w_b: torch.Tensor,
    intensity_air: float,
    w_ref: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref = max(float(w_ref), 1e-6)
    v_b = (w_b.to(dtype=torch.float32) / ref).clamp(0.0, 1.0)
    air = torch.full_like(v_b, float(intensity_air))
    mu = v_b * a_b.to(dtype=torch.float32) + (1.0 - v_b) * air
    return mu.clamp(0.0, 1.0), v_b


def _query_points_fields(points: torch.Tensor, context, args, training_state) -> dict[str, torch.Tensor]:
    signed_distance = _sample_signed_distance(context.signed_distance_field, points).to(dtype=torch.float32)
    return query_ct_fields_unified(
        points,
        training_state,
        signed_distance=signed_distance,
        config=args,
        intensity_air=float(context.intensity_air),
        include_surface=False,
        train_ct_value=False,
        detach_value_geometry=True,
    )


def _sample_pool_readout(
    pool_key: str,
    sample_count: int,
    context,
    args,
    training_state,
    w_ref: float,
) -> dict[str, float]:
    candidates = context.field_pools.get(pool_key)
    if _candidate_count(candidates) <= 0 or int(sample_count) <= 0:
        return {
            "count": 0,
            "A_b_p95": float("nan"),
            "mu_p95": float("nan"),
            "V_b_p95": float("nan"),
        }

    device = getattr(training_state.xyz, "device", torch.device("cuda"))
    points = _sample_occupancy_points(candidates, int(sample_count), context.spacing_zyx, device=device)
    if points.numel() == 0:
        return {
            "count": 0,
            "A_b_p95": float("nan"),
            "mu_p95": float("nan"),
            "V_b_p95": float("nan"),
        }

    with torch.no_grad():
        fields = _query_points_fields(points, context, args, training_state)
        raw_i_b = fields.get("I_b_raw", fields["I_b"]).to(dtype=torch.float32)
        den_b = fields["den_b"].to(dtype=torch.float32)
        a_b = fields.get("A_b", bulk_intensity_readout(raw_i_b, den_b)).to(dtype=torch.float32)
        mu, v_b = _volume_fraction_readout(a_b, den_b, float(context.intensity_air), w_ref)

    return {
        "count": int(points.shape[0]),
        "A_b_p95": _finite_quantile(_to_numpy(a_b), 0.95),
        "mu_p95": _finite_quantile(_to_numpy(mu), 0.95),
        "V_b_p95": _finite_quantile(_to_numpy(v_b), 0.95),
    }


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

    training_state = prepare_ct_training_state(
        context.gaussians,
        spacing_zyx=context.spacing_zyx,
        truncation_sigma=float(args.ct_gaussian_truncation_sigma),
        grid_cell_voxels=int(args.ct_grid_cell_voxels),
        signed_distance_field=context.signed_distance_field,
        curvature_field=context.intensity_field_cache.get("curvature_proxy"),
    )
    device = getattr(training_state.xyz, "device", torch.device("cuda"))
    intensity_air = float(context.intensity_air)
    interior_margin = max(float(cli_args.interior_margin_vox), 0.0)

    with torch.no_grad():
        interior_points, interior_sdf = _sample_filtered_from_candidate_sets(
            (context.field_pools.get("support"),),
            int(cli_args.interior_sample_count),
            context,
            device=device,
            signed_distance_predicate=lambda sdf: sdf < -float(interior_margin),
            oversample=6,
        )
        if interior_points.numel() == 0:
            raise RuntimeError("No confident material interior points were available for W_ref estimation.")
        interior_fields = query_ct_fields_unified(
            interior_points,
            training_state,
            signed_distance=interior_sdf,
            config=args,
            intensity_air=intensity_air,
            include_surface=False,
            train_ct_value=False,
            detach_value_geometry=True,
        )
        interior_den = interior_fields["den_b"].to(dtype=torch.float32)
        interior_den_np = _to_numpy(interior_den)
        w_refs = {
            "p50": _finite_quantile(interior_den_np, 0.50),
            "p75": _finite_quantile(interior_den_np, 0.75),
            "p90": _finite_quantile(interior_den_np, 0.90),
        }

        slice_idx = int(cli_args.slice_idx) if cli_args.slice_idx is not None else int(context.volume_shape[0] // 2)
        patch_size = (int(context.volume_shape[1]), int(context.volume_shape[2]))
        origin = (0, 0)

        gt = sample_gt_slice_patch(context.volume_cuda, "z", slice_idx, origin, patch_size).to(dtype=torch.float32)
        rr, cc = torch.meshgrid(
            torch.arange(patch_size[0], dtype=torch.float32, device=device),
            torch.arange(patch_size[1], dtype=torch.float32, device=device),
            indexing="ij",
        )
        points = _build_query_points_from_base(rr, cc, 0, slice_idx, origin, context.spacing_zyx)
        signed_distance = _sample_signed_distance(context.signed_distance_field, points).to(dtype=torch.float32)
        fields = query_ct_fields_unified(
            points,
            training_state,
            signed_distance=signed_distance,
            config=args,
            intensity_air=intensity_air,
            include_surface=False,
            train_ct_value=False,
            detach_value_geometry=True,
        )

    signed_distance_2d = signed_distance.reshape(patch_size).to(dtype=torch.float32)
    material_mask = signed_distance_2d <= 0.0
    interior_mask = signed_distance_2d < -float(interior_margin)
    boundary_mask = torch.abs(signed_distance_2d) <= float(interior_margin)
    air_value = torch.full(patch_size, intensity_air, dtype=torch.float32, device=device)

    raw_i_b = fields.get("I_b_raw", fields["I_b"]).reshape(patch_size).to(dtype=torch.float32)
    den_b = fields["den_b"].reshape(patch_size).to(dtype=torch.float32)
    a_b = fields.get("A_b", bulk_intensity_readout(raw_i_b, den_b)).reshape(patch_size).to(dtype=torch.float32)

    plain_a_b = a_b.clamp(0.0, 1.0)
    masked_a_b = torch.where(material_mask, plain_a_b, air_value)
    den_np = _to_numpy(den_b)

    report_rows: list[tuple[str, object]] = [
        ("checkpoint", str(checkpoint_path)),
        ("command_file", str(command_file)),
        ("slice_idx", slice_idx),
        ("intensity_air", intensity_air),
        ("interior_margin_vox", interior_margin),
        ("interior_sample_count", int(interior_points.shape[0])),
        ("slice_material_pixel_count", int(material_mask.sum().item())),
        ("slice_interior_pixel_count", int(interior_mask.sum().item())),
        ("slice_boundary_pixel_count", int(boundary_mask.sum().item())),
        ("w_ref_interior_p50", w_refs["p50"]),
        ("w_ref_interior_p75", w_refs["p75"]),
        ("w_ref_interior_p90", w_refs["p90"]),
    ]

    plain_metrics = _full_slice_metrics(plain_a_b, gt)
    masked_metrics = _full_slice_metrics(masked_a_b, gt)
    _append_metrics(report_rows, "plain_A_b", plain_metrics)
    _append_metrics(report_rows, "masked_A_b", masked_metrics)
    report_rows.extend(
        [
            ("plain_A_b_air_slice_p95", _finite_quantile(_to_numpy(plain_a_b[~material_mask]), 0.95)),
            ("masked_A_b_air_slice_p95", _finite_quantile(_to_numpy(masked_a_b[~material_mask]), 0.95)),
            ("plain_A_b_boundary_shell_mae", float(torch.mean(torch.abs(plain_a_b[boundary_mask] - gt[boundary_mask])).item()) if torch.any(boundary_mask) else float("nan")),
            ("masked_A_b_boundary_shell_mae", float(torch.mean(torch.abs(masked_a_b[boundary_mask] - gt[boundary_mask])).item()) if torch.any(boundary_mask) else float("nan")),
        ]
    )

    variant_images: dict[str, torch.Tensor] = {}
    variant_vb: dict[str, torch.Tensor] = {}
    pool_keys = ("void_air", "exterior_air_near_band", "cavity_air_shell", "cavity_material_shell")
    for label, w_ref in w_refs.items():
        mu_pred, v_b = _volume_fraction_readout(plain_a_b, den_b, intensity_air, w_ref)
        variant_images[label] = mu_pred
        variant_vb[label] = v_b
        metrics = _full_slice_metrics(mu_pred, gt)
        prefix = f"vf_clamp_{label}"
        _append_metrics(report_rows, prefix, metrics)
        report_rows.extend(
            [
                (f"{prefix}_w_ref", w_ref),
                (f"{prefix}_air_slice_p95", _finite_quantile(_to_numpy(mu_pred[~material_mask]), 0.95)),
                (f"{prefix}_interior_slice_p50", _finite_quantile(_to_numpy(mu_pred[interior_mask]), 0.50)),
                (f"{prefix}_boundary_shell_mae", float(torch.mean(torch.abs(mu_pred[boundary_mask] - gt[boundary_mask])).item()) if torch.any(boundary_mask) else float("nan")),
                (f"{prefix}_V_b_air_slice_p95", _finite_quantile(_to_numpy(v_b[~material_mask]), 0.95)),
                (f"{prefix}_V_b_interior_slice_p50", _finite_quantile(_to_numpy(v_b[interior_mask]), 0.50)),
            ]
        )
        for pool_key in pool_keys:
            pool_metrics = _sample_pool_readout(
                pool_key,
                int(cli_args.pool_sample_count),
                context,
                args,
                training_state,
                w_ref,
            )
            report_rows.extend(
                [
                    (f"{prefix}_{pool_key}_sample_count", pool_metrics["count"]),
                    (f"{prefix}_{pool_key}_A_b_p95", pool_metrics["A_b_p95"]),
                    (f"{prefix}_{pool_key}_mu_p95", pool_metrics["mu_p95"]),
                    (f"{prefix}_{pool_key}_V_b_p95", pool_metrics["V_b_p95"]),
                ]
            )

    best_label = max(w_refs.keys(), key=lambda name: _full_slice_metrics(variant_images[name], gt)["ssim"])
    best_metrics = _full_slice_metrics(variant_images[best_label], gt)
    report_rows.append(("best_variant_by_ssim", best_label))
    _append_metrics(report_rows, "best_variant", best_metrics)

    metrics_path = output_dir / f"volume_fraction_readout_slice_z{slice_idx:03d}.txt"
    write_key_value_report(metrics_path, report_rows)

    figure_path = output_dir / f"volume_fraction_readout_slice_z{slice_idx:03d}.png"
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=False)
    panels = [
        ("GT", _to_numpy(gt), None),
        (f"A_b plain\nMAE={plain_metrics['mae']:.4f} SSIM={plain_metrics['ssim']:.4f}", _to_numpy(plain_a_b), None),
        (f"Current masked A_b\nMAE={masked_metrics['mae']:.4f} SSIM={masked_metrics['ssim']:.4f}", _to_numpy(masked_a_b), None),
        ("Kernel mass W_b=den_b", den_np, "viridis"),
        (
            f"mu_pred clamp p50\nW_ref={w_refs['p50']:.3f}\nMAE={_full_slice_metrics(variant_images['p50'], gt)['mae']:.4f} SSIM={_full_slice_metrics(variant_images['p50'], gt)['ssim']:.4f}",
            _to_numpy(variant_images["p50"]),
            None,
        ),
        (
            f"mu_pred clamp p75\nW_ref={w_refs['p75']:.3f}\nMAE={_full_slice_metrics(variant_images['p75'], gt)['mae']:.4f} SSIM={_full_slice_metrics(variant_images['p75'], gt)['ssim']:.4f}",
            _to_numpy(variant_images["p75"]),
            None,
        ),
        (
            f"mu_pred clamp p90\nW_ref={w_refs['p90']:.3f}\nMAE={_full_slice_metrics(variant_images['p90'], gt)['mae']:.4f} SSIM={_full_slice_metrics(variant_images['p90'], gt)['ssim']:.4f}",
            _to_numpy(variant_images["p90"]),
            None,
        ),
        (
            f"V_b ({best_label})\nW_ref={w_refs[best_label]:.3f}",
            _to_numpy(variant_vb[best_label]),
            "magma",
        ),
    ]

    for ax, (title, image, cmap) in zip(axes.reshape(-1), panels):
        if cmap is None:
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(image, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle("Post-hoc bulk volume-fraction readout sweep", fontsize=12)
    plt.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
