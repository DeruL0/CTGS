from __future__ import annotations

import argparse
import re
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
from ct_pipeline.rendering.fields import query_ct_fields_unified
from ct_pipeline.training.config import build_parser, validate_ct_training_args
from ct_pipeline.training.preview import _correct_bulk_intensity_preview
from ct_pipeline.training.bootstrap import prepare_ct_training_bootstrap
from ct_pipeline.training.objectives.modes import _intensity_geometry_flags
from ct_pipeline.training.objectives.prediction import _bulk_intensity_from_fields
from ct_pipeline.training.sampling import _sample_signed_distance
from ct_pipeline.training.utils import write_key_value_report
from ct_pipeline.training.losses import sample_volume_field

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CTGS training and preview readouts on the same slice grid.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth.")
    parser.add_argument("--command-file", type=str, default=None, help="Path to original command.txt. Defaults to sibling command.txt.")
    parser.add_argument("--axis", type=str, default="z", choices=("z", "y", "x"), help="Slice axis.")
    parser.add_argument("--slice-idx", type=int, default=None, help="Slice index along the chosen axis.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for report, npz, and png.")
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


def _axis_to_index(axis: str) -> int:
    axis_name = str(axis).lower()
    mapping = {"z": 0, "y": 1, "x": 2}
    if axis_name not in mapping:
        raise ValueError(f"Unsupported axis {axis!r}.")
    return mapping[axis_name]


def _infer_iteration_from_checkpoint(checkpoint: Path) -> int:
    match = re.search(r"(\d+)", checkpoint.stem)
    return int(match.group(1)) if match else 0


def _resolve_command_file(checkpoint: Path, command_file: str | None) -> Path:
    if command_file is not None:
        return Path(command_file).resolve()
    candidate = checkpoint.resolve().parent / "command.txt"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("command.txt not provided and not found next to checkpoint.")


def _default_slice_idx(volume_shape: tuple[int, int, int], axis: str) -> int:
    axis_index = _axis_to_index(axis)
    if axis_index == 0:
        return int(volume_shape[0] // 2)
    if axis_index == 1:
        return int(volume_shape[1] // 2)
    return int(volume_shape[2] // 2)


def _patch_size_for_axis(volume_shape: tuple[int, int, int], axis: str) -> tuple[int, int]:
    axis_index = _axis_to_index(axis)
    if axis_index == 0:
        return int(volume_shape[1]), int(volume_shape[2])
    if axis_index == 1:
        return int(volume_shape[0]), int(volume_shape[2])
    return int(volume_shape[0]), int(volume_shape[1])


def _slice_tensor(volume_like, axis: str, slice_idx: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    data = torch.as_tensor(volume_like, dtype=dtype, device=device)
    axis_index = _axis_to_index(axis)
    if axis_index == 0:
        return data[int(slice_idx)]
    if axis_index == 1:
        return data[:, int(slice_idx), :]
    return data[:, :, int(slice_idx)]


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _diff_stats(a: torch.Tensor, b: torch.Tensor, prefix: str) -> dict[str, float]:
    diff = torch.abs(a.to(dtype=torch.float32) - b.to(dtype=torch.float32)).reshape(-1)
    valid = torch.isfinite(diff)
    if not torch.any(valid):
        return {
            f"{prefix}_max": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p95": float("nan"),
        }
    diff = diff[valid]
    return {
        f"{prefix}_max": float(diff.max().item()),
        f"{prefix}_mean": float(diff.mean().item()),
        f"{prefix}_p95": float(torch.quantile(diff, 0.95).item()),
    }


def _mask_stats(lhs: torch.Tensor, rhs: torch.Tensor, prefix: str) -> dict[str, float]:
    lhs_bool = lhs.to(dtype=torch.bool)
    rhs_bool = rhs.to(dtype=torch.bool)
    agreement = lhs_bool == rhs_bool
    lhs_count = int(lhs_bool.sum().item())
    rhs_count = int(rhs_bool.sum().item())
    mismatch = ~agreement
    false_positive = lhs_bool & (~rhs_bool)
    false_negative = (~lhs_bool) & rhs_bool
    return {
        f"{prefix}_agreement": float(agreement.to(dtype=torch.float32).mean().item()),
        f"{prefix}_mismatch_ratio": float(mismatch.to(dtype=torch.float32).mean().item()),
        f"{prefix}_false_positive_ratio": float(false_positive.to(dtype=torch.float32).mean().item()),
        f"{prefix}_false_negative_ratio": float(false_negative.to(dtype=torch.float32).mean().item()),
        f"{prefix}_lhs_count": float(lhs_count),
        f"{prefix}_rhs_count": float(rhs_count),
    }


def _save_debug_png(output_path: Path, arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=False)
    panels = [
        ("GT indexed", arrays["gt_slice"], "gray", 0.0, 1.0),
        ("GT sampled", arrays["gt_sampled"], "gray", 0.0, 1.0),
        ("|GT sampled - indexed|", arrays["gt_sample_abs_diff"], "magma", 0.0, max(1e-6, float(arrays["gt_sample_abs_diff"].max()))),
        ("Sampled SDF", arrays["sdf_sampled"], "coolwarm", float(np.min(arrays["sdf_sampled"])), float(np.max(arrays["sdf_sampled"]))),
        ("raw A_b train", arrays["raw_A_b_train_path"], "gray", 0.0, 1.0),
        ("raw A_b preview", arrays["raw_A_b_preview_path"], "gray", 0.0, 1.0),
        ("|A_b train-preview|", arrays["raw_A_b_abs_diff"], "magma", 0.0, max(1e-6, float(arrays["raw_A_b_abs_diff"].max()))),
        ("raw W_b train", arrays["raw_W_b_train_path"], "viridis", float(np.min(arrays["raw_W_b_train_path"])), float(np.max(arrays["raw_W_b_train_path"]))),
        ("raw rho_b train", arrays["raw_rho_b_train_path"], "viridis", float(np.min(arrays["raw_rho_b_train_path"])), float(np.max(arrays["raw_rho_b_train_path"]))),
        ("hard mask", arrays["hard_mask"].astype(np.float32), "gray", 0.0, 1.0),
        ("current bulk_only", arrays["bulk_only_after_mask"], "gray", 0.0, 1.0),
        ("|bulk_only train-preview|", arrays["bulk_only_abs_diff"], "magma", 0.0, max(1e-6, float(arrays["bulk_only_abs_diff"].max()))),
    ]
    for ax, (title, image, cmap, vmin, vmax) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def debug_compare_training_vs_preview_readout(
    checkpoint: str | Path,
    axis: str = "z",
    slice_idx: int | None = None,
    *,
    command_file: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint).resolve()
    command_path = _resolve_command_file(checkpoint_path, str(command_file) if command_file is not None else None)
    output_root = Path(output_dir).resolve() if output_dir is not None else checkpoint_path.resolve().parent / "diagnostics" / "preview_alignment"
    bootstrap_dir = output_root / "_bootstrap_runtime"
    output_root.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    args = _load_training_args(command_path, checkpoint_path, bootstrap_dir)
    dataset = extract_ct_model_args(args)
    opt = extract_ct_optimization_args(args)
    context = prepare_ct_training_bootstrap(dataset, opt, args, args.start_checkpoint)
    if context.tb_writer is not None:
        context.tb_writer.close()

    volume_shape = tuple(int(value) for value in context.volume_shape)
    axis_name = str(axis).lower()
    axis_index = _axis_to_index(axis_name)
    chosen_slice_idx = int(slice_idx) if slice_idx is not None else _default_slice_idx(volume_shape, axis_name)
    patch_size = _patch_size_for_axis(volume_shape, axis_name)
    origin = (0, 0)
    iteration = _infer_iteration_from_checkpoint(checkpoint_path)

    training_state = prepare_ct_training_state(
        context.gaussians,
        spacing_zyx=context.spacing_zyx,
        truncation_sigma=float(args.ct_gaussian_truncation_sigma),
        grid_cell_voxels=int(args.ct_grid_cell_voxels),
        signed_distance_field=context.signed_distance_field,
        curvature_field=context.intensity_field_cache.get("curvature_proxy"),
    )

    gt_slice = sample_gt_slice_patch(context.volume_cuda, axis_name, chosen_slice_idx, origin, patch_size).to(dtype=torch.float32)
    rr, cc = torch.meshgrid(
        torch.arange(patch_size[0], dtype=torch.float32, device=gt_slice.device),
        torch.arange(patch_size[1], dtype=torch.float32, device=gt_slice.device),
        indexing="ij",
    )
    points_xyz = _build_query_points_from_base(rr, cc, axis_index, chosen_slice_idx, origin, context.spacing_zyx)
    signed_distance = _sample_signed_distance(context.signed_distance_field, points_xyz).to(dtype=torch.float32)
    flags = _intensity_geometry_flags(args, iteration)
    volume_field = context.volume_cuda.reshape(1, 1, *volume_shape)

    with torch.no_grad():
        train_fields = query_ct_fields_unified(
            points_xyz,
            training_state,
            signed_distance=signed_distance,
            config=args,
            intensity_air=float(context.intensity_air),
            include_surface=False,
            bulk_train_opacity=bool(flags.get("bulk_train_opacity", False)),
            bulk_train_scale=bool(flags.get("bulk_train_scale", False)),
            bulk_scale_grad=float(flags.get("bulk_scale_grad", 1.0)),
            train_ct_value=False,
        )
        preview_fields = query_ct_fields_unified(
            points_xyz,
            training_state,
            signed_distance=signed_distance,
            config=args,
            intensity_air=float(context.intensity_air),
            include_surface=True,
            train_ct_value=True,
            detach_value_geometry=True,
        )
        preview_corrected = _correct_bulk_intensity_preview(
            preview_fields,
            signed_distance,
            float(context.intensity_air),
            args,
        )
        gt_sampled = sample_volume_field(volume_field, points_xyz, context.spacing_zyx).reshape(patch_size).to(dtype=torch.float32)

    raw_A_b_train = _bulk_intensity_from_fields(train_fields).reshape(patch_size).to(dtype=torch.float32)
    raw_W_b_train = train_fields["W_b"].reshape(patch_size).to(dtype=torch.float32)
    raw_rho_b_train = train_fields["rho_b"].reshape(patch_size).to(dtype=torch.float32)
    raw_A_b_preview = _bulk_intensity_from_fields(preview_fields).reshape(patch_size).to(dtype=torch.float32)
    raw_W_b_preview = preview_fields["W_b"].reshape(patch_size).to(dtype=torch.float32)
    raw_rho_b_preview = preview_fields["rho_b"].reshape(patch_size).to(dtype=torch.float32)
    bulk_only_after_mask = preview_corrected["bulk_only"].reshape(patch_size).to(dtype=torch.float32)

    sdf_sampled = signed_distance.reshape(patch_size).to(dtype=torch.float32)
    sdf_native = _slice_tensor(
        context.signed_distance_field["signed_distance_native"],
        axis_name,
        chosen_slice_idx,
        device=gt_slice.device,
        dtype=torch.float32,
    )
    native_sdf_hard_mask = sdf_native <= 0.0
    native_material_mask_np = np.asarray(context.analysis["material_mask"], dtype=bool)
    if axis_index == 0:
        native_material_mask = torch.as_tensor(native_material_mask_np[chosen_slice_idx], device=gt_slice.device, dtype=torch.bool)
    elif axis_index == 1:
        native_material_mask = torch.as_tensor(native_material_mask_np[:, chosen_slice_idx, :], device=gt_slice.device, dtype=torch.bool)
    else:
        native_material_mask = torch.as_tensor(native_material_mask_np[:, :, chosen_slice_idx], device=gt_slice.device, dtype=torch.bool)
    sampled_hard_mask = sdf_sampled <= 0.0
    train_bulk_only = torch.where(
        sampled_hard_mask,
        raw_A_b_train.clamp(0.0, 1.0),
        torch.full_like(raw_A_b_train, float(context.intensity_air)),
    )

    point_ranges = {
        "point_x_min": float(points_xyz[:, 0].min().item()),
        "point_x_max": float(points_xyz[:, 0].max().item()),
        "point_y_min": float(points_xyz[:, 1].min().item()),
        "point_y_max": float(points_xyz[:, 1].max().item()),
        "point_z_min": float(points_xyz[:, 2].min().item()),
        "point_z_max": float(points_xyz[:, 2].max().item()),
    }

    report_rows: list[tuple[str, object]] = [
        ("checkpoint", str(checkpoint_path)),
        ("command_file", str(command_path)),
        ("axis", axis_name),
        ("axis_index", axis_index),
        ("slice_idx", chosen_slice_idx),
        ("iteration", iteration),
        ("patch_h", int(patch_size[0])),
        ("patch_w", int(patch_size[1])),
        ("spacing_z", float(context.spacing_zyx[0])),
        ("spacing_y", float(context.spacing_zyx[1])),
        ("spacing_x", float(context.spacing_zyx[2])),
        ("preview_mode_expected", "bulk_raw_A_b"),
    ]
    report_rows.extend(point_ranges.items())
    report_rows.extend(_diff_stats(raw_A_b_train, raw_A_b_preview, "readout_diff_A_b").items())
    report_rows.extend(_diff_stats(raw_W_b_train, raw_W_b_preview, "readout_diff_W_b").items())
    report_rows.extend(_diff_stats(raw_rho_b_train, raw_rho_b_preview, "readout_diff_rho_b").items())
    report_rows.extend(_diff_stats(train_bulk_only, bulk_only_after_mask, "readout_diff_bulk_only").items())
    report_rows.extend(_diff_stats(gt_sampled, gt_slice, "coord_diff_gt_sampled_vs_indexed").items())
    report_rows.extend(_diff_stats(sdf_sampled, sdf_native, "coord_diff_sdf_sampled_vs_native").items())
    report_rows.extend(_mask_stats(sampled_hard_mask, native_material_mask, "mask_agreement_native_mask").items())
    report_rows.extend(_mask_stats(sampled_hard_mask, native_sdf_hard_mask, "mask_agreement_native_sdf").items())
    report_rows.extend(_mask_stats(native_sdf_hard_mask, native_material_mask, "phase1_internal_mask_vs_sdf").items())

    arrays = {
        "gt_slice": _to_numpy(gt_slice),
        "gt_sampled": _to_numpy(gt_sampled),
        "gt_sample_abs_diff": _to_numpy(torch.abs(gt_sampled - gt_slice)),
        "raw_A_b_train_path": _to_numpy(raw_A_b_train),
        "raw_W_b_train_path": _to_numpy(raw_W_b_train),
        "raw_rho_b_train_path": _to_numpy(raw_rho_b_train),
        "raw_A_b_preview_path": _to_numpy(raw_A_b_preview),
        "raw_W_b_preview_path": _to_numpy(raw_W_b_preview),
        "raw_rho_b_preview_path": _to_numpy(raw_rho_b_preview),
        "raw_A_b_abs_diff": _to_numpy(torch.abs(raw_A_b_train - raw_A_b_preview)),
        "raw_W_b_abs_diff": _to_numpy(torch.abs(raw_W_b_train - raw_W_b_preview)),
        "raw_rho_b_abs_diff": _to_numpy(torch.abs(raw_rho_b_train - raw_rho_b_preview)),
        "sdf_sampled": _to_numpy(sdf_sampled),
        "sdf_native_slice": _to_numpy(sdf_native),
        "hard_mask": _to_numpy(sampled_hard_mask.to(dtype=torch.float32)),
        "hard_mask_native": _to_numpy(native_material_mask.to(dtype=torch.float32)),
        "hard_mask_native_sdf": _to_numpy(native_sdf_hard_mask.to(dtype=torch.float32)),
        "bulk_only_after_mask": _to_numpy(bulk_only_after_mask),
        "bulk_only_train_composed": _to_numpy(train_bulk_only),
        "bulk_only_abs_diff": _to_numpy(torch.abs(train_bulk_only - bulk_only_after_mask)),
    }

    stem = f"{checkpoint_path.stem}_{axis_name}{chosen_slice_idx:03d}"
    report_path = output_root / f"preview_alignment_{stem}.txt"
    npz_path = output_root / f"preview_alignment_{stem}.npz"
    png_path = output_root / f"preview_alignment_{stem}.png"
    write_key_value_report(report_path, report_rows)
    np.savez_compressed(npz_path, **arrays)
    _save_debug_png(png_path, arrays)

    return {
        "report_path": str(report_path),
        "npz_path": str(npz_path),
        "png_path": str(png_path),
        "slice_idx": chosen_slice_idx,
        "axis": axis_name,
    }


def main() -> int:
    cli_args = _parse_args()
    result = debug_compare_training_vs_preview_readout(
        checkpoint=cli_args.checkpoint,
        axis=cli_args.axis,
        slice_idx=cli_args.slice_idx,
        command_file=cli_args.command_file,
        output_dir=cli_args.output_dir,
    )
    print(f"report: {result['report_path']}")
    print(f"npz: {result['npz_path']}")
    print(f"png: {result['png_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
