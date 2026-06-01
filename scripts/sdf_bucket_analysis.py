"""SDF-bucketed error analysis and preview mode comparison.

Usage:
    python scripts/sdf_bucket_analysis.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_volume(volume_path: str, volume_format: str):
    from ct_pipeline.data import CTVolumeLoader
    loader = CTVolumeLoader()
    volume = loader.load(volume_path, fmt=volume_format)
    spacing_zyx = loader.get_voxel_spacing()
    return volume, spacing_zyx


def _load_phase1_sdf(analysis_npz_path: str, spacing_zyx=None) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    with np.load(analysis_npz_path) as data:
        if "material_signed_distance" in data:
            return data["material_signed_distance"].astype(np.float32)
        if "signed_distance" in data:
            return data["signed_distance"].astype(np.float32)
        # Compute from material_mask
        mask_key = "material_mask" if "material_mask" in data else "coarse_support_mask"
        if mask_key not in data:
            raise KeyError("No SDF or mask found in analysis.npz")
        mask = data[mask_key].astype(bool)
    print(f"  Computing SDF from {mask_key} (shape={mask.shape})...")
    sampling = tuple(float(v) for v in spacing_zyx) if spacing_zyx else None
    dist_in = distance_transform_edt(mask, sampling=sampling).astype(np.float32)
    dist_out = distance_transform_edt(~mask, sampling=sampling).astype(np.float32)
    sdf = np.where(mask, -dist_in, dist_out).astype(np.float32)
    return sdf


def _render_slice_from_ply(ply_path: str, axis: str, t: float, size: int = 512) -> np.ndarray:
    from scene.ct_gaussian_model import CTGaussianModel
    from ct_pipeline.viewer.session import ViewerSession, load_viewer_session
    session = load_viewer_session(Path(ply_path))
    return session.render_slice(axis=axis, t=t, layer="all", size=size)


def sdf_bucket_analysis(
    volume_path: str,
    volume_format: str,
    analysis_npz: str,
    ply_path: str,
    slice_z: int | None = None,
    output_dir: str | None = None,
):
    print("Loading GT volume...")
    volume, spacing_zyx = _load_volume(volume_path, volume_format)
    volume_norm = volume.astype(np.float32)
    if volume_norm.max() > 1.0:
        volume_norm = (volume_norm - volume_norm.min()) / max(volume_norm.max() - volume_norm.min(), 1e-8)

    print("Loading Phase1 SDF...")
    sdf = _load_phase1_sdf(analysis_npz, spacing_zyx=spacing_zyx)

    D, H, W = volume_norm.shape
    if slice_z is None:
        slice_z = D // 2
    print(f"Using slice z={slice_z} (D={D})")

    gt_slice = volume_norm[slice_z]          # (H, W)
    # Convert SDF from world units to voxel units for threshold comparisons
    min_spacing = float(min(spacing_zyx))
    sdf_voxels = sdf[slice_z] / max(min_spacing, 1e-8)  # now in voxel units
    sdf_slice = sdf_voxels

    print("Rendering GS occupancy slice...")
    t = float(slice_z) / max(D - 1, 1)
    pred_occ = _render_slice_from_ply(ply_path, axis="z", t=t, size=max(H, W))
    if pred_occ.shape != gt_slice.shape:
        from PIL import Image
        pred_img = Image.fromarray((pred_occ * 255).clip(0, 255).astype(np.uint8))
        pred_img = pred_img.resize((W, H), Image.BILINEAR)
        pred_occ = np.array(pred_img).astype(np.float32) / 255.0

    # --- SDF bucket statistics ---
    # sdf < 0 → inside material, sdf > 0 → outside
    buckets = [
        ("deep_inside  sdf<-3", sdf_slice < -3),
        ("inner_shell  -3<=sdf<-1.5", (sdf_slice >= -3) & (sdf_slice < -1.5)),
        ("boundary_in  -1.5<=sdf<-0.5", (sdf_slice >= -1.5) & (sdf_slice < -0.5)),
        ("boundary_band |sdf|<=0.5", np.abs(sdf_slice) <= 0.5),
        ("boundary_out  0.5<sdf<=1.5", (sdf_slice > 0.5) & (sdf_slice <= 1.5)),
        ("outer_shell  1.5<sdf<=3", (sdf_slice > 1.5) & (sdf_slice <= 3)),
        ("exterior     sdf>3", sdf_slice > 3),
    ]

    # Treat material_mask occupancy target: inside (sdf<=0) → target=1, outside → target=0
    occ_target = (sdf_slice <= 0).astype(np.float32)
    error = pred_occ - occ_target          # signed: positive = over-predicted
    abs_error = np.abs(error)

    print("\n=== SDF-bucketed Occupancy Error ===")
    print(f"{'bucket':<35} {'N':>7} {'MAE':>8} {'bias':>8} {'top10%':>8} {'pred_mean':>10} {'gt_mean':>8}")
    print("-" * 90)
    rows = []
    for label, mask in buckets:
        n = int(mask.sum())
        if n == 0:
            continue
        mae = float(abs_error[mask].mean())
        bias = float(error[mask].mean())
        top10 = float(np.percentile(abs_error[mask], 90))
        pred_mean = float(pred_occ[mask].mean())
        gt_mean = float(occ_target[mask].mean())
        print(f"{label:<35} {n:>7} {mae:>8.4f} {bias:>8.4f} {top10:>8.4f} {pred_mean:>10.4f} {gt_mean:>8.4f}")
        rows.append(dict(label=label, n=n, mae=mae, bias=bias, top10=top10, pred_mean=pred_mean, gt_mean=gt_mean))

    # Peak-error bucket identification
    mae_vals = [r["mae"] for r in rows]
    peak_idx = int(np.argmax(mae_vals))
    print(f"\n>>> Peak MAE bucket: '{rows[peak_idx]['label']}' (MAE={rows[peak_idx]['mae']:.4f})")
    shell_mae = sum(r["mae"] for r in rows if "boundary" in r["label"] or "shell" in r["label"] if r["n"] > 0)
    shell_n = sum(1 for r in rows if "boundary" in r["label"] or "shell" in r["label"] if r["n"] > 0)
    print(f">>> Mean MAE over shell+boundary buckets: {shell_mae/max(shell_n,1):.4f}")
    deep_rows = [r for r in rows if "deep_inside" in r["label"]]
    if deep_rows:
        print(f">>> Deep interior MAE: {deep_rows[0]['mae']:.4f}  bias: {deep_rows[0]['bias']:.4f}")

    if output_dir:
        import json
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sdf_bucket_results.json").write_text(json.dumps(rows, indent=2))
        _save_visualization(gt_slice, pred_occ, sdf_slice, error, out, slice_z)
        print(f"\nResults saved to {out}")

    return rows


def _save_visualization(gt, pred, sdf, error, out_dir: Path, slice_z: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title(f"GT occupancy target z={slice_z}")
        axes[0, 1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title("GS raw_occupancy prediction")
        axes[0, 2].imshow(np.abs(error), cmap="hot", vmin=0, vmax=0.5)
        axes[0, 2].set_title("Abs error")
        axes[1, 0].imshow(sdf, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5))
        axes[1, 0].set_title("Phase1 SDF (red=inside, blue=outside)")
        # signed error coloured by SDF zone
        sdf_clipped = np.clip(sdf, -5, 5)
        axes[1, 1].imshow(error, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=0.5))
        axes[1, 1].set_title("Signed error (red=over, blue=under)")
        # MAE masked to |sdf| <= 2
        shell_mask = np.abs(sdf) <= 2
        shell_err = np.where(shell_mask, np.abs(error), 0)
        axes[1, 2].imshow(shell_err, cmap="hot", vmin=0, vmax=0.5)
        axes[1, 2].set_title("|sdf|<=2 shell abs error")
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "sdf_bucket_visualization.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[viz] skipped: {exc}")


def preview_mode_comparison(
    volume_path: str,
    volume_format: str,
    checkpoint_path: str,
    phase1_dir: str,
    slice_z: int | None = None,
    output_dir: str | None = None,
):
    """Generate both raw_occupancy and bounded_bulk_intensity previews from same checkpoint."""
    import torch
    print("Generating preview mode comparison...")
    env = dict(
        CT_DISABLE_TENSORBOARD="1",
    )
    import os, subprocess

    out = Path(output_dir) if output_dir else Path("outputs/preview_comparison")
    out.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable, "train_ct.py",
        "--ct_phase1_dir", phase1_dir,
        "--ct_volume_path", volume_path,
        "--ct_volume_format", volume_format,
        "--start_checkpoint", checkpoint_path,
        "--iterations", "0",
        "--save_iterations", "0",
        "--checkpoint_iterations", "0",
        "--model_path", str(out),
        "--skip_export_mesh",
        "--skip_export_sdf",
    ]
    run_env = {**os.environ, **env}
    print("Generating preview at iteration 0 from checkpoint...")
    result = subprocess.run(base_cmd, cwd=str(REPO_ROOT), env=run_env, capture_output=True, text=True)
    if result.returncode != 0:
        print("[WARN] train_ct exited non-zero:")
        print(result.stderr[-2000:])
    print("Done. Check", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", default=str(REPO_ROOT / "outputs/bunny_smoke/phase1_surface_complete_grid3_margin1_20260522"))
    parser.add_argument("--volume-path", default=str(REPO_ROOT / "assets/bunny"))
    parser.add_argument("--volume-format", default="dicom")
    parser.add_argument("--ply", default=str(REPO_ROOT / "outputs/bulk_diagnostic_sweep_20260529_020703/iter1000/variant_base/point_cloud/iteration_1000/point_cloud.ply"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs/bulk_diagnostic_sweep_20260529_020703/iter1000/variant_base/chkpnt1000.pth"))
    parser.add_argument("--slice-z", type=int, default=101)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/sdf_bucket_analysis"))
    parser.add_argument("--skip-preview-compare", action="store_true")
    args = parser.parse_args()

    analysis_npz = str(Path(args.phase1_dir) / "analysis.npz")

    print("=" * 60)
    print("EXPERIMENT 1: SDF-bucketed error analysis")
    print("=" * 60)
    sdf_bucket_analysis(
        volume_path=args.volume_path,
        volume_format=args.volume_format,
        analysis_npz=analysis_npz,
        ply_path=args.ply,
        slice_z=args.slice_z,
        output_dir=args.output_dir,
    )

    if not args.skip_preview_compare:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Preview mode comparison")
        print("=" * 60)
        preview_mode_comparison(
            volume_path=args.volume_path,
            volume_format=args.volume_format,
            checkpoint_path=args.checkpoint,
            phase1_dir=args.phase1_dir,
            slice_z=args.slice_z,
            output_dir=str(Path(args.output_dir) / "preview_compare"),
        )


if __name__ == "__main__":
    main()
