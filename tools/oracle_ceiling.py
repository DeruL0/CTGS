"""Estimate ceilings (PSNR/SSIM/LPIPS/MAE/RMSE) of the current CT representation
parameterization.

For each ceiling level we compute metrics over both the full 3D volume (PSNR only)
and the middle z-slice (all metrics — matches the training preview metric).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ct_pipeline.data import CTVolumeLoader


def _mse(pred: np.ndarray, gt: np.ndarray) -> float:
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    return float(np.mean(diff * diff))


def _mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float64) - gt.astype(np.float64))))


def _psnr(pred: np.ndarray, gt: np.ndarray, peak: float = 1.0) -> float:
    mse = _mse(pred, gt)
    if mse <= 0.0:
        return float("inf")
    return 10.0 * float(np.log10((peak * peak) / mse))


def _ssim_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim_fn
    return float(ssim_fn(gt, pred, data_range=1.0))


def _lpips_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    import torch
    import lpips
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    def _prep(image):
        tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        return tensor.expand(-1, 3, -1, -1)
    with torch.no_grad():
        value = model(_prep(pred), _prep(gt))
    return float(value.item())


def _calibrate(volume: np.ndarray, support: np.ndarray, air: np.ndarray) -> tuple[float, float]:
    finite_volume = volume[np.isfinite(volume)]
    material_values = volume[np.logical_and(support, np.isfinite(volume))]
    air_values = volume[np.logical_and(air, np.isfinite(volume))]
    i_air = float(np.quantile(air_values, 0.5)) if air_values.size > 0 else float(np.quantile(finite_volume, 0.05))
    i_mat = float(np.quantile(material_values, 0.5)) if material_values.size > 0 else float(np.quantile(finite_volume, 0.95))
    return i_air, i_mat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_dir", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--volume_format", default="dicom")
    args = parser.parse_args()

    phase1_dir = Path(args.phase1_dir)
    analysis = dict(np.load(str(phase1_dir / "analysis.npz")))
    with (phase1_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    loader = CTVolumeLoader()
    volume_np = loader.load(args.volume_path, fmt=args.volume_format)
    spacing_zyx = loader.get_voxel_spacing()
    volume_np = np.asarray(volume_np, dtype=np.float32)
    print(f"Volume shape: {volume_np.shape}, range [{float(volume_np.min()):.4f}, {float(volume_np.max()):.4f}]")

    support = np.asarray(analysis.get("material_mask"), dtype=bool)
    void = np.asarray(analysis.get("void_mask"), dtype=bool) if "void_mask" in analysis else np.zeros_like(support)
    air = np.logical_or(~support, void)

    i_air, i_mat = _calibrate(volume_np, support, air)
    print(f"Fixed calibration: I_air={i_air:.6f}, I_mat={i_mat:.6f}")

    # Build oracles
    oracle_fixed = (i_air + (i_mat - i_air) * support.astype(np.float32)).astype(np.float32)

    flat_mask = support.reshape(-1).astype(np.float64)
    flat_gt = volume_np.reshape(-1).astype(np.float64)
    A = np.stack([np.ones_like(flat_mask), flat_mask], axis=1)
    coef, _residuals, _rank, _sv = np.linalg.lstsq(A, flat_gt, rcond=None)
    a_ols, b_ols = float(coef[0]), float(coef[1])
    oracle_ols = (a_ols + b_ols * support.astype(np.float32)).astype(np.float32)
    print(f"OLS calibration:   I_air={a_ols:.6f}, I_mat={a_ols + b_ols:.6f}")

    mean_support = float(volume_np[support].mean()) if support.any() else 0.0
    mean_air = float(volume_np[~support].mean()) if (~support).any() else 0.0
    oracle_two_bucket = np.where(support, mean_support, mean_air).astype(np.float32)

    oracle_perfect_inside = np.where(support, volume_np, mean_air).astype(np.float32)

    # Full volume PSNR
    print()
    print("=" * 74)
    print("Full volume (peak=1.0)")
    print("=" * 74)
    print(f"{'ceiling':52s} {'PSNR':>8s} {'MAE':>8s} {'RMSE':>8s}")
    for name, pred in [
        ("C1 fixed quantile (current pipeline)", oracle_fixed),
        ("C2 OLS-optimal affine", oracle_ols),
        ("C3 two-bucket mean", oracle_two_bucket),
        ("C4 per-voxel inside, mean air outside", oracle_perfect_inside),
    ]:
        psnr = _psnr(pred, volume_np)
        mae = _mae(pred, volume_np)
        rmse = float(np.sqrt(_mse(pred, volume_np)))
        print(f"  {name:50s} {psnr:8.3f} {mae:8.5f} {rmse:8.5f}")

    # Middle-slice all metrics
    slice_idx = volume_np.shape[0] // 2
    gt_slice = volume_np[slice_idx]
    print()
    print("=" * 74)
    print(f"Middle z-slice (idx={slice_idx}, peak=1.0) — matches preview metric")
    print("=" * 74)
    print(f"{'ceiling':52s} {'PSNR':>8s} {'SSIM':>7s} {'LPIPS':>7s} {'MAE':>8s} {'RMSE':>8s}")

    candidates = [
        ("C1 fixed quantile (current pipeline)", oracle_fixed[slice_idx]),
        ("C2 OLS-optimal affine", oracle_ols[slice_idx]),
        ("C3 two-bucket mean", oracle_two_bucket[slice_idx]),
        ("C4 per-voxel inside, mean air outside", oracle_perfect_inside[slice_idx]),
    ]
    # Pre-clip to [0, 1] so SSIM/LPIPS see valid range
    for name, pred in candidates:
        pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
        psnr = _psnr(pred, gt_slice)
        ssim = _ssim_2d(pred, gt_slice)
        lpips_val = _lpips_2d(pred, gt_slice)
        mae = _mae(pred, gt_slice)
        rmse = float(np.sqrt(_mse(pred, gt_slice)))
        print(f"  {name:50s} {psnr:8.3f} {ssim:7.4f} {lpips_val:7.4f} {mae:8.5f} {rmse:8.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
