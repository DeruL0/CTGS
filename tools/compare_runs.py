"""Scan outputs/ for all training runs and tabulate their final preview metrics.

Each training run has a previews/ subfolder with preview_iter_*.txt files
containing iteration / mae / rmse / psnr / ssim / lpips.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _parse_preview(path: Path) -> dict | None:
    metrics: dict = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue
    except Exception:
        return None
    return metrics if metrics else None


def _final_preview(run_dir: Path) -> tuple[int, dict] | None:
    previews_dir = run_dir / "previews"
    if not previews_dir.is_dir():
        return None
    best_iter = -1
    best_metrics = None
    for preview_path in previews_dir.glob("preview_iter_*.txt"):
        match = re.search(r"preview_iter_(\d+)\.txt", preview_path.name)
        if match is None:
            continue
        iter_value = int(match.group(1))
        if iter_value <= best_iter:
            continue
        metrics = _parse_preview(preview_path)
        if metrics is None:
            continue
        best_iter = iter_value
        best_metrics = metrics
    if best_metrics is None or best_iter < 0:
        return None
    return best_iter, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--filter", default=None, help="Substring filter on run path")
    parser.add_argument("--sort_by", default="psnr", choices=["psnr", "ssim", "lpips", "rmse", "mae", "iter", "name"])
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for run_dir in sorted(root.rglob("previews")):
        if not run_dir.is_dir():
            continue
        if args.filter and args.filter not in str(run_dir):
            continue
        parent = run_dir.parent
        result = _final_preview(parent)
        if result is None:
            continue
        iter_value, metrics = result
        rows.append({
            "name": str(parent.relative_to(root)),
            "iter": iter_value,
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "lpips": metrics.get("lpips"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
        })

    def sort_key(row):
        value = row.get(args.sort_by)
        if value is None:
            return (1, 0)
        if args.sort_by in ("psnr", "ssim"):
            return (0, -value)
        if args.sort_by == "name":
            return (0, str(row["name"]))
        return (0, value)

    rows.sort(key=sort_key)

    header = f"{'run':70s} {'iter':>6s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'RMSE':>8s} {'MAE':>8s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        name = row["name"][:70]
        iter_str = f"{row['iter']}"
        psnr_str = f"{row['psnr']:7.3f}" if row.get("psnr") is not None else "    -  "
        ssim_str = f"{row['ssim']:7.4f}" if row.get("ssim") is not None else "    -  "
        lpips_str = f"{row['lpips']:7.4f}" if row.get("lpips") is not None else "    -  "
        rmse_str = f"{row['rmse']:8.5f}" if row.get("rmse") is not None else "    -   "
        mae_str = f"{row['mae']:8.5f}" if row.get("mae") is not None else "    -   "
        print(f"{name:70s} {iter_str:>6s} {psnr_str} {ssim_str} {lpips_str} {rmse_str} {mae_str}")

    print()
    print(f"Total runs scanned: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
