"""Ablation over the four CTGS-vFinal "targeted mechanisms" (对症机制).

Mechanisms toggled:
  M1 atten_only  : --ct_train_bulk_atten_only        (freeze bulk geometry, train a_i only)
  M2 interior    : --ct_intensity_sample_mode         (material_interior_only vs full_band)
  M3 den_min     : --ct_intensity_den_min             (coverage gate on intensity loss)
  M4 adaptive_s  : --ct_bulk_adaptive_mode             (scale vs fixed -> residual-driven sigma)

Runs a cumulative ladder (A0..A4) plus an adaptive-only isolation (A5),
collects per-iteration preview metrics (raw A_b, masked hard, sdf-soft),
and writes a single summary TSV.

Usage:
  python -m scripts.run_targeted_mechanism_ablation --iterations 2000
  python -m scripts.run_targeted_mechanism_ablation --iterations 500 --variants A0 A3 A4
"""
from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path

PHASE1_DIR = "outputs/bunny_smoke/phase1_surface_complete_grid3_margin1_20260522"
VOLUME_PATH = "assets/bunny"

BASE_CMD = [
    sys.executable,
    "train_ct.py",
    "--ct_phase1_dir", PHASE1_DIR,
    "--ct_volume_path", VOLUME_PATH,
    "--ct_volume_format", "auto",
    "--skip_export_mesh",
    "--skip_export_sdf",
    "--quiet",
]

# Historical coverage variants now only keep the sigma floor lever; query-time
# compact support was removed from the training CLI.
COVERAGE_FIX = ["--ct_bulk_sigma_min_mm", "0.7"]

# Each variant = override flags relative to code defaults.
# Defaults: atten_only=off, full_band, den_min=0, adaptive=fixed.
VARIANTS: dict[str, dict] = {
    "B0_speckled": {
        "label": "current default (no sigma floor)",
        "flags": [],
    },
    "B1_coverage_fix": {
        "label": "coverage fix: sigma floor 0.7mm",
        "flags": list(COVERAGE_FIX),
    },
    "B2_freeze": {
        "label": "+freeze geometry (atten_only)",
        "flags": COVERAGE_FIX + ["--ct_train_bulk_atten_only"],
    },
    "B3_interior_denmin": {
        "label": "+interior_only sampling + den_min gate",
        "flags": COVERAGE_FIX + ["--ct_train_bulk_atten_only",
                                 "--ct_intensity_sample_mode", "material_interior_only",
                                 "--ct_intensity_den_min", "0.3"],
    },
    "B4_adaptive": {
        "label": "+adaptive sigma (root-cause over-smoothing fix)",
        "flags": COVERAGE_FIX + ["--ct_train_bulk_atten_only",
                                 "--ct_intensity_sample_mode", "material_interior_only",
                                 "--ct_intensity_den_min", "0.3",
                                 "--ct_bulk_adaptive_mode", "scale"],
    },
}

LADDER = ["B0_speckled", "B1_coverage_fix", "B2_freeze", "B3_interior_denmin", "B4_adaptive"]

# metrics we extract from preview txts, per save iteration
MAIN_KEYS = ["preview_mode", "ssim", "mae", "psnr"]            # raw A_b primary
DUAL_KEYS = ["hybrid_hard_ssim", "hybrid_hard_mae",
             "hybrid_soft_ssim", "hybrid_soft_mae",
             "bulk_only_ssim", "bulk_only_mae"]


def _read_kv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " not in line:
            continue
        k, v = line.split(" = ", 1)
        data[k.strip()] = v.strip()
    return data


def _collect(model_path: Path, save_iters: list[int]) -> list[dict[str, str]]:
    rows = []
    prev = model_path / "previews"
    for it in save_iters:
        main = _read_kv(prev / f"preview_iter_{it:06d}.txt")
        dual = _read_kv(prev / f"preview_iter_{it:06d}_dual.txt")
        if not main and not dual:
            continue
        row = {"iter": str(it)}
        for k in MAIN_KEYS:
            row[k] = main.get(k, "")
        for k in DUAL_KEYS:
            row[k] = dual.get(k, "")
        rows.append(row)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--save_iterations", type=int, nargs="+", default=None)
    p.add_argument("--variants", type=str, nargs="+", default=None,
                   help="Subset of variant IDs; default runs the full ladder.")
    args = p.parse_args(argv)

    cwd = Path(__file__).resolve().parents[1]
    save_iters = args.save_iterations or [100, 300, 500, args.iterations]
    save_iters = sorted(set(int(x) for x in save_iters if int(x) <= args.iterations))
    variant_ids = args.variants or LADDER

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = cwd / "outputs" / "bunny_smoke" / f"mechanism_ablation_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_lines = ["variant\tlabel\titer\t" + "\t".join(MAIN_KEYS + DUAL_KEYS)]
    failures = []
    for vid in variant_ids:
        if vid not in VARIANTS:
            print(f"unknown variant {vid}")
            return 2
        model_path = out_root / vid
        cmd = BASE_CMD + [
            "--iterations", str(args.iterations),
            "--save_iterations", *[str(x) for x in save_iters],
            "--model_path", str(model_path),
        ] + list(VARIANTS[vid]["flags"])
        print(f"\n=== {vid}: {VARIANTS[vid]['label']} ===")
        print("$ " + " ".join(shlex.quote(c) for c in cmd))
        rc = subprocess.run(cmd, cwd=str(cwd)).returncode
        if rc != 0:
            failures.append(f"{vid}(rc={rc})")
            continue
        for row in _collect(model_path, save_iters):
            fields = [vid, str(VARIANTS[vid]["label"]), row["iter"]]
            fields += [row.get(k, "") for k in MAIN_KEYS + DUAL_KEYS]
            summary_lines.append("\t".join(fields))

    summary_path = out_root / "summary.tsv"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print("\n".join(summary_lines))
    if failures:
        print("\nFAILURES: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
