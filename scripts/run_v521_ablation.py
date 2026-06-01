"""CTGS v5.2.1 ablation runner for V0-V5.

V0 is the rollback baseline.
V1-V5 isolate A/B/C combinations described in the v5.2.1 plan.

Examples:
    python -m scripts.run_v521_ablation --variant V1 --iterations 1000
    python -m scripts.run_v521_ablation --variant all --iterations 1000
"""
from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path


BASE_CMD = [
    sys.executable,
    "train_ct.py",
    "--ct_phase1_dir",
    "outputs/bunny_smoke/phase1_surface_complete_grid3_margin1_20260522",
    "--ct_volume_path",
    "assets/bunny",
    "--ct_volume_format",
    "auto",
    "--skip_export_mesh",
    "--skip_export_sdf",
    "--quiet",
]


def _rollback_flags() -> list[str]:
    return [
        "--ct_loss_band_in_weight", "0.25",
        "--ct_reseed_sigma_init_factor", "0.5",
        "--ct_reseed_sigma_init_floor_ratio", "0.05",
        "--ct_reseed_atten_init_boost", "1.0",
        "--ct_containment_ramp_weight_1", "0.25",
        "--ct_containment_ramp_iter_2", "3000",
        "--ct_containment_ramp_iter_3", "6000",
    ]


VARIANTS: dict[str, dict[str, object]] = {
    "V0": {
        "label": "rollback baseline",
        "flags": _rollback_flags(),
    },
    "V1": {
        "label": "A only: band in weight",
        "flags": _rollback_flags() + ["--ct_loss_band_in_weight", "1.0"],
    },
    "V2": {
        "label": "B only: reseed init upgrade",
        "flags": _rollback_flags() + [
            "--ct_reseed_sigma_init_factor", "1.0",
            "--ct_reseed_sigma_init_floor_ratio", "0.5",
            "--ct_reseed_atten_init_boost", "2.0",
        ],
    },
    "V3": {
        "label": "C only: deferred containment ramp",
        "flags": _rollback_flags() + [
            "--ct_containment_ramp_weight_1", "0.05",
            "--ct_containment_ramp_iter_2", "5000",
            "--ct_containment_ramp_iter_3", "8000",
        ],
    },
    "V4": {
        "label": "AB: asymmetric band + reseed init upgrade",
        "flags": _rollback_flags() + [
            "--ct_loss_band_in_weight", "1.0",
            "--ct_reseed_sigma_init_factor", "1.0",
            "--ct_reseed_sigma_init_floor_ratio", "0.5",
            "--ct_reseed_atten_init_boost", "2.0",
        ],
    },
    "V5": {
        "label": "ABC: full v5.2.1 package",
        "flags": [
            "--ct_loss_band_in_weight", "1.0",
            "--ct_reseed_sigma_init_factor", "1.0",
            "--ct_reseed_sigma_init_floor_ratio", "0.5",
            "--ct_reseed_atten_init_boost", "2.0",
            "--ct_containment_ramp_weight_1", "0.05",
            "--ct_containment_ramp_iter_2", "5000",
            "--ct_containment_ramp_iter_3", "8000",
        ],
    },
}


SUMMARY_KEYS = [
    "boundary_miss_rate",
    "tau_current",
    "bulk_material_occupancy_p10",
    "material_boundary_shell_occ_b_raw_p10",
    "combined_material_occ_union_raw_p10",
    "void_air_occ_b_raw_p95",
    "cavity_void_false_fill_ratio",
    "bulk_reseed_added",
    "bulk_reseed_sigma_init_mean",
    "bulk_reseed_atten_init_mean",
    "bulk_surface_gap_distance_p50",
    "bulk_surface_gap_distance_p90",
]


def _build_command(variant: str, iterations: int, save_iters: list[int]) -> tuple[list[str], str]:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"outputs/bunny_smoke/ctgs_v521_{variant}_{timestamp}"
    iter_flags = [
        "--iterations", str(iterations),
        "--save_iterations", *[str(it) for it in save_iters],
        "--checkpoint_iterations", *[str(it) for it in save_iters],
        "--model_path", model_path,
    ]
    return BASE_CMD + iter_flags + list(VARIANTS[variant]["flags"]), model_path


def _run(cmd: list[str], cwd: Path, label: str) -> int:
    print(f"\n=== {label} ===")
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    return int(subprocess.run(cmd, cwd=str(cwd)).returncode)


def _read_kv_report(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        data[key.strip()] = value.strip()
    return data


def _write_summary(output_root: Path, rows: list[dict[str, str]]) -> None:
    lines = ["variant\tlabel\tmodel_path\t" + "\t".join(SUMMARY_KEYS)]
    for row in rows:
        fields = [row.get("variant", ""), row.get("label", ""), row.get("model_path", "")]
        fields.extend(row.get(key, "") for key in SUMMARY_KEYS)
        lines.append("\t".join(fields))
    (output_root / "variant_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=str, default="V5", help="Variant ID (V0..V5) or 'all' to run V1..V5.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save_iterations", type=int, nargs="+", default=None, help="Defaults to [iterations].")
    args = parser.parse_args(argv)

    cwd = Path(__file__).resolve().parents[1]
    save_iters = args.save_iterations or [args.iterations]
    if args.variant.lower() == "all":
        variant_ids = ["V1", "V2", "V3", "V4", "V5"]
    else:
        variant_ids = [args.variant]

    failures: list[str] = []
    summary_rows: list[dict[str, str]] = []
    for vid in variant_ids:
        if vid not in VARIANTS:
            print(f"unknown variant: {vid}")
            return 2
        cmd, model_path = _build_command(vid, args.iterations, save_iters)
        rc = _run(cmd, cwd, label=f"{vid}: {VARIANTS[vid]['label']} -> {model_path}")
        if rc != 0:
            failures.append(f"{vid} (rc={rc})")
            continue
        drift_path = cwd / model_path / "diagnostics" / f"drift_iter_{int(save_iters[-1]):06d}.txt"
        drift = _read_kv_report(drift_path)
        row = {"variant": vid, "label": str(VARIANTS[vid]["label"]), "model_path": model_path}
        for key in SUMMARY_KEYS:
            row[key] = drift.get(key, "")
        summary_rows.append(row)

    if summary_rows:
        output_root = cwd / "outputs" / "bunny_smoke"
        _write_summary(output_root, summary_rows)
        print(f"\nWrote summary: {output_root / 'variant_summary.tsv'}")
    if failures:
        print("\nFAILURES: " + ", ".join(failures))
        return 1
    print("\nAll variants completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
