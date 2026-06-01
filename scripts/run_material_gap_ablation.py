"""Run a focused CTGS ablation for material-side intensity and thin-gap fill."""
from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = "outputs/bunny_smoke/phase1_surface_complete_grid3_margin1_20260522"
VOLUME_PATH = "assets/bunny"


BASE_FLAGS = [
    "--ct_phase1_dir", PHASE1_DIR,
    "--ct_volume_path", VOLUME_PATH,
    "--ct_volume_format", "auto",
    "--skip_export_mesh",
    "--skip_export_sdf",
    "--quiet",
    "--ct_bulk_init_mode", "contained_lattice",
    "--ct_train_bulk_atten_only",
    "--ct_freeze_surface",
    "--ct_freeze_bulk_geometry",
    "--no-ct_enable_densification",
    "--no-ct_enable_surface_reseeding",
    "--no-ct_enable_bulk_reseeding",
    "--no-ct_atten_only_early_stop",
]


VARIANTS = {
    "dense_2p5": {
        "label": "denser lattice for thin material gaps",
        "flags": [
            "--ct_bulk_lattice_spacing_vox", "2.5",
            "--ct_bulk_lattice_sigma_vox", "1.6",
            "--ct_bulk_lattice_margin_vox", "0.02",
            "--ct_bulk_prune_warmup", "100",
            "--ct_bulk_prune_interval", "100",
        ],
    },
    "dense_2p0": {
        "label": "aggressive lattice for narrowest gaps",
        "flags": [
            "--ct_bulk_lattice_spacing_vox", "2.0",
            "--ct_bulk_lattice_sigma_vox", "1.3",
            "--ct_bulk_lattice_margin_vox", "0.0",
            "--ct_bulk_prune_warmup", "100",
            "--ct_bulk_prune_interval", "100",
        ],
    },
    "dense_2p5_no_prune": {
        "label": "test whether pruning removes thin-gap candidates",
        "flags": [
            "--ct_bulk_lattice_spacing_vox", "2.5",
            "--ct_bulk_lattice_sigma_vox", "1.6",
            "--ct_bulk_lattice_margin_vox", "0.02",
            "--ct_bulk_prune_interval", "0",
        ],
    },
    "dense_2p5_material_only": {
        "label": "material-only intensity fitting on dense lattice",
        "flags": [
            "--ct_bulk_lattice_spacing_vox", "2.5",
            "--ct_bulk_lattice_sigma_vox", "1.6",
            "--ct_bulk_lattice_margin_vox", "0.02",
            "--ct_intensity_sample_mode", "material_interior_only",
            "--ct_intensity_den_min", "0.3",
            "--ct_bulk_prune_warmup", "100",
            "--ct_bulk_prune_interval", "100",
        ],
    },
}


SUMMARY_KEYS = [
    "total_count",
    "bulk_count",
    "bulk_pruned_count",
    "bulk_interior_coverage_gap_ratio",
    "bulk_material_coverage_gap_ratio",
    "bulk_material_A_b_p10",
    "bulk_material_A_b_p50",
    "material_boundary_shell_occ_b_raw_p10",
    "cavity_material_shell_occ_b_raw_p10",
    "false_hole_active_ratio",
    "false_hole_bulk_occ_p10",
    "cavity_void_false_fill_ratio",
]

INTENSITY_KEYS = [
    "avg_material_mae",
    "avg_material_ssim",
    "affine_avg_material_mae",
    "affine_avg_material_ssim",
    "material_corr_avg_attn_vs_gt",
    "avg_attn_material_p10",
    "avg_attn_material_p50",
    "raw_i_b_material_p10",
    "raw_i_b_material_p50",
]


def read_kv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        data[key.strip()] = value.strip()
    return data


def run_checked(cmd: list[str], label: str) -> int:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else REPO_ROOT / "outputs" / "bunny_smoke" / f"material_gap_ablation_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    failures: list[str] = []

    for variant_id in args.variants:
        if variant_id not in VARIANTS:
            raise SystemExit(f"unknown variant: {variant_id}")
        variant = VARIANTS[variant_id]
        model_dir = output_root / variant_id
        train_cmd = [
            sys.executable,
            "train_ct.py",
            *BASE_FLAGS,
            "--iterations", str(args.iterations),
            "--save_iterations", str(args.iterations),
            "--checkpoint_iterations", str(args.iterations),
            "--model_path", str(model_dir),
            *variant["flags"],
        ]
        rc = run_checked(train_cmd, f"{variant_id}: {variant['label']}")
        if rc != 0:
            failures.append(f"{variant_id}: train rc={rc}")
            rows.append({"variant": variant_id, "label": variant["label"], "status": f"train_failed_{rc}", "model_path": str(model_dir)})
            continue

        ckpt = model_dir / f"chkpnt{args.iterations}.pth"
        diag_dir = model_dir / "material_intensity_diag"
        diag_cmd = [
            sys.executable,
            "scripts/diagnose_ct_intensity_preview.py",
            "--command-file", str(model_dir / "command.txt"),
            "--checkpoint", str(ckpt),
            "--output-dir", str(diag_dir),
        ]
        rc = run_checked(diag_cmd, f"{variant_id}: material intensity diagnostic")
        if rc != 0:
            failures.append(f"{variant_id}: diag rc={rc}")

        drift = read_kv(model_dir / "diagnostics" / f"drift_iter_{args.iterations:06d}.txt")
        intensity = read_kv(diag_dir / "intensity_diag_slice_z101.txt")
        row = {
            "variant": variant_id,
            "label": variant["label"],
            "status": "ok" if rc == 0 else f"diag_failed_{rc}",
            "model_path": str(model_dir),
        }
        for key in SUMMARY_KEYS:
            row[key] = drift.get(key, "")
        for key in INTENSITY_KEYS:
            row[key] = intensity.get(key, "")
        rows.append(row)
        write_summary(output_root, rows)

    write_summary(output_root, rows)
    if failures:
        print("\nFAILURES: " + "; ".join(failures), flush=True)
        return 1
    print(f"\nWrote summary: {output_root / 'material_gap_summary.tsv'}", flush=True)
    return 0


def write_summary(output_root: Path, rows: list[dict[str, str]]) -> None:
    columns = ["variant", "label", "status", "model_path", *SUMMARY_KEYS, *INTENSITY_KEYS]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in columns))
    (output_root / "material_gap_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
