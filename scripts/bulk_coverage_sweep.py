"""Controlled sweep for diagnosing incomplete CT bulk coverage.

The sweep is intentionally centered on bulk coverage metrics rather than preview
quality. It writes partial summaries after every variant so a long run can be
inspected while still in progress.

Usage:
    python scripts/bulk_coverage_sweep.py --run-confirmation
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1_DIR = REPO_ROOT / "outputs" / "bunny_smoke" / "phase1_surface_complete_grid3_margin1_20260522"
DEFAULT_VOLUME_PATH = REPO_ROOT / "assets" / "bunny"


VARIANTS: list[dict[str, Any]] = [
    {
        "group": "Control",
        "name": "base",
        "params": {},
        "purpose": "Reference point with bulk xyz frozen by default.",
        "expected": "If freeze is the right default, deep material gap should stay much lower than the old base.",
    },
    {
        "group": "Boundary compose",
        "name": "surface_gate_narrow",
        "params": {"ct_surface_material_gate_sigma": 0.25},
        "purpose": "Narrow material-side surface ownership so surface only dominates very near the boundary.",
        "expected": "If surface owns too much of the material shell, boundary error should drop without hurting deep bulk.",
    },
    {
        "group": "Boundary compose",
        "name": "bulk_dominant_material_shell",
        "params": {"ct_material_compose_mode": "bulk_first_material"},
        "purpose": "Keep total OR occupancy but give material-side intensity ownership to bulk before surface.",
        "expected": "If surface intensity ownership is the issue, shell artifacts/MAE should improve while counts stay fixed.",
    },
    {
        "group": "Boundary compose",
        "name": "surface_tangent_cap_0p8",
        "params": {"ct_surface_max_scale": 0.8},
        "purpose": "Constrain wide tangential surface footprints.",
        "expected": "If thin surfaces spread too far tangentially, boundary error should drop and surface max scale should fall.",
    },
    {
        "group": "Motion",
        "name": "unfreeze_bulk_xyz",
        "params": {"ct_freeze_bulk_xyz": False},
        "purpose": "Single-variable control for bulk geometry drift.",
        "expected": "If gap returns, bulk xyz motion is the main coverage failure.",
    },
    {
        "group": "Init density",
        "name": "bulk_aug_1x",
        "params": {"ct_bulk_augment_factor": 1.0},
        "purpose": "Reduce initial bulk count.",
        "expected": "A larger gap means initial density matters.",
    },
    {
        "group": "Init density",
        "name": "bulk_aug_2x",
        "params": {"ct_bulk_augment_factor": 2.0},
        "purpose": "Explicit current default.",
        "expected": "Should align with base.",
    },
    {
        "group": "Init density",
        "name": "bulk_aug_4x",
        "params": {"ct_bulk_augment_factor": 4.0},
        "purpose": "Increase initial bulk count.",
        "expected": "A lower gap points to insufficient initial sampling.",
    },
    {
        "group": "Init density",
        "name": "bulk_aug_8x",
        "params": {"ct_bulk_augment_factor": 8.0},
        "purpose": "Stress-test density saturation.",
        "expected": "Continued improvement means bulk density is still not saturated.",
    },
    {
        "group": "Init placement",
        "name": "bulk_grid_no_jitter",
        "params": {"ct_bulk_continuous_init": False},
        "purpose": "Separate jitter effects from sparse sampling effects.",
        "expected": "A lower gap means continuous jitter is creating holes.",
    },
    {
        "group": "Growth",
        "name": "bulk_reseed_on",
        "params": {"ct_enable_bulk_reseeding": True},
        "purpose": "Test whether default bulk reseeding fills low-coverage material, including near-boundary material.",
        "expected": "A lower gap plus higher bulk_count means reseeding works.",
    },
    {
        "group": "Growth",
        "name": "bulk_reseed_aggressive",
        "params": {
            "ct_enable_bulk_reseeding": True,
            "ct_bulk_reseed_occupancy_threshold": 0.85,
            "ct_bulk_reseed_max_per_iter": 6000,
            "ct_bulk_reseed_interval": 250,
        },
        "purpose": "Force low-coverage material refill, including the material-side boundary shell.",
        "expected": "No change means candidate filtering or insertion logic is ineffective.",
    },
    {
        "group": "Growth",
        "name": "densify_early",
        "params": {
            "ct_densify_from_iter": 250,
            "ct_densify_interval": 250,
            "ct_densify_bulk_percent": 0.05,
        },
        "purpose": "Test whether bulk splitting can fill holes.",
        "expected": "bulk_count growth without gap reduction means splitting stays local.",
    },
    {
        "group": "Loss",
        "name": "volume_dominant",
        "params": {
            "ct_lambda_volume": 10.0,
            "ct_lambda_occupancy": 0.05,
            "ct_surface_regularizer_weight": 0.05,
            "ct_bulk_coverage_weight": 0.0,
        },
        "purpose": "Check whether volume loss can repair geometry coverage.",
        "expected": "MAE may move, but unchanged gap means volume mostly trains intensity.",
    },
    {
        "group": "Loss",
        "name": "bulk_coverage_dominant",
        "params": {
            "ct_lambda_volume": 0.2,
            "ct_lambda_occupancy": 2.0,
            "ct_bulk_coverage_weight": 3.0,
            "ct_surface_regularizer_weight": 0.2,
        },
        "purpose": "Check whether coverage loss can push gap down.",
        "expected": "No gap change means weak geometry gradients or poor sample candidates.",
    },
]


def parse_kv_txt(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def parse_ply_vertex_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def append_param_args(cmd: list[str], params: dict[str, Any]) -> None:
    for key, value in params.items():
        if isinstance(value, bool):
            cmd.append(f"--{key}" if value else f"--no-{key}")
        else:
            cmd.extend([f"--{key}", str(value)])


def command_text(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def run_parser_smoke(python_exe: str, base_args: list[str], variants: list[dict[str, Any]]) -> None:
    script = r"""
from ct_pipeline.training.config import build_parser, validate_ct_training_args
base_args = __BASE_ARGS__
variants = __VARIANTS__
parser = build_parser()
for variant in variants:
    argv = list(base_args)
    for key, value in variant["params"].items():
        if isinstance(value, bool):
            argv.append(f"--{key}" if value else f"--no-{key}")
        else:
            argv.extend([f"--{key}", str(value)])
    args = parser.parse_args(argv)
    validate_ct_training_args(args)
print(f"parser smoke ok: {len(variants)} variants")
"""
    script = script.replace("__BASE_ARGS__", repr(base_args)).replace("__VARIANTS__", repr(variants))
    subprocess.run([python_exe, "-c", script], cwd=str(REPO_ROOT), check=True)


def run_variant(
    variant: dict[str, Any],
    phase: str,
    iterations: int,
    output_root: Path,
    base_args: list[str],
    python_exe: str,
) -> dict[str, Any]:
    name = variant["name"]
    model_dir = output_root / phase / f"variant_{name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        "train_ct.py",
        *base_args,
        "--iterations", str(iterations),
        "--save_iterations", str(iterations),
        "--checkpoint_iterations", str(iterations),
        "--model_path", str(model_dir),
        "--skip_export_mesh",
        "--skip_export_sdf",
        "--quiet",
    ]
    append_param_args(cmd, variant["params"])
    (model_dir / "command.txt").write_text(command_text(cmd[1:]) + "\n", encoding="utf-8")

    print(f"\n[VARIANT] {phase}/{name}")
    print(command_text(cmd[1:]))
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        return {
            "phase": phase,
            **variant,
            "status": "failed",
            "returncode": exc.returncode,
            "model_dir": str(model_dir),
        }

    preview_txt = model_dir / "previews" / f"preview_iter_{int(iterations):06d}.txt"
    drift_txt = model_dir / "diagnostics" / f"drift_iter_{int(iterations):06d}.txt"
    ply_path = model_dir / "point_cloud" / f"iteration_{int(iterations)}" / "point_cloud.ply"
    preview_metrics = parse_kv_txt(preview_txt)
    drift_metrics = parse_kv_txt(drift_txt)
    vertex_count = parse_ply_vertex_count(ply_path)
    if vertex_count is not None and "total_count" not in drift_metrics:
        drift_metrics["total_count"] = float(vertex_count)
    return {
        "phase": phase,
        **variant,
        "status": "ok",
        "preview": preview_metrics,
        "drift": drift_metrics,
        "model_dir": str(model_dir),
    }


def metric_value(result: dict[str, Any], key: str) -> Any:
    if result.get("status") != "ok":
        return None
    if key in result.get("drift", {}):
        return result["drift"][key]
    return result.get("preview", {}).get(key)


def format_value(value: Any, key: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if key.endswith("_count") or key.endswith("_added") or key.endswith("_split"):
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def format_table(results: list[dict[str, Any]]) -> str:
    columns = [
        ("phase", "phase"),
        ("group", "group"),
        ("variant", "name"),
        ("status", "status"),
        ("bulk_gap", "bulk_interior_coverage_gap_ratio"),
        ("deep_gap", "bulk_deep_coverage_gap_ratio"),
        ("deep_p10", "bulk_deep_occupancy_p10"),
        ("deep_p50", "bulk_deep_occupancy_p50"),
        ("deep_1p5_3", "bulk_deep_sdf_1p5_3_gap_ratio"),
        ("deep_3_6", "bulk_deep_sdf_3_6_gap_ratio"),
        ("deep_6p", "bulk_deep_sdf_6_plus_gap_ratio"),
        ("low_nn_p50", "bulk_deep_low_occ_nearest_bulk_distance_p50"),
        ("low_sdf_p50", "bulk_deep_low_occ_sdf_depth_p50"),
        ("occ_p10", "bulk_interior_occupancy_p10"),
        ("occ_p50", "bulk_interior_occupancy_p50"),
        ("total", "total_count"),
        ("bulk", "bulk_count"),
        ("surface", "surface_count"),
        ("bulk_reseed", "bulk_reseed_added"),
        ("densify_bulk", "densify_bulk_split"),
        ("densify_children", "densify_children_added"),
        ("mae", "mae"),
        ("psnr", "psnr"),
    ]
    rows: list[list[str]] = []
    for result in results:
        row = []
        for title, key in columns:
            if key in {"phase", "group", "name", "status"}:
                value = result.get(key)
            else:
                value = metric_value(result, key)
            row.append(format_value(value, key))
        rows.append(row)

    widths = [max(len(title), *(len(row[i]) for row in rows)) for i, (title, _key) in enumerate(columns)]
    header = " | ".join(columns[i][0].ljust(widths[i]) for i in range(len(columns)))
    sep = " | ".join("-" * width for width in widths)
    body = "\n".join(" | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) for row in rows)
    return f"{header}\n{sep}\n{body}" if body else f"{header}\n{sep}"


def sortable_gap(result: dict[str, Any]) -> float:
    value = metric_value(result, "bulk_interior_coverage_gap_ratio")
    return float(value) if isinstance(value, float) else float("inf")


def write_outputs(output_root: Path, variants: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    (output_root / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "summary.txt").write_text(format_table(results) + "\n", encoding="utf-8")
    sorted_results = sorted(results, key=sortable_gap)
    (output_root / "summary_by_bulk_gap.txt").write_text(format_table(sorted_results) + "\n", encoding="utf-8")

    lines = ["group\tvariant\tparams\tpurpose\texpected"]
    for variant in variants:
        lines.append(
            "{group}\t{name}\t{params}\t{purpose}\t{expected}".format(
                group=variant["group"],
                name=variant["name"],
                params=json.dumps(variant["params"], sort_keys=True),
                purpose=variant["purpose"],
                expected=variant["expected"],
            )
        )
    (output_root / "variant_plan.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_confirmation_variants(results: list[dict[str, Any]], variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_results = [result for result in results if result.get("phase") == "iter1000" and result.get("status") == "ok"]
    selected_names = ["base", "volume_dominant"]
    if ok_results:
        selected_names.append(min(ok_results, key=sortable_gap)["name"])
        selected_names.append(max(ok_results, key=sortable_gap)["name"])
    unique_names = []
    for name in selected_names:
        if name not in unique_names:
            unique_names.append(name)
    by_name = {variant["name"]: variant for variant in variants}
    return [by_name[name] for name in unique_names if name in by_name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", default=str(DEFAULT_PHASE1_DIR))
    parser.add_argument("--volume-path", default=str(DEFAULT_VOLUME_PATH))
    parser.add_argument("--volume-format", default="dicom")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--confirm-iterations", type=int, default=2500)
    parser.add_argument("--run-confirmation", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--only", nargs="*", default=None, help="Optional variant name subset.")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    variants = VARIANTS
    if args.only:
        wanted = set(args.only)
        variants = [variant for variant in variants if variant["name"] in wanted]
        missing = sorted(wanted - {variant["name"] for variant in variants})
        if missing:
            raise SystemExit(f"Unknown variants: {', '.join(missing)}")

    base_args = [
        "--ct_phase1_dir", args.phase1_dir,
        "--ct_volume_path", args.volume_path,
        "--ct_volume_format", args.volume_format,
        "--ct_surface_boundary_sample_ratio", "0.5",
        "--ct_lambda_volume", "1.0",
        "--ct_lambda_occupancy", "0.5",
        "--ct_surface_regularizer_weight", "0.7",
        "--ct_bulk_coverage_weight", "0.5",
        "--ct_bulk_coverage_target", "0.85",
    ]

    run_parser_smoke(args.python_exe, base_args, variants)
    if args.smoke_only:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else REPO_ROOT / "outputs" / f"bulk_diagnostic_sweep_{ts}"
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for variant in variants:
        results.append(
            run_variant(
                variant,
                phase=f"iter{int(args.iterations)}",
                iterations=args.iterations,
                output_root=output_root,
                base_args=base_args,
                python_exe=args.python_exe,
            )
        )
        write_outputs(output_root, variants, results)

    if args.run_confirmation:
        for variant in select_confirmation_variants(results, variants):
            results.append(
                run_variant(
                    variant,
                    phase=f"iter{int(args.confirm_iterations)}",
                    iterations=args.confirm_iterations,
                    output_root=output_root,
                    base_args=base_args,
                    python_exe=args.python_exe,
                )
            )
            write_outputs(output_root, variants, results)

    print("\n=== Bulk Coverage Summary ===")
    print(format_table(sorted(results, key=sortable_gap)))
    print(f"\nResults saved to {output_root}")


if __name__ == "__main__":
    main()
