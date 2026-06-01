from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    name: str
    params: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    lower = value.lower()
    if lower in {"nan", "+nan", "-nan"}:
        return float("nan")
    if lower in {"inf", "+inf"}:
        return float("inf")
    if lower == "-inf":
        return float("-inf")
    try:
        if any(token in value for token in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_kv_report(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key.strip()] = _parse_scalar(value)
    return metrics


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _run(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(f"[RUN] {printable}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


_DENSIFY_DEFAULTS = {
    "ct_enable_densification": True,
}


def _with_densify(params: dict) -> dict:
    merged = dict(params)
    merged.update(_DENSIFY_DEFAULTS)
    return merged


def _candidate_sweep_configs() -> list[Candidate]:
    return [
        Candidate(
            name="balanced_anchor",
            params=_with_densify({
                "position_lr_init": 2.0e-5,
                "position_lr_final": 2.0e-7,
                "ct_volume_sample_count": 16384,
                "ct_volume_jitter": 0.75,
                "ct_surface_regularizer_weight": 0.8,
                "ct_surface_boundary_sample_ratio": 0.25,
            }),
        ),
        Candidate(
            name="tight_anchor",
            params=_with_densify({
                "position_lr_init": 1.5e-5,
                "position_lr_final": 1.5e-7,
                "ct_volume_sample_count": 16384,
                "ct_volume_jitter": 0.75,
                "ct_surface_regularizer_weight": 1.0,
                "ct_boundary_band": 1.25,
            }),
        ),
        Candidate(
            name="connectivity_relaxed",
            params=_with_densify({
                "position_lr_init": 2.5e-5,
                "position_lr_final": 2.5e-7,
                "ct_volume_sample_count": 16384,
                "ct_volume_jitter": 0.7,
                "ct_surface_regularizer_weight": 0.7,
                "ct_surface_boundary_sample_ratio": 0.2,
            }),
        ),
        Candidate(
            name="high_sample_balanced",
            params=_with_densify({
                "position_lr_init": 2.0e-5,
                "position_lr_final": 2.0e-7,
                "ct_volume_sample_count": 24576,
                "ct_volume_jitter": 0.5,
                "ct_surface_regularizer_weight": 0.9,
                "ct_surface_boundary_sample_ratio": 0.3,
            }),
        ),
    ]


def _base_train_args(args: argparse.Namespace) -> list[str]:
    return [
        "--ct_phase1_dir",
        args.phase1_dir,
        "--ct_volume_path",
        args.volume_path,
        "--ct_volume_format",
        args.volume_format,
        "--ct_raw_meta",
        args.raw_meta,
        "--ct_air_sample_count",
        "18000",
        "--ct_lambda_volume",
        "1.0",
        "--ct_lambda_occupancy",
        "0.5",
        "--ct_bulk_max_scale",
        "3.2",
        "--skip_export_mesh",
        "--skip_export_sdf",
        "--quiet",
    ]


def _append_param_args(cmd: list[str], params: dict[str, Any]) -> None:
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cmd.append(f"--{key}" if value else f"--no-{key}")
            continue
        cmd.extend([f"--{key}", str(value)])


def _train_command(
    python_exe: str,
    args: argparse.Namespace,
    model_path: Path,
    iterations: int,
    save_iterations: list[int],
    checkpoint_iterations: list[int],
    params: dict[str, Any],
    start_checkpoint: Path | None = None,
) -> list[str]:
    output_gs = model_path / f"display_iter{int(iterations)}.ply"
    cmd = [
        python_exe,
        "train_ct.py",
        "--model_path",
        str(model_path),
        "--iterations",
        str(int(iterations)),
        "--save_iterations",
        *[str(int(value)) for value in save_iterations],
        "--checkpoint_iterations",
        *[str(int(value)) for value in checkpoint_iterations],
        "--output_gs",
        str(output_gs),
    ]
    if start_checkpoint is not None:
        cmd.extend(["--start_checkpoint", str(start_checkpoint)])
    cmd.extend(_base_train_args(args))
    _append_param_args(cmd, params)
    return cmd


def _mesh_command(
    python_exe: str,
    point_cloud: Path,
    output_mesh: Path,
    resolution: float,
    seed: int,
) -> list[str]:
    return [
        python_exe,
        "mesher.py",
        "--input",
        str(point_cloud),
        "--output",
        str(output_mesh),
        "--method",
        "sugar",
        "--resolution",
        str(resolution),
        "--sugar_poisson_depth",
        "9",
        "--sugar_max_points",
        "160000",
        "--sugar_tangent_scale",
        "0.75",
        "--sugar_density_quantile",
        "0.08",
        "--sugar_normal_consistency_k",
        "30",
        "--sugar_component_min_face_ratio",
        "0.005",
        "--seed",
        str(int(seed)),
    ]


def _mesh_eval_command(
    python_exe: str,
    mesh_path: Path,
    phase1_dir: str,
    output_json: Path,
    seed: int,
) -> list[str]:
    return [
        python_exe,
        "-m",
        "tools.mesh_evaluator",
        "--mesh",
        str(mesh_path),
        "--phase1",
        phase1_dir,
        "--output",
        str(output_json),
        "--sample-count",
        "100000",
        "--reference-sample-count",
        "100000",
        "--seed",
        str(int(seed)),
    ]


def _score_candidate(drift: dict[str, Any], preview: dict[str, Any], mesh: dict[str, Any]) -> float:
    largest_component = float(mesh.get("largest_component_face_ratio", 0.0) or 0.0)
    score = 0.0
    score += 1.2 * float(mesh.get("symmetric_chamfer_l1_mean", 1e9))
    score += 1.5 * float(mesh.get("symmetric_hausdorff_p95", 1e9))
    score += 0.6 * float(mesh.get("mesh_sample_outside_support_ratio", 1e9))
    score += 0.5 * float(mesh.get("mesh_outside_distance_p99", 1e9))
    score += 0.4 * float(drift.get("surface_to_phase1_boundary_distance_p99", 1e9))
    score += 0.2 * float(drift.get("surface_outside_support_ratio", 1e9))
    score += 2.0 * max(0.0, 0.98 - largest_component)
    score += 0.2 * max(0.0, 29.5 - float(preview.get("psnr", 0.0) or 0.0))
    return float(score)


def _evaluate_run(
    repo_root: Path,
    python_exe: str,
    args: argparse.Namespace,
    model_dir: Path,
    iteration: int,
    candidate: Candidate,
) -> dict[str, Any]:
    point_cloud = model_dir / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    if not point_cloud.exists():
        raise FileNotFoundError(f"Missing point cloud for evaluation: {point_cloud}")

    mesh_path = model_dir / f"mesh_iter{int(iteration)}_sugar_eval.ply"
    mesh_metrics_json = model_dir / f"mesh_iter{int(iteration)}_sugar_eval_metrics.json"
    mesh_metrics_txt = model_dir / f"mesh_iter{int(iteration)}_sugar_eval_metrics.txt"
    drift_path = model_dir / "diagnostics" / f"drift_iter_{int(iteration):06d}.txt"
    preview_path = model_dir / "previews" / f"preview_iter_{int(iteration):06d}.txt"

    _run(
        _mesh_command(
            python_exe=python_exe,
            point_cloud=point_cloud,
            output_mesh=mesh_path,
            resolution=args.mesh_resolution,
            seed=args.seed,
        ),
        cwd=repo_root,
    )
    _run(
        _mesh_eval_command(
            python_exe=python_exe,
            mesh_path=mesh_path,
            phase1_dir=args.phase1_dir,
            output_json=mesh_metrics_json,
            seed=args.seed,
        ),
        cwd=repo_root,
    )

    drift = _parse_kv_report(drift_path)
    preview = _parse_kv_report(preview_path)
    mesh = _parse_kv_report(mesh_metrics_txt)
    score = _score_candidate(drift=drift, preview=preview, mesh=mesh)
    return {
        "candidate": asdict(candidate),
        "iteration": int(iteration),
        "model_dir": model_dir,
        "point_cloud": point_cloud,
        "mesh_path": mesh_path,
        "mesh_metrics_json": mesh_metrics_json,
        "mesh_metrics_txt": mesh_metrics_txt,
        "drift_path": drift_path,
        "preview_path": preview_path,
        "score": score,
        "drift": drift,
        "preview": preview,
        "mesh": mesh,
    }


def _write_summary(root: Path, payload: dict[str, Any]) -> None:
    summary_json = root / "summary.json"
    summary_txt = root / "summary.txt"
    summary_json.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")

    lines = [
        f"run_root = {root}",
        f"sweep_iterations = {payload['sweep_iterations']}",
        f"final_iterations = {payload['final_iterations']}",
        "",
        "[sweep_results]",
    ]
    for item in payload["sweep_results"]:
        lines.extend(
            [
                f"name = {item['candidate']['name']}",
                f"score = {item['score']:.6f}",
                f"surface_outside = {float(item['drift'].get('surface_outside_support_ratio', float('nan'))):.6f}",
                f"surface_boundary_p99 = {float(item['drift'].get('surface_to_phase1_boundary_distance_p99', float('nan'))):.6f}",
                f"mesh_outside = {float(item['mesh'].get('mesh_sample_outside_support_ratio', float('nan'))):.6f}",
                f"mesh_outside_p99 = {float(item['mesh'].get('mesh_outside_distance_p99', float('nan'))):.6f}",
                f"mesh_chamfer = {float(item['mesh'].get('symmetric_chamfer_l1_mean', float('nan'))):.6f}",
                f"mesh_hausdorff_p95 = {float(item['mesh'].get('symmetric_hausdorff_p95', float('nan'))):.6f}",
                f"largest_component = {float(item['mesh'].get('largest_component_face_ratio', float('nan'))):.6f}",
                f"preview_psnr = {float(item['preview'].get('psnr', float('nan'))):.6f}",
                f"model_dir = {item['model_dir']}",
                "",
            ]
        )
    best = payload.get("best_sweep")
    if best is not None:
        lines.extend(
            [
                "[best_sweep]",
                f"name = {best['candidate']['name']}",
                f"score = {best['score']:.6f}",
                f"model_dir = {best['model_dir']}",
                "",
            ]
        )
    final_result = payload.get("final_result")
    if final_result is not None:
        lines.extend(
            [
                "[final_result]",
                f"name = {final_result['candidate']['name']}",
                f"score = {final_result['score']:.6f}",
                f"surface_outside = {float(final_result['drift'].get('surface_outside_support_ratio', float('nan'))):.6f}",
                f"surface_boundary_p99 = {float(final_result['drift'].get('surface_to_phase1_boundary_distance_p99', float('nan'))):.6f}",
                f"mesh_outside = {float(final_result['mesh'].get('mesh_sample_outside_support_ratio', float('nan'))):.6f}",
                f"mesh_outside_p99 = {float(final_result['mesh'].get('mesh_outside_distance_p99', float('nan'))):.6f}",
                f"mesh_chamfer = {float(final_result['mesh'].get('symmetric_chamfer_l1_mean', float('nan'))):.6f}",
                f"mesh_hausdorff_p95 = {float(final_result['mesh'].get('symmetric_hausdorff_p95', float('nan'))):.6f}",
                f"largest_component = {float(final_result['mesh'].get('largest_component_face_ratio', float('nan'))):.6f}",
                f"preview_psnr = {float(final_result['preview'].get('psnr', float('nan'))):.6f}",
                f"model_dir = {final_result['model_dir']}",
                f"mesh_path = {final_result['mesh_path']}",
            ]
        )
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-tune and run the best CT figure training recipe.")
    parser.add_argument("--phase1-dir", default="outputs/figure_153505_test/phase1_dense")
    parser.add_argument("--volume-path", default="outputs/figure_153505_test/figure_153505_ds4x4x4_float32_norm.raw")
    parser.add_argument("--volume-format", default="raw")
    parser.add_argument("--raw-meta", default="outputs/figure_153505_test/figure_153505_ds4x4x4_float32_norm.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--sweep-iterations", type=int, default=2500)
    parser.add_argument("--final-iterations", type=int, default=5000)
    parser.add_argument("--mesh-resolution", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-final", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = _repo_root()
    python_exe = sys.executable
    if args.output_root is None:
        output_root = repo_root / "outputs" / "figure_153505_test" / f"auto_tune_{_timestamp()}"
    else:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = (repo_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sweep_results: list[dict[str, Any]] = []
    candidates = _candidate_sweep_configs()
    for candidate in candidates:
        model_dir = output_root / f"sweep_{candidate.name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        _run(
            _train_command(
                python_exe=python_exe,
                args=args,
                model_path=model_dir,
                iterations=args.sweep_iterations,
                save_iterations=[args.sweep_iterations],
                checkpoint_iterations=[args.sweep_iterations],
                params=candidate.params,
            ),
            cwd=repo_root,
        )
        result = _evaluate_run(
            repo_root=repo_root,
            python_exe=python_exe,
            args=args,
            model_dir=model_dir,
            iteration=args.sweep_iterations,
            candidate=candidate,
        )
        sweep_results.append(result)
        sweep_results.sort(key=lambda item: item["score"])
        _write_summary(
            output_root,
            {
                "sweep_iterations": args.sweep_iterations,
                "final_iterations": args.final_iterations,
                "sweep_results": sweep_results,
                "best_sweep": sweep_results[0],
                "final_result": None,
            },
        )

    sweep_results.sort(key=lambda item: item["score"])
    best_sweep = sweep_results[0]
    final_result: dict[str, Any] | None = None

    if not args.skip_final:
        best_candidate = Candidate(**best_sweep["candidate"])
        final_model_dir = output_root / f"winner_{best_candidate.name}_iter{int(args.final_iterations)}"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(best_sweep["model_dir"]) / f"chkpnt{int(args.sweep_iterations)}.pth"
        _run(
            _train_command(
                python_exe=python_exe,
                args=args,
                model_path=final_model_dir,
                iterations=args.final_iterations,
                save_iterations=[args.final_iterations],
                checkpoint_iterations=[args.final_iterations],
                params=best_candidate.params,
                start_checkpoint=checkpoint_path,
            ),
            cwd=repo_root,
        )
        final_result = _evaluate_run(
            repo_root=repo_root,
            python_exe=python_exe,
            args=args,
            model_dir=final_model_dir,
            iteration=args.final_iterations,
            candidate=best_candidate,
        )

    _write_summary(
        output_root,
        {
            "sweep_iterations": args.sweep_iterations,
            "final_iterations": args.final_iterations,
            "sweep_results": sweep_results,
            "best_sweep": best_sweep,
            "final_result": final_result,
        },
    )
    print(f"[DONE] Summary written to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
