from __future__ import annotations

import argparse
import shlex
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ct_pipeline import extract_ct_model_args, extract_ct_optimization_args
from ct_pipeline.training.config import build_parser, validate_ct_training_args
from ct_pipeline.training.bootstrap import prepare_ct_training_bootstrap
from ct_pipeline.training.utils import write_key_value_report
from scene.ct_bulk_initialization import _apply_directional_clearance_scales
from utils.rotation_utils import quaternion_to_matrix


def _load_training_args(command_file: Path, checkpoint: Path, bootstrap_dir: Path):
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
    args.ct_init_preflight_abort = False
    validate_ct_training_args(args)
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply directional-clearance bulk scale contraction to a CTGS checkpoint.")
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--q-cont", type=float, default=4.0)
    parser.add_argument("--safety", type=float, default=0.85)
    args_cli = parser.parse_args()

    command_file = Path(args_cli.command_file).resolve()
    checkpoint = Path(args_cli.checkpoint).resolve()
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir = output_dir / "_bootstrap_runtime"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    args = _load_training_args(command_file, checkpoint, bootstrap_dir)
    dataset = extract_ct_model_args(args)
    opt = extract_ct_optimization_args(args)
    context = prepare_ct_training_bootstrap(dataset, opt, args, args.start_checkpoint)
    if context.tb_writer is not None:
        context.tb_writer.close()

    gaussians = context.gaussians
    bulk_mask = (gaussians.get_region_type.reshape(-1) == 1).detach()
    if not torch.any(bulk_mask):
        raise RuntimeError("checkpoint has no bulk Gaussians")

    old_scales_t = gaussians.get_scaling.detach()[bulk_mask].to(dtype=torch.float32)
    bulk_xyz = gaussians.get_xyz.detach()[bulk_mask].cpu().numpy()
    bulk_scales = old_scales_t.cpu().numpy()
    bulk_rotations = quaternion_to_matrix(gaussians.get_rotation.detach()[bulk_mask]).to(dtype=torch.float32).cpu().numpy()
    material_mask = context.analysis.get("material_mask", context.analysis.get("coarse_support_mask"))
    signed_distance = context.signed_distance_field.get("signed_distance_native")
    if signed_distance is None:
        signed_distance = context.signed_distance_field.get("signed_distance")
    if torch.is_tensor(signed_distance):
        signed_distance = signed_distance.detach().reshape(*tuple(int(v) for v in signed_distance.shape[-3:])).cpu().numpy()
    if signed_distance is None:
        signed_distance = context.analysis.get("material_signed_distance")
    new_scales, clearance_stats = _apply_directional_clearance_scales(
        bulk_xyz,
        bulk_scales,
        bulk_rotations,
        material_mask,
        context.spacing_zyx,
        float(min(context.spacing_zyx)),
        signed_distance_volume=signed_distance,
        q_cont=float(args_cli.q_cont),
        safety=float(args_cli.safety),
    )

    new_scales_t = torch.as_tensor(new_scales, dtype=gaussians._scaling.dtype, device=gaussians._scaling.device).clamp_min(1e-8)
    with torch.no_grad():
        gaussians._scaling[bulk_mask] = torch.log(new_scales_t)

    ratios = (new_scales_t.to(dtype=torch.float32) / old_scales_t.to(device=new_scales_t.device).clamp_min(1e-8)).detach()
    rows = [
        ("bulk_count", int(bulk_mask.sum().item())),
        ("q_cont", float(args_cli.q_cont)),
        ("safety", float(args_cli.safety)),
        ("scale_ratio_p10", float(torch.quantile(ratios.reshape(-1), 0.10).item())),
        ("scale_ratio_p50", float(torch.quantile(ratios.reshape(-1), 0.50).item())),
        ("scale_ratio_p90", float(torch.quantile(ratios.reshape(-1), 0.90).item())),
    ]
    rows.extend(clearance_stats.items())
    write_key_value_report(output_dir / "directional_clearance_contraction.txt", rows)

    iteration = int(args_cli.iteration) if args_cli.iteration is not None else 0
    ckpt_path = output_dir / f"chkpnt{iteration}_contracted.pth"
    torch.save((gaussians.capture(), iteration), ckpt_path)
    ply_dir = output_dir / "point_cloud" / f"iteration_{iteration}"
    ply_dir.mkdir(parents=True, exist_ok=True)
    ply_path = ply_dir / "point_cloud.ply"
    gaussians.save_ply(str(ply_path))

    shutil.copy2(command_file, output_dir / "command.txt")
    source_cfg = command_file.parent / "cfg_args"
    if source_cfg.exists():
        shutil.copy2(source_cfg, output_dir / "cfg_args")
    print(f"wrote_checkpoint={ckpt_path}")
    print(f"wrote_ply={ply_path}")
    print(f"wrote_stats={output_dir / 'directional_clearance_contraction.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
