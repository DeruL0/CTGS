from __future__ import annotations

import os
from argparse import ArgumentParser
from types import SimpleNamespace


def add_ct_model_args(parser: ArgumentParser) -> None:
    parser.add_argument("--model_path", type=str, default="", help="Directory for checkpoints, point clouds, and exports.")


def add_ct_optimization_args(parser: ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--position_lr_max_steps", type=int, default=30_000)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)
    parser.add_argument("--percent_dense", type=float, default=0.01)
    parser.add_argument("--primitive_harden_iter", type=int, default=2000)
    parser.add_argument("--planar_thickness_max", type=float, default=None)


def extract_ct_model_args(args) -> SimpleNamespace:
    model_path = str(args.model_path).strip()
    if model_path:
        model_path = os.path.abspath(model_path)
    return SimpleNamespace(
        model_path=model_path,
        sh_degree=0,
        data_device="cuda",
    )


def extract_ct_optimization_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        iterations=int(args.iterations),
        percent_dense=float(args.percent_dense),
        position_lr_init=float(args.position_lr_init),
        position_lr_final=float(args.position_lr_final),
        position_lr_delay_mult=float(args.position_lr_delay_mult),
        position_lr_max_steps=int(args.position_lr_max_steps),
        feature_lr=float(args.feature_lr),
        opacity_lr=float(args.opacity_lr),
        scaling_lr=float(args.scaling_lr),
        rotation_lr=float(args.rotation_lr),
        primitive_harden_iter=int(args.primitive_harden_iter),
        planar_thickness_max=args.planar_thickness_max,
    )
