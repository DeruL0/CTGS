"""Bulk kernel mass probe for radiative CT Gaussians.

Loads a checkpoint and phases-1 SDF, samples probe points by region, and
reports W_b / mu_raw / mu_norm / attenuation statistics WITHOUT running any
training step.

Usage
-----
    python tools/ct_probe_bulk_coverage.py \
        --checkpoint outputs/bunny_smoke/ctgs_v521_V2_tauopen_20260531_000952/chkpnt1000.pth \
        --phase1_dir  outputs/bunny_smoke/phase1_surface_complete_grid3_margin1_20260522 \
        --output      outputs/bunny_smoke/probe_V2_tauopen_1000 \
        --num_points  200000

Post-hoc experiments (no re-training):
    --bulk_scale_multiplier 2.0       # scale × k before kernel eval
    --bulk_atten_multiplier 2.0       # attenuation × k before kernel eval
    --compact_support_posthoc         # apply (1-q²)² compact taper
    --support_radius_factor 2.0       # R_i = min(factor*sigma, inside_sdf_factor*inside_dist)
    --support_inside_sdf_factor 0.8   # clamp R_i to 0.8 × inside distance
    --render_preview                  # save PNG panel alongside TSV
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# repo root on path
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ct_pipeline.training.bootstrap.analysis import (
    _load_ct_analysis_bundle,
    _prepare_signed_distance_field,
)
from utils.rotation_utils import quaternion_to_matrix


# ---------------------------------------------------------------------------
# Checkpoint loading (no GaussianModel instantiation needed)
# ---------------------------------------------------------------------------

def _load_raw_state(chkpnt_path: Path, device: str) -> dict:
    payload, iteration = torch.load(str(chkpnt_path), map_location=device, weights_only=False)
    if len(payload) not in (18, 19, 20):
        raise ValueError(f"Unexpected checkpoint tuple length {len(payload)}")
    # unpack by position – matches capture_state order in gaussian_model_io.py
    (
        _active_sh_degree,
        xyz,           # (N,3)
        _fdc,
        _frest,
        scaling_raw,   # (N,3) log-scale
        rotation_quat, # (N,4)
        opacity_logit, # (N,1)
        _prim_type,
        _normal,
        _mat_id,
        _planarity,
        region_type,   # (N,1) int  0=surface 1=bulk
        ct_value_logit,
        *rest,
    ) = payload
    # rest[0] may be atten_logit (len-20) or max_radii2D (len-18/19)
    atten_logit = None
    if len(payload) == 20:
        atten_logit = rest[0]  # (N,1) or (N,)
    elif len(payload) == 19:
        # atten_logit was absent; derive from opacity * ct_value as in restore_state
        atten_init = torch.clamp(
            torch.sigmoid(opacity_logit.float()) * torch.sigmoid(ct_value_logit.float()),
            min=1e-6,
        )
        atten_logit = _inverse_softplus(atten_init)

    xyz = xyz.to(device=device, dtype=torch.float32).detach()
    scaling_raw = scaling_raw.to(device=device, dtype=torch.float32).detach()
    rotation_quat = rotation_quat.to(device=device, dtype=torch.float32).detach()
    region_type = region_type.to(device=device).reshape(-1).long()
    if atten_logit is not None:
        atten_logit = atten_logit.to(device=device, dtype=torch.float32).detach().reshape(-1)

    scales = torch.exp(scaling_raw).clamp_min(1e-6)          # (N,3)
    rotation_mats = quaternion_to_matrix(rotation_quat)       # (N,3,3)

    bulk_mask = region_type == 1
    return {
        "iteration": iteration,
        "xyz": xyz,
        "scales": scales,
        "rotation_mats": rotation_mats,
        "region_type": region_type,
        "bulk_mask": bulk_mask,
        "bulk_xyz": xyz[bulk_mask],
        "bulk_scales": scales[bulk_mask],
        "bulk_rotation_mats": rotation_mats[bulk_mask],
        "bulk_atten_logit": atten_logit[bulk_mask] if atten_logit is not None else None,
    }


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x.clamp_min(1e-6))))


# ---------------------------------------------------------------------------
# Isotropic radiative bulk kernel  (mirrors _query_radiative_bulk_field)
# ---------------------------------------------------------------------------

def _compute_support_radius(
    bulk_scales: torch.Tensor,       # (M,3)
    bulk_center_sdf: torch.Tensor,   # (M,) — negative inside material
    support_radius_factor: float = 2.0,
    support_inside_sdf_factor: float = 0.8,
) -> torch.Tensor:
    """R_i = min(factor * sigma_i, inside_sdf_factor * inside_dist_i)."""
    sigma = bulk_scales.mean(dim=1).clamp_min(1e-6)
    inside_dist = (-bulk_center_sdf).clamp_min(1e-6)
    return torch.minimum(
        support_radius_factor * sigma,
        support_inside_sdf_factor * inside_dist,
    )


def _build_bulk_kdtree(bulk_xyz_np: np.ndarray):
    """Build a cKDTree from bulk Gaussian centres (CPU numpy)."""
    from scipy.spatial import cKDTree
    return cKDTree(bulk_xyz_np)


def _query_bulk_field_iso(
    bulk_xyz: torch.Tensor,          # (M,3)
    bulk_scales: torch.Tensor,       # (M,3)
    bulk_attenuation: torch.Tensor,  # (M,)
    probe_xyz: torch.Tensor,         # (P,3)
    chunk_size: int = 512,
    tau: float = 0.5,
    bulk_center_sdf: torch.Tensor | None = None,
    bulk_center_normals: torch.Tensor | None = None,
    skip_depth: float = 2.0,
    compact_support: bool = False,
    support_radius: torch.Tensor | None = None,  # (M,)
    kdtree=None,          # pre-built cKDTree; built on first call if None
    search_radius: float | None = None,  # KD-tree query radius; auto if None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mu_raw [P,], W_b [P,]).

    Uses KD-tree radius search to avoid O(P*M) memory when M is large.
    compact_support: apply C(q)=(1-q²)² taper, zero for q>=1.
    """
    from scipy.spatial import cKDTree

    device = probe_xyz.device
    P = probe_xyz.shape[0]
    M = bulk_xyz.shape[0]

    sigma_np = bulk_scales.mean(dim=1).clamp_min(1e-6).cpu().numpy()  # (M,)
    sigma_max = float(sigma_np.max())

    # auto search radius: 3*sigma_max covers ~99% of Gaussian mass
    if search_radius is None:
        if compact_support and support_radius is not None:
            search_radius = float(support_radius.max().item()) * 1.01
        else:
            search_radius = 3.0 * sigma_max

    # build KD-tree on CPU from bulk centres
    bulk_xyz_np = bulk_xyz.detach().cpu().numpy()
    if kdtree is None:
        kdtree = cKDTree(bulk_xyz_np)

    # numpy arrays for CPU-side lookup
    atten_np = bulk_attenuation.detach().cpu().numpy()
    center_sdf_np = bulk_center_sdf.detach().cpu().numpy() if bulk_center_sdf is not None else None
    center_normals_np = (
        F.normalize(bulk_center_normals, dim=-1, eps=1e-6).detach().cpu().numpy()
        if bulk_center_normals is not None else None
    )
    R_np = support_radius.detach().cpu().numpy() if (compact_support and support_radius is not None) else None

    probe_np = probe_xyz.detach().cpu().numpy()
    mu_out = np.zeros(P, dtype=np.float32)
    W_out = np.zeros(P, dtype=np.float32)

    # process in CPU chunks to avoid any GPU memory spike
    for p_start in range(0, P, chunk_size):
        p_end = min(p_start + chunk_size, P)
        qpts = probe_np[p_start:p_end]  # (Cp, 3)

        # radius search: list of arrays, one per probe point
        nbr_lists = kdtree.query_ball_point(qpts, r=search_radius, workers=1)

        for local_i, nbr_ids in enumerate(nbr_lists):
            if len(nbr_ids) == 0:
                continue
            ids = np.array(nbr_ids, dtype=np.int64)
            diff = qpts[local_i] - bulk_xyz_np[ids]          # (K,3)
            d2 = (diff * diff).sum(-1)                         # (K,)
            s = sigma_np[ids]
            gauss = np.exp(-0.5 * d2 / np.maximum(s * s, 1e-8))

            if center_sdf_np is not None:
                lin_sdf = center_sdf_np[ids]
                if center_normals_np is not None:
                    lin_sdf = lin_sdf + (diff * center_normals_np[ids]).sum(-1)
                gate = 1.0 / (1.0 + np.exp(lin_sdf / float(tau)))  # sigmoid(-sdf/tau)
                deep = center_sdf_np[ids] <= -float(skip_depth)
                gate = np.where(deep, 1.0, gate)
                kernel = gauss * gate
            else:
                kernel = gauss

            if R_np is not None:
                R = R_np[ids]
                q2 = d2 / np.maximum(R * R, 1e-12)
                taper = np.maximum(1.0 - q2, 0.0) ** 2
                kernel = kernel * taper

            W_out[p_start + local_i] = kernel.sum()
            mu_out[p_start + local_i] = (kernel * atten_np[ids]).sum()

    return (
        torch.from_numpy(mu_out).to(device=device, dtype=torch.float32),
        torch.from_numpy(W_out).to(device=device, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Region sampling from SDF
# ---------------------------------------------------------------------------

def _world_coords_from_sdf(sdf_native: np.ndarray, spacing_zyx, device: str) -> torch.Tensor:
    """Return world-space (x,y,z) coords for every voxel centre (D*H*W, 3)."""
    D, H, W = sdf_native.shape
    sz, sy, sx = spacing_zyx
    zz = (np.arange(D) + 0.5) * sz
    yy = (np.arange(H) + 0.5) * sy
    xx = (np.arange(W) + 0.5) * sx
    gz, gy, gx = np.meshgrid(zz, yy, xx, indexing="ij")
    # world = (x, y, z) matching CTGS convention
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)
    return torch.from_numpy(coords).to(device)


def _sample_region(
    sdf_native: np.ndarray,
    spacing_zyx,
    device: str,
    sdf_min: float,
    sdf_max: float,
    n: int,
    rng: np.random.Generator,
) -> torch.Tensor | None:
    mask = (sdf_native >= sdf_min) & (sdf_native < sdf_max)
    idx = np.where(mask.ravel())[0]
    if idx.size == 0:
        return None
    chosen = rng.choice(idx, size=min(n, idx.size), replace=False)
    D, H, W = sdf_native.shape
    sz, sy, sx = spacing_zyx
    z_idx = chosen // (H * W)
    y_idx = (chosen % (H * W)) // W
    x_idx = chosen % W
    x = (x_idx + 0.5) * sx
    y = (y_idx + 0.5) * sy
    z = (z_idx + 0.5) * sz
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    return torch.from_numpy(pts).to(device)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _pct(t: torch.Tensor, p: int) -> float:
    if t.numel() == 0:
        return float("nan")
    return float(torch.quantile(t.float(), p / 100.0).item())


def _stats(t: torch.Tensor) -> dict:
    if t.numel() == 0:
        return {k: float("nan") for k in ("p10", "p50", "p90", "mean")}
    return {
        "p10": _pct(t, 10),
        "p50": _pct(t, 50),
        "p90": _pct(t, 90),
        "mean": float(t.float().mean().item()),
    }


# ---------------------------------------------------------------------------
# Top-k contribution ratio
# ---------------------------------------------------------------------------

def _topk_ratio(
    bulk_xyz: torch.Tensor,
    bulk_scales: torch.Tensor,
    bulk_attenuation: torch.Tensor,
    probe_pts: torch.Tensor,
    k: int,
    chunk: int = 512,
    tau: float = 0.5,
    compact_support: bool = False,
    support_radius: torch.Tensor | None = None,
    kdtree=None,
    search_radius: float | None = None,
) -> torch.Tensor:
    """For each probe point: sum(top-k a_i K_i) / mu_raw.  Uses KD-tree for large M."""
    from scipy.spatial import cKDTree

    sigma_np = bulk_scales.mean(dim=1).clamp_min(1e-6).detach().cpu().numpy()
    sigma_max = float(sigma_np.max())
    if search_radius is None:
        if compact_support and support_radius is not None:
            search_radius = float(support_radius.max().item()) * 1.01
        else:
            search_radius = 3.0 * sigma_max

    bulk_xyz_np = bulk_xyz.detach().cpu().numpy()
    if kdtree is None:
        kdtree = cKDTree(bulk_xyz_np)
    atten_np = bulk_attenuation.detach().cpu().numpy()
    R_np = support_radius.detach().cpu().numpy() if (compact_support and support_radius is not None) else None
    probe_np = probe_pts.detach().cpu().numpy()
    P = probe_np.shape[0]
    ratios = np.zeros(P, dtype=np.float32)

    for p_start in range(0, P, chunk):
        p_end = min(p_start + chunk, P)
        qpts = probe_np[p_start:p_end]
        nbr_lists = kdtree.query_ball_point(qpts, r=search_radius, workers=1)
        for local_i, nbr_ids in enumerate(nbr_lists):
            if len(nbr_ids) == 0:
                continue
            ids = np.array(nbr_ids, dtype=np.int64)
            diff = qpts[local_i] - bulk_xyz_np[ids]
            d2 = (diff * diff).sum(-1)
            s = sigma_np[ids]
            gauss = np.exp(-0.5 * d2 / np.maximum(s * s, 1e-8))
            if R_np is not None:
                R = R_np[ids]
                q2 = d2 / np.maximum(R * R, 1e-12)
                gauss = gauss * np.maximum(1.0 - q2, 0.0) ** 2
            contrib = gauss * atten_np[ids]
            total = contrib.sum()
            if total < 1e-12:
                continue
            top_k = np.partition(contrib, -min(k, len(contrib)))[-min(k, len(contrib)):]
            ratios[p_start + local_i] = top_k.sum() / total

    return torch.from_numpy(ratios).to(device=probe_pts.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Per-region probe
# ---------------------------------------------------------------------------

REGIONS = {
    # name: (sdf_min, sdf_max)
    "material":                  (-99.0,  -0.5),
    "material_boundary_shell":   (-1.5,    0.0),
    "deep_material":             (-99.0,  -3.0),
    "void_air":                  (0.5,    99.0),
    "cavity_void":               (0.5,    3.0),
    "false_hole":                (-2.0,   0.0),   # same as inner boundary for diagnosis
}


def probe_region(
    region_name: str,
    sdf_min: float,
    sdf_max: float,
    sdf_native: np.ndarray,
    spacing_zyx,
    device: str,
    bulk_xyz: torch.Tensor,
    bulk_scales: torch.Tensor,
    bulk_attenuation: torch.Tensor,
    bulk_center_sdf: torch.Tensor | None,
    bulk_center_normals: torch.Tensor | None,
    n_per_region: int,
    tau: float,
    skip_depth: float,
    rng: np.random.Generator,
    compact_support: bool = False,
    support_radius: torch.Tensor | None = None,
    kdtree=None,
    search_radius: float | None = None,
) -> dict:
    pts = _sample_region(sdf_native, spacing_zyx, device, sdf_min, sdf_max, n_per_region, rng)
    if pts is None or pts.numel() == 0:
        return {"region": region_name, "n": 0}

    mu_raw, W_b = _query_bulk_field_iso(
        bulk_xyz, bulk_scales, bulk_attenuation, pts,
        tau=tau, bulk_center_sdf=bulk_center_sdf,
        bulk_center_normals=bulk_center_normals, skip_depth=skip_depth,
        compact_support=compact_support, support_radius=support_radius,
        kdtree=kdtree, search_radius=search_radius,
    )
    eps = 1e-9
    mu_norm = mu_raw / (W_b + eps)

    r = {"region": region_name, "n": int(pts.shape[0])}
    for prefix, arr in [("W_b", W_b), ("mu_raw", mu_raw), ("mu_norm", mu_norm), ("atten", bulk_attenuation)]:
        for k, v in _stats(arr).items():
            r[f"{prefix}_{k}"] = v

    # top-k concentration
    if bulk_xyz.shape[0] > 0:
        for k_val in (1, 4, 8):
            ratio = _topk_ratio(
                bulk_xyz, bulk_scales, bulk_attenuation, pts, k=k_val, tau=tau,
                compact_support=compact_support, support_radius=support_radius,
                kdtree=kdtree, search_radius=search_radius,
            )
            r[f"top{k_val}_contrib_ratio_p50"] = _pct(ratio, 50)

    return r


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------

def write_tsv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    all_keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    lines = ["\t".join(all_keys)]
    for row in rows:
        lines.append("\t".join(str(row.get(k, "")) for k in all_keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# PNG diagnostics
# ---------------------------------------------------------------------------

def _render_single_slice(
    sdf_native: np.ndarray,
    spacing_zyx,
    device: str,
    bulk_xyz: torch.Tensor,
    bulk_scales: torch.Tensor,
    bulk_attenuation: torch.Tensor,
    bulk_center_sdf: torch.Tensor | None,
    bulk_center_normals: torch.Tensor | None,
    tau: float,
    skip_depth: float,
    compact_support: bool = False,
    support_radius: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a central Z slice of W_b, mu_raw, mu_norm.  Returns (H,W) arrays."""
    D, H, W = sdf_native.shape
    sz, sy, sx = spacing_zyx
    z_slice = D // 2
    yy = (np.arange(H) + 0.5) * sy
    xx = (np.arange(W) + 0.5) * sx
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    gz = np.full_like(gx, (z_slice + 0.5) * sz)
    pts_np = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)
    pts = torch.from_numpy(pts_np).to(device)

    with torch.no_grad():
        mu_raw, W_b = _query_bulk_field_iso(
            bulk_xyz, bulk_scales, bulk_attenuation, pts,
            tau=tau, bulk_center_sdf=bulk_center_sdf,
            bulk_center_normals=bulk_center_normals, skip_depth=skip_depth,
            chunk_size=1024,
            compact_support=compact_support, support_radius=support_radius,
        )
    eps = 1e-9
    mu_norm = mu_raw / (W_b + eps)
    return (
        W_b.cpu().numpy().reshape(H, W),
        mu_raw.cpu().numpy().reshape(H, W),
        mu_norm.cpu().numpy().reshape(H, W),
    )


def save_png_panel(
    W_b: np.ndarray,
    mu_raw: np.ndarray,
    mu_norm: np.ndarray,
    sdf_slice: np.ndarray,
    out_dir: Path,
    suffix: str = "",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping PNG output.")
        return

    inside = sdf_slice < 0.0

    def _clip(arr: np.ndarray, p99: float) -> np.ndarray:
        return np.clip(arr, 0.0, max(p99, 1e-6)) / max(p99, 1e-6)

    W_p99 = float(np.percentile(W_b[inside], 99)) if inside.any() else 1.0
    mu_p99 = float(np.percentile(mu_raw[inside], 99)) if inside.any() else 1.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles = [
        f"W_b (kernel mass)\np99={W_p99:.3f}",
        f"mu_raw = Σa·K\np99={mu_p99:.3f}",
        "mu_norm = mu_raw / W_b",
        "SDF (inside<0)",
    ]
    images = [
        _clip(W_b, W_p99),
        _clip(mu_raw, mu_p99),
        np.clip(mu_norm / max(float(mu_norm[inside].max()), 1e-6), 0, 1) if inside.any() else mu_norm,
        np.clip(-sdf_slice / max(1.0, float(-sdf_slice[inside].min() if inside.any() else 1.0)), 0, 1),
    ]
    cmaps = ["hot", "hot", "viridis", "gray"]
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    tag = f"_{suffix}" if suffix else ""
    path = out_dir / f"bulk_field_slice{tag}.png"
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to chkpntNNN.pth")
    parser.add_argument("--phase1_dir", required=True, help="Phase-1 output dir (contains analysis.npz + metadata.json)")
    parser.add_argument("--output", required=True, help="Output directory for TSV and PNGs")
    parser.add_argument("--num_points", type=int, default=200_000, help="Total probe points (split across regions)")
    parser.add_argument("--bulk_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--bulk_atten_multiplier", type=float, default=1.0)
    parser.add_argument("--compact_support_posthoc", action="store_true", default=False,
                        help="Apply (1-q²)² compact taper at render time")
    parser.add_argument("--support_radius_factor", type=float, default=2.0,
                        help="R_i = min(factor*sigma_i, inside_sdf_factor*inside_dist_i)")
    parser.add_argument("--support_inside_sdf_factor", type=float, default=0.8)
    parser.add_argument("--render_preview", action="store_true", default=False)
    parser.add_argument("--tau", type=float, default=0.5, help="Half-space gate tau (match training ct_bulk_halfspace_tau_init)")
    parser.add_argument("--skip_depth", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Device: {device}")

    # ---- load checkpoint ---------------------------------------------------
    chkpnt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {chkpnt_path}")
    state = _load_raw_state(chkpnt_path, device)
    iteration = state["iteration"]
    print(f"  iteration={iteration}  total_gaussians={state['xyz'].shape[0]}"
          f"  bulk={state['bulk_xyz'].shape[0]}")

    bulk_xyz = state["bulk_xyz"]
    bulk_scales = state["bulk_scales"].clone()
    bulk_attenuation = (
        F.softplus(state["bulk_atten_logit"]).clone()
        if state["bulk_atten_logit"] is not None
        else torch.ones(bulk_xyz.shape[0], device=device)
    )

    # ---- apply post-hoc multipliers ----------------------------------------
    if args.bulk_scale_multiplier != 1.0:
        bulk_scales = bulk_scales * args.bulk_scale_multiplier
        print(f"  [post-hoc] scale × {args.bulk_scale_multiplier}")
    if args.bulk_atten_multiplier != 1.0:
        bulk_attenuation = bulk_attenuation * args.bulk_atten_multiplier
        print(f"  [post-hoc] attenuation × {args.bulk_atten_multiplier}")
    compact_posthoc = args.compact_support_posthoc
    if compact_posthoc:
        print(f"  [post-hoc] compact support  radius_factor={args.support_radius_factor}"
              f"  inside_sdf_factor={args.support_inside_sdf_factor}")

    # ---- load SDF ----------------------------------------------------------
    phase1_dir = Path(args.phase1_dir)
    print(f"Loading phase1 bundle: {phase1_dir}")
    analysis, metadata = _load_ct_analysis_bundle_with_meta(phase1_dir)
    spacing_zyx = tuple(float(v) for v in metadata["spacing_zyx"])
    sdf_field = _prepare_signed_distance_field(analysis, spacing_zyx, device=device)
    sdf_native = sdf_field["signed_distance_native"]           # (D,H,W) numpy float32
    print(f"  volume shape={sdf_native.shape}  spacing={spacing_zyx}")

    # ---- sample SDF at bulk Gaussian centres --------------------------------
    from ct_pipeline.training.losses import sample_volume_field, sample_sdf_normals
    sdf_tensor = sdf_field["signed_distance"]  # (1,1,D,H,W)
    bulk_center_sdf = sample_volume_field(sdf_tensor, bulk_xyz, spacing_zyx).reshape(-1).to(device=device, dtype=torch.float32)
    if "sdf_normal" in sdf_field:
        bulk_center_normals = sample_volume_field(sdf_field["sdf_normal"], bulk_xyz, spacing_zyx).to(device=device, dtype=torch.float32)
    else:
        bulk_center_normals = sample_sdf_normals(sdf_tensor, bulk_xyz, spacing_zyx).to(device=device, dtype=torch.float32)
    bulk_center_normals = F.normalize(bulk_center_normals, dim=-1, eps=1e-6)

    # ---- compute per-Gaussian support radius (used by compact_support) ------
    support_radius: torch.Tensor | None = None
    if compact_posthoc:
        support_radius = _compute_support_radius(
            bulk_scales, bulk_center_sdf,
            support_radius_factor=args.support_radius_factor,
            support_inside_sdf_factor=args.support_inside_sdf_factor,
        )
        sr = support_radius
        print(f"  support_radius  p10={_pct(sr,10):.4f}  p50={_pct(sr,50):.4f}  p90={_pct(sr,90):.4f}")

    # ---- print bulk Gaussian parameter stats --------------------------------
    sigma_vals = bulk_scales.mean(dim=1)
    print(f"\nBulk Gaussian stats (N={bulk_xyz.shape[0]}):")
    print(f"  sigma  p10={_pct(sigma_vals,10):.4f}  p50={_pct(sigma_vals,50):.4f}  p90={_pct(sigma_vals,90):.4f}")
    print(f"  atten  p10={_pct(bulk_attenuation,10):.4f}  p50={_pct(bulk_attenuation,50):.4f}  p90={_pct(bulk_attenuation,90):.4f}  mean={bulk_attenuation.mean():.4f}")
    print(f"  center_sdf  p10={_pct(bulk_center_sdf,10):.3f}  p50={_pct(bulk_center_sdf,50):.3f}  p90={_pct(bulk_center_sdf,90):.3f}")

    # ---- build KD-tree once for efficient radius search --------------------
    print("\nBuilding KD-tree on bulk centres ...")
    from scipy.spatial import cKDTree
    bulk_kdtree = cKDTree(bulk_xyz.detach().cpu().numpy())
    sigma_max = float(bulk_scales.mean(dim=1).clamp_min(1e-6).max().item())
    if support_radius is not None:
        _search_radius = float(support_radius.max().item()) * 1.01
    else:
        _search_radius = 3.0 * sigma_max
    print(f"  search_radius={_search_radius:.4f}  sigma_max={sigma_max:.4f}")

    # ---- sample probe points by region -------------------------------------
    rng = np.random.default_rng(42)
    n_per = max(1000, args.num_points // len(REGIONS))

    rows = []
    with torch.no_grad():
        for rname, (smin, smax) in REGIONS.items():
            print(f"\nProbing region '{rname}' (sdf ∈ [{smin}, {smax}))  ...")
            row = probe_region(
                region_name=rname,
                sdf_min=smin, sdf_max=smax,
                sdf_native=sdf_native, spacing_zyx=spacing_zyx, device=device,
                bulk_xyz=bulk_xyz, bulk_scales=bulk_scales, bulk_attenuation=bulk_attenuation,
                bulk_center_sdf=bulk_center_sdf, bulk_center_normals=bulk_center_normals,
                n_per_region=n_per, tau=args.tau, skip_depth=args.skip_depth, rng=rng,
                compact_support=compact_posthoc, support_radius=support_radius,
                kdtree=bulk_kdtree, search_radius=_search_radius,
            )
            n = row.get("n", 0)
            if n == 0:
                print(f"  no points in this region")
                rows.append(row)
                continue
            print(f"  n={n}")
            print(f"  W_b   p10={row.get('W_b_p10',float('nan')):.4f}  p50={row.get('W_b_p50',float('nan')):.4f}  p90={row.get('W_b_p90',float('nan')):.4f}")
            print(f"  mu_raw p10={row.get('mu_raw_p10',float('nan')):.4f}  p50={row.get('mu_raw_p50',float('nan')):.4f}  p90={row.get('mu_raw_p90',float('nan')):.4f}")
            print(f"  mu_norm p10={row.get('mu_norm_p10',float('nan')):.4f}  p50={row.get('mu_norm_p50',float('nan')):.4f}  p90={row.get('mu_norm_p90',float('nan')):.4f}")
            rows.append(row)

    # ---- write TSV ---------------------------------------------------------
    tsv_path = out_dir / "bulk_probe_summary.tsv"
    write_tsv(rows, tsv_path)
    print(f"\nWrote TSV: {tsv_path}")

    # ---- PNG panel ---------------------------------------------------------
    if args.render_preview:
        print("\nRendering slice PNG ...")
        D = sdf_native.shape[0]
        z_slice = D // 2
        sdf_slice = sdf_native[z_slice]
        with torch.no_grad():
            W_b_img, mu_raw_img, mu_norm_img = _render_single_slice(
                sdf_native, spacing_zyx, device,
                bulk_xyz, bulk_scales, bulk_attenuation,
                bulk_center_sdf, bulk_center_normals,
                tau=args.tau, skip_depth=args.skip_depth,
                compact_support=compact_posthoc, support_radius=support_radius,
            )
        tag = ""
        if args.bulk_scale_multiplier != 1.0:
            tag += f"scale_x{args.bulk_scale_multiplier}"
        if args.bulk_atten_multiplier != 1.0:
            tag += f"atten_x{args.bulk_atten_multiplier}"
        if compact_posthoc:
            tag += f"_compact_r{args.support_radius_factor}_s{args.support_inside_sdf_factor}"
        save_png_panel(W_b_img, mu_raw_img, mu_norm_img, sdf_slice, out_dir, suffix=tag)

    print("\nDone.")
    return 0


# ---------------------------------------------------------------------------
# phase1 bundle loader (analysis + metadata together)
# ---------------------------------------------------------------------------

def _load_ct_analysis_bundle_with_meta(phase1_dir: Path) -> tuple[dict, dict]:
    analysis_path = phase1_dir / "analysis.npz"
    metadata_path = phase1_dir / "metadata.json"
    if not analysis_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Phase1 dir must contain analysis.npz and metadata.json: {phase1_dir}")
    with np.load(str(analysis_path)) as npz:
        analysis = {k: npz[k] for k in npz.files}
    with open(str(metadata_path), "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return analysis, metadata


if __name__ == "__main__":
    raise SystemExit(main())
