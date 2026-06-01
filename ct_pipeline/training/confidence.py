"""Static artifact-aware confidence maps for CTGS role-separated training.

Splits the CT volume into three regions:
  mat_conf  — high-confidence material (large connected components above T_mat)
  air_conf  — high-confidence air/void (large connected components below T_air)
  unknown   — uncertain boundary band / artifact / partial-volume

These maps drive two loss weights:
  int_weight — confidence weight for bulk intensity fitting
  geo_weight — confidence weight for surface phase loss
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


# ---------------------------------------------------------------------------
# Threshold estimation helpers
# ---------------------------------------------------------------------------

def _otsu_two_class(volume_np: np.ndarray) -> tuple[float, float]:
    """Return (low_threshold, high_threshold) via two-class Otsu on a flat histogram."""
    from skimage.filters import threshold_otsu
    flat = volume_np.ravel().astype(np.float32)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.25, 0.75
    t = float(threshold_otsu(finite))
    return t * 0.5, t


def _percentile_thresholds(
    volume_np: np.ndarray,
    air_percentile: float = 90.0,
    mat_percentile: float = 10.0,
) -> tuple[float, float]:
    """Estimate air/material thresholds from the lower and upper tails."""
    flat = volume_np.ravel().astype(np.float32)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.2, 0.6
    # Air: upper-tail of the low-intensity cluster
    low_half = finite[finite <= np.median(finite)]
    t_air = float(np.percentile(low_half, air_percentile)) if low_half.size > 0 else 0.2
    # Material: lower-tail of the high-intensity cluster
    high_half = finite[finite > np.median(finite)]
    t_mat = float(np.percentile(high_half, mat_percentile)) if high_half.size > 0 else 0.6
    return t_air, t_mat


# ---------------------------------------------------------------------------
# Connected-component filter
# ---------------------------------------------------------------------------

def _keep_large_components(
    mask: np.ndarray,
    min_size: int,
    connectivity: int = 1,
) -> np.ndarray:
    """Keep only 3-D connected components with voxel count >= min_size."""
    if not np.any(mask):
        return mask
    structure = ndimage.generate_binary_structure(3, connectivity)
    labeled, n = ndimage.label(mask.astype(bool), structure=structure)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = ndimage.sum(np.ones_like(labeled, dtype=np.int64), labeled, range(1, n + 1))
    keep_labels = np.where(np.asarray(counts) >= int(min_size))[0] + 1
    if keep_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)
    result = np.isin(labeled, keep_labels)
    return result.astype(bool)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def build_ct_confidence_maps(
    volume_np: np.ndarray,
    spacing_zyx,
    args=None,
    analysis: dict | None = None,
) -> dict:
    """Compute static confidence maps from the CT intensity volume.

    Parameters
    ----------
    volume_np : (D, H, W) float32 array, normalised to [0, 1]
    spacing_zyx : voxel spacing tuple
    args : argparse namespace with ct_conf_* parameters (optional)
    analysis : phase1 analysis dict (optional, used for roi masking)

    Returns
    -------
    dict with keys:
        mat_conf         : (D,H,W) bool  — high-confidence material
        air_conf         : (D,H,W) bool  — high-confidence air/void
        unknown          : (D,H,W) bool  — uncertain region
        int_weight       : (D,H,W) float32 — confidence weight for intensity loss
        geo_weight       : (D,H,W) float32 — confidence weight for surface phase loss
        mu_air           : float — estimated air intensity mean
        t_air            : float — used air threshold
        t_mat            : float — used material threshold
    """
    volume = np.asarray(volume_np, dtype=np.float32)

    # --- thresholds ---
    mode = "auto_percentile"
    if args is not None:
        mode = str(getattr(args, "ct_confidence_mode", mode))

    if mode == "static_threshold":
        t_air = float(getattr(args, "ct_conf_air_threshold", 0.25))
        t_mat = float(getattr(args, "ct_conf_material_threshold", 0.55))
    else:
        t_air, t_mat = _percentile_thresholds(volume)
        if args is not None:
            t_air = float(getattr(args, "ct_conf_air_threshold", t_air))
            t_mat = float(getattr(args, "ct_conf_material_threshold", t_mat))

    band_width = 0.0 if args is None else float(getattr(args, "ct_conf_unknown_band_width", 0.0))
    t_air_lo = max(0.0, t_air - band_width)
    t_mat_hi = min(1.0, t_mat + band_width)

    mat_raw = volume >= t_mat_hi
    air_raw = volume <= t_air_lo

    # --- connectivity filter ---
    min_size_vox = 500 if args is None else int(getattr(args, "ct_conf_min_component_size", 500))
    mat_conf = _keep_large_components(mat_raw, min_size_vox)
    air_conf = _keep_large_components(air_raw, min_size_vox)

    # --- apply roi mask if available ---
    if analysis is not None:
        roi = analysis.get("coarse_support_mask", analysis.get("material_mask"))
        if roi is not None:
            roi_np = np.asarray(roi, dtype=bool)
            if roi_np.shape == volume.shape:
                # material must be inside roi; exterior air already handled
                mat_conf = mat_conf & roi_np

    unknown = ~(mat_conf | air_conf)

    # --- intensity confidence weight ---
    # high weight on confident regions, low on unknown
    int_weight = np.where(mat_conf | air_conf, 1.0, 0.1).astype(np.float32)

    # --- geometry confidence weight ---
    # only phase boundary between mat_conf and air_conf drives surface
    structure = ndimage.generate_binary_structure(3, 1)
    mat_dilated = ndimage.binary_dilation(mat_conf, structure=structure, iterations=2)
    air_dilated = ndimage.binary_dilation(air_conf, structure=structure, iterations=2)
    geo_active = mat_dilated & air_dilated  # near-boundary only
    geo_weight = np.where(geo_active, 1.0, 0.0).astype(np.float32)

    # --- air intensity estimate ---
    air_vals = volume[air_conf] if np.any(air_conf) else volume[volume < t_air]
    mu_air = float(np.mean(air_vals)) if air_vals.size > 0 else 0.0

    return {
        "mat_conf": mat_conf,
        "air_conf": air_conf,
        "unknown": unknown,
        "int_weight": int_weight,
        "geo_weight": geo_weight,
        "mu_air": mu_air,
        "t_air": t_air,
        "t_mat": t_mat,
    }


def confidence_maps_to_gpu(maps: dict, device="cuda") -> dict:
    """Move numpy arrays in the confidence maps dict to GPU tensors."""
    out = {}
    for k, v in maps.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.as_tensor(v, device=device)
        else:
            out[k] = v
    return out


def sample_confidence_at_points(
    conf_volume: torch.Tensor,
    points_xyz: torch.Tensor,
    spacing_zyx,
) -> torch.Tensor:
    """Sample a (D,H,W) confidence volume at world-space points.

    Returns a (N,) float32 tensor of sampled confidence weights.
    """
    from ct_pipeline.training.losses import sample_volume_field
    vol = conf_volume.to(dtype=torch.float32)
    if vol.ndim == 3:
        vol = vol.unsqueeze(0).unsqueeze(0)
    return sample_volume_field(vol, points_xyz, spacing_zyx).reshape(-1).to(dtype=torch.float32)
