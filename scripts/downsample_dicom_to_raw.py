from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


def _collect_sorted_slices(dicom_dir: Path) -> list[Path]:
    if pydicom is None:
        raise ImportError("pydicom is required for DICOM downsampling.")

    files = sorted(p for p in dicom_dir.iterdir() if p.is_file())
    if not files:
        raise ValueError(f"No files found in {dicom_dir}")

    headers: list[tuple[float, Path]] = []
    for path in files:
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not hasattr(header, "Rows") or not hasattr(header, "Columns"):
            continue
        ipp = getattr(header, "ImagePositionPatient", None)
        if ipp is not None:
            key = float(ipp[2])
        else:
            key = float(getattr(header, "InstanceNumber", 0))
        headers.append((key, path))

    headers.sort(key=lambda item: item[0])
    return [path for _, path in headers]


def _streaming_min_max(paths: list[Path]) -> tuple[float, float, float, float]:
    """Compute global min/max after slope/intercept across all slices."""
    g_min = np.inf
    g_max = -np.inf
    slope = 1.0
    intercept = 0.0
    for path in paths:
        dataset = pydicom.dcmread(str(path), force=True)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        arr = dataset.pixel_array.astype(np.float32) * slope + intercept
        g_min = min(g_min, float(arr.min()))
        g_max = max(g_max, float(arr.max()))
    return g_min, g_max, slope, intercept


def _downsample_2d_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape
    h_trim = (h // factor) * factor
    w_trim = (w // factor) * factor
    arr = arr[:h_trim, :w_trim]
    arr = arr.reshape(h_trim // factor, factor, w_trim // factor, factor).mean(axis=(1, 3))
    return arr.astype(np.float32, copy=False)


def downsample(
    dicom_dir: Path,
    output_raw: Path,
    output_json: Path,
    factor_zyx: tuple[int, int, int],
) -> None:
    fz, fy, fx = factor_zyx
    if pydicom is None:
        raise ImportError("pydicom is required for DICOM downsampling.")

    paths = _collect_sorted_slices(dicom_dir)
    if not paths:
        raise ValueError(f"No DICOM slices found in {dicom_dir}")

    print(f"[INFO] Found {len(paths)} DICOM slices in {dicom_dir}", flush=True)

    reference = pydicom.dcmread(str(paths[0]), force=True)
    rows = int(reference.Rows)
    cols = int(reference.Columns)
    pixel_spacing = getattr(reference, "PixelSpacing", None)
    if pixel_spacing is None:
        pixel_spacing = getattr(reference, "ImagerPixelSpacing", None)
    sy = float(pixel_spacing[0])
    sx = float(pixel_spacing[1])

    positions: list[float] = []
    for path in paths:
        h = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        ipp = getattr(h, "ImagePositionPatient", None)
        if ipp is not None:
            positions.append(float(ipp[2]))
    if len(positions) >= 2:
        diffs = np.diff(np.asarray(positions, dtype=np.float64))
        nonzero = np.abs(diffs[np.abs(diffs) > 1e-6])
        sz = float(np.median(nonzero)) if nonzero.size else float(getattr(reference, "SliceThickness", 1.0))
    else:
        sz = float(getattr(reference, "SpacingBetweenSlices", getattr(reference, "SliceThickness", 1.0)))

    source_shape = (len(paths), rows, cols)
    print(f"[INFO] Source shape (D,H,W) = {source_shape}, spacing (z,y,x) = ({sz}, {sy}, {sx})", flush=True)
    print(f"[INFO] Downsample factor (z,y,x) = ({fz}, {fy}, {fx})", flush=True)

    print(f"[INFO] Pass 1/2: computing global min/max ...", flush=True)
    g_min, g_max, slope, intercept = _streaming_min_max(paths)
    print(f"[INFO] global min = {g_min}, max = {g_max}, rescale_slope = {slope}, intercept = {intercept}", flush=True)
    denominator = max(g_max - g_min, 1e-8)

    n_groups_z = len(paths) // fz
    output_d = n_groups_z
    output_h = (rows // fy)
    output_w = (cols // fx)
    output_shape = (output_d, output_h, output_w)
    print(f"[INFO] Output shape (D,H,W) = {output_shape}", flush=True)

    output_raw.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Pass 2/2: streaming downsample to {output_raw}", flush=True)

    buffer = np.zeros((output_h, output_w), dtype=np.float64)
    written = 0
    with output_raw.open("wb") as handle:
        for zg in range(n_groups_z):
            buffer.fill(0.0)
            for zo in range(fz):
                slice_idx = zg * fz + zo
                dataset = pydicom.dcmread(str(paths[slice_idx]), force=True)
                arr = dataset.pixel_array.astype(np.float32) * slope + intercept
                arr = (arr - g_min) / denominator
                arr = np.clip(arr, 0.0, 1.0)
                pooled = _downsample_2d_mean(arr, fy)
                if pooled.shape != (output_h, output_w):
                    pooled = pooled[:output_h, :output_w]
                buffer += pooled.astype(np.float64)
            avg = (buffer / float(fz)).astype("<f4")
            avg.tofile(handle)
            written += avg.size * 4
            if (zg + 1) % 25 == 0 or zg == n_groups_z - 1:
                print(f"[INFO] wrote slice {zg + 1}/{n_groups_z}", flush=True)

    sidecar = {
        "shape": [int(output_d), int(output_h), int(output_w)],
        "dtype": "float32",
        "endianness": "little",
        "spacing": [float(sz * fz), float(sy * fy), float(sx * fx)],
        "origin": [0.0, 0.0, 0.0],
        "direction": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "scan_params": {
            "source_asset": str(dicom_dir),
            "source_shape_dhw": [int(v) for v in source_shape],
            "downsample_factor_zyx": [int(fz), int(fy), int(fx)],
            "source_normalization": {
                "method": "dicom_rescale_streamed",
                "rescale_slope": float(slope),
                "rescale_intercept": float(intercept),
                "input_min": float(g_min),
                "input_max": float(g_max),
            },
        },
        "normalization": {
            "method": "min_max",
            "input_min": float(g_min),
            "input_max": float(g_max),
            "output_range": [0.0, 1.0],
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"[DONE] raw = {output_raw} ({written / (1024 * 1024):.1f} MiB)", flush=True)
    print(f"[DONE] json = {output_json}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Downsample a DICOM series into a float32 raw + JSON sidecar.")
    parser.add_argument("--input", required=True, help="DICOM series directory.")
    parser.add_argument("--output-raw", required=True, help="Output .raw file path.")
    parser.add_argument("--output-json", required=True, help="Output sidecar .json file path.")
    parser.add_argument("--factor-z", type=int, default=2)
    parser.add_argument("--factor-y", type=int, default=2)
    parser.add_argument("--factor-x", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    downsample(
        Path(args.input),
        Path(args.output_raw),
        Path(args.output_json),
        (int(args.factor_z), int(args.factor_y), int(args.factor_x)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
