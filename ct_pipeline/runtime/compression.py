from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .acceleration import _prune_gaussian_like, clone_gaussian_like


def _tensor_to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _choose_uint_dtype(bits: int):
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    raise ValueError("bits must be <= 32.")


def _uniform_quantize(array: np.ndarray, bits: int):
    array = np.asarray(array, dtype=np.float32)
    dtype = _choose_uint_dtype(bits)
    levels = (1 << int(bits)) - 1
    if array.size == 0:
        return {
            "data": array.astype(dtype),
            "min": np.array(0.0, dtype=np.float32),
            "max": np.array(0.0, dtype=np.float32),
            "scale": np.array(1.0, dtype=np.float32),
            "bits": np.array(bits, dtype=np.int32),
        }

    min_value = np.min(array).astype(np.float32)
    max_value = np.max(array).astype(np.float32)
    if float(max_value - min_value) <= 1e-8:
        quantized = np.zeros_like(array, dtype=dtype)
        scale = np.array(1.0, dtype=np.float32)
    else:
        scale = np.array((max_value - min_value) / levels, dtype=np.float32)
        quantized = np.clip(np.round((array - min_value) / scale), 0, levels).astype(dtype)

    return {
        "data": quantized,
        "min": np.array(min_value, dtype=np.float32),
        "max": np.array(max_value, dtype=np.float32),
        "scale": scale,
        "bits": np.array(bits, dtype=np.int32),
    }


def _run_kmeans(data: np.ndarray, n_clusters: int, max_iter: int = 25, seed: int = 0):
    if data.ndim != 2:
        raise ValueError("K-means input must have shape (N, D).")
    if data.shape[0] == 0:
        return np.zeros((0, data.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    rng = np.random.default_rng(seed)
    cluster_count = max(1, min(int(n_clusters), data.shape[0]))
    initial_indices = rng.choice(data.shape[0], size=cluster_count, replace=False)
    centers = data[initial_indices].astype(np.float32, copy=True)

    for _ in range(max_iter):
        distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for cluster_index in range(cluster_count):
            mask = assignments == cluster_index
            if np.any(mask):
                new_centers[cluster_index] = data[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assignments = np.argmin(distances, axis=1).astype(np.int32)
    return centers.astype(np.float32), assignments


class GSCompressor:
    def __init__(self):
        self.latest_quantized = None
        self.latest_codebook = None

    def quantize_attributes(self, model, bits=8):
        quantized = {
            "sh": _uniform_quantize(_tensor_to_numpy(model.get_features), bits),
            "opacity": _uniform_quantize(_tensor_to_numpy(model.get_opacity), bits),
            "scaling": _uniform_quantize(_tensor_to_numpy(model.get_scaling), bits),
        }
        quantized["bits"] = int(bits)
        self.latest_quantized = quantized
        return quantized

    def prune_low_contribution(self, model, threshold=0.01):
        contribution = model.get_opacity.squeeze(-1) * model.get_scaling.max(dim=1).values
        keep_mask = contribution >= float(threshold)
        if int(keep_mask.sum().item()) == 0 and contribution.numel() > 0:
            keep_mask[contribution.argmax()] = True
        return _prune_gaussian_like(model, keep_mask)

    def codebook_compress(self, model, n_clusters=256):
        sh = _tensor_to_numpy(model.get_features).reshape(model.get_features.shape[0], -1).astype(np.float32, copy=False)
        centers, indices = _run_kmeans(sh, n_clusters=n_clusters)
        compressed = {
            "centers": centers,
            "indices": indices.astype(np.int32),
            "n_clusters": int(centers.shape[0]),
        }
        self.latest_codebook = compressed
        return compressed

    def save_compressed(self, model, path):
        export_path = Path(path)
        if export_path.suffix == "":
            export_path = export_path.with_suffix(".npz")
        export_path.parent.mkdir(parents=True, exist_ok=True)

        quantized = self.latest_quantized if self.latest_quantized is not None else self.quantize_attributes(model, bits=8)
        codebook = self.latest_codebook if self.latest_codebook is not None else self.codebook_compress(model, n_clusters=256)

        payload = {
            "xyz": _tensor_to_numpy(model.get_xyz).astype(np.float32),
            "rotation": _tensor_to_numpy(model.get_rotation).astype(np.float32),
            "normal": _tensor_to_numpy(model.get_normals()).astype(np.float32),
            "primitive_type": _tensor_to_numpy(getattr(model, "_primitive_type")).astype(np.float32),
            "material_id": _tensor_to_numpy(model.get_material_id).astype(np.int32),
            "planarity": _tensor_to_numpy(model.get_planarity).astype(np.float32),
            "region_type": _tensor_to_numpy(model.get_region_type).astype(np.int32),
            "quantized_sh": quantized["sh"]["data"],
            "quantized_opacity": quantized["opacity"]["data"],
            "quantized_scaling": quantized["scaling"]["data"],
            "sh_min": quantized["sh"]["min"],
            "sh_max": quantized["sh"]["max"],
            "opacity_min": quantized["opacity"]["min"],
            "opacity_max": quantized["opacity"]["max"],
            "scaling_min": quantized["scaling"]["min"],
            "scaling_max": quantized["scaling"]["max"],
            "sh_scale": quantized["sh"]["scale"],
            "opacity_scale": quantized["opacity"]["scale"],
            "scaling_scale": quantized["scaling"]["scale"],
            "codebook_centers": codebook["centers"],
            "codebook_indices": codebook["indices"],
            "metadata_json": np.array(
                json.dumps(
                    {
                        "format": "ct_gs_compressed_v1",
                        "quantization_bits": int(quantized["bits"]),
                        "num_gaussians": int(model.get_xyz.shape[0]),
                        "codebook_clusters": int(codebook["n_clusters"]),
                    }
                )
            ),
        }
        np.savez_compressed(str(export_path), **payload)
        return export_path
