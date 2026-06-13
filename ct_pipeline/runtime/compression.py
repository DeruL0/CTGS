from __future__ import annotations

import numpy as np
import torch


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


def quantize_attributes(model, bits=8):
    return {
        "sh": _uniform_quantize(_tensor_to_numpy(model.get_features), bits),
        "opacity": _uniform_quantize(_tensor_to_numpy(model.get_opacity), bits),
        "scaling": _uniform_quantize(_tensor_to_numpy(model.get_scaling), bits),
        "bits": int(bits),
    }
