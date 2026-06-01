from __future__ import annotations

from pathlib import Path

import torch


def as_device_tensor(value, *, device=None, dtype=None, reshape=None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device if device is not None else tensor.device, dtype=dtype if dtype is not None else tensor.dtype)
    if reshape is not None:
        tensor = tensor.reshape(*reshape)
    return tensor


def format_metric_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_key_value_report(path, entries) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {format_metric_value(value)}" for key, value in entries]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
