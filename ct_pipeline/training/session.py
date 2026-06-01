from __future__ import annotations

import os
import sys
import uuid
import warnings
from argparse import Namespace

import torch

try:
    import nvidia_smi
except ImportError:  # pragma: no cover
    try:
        import pynvml as nvidia_smi
    except ImportError:  # pragma: no cover
        nvidia_smi = None

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:  # pragma: no cover
    TENSORBOARD_FOUND = False


_NVML_INITIALIZED = False
_LPIPS_MODEL_CACHE = {}


def initialize_nvml():
    global _NVML_INITIALIZED
    if nvidia_smi is None:
        return False
    if _NVML_INITIALIZED:
        return True
    try:
        nvidia_smi.nvmlInit()
    except Exception:
        return False
    _NVML_INITIALIZED = True
    return True


def shutdown_nvml():
    global _NVML_INITIALIZED
    if not _NVML_INITIALIZED:
        return
    try:
        nvidia_smi.nvmlShutdown()
    except Exception:
        pass
    _NVML_INITIALIZED = False


def get_lpips_model(device):
    device = torch.device(device)
    cache_key = str(device)
    if cache_key in _LPIPS_MODEL_CACHE:
        return _LPIPS_MODEL_CACHE[cache_key]
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"LPIPS metric disabled during CT preview generation: {exc!r}", RuntimeWarning, stacklevel=2)
        model = None
    _LPIPS_MODEL_CACHE[cache_key] = model
    return model


def save_command(path):
    command = " ".join(sys.argv)
    with open(os.path.join(path, "command.txt"), "a", encoding="utf-8") as handle:
        handle.write(command + "\n")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    disable_tensorboard = os.getenv("CT_DISABLE_TENSORBOARD", "").lower() in {"1", "true", "yes", "on"}
    if disable_tensorboard:
        print("Tensorboard disabled via CT_DISABLE_TENSORBOARD")
        return None
    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: not logging progress")
    return None


def ct_log_gpu_memory():
    if not initialize_nvml():
        return None
    try:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    except Exception:
        return None
    return info.used / (1024 ** 2 * 1000)


def build_renderer_autocast_kwargs() -> dict:
    return {
        "device_type": "cuda",
        "dtype": torch.bfloat16,
        "enabled": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    }
