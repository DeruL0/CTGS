from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ct_pipeline.data import CTVolumeLoader
from scene import CTGaussianModel


@dataclass
class CTTrainingBootstrap:
    first_iter: int
    tb_writer: object | None
    loader: CTVolumeLoader
    volume_np: np.ndarray
    volume_cuda: torch.Tensor
    volume_shape: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]
    analysis: dict
    analysis_gpu: dict
    analysis_path: str
    metadata_path: str
    gaussians: CTGaussianModel
    renderer_autocast_kwargs: dict
    field_pools: dict
    support_sample_count: int
    air_sample_count: int
    support_distance_field: dict
    signed_distance_field: dict
    intensity_field_cache: dict
    intensity_air: float
    intensity_mat: float
    initial_gaussian_count: int
    preferred_air_candidates: torch.Tensor
    exterior_air_sample_ratio: float
