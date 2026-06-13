from __future__ import annotations

import torch
from torch import nn


def _clone_tensor(value):
    if not isinstance(value, torch.Tensor):
        return value
    cloned = value.detach().clone()
    if isinstance(value, nn.Parameter):
        return nn.Parameter(cloned.requires_grad_(value.requires_grad))
    return cloned


def _instantiate_like(model):
    model_type = type(model)
    try:
        if hasattr(model, "max_sh_degree"):
            return model_type(model.max_sh_degree)
        return model_type()
    except TypeError:
        clone = model_type.__new__(model_type)
        if hasattr(clone, "setup_functions"):
            clone.setup_functions()
        return clone


def clone_gaussian_like(model):
    clone = _instantiate_like(model)
    tensor_attrs = [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "_primitive_type",
        "_normal",
        "_material_id",
        "_planarity",
        "_region_type",
        "max_radii2D",
        "xyz_gradient_accum",
        "denom",
    ]
    scalar_attrs = [
        "active_sh_degree",
        "max_sh_degree",
        "percent_dense",
        "spatial_lr_scale",
        "pose_lr_joint",
        "planar_thickness_max",
        "planar_logit_value",
        "nonplanar_logit_value",
        "single_material_fallback",
    ]

    for attr in tensor_attrs:
        if hasattr(model, attr):
            setattr(clone, attr, _clone_tensor(getattr(model, attr)))
    for attr in scalar_attrs:
        if hasattr(model, attr):
            setattr(clone, attr, getattr(model, attr))

    if hasattr(clone, "optimizer"):
        clone.optimizer = None
    return clone


def _prune_gaussian_like(model, keep_mask: torch.Tensor):
    keep_mask = torch.as_tensor(keep_mask, dtype=torch.bool)
    clone = clone_gaussian_like(model)

    for attr in [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "_primitive_type",
        "_normal",
        "_material_id",
        "_planarity",
        "_region_type",
        "max_radii2D",
        "xyz_gradient_accum",
        "denom",
    ]:
        if not hasattr(clone, attr):
            continue
        tensor = getattr(clone, attr)
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 0 or tensor.shape[0] != keep_mask.shape[0]:
            continue
        sliced = tensor[keep_mask.to(device=tensor.device)]
        if isinstance(tensor, nn.Parameter):
            sliced = nn.Parameter(sliced.requires_grad_(tensor.requires_grad))
        setattr(clone, attr, sliced)

    return clone
