from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from utils.rotation_utils import quaternion_to_matrix


def default_device(model) -> torch.device:
    override = getattr(model, "_device_override", None)
    if override is not None:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_device(model) -> torch.device:
    for tensor in (
        model._xyz,
        model._scaling,
        model._rotation,
        model._opacity,
        model._primitive_type,
        model._normal,
    ):
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            return tensor.device
    return default_device(model)


def is_initialized(model) -> bool:
    return isinstance(model._xyz, torch.Tensor) and model._xyz.numel() > 0


def default_primitive_type_logits(model, count: int, device: torch.device) -> torch.Tensor:
    return torch.full((int(count), 1), model.nonplanar_logit_value, dtype=torch.float32, device=device)


def sanitize_material_id(model, material_id, count: int, device: torch.device, fill_value: int = -1) -> torch.Tensor:
    del model
    if material_id is None:
        return torch.full((int(count), 1), fill_value, dtype=torch.long, device=device)
    if not isinstance(material_id, torch.Tensor):
        material_id = torch.as_tensor(material_id)
    return material_id.to(device=device, dtype=torch.long).reshape(int(count), 1)


def sanitize_planarity(model, planarity, count: int, device: torch.device, fill_value: float = 0.0) -> torch.Tensor:
    del model
    if planarity is None:
        return torch.full((int(count), 1), fill_value, dtype=torch.float32, device=device)
    if not isinstance(planarity, torch.Tensor):
        planarity = torch.as_tensor(planarity)
    return planarity.to(device=device, dtype=torch.float32).reshape(int(count), 1)


def sanitize_region_type(model, region_type, count: int, device: torch.device, fill_value: int = 0) -> torch.Tensor:
    del model
    if region_type is None:
        return torch.full((int(count), 1), fill_value, dtype=torch.long, device=device)
    if not isinstance(region_type, torch.Tensor):
        region_type = torch.as_tensor(region_type)
    return region_type.to(device=device, dtype=torch.long).reshape(int(count), 1)


def rotation_matrices_from_quaternions(model, rotations: torch.Tensor) -> torch.Tensor:
    if rotations.numel() == 0:
        return torch.empty((0, 3, 3), dtype=rotations.dtype, device=rotations.device)
    return quaternion_to_matrix(model.rotation_activation(rotations))


def derive_normals_from_rotations(model, rotations: torch.Tensor) -> torch.Tensor:
    if rotations.numel() == 0:
        return torch.empty((0, 3), dtype=rotations.dtype, device=rotations.device)
    rotation_matrices = rotation_matrices_from_quaternions(model, rotations)
    return F.normalize(rotation_matrices[:, :, 2], dim=1)


def initialize_hybrid_metadata(
    model,
    count,
    device,
    rotations=None,
    primitive_type_logits=None,
    normals=None,
    material_id=None,
    planarity=None,
    region_type=None,
) -> None:
    count = int(count)
    if primitive_type_logits is None:
        primitive_type_logits = default_primitive_type_logits(model, count, device)
    elif not isinstance(primitive_type_logits, torch.Tensor):
        primitive_type_logits = torch.as_tensor(primitive_type_logits)
    primitive_type_logits = primitive_type_logits.to(device=device, dtype=torch.float32).reshape(count, 1)

    if normals is None:
        if rotations is not None:
            normals = derive_normals_from_rotations(model, rotations)
        else:
            normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device).repeat(count, 1)
    elif not isinstance(normals, torch.Tensor):
        normals = torch.as_tensor(normals)
    normals = F.normalize(normals.to(device=device, dtype=torch.float32).reshape(count, 3), dim=1)

    model._primitive_type = nn.Parameter(primitive_type_logits.requires_grad_(True))
    model._normal = normals.detach().requires_grad_(False)
    model._material_id = sanitize_material_id(model, material_id, count, device)
    model._planarity = sanitize_planarity(model, planarity, count, device)
    model._region_type = sanitize_region_type(model, region_type, count, device)


def freeze_primitive_type_parameter(model) -> None:
    if not isinstance(model._primitive_type, nn.Parameter):
        return
    model._primitive_type.requires_grad_(False)
    if model.optimizer is None:
        return
    for group in model.optimizer.param_groups:
        if group["name"] == "primitive_type":
            group["lr"] = 0.0
            if len(group["params"]) == 1:
                group["params"][0].requires_grad_(False)
            break


def primitive_type_requires_grad(model) -> bool:
    return not model.freeze_primitive_type
