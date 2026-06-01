from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from torch import nn

from scene.gaussian_model_metadata import initialize_hybrid_metadata
from utils.system_utils import mkdir_p


def capture_state(model):
    return (
        model.active_sh_degree,
        model._xyz,
        model._features_dc,
        model._features_rest,
        model._scaling,
        model._rotation,
        model._opacity,
        model._primitive_type,
        model._normal,
        model._material_id,
        model._planarity,
        model._region_type,
        model._ct_value_logit,
        model._atten_logit,
        model.max_radii2D,
        model.xyz_gradient_accum,
        model.denom,
        None if model.optimizer is None else model.optimizer.state_dict(),
        model.spatial_lr_scale,
        model.planar_thickness_max,
    )


def restore_state(model, model_args, training_args=None):
    if len(model_args) not in (18, 19, 20):
        raise ValueError(
            "Unsupported checkpoint payload for GaussianModel.restore(). "
            "This repository only supports checkpoints written by the current CTGS pipeline."
        )
    if len(model_args) == 20:
        (
            model.active_sh_degree,
            model._xyz,
            model._features_dc,
            model._features_rest,
            model._scaling,
            model._rotation,
            model._opacity,
            model._primitive_type,
            model._normal,
            model._material_id,
            model._planarity,
            model._region_type,
            model._ct_value_logit,
            model._atten_logit,
            model.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            model.spatial_lr_scale,
            model.planar_thickness_max,
        ) = model_args
    elif len(model_args) == 19:
        (
            model.active_sh_degree,
            model._xyz,
            model._features_dc,
            model._features_rest,
            model._scaling,
            model._rotation,
            model._opacity,
            model._primitive_type,
            model._normal,
            model._material_id,
            model._planarity,
            model._region_type,
            model._ct_value_logit,
            model.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            model.spatial_lr_scale,
            model.planar_thickness_max,
        ) = model_args
        atten_init = torch.clamp(
            torch.sigmoid(model._opacity.detach()) * torch.sigmoid(model._ct_value_logit.detach()),
            min=1e-6,
        )
        model._atten_logit = nn.Parameter(model._inverse_softplus(atten_init).requires_grad_(True))
    else:
        (
            model.active_sh_degree,
            model._xyz,
            model._features_dc,
            model._features_rest,
            model._scaling,
            model._rotation,
            model._opacity,
            model._primitive_type,
            model._normal,
            model._material_id,
            model._planarity,
            model._region_type,
            model.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            model.spatial_lr_scale,
            model.planar_thickness_max,
        ) = model_args
        model._ct_value_logit = nn.Parameter(
            torch.empty((0, 1), dtype=torch.float32, device=model._xyz.device).requires_grad_(True)
        )
        atten_init = torch.clamp(torch.sigmoid(model._opacity.detach()), min=1e-6)
        model._atten_logit = nn.Parameter(model._inverse_softplus(atten_init).requires_grad_(True))
    model.surface_thickness_max = model.planar_thickness_max
    model._normal = F.normalize(
        torch.as_tensor(model._normal, device=model._xyz.device, dtype=torch.float32).reshape(-1, 3),
        dim=1,
        eps=1e-8,
    ).detach().requires_grad_(False)

    if training_args is not None:
        model.training_setup(training_args)
        if opt_dict is not None:
            try:
                model.optimizer.load_state_dict(opt_dict)
            except ValueError:
                print("Warning: optimizer state is incompatible with the current GaussianModel layout. Continuing with a fresh optimizer.")
        if model.freeze_primitive_type:
            model._freeze_primitive_type_parameter()

    model.xyz_gradient_accum = xyz_gradient_accum
    model.denom = denom


def construct_list_of_attributes(model):
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for index in range(model._features_dc.shape[1] * model._features_dc.shape[2]):
        attributes.append(f"f_dc_{index}")
    for index in range(model._features_rest.shape[1] * model._features_rest.shape[2]):
        attributes.append(f"f_rest_{index}")
    attributes.append("opacity")
    for index in range(model._scaling.shape[1]):
        attributes.append(f"scale_{index}")
    for index in range(model._rotation.shape[1]):
        attributes.append(f"rot_{index}")
    attributes.append("primitive_type")
    for index in range(3):
        attributes.append(f"normal_{index}")
    attributes.append("material_id")
    attributes.append("planarity")
    attributes.append("region_type")
    if model._ct_value_logit.numel() > 0:
        attributes.append("ct_value_logit")
    if model._atten_logit.numel() > 0:
        attributes.append("atten_logit")
    return attributes


def save_ply_file(model, path):
    mkdir_p(os.path.dirname(path))
    xyz = model._xyz.detach().cpu().numpy()
    normals = model.get_normals().detach().cpu().numpy()
    f_dc = model._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = model._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = model._opacity.detach().cpu().numpy()
    scale = model._scaling.detach().cpu().numpy()
    rotation = model._rotation.detach().cpu().numpy()
    primitive_type = model._primitive_type.detach().cpu().numpy()
    explicit_normals = F.normalize(model._normal.detach(), dim=1).cpu().numpy()
    material_id = model._material_id.detach().cpu().numpy().astype(np.float32)
    planarity = model._planarity.detach().cpu().numpy()
    region_type = model._region_type.detach().cpu().numpy().astype(np.float32)
    ct_value_logit = (
        model._ct_value_logit.detach().cpu().numpy().astype(np.float32)
        if model._ct_value_logit.numel() > 0
        else None
    )
    atten_logit = (
        model._atten_logit.detach().cpu().numpy().astype(np.float32)
        if model._atten_logit.numel() > 0
        else None
    )

    attribute_names = construct_list_of_attributes(model)
    dtype_full = [(attribute, "f4") for attribute in attribute_names]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    parts = [xyz, normals, f_dc, f_rest, opacities, scale, rotation, primitive_type, explicit_normals, material_id, planarity, region_type]
    if ct_value_logit is not None:
        parts.append(ct_value_logit)
    if atten_logit is not None:
        parts.append(atten_logit)
    attributes = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if attributes.shape[1] != len(attribute_names):
        raise RuntimeError(
            f"PLY attribute mismatch: got {attributes.shape[1]} columns for {len(attribute_names)} attributes."
        )
    for column_index, attribute in enumerate(attribute_names):
        elements[attribute] = attributes[:, column_index]
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def load_ply_file(model, path):
    plydata = PlyData.read(path)
    properties = plydata.elements[0].data.dtype.names

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
    for channel in range(3):
        features_dc[:, channel, 0] = np.asarray(plydata.elements[0][f"f_dc_{channel}"])

    extra_f_names = sorted(
        [name for name in properties if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if len(extra_f_names) % 3 != 0:
        raise ValueError("Invalid PLY feature_rest layout.")
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for index, attr_name in enumerate(extra_f_names):
        features_extra[:, index] = np.asarray(plydata.elements[0][attr_name])
    coeffs_minus_dc = len(extra_f_names) // 3
    features_extra = features_extra.reshape((xyz.shape[0], 3, coeffs_minus_dc))

    if coeffs_minus_dc > 0:
        inferred_degree = int(round(np.sqrt(coeffs_minus_dc + 1) - 1))
        if (inferred_degree + 1) ** 2 - 1 == coeffs_minus_dc:
            model.max_sh_degree = max(model.max_sh_degree, inferred_degree)

    scale_names = sorted(
        [name for name in properties if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for index, attr_name in enumerate(scale_names):
        scales[:, index] = np.asarray(plydata.elements[0][attr_name])

    rot_names = sorted(
        [name for name in properties if name.startswith("rot_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for index, attr_name in enumerate(rot_names):
        rots[:, index] = np.asarray(plydata.elements[0][attr_name])

    required_ct_properties = {"primitive_type", "material_id", "planarity", "region_type"}
    missing = sorted(required_ct_properties.difference(properties))
    if missing:
        raise ValueError(
            "PLY payload is missing required CTGS metadata fields: " + ", ".join(missing)
        )

    primitive_type = np.asarray(plydata.elements[0]["primitive_type"])[..., np.newaxis]
    normal_names = sorted(
        [name for name in properties if name.startswith("normal_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if len(normal_names) != 3:
        raise ValueError("PLY payload is missing required explicit normal_* fields.")
    normals = np.stack([np.asarray(plydata.elements[0][name]) for name in normal_names], axis=1)
    material_id = np.asarray(plydata.elements[0]["material_id"])[..., np.newaxis]
    planarity = np.asarray(plydata.elements[0]["planarity"])[..., np.newaxis]
    region_type = np.asarray(plydata.elements[0]["region_type"])[..., np.newaxis]

    device = model._default_device()
    model._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device).requires_grad_(True))
    model._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float32, device=device).transpose(1, 2).contiguous().requires_grad_(True))
    model._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float32, device=device).transpose(1, 2).contiguous().requires_grad_(True))
    model._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=device).requires_grad_(True))
    model._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device).requires_grad_(True))
    model._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device=device).requires_grad_(True))
    initialize_hybrid_metadata(
        model,
        xyz.shape[0],
        device,
        rotations=model._rotation.detach(),
        primitive_type_logits=torch.tensor(primitive_type, dtype=torch.float32, device=device),
        normals=torch.tensor(normals, dtype=torch.float32, device=device),
        material_id=torch.tensor(material_id, dtype=torch.long, device=device),
        planarity=torch.tensor(planarity, dtype=torch.float32, device=device),
        region_type=torch.tensor(region_type, dtype=torch.long, device=device),
    )
    model.max_radii2D = torch.zeros((model.get_xyz.shape[0],), device=device)
    model.xyz_gradient_accum = torch.zeros((model.get_xyz.shape[0], 1), device=device)
    model.denom = torch.zeros((model.get_xyz.shape[0], 1), device=device)
    model.active_sh_degree = model.max_sh_degree
    if "ct_value_logit" in properties:
        ct_value_logit = np.asarray(plydata.elements[0]["ct_value_logit"])[..., np.newaxis]
        model._ct_value_logit = nn.Parameter(
            torch.tensor(ct_value_logit, dtype=torch.float32, device=device).requires_grad_(True)
        )
    else:
        model._ct_value_logit = nn.Parameter(
            torch.empty((0, 1), dtype=torch.float32, device=device).requires_grad_(True)
        )
    if "atten_logit" in properties:
        atten_logit = np.asarray(plydata.elements[0]["atten_logit"])[..., np.newaxis]
        model._atten_logit = nn.Parameter(
            torch.tensor(atten_logit, dtype=torch.float32, device=device).requires_grad_(True)
        )
    else:
        if model._ct_value_logit.numel() > 0:
            atten_init = torch.clamp(torch.sigmoid(model._opacity.detach()) * torch.sigmoid(model._ct_value_logit.detach()), min=1e-6)
        else:
            atten_init = torch.clamp(torch.sigmoid(model._opacity.detach()), min=1e-6)
        model._atten_logit = nn.Parameter(model._inverse_softplus(atten_init).requires_grad_(True))
