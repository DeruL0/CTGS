from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from scene.gaussian_model_metadata import freeze_primitive_type_parameter, primitive_type_requires_grad
from utils.general_utils import get_expon_lr_func, inverse_sigmoid


def find_optimizer_group(model, name: str):
    if model.optimizer is None:
        return None
    for group in model.optimizer.param_groups:
        if group["name"] == name:
            return group
    return None


def assign_parameter(model, attr_name: str, tensor, optimizer_name: str | None = None, requires_grad: bool = True):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.to(device=model._device())
    if attr_name == "_normal":
        normal = F.normalize(tensor.to(dtype=torch.float32).reshape(-1, 3), dim=1, eps=1e-8)
        setattr(model, attr_name, normal.detach().requires_grad_(False))
        return getattr(model, attr_name)
    if optimizer_name is not None and model.optimizer is not None:
        optimizable = replace_tensor_to_optimizer(model, tensor, optimizer_name, requires_grad=requires_grad)
        if optimizer_name in optimizable:
            setattr(model, attr_name, optimizable[optimizer_name])
            return optimizable[optimizer_name]
    param = nn.Parameter(tensor.requires_grad_(requires_grad), requires_grad=requires_grad)
    setattr(model, attr_name, param)
    return param


def training_setup(model, training_args):
    model.percent_dense = training_args.percent_dense
    model.freeze_primitive_type = bool(getattr(training_args, "ct_freeze_primitive_type", model.freeze_primitive_type))
    surface_thickness_max = getattr(training_args, "surface_thickness_max", None)
    if surface_thickness_max is not None:
        model.surface_thickness_max = surface_thickness_max
        model.planar_thickness_max = surface_thickness_max
    if getattr(training_args, "planar_thickness_max", None) is not None:
        model.planar_thickness_max = training_args.planar_thickness_max
        model.surface_thickness_max = training_args.planar_thickness_max

    device = model._device()
    model.xyz_gradient_accum = torch.zeros((model.get_xyz.shape[0], 1), device=device)
    model.denom = torch.zeros((model.get_xyz.shape[0], 1), device=device)

    param_groups = [
        (model._xyz, training_args.position_lr_init * model.spatial_lr_scale, "xyz"),
        (model._features_dc, training_args.feature_lr, "f_dc"),
        (model._features_rest, training_args.feature_lr / 20.0, "f_rest"),
        (model._opacity, training_args.opacity_lr, "opacity"),
        (model._scaling, training_args.scaling_lr, "scaling"),
        (model._rotation, training_args.rotation_lr, "rotation"),
        (model._primitive_type, 0.0 if model.freeze_primitive_type else training_args.opacity_lr, "primitive_type"),
    ]
    if model._ct_value_logit.numel() > 0:
        ct_value_lr = float(getattr(training_args, "ct_value_lr", 1e-3))
        param_groups.append((model._ct_value_logit, ct_value_lr, "ct_value"))
    if model._atten_logit.numel() > 0:
        atten_lr = float(getattr(training_args, "ct_attenuation_lr", getattr(training_args, "ct_value_lr", 1e-3)))
        param_groups.append((model._atten_logit, atten_lr, "attenuation"))
    adaptive_mode = str(getattr(training_args, "ct_bulk_adaptive_mode", "fixed"))
    if adaptive_mode in ("scale", "scale_offset"):
        # bulk scale: trained via existing _scaling param, but only for bulk Gaussians
        # We add a separate lr for the bulk scale channel (handled in train loop via mask)
        pass  # gradient is enabled/disabled per-mask in the training loop
    if adaptive_mode == "scale_offset" and model._bulk_offset.numel() > 0:
        offset_lr = atten_lr * 0.05 if model._atten_logit.numel() > 0 else 1e-4
        param_groups.append((model._bulk_offset, offset_lr, "bulk_offset"))

    model._primitive_type.requires_grad_(primitive_type_requires_grad(model))
    optim_params = []
    for parameter, lr, name in param_groups:
        parameter.requires_grad_(lr > 0.0 and parameter.requires_grad)
        optim_params.append({"params": [parameter], "lr": lr, "name": name})
    if isinstance(model._normal, torch.Tensor):
        model._normal = model._normal.detach().requires_grad_(False)

    model.optimizer = torch.optim.Adam(optim_params)
    model.xyz_scheduler_args = get_expon_lr_func(
        lr_init=training_args.position_lr_init * model.spatial_lr_scale,
        lr_final=training_args.position_lr_final * model.spatial_lr_scale,
        lr_delay_mult=training_args.position_lr_delay_mult,
        max_steps=training_args.position_lr_max_steps,
    )

    if model.freeze_primitive_type:
        freeze_primitive_type_parameter(model)


def update_learning_rate(model, iteration):
    if model.optimizer is None:
        return None
    group = find_optimizer_group(model, "xyz")
    if group is None:
        return None
    lr = model.xyz_scheduler_args(iteration)
    group["lr"] = lr
    return lr


def post_optimizer_step(model, iteration):
    del model, iteration
    return


def reset_opacity(model):
    opacities_new = inverse_sigmoid(torch.min(model.get_opacity, torch.ones_like(model.get_opacity) * 0.01))
    optimizable_tensors = replace_tensor_to_optimizer(model, opacities_new, "opacity")
    model._opacity = optimizable_tensors["opacity"]


def replace_tensor_to_optimizer(model, tensor, name, requires_grad=True):
    optimizable_tensors = {}
    group = find_optimizer_group(model, name)
    if group is None:
        return optimizable_tensors
    stored_state = model.optimizer.state.get(group["params"][0], None)
    if stored_state is not None:
        stored_state["exp_avg"] = torch.zeros_like(tensor)
        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
        del model.optimizer.state[group["params"][0]]
    group["params"][0] = nn.Parameter(tensor.requires_grad_(requires_grad), requires_grad=requires_grad)
    if stored_state is not None:
        model.optimizer.state[group["params"][0]] = stored_state
    optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def prune_optimizer(model, mask):
    optimizable_tensors = {}
    if model.optimizer is None:
        return optimizable_tensors
    for group in model.optimizer.param_groups:
        requires_grad = group["params"][0].requires_grad
        stored_state = model.optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
            del model.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(requires_grad), requires_grad=requires_grad)
            model.optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(requires_grad), requires_grad=requires_grad)
        optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def prune_points(model, mask):
    valid_points_mask = ~mask
    optimizable_tensors = prune_optimizer(model, valid_points_mask)
    if model.optimizer is not None:
        model._xyz = optimizable_tensors["xyz"]
        model._features_dc = optimizable_tensors["f_dc"]
        model._features_rest = optimizable_tensors["f_rest"]
        model._opacity = optimizable_tensors["opacity"]
        model._scaling = optimizable_tensors["scaling"]
        model._rotation = optimizable_tensors["rotation"]
        model._primitive_type = optimizable_tensors["primitive_type"]
        if "ct_value" in optimizable_tensors:
            model._ct_value_logit = optimizable_tensors["ct_value"]
        if "attenuation" in optimizable_tensors:
            model._atten_logit = optimizable_tensors["attenuation"]
    else:
        model._xyz = nn.Parameter(model._xyz[valid_points_mask].requires_grad_(True))
        model._features_dc = nn.Parameter(model._features_dc[valid_points_mask].requires_grad_(True))
        model._features_rest = nn.Parameter(model._features_rest[valid_points_mask].requires_grad_(True))
        model._opacity = nn.Parameter(model._opacity[valid_points_mask].requires_grad_(True))
        model._scaling = nn.Parameter(model._scaling[valid_points_mask].requires_grad_(True))
        model._rotation = nn.Parameter(model._rotation[valid_points_mask].requires_grad_(True))
        primitive_requires_grad = primitive_type_requires_grad(model)
        model._primitive_type = nn.Parameter(
            model._primitive_type[valid_points_mask].requires_grad_(primitive_requires_grad),
            requires_grad=primitive_requires_grad,
        )
        if model._ct_value_logit.numel() > 0:
            model._ct_value_logit = nn.Parameter(model._ct_value_logit[valid_points_mask].requires_grad_(True))
        if model._atten_logit.numel() > 0:
            model._atten_logit = nn.Parameter(model._atten_logit[valid_points_mask].requires_grad_(True))

    model._normal = model._normal[valid_points_mask].detach().requires_grad_(False)
    model._material_id = model._material_id[valid_points_mask]
    model._planarity = model._planarity[valid_points_mask]
    model._region_type = model._region_type[valid_points_mask]
    model.xyz_gradient_accum = model.xyz_gradient_accum[valid_points_mask]
    model.denom = model.denom[valid_points_mask]
    model.max_radii2D = model.max_radii2D[valid_points_mask]


def cat_tensors_to_optimizer(model, tensors_dict):
    optimizable_tensors = {}
    if model.optimizer is None:
        return optimizable_tensors
    for group in model.optimizer.param_groups:
        if group["name"] not in tensors_dict:
            continue
        extension_tensor = tensors_dict[group["name"]]
        requires_grad = group["params"][0].requires_grad
        stored_state = model.optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
            del model.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(requires_grad),
                requires_grad=requires_grad,
            )
            model.optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(requires_grad),
                requires_grad=requires_grad,
            )
        optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def densification_postfix(
    model,
    new_xyz,
    new_features_dc,
    new_features_rest,
    new_opacities,
    new_scaling,
    new_rotation,
    new_primitive_type,
    new_normal,
    new_material_id,
    new_planarity,
    new_region_type,
    new_ct_value_logit=None,
    new_atten_logit=None,
):
    if new_ct_value_logit is None and model._ct_value_logit.numel() > 0:
        # If caller didn't provide ct_value_logit, default to logit(0.5) for new entries.
        new_count = int(new_xyz.shape[0])
        new_ct_value_logit = torch.zeros(
            (new_count, 1), dtype=model._ct_value_logit.dtype, device=model._ct_value_logit.device
        )
    if new_atten_logit is None and model._atten_logit.numel() > 0:
        new_count = int(new_xyz.shape[0])
        new_atten_logit = torch.zeros(
            (new_count, 1), dtype=model._atten_logit.dtype, device=model._atten_logit.device
        )
    tensors = {
        "xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling": new_scaling,
        "rotation": new_rotation,
        "primitive_type": new_primitive_type,
    }
    if new_ct_value_logit is not None:
        tensors["ct_value"] = new_ct_value_logit
    if new_atten_logit is not None:
        tensors["attenuation"] = new_atten_logit

    if model.optimizer is not None:
        optimizable_tensors = cat_tensors_to_optimizer(model, tensors)
        model._xyz = optimizable_tensors["xyz"]
        model._features_dc = optimizable_tensors["f_dc"]
        model._features_rest = optimizable_tensors["f_rest"]
        model._opacity = optimizable_tensors["opacity"]
        model._scaling = optimizable_tensors["scaling"]
        model._rotation = optimizable_tensors["rotation"]
        model._primitive_type = optimizable_tensors["primitive_type"]
        if "ct_value" in optimizable_tensors:
            model._ct_value_logit = optimizable_tensors["ct_value"]
        if "attenuation" in optimizable_tensors:
            model._atten_logit = optimizable_tensors["attenuation"]
    else:
        model._xyz = nn.Parameter(torch.cat((model._xyz, new_xyz), dim=0).requires_grad_(True))
        model._features_dc = nn.Parameter(torch.cat((model._features_dc, new_features_dc), dim=0).requires_grad_(True))
        model._features_rest = nn.Parameter(torch.cat((model._features_rest, new_features_rest), dim=0).requires_grad_(True))
        model._opacity = nn.Parameter(torch.cat((model._opacity, new_opacities), dim=0).requires_grad_(True))
        model._scaling = nn.Parameter(torch.cat((model._scaling, new_scaling), dim=0).requires_grad_(True))
        model._rotation = nn.Parameter(torch.cat((model._rotation, new_rotation), dim=0).requires_grad_(True))
        primitive_requires_grad = primitive_type_requires_grad(model)
        model._primitive_type = nn.Parameter(
            torch.cat((model._primitive_type, new_primitive_type), dim=0).requires_grad_(primitive_requires_grad),
            requires_grad=primitive_requires_grad,
        )
        if new_ct_value_logit is not None:
            model._ct_value_logit = nn.Parameter(
                torch.cat((model._ct_value_logit, new_ct_value_logit), dim=0).requires_grad_(True)
            )
        if new_atten_logit is not None:
            model._atten_logit = nn.Parameter(
                torch.cat((model._atten_logit, new_atten_logit), dim=0).requires_grad_(True)
            )

    normal_device = model._normal.device if isinstance(model._normal, torch.Tensor) and model._normal.numel() > 0 else model._device()
    new_normal = F.normalize(new_normal.to(device=normal_device, dtype=torch.float32).reshape(-1, 3), dim=1, eps=1e-8)
    existing_normal = model._normal.detach().to(device=normal_device, dtype=torch.float32).reshape(-1, 3)
    model._normal = torch.cat((existing_normal, new_normal), dim=0).detach().requires_grad_(False)
    model._material_id = torch.cat((model._material_id, new_material_id.to(device=model._material_id.device, dtype=torch.long)), dim=0)
    model._planarity = torch.cat((model._planarity, new_planarity.to(device=model._planarity.device, dtype=torch.float32)), dim=0)
    model._region_type = torch.cat((model._region_type, new_region_type.to(device=model._region_type.device, dtype=torch.long)), dim=0)
    device = model._device()
    model.xyz_gradient_accum = torch.zeros((model.get_xyz.shape[0], 1), device=device)
    model.denom = torch.zeros((model.get_xyz.shape[0], 1), device=device)
    model.max_radii2D = torch.zeros((model.get_xyz.shape[0],), device=device)
