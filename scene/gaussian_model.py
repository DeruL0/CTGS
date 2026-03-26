from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from torch import nn

from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.rotation_utils import matrix_to_quaternion, quaternion_to_matrix
from utils.system_utils import mkdir_p


class GaussianModel:
    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = int(sh_degree)
        self._xyz = torch.empty((0, 3), dtype=torch.float32)
        self._features_dc = torch.empty((0, 1, 3), dtype=torch.float32)
        self._features_rest = torch.empty((0, 0, 3), dtype=torch.float32)
        self._scaling = torch.empty((0, 3), dtype=torch.float32)
        self._rotation = torch.empty((0, 4), dtype=torch.float32)
        self._opacity = torch.empty((0, 1), dtype=torch.float32)
        self._primitive_type = torch.empty((0, 1), dtype=torch.float32)
        self._normal = torch.empty((0, 3), dtype=torch.float32)
        self._material_id = torch.empty((0, 1), dtype=torch.long)
        self._planarity = torch.empty((0, 1), dtype=torch.float32)
        self._region_type = torch.empty((0, 1), dtype=torch.long)
        self.max_radii2D = torch.empty((0,), dtype=torch.float32)
        self.xyz_gradient_accum = torch.empty((0, 1), dtype=torch.float32)
        self.denom = torch.empty((0, 1), dtype=torch.float32)
        self.optimizer = None
        self.percent_dense = 0.0
        self.spatial_lr_scale = 0.0
        self.primitive_harden_iter = 2000
        self.primitive_types_hardened = False
        self.planar_thickness_max = None
        self.planar_logit_value = 8.0
        self.nonplanar_logit_value = -8.0
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return strip_symmetric(actual_covariance)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def _default_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _device(self) -> torch.device:
        for tensor in (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._primitive_type,
            self._normal,
        ):
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                return tensor.device
        return self._default_device()

    def is_initialized(self) -> bool:
        return isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0

    def _default_primitive_type_logits(self, count: int, device: torch.device) -> torch.Tensor:
        return torch.full((int(count), 1), self.nonplanar_logit_value, dtype=torch.float32, device=device)

    def _sanitize_material_id(self, material_id, count: int, device: torch.device, fill_value: int = -1) -> torch.Tensor:
        if material_id is None:
            return torch.full((int(count), 1), fill_value, dtype=torch.long, device=device)
        if not isinstance(material_id, torch.Tensor):
            material_id = torch.as_tensor(material_id)
        return material_id.to(device=device, dtype=torch.long).reshape(int(count), 1)

    def _sanitize_planarity(self, planarity, count: int, device: torch.device, fill_value: float = 0.0) -> torch.Tensor:
        if planarity is None:
            return torch.full((int(count), 1), fill_value, dtype=torch.float32, device=device)
        if not isinstance(planarity, torch.Tensor):
            planarity = torch.as_tensor(planarity)
        return planarity.to(device=device, dtype=torch.float32).reshape(int(count), 1)

    def _sanitize_region_type(self, region_type, count: int, device: torch.device, fill_value: int = 0) -> torch.Tensor:
        if region_type is None:
            return torch.full((int(count), 1), fill_value, dtype=torch.long, device=device)
        if not isinstance(region_type, torch.Tensor):
            region_type = torch.as_tensor(region_type)
        return region_type.to(device=device, dtype=torch.long).reshape(int(count), 1)

    def _rotation_matrices_from_quaternions(self, rotations: torch.Tensor) -> torch.Tensor:
        if rotations.numel() == 0:
            return torch.empty((0, 3, 3), dtype=rotations.dtype, device=rotations.device)
        return quaternion_to_matrix(self.rotation_activation(rotations))

    def _derive_normals_from_rotations(self, rotations: torch.Tensor) -> torch.Tensor:
        if rotations.numel() == 0:
            return torch.empty((0, 3), dtype=rotations.dtype, device=rotations.device)
        rotation_matrices = self._rotation_matrices_from_quaternions(rotations)
        return F.normalize(rotation_matrices[:, :, 2], dim=1)

    def _initialize_hybrid_metadata(
        self,
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
            primitive_type_logits = self._default_primitive_type_logits(count, device)
        elif not isinstance(primitive_type_logits, torch.Tensor):
            primitive_type_logits = torch.as_tensor(primitive_type_logits)
        primitive_type_logits = primitive_type_logits.to(device=device, dtype=torch.float32).reshape(count, 1)

        if normals is None:
            if rotations is not None:
                normals = self._derive_normals_from_rotations(rotations)
            else:
                normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device).repeat(count, 1)
        elif not isinstance(normals, torch.Tensor):
            normals = torch.as_tensor(normals)
        normals = F.normalize(normals.to(device=device, dtype=torch.float32).reshape(count, 3), dim=1)

        self._primitive_type = nn.Parameter(primitive_type_logits.requires_grad_(True))
        self._normal = nn.Parameter(normals.requires_grad_(True))
        self._material_id = self._sanitize_material_id(material_id, count, device)
        self._planarity = self._sanitize_planarity(planarity, count, device)
        self._region_type = self._sanitize_region_type(region_type, count, device)

    def _freeze_primitive_type_parameter(self) -> None:
        if not isinstance(self._primitive_type, nn.Parameter):
            return
        self._primitive_type.requires_grad_(False)
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group["name"] == "primitive_type":
                group["lr"] = 0.0
                if len(group["params"]) == 1:
                    group["params"][0].requires_grad_(False)
                break

    def _assign_parameter(self, attr_name: str, tensor, optimizer_name: str | None = None, requires_grad: bool = True):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.to(device=self._device())
        if optimizer_name is not None and self.optimizer is not None:
            optimizable = self.replace_tensor_to_optimizer(tensor, optimizer_name, requires_grad=requires_grad)
            if optimizer_name in optimizable:
                setattr(self, attr_name, optimizable[optimizer_name])
                return optimizable[optimizer_name]
        param = nn.Parameter(tensor.requires_grad_(requires_grad))
        setattr(self, attr_name, param)
        return param

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._primitive_type,
            self._normal,
            self._material_id,
            self._planarity,
            self._region_type,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            None if self.optimizer is None else self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.primitive_harden_iter,
            self.primitive_types_hardened,
            self.planar_thickness_max,
        )

    def restore(self, model_args, training_args=None):
        if len(model_args) == 12:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            self.primitive_harden_iter = 2000
            self.primitive_types_hardened = False
            self.planar_thickness_max = None
            self._initialize_hybrid_metadata(
                self._xyz.shape[0],
                self._xyz.device,
                rotations=self._rotation.detach(),
            )
        elif len(model_args) == 19:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._primitive_type,
                self._normal,
                self._material_id,
                self._planarity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.primitive_harden_iter,
                self.primitive_types_hardened,
                self.planar_thickness_max,
            ) = model_args
            self._region_type = torch.zeros_like(self._material_id, dtype=torch.long, device=self._material_id.device)
        else:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._primitive_type,
                self._normal,
                self._material_id,
                self._planarity,
                self._region_type,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.primitive_harden_iter,
                self.primitive_types_hardened,
                self.planar_thickness_max,
            ) = model_args

        if training_args is not None:
            self.training_setup(training_args)
            if opt_dict is not None:
                try:
                    self.optimizer.load_state_dict(opt_dict)
                except ValueError:
                    print("Warning: optimizer state is incompatible with the current GaussianModel layout. Continuing with a fresh optimizer.")
            if self.primitive_types_hardened:
                self._freeze_primitive_type_parameter()

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

    @property
    def get_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        planar_mask = self.get_is_planar.squeeze(-1) if self._primitive_type.numel() > 0 else torch.zeros((0,), dtype=torch.bool, device=scaling.device)
        if self.planar_thickness_max is None or scaling.numel() == 0 or not torch.any(planar_mask):
            return scaling
        scaling = scaling.clone()
        thickness_limit = torch.as_tensor(self.planar_thickness_max, dtype=scaling.dtype, device=scaling.device)
        scaling[planar_mask, 2] = torch.minimum(scaling[planar_mask, 2], thickness_limit)
        return scaling

    @property
    def get_rotation(self):
        return self.get_effective_rotation()

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_material_id(self):
        return self._material_id

    @property
    def get_planarity(self):
        return self._planarity

    @property
    def get_region_type(self):
        return self._region_type

    @property
    def get_raw_scaling(self):
        return self._scaling

    @property
    def get_primitive_type_prob(self):
        if self._primitive_type.numel() == 0:
            return self._primitive_type
        if self.primitive_types_hardened:
            return (self._primitive_type >= 0).float()
        return torch.sigmoid(self._primitive_type)

    @property
    def get_is_planar(self):
        if self._primitive_type.numel() == 0:
            return torch.zeros((0, 1), dtype=torch.bool, device=self._device())
        return self.get_primitive_type_prob >= 0.5

    def get_normals(self):
        if self._normal.numel() == 0 and self._rotation.numel() == 0:
            return self._normal

        normals = self._derive_normals_from_rotations(self._rotation) if self._rotation.numel() > 0 else torch.empty_like(self._normal)
        if self._normal.numel() == 0:
            return normals

        explicit_normals = F.normalize(self._normal, dim=1)
        if normals.numel() == 0:
            return explicit_normals

        planar_mask = self.get_is_planar.squeeze(-1)
        if torch.any(planar_mask):
            normals = normals.clone()
            normals[planar_mask] = explicit_normals[planar_mask]
        return normals

    def get_effective_rotation(self):
        if self._rotation.numel() == 0:
            return self._rotation

        rotations = self.rotation_activation(self._rotation)
        planar_mask = self.get_is_planar.squeeze(-1) if self._primitive_type.numel() > 0 else torch.zeros((rotations.shape[0],), dtype=torch.bool, device=rotations.device)
        if not torch.any(planar_mask):
            return rotations

        rotation_matrices = self._rotation_matrices_from_quaternions(rotations)
        normals = self.get_normals()
        tangent_u = rotation_matrices[planar_mask, :, 0]
        planar_normals = normals[planar_mask]
        tangent_u = tangent_u - torch.sum(tangent_u * planar_normals, dim=1, keepdim=True) * planar_normals
        tangent_u_norm = torch.linalg.norm(tangent_u, dim=1, keepdim=True)

        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=rotation_matrices.dtype, device=rotation_matrices.device).repeat(planar_normals.shape[0], 1)
        parallel_x = torch.abs(torch.sum(fallback * planar_normals, dim=1, keepdim=True)) > 0.9
        fallback[parallel_x.squeeze(-1)] = torch.tensor([0.0, 1.0, 0.0], dtype=rotation_matrices.dtype, device=rotation_matrices.device)
        fallback = fallback - torch.sum(fallback * planar_normals, dim=1, keepdim=True) * planar_normals
        fallback = F.normalize(fallback, dim=1)

        tangent_u = torch.where(tangent_u_norm > 1e-6, tangent_u / tangent_u_norm.clamp_min(1e-6), fallback)
        tangent_v = F.normalize(torch.cross(planar_normals, tangent_u, dim=1), dim=1)
        tangent_u = F.normalize(torch.cross(tangent_v, planar_normals, dim=1), dim=1)

        planar_rotation = torch.stack((tangent_u, tangent_v, planar_normals), dim=2)
        rotation_matrices = rotation_matrices.clone()
        rotation_matrices[planar_mask] = planar_rotation
        return matrix_to_quaternion(rotation_matrices)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_effective_rotation())

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.primitive_harden_iter = getattr(training_args, "primitive_harden_iter", self.primitive_harden_iter)
        if getattr(training_args, "planar_thickness_max", None) is not None:
            self.planar_thickness_max = training_args.planar_thickness_max

        device = self._device()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

        param_groups = [
            (self._xyz, training_args.position_lr_init * self.spatial_lr_scale, "xyz"),
            (self._features_dc, training_args.feature_lr, "f_dc"),
            (self._features_rest, training_args.feature_lr / 20.0, "f_rest"),
            (self._opacity, training_args.opacity_lr, "opacity"),
            (self._scaling, training_args.scaling_lr, "scaling"),
            (self._rotation, training_args.rotation_lr, "rotation"),
            (self._normal, training_args.rotation_lr, "normal"),
            (self._primitive_type, training_args.opacity_lr, "primitive_type"),
        ]

        optim_params = []
        for parameter, lr, name in param_groups:
            parameter.requires_grad_(lr > 0.0 and parameter.requires_grad)
            optim_params.append({"params": [parameter], "lr": lr, "name": name})

        self.optimizer = torch.optim.Adam(optim_params)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        if self.primitive_types_hardened:
            self._freeze_primitive_type_parameter()

    def update_learning_rate(self, iteration):
        if self.optimizer is None:
            return None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
        return None

    def post_optimizer_step(self, iteration):
        del iteration
        return

    def construct_list_of_attributes(self):
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for index in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attributes.append(f"f_dc_{index}")
        for index in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attributes.append(f"f_rest_{index}")
        attributes.append("opacity")
        for index in range(self._scaling.shape[1]):
            attributes.append(f"scale_{index}")
        for index in range(self._rotation.shape[1]):
            attributes.append(f"rot_{index}")
        attributes.append("primitive_type")
        for index in range(3):
            attributes.append(f"normal_{index}")
        attributes.append("material_id")
        attributes.append("planarity")
        attributes.append("region_type")
        return attributes

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = self.get_normals().detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        primitive_type = self._primitive_type.detach().cpu().numpy()
        explicit_normals = F.normalize(self._normal.detach(), dim=1).cpu().numpy()
        material_id = self._material_id.detach().cpu().numpy().astype(np.float32)
        planarity = self._planarity.detach().cpu().numpy()
        region_type = self._region_type.detach().cpu().numpy().astype(np.float32)

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, primitive_type, explicit_normals, material_id, planarity, region_type),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
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
                self.max_sh_degree = max(self.max_sh_degree, inferred_degree)

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

        primitive_type = np.asarray(plydata.elements[0]["primitive_type"])[..., np.newaxis] if "primitive_type" in properties else None
        normal_names = sorted(
            [name for name in properties if name.startswith("normal_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        normals = None
        if len(normal_names) == 3:
            normals = np.stack([np.asarray(plydata.elements[0][name]) for name in normal_names], axis=1)
        material_id = np.asarray(plydata.elements[0]["material_id"])[..., np.newaxis] if "material_id" in properties else None
        planarity = np.asarray(plydata.elements[0]["planarity"])[..., np.newaxis] if "planarity" in properties else None
        region_type = np.asarray(plydata.elements[0]["region_type"])[..., np.newaxis] if "region_type" in properties else None

        device = self._default_device()
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float32, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float32, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device=device).requires_grad_(True))
        self._initialize_hybrid_metadata(
            xyz.shape[0],
            device,
            rotations=self._rotation.detach(),
            primitive_type_logits=None if primitive_type is None else torch.tensor(primitive_type, dtype=torch.float32, device=device),
            normals=None if normals is None else torch.tensor(normals, dtype=torch.float32, device=device),
            material_id=None if material_id is None else torch.tensor(material_id, dtype=torch.long, device=device),
            planarity=None if planarity is None else torch.tensor(planarity, dtype=torch.float32, device=device),
            region_type=None if region_type is None else torch.tensor(region_type, dtype=torch.long, device=device),
        )
        self.primitive_types_hardened = False
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0],), device=device)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name, requires_grad=True):
        optimizable_tensors = {}
        if self.optimizer is None:
            return optimizable_tensors
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(requires_grad))
            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        if self.optimizer is None:
            return optimizable_tensors
        for group in self.optimizer.param_groups:
            requires_grad = group["params"][0].requires_grad
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(requires_grad))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(requires_grad))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        if self.optimizer is not None:
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._normal = optimizable_tensors["normal"]
            self._primitive_type = optimizable_tensors["primitive_type"]
        else:
            self._xyz = nn.Parameter(self._xyz[valid_points_mask].requires_grad_(True))
            self._features_dc = nn.Parameter(self._features_dc[valid_points_mask].requires_grad_(True))
            self._features_rest = nn.Parameter(self._features_rest[valid_points_mask].requires_grad_(True))
            self._opacity = nn.Parameter(self._opacity[valid_points_mask].requires_grad_(True))
            self._scaling = nn.Parameter(self._scaling[valid_points_mask].requires_grad_(True))
            self._rotation = nn.Parameter(self._rotation[valid_points_mask].requires_grad_(True))
            self._normal = nn.Parameter(self._normal[valid_points_mask].requires_grad_(True))
            self._primitive_type = nn.Parameter(self._primitive_type[valid_points_mask].requires_grad_(not self.primitive_types_hardened))

        self._material_id = self._material_id[valid_points_mask]
        self._planarity = self._planarity[valid_points_mask]
        self._region_type = self._region_type[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        if self.optimizer is None:
            return optimizable_tensors
        for group in self.optimizer.param_groups:
            extension_tensor = tensors_dict[group["name"]]
            requires_grad = group["params"][0].requires_grad
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(requires_grad))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(requires_grad))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self,
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
    ):
        tensors = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "normal": new_normal,
            "primitive_type": new_primitive_type,
        }

        if self.optimizer is not None:
            optimizable_tensors = self.cat_tensors_to_optimizer(tensors)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._normal = optimizable_tensors["normal"]
            self._primitive_type = optimizable_tensors["primitive_type"]
        else:
            self._xyz = nn.Parameter(torch.cat((self._xyz, new_xyz), dim=0).requires_grad_(True))
            self._features_dc = nn.Parameter(torch.cat((self._features_dc, new_features_dc), dim=0).requires_grad_(True))
            self._features_rest = nn.Parameter(torch.cat((self._features_rest, new_features_rest), dim=0).requires_grad_(True))
            self._opacity = nn.Parameter(torch.cat((self._opacity, new_opacities), dim=0).requires_grad_(True))
            self._scaling = nn.Parameter(torch.cat((self._scaling, new_scaling), dim=0).requires_grad_(True))
            self._rotation = nn.Parameter(torch.cat((self._rotation, new_rotation), dim=0).requires_grad_(True))
            self._normal = nn.Parameter(torch.cat((self._normal, new_normal), dim=0).requires_grad_(True))
            self._primitive_type = nn.Parameter(
                torch.cat((self._primitive_type, new_primitive_type), dim=0).requires_grad_(not self.primitive_types_hardened)
            )

        self._material_id = torch.cat((self._material_id, new_material_id.to(device=self._material_id.device, dtype=torch.long)), dim=0)
        self._planarity = torch.cat((self._planarity, new_planarity.to(device=self._planarity.device, dtype=torch.float32)), dim=0)
        self._region_type = torch.cat((self._region_type, new_region_type.to(device=self._region_type.device, dtype=torch.long)), dim=0)
        device = self._device()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0],), device=device)

    def densify_and_split(self, grads, grad_threshold, split_clone_size, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points,), device=self._device())
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > split_clone_size)

        if selected_pts_mask.sum() == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.shape[0], 3), device=self._device())
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        self.densification_postfix(
            new_xyz,
            self._features_dc[selected_pts_mask].repeat(N, 1, 1),
            self._features_rest[selected_pts_mask].repeat(N, 1, 1),
            self._opacity[selected_pts_mask].repeat(N, 1),
            new_scaling,
            self._rotation[selected_pts_mask].repeat(N, 1),
            self._primitive_type[selected_pts_mask].repeat(N, 1),
            self._normal[selected_pts_mask].repeat(N, 1),
            self._material_id[selected_pts_mask].repeat(N, 1),
            self._planarity[selected_pts_mask].repeat(N, 1),
            self._region_type[selected_pts_mask].repeat(N, 1),
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), dtype=torch.bool, device=self._device())))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, split_clone_size):
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= split_clone_size)
        self.densification_postfix(
            self._xyz[selected_pts_mask],
            self._features_dc[selected_pts_mask],
            self._features_rest[selected_pts_mask],
            self._opacity[selected_pts_mask],
            self._scaling[selected_pts_mask],
            self._rotation[selected_pts_mask],
            self._primitive_type[selected_pts_mask],
            self._normal[selected_pts_mask],
            self._material_id[selected_pts_mask],
            self._planarity[selected_pts_mask],
            self._region_type[selected_pts_mask],
        )

    def densify_and_prune(self, max_grad, min_opacity, split_clone_size, max_scale, min_scale, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, split_clone_size)
        self.densify_and_split(grads, max_grad, split_clone_size)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            small_points_ws = self.get_scaling.max(dim=1).values < min_scale
            prune_mask = torch.logical_or(torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws), small_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def prune_all(self):
        prune_mask = (self.get_opacity != 0).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
