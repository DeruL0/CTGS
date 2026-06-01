from __future__ import annotations

import torch
import torch.nn.functional as F

from scene.gaussian_model_io import (
    capture_state,
    construct_list_of_attributes,
    load_ply_file,
    restore_state,
    save_ply_file,
)
from scene.gaussian_model_metadata import (
    default_device,
    default_primitive_type_logits,
    derive_normals_from_rotations,
    freeze_primitive_type_parameter,
    initialize_hybrid_metadata,
    is_initialized,
    model_device,
    primitive_type_requires_grad,
    rotation_matrices_from_quaternions,
    sanitize_material_id,
    sanitize_planarity,
    sanitize_region_type,
)
from scene.gaussian_model_optimizer import (
    assign_parameter,
    cat_tensors_to_optimizer,
    densification_postfix,
    post_optimizer_step,
    prune_optimizer,
    prune_points,
    replace_tensor_to_optimizer,
    reset_opacity,
    training_setup,
    update_learning_rate,
)
from ct_pipeline.geometry.coordinates import world_xyz_to_voxel_indices_floor_numpy
from utils.general_utils import build_scaling_rotation, inverse_sigmoid, strip_symmetric
from utils.rotation_utils import matrix_to_quaternion


class GaussianModel:
    def __init__(self, sh_degree: int, device=None):
        self._device_override = None if device is None else torch.device(device)
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
        self._ct_value_logit = torch.empty((0, 1), dtype=torch.float32)
        self._atten_logit = torch.empty((0, 1), dtype=torch.float32)
        self._bulk_offset = torch.empty((0, 3), dtype=torch.float32)  # bounded center offset for adaptive bulk
        self.max_radii2D = torch.empty((0,), dtype=torch.float32)
        self.xyz_gradient_accum = torch.empty((0, 1), dtype=torch.float32)
        self.denom = torch.empty((0, 1), dtype=torch.float32)
        self.optimizer = None
        self.percent_dense = 0.0
        self.spatial_lr_scale = 0.0
        self.freeze_primitive_type = False
        self.planar_thickness_max = None
        self.surface_thickness_max = None
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

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(value)
        return torch.where(
            value > 20.0,
            value,
            torch.log(torch.expm1(value.clamp_min(1e-8))),
        )

    def _default_device(self) -> torch.device:
        return default_device(self)

    def _device(self) -> torch.device:
        return model_device(self)

    def is_initialized(self) -> bool:
        return is_initialized(self)

    def _default_primitive_type_logits(self, count: int, device: torch.device) -> torch.Tensor:
        return default_primitive_type_logits(self, count, device)

    def _sanitize_material_id(self, material_id, count: int, device: torch.device, fill_value: int = -1) -> torch.Tensor:
        return sanitize_material_id(self, material_id, count, device, fill_value=fill_value)

    def _sanitize_planarity(self, planarity, count: int, device: torch.device, fill_value: float = 0.0) -> torch.Tensor:
        return sanitize_planarity(self, planarity, count, device, fill_value=fill_value)

    def _sanitize_region_type(self, region_type, count: int, device: torch.device, fill_value: int = 0) -> torch.Tensor:
        return sanitize_region_type(self, region_type, count, device, fill_value=fill_value)

    def _rotation_matrices_from_quaternions(self, rotations: torch.Tensor) -> torch.Tensor:
        return rotation_matrices_from_quaternions(self, rotations)

    def _derive_normals_from_rotations(self, rotations: torch.Tensor) -> torch.Tensor:
        return derive_normals_from_rotations(self, rotations)

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
        initialize_hybrid_metadata(
            self,
            count,
            device,
            rotations=rotations,
            primitive_type_logits=primitive_type_logits,
            normals=normals,
            material_id=material_id,
            planarity=planarity,
            region_type=region_type,
        )

    def initialize_ct_value_from_volume(self, volume_np, spacing_zyx) -> None:
        """Initialize CT appearance parameters from the reference CT volume.

        `_ct_value_logit` stays as a legacy/debug channel. `_atten_logit` is the
        v5.2 bulk attenuation amplitude and is seeded from the same normalized
        CT sample so the bulk field starts on the right intensity scale.
        """
        import numpy as np

        device = self._xyz.device
        count = int(self._xyz.shape[0])
        if count == 0:
            self._ct_value_logit = torch.nn.Parameter(
                torch.empty((0, 1), dtype=torch.float32, device=device).requires_grad_(True)
            )
            self._atten_logit = torch.nn.Parameter(
                torch.empty((0, 1), dtype=torch.float32, device=device).requires_grad_(True)
            )
            return

        xyz = self._xyz.detach().cpu().numpy()
        sz, sy, sx = (float(value) for value in spacing_zyx)
        volume_np = np.asarray(volume_np, dtype=np.float32)
        depth, height, width = volume_np.shape

        del sz, sy, sx
        zi, yi, xi = world_xyz_to_voxel_indices_floor_numpy(xyz, spacing_zyx, shape_dhw=(depth, height, width))
        sampled = volume_np[zi, yi, xi]
        sampled = np.clip(sampled, 1e-4, 1.0 - 1e-4).astype(np.float32)
        logits = np.log(sampled / (1.0 - sampled)).reshape(count, 1)
        ct_value_logit = torch.tensor(logits, dtype=torch.float32, device=device)
        self._ct_value_logit = torch.nn.Parameter(ct_value_logit.requires_grad_(True))
        attenuation = torch.tensor(sampled.reshape(count, 1), dtype=torch.float32, device=device).clamp_min(1e-6)
        atten_logit = self._inverse_softplus(attenuation)
        self._atten_logit = torch.nn.Parameter(atten_logit.requires_grad_(True))

    def _freeze_primitive_type_parameter(self) -> None:
        freeze_primitive_type_parameter(self)

    def _primitive_type_requires_grad(self) -> bool:
        return primitive_type_requires_grad(self)

    def _assign_parameter(self, attr_name: str, tensor, optimizer_name: str | None = None, requires_grad: bool = True):
        return assign_parameter(self, attr_name, tensor, optimizer_name=optimizer_name, requires_grad=requires_grad)

    def capture(self):
        return capture_state(self)

    def restore(self, model_args, training_args=None):
        restore_state(self, model_args, training_args=training_args)

    @property
    def get_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        if scaling.numel() == 0:
            return scaling

        planar_mask = self.get_is_planar.squeeze(-1) if self._primitive_type.numel() > 0 else torch.zeros((scaling.shape[0],), dtype=torch.bool, device=scaling.device)
        surface_mask = self.get_region_type.squeeze(-1) == 0 if self._region_type.numel() > 0 else torch.zeros((scaling.shape[0],), dtype=torch.bool, device=scaling.device)
        if (
            (self.planar_thickness_max is None or not torch.any(planar_mask))
            and (self.surface_thickness_max is None or not torch.any(surface_mask))
        ):
            return scaling

        scaling = scaling.clone()
        if self.planar_thickness_max is not None and torch.any(planar_mask):
            planar_limit = torch.as_tensor(self.planar_thickness_max, dtype=scaling.dtype, device=scaling.device)
            scaling[planar_mask, 2] = torch.minimum(scaling[planar_mask, 2], planar_limit)
        if self.surface_thickness_max is not None and torch.any(surface_mask):
            surface_limit = torch.as_tensor(self.surface_thickness_max, dtype=scaling.dtype, device=scaling.device)
            scaling[surface_mask, 2] = torch.minimum(scaling[surface_mask, 2], surface_limit)
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
    def get_ct_value(self):
        if self._ct_value_logit.numel() == 0:
            return self._ct_value_logit
        return torch.sigmoid(self._ct_value_logit)

    @property
    def get_attenuation(self):
        if self._atten_logit.numel() == 0:
            return self._atten_logit
        return F.softplus(self._atten_logit)

    @property
    def get_bulk_offset(self):
        """Bounded center offset for adaptive bulk mode.  Returns (N_bulk, 3) or empty."""
        return self._bulk_offset

    @property
    def get_raw_scaling(self):
        return self._scaling

    @property
    def get_primitive_type_prob(self):
        if self._primitive_type.numel() == 0:
            return self._primitive_type
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
        surface_mask = self.get_region_type.squeeze(-1) == 0 if self._region_type.numel() > 0 else torch.zeros_like(planar_mask)
        override_mask = planar_mask | surface_mask
        if torch.any(override_mask):
            normals = normals.clone()
            normals[override_mask] = explicit_normals[override_mask]
        return normals

    def get_effective_rotation(self):
        if self._rotation.numel() == 0:
            return self._rotation

        rotations = self.rotation_activation(self._rotation)
        active_mask = self.get_is_planar.squeeze(-1) if self._primitive_type.numel() > 0 else torch.zeros((rotations.shape[0],), dtype=torch.bool, device=rotations.device)
        if not torch.any(active_mask):
            return rotations

        rotation_matrices = self._rotation_matrices_from_quaternions(rotations)
        normals = self.get_normals()
        tangent_u = rotation_matrices[active_mask, :, 0]
        active_normals = normals[active_mask]
        tangent_u = tangent_u - torch.sum(tangent_u * active_normals, dim=1, keepdim=True) * active_normals
        tangent_u_norm = torch.linalg.norm(tangent_u, dim=1, keepdim=True)

        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=rotation_matrices.dtype, device=rotation_matrices.device).repeat(active_normals.shape[0], 1)
        parallel_x = torch.abs(torch.sum(fallback * active_normals, dim=1, keepdim=True)) > 0.9
        fallback[parallel_x.squeeze(-1)] = torch.tensor([0.0, 1.0, 0.0], dtype=rotation_matrices.dtype, device=rotation_matrices.device)
        fallback = fallback - torch.sum(fallback * active_normals, dim=1, keepdim=True) * active_normals
        fallback = F.normalize(fallback, dim=1)

        tangent_u = torch.where(tangent_u_norm > 1e-6, tangent_u / tangent_u_norm.clamp_min(1e-6), fallback)
        tangent_v = F.normalize(torch.cross(active_normals, tangent_u, dim=1), dim=1)
        tangent_u = F.normalize(torch.cross(tangent_v, active_normals, dim=1), dim=1)

        planar_rotation = torch.stack((tangent_u, tangent_v, active_normals), dim=2)
        rotation_matrices = rotation_matrices.clone()
        rotation_matrices[active_mask] = planar_rotation
        return matrix_to_quaternion(rotation_matrices)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_effective_rotation())

    def training_setup(self, training_args):
        training_setup(self, training_args)

    def update_learning_rate(self, iteration):
        return update_learning_rate(self, iteration)

    def post_optimizer_step(self, iteration):
        return post_optimizer_step(self, iteration)

    def construct_list_of_attributes(self):
        return construct_list_of_attributes(self)

    def save_ply(self, path):
        save_ply_file(self, path)

    def reset_opacity(self):
        reset_opacity(self)

    def load_ply(self, path):
        load_ply_file(self, path)

    def replace_tensor_to_optimizer(self, tensor, name, requires_grad=True):
        return replace_tensor_to_optimizer(self, tensor, name, requires_grad=requires_grad)

    def _prune_optimizer(self, mask):
        return prune_optimizer(self, mask)

    def prune_points(self, mask):
        prune_points(self, mask)

    def cat_tensors_to_optimizer(self, tensors_dict):
        return cat_tensors_to_optimizer(self, tensors_dict)

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
        new_ct_value_logit=None,
        new_atten_logit=None,
    ):
        densification_postfix(
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
            new_ct_value_logit=new_ct_value_logit,
            new_atten_logit=new_atten_logit,
        )
