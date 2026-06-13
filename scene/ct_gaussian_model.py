import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree
from torch import nn

from utils.rotation_utils import matrix_to_quaternion
from .gaussian_model import GaussianModel

from .ct_bulk_initialization import (
    CT_BULK_LATTICE_ANISOTROPIC,
    CT_BULK_LATTICE_ATTEN_INIT,
    CT_BULK_LATTICE_MARGIN_VOX,
    CT_BULK_LATTICE_SIGMA_N_VOX,
    CT_BULK_LATTICE_SIGMA_T_VOX,
    CT_BULK_LATTICE_SIGMA_VOX,
    CT_BULK_LATTICE_SPACING_VOX,
    CT_DENSE_INIT_BULK_MAX_SCALE_RATIO,
    CT_DENSE_INIT_BULK_MIN_SCALE_RATIO,
    CT_DENSE_INIT_BULK_RADIUS_RATIO,
    CT_DENSE_INIT_SURFACE_MIN_SCALE_RATIO,
    CT_DENSE_INIT_SURFACE_POISSON_BASE_RATIO,
    CT_DENSE_INIT_SURFACE_TANGENT_RATIO,
    CT_DENSE_INIT_SURFACE_THICKNESS_RATIO,
    CT_FEATURE_ADAPTIVE_BLUR_SIGMA_VOX,
    CT_FEATURE_ADAPTIVE_DIRECTIONAL_CLEARANCE,
    CT_FEATURE_ADAPTIVE_CLEARANCE_Q_CONT,
    CT_FEATURE_ADAPTIVE_CLEARANCE_SAFETY,
    CT_FEATURE_ADAPTIVE_JITTER,
    CT_FEATURE_ADAPTIVE_R_SHELL_VOX,
    CT_FEATURE_ADAPTIVE_PROBE_CONTAINMENT,
    CT_FEATURE_ADAPTIVE_SEED,
    CT_FEATURE_ADAPTIVE_SPACING_HIGH_VOX,
    CT_FEATURE_ADAPTIVE_SPACING_LOW_VOX,
    CT_FEATURE_ADAPTIVE_SPACING_MID_VOX,
    _apply_directional_clearance_scales,
    _augment_interior_points,
    _build_contained_lattice_points,
    _build_feature_adaptive_bulk_attributes,
    _build_feature_adaptive_bulk_points,
    _build_frame_from_normal,
    _build_sdf_aligned_bulk_attributes,
    _contain_initial_bulk_attributes,
    _normalize_np,
    _poisson_disk_filter_boundary,
    _probe_correct_feature_adaptive_bulk_attributes,
    _sample_sdf_and_gradient_at_points,
)


class CTGaussianModel(GaussianModel):
    """CT-aware GaussianModel that initializes hybrid primitives from Phase 1 analysis."""

    def __init__(self, sh_degree: int, device=None):
        super().__init__(sh_degree, device=device)
        self.single_material_fallback = False
        self.surface_thickness_max = None
        self.ct_feature_adaptive_init_stats = {}

    def create_from_phase1_bundle(
        self,
        analysis_path,
        metadata_path,
        spatial_lr_scale,
        surface_thickness_max=None,
        planar_thickness_max=None,
        bulk_continuous_init=True,
        bulk_augment_factor: float = 1.0,
        bulk_init_mode: str = "sparse_reseed",
        bulk_lattice_spacing_vox: float = CT_BULK_LATTICE_SPACING_VOX,
        bulk_lattice_sigma_vox: float = CT_BULK_LATTICE_SIGMA_VOX,
        bulk_lattice_margin_vox: float = CT_BULK_LATTICE_MARGIN_VOX,
        bulk_lattice_atten_init: float = CT_BULK_LATTICE_ATTEN_INIT,
        bulk_lattice_anisotropic: bool = CT_BULK_LATTICE_ANISOTROPIC,
        bulk_lattice_sigma_t_vox: float = CT_BULK_LATTICE_SIGMA_T_VOX,
        bulk_lattice_sigma_n_vox: float = CT_BULK_LATTICE_SIGMA_N_VOX,
        intensity_volume=None,
        feature_adaptive_jitter: bool = CT_FEATURE_ADAPTIVE_JITTER,
        feature_adaptive_seed: int = CT_FEATURE_ADAPTIVE_SEED,
        feature_adaptive_r_shell_vox: float = CT_FEATURE_ADAPTIVE_R_SHELL_VOX,
        feature_adaptive_blur_sigma_vox: float = CT_FEATURE_ADAPTIVE_BLUR_SIGMA_VOX,
        feature_adaptive_spacing_high_vox: int = CT_FEATURE_ADAPTIVE_SPACING_HIGH_VOX,
        feature_adaptive_spacing_mid_vox: int = CT_FEATURE_ADAPTIVE_SPACING_MID_VOX,
        feature_adaptive_spacing_low_vox: int = CT_FEATURE_ADAPTIVE_SPACING_LOW_VOX,
        feature_adaptive_directional_clearance: bool = CT_FEATURE_ADAPTIVE_DIRECTIONAL_CLEARANCE,
        feature_adaptive_probe_containment: bool = CT_FEATURE_ADAPTIVE_PROBE_CONTAINMENT,
    ):
        analysis_path = Path(analysis_path)
        metadata_path = Path(metadata_path)
        with np.load(str(analysis_path)) as analysis_npz:
            analysis = {key: analysis_npz[key] for key in analysis_npz.files}
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        spacing = tuple(float(value) for value in metadata["spacing_zyx"])
        boundary_points, boundary_normals, boundary_tangent_u, boundary_tangent_v, boundary_strength, boundary_material_id = self._resolve_boundary_samples(
            analysis,
        )
        interior_points, interior_density_seed, interior_material_id = self._resolve_interior_samples(
            analysis,
        )
        support_volume = analysis.get("coarse_support_mask")
        if support_volume is None:
            support_volume = analysis.get("material_mask")
        material_mask_volume = analysis.get("material_mask")
        if material_mask_volume is None:
            material_mask_volume = support_volume
        volume_shape_dhw = None if support_volume is None else tuple(int(value) for value in np.asarray(support_volume).shape)
        signed_distance_volume = analysis.get("material_signed_distance")
        if signed_distance_volume is None:
            _material_mask = analysis.get("material_mask")
            if _material_mask is not None:
                from ct_pipeline.data.preprocessing import build_support_signed_distance
                signed_distance_volume = build_support_signed_distance(
                    np.asarray(_material_mask, dtype=bool),
                    tuple(float(v) for v in spacing),
                )

        material_mask_for_augment = analysis.get("material_mask")
        if material_mask_for_augment is None:
            material_mask_for_augment = analysis.get("coarse_support_mask")
        if (
            material_mask_for_augment is not None
            and interior_points.shape[0] > 0
            and float(bulk_augment_factor) > 1.0
        ):
            target_count = int(round(float(interior_points.shape[0]) * float(bulk_augment_factor)))
            interior_points, interior_density_seed, interior_material_id = _augment_interior_points(
                interior_points,
                interior_density_seed,
                interior_material_id,
                material_mask_for_augment,
                spacing,
                target_count,
            )

        self._create_from_analysis(
            boundary_points=boundary_points,
            boundary_normals=boundary_normals,
            boundary_tangent_u=boundary_tangent_u,
            boundary_tangent_v=boundary_tangent_v,
            boundary_strength=boundary_strength,
            spacing=spacing,
            spatial_lr_scale=spatial_lr_scale,
            surface_thickness_max=surface_thickness_max if surface_thickness_max is not None else planar_thickness_max,
            material_id=boundary_material_id,
            interior_points=interior_points,
            interior_density_seed=interior_density_seed,
            interior_material_id=interior_material_id,
            bulk_continuous_init=bulk_continuous_init,
            volume_shape_dhw=volume_shape_dhw,
            signed_distance_volume=signed_distance_volume,
            material_mask_volume=material_mask_volume,
            bulk_init_mode=bulk_init_mode,
            bulk_lattice_spacing_vox=bulk_lattice_spacing_vox,
            bulk_lattice_sigma_vox=bulk_lattice_sigma_vox,
            bulk_lattice_margin_vox=bulk_lattice_margin_vox,
            bulk_lattice_atten_init=bulk_lattice_atten_init,
            bulk_lattice_anisotropic=bulk_lattice_anisotropic,
            bulk_lattice_sigma_t_vox=bulk_lattice_sigma_t_vox,
            bulk_lattice_sigma_n_vox=bulk_lattice_sigma_n_vox,
            intensity_volume=intensity_volume,
            feature_adaptive_jitter=feature_adaptive_jitter,
            feature_adaptive_seed=feature_adaptive_seed,
            feature_adaptive_r_shell_vox=feature_adaptive_r_shell_vox,
            feature_adaptive_blur_sigma_vox=feature_adaptive_blur_sigma_vox,
            feature_adaptive_spacing_high_vox=feature_adaptive_spacing_high_vox,
            feature_adaptive_spacing_mid_vox=feature_adaptive_spacing_mid_vox,
            feature_adaptive_spacing_low_vox=feature_adaptive_spacing_low_vox,
            feature_adaptive_directional_clearance=feature_adaptive_directional_clearance,
            feature_adaptive_probe_containment=feature_adaptive_probe_containment,
        )

    def clamp_surface_thickness(self, max_thickness: float):
        self.surface_thickness_max = float(max_thickness)
        self.planar_thickness_max = float(max_thickness)
        if self._scaling.numel() == 0:
            return

        surface_mask = self.get_region_type.squeeze(-1) == 0
        if not torch.any(surface_mask):
            return

        scales = self.scaling_activation(self._scaling.detach())
        thickness_limit = torch.as_tensor(self.surface_thickness_max, dtype=scales.dtype, device=scales.device)
        scales[surface_mask, 2] = torch.minimum(scales[surface_mask, 2], thickness_limit)
        clamped = torch.clamp(scales, min=1e-8)
        new_scaling = self.scaling_inverse_activation(clamped)
        self._assign_parameter("_scaling", new_scaling, optimizer_name="scaling", requires_grad=True)

    def get_normals(self) -> torch.Tensor:
        return super().get_normals()

    def post_optimizer_step(self, iteration):
        del iteration
        if self.surface_thickness_max is not None:
            self.clamp_surface_thickness(self.surface_thickness_max)
        self._sync_surface_normals_from_geometry()

    @torch.no_grad()
    def _sync_surface_normals_from_geometry(self):
        if self._xyz.numel() == 0 or self._rotation.numel() == 0 or self._scaling.numel() == 0:
            return
        surface_mask = self.get_region_type.squeeze(-1) == 0
        if not torch.any(surface_mask):
            return

        normal_device = self._xyz.device
        normal_count = int(self._xyz.shape[0])
        if not isinstance(self._normal, torch.Tensor) or self._normal.shape[0] != normal_count:
            self._normal = self._derive_normals_from_rotations(self._rotation).detach()
        self._normal = self._normal.detach().to(device=normal_device, dtype=torch.float32)

        surface_scales = self.get_scaling.detach()[surface_mask]
        surface_rotations = self._rotation_matrices_from_quaternions(self._rotation.detach()[surface_mask])
        min_axis = surface_scales.argmin(dim=-1)
        local_axis = F.one_hot(min_axis, num_classes=3).to(device=surface_scales.device, dtype=surface_scales.dtype)
        derived = torch.einsum("nij,nj->ni", surface_rotations, local_axis)
        derived = F.normalize(derived, dim=-1, eps=1e-8)

        previous = F.normalize(self._normal[surface_mask].to(device=derived.device, dtype=derived.dtype), dim=-1, eps=1e-8)
        sign = torch.where(
            torch.sum(derived * previous, dim=-1, keepdim=True) < 0.0,
            -torch.ones((derived.shape[0], 1), dtype=derived.dtype, device=derived.device),
            torch.ones((derived.shape[0], 1), dtype=derived.dtype, device=derived.device),
        )
        self._normal[surface_mask] = derived * sign

    def _create_from_analysis(
        self,
        boundary_points,
        boundary_normals,
        boundary_tangent_u,
        boundary_tangent_v,
        boundary_strength,
        spacing,
        spatial_lr_scale,
        surface_thickness_max,
        material_id,
        interior_points,
        interior_density_seed,
        interior_material_id,
        bulk_continuous_init=True,
        volume_shape_dhw=None,
        signed_distance_volume=None,
        material_mask_volume=None,
        bulk_init_mode: str = "sparse_reseed",
        bulk_lattice_spacing_vox: float = CT_BULK_LATTICE_SPACING_VOX,
        bulk_lattice_sigma_vox: float = CT_BULK_LATTICE_SIGMA_VOX,
        bulk_lattice_margin_vox: float = CT_BULK_LATTICE_MARGIN_VOX,
        bulk_lattice_atten_init: float = CT_BULK_LATTICE_ATTEN_INIT,
        bulk_lattice_anisotropic: bool = CT_BULK_LATTICE_ANISOTROPIC,
        bulk_lattice_sigma_t_vox: float = CT_BULK_LATTICE_SIGMA_T_VOX,
        bulk_lattice_sigma_n_vox: float = CT_BULK_LATTICE_SIGMA_N_VOX,
        intensity_volume=None,
        feature_adaptive_jitter: bool = CT_FEATURE_ADAPTIVE_JITTER,
        feature_adaptive_seed: int = CT_FEATURE_ADAPTIVE_SEED,
        feature_adaptive_r_shell_vox: float = CT_FEATURE_ADAPTIVE_R_SHELL_VOX,
        feature_adaptive_blur_sigma_vox: float = CT_FEATURE_ADAPTIVE_BLUR_SIGMA_VOX,
        feature_adaptive_spacing_high_vox: int = CT_FEATURE_ADAPTIVE_SPACING_HIGH_VOX,
        feature_adaptive_spacing_mid_vox: int = CT_FEATURE_ADAPTIVE_SPACING_MID_VOX,
        feature_adaptive_spacing_low_vox: int = CT_FEATURE_ADAPTIVE_SPACING_LOW_VOX,
        feature_adaptive_directional_clearance: bool = CT_FEATURE_ADAPTIVE_DIRECTIONAL_CLEARANCE,
        feature_adaptive_probe_containment: bool = CT_FEATURE_ADAPTIVE_PROBE_CONTAINMENT,
    ):
        boundary_points = np.asarray(boundary_points, dtype=np.float32)
        boundary_normals = np.asarray(boundary_normals, dtype=np.float32)
        boundary_tangent_u = np.asarray(boundary_tangent_u, dtype=np.float32)
        boundary_tangent_v = np.asarray(boundary_tangent_v, dtype=np.float32)
        boundary_strength = np.asarray(boundary_strength, dtype=np.float32).reshape(-1, 1)
        spacing = tuple(float(value) for value in spacing)
        interior_points = np.asarray(interior_points, dtype=np.float32).reshape(-1, 3)
        interior_density_seed = np.asarray(interior_density_seed, dtype=np.float32).reshape(-1, 1)
        interior_material_id = np.asarray(interior_material_id, dtype=np.int64).reshape(-1, 1)

        if boundary_points.shape[0] > 1:
            poisson_base = CT_DENSE_INIT_SURFACE_POISSON_BASE_RATIO * float(min(spacing))
            keep_idx = _poisson_disk_filter_boundary(
                boundary_points,
                boundary_strength,
                base_spacing=poisson_base,
            )
            if 0 < keep_idx.shape[0] < boundary_points.shape[0]:
                boundary_points = boundary_points[keep_idx]
                boundary_normals = boundary_normals[keep_idx]
                boundary_tangent_u = boundary_tangent_u[keep_idx]
                boundary_tangent_v = boundary_tangent_v[keep_idx]
                boundary_strength = boundary_strength[keep_idx]
                if material_id is not None:
                    material_id_array = np.asarray(material_id).reshape(-1, 1)
                    if material_id_array.shape[0] >= int(keep_idx.max()) + 1:
                        material_id = material_id_array[keep_idx]

        num_points = boundary_points.shape[0]
        if num_points == 0:
            raise ValueError("Phase 1 analysis bundle does not contain any boundary points.")

        self.spatial_lr_scale = spatial_lr_scale
        min_spacing = float(min(spacing))
        self.surface_thickness_max = (
            float(surface_thickness_max)
            if surface_thickness_max is not None
            else CT_DENSE_INIT_SURFACE_THICKNESS_RATIO * min_spacing
        )
        self.planar_thickness_max = self.surface_thickness_max

        if material_id is None:
            material_id = np.zeros((num_points, 1), dtype=np.int64)
            self.single_material_fallback = True
        else:
            material_id = np.asarray(material_id).reshape(num_points, 1).astype(np.int64)
            self.single_material_fallback = False
        if interior_points.shape[0] == 0:
            interior_density_seed = np.empty((0, 1), dtype=np.float32)
            interior_material_id = np.empty((0, 1), dtype=np.int64)
        elif interior_material_id.shape[0] == 0:
            interior_material_id = np.zeros((interior_points.shape[0], 1), dtype=np.int64)
        adaptive_bulk_meta = None
        self.ct_feature_adaptive_init_stats = {}
        if (
            bulk_init_mode in ("feature_adaptive", "fasj")
            and material_mask_volume is not None
            and signed_distance_volume is not None
        ):
            adaptive_pts, adaptive_bulk_meta = _build_feature_adaptive_bulk_points(
                material_mask_volume,
                signed_distance_volume,
                intensity_volume,
                spacing,
                r_shell_vox=float(feature_adaptive_r_shell_vox),
                blur_sigma_vox=float(feature_adaptive_blur_sigma_vox),
                jitter=bool(feature_adaptive_jitter),
                seed=int(feature_adaptive_seed),
                spacing_high_vox=int(feature_adaptive_spacing_high_vox),
                spacing_mid_vox=int(feature_adaptive_spacing_mid_vox),
                spacing_low_vox=int(feature_adaptive_spacing_low_vox),
            )
            if adaptive_pts.shape[0] > 0:
                interior_points = adaptive_pts
                interior_density_seed = np.full(
                    (adaptive_pts.shape[0], 1), float(bulk_lattice_atten_init), dtype=np.float32
                )
                interior_material_id = np.zeros((adaptive_pts.shape[0], 1), dtype=np.int64)
                self.ct_feature_adaptive_init_stats = {
                    "num_init_candidates_total": int(adaptive_pts.shape[0]),
                    "num_init_candidates_shrunk": 0,
                    "num_init_candidates_downgraded": 0,
                    "num_init_candidates_rejected": 0,
                }
        elif bulk_init_mode in ("contained_lattice", "conservative_envelope") and (
            signed_distance_volume is not None or material_mask_volume is not None
        ):
            if bulk_init_mode == "conservative_envelope":
                # dilate the confident material mask before building lattice
                if material_mask_volume is not None:
                    _mat_mask = np.asarray(material_mask_volume, dtype=bool)
                else:
                    _mat_mask = (np.asarray(signed_distance_volume) < 0).astype(bool)
                _dilate_vox = max(1, int(round(float(bulk_lattice_margin_vox) * 4)))
                _structure = ndimage.generate_binary_structure(3, 1)
                _envelope = ndimage.binary_dilation(_mat_mask, structure=_structure, iterations=_dilate_vox)
                # Build SDF from envelope for radius estimates, but preserve binary ownership from the envelope.
                from ct_pipeline.data.preprocessing import build_support_signed_distance as _bssd
                _envelope_sdf = _bssd(_envelope, tuple(float(v) for v in spacing))
                _sdf_for_lattice = _envelope_sdf
                _mask_for_lattice = _envelope
            else:
                _sdf_for_lattice = signed_distance_volume
                _mask_for_lattice = material_mask_volume
            lattice_pts, _ = _build_contained_lattice_points(
                _sdf_for_lattice,
                spacing,
                material_mask_volume=_mask_for_lattice,
                spacing_vox=float(bulk_lattice_spacing_vox),
                margin_vox=float(bulk_lattice_margin_vox),
            )
            if lattice_pts.shape[0] > 0:
                interior_points = lattice_pts
                interior_density_seed = np.full(
                    (lattice_pts.shape[0], 1), float(bulk_lattice_atten_init), dtype=np.float32
                )
                interior_material_id = np.zeros((lattice_pts.shape[0], 1), dtype=np.int64)
        elif bulk_continuous_init and interior_points.shape[0] > 0:
            interior_points = self._apply_bulk_continuous_jitter(
                interior_points,
                spacing,
                volume_shape_dhw=volume_shape_dhw,
            )

        nearest_neighbor_distance = self._estimate_nearest_neighbor_distance(boundary_points, default_value=min_spacing)
        rotation_matrices = np.zeros((num_points, 3, 3), dtype=np.float32)
        explicit_normals = np.zeros((num_points, 3), dtype=np.float32)
        local_scales = np.zeros((num_points, 3), dtype=np.float32)
        for point_index in range(num_points):
            tangent_u, tangent_v, normal = _build_frame_from_normal(
                boundary_normals[point_index],
                tangent_hint=boundary_tangent_u[point_index],
            )
            if np.all(np.isfinite(boundary_tangent_v[point_index])):
                tangent_v = boundary_tangent_v[point_index]
                tangent_v = tangent_v - np.dot(tangent_v, normal) * normal
                tangent_v = _normalize_np(tangent_v)
                if tangent_v is not None:
                    tangent_u = _normalize_np(np.cross(tangent_v, normal))
                    tangent_v = _normalize_np(np.cross(normal, tangent_u))
                    tangent_u = tangent_u.astype(np.float32)
                    tangent_v = tangent_v.astype(np.float32)
            rotation_matrices[point_index] = np.stack((tangent_u, tangent_v, normal), axis=1)
            explicit_normals[point_index] = normal
            tangential_scale = max(
                float(nearest_neighbor_distance[point_index]) * CT_DENSE_INIT_SURFACE_TANGENT_RATIO,
                CT_DENSE_INIT_SURFACE_MIN_SCALE_RATIO * min_spacing,
            )
            local_scales[point_index] = np.array(
                [tangential_scale, tangential_scale, self.surface_thickness_max],
                dtype=np.float32,
            )

        local_scales = np.clip(local_scales, a_min=min_spacing * CT_DENSE_INIT_SURFACE_MIN_SCALE_RATIO, a_max=None)
        local_scales[:, 2] = np.minimum(local_scales[:, 2], self.surface_thickness_max)
        primitive_logits = np.full((num_points, 1), self.nonplanar_logit_value, dtype=np.float32)
        planarity = np.zeros((num_points, 1), dtype=np.float32)

        bulk_count = interior_points.shape[0]
        if bulk_count > 0:
            bulk_nn = self._estimate_nearest_neighbor_distance(interior_points, default_value=min_spacing)
            if bulk_init_mode in ("feature_adaptive", "fasj") and adaptive_bulk_meta is not None:
                has_clearance_domain = (
                    bool(feature_adaptive_directional_clearance)
                    and (signed_distance_volume is not None or material_mask_volume is not None)
                )
                bulk_scales, bulk_rotations, bulk_normals = _build_feature_adaptive_bulk_attributes(
                    adaptive_bulk_meta,
                    min_spacing,
                    anisotropic=bool(bulk_lattice_anisotropic) or bool(has_clearance_domain),
                    interior_points=interior_points,
                    signed_distance_volume=signed_distance_volume,
                    spacing_zyx=spacing,
                )
                if has_clearance_domain:
                    bulk_scales, clearance_stats = _apply_directional_clearance_scales(
                        interior_points,
                        bulk_scales,
                        bulk_rotations,
                        material_mask_volume,
                        spacing,
                        min_spacing,
                        signed_distance_volume=signed_distance_volume,
                    )
                    self.ct_feature_adaptive_init_stats.update(clearance_stats)
                    self.ct_feature_adaptive_init_stats["init_directional_clearance_enforced"] = True
                    self.ct_feature_adaptive_init_stats["init_directional_clearance_q_cont"] = float(
                        CT_FEATURE_ADAPTIVE_CLEARANCE_Q_CONT
                    )
                    self.ct_feature_adaptive_init_stats["init_directional_clearance_safety"] = float(
                        CT_FEATURE_ADAPTIVE_CLEARANCE_SAFETY
                    )
                else:
                    self.ct_feature_adaptive_init_stats["init_directional_clearance_enforced"] = False
                self.ct_feature_adaptive_init_stats["init_probe_final_check"] = bool(
                    feature_adaptive_probe_containment and has_clearance_domain
                )
                if bool(feature_adaptive_probe_containment) and has_clearance_domain:
                    bulk_scales, bulk_rotations, bulk_normals, bulk_keep, probe_stats = (
                        _probe_correct_feature_adaptive_bulk_attributes(
                            interior_points,
                            bulk_scales,
                            bulk_rotations,
                            bulk_normals,
                            signed_distance_volume,
                            material_mask_volume,
                            spacing,
                            min_spacing,
                        )
                    )
                    self.ct_feature_adaptive_init_stats.update(probe_stats)
                    if not np.all(bulk_keep):
                        interior_points = interior_points[bulk_keep]
                        interior_density_seed = interior_density_seed[bulk_keep]
                        interior_material_id = interior_material_id[bulk_keep]
                        bulk_count = int(interior_points.shape[0])
                        bulk_scales = bulk_scales[bulk_keep]
                        bulk_rotations = bulk_rotations[bulk_keep]
                        bulk_normals = bulk_normals[bulk_keep]
            elif bulk_init_mode in ("contained_lattice", "conservative_envelope") and bool(bulk_lattice_anisotropic) and signed_distance_volume is not None:
                sdf_values, sdf_gradients = _sample_sdf_and_gradient_at_points(
                    interior_points,
                    signed_distance_volume,
                    spacing,
                )
                inside_distance = np.maximum(-sdf_values.astype(np.float32), 0.0)
                tangent_sigma = max(float(bulk_lattice_sigma_t_vox), 1e-6) * min_spacing
                normal_sigma = max(float(bulk_lattice_sigma_n_vox), 1e-6) * min_spacing
                min_normal = max(0.10 * min_spacing, 1e-6)
                normal_radius = np.minimum(normal_sigma, np.maximum(inside_distance * 0.85, min_normal)).astype(np.float32)
                tangent_radius = np.full((bulk_count,), tangent_sigma, dtype=np.float32)
                bulk_scales = np.stack((tangent_radius, tangent_radius, normal_radius), axis=1).astype(np.float32)
                bulk_rotations = np.repeat(np.eye(3, dtype=np.float32)[np.newaxis, :, :], bulk_count, axis=0)
                bulk_normals = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), bulk_count, axis=0)
                gradient_norm = np.linalg.norm(sdf_gradients, axis=1)
                trusted = gradient_norm > float(CT_DENSE_INIT_BULK_GRADIENT_THRESHOLD)
                for index in np.nonzero(trusted)[0]:
                    normal = sdf_gradients[index] / max(float(gradient_norm[index]), 1e-8)
                    tangent_u, tangent_v, normal = _build_frame_from_normal(normal)
                    bulk_rotations[index] = np.stack((tangent_u, tangent_v, normal), axis=1)
                    bulk_normals[index] = normal
            elif bulk_init_mode in ("contained_lattice", "conservative_envelope"):
                # isotropic sigma = sigma_vox * min_spacing, capped by half lattice step
                sigma = float(bulk_lattice_sigma_vox) * min_spacing
                bulk_radius = np.full((bulk_count,), sigma, dtype=np.float32)
                bulk_scales = np.repeat(bulk_radius[:, np.newaxis], 3, axis=1)
                bulk_rotations = np.repeat(np.eye(3, dtype=np.float32)[np.newaxis, :, :], bulk_count, axis=0)
                bulk_normals = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), bulk_count, axis=0)
            elif signed_distance_volume is not None:
                bulk_scales, bulk_rotations, bulk_normals = _build_sdf_aligned_bulk_attributes(
                    interior_points=interior_points,
                    signed_distance_volume=signed_distance_volume,
                    spacing_zyx=spacing,
                    nn_distance=bulk_nn,
                    min_spacing=min_spacing,
                )
                bulk_scales, bulk_rotations, bulk_normals, bulk_keep = _contain_initial_bulk_attributes(
                    interior_points,
                    bulk_scales,
                    bulk_rotations,
                    bulk_normals,
                    signed_distance_volume,
                    spacing,
                    min_spacing,
                )
                if not np.all(bulk_keep):
                    interior_points = interior_points[bulk_keep]
                    interior_density_seed = interior_density_seed[bulk_keep]
                    interior_material_id = interior_material_id[bulk_keep]
                    bulk_count = int(interior_points.shape[0])
                    bulk_scales = bulk_scales[bulk_keep]
                    bulk_rotations = bulk_rotations[bulk_keep]
                    bulk_normals = bulk_normals[bulk_keep]
            else:
                bulk_radius = np.clip(
                    CT_DENSE_INIT_BULK_RADIUS_RATIO * bulk_nn,
                    a_min=CT_DENSE_INIT_BULK_MIN_SCALE_RATIO * min_spacing,
                    a_max=CT_DENSE_INIT_BULK_MAX_SCALE_RATIO * min_spacing,
                ).astype(np.float32)
                bulk_scales = np.repeat(bulk_radius[:, np.newaxis], 3, axis=1)
                bulk_rotations = np.repeat(np.eye(3, dtype=np.float32)[np.newaxis, :, :], bulk_count, axis=0)
                bulk_normals = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), bulk_count, axis=0)
            bulk_primitive_logits = np.full((bulk_count, 1), self.nonplanar_logit_value, dtype=np.float32)
            bulk_planarity = np.zeros((bulk_count, 1), dtype=np.float32)
            bulk_region_type = np.ones((bulk_count, 1), dtype=np.int64)
            bulk_opacity_seed = np.clip(interior_density_seed, 0.2, 0.9).astype(np.float32)
        else:
            bulk_scales = np.empty((0, 3), dtype=np.float32)
            bulk_rotations = np.empty((0, 3, 3), dtype=np.float32)
            bulk_normals = np.empty((0, 3), dtype=np.float32)
            bulk_primitive_logits = np.empty((0, 1), dtype=np.float32)
            bulk_planarity = np.empty((0, 1), dtype=np.float32)
            bulk_region_type = np.empty((0, 1), dtype=np.int64)
            bulk_opacity_seed = np.empty((0, 1), dtype=np.float32)

        all_points = np.concatenate((boundary_points, interior_points), axis=0)
        all_scales = np.concatenate((local_scales, bulk_scales), axis=0)
        all_rotation_matrices = np.concatenate((rotation_matrices, bulk_rotations), axis=0)
        all_normals = np.concatenate((explicit_normals, bulk_normals), axis=0)
        all_primitive_logits = np.concatenate((primitive_logits, bulk_primitive_logits), axis=0)
        all_planarity = np.concatenate((planarity, bulk_planarity), axis=0)
        all_material_id = np.concatenate((material_id, interior_material_id), axis=0)
        all_region_type = np.concatenate(
            (np.zeros((num_points, 1), dtype=np.int64), bulk_region_type),
            axis=0,
        )

        device = self._default_device()
        rotation_quaternions = matrix_to_quaternion(
            torch.tensor(all_rotation_matrices, dtype=torch.float32, device=device)
        ).detach()

        total_points = all_points.shape[0]
        fused_point_cloud = torch.tensor(all_points, dtype=torch.float32, device=device)
        feature_count = max(1, (self.max_sh_degree + 1) ** 2)
        features = torch.zeros((total_points, feature_count, 3), dtype=torch.float32, device=device)
        features[:, 0, :] = 0.5

        boundary_strength = np.clip(boundary_strength, 0.2, 0.9).astype(np.float32)
        if not np.any(np.isfinite(boundary_strength)):
            boundary_strength = np.full((num_points, 1), 0.4, dtype=np.float32)
        surface_opacities = torch.tensor(boundary_strength, dtype=torch.float32, device=device)
        if bulk_count > 0:
            bulk_opacities = torch.tensor(bulk_opacity_seed, dtype=torch.float32, device=device)
            opacity_values = torch.cat((surface_opacities, bulk_opacities), dim=0)
        else:
            opacity_values = surface_opacities
        opacities = self.inverse_opacity_activation(opacity_values)
        scales = torch.log(torch.tensor(all_scales, dtype=torch.float32, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, 0:1, :].contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, 1:, :].contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotation_quaternions.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._initialize_hybrid_metadata(
            total_points,
            fused_point_cloud.device,
            primitive_type_logits=torch.tensor(all_primitive_logits, dtype=torch.float32, device=device),
            normals=torch.tensor(all_normals, dtype=torch.float32, device=device),
            material_id=torch.tensor(all_material_id, dtype=torch.long, device=device),
            planarity=torch.tensor(all_planarity, dtype=torch.float32, device=device),
            region_type=torch.tensor(all_region_type, dtype=torch.long, device=device),
        )
        self.max_radii2D = torch.zeros((total_points,), dtype=torch.float32, device=device)
        self.clamp_surface_thickness(self.surface_thickness_max)

    @staticmethod
    def _apply_bulk_continuous_jitter(interior_points, spacing, volume_shape_dhw=None):
        points = np.asarray(interior_points, dtype=np.float32).reshape(-1, 3)
        if points.shape[0] == 0:
            return points
        spacing_z, spacing_y, spacing_x = [float(value) for value in spacing]
        spacing_xyz = np.asarray([spacing_x, spacing_y, spacing_z], dtype=np.float32)
        jitter = (np.random.random(points.shape).astype(np.float32) - 0.5) * spacing_xyz[np.newaxis, :]
        jittered = points + jitter
        if volume_shape_dhw is not None:
            depth, height, width = [int(value) for value in volume_shape_dhw]
            upper = np.asarray(
                [
                    max(width - 1, 0) * spacing_x,
                    max(height - 1, 0) * spacing_y,
                    max(depth - 1, 0) * spacing_z,
                ],
                dtype=np.float32,
            )
            jittered = np.minimum(np.maximum(jittered, np.zeros((3,), dtype=np.float32)), upper[np.newaxis, :])
        return jittered.astype(np.float32)

    def _resolve_boundary_samples(self, analysis):
        required = (
            "boundary_points",
            "boundary_normals",
            "boundary_tangent_u",
            "boundary_tangent_v",
            "boundary_strength",
            "boundary_material_id",
        )
        missing = [key for key in required if key not in analysis]
        if missing:
            raise ValueError(
                "Canonical Phase 1 bundle is missing boundary fields: " + ", ".join(missing)
            )
        return (
            np.asarray(analysis["boundary_points"], dtype=np.float32).reshape(-1, 3),
            np.asarray(analysis["boundary_normals"], dtype=np.float32).reshape(-1, 3),
            np.asarray(analysis["boundary_tangent_u"], dtype=np.float32).reshape(-1, 3),
            np.asarray(analysis["boundary_tangent_v"], dtype=np.float32).reshape(-1, 3),
            np.asarray(analysis["boundary_strength"], dtype=np.float32).reshape(-1, 1),
            np.asarray(analysis["boundary_material_id"], dtype=np.int64).reshape(-1, 1),
        )

    def _resolve_interior_samples(self, analysis):
        required = ("interior_points", "interior_density_seed", "interior_material_id")
        missing = [key for key in required if key not in analysis]
        if missing:
            raise ValueError(
                "Canonical Phase 1 bundle is missing interior fields: " + ", ".join(missing)
            )
        return (
            np.asarray(analysis["interior_points"], dtype=np.float32).reshape(-1, 3),
            np.asarray(analysis["interior_density_seed"], dtype=np.float32).reshape(-1, 1),
            np.asarray(analysis["interior_material_id"], dtype=np.int64).reshape(-1, 1),
        )

    def _estimate_nearest_neighbor_distance(self, points, default_value):
        if points.shape[0] <= 1:
            return np.full((points.shape[0],), default_value, dtype=np.float32)
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        nearest = distances[:, 1].astype(np.float32)
        nearest[nearest <= 1e-8] = default_value
        return nearest
