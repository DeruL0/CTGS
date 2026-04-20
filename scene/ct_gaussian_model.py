import json
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from ct_pipeline.ct_preprocessor import CTPreprocessor
from ct_pipeline.geometry_analyzer import GeometryAnalyzer
from utils.rotation_utils import matrix_to_quaternion
from .gaussian_model import GaussianModel


def _normalize_np(vector):
    norm = np.linalg.norm(vector)
    if norm <= 1e-8:
        return None
    return vector / norm


def _orthogonal_hint(normal):
    hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(hint, normal)) > 0.9:
        hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent = hint - np.dot(hint, normal) * normal
    tangent = _normalize_np(tangent)
    if tangent is None:
        tangent = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        tangent = tangent - np.dot(tangent, normal) * normal
        tangent = _normalize_np(tangent)
    return tangent


def _build_frame_from_normal(normal, tangent_hint=None):
    normal = _normalize_np(np.asarray(normal, dtype=np.float32))
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    if tangent_hint is None:
        tangent_u = _orthogonal_hint(normal)
    else:
        tangent_u = np.asarray(tangent_hint, dtype=np.float32)
        tangent_u = tangent_u - np.dot(tangent_u, normal) * normal
        tangent_u = _normalize_np(tangent_u)
        if tangent_u is None:
            tangent_u = _orthogonal_hint(normal)

    tangent_v = _normalize_np(np.cross(normal, tangent_u))
    if tangent_v is None:
        tangent_u = _orthogonal_hint(normal)
        tangent_v = _normalize_np(np.cross(normal, tangent_u))
    tangent_u = _normalize_np(np.cross(tangent_v, normal))
    return tangent_u.astype(np.float32), tangent_v.astype(np.float32), normal.astype(np.float32)


class CTGaussianModel(GaussianModel):
    """CT-aware GaussianModel that initializes hybrid primitives from Phase 1 analysis."""

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.single_material_fallback = False
        self.surface_thickness_max = None

    def create_from_phase1_bundle(
        self,
        analysis_path,
        metadata_path,
        spatial_lr_scale,
        surface_thickness_max=None,
        planar_thickness_max=None,
        volume=None,
        bulk_points_ratio: float = 1.0,
        bulk_boundary_margin_voxels: int = 2,
    ):
        analysis_path = Path(analysis_path)
        metadata_path = Path(metadata_path)
        with np.load(str(analysis_path)) as analysis_npz:
            analysis = {key: analysis_npz[key] for key in analysis_npz.files}
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        spacing = tuple(float(value) for value in metadata["spacing_zyx"])
        max_material_classes = int(metadata.get("parameters", {}).get("max_material_classes", 3))
        sigma = float(metadata.get("parameters", {}).get("sigma", 1.0))
        boundary_points, boundary_normals, boundary_tangent_u, boundary_tangent_v, boundary_strength, boundary_material_id = self._resolve_boundary_samples(
            analysis,
            volume=volume,
            spacing=spacing,
            sigma=sigma,
            max_material_classes=max_material_classes,
        )
        interior_points, interior_density_seed, interior_material_id = self._resolve_interior_samples(
            analysis,
            volume=volume,
            spacing=spacing,
            bulk_points_ratio=bulk_points_ratio,
            bulk_boundary_margin_voxels=bulk_boundary_margin_voxels,
            max_material_classes=max_material_classes,
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
        )

    def create_from_ct_volume(
        self,
        volume,
        spacing,
        analyzer,
        spatial_lr_scale=1.0,
        surface_thickness_max=None,
        planar_thickness_max=None,
        bulk_points_ratio: float = 1.0,
        bulk_boundary_margin_voxels: int = 2,
        max_material_classes: int = 3,
        support_threshold_mode: str = "otsu",
        surface_gradient_percentile: float = 60.0,
    ):
        preprocessor = CTPreprocessor()
        support = preprocessor.segment_coarse_support(volume, threshold_mode=support_threshold_mode)
        support_mask = support["support_mask"]
        if not np.any(support_mask):
            raise ValueError("CT volume support extraction produced an empty solid mask.")

        boundary_points = preprocessor.extract_intensity_surface_points(
            volume,
            support_mask,
            spacing,
            sigma=float(getattr(analyzer, "sigma", 1.0)),
            gradient_percentile=surface_gradient_percentile,
        )
        boundary_material_id = np.zeros((boundary_points.shape[0], 1), dtype=np.int64)
        interior_target_count = max(1, int(round(boundary_points.shape[0] * float(bulk_points_ratio))))
        interior_points, interior_density_seed, interior_material_id = preprocessor.sample_support_points(
            support_mask,
            volume,
            spacing,
            target_count=interior_target_count,
            boundary_margin_voxels=bulk_boundary_margin_voxels,
        )
        boundary_normals, boundary_tangent_u, boundary_tangent_v, boundary_strength = analyzer.estimate_boundary_geometry(
            boundary_points,
            volume,
            spacing,
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

    def clamp_planar_thickness(self, max_thickness: float):
        self.clamp_surface_thickness(max_thickness)

    def get_normals(self) -> torch.Tensor:
        return super().get_normals()

    def harden_primitive_types(self):
        self.primitive_types_hardened = True
        self._freeze_primitive_type_parameter()

    def post_optimizer_step(self, iteration):
        del iteration
        if self.surface_thickness_max is not None:
            self.clamp_surface_thickness(self.surface_thickness_max)

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

        num_points = boundary_points.shape[0]
        if num_points == 0:
            raise ValueError("Phase 1 analysis bundle does not contain any boundary points.")

        self.spatial_lr_scale = spatial_lr_scale
        min_spacing = float(min(spacing))
        self.surface_thickness_max = float(surface_thickness_max) if surface_thickness_max is not None else 0.5 * min_spacing
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
            tangential_scale = max(float(nearest_neighbor_distance[point_index]), min_spacing)
            local_scales[point_index] = np.array(
                [tangential_scale, tangential_scale, self.surface_thickness_max],
                dtype=np.float32,
            )

        local_scales = np.clip(local_scales, a_min=min_spacing * 0.25, a_max=None)
        local_scales[:, 2] = np.minimum(local_scales[:, 2], self.surface_thickness_max)
        primitive_logits = np.full((num_points, 1), self.nonplanar_logit_value, dtype=np.float32)
        planarity = np.zeros((num_points, 1), dtype=np.float32)

        bulk_count = interior_points.shape[0]
        if bulk_count > 0:
            bulk_nn = self._estimate_nearest_neighbor_distance(interior_points, default_value=min_spacing)
            bulk_radius = np.clip(0.75 * bulk_nn, a_min=0.5 * min_spacing, a_max=2.0 * min_spacing).astype(np.float32)
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

        rotation_quaternions = matrix_to_quaternion(
            torch.tensor(all_rotation_matrices, dtype=torch.float32, device="cuda")
        ).detach()

        total_points = all_points.shape[0]
        fused_point_cloud = torch.tensor(all_points, dtype=torch.float32, device="cuda")
        feature_count = max(1, (self.max_sh_degree + 1) ** 2)
        features = torch.zeros((total_points, feature_count, 3), dtype=torch.float32, device="cuda")
        features[:, 0, :] = 0.5

        boundary_strength = np.clip(boundary_strength, 0.2, 0.9).astype(np.float32)
        if not np.any(np.isfinite(boundary_strength)):
            boundary_strength = np.full((num_points, 1), 0.4, dtype=np.float32)
        surface_opacities = torch.tensor(boundary_strength, dtype=torch.float32, device="cuda")
        if bulk_count > 0:
            bulk_opacities = torch.tensor(bulk_opacity_seed, dtype=torch.float32, device="cuda")
            opacity_values = torch.cat((surface_opacities, bulk_opacities), dim=0)
        else:
            opacity_values = surface_opacities
        opacities = self.inverse_opacity_activation(opacity_values)
        scales = torch.log(torch.tensor(all_scales, dtype=torch.float32, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, 0:1, :].contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, 1:, :].contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotation_quaternions.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._initialize_hybrid_metadata(
            total_points,
            fused_point_cloud.device,
            primitive_type_logits=torch.tensor(all_primitive_logits, dtype=torch.float32, device="cuda"),
            normals=torch.tensor(all_normals, dtype=torch.float32, device="cuda"),
            material_id=torch.tensor(all_material_id, dtype=torch.long, device="cuda"),
            planarity=torch.tensor(all_planarity, dtype=torch.float32, device="cuda"),
            region_type=torch.tensor(all_region_type, dtype=torch.long, device="cuda"),
        )
        self.primitive_types_hardened = True
        self.max_radii2D = torch.zeros((total_points,), dtype=torch.float32, device="cuda")
        self.clamp_surface_thickness(self.surface_thickness_max)

    def _resolve_boundary_samples(self, analysis, volume, spacing, sigma, max_material_classes):
        if "boundary_points" in analysis and analysis["boundary_points"].size > 0:
            boundary_points = np.asarray(analysis["boundary_points"], dtype=np.float32).reshape(-1, 3)
            boundary_normals = np.asarray(analysis["boundary_normals"], dtype=np.float32).reshape(-1, 3)
            boundary_tangent_u = np.asarray(analysis["boundary_tangent_u"], dtype=np.float32).reshape(-1, 3)
            boundary_tangent_v = np.asarray(analysis["boundary_tangent_v"], dtype=np.float32).reshape(-1, 3)
            boundary_strength = np.asarray(analysis["boundary_strength"], dtype=np.float32).reshape(-1, 1)
            boundary_material_id = np.asarray(analysis["boundary_material_id"], dtype=np.int64).reshape(-1, 1)
            return (
                boundary_points,
                boundary_normals,
                boundary_tangent_u,
                boundary_tangent_v,
                boundary_strength,
                boundary_material_id,
            )

        if volume is not None:
            warnings.warn(
                "Phase 1 bundle is missing intensity-driven surface seeds; recomputing them from the CT volume.",
                RuntimeWarning,
                stacklevel=2,
            )
            preprocessor = CTPreprocessor()
            analyzer = GeometryAnalyzer(sigma=sigma)
            if "coarse_support_mask" in analysis:
                support_mask = np.asarray(analysis["coarse_support_mask"], dtype=bool)
            elif "material_mask" in analysis:
                support_mask = np.asarray(analysis["material_mask"], dtype=bool)
            else:
                support = preprocessor.segment_coarse_support(
                    np.asarray(volume, dtype=np.float32),
                    threshold_mode="otsu",
                )
                support_mask = support["support_mask"]
            boundary_points = preprocessor.extract_intensity_surface_points(
                np.asarray(volume, dtype=np.float32),
                support_mask,
                spacing,
                sigma=sigma,
                gradient_percentile=60.0,
            )
            boundary_material_id = np.zeros((boundary_points.shape[0], 1), dtype=np.int64)
            boundary_normals, boundary_tangent_u, boundary_tangent_v, boundary_strength = analyzer.estimate_boundary_geometry(
                boundary_points,
                np.asarray(volume, dtype=np.float32),
                spacing,
            )
            return (
                boundary_points,
                boundary_normals,
                boundary_tangent_u,
                boundary_tangent_v,
                boundary_strength,
                boundary_material_id,
            )

        if "surface_points" in analysis and analysis["surface_points"].size > 0:
            warnings.warn(
                "Phase 1 bundle is legacy surface-first data; surface samples are being used as a fallback boundary initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
            boundary_points = np.asarray(analysis["surface_points"], dtype=np.float32).reshape(-1, 3)
            boundary_normals = np.asarray(
                analysis["surface_normals"] if "surface_normals" in analysis else np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (boundary_points.shape[0], 1)),
                dtype=np.float32,
            ).reshape(-1, 3)
            boundary_tangent_u = np.zeros_like(boundary_normals, dtype=np.float32)
            boundary_tangent_v = np.zeros_like(boundary_normals, dtype=np.float32)
            for point_index in range(boundary_points.shape[0]):
                tangent_u, tangent_v, normal = _build_frame_from_normal(boundary_normals[point_index])
                boundary_normals[point_index] = normal
                boundary_tangent_u[point_index] = tangent_u
                boundary_tangent_v[point_index] = tangent_v
            boundary_strength = np.full((boundary_points.shape[0], 1), 0.5, dtype=np.float32)
            boundary_material_id = np.asarray(
                analysis["surface_material_id"] if "surface_material_id" in analysis else analysis["material_id"] if "material_id" in analysis else np.zeros((boundary_points.shape[0], 1), dtype=np.int64),
                dtype=np.int64,
            ).reshape(-1, 1)
            return (
                boundary_points,
                boundary_normals,
                boundary_tangent_u,
                boundary_tangent_v,
                boundary_strength,
                boundary_material_id,
            )

        raise ValueError("Unable to resolve boundary samples from Phase 1 bundle.")

    def _resolve_interior_samples(self, analysis, volume, spacing, bulk_points_ratio, bulk_boundary_margin_voxels, max_material_classes):
        if "interior_points" in analysis and analysis["interior_points"].size > 0:
            density_seed = analysis["interior_density_seed"] if "interior_density_seed" in analysis else np.full(
                (analysis["interior_points"].shape[0], 1),
                0.5,
                dtype=np.float32,
            )
            material_id = analysis["interior_material_id"] if "interior_material_id" in analysis else np.zeros(
                (analysis["interior_points"].shape[0], 1),
                dtype=np.int64,
            )
            return (
                np.asarray(analysis["interior_points"], dtype=np.float32).reshape(-1, 3),
                np.asarray(density_seed, dtype=np.float32).reshape(-1, 1),
                np.asarray(material_id, dtype=np.int64).reshape(-1, 1),
            )

        if volume is None:
            warnings.warn(
                "Phase 1 bundle does not contain interior_points; continuing with a surface-only initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.int64),
            )

        if "boundary_points" in analysis:
            surface_count = int(np.asarray(analysis["boundary_points"]).shape[0])
        elif "surface_points" in analysis:
            surface_count = int(np.asarray(analysis["surface_points"]).shape[0])
        else:
            surface_count = 0
        target_count = max(1, int(round(surface_count * float(bulk_points_ratio))))
        preprocessor = CTPreprocessor()
        if "coarse_support_mask" in analysis:
            support_mask = np.asarray(analysis["coarse_support_mask"], dtype=bool)
        elif "material_mask" in analysis:
            support_mask = np.asarray(analysis["material_mask"], dtype=bool)
        else:
            support = preprocessor.segment_coarse_support(
                np.asarray(volume, dtype=np.float32),
                threshold_mode="otsu",
            )
            support_mask = support["support_mask"]
            warnings.warn(
                "Phase 1 bundle is legacy surface-only data; coarse solid support was recomputed from the CT volume for bulk initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
        interior_points, interior_density_seed, interior_material_id = preprocessor.sample_support_points(
            support_mask,
            np.asarray(volume, dtype=np.float32),
            spacing,
            target_count=target_count,
            boundary_margin_voxels=bulk_boundary_margin_voxels,
        )
        return interior_points, interior_density_seed, interior_material_id

    def _estimate_nearest_neighbor_distance(self, points, default_value):
        if points.shape[0] <= 1:
            return np.full((points.shape[0],), default_value, dtype=np.float32)
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        nearest = distances[:, 1].astype(np.float32)
        nearest[nearest <= 1e-8] = default_value
        return nearest

    def _estimate_local_gaussian_geometry(self, points, surface_normals, nearest_neighbor_distance, min_spacing, k_neighbors=12):
        num_points = points.shape[0]
        local_scales = np.zeros((num_points, 3), dtype=np.float32)
        rotation_matrices = np.zeros((num_points, 3, 3), dtype=np.float32)

        if num_points == 1:
            tangent_u, tangent_v, normal = _build_frame_from_normal(surface_normals[0])
            rotation_matrices[0] = np.stack((tangent_u, tangent_v, normal), axis=1)
            local_scales[0] = np.array([min_spacing, min_spacing, min_spacing], dtype=np.float32)
            return local_scales, rotation_matrices

        tree = cKDTree(points)
        k = min(max(k_neighbors, 3), num_points)
        _, neighbor_indices = tree.query(points, k=k)
        if neighbor_indices.ndim == 1:
            neighbor_indices = neighbor_indices[:, np.newaxis]

        min_scale = max(min_spacing * 0.25, 1e-4)
        for point_index in range(num_points):
            neighborhood = points[np.unique(neighbor_indices[point_index])]
            centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / max(neighborhood.shape[0] - 1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.clip(eigenvalues[order], a_min=min_scale ** 2, a_max=None)
            tangent_hint = eigenvectors[:, order[0]]
            tangent_u, tangent_v, normal = _build_frame_from_normal(surface_normals[point_index], tangent_hint=tangent_hint)
            rotation_matrices[point_index] = np.stack((tangent_u, tangent_v, normal), axis=1)
            local_scales[point_index] = np.maximum(np.sqrt(eigenvalues).astype(np.float32), nearest_neighbor_distance[point_index])

        return local_scales, rotation_matrices
