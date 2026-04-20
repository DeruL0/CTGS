import warnings
import unittest
from unittest import mock

import numpy as np
import torch
from scipy import ndimage

from ct_pipeline.field_query import query_ct_density_python
from ct_pipeline import native_backend
from ct_pipeline.ct_slice_renderer import CTRenderState, prepare_ct_render_state, render_ct_slice_patch
from utils.ct_losses import material_boundary_loss, point_to_plane_loss_from_cache, prepare_point_to_plane_cache, sample_volume_field


class DummyGaussians:
    def __init__(self, xyz, scaling, rotation, opacity):
        self._xyz = xyz
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_opacity(self):
        return self._opacity


class CTNativeBackendTests(unittest.TestCase):
    def _build_signed_field_reference(self, material_mask: torch.Tensor, band_voxels: int) -> torch.Tensor:
        material_np = material_mask.detach().cpu().numpy().astype(bool)
        boundary = np.zeros_like(material_np, dtype=bool)
        for axis in range(3):
            slicer_a = [slice(None)] * 3
            slicer_b = [slice(None)] * 3
            slicer_a[axis] = slice(1, None)
            slicer_b[axis] = slice(None, -1)
            current = material_np[tuple(slicer_a)]
            previous = material_np[tuple(slicer_b)]
            change = current != previous
            boundary[tuple(slicer_a)] |= change
            boundary[tuple(slicer_b)] |= change
        if not np.any(boundary):
            return torch.zeros_like(material_mask, dtype=torch.float32)
        distance = ndimage.distance_transform_edt(~boundary).astype("float32")
        distance = np.clip(distance, 0.0, float(band_voxels))
        signed = np.where(material_np, distance, -distance)
        signed[boundary] = 0.0
        return torch.from_numpy(signed)

    def test_resolve_auto_falls_back_to_python_without_extension(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.object(native_backend, "has_ct_native_backend", return_value=False):
                backend = native_backend.resolve_ct_backend("auto")
        self.assertEqual(backend, "python")
        self.assertTrue(caught)

    def test_resolve_cuda_raises_without_extension(self):
        with mock.patch.object(native_backend, "has_ct_native_backend", return_value=False):
            with mock.patch("torch.cuda.is_available", return_value=True):
                with self.assertRaises(RuntimeError):
                    native_backend.resolve_ct_backend("cuda")

    def test_prepare_point_to_plane_cache_backend_matches_python_path(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(4, 1)
        planarity = torch.ones((4, 1), dtype=torch.float32)
        primitive_type = torch.ones((4, 1), dtype=torch.float32)
        material_ids = torch.zeros((4, 1), dtype=torch.long)
        neighbor_index = torch.tensor(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
            dtype=torch.long,
        )

        cache_python = prepare_point_to_plane_cache(
            xyz,
            normals,
            planarity,
            material_ids=material_ids,
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
        )
        cache_backend = native_backend.prepare_point_to_plane_cache_backend(
            "python",
            xyz,
            normals,
            planarity,
            material_ids=material_ids,
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
        )

        self.assertTrue(torch.equal(cache_python.active_indices, cache_backend.active_indices))
        self.assertTrue(torch.allclose(cache_python.centroids, cache_backend.centroids))
        self.assertTrue(torch.allclose(cache_python.fitted_normals, cache_backend.fitted_normals))
        self.assertTrue(torch.allclose(cache_python.weights, cache_backend.weights))
        self.assertTrue(torch.allclose(point_to_plane_loss_from_cache(xyz, cache_python), point_to_plane_loss_from_cache(xyz, cache_backend)))

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_native_density_query_matches_python_and_has_finite_gradients(self):
        means = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        rotations = torch.eye(3, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(2, 1, 1).requires_grad_(True)
        scales = torch.tensor([[1.0, 1.0, 1.0], [0.8, 0.9, 1.1]], dtype=torch.float32, device="cuda", requires_grad=True)
        opacity = torch.tensor([0.9, 0.6], dtype=torch.float32, device="cuda", requires_grad=True)
        query_points = torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.2, 0.1]], dtype=torch.float32, device="cuda")

        python_density = query_ct_density_python(means, rotations, scales, opacity, query_points)
        native_density = native_backend.query_ct_density_backend("cuda", means, rotations, scales, opacity, query_points)
        self.assertTrue(torch.allclose(python_density, native_density, atol=1e-4, rtol=1e-4))

        native_density.sum().backward()
        self.assertTrue(torch.isfinite(means.grad).all())
        self.assertTrue(torch.isfinite(rotations.grad).all())
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertTrue(torch.isfinite(opacity.grad).all())

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_local_density_query_matches_full_scan_and_has_finite_gradients(self):
        means = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [8.0, 8.0, 8.0]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        rotations = torch.eye(3, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(3, 1, 1).requires_grad_(True)
        scales = torch.tensor([[1.0, 1.0, 1.0], [0.7, 1.1, 0.8], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda", requires_grad=True)
        opacity = torch.tensor([0.9, 0.6, 0.2], dtype=torch.float32, device="cuda", requires_grad=True)
        query_points = torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.2, 0.1]], dtype=torch.float32, device="cuda")

        spatial_grid = native_backend.build_ct_spatial_grid(
            means.detach(),
            rotations.detach(),
            scales.detach(),
            spacing_zyx=(1.0, 1.0, 1.0),
            truncation_sigma=4.0,
            grid_cell_voxels=8,
        )
        self.assertIsNotNone(spatial_grid)

        full_density = native_backend.query_ct_density_native(means, rotations, scales, opacity, query_points)
        local_density = native_backend.query_ct_density_native(
            means,
            rotations,
            scales,
            opacity,
            query_points,
            spatial_grid=spatial_grid,
            support_extent=spatial_grid.support_extent,
        )
        self.assertTrue(torch.allclose(full_density, local_density, atol=1e-4, rtol=1e-4))

        local_density.sum().backward()
        self.assertTrue(torch.isfinite(means.grad).all())
        self.assertTrue(torch.isfinite(rotations.grad).all())
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertTrue(torch.isfinite(opacity.grad).all())

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_native_signed_field_matches_cpu_reference(self):
        material_mask = torch.zeros((7, 7, 7), dtype=torch.bool, device="cuda")
        material_mask[2:5, 2:5, 2:5] = True
        material_mask[3, 3, 3] = False

        native_signed = native_backend.build_signed_field_backend("cuda", material_mask, band_voxels=3).cpu()
        reference_signed = self._build_signed_field_reference(material_mask.cpu(), band_voxels=3)

        self.assertTrue(torch.allclose(native_signed, reference_signed, atol=1e-4, rtol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_native_neighbor_index_matches_exact_cpu_knn(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        native_neighbors = native_backend.build_neighbor_index_backend("cuda", xyz, k=2, tile_size=2).cpu()
        cpu_dist = torch.cdist(xyz.cpu(), xyz.cpu())
        cpu_dist.fill_diagonal_(float("inf"))
        cpu_neighbors = torch.topk(cpu_dist, k=2, dim=1, largest=False, sorted=True).indices
        self.assertTrue(torch.equal(native_neighbors, cpu_neighbors))

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_native_plane_cache_and_loss_match_python(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda").repeat(4, 1)
        planarity = torch.ones((4, 1), dtype=torch.float32, device="cuda")
        primitive_type = torch.ones((4, 1), dtype=torch.float32, device="cuda")
        material_ids = torch.zeros((4, 1), dtype=torch.long, device="cuda")
        neighbor_index = torch.tensor(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
            dtype=torch.long,
            device="cuda",
        )

        python_cache = prepare_point_to_plane_cache(
            xyz,
            normals,
            planarity,
            material_ids=material_ids,
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
        )
        native_cache = native_backend.prepare_point_to_plane_cache_backend(
            "cuda",
            xyz,
            normals,
            planarity,
            material_ids=material_ids,
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
        )
        self.assertTrue(torch.equal(python_cache.active_indices, native_cache.active_indices))
        self.assertTrue(torch.allclose(python_cache.centroids, native_cache.centroids, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(python_cache.fitted_normals, native_cache.fitted_normals, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(python_cache.weights, native_cache.weights, atol=1e-6, rtol=1e-6))

        native_loss = native_backend.point_to_plane_loss_backend("cuda", xyz, native_cache)
        python_loss = point_to_plane_loss_from_cache(xyz, python_cache)
        self.assertTrue(torch.allclose(native_loss, python_loss, atol=1e-5, rtol=1e-5))
        native_loss.backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_native_renderer_matches_python_and_has_finite_gradients(self):
        gaussians = DummyGaussians(
            xyz=torch.tensor([[4.0, 4.0, 4.0], [5.0, 4.0, 4.0]], dtype=torch.float32, device="cuda", requires_grad=True),
            scaling=torch.tensor([[1.0, 1.0, 1.0], [0.75, 1.25, 1.0]], dtype=torch.float32, device="cuda", requires_grad=True),
            rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda", requires_grad=True),
            opacity=torch.tensor([[1.0], [0.6]], dtype=torch.float32, device="cuda", requires_grad=True),
        )
        render_state = prepare_ct_render_state(gaussians)

        python_patch = render_ct_slice_patch(
            render_state,
            axis="z",
            slice_idx=4,
            patch_origin_hw=(0, 0),
            patch_size_hw=(9, 9),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_dhw=(9, 9, 9),
            gaussians_per_chunk=64,
        )
        native_patch = native_backend.render_ct_slice_patch_native(
            render_state,
            axis="z",
            slice_idx=4,
            patch_origin_hw=(0, 0),
            patch_size_hw=(9, 9),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_dhw=(9, 9, 9),
            gaussians_per_chunk=64,
        )
        self.assertTrue(torch.allclose(python_patch, native_patch, atol=1e-4, rtol=1e-4))

        native_patch.sum().backward()
        self.assertTrue(torch.isfinite(gaussians.get_xyz.grad).all())
        self.assertTrue(torch.isfinite(gaussians.get_scaling.grad).all())
        self.assertTrue(torch.isfinite(gaussians.get_rotation.grad).all())
        self.assertTrue(torch.isfinite(gaussians.get_opacity.grad).all())

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_local_renderer_matches_full_scan_and_has_finite_gradients(self):
        means = torch.tensor([[4.0, 4.0, 4.0], [5.0, 4.0, 4.0], [20.0, 20.0, 20.0]], dtype=torch.float32, device="cuda", requires_grad=True)
        rotations = torch.eye(3, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(3, 1, 1).requires_grad_(True)
        scales = torch.tensor([[1.0, 1.0, 1.0], [0.75, 1.25, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda", requires_grad=True)
        opacity = torch.tensor([1.0, 0.6, 0.1], dtype=torch.float32, device="cuda", requires_grad=True)

        spatial_grid = native_backend.build_ct_spatial_grid(
            means.detach(),
            rotations.detach(),
            scales.detach(),
            spacing_zyx=(1.0, 1.0, 1.0),
            truncation_sigma=4.0,
            grid_cell_voxels=8,
        )
        self.assertIsNotNone(spatial_grid)

        full_state = CTRenderState(
            means=means,
            rotations=rotations,
            scales=scales,
            opacity=opacity,
            radius=torch.ones((3,), dtype=torch.float32, device="cuda"),
        )
        local_state = CTRenderState(
            means=means,
            rotations=rotations,
            scales=scales,
            opacity=opacity,
            radius=torch.ones((3,), dtype=torch.float32, device="cuda"),
            support_extent=spatial_grid.support_extent,
            spatial_grid=spatial_grid,
            truncation_sigma=4.0,
        )

        full_patch = native_backend.render_ct_slice_patch_native(
            full_state,
            axis="z",
            slice_idx=4,
            patch_origin_hw=(0, 0),
            patch_size_hw=(9, 9),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_dhw=(32, 32, 32),
            gaussians_per_chunk=64,
            slice_tile_size=8,
        )
        local_patch = native_backend.render_ct_slice_patch_native(
            local_state,
            axis="z",
            slice_idx=4,
            patch_origin_hw=(0, 0),
            patch_size_hw=(9, 9),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_dhw=(32, 32, 32),
            gaussians_per_chunk=64,
            slice_tile_size=8,
        )
        self.assertTrue(torch.allclose(full_patch, local_patch, atol=1e-4, rtol=1e-4))

        local_patch.sum().backward()
        self.assertTrue(torch.isfinite(means.grad).all())
        self.assertTrue(torch.isfinite(rotations.grad).all())
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertTrue(torch.isfinite(opacity.grad).all())

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_native_boundary_field_sampling_matches_python_and_has_finite_gradients(self):
        strength_volume = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32, device="cuda")
        strength_volume[0, 0, 1:3, 1:3, 1:3] = 1.0
        normal_volume = torch.zeros((1, 3, 4, 4, 4), dtype=torch.float32, device="cuda")
        normal_volume[0, 0] = 1.0
        query_points = torch.tensor(
            [[1.5, 1.5, 1.5], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )

        python_strength = sample_volume_field(strength_volume, query_points, spacing_zyx=(1.0, 1.0, 1.0)).reshape(-1)
        python_normals = sample_volume_field(normal_volume, query_points, spacing_zyx=(1.0, 1.0, 1.0))
        native_strength, native_normals = native_backend.sample_boundary_field_backend(
            "cuda",
            strength_volume,
            normal_volume,
            query_points,
            spacing_zyx=(1.0, 1.0, 1.0),
        )

        self.assertTrue(torch.allclose(python_strength, native_strength, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(python_normals, native_normals, atol=1e-4, rtol=1e-4))

        (native_strength.sum() + native_normals.sum()).backward()
        self.assertTrue(torch.isfinite(query_points.grad).all())

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_native_surface_thickness_loss_matches_python_and_has_finite_gradients(self):
        raw_scaling = torch.tensor(
            [[-0.4, -0.3, -1.8], [-0.2, -0.1, -1.2]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        rotation_mats = torch.eye(3, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(2, 1, 1).requires_grad_(True)
        normals_raw = torch.tensor(
            [[0.0, 0.0, 1.0], [0.2, 0.0, 0.98]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        normals = torch.nn.functional.normalize(normals_raw, dim=-1)

        scales = torch.exp(raw_scaling)
        local_normals = torch.einsum("nij,nj->ni", rotation_mats.transpose(1, 2), torch.nn.functional.normalize(normals, dim=-1))
        python_loss = torch.relu(torch.sqrt(torch.sum((local_normals * scales) ** 2, dim=-1).clamp_min(1e-8)) - 0.1).mean()
        native_loss = native_backend.surface_thickness_loss_backend(
            "cuda",
            raw_scaling,
            rotation_mats,
            normals,
            max_thickness=0.1,
        )
        self.assertTrue(torch.allclose(python_loss, native_loss, atol=1e-5, rtol=1e-5))

        native_loss.backward()
        self.assertTrue(torch.isfinite(raw_scaling.grad).all())
        self.assertTrue(torch.isfinite(rotation_mats.grad).all())
        self.assertTrue(torch.isfinite(normals_raw.grad).all())

    @unittest.skipUnless(torch.cuda.is_available() and native_backend.has_ct_native_backend(), "CUDA and ct_native_backend are required")
    def test_native_material_boundary_loss_matches_python_and_has_finite_gradients(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        material_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.long, device="cuda")
        opacity = torch.tensor([[0.2], [0.8], [0.4], [0.9]], dtype=torch.float32, device="cuda", requires_grad=True)
        neighbor_index = torch.tensor(
            [
                [1, 2],
                [0, 3],
                [0, 3],
                [1, 2],
            ],
            dtype=torch.long,
            device="cuda",
        )

        python_loss = material_boundary_loss(
            xyz,
            material_ids,
            opacity,
            neighbor_index=neighbor_index,
            target_opacity=0.5,
        )
        native_loss = native_backend.material_boundary_loss_backend(
            "cuda",
            xyz,
            material_ids.reshape(-1),
            opacity.reshape(-1),
            neighbor_index=neighbor_index,
            target_opacity=0.5,
        )
        self.assertTrue(torch.allclose(python_loss, native_loss, atol=1e-5, rtol=1e-5))

        native_loss.backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())
        self.assertTrue(torch.isfinite(opacity.grad).all())
