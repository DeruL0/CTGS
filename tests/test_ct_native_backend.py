import warnings
import unittest
from unittest import mock

import torch

from ct_pipeline.field_query import query_ct_density_python
from ct_pipeline import native_backend
from ct_pipeline.ct_slice_renderer import CTRenderState, prepare_ct_render_state, render_ct_slice_patch
from utils.ct_losses import point_to_plane_loss_from_cache, prepare_point_to_plane_cache


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
