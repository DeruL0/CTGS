import os
import unittest
from unittest import mock

import torch

from ct_pipeline import backend
from ct_pipeline.backend.query import _normalize_boundary_volumes
from ct_pipeline.rendering.fields import query_ct_density_python


class CTNativeBackendContractTests(unittest.TestCase):
    def test_public_backend_package_exposes_active_api(self):
        self.assertTrue(callable(backend.prepare_ct_training_state))
        self.assertTrue(callable(backend.query_ct_density_native))
        self.assertTrue(callable(backend.render_ct_slice_patch_native))
        self.assertFalse(hasattr(backend, "build_neighbor_index_backend"))
        self.assertFalse(hasattr(backend, "point_to_plane_loss_backend"))

    def test_require_backend_rejects_missing_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                backend.require_ct_native_backend()

    def test_native_density_query_rejects_cpu_tensors(self):
        means = torch.zeros((1, 3))
        rotations = torch.eye(3).reshape(1, 3, 3)
        scales = torch.ones((1, 3))
        opacity = torch.ones((1,))
        points = torch.zeros((1, 3))
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA tensor"):
            backend.query_ct_density_native(means, rotations, scales, opacity, points)

    def test_normalize_boundary_volumes_accepts_channel_first(self):
        strength = torch.zeros((1, 1, 2, 3, 4))
        normals = torch.zeros((1, 3, 2, 3, 4))
        normalized_strength, normalized_normals = _normalize_boundary_volumes(strength, normals)
        self.assertEqual(tuple(normalized_strength.shape), (2, 3, 4))
        self.assertEqual(tuple(normalized_normals.shape), (2, 3, 4, 3))


@unittest.skipUnless(
    os.environ.get("CTGS_RUN_CUDA_TESTS") == "1"
    and torch.cuda.is_available()
    and backend.has_ct_native_backend(),
    "Set CTGS_RUN_CUDA_TESTS=1 with CUDA and ct_native_backend available",
)
class CTNativeBackendCudaTests(unittest.TestCase):
    def test_native_density_matches_python_and_has_finite_gradients(self):
        means = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], device="cuda", requires_grad=True)
        rotations = torch.eye(3, device="cuda").reshape(1, 3, 3).repeat(2, 1, 1).requires_grad_(True)
        scales = torch.tensor([[1.0, 1.0, 1.0], [0.8, 0.9, 1.1]], device="cuda", requires_grad=True)
        opacity = torch.tensor([0.9, 0.6], device="cuda", requires_grad=True)
        points = torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.2, 0.1]], device="cuda")

        expected = query_ct_density_python(means, rotations, scales, opacity, points)
        actual = backend.query_ct_density_native(means, rotations, scales, opacity, points)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-4, rtol=1e-4))

        actual.sum().backward()
        self.assertTrue(torch.isfinite(means.grad).all())
        self.assertTrue(torch.isfinite(rotations.grad).all())
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertTrue(torch.isfinite(opacity.grad).all())

    def test_signed_field_builder_returns_expected_shape(self):
        material_mask = torch.zeros((5, 5, 5), dtype=torch.bool, device="cuda")
        material_mask[1:4, 1:4, 1:4] = True
        signed = backend.build_signed_field_native(material_mask, band_voxels=3)
        self.assertEqual(tuple(signed.shape), tuple(material_mask.shape))
        self.assertTrue(torch.isfinite(signed).all())


if __name__ == "__main__":
    unittest.main()
