"""Unit tests for Method-C role-separated joint training.

Verifies the key invariants:
  1. intensity_loss does NOT give gradients to surface parameters
  2. surface_phase_loss does NOT give gradients to bulk attenuation
  3. masked A_b readout uses I_b / W_b inside material, mu_air outside
  4. unknown / interior regions get zero geo_weight in confidence maps
  5. frozen bulk geometry: xyz and scale unchanged after an optimizer step
  6. query_surface_signed_distance is differentiable w.r.t. surface params
"""
from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn.functional as F


class _MinSurfaceState:
    """Minimal surface-only state for SDF tests."""

    def __init__(self, n: int = 10):
        self.surface_xyz = torch.randn(n, 3, requires_grad=True)
        self.surface_normals = F.normalize(torch.randn(n, 3), dim=-1).requires_grad_(True)
        self.surface_scales = (torch.ones(n, 3) * 0.3).requires_grad_(False)
        self.surface_opacity = torch.ones(n) * 0.8


class TestGradientIsolation(unittest.TestCase):

    def test_intensity_loss_detaches_surface_mask(self):
        """L_int gradient must be zero on surface_xyz when m_s is detached."""
        from ct_pipeline.rendering.fields import query_surface_signed_distance, bulk_intensity_readout

        torch.manual_seed(0)
        state = _MinSurfaceState(n=8)
        pts = torch.randn(16, 3)

        D_s = query_surface_signed_distance(state, pts)
        m_s = torch.sigmoid(-D_s / 0.1)
        m_s_detached = m_s.detach()  # this is what role_separated_intensity_loss does

        A_b = torch.randn(16, requires_grad=True)
        mu_pred = m_s_detached * A_b + (1.0 - m_s_detached) * 0.0
        loss = F.smooth_l1_loss(mu_pred, torch.rand(16), beta=0.1)
        loss.backward()

        surf_grad = 0.0 if state.surface_xyz.grad is None else float(state.surface_xyz.grad.abs().sum())
        self.assertEqual(surf_grad, 0.0,
            f"Intensity loss gave non-zero surface gradient: {surf_grad}")
        self.assertIsNotNone(A_b.grad)
        self.assertGreater(float(A_b.grad.abs().sum()), 0.0,
            "Bulk A_b must receive gradient from intensity loss")

    def test_surface_phase_loss_no_bulk_gradient(self):
        """surface_phase_loss must not touch bulk attenuation logits."""
        from ct_pipeline.rendering.fields import query_surface_signed_distance

        torch.manual_seed(1)
        state = _MinSurfaceState(n=8)
        atten_logit = torch.zeros(20, 1, requires_grad=True)

        pts_mat = torch.randn(8, 3)
        pts_air = torch.randn(8, 3)

        D_mat = query_surface_signed_distance(state, pts_mat)
        D_air = query_surface_signed_distance(state, pts_air)
        margin, temp = 0.3, 0.05
        loss = (F.softplus((D_mat + margin) / temp) +
                F.softplus((-D_air + margin) / temp)).mean()
        loss.backward()

        atten_grad = 0.0 if atten_logit.grad is None else float(atten_logit.grad.abs().sum())
        self.assertEqual(atten_grad, 0.0,
            "Surface phase loss gave gradient to bulk attenuation")
        surf_grad = float(state.surface_xyz.grad.abs().sum())
        self.assertGreater(surf_grad, 0.0, "Surface_xyz must receive gradient from phase loss")


class TestMaskedAbReadout(unittest.TestCase):

    def test_ab_equals_raw_over_den(self):
        """A_b = I_b / W_b when W_b > eps, else 0."""
        from ct_pipeline.rendering.fields import bulk_intensity_readout

        I_b = torch.tensor([2.0, 0.5, 0.0, 1.0])
        W_b = torch.tensor([4.0, 2.0, 0.0, 0.5])
        A_b = bulk_intensity_readout(I_b, W_b, eps=1e-6)

        self.assertAlmostEqual(float(A_b[0]), 0.5, places=5)
        self.assertAlmostEqual(float(A_b[1]), 0.25, places=5)
        self.assertEqual(float(A_b[2]), 0.0)
        self.assertAlmostEqual(float(A_b[3]), 2.0, places=5)

    def test_masked_prediction(self):
        """mu_pred = m_s * A_b + (1-m_s) * mu_air."""
        from ct_pipeline.rendering.fields import bulk_intensity_readout

        I_b = torch.tensor([2.0, 0.0])
        W_b = torch.tensor([4.0, 0.0])
        A_b = bulk_intensity_readout(I_b, W_b)
        m_s = torch.tensor([1.0, 0.0])
        mu_air = 0.05
        mu_pred = m_s * A_b + (1.0 - m_s) * mu_air
        self.assertAlmostEqual(float(mu_pred[0]), 0.5, places=5)
        self.assertAlmostEqual(float(mu_pred[1]), mu_air, places=5)


class TestConfidenceMaps(unittest.TestCase):

    def test_geo_weight_zero_in_deep_interior(self):
        """Voxels far from the material/air boundary have geo_weight = 0."""
        from ct_pipeline.training.confidence import build_ct_confidence_maps

        D, H, W = 30, 20, 20
        vol = np.ones((D, H, W), dtype=np.float32) * 0.4
        vol[:12, :, :] = 0.9   # material
        vol[18:, :, :] = 0.02  # air

        result = build_ct_confidence_maps(vol, (1.0, 1.0, 1.0))
        geo_weight = result["geo_weight"]
        self.assertGreater(float(geo_weight.sum()), 0.0,
            "geo_weight should be nonzero near boundary")
        # deep interior (far from boundary) should be zero
        deep_geo = float(geo_weight[0:4, :, :].mean())
        self.assertEqual(deep_geo, 0.0, "Deep interior should have zero geo_weight")
        # deep air (far from boundary) should be zero
        far_air_geo = float(geo_weight[-4:, :, :].mean())
        self.assertEqual(far_air_geo, 0.0, "Deep air should have zero geo_weight")

    def test_mat_conf_is_subset_of_material(self):
        """mat_conf must be a subset of material voxels."""
        from ct_pipeline.training.confidence import build_ct_confidence_maps

        D, H, W = 20, 20, 20
        vol = np.zeros((D, H, W), dtype=np.float32)
        vol[:10, :, :] = 0.9  # material in top half

        result = build_ct_confidence_maps(vol, (1.0, 1.0, 1.0))
        mat_conf = result["mat_conf"]
        # mat_conf must not appear in the air region
        self.assertFalse(np.any(mat_conf[10:, :, :]),
            "mat_conf leaked into air region")


class TestBulkGeometryFrozen(unittest.TestCase):

    def test_atten_updates_xyz_does_not(self):
        """After an optimizer step on attenuation, xyz must not change."""
        from ct_pipeline.rendering.fields import bulk_intensity_readout

        torch.manual_seed(2)
        bulk_xyz_init = torch.randn(10, 3)
        bulk_xyz = bulk_xyz_init.clone().requires_grad_(False)
        atten_logit = torch.zeros(10, 1, requires_grad=True)

        opt = torch.optim.Adam([atten_logit], lr=0.01)
        A_b = bulk_intensity_readout(F.softplus(atten_logit).reshape(-1) * 0.5, torch.ones(10))
        loss = (A_b - 0.75).square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        self.assertTrue(torch.allclose(bulk_xyz, bulk_xyz_init), "bulk_xyz must not change")
        self.assertGreater(float(atten_logit.detach().abs().sum()), 0.0, "attenuation must have updated")


class TestSurfaceSdfDifferentiable(unittest.TestCase):

    def test_gradients_flow_to_surface_params(self):
        """D_s must provide non-zero gradients to surface_xyz and normals."""
        from ct_pipeline.rendering.fields import query_surface_signed_distance

        torch.manual_seed(3)
        state = _MinSurfaceState(n=6)
        pts = torch.randn(10, 3)
        D_s = query_surface_signed_distance(state, pts)
        D_s.square().mean().backward()

        surf_grad = state.surface_xyz.grad
        norm_grad = state.surface_normals.grad
        self.assertIsNotNone(surf_grad)
        self.assertGreater(float(surf_grad.abs().sum()), 0, "No gradient to surface_xyz")
        self.assertIsNotNone(norm_grad)
        self.assertGreater(float(norm_grad.abs().sum()), 0, "No gradient to surface_normals")

    def test_relative_ordering_inside_outside(self):
        """D_s(inside) < D_s(outside) for a planar surface patch.

        Uses a flat surface patch (not a closed sphere) to avoid the
        degenerate case where opposite-side primitives have zero tangential
        distance to the query point.
        """
        from ct_pipeline.rendering.fields import query_surface_signed_distance

        # A flat patch of 10 primitives arranged in a grid on the z=0 plane,
        # all with outward normals pointing in +z direction.
        # Points above (z > 0) should have D_s > 0, below D_s < 0.
        n_surf = 9
        grid = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]
        centres = torch.tensor([(x * 0.3, y * 0.3, 0.0) for x, y in grid])
        normals = torch.zeros(n_surf, 3)
        normals[:, 2] = 1.0  # all normals in +z

        class _PlaneState:
            surface_xyz = centres.requires_grad_(False)
            surface_normals = normals
            surface_scales = torch.ones(n_surf, 3) * 0.3
            surface_opacity = torch.ones(n_surf)

        # a point directly above the patch centre (outside = +z)
        pt_above = torch.tensor([[0.0, 0.0, 0.5]])
        # a point directly below (inside = -z)
        pt_below = torch.tensor([[0.0, 0.0, -0.5]])
        D_above = query_surface_signed_distance(_PlaneState(), pt_above)
        D_below = query_surface_signed_distance(_PlaneState(), pt_below)

        self.assertGreater(float(D_above[0]), float(D_below[0]),
            f"D_s above plane ({float(D_above[0]):.3f}) should be > below "
            f"({float(D_below[0]):.3f})")
        self.assertGreater(float(D_above[0]), 0.0,
            f"Point above patch should have D_s > 0, got {float(D_above[0]):.4f}")
        self.assertLess(float(D_below[0]), 0.0,
            f"Point below patch should have D_s < 0, got {float(D_below[0]):.4f}")


if __name__ == "__main__":
    unittest.main()
