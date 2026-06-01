import unittest

import torch

from ct_pipeline.training.losses import (
    asymmetric_binary_focal_loss,
    binary_focal_loss,
    calibrated_render_huber_loss,
    eagle_patch_loss,
    sample_sdf_normals,
    sample_volume_field,
    soft_occupancy_loss,
    surface_sdf_thickness_loss,
)


class CTFocalLossTests(unittest.TestCase):
    def test_binary_focal_loss_prefers_correct_predictions(self):
        target = torch.tensor([1.0, 0.0])
        good = binary_focal_loss(torch.tensor([0.9, 0.1]), target)
        bad = binary_focal_loss(torch.tensor([0.1, 0.9]), target)
        self.assertLess(float(good), float(bad))

    def test_asymmetric_focal_loss_supports_weighted_reduction(self):
        pred = torch.tensor([0.8, 0.2])
        target = torch.tensor([1.0, 0.0])
        weights = torch.tensor([2.0, 0.0])
        weighted = asymmetric_binary_focal_loss(pred, target, sample_weights=weights)
        positive_only = asymmetric_binary_focal_loss(pred[:1], target[:1])
        self.assertTrue(torch.allclose(weighted, positive_only))

    def test_focal_loss_rejects_invalid_gamma(self):
        with self.assertRaisesRegex(ValueError, "gamma"):
            binary_focal_loss(torch.tensor([0.5]), torch.tensor([1.0]), gamma=-1.0)


class CTRenderLossTests(unittest.TestCase):
    def test_calibrated_huber_is_zero_for_matching_intensity(self):
        occupancy = torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[0.1, 0.9]])
        loss = calibrated_render_huber_loss(occupancy, target, intensity_air=0.1, intensity_mat=0.9)
        self.assertEqual(float(loss), 0.0)

    def test_eagle_patch_loss_is_zero_for_matching_patch(self):
        patch = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64.0
        self.assertLess(float(eagle_patch_loss(patch, patch, block_size=3)), 1e-7)

    def test_soft_occupancy_loss_prefers_sdf_target(self):
        sdf = torch.tensor([-2.0, 2.0])
        expected = torch.sigmoid(-sdf)
        good = soft_occupancy_loss(expected, sdf, tau=1.0)
        bad = soft_occupancy_loss(1.0 - expected, sdf, tau=1.0)
        self.assertLess(float(good), float(bad))


class CTFieldSamplingTests(unittest.TestCase):
    def test_sample_volume_field_uses_voxel_centers(self):
        volume = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
        points = torch.tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        sampled = sample_volume_field(volume, points, spacing_zyx=(1.0, 1.0, 1.0)).reshape(-1)
        self.assertTrue(torch.allclose(sampled, torch.tensor([0.0, 7.0])))

    def test_sample_sdf_normals_tracks_x_gradient(self):
        x = torch.arange(5, dtype=torch.float32).reshape(1, 1, 1, 1, 5)
        sdf = x.expand(1, 1, 5, 5, 5).contiguous()
        normals = sample_sdf_normals(sdf, torch.tensor([[2.5, 2.5, 2.5]]), spacing_zyx=(1.0, 1.0, 1.0))
        self.assertGreater(float(normals[0, 0]), 0.99)

    def test_surface_regularizer_penalizes_thick_gaussian(self):
        sdf = torch.zeros((1, 1, 5, 5, 5), dtype=torch.float32)
        xyz = torch.tensor([[2.5, 2.5, 2.5]])
        raw_scaling = torch.log(torch.tensor([[0.2, 0.2, 1.0]]))
        rotations = torch.eye(3).reshape(1, 3, 3)
        normals = torch.tensor([[0.0, 0.0, 1.0]])
        loss = surface_sdf_thickness_loss(
            xyz,
            raw_scaling,
            rotations,
            sdf,
            spacing_zyx=(1.0, 1.0, 1.0),
            max_normal_thickness=0.1,
            sdf_normals=normals,
        )
        self.assertGreater(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
