import unittest

import torch

from utils.ct_losses import (
    edge_split_criterion,
    material_boundary_loss,
    normal_alignment_loss,
    occupancy_loss,
    point_to_plane_loss,
    thickness_penalty,
    volume_rendering_loss,
)


class CTLossTests(unittest.TestCase):
    def test_volume_rendering_loss_is_near_zero_for_identical_slices(self):
        slice_tensor = torch.linspace(0.0, 1.0, 64, dtype=torch.float32).reshape(8, 8)
        loss = volume_rendering_loss(slice_tensor, slice_tensor)
        self.assertLess(float(loss), 1e-6)

    def test_occupancy_loss_prefers_correct_binary_predictions(self):
        good_pred = torch.tensor([0.95, 0.9, 0.1, 0.05], dtype=torch.float32)
        bad_pred = torch.tensor([0.2, 0.1, 0.8, 0.9], dtype=torch.float32)
        target = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        self.assertLess(float(occupancy_loss(good_pred, target)), float(occupancy_loss(bad_pred, target)))

    def test_occupancy_loss_can_upweight_void_negatives(self):
        pred = torch.tensor([0.95, 0.75, 0.10], dtype=torch.float32)
        target = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        unweighted = occupancy_loss(pred, target)
        weighted = occupancy_loss(pred, target, sample_weights=torch.tensor([1.0, 3.0, 1.0], dtype=torch.float32))
        self.assertGreater(float(weighted), float(unweighted))

    def test_point_to_plane_loss_increases_for_off_plane_point(self):
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
        neighbor_index = torch.tensor(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
            dtype=torch.long,
        )

        on_plane = point_to_plane_loss(
            xyz,
            normals,
            planarity,
            material_ids=torch.zeros((4, 1), dtype=torch.long),
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
        )

        xyz[3, 2] = 0.5
        off_plane = point_to_plane_loss(
            xyz,
            normals,
            planarity,
            material_ids=torch.zeros((4, 1), dtype=torch.long),
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
        )

        self.assertLess(float(on_plane), 1e-6)
        self.assertGreater(float(off_plane), float(on_plane) + 1e-2)

    def test_point_to_plane_loss_handles_degenerate_neighbors(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(4, 1)
        planarity = torch.ones((4, 1), dtype=torch.float32)
        primitive_type = torch.ones((4, 1), dtype=torch.float32)
        neighbor_index = torch.tensor(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
            dtype=torch.long,
        )

        loss = point_to_plane_loss(
            xyz,
            normals,
            planarity,
            material_ids=torch.zeros((4, 1), dtype=torch.long),
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
        )

        self.assertTrue(torch.isfinite(loss))

    def test_point_to_plane_loss_backward_is_finite_for_problematic_planar_neighborhood(self):
        xyz = torch.tensor(
            [
                [27.5968, 36.3776, 16.8560],
                [27.5968, 36.6912, 16.8560],
                [27.5968, 36.0640, 16.8560],
                [27.2832, 36.6912, 16.8560],
                [26.9696, 36.3776, 16.8560],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(5, 1)
        planarity = torch.ones((5, 1), dtype=torch.float32)
        primitive_type = torch.ones((5, 1), dtype=torch.float32)
        neighbor_index = torch.tensor(
            [
                [1, 2, 3, 4],
                [0, 2, 3, 4],
                [0, 1, 3, 4],
                [0, 1, 2, 4],
                [0, 1, 2, 3],
            ],
            dtype=torch.long,
        )

        loss = point_to_plane_loss(
            xyz,
            normals,
            planarity,
            material_ids=torch.zeros((5, 1), dtype=torch.long),
            primitive_type=primitive_type,
            neighbor_index=neighbor_index,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(xyz.grad)
        self.assertTrue(torch.isfinite(xyz.grad).all())

    def test_normal_alignment_loss_handles_flipped_and_orthogonal_normals(self):
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        aligned_neighbors = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            ],
            dtype=torch.float32,
        )
        orthogonal_neighbors = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        weights = torch.ones((2, 2), dtype=torch.float32)

        aligned_loss = normal_alignment_loss(normals, aligned_neighbors, weights)
        orthogonal_loss = normal_alignment_loss(normals, orthogonal_neighbors, weights)

        self.assertLess(float(aligned_loss), 1e-6)
        self.assertGreater(float(orthogonal_loss), 0.9)

    def test_thickness_penalty_only_applies_to_planar_primitives(self):
        raw_scaling = torch.log(torch.tensor([[1.0, 1.0, 0.2], [1.0, 1.0, 0.2]], dtype=torch.float32))
        primitive_type = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        penalty = thickness_penalty(raw_scaling, None, None, primitive_type, max_thickness=0.1)
        self.assertGreater(float(penalty), 0.09)

        non_planar = thickness_penalty(raw_scaling, None, None, torch.zeros_like(primitive_type), max_thickness=0.1)
        self.assertEqual(float(non_planar), 0.0)

    def test_material_boundary_loss_uses_cross_material_neighbors(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        opacity = torch.tensor([[0.2], [0.2], [0.8]], dtype=torch.float32)
        neighbor_index = torch.tensor([[1, 2], [0, 2], [1, 0]], dtype=torch.long)

        single_material = material_boundary_loss(xyz, torch.zeros((3, 1), dtype=torch.long), opacity, neighbor_index=neighbor_index)
        cross_material = material_boundary_loss(
            xyz,
            torch.tensor([[0], [1], [1]], dtype=torch.long),
            opacity,
            neighbor_index=neighbor_index,
        )

        self.assertEqual(float(single_material), 0.0)
        self.assertGreater(float(cross_material), 0.0)

    def test_edge_split_criterion_reduces_threshold_for_high_curvature(self):
        grads = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        xyz = torch.zeros((2, 3), dtype=torch.float32)
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        curvature = torch.tensor([[0.1], [10.0]], dtype=torch.float32)
        multiplier = edge_split_criterion(grads, xyz, normals, curvature)

        self.assertEqual(multiplier.shape, (2, 1))
        self.assertGreaterEqual(float(multiplier[0]), 1.0)
        self.assertLess(float(multiplier[1]), float(multiplier[0]))
