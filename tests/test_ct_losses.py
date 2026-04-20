import unittest

import torch

from utils.ct_losses import (
    boundary_ridge_alignment_loss,
    boundary_center_loss,
    boundary_normal_loss,
    bulk_overlap_regularizer,
    bulk_regularization_loss,
    bulk_scale_regularizer,
    continuous_field_loss,
    edge_split_criterion,
    material_boundary_compact_loss,
    material_boundary_loss,
    normal_alignment_loss,
    occupancy_loss,
    point_to_plane_loss,
    sample_volume_field,
    signed_surface_loss,
    surface_opacity_regularizer,
    surface_shape_loss,
    surface_tangential_scale_loss,
    surface_thickness_loss,
    thickness_penalty,
    void_boundary_loss,
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

    def test_occupancy_loss_treats_input_as_probability(self):
        probability_pred = torch.tensor([0.0, 1.0], dtype=torch.float32)
        neutral_pred = torch.tensor([0.5, 0.5], dtype=torch.float32)
        target = torch.tensor([0.0, 1.0], dtype=torch.float32)

        self.assertLess(float(occupancy_loss(probability_pred, target)), float(occupancy_loss(neutral_pred, target)))

    def test_sample_volume_field_reads_voxel_center_values(self):
        volume = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
        volume[0, 0, 1, 2, 3] = 0.75
        points = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
        sampled = sample_volume_field(volume, points, spacing_zyx=(1.0, 1.0, 1.0))
        self.assertGreater(float(sampled[0, 0]), 0.7)

    def test_continuous_field_loss_prefers_closer_signed_field_prediction(self):
        good_pred = torch.tensor([0.8, 0.6, -0.7, -0.9], dtype=torch.float32)
        bad_pred = torch.tensor([-0.3, 0.1, 0.4, 0.2], dtype=torch.float32)
        target = torch.tensor([0.9, 0.5, -0.8, -1.0], dtype=torch.float32)
        self.assertLess(float(continuous_field_loss(good_pred, target)), float(continuous_field_loss(bad_pred, target)))

    def test_boundary_center_loss_prefers_high_boundary_strength(self):
        xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        strong = boundary_center_loss(xyz, torch.tensor([1.0, 0.9], dtype=torch.float32))
        weak = boundary_center_loss(xyz, torch.tensor([0.2, 0.1], dtype=torch.float32))
        self.assertLess(float(strong), float(weak))

    def test_boundary_normal_loss_prefers_aligned_normals(self):
        surface = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        aligned = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float32)
        orthogonal = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.assertLess(float(boundary_normal_loss(surface, aligned, weight)), 1e-6)
        self.assertGreater(float(boundary_normal_loss(surface, orthogonal, weight)), 0.9)

    def test_void_boundary_loss_prefers_sharp_material_void_contrast(self):
        good_inner = torch.tensor([0.95, 0.9, 0.88], dtype=torch.float32)
        good_outer = torch.tensor([0.04, 0.08, 0.12], dtype=torch.float32)
        bad_inner = torch.tensor([0.62, 0.58, 0.55], dtype=torch.float32)
        bad_outer = torch.tensor([0.52, 0.49, 0.45], dtype=torch.float32)
        weight = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        good_loss = void_boundary_loss(good_inner, good_outer, weight, boundary_strength=weight, margin=0.25)
        bad_loss = void_boundary_loss(bad_inner, bad_outer, weight, boundary_strength=weight, margin=0.25)

        self.assertLess(float(good_loss), float(bad_loss))
        self.assertLess(float(good_loss), 0.3)

    def test_signed_surface_loss_prefers_zero_level_set_and_aligned_gradients(self):
        surface_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        aligned_values = torch.tensor([0.0, 0.05], dtype=torch.float32)
        aligned_gradients = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float32)
        off_values = torch.tensor([0.8, -0.7], dtype=torch.float32)
        off_gradients = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        weights = torch.tensor([1.0, 0.8], dtype=torch.float32)

        aligned_loss = signed_surface_loss(
            surface_xyz,
            normals,
            aligned_values,
            aligned_gradients,
            boundary_strength=weights,
            band_width_voxels=3.0,
        )
        off_loss = signed_surface_loss(
            surface_xyz,
            normals,
            off_values,
            off_gradients,
            boundary_strength=weights,
            band_width_voxels=3.0,
        )

        self.assertLess(float(aligned_loss), float(off_loss))
        self.assertLess(float(aligned_loss), 0.1)

    def test_boundary_ridge_alignment_loss_prefers_high_strength_stationary_aligned_surface(self):
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        good_loss = boundary_ridge_alignment_loss(
            normals,
            ridge_directional_derivative=torch.tensor([0.01, -0.02], dtype=torch.float32),
            ridge_strength=torch.tensor([0.9, 0.8], dtype=torch.float32),
            ridge_normals=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float32),
        )
        bad_loss = boundary_ridge_alignment_loss(
            normals,
            ridge_directional_derivative=torch.tensor([0.6, -0.7], dtype=torch.float32),
            ridge_strength=torch.tensor([0.05, 0.1], dtype=torch.float32),
            ridge_normals=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        )
        self.assertLess(float(good_loss), float(bad_loss))

    def test_surface_thickness_loss_only_applies_to_surface_region(self):
        raw_scaling = torch.log(torch.tensor([[0.2, 0.2, 0.3], [0.2, 0.2, 0.3]], dtype=torch.float32))
        rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        region_type = torch.tensor([[0], [1]], dtype=torch.long)
        loss = surface_thickness_loss(raw_scaling, rotation, normals, region_type, max_thickness=0.1)
        self.assertGreater(float(loss), 0.15)

    def test_surface_tangential_scale_loss_prefers_smaller_in_plane_support(self):
        raw_scaling_small = torch.log(torch.tensor([[0.6, 0.6, 0.1]], dtype=torch.float32))
        raw_scaling_large = torch.log(torch.tensor([[2.5, 2.5, 0.1]], dtype=torch.float32))
        rotation = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

        small_loss = surface_tangential_scale_loss(raw_scaling_small, rotation, normals, max_tangential_scale=1.0)
        large_loss = surface_tangential_scale_loss(raw_scaling_large, rotation, normals, max_tangential_scale=1.0)

        self.assertLess(float(small_loss), float(large_loss))
        self.assertLess(float(small_loss), 1e-6)

    def test_surface_opacity_regularizer_only_penalizes_near_saturation(self):
        modest = torch.tensor([0.5, 0.75, 0.85], dtype=torch.float32)
        saturated = torch.tensor([0.91, 0.97, 0.99], dtype=torch.float32)

        modest_loss = surface_opacity_regularizer(modest, target_opacity=0.9)
        saturated_loss = surface_opacity_regularizer(saturated, target_opacity=0.9)

        self.assertLess(float(modest_loss), 1e-6)
        self.assertGreater(float(saturated_loss), float(modest_loss))

    def test_surface_shape_loss_penalizes_surface_anisotropy_and_low_opacity(self):
        rotation = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        compact = surface_shape_loss(
            torch.log(torch.tensor([[0.5, 0.5, 0.05]], dtype=torch.float32)),
            rotation,
            normals,
            torch.tensor([0.9], dtype=torch.float32),
            max_thickness=0.1,
            max_tangential_scale=1.0,
            min_opacity=0.8,
        )
        bloated = surface_shape_loss(
            torch.log(torch.tensor([[2.0, 2.0, 0.4]], dtype=torch.float32)),
            rotation,
            normals,
            torch.tensor([0.3], dtype=torch.float32),
            max_thickness=0.1,
            max_tangential_scale=1.0,
            min_opacity=0.8,
        )

        self.assertLess(float(compact), 1e-6)
        self.assertGreater(float(bloated), float(compact) + 0.5)

    def test_bulk_scale_regularizer_only_penalizes_large_bulk_support(self):
        raw_scaling = torch.log(
            torch.tensor(
                [
                    [0.5, 0.5, 0.5],
                    [6.0, 0.5, 0.5],
                ],
                dtype=torch.float32,
            )
        )
        region_type = torch.tensor([[0], [1]], dtype=torch.long)
        low_loss = bulk_scale_regularizer(raw_scaling[:1], region_type[:1], max_bulk_scale=4.0)
        high_loss = bulk_scale_regularizer(raw_scaling, region_type, max_bulk_scale=4.0)

        self.assertLess(float(low_loss), 1e-6)
        self.assertGreater(float(high_loss), float(low_loss))

    def test_bulk_overlap_regularizer_penalizes_neighbor_overlap(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        raw_scaling_small = torch.log(torch.tensor([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]], dtype=torch.float32))
        raw_scaling_large = torch.log(torch.tensor([[0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.2, 0.2, 0.2]], dtype=torch.float32))
        neighbor_index = torch.tensor([[1], [0], [1]], dtype=torch.long)

        small_loss = bulk_overlap_regularizer(xyz, raw_scaling_small, neighbor_index=neighbor_index)
        large_loss = bulk_overlap_regularizer(xyz, raw_scaling_large, neighbor_index=neighbor_index)

        self.assertLess(float(small_loss), float(large_loss))
        self.assertLess(float(small_loss), 1e-6)

    def test_bulk_regularization_loss_penalizes_large_scale_and_local_density(self):
        xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float32)
        rotation = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        neighbor_index = torch.tensor([[1], [0], [1]], dtype=torch.long)
        low_density = bulk_regularization_loss(
            xyz,
            torch.log(torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=torch.float32)),
            rotation,
            torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32),
            neighbor_index=neighbor_index,
            max_bulk_scale=1.0,
            density_cap=0.5,
        )
        high_density = bulk_regularization_loss(
            xyz,
            torch.log(torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=torch.float32)),
            rotation,
            torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32),
            neighbor_index=neighbor_index,
            max_bulk_scale=1.0,
            density_cap=0.2,
        )
        large_scale = bulk_regularization_loss(
            xyz[:1],
            torch.log(torch.tensor([[2.0, 0.5, 0.5]], dtype=torch.float32)),
            rotation[:1],
            torch.tensor([0.05], dtype=torch.float32),
            max_bulk_scale=1.0,
            density_cap=3.0,
        )

        self.assertLess(float(low_density), float(high_density))
        self.assertGreater(float(large_scale), 0.9)

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

    def test_material_boundary_compact_loss_smooths_same_material_and_aligns_cross_material(self):
        xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        normals_aligned = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        normals_bad = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        neighbor_index = torch.tensor([[1, 2], [0, 2], [1, 0]], dtype=torch.long)

        smooth_same = material_boundary_compact_loss(
            xyz,
            normals_aligned,
            torch.zeros((3, 1), dtype=torch.long),
            torch.tensor([0.5, 0.52, 0.51], dtype=torch.float32),
            neighbor_index=neighbor_index,
        )
        rough_same = material_boundary_compact_loss(
            xyz,
            normals_aligned,
            torch.zeros((3, 1), dtype=torch.long),
            torch.tensor([0.1, 0.9, 0.2], dtype=torch.float32),
            neighbor_index=neighbor_index,
        )
        aligned_cross = material_boundary_compact_loss(
            xyz,
            normals_aligned,
            torch.tensor([[0], [0], [1]], dtype=torch.long),
            torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32),
            neighbor_index=neighbor_index,
        )
        bad_cross = material_boundary_compact_loss(
            xyz,
            normals_bad,
            torch.tensor([[0], [0], [1]], dtype=torch.long),
            torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32),
            neighbor_index=neighbor_index,
        )
        single_material_flat = material_boundary_compact_loss(
            xyz,
            normals_bad,
            torch.zeros((3, 1), dtype=torch.long),
            torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32),
            neighbor_index=neighbor_index,
        )

        self.assertLess(float(smooth_same), float(rough_same))
        self.assertLess(float(aligned_cross), float(bad_cross))
        self.assertLess(float(single_material_flat), 1e-6)

    def test_edge_split_criterion_reduces_threshold_for_high_curvature(self):
        grads = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        xyz = torch.zeros((2, 3), dtype=torch.float32)
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        curvature = torch.tensor([[0.1], [10.0]], dtype=torch.float32)
        multiplier = edge_split_criterion(grads, xyz, normals, curvature)

        self.assertEqual(multiplier.shape, (2, 1))
        self.assertGreaterEqual(float(multiplier[0]), 1.0)
        self.assertLess(float(multiplier[1]), float(multiplier[0]))
