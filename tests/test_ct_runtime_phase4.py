import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ct_pipeline.acceleration import ClipPlaneManager, LODManager, OccupancyGrid
from ct_pipeline.compression import GSCompressor


class DummyCTGaussianModel:
    def __init__(self, sh_degree=0):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.planar_thickness_max = 0.1
        self.primitive_harden_iter = 2000
        self.primitive_types_hardened = False
        self.single_material_fallback = False
        self.optimizer = None

        self._xyz = nn.Parameter(torch.empty((0, 3), dtype=torch.float32))
        self._features_dc = nn.Parameter(torch.empty((0, 1, 3), dtype=torch.float32))
        self._features_rest = nn.Parameter(torch.empty((0, 0, 3), dtype=torch.float32))
        self._scaling = nn.Parameter(torch.empty((0, 3), dtype=torch.float32))
        self._rotation = nn.Parameter(torch.empty((0, 4), dtype=torch.float32))
        self._opacity = nn.Parameter(torch.empty((0, 1), dtype=torch.float32))
        self._primitive_type = nn.Parameter(torch.empty((0, 1), dtype=torch.float32))
        self._normal = nn.Parameter(torch.empty((0, 3), dtype=torch.float32))
        self._material_id = torch.empty((0, 1), dtype=torch.long)
        self._planarity = torch.empty((0, 1), dtype=torch.float32)
        self._region_type = torch.empty((0, 1), dtype=torch.long)
        self.max_radii2D = torch.empty((0,), dtype=torch.float32)
        self.xyz_gradient_accum = torch.empty((0, 1), dtype=torch.float32)
        self.denom = torch.empty((0, 1), dtype=torch.float32)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        if self._features_rest.numel() == 0:
            return self._features_dc
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_material_id(self):
        return self._material_id

    @property
    def get_planarity(self):
        return self._planarity

    @property
    def get_region_type(self):
        return self._region_type

    def get_normals(self):
        normals = self._normal
        norm = torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-8)
        return normals / norm


def build_dummy_model():
    model = DummyCTGaussianModel(sh_degree=1)
    xyz = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.4, 1.2, 1.0],
            [4.0, 4.0, 4.0],
            [8.0, 8.0, 8.0],
        ],
        dtype=torch.float32,
    )
    feature_dc = torch.tensor(
        [
            [[0.1, 0.2, 0.3]],
            [[0.12, 0.19, 0.29]],
            [[0.5, 0.4, 0.3]],
            [[0.8, 0.7, 0.6]],
        ],
        dtype=torch.float32,
    )
    feature_rest = torch.zeros((4, 3, 3), dtype=torch.float32)
    scaling = torch.log(
        torch.tensor(
            [
                [0.8, 0.8, 0.2],
                [0.7, 0.7, 0.2],
                [0.4, 0.4, 0.4],
                [0.15, 0.15, 0.15],
            ],
            dtype=torch.float32,
        )
    )
    rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    opacity_logits = torch.logit(torch.tensor([[0.8], [0.75], [0.4], [0.02]], dtype=torch.float32), eps=1e-6)
    primitive_type = torch.tensor([[8.0], [8.0], [-8.0], [-8.0]], dtype=torch.float32)
    normal = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    material_id = torch.tensor([[1], [1], [2], [2]], dtype=torch.long)
    planarity = torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)
    region_type = torch.tensor([[0], [0], [1], [1]], dtype=torch.long)

    model._xyz = nn.Parameter(xyz)
    model._features_dc = nn.Parameter(feature_dc)
    model._features_rest = nn.Parameter(feature_rest)
    model._scaling = nn.Parameter(scaling)
    model._rotation = nn.Parameter(rotation)
    model._opacity = nn.Parameter(opacity_logits)
    model._primitive_type = nn.Parameter(primitive_type)
    model._normal = nn.Parameter(normal)
    model._material_id = material_id
    model._planarity = planarity
    model._region_type = region_type
    model.max_radii2D = torch.zeros((4,), dtype=torch.float32)
    model.xyz_gradient_accum = torch.zeros((4, 1), dtype=torch.float32)
    model.denom = torch.zeros((4, 1), dtype=torch.float32)
    return model


class RuntimeAccelerationTests(unittest.TestCase):
    def test_occupancy_grid_marks_blocks_and_queries_rays(self):
        model = build_dummy_model()
        grid = OccupancyGrid(bbox=np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float32), block_size=2.0)
        occupancy = grid.update(model)

        self.assertTrue(occupancy.any())
        rays_hit = grid.query(
            np.array([[0.0, 0.0, 0.0], [0.0, 9.5, 0.0]], dtype=np.float32),
            np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        )
        self.assertTrue(bool(rays_hit[0]))
        self.assertFalse(bool(rays_hit[1]))

    def test_clip_plane_manager_filters_by_plane_halfspace(self):
        model = build_dummy_model()
        manager = ClipPlaneManager()
        manager.add_plane([1.0, 0.0, 0.0], 2.0)
        manager.add_plane([0.0, 0.0, 1.0], 1.0)
        mask = manager.clip_gaussians(model)
        self.assertEqual(mask.tolist(), [False, False, True, True])

    def test_lod_manager_builds_nonincreasing_levels(self):
        model = build_dummy_model()
        manager = LODManager(levels=3)
        lod_models = manager.build_lod(model)

        self.assertEqual(len(lod_models), 3)
        counts = [lod_model.get_xyz.shape[0] for lod_model in lod_models]
        self.assertGreaterEqual(counts[0], counts[1])
        self.assertGreaterEqual(counts[1], counts[2])
        self.assertEqual(manager.select_lod(0.1), 0)
        self.assertIn(manager.select_lod(100.0), {1, 2})


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_quantize_attributes_and_codebook_compress(self):
        model = build_dummy_model()
        compressor = GSCompressor()
        quantized = compressor.quantize_attributes(model, bits=8)
        codebook = compressor.codebook_compress(model, n_clusters=2)

        self.assertEqual(int(quantized["bits"]), 8)
        self.assertEqual(quantized["sh"]["data"].dtype, np.uint8)
        self.assertEqual(codebook["centers"].shape[0], 2)
        self.assertEqual(codebook["indices"].shape[0], model.get_xyz.shape[0])

    def test_prune_low_contribution_reduces_model(self):
        model = build_dummy_model()
        compressor = GSCompressor()
        pruned = compressor.prune_low_contribution(model, threshold=0.05)
        self.assertLess(pruned.get_xyz.shape[0], model.get_xyz.shape[0])
        self.assertGreater(pruned.get_xyz.shape[0], 0)

    def test_save_compressed_writes_npz_payload(self):
        model = build_dummy_model()
        compressor = GSCompressor()
        output_path = compressor.save_compressed(model, self.temp_dir / "compressed_model")
        self.assertTrue(output_path.exists())

        payload = np.load(output_path)
        self.assertIn("xyz", payload.files)
        self.assertIn("region_type", payload.files)
        self.assertIn("quantized_sh", payload.files)
        self.assertIn("codebook_centers", payload.files)
