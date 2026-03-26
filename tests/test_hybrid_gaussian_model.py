import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

import scene as scene_module
from scene.ct_gaussian_model import CTGaussianModel
from scene.gaussian_model import GaussianModel
from utils.rotation_utils import quaternion_to_matrix


def build_training_args():
    return SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        primitive_harden_iter=2000,
        planar_thickness_max=None,
    )


def seed_gaussian_core(model: GaussianModel):
    device = torch.device("cuda")
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=torch.float32,
        device=device,
    )
    count = xyz.shape[0]
    model.spatial_lr_scale = 1.0
    model._xyz = nn.Parameter(xyz.requires_grad_(True))
    model._features_dc = nn.Parameter(torch.full((count, 1, 3), 0.5, dtype=torch.float32, device=device).requires_grad_(True))
    model._features_rest = nn.Parameter(torch.zeros((count, 0, 3), dtype=torch.float32, device=device).requires_grad_(True))
    model._scaling = nn.Parameter(torch.log(torch.full((count, 3), 0.1, dtype=torch.float32, device=device)).requires_grad_(True))
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(count, 1)
    model._rotation = nn.Parameter(rotation.requires_grad_(True))
    model._opacity = nn.Parameter(torch.logit(torch.full((count, 1), 0.1, dtype=torch.float32, device=device), eps=1e-6).requires_grad_(True))
    model._initialize_hybrid_metadata(count, device, rotations=model._rotation.detach())
    model.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=device)
    model.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.denom = torch.ones((count, 1), dtype=torch.float32, device=device)
    return model


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for hybrid Gaussian model tests")
class HybridGaussianModelTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_scene_package_is_ct_only(self):
        self.assertTrue(hasattr(scene_module, "GaussianModel"))
        self.assertTrue(hasattr(scene_module, "CTGaussianModel"))
        self.assertFalse(hasattr(scene_module, "build_gaussian_model"))
        self.assertFalse(hasattr(scene_module, "Scene"))

    def test_core_model_initializes_hybrid_defaults(self):
        model = seed_gaussian_core(GaussianModel(sh_degree=0))
        self.assertEqual(model._primitive_type.shape[0], model.get_xyz.shape[0])
        self.assertEqual(model._normal.shape[0], model.get_xyz.shape[0])
        self.assertTrue(torch.all(model.get_is_planar == 0))
        self.assertTrue(torch.all(model._material_id == -1))
        self.assertTrue(torch.allclose(model._planarity, torch.zeros_like(model._planarity)))
        self.assertTrue(torch.all(model.get_region_type == 0))

    def test_capture_and_restore_preserve_hybrid_fields(self):
        model = seed_gaussian_core(GaussianModel(sh_degree=0))
        model._primitive_type.data[:2] = model.planar_logit_value
        model._normal.data[:2] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
        model._material_id[:2] = torch.tensor([[3], [4]], device="cuda")
        model._planarity[:2] = 1.0
        model._region_type[2:] = 1
        model.planar_thickness_max = 0.02

        restored = GaussianModel(sh_degree=0)
        restored.restore(model.capture())
        self.assertTrue(torch.allclose(restored._primitive_type, model._primitive_type))
        self.assertTrue(torch.allclose(restored.get_normals(), model.get_normals()))
        self.assertTrue(torch.equal(restored._material_id, model._material_id))
        self.assertTrue(torch.allclose(restored._planarity, model._planarity))
        self.assertTrue(torch.equal(restored.get_region_type, model.get_region_type))
        self.assertEqual(restored.planar_thickness_max, 0.02)

    def test_legacy_checkpoint_restore_fills_ct_defaults(self):
        model = seed_gaussian_core(GaussianModel(sh_degree=0))
        legacy_args = (
            model.active_sh_degree,
            model._xyz,
            model._features_dc,
            model._features_rest,
            model._scaling,
            model._rotation,
            model._opacity,
            model.max_radii2D,
            model.xyz_gradient_accum,
            model.denom,
            None,
            model.spatial_lr_scale,
        )

        restored = GaussianModel(sh_degree=0)
        restored.restore(legacy_args)
        self.assertTrue(torch.all(restored.get_is_planar == 0))
        self.assertTrue(torch.all(restored._material_id == -1))
        self.assertTrue(torch.allclose(restored._planarity, torch.zeros_like(restored._planarity)))
        self.assertTrue(torch.all(restored.get_region_type == 0))

    def test_prune_and_densify_keep_hybrid_tensors_aligned(self):
        model = seed_gaussian_core(GaussianModel(sh_degree=0))
        model.training_setup(build_training_args())
        model._primitive_type.data[0] = model.planar_logit_value
        model._material_id[:, 0] = torch.arange(model.get_xyz.shape[0], device="cuda", dtype=torch.long)
        model._planarity[:, 0] = torch.linspace(0.0, 1.0, model.get_xyz.shape[0], device="cuda")
        model._normal.data = model.get_normals().detach().clone()

        grads = torch.ones((model.get_xyz.shape[0], 1), device="cuda")
        model.densify_and_clone(grads, grad_threshold=0.0, split_clone_size=10.0)
        cloned_count = model.get_xyz.shape[0]
        self.assertEqual(model._material_id.shape[0], cloned_count)
        self.assertEqual(model._primitive_type.shape[0], cloned_count)
        self.assertEqual(model.get_region_type.shape[0], cloned_count)

        model._scaling.data[:] = 0.0
        model.densify_and_split(grads=torch.ones((cloned_count, 1), device="cuda"), grad_threshold=0.0, split_clone_size=0.5, N=2)
        self.assertEqual(model.get_xyz.shape[0], model._normal.shape[0])
        self.assertEqual(model.get_xyz.shape[0], model._material_id.shape[0])
        self.assertEqual(model.get_xyz.shape[0], model._planarity.shape[0])
        self.assertEqual(model.get_xyz.shape[0], model.get_region_type.shape[0])

        prune_mask = torch.zeros((model.get_xyz.shape[0],), dtype=torch.bool, device="cuda")
        prune_mask[0] = True
        model.prune_points(prune_mask)
        self.assertEqual(model.get_xyz.shape[0], model._primitive_type.shape[0])
        self.assertEqual(model.get_xyz.shape[0], model._normal.shape[0])
        self.assertEqual(model.get_xyz.shape[0], model._material_id.shape[0])
        self.assertEqual(model.get_xyz.shape[0], model.get_region_type.shape[0])

    def test_effective_rotation_and_planar_thickness_clamp(self):
        model = seed_gaussian_core(GaussianModel(sh_degree=0))
        model._primitive_type.data[0] = model.planar_logit_value
        model._normal.data[0] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
        model._scaling.data[0] = torch.log(torch.tensor([0.2, 0.2, 0.2], device="cuda"))
        model.planar_thickness_max = 0.01

        normals = model.get_normals()
        rotation_matrix = quaternion_to_matrix(model.get_rotation)[0]
        scaling = model.get_scaling
        self.assertGreater(float(torch.abs(normals[0, 0]).detach()), 0.99)
        self.assertGreater(float(torch.abs(rotation_matrix[0, 2]).detach()), 0.99)
        self.assertLessEqual(float(scaling[0, 2].detach()), 0.010001)

    def test_ct_model_hardens_primitive_types(self):
        model = seed_gaussian_core(CTGaussianModel(sh_degree=0))
        model.training_setup(build_training_args())
        model._primitive_type.data[:2] = torch.tensor([[2.0], [-2.0]], device="cuda")

        model.post_optimizer_step(2000)
        self.assertTrue(model.primitive_types_hardened)
        self.assertFalse(model._primitive_type.requires_grad)
        self.assertGreater(float(model.get_primitive_type_prob[0]), 0.5)
        self.assertLess(float(model.get_primitive_type_prob[1]), 0.5)

    def test_ct_bundle_initialization_and_ply_roundtrip(self):
        analysis_path, metadata_path = self._write_phase1_bundle()
        model = CTGaussianModel(sh_degree=0)
        model.create_from_phase1_bundle(analysis_path, metadata_path, spatial_lr_scale=1.0)

        planar_mask = model.get_is_planar.squeeze(-1)
        self.assertEqual(int(planar_mask.sum().item()), 2)
        self.assertTrue(torch.all(model._material_id == 0))
        self.assertGreater(int((model.get_region_type == 1).sum().item()), 0)
        self.assertLessEqual(float(model.get_scaling[planar_mask, 2].max().detach()), model.planar_thickness_max + 1e-6)

        ply_path = self.temp_dir / "hybrid.ply"
        model.save_ply(str(ply_path))

        reloaded = CTGaussianModel(sh_degree=0)
        reloaded.load_ply(str(ply_path))
        self.assertTrue(torch.allclose(reloaded._primitive_type, model._primitive_type))
        self.assertTrue(torch.allclose(reloaded.get_normals(), model.get_normals(), atol=1e-5))
        self.assertTrue(torch.equal(reloaded._material_id, model._material_id))
        self.assertTrue(torch.allclose(reloaded._planarity, model._planarity))
        self.assertTrue(torch.equal(reloaded.get_region_type, model.get_region_type))

    def test_old_ply_load_defaults_to_ct_metadata(self):
        old_ply_path = self.temp_dir / "old_style.ply"
        self._write_old_style_ply(old_ply_path)

        model = CTGaussianModel(sh_degree=0)
        model.load_ply(str(old_ply_path))
        self.assertTrue(torch.all(model.get_is_planar == 0))
        self.assertTrue(torch.all(model._material_id == -1))
        self.assertTrue(torch.allclose(model._planarity, torch.zeros_like(model._planarity)))
        self.assertTrue(torch.all(model.get_region_type == 0))

    def _write_phase1_bundle(self):
        analysis_path = self.temp_dir / "analysis.npz"
        metadata_path = self.temp_dir / "metadata.json"
        surface_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        surface_normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.savez_compressed(
            analysis_path,
            material_mask=np.ones((2, 2, 2), dtype=bool),
            void_mask=np.zeros((2, 2, 2), dtype=bool),
            foreground_mask=np.ones((2, 2, 2), dtype=bool),
            material_label_volume=np.ones((2, 2, 2), dtype=np.int32),
            surface_points=surface_points,
            surface_material_id=np.zeros((surface_points.shape[0], 1), dtype=np.int64),
            surface_normals=surface_normals,
            mask_planar=np.array([True, True, False]),
            mask_edge=np.array([False, False, True]),
            mask_curved=np.array([False, False, False]),
            plane_normals=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [np.nan, np.nan, np.nan],
                ],
                dtype=np.float32,
            ),
            plane_tangent_u=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [np.nan, np.nan, np.nan],
                ],
                dtype=np.float32,
            ),
            plane_tangent_v=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [np.nan, np.nan, np.nan],
                ],
                dtype=np.float32,
            ),
            plane_residuals=np.array([0.0, 0.0, np.inf], dtype=np.float32),
            interior_points=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
            interior_density_seed=np.array([[0.8]], dtype=np.float32),
            interior_material_id=np.array([[0]], dtype=np.int64),
        )
        metadata_path.write_text(json.dumps({"spacing_zyx": [1.0, 1.0, 1.0]}), encoding="utf-8")
        return analysis_path, metadata_path

    def _write_old_style_ply(self, path: Path):
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
        element = np.empty(1, dtype=dtype)
        element[0] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        PlyData([PlyElement.describe(element, "vertex")]).write(str(path))


if __name__ == "__main__":
    unittest.main()
