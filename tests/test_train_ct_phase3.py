import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

try:
    import train_ct as train_module
except Exception as import_error:  # pragma: no cover
    train_module = None
    TRAIN_IMPORT_ERROR = import_error
else:  # pragma: no cover
    TRAIN_IMPORT_ERROR = None


def build_ct_opt(iterations):
    return SimpleNamespace(
        iterations=iterations,
        percent_dense=0.01,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        pose_lr_init=0.001,
        pose_lr_final=0.001,
        pose_lr_delay_mult=0.01,
        pose_lr_max_steps=30000,
        pose_lr_joint=0.001,
        primitive_harden_iter=2000,
        planar_thickness_max=None,
    )


def build_dataset(model_path):
    return SimpleNamespace(model_path=str(model_path), sh_degree=0, white_background=False)


def build_args(phase1_dir, volume_path, raw_meta_path, model_path):
    return SimpleNamespace(
        ct_phase1_dir=str(phase1_dir),
        ct_volume_path=str(volume_path),
        ct_volume_format="raw",
        ct_raw_meta=str(raw_meta_path),
        ct_patch_size=4,
        ct_slice_batch_size=1,
        ct_neighbor_k=3,
        ct_neighbor_refresh_interval=1,
        ct_backend="python",
        ct_render_chunk_gaussians=64,
        ct_compile_renderer=False,
        ct_bulk_points_ratio=1.0,
        ct_bulk_boundary_margin_voxels=1,
        ct_material_query_count=16,
        ct_void_query_count=16,
        ct_interior_query_count=16,
        ct_exterior_query_count=16,
        ct_max_material_classes=3,
        ct_void_negative_weight=2.0,
        ct_density_query_tile_points=32,
        ct_knn_tile_size=16,
        ct_lambda_slice=1.0,
        ct_lambda_occupancy=1.0,
        ct_lambda_plane=0.2,
        ct_lambda_normal=0.1,
        ct_lambda_thickness=0.05,
        ct_lambda_material=0.1,
        primitive_harden_iter=2000,
        planar_thickness_max=0.25,
        load_ply=False,
        wandb=False,
        save_iterations=[1, 2],
        iterations=2,
        model_path=str(model_path),
        quiet=True,
        checkpoint_iterations=[1, 2],
        start_checkpoint=None,
        output_gs=str(model_path / "exports" / "display.ply"),
        output_mesh=None,
        output_sdf=None,
        export_mesh_resolution=0.05,
        export_sdf_resolution=32,
        skip_export_mesh=True,
        skip_export_sdf=True,
    )


@unittest.skipUnless(torch.cuda.is_available() and train_module is not None, "CUDA and train_ct.py dependencies are required for CT training tests")
class CTTrainingPhase3Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.phase1_dir = self.temp_dir / "phase1"
        self.phase1_dir.mkdir()
        self.model_dir = self.temp_dir / "output"
        self.model_dir.mkdir()
        self.raw_path, self.raw_meta_path = self._write_ct_volume()
        self._write_phase1_bundle()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_ct_training_requires_phase1_and_volume_path(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_volume_path = None
        with self.assertRaises(ValueError):
            train_module.validate_ct_training_args(args)

    def test_ct_training_completes_minimal_run_and_exports_requested_outputs(self):
        dataset = build_dataset(self.model_dir)
        opt = build_ct_opt(iterations=1)
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.iterations = 1
        args.save_iterations = [1]
        args.checkpoint_iterations = [1]

        result = train_module.training_ct(dataset, opt, [1], [1], None, args)

        self.assertEqual(result["branch"], "ct")
        self.assertTrue((self.model_dir / "point_cloud" / "iteration_1" / "point_cloud.ply").exists())
        self.assertTrue((self.model_dir / "chkpnt1.pth").exists())
        self.assertTrue((self.model_dir / "exports" / "display.ply").exists())

    def test_ct_checkpoint_resume_runs(self):
        dataset = build_dataset(self.model_dir)
        opt = build_ct_opt(iterations=1)
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.iterations = 1
        args.save_iterations = [1]
        args.checkpoint_iterations = [1]

        train_module.training_ct(dataset, opt, [1], [1], None, args)
        checkpoint_path = self.model_dir / "chkpnt1.pth"
        self.assertTrue(checkpoint_path.exists())

        resumed_opt = build_ct_opt(iterations=2)
        resumed_args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        resumed_args.iterations = 2
        resumed_args.save_iterations = [2]
        resumed_args.checkpoint_iterations = [2]
        result = train_module.training_ct(dataset, resumed_opt, [2], [2], str(checkpoint_path), resumed_args)
        self.assertEqual(result["branch"], "ct")
        self.assertTrue((self.model_dir / "point_cloud" / "iteration_2" / "point_cloud.ply").exists())

    def test_ct_training_runs_without_output_gs(self):
        dataset = build_dataset(self.model_dir)
        opt = build_ct_opt(iterations=1)
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.iterations = 1
        args.save_iterations = [1]
        args.checkpoint_iterations = [1]
        args.output_gs = None

        result = train_module.training_ct(dataset, opt, [1], [1], None, args)
        self.assertEqual(result["branch"], "ct")
        self.assertNotIn("output_gs", result)

    def test_ct_training_auto_backend_resolves_and_runs(self):
        dataset = build_dataset(self.model_dir)
        opt = build_ct_opt(iterations=1)
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.iterations = 1
        args.save_iterations = [1]
        args.checkpoint_iterations = [1]
        args.ct_backend = "auto"

        result = train_module.training_ct(dataset, opt, [1], [1], None, args)
        self.assertEqual(result["branch"], "ct")
        self.assertIn(result["backend"], {"python", "cuda"})

    def test_void_aware_occupancy_pools_keep_cavity_negative(self):
        analysis = {
            "foreground_mask": np.load(str(self.phase1_dir / "analysis.npz"))["foreground_mask"],
            "material_mask": np.load(str(self.phase1_dir / "analysis.npz"))["material_mask"],
            "void_mask": np.load(str(self.phase1_dir / "analysis.npz"))["void_mask"],
            "roi_bbox": np.load(str(self.phase1_dir / "analysis.npz"))["roi_bbox"],
        }
        pools = train_module._prepare_occupancy_voxel_pools(analysis, (8, 8, 8), boundary_margin_voxels=1)
        self.assertGreater(pools["material"].shape[0], 0)
        self.assertGreater(pools["void"].shape[0], 0)
        self.assertTrue(any(np.all(index == np.array([3, 3, 3], dtype=np.int32)) for index in pools["void"]))
        self.assertTrue(all(not np.all(index == np.array([3, 3, 3], dtype=np.int32)) for index in pools["material"]))

    def test_legacy_interior_query_alias_maps_to_material_query_count(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_material_query_count = None
        args.ct_interior_query_count = 23
        train_module.validate_ct_training_args(args)
        self.assertEqual(args.ct_material_query_count, 23)
        self.assertEqual(args.ct_interior_query_count, 23)

    def test_ct_training_cuda_backend_resolution_error_propagates(self):
        dataset = build_dataset(self.model_dir)
        opt = build_ct_opt(iterations=1)
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_backend = "cuda"

        with mock.patch.object(train_module, "resolve_ct_backend", side_effect=RuntimeError("native backend missing")):
            with self.assertRaisesRegex(RuntimeError, "native backend missing"):
                train_module.training_ct(dataset, opt, [1], [1], None, args)

    def _write_ct_volume(self):
        volume = np.zeros((8, 8, 8), dtype=np.uint16)
        volume[2:6, 2:6, 2:6] = 400
        raw_path = self.temp_dir / "volume.raw"
        sidecar_path = self.temp_dir / "volume.json"
        volume.tofile(str(raw_path))
        sidecar = {
            "shape": [8, 8, 8],
            "dtype": "uint16",
            "spacing": [1.0, 1.0, 1.0],
            "endianness": "little",
        }
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        return raw_path, sidecar_path

    def _write_phase1_bundle(self):
        foreground_mask = np.zeros((8, 8, 8), dtype=bool)
        foreground_mask[2:6, 2:6, 2:6] = True
        material_mask = foreground_mask.copy()
        material_mask[3:5, 3:5, 3:5] = False
        void_mask = np.logical_and(foreground_mask, np.logical_not(material_mask))
        material_label_volume = np.zeros((8, 8, 8), dtype=np.int32)
        material_label_volume[material_mask] = 1
        roi_bbox = np.array([[2, 6], [2, 6], [2, 6]], dtype=np.int32)
        surface_points = np.array(
            [
                [2.0, 2.0, 2.0],
                [5.0, 2.0, 2.0],
                [2.0, 5.0, 2.0],
                [5.0, 5.0, 2.0],
            ],
            dtype=np.float32,
        )
        surface_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
        np.savez_compressed(
            self.phase1_dir / "analysis.npz",
            material_mask=material_mask,
            void_mask=void_mask,
            foreground_mask=foreground_mask,
            roi_bbox=roi_bbox,
            material_label_volume=material_label_volume,
            surface_points=surface_points,
            surface_material_id=np.zeros((4, 1), dtype=np.int64),
            surface_normals=surface_normals,
            mask_planar=np.array([True, True, True, True]),
            mask_edge=np.array([False, False, False, False]),
            mask_curved=np.array([False, False, False, False]),
            plane_normals=surface_normals,
            plane_tangent_u=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
            plane_tangent_v=np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (4, 1)),
            plane_residuals=np.zeros((4,), dtype=np.float32),
            material_id=np.zeros((4, 1), dtype=np.int64),
            interior_points=np.array([[3.5, 3.5, 3.5]], dtype=np.float32),
            interior_density_seed=np.array([[1.0]], dtype=np.float32),
            interior_material_id=np.array([[0]], dtype=np.int64),
        )
        (self.phase1_dir / "metadata.json").write_text(json.dumps({"spacing_zyx": [1.0, 1.0, 1.0]}), encoding="utf-8")
