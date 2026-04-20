import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn
from ct_pipeline import native_backend

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
        surface_thickness_max=None,
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
        ct_material_k=3,
        ct_neighbor_refresh_interval=1,
        ct_backend="cuda",
        ct_render_chunk_gaussians=64,
        ct_compile_renderer=False,
        ct_bulk_points_ratio=1.0,
        ct_bulk_boundary_margin_voxels=1,
        ct_support_threshold_mode="otsu",
        ct_support_sample_count=16,
        ct_air_sample_count=16,
        ct_gradient_sigma_voxels=1.0,
        ct_material_query_count=16,
        ct_void_query_count=16,
        ct_interior_query_count=16,
        ct_exterior_query_count=16,
        ct_max_material_classes=3,
        ct_void_negative_weight=4.0,
        ct_cavity_patch_bias=0.6,
        ct_void_boundary_offset_scale=0.75,
        ct_void_boundary_margin=0.25,
        ct_signed_field_band_voxels=3,
        ct_density_query_tile_points=32,
        ct_knn_tile_size=16,
        ct_gaussian_truncation_sigma=4.0,
        ct_slice_tile_size=8,
        ct_grid_cell_voxels=8,
        ct_freeze_bulk_xyz=True,
        ct_bulk_edt_alpha=1.0,
        ct_auto_preview=False,
        ct_lambda_render=1.0,
        ct_lambda_slice=1.0,
        ct_lambda_field_recon=None,
        ct_lambda_boundary_ridge=0.15,
        ct_lambda_occupancy=0.5,
        ct_lambda_shape=0.02,
        ct_lambda_bulk=0.01,
        ct_lambda_boundary_center=0.2,
        ct_lambda_boundary_normal=0.1,
        ct_lambda_signed_surface=0.15,
        ct_lambda_void_boundary=0.15,
        ct_lambda_surface_thickness=None,
        ct_lambda_surface_tangential_scale=None,
        ct_lambda_surface_opacity=None,
        ct_lambda_bulk_scale=None,
        ct_lambda_bulk_overlap=None,
        ct_lambda_plane=0.2,
        ct_lambda_normal=0.1,
        ct_lambda_thickness=None,
        ct_lambda_material=0.1,
        ct_align_weight=0.3,
        ct_cross_floor_weight=0.2,
        ct_thickness_max=0.25,
        ct_tangential_max_scale=4.0,
        ct_surface_tangential_max_scale=4.0,
        ct_surface_min_opacity=0.8,
        ct_surface_target_opacity=0.9,
        ct_bulk_max_scale=4.0,
        ct_bulk_density_cap=3.0,
        ct_bulk_k=4,
        ct_bulk_overlap_k=4,
        primitive_harden_iter=2000,
        surface_thickness_max=0.25,
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


@unittest.skipUnless(
    torch.cuda.is_available() and train_module is not None and native_backend.has_ct_native_backend(),
    "CUDA, ct_native_backend, and train_ct.py dependencies are required for CT training tests",
)
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
        self.assertEqual(result["backend"], "cuda")

    def test_field_sample_pools_keep_cavity_air_negative(self):
        analysis = {
            "foreground_mask": np.load(str(self.phase1_dir / "analysis.npz"))["foreground_mask"],
            "material_mask": np.load(str(self.phase1_dir / "analysis.npz"))["material_mask"],
            "void_mask": np.load(str(self.phase1_dir / "analysis.npz"))["void_mask"],
            "coarse_support_mask": np.load(str(self.phase1_dir / "analysis.npz"))["coarse_support_mask"],
            "roi_bbox": np.load(str(self.phase1_dir / "analysis.npz"))["roi_bbox"],
        }
        pools = train_module._prepare_field_sample_pools(analysis, (8, 8, 8), boundary_margin_voxels=1)
        self.assertGreater(pools["support"].shape[0], 0)
        self.assertGreater(pools["air_shell"].shape[0], 0)
        self.assertGreater(pools["air"].shape[0], 0)
        self.assertTrue(any(np.all(index == np.array([3, 3, 3], dtype=np.int32)) for index in pools["air"]))
        self.assertTrue(any(np.all(index == np.array([3, 3, 3], dtype=np.int32)) for index in pools["air_shell"]))
        self.assertEqual(float(pools["air_shell_band_ratio"]), 1.0)
        self.assertTrue(all(not np.all(index == np.array([3, 3, 3], dtype=np.int32)) for index in pools["support"]))

    def test_resolve_air_sampling_candidates_keeps_existing_shell_when_ratio_is_high(self):
        analysis = {
            "foreground_mask": np.load(str(self.phase1_dir / "analysis.npz"))["foreground_mask"],
            "material_mask": np.load(str(self.phase1_dir / "analysis.npz"))["material_mask"],
            "void_mask": np.load(str(self.phase1_dir / "analysis.npz"))["void_mask"],
            "coarse_support_mask": np.load(str(self.phase1_dir / "analysis.npz"))["coarse_support_mask"],
            "roi_bbox": np.load(str(self.phase1_dir / "analysis.npz"))["roi_bbox"],
        }
        pools = train_module._prepare_field_sample_pools(analysis, (8, 8, 8), boundary_margin_voxels=1)
        candidates, ratio, used_band = train_module._resolve_air_sampling_candidates(pools)
        np.testing.assert_array_equal(candidates, pools["air_shell"])
        self.assertEqual(float(ratio), 1.0)
        self.assertFalse(used_band)

    def test_legacy_interior_query_alias_maps_to_material_query_count(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_material_query_count = None
        args.ct_interior_query_count = 23
        train_module.validate_ct_training_args(args)
        self.assertEqual(args.ct_material_query_count, 23)
        self.assertEqual(args.ct_interior_query_count, 23)

    def test_legacy_loss_aliases_map_to_compact_loss_weights(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_lambda_render = None
        args.ct_lambda_slice = 0.7
        args.ct_lambda_field_recon = 0.6
        args.ct_lambda_shape = None
        args.ct_lambda_surface_thickness = 0.03
        args.ct_lambda_surface_tangential_scale = 0.04
        args.ct_lambda_surface_opacity = 0.02
        args.ct_lambda_bulk = None
        args.ct_lambda_bulk_scale = 0.005
        args.ct_lambda_bulk_overlap = 0.012
        args.ct_material_k = None
        args.ct_neighbor_k = 5
        args.ct_bulk_k = None
        args.ct_bulk_overlap_k = 6

        train_module.validate_ct_training_args(args)

        self.assertAlmostEqual(args.ct_lambda_render, 0.7)
        self.assertAlmostEqual(args.ct_lambda_occupancy, 0.6)
        self.assertAlmostEqual(args.ct_lambda_shape, 0.04)
        self.assertAlmostEqual(args.ct_lambda_bulk, 0.012)
        self.assertEqual(args.ct_material_k, 5)
        self.assertEqual(args.ct_bulk_k, 6)

    def test_cavity_aware_patch_sampling_biases_slice_toward_void_region(self):
        with np.load(str(self.phase1_dir / "analysis.npz")) as analysis_npz:
            analysis = {key: analysis_npz[key] for key in analysis_npz.files}
        np.random.seed(0)
        for _ in range(8):
            axis, slice_idx, _, _ = train_module._sample_ct_patch_spec(
                analysis,
                (8, 8, 8),
                requested_patch_size=4,
                spacing_zyx=(1.0, 1.0, 1.0),
                cavity_patch_bias=1.0,
            )
            self.assertIn(axis, {"z", "y", "x"})
            self.assertIn(slice_idx, {3, 4})

    def test_ct_training_python_backend_is_rejected_for_active_path(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_backend = "python"
        with self.assertRaisesRegex(ValueError, "requires the CUDA/native backend"):
            train_module.validate_ct_training_args(args)

    def test_ct_training_validates_surface_regularizer_args(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_surface_target_opacity = 1.0
        with self.assertRaisesRegex(ValueError, "ct_surface_target_opacity"):
            train_module.validate_ct_training_args(args)

        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_surface_min_opacity = 1.0
        with self.assertRaisesRegex(ValueError, "ct_surface_min_opacity"):
            train_module.validate_ct_training_args(args)

    def test_ct_training_validates_bulk_edt_alpha(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_bulk_edt_alpha = 0.0
        with self.assertRaisesRegex(ValueError, "ct_bulk_edt_alpha"):
            train_module.validate_ct_training_args(args)

    def test_ct_training_validates_bulk_overlap_k(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_bulk_overlap_k = 0
        with self.assertRaisesRegex(ValueError, "ct_bulk_overlap_k"):
            train_module.validate_ct_training_args(args)

        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_bulk_k = 0
        with self.assertRaisesRegex(ValueError, "ct_bulk_k"):
            train_module.validate_ct_training_args(args)

    def test_ct_training_validates_compact_loss_weights(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_lambda_shape = -0.1
        with self.assertRaisesRegex(ValueError, "lambda"):
            train_module.validate_ct_training_args(args)

        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_align_weight = -0.1
        with self.assertRaisesRegex(ValueError, "lambda"):
            train_module.validate_ct_training_args(args)

    def test_auto_field_sample_budget_scales_with_gaussian_count(self):
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_support_sample_count = None
        args.ct_air_sample_count = None
        train_module.validate_ct_training_args(args)

        support_small, air_small = train_module._resolve_field_sample_counts(args, total_gaussians=17_478)
        support_large, air_large = train_module._resolve_field_sample_counts(args, total_gaussians=69_912)

        self.assertEqual(support_small, 2184)
        self.assertEqual(air_small, 2184)
        self.assertGreater(support_large, support_small)
        self.assertGreater(air_large, air_small)

    def test_ct_training_cuda_backend_resolution_error_propagates(self):
        dataset = build_dataset(self.model_dir)
        opt = build_ct_opt(iterations=1)
        args = build_args(self.phase1_dir, self.raw_path, self.raw_meta_path, self.model_dir)
        args.ct_backend = "cuda"

        with mock.patch.object(train_module, "resolve_ct_backend", side_effect=RuntimeError("native backend missing")):
            with self.assertRaisesRegex(RuntimeError, "native backend missing"):
                train_module.training_ct(dataset, opt, [1], [1], None, args)

    def test_bulk_scale_hard_projection_clamps_bulk_only_and_clears_scaling_momentum(self):
        class DummyGaussians:
            def __init__(self):
                self._xyz = nn.Parameter(
                    torch.tensor(
                        [
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                        ],
                        dtype=torch.float32,
                        device="cuda",
                    )
                )
                self._scaling = nn.Parameter(
                    torch.log(
                        torch.tensor(
                            [
                                [0.8, 0.9, 0.7],
                                [2.0, 0.8, 1.5],
                                [1.6, 1.7, 0.6],
                            ],
                            dtype=torch.float32,
                            device="cuda",
                        )
                    )
                )
                self._region_type = torch.tensor([[0], [1], [1]], dtype=torch.long, device="cuda")
                self.optimizer = torch.optim.Adam([{"params": [self._scaling], "lr": 0.1, "name": "scaling"}])
                self.optimizer.state[self._scaling] = {
                    "exp_avg": torch.ones_like(self._scaling),
                    "exp_avg_sq": torch.ones_like(self._scaling),
                }

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_raw_scaling(self):
                return self._scaling

            @property
            def get_region_type(self):
                return self._region_type

        gaussians = DummyGaussians()
        support_distance = torch.full((1, 1, 4, 4, 4), 1.2, dtype=torch.float32, device="cuda")
        support_distance_field = {
            "support_distance": support_distance,
            "spacing_zyx": (1.0, 1.0, 1.0),
        }

        projected = train_module._apply_bulk_scale_hard_projection(
            gaussians,
            support_distance_field,
            spacing_zyx=(1.0, 1.0, 1.0),
            edt_alpha=1.0,
            bulk_max_scale=4.0,
        )
        cleared = train_module._clear_bulk_scaling_optimizer_momentum(gaussians)

        scales = torch.exp(gaussians._scaling.detach())
        self.assertEqual(projected, 2)
        self.assertEqual(cleared, 2)
        self.assertTrue(torch.allclose(scales[0], torch.tensor([0.8, 0.9, 0.7], device="cuda")))
        self.assertTrue(torch.allclose(scales[1], torch.tensor([1.2, 0.8, 1.2], device="cuda"), atol=1e-5))
        self.assertTrue(torch.allclose(scales[2], torch.tensor([1.2, 1.2, 0.6], device="cuda"), atol=1e-5))
        state = gaussians.optimizer.state[gaussians._scaling]
        self.assertTrue(torch.all(state["exp_avg"][1:] == 0))
        self.assertTrue(torch.all(state["exp_avg_sq"][1:] == 0))
        self.assertTrue(torch.all(state["exp_avg"][0] == 1))
        self.assertTrue(torch.all(state["exp_avg_sq"][0] == 1))

    def test_bulk_barrier_diagnostics_warn_when_cap_is_tighter_than_initial_bulk_scale(self):
        class DummyGaussians:
            def __init__(self):
                self._xyz = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda"))
                self._scaling = nn.Parameter(torch.log(torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32, device="cuda")))
                self._region_type = torch.tensor([[1]], dtype=torch.long, device="cuda")

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_raw_scaling(self):
                return self._scaling

            @property
            def get_region_type(self):
                return self._region_type

        gaussians = DummyGaussians()
        support_distance = torch.full((1, 1, 4, 4, 4), 0.6, dtype=torch.float32, device="cuda")
        support_distance_field = {
            "support_distance": support_distance,
            "spacing_zyx": (1.0, 1.0, 1.0),
        }

        with self.assertWarnsRegex(RuntimeWarning, "Bulk barrier cap is tighter"):
            train_module._log_bulk_barrier_diagnostics(
                gaussians,
                support_distance_field,
                spacing_zyx=(1.0, 1.0, 1.0),
                edt_alpha=1.0,
                bulk_max_scale=4.0,
            )

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
        boundary_points = np.array(
            [
                [2.5, 2.5, 2.5],
                [5.5, 2.5, 2.5],
                [2.5, 5.5, 2.5],
                [5.5, 5.5, 2.5],
            ],
            dtype=np.float32,
        )
        boundary_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
        np.savez_compressed(
            self.phase1_dir / "analysis.npz",
            coarse_support_mask=material_mask,
            support_threshold=np.array([0.5], dtype=np.float32),
            material_mask=material_mask,
            void_mask=void_mask,
            foreground_mask=foreground_mask,
            roi_bbox=roi_bbox,
            material_label_volume=material_label_volume,
            boundary_points=boundary_points,
            boundary_normals=boundary_normals,
            boundary_tangent_u=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
            boundary_tangent_v=np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (4, 1)),
            boundary_strength=np.ones((4, 1), dtype=np.float32),
            boundary_material_id=np.zeros((4, 1), dtype=np.int64),
            interior_points=np.array([[3.5, 3.5, 3.5]], dtype=np.float32),
            interior_density_seed=np.array([[1.0]], dtype=np.float32),
            interior_material_id=np.array([[0]], dtype=np.int64),
        )
        (self.phase1_dir / "metadata.json").write_text(json.dumps({"spacing_zyx": [1.0, 1.0, 1.0]}), encoding="utf-8")
