import importlib
import unittest


class CTPipelineImportTests(unittest.TestCase):
    def test_new_ct_public_paths_import(self):
        self.assertIsNotNone(importlib.import_module("ct_pipeline"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.config"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.data"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.geometry"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.rendering"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.exporting"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.runtime"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.training.losses"))

    def test_retired_ct_pipeline_flat_modules_are_no_longer_importable(self):
        for module_name in [
            "ct_pipeline.ct_args",
            "ct_pipeline.ct_exporter",
            "ct_pipeline.ct_loader",
            "ct_pipeline.ct_preprocessor",
            "ct_pipeline.ct_slice_renderer",
            "ct_pipeline.field_query",
            "ct_pipeline.geometry_analyzer",
            "ct_pipeline.acceleration",
            "ct_pipeline.compression",
            "ct_pipeline.backend.native_backend_core",
            "ct_pipeline.backend.native_backend_grid",
            "ct_pipeline.backend.native_backend_query",
            "ct_pipeline.backend.native_backend_render",
            "ct_pipeline.training.ct_confidence",
            "ct_pipeline.training.train_ct_config",
            "ct_pipeline.training.train_ct_densify",
            "ct_pipeline.training.train_ct_grid_cache",
            "ct_pipeline.training.train_ct_reporting",
            "ct_pipeline.training.train_ct_runtime",
            "ct_pipeline.training.train_ct_sampling",
            "ct_pipeline.training.train_ct_utils",
            "utils.ct_geometry",
            "utils.ct_losses",
        ]:
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(module_name)

    def test_legacy_top_level_ct_modules_are_no_longer_importable(self):
        for module_name in [
            "ct_loader",
            "ct_preprocessor",
            "geometry_analyzer",
            "acceleration",
            "compression",
            "ct_slice_renderer",
            "ct_losses",
        ]:
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(module_name)

    def test_removed_standard_gs_modules_are_no_longer_importable(self):
        for module_name in [
            "arguments",
            "gaussian_renderer",
            "train",
            "render",
            "scene.dataset_readers",
            "scene.cameras",
            "scene.colmap_loader",
            "scene.gen_pseudo_cam_poses",
            "scene.initialize_utils",
            "utils.camera_utils",
        ]:
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(module_name)
