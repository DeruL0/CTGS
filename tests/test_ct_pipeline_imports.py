import importlib
import unittest


class CTPipelineImportTests(unittest.TestCase):
    def test_new_ct_public_paths_import(self):
        self.assertIsNotNone(importlib.import_module("ct_pipeline"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.ct_loader"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.ct_preprocessor"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.geometry_analyzer"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.acceleration"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.compression"))
        self.assertIsNotNone(importlib.import_module("ct_pipeline.ct_slice_renderer"))
        self.assertIsNotNone(importlib.import_module("utils.ct_losses"))

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
