import io
import unittest
from contextlib import redirect_stderr

import train_ct
from ct_pipeline.training import build_parser, validate_ct_training_args


def parse_minimal_args(*extra):
    parser = build_parser()
    return parser.parse_args(
        [
            "--ct_phase1_dir",
            "phase1",
            "--ct_volume_path",
            "volume.tiff",
            *extra,
        ]
    )


class CTTrainingConfigTests(unittest.TestCase):
    def test_train_entrypoint_uses_training_package_contract(self):
        self.assertIs(train_ct.build_parser, build_parser)
        self.assertIs(train_ct.validate_ct_training_args, validate_ct_training_args)
        self.assertTrue(callable(train_ct.training_ct))

    def test_minimal_args_validate_and_receive_runtime_defaults(self):
        args = parse_minimal_args()
        validate_ct_training_args(args)
        self.assertGreater(args.ct_bulk_query_truncation_sigma, 0.0)
        self.assertFalse(hasattr(args, "ct_occ_sample_count"))
        self.assertFalse(hasattr(args, "ct_cavity_patch_bias"))

    def test_alias_flags_normalize_to_single_runtime_fields(self):
        args = parse_minimal_args(
            "--ct_volume_jitter_voxels",
            "0.25",
            "--ct_boundary_band_voxels",
            "2.5",
            "--ct_surface_sigma_n_max_voxels",
            "0.35",
        )
        validate_ct_training_args(args)
        self.assertEqual(args.ct_volume_jitter, 0.25)
        self.assertEqual(args.ct_boundary_band, 2.5)
        self.assertEqual(args.ct_surface_sigma_n_max, 0.35)

    def test_raw_volume_requires_sidecar(self):
        args = parse_minimal_args("--ct_volume_format", "raw")
        with self.assertRaisesRegex(ValueError, "--ct_raw_meta"):
            validate_ct_training_args(args)

    def test_invalid_boundary_ratio_is_rejected(self):
        args = parse_minimal_args("--ct_surface_boundary_sample_ratio", "1.5")
        with self.assertRaisesRegex(ValueError, "surface_boundary_sample_ratio"):
            validate_ct_training_args(args)

    def test_removed_backend_flag_is_not_registered(self):
        parser = build_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--ct_backend", "python"])

    def test_clearance_balanced_preset_applies_active_recipe(self):
        args = parse_minimal_args("--ct_preset", "clearance_balanced_246")
        self.assertFalse(args.quiet)
        self.assertFalse(args.skip_export_mesh)
        self.assertFalse(args.skip_export_sdf)
        self.assertTrue(args.ct_train_bulk_atten_only)
        self.assertTrue(args.ct_freeze_surface)
        self.assertTrue(args.ct_freeze_bulk_geometry)
        self.assertFalse(args.ct_enable_densification)
        self.assertFalse(args.ct_enable_surface_reseeding)
        self.assertFalse(args.ct_atten_only_early_stop)
        self.assertTrue(args.ct_gap_aware_reseed)
        self.assertEqual(args.ct_bulk_init_mode, "feature_adaptive")
        self.assertEqual(args.ct_feature_adaptive_spacing_mid_vox, 4)
        self.assertEqual(args.ct_feature_adaptive_spacing_low_vox, 6)
        self.assertEqual(args.ct_bulk_reseed_max_per_iter, 6_000)

    def test_explicit_flags_override_preset_without_leaking_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--ct_preset",
                "clearance_balanced_246",
                "--ct_bulk_reseed_interval",
                "25",
                "--ct_enable_surface_reseeding",
            ]
        )
        self.assertEqual(args.ct_bulk_reseed_interval, 25)
        self.assertTrue(args.ct_enable_surface_reseeding)

        default_args = parser.parse_args([])
        self.assertEqual(default_args.ct_bulk_reseed_interval, 500)
        self.assertTrue(default_args.ct_enable_surface_reseeding)
        self.assertFalse(default_args.quiet)

    def test_retired_historical_recipe_flags_are_not_registered(self):
        parser = build_parser()
        retired_flags = [
            "--ct_bulk_support_inside_sdf_factor",
            "--no-ct_query_compact_support",
            "--ct_repair_use_true_anisotropic_den",
            "--ct_bulk_max_offset_vox",
            "--ct_occ_tau",
            "--ct_occ_tau_voxels",
        ]
        for flag in retired_flags:
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([flag])


if __name__ == "__main__":
    unittest.main()
