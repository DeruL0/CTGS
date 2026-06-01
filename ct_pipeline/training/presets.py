"""Named CT training recipes for repeatable command-line experiments."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence


CT_TRAINING_PRESETS = {
    "clearance_balanced_246": {
        "description": (
            "Feature-adaptive bulk attenuation-only recipe with 2/4/6 voxel "
            "spacing, anisotropic clearance, and gap-aware reseeding."
        ),
        "defaults": {
            "ct_bulk_lattice_margin_vox": 0.0,
            "ct_train_bulk_atten_only": True,
            "ct_freeze_surface": True,
            "ct_freeze_bulk_geometry": True,
            "ct_enable_densification": False,
            "ct_enable_surface_reseeding": False,
            "ct_atten_only_early_stop": False,
            "ct_gap_aware_reseed": True,
            "ct_bulk_reseed_from_iter": 50,
            "ct_bulk_reseed_until_iter": 500,
            "ct_bulk_reseed_interval": 100,
            "ct_bulk_reseed_sample_count": 12_000,
            "ct_bulk_reseed_max_per_iter": 6_000,
            "ct_bulk_reseed_max_gaussian_ratio": 3.0,
            "ct_gap_reseed_protect_prune": True,
            "ct_gap_reseed_boundary_subvoxel": True,
            "ct_bulk_init_mode": "feature_adaptive",
            "ct_feature_adaptive_spacing_mid_vox": 4,
            "ct_feature_adaptive_spacing_low_vox": 6,
            "ct_bulk_lattice_anisotropic": True,
            "ct_repair_gain_ratio_min": 0.05,
            "ct_repair_exclusion_radius_vox": 0.35,
            "ct_repair_max_new_fraction": 0.02,
            "ct_repair_max_new_per_pass": 3_000,
            "ct_repair_min_component_points": 4,
            "ct_init_preflight_max_material_coverage_gap": 0.25,
            "ct_init_preflight_min_material_a_b_p10": 0.15,
        },
    },
}


def available_ct_training_presets() -> tuple[str, ...]:
    return tuple(sorted(CT_TRAINING_PRESETS))


def get_ct_training_preset_defaults(name: str) -> dict[str, object]:
    return dict(CT_TRAINING_PRESETS[name]["defaults"])


def _requested_ct_training_preset(args: Sequence[str] | None) -> str | None:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--ct_preset", choices=available_ct_training_presets())
    parsed, _ = parser.parse_known_args(args)
    return parsed.ct_preset


class CTTrainingArgumentParser(ArgumentParser):
    """Argument parser that applies a named recipe before explicit overrides."""

    def parse_args(self, args: Sequence[str] | None = None, namespace=None):
        preset_name = _requested_ct_training_preset(args)
        if preset_name is None:
            return super().parse_args(args, namespace)

        preset_defaults = get_ct_training_preset_defaults(preset_name)
        original_defaults = {key: self.get_default(key) for key in preset_defaults}
        self.set_defaults(**preset_defaults)
        try:
            return super().parse_args(args, namespace)
        finally:
            self.set_defaults(**original_defaults)
