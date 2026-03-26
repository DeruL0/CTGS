import unittest
from importlib import util as importlib_util

import numpy as np
import torch

from ct_pipeline.ct_slice_renderer import (
    CTPatchGridCache,
    build_ct_patch_renderer,
    prepare_ct_render_state,
    render_ct_slice_patch,
    sample_gt_slice_patch,
)


class DummyGaussians:
    def __init__(self, xyz, scaling, rotation, opacity):
        self._xyz = xyz
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_opacity(self):
        return self._opacity


class CTSliceRendererTests(unittest.TestCase):
    def setUp(self):
        self.gaussians = DummyGaussians(
            xyz=torch.tensor([[4.0, 4.0, 4.0]], dtype=torch.float32),
            scaling=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            opacity=torch.tensor([[1.0]], dtype=torch.float32),
        )

    def test_single_gaussian_peaks_at_slice_center(self):
        patch = render_ct_slice_patch(
            self.gaussians,
            axis="z",
            slice_idx=4,
            patch_origin_hw=(0, 0),
            patch_size_hw=(9, 9),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_dhw=(9, 9, 9),
        )

        max_index = torch.nonzero(patch == patch.max(), as_tuple=False)[0]
        self.assertEqual(tuple(max_index.tolist()), (4, 4))

    def test_response_decays_away_from_matching_slice(self):
        center_patch = render_ct_slice_patch(
            self.gaussians,
            axis="z",
            slice_idx=4,
            patch_origin_hw=(0, 0),
            patch_size_hw=(9, 9),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_dhw=(9, 9, 9),
        )
        offset_patch = render_ct_slice_patch(
            self.gaussians,
            axis="z",
            slice_idx=6,
            patch_origin_hw=(0, 0),
            patch_size_hw=(9, 9),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_dhw=(9, 9, 9),
        )

        self.assertGreater(float(center_patch.max()), float(offset_patch.max()))

    def test_sample_gt_slice_patch_matches_volume_indexing(self):
        volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        patch = sample_gt_slice_patch(volume, axis="y", slice_idx=2, patch_origin_hw=(1, 3), patch_size_hw=(2, 2))
        expected = volume[1:3, 2, 3:5]
        self.assertTrue(np.array_equal(patch, expected))

    def test_precomputed_render_state_matches_direct_render(self):
        eager_renderer = build_ct_patch_renderer(compile_renderer=False)
        render_state = prepare_ct_render_state(self.gaussians)
        patch_grid_cache = CTPatchGridCache()

        direct = render_ct_slice_patch(
            self.gaussians,
            axis="z",
            slice_idx=4,
            patch_origin_hw=(0, 0),
            patch_size_hw=(9, 9),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_dhw=(9, 9, 9),
            gaussians_per_chunk=64,
        )
        cached = eager_renderer(
            render_state,
            axis="z",
            slice_idx=4,
            patch_origin_hw=(0, 0),
            patch_size_hw=(9, 9),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_dhw=(9, 9, 9),
            gaussians_per_chunk=64,
            patch_grid_cache=patch_grid_cache,
        )

        self.assertTrue(torch.allclose(direct, cached, atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(
        torch.cuda.is_available() and hasattr(torch, "compile") and importlib_util.find_spec("triton") is not None,
        "CUDA, torch.compile, and Triton are required",
    )
    def test_compiled_renderer_matches_eager_with_autocast(self):
        gaussians = DummyGaussians(
            xyz=torch.tensor([[4.0, 4.0, 4.0], [5.0, 4.0, 4.0]], dtype=torch.float32, device="cuda"),
            scaling=torch.tensor([[1.0, 1.0, 1.0], [0.75, 1.25, 1.0]], dtype=torch.float32, device="cuda"),
            rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"),
            opacity=torch.tensor([[1.0], [0.6]], dtype=torch.float32, device="cuda"),
        )
        render_state = prepare_ct_render_state(gaussians)
        patch_grid_cache = CTPatchGridCache()
        eager_renderer = build_ct_patch_renderer(compile_renderer=False)
        compiled_renderer = build_ct_patch_renderer(compile_renderer=True)

        autocast_kwargs = {
            "device_type": "cuda",
            "dtype": torch.bfloat16,
            "enabled": torch.cuda.is_bf16_supported(),
        }
        with torch.autocast(**autocast_kwargs):
            eager = eager_renderer(
                render_state,
                axis="z",
                slice_idx=4,
                patch_origin_hw=(0, 0),
                patch_size_hw=(9, 9),
                spacing_zyx=(1.0, 1.0, 1.0),
                volume_shape_dhw=(9, 9, 9),
                gaussians_per_chunk=64,
                patch_grid_cache=patch_grid_cache,
            )
            compiled = compiled_renderer(
                render_state,
                axis="z",
                slice_idx=4,
                patch_origin_hw=(0, 0),
                patch_size_hw=(9, 9),
                spacing_zyx=(1.0, 1.0, 1.0),
                volume_shape_dhw=(9, 9, 9),
                gaussians_per_chunk=64,
                patch_grid_cache=patch_grid_cache,
            )

        self.assertTrue(torch.isfinite(eager).all())
        self.assertTrue(torch.isfinite(compiled).all())
        self.assertTrue(torch.allclose(eager.float(), compiled.float(), atol=5e-3, rtol=5e-3))
