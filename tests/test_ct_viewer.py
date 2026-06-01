import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from ct_pipeline.viewer.server import create_viewer_app
from ct_pipeline.viewer.session import BUFFER_STRIDE_BYTES, load_viewer_session

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _write_test_ply(path: Path) -> None:
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
        ("primitive_type", "f4"),
        ("normal_0", "f4"),
        ("normal_1", "f4"),
        ("normal_2", "f4"),
        ("material_id", "f4"),
        ("planarity", "f4"),
        ("region_type", "f4"),
    ]
    elements = np.empty(2, dtype=dtype)
    elements[0] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.5,
        0.5,
        0.5,
        _logit(0.9),
        0.0,
        0.0,
        math.log(0.5),
        1.0,
        0.0,
        0.0,
        0.0,
        -8.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    elements[1] = (
        0.4,
        0.2,
        0.0,
        0.0,
        0.0,
        1.0,
        0.5,
        0.5,
        0.5,
        _logit(0.65),
        math.log(0.8),
        math.log(0.6),
        math.log(0.8),
        1.0,
        0.0,
        0.0,
        0.0,
        -8.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
    )
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


class CTViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.ply_path = Path(self.temp_dir.name) / "viewer_test.ply"
        _write_test_ply(self.ply_path)

    def test_load_viewer_session_parses_counts_and_buffer_layout(self):
        session = load_viewer_session(self.ply_path, device="cpu")

        self.assertEqual(session.surface_count, 1)
        self.assertEqual(session.bulk_count, 1)
        self.assertEqual(session.gaussian_meta_payload()["stride_bytes"], BUFFER_STRIDE_BYTES)
        self.assertEqual(session.gaussian_meta_payload()["fields"]["normal"]["offset"], 10)
        self.assertEqual(session.gaussian_meta_payload()["fields"]["attenuation"]["offset"], 17)
        self.assertIn("bulk_intensity", session.session_payload())
        self.assertIn("surface_intensity", session.session_payload())
        self.assertIn("intensity", session.session_payload()["render_modes"])

    def test_render_slice_supports_all_surface_and_bulk_layers(self):
        session = load_viewer_session(self.ply_path, device="cpu")

        all_slice = session.render_slice(axis="z", t=0.5, layer="all", size=96)
        surface_slice = session.render_slice(axis="z", t=0.5, layer="surface", size=96)
        bulk_slice = session.render_slice(axis="z", t=0.5, layer="bulk", size=96)
        clipped_bulk_slice = session.render_slice(
            axis="z",
            t=0.5,
            layer="bulk",
            size=96,
            intensity_clip=True,
            intensity_min=0.9,
            intensity_max=1.0,
            preview_mode="clipped",
        )
        mask_bulk_slice = session.render_slice(
            axis="z",
            t=0.5,
            layer="bulk",
            size=96,
            intensity_clip=True,
            intensity_min=0.0,
            intensity_max=0.8,
            preview_mode="mask",
        )

        self.assertEqual(all_slice.shape, (96, 96))
        self.assertTrue(np.all(all_slice >= 0.0))
        self.assertTrue(np.all(all_slice <= 1.0))
        self.assertGreater(float(surface_slice.max()), 0.0)
        self.assertGreater(float(bulk_slice.max()), 0.0)
        self.assertFalse(np.allclose(surface_slice, bulk_slice))
        self.assertEqual(float(clipped_bulk_slice.max()), 0.0)
        self.assertGreater(float(mask_bulk_slice.max()), 0.0)

    @unittest.skipIf(TestClient is None, "FastAPI test client is unavailable")
    def test_viewer_api_serves_session_buffer_and_slice(self):
        session = load_viewer_session(self.ply_path, device="cpu")
        app = create_viewer_app(session)
        client = TestClient(app)

        session_response = client.get("/api/session")
        self.assertEqual(session_response.status_code, 200)
        self.assertEqual(session_response.json()["gaussian_count"], 2)
        self.assertEqual(session_response.json()["defaults"]["sliceSize"], 512)

        buffer_response = client.get("/api/gaussians/buffer")
        self.assertEqual(buffer_response.status_code, 200)
        self.assertEqual(len(buffer_response.content), 2 * BUFFER_STRIDE_BYTES)

        slice_response = client.get("/api/slice/gs", params={"axis": "z", "t": 0.5, "layer": "all", "size": 96})
        self.assertEqual(slice_response.status_code, 200)
        self.assertEqual(slice_response.headers["content-type"], "image/png")

        clipped_response = client.get(
            "/api/slice/gs",
            params={
                "axis": "z",
                "t": 0.5,
                "layer": "bulk",
                "size": 96,
                "intensity_clip": "true",
                "intensity_min": 0.9,
                "intensity_max": 1.0,
                "preview": "mask",
            },
        )
        self.assertEqual(clipped_response.status_code, 200)
        self.assertEqual(clipped_response.headers["content-type"], "image/png")

        high_res_slice_response = client.get("/api/slice/gs", params={"axis": "z", "t": 0.5, "layer": "all", "size": 1024})
        self.assertEqual(high_res_slice_response.status_code, 200)
        self.assertEqual(high_res_slice_response.headers["content-type"], "image/png")

    @unittest.skipIf(TestClient is None, "FastAPI test client is unavailable")
    def test_viewer_api_can_load_ply_from_upload(self):
        app = create_viewer_app(load_device="cpu")
        client = TestClient(app)

        missing_session = client.get("/api/session")
        self.assertEqual(missing_session.status_code, 404)

        load_response = client.post(
            "/api/session/load",
            params={"filename": "browser_selected.ply"},
            content=self.ply_path.read_bytes(),
        )
        self.assertEqual(load_response.status_code, 200)
        self.assertEqual(load_response.json()["gaussian_count"], 2)

        buffer_response = client.get("/api/gaussians/buffer")
        self.assertEqual(buffer_response.status_code, 200)
        self.assertEqual(len(buffer_response.content), 2 * BUFFER_STRIDE_BYTES)
