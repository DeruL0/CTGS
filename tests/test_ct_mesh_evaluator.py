import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.mesh_evaluator import (
    MeshData,
    evaluate_mesh_extraction,
    main as mesh_evaluator_main,
    mesh_from_support_mask,
    write_mesh_evaluation,
    _write_mesh_ply,
)


class CTMeshEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.phase1_dir = self.temp_dir / "phase1"
        self.phase1_dir.mkdir()

        self.spacing_zyx = (1.0, 1.0, 1.0)
        support_mask = np.zeros((8, 8, 8), dtype=bool)
        support_mask[2:6, 2:6, 2:6] = True
        self.reference_mesh = mesh_from_support_mask(support_mask, self.spacing_zyx)

        np.savez(
            self.phase1_dir / "analysis.npz",
            coarse_support_mask=support_mask,
            material_mask=support_mask,
            boundary_points=self.reference_mesh.vertices,
        )
        (self.phase1_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "spacing_zyx": list(self.spacing_zyx),
                    "origin_xyz": [0.0, 0.0, 0.0],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_matching_mesh_has_near_zero_bidirectional_distance(self):
        metrics = evaluate_mesh_extraction(
            self.reference_mesh,
            self.phase1_dir,
            sample_count=0,
            reference_sample_count=0,
        )

        self.assertEqual(metrics["mesh_vertex_count"], self.reference_mesh.vertices.shape[0])
        self.assertLess(metrics["symmetric_chamfer_l1_mean"], 1e-6)
        self.assertLess(metrics["symmetric_hausdorff_distance"], 1e-6)
        self.assertIn("mesh_sample_outside_support_ratio", metrics)

    def test_shifted_mesh_reports_large_error_and_outside_support(self):
        shifted_mesh = MeshData(
            vertices=self.reference_mesh.vertices + np.array([5.0, 0.0, 0.0], dtype=np.float32),
            faces=self.reference_mesh.faces,
            normals=self.reference_mesh.normals,
            material_id=self.reference_mesh.material_id,
        )

        metrics = evaluate_mesh_extraction(
            shifted_mesh,
            self.phase1_dir,
            sample_count=0,
            reference_sample_count=0,
        )

        self.assertGreater(metrics["symmetric_chamfer_l1_mean"], 2.0)
        self.assertGreater(metrics["symmetric_hausdorff_p95"], 4.0)
        self.assertGreater(metrics["mesh_sample_outside_support_ratio"], 0.5)

    def test_writes_json_and_text_reports(self):
        metrics = evaluate_mesh_extraction(
            self.reference_mesh,
            self.phase1_dir,
            sample_count=0,
            reference_sample_count=0,
        )
        json_path, txt_path = write_mesh_evaluation(metrics, self.temp_dir / "metrics.json")

        self.assertTrue(json_path.exists())
        self.assertTrue(txt_path.exists())
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIn("symmetric_chamfer_l1_mean", loaded)
        self.assertIn("mesh_sample_outside_support_ratio", txt_path.read_text(encoding="utf-8"))

    def test_cli_evaluates_existing_mesh_ply(self):
        mesh_path = _write_mesh_ply(self.temp_dir / "predicted_mesh.ply", self.reference_mesh)
        output_path = self.temp_dir / "cli_metrics.json"

        exit_code = mesh_evaluator_main(
            [
                "--mesh",
                str(mesh_path),
                "--phase1",
                str(self.phase1_dir),
                "--output",
                str(output_path),
                "--sample-count",
                "0",
                "--reference-sample-count",
                "0",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_path.exists())
        self.assertTrue(output_path.with_suffix(".txt").exists())


if __name__ == "__main__":
    unittest.main()
