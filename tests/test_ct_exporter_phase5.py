import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from torch import nn

from ct_pipeline.ct_exporter import CTExporter
from mesher import meshing_ct
from scene.ct_gaussian_model import CTGaussianModel


class ExportDummyModel:
    def __init__(self, xyz, scaling, opacity, material_id, region_type=None):
        count = xyz.shape[0]
        self.max_sh_degree = 1
        self.active_sh_degree = 1
        self.planar_thickness_max = 0.1
        self.primitive_harden_iter = 2000
        self.primitive_types_hardened = False
        self.planar_logit_value = 8.0
        self.nonplanar_logit_value = -8.0
        self.single_material_fallback = False
        self.optimizer = None
        self.percent_dense = 0.01
        self.spatial_lr_scale = 1.0
        self.pose_lr_joint = 0.0

        self._xyz = nn.Parameter(torch.as_tensor(xyz, dtype=torch.float32))
        self._features_dc = nn.Parameter(torch.full((count, 1, 3), 0.1, dtype=torch.float32))
        self._features_rest = nn.Parameter(torch.zeros((count, 3, 3), dtype=torch.float32))
        self._scaling = nn.Parameter(torch.log(torch.as_tensor(scaling, dtype=torch.float32)))
        self._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(count, 1))
        self._opacity = nn.Parameter(torch.logit(torch.as_tensor(opacity, dtype=torch.float32), eps=1e-6))
        self._primitive_type = nn.Parameter(torch.full((count, 1), -8.0, dtype=torch.float32))
        self._normal = nn.Parameter(torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(count, 1))
        self._material_id = torch.as_tensor(material_id, dtype=torch.long).reshape(count, 1)
        self._planarity = torch.zeros((count, 1), dtype=torch.float32)
        if region_type is None:
            region_type = np.zeros((count, 1), dtype=np.int64)
        self._region_type = torch.as_tensor(region_type, dtype=torch.long).reshape(count, 1)
        self.max_radii2D = torch.zeros((count,), dtype=torch.float32)
        self.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32)
        self.denom = torch.zeros((count, 1), dtype=torch.float32)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_material_id(self):
        return self._material_id

    @property
    def get_planarity(self):
        return self._planarity

    @property
    def get_region_type(self):
        return self._region_type

    def get_normals(self):
        normals = self._normal
        norm = torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-8)
        return normals / norm

    def save_ply(self, path):
        from plyfile import PlyElement

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = self.get_normals().detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        primitive_type = self._primitive_type.detach().cpu().numpy()
        explicit_normals = self.get_normals().detach().cpu().numpy()
        material_id = self._material_id.detach().cpu().numpy().astype(np.float32)
        planarity = self._planarity.detach().cpu().numpy()
        region_type = self._region_type.detach().cpu().numpy().astype(np.float32)

        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        attrs += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
        attrs += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        attrs += ["opacity"]
        attrs += [f"scale_{i}" for i in range(scale.shape[1])]
        attrs += [f"rot_{i}" for i in range(rotation.shape[1])]
        attrs += ["primitive_type", "normal_0", "normal_1", "normal_2", "material_id", "planarity", "region_type"]
        dtype_full = [(attribute, "f4") for attribute in attrs]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, primitive_type, explicit_normals, material_id, planarity, region_type),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def build_single_material_model():
    return ExportDummyModel(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scaling=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        opacity=np.array([[0.95]], dtype=np.float32),
        material_id=np.array([[0]], dtype=np.int64),
    )


def build_multi_material_model():
    return ExportDummyModel(
        xyz=np.array(
            [
                [-0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        scaling=np.array(
            [
                [0.9, 0.9, 0.9],
                [0.9, 0.9, 0.9],
            ],
            dtype=np.float32,
        ),
        opacity=np.array([[0.95], [0.95]], dtype=np.float32),
        material_id=np.array([[1], [2]], dtype=np.int64),
    )


def build_surface_bulk_model():
    return ExportDummyModel(
        xyz=np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.3]], dtype=np.float32),
        scaling=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        opacity=np.array([[0.95], [0.004]], dtype=np.float32),
        material_id=np.array([[0], [0]], dtype=np.int64),
        region_type=np.array([[0], [1]], dtype=np.int64),
    )


class CTExporterPhase5Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.exporter = CTExporter()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_export_display_gs_prunes_and_writes_ply(self):
        model = build_single_material_model()
        output_path = self.exporter.export_display_gs(model, self.temp_dir / "display_model", compress=True)
        self.assertTrue(output_path.exists())
        vertex_count = len(PlyData.read(str(output_path)).elements[0].data)
        self.assertLessEqual(vertex_count, model.get_xyz.shape[0])

    def test_export_display_gs_preserves_bulk_points_with_region_aware_pruning(self):
        model = build_surface_bulk_model()
        output_path = self.exporter.export_display_gs(model, self.temp_dir / "display_bulk_model", compress=True)
        vertex_count = len(PlyData.read(str(output_path)).elements[0].data)
        self.assertEqual(vertex_count, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to load exported hybrid PLYs into CTGaussianModel.")
    def test_export_display_gs_round_trips_into_ct_gaussian_model(self):
        model = build_single_material_model()
        output_path = self.exporter.export_display_gs(model, self.temp_dir / "display_roundtrip.ply", compress=False)
        reloaded = CTGaussianModel(sh_degree=1)
        reloaded.load_ply(str(output_path))
        self.assertEqual(reloaded.get_xyz.shape[0], model.get_xyz.shape[0])

    def test_export_metrology_mesh_single_material_is_nonempty(self):
        model = build_single_material_model()
        output_path = self.exporter.export_metrology_mesh(model, self.temp_dir / "mesh_single", resolution=0.2)
        ply = PlyData.read(str(output_path))
        self.assertGreater(len(ply["vertex"].data), 0)
        self.assertGreater(len(ply["face"].data), 0)
        self.assertIn("material_id", ply["vertex"].data.dtype.names)

    def test_export_metrology_mesh_multi_material_preserves_labels(self):
        model = build_multi_material_model()
        output_path = self.exporter.export_metrology_mesh(model, self.temp_dir / "mesh_multi", resolution=0.2)
        ply = PlyData.read(str(output_path))
        labels = np.unique(np.asarray(ply["vertex"].data["material_id"], dtype=np.int32))
        self.assertGreaterEqual(labels.size, 2)

    def test_export_sdf_writes_npy_and_json_with_expected_signs(self):
        model = build_single_material_model()
        sdf_path, sidecar_path = self.exporter.export_sdf(model, self.temp_dir / "field", grid_resolution=32)
        self.assertTrue(sdf_path.exists())
        self.assertTrue(sidecar_path.exists())

        sdf = np.load(str(sdf_path))
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(tuple(meta["shape_zyx"]), sdf.shape)

        origin = np.asarray(meta["origin_xyz"], dtype=np.float32)
        spacing = np.asarray(meta["spacing_xyz"], dtype=np.float32)
        center_idx_xyz = np.round((np.array([0.0, 0.0, 0.0], dtype=np.float32) - origin) / spacing).astype(int)
        center_idx_zyx = (int(center_idx_xyz[2]), int(center_idx_xyz[1]), int(center_idx_xyz[0]))
        center_idx_zyx = tuple(np.clip(center_idx_zyx, 0, np.array(sdf.shape) - 1))

        self.assertLess(float(sdf[center_idx_zyx]), 0.0)
        self.assertGreater(float(sdf[0, 0, 0]), 0.0)


class MeshingCTPhase5Tests(unittest.TestCase):
    def test_meshing_ct_accepts_dataset_none_and_refines_boundary_region(self):
        model = build_multi_material_model()
        mesh = meshing_ct(None, model, resolution=0.35, threshold=0.25)
        self.assertTrue(mesh["boundary_refined"])
        self.assertGreater(mesh["faces"].shape[0], mesh["coarse_face_count"])
        self.assertGreater(mesh["vertices"].shape[0], 0)
