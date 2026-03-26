import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
except ImportError:  # pragma: no cover
    pydicom = None

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

from ct_pipeline.ct_loader import CTVolumeLoader
from ct_pipeline.ct_preprocessor import CTPreprocessor
from ct_pipeline.geometry_analyzer import GeometryAnalyzer
from run_ct_phase1 import main as run_ct_phase1_main


def create_cube_volume(shape=(32, 32, 32), bounds=((8, 24), (8, 24), (8, 24))):
    volume = np.zeros(shape, dtype=np.float32)
    (z0, z1), (y0, y1), (x0, x1) = bounds
    volume[z0:z1, y0:y1, x0:x1] = 1.0
    return volume


def create_sphere_volume(shape=(32, 32, 32), center=(16, 16, 16), radius=9):
    zz, yy, xx = np.indices(shape)
    squared_distance = (
        (zz - center[0]) ** 2 +
        (yy - center[1]) ** 2 +
        (xx - center[2]) ** 2
    )
    return (squared_distance <= radius ** 2).astype(np.float32)


def create_hollow_cube_volume(shape=(32, 32, 32), outer=((8, 24), (8, 24), (8, 24)), inner=((12, 20), (12, 20), (12, 20))):
    volume = create_cube_volume(shape=shape, bounds=outer)
    (z0, z1), (y0, y1), (x0, x1) = inner
    volume[z0:z1, y0:y1, x0:x1] = 0.0
    return volume


def create_through_hole_block(shape=(32, 32, 32), bounds=((8, 24), (8, 24), (8, 24)), hole_radius=3):
    volume = create_cube_volume(shape=shape, bounds=bounds)
    zz, yy, xx = np.indices(shape)
    center_y = int(round((bounds[1][0] + bounds[1][1] - 1) * 0.5))
    center_z = int(round((bounds[0][0] + bounds[0][1] - 1) * 0.5))
    radial = (yy - center_y) ** 2 + (zz - center_z) ** 2
    hole_mask = (radial <= int(hole_radius) ** 2) & (xx >= bounds[2][0]) & (xx < bounds[2][1])
    x0, x1 = bounds[2]
    del x0, x1
    volume[hole_mask] = 0.0
    return volume


def create_multi_material_slab(shape=(24, 24, 24)):
    volume = np.zeros(shape, dtype=np.float32)
    volume[4:20, 4:12, 4:20] = 0.45
    volume[4:20, 12:20, 4:20] = 0.85
    return volume


@unittest.skipUnless(tifffile is not None, "tifffile is required for TIFF tests")
class CTLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_raw_volume_with_sidecar(self):
        raw_path = self.temp_dir / "volume.raw"
        sidecar_path = self.temp_dir / "volume.json"

        raw_volume = np.arange(24, dtype=">u2").reshape(2, 3, 4)
        raw_volume.tofile(str(raw_path))
        sidecar = {
            "shape": [2, 3, 4],
            "dtype": "uint16",
            "spacing": [1.5, 0.75, 0.5],
            "endianness": "big",
            "origin": [0.0, 1.0, 2.0],
        }
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

        loader = CTVolumeLoader()
        volume = loader.load(str(raw_path), fmt="raw", raw_meta_path=str(sidecar_path))
        metadata = loader.get_metadata()

        self.assertEqual(volume.shape, (2, 3, 4))
        self.assertEqual(volume.dtype, np.float32)
        self.assertAlmostEqual(float(volume.min()), 0.0)
        self.assertAlmostEqual(float(volume.max()), 1.0)
        self.assertEqual(loader.get_voxel_spacing(), (1.5, 0.75, 0.5))
        self.assertEqual(metadata["origin_xyz"], [0.0, 1.0, 2.0])

    def test_tiff_multipage_loads_with_embedded_spacing(self):
        tiff_path = self.temp_dir / "stack.tiff"
        volume = (np.arange(4 * 6 * 8).reshape(4, 6, 8) * 4).astype(np.uint16)
        tifffile.imwrite(
            str(tiff_path),
            volume,
            imagej=True,
            metadata={"spacing": 1.5, "unit": "mm"},
            resolution=(4.0, 2.0),
            resolutionunit="CENTIMETER",
        )

        loader = CTVolumeLoader()
        loaded = loader.load(str(tiff_path), fmt="tiff")

        self.assertEqual(loaded.shape, volume.shape)
        self.assertEqual(loader.get_voxel_spacing(), (1.5, 5.0, 2.5))
        self.assertAlmostEqual(float(loaded.min()), 0.0)
        self.assertAlmostEqual(float(loaded.max()), 1.0)

    def test_tiff_directory_loads_with_sidecar(self):
        slice_dir = self.temp_dir / "tiff_slices"
        slice_dir.mkdir()
        for index in range(3):
            slice_array = np.full((5, 7), fill_value=index * 100, dtype=np.uint16)
            tifffile.imwrite(str(slice_dir / "{0:02d}.tif".format(index)), slice_array)

        sidecar_path = self.temp_dir / "tiff_spacing.json"
        sidecar_path.write_text(
            json.dumps({"spacing": [2.0, 0.8, 0.4], "origin": [0.0, 0.0, 0.0]}),
            encoding="utf-8",
        )

        loader = CTVolumeLoader()
        loaded = loader.load(str(slice_dir), fmt="tiff", raw_meta_path=str(sidecar_path))
        self.assertEqual(loaded.shape, (3, 5, 7))
        self.assertEqual(loader.get_voxel_spacing(), (2.0, 0.8, 0.4))

    def test_tiff_missing_spacing_raises(self):
        tiff_path = self.temp_dir / "no_spacing.tiff"
        tifffile.imwrite(str(tiff_path), np.ones((2, 4, 4), dtype=np.uint16))

        loader = CTVolumeLoader()
        with self.assertRaises(ValueError):
            loader.load(str(tiff_path), fmt="tiff")

    @unittest.skipUnless(pydicom is not None, "pydicom is required for DICOM tests")
    def test_dicom_series_loads_and_sorts_by_position(self):
        dicom_dir = self.temp_dir / "dicom"
        dicom_dir.mkdir()

        series_uid = generate_uid()
        slices = [
            ("slice_c.dcm", 2, 10.0, np.full((6, 7), 300, dtype=np.int16)),
            ("slice_a.dcm", 0, 0.0, np.full((6, 7), 100, dtype=np.int16)),
            ("slice_b.dcm", 1, 5.0, np.full((6, 7), 200, dtype=np.int16)),
        ]
        for filename, instance_number, position_z, pixel_array in slices:
            self._write_dicom_slice(
                dicom_dir / filename,
                pixel_array=pixel_array,
                series_uid=series_uid,
                instance_number=instance_number,
                position_z=position_z,
            )

        loader = CTVolumeLoader()
        volume = loader.load(str(dicom_dir), fmt="dicom")

        self.assertEqual(volume.shape, (3, 6, 7))
        self.assertEqual(loader.get_voxel_spacing(), (5.0, 0.5, 0.25))
        self.assertLess(float(volume[0].mean()), float(volume[1].mean()))
        self.assertLess(float(volume[1].mean()), float(volume[2].mean()))

    def _write_dicom_slice(self, path, pixel_array, series_uid, instance_number, position_z):
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.Modality = "CT"
        dataset.SeriesInstanceUID = series_uid
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Rows = int(pixel_array.shape[0])
        dataset.Columns = int(pixel_array.shape[1])
        dataset.InstanceNumber = int(instance_number)
        dataset.ImagePositionPatient = [0.0, 0.0, float(position_z)]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.PixelSpacing = [0.5, 0.25]
        dataset.SliceThickness = 5.0
        dataset.RescaleSlope = 1.0
        dataset.RescaleIntercept = -1000.0
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.SamplesPerPixel = 1
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.PixelData = pixel_array.tobytes()
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.save_as(str(path))


class CTPreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.volume = create_cube_volume(shape=(24, 24, 24), bounds=((5, 19), (6, 18), (7, 15)))
        self.true_mask = self.volume.astype(bool)
        self.spacing = (2.0, 1.0, 0.5)
        self.preprocessor = CTPreprocessor()

    def test_segment_foreground_recovers_cube(self):
        mask = self.preprocessor.segment_foreground(self.volume)
        self.assertTrue(np.any(mask))
        overlap = np.logical_and(mask, self.true_mask).sum()
        self.assertGreater(overlap / float(self.true_mask.sum()), 0.98)

    def test_boundary_band_is_exterior_dilation(self):
        boundary = self.preprocessor.dilate_boundary(self.true_mask, width_voxels=2)
        expected = np.logical_and(
            nd_binary_dilation(self.true_mask, iterations=2),
            np.logical_not(self.true_mask),
        )
        self.assertTrue(np.array_equal(boundary, expected))

    def test_extract_surface_points_use_physical_spacing(self):
        points = self.preprocessor.extract_surface_points(self.true_mask, self.spacing)
        self.assertGreater(points.shape[0], 0)

        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)
        self.assertAlmostEqual(x_min, (7 - 0.5) * self.spacing[2], places=4)
        self.assertAlmostEqual(x_max, (15 - 0.5) * self.spacing[2], places=4)
        self.assertAlmostEqual(y_min, (6 - 0.5) * self.spacing[1], places=4)
        self.assertAlmostEqual(y_max, (18 - 0.5) * self.spacing[1], places=4)
        self.assertAlmostEqual(z_min, (5 - 0.5) * self.spacing[0], places=4)
        self.assertAlmostEqual(z_max, (19 - 0.5) * self.spacing[0], places=4)

    def test_sample_interior_points_stay_inside_foreground_and_keep_density_seed(self):
        points, density_seed, material_id = self.preprocessor.sample_interior_points(
            self.true_mask,
            self.volume,
            self.spacing,
            target_count=64,
            boundary_margin_voxels=2,
        )
        self.assertGreater(points.shape[0], 0)
        self.assertEqual(points.shape[0], density_seed.shape[0])
        self.assertEqual(points.shape[0], material_id.shape[0])
        self.assertTrue(np.allclose(density_seed, 1.0))
        self.assertTrue(np.all(material_id == 0))

        voxel_x = np.floor(points[:, 0] / self.spacing[2]).astype(np.int32)
        voxel_y = np.floor(points[:, 1] / self.spacing[1]).astype(np.int32)
        voxel_z = np.floor(points[:, 2] / self.spacing[0]).astype(np.int32)
        self.assertTrue(np.all(self.true_mask[voxel_z, voxel_y, voxel_x]))

    def test_segment_material_void_preserves_hollow_cavity(self):
        volume = create_hollow_cube_volume()
        segmentation = self.preprocessor.segment_material_void(volume, max_material_classes=3)
        material_mask = segmentation["material_mask"]
        void_mask = segmentation["void_mask"]
        foreground_mask = segmentation["foreground_mask"]

        self.assertTrue(np.any(material_mask))
        self.assertTrue(np.any(void_mask))
        self.assertTrue(np.array_equal(foreground_mask, np.logical_or(material_mask, void_mask)))
        self.assertTrue(void_mask[16, 16, 16])
        self.assertFalse(material_mask[16, 16, 16])

    def test_segment_material_void_preserves_through_hole(self):
        volume = create_through_hole_block()
        segmentation = self.preprocessor.segment_material_void(volume, max_material_classes=3)
        void_mask = segmentation["void_mask"]
        self.assertTrue(np.any(void_mask))
        self.assertTrue(void_mask[16, 16, 8])
        self.assertTrue(void_mask[16, 16, 23])

    def test_material_surface_extraction_returns_material_ids(self):
        volume = create_multi_material_slab()
        segmentation = self.preprocessor.segment_material_void(volume, max_material_classes=3)
        points, surface_material_id = self.preprocessor.extract_material_surface_points(
            segmentation["material_label_volume"],
            self.spacing,
        )
        self.assertGreater(points.shape[0], 0)
        self.assertEqual(points.shape[0], surface_material_id.shape[0])
        self.assertGreaterEqual(np.unique(segmentation["material_label_volume"][segmentation["material_label_volume"] > 0]).size, 2)
        self.assertGreaterEqual(np.unique(surface_material_id).size, 2)

    def test_sample_interior_points_uses_real_material_labels(self):
        volume = create_multi_material_slab()
        segmentation = self.preprocessor.segment_material_void(volume, max_material_classes=3)
        points, _, material_id = self.preprocessor.sample_interior_points(
            segmentation["material_mask"],
            volume,
            spacing=(1.0, 1.0, 1.0),
            target_count=128,
            boundary_margin_voxels=1,
            material_label_volume=segmentation["material_label_volume"],
        )
        self.assertGreater(points.shape[0], 0)
        self.assertGreaterEqual(np.unique(material_id).size, 2)


class GeometryAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = CTPreprocessor()

    def test_cube_faces_are_mostly_planar_and_edges_are_mostly_edge(self):
        volume = create_cube_volume(shape=(32, 32, 32), bounds=((8, 24), (8, 24), (8, 24)))
        spacing = (1.0, 1.0, 1.0)
        points = self.preprocessor.extract_surface_points(volume.astype(bool), spacing)

        analyzer = GeometryAnalyzer(sigma=1.0, planar_residual_threshold=0.003)
        mask_planar, mask_edge, _ = analyzer.classify_regions(points, volume, spacing)

        x_min, y_min, z_min = points.min(axis=0)
        y_max, z_max = points[:, 1].max(), points[:, 2].max()
        face_band = (
            (np.abs(points[:, 0] - x_min) < 0.55) &
            (points[:, 1] > y_min + 2.0) &
            (points[:, 1] < y_max - 2.0) &
            (points[:, 2] > z_min + 2.0) &
            (points[:, 2] < z_max - 2.0)
        )
        edge_band = (
            (np.abs(points[:, 0] - x_min) < 0.55) &
            (np.abs(points[:, 1] - y_min) < 0.55) &
            (points[:, 2] > z_min + 2.0) &
            (points[:, 2] < z_max - 2.0)
        )

        self.assertGreater(face_band.sum(), 10)
        self.assertGreater(edge_band.sum(), 10)
        self.assertGreater(mask_planar[face_band].mean(), 0.7)
        self.assertGreater(mask_edge[edge_band].mean(), 0.5)

    def test_sphere_surface_is_mostly_curved(self):
        volume = create_sphere_volume(shape=(36, 36, 36), center=(18, 18, 18), radius=10)
        spacing = (1.0, 1.0, 1.0)
        points = self.preprocessor.extract_surface_points(volume.astype(bool), spacing)

        analyzer = GeometryAnalyzer(sigma=1.0, planar_residual_threshold=0.003)
        _, _, mask_curved = analyzer.classify_regions(points, volume, spacing)
        self.assertGreater(mask_curved.mean(), 0.55)

    def test_anisotropic_spacing_keeps_normals_and_plane_fits_stable(self):
        volume = create_cube_volume(shape=(28, 28, 28), bounds=((6, 22), (7, 21), (8, 20)))
        spacing = (2.0, 1.0, 0.5)
        points = self.preprocessor.extract_surface_points(volume.astype(bool), spacing)

        analyzer = GeometryAnalyzer(sigma=1.0, planar_residual_threshold=0.003)
        normals = analyzer.estimate_surface_normals(points, volume, spacing)

        x_min = points[:, 0].min()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        face_band = (
            (np.abs(points[:, 0] - x_min) < 0.30) &
            (points[:, 1] > y_min + 2.0) &
            (points[:, 1] < y_max - 2.0) &
            (points[:, 2] > z_min + 4.0) &
            (points[:, 2] < z_max - 4.0)
        )

        self.assertGreater(face_band.sum(), 10)
        self.assertGreater(np.mean(np.abs(normals[face_band, 0])), 0.85)
        self.assertLess(np.mean(np.abs(normals[face_band, 1])), 0.2)
        self.assertLess(np.mean(np.abs(normals[face_band, 2])), 0.2)

        plane_params, residuals = analyzer.fit_local_planes(points[face_band], normals[face_band], k_neighbors=20)
        valid_residuals = residuals[np.isfinite(residuals)]
        self.assertGreater(valid_residuals.size, 0)
        self.assertLess(np.median(valid_residuals), 1e-3)
        valid_normals = plane_params["normal"][np.isfinite(residuals)]
        self.assertGreater(np.mean(np.abs(valid_normals[:, 0])), 0.85)


class RunCTPhase1Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_cli_failure_does_not_write_partial_outputs(self):
        raw_path = self.temp_dir / "empty.raw"
        sidecar_path = self.temp_dir / "empty.json"
        output_dir = self.temp_dir / "output"

        np.zeros((8, 8, 8), dtype=np.uint16).tofile(str(raw_path))
        sidecar = {
            "shape": [8, 8, 8],
            "dtype": "uint16",
            "spacing": [1.0, 1.0, 1.0],
            "endianness": "little",
        }
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

        exit_code = run_ct_phase1_main(
            [
                "--input", str(raw_path),
                "--fmt", "raw",
                "--raw-meta", str(sidecar_path),
                "--output", str(output_dir),
            ]
        )

        self.assertEqual(exit_code, 1)
        self.assertFalse(output_dir.exists())

    def test_success_bundle_contains_interior_samples(self):
        raw_path = self.temp_dir / "cube.raw"
        sidecar_path = self.temp_dir / "cube.json"
        output_dir = self.temp_dir / "output_ok"

        volume = np.zeros((12, 12, 12), dtype=np.uint16)
        volume[3:9, 3:9, 3:9] = 500
        volume.tofile(str(raw_path))
        sidecar_path.write_text(
            json.dumps(
                {
                    "shape": [12, 12, 12],
                    "dtype": "uint16",
                    "spacing": [1.0, 1.0, 1.0],
                    "endianness": "little",
                }
            ),
            encoding="utf-8",
        )

        exit_code = run_ct_phase1_main(
            [
                "--input", str(raw_path),
                "--fmt", "raw",
                "--raw-meta", str(sidecar_path),
                "--output", str(output_dir),
            ]
        )

        self.assertEqual(exit_code, 0)
        with np.load(str(output_dir / "analysis.npz")) as analysis:
            self.assertIn("material_mask", analysis.files)
            self.assertIn("void_mask", analysis.files)
            self.assertIn("material_label_volume", analysis.files)
            self.assertIn("surface_material_id", analysis.files)
            self.assertIn("interior_points", analysis.files)
            self.assertIn("interior_density_seed", analysis.files)
            self.assertIn("interior_material_id", analysis.files)
            self.assertGreater(analysis["interior_points"].shape[0], 0)


def nd_binary_dilation(mask, iterations):
    from scipy import ndimage

    return ndimage.binary_dilation(mask, iterations=iterations)


if __name__ == "__main__":
    unittest.main()
