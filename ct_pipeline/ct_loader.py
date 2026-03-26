import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover - exercised when dependency is missing
    pydicom = None

try:
    import tifffile
except ImportError:  # pragma: no cover - exercised when dependency is missing
    tifffile = None


def _as_float_list(values: Sequence[Any], expected_len: int, field_name: str) -> List[float]:
    if len(values) != expected_len:
        raise ValueError("{0} must contain {1} values, got {2}".format(field_name, expected_len, len(values)))
    return [float(value) for value in values]


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError("Encountered a zero-length orientation vector.")
    return vector / norm


def _build_affine(origin_xyz: Sequence[float], direction_3x3: Sequence[Sequence[float]], spacing_zyx: Sequence[float]) -> np.ndarray:
    spacing_zyx = np.asarray(spacing_zyx, dtype=np.float64)
    direction = np.asarray(direction_3x3, dtype=np.float64)
    if direction.shape != (3, 3):
        raise ValueError("direction_3x3 must be a 3x3 matrix.")

    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = direction[:, 0] * float(spacing_zyx[2])
    affine[:3, 1] = direction[:, 1] * float(spacing_zyx[1])
    affine[:3, 2] = direction[:, 2] * float(spacing_zyx[0])
    affine[:3, 3] = np.asarray(origin_xyz, dtype=np.float64)
    return affine


def _normalize_volume(volume: np.ndarray, method: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    volume = volume.astype(np.float32, copy=False)
    min_value = float(np.min(volume))
    max_value = float(np.max(volume))

    if max_value <= min_value:
        normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        denominator = max_value - min_value
        if volume.flags.writeable:
            normalized = volume
        else:
            normalized = volume.copy()
        normalized -= min_value
        normalized /= denominator

    return normalized.astype(np.float32, copy=False), {
        "method": method,
        "input_min": min_value,
        "input_max": max_value,
        "output_range": [0.0, 1.0],
    }


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _unit_scale_to_mm(unit: Optional[str]) -> Optional[float]:
    if unit is None:
        return None

    normalized = unit.strip().lower()
    unit_map = {
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "um": 1e-3,
        "µm": 1e-3,
        "micrometer": 1e-3,
        "micrometers": 1e-3,
        "micron": 1e-3,
        "microns": 1e-3,
        "cm": 10.0,
        "centimeter": 10.0,
        "centimeters": 10.0,
        "m": 1000.0,
        "meter": 1000.0,
        "meters": 1000.0,
        "nm": 1e-6,
        "nanometer": 1e-6,
        "nanometers": 1e-6,
        "inch": 25.4,
        "inches": 25.4,
    }
    return unit_map.get(normalized)


def _resolution_unit_scale_to_mm(resolution_unit: Optional[Any]) -> Optional[float]:
    if resolution_unit is None:
        return None

    normalized = str(resolution_unit).strip().upper()
    if "." in normalized:
        normalized = normalized.split(".")[-1]
    unit_map = {
        "INCH": 25.4,
        "CENTIMETER": 10.0,
        "MM": 1.0,
        "MILLIMETER": 1.0,
        "NONE": None,
    }
    return unit_map.get(normalized)


class CTVolumeLoader:
    """Load CT volumes from DICOM series, RAW binaries, or TIFF stacks."""

    def __init__(self) -> None:
        self._metadata: Dict[str, Any] = {}
        self._spacing_zyx: Optional[Tuple[float, float, float]] = None

    def load(self, path: str, fmt: str = "auto", raw_meta_path: Optional[str] = None) -> np.ndarray:
        input_path = Path(path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError("CT input path does not exist: {0}".format(input_path))

        resolved_format = self._resolve_format(input_path, fmt)
        sidecar = None
        if raw_meta_path is not None:
            sidecar = _read_json(Path(raw_meta_path).expanduser().resolve())

        if resolved_format == "raw":
            volume, metadata = self._load_raw(input_path, sidecar)
        elif resolved_format == "tiff":
            volume, metadata = self._load_tiff(input_path, sidecar)
        elif resolved_format == "dicom":
            volume, metadata = self._load_dicom(input_path)
        else:
            raise ValueError("Unsupported CT format: {0}".format(resolved_format))

        self._metadata = metadata
        self._spacing_zyx = tuple(float(value) for value in metadata["spacing_zyx"])
        return volume

    def get_voxel_spacing(self) -> Tuple[float, float, float]:
        if self._spacing_zyx is None:
            raise RuntimeError("No CT volume has been loaded yet.")
        return self._spacing_zyx

    def get_metadata(self) -> Dict[str, Any]:
        if not self._metadata:
            raise RuntimeError("No CT volume has been loaded yet.")
        return dict(self._metadata)

    def _resolve_format(self, path: Path, fmt: str) -> str:
        normalized = fmt.lower()
        if normalized != "auto":
            if normalized not in {"dicom", "raw", "tiff"}:
                raise ValueError("fmt must be one of: auto, dicom, raw, tiff.")
            return normalized

        suffix = path.suffix.lower()
        if suffix in {".raw", ".bin"}:
            return "raw"
        if suffix in {".tif", ".tiff"}:
            return "tiff"
        if path.is_dir():
            tiff_files = list(path.glob("*.tif")) + list(path.glob("*.tiff"))
            if tiff_files:
                return "tiff"
            return "dicom"
        return "dicom"

    def _load_raw(self, path: Path, sidecar: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if sidecar is None:
            raise ValueError("RAW volumes require a JSON sidecar with shape, dtype, spacing, and endianness.")

        required_fields = ["shape", "dtype", "spacing", "endianness"]
        missing_fields = [field for field in required_fields if field not in sidecar]
        if missing_fields:
            raise ValueError("RAW sidecar is missing required fields: {0}".format(", ".join(missing_fields)))

        shape_dhw = [int(value) for value in sidecar["shape"]]
        if len(shape_dhw) != 3:
            raise ValueError("RAW sidecar shape must be [D, H, W].")

        spacing_zyx = _as_float_list(sidecar["spacing"], 3, "spacing")
        endianness = str(sidecar["endianness"]).strip().lower()
        if endianness not in {"little", "big", "native", "<", ">", "="}:
            raise ValueError("RAW sidecar endianness must be little, big, native, <, >, or =.")

        dtype = np.dtype(str(sidecar["dtype"]))
        if dtype.byteorder == "|":
            byteorder = "|"
        else:
            byteorder = {"little": "<", "big": ">", "native": "=", "<": "<", ">": ">", "=": "="}[endianness]
        raw_dtype = np.dtype(byteorder + dtype.str.lstrip("<>="))

        offset_bytes = int(sidecar.get("offset_bytes", 0))
        expected_voxels = int(np.prod(shape_dhw))
        with path.open("rb") as handle:
            if offset_bytes:
                handle.seek(offset_bytes)
            flat = np.fromfile(handle, dtype=raw_dtype, count=expected_voxels)

        if flat.size != expected_voxels:
            raise ValueError(
                "RAW file size does not match sidecar shape. Expected {0} voxels, found {1}.".format(
                    expected_voxels, flat.size
                )
            )

        volume = flat.reshape(shape_dhw).astype(np.float32)
        volume, normalization = _normalize_volume(volume, method="min_max")

        origin_xyz = _as_float_list(sidecar.get("origin", [0.0, 0.0, 0.0]), 3, "origin")
        direction_3x3 = sidecar.get("direction", np.eye(3, dtype=np.float64).tolist())
        affine = _build_affine(origin_xyz, direction_3x3, spacing_zyx)

        metadata = {
            "format": "raw",
            "shape_dhw": shape_dhw,
            "spacing_zyx": spacing_zyx,
            "origin_xyz": origin_xyz,
            "direction_3x3": np.asarray(direction_3x3, dtype=np.float64).tolist(),
            "voxel_to_world_4x4": affine.tolist(),
            "dtype": "float32",
            "scan_params": sidecar.get("scan_params", {}),
            "normalization": normalization,
        }
        return volume, metadata

    def _load_tiff(self, path: Path, sidecar: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if tifffile is None:
            raise ImportError("tifffile is required to load TIFF CT volumes.")

        if path.is_dir():
            tiff_paths = sorted(list(path.glob("*.tif")) + list(path.glob("*.tiff")))
            if not tiff_paths:
                raise ValueError("No TIFF slices were found in directory: {0}".format(path))

            slices = [tifffile.imread(str(slice_path)) for slice_path in tiff_paths]
            volume = np.stack([np.asarray(slice_array) for slice_array in slices], axis=0)
            metadata_file = tiff_paths[0]
        else:
            volume = np.asarray(tifffile.imread(str(path)))
            metadata_file = path

        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        if volume.ndim != 3:
            raise ValueError("TIFF CT volume must be 3D after loading, got shape {0}.".format(volume.shape))

        volume = volume.astype(np.float32)

        embedded_spacing = self._read_tiff_spacing(metadata_file)
        if sidecar is not None and "spacing" in sidecar:
            spacing_zyx = _as_float_list(sidecar["spacing"], 3, "spacing")
        elif embedded_spacing is not None:
            spacing_zyx = embedded_spacing
        else:
            raise ValueError(
                "TIFF spacing was not found in embedded metadata. Provide a JSON sidecar with spacing."
            )

        origin_xyz = [0.0, 0.0, 0.0]
        direction_3x3 = np.eye(3, dtype=np.float64).tolist()
        scan_params: Dict[str, Any] = {}
        if sidecar is not None:
            origin_xyz = _as_float_list(sidecar.get("origin", origin_xyz), 3, "origin")
            direction_3x3 = sidecar.get("direction", direction_3x3)
            scan_params = sidecar.get("scan_params", {})

        affine = _build_affine(origin_xyz, direction_3x3, spacing_zyx)
        volume, normalization = _normalize_volume(volume, method="min_max")

        metadata = {
            "format": "tiff",
            "shape_dhw": [int(value) for value in volume.shape],
            "spacing_zyx": spacing_zyx,
            "origin_xyz": origin_xyz,
            "direction_3x3": np.asarray(direction_3x3, dtype=np.float64).tolist(),
            "voxel_to_world_4x4": affine.tolist(),
            "dtype": "float32",
            "scan_params": scan_params,
            "normalization": normalization,
        }
        return volume, metadata

    def _read_tiff_spacing(self, path: Path) -> Optional[List[float]]:
        if tifffile is None:
            return None

        with tifffile.TiffFile(str(path)) as tif:
            ome_spacing = self._read_ome_spacing(tif)
            if ome_spacing is not None:
                return ome_spacing

            imagej_spacing = self._read_imagej_spacing(tif)
            if imagej_spacing is not None:
                return imagej_spacing

        return None

    def _read_ome_spacing(self, tif: Any) -> Optional[List[float]]:
        ome_metadata = getattr(tif, "ome_metadata", None)
        if not ome_metadata:
            return None

        try:
            root = ET.fromstring(ome_metadata)
        except ET.ParseError:
            return None

        pixels_node = root.find(".//{*}Pixels")
        if pixels_node is None:
            return None

        physical_size_x = pixels_node.attrib.get("PhysicalSizeX")
        physical_size_y = pixels_node.attrib.get("PhysicalSizeY")
        physical_size_z = pixels_node.attrib.get("PhysicalSizeZ")
        if physical_size_x is None or physical_size_y is None or physical_size_z is None:
            return None

        scale_x = _unit_scale_to_mm(pixels_node.attrib.get("PhysicalSizeXUnit", "mm"))
        scale_y = _unit_scale_to_mm(pixels_node.attrib.get("PhysicalSizeYUnit", "mm"))
        scale_z = _unit_scale_to_mm(pixels_node.attrib.get("PhysicalSizeZUnit", "mm"))
        if scale_x is None or scale_y is None or scale_z is None:
            return None

        return [
            float(physical_size_z) * scale_z,
            float(physical_size_y) * scale_y,
            float(physical_size_x) * scale_x,
        ]

    def _read_imagej_spacing(self, tif: Any) -> Optional[List[float]]:
        imagej_metadata = getattr(tif, "imagej_metadata", None) or {}
        z_spacing = imagej_metadata.get("spacing")
        if z_spacing is None:
            return None

        first_page = tif.pages[0]
        x_resolution = first_page.tags.get("XResolution")
        y_resolution = first_page.tags.get("YResolution")
        resolution_unit_tag = first_page.tags.get("ResolutionUnit")
        if x_resolution is None or y_resolution is None:
            return None

        unit_scale = _resolution_unit_scale_to_mm(
            resolution_unit_tag.value if resolution_unit_tag is not None else None
        )
        z_unit_scale = _unit_scale_to_mm(imagej_metadata.get("unit", "mm"))
        if unit_scale is None or z_unit_scale is None:
            return None

        x_value = x_resolution.value
        y_value = y_resolution.value
        x_resolution_value = float(x_value[0]) / float(x_value[1]) if isinstance(x_value, tuple) else float(x_value)
        y_resolution_value = float(y_value[0]) / float(y_value[1]) if isinstance(y_value, tuple) else float(y_value)
        if x_resolution_value <= 0 or y_resolution_value <= 0:
            return None

        spacing_x = unit_scale / x_resolution_value
        spacing_y = unit_scale / y_resolution_value
        spacing_z = float(z_spacing) * z_unit_scale
        return [spacing_z, spacing_y, spacing_x]

    def _load_dicom(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        if pydicom is None:
            raise ImportError("pydicom is required to load DICOM CT volumes.")

        headers = self._collect_dicom_headers(path)
        if not headers:
            raise ValueError("No readable DICOM slices were found at: {0}".format(path))

        reference_header = headers[0][1]
        x_dir, y_dir, z_dir = self._extract_dicom_orientation(reference_header)
        sorted_headers = self._sort_dicom_headers(headers, z_dir)

        slices = []
        original_dtype = None
        positions = []
        for file_path, _ in sorted_headers:
            dataset = pydicom.dcmread(str(file_path), force=True)
            pixel_array = dataset.pixel_array
            if pixel_array.ndim != 2:
                raise ValueError("Expected 2D DICOM slices, got shape {0}.".format(pixel_array.shape))
            if original_dtype is None:
                original_dtype = str(pixel_array.dtype)

            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            slices.append(pixel_array.astype(np.float32) * slope + intercept)

            image_position = getattr(dataset, "ImagePositionPatient", None)
            if image_position is not None:
                positions.append([float(value) for value in image_position])

        volume = np.stack(slices, axis=0).astype(np.float32)
        spacing_zyx = self._extract_dicom_spacing(sorted_headers, z_dir)
        origin_xyz = positions[0] if positions else [0.0, 0.0, 0.0]
        direction_3x3 = np.column_stack((x_dir, y_dir, z_dir))
        affine = _build_affine(origin_xyz, direction_3x3, spacing_zyx)

        scan_fields = [
            "SeriesDescription",
            "Modality",
            "Manufacturer",
            "ManufacturerModelName",
            "KVP",
            "TubeCurrent",
            "ExposureTime",
            "ConvolutionKernel",
            "SliceThickness",
            "SpacingBetweenSlices",
        ]
        scan_params = {}
        for field_name in scan_fields:
            if hasattr(reference_header, field_name):
                scan_params[field_name] = str(getattr(reference_header, field_name))
        if original_dtype is not None:
            scan_params["source_dtype"] = original_dtype

        volume, normalization = _normalize_volume(volume, method="dicom_rescale_min_max")
        metadata = {
            "format": "dicom",
            "shape_dhw": [int(value) for value in volume.shape],
            "spacing_zyx": [float(value) for value in spacing_zyx],
            "origin_xyz": [float(value) for value in origin_xyz],
            "direction_3x3": direction_3x3.astype(np.float64).tolist(),
            "voxel_to_world_4x4": affine.tolist(),
            "dtype": "float32",
            "scan_params": scan_params,
            "normalization": normalization,
        }
        return volume, metadata

    def _collect_dicom_headers(self, path: Path) -> List[Tuple[Path, Any]]:
        if pydicom is None:
            raise ImportError("pydicom is required to load DICOM CT volumes.")

        if path.is_dir():
            candidate_files = sorted([candidate for candidate in path.iterdir() if candidate.is_file()])
            reference_series_uid = None
        else:
            candidate_files = sorted([candidate for candidate in path.parent.iterdir() if candidate.is_file()])
            reference_header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            reference_series_uid = getattr(reference_header, "SeriesInstanceUID", None)

        headers = []
        for candidate_file in candidate_files:
            try:
                header = pydicom.dcmread(str(candidate_file), stop_before_pixels=True, force=True)
            except Exception:
                continue
            if not hasattr(header, "Rows") or not hasattr(header, "Columns"):
                continue
            if reference_series_uid is not None:
                candidate_series_uid = getattr(header, "SeriesInstanceUID", None)
                if candidate_series_uid != reference_series_uid:
                    continue
            headers.append((candidate_file, header))

        if path.is_dir():
            series_ids = {getattr(header, "SeriesInstanceUID", None) for _, header in headers if getattr(header, "SeriesInstanceUID", None) is not None}
            if len(series_ids) > 1:
                raise ValueError(
                    "Multiple DICOM series were found in {0}. Pass a representative slice file instead.".format(path)
                )

        return headers

    def _extract_dicom_orientation(self, header: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        orientation = getattr(header, "ImageOrientationPatient", None)
        if orientation is None:
            x_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            y_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            z_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            return x_dir, y_dir, z_dir

        x_dir = _normalize_vector(np.asarray([float(value) for value in orientation[:3]], dtype=np.float64))
        y_dir = _normalize_vector(np.asarray([float(value) for value in orientation[3:]], dtype=np.float64))
        z_dir = _normalize_vector(np.cross(x_dir, y_dir))
        return x_dir, y_dir, z_dir

    def _sort_dicom_headers(self, headers: List[Tuple[Path, Any]], z_dir: np.ndarray) -> List[Tuple[Path, Any]]:
        sortable: List[Tuple[int, Any, Tuple[Path, Any]]] = []
        for item in headers:
            file_path, header = item
            image_position = getattr(header, "ImagePositionPatient", None)
            instance_number = getattr(header, "InstanceNumber", None)

            if image_position is not None:
                location = float(np.dot(np.asarray([float(value) for value in image_position], dtype=np.float64), z_dir))
                sortable.append((0, location, item))
            elif instance_number is not None:
                sortable.append((1, float(instance_number), item))
            else:
                sortable.append((2, file_path.name, item))

        sortable.sort(key=lambda entry: (entry[0], entry[1]))
        return [entry[2] for entry in sortable]

    def _extract_dicom_spacing(self, headers: List[Tuple[Path, Any]], z_dir: np.ndarray) -> List[float]:
        first_header = headers[0][1]
        pixel_spacing = getattr(first_header, "PixelSpacing", None)
        if pixel_spacing is None:
            pixel_spacing = getattr(first_header, "ImagerPixelSpacing", None)
        if pixel_spacing is None:
            raise ValueError("DICOM slices are missing PixelSpacing metadata.")

        sy = float(pixel_spacing[0])
        sx = float(pixel_spacing[1])
        sz: Optional[float] = None

        positions = []
        for _, header in headers:
            image_position = getattr(header, "ImagePositionPatient", None)
            if image_position is None:
                positions = []
                break
            positions.append(np.asarray([float(value) for value in image_position], dtype=np.float64))

        if len(positions) >= 2:
            locations = np.asarray([np.dot(position, z_dir) for position in positions], dtype=np.float64)
            diffs = np.diff(locations)
            nonzero_diffs = np.abs(diffs[np.abs(diffs) > 1e-6])
            if nonzero_diffs.size > 0:
                sz = float(np.median(nonzero_diffs))

        if sz is None:
            if hasattr(first_header, "SpacingBetweenSlices"):
                sz = float(first_header.SpacingBetweenSlices)
            elif hasattr(first_header, "SliceThickness"):
                sz = float(first_header.SliceThickness)
            else:
                raise ValueError("DICOM slices are missing slice spacing metadata.")

        return [float(sz), sy, sx]
