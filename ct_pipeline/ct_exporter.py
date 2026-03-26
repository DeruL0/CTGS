from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.ndimage import distance_transform_edt
from torch import nn

from ct_pipeline.field_query import query_ct_density
from mesher import meshing_ct

from .acceleration import _prune_gaussian_like, clone_gaussian_like
from .compression import GSCompressor


def _dequantize(payload):
    data = np.asarray(payload["data"], dtype=np.float32)
    minimum = float(np.asarray(payload["min"], dtype=np.float32))
    scale = float(np.asarray(payload["scale"], dtype=np.float32))
    return minimum + data * scale


def _model_bbox(model, padding_factor: float = 3.0):
    xyz = model.get_xyz.detach()
    scaling = model.get_scaling.detach()
    if xyz.numel() == 0:
        raise ValueError("Cannot export an empty CTGaussianModel.")
    radius = padding_factor * scaling.max(dim=1).values.unsqueeze(1)
    lower = (xyz - radius).min(dim=0).values
    upper = (xyz + radius).max(dim=0).values
    return torch.stack((lower, upper), dim=0)


def _build_axis_samples(lower: float, upper: float, count: int):
    count = max(2, int(count))
    return np.linspace(float(lower), float(upper), count, dtype=np.float32)


def _grid_counts_from_bbox(bbox, grid_resolution: int):
    lengths = np.maximum(np.asarray(bbox[1]) - np.asarray(bbox[0]), 1e-6)
    longest = float(lengths.max())
    target_longest = max(2, int(grid_resolution))
    counts = np.maximum(2, np.ceil(lengths / longest * (target_longest - 1)).astype(np.int32) + 1)
    return counts


def _clone_for_export(model):
    cloned = clone_gaussian_like(model)
    if getattr(cloned, "optimizer", None) is not None:
        cloned.optimizer = None
    return cloned


def _apply_quantized_attributes(model, quantized):
    sh = _dequantize(quantized["sh"])
    opacity = np.clip(_dequantize(quantized["opacity"]), 1e-6, 1.0 - 1e-6)
    scaling = np.clip(_dequantize(quantized["scaling"]), 1e-6, None)

    sh_tensor = torch.as_tensor(sh, dtype=model._features_dc.dtype, device=model._features_dc.device)
    opacity_tensor = torch.as_tensor(opacity, dtype=model._opacity.dtype, device=model._opacity.device)
    scaling_tensor = torch.as_tensor(scaling, dtype=model._scaling.dtype, device=model._scaling.device)

    dc_channels = model._features_dc.shape[1]
    model._features_dc = nn.Parameter(sh_tensor[:, :dc_channels, :].contiguous().requires_grad_(True))
    model._features_rest = nn.Parameter(sh_tensor[:, dc_channels:, :].contiguous().requires_grad_(True))
    model._opacity = nn.Parameter(torch.logit(opacity_tensor, eps=1e-6).requires_grad_(True))
    model._scaling = nn.Parameter(torch.log(scaling_tensor).requires_grad_(True))


def _region_aware_keep_mask(model, surface_threshold: float, bulk_threshold_ratio: float = 0.25):
    contribution = model.get_opacity.squeeze(-1) * model.get_scaling.max(dim=1).values
    region_type = model.get_region_type.reshape(-1) if hasattr(model, "get_region_type") else torch.zeros_like(contribution, dtype=torch.long)
    surface_mask = region_type == 0
    bulk_mask = region_type == 1

    keep_mask = torch.zeros_like(surface_mask, dtype=torch.bool)
    keep_mask[surface_mask] = contribution[surface_mask] >= float(surface_threshold)
    keep_mask[bulk_mask] = contribution[bulk_mask] >= float(surface_threshold) * float(bulk_threshold_ratio)

    if contribution.numel() > 0 and int(keep_mask.sum().item()) == 0:
        keep_mask[contribution.argmax()] = True

    if torch.any(bulk_mask):
        material_id = model.get_material_id.reshape(-1) if hasattr(model, "get_material_id") else torch.zeros_like(region_type)
        unique_materials = torch.unique(material_id[bulk_mask]) if torch.any(bulk_mask) else torch.zeros((0,), dtype=torch.long, device=contribution.device)
        for material in unique_materials.tolist():
            group_mask = bulk_mask & (material_id == material)
            if not torch.any(group_mask) or torch.any(keep_mask[group_mask]):
                continue
            group_indices = torch.nonzero(group_mask, as_tuple=False).reshape(-1)
            best_local = contribution[group_indices].argmax()
            keep_mask[group_indices[best_local]] = True
    return keep_mask


def _write_mesh_ply(path, vertices, faces, normals, material_id):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("material_id", "i4"),
    ]
    vertices_np = np.asarray(vertices, dtype=np.float32)
    normals_np = np.asarray(normals, dtype=np.float32)
    material_np = np.asarray(material_id, dtype=np.int32).reshape(-1)

    vertex_array = np.empty(vertices_np.shape[0], dtype=vertex_dtype)
    vertex_array["x"] = vertices_np[:, 0]
    vertex_array["y"] = vertices_np[:, 1]
    vertex_array["z"] = vertices_np[:, 2]
    vertex_array["nx"] = normals_np[:, 0]
    vertex_array["ny"] = normals_np[:, 1]
    vertex_array["nz"] = normals_np[:, 2]
    vertex_array["material_id"] = material_np

    face_dtype = [("vertex_indices", "i4", (3,))]
    face_array = np.empty(len(faces), dtype=face_dtype)
    face_array["vertex_indices"] = np.asarray(faces, dtype=np.int32)

    PlyData(
        [
            PlyElement.describe(vertex_array, "vertex"),
            PlyElement.describe(face_array, "face"),
        ]
    ).write(str(path))
    return path


class CTExporter:
    def __init__(self):
        self.compressor = GSCompressor()

    def export_display_gs(self, model, path, compress=True):
        export_model = _clone_for_export(model)
        if compress:
            keep_mask = _region_aware_keep_mask(export_model, surface_threshold=0.01, bulk_threshold_ratio=0.25)
            export_model = _prune_gaussian_like(export_model, keep_mask)
            quantized = self.compressor.quantize_attributes(export_model, bits=8)
            _apply_quantized_attributes(export_model, quantized)

        export_path = Path(path)
        if export_path.suffix == "":
            export_path = export_path.with_suffix(".ply")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_model.save_ply(str(export_path))
        return export_path

    def export_metrology_mesh(self, model, path, resolution=0.05):
        mesh = meshing_ct(None, model, resolution=resolution, threshold=0.5)
        export_path = Path(path)
        if export_path.suffix == "":
            export_path = export_path.with_suffix(".ply")
        return _write_mesh_ply(export_path, mesh["vertices"], mesh["faces"], mesh["normals"], mesh["material_id"])

    def export_sdf(self, model, path, grid_resolution=256):
        bbox = _model_bbox(model).detach().cpu().numpy()
        counts = _grid_counts_from_bbox(bbox, grid_resolution=grid_resolution)
        xs = _build_axis_samples(bbox[0, 0], bbox[1, 0], counts[0])
        ys = _build_axis_samples(bbox[0, 1], bbox[1, 1], counts[1])
        zs = _build_axis_samples(bbox[0, 2], bbox[1, 2], counts[2])

        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
        device = model.get_xyz.device
        density = query_ct_density(model, torch.as_tensor(points, dtype=model.get_xyz.dtype, device=device)).detach().cpu().numpy()
        density_xyz = density.reshape(len(xs), len(ys), len(zs))
        occupancy_xyz = density_xyz >= 0.5

        occupancy_zyx = np.transpose(occupancy_xyz, (2, 1, 0))
        spacing_xyz = [
            float(xs[1] - xs[0]) if len(xs) > 1 else 1.0,
            float(ys[1] - ys[0]) if len(ys) > 1 else 1.0,
            float(zs[1] - zs[0]) if len(zs) > 1 else 1.0,
        ]
        sampling_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        outside_dt = distance_transform_edt(~occupancy_zyx, sampling=sampling_zyx)
        inside_dt = distance_transform_edt(occupancy_zyx, sampling=sampling_zyx)
        sdf_zyx = (outside_dt - inside_dt).astype(np.float32)

        export_path = Path(path)
        if export_path.suffix == "":
            export_path = export_path.with_suffix(".npy")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(export_path), sdf_zyx)

        sidecar = {
            "origin_xyz": [float(bbox[0, 0]), float(bbox[0, 1]), float(bbox[0, 2])],
            "spacing_xyz": spacing_xyz,
            "shape_zyx": list(sdf_zyx.shape),
        }
        sidecar_path = export_path.with_suffix(".json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return export_path, sidecar_path
