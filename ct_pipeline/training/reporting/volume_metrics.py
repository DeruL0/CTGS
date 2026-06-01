import numpy as np
import torch
from scipy import ndimage

from ct_pipeline.rendering.fields import (
    bulk_intensity_readout,
    compose_signed_overlap_occupancy,
    density_to_occupancy,
    query_ct_density_from_state_by_region,
    query_ct_fields_unified,
    query_ct_local_intensity_from_state,
)
from ct_pipeline.training.losses import sample_volume_field
from ct_pipeline.training.sampling import _interior_void_air_mask_np

from .common import (
    _mask_to_np,
    _quantile_metrics,
    _roi_bbox_from_analysis,
    _roi_window_from_analysis,
    _sample_mask_points,
)


def _record_volume_material_intensity_metrics(
    metrics,
    training_state,
    analysis,
    spacing_zyx,
    device,
    dtype,
    *,
    signed_distance_field=None,
    volume_cuda=None,
    config=None,
    intensity_air: float = 0.0,
    max_count: int = 16384,
):
    """Record the primary material-side intensity metrics over the 3D volume."""
    material_mask = _mask_to_np(analysis.get("material_mask", analysis.get("coarse_support_mask")))
    if material_mask is None or volume_cuda is None or not np.any(material_mask):
        return
    points = _sample_mask_points(material_mask, spacing_zyx, device, dtype, int(max_count))
    if points is None or points.numel() == 0:
        return
    volume_field = volume_cuda.reshape(1, 1, *tuple(int(value) for value in volume_cuda.shape[-3:]))
    target = sample_volume_field(volume_field, points, spacing_zyx).reshape(-1).to(dtype=torch.float32)
    signed_distance = None
    if signed_distance_field is not None:
        signed_distance = sample_volume_field(
            signed_distance_field["signed_distance"],
            points,
            signed_distance_field["spacing_zyx"],
        ).reshape(-1).to(dtype=torch.float32)
    fields = query_ct_fields_unified(
        points,
        training_state,
        signed_distance=signed_distance,
        config=config,
        intensity_air=float(intensity_air),
        include_surface=False,
        train_ct_value=False,
        apply_bulk_gate=False,
    )
    pred = fields.get("A_b")
    if pred is None:
        pred = bulk_intensity_readout(
            fields.get("I_b_raw", fields["I_b"]).to(dtype=torch.float32),
            fields["den_b"].to(dtype=torch.float32),
        )
    pred = pred.to(dtype=torch.float32)
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if not torch.any(valid):
        return
    pred = pred[valid]
    target = target[valid]
    residual = pred - target
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    corr_den = torch.sqrt(torch.sum(pred_centered * pred_centered) * torch.sum(target_centered * target_centered))
    corr = torch.sum(pred_centered * target_centered) / corr_den.clamp_min(1e-12)
    metrics["material_volume_MAE"] = float(torch.mean(torch.abs(residual)).item())
    metrics["material_volume_RMSE"] = float(torch.sqrt(torch.mean(residual * residual)).item())
    metrics["material_volume_corr"] = float(corr.item()) if float(corr_den.item()) > 1e-12 else float("nan")
    metrics["material_volume_sample_count"] = int(valid.sum().item())
    for suffix, value in _quantile_metrics(pred.detach().cpu().numpy(), ("p10", "p50", "p90")).items():
        metrics[f"material_volume_A_b_{suffix}"] = value

def _record_high_gradient_material_metrics(
    metrics,
    training_state,
    analysis,
    spacing_zyx,
    device,
    dtype,
    *,
    signed_distance_field=None,
    volume_cuda=None,
    config=None,
    intensity_air: float = 0.0,
    max_count: int = 4096,
):
    material_mask = _mask_to_np(analysis.get("material_mask", analysis.get("coarse_support_mask")))
    if material_mask is None or volume_cuda is None or not np.any(material_mask):
        return
    volume_np = volume_cuda.detach().cpu().numpy().astype(np.float32, copy=False)
    if volume_np.shape != material_mask.shape:
        return
    blur_sigma = float(getattr(config, "ct_feature_adaptive_blur_sigma_vox", 0.75)) if config is not None else 0.75
    smoothed = ndimage.gaussian_filter(volume_np, sigma=max(blur_sigma, 0.0)).astype(np.float32)
    grad_z, grad_y, grad_x = np.gradient(smoothed)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).astype(np.float32)
    material_grad = grad_mag[material_mask & np.isfinite(grad_mag)]
    if material_grad.size == 0:
        return
    q75 = float(np.quantile(material_grad, 0.75))
    q95 = float(np.quantile(material_grad, 0.95))
    if q95 <= q75:
        return
    grad_score = np.clip((grad_mag - q75) / max(q95 - q75, 1e-8), 0.0, 1.0)
    high_gradient = material_mask & (grad_score > 0.75)
    points = _sample_mask_points(high_gradient, spacing_zyx, device, dtype, int(max_count))
    if points is None or points.numel() == 0:
        return
    volume_field = volume_cuda.reshape(1, 1, *tuple(int(value) for value in volume_cuda.shape[-3:]))
    target = sample_volume_field(volume_field, points, spacing_zyx).reshape(-1).to(dtype=torch.float32)
    signed_distance = None
    if signed_distance_field is not None:
        signed_distance = sample_volume_field(
            signed_distance_field["signed_distance"],
            points,
            signed_distance_field["spacing_zyx"],
        ).reshape(-1).to(dtype=torch.float32)
    fields = query_ct_fields_unified(
        points,
        training_state,
        signed_distance=signed_distance,
        config=config,
        intensity_air=float(intensity_air),
        include_surface=True,
        train_ct_value=False,
        apply_bulk_gate=False,
    )
    if "A_b" in fields:
        pred = fields["A_b"].to(dtype=torch.float32)
    else:
        pred = bulk_intensity_readout(fields["mu_raw"], fields["W_b"]).to(dtype=torch.float32)
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if torch.any(valid):
        metrics["high_gradient_material_MAE"] = float(torch.mean(torch.abs(pred[valid] - target[valid])).item())
        metrics["high_gradient_material_sample_count"] = int(valid.sum().item())

def _record_false_hole_diagnostics(
    metrics,
    training_state,
    analysis,
    spacing_zyx,
    device,
    dtype,
    *,
    signed_distance_field=None,
    volume_cuda=None,
    intensity_air: float = 0.0,
    intensity_mat: float = 1.0,
    boundary_band_distance: float = 1.5,
    false_hole_sample_count: int = 4096,
    false_hole_boundary_band: float = 2.0,
    false_hole_material_threshold: float = 0.65,
    false_hole_dark_margin: float = 0.15,
    surface_material_gate_sigma=None,
    material_compose_mode: str = "bulk_first_material",
    config=None,
    use_unified_compositor: bool = True,
):
    material_np = _mask_to_np(analysis.get("material_mask", analysis.get("coarse_support_mask")))
    if material_np is None or not np.any(material_np):
        return
    roi_window = _roi_window_from_analysis(analysis, material_np.shape)
    roi_bbox = _roi_bbox_from_analysis(analysis, material_np.shape)
    air_np = np.logical_and(roi_window, np.logical_not(material_np))
    interior_void_np = _interior_void_air_mask_np(air_np, roi_bbox=roi_bbox)
    structure = ndimage.generate_binary_structure(3, 1)
    cavity_material_shell = np.logical_and(
        material_np,
        ndimage.binary_dilation(interior_void_np, structure=structure, iterations=3),
    )
    cavity_void_shell = np.logical_and(
        interior_void_np,
        ndimage.binary_dilation(material_np, structure=structure, iterations=1),
    )

    points = _sample_mask_points(cavity_material_shell, spacing_zyx, device, dtype, int(false_hole_sample_count))
    if points is not None:
        metrics["false_hole_candidate_count"] = int(points.shape[0])
    if points is not None and volume_cuda is not None:
        volume_field = volume_cuda.reshape(1, 1, *tuple(int(value) for value in volume_cuda.shape[-3:]))
        target_intensity = sample_volume_field(volume_field, points, spacing_zyx).reshape(-1).to(dtype=torch.float32)
        bulk_density = query_ct_density_from_state_by_region(training_state, points, region="bulk", detach=True).to(dtype=torch.float32)
        surface_density = query_ct_density_from_state_by_region(training_state, points, region="surface", detach=True).to(dtype=torch.float32)
        if bool(use_unified_compositor):
            signed_distance_values = None
            if signed_distance_field is not None:
                signed_distance_values = sample_volume_field(
                    signed_distance_field["signed_distance"],
                    points,
                    signed_distance_field["spacing_zyx"],
                ).reshape(-1).to(dtype=torch.float32)
            unified = query_ct_fields_unified(
                points,
                training_state,
                signed_distance=signed_distance_values,
                config=config,
                intensity_air=float(intensity_air),
                include_surface=True,
                train_ct_value=False,
                apply_bulk_gate=False,
            )
            predicted_intensity = unified["I_pred"].to(dtype=torch.float32)
        else:
            combined_occupancy, _, _ = compose_signed_overlap_occupancy(
                bulk_density,
                surface_density,
                None,
                boundary_band_distance,
                surface_material_gate_sigma=surface_material_gate_sigma,
                material_compose_mode=material_compose_mode,
            )
            local_intensity = query_ct_local_intensity_from_state(
                training_state,
                points,
                detach_geometry=True,
                include_surface=True,
                signed_distance=None,
                boundary_band_distance=boundary_band_distance,
                surface_material_gate_sigma=surface_material_gate_sigma,
                material_compose_mode=material_compose_mode,
            ).to(dtype=torch.float32)
            predicted_intensity = float(intensity_air) + (local_intensity - float(intensity_air)) * combined_occupancy
        intensity_range = max(abs(float(intensity_mat) - float(intensity_air)), 1e-6)
        material_threshold = float(intensity_air) + float(false_hole_material_threshold) * (float(intensity_mat) - float(intensity_air))
        dark_margin = float(false_hole_dark_margin) * intensity_range
        active = (
            torch.isfinite(target_intensity)
            & torch.isfinite(predicted_intensity)
            & (target_intensity >= material_threshold)
            & ((target_intensity - predicted_intensity) > dark_margin)
        )
        metrics["false_hole_active_ratio"] = float(active.float().mean().item()) if active.numel() > 0 else 0.0
        for suffix, value in _quantile_metrics(predicted_intensity[active].detach().cpu().numpy(), ("p10", "p50")).items():
            metrics[f"false_hole_pred_intensity_{suffix}"] = value
        metrics["false_hole_target_intensity_p50"] = _quantile_metrics(
            target_intensity[active].detach().cpu().numpy(),
            ("p50",),
        )["p50"]
        bulk_occ = density_to_occupancy(bulk_density)
        for suffix, value in _quantile_metrics(bulk_occ[active].detach().cpu().numpy(), ("p10", "p50")).items():
            metrics[f"false_hole_bulk_occ_{suffix}"] = value

        # surface-owned false hole ratio: fraction of active false holes in boundary shell (sdf 鈭?[-band, 0])
        if signed_distance_field is not None and active.any():
            sdf_at_points = sample_volume_field(
                signed_distance_field["signed_distance"],
                points,
                signed_distance_field["spacing_zyx"],
            ).reshape(-1).to(dtype=torch.float32)
            in_boundary_shell = (sdf_at_points >= -float(false_hole_boundary_band)) & (sdf_at_points <= 0.0)
            active_in_shell = active & in_boundary_shell
            metrics["surface_owned_false_hole_ratio"] = float(active_in_shell.float().sum().item() / max(1, int(active.float().sum().item())))

    void_points = _sample_mask_points(cavity_void_shell, spacing_zyx, device, dtype, int(false_hole_sample_count))
    if void_points is not None:
        bulk_occ = density_to_occupancy(
            query_ct_density_from_state_by_region(training_state, void_points, region="bulk", detach=True)
        ).to(dtype=torch.float32)
        metrics["cavity_void_false_fill_ratio"] = float((bulk_occ > 0.35).float().mean().item())
