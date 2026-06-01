from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from ct_pipeline.backend import prepare_ct_training_state, render_ct_slice_patch_native
from ct_pipeline.exporting import CTExporter
from ct_pipeline.rendering.bulk_support import resolve_bulk_query_truncation_sigma
from ct_pipeline.rendering.fields import bulk_intensity_readout, is_bulk_intensity_field_mode, query_ct_fields_unified
from ct_pipeline.rendering.slices import _build_query_points_from_base, sample_gt_slice_patch
from ct_pipeline.training.sampling import _sample_signed_distance
from ct_pipeline.training.session import build_renderer_autocast_kwargs, get_lpips_model
from ct_pipeline.training.utils import write_key_value_report
from utils.loss_utils import ssim


def _correct_bulk_intensity_preview(
    fields: dict[str, torch.Tensor],
    signed_distance: torch.Tensor | None,
    intensity_air: float,
    config=None,
    *,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    del signed_distance, intensity_air, config
    # CTGS-vFinal diagnostics/rendering are always ungated raw readouts.
    # Do not emit hard-mask or SDF-soft variants here; those hide leakage.
    surface_only = fields["I_s"].to(dtype=torch.float32).clamp(0.0, 1.0)
    bulk_only = fields.get("A_b")
    if bulk_only is None:
        raw_bulk = fields.get("I_b_raw", fields["I_b"]).to(dtype=torch.float32)
        bulk_only = bulk_intensity_readout(raw_bulk, fields["den_b"], eps=eps)
    bulk_only = bulk_only.to(dtype=torch.float32).clamp(0.0, 1.0)
    return {
        "surface_only": surface_only,
        "bulk_only": bulk_only,
        "unified": bulk_only,
    }


def _export_ct_outputs(gaussians, args):
    exporter = CTExporter()
    outputs = {}
    if args.output_gs:
        outputs["output_gs"] = str(exporter.export_display_gs(gaussians, args.output_gs, compress=True))
    if not args.skip_export_mesh and args.output_mesh:
        outputs["output_mesh"] = str(
            exporter.export_metrology_mesh(gaussians, args.output_mesh, resolution=args.export_mesh_resolution)
        )
    if not args.skip_export_sdf and args.output_sdf:
        sdf_path, sdf_meta = exporter.export_sdf(gaussians, args.output_sdf, grid_resolution=args.export_sdf_resolution)
        outputs["output_sdf"] = str(sdf_path)
        outputs["output_sdf_metadata"] = str(sdf_meta)
    return outputs


def _save_ct_middle_slice_preview(
    gaussians,
    volume_cuda,
    spacing_zyx,
    model_path,
    slice_tile_size,
    truncation_sigma,
    grid_cell_voxels,
    bulk_truncation_sigma=None,
    iteration=None,
    output_dir=None,
    intensity_air=0.0,
    intensity_mat=1.0,
    **_kwargs,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Skipping CT middle-slice preview generation: {exc!r}", RuntimeWarning, stacklevel=2)
        return {}

    volume_shape = tuple(int(value) for value in volume_cuda.shape)
    if len(volume_shape) != 3 or volume_shape[0] < 1:
        return {}

    preview_axis = "z"
    slice_idx = int(volume_shape[0] // 2)
    patch_origin = (0, 0)
    patch_size = (int(volume_shape[1]), int(volume_shape[2]))
    training_state = prepare_ct_training_state(
        gaussians,
        spacing_zyx=spacing_zyx,
        truncation_sigma=truncation_sigma,
        bulk_truncation_sigma=(
            resolve_bulk_query_truncation_sigma(_kwargs.get("config", None))
            if bulk_truncation_sigma is None
            else float(bulk_truncation_sigma)
        ),
        grid_cell_voxels=grid_cell_voxels,
        signed_distance_field=_kwargs.get("signed_distance_field"),
        curvature_field=_kwargs.get("curvature_field"),
    )
    gt_patch = sample_gt_slice_patch(volume_cuda, preview_axis, slice_idx, patch_origin, patch_size).to(dtype=torch.float32)

    with torch.no_grad():
        signed_distance_field = _kwargs.get("signed_distance_field")
        has_ct_value = getattr(training_state, "ct_value", None) is not None and training_state.ct_value.numel() > 0
        dual_preview_patches = None
        if has_ct_value and signed_distance_field is not None:
            use_unified = bool(_kwargs.get("ct_use_unified_compositor", True))
            if use_unified:
                tile_size = max(1, int(slice_tile_size))
                rendered_patch = torch.empty(patch_size, dtype=torch.float32, device=gt_patch.device)
                bulk_intensity_mode = is_bulk_intensity_field_mode(
                    getattr(_kwargs.get("config", None), "ct_bulk_field_mode", "")
                )
                dual_preview_patches = {
                    "surface_only": torch.empty(patch_size, dtype=torch.float32, device=gt_patch.device),
                    "bulk_only": torch.empty(patch_size, dtype=torch.float32, device=gt_patch.device),
                }
                for origin_h in range(0, int(patch_size[0]), tile_size):
                    for origin_w in range(0, int(patch_size[1]), tile_size):
                        tile_h = min(tile_size, int(patch_size[0]) - int(origin_h))
                        tile_w = min(tile_size, int(patch_size[1]) - int(origin_w))
                        tile_origin = (int(origin_h), int(origin_w))
                        tile_size_hw = (int(tile_h), int(tile_w))
                        rr, cc = torch.meshgrid(
                            torch.arange(tile_h, dtype=torch.float32, device=gt_patch.device),
                            torch.arange(tile_w, dtype=torch.float32, device=gt_patch.device),
                            indexing="ij",
                        )
                        points_xyz = _build_query_points_from_base(
                            rr,
                            cc,
                            0,
                            slice_idx,
                            tile_origin,
                            spacing_zyx,
                        )
                        signed_distance = _sample_signed_distance(signed_distance_field, points_xyz)
                        fields = query_ct_fields_unified(
                            points_xyz,
                            training_state,
                            signed_distance=signed_distance,
                            config=_kwargs.get("config", None),
                            intensity_air=float(intensity_air),
                            include_surface=True,
                            train_ct_value=True,
                            detach_value_geometry=True,
                            apply_bulk_gate=False,
                        )
                        if bulk_intensity_mode:
                            corrected_preview = _correct_bulk_intensity_preview(
                                fields,
                                signed_distance,
                                float(intensity_air),
                                _kwargs.get("config", None),
                            )
                            rendered_tile = corrected_preview["unified"].reshape(tile_size_hw).to(dtype=torch.float32).clamp(0.0, 1.0)
                            rendered_patch[origin_h : origin_h + tile_h, origin_w : origin_w + tile_w] = rendered_tile
                            for panel_name in ("surface_only", "bulk_only"):
                                dual_preview_patches[panel_name][origin_h : origin_h + tile_h, origin_w : origin_w + tile_w] = (
                                    corrected_preview[panel_name].reshape(tile_size_hw).to(dtype=torch.float32).clamp(0.0, 1.0)
                                )
                            preview_mode = "bulk_raw_A_b"
                        else:
                            rendered_tile = fields["I_pred"].reshape(tile_size_hw).to(dtype=torch.float32).clamp(0.0, 1.0)
                            rendered_patch[origin_h : origin_h + tile_h, origin_w : origin_w + tile_w] = rendered_tile
                            dual_preview_patches["surface_only"][origin_h : origin_h + tile_h, origin_w : origin_w + tile_w] = (
                                fields["I_s"].reshape(tile_size_hw).to(dtype=torch.float32).clamp(0.0, 1.0)
                            )
                            dual_preview_patches["bulk_only"][origin_h : origin_h + tile_h, origin_w : origin_w + tile_w] = (
                                fields["I_b"].reshape(tile_size_hw).to(dtype=torch.float32).clamp(0.0, 1.0)
                            )
                            preview_mode = "unified_raw_bulk_intensity"
            else:
                from ct_pipeline.training.objectives.prediction import compute_surface_bounded_bulk_volume_prediction

                rendered_patch = compute_surface_bounded_bulk_volume_prediction(
                    training_state,
                    points_xyz,
                    signed_distance=signed_distance,
                    intensity_air=intensity_air,
                    boundary_band_distance=float(_kwargs.get("boundary_band_distance", 1.5)),
                    surface_material_gate_sigma=_kwargs.get("surface_material_gate_sigma", None),
                    material_compose_mode=_kwargs.get("material_compose_mode", "bulk_first_material"),
                    config=_kwargs.get("config", None),
                    use_unified=False,
                ).reshape(patch_size).to(dtype=torch.float32).clamp(0.0, 1.0)
                preview_mode = "bounded_bulk_intensity"
        else:
            with torch.autocast(**build_renderer_autocast_kwargs()):
                rendered_patch = render_ct_slice_patch_native(
                    training_state.render_state,
                    preview_axis,
                    slice_idx,
                    patch_origin,
                    patch_size,
                    spacing_zyx,
                    volume_shape,
                    slice_tile_size=slice_tile_size,
                )
            rendered_occupancy = rendered_patch.to(dtype=torch.float32).clamp(0.0, 1.0)
            rendered_patch = float(intensity_air) + (float(intensity_mat) - float(intensity_air)) * rendered_occupancy
            preview_mode = "raw_occupancy"
        abs_error = torch.abs(gt_patch - rendered_patch)
        gt_patch_bchw = gt_patch.unsqueeze(0).unsqueeze(0)
        rendered_patch_bchw = rendered_patch.unsqueeze(0).unsqueeze(0)
        mse = torch.mean((gt_patch_bchw - rendered_patch_bchw) ** 2)
        mae = float(abs_error.mean().item())
        rmse = float(torch.sqrt(mse).item())
        psnr = float((-10.0 * torch.log10(mse.clamp_min(1e-10))).item())
        ssim_value = float(ssim(rendered_patch_bchw, gt_patch_bchw).item())
        lpips_value = None
        lpips_model = get_lpips_model(gt_patch.device)
        if lpips_model is not None:
            gt_patch_lpips = gt_patch_bchw.repeat(1, 3, 1, 1) * 2.0 - 1.0
            rendered_patch_lpips = rendered_patch_bchw.repeat(1, 3, 1, 1) * 2.0 - 1.0
            lpips_value = float(lpips_model(rendered_patch_lpips, gt_patch_lpips).mean().item())
        material_metrics = {}
        signed_distance_field = _kwargs.get("signed_distance_field")
        if signed_distance_field is not None:
            rr, cc = torch.meshgrid(
                torch.arange(patch_size[0], dtype=torch.float32, device=gt_patch.device),
                torch.arange(patch_size[1], dtype=torch.float32, device=gt_patch.device),
                indexing="ij",
            )
            points_xyz = _build_query_points_from_base(rr, cc, 0, slice_idx, patch_origin, spacing_zyx)
            material_sdf = _sample_signed_distance(signed_distance_field, points_xyz).reshape(patch_size)
            material_mask = material_sdf <= 0.0
            if torch.any(material_mask):
                material_pred = rendered_patch[material_mask].to(dtype=torch.float32)
                material_gt = gt_patch[material_mask].to(dtype=torch.float32)
                material_abs = torch.abs(material_pred - material_gt)
                material_pred_centered = material_pred - material_pred.mean()
                material_gt_centered = material_gt - material_gt.mean()
                denom = torch.sqrt(torch.sum(material_pred_centered ** 2) * torch.sum(material_gt_centered ** 2)).clamp_min(1e-8)
                material_corr = torch.sum(material_pred_centered * material_gt_centered) / denom
                material_rendered = torch.where(material_mask, rendered_patch, torch.full_like(rendered_patch, float(intensity_air)))
                material_gt_patch = torch.where(material_mask, gt_patch, torch.full_like(gt_patch, float(intensity_air)))
                material_metrics = {
                    "material_mae": float(material_abs.mean().item()),
                    "material_ssim": float(
                        ssim(material_rendered.unsqueeze(0).unsqueeze(0), material_gt_patch.unsqueeze(0).unsqueeze(0)).item()
                    ),
                    "material_corr": float(material_corr.item()),
                    "material_A_b_p10": float(torch.quantile(material_pred, 0.10).item()),
                    "material_A_b_p50": float(torch.quantile(material_pred, 0.50).item()),
                }
        dual_preview_metrics = {}
        if dual_preview_patches is not None:
            for dual_name, dual_patch in dual_preview_patches.items():
                dual_bchw = dual_patch.unsqueeze(0).unsqueeze(0)
                dual_mse = torch.mean((gt_patch_bchw - dual_bchw) ** 2)
                dual_preview_metrics[dual_name] = {
                    "mae": float(torch.mean(torch.abs(gt_patch - dual_patch)).item()),
                    "rmse": float(torch.sqrt(dual_mse).item()),
                    "psnr": float((-10.0 * torch.log10(dual_mse.clamp_min(1e-10))).item()),
                    "ssim": float(ssim(dual_bchw, gt_patch_bchw).item()),
                }

    if output_dir is None:
        output_dir = Path(model_path).resolve().parent if iteration is None else Path(model_path).resolve() / "previews"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"preview_{Path(model_path).name}" if iteration is None else f"preview_iter_{int(iteration):06d}"
    image_path = output_dir / f"{stem}.png"
    metrics_path = output_dir / f"{stem}.txt"
    dual_image_path = output_dir / f"{stem}_dual.png"
    dual_metrics_path = output_dir / f"{stem}_dual.txt"

    gt_np = gt_patch.detach().cpu().numpy()
    rendered_np = rendered_patch.detach().cpu().numpy()
    error_np = abs_error.detach().cpu().numpy()

    if dual_preview_patches is not None:
        dual_report_rows = [
            ("axis", preview_axis),
            ("slice_idx", slice_idx),
            ("iteration", int(iteration) if iteration is not None else "final"),
            ("preview_mode", f"{preview_mode}_dual"),
        ]
        for dual_name, dual_metrics in dual_preview_metrics.items():
            for metric_name, metric_value in dual_metrics.items():
                dual_report_rows.append((f"{dual_name}_{metric_name}", metric_value))
        write_key_value_report(dual_metrics_path, dual_report_rows)

    dual_image_value = None
    dual_metrics_value = None
    bulk_mae = float("nan")
    bulk_ssim = float("nan")
    if dual_preview_patches is not None:
        dual_np = {name: patch.detach().cpu().numpy() for name, patch in dual_preview_patches.items()}
        bulk_metrics = dual_preview_metrics.get("bulk_only", {})
        bulk_mae = bulk_metrics.get("mae", float("nan"))
        bulk_ssim = bulk_metrics.get("ssim", float("nan"))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=False)
        panels = [
            ("GT", None, gt_np),
            ("Surface geometry\n(diagnostic)", "surface_only", dual_np["surface_only"]),
            ("Raw bulk A_b\n(primary)", "bulk_only", dual_np["bulk_only"]),
        ]
        for ax, (title, metric_key, image_np) in zip(axes, panels):
            ax.imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
            if metric_key in dual_preview_metrics:
                m = dual_preview_metrics[metric_key]
                title = f"{title}\nMAE={m['mae']:.4f} SSIM={m['ssim']:.4f}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        try:
            fig.savefig(dual_image_path, dpi=100)
        except (MemoryError, RuntimeError, np.core._exceptions._ArrayMemoryError) as exc:
            warnings.warn(f"Skipping CT dual preview image save: {exc!r}", RuntimeWarning, stacklevel=2)
            dual_image_path = None
        finally:
            plt.close(fig)
        dual_image_value = str(dual_image_path) if dual_image_path is not None else "unavailable"
        dual_metrics_value = str(dual_metrics_path)

    secondary_image_path = output_dir / f"{stem}_unified.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=False)
    axes[0].imshow(gt_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"GT z={slice_idx}")
    axes[0].axis("off")
    axes[1].imshow(rendered_np, cmap="gray", vmin=0.0, vmax=1.0)
    preview_title = "Prediction"
    if preview_mode == "bulk_raw_A_b":
        preview_title = "Raw bulk A_b"
    elif preview_mode == "unified_raw_bulk_intensity":
        preview_title = "Unified raw bulk"
    elif preview_mode == "bounded_bulk_intensity":
        preview_title = "Bounded bulk"
    elif preview_mode == "raw_occupancy":
        preview_title = "Raw occupancy"
    axes[1].set_title(f"{preview_title}\nPSNR={psnr:.2f} SSIM={ssim_value:.4f}")
    axes[1].axis("off")
    image_err = axes[2].imshow(error_np, cmap="magma", vmin=0.0, vmax=max(1e-6, float(error_np.max())))
    axes[2].set_title(f"Abs Error\nMAE={mae:.4f} RMSE={rmse:.4f}")
    axes[2].axis("off")
    fig.colorbar(image_err, ax=axes[2], fraction=0.046, pad=0.04)
    try:
        fig.savefig(image_path, dpi=100)
        fig.savefig(secondary_image_path, dpi=100)
    except (MemoryError, RuntimeError, np.core._exceptions._ArrayMemoryError):
        image_path = None
        secondary_image_path = None
    finally:
        plt.close(fig)

    metric_rows = [
        ("axis", preview_axis),
        ("slice_idx", slice_idx),
        ("iteration", int(iteration) if iteration is not None else "final"),
        ("preview_mode", preview_mode),
    ]
    if dual_preview_patches is not None:
        metric_rows.extend(
            [
                ("bulk_only_mae", bulk_mae),
                ("bulk_only_ssim", bulk_ssim),
            ]
        )
    metric_rows.extend(
        [
            ("mae", mae),
            ("rmse", rmse),
            ("psnr", psnr),
            ("ssim", ssim_value),
            ("lpips", lpips_value if lpips_value is not None else "unavailable"),
        ]
    )
    for key in ("material_mae", "material_ssim", "material_corr", "material_A_b_p10", "material_A_b_p50"):
        if key in material_metrics:
            metric_rows.append((key, material_metrics[key]))
    write_key_value_report(metrics_path, metric_rows)

    result = {
        "preview_image": str(image_path) if image_path is not None else "unavailable",
        "preview_metrics": str(metrics_path),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "psnr": psnr,
            "ssim": ssim_value,
            "lpips": lpips_value if lpips_value is not None else float("nan"),
            **material_metrics,
        },
    }
    if dual_preview_patches is not None:
        result["preview_dual_image"] = dual_image_value
        result["preview_dual_metrics"] = dual_metrics_value
        result["bulk_only_metrics"] = {
            "mae": bulk_mae,
            "ssim": bulk_ssim,
        }
    return result
