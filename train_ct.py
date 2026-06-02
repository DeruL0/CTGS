import os
import time
import uuid

import numpy as np
import torch
from tqdm import tqdm

from ct_pipeline.config import extract_ct_model_args, extract_ct_optimization_args
from ct_pipeline.rendering.bulk_support import resolve_bulk_query_truncation_sigma
from ct_pipeline.backend import prepare_ct_training_state, require_ct_native_backend
from ct_pipeline.training import (
    CTGridCacheManager,
    build_parser,
    compute_ct_loss_terms,
    prepare_ct_training_bootstrap,
    validate_ct_training_args,
)
from ct_pipeline.training.mutations.bulk_pruning import _apply_bulk_pruning
from ct_pipeline.training.mutations.bulk_reseeding import (
    _apply_bulk_coverage_reseeding,
    _apply_material_coverage_completion,
)
from ct_pipeline.training.mutations.densification import _apply_ct_densification
from ct_pipeline.training.mutations.helpers import (
    _apply_surface_scale_hard_projection,
    _freeze_bulk_xyz_gradients,
    _log_air_shell_diagnostics,
)
from ct_pipeline.training.mutations.schedules import (
    _ct_effective_loss_weights,
    _ct_should_densify,
    _ct_should_prune_bulk,
    _ct_should_reseed_bulk,
    _ct_should_reseed_surface,
)
from ct_pipeline.training.mutations.surface_reseeding import _apply_surface_reseeding
from ct_pipeline.training.reporting import _compute_surface_drift_diagnostics, _save_surface_drift_diagnostics
from ct_pipeline.training.preview import _export_ct_outputs, _save_ct_middle_slice_preview
from ct_pipeline.training.control import (
    _apply_bulk_atten_only_optimizer_mode,
    _attenuation_only_preview_early_stop_enabled,
    _bulk_attenuation_grad_stats,
    _restore_best_bulk_attenuation,
    _sanitize_xyz_parameter,
    _save_ct_gaussians,
    maybe_apply_stage1_freeze,
)
from ct_pipeline.training.session import (
    ct_log_gpu_memory,
    initialize_nvml,
    save_command,
    shutdown_nvml,
)
from utils.general_utils import safe_state

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def training_ct(dataset, opt, saving_iterations, checkpoint_iterations, checkpoint, args):
    validate_ct_training_args(args)
    require_ct_native_backend()

    context = prepare_ct_training_bootstrap(dataset, opt, args, checkpoint)
    gaussians = context.gaussians
    tb_writer = context.tb_writer

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(total=max(0, int(opt.iterations) - int(context.first_iter)), desc="CT training progress")
    ema_loss_for_log = 0.0
    total_computing_time = 0.0
    last_ct_densify_iter = None
    last_surface_reseed_added = 0
    last_bulk_reseed_stats = {"added": 0, "candidates": 0, "low_coverage_ratio": 0.0}
    last_bulk_prune_stats = {"pruned": 0}
    last_densify_stats = {
        "surface_split": 0,
        "bulk_split": 0,
        "children_added": 0,
        "net_added": 0,
    }
    final_iteration = int(opt.iterations)
    bulk_truncation_sigma = resolve_bulk_query_truncation_sigma(args)
    atten_early_stop_enabled = _attenuation_only_preview_early_stop_enabled(args)
    atten_early_stop_best_ssim = float("-inf")
    atten_early_stop_best_iter = 0
    atten_early_stop_best_atten = None
    atten_early_stop_best_opacity = None
    atten_early_stop_bad_evals = 0
    grid_cache = CTGridCacheManager.from_args(args, context.spacing_zyx)

    _log_air_shell_diagnostics(context.field_pools)

    if bool(getattr(args, "ct_material_coverage_completion", False)) and bool(getattr(args, "ct_completion_init", True)):
        max_passes = int(getattr(args, "ct_completion_max_init_passes", 0))
        for pass_index in range(max_passes):
            init_state = prepare_ct_training_state(
                gaussians,
                spacing_zyx=context.spacing_zyx,
                truncation_sigma=args.ct_gaussian_truncation_sigma,
                bulk_truncation_sigma=bulk_truncation_sigma,
                grid_cell_voxels=args.ct_grid_cell_voxels,
                build_full_grid=False,
                build_region_grids=True,
                signed_distance_field=context.signed_distance_field,
                curvature_field=context.intensity_field_cache.get("curvature_proxy"),
            )
            completion_stats = _apply_material_coverage_completion(
                gaussians,
                args,
                init_state,
                context.spacing_zyx,
                context.analysis_gpu,
                volume_field=context.volume_cuda.reshape(1, 1, *tuple(int(value) for value in context.volume_shape)),
                signed_distance_field=context.signed_distance_field,
                initial_gaussian_count=context.initial_gaussian_count,
                iteration=0,
                pass_index=pass_index,
            )
            last_bulk_reseed_stats = dict(completion_stats)
            print(
                "[INIT] Material coverage completion pass={0}: components={1} max_component={2} "
                "added={3} low_cov={4:.6f} count={5}->{6}".format(
                    pass_index + 1,
                    completion_stats.get("num_uncovered_components", 0),
                    completion_stats.get("max_uncovered_component_voxels", 0),
                    completion_stats.get("added", 0),
                    completion_stats.get("low_coverage_ratio", 0.0),
                    completion_stats.get("count_before", 0),
                    completion_stats.get("count_after", 0),
                ),
                flush=True,
            )
            if int(completion_stats.get("num_uncovered_components", 0)) <= 0 or int(completion_stats.get("added", 0)) <= 0:
                break
            grid_cache.invalidate_all()

    for iteration in range(int(context.first_iter) + 1, int(opt.iterations) + 1):
        tic = time.time()
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        maybe_apply_stage1_freeze(gaussians, args, iteration)
        _apply_bulk_atten_only_optimizer_mode(gaussians, args, iteration=iteration, max_iter=int(opt.iterations))
        bad_xyz_count = _sanitize_xyz_parameter(gaussians)
        if bad_xyz_count > 0:
            print(f"[WARN] Replaced {bad_xyz_count} non-finite Gaussian centers before the training step.")

        training_state = prepare_ct_training_state(
            gaussians,
            spacing_zyx=context.spacing_zyx,
            truncation_sigma=args.ct_gaussian_truncation_sigma,
            bulk_truncation_sigma=bulk_truncation_sigma,
            grid_cell_voxels=args.ct_grid_cell_voxels,
            build_full_grid=False,
            build_region_grids=not grid_cache.enabled,
            signed_distance_field=context.signed_distance_field,
            curvature_field=context.intensity_field_cache.get("curvature_proxy"),
        )
        if grid_cache.enabled:
            grid_cache.refresh(training_state, iteration)
            grid_cache.attach(training_state)
        loss_terms = compute_ct_loss_terms(context, args, training_state, iteration=iteration)
        effective_weights = _ct_effective_loss_weights(args, iteration, last_ct_densify_iter)
        loss = (
            effective_weights.volume * loss_terms.volume
            + effective_weights.occupancy * loss_terms.occupancy
            + effective_weights.surface * loss_terms.surface
        )
        loss.backward()
        atten_grad_stats = _bulk_attenuation_grad_stats(gaussians)

        densify_xyz_grad_norm = None
        if getattr(args, "ct_enable_densification", False) and gaussians._xyz.grad is not None:
            densify_xyz_grad_norm = gaussians._xyz.grad.detach().norm(dim=1)
        if args.ct_freeze_bulk_xyz:
            _freeze_bulk_xyz_gradients(gaussians)

        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
            progress_bar.update(1)

            elapsed = iter_start.elapsed_time(iter_end)
            total_computing_time += time.time() - tic
            if tb_writer:
                tb_writer.add_scalar("ct_loss/volume", loss_terms.volume.item(), iteration)
                tb_writer.add_scalar("ct_loss/render", loss_terms.render.item(), iteration)
                tb_writer.add_scalar("ct_loss/occupancy", loss_terms.occupancy.item(), iteration)
                tb_writer.add_scalar("ct_loss/surface", loss_terms.surface.item(), iteration)
                tb_writer.add_scalar("ct_loss/total", loss.item(), iteration)
                if getattr(args, "ct_enable_densification", False):
                    tb_writer.add_scalar("ct_loss_effective/volume_lambda", effective_weights.volume, iteration)
                    tb_writer.add_scalar("ct_loss_effective/occupancy_lambda", effective_weights.occupancy, iteration)
                    tb_writer.add_scalar("ct_loss_effective/surface_lambda", effective_weights.surface, iteration)
                for metric_name, metric_value in atten_grad_stats.items():
                    tb_writer.add_scalar(f"ct_grad/{metric_name}", metric_value, iteration)
                tb_writer.add_scalar("iter_time", elapsed, iteration)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                _save_ct_gaussians(gaussians, dataset.model_path, iteration)
                try:
                    _save_surface_drift_diagnostics(
                        training_state,
                        context.analysis_gpu,
                        context.spacing_zyx,
                        dataset.model_path,
                        iteration,
                        tb_writer=tb_writer,
                        curvature_field=context.intensity_field_cache.get("curvature_proxy"),
                        surface_reseed_added=last_surface_reseed_added,
                        bulk_reseed_stats=last_bulk_reseed_stats,
                        bulk_prune_stats=last_bulk_prune_stats,
                        densify_stats=last_densify_stats,
                        boundary_band_distance=args.ct_boundary_band,
                        signed_distance_field=context.signed_distance_field,
                        volume_cuda=context.volume_cuda,
                        intensity_air=context.intensity_air,
                        intensity_mat=context.intensity_mat,
                        false_hole_sample_count=getattr(args, "ct_false_hole_sample_count", 4096),
                        false_hole_boundary_band=getattr(args, "ct_false_hole_boundary_band", 2.0),
                        false_hole_material_threshold=getattr(args, "ct_false_hole_material_threshold", 0.65),
                        false_hole_dark_margin=getattr(args, "ct_false_hole_dark_margin", 0.15),
                        surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                        material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
                        config=args,
                        use_unified_compositor=getattr(args, "ct_use_unified_compositor", True),
                    )
                except (MemoryError, RuntimeError, np.core._exceptions._ArrayMemoryError) as exc:
                    print(f"[WARN] Skipping drift diagnostics at iter {iteration}: {exc!r}")
                preview_result = None
                if getattr(args, "ct_auto_preview", True):
                    preview_result = _save_ct_middle_slice_preview(
                        gaussians,
                        context.volume_cuda,
                        context.spacing_zyx,
                        dataset.model_path,
                        slice_tile_size=args.ct_slice_tile_size,
                        truncation_sigma=args.ct_gaussian_truncation_sigma,
                        bulk_truncation_sigma=bulk_truncation_sigma,
                        grid_cell_voxels=args.ct_grid_cell_voxels,
                        iteration=iteration,
                        intensity_air=context.intensity_air,
                        intensity_mat=context.intensity_mat,
                        signed_distance_field=context.signed_distance_field,
                        support_mask=context.analysis_gpu.get("material_mask"),
                        boundary_band_distance=args.ct_boundary_band,
                        surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                        material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
                        config=args,
                        ct_use_unified_compositor=getattr(args, "ct_use_unified_compositor", True),
                    )
            else:
                preview_result = None

            # Emit one extra preview at the first training iteration so
            # initialization leakage is visible before later updates can mask it.
            if (
                bool(getattr(args, "ct_preview_first_iter", True))
                and preview_result is None
                and getattr(args, "ct_auto_preview", True)
                and int(iteration) == int(context.first_iter) + 1
            ):
                preview_result = _save_ct_middle_slice_preview(
                    gaussians,
                    context.volume_cuda,
                    context.spacing_zyx,
                    dataset.model_path,
                    slice_tile_size=args.ct_slice_tile_size,
                    truncation_sigma=args.ct_gaussian_truncation_sigma,
                    bulk_truncation_sigma=bulk_truncation_sigma,
                    grid_cell_voxels=args.ct_grid_cell_voxels,
                    iteration=iteration,
                    intensity_air=context.intensity_air,
                    intensity_mat=context.intensity_mat,
                    signed_distance_field=context.signed_distance_field,
                    support_mask=context.analysis_gpu.get("material_mask"),
                    boundary_band_distance=args.ct_boundary_band,
                    surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                    material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
                    config=args,
                    ct_use_unified_compositor=getattr(args, "ct_use_unified_compositor", True),
                )

            preview_interval = max(int(getattr(args, "ct_preview_interval", 0)), 0)
            if (
                preview_interval > 0
                and preview_result is None
                and getattr(args, "ct_auto_preview", True)
                and int(iteration) % preview_interval == 0
            ):
                preview_result = _save_ct_middle_slice_preview(
                    gaussians,
                    context.volume_cuda,
                    context.spacing_zyx,
                    dataset.model_path,
                    slice_tile_size=args.ct_slice_tile_size,
                    truncation_sigma=args.ct_gaussian_truncation_sigma,
                    bulk_truncation_sigma=bulk_truncation_sigma,
                    grid_cell_voxels=args.ct_grid_cell_voxels,
                    iteration=iteration,
                    intensity_air=context.intensity_air,
                    intensity_mat=context.intensity_mat,
                    signed_distance_field=context.signed_distance_field,
                    support_mask=context.analysis_gpu.get("material_mask"),
                    boundary_band_distance=args.ct_boundary_band,
                    surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                    material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
                    config=args,
                    ct_use_unified_compositor=getattr(args, "ct_use_unified_compositor", True),
                )

            eval_interval = max(int(getattr(args, "ct_atten_only_early_stop_eval_interval", 100)), 1)
            monitor_preview = atten_early_stop_enabled and iteration % eval_interval == 0
            if monitor_preview and preview_result is None:
                preview_result = _save_ct_middle_slice_preview(
                    gaussians,
                    context.volume_cuda,
                    context.spacing_zyx,
                    dataset.model_path,
                    slice_tile_size=args.ct_slice_tile_size,
                    truncation_sigma=args.ct_gaussian_truncation_sigma,
                    bulk_truncation_sigma=bulk_truncation_sigma,
                    grid_cell_voxels=args.ct_grid_cell_voxels,
                    iteration=iteration,
                    intensity_air=context.intensity_air,
                    intensity_mat=context.intensity_mat,
                    signed_distance_field=context.signed_distance_field,
                    support_mask=context.analysis_gpu.get("material_mask"),
                    boundary_band_distance=args.ct_boundary_band,
                    surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                    material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
                    config=args,
                    ct_use_unified_compositor=getattr(args, "ct_use_unified_compositor", True),
                )

            if monitor_preview and preview_result is not None:
                preview_metrics = preview_result.get("bulk_only_metrics", preview_result.get("metrics", {}))
                current_ssim = float(preview_metrics.get("ssim", float("nan")))
                if np.isfinite(current_ssim):
                    min_delta = float(getattr(args, "ct_atten_only_early_stop_min_delta", 1e-4))
                    if atten_early_stop_best_atten is None or current_ssim > (atten_early_stop_best_ssim + min_delta):
                        atten_early_stop_best_ssim = current_ssim
                        atten_early_stop_best_iter = int(iteration)
                        atten_early_stop_best_atten = gaussians._atten_logit.detach().clone()
                        atten_early_stop_best_opacity = gaussians._opacity.detach().clone()
                        atten_early_stop_bad_evals = 0
                    elif int(iteration) >= int(getattr(args, "ct_atten_only_early_stop_warmup_iters", 100)):
                        atten_early_stop_bad_evals += 1

                    if tb_writer:
                        tb_writer.add_scalar("ct_early_stop/current_ssim", current_ssim, iteration)
                        tb_writer.add_scalar("ct_early_stop/best_ssim", atten_early_stop_best_ssim, iteration)

                    patience = max(int(getattr(args, "ct_atten_only_early_stop_patience", 2)), 1)
                    if atten_early_stop_bad_evals >= patience and atten_early_stop_best_atten is not None:
                        restored = _restore_best_bulk_attenuation(
                            gaussians,
                            atten_early_stop_best_atten,
                            opacity_logit=atten_early_stop_best_opacity,
                        )
                        final_iteration = int(iteration)
                        print(
                            "[ITER {0}] Early stop bulk attenuation: restore iter={1} best_ssim={2:.6f} current_ssim={3:.6f}".format(
                                iteration,
                                atten_early_stop_best_iter,
                                atten_early_stop_best_ssim,
                                current_ssim,
                            )
                        )
                        if tb_writer:
                            tb_writer.add_scalar("ct_early_stop/trigger_iter", float(iteration), iteration)
                            tb_writer.add_scalar("ct_early_stop/restored", float(restored), iteration)
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        break

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.post_optimizer_step(iteration)
                _apply_surface_scale_hard_projection(
                    gaussians,
                    spacing_zyx=context.spacing_zyx,
                    max_scale=args.ct_surface_max_scale,
                )

                if _ct_should_reseed_surface(args, iteration):
                    reseed_stats = _apply_surface_reseeding(
                        gaussians,
                        args,
                        training_state,
                        context.spacing_zyx,
                        context.analysis_gpu,
                        initial_gaussian_count=context.initial_gaussian_count,
                        signed_distance_field=context.signed_distance_field,
                        curvature_field=context.intensity_field_cache.get("curvature_proxy"),
                        volume_field=context.volume_cuda.reshape(1, 1, *tuple(int(value) for value in context.volume_shape)),
                    )
                    if reseed_stats["added"] > 0:
                        last_surface_reseed_added = int(reseed_stats["added"])
                        grid_cache.invalidate_all()
                        print(
                            "[ITER {0}] CT surface reseeding: added={1} candidates={2} "
                            "gap={3:.4f} bulk_owned={4:.4f} count={5}->{6}".format(
                                iteration,
                                reseed_stats["added"],
                                reseed_stats["candidates"],
                                reseed_stats["coverage_gap_ratio"],
                                reseed_stats["bulk_owned_ratio"],
                                reseed_stats["count_before"],
                                reseed_stats["count_after"],
                            )
                        )
                    else:
                        last_surface_reseed_added = 0
                    if tb_writer:
                        tb_writer.add_scalar("ct_surface_reseed/added", reseed_stats["added"], iteration)
                        tb_writer.add_scalar("ct_surface_reseed/coverage_gap_ratio", reseed_stats["coverage_gap_ratio"], iteration)
                        tb_writer.add_scalar("ct_surface_reseed/bulk_owned_ratio", reseed_stats["bulk_owned_ratio"], iteration)
                        tb_writer.add_scalar("ct_surface_reseed/count", reseed_stats["count_after"], iteration)

                if _ct_should_reseed_bulk(args, iteration):
                    bulk_reseed_stats = _apply_bulk_coverage_reseeding(
                        gaussians,
                        args,
                        training_state,
                        context.spacing_zyx,
                        context.analysis_gpu,
                        support_distance_field=context.support_distance_field,
                        initial_gaussian_count=context.initial_gaussian_count,
                        volume_field=context.volume_cuda.reshape(1, 1, *tuple(int(value) for value in context.volume_shape)),
                        signed_distance_field=context.signed_distance_field,
                        field_pools=context.field_pools,
                        iteration=iteration,
                    )
                    last_bulk_reseed_stats = dict(bulk_reseed_stats)
                    if bulk_reseed_stats["added"] > 0 or bulk_reseed_stats.get("repair_stretched_count", 0) > 0:
                        grid_cache.invalidate_all()
                        print(
                            "[ITER {0}] CT bulk reseeding: added={1} candidates={2} "
                            "low_cov={3:.4f} grown={4} components={5} max_component={6} count={7}->{8} "
                            "stretch={9} excl={10} low_gain={11} containment={12} overfill={13}".format(
                                iteration,
                                bulk_reseed_stats["added"],
                                bulk_reseed_stats["candidates"],
                                bulk_reseed_stats["low_coverage_ratio"],
                                bulk_reseed_stats.get("bulk_grown_count", 0),
                                bulk_reseed_stats.get("num_uncovered_components", 0),
                                bulk_reseed_stats.get("max_uncovered_component_voxels", 0),
                                bulk_reseed_stats["count_before"],
                                bulk_reseed_stats["count_after"],
                                bulk_reseed_stats.get("repair_stretched_count", 0),
                                bulk_reseed_stats.get("repair_skipped_exclusion", 0),
                                bulk_reseed_stats.get("repair_skipped_low_gain", 0),
                                bulk_reseed_stats.get("repair_skipped_containment", 0),
                                bulk_reseed_stats.get("repair_skipped_overfill", 0),
                            )
                        )

                if _ct_should_densify(args, iteration):
                    densify_stats = _apply_ct_densification(
                        gaussians,
                        args,
                        densify_xyz_grad_norm,
                        context.support_distance_field,
                        context.spacing_zyx,
                        context.analysis_gpu,
                        initial_gaussian_count=context.initial_gaussian_count,
                        signed_distance_field=context.signed_distance_field,
                        curvature_field=context.intensity_field_cache.get("curvature_proxy"),
                    )
                    last_densify_stats = dict(densify_stats)
                    if densify_stats["children_added"] > 0:
                        last_ct_densify_iter = iteration
                        grid_cache.invalidate_all()
                        _apply_surface_scale_hard_projection(
                            gaussians,
                            spacing_zyx=context.spacing_zyx,
                            max_scale=args.ct_surface_max_scale,
                        )
                        print(
                            "[ITER {0}] CT densification: surface_split={1} bulk_split={2} "
                            "children={3} net_added={4} count={5}->{6}".format(
                                iteration,
                                densify_stats["surface_split"],
                                densify_stats["bulk_split"],
                                densify_stats["children_added"],
                                densify_stats["net_added"],
                                densify_stats["count_before"],
                                densify_stats["count_after"],
                            )
                        )
                        if tb_writer:
                            tb_writer.add_scalar("ct_densify/surface_split", densify_stats["surface_split"], iteration)
                            tb_writer.add_scalar("ct_densify/bulk_split", densify_stats["bulk_split"], iteration)
                            tb_writer.add_scalar("ct_densify/net_added", densify_stats["net_added"], iteration)
                            tb_writer.add_scalar("ct_densify/count", densify_stats["count_after"], iteration)

                if _ct_should_prune_bulk(args, iteration):
                    prune_state = prepare_ct_training_state(
                        gaussians,
                        spacing_zyx=context.spacing_zyx,
                        truncation_sigma=args.ct_gaussian_truncation_sigma,
                        bulk_truncation_sigma=bulk_truncation_sigma,
                        grid_cell_voxels=args.ct_grid_cell_voxels,
                        build_full_grid=False,
                        build_region_grids=True,
                        signed_distance_field=context.signed_distance_field,
                        curvature_field=context.intensity_field_cache.get("curvature_proxy"),
                    )
                    prune_stats = _apply_bulk_pruning(
                        gaussians,
                        args,
                        prune_state,
                        context.spacing_zyx,
                        context.field_pools,
                        context.signed_distance_field,
                        material_mask=context.analysis_gpu.get("material_mask"),
                        iteration=iteration,
                    )
                    last_bulk_prune_stats = dict(prune_stats)
                    if prune_stats["pruned"] > 0:
                        grid_cache.invalidate_all()
                        print(
                            "[ITER {0}] CT bulk pruning: pruned={1} low_opacity={2} air_center={3} "
                            "raw_air_owner={4} isolated={5} count={6}->{7}".format(
                                iteration,
                                prune_stats["pruned"],
                                prune_stats["low_opacity"],
                                prune_stats["air_center"],
                                prune_stats["raw_air_owner"],
                                prune_stats["isolated"],
                                prune_stats["count_before"],
                                prune_stats["count_after"],
                            )
                        )

                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), dataset.model_path + "/chkpnt" + str(iteration) + ".pth")

        if args.wandb and wandb is not None:
            wandb_logs = {
                "loss": loss.item(),
                "volume_loss": loss_terms.volume.item(),
                "render_loss": loss_terms.render.item(),
                "occupancy_loss": loss_terms.occupancy.item(),
                "surface_loss": loss_terms.surface.item(),
                "t": total_computing_time,
                "num_gaussian": len(gaussians.get_xyz),
            }
            gpu_memory = ct_log_gpu_memory()
            if gpu_memory is not None:
                wandb_logs["gpu"] = gpu_memory
            wandb.log(wandb_logs, commit=True)

    progress_bar.close()

    exports = _export_ct_outputs(gaussians, args)
    if getattr(args, "ct_auto_preview", True):
        exports.update(
            _save_ct_middle_slice_preview(
                gaussians,
                context.volume_cuda,
                context.spacing_zyx,
                dataset.model_path,
                slice_tile_size=args.ct_slice_tile_size,
                truncation_sigma=args.ct_gaussian_truncation_sigma,
                bulk_truncation_sigma=bulk_truncation_sigma,
                grid_cell_voxels=args.ct_grid_cell_voxels,
                intensity_air=context.intensity_air,
                intensity_mat=context.intensity_mat,
                signed_distance_field=context.signed_distance_field,
                support_mask=context.analysis_gpu.get("material_mask"),
                boundary_band_distance=args.ct_boundary_band,
                surface_material_gate_sigma=getattr(args, "ct_surface_material_gate_sigma", None),
                material_compose_mode=getattr(args, "ct_material_compose_mode", "bulk_first_material"),
                config=args,
                ct_use_unified_compositor=getattr(args, "ct_use_unified_compositor", True),
            )
        )
    if tb_writer:
        tb_writer.close()
    return {"branch": "ct", "iterations": final_iteration, **exports}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[:10])

    print("Optimizing " + args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    save_command(args.model_path)
    safe_state(args.quiet)

    if args.wandb and wandb is not None:
        wandb.login()
        wandb.init(
            project="gaussian_splatting",
            name=os.path.basename(args.model_path.rstrip("/\\")),
            config={},
            save_code=True,
            notes="",
            mode="online",
        )

    initialize_nvml()
    torch.autograd.set_detect_anomaly(False)
    result = training_ct(
        extract_ct_model_args(args),
        extract_ct_optimization_args(args),
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args,
    )
    shutdown_nvml()
    print("\nTraining complete.")
    return result


if __name__ == "__main__":
    main()
