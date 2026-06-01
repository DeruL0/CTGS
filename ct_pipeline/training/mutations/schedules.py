from __future__ import annotations

from types import SimpleNamespace


def _ct_densification_active(args, iteration: int) -> bool:
    return (
        bool(getattr(args, "ct_enable_densification", False))
        and int(args.ct_densify_from_iter) <= int(iteration) <= int(args.ct_densify_until_iter)
    )


def _ct_should_densify(args, iteration: int) -> bool:
    if not _ct_densification_active(args, iteration):
        return False
    return (int(iteration) - int(args.ct_densify_from_iter)) % int(args.ct_densify_interval) == 0


def _ct_surface_reseeding_active(args, iteration: int) -> bool:
    return (
        bool(getattr(args, "ct_enable_surface_reseeding", False))
        and int(args.ct_surface_reseed_from_iter) <= int(iteration) <= int(args.ct_surface_reseed_until_iter)
        and int(getattr(args, "ct_surface_reseed_max_new_per_iter", 0)) > 0
    )


def _ct_should_reseed_surface(args, iteration: int) -> bool:
    if not _ct_surface_reseeding_active(args, iteration):
        return False
    return (int(iteration) - int(args.ct_surface_reseed_from_iter)) % int(args.ct_surface_reseed_interval) == 0


def _ct_effective_loss_weights(args, iteration: int, last_densify_iter=None):
    del iteration, last_densify_iter
    volume_weight = float(getattr(args, "ct_lambda_volume", 1.0))
    return SimpleNamespace(
        volume=volume_weight,
        render=volume_weight,
        occupancy=float(args.ct_lambda_occupancy),
        surface=float(args.ct_surface_regularizer_weight),
    )


def _ct_should_reseed_bulk(args, iteration: int) -> bool:
    if not bool(getattr(args, "ct_enable_bulk_reseeding", False)):
        return False
    from_iter = int(getattr(args, "ct_bulk_reseed_from_iter", 500))
    until_iter = int(getattr(args, "ct_bulk_reseed_until_iter", 2000))
    interval = int(getattr(args, "ct_bulk_reseed_interval", 500))
    if not (from_iter <= int(iteration) <= until_iter):
        return False
    return (int(iteration) - from_iter) % interval == 0


def _ct_should_prune_bulk(args, iteration: int) -> bool:
    interval = int(getattr(args, "ct_bulk_prune_interval", 0))
    if interval <= 0:
        return False
    if int(iteration) < int(getattr(args, "ct_bulk_prune_warmup", 1000)):
        return False
    return int(iteration) % interval == 0
