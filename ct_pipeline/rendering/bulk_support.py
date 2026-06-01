import math


DEFAULT_BULK_CONTAINMENT_Q_SUPPORT = 4.0
DEFAULT_BULK_CLEARANCE_SAFETY = 0.85


def bulk_query_sigma_from_q(q_support: float) -> float:
    return math.sqrt(max(float(q_support), 1e-8))


DEFAULT_BULK_QUERY_TRUNCATION_SIGMA = bulk_query_sigma_from_q(DEFAULT_BULK_CONTAINMENT_Q_SUPPORT)


def resolve_bulk_containment_q(config=None) -> float:
    if config is None:
        return DEFAULT_BULK_CONTAINMENT_Q_SUPPORT
    return max(float(getattr(config, "ct_bulk_containment_q_support", DEFAULT_BULK_CONTAINMENT_Q_SUPPORT)), 1e-8)


def resolve_bulk_query_truncation_sigma(config=None) -> float:
    if config is None:
        return DEFAULT_BULK_QUERY_TRUNCATION_SIGMA
    value = getattr(config, "ct_bulk_query_truncation_sigma", None)
    if value is None:
        return bulk_query_sigma_from_q(resolve_bulk_containment_q(config))
    return max(float(value), 1e-8)


def ellipsoid_probe_directions():
    """Deterministic local directions used to test contained q-support."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    directions = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for si in (-1.0, 1.0):
            for sj in (-1.0, 1.0):
                direction = [0.0, 0.0, 0.0]
                direction[i] = si * inv_sqrt2
                direction[j] = sj * inv_sqrt2
                directions.append(tuple(direction))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                directions.append((sx * inv_sqrt3, sy * inv_sqrt3, sz * inv_sqrt3))
    return tuple(directions)
