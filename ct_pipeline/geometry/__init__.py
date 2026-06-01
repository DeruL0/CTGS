"""Geometry analysis helpers shared by preprocessing and training."""

from .analysis import GeometryAnalyzer
from .curvature import compute_curvature_proxy_np

__all__ = [
    "GeometryAnalyzer",
    "compute_curvature_proxy_np",
]
