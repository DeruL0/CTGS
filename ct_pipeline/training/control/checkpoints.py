from __future__ import annotations

import os

def _save_ct_gaussians(gaussians, model_path, iteration):
    point_cloud_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    os.makedirs(point_cloud_path, exist_ok=True)
    ply_path = os.path.join(point_cloud_path, "point_cloud.ply")
    try:
        gaussians.save_ply(ply_path)
    except Exception as exc:
        if exc.__class__.__name__ not in {"MemoryError", "_ArrayMemoryError"}:
            raise
        warning_path = os.path.join(point_cloud_path, "point_cloud_save_warning.txt")
        with open(warning_path, "w", encoding="utf-8") as handle:
            handle.write(f"Skipped PLY save at iteration {iteration}: {exc.__class__.__name__}: {exc}\n")
        print(f"[WARN] Skipped PLY save at iter {iteration}: {exc.__class__.__name__}")
