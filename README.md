# CTGS

`CTGS` is a CT-only Gaussian representation pipeline for industrial volume data.

Current workflow:

1. Phase 1 CT ingestion, segmentation, and geometric analysis
2. CT-only hybrid Gaussian training
3. Native CUDA acceleration for CT slice rendering, density query, and KNN refresh
4. Dual-output export for display GS, mesh, and SDF

## What The Current Pipeline Learns

The training pipeline is now `void-aware`, not solid-fill by default.

- `material_mask` is the occupancy positive set
- `void_mask` and exterior voxels are occupancy negatives
- `foreground_mask` is retained only as a compatibility / ROI mask
- surface primitives and bulk primitives are both used
- internal cavities and through-holes are intended to remain visible

Primitive semantics:

- `primitive_type = 0`: anisotropic 3D Gaussian
- `primitive_type = 1`: planar Gaussian
- `region_type = 0`: surface primitive
- `region_type = 1`: bulk primitive

## Setup

```powershell
conda env create --file environment.yml
conda activate gaussian_splatting
```

The environment installs the CT native backend from `submodules/ct-native-backend`.

## Repository Scope

Removed from this repository:

- standard SAD-GS / traditional GS train-render-eval entrypoints
- camera / dataset scene pipeline
- COLMAP / Replica / Blender style scene loading
- standard image-space rasterizer training path

Retained compatibility:

- `CTGaussianModel` can still best-effort load older GS PLY / checkpoint payloads and resave them as CT-side hybrid PLYs

## Phase 1: CT Ingestion And Analysis

Run standalone CT ingestion and analysis:

```powershell
python run_ct_phase1.py ^
  --input D:\path\to\ct_data ^
  --fmt auto ^
  --output D:\path\to\phase1_out ^
  --max-material-classes 3
```

Supported inputs:

- DICOM series directory or representative slice
- RAW binary volume with JSON sidecar
- TIFF stack file or TIFF slice directory

Important Phase 1 behavior:

- global hole filling is disabled
- material classes are split automatically with `threshold_multiotsu`
- `void_mask` preserves internal cavities and through-holes inside the object ROI
- material surfaces are extracted per material label and merged

Main outputs:

- `analysis.npz`
- `metadata.json`

Important arrays in `analysis.npz`:

- `material_mask`
- `void_mask`
- `foreground_mask`
- `material_label_volume`
- `surface_points`
- `surface_material_id`
- `surface_normals`
- `interior_points`
- `interior_density_seed`
- `interior_material_id`

`foreground_mask` should be treated as a coarse ROI mask only. It is not the occupancy positive set anymore.

## CT Training

Train the hybrid CT Gaussian model:

```powershell
python train_ct.py ^
  --model_path D:\path\to\train_out ^
  --ct_phase1_dir D:\path\to\phase1_out ^
  --ct_volume_path D:\path\to\ct_data ^
  --ct_volume_format auto ^
  --ct_backend auto ^
  --ct_material_query_count 4096 ^
  --ct_void_query_count 4096 ^
  --ct_exterior_query_count 4096 ^
  --ct_void_negative_weight 2.0 ^
  --output_gs D:\path\to\train_out\display.ply ^
  --output_mesh D:\path\to\train_out\mesh.ply ^
  --output_sdf D:\path\to\train_out\sdf.npy
```

Backends:

- `--ct_backend auto`
- `--ct_backend python`
- `--ct_backend cuda`

`auto` prefers the native CUDA backend and falls back to the Python backend when the extension is unavailable.

Training requirements and behavior:

- CUDA is required for the current training implementation
- SH/color features are frozen; CT training optimizes density geometry only
- occupancy positives come only from `material_mask`
- occupancy negatives come from `void_mask` and exterior voxels
- `--ct_interior_query_count` is retained as a deprecated alias for `--ct_material_query_count`

Useful training flags:

- `--ct_patch_size`
- `--ct_slice_batch_size`
- `--ct_neighbor_k`
- `--ct_neighbor_refresh_interval`
- `--ct_bulk_points_ratio`
- `--ct_bulk_boundary_margin_voxels`
- `--ct_max_material_classes`
- `--ct_void_negative_weight`
- `--primitive_harden_iter`
- `--planar_thickness_max`

## Native CUDA Backend

The native backend is CT-specific and separate from the removed old GS rasterizer stack.

Currently accelerated:

- slice patch render forward / backward
- occupancy density query
- KNN neighbor refresh
- cached point-to-plane target preparation
- point-to-plane loss application

If the native extension is not available:

- `--ct_backend auto` falls back to Python
- `--ct_backend cuda` raises an error

## Export

`train_ct.py` can export directly at the end of training:

- display GS `.ply`
- metrology mesh `.ply`
- SDF `.npy` with `.json` sidecar

Display export is for interactive viewing and lightweight distribution.
Mesh and SDF are the analysis-oriented outputs.

## Mesh Extraction

Extract a CT mesh from a saved CTGS point cloud:

```powershell
python mesher.py ^
  --input D:\path\to\train_out ^
  --output D:\path\to\mesh.ply ^
  --iteration -1 ^
  --resolution 0.05 ^
  --threshold 0.5
```

`--input` may point either to:

- a training output directory containing `point_cloud/iteration_*/point_cloud.ply`
- a direct hybrid GS `.ply`

`mesher.py` is CT-aware:

- higher-resolution marching cubes over the CTGS density field
- material-aware vertex labeling
- local boundary refinement near material transitions

## Large CT Volumes

Full Phase 1 output can be very large for dense CT scans.
For training, it is often practical to keep:

- full voxel masks and metadata
- subsampled `surface_points`
- subsampled `interior_points`

This is the workflow used for the `assets/bunny` smoke runs under `outputs/bunny_smoke`.

## Tests

Run the CTGS regression suite:

```powershell
python -m unittest discover -s tests
```

The test suite covers:

- Phase 1 loading, segmentation, and geometry analysis
- hybrid Gaussian persistence and initialization
- CT losses
- native backend parity and safety
- CT training smoke runs
- exporter and mesher behavior
