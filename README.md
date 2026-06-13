# CTGS: Hybrid Gaussian Splatting for Compact Industrial CT Representation

CTGS is a CT-only Gaussian representation pipeline for reconstructed industrial CT
volumes. It turns a dense CT volume into a compact, role-separated 3D Gaussian
model that can be trained, exported, meshed, and viewed.

> Shuyang Zhu and Yukie Nagai. *CTGS: Hybrid Gaussian Splatting for Compact
> Industrial CT Representation.* School of Engineering, The University of Tokyo.

## Motivation

Industrial CT produces dense voxel volumes that are faithful to material-air
boundaries but expensive to store, stream, and analyze. A surface mesh makes the
boundary explicit but discards the interior attenuation field, while a single
undifferentiated Gaussian role tends to mix two different tasks — representing a
thin material boundary and representing a volumetric material interior — which
blurs material-air transitions and spreads attenuation into void regions.

CTGS addresses this with a hybrid representation built from two complementary
primitive roles:

- **Surface Gaussians** (`region_type == 0`): thin, anisotropic primitives
  aligned with material boundaries to represent surface geometry, normals, and
  topology.
- **Bulk Gaussians** (`region_type == 1`): compact volumetric primitives placed
  inside material regions to represent the CT intensity field within the object.

Given a reconstructed CT volume, CTGS initializes surface primitives from
material boundaries and bulk primitives as a contained lattice inside the
material domain, then optimizes the two branches with separated roles. The result
is significant data reduction versus dense volumes while preserving surface
geometry for downstream analysis and a compact bulk field for fast intensity
visualization.

## Pipeline Overview

The pipeline is organized around one workflow:

1. run Phase 1 geometric analysis on a CT volume
2. initialize a hybrid Gaussian model from the Phase 1 bundle
3. train with the CT objective
4. export the trained representation for display or downstream analysis

Capabilities:

- CT volume loading from DICOM, RAW, and TIFF
- Phase 1 preprocessing and boundary analysis (material mask + signed distance field)
- hybrid Gaussian initialization with surface and bulk roles
- CT-only training in [`train_ct.py`](train_ct.py)
- export through [`ct_pipeline/exporting/`](ct_pipeline/exporting)

A small legacy spherical-harmonics (SH) payload is retained for PLY/viewer
compatibility, but it is frozen and is not an active CT training signal.

## Repository Layout

- [`ct_pipeline/backend/`](ct_pipeline/backend): native CUDA wrapper split by core state, grids, queries, and rendering
- [`ct_pipeline/config/`](ct_pipeline/config): model and optimization argument definitions
- [`ct_pipeline/data/`](ct_pipeline/data): CT volume loading and Phase 1 preprocessing
- [`ct_pipeline/geometry/`](ct_pipeline/geometry): geometry analysis and curvature helpers
- [`ct_pipeline/rendering/`](ct_pipeline/rendering): field queries, bulk support rules, and slice rendering
- [`ct_pipeline/exporting/`](ct_pipeline/exporting): PLY, mesh, and SDF export
- [`ct_pipeline/runtime/`](ct_pipeline/runtime): runtime acceleration and compression utilities
- [`ct_pipeline/training/`](ct_pipeline/training): parser defaults, losses, sampling, densification, grid caching, and reporting
- [`ct_pipeline/viewer/`](ct_pipeline/viewer): local viewer session loading and HTTP API
- [`scene/`](scene): base Gaussian storage plus CT-specific initialization
- [`viewer/`](viewer): browser frontend source; generated `dist/` assets remain untracked

## Representation

CTGS uses two primitive roles:

- `region_type == 0`: surface Gaussians model material boundaries
- `region_type == 1`: bulk Gaussians model material interiors

`region_type` is the only role signal used by the active training path.
`primitive_type` is still stored in checkpoints and PLY payloads, but it is
frozen by default and does not decide surface-versus-bulk behavior. SH feature
tensors are retained only as frozen PLY/viewer compatibility payloads.

## Training Objective

Training uses calibrated CT intensity reconstruction, phase occupancy
supervision, and a compact surface regularizer:

```text
L_total =
  ct_lambda_volume              * L_volume
+ ct_lambda_occupancy           * L_occupancy
+ ct_surface_regularizer_weight * L_surface_regularizer
```

Where:

- `L_volume`: Huber reconstruction at sampled CT points after fixed air/material intensity calibration
- `L_occupancy`: combined Gaussian occupancy supervised against the Phase 1 `material_mask`, with SDF-weighted boundary samples
- `L_surface_regularizer`: coarse-SDF normal alignment, normal-thickness, and tangential-spread control

Occupancy uses the Phase 1 `material_mask` as the phase target. Sampling is split
across boundary, deep material, and air; air sampling keeps an explicit void bias
so cavity air is preserved. At the boundary band, surface Gaussians own the
prediction:

```text
w = smooth_boundary_weight(|sdf|, boundary_band)
pred_occ = (1 - w) * bulk_occ + w * surface_occ
```

Bulk scale is capped by a global max clamp. Surface and bulk reseeding are
enabled by default to add missing Gaussians where boundary anchors are still
bulk-owned.

## Densification

The CT-aware densification path provides:

- tangential split for surface Gaussians
- scale-aware split for bulk Gaussians

The switch is enabled by default, but both default split percentages are zero,
so it does not create new Gaussians unless explicitly configured.

Alternative bulk initialization modes remain explicit experiment options
selected with `--ct_bulk_init_mode`; the maintained default is `sparse_reseed`.

## Usage

Replace the placeholder paths below with your own.

### Phase 1

```powershell
python run_ct_phase1.py `
  --input <ct_data_dir> `
  --fmt auto `
  --output <phase1_out_dir>
```

### Training

```powershell
python train_ct.py `
  --model_path <train_out_dir> `
  --ct_phase1_dir <phase1_out_dir> `
  --ct_volume_path <ct_data_path> `
  --ct_volume_format auto `
  --ct_lambda_volume 1.0 `
  --ct_lambda_occupancy 0.5 `
  --ct_surface_regularizer_weight 0.7 `
  --output_gs <train_out_dir>\display.ply
```

The CT volume format is detected from the file suffix automatically. Named
presets can be applied with `--ct_preset <name>`; preset values are applied
first, so a later explicit flag still overrides the recipe.

Notes:

- the active training path requires CUDA plus the `ct_native_backend` extension
- a small set of parser aliases is accepted for command compatibility
- legacy Phase 1 bundles that contain `material_mask` but not `coarse_support_mask` are accepted through a compatibility alias
- SH feature fields are kept only for display/PLY compatibility and are frozen during CT training

## Environment

The baseline environment is defined in [`environment.yml`](environment.yml).
The repository expects:

- a CUDA-capable PyTorch installation
- a working local CUDA toolchain for the native backend
- enough GPU memory for the chosen CT volume

After setup, verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import ct_native_backend._C as C; print('ct_native_backend ok')"
```

## Outputs

Training can produce:

- checkpoints
- PLY exports for display
- mesh exports
- SDF exports
- preview slices and drift diagnostics

## Local Viewer

Serve a trained CTGS PLY with:

```powershell
python -m ct_pipeline.viewer serve --ply <display.ply> --device auto
```

Use `--device cpu` when CUDA memory should remain reserved for training.

## Mesh Extraction

The default CT mesh extractor is SuGaR-style:

1. keep surface Gaussians,
2. use each Gaussian's shortest covariance axis as the oriented surface normal,
3. sample a small local tangent diamond around each center,
4. reconstruct the surface with Poisson reconstruction,
5. optionally project mesh vertices back onto the sampled Gaussian surface points.

A density-volume marching-cubes extractor is also available for A/B runs:

```powershell
python mesher.py `
  --input <train_out_dir> `
  --output <mesh_density.ply> `
  --method density
```

## License

See [`LICENSE.md`](LICENSE.md).
