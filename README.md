# CTGS

CTGS is a CT-only Gaussian representation pipeline for industrial reconstructed volumes.
The repository is intentionally centered on one workflow:

1. run Phase 1 geometric analysis on a CT volume
2. initialize a hybrid Gaussian model from the canonical Phase 1 bundle
3. train with the minimal CT objective
4. export the trained representation for display or downstream analysis

This is no longer a mixed scene-training repository. A small legacy SH payload is still retained for PLY/viewer compatibility, but it is frozen and is not an active CT training signal.

## Active Pipeline

The maintained path is:

- CT volume loading from DICOM, RAW, and TIFF
- Phase 1 preprocessing and boundary analysis
- hybrid Gaussian initialization with surface and bulk roles
- CT-only training in [`train_ct.py`](/d:/Projects/3.3DGS/SAD-GS/train_ct.py)
- export through [`ct_pipeline/exporting/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/exporting)

The maintained high-level docs are:

- [`README.md`](/d:/Projects/3.3DGS/SAD-GS/README.md)
- [`ICPE_REPORT.md`](/d:/Projects/3.3DGS/SAD-GS/ICPE_REPORT.md)

## Repository Layout

- [`ct_pipeline/backend/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/backend): native CUDA wrapper split by core state, grids, queries, and rendering
- [`ct_pipeline/config/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/config): model and optimization argument definitions
- [`ct_pipeline/data/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/data): CT volume loading and Phase 1 preprocessing
- [`ct_pipeline/geometry/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/geometry): geometry analysis and curvature helpers
- [`ct_pipeline/rendering/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/rendering): field queries, bulk support rules, and slice rendering
- [`ct_pipeline/exporting/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/exporting): PLY, mesh, and SDF export
- [`ct_pipeline/runtime/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/runtime): runtime acceleration and compression utilities
- [`ct_pipeline/training/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/training): parser defaults, bootstrap/runtime losses, sampling, densification, grid caching, and reporting
- [`ct_pipeline/viewer/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline/viewer): local viewer session loading and HTTP API
- [`scene/`](/d:/Projects/3.3DGS/SAD-GS/scene): base Gaussian storage plus CT-specific initialization
- [`tools/`](/d:/Projects/3.3DGS/SAD-GS/tools): reusable evaluation and comparison commands
- [`scripts/`](/d:/Projects/3.3DGS/SAD-GS/scripts): reproducible diagnostics and ablation launchers
- [`viewer/`](/d:/Projects/3.3DGS/SAD-GS/viewer): browser frontend source; generated `dist/` assets remain untracked
- [`tests/`](/d:/Projects/3.3DGS/SAD-GS/tests): regression and contract tests for the maintained CTGS path

## Representation

CTGS uses two primitive roles:

- `region_type == 0`: surface Gaussians
- `region_type == 1`: bulk Gaussians

`region_type` is the only role signal used by the active training path.
`primitive_type` is still stored in checkpoints and PLY payloads, but it is frozen by default and is not used to decide surface-versus-bulk behavior.

## Current Algorithm

The maintained default uses `sparse_reseed` bulk initialization. Alternative
bulk initialization modes, including FASJ, remain explicit experiment options
selected with `--ct_bulk_init_mode`; they are not the default training path.

The default execution shape is:

- `material_mask` defines the coarse material support and the phase target.
- SDF provides boundary distance, containment checks, and sampling regions.
- surface Gaussians represent the material boundary.
- bulk Gaussians represent material intensity in the interior.
- training reads bulk intensity through the unified compositor with its
  configured material gate.
- saved raw-bulk diagnostics intentionally disable that gate so leakage remains
  visible instead of being hidden by the display path.
- surface reseeding and bulk reseeding are enabled by default.
- traditional split densification is available, but its default surface and
  bulk split percentages are both zero.

FASJ and coverage-repair options are retained for controlled ablations. Their
clearance and containment rules should not be described as unconditional
properties of the default run.

The current research path is a role-separated CT Gaussian model:

- `region_type == 0` surface Gaussians model material boundaries
- `region_type == 1` bulk Gaussians model material interiors
- SH feature tensors are retained only as frozen PLY/viewer compatibility payloads

Training uses calibrated CT intensity reconstruction, unified phase occupancy supervision, a compact surface regularizer, and role-specific scale clamps:

```text
L_total =
  ct_lambda_volume                  * L_volume
+ ct_lambda_occupancy               * L_occupancy
+ ct_surface_regularizer_weight     * L_surface_regularizer
```

Where:

- `L_volume`: Huber reconstruction at sampled CT volume points after fixed air/material intensity calibration
- `L_occupancy`: raw combined Gaussian occupancy supervised against the material mask, with SDF-weighted boundary samples
- `L_surface_regularizer`: coarse-SDF normal alignment, normal-thickness, and tangential-spread control
- bulk scale is capped by a global max clamp, not by EDT containment projection or a separate bulk-only loss

The occupancy path uses the Phase 1 `material_mask` as the only phase target. Sampling is split across boundary, deep material, and air; air sampling keeps an explicit void bias so cavity air is preserved without a separate bulk-only objective.

At the boundary band, surface Gaussians own the prediction:

```text
w = smooth_boundary_weight(|sdf|, boundary_band)
pred_occ = (1 - w) * bulk_occ + w * surface_occ
```

Surface reseeding is enabled by default to add missing surface Gaussians where
boundary anchors are still bulk-owned. Bulk reseeding is also enabled by
default. Traditional densification remains configurable, but its default split
percentages are zero.

Current experimental status:

- cavity/void occupancy is much healthier after explicit `void_air` sampling
- surface placement is the acceptance-critical metric for v4: renders can look good while surface drift and scale quantiles remain poor
- traditional densification does not create splits under default percentages

## Densification

The repository keeps the current CT-aware densification path:

- tangential split for surface Gaussians
- scale-aware split for bulk Gaussians

The switch is enabled by default, but both default split percentages are zero,
so it does not create new Gaussians unless explicitly configured.
Current bunny tuning results do not support enabling it as the best default path yet.

## Canonical CLI

### Phase 1

```powershell
python run_ct_phase1.py `
  --input D:\path\to\ct_data `
  --fmt auto `
  --output D:\path\to\phase1_out
```

### Training

```powershell
python train_ct.py `
  --model_path D:\path\to\train_out `
  --ct_phase1_dir D:\path\to\phase1_out `
  --ct_volume_path D:\path\to\ct_data `
  --ct_volume_format auto `
  --ct_lambda_volume 1.0 `
  --ct_lambda_occupancy 0.5 `
  --ct_surface_regularizer_weight 0.7 `
  --output_gs D:\path\to\train_out\display.ply
```

For the June 2026 `clearance_balanced_246` bulk attenuation experiment, use the
named preset instead of repeating the full tuning matrix:

```powershell
python train_ct.py `
  --ct_preset clearance_balanced_246 `
  --ct_phase1_dir outputs\figure_153505_test\phase1_ds4x4x4_keep_components_003_surf2 `
  --ct_volume_path outputs\figure_153505_test\figure_153505_ds4x4x4_float32_norm.raw `
  --ct_raw_meta outputs\figure_153505_test\figure_153505_ds4x4x4_float32_norm.json `
  --skip_export_mesh `
  --skip_export_sdf `
  --quiet `
  --iterations 500 `
  --save_iterations 500 `
  --checkpoint_iterations 500 `
  --model_path outputs\figure_153505_test\clearance_balanced_246_fixed_phase1_500_20260601
```

The `.raw` suffix is detected automatically. Preset values are applied first,
so a later explicit flag still overrides the recipe for focused ablations.

Important notes:

- the active training path requires CUDA plus the `ct_native_backend` extension
- a small set of parser aliases is still accepted for command compatibility
- legacy Phase 1 bundles that contain `material_mask` but not `coarse_support_mask` are accepted through a compatibility alias
- SH feature fields are kept only for display/PLY compatibility and are frozen during CT training

## Environment

The baseline environment is defined in [`environment.yml`](/d:/Projects/3.3DGS/SAD-GS/environment.yml).
The repository expects:

- a CUDA-capable PyTorch installation
- a working local CUDA toolchain for the native backend
- enough GPU memory for the chosen CT volume

After environment setup, verify:

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
python -m ct_pipeline.viewer serve --ply D:\path\to\display.ply --device auto
```

Use `--device cpu` when CUDA memory should remain reserved for training.

## Mesh Extraction

The default CT mesh extractor is SuGaR-style:

1. keep surface Gaussians,
2. use each Gaussian's shortest covariance axis as the oriented surface normal,
3. sample a small local tangent diamond around each center,
4. reconstruct the surface with Poisson reconstruction,
5. optionally project mesh vertices back onto the sampled Gaussian surface points.

The older density-volume marching-cubes extractor is still available for A/B runs:

```powershell
python mesher.py `
  --input D:\path\to\train_out `
  --output D:\path\to\mesh_density.ply `
  --method density
```

## Mesh Evaluation

Evaluate an extracted mesh against the Phase 1 support boundary:

```powershell
python -m tools.mesh_evaluator `
  --mesh D:\path\to\mesh.ply `
  --phase1 D:\path\to\phase1 `
  --output D:\path\to\mesh_metrics.json
```

Or extract and evaluate directly from a CTGS PLY/training output:

```powershell
python -m tools.mesh_evaluator `
  --input D:\path\to\train_out `
  --phase1 D:\path\to\phase1 `
  --mesh-output D:\path\to\extracted_mesh.ply `
  --output D:\path\to\mesh_metrics.json
```

The evaluator reports bidirectional distance metrics, symmetric Chamfer/Hausdorff, support-SDF outside ratio, and mesh component statistics.

## Tests

Run the focused regression suites with:

```powershell
python -m unittest discover -s tests -v
```

Run native CUDA parity checks explicitly when the GPU is available:

```powershell
$env:CTGS_RUN_CUDA_TESTS = "1"
python -m unittest discover -s tests -p test_ct_native_backend.py -v
```
