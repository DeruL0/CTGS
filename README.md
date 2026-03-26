# CTGS

`CTGS` is a CT-only Gaussian representation repository for industrial volume data.
It focuses on the full pipeline around CT volumes: ingestion, geometric analysis, hybrid Gaussian modeling, CT-specific training, runtime acceleration, and export to display and analysis formats.

This repository is no longer a general SAD-GS / NeRF-style scene training project.
It is organized around one target problem: turning reconstructed CT volumes into an interactive Gaussian representation that remains useful for inspection, slicing, meshing, and downstream geometric analysis.

In practical terms, the repository is meant to make CT data easier to:

- compress into a lighter interactive representation
- display quickly for browsing and inspection
- export into analysis-oriented geometry formats

## Overview

The repository provides:

- CT volume loading from DICOM, RAW, and TIFF
- Phase 1 preprocessing and geometry analysis
- hybrid Gaussian primitives for surface-aware and bulk-aware CT modeling
- CT-only training with slice supervision and geometric regularization
- native CUDA acceleration for the main CT training bottlenecks
- export paths for display GS, mesh, and SDF

At a high level, the pipeline is:

1. load a reconstructed CT volume
2. segment material / void structure and analyze local geometry
3. initialize a hybrid CT Gaussian model
4. train the model with CT-specific losses
5. export the trained representation for viewing or analysis

## Design Goals

CTGS is built around a few practical assumptions:

- CT is not treated like a camera-scene dataset
- the training target is a volumetric CT field, not RGB novel-view synthesis
- surface structure matters, but internal material / void structure matters too
- fast display and geometric analysis do not have to use the exact same output format
- compact representation and fast browsing are first-class goals, not afterthoughts

This is why the repository keeps:

- a hybrid Gaussian model for interactive representation
- CT-specific losses instead of image-space GS losses
- mesh / SDF export for analysis-oriented downstream use

## Repository Layout

Core directories and entrypoints:

- [`ct_pipeline/`](/d:/Projects/3.3DGS/SAD-GS/ct_pipeline)
  CT data pipeline modules: loading, preprocessing, geometry analysis, acceleration, compression, exporters, field queries, and backend wrappers
- [`scene/`](/d:/Projects/3.3DGS/SAD-GS/scene)
  Gaussian model core and the CT-specific hybrid model
- [`submodules/ct-native-backend/`](/d:/Projects/3.3DGS/SAD-GS/submodules/ct-native-backend)
  Native CUDA extension for CT training acceleration
- [`tests/`](/d:/Projects/3.3DGS/SAD-GS/tests)
  Regression and smoke tests for the CTGS pipeline
- [`run_ct_phase1.py`](/d:/Projects/3.3DGS/SAD-GS/run_ct_phase1.py)
  Standalone Phase 1 ingestion and analysis entrypoint
- [`train_ct.py`](/d:/Projects/3.3DGS/SAD-GS/train_ct.py)
  Main CT training entrypoint
- [`mesher.py`](/d:/Projects/3.3DGS/SAD-GS/mesher.py)
  CT mesh extraction utility

## Main Components

### 1. CT ingestion and preprocessing

Phase 1 converts a CT dataset into a structured analysis bundle.
This bundle contains the masks, surface samples, interior samples, and geometric annotations needed by the later training stages.

The current preprocessing path is `void-aware`:

- `material_mask` represents occupied material
- `void_mask` represents low-density internal or ROI-contained empty regions
- `foreground_mask` is retained as a coarse ROI / compatibility mask

### 2. Hybrid Gaussian representation

The model combines:

- planar primitives for locally planar surface regions
- anisotropic 3D primitives for non-planar surface regions and bulk material regions

The model persists CT-specific metadata such as:

- primitive type
- normals
- material id
- planarity
- region type

### 3. CT-specific training

Training is based on CT slice supervision and geometry-aware regularization rather than camera rendering.

The training loop includes:

- slice consistency loss
- occupancy supervision
- point-to-plane regularization
- normal alignment regularization
- thickness penalty for planar primitives
- material boundary regularization

### 4. Native CUDA backend

The repository includes a dedicated CT CUDA backend for the main training bottlenecks.
This backend is separate from the removed old GS rasterizer stack.

Current native acceleration covers:

- slice patch rendering
- density query
- KNN refresh
- cached point-to-plane operations

### 5. Dual-output export

The trained CTGS model can be exported to different representations depending on the use case:

- display GS for lightweight viewing and compressed distribution
- mesh for geometry-oriented inspection
- SDF for downstream analysis workflows

## Rendering And Viewing

This repository does not include a full standalone rendering application or an online viewer service.
Its role is to produce the CTGS representation and export formats, not to ship a complete end-user visualization product.

In practice, the recommended usage is:

- train and export a lightweight display GS from this repository
- view that output in an external viewer, custom frontend, or online visualization pipeline
- use mesh / SDF exports when a pure Gaussian display is not the right downstream format

So the repository should be understood as the CT representation, training, and export stack, not the final viewer itself.

## Installation

```powershell
conda env create --file environment.yml
conda activate gaussian_splatting
```

The environment installs the CT native backend from `submodules/ct-native-backend`.

## Typical Workflow

### Phase 1

```powershell
python run_ct_phase1.py ^
  --input D:\path\to\ct_data ^
  --fmt auto ^
  --output D:\path\to\phase1_out
```

Supported inputs:

- DICOM series directory or representative slice
- RAW volume with JSON sidecar
- TIFF stack file or TIFF slice directory

### Training

```powershell
python train_ct.py ^
  --model_path D:\path\to\train_out ^
  --ct_phase1_dir D:\path\to\phase1_out ^
  --ct_volume_path D:\path\to\ct_data ^
  --ct_volume_format auto ^
  --ct_backend auto ^
  --output_gs D:\path\to\train_out\display.ply
```

Backends:

- `--ct_backend auto`
- `--ct_backend python`
- `--ct_backend cuda`

`auto` prefers the native backend and falls back when it is unavailable.

### Mesh extraction

```powershell
python mesher.py ^
  --input D:\path\to\train_out ^
  --output D:\path\to\mesh.ply ^
  --iteration -1 ^
  --resolution 0.05 ^
  --threshold 0.5
```

## Outputs

Phase 1 produces:

- `analysis.npz`
- `metadata.json`

Training produces:

- checkpoints
- per-iteration point clouds
- optional display GS export
- optional mesh export
- optional SDF export

## Scope

This repository intentionally does not include the old scene-based SAD-GS stack anymore.
Removed categories include:

- camera scene loading
- standard GS train / render / eval scripts
- COLMAP / Replica / Blender style pipelines
- the old general-purpose rasterizer training path

What remains is the CTGS stack only.

## Compatibility

`CTGaussianModel` still provides best-effort loading for older GS-style PLY / checkpoint payloads so they can be read and resaved into the CTGS representation.
That compatibility is limited to model payload handling; the old training and rendering entrypoints are not part of this repository anymore.

## Tests

Run the CTGS regression suite with:

```powershell
python -m unittest discover -s tests
```

The tests cover:

- Phase 1 loading and preprocessing
- geometry analysis
- hybrid model persistence and initialization
- CT losses
- native backend parity and stability
- training smoke runs
- exporter and mesher behavior
