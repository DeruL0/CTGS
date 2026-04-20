# CTGS Research Description

## 1. Problem Statement

This project studies how to convert reconstructed industrial CT volumes into a compact Gaussian representation that remains useful for:

- fast browsing and visualization
- slice inspection
- preservation of internal cavities and material structure
- export to analysis-oriented formats such as mesh and SDF

Unlike the original 3DGS, the target is **not** RGB novel-view synthesis.
Unlike X-Gaussian, the target is **not** X-ray projection synthesis from multi-view projections.
The target here is a **CT volume representation framework** learned directly from reconstructed CT data.

## 2. Core Idea

The main idea is to represent a CT object using two kinds of Gaussian primitives:

- **surface Gaussians**, which model material boundaries
- **bulk Gaussians**, which model material interior

This decomposition is important because a CT object is not only an outer shell.
For inspection-oriented use cases, we also care about internal material occupancy, voids, cavities, and material boundaries.

The current active pipeline is:

1. load a reconstructed CT volume
2. separate material and void regions
3. extract boundary samples and interior samples
4. initialize surface and bulk Gaussians
5. optimize them with CT-specific losses
6. export the trained representation for visualization or analysis

## 3. What Changed Compared with Original 3DGS

From the perspective of the original 3DGS, the current framework changes several fundamental assumptions.

### 3.1 Task definition

Original 3DGS is designed for RGB scene rendering and novel-view synthesis.
Its goal is to render realistic color images from unseen camera poses.

This framework is designed for CT volumes.
Its goal is to learn a compact volumetric representation that preserves boundary geometry and internal material/void structure.

### 3.2 Input modality

Original 3DGS uses multi-view RGB images.

This framework uses a reconstructed CT volume as input.
Therefore, training is driven by volumetric structure rather than by multi-view appearance.

### 3.3 Representation

Original 3DGS uses a generic set of 3D Gaussians for a scene.

This framework uses a **surface + bulk** decomposition:

- surface Gaussians model boundaries
- bulk Gaussians model material occupancy

### 3.4 Training signals

Original 3DGS mainly relies on photometric supervision in image space.

This framework uses CT-specific training signals, but they are better understood as a small number of grouped objectives rather than a long flat list of losses.

The active training path can be summarized by four main components:

- **slice reconstruction**
  - encourages rendered orthogonal CT slices to match the original volume slices
- **void-aware occupancy supervision**
  - encourages material regions to stay occupied and void / exterior regions to stay empty
- **signed-surface alignment**
  - aligns surface Gaussians with the material boundary through a signed-field formulation
- **surface shape regularization**
  - stabilizes the surface layer so it does not become too thick, too elongated, or too opaque

## 4. Main Difference from X-Gaussian

Although both methods are related to X-ray/CT and both start from Gaussian representations, the problem setting is substantially different.

### 4.1 X-Gaussian

X-Gaussian is designed for **efficient X-ray novel-view synthesis**.
Its goal is to render new X-ray projections from projection images acquired under scanner geometry.

Its key ideas include:

- an isotropic radiative Gaussian model
- differentiable radiative rasterization (DRR)
- ACUI initialization from scanner geometry

### 4.2 Our framework

Our framework is designed for **reconstructed CT volume representation**.
It does not primarily aim to render new cone-beam X-ray views.
Instead, it aims to learn a compact CT representation that supports:

- slicing
- cavity preservation
- geometry-aware inspection
- export to display GS, mesh, and SDF

### 4.3 Practical summary

X-Gaussian answers:

> how can Gaussian splatting be adapted for fast X-ray projection synthesis?

Our framework answers:

> how can a reconstructed CT volume be converted into a compact, geometry-aware Gaussian representation that preserves material and void structure?

So the two methods are related, but they are not solving the same problem.

## 5. Difference vs. Novelty

### 5.1 Difference

A **difference** means anything that is not the same as the baseline or prior work.

Examples:

- using CT volumes instead of RGB images
- adding a CUDA backend

All of these are differences.

### 5.2 Novelty

A **novelty** is a difference that is central enough to be claimed as a research contribution.

Not every difference is a novelty.

For this project, the likely novelty candidates are:

- a CT-specific surface + bulk Gaussian representation
- void-aware occupancy and signed-field boundary training

## 6. Why These Changes Can Lead to Better Results

### 6.1 Surface + bulk decomposition

If a model uses only a generic Gaussian cloud, it must use the same primitives to represent both boundaries and interior structure.
This makes optimization harder because boundary geometry and material occupancy have different roles.

By separating surface and bulk:

- surface Gaussians can focus on boundary localization
- bulk Gaussians can focus on material occupancy

This reduces representational conflict and makes it easier to preserve both shape and interior structure.

### 6.2 Void-aware occupancy supervision

Internal cavities are important in CT.
If the training signal only distinguishes foreground from background in a coarse way, the model may fill internal holes.

By explicitly treating:

- material as positive
- void as negative
- exterior as negative

the model is pushed to preserve internal cavities instead of collapsing them into solid material.

### 6.3 Signed-surface alignment

Slice loss and occupancy loss alone are not sufficient to force sharp and well-localized boundaries.
They can produce boundaries that are approximately correct but still soft or spatially shifted.

Signed-surface alignment helps because it gives a direct geometric signal:

- surface centers should lie close to the zero level set
- surface normals should align with the local signed-field gradient

This makes surface Gaussians more likely to stay on the actual boundary, keeping the surface aligned with the true boundary.

### 6.4 Surface shape regularization

In longer training runs, surface Gaussians can become:

- too elongated along tangential directions
- too opaque

This tends to create unstable viewer artifacts such as streaks or star-like splats.

This is why the method benefits from a grouped **surface shape regularization** term.
In implementation, this group currently includes:

- thickness control along the boundary normal
- tangential scale control in the local surface plane
- opacity control to avoid saturation

These are best treated as stabilizers, making long runs more stable.

## 8. Current Strengths

At the current stage, the framework already shows several strengths:

- the outer contour is generally stable for novel view and mesh extraction
- large internal cavities can be preserved
- the trained representation can be exported to display GS, mesh, and SDF

## 9. Current Weaknesses

The current limitations should also be stated clearly.

- small cavities and thin internal structures are still harder to preserve
- long training can still destabilize surface Gaussians if regularization is weak
- the framework is not a full X-ray novel-view synthesis method
- the active path is specialized for CT volumes rather than general scenes

## 10. Terminology Notes

The English term often used in this context is:

- **supervision**

However, directly translating it as Japanese `監督` is unnatural in technical writing.
Depending on context, better Japanese equivalents are:

- 教師信号
- 学習信号
- 制約
- 正則化

Examples:

- `slice supervision` -> 切片に基づく教師信号 / 切片一致制約
- `occupancy supervision` -> 占有に関する教師信号
- `boundary supervision` -> 境界整列の教師信号

In this framework, it is often even clearer to avoid the word "supervision" in Japanese and explain the role directly in terms of:

- what is being matched
- what is being constrained
- what instability is being regularized

## 12. Short Summary

In one sentence, this research is not simply "3DGS for CT."
It is a **CT-oriented Gaussian representation framework** that combines:

- boundary-aware surface modeling
- bulk occupancy modeling
- void-aware supervision
- signed-field alignment
- GPU local-query acceleration

to produce a compact representation suitable for slicing, visualization, and geometry-aware CT analysis.
