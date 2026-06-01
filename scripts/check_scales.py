from scene.ct_gaussian_model import CTGaussianModel
import torch
import numpy as np

model = CTGaussianModel(0)
model.load_ply('outputs/bulk_no_internal_gate_sweep_20260529_031050/iter1000/variant_base/point_cloud/iteration_1000/point_cloud.ply')
region = model.get_region_type.reshape(-1)
surface_mask = region == 0
bulk_mask = region == 1
scales = model.get_scaling.detach().cpu()
s = scales[surface_mask.cpu()]
b = scales[bulk_mask.cpu()]
sp = 0.3906

def q(t, ps=[0.5, 0.9, 0.99]):
    return [round(float(torch.quantile(t, p)), 4) for p in ps]

st = s[:, :2].max(dim=1).values
sn = s[:, 2]
bm = b.max(dim=1).values

print(f"Surface tangent σ (mm) p50/p90/p99: {q(st)}")
print(f"Surface normal  σ (mm) p50/p90/p99: {q(sn)}")
print(f"Surface tangent 4σ p50 = {q(st,[0.5])[0]*4/sp:.2f} voxels")
print(f"Surface normal  4σ p50 = {q(sn,[0.5])[0]*4/sp:.2f} voxels")
print(f"Surface count: {int(surface_mask.sum())}")
print()
print(f"Bulk max σ (mm) p50/p90/p99: {q(bm)}")
print(f"Bulk max  4σ p50 = {q(bm,[0.5])[0]*4/sp:.2f} voxels")
print(f"Bulk count: {int(bulk_mask.sum())}")

# Overlap analysis: how far does surface 4σ extend inward?
# Surface at sdf=0, tangent 4σ in-plane => how many voxels it "covers" in tangent plane
print()
print(f"Surface tangent 4σ p50 coverage radius = {q(st,[0.5])[0]*4/sp:.2f} voxels")
print(f"  -> each surface gaussian covers ~{(q(st,[0.5])[0]*4/sp)**2 * 3.14:.1f} voxels^2 in tangent plane")
print(f"Surface density = {int(surface_mask.sum())} gaussians")
print(f"Material surface area ~ {int(surface_mask.sum())} boundary voxels")
print()
print("Key question: does surface tangent footprint extend FAR INTO material interior?")
print(f"  If bulk starts at ~sdf=-1.5 voxel gap,")
print(f"  and surface tangent 4sigma = {q(st,[0.5])[0]*4/sp:.2f} voxels,")
print(f"  surface gaussian centered at sdf=0 contributes exp(-2*({q(st,[0.5])[0]*4/sp:.1f}/4)^2)")
at_15 = float(torch.exp(torch.tensor(-0.5 * (1.5 / q(st,[0.5])[0])**2)))
print(f"  at sdf=-1.5 voxels (deep material): occupancy contribution ≈ {at_15:.4f}")
