#include <torch/extension.h>

#include <limits>
#include <tuple>
#include <vector>

namespace {

void check_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
}

std::vector<torch::Tensor> empty_plane_targets(const torch::Tensor& xyz) {
    auto long_options = torch::TensorOptions().device(xyz.device()).dtype(torch::kLong);
    auto float_options = xyz.options();
    return {
        torch::empty({0}, long_options),
        torch::empty({0, 3}, float_options),
        torch::empty({0, 3}, float_options),
        torch::empty({0}, float_options),
    };
}

torch::Tensor normalize_rows(torch::Tensor tensor) {
    return tensor / tensor.norm(2, -1, true).clamp_min(1e-12);
}

}  // namespace

std::vector<torch::Tensor> build_plane_targets_cuda(
    torch::Tensor xyz,
    torch::Tensor normals,
    torch::Tensor planarity,
    torch::Tensor material_ids,
    torch::Tensor planar_mask,
    torch::Tensor neighbor_index) {
    check_cuda(xyz, "xyz");
    check_cuda(normals, "normals");
    check_cuda(planarity, "planarity");
    check_cuda(planar_mask, "planar_mask");
    check_cuda(neighbor_index, "neighbor_index");

    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must have shape (N, 3).");
    TORCH_CHECK(normals.dim() == 2 && normals.size(1) == 3, "normals must have shape (N, 3).");
    TORCH_CHECK(neighbor_index.dim() == 2, "neighbor_index must have shape (N, K).");
    if (xyz.numel() == 0 || neighbor_index.numel() == 0) {
        return empty_plane_targets(xyz);
    }

    auto work_planarity = planarity.reshape({-1});
    auto work_material_ids = material_ids.numel() == 0 ? torch::full({xyz.size(0)}, -1, torch::TensorOptions().device(xyz.device()).dtype(torch::kLong)) : material_ids.reshape({-1});
    auto work_planar_mask = planar_mask.reshape({-1}).to(torch::kBool);

    auto active_indices = torch::nonzero(work_planar_mask & work_planarity.gt(0.0)).reshape({-1});
    if (active_indices.numel() == 0) {
        return empty_plane_targets(xyz);
    }

    auto active_neighbors = neighbor_index.index_select(0, active_indices).to(torch::kLong);
    auto clamped_neighbors = active_neighbors.clamp_min(0);
    auto valid = active_neighbors.ge(0) & active_neighbors.lt(xyz.size(0));
    valid = valid & clamped_neighbors.ne(active_indices.unsqueeze(1));

    auto planar_neighbors = work_planar_mask.index_select(0, clamped_neighbors.reshape({-1})).reshape_as(clamped_neighbors);
    valid = valid & planar_neighbors;

    auto center_material = work_material_ids.index_select(0, active_indices);
    auto neighbor_material = work_material_ids.index_select(0, clamped_neighbors.reshape({-1})).reshape_as(clamped_neighbors);
    auto material_valid = center_material.unsqueeze(1).lt(0) | neighbor_material.eq(center_material.unsqueeze(1));
    valid = valid & material_valid;

    auto counts = valid.sum(1);
    auto valid_rows = counts.ge(3);
    auto valid_row_indices = torch::nonzero(valid_rows).reshape({-1});
    if (valid_row_indices.numel() == 0) {
        return empty_plane_targets(xyz);
    }

    active_indices = active_indices.index_select(0, valid_row_indices);
    counts = counts.index_select(0, valid_row_indices);
    clamped_neighbors = clamped_neighbors.index_select(0, valid_row_indices);
    valid = valid.index_select(0, valid_row_indices);

    auto neighbor_points = xyz.index_select(0, clamped_neighbors.reshape({-1})).reshape({clamped_neighbors.size(0), clamped_neighbors.size(1), 3});
    auto valid_float = valid.to(xyz.scalar_type()).unsqueeze(-1);
    auto counts_float = counts.to(xyz.scalar_type()).unsqueeze(-1);
    auto centroids = (neighbor_points * valid_float).sum(1) / counts_float;

    auto centered = (neighbor_points - centroids.unsqueeze(1)) * valid_float;
    auto denom = (counts - 1).clamp_min(1).to(xyz.scalar_type()).view({-1, 1, 1});
    auto covariance = centered.transpose(1, 2).matmul(centered) / denom;

    auto trace = covariance.diagonal(0, 1, 2).sum(-1).abs();
    auto jitter = torch::clamp(trace * 1e-6, 1e-8).view({-1, 1, 1});
    auto eye = torch::eye(3, xyz.options()).unsqueeze(0);
    covariance = covariance + eye * jitter;

    auto eigh_result = torch::linalg_eigh(covariance);
    auto eigenvectors = std::get<1>(eigh_result);
    auto fitted_normals = normalize_rows(eigenvectors.select(2, 0));

    auto reference_normals = normalize_rows(normals.index_select(0, active_indices).detach());
    auto alignment = (fitted_normals * reference_normals).sum(-1);
    auto signs = torch::where(alignment.lt(0.0), -torch::ones_like(alignment), torch::ones_like(alignment)).unsqueeze(-1);
    fitted_normals = fitted_normals * signs;

    auto finite_mask = torch::isfinite(fitted_normals).all(1) & torch::isfinite(centroids).all(1);
    auto finite_indices = torch::nonzero(finite_mask).reshape({-1});
    if (finite_indices.numel() == 0) {
        return empty_plane_targets(xyz);
    }

    active_indices = active_indices.index_select(0, finite_indices);
    centroids = centroids.index_select(0, finite_indices);
    fitted_normals = fitted_normals.index_select(0, finite_indices);
    auto weights = work_planarity.index_select(0, active_indices).to(xyz.scalar_type());

    return {active_indices, centroids, fitted_normals, weights};
}

torch::Tensor point_to_plane_loss_cuda(
    torch::Tensor xyz,
    torch::Tensor active_indices,
    torch::Tensor centroids,
    torch::Tensor fitted_normals,
    torch::Tensor weights) {
    check_cuda(xyz, "xyz");
    check_cuda(active_indices, "active_indices");
    check_cuda(centroids, "centroids");
    check_cuda(fitted_normals, "fitted_normals");
    check_cuda(weights, "weights");

    if (xyz.numel() == 0 || active_indices.numel() == 0) {
        return torch::zeros({}, xyz.options());
    }

    auto centers = xyz.index_select(0, active_indices.to(torch::kLong));
    auto distances = torch::abs(((centers - centroids) * fitted_normals).sum(-1));
    return (distances * weights.to(xyz.scalar_type())).mean();
}
