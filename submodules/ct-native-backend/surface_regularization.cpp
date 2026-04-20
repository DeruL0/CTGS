#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace {

void check_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
}

void check_float(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat, name, " must be float32.");
}

torch::Tensor zeros_like_scalar(const torch::Tensor& reference) {
    return torch::zeros({}, reference.options());
}

std::vector<torch::Tensor> empty_surface_thickness_grads(
    const torch::Tensor& raw_scaling,
    const torch::Tensor& rotation_mats,
    const torch::Tensor& normals) {
    return {
        torch::zeros_like(raw_scaling),
        torch::zeros_like(rotation_mats),
        torch::zeros_like(normals),
    };
}

std::vector<torch::Tensor> empty_material_boundary_grads(
    const torch::Tensor& xyz,
    const torch::Tensor& opacity) {
    return {
        torch::zeros_like(xyz),
        torch::zeros_like(opacity),
    };
}

}  // namespace

torch::Tensor surface_thickness_loss_forward_cuda(
    torch::Tensor raw_scaling,
    torch::Tensor rotation_mats,
    torch::Tensor normals,
    double max_thickness) {
    check_cuda(raw_scaling, "raw_scaling");
    check_cuda(rotation_mats, "rotation_mats");
    check_cuda(normals, "normals");
    check_float(raw_scaling, "raw_scaling");
    check_float(rotation_mats, "rotation_mats");
    check_float(normals, "normals");

    TORCH_CHECK(raw_scaling.dim() == 2 && raw_scaling.size(1) == 3, "raw_scaling must have shape (N, 3).");
    TORCH_CHECK(rotation_mats.dim() == 3 && rotation_mats.size(1) == 3 && rotation_mats.size(2) == 3, "rotation_mats must have shape (N, 3, 3).");
    TORCH_CHECK(normals.dim() == 2 && normals.size(1) == 3, "normals must have shape (N, 3).");
    TORCH_CHECK(raw_scaling.size(0) == rotation_mats.size(0) && raw_scaling.size(0) == normals.size(0), "surface_thickness inputs must share the same batch dimension.");

    if (raw_scaling.numel() == 0) {
        return zeros_like_scalar(raw_scaling);
    }

    auto scales = torch::exp(raw_scaling);
    auto local_normals = rotation_mats.transpose(1, 2).matmul(normals.unsqueeze(-1)).squeeze(-1);
    auto variance = torch::sum(torch::pow(local_normals * scales, 2), 1);
    auto thickness = torch::sqrt(torch::clamp_min(variance, 1e-8));
    return torch::relu(thickness - static_cast<float>(max_thickness)).mean();
}

std::vector<torch::Tensor> surface_thickness_loss_backward_cuda(
    torch::Tensor raw_scaling,
    torch::Tensor rotation_mats,
    torch::Tensor normals,
    double max_thickness,
    torch::Tensor grad_output) {
    check_cuda(raw_scaling, "raw_scaling");
    check_cuda(rotation_mats, "rotation_mats");
    check_cuda(normals, "normals");
    check_cuda(grad_output, "grad_output");
    check_float(raw_scaling, "raw_scaling");
    check_float(rotation_mats, "rotation_mats");
    check_float(normals, "normals");

    if (raw_scaling.numel() == 0) {
        return empty_surface_thickness_grads(raw_scaling, rotation_mats, normals);
    }

    auto scales = torch::exp(raw_scaling);
    auto local_normals = rotation_mats.transpose(1, 2).matmul(normals.unsqueeze(-1)).squeeze(-1);
    auto variance = torch::sum(torch::pow(local_normals * scales, 2), 1);
    auto thickness = torch::sqrt(torch::clamp_min(variance, 1e-8));
    auto active = thickness.gt(static_cast<float>(max_thickness)).to(raw_scaling.scalar_type());
    if (active.sum().item<int64_t>() == 0) {
        return empty_surface_thickness_grads(raw_scaling, rotation_mats, normals);
    }

    auto grad_scale = active * grad_output.to(raw_scaling.scalar_type()) / static_cast<float>(raw_scaling.size(0));
    auto safe_thickness = torch::clamp_min(thickness, 1e-8).unsqueeze(-1);
    auto scale_sq = torch::pow(scales, 2);
    auto grad_local = grad_scale.unsqueeze(-1) * local_normals * scale_sq / safe_thickness;
    auto grad_raw = grad_scale.unsqueeze(-1) * torch::pow(local_normals, 2) * scale_sq / safe_thickness;
    auto grad_rotation = normals.unsqueeze(2) * grad_local.unsqueeze(1);
    auto grad_normals = rotation_mats.matmul(grad_local.unsqueeze(-1)).squeeze(-1);
    return {grad_raw, grad_rotation, grad_normals};
}

torch::Tensor material_boundary_loss_forward_cuda(
    torch::Tensor xyz,
    torch::Tensor material_ids,
    torch::Tensor opacity,
    torch::Tensor neighbor_index,
    double target_opacity) {
    check_cuda(xyz, "xyz");
    check_cuda(material_ids, "material_ids");
    check_cuda(opacity, "opacity");
    check_cuda(neighbor_index, "neighbor_index");
    check_float(xyz, "xyz");
    check_float(opacity, "opacity");

    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must have shape (N, 3).");
    TORCH_CHECK(neighbor_index.dim() == 2, "neighbor_index must have shape (N, K).");
    if (xyz.numel() == 0 || neighbor_index.numel() == 0) {
        return zeros_like_scalar(xyz);
    }

    auto work_material_ids = material_ids.reshape({-1}).to(torch::kLong);
    auto nonnegative_ids = work_material_ids.masked_select(work_material_ids.ge(0));
    if (nonnegative_ids.numel() == 0 || torch::equal(nonnegative_ids.min(), nonnegative_ids.max())) {
        return zeros_like_scalar(xyz);
    }

    auto work_opacity = opacity.reshape({-1}).to(xyz.scalar_type());
    auto clamped_neighbors = neighbor_index.clamp_min(0).to(torch::kLong);
    auto center_material = work_material_ids.unsqueeze(1);
    auto neighbor_material = work_material_ids.index_select(0, clamped_neighbors.reshape({-1})).reshape_as(clamped_neighbors);
    auto valid_pairs = center_material.ge(0) & neighbor_material.ge(0) & center_material.ne(neighbor_material);
    if (valid_pairs.sum().item<int64_t>() == 0) {
        return zeros_like_scalar(xyz);
    }

    auto neighbor_xyz = xyz.index_select(0, clamped_neighbors.reshape({-1})).reshape({xyz.size(0), clamped_neighbors.size(1), 3});
    auto distances = torch::norm(xyz.unsqueeze(1) - neighbor_xyz, 2, -1);
    auto valid_distances = distances.masked_select(valid_pairs);
    auto mean_spacing = valid_distances.mean().clamp_min(1e-6);
    auto neighbor_opacity = work_opacity.index_select(0, clamped_neighbors.reshape({-1})).reshape_as(clamped_neighbors).to(xyz.scalar_type());
    auto pair_opacity = torch::minimum(work_opacity.unsqueeze(1), neighbor_opacity);
    auto losses = torch::relu(static_cast<float>(target_opacity) - pair_opacity) * torch::exp(-distances / mean_spacing);
    return losses.masked_select(valid_pairs).mean();
}

std::vector<torch::Tensor> material_boundary_loss_backward_cuda(
    torch::Tensor xyz,
    torch::Tensor material_ids,
    torch::Tensor opacity,
    torch::Tensor neighbor_index,
    double target_opacity,
    torch::Tensor grad_output) {
    check_cuda(xyz, "xyz");
    check_cuda(material_ids, "material_ids");
    check_cuda(opacity, "opacity");
    check_cuda(neighbor_index, "neighbor_index");
    check_cuda(grad_output, "grad_output");
    check_float(xyz, "xyz");
    check_float(opacity, "opacity");

    if (xyz.numel() == 0 || neighbor_index.numel() == 0) {
        return empty_material_boundary_grads(xyz, opacity);
    }

    auto work_material_ids = material_ids.reshape({-1}).to(torch::kLong);
    auto nonnegative_ids = work_material_ids.masked_select(work_material_ids.ge(0));
    if (nonnegative_ids.numel() == 0 || torch::equal(nonnegative_ids.min(), nonnegative_ids.max())) {
        return empty_material_boundary_grads(xyz, opacity);
    }

    auto work_opacity = opacity.reshape({-1}).to(xyz.scalar_type());
    auto clamped_neighbors = neighbor_index.clamp_min(0).to(torch::kLong);
    auto center_material = work_material_ids.unsqueeze(1);
    auto neighbor_material = work_material_ids.index_select(0, clamped_neighbors.reshape({-1})).reshape_as(clamped_neighbors);
    auto valid_pairs = center_material.ge(0) & neighbor_material.ge(0) & center_material.ne(neighbor_material);
    auto valid_count = valid_pairs.sum().item<int64_t>();
    if (valid_count == 0) {
        return empty_material_boundary_grads(xyz, opacity);
    }

    auto neighbor_xyz = xyz.index_select(0, clamped_neighbors.reshape({-1})).reshape({xyz.size(0), clamped_neighbors.size(1), 3});
    auto diff = xyz.unsqueeze(1) - neighbor_xyz;
    auto distances = torch::norm(diff, 2, -1);
    auto safe_distances = torch::clamp_min(distances, 1e-8);
    auto valid_distances = distances.masked_select(valid_pairs);
    auto mean_spacing = valid_distances.mean().clamp_min(1e-6);

    auto neighbor_opacity = work_opacity.index_select(0, clamped_neighbors.reshape({-1})).reshape_as(clamped_neighbors).to(xyz.scalar_type());
    auto center_opacity = work_opacity.unsqueeze(1);
    auto pair_opacity = torch::minimum(center_opacity, neighbor_opacity);
    auto opacity_margin = torch::relu(static_cast<float>(target_opacity) - pair_opacity);
    auto weights = torch::exp(-distances / mean_spacing);
    auto losses = opacity_margin * weights;

    auto valid_float = valid_pairs.to(xyz.scalar_type());
    auto pair_count = static_cast<float>(valid_count);
    auto sum_weighted_distance = (losses * distances * valid_float).sum();
    auto grad_dist = valid_float * (
        -losses / (pair_count * mean_spacing) +
        sum_weighted_distance / (pair_count * pair_count * mean_spacing * mean_spacing)
    );

    auto direction = diff / safe_distances.unsqueeze(-1);
    auto grad_xyz = (grad_dist.unsqueeze(-1) * direction).sum(1);
    auto grad_neighbor_xyz = -(grad_dist.unsqueeze(-1) * direction);
    grad_xyz.index_add_(0, clamped_neighbors.reshape({-1}), grad_neighbor_xyz.reshape({-1, 3}));

    auto opacity_grad_pair = -weights * valid_float * (pair_opacity.lt(static_cast<float>(target_opacity))).to(xyz.scalar_type()) / pair_count;
    auto center_is_min = center_opacity.lt(neighbor_opacity).to(xyz.scalar_type());
    auto neighbor_is_min = neighbor_opacity.lt(center_opacity).to(xyz.scalar_type());
    auto equal_mask = (1.0 - center_is_min - neighbor_is_min).clamp_min(0.0);
    auto grad_center_opacity = opacity_grad_pair * (center_is_min + 0.5 * equal_mask);
    auto grad_neighbor_opacity = opacity_grad_pair * (neighbor_is_min + 0.5 * equal_mask);
    auto grad_opacity = grad_center_opacity.sum(1);
    grad_opacity.index_add_(0, clamped_neighbors.reshape({-1}), grad_neighbor_opacity.reshape({-1}));

    auto grad_scale = grad_output.to(xyz.scalar_type());
    grad_xyz = grad_xyz * grad_scale;
    grad_opacity = grad_opacity * grad_scale;
    return {grad_xyz, grad_opacity.reshape_as(opacity)};
}
