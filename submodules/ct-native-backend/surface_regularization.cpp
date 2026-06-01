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
