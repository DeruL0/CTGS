#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

void check_cuda(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
}

void check_float(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32.");
}

__global__ void query_density_forward_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const float* query_points,
    int64_t num_gaussians,
    int64_t num_queries,
    float* output) {
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= num_queries) {
        return;
    }

    const float* query_point = query_points + query_index * 3;
    float qx = query_point[0];
    float qy = query_point[1];
    float qz = query_point[2];

    float accum = 0.0f;
    for (int64_t gaussian_idx = 0; gaussian_idx < num_gaussians; ++gaussian_idx) {
        const float* mean = means + gaussian_idx * 3;
        const float* rotation = rotations + gaussian_idx * 9;
        const float* scale = scales + gaussian_idx * 3;

        float dx = qx - mean[0];
        float dy = qy - mean[1];
        float dz = qz - mean[2];

        float local_x = dx * rotation[0] + dy * rotation[3] + dz * rotation[6];
        float local_y = dx * rotation[1] + dy * rotation[4] + dz * rotation[7];
        float local_z = dx * rotation[2] + dy * rotation[5] + dz * rotation[8];

        float nx = local_x / scale[0];
        float ny = local_y / scale[1];
        float nz = local_z / scale[2];
        float exponent = -0.5f * (nx * nx + ny * ny + nz * nz);
        accum += expf(exponent) * opacity[gaussian_idx];
    }

    output[query_index] = accum;
}

__global__ void query_density_backward_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const float* query_points,
    const float* grad_output,
    int64_t num_gaussians,
    int64_t num_queries,
    float* grad_means,
    float* grad_rotations,
    float* grad_scales,
    float* grad_opacity) {
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= num_queries) {
        return;
    }

    float upstream = grad_output[query_index];
    if (upstream == 0.0f) {
        return;
    }

    const float* query_point = query_points + query_index * 3;
    float qx = query_point[0];
    float qy = query_point[1];
    float qz = query_point[2];

    for (int64_t gaussian_idx = 0; gaussian_idx < num_gaussians; ++gaussian_idx) {
        const float* mean = means + gaussian_idx * 3;
        const float* rotation = rotations + gaussian_idx * 9;
        const float* scale = scales + gaussian_idx * 3;

        float dx = qx - mean[0];
        float dy = qy - mean[1];
        float dz = qz - mean[2];

        float local_x = dx * rotation[0] + dy * rotation[3] + dz * rotation[6];
        float local_y = dx * rotation[1] + dy * rotation[4] + dz * rotation[7];
        float local_z = dx * rotation[2] + dy * rotation[5] + dz * rotation[8];

        float inv_sx = 1.0f / scale[0];
        float inv_sy = 1.0f / scale[1];
        float inv_sz = 1.0f / scale[2];
        float nx = local_x * inv_sx;
        float ny = local_y * inv_sy;
        float nz = local_z * inv_sz;
        float exponent = -0.5f * (nx * nx + ny * ny + nz * nz);
        float exp_value = expf(exponent);

        float grad_opacity_value = upstream * exp_value;
        float common = grad_opacity_value * opacity[gaussian_idx];

        float inv_sx2 = inv_sx * inv_sx;
        float inv_sy2 = inv_sy * inv_sy;
        float inv_sz2 = inv_sz * inv_sz;
        float local_grad_x = common * (-local_x * inv_sx2);
        float local_grad_y = common * (-local_y * inv_sy2);
        float local_grad_z = common * (-local_z * inv_sz2);

        float grad_mean_x = -(local_grad_x * rotation[0] + local_grad_y * rotation[1] + local_grad_z * rotation[2]);
        float grad_mean_y = -(local_grad_x * rotation[3] + local_grad_y * rotation[4] + local_grad_z * rotation[5]);
        float grad_mean_z = -(local_grad_x * rotation[6] + local_grad_y * rotation[7] + local_grad_z * rotation[8]);

        atomicAdd(grad_means + gaussian_idx * 3 + 0, grad_mean_x);
        atomicAdd(grad_means + gaussian_idx * 3 + 1, grad_mean_y);
        atomicAdd(grad_means + gaussian_idx * 3 + 2, grad_mean_z);

        atomicAdd(grad_rotations + gaussian_idx * 9 + 0, local_grad_x * dx);
        atomicAdd(grad_rotations + gaussian_idx * 9 + 3, local_grad_x * dy);
        atomicAdd(grad_rotations + gaussian_idx * 9 + 6, local_grad_x * dz);
        atomicAdd(grad_rotations + gaussian_idx * 9 + 1, local_grad_y * dx);
        atomicAdd(grad_rotations + gaussian_idx * 9 + 4, local_grad_y * dy);
        atomicAdd(grad_rotations + gaussian_idx * 9 + 7, local_grad_y * dz);
        atomicAdd(grad_rotations + gaussian_idx * 9 + 2, local_grad_z * dx);
        atomicAdd(grad_rotations + gaussian_idx * 9 + 5, local_grad_z * dy);
        atomicAdd(grad_rotations + gaussian_idx * 9 + 8, local_grad_z * dz);

        float grad_scale_x = common * (local_x * local_x) * inv_sx2 * inv_sx;
        float grad_scale_y = common * (local_y * local_y) * inv_sy2 * inv_sy;
        float grad_scale_z = common * (local_z * local_z) * inv_sz2 * inv_sz;
        atomicAdd(grad_scales + gaussian_idx * 3 + 0, grad_scale_x);
        atomicAdd(grad_scales + gaussian_idx * 3 + 1, grad_scale_y);
        atomicAdd(grad_scales + gaussian_idx * 3 + 2, grad_scale_z);

        atomicAdd(grad_opacity + gaussian_idx, grad_opacity_value);
    }
}

}  // namespace

at::Tensor query_density_forward_cuda(
    at::Tensor means,
    at::Tensor rotations,
    at::Tensor scales,
    at::Tensor opacity,
    at::Tensor query_points) {
    check_cuda(means, "means");
    check_cuda(rotations, "rotations");
    check_cuda(scales, "scales");
    check_cuda(opacity, "opacity");
    check_cuda(query_points, "query_points");
    check_float(means, "means");
    check_float(rotations, "rotations");
    check_float(scales, "scales");
    check_float(opacity, "opacity");
    check_float(query_points, "query_points");

    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must have shape (N, 3).");
    TORCH_CHECK(rotations.dim() == 3 && rotations.size(1) == 3 && rotations.size(2) == 3, "rotations must have shape (N, 3, 3).");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must have shape (N, 3).");
    TORCH_CHECK(opacity.dim() == 1, "opacity must have shape (N,).");
    TORCH_CHECK(query_points.dim() == 2 && query_points.size(1) == 3, "query_points must have shape (Q, 3).");

    c10::cuda::CUDAGuard device_guard(means.device());
    auto output = at::zeros({query_points.size(0)}, means.options());
    if (means.numel() == 0 || query_points.numel() == 0) {
        return output;
    }

    constexpr int kThreads = 256;
    int blocks = static_cast<int>((query_points.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();

    query_density_forward_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        means.size(0),
        query_points.size(0),
        output.data_ptr<float>());

    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "query_density_forward kernel launch failed: ", cudaGetErrorString(error));
    return output;
}

std::vector<at::Tensor> query_density_backward_cuda(
    at::Tensor means,
    at::Tensor rotations,
    at::Tensor scales,
    at::Tensor opacity,
    at::Tensor query_points,
    at::Tensor grad_output) {
    check_cuda(means, "means");
    check_cuda(rotations, "rotations");
    check_cuda(scales, "scales");
    check_cuda(opacity, "opacity");
    check_cuda(query_points, "query_points");
    check_cuda(grad_output, "grad_output");
    check_float(means, "means");
    check_float(rotations, "rotations");
    check_float(scales, "scales");
    check_float(opacity, "opacity");
    check_float(query_points, "query_points");
    check_float(grad_output, "grad_output");

    TORCH_CHECK(grad_output.dim() == 1 && grad_output.size(0) == query_points.size(0), "grad_output must have shape (Q,).");

    c10::cuda::CUDAGuard device_guard(means.device());
    auto grad_means = at::zeros_like(means);
    auto grad_rotations = at::zeros_like(rotations);
    auto grad_scales = at::zeros_like(scales);
    auto grad_opacity = at::zeros_like(opacity);
    if (means.numel() == 0 || query_points.numel() == 0) {
        return {grad_means, grad_rotations, grad_scales, grad_opacity};
    }

    constexpr int kThreads = 256;
    int blocks = static_cast<int>((query_points.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();

    query_density_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        grad_output.contiguous().data_ptr<float>(),
        means.size(0),
        query_points.size(0),
        grad_means.data_ptr<float>(),
        grad_rotations.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_opacity.data_ptr<float>());

    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "query_density_backward kernel launch failed: ", cudaGetErrorString(error));
    return {grad_means, grad_rotations, grad_scales, grad_opacity};
}
