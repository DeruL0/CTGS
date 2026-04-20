#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

void check_cuda(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
}

void check_float(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32.");
}

__device__ inline float read_strength(
    const float* strength_volume,
    int64_t depth,
    int64_t height,
    int64_t width,
    int x,
    int y,
    int z) {
    if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
        return 0.0f;
    }
    return strength_volume[(static_cast<int64_t>(z) * height + y) * width + x];
}

__device__ inline float read_normal_component(
    const float* normal_volume,
    int64_t depth,
    int64_t height,
    int64_t width,
    int x,
    int y,
    int z,
    int c) {
    if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
        return 0.0f;
    }
    return normal_volume[(((static_cast<int64_t>(z) * height + y) * width + x) * 3) + c];
}

__device__ inline float lerp(float a, float b, float w) {
    return a + (b - a) * w;
}

__global__ void sample_boundary_field_forward_kernel(
    const float* strength_volume,
    const float* normal_volume,
    const float* query_points,
    int64_t num_queries,
    int64_t depth,
    int64_t height,
    int64_t width,
    float spacing_z,
    float spacing_y,
    float spacing_x,
    float* output_strength,
    float* output_normals) {
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= num_queries) {
        return;
    }

    const float* query_point = query_points + query_index * 3;
    float x = query_point[0] / spacing_x;
    float y = query_point[1] / spacing_y;
    float z = query_point[2] / spacing_z;

    if (x <= -1.0f || x >= static_cast<float>(width) || y <= -1.0f || y >= static_cast<float>(height) || z <= -1.0f || z >= static_cast<float>(depth)) {
        output_strength[query_index] = 0.0f;
        output_normals[query_index * 3 + 0] = 0.0f;
        output_normals[query_index * 3 + 1] = 0.0f;
        output_normals[query_index * 3 + 2] = 0.0f;
        return;
    }

    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int z0 = static_cast<int>(floorf(z));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float wx = x - static_cast<float>(x0);
    float wy = y - static_cast<float>(y0);
    float wz = z - static_cast<float>(z0);

    float s000 = read_strength(strength_volume, depth, height, width, x0, y0, z0);
    float s100 = read_strength(strength_volume, depth, height, width, x1, y0, z0);
    float s010 = read_strength(strength_volume, depth, height, width, x0, y1, z0);
    float s110 = read_strength(strength_volume, depth, height, width, x1, y1, z0);
    float s001 = read_strength(strength_volume, depth, height, width, x0, y0, z1);
    float s101 = read_strength(strength_volume, depth, height, width, x1, y0, z1);
    float s011 = read_strength(strength_volume, depth, height, width, x0, y1, z1);
    float s111 = read_strength(strength_volume, depth, height, width, x1, y1, z1);

    float s00 = lerp(s000, s100, wx);
    float s10 = lerp(s010, s110, wx);
    float s01 = lerp(s001, s101, wx);
    float s11 = lerp(s011, s111, wx);
    float s0 = lerp(s00, s10, wy);
    float s1 = lerp(s01, s11, wy);
    output_strength[query_index] = lerp(s0, s1, wz);

    for (int c = 0; c < 3; ++c) {
        float n000 = read_normal_component(normal_volume, depth, height, width, x0, y0, z0, c);
        float n100 = read_normal_component(normal_volume, depth, height, width, x1, y0, z0, c);
        float n010 = read_normal_component(normal_volume, depth, height, width, x0, y1, z0, c);
        float n110 = read_normal_component(normal_volume, depth, height, width, x1, y1, z0, c);
        float n001 = read_normal_component(normal_volume, depth, height, width, x0, y0, z1, c);
        float n101 = read_normal_component(normal_volume, depth, height, width, x1, y0, z1, c);
        float n011 = read_normal_component(normal_volume, depth, height, width, x0, y1, z1, c);
        float n111 = read_normal_component(normal_volume, depth, height, width, x1, y1, z1, c);

        float n00 = lerp(n000, n100, wx);
        float n10 = lerp(n010, n110, wx);
        float n01 = lerp(n001, n101, wx);
        float n11 = lerp(n011, n111, wx);
        float n0 = lerp(n00, n10, wy);
        float n1 = lerp(n01, n11, wy);
        output_normals[query_index * 3 + c] = lerp(n0, n1, wz);
    }
}

__global__ void sample_boundary_field_backward_kernel(
    const float* strength_volume,
    const float* normal_volume,
    const float* query_points,
    const float* grad_strength,
    const float* grad_normals,
    int64_t num_queries,
    int64_t depth,
    int64_t height,
    int64_t width,
    float spacing_z,
    float spacing_y,
    float spacing_x,
    float* grad_points) {
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= num_queries) {
        return;
    }

    const float* query_point = query_points + query_index * 3;
    float x = query_point[0] / spacing_x;
    float y = query_point[1] / spacing_y;
    float z = query_point[2] / spacing_z;

    if (x <= -1.0f || x >= static_cast<float>(width) || y <= -1.0f || y >= static_cast<float>(height) || z <= -1.0f || z >= static_cast<float>(depth)) {
        grad_points[query_index * 3 + 0] = 0.0f;
        grad_points[query_index * 3 + 1] = 0.0f;
        grad_points[query_index * 3 + 2] = 0.0f;
        return;
    }

    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int z0 = static_cast<int>(floorf(z));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float wx = x - static_cast<float>(x0);
    float wy = y - static_cast<float>(y0);
    float wz = z - static_cast<float>(z0);

    float s000 = read_strength(strength_volume, depth, height, width, x0, y0, z0);
    float s100 = read_strength(strength_volume, depth, height, width, x1, y0, z0);
    float s010 = read_strength(strength_volume, depth, height, width, x0, y1, z0);
    float s110 = read_strength(strength_volume, depth, height, width, x1, y1, z0);
    float s001 = read_strength(strength_volume, depth, height, width, x0, y0, z1);
    float s101 = read_strength(strength_volume, depth, height, width, x1, y0, z1);
    float s011 = read_strength(strength_volume, depth, height, width, x0, y1, z1);
    float s111 = read_strength(strength_volume, depth, height, width, x1, y1, z1);

    float ds_dx = lerp(
        lerp(s100 - s000, s110 - s010, wy),
        lerp(s101 - s001, s111 - s011, wy),
        wz);
    float ds_dy = lerp(
        lerp(s010 - s000, s110 - s100, wx),
        lerp(s011 - s001, s111 - s101, wx),
        wz);
    float ds_dz = lerp(
        lerp(s001 - s000, s101 - s100, wx),
        lerp(s011 - s010, s111 - s110, wx),
        wy);

    float grad_x = grad_strength[query_index] * ds_dx;
    float grad_y = grad_strength[query_index] * ds_dy;
    float grad_z = grad_strength[query_index] * ds_dz;

    for (int c = 0; c < 3; ++c) {
        float n000 = read_normal_component(normal_volume, depth, height, width, x0, y0, z0, c);
        float n100 = read_normal_component(normal_volume, depth, height, width, x1, y0, z0, c);
        float n010 = read_normal_component(normal_volume, depth, height, width, x0, y1, z0, c);
        float n110 = read_normal_component(normal_volume, depth, height, width, x1, y1, z0, c);
        float n001 = read_normal_component(normal_volume, depth, height, width, x0, y0, z1, c);
        float n101 = read_normal_component(normal_volume, depth, height, width, x1, y0, z1, c);
        float n011 = read_normal_component(normal_volume, depth, height, width, x0, y1, z1, c);
        float n111 = read_normal_component(normal_volume, depth, height, width, x1, y1, z1, c);

        float dn_dx = lerp(
            lerp(n100 - n000, n110 - n010, wy),
            lerp(n101 - n001, n111 - n011, wy),
            wz);
        float dn_dy = lerp(
            lerp(n010 - n000, n110 - n100, wx),
            lerp(n011 - n001, n111 - n101, wx),
            wz);
        float dn_dz = lerp(
            lerp(n001 - n000, n101 - n100, wx),
            lerp(n011 - n010, n111 - n110, wx),
            wy);

        float upstream = grad_normals[query_index * 3 + c];
        grad_x += upstream * dn_dx;
        grad_y += upstream * dn_dy;
        grad_z += upstream * dn_dz;
    }

    grad_points[query_index * 3 + 0] = grad_x / spacing_x;
    grad_points[query_index * 3 + 1] = grad_y / spacing_y;
    grad_points[query_index * 3 + 2] = grad_z / spacing_z;
}

}  // namespace

std::vector<at::Tensor> sample_boundary_field_forward_cuda(
    at::Tensor strength_volume,
    at::Tensor normal_volume,
    at::Tensor query_points,
    double spacing_z,
    double spacing_y,
    double spacing_x) {
    check_cuda(strength_volume, "strength_volume");
    check_cuda(normal_volume, "normal_volume");
    check_cuda(query_points, "query_points");
    check_float(strength_volume, "strength_volume");
    check_float(normal_volume, "normal_volume");
    check_float(query_points, "query_points");
    TORCH_CHECK(strength_volume.dim() == 3, "strength_volume must have shape (D, H, W).");
    TORCH_CHECK(normal_volume.dim() == 4 && normal_volume.size(3) == 3, "normal_volume must have shape (D, H, W, 3).");
    TORCH_CHECK(query_points.dim() == 2 && query_points.size(1) == 3, "query_points must have shape (Q, 3).");
    TORCH_CHECK(strength_volume.size(0) == normal_volume.size(0) && strength_volume.size(1) == normal_volume.size(1) && strength_volume.size(2) == normal_volume.size(2),
        "strength_volume and normal_volume spatial dimensions must match.");

    c10::cuda::CUDAGuard device_guard(query_points.device());
    auto sampled_strength = at::zeros({query_points.size(0)}, query_points.options());
    auto sampled_normals = at::zeros({query_points.size(0), 3}, query_points.options());
    if (query_points.numel() == 0) {
        return {sampled_strength, sampled_normals};
    }

    constexpr int kThreads = 256;
    int blocks = static_cast<int>((query_points.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    sample_boundary_field_forward_kernel<<<blocks, kThreads, 0, stream>>>(
        strength_volume.contiguous().data_ptr<float>(),
        normal_volume.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        query_points.size(0),
        strength_volume.size(0),
        strength_volume.size(1),
        strength_volume.size(2),
        static_cast<float>(spacing_z),
        static_cast<float>(spacing_y),
        static_cast<float>(spacing_x),
        sampled_strength.data_ptr<float>(),
        sampled_normals.data_ptr<float>());

    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "sample_boundary_field_forward kernel launch failed: ", cudaGetErrorString(error));
    return {sampled_strength, sampled_normals};
}

at::Tensor sample_boundary_field_backward_cuda(
    at::Tensor strength_volume,
    at::Tensor normal_volume,
    at::Tensor query_points,
    at::Tensor grad_strength,
    at::Tensor grad_normals,
    double spacing_z,
    double spacing_y,
    double spacing_x) {
    check_cuda(strength_volume, "strength_volume");
    check_cuda(normal_volume, "normal_volume");
    check_cuda(query_points, "query_points");
    check_cuda(grad_strength, "grad_strength");
    check_cuda(grad_normals, "grad_normals");
    check_float(strength_volume, "strength_volume");
    check_float(normal_volume, "normal_volume");
    check_float(query_points, "query_points");
    check_float(grad_strength, "grad_strength");
    check_float(grad_normals, "grad_normals");
    TORCH_CHECK(query_points.dim() == 2 && query_points.size(1) == 3, "query_points must have shape (Q, 3).");
    TORCH_CHECK(grad_strength.dim() == 1 && grad_strength.size(0) == query_points.size(0), "grad_strength must have shape (Q,).");
    TORCH_CHECK(grad_normals.dim() == 2 && grad_normals.size(0) == query_points.size(0) && grad_normals.size(1) == 3, "grad_normals must have shape (Q, 3).");

    c10::cuda::CUDAGuard device_guard(query_points.device());
    auto grad_points = at::zeros_like(query_points);
    if (query_points.numel() == 0) {
        return grad_points;
    }

    constexpr int kThreads = 256;
    int blocks = static_cast<int>((query_points.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    sample_boundary_field_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        strength_volume.contiguous().data_ptr<float>(),
        normal_volume.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        grad_strength.contiguous().data_ptr<float>(),
        grad_normals.contiguous().data_ptr<float>(),
        query_points.size(0),
        strength_volume.size(0),
        strength_volume.size(1),
        strength_volume.size(2),
        static_cast<float>(spacing_z),
        static_cast<float>(spacing_y),
        static_cast<float>(spacing_x),
        grad_points.data_ptr<float>());

    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "sample_boundary_field_backward kernel launch failed: ", cudaGetErrorString(error));
    return grad_points;
}
