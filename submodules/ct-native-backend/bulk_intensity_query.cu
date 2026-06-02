#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreads = 256;

void check_cuda(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
}

void check_float(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32.");
}

void check_int32(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32.");
}

void check_int64(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64.");
}

__device__ __forceinline__ int64_t flatten_cell(int x, int y, int z, int dim_x, int dim_y) {
    return static_cast<int64_t>(x) + static_cast<int64_t>(dim_x) * (
        static_cast<int64_t>(y) + static_cast<int64_t>(dim_y) * static_cast<int64_t>(z)
    );
}

__device__ __forceinline__ bool world_to_cell(
    const float* point,
    const float* world_min,
    float cell_size,
    int dim_x,
    int dim_y,
    int dim_z,
    int* cell_x,
    int* cell_y,
    int* cell_z) {
    int x = static_cast<int>(floorf((point[0] - world_min[0]) / cell_size));
    int y = static_cast<int>(floorf((point[1] - world_min[1]) / cell_size));
    int z = static_cast<int>(floorf((point[2] - world_min[2]) / cell_size));
    if (x < 0 || x >= dim_x || y < 0 || y >= dim_y || z < 0 || z >= dim_z) {
        return false;
    }
    *cell_x = x;
    *cell_y = y;
    *cell_z = z;
    return true;
}

__device__ __forceinline__ float gaussian_q_value(
    const float* mean,
    const float* rotation,
    const float* scale,
    const float* query,
    float* dx_out,
    float* dy_out,
    float* dz_out,
    float* local_x_out,
    float* local_y_out,
    float* local_z_out) {
    float dx = query[0] - mean[0];
    float dy = query[1] - mean[1];
    float dz = query[2] - mean[2];

    float local_x = dx * rotation[0] + dy * rotation[3] + dz * rotation[6];
    float local_y = dx * rotation[1] + dy * rotation[4] + dz * rotation[7];
    float local_z = dx * rotation[2] + dy * rotation[5] + dz * rotation[8];

    float nx = local_x / scale[0];
    float ny = local_y / scale[1];
    float nz = local_z / scale[2];

    if (dx_out != nullptr) {
        *dx_out = dx;
        *dy_out = dy;
        *dz_out = dz;
        *local_x_out = local_x;
        *local_y_out = local_y;
        *local_z_out = local_z;
    }
    return nx * nx + ny * ny + nz * nz;
}

__global__ void count_qcut_neighbors_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* support_extent,
    const float* query_points,
    const float* world_min,
    const int32_t* grid_dims,
    float cell_size,
    const int64_t* cell_offsets,
    const int32_t* cell_gaussian_ids,
    int64_t num_queries,
    float q_cut,
    int64_t* counts) {
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= num_queries) {
        return;
    }

    const float* query = query_points + query_index * 3;
    int cell_x = 0;
    int cell_y = 0;
    int cell_z = 0;
    if (!world_to_cell(query, world_min, cell_size, grid_dims[0], grid_dims[1], grid_dims[2], &cell_x, &cell_y, &cell_z)) {
        counts[query_index] = 0;
        return;
    }

    int64_t flat = flatten_cell(cell_x, cell_y, cell_z, grid_dims[0], grid_dims[1]);
    int64_t start = cell_offsets[flat];
    int64_t end = cell_offsets[flat + 1];
    int64_t count = 0;
    for (int64_t cursor = start; cursor < end; ++cursor) {
        int32_t gaussian_idx = cell_gaussian_ids[cursor];
        const float* mean = means + gaussian_idx * 3;
        const float* extent = support_extent + gaussian_idx * 3;
        if (fabsf(query[0] - mean[0]) > extent[0] || fabsf(query[1] - mean[1]) > extent[1] || fabsf(query[2] - mean[2]) > extent[2]) {
            continue;
        }
        const float* rotation = rotations + gaussian_idx * 9;
        const float* scale = scales + gaussian_idx * 3;
        float q = gaussian_q_value(mean, rotation, scale, query, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (q <= q_cut) {
            ++count;
        }
    }
    counts[query_index] = count;
}

__global__ void fill_qcut_density_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const float* support_extent,
    const float* query_points,
    const float* world_min,
    const int32_t* grid_dims,
    float cell_size,
    const int64_t* cell_offsets,
    const int32_t* cell_gaussian_ids,
    const int64_t* query_offsets,
    int64_t num_queries,
    float q_cut,
    int32_t* query_gaussian_ids,
    float* density) {
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= num_queries) {
        return;
    }

    const float* query = query_points + query_index * 3;
    int cell_x = 0;
    int cell_y = 0;
    int cell_z = 0;
    if (!world_to_cell(query, world_min, cell_size, grid_dims[0], grid_dims[1], grid_dims[2], &cell_x, &cell_y, &cell_z)) {
        density[query_index] = 0.0f;
        return;
    }

    int64_t flat = flatten_cell(cell_x, cell_y, cell_z, grid_dims[0], grid_dims[1]);
    int64_t start = cell_offsets[flat];
    int64_t end = cell_offsets[flat + 1];
    int64_t write_offset = query_offsets[query_index];
    float accum = 0.0f;

    for (int64_t cursor = start; cursor < end; ++cursor) {
        int32_t gaussian_idx = cell_gaussian_ids[cursor];
        const float* mean = means + gaussian_idx * 3;
        const float* extent = support_extent + gaussian_idx * 3;
        if (fabsf(query[0] - mean[0]) > extent[0] || fabsf(query[1] - mean[1]) > extent[1] || fabsf(query[2] - mean[2]) > extent[2]) {
            continue;
        }
        const float* rotation = rotations + gaussian_idx * 9;
        const float* scale = scales + gaussian_idx * 3;
        float q = gaussian_q_value(mean, rotation, scale, query, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (q > q_cut) {
            continue;
        }
        accum += expf(-0.5f * q) * opacity[gaussian_idx];
        query_gaussian_ids[write_offset++] = gaussian_idx;
    }
    density[query_index] = accum;
}

__global__ void qcut_density_backward_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const float* query_points,
    const int64_t* query_offsets,
    const int32_t* query_gaussian_ids,
    const float* grad_output,
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

    const float* query = query_points + query_index * 3;
    int64_t start = query_offsets[query_index];
    int64_t end = query_offsets[query_index + 1];
    for (int64_t cursor = start; cursor < end; ++cursor) {
        int32_t gaussian_idx = query_gaussian_ids[cursor];
        const float* mean = means + gaussian_idx * 3;
        const float* rotation = rotations + gaussian_idx * 9;
        const float* scale = scales + gaussian_idx * 3;

        float dx, dy, dz, local_x, local_y, local_z;
        float q = gaussian_q_value(mean, rotation, scale, query, &dx, &dy, &dz, &local_x, &local_y, &local_z);
        float exp_value = expf(-0.5f * q);
        float grad_opacity_value = upstream * exp_value;
        float common = grad_opacity_value * opacity[gaussian_idx];

        float inv_sx = 1.0f / scale[0];
        float inv_sy = 1.0f / scale[1];
        float inv_sz = 1.0f / scale[2];
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

        atomicAdd(grad_scales + gaussian_idx * 3 + 0, common * local_x * local_x * inv_sx2 * inv_sx);
        atomicAdd(grad_scales + gaussian_idx * 3 + 1, common * local_y * local_y * inv_sy2 * inv_sy);
        atomicAdd(grad_scales + gaussian_idx * 3 + 2, common * local_z * local_z * inv_sz2 * inv_sz);
        atomicAdd(grad_opacity + gaussian_idx, grad_opacity_value);
    }
}

__device__ __forceinline__ float bulk_gate_value(
    bool apply_gate,
    bool has_membership,
    const float* material_membership,
    int query_index,
    const float* center_sdf,
    const float* center_normals,
    int gaussian_idx,
    float dx,
    float dy,
    float dz,
    float tau,
    float skip_depth,
    float* dgate_dlinear_out) {
    *dgate_dlinear_out = 0.0f;
    if (!apply_gate) {
        return 1.0f;
    }
    if (has_membership) {
        float m = material_membership[query_index];
        return fminf(fmaxf(m, 0.0f), 1.0f);
    }
    float sdf = center_sdf[gaussian_idx];
    if (sdf <= -skip_depth) {
        return 1.0f;
    }
    const float* normal = center_normals + gaussian_idx * 3;
    float linearized = sdf + dx * normal[0] + dy * normal[1] + dz * normal[2];
    float gate = 1.0f / (1.0f + expf(linearized / fmaxf(tau, 1e-6f)));
    *dgate_dlinear_out = -(gate * (1.0f - gate)) / fmaxf(tau, 1e-6f);
    return gate;
}

__global__ void fill_bulk_intensity_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const float* attenuation,
    const float* center_sdf,
    const float* center_normals,
    const float* material_membership,
    const float* support_extent,
    const float* query_points,
    const float* world_min,
    const int32_t* grid_dims,
    float cell_size,
    const int64_t* cell_offsets,
    const int32_t* cell_gaussian_ids,
    const int64_t* query_offsets,
    int64_t num_queries,
    float q_cut,
    float tau,
    float skip_depth,
    bool apply_gate,
    bool has_membership,
    int32_t* query_gaussian_ids,
    float* raw_bulk,
    float* density) {
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= num_queries) {
        return;
    }

    const float* query = query_points + query_index * 3;
    int cell_x = 0;
    int cell_y = 0;
    int cell_z = 0;
    if (!world_to_cell(query, world_min, cell_size, grid_dims[0], grid_dims[1], grid_dims[2], &cell_x, &cell_y, &cell_z)) {
        raw_bulk[query_index] = 0.0f;
        density[query_index] = 0.0f;
        return;
    }

    int64_t flat = flatten_cell(cell_x, cell_y, cell_z, grid_dims[0], grid_dims[1]);
    int64_t start = cell_offsets[flat];
    int64_t end = cell_offsets[flat + 1];
    int64_t write_offset = query_offsets[query_index];
    float raw_accum = 0.0f;
    float den_accum = 0.0f;

    for (int64_t cursor = start; cursor < end; ++cursor) {
        int32_t gaussian_idx = cell_gaussian_ids[cursor];
        const float* mean = means + gaussian_idx * 3;
        const float* extent = support_extent + gaussian_idx * 3;
        if (fabsf(query[0] - mean[0]) > extent[0] || fabsf(query[1] - mean[1]) > extent[1] || fabsf(query[2] - mean[2]) > extent[2]) {
            continue;
        }
        const float* rotation = rotations + gaussian_idx * 9;
        const float* scale = scales + gaussian_idx * 3;
        float dx, dy, dz, local_x, local_y, local_z;
        float q = gaussian_q_value(mean, rotation, scale, query, &dx, &dy, &dz, &local_x, &local_y, &local_z);
        if (q > q_cut) {
            continue;
        }
        float dgate_dlinear = 0.0f;
        float gate = bulk_gate_value(
            apply_gate,
            has_membership,
            material_membership,
            query_index,
            center_sdf,
            center_normals,
            gaussian_idx,
            dx,
            dy,
            dz,
            tau,
            skip_depth,
            &dgate_dlinear);
        float kernel = expf(-0.5f * q) * gate * opacity[gaussian_idx];
        raw_accum += kernel * attenuation[gaussian_idx];
        den_accum += kernel;
        query_gaussian_ids[write_offset++] = gaussian_idx;
    }
    raw_bulk[query_index] = raw_accum;
    density[query_index] = den_accum;
}

__global__ void bulk_intensity_backward_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const float* attenuation,
    const float* center_sdf,
    const float* center_normals,
    const float* material_membership,
    const float* query_points,
    const int64_t* query_offsets,
    const int32_t* query_gaussian_ids,
    const float* grad_raw,
    const float* grad_den,
    int64_t num_queries,
    float q_cut,
    float tau,
    float skip_depth,
    bool apply_gate,
    bool has_membership,
    float* grad_means,
    float* grad_rotations,
    float* grad_scales,
    float* grad_opacity,
    float* grad_attenuation) {
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= num_queries) {
        return;
    }

    float upstream_raw = grad_raw[query_index];
    float upstream_den = grad_den[query_index];
    if (upstream_raw == 0.0f && upstream_den == 0.0f) {
        return;
    }

    const float* query = query_points + query_index * 3;
    int64_t start = query_offsets[query_index];
    int64_t end = query_offsets[query_index + 1];
    for (int64_t cursor = start; cursor < end; ++cursor) {
        int32_t gaussian_idx = query_gaussian_ids[cursor];
        const float* mean = means + gaussian_idx * 3;
        const float* rotation = rotations + gaussian_idx * 9;
        const float* scale = scales + gaussian_idx * 3;

        float dx, dy, dz, local_x, local_y, local_z;
        float q = gaussian_q_value(mean, rotation, scale, query, &dx, &dy, &dz, &local_x, &local_y, &local_z);
        if (q > q_cut) {
            continue;
        }

        float dgate_dlinear = 0.0f;
        float gate = bulk_gate_value(
            apply_gate,
            has_membership,
            material_membership,
            query_index,
            center_sdf,
            center_normals,
            gaussian_idx,
            dx,
            dy,
            dz,
            tau,
            skip_depth,
            &dgate_dlinear);
        float exp_value = expf(-0.5f * q);
        float alpha = opacity[gaussian_idx];
        float atten = attenuation[gaussian_idx];
        float base = exp_value * gate * alpha;
        float upstream_base = upstream_den + upstream_raw * atten;

        atomicAdd(grad_attenuation + gaussian_idx, upstream_raw * base);
        atomicAdd(grad_opacity + gaussian_idx, upstream_base * exp_value * gate);

        float common = upstream_base * alpha * gate * exp_value;
        float inv_sx = 1.0f / scale[0];
        float inv_sy = 1.0f / scale[1];
        float inv_sz = 1.0f / scale[2];
        float inv_sx2 = inv_sx * inv_sx;
        float inv_sy2 = inv_sy * inv_sy;
        float inv_sz2 = inv_sz * inv_sz;
        float local_grad_x = common * (-local_x * inv_sx2);
        float local_grad_y = common * (-local_y * inv_sy2);
        float local_grad_z = common * (-local_z * inv_sz2);

        float gate_diff_x = 0.0f;
        float gate_diff_y = 0.0f;
        float gate_diff_z = 0.0f;
        if (apply_gate && !has_membership && dgate_dlinear != 0.0f) {
            const float* normal = center_normals + gaussian_idx * 3;
            float gate_upstream = upstream_base * alpha * exp_value;
            gate_diff_x = gate_upstream * dgate_dlinear * normal[0];
            gate_diff_y = gate_upstream * dgate_dlinear * normal[1];
            gate_diff_z = gate_upstream * dgate_dlinear * normal[2];
        }

        float grad_mean_x = -(local_grad_x * rotation[0] + local_grad_y * rotation[1] + local_grad_z * rotation[2] + gate_diff_x);
        float grad_mean_y = -(local_grad_x * rotation[3] + local_grad_y * rotation[4] + local_grad_z * rotation[5] + gate_diff_y);
        float grad_mean_z = -(local_grad_x * rotation[6] + local_grad_y * rotation[7] + local_grad_z * rotation[8] + gate_diff_z);

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

        atomicAdd(grad_scales + gaussian_idx * 3 + 0, common * local_x * local_x * inv_sx2 * inv_sx);
        atomicAdd(grad_scales + gaussian_idx * 3 + 1, common * local_y * local_y * inv_sy2 * inv_sy);
        atomicAdd(grad_scales + gaussian_idx * 3 + 2, common * local_z * local_z * inv_sz2 * inv_sz);
    }
}

std::vector<at::Tensor> run_qcut_forward(
    at::Tensor means,
    at::Tensor rotations,
    at::Tensor scales,
    at::Tensor support_extent,
    at::Tensor query_points,
    at::Tensor grid_world_min,
    at::Tensor grid_dims,
    double cell_size,
    at::Tensor cell_offsets,
    at::Tensor cell_gaussian_ids,
    double q_cut,
    bool allocate_density,
    at::Tensor opacity) {
    c10::cuda::CUDAGuard device_guard(means.device());
    auto options_float = means.options();
    auto options_long = at::TensorOptions().device(means.device()).dtype(at::kLong);
    auto options_int = at::TensorOptions().device(means.device()).dtype(at::kInt);

    auto density = at::zeros({query_points.size(0)}, options_float);
    auto query_offsets = at::zeros({query_points.size(0) + 1}, options_long);
    if (means.numel() == 0 || query_points.numel() == 0) {
        return {density, query_offsets, at::empty({0}, options_int)};
    }

    auto counts = at::zeros({query_points.size(0)}, options_long);
    int blocks = static_cast<int>((query_points.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    count_qcut_neighbors_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        support_extent.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        grid_world_min.contiguous().data_ptr<float>(),
        grid_dims.contiguous().data_ptr<int32_t>(),
        static_cast<float>(cell_size),
        cell_offsets.contiguous().data_ptr<int64_t>(),
        cell_gaussian_ids.contiguous().data_ptr<int32_t>(),
        query_points.size(0),
        static_cast<float>(q_cut),
        counts.data_ptr<int64_t>());
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "count_qcut_neighbors kernel launch failed: ", cudaGetErrorString(error));

    query_offsets.slice(0, 1).copy_(at::cumsum(counts, 0));
    const int64_t total_neighbors = query_offsets[-1].item<int64_t>();
    auto query_gaussian_ids = at::empty({total_neighbors}, options_int);
    if (total_neighbors == 0 || !allocate_density) {
        return {density, query_offsets, query_gaussian_ids};
    }

    fill_qcut_density_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        support_extent.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        grid_world_min.contiguous().data_ptr<float>(),
        grid_dims.contiguous().data_ptr<int32_t>(),
        static_cast<float>(cell_size),
        cell_offsets.contiguous().data_ptr<int64_t>(),
        cell_gaussian_ids.contiguous().data_ptr<int32_t>(),
        query_offsets.contiguous().data_ptr<int64_t>(),
        query_points.size(0),
        static_cast<float>(q_cut),
        query_gaussian_ids.data_ptr<int32_t>(),
        density.data_ptr<float>());
    error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "fill_qcut_density kernel launch failed: ", cudaGetErrorString(error));
    return {density, query_offsets, query_gaussian_ids};
}

}  // namespace

std::vector<at::Tensor> query_density_qcut_local_forward_cuda(
    at::Tensor means,
    at::Tensor rotations,
    at::Tensor scales,
    at::Tensor opacity,
    at::Tensor support_extent,
    at::Tensor query_points,
    at::Tensor grid_world_min,
    at::Tensor grid_dims,
    double cell_size,
    at::Tensor cell_offsets,
    at::Tensor cell_gaussian_ids,
    double q_cut) {
    check_cuda(means, "means");
    check_cuda(rotations, "rotations");
    check_cuda(scales, "scales");
    check_cuda(opacity, "opacity");
    check_cuda(support_extent, "support_extent");
    check_cuda(query_points, "query_points");
    check_cuda(grid_world_min, "grid_world_min");
    check_cuda(grid_dims, "grid_dims");
    check_cuda(cell_offsets, "cell_offsets");
    check_cuda(cell_gaussian_ids, "cell_gaussian_ids");
    check_float(means, "means");
    check_float(rotations, "rotations");
    check_float(scales, "scales");
    check_float(opacity, "opacity");
    check_float(support_extent, "support_extent");
    check_float(query_points, "query_points");
    check_float(grid_world_min, "grid_world_min");
    check_int32(grid_dims, "grid_dims");
    check_int64(cell_offsets, "cell_offsets");
    check_int32(cell_gaussian_ids, "cell_gaussian_ids");

    return run_qcut_forward(
        means,
        rotations,
        scales,
        support_extent,
        query_points,
        grid_world_min,
        grid_dims,
        cell_size,
        cell_offsets,
        cell_gaussian_ids,
        q_cut,
        true,
        opacity);
}

std::vector<at::Tensor> query_density_qcut_local_backward_cuda(
    at::Tensor means,
    at::Tensor rotations,
    at::Tensor scales,
    at::Tensor opacity,
    at::Tensor query_points,
    at::Tensor query_offsets,
    at::Tensor query_gaussian_ids,
    at::Tensor grad_output) {
    check_cuda(means, "means");
    check_cuda(rotations, "rotations");
    check_cuda(scales, "scales");
    check_cuda(opacity, "opacity");
    check_cuda(query_points, "query_points");
    check_cuda(query_offsets, "query_offsets");
    check_cuda(query_gaussian_ids, "query_gaussian_ids");
    check_cuda(grad_output, "grad_output");
    check_float(means, "means");
    check_float(rotations, "rotations");
    check_float(scales, "scales");
    check_float(opacity, "opacity");
    check_float(query_points, "query_points");
    check_float(grad_output, "grad_output");
    check_int64(query_offsets, "query_offsets");
    check_int32(query_gaussian_ids, "query_gaussian_ids");

    c10::cuda::CUDAGuard device_guard(means.device());
    auto grad_means = at::zeros_like(means);
    auto grad_rotations = at::zeros_like(rotations);
    auto grad_scales = at::zeros_like(scales);
    auto grad_opacity = at::zeros_like(opacity);
    if (means.numel() == 0 || query_points.numel() == 0 || query_gaussian_ids.numel() == 0) {
        return {grad_means, grad_rotations, grad_scales, grad_opacity};
    }

    int blocks = static_cast<int>((query_points.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    qcut_density_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        query_offsets.contiguous().data_ptr<int64_t>(),
        query_gaussian_ids.contiguous().data_ptr<int32_t>(),
        grad_output.contiguous().data_ptr<float>(),
        query_points.size(0),
        grad_means.data_ptr<float>(),
        grad_rotations.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_opacity.data_ptr<float>());
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "qcut_density_backward kernel launch failed: ", cudaGetErrorString(error));
    return {grad_means, grad_rotations, grad_scales, grad_opacity};
}

std::vector<at::Tensor> query_bulk_intensity_local_forward_cuda(
    at::Tensor means,
    at::Tensor rotations,
    at::Tensor scales,
    at::Tensor opacity,
    at::Tensor attenuation,
    at::Tensor center_sdf,
    at::Tensor center_normals,
    at::Tensor material_membership,
    at::Tensor support_extent,
    at::Tensor query_points,
    at::Tensor grid_world_min,
    at::Tensor grid_dims,
    double cell_size,
    at::Tensor cell_offsets,
    at::Tensor cell_gaussian_ids,
    double q_cut,
    double tau,
    double skip_depth,
    bool apply_gate,
    bool has_membership) {
    check_cuda(means, "means");
    check_cuda(rotations, "rotations");
    check_cuda(scales, "scales");
    check_cuda(opacity, "opacity");
    check_cuda(attenuation, "attenuation");
    check_cuda(center_sdf, "center_sdf");
    check_cuda(center_normals, "center_normals");
    check_cuda(material_membership, "material_membership");
    check_cuda(support_extent, "support_extent");
    check_cuda(query_points, "query_points");
    check_cuda(grid_world_min, "grid_world_min");
    check_cuda(grid_dims, "grid_dims");
    check_cuda(cell_offsets, "cell_offsets");
    check_cuda(cell_gaussian_ids, "cell_gaussian_ids");
    check_float(means, "means");
    check_float(rotations, "rotations");
    check_float(scales, "scales");
    check_float(opacity, "opacity");
    check_float(attenuation, "attenuation");
    check_float(center_sdf, "center_sdf");
    check_float(center_normals, "center_normals");
    check_float(material_membership, "material_membership");
    check_float(support_extent, "support_extent");
    check_float(query_points, "query_points");
    check_float(grid_world_min, "grid_world_min");
    check_int32(grid_dims, "grid_dims");
    check_int64(cell_offsets, "cell_offsets");
    check_int32(cell_gaussian_ids, "cell_gaussian_ids");

    c10::cuda::CUDAGuard device_guard(means.device());
    auto options_float = means.options();
    auto empty_opacity = at::ones({means.size(0)}, options_float);
    auto neighbor_data = run_qcut_forward(
        means,
        rotations,
        scales,
        support_extent,
        query_points,
        grid_world_min,
        grid_dims,
        cell_size,
        cell_offsets,
        cell_gaussian_ids,
        q_cut,
        false,
        empty_opacity);
    auto raw_bulk = at::zeros({query_points.size(0)}, options_float);
    auto density = at::zeros({query_points.size(0)}, options_float);
    auto query_offsets = neighbor_data[1];
    auto query_gaussian_ids = neighbor_data[2];
    if (query_gaussian_ids.numel() == 0) {
        return {raw_bulk, density, query_offsets, query_gaussian_ids};
    }

    int blocks = static_cast<int>((query_points.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    fill_bulk_intensity_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        attenuation.contiguous().data_ptr<float>(),
        center_sdf.contiguous().data_ptr<float>(),
        center_normals.contiguous().data_ptr<float>(),
        material_membership.contiguous().data_ptr<float>(),
        support_extent.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        grid_world_min.contiguous().data_ptr<float>(),
        grid_dims.contiguous().data_ptr<int32_t>(),
        static_cast<float>(cell_size),
        cell_offsets.contiguous().data_ptr<int64_t>(),
        cell_gaussian_ids.contiguous().data_ptr<int32_t>(),
        query_offsets.contiguous().data_ptr<int64_t>(),
        query_points.size(0),
        static_cast<float>(q_cut),
        static_cast<float>(tau),
        static_cast<float>(skip_depth),
        bool(apply_gate),
        bool(has_membership),
        query_gaussian_ids.data_ptr<int32_t>(),
        raw_bulk.data_ptr<float>(),
        density.data_ptr<float>());
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "fill_bulk_intensity kernel launch failed: ", cudaGetErrorString(error));
    return {raw_bulk, density, query_offsets, query_gaussian_ids};
}

std::vector<at::Tensor> query_bulk_intensity_local_backward_cuda(
    at::Tensor means,
    at::Tensor rotations,
    at::Tensor scales,
    at::Tensor opacity,
    at::Tensor attenuation,
    at::Tensor center_sdf,
    at::Tensor center_normals,
    at::Tensor material_membership,
    at::Tensor query_points,
    at::Tensor query_offsets,
    at::Tensor query_gaussian_ids,
    at::Tensor grad_raw,
    at::Tensor grad_den,
    double q_cut,
    double tau,
    double skip_depth,
    bool apply_gate,
    bool has_membership) {
    check_cuda(means, "means");
    check_cuda(rotations, "rotations");
    check_cuda(scales, "scales");
    check_cuda(opacity, "opacity");
    check_cuda(attenuation, "attenuation");
    check_cuda(center_sdf, "center_sdf");
    check_cuda(center_normals, "center_normals");
    check_cuda(material_membership, "material_membership");
    check_cuda(query_points, "query_points");
    check_cuda(query_offsets, "query_offsets");
    check_cuda(query_gaussian_ids, "query_gaussian_ids");
    check_cuda(grad_raw, "grad_raw");
    check_cuda(grad_den, "grad_den");
    check_float(means, "means");
    check_float(rotations, "rotations");
    check_float(scales, "scales");
    check_float(opacity, "opacity");
    check_float(attenuation, "attenuation");
    check_float(center_sdf, "center_sdf");
    check_float(center_normals, "center_normals");
    check_float(material_membership, "material_membership");
    check_float(query_points, "query_points");
    check_float(grad_raw, "grad_raw");
    check_float(grad_den, "grad_den");
    check_int64(query_offsets, "query_offsets");
    check_int32(query_gaussian_ids, "query_gaussian_ids");

    c10::cuda::CUDAGuard device_guard(means.device());
    auto grad_means = at::zeros_like(means);
    auto grad_rotations = at::zeros_like(rotations);
    auto grad_scales = at::zeros_like(scales);
    auto grad_opacity = at::zeros_like(opacity);
    auto grad_attenuation = at::zeros_like(attenuation);
    if (means.numel() == 0 || query_points.numel() == 0 || query_gaussian_ids.numel() == 0) {
        return {grad_means, grad_rotations, grad_scales, grad_opacity, grad_attenuation};
    }

    int blocks = static_cast<int>((query_points.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    bulk_intensity_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        attenuation.contiguous().data_ptr<float>(),
        center_sdf.contiguous().data_ptr<float>(),
        center_normals.contiguous().data_ptr<float>(),
        material_membership.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        query_offsets.contiguous().data_ptr<int64_t>(),
        query_gaussian_ids.contiguous().data_ptr<int32_t>(),
        grad_raw.contiguous().data_ptr<float>(),
        grad_den.contiguous().data_ptr<float>(),
        query_points.size(0),
        static_cast<float>(q_cut),
        static_cast<float>(tau),
        static_cast<float>(skip_depth),
        bool(apply_gate),
        bool(has_membership),
        grad_means.data_ptr<float>(),
        grad_rotations.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_opacity.data_ptr<float>(),
        grad_attenuation.data_ptr<float>());
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "bulk_intensity_backward kernel launch failed: ", cudaGetErrorString(error));
    return {grad_means, grad_rotations, grad_scales, grad_opacity, grad_attenuation};
}
