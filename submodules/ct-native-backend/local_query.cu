#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
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

inline int clamp_int_host(int value, int low, int high) {
    return std::max(low, std::min(value, high));
}

__device__ __forceinline__ int clamp_int(int value, int low, int high) {
    return max(low, min(value, high));
}

__device__ __forceinline__ int64_t flatten_cell(int x, int y, int z, int dim_x, int dim_y) {
    return static_cast<int64_t>(x) + static_cast<int64_t>(dim_x) * (static_cast<int64_t>(y) + static_cast<int64_t>(dim_y) * static_cast<int64_t>(z));
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

__global__ void gather_cell_entries_kernel(
    const int64_t* cell_ids,
    const int64_t* cell_offsets,
    const int32_t* cell_gaussian_ids,
    const int64_t* write_offsets,
    int64_t num_cells,
    int32_t* output_gaussian_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) {
        return;
    }
    int64_t cell_id = cell_ids[idx];
    int64_t start = cell_offsets[cell_id];
    int64_t end = cell_offsets[cell_id + 1];
    int64_t write_offset = write_offsets[idx];
    for (int64_t cursor = start; cursor < end; ++cursor) {
        output_gaussian_ids[write_offset + (cursor - start)] = cell_gaussian_ids[cursor];
    }
}

__global__ void count_patch_active_kernel(
    const float* means,
    const float* support_extent,
    const int32_t* candidate_ids,
    int64_t num_candidates,
    int plane_axis,
    int dim_h,
    int dim_w,
    float plane_coord,
    float patch_min_h,
    float patch_max_h,
    float patch_min_w,
    float patch_max_w,
    int64_t* counts);

__global__ void fill_patch_active_kernel(
    const int32_t* candidate_ids,
    const int64_t* counts_prefix,
    const int64_t* counts,
    int64_t num_candidates,
    int32_t* active_ids);

__global__ void render_slice_patch_local_forward_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const int64_t* tile_offsets,
    const int32_t* tile_gaussian_ids,
    int axis_index,
    int slice_idx,
    int origin_h,
    int origin_w,
    int patch_h,
    int patch_w,
    float spacing_z,
    float spacing_y,
    float spacing_x,
    int tile_size,
    int tile_w_count,
    float* output);

__global__ void render_slice_patch_local_backward_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const int64_t* tile_offsets,
    const int32_t* tile_gaussian_ids,
    const float* grad_output,
    int axis_index,
    int slice_idx,
    int origin_h,
    int origin_w,
    int patch_h,
    int patch_w,
    float spacing_z,
    float spacing_y,
    float spacing_x,
    int tile_size,
    int tile_w_count,
    float* grad_means,
    float* grad_rotations,
    float* grad_scales,
    float* grad_opacity);

std::vector<at::Tensor> build_slice_tile_lists(
    at::Tensor means,
    at::Tensor support_extent,
    at::Tensor active_ids,
    int dim_h,
    int dim_w,
    float spacing_h,
    float spacing_w,
    int origin_h,
    int origin_w,
    int patch_h,
    int patch_w,
    int tile_size);

at::Tensor unique_sorted_int32(at::Tensor sorted_ids) {
    TORCH_CHECK(sorted_ids.dim() == 1, "sorted_ids must be a 1D tensor.");
    if (sorted_ids.numel() == 0) {
        return sorted_ids;
    }

    auto keep = at::zeros({sorted_ids.size(0)}, sorted_ids.options().dtype(at::kBool));
    keep[0] = true;
    if (sorted_ids.size(0) > 1) {
        auto current = sorted_ids.slice(0, 1);
        auto previous = sorted_ids.slice(0, 0, sorted_ids.size(0) - 1);
        keep.slice(0, 1).copy_(current != previous);
    }
    return sorted_ids.masked_select(keep);
}

at::Tensor gather_candidate_ids(
    const at::Tensor& cell_ids,
    const at::Tensor& cell_offsets,
    const at::Tensor& cell_gaussian_ids) {
    auto options_long = at::TensorOptions().device(cell_ids.device()).dtype(at::kLong);
    auto options_int = at::TensorOptions().device(cell_ids.device()).dtype(at::kInt);

    if (cell_ids.numel() == 0) {
        return at::empty({0}, options_int);
    }

    auto start = cell_offsets.index_select(0, cell_ids);
    auto end = cell_offsets.index_select(0, cell_ids + 1);
    auto lengths = end - start;
    auto write_offsets = at::zeros({cell_ids.size(0) + 1}, options_long);
    write_offsets.slice(0, 1).copy_(at::cumsum(lengths, 0));
    const int64_t total = write_offsets[-1].item<int64_t>();
    if (total == 0) {
        return at::empty({0}, options_int);
    }

    auto output = at::empty({total}, options_int);
    int blocks = static_cast<int>((cell_ids.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    gather_cell_entries_kernel<<<blocks, kThreads, 0, stream>>>(
        cell_ids.contiguous().data_ptr<int64_t>(),
        cell_offsets.contiguous().data_ptr<int64_t>(),
        cell_gaussian_ids.contiguous().data_ptr<int32_t>(),
        write_offsets.contiguous().data_ptr<int64_t>(),
        cell_ids.size(0),
        output.data_ptr<int32_t>());
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "gather_cell_entries kernel launch failed: ", cudaGetErrorString(error));

    auto sort_result = at::sort(output, 0, false);
    return unique_sorted_int32(std::get<0>(sort_result));
}

__global__ void count_local_neighbors_kernel(
    const float* means,
    const float* support_extent,
    const float* query_points,
    const float* world_min,
    const int32_t* grid_dims,
    float cell_size,
    const int64_t* cell_offsets,
    const int32_t* cell_gaussian_ids,
    int64_t num_queries,
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
        ++count;
    }
    counts[query_index] = count;
}

__global__ void fill_local_neighbors_and_density_kernel(
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
        float dx = query[0] - mean[0];
        float dy = query[1] - mean[1];
        float dz = query[2] - mean[2];
        float local_x = dx * rotation[0] + dy * rotation[3] + dz * rotation[6];
        float local_y = dx * rotation[1] + dy * rotation[4] + dz * rotation[7];
        float local_z = dx * rotation[2] + dy * rotation[5] + dz * rotation[8];
        float nx = local_x / scale[0];
        float ny = local_y / scale[1];
        float nz = local_z / scale[2];
        float exponent = -0.5f * (nx * nx + ny * ny + nz * nz);
        accum += expf(exponent) * opacity[gaussian_idx];
        query_gaussian_ids[write_offset++] = gaussian_idx;
    }

    density[query_index] = accum;
}

__global__ void query_density_local_backward_kernel(
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

        float dx = query[0] - mean[0];
        float dy = query[1] - mean[1];
        float dz = query[2] - mean[2];

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

std::vector<at::Tensor> render_slice_patch_local_forward_cuda(
    at::Tensor means,
    at::Tensor rotations,
    at::Tensor scales,
    at::Tensor opacity,
    at::Tensor support_extent,
    at::Tensor grid_world_min,
    at::Tensor grid_dims,
    double cell_size,
    at::Tensor cell_offsets,
    at::Tensor cell_gaussian_ids,
    int64_t axis_index,
    int64_t slice_idx,
    int64_t origin_h,
    int64_t origin_w,
    int64_t patch_h,
    int64_t patch_w,
    double spacing_z,
    double spacing_y,
    double spacing_x,
    int64_t tile_size) {
    check_cuda(means, "means");
    check_cuda(rotations, "rotations");
    check_cuda(scales, "scales");
    check_cuda(opacity, "opacity");
    check_cuda(support_extent, "support_extent");
    check_cuda(grid_world_min, "grid_world_min");
    check_cuda(grid_dims, "grid_dims");
    check_cuda(cell_offsets, "cell_offsets");
    check_cuda(cell_gaussian_ids, "cell_gaussian_ids");
    check_float(means, "means");
    check_float(rotations, "rotations");
    check_float(scales, "scales");
    check_float(opacity, "opacity");
    check_float(support_extent, "support_extent");
    check_float(grid_world_min, "grid_world_min");
    check_int32(grid_dims, "grid_dims");
    check_int64(cell_offsets, "cell_offsets");
    check_int32(cell_gaussian_ids, "cell_gaussian_ids");

    c10::cuda::CUDAGuard device_guard(means.device());
    auto options_float = means.options();
    auto options_long = at::TensorOptions().device(means.device()).dtype(at::kLong);
    auto options_int = at::TensorOptions().device(means.device()).dtype(at::kInt);
    auto output = at::zeros({patch_h, patch_w}, options_float);
    const int tile_h_count = (static_cast<int>(patch_h) + static_cast<int>(tile_size) - 1) / static_cast<int>(tile_size);
    const int tile_w_count = (static_cast<int>(patch_w) + static_cast<int>(tile_size) - 1) / static_cast<int>(tile_size);
    auto empty_offsets = at::zeros({tile_h_count * tile_w_count + 1}, options_long);
    if (means.numel() == 0 || patch_h == 0 || patch_w == 0) {
        return {output, empty_offsets, at::empty({0}, options_int)};
    }

    int plane_axis = 0;
    int dim_h = 0;
    int dim_w = 0;
    float plane_coord = 0.0f;
    float patch_min_h = 0.0f;
    float patch_max_h = 0.0f;
    float patch_min_w = 0.0f;
    float patch_max_w = 0.0f;
    float spacing_h = 0.0f;
    float spacing_w = 0.0f;
    if (axis_index == 0) {
        plane_axis = 2;
        dim_h = 1;
        dim_w = 0;
        plane_coord = static_cast<float>(slice_idx) * static_cast<float>(spacing_z);
        patch_min_h = static_cast<float>(origin_h) * static_cast<float>(spacing_y);
        patch_max_h = static_cast<float>(origin_h + patch_h - 1) * static_cast<float>(spacing_y);
        patch_min_w = static_cast<float>(origin_w) * static_cast<float>(spacing_x);
        patch_max_w = static_cast<float>(origin_w + patch_w - 1) * static_cast<float>(spacing_x);
        spacing_h = static_cast<float>(spacing_y);
        spacing_w = static_cast<float>(spacing_x);
    } else if (axis_index == 1) {
        plane_axis = 1;
        dim_h = 2;
        dim_w = 0;
        plane_coord = static_cast<float>(slice_idx) * static_cast<float>(spacing_y);
        patch_min_h = static_cast<float>(origin_h) * static_cast<float>(spacing_z);
        patch_max_h = static_cast<float>(origin_h + patch_h - 1) * static_cast<float>(spacing_z);
        patch_min_w = static_cast<float>(origin_w) * static_cast<float>(spacing_x);
        patch_max_w = static_cast<float>(origin_w + patch_w - 1) * static_cast<float>(spacing_x);
        spacing_h = static_cast<float>(spacing_z);
        spacing_w = static_cast<float>(spacing_x);
    } else {
        plane_axis = 0;
        dim_h = 2;
        dim_w = 1;
        plane_coord = static_cast<float>(slice_idx) * static_cast<float>(spacing_x);
        patch_min_h = static_cast<float>(origin_h) * static_cast<float>(spacing_z);
        patch_max_h = static_cast<float>(origin_h + patch_h - 1) * static_cast<float>(spacing_z);
        patch_min_w = static_cast<float>(origin_w) * static_cast<float>(spacing_y);
        patch_max_w = static_cast<float>(origin_w + patch_w - 1) * static_cast<float>(spacing_y);
        spacing_h = static_cast<float>(spacing_z);
        spacing_w = static_cast<float>(spacing_y);
    }

    const auto dim_x = grid_dims[0].item<int32_t>();
    const auto dim_y = grid_dims[1].item<int32_t>();
    const auto dim_z = grid_dims[2].item<int32_t>();
    const auto world_min_host = grid_world_min.contiguous().cpu();
    const float* world_min_ptr = world_min_host.data_ptr<float>();
    int plane_cell = clamp_int_host(static_cast<int>(floor((plane_coord - world_min_ptr[plane_axis]) / static_cast<float>(cell_size))), 0, (plane_axis == 0 ? dim_x : (plane_axis == 1 ? dim_y : dim_z)) - 1);
    int cell_h_min = clamp_int_host(static_cast<int>(floor((patch_min_h - world_min_ptr[dim_h]) / static_cast<float>(cell_size))), 0, (dim_h == 0 ? dim_x : (dim_h == 1 ? dim_y : dim_z)) - 1);
    int cell_h_max = clamp_int_host(static_cast<int>(floor((patch_max_h - world_min_ptr[dim_h]) / static_cast<float>(cell_size))), 0, (dim_h == 0 ? dim_x : (dim_h == 1 ? dim_y : dim_z)) - 1);
    int cell_w_min = clamp_int_host(static_cast<int>(floor((patch_min_w - world_min_ptr[dim_w]) / static_cast<float>(cell_size))), 0, (dim_w == 0 ? dim_x : (dim_w == 1 ? dim_y : dim_z)) - 1);
    int cell_w_max = clamp_int_host(static_cast<int>(floor((patch_max_w - world_min_ptr[dim_w]) / static_cast<float>(cell_size))), 0, (dim_w == 0 ? dim_x : (dim_w == 1 ? dim_y : dim_z)) - 1);

    auto h_cells = at::arange(cell_h_min, cell_h_max + 1, options_long);
    auto w_cells = at::arange(cell_w_min, cell_w_max + 1, options_long);
    if (h_cells.numel() == 0 || w_cells.numel() == 0) {
        return {output, empty_offsets, at::empty({0}, options_int)};
    }
    auto h_grid = h_cells.repeat_interleave(w_cells.size(0));
    auto w_grid = w_cells.repeat({h_cells.size(0)});
    auto plane_grid = at::full_like(h_grid, static_cast<int64_t>(plane_cell));

    at::Tensor cell_x;
    at::Tensor cell_y;
    at::Tensor cell_z;
    if (axis_index == 0) {
        cell_x = w_grid;
        cell_y = h_grid;
        cell_z = plane_grid;
    } else if (axis_index == 1) {
        cell_x = w_grid;
        cell_y = plane_grid;
        cell_z = h_grid;
    } else {
        cell_x = plane_grid;
        cell_y = w_grid;
        cell_z = h_grid;
    }

    auto cell_ids = cell_x + static_cast<int64_t>(dim_x) * (cell_y + static_cast<int64_t>(dim_y) * cell_z);
    auto candidate_ids = gather_candidate_ids(cell_ids, cell_offsets, cell_gaussian_ids);
    if (candidate_ids.numel() == 0) {
        return {output, empty_offsets, at::empty({0}, options_int)};
    }

    auto counts = at::zeros({candidate_ids.size(0)}, options_long);
    int blocks = static_cast<int>((candidate_ids.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    count_patch_active_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        support_extent.contiguous().data_ptr<float>(),
        candidate_ids.contiguous().data_ptr<int32_t>(),
        candidate_ids.size(0),
        plane_axis,
        dim_h,
        dim_w,
        plane_coord,
        patch_min_h,
        patch_max_h,
        patch_min_w,
        patch_max_w,
        counts.data_ptr<int64_t>());
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "count_patch_active kernel launch failed: ", cudaGetErrorString(error));

    auto active_offsets = at::zeros({candidate_ids.size(0) + 1}, options_long);
    active_offsets.slice(0, 1).copy_(at::cumsum(counts, 0));
    int64_t active_count = active_offsets[-1].item<int64_t>();
    auto active_ids = at::empty({active_count}, options_int);
    if (active_count == 0) {
        return {output, empty_offsets, active_ids};
    }

    fill_patch_active_kernel<<<blocks, kThreads, 0, stream>>>(
        candidate_ids.contiguous().data_ptr<int32_t>(),
        active_offsets.contiguous().data_ptr<int64_t>(),
        counts.contiguous().data_ptr<int64_t>(),
        candidate_ids.size(0),
        active_ids.data_ptr<int32_t>());
    error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "fill_patch_active kernel launch failed: ", cudaGetErrorString(error));

    auto tile_data = build_slice_tile_lists(
        means,
        support_extent,
        active_ids,
        dim_h,
        dim_w,
        spacing_h,
        spacing_w,
        static_cast<int>(origin_h),
        static_cast<int>(origin_w),
        static_cast<int>(patch_h),
        static_cast<int>(patch_w),
        static_cast<int>(tile_size));
    auto tile_offsets = tile_data[0];
    auto tile_gaussian_ids = tile_data[1];
    if (tile_gaussian_ids.numel() == 0) {
        return {output, tile_offsets, tile_gaussian_ids};
    }

    int total_pixels = static_cast<int>(patch_h * patch_w);
    blocks = (total_pixels + kThreads - 1) / kThreads;
    render_slice_patch_local_forward_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        tile_offsets.contiguous().data_ptr<int64_t>(),
        tile_gaussian_ids.contiguous().data_ptr<int32_t>(),
        static_cast<int>(axis_index),
        static_cast<int>(slice_idx),
        static_cast<int>(origin_h),
        static_cast<int>(origin_w),
        static_cast<int>(patch_h),
        static_cast<int>(patch_w),
        static_cast<float>(spacing_z),
        static_cast<float>(spacing_y),
        static_cast<float>(spacing_x),
        static_cast<int>(tile_size),
        tile_w_count,
        output.data_ptr<float>());
    error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "render_slice_patch_local_forward kernel launch failed: ", cudaGetErrorString(error));
    return {output, tile_offsets, tile_gaussian_ids};
}

std::vector<at::Tensor> render_slice_patch_local_backward_cuda(
    at::Tensor means,
    at::Tensor rotations,
    at::Tensor scales,
    at::Tensor opacity,
    at::Tensor tile_offsets,
    at::Tensor tile_gaussian_ids,
    at::Tensor grad_output,
    int64_t axis_index,
    int64_t slice_idx,
    int64_t origin_h,
    int64_t origin_w,
    int64_t patch_h,
    int64_t patch_w,
    double spacing_z,
    double spacing_y,
    double spacing_x,
    int64_t tile_size) {
    check_cuda(means, "means");
    check_cuda(rotations, "rotations");
    check_cuda(scales, "scales");
    check_cuda(opacity, "opacity");
    check_cuda(tile_offsets, "tile_offsets");
    check_cuda(tile_gaussian_ids, "tile_gaussian_ids");
    check_cuda(grad_output, "grad_output");
    check_float(means, "means");
    check_float(rotations, "rotations");
    check_float(scales, "scales");
    check_float(opacity, "opacity");
    check_int64(tile_offsets, "tile_offsets");
    check_int32(tile_gaussian_ids, "tile_gaussian_ids");
    check_float(grad_output, "grad_output");

    c10::cuda::CUDAGuard device_guard(means.device());
    auto grad_means = at::zeros_like(means);
    auto grad_rotations = at::zeros_like(rotations);
    auto grad_scales = at::zeros_like(scales);
    auto grad_opacity = at::zeros_like(opacity);
    if (means.numel() == 0 || tile_gaussian_ids.numel() == 0 || patch_h == 0 || patch_w == 0) {
        return {grad_means, grad_rotations, grad_scales, grad_opacity};
    }

    const int tile_w_count = (static_cast<int>(patch_w) + static_cast<int>(tile_size) - 1) / static_cast<int>(tile_size);
    int total_pixels = static_cast<int>(patch_h * patch_w);
    int blocks = (total_pixels + kThreads - 1) / kThreads;
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    render_slice_patch_local_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        tile_offsets.contiguous().data_ptr<int64_t>(),
        tile_gaussian_ids.contiguous().data_ptr<int32_t>(),
        grad_output.contiguous().data_ptr<float>(),
        static_cast<int>(axis_index),
        static_cast<int>(slice_idx),
        static_cast<int>(origin_h),
        static_cast<int>(origin_w),
        static_cast<int>(patch_h),
        static_cast<int>(patch_w),
        static_cast<float>(spacing_z),
        static_cast<float>(spacing_y),
        static_cast<float>(spacing_x),
        static_cast<int>(tile_size),
        tile_w_count,
        grad_means.data_ptr<float>(),
        grad_rotations.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_opacity.data_ptr<float>());
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "render_slice_patch_local_backward kernel launch failed: ", cudaGetErrorString(error));
    return {grad_means, grad_rotations, grad_scales, grad_opacity};
}

std::vector<at::Tensor> query_density_local_forward_cuda(
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
    at::Tensor cell_gaussian_ids) {
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

    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must have shape (N, 3).");
    TORCH_CHECK(rotations.dim() == 3 && rotations.size(1) == 3 && rotations.size(2) == 3, "rotations must have shape (N, 3, 3).");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must have shape (N, 3).");
    TORCH_CHECK(opacity.dim() == 1 && opacity.size(0) == means.size(0), "opacity must have shape (N,).");
    TORCH_CHECK(support_extent.dim() == 2 && support_extent.size(0) == means.size(0) && support_extent.size(1) == 3, "support_extent must have shape (N, 3).");
    TORCH_CHECK(query_points.dim() == 2 && query_points.size(1) == 3, "query_points must have shape (Q, 3).");
    TORCH_CHECK(grid_world_min.dim() == 1 && grid_world_min.size(0) == 3, "grid_world_min must have shape (3,).");
    TORCH_CHECK(grid_dims.dim() == 1 && grid_dims.size(0) == 3, "grid_dims must have shape (3,).");

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
    count_local_neighbors_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        support_extent.contiguous().data_ptr<float>(),
        query_points.contiguous().data_ptr<float>(),
        grid_world_min.contiguous().data_ptr<float>(),
        grid_dims.contiguous().data_ptr<int32_t>(),
        static_cast<float>(cell_size),
        cell_offsets.contiguous().data_ptr<int64_t>(),
        cell_gaussian_ids.contiguous().data_ptr<int32_t>(),
        query_points.size(0),
        counts.data_ptr<int64_t>());
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "count_local_neighbors kernel launch failed: ", cudaGetErrorString(error));

    query_offsets.slice(0, 1).copy_(at::cumsum(counts, 0));
    int64_t total_neighbors = query_offsets[-1].item<int64_t>();
    auto query_gaussian_ids = at::empty({total_neighbors}, options_int);
    if (total_neighbors == 0) {
        return {density, query_offsets, query_gaussian_ids};
    }

    fill_local_neighbors_and_density_kernel<<<blocks, kThreads, 0, stream>>>(
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
        query_gaussian_ids.data_ptr<int32_t>(),
        density.data_ptr<float>());
    error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "fill_local_neighbors_and_density kernel launch failed: ", cudaGetErrorString(error));
    return {density, query_offsets, query_gaussian_ids};
}

std::vector<at::Tensor> query_density_local_backward_cuda(
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
    check_int64(query_offsets, "query_offsets");
    check_int32(query_gaussian_ids, "query_gaussian_ids");
    check_float(grad_output, "grad_output");

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
    query_density_local_backward_kernel<<<blocks, kThreads, 0, stream>>>(
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
    TORCH_CHECK(error == cudaSuccess, "query_density_local_backward kernel launch failed: ", cudaGetErrorString(error));
    return {grad_means, grad_rotations, grad_scales, grad_opacity};
}

namespace {

__global__ void count_patch_active_kernel(
    const float* means,
    const float* support_extent,
    const int32_t* candidate_ids,
    int64_t num_candidates,
    int plane_axis,
    int dim_h,
    int dim_w,
    float plane_coord,
    float patch_min_h,
    float patch_max_h,
    float patch_min_w,
    float patch_max_w,
    int64_t* counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) {
        return;
    }

    int32_t gaussian_idx = candidate_ids[idx];
    const float* mean = means + gaussian_idx * 3;
    const float* extent = support_extent + gaussian_idx * 3;

    bool active = fabsf(plane_coord - mean[plane_axis]) <= extent[plane_axis];
    active = active && (mean[dim_h] + extent[dim_h] >= patch_min_h);
    active = active && (mean[dim_h] - extent[dim_h] <= patch_max_h);
    active = active && (mean[dim_w] + extent[dim_w] >= patch_min_w);
    active = active && (mean[dim_w] - extent[dim_w] <= patch_max_w);
    counts[idx] = active ? 1 : 0;
}

__global__ void fill_patch_active_kernel(
    const int32_t* candidate_ids,
    const int64_t* counts_prefix,
    const int64_t* counts,
    int64_t num_candidates,
    int32_t* active_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates || counts[idx] == 0) {
        return;
    }
    active_ids[counts_prefix[idx]] = candidate_ids[idx];
}

__device__ __forceinline__ void project_patch_ranges(
    const float* mean,
    const float* extent,
    int dim_h,
    int dim_w,
    float patch_spacing_h,
    float patch_spacing_w,
    int origin_h,
    int origin_w,
    int patch_h,
    int patch_w,
    int tile_size,
    int* tile_h_min,
    int* tile_h_max,
    int* tile_w_min,
    int* tile_w_max) {
    float min_h_world = mean[dim_h] - extent[dim_h];
    float max_h_world = mean[dim_h] + extent[dim_h];
    float min_w_world = mean[dim_w] - extent[dim_w];
    float max_w_world = mean[dim_w] + extent[dim_w];

    int pix_h_min = static_cast<int>(floorf(min_h_world / patch_spacing_h)) - origin_h;
    int pix_h_max = static_cast<int>(ceilf(max_h_world / patch_spacing_h)) - origin_h;
    int pix_w_min = static_cast<int>(floorf(min_w_world / patch_spacing_w)) - origin_w;
    int pix_w_max = static_cast<int>(ceilf(max_w_world / patch_spacing_w)) - origin_w;

    pix_h_min = clamp_int(pix_h_min, 0, patch_h - 1);
    pix_h_max = clamp_int(pix_h_max, 0, patch_h - 1);
    pix_w_min = clamp_int(pix_w_min, 0, patch_w - 1);
    pix_w_max = clamp_int(pix_w_max, 0, patch_w - 1);

    *tile_h_min = pix_h_min / tile_size;
    *tile_h_max = pix_h_max / tile_size;
    *tile_w_min = pix_w_min / tile_size;
    *tile_w_max = pix_w_max / tile_size;
}

__global__ void count_tile_entries_kernel(
    const float* means,
    const float* support_extent,
    const int32_t* active_ids,
    int64_t num_active,
    int dim_h,
    int dim_w,
    float spacing_h,
    float spacing_w,
    int origin_h,
    int origin_w,
    int patch_h,
    int patch_w,
    int tile_size,
    int64_t* counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active) {
        return;
    }
    int32_t gaussian_idx = active_ids[idx];
    const float* mean = means + gaussian_idx * 3;
    const float* extent = support_extent + gaussian_idx * 3;
    int tile_h_min = 0;
    int tile_h_max = 0;
    int tile_w_min = 0;
    int tile_w_max = 0;
    project_patch_ranges(mean, extent, dim_h, dim_w, spacing_h, spacing_w, origin_h, origin_w, patch_h, patch_w, tile_size, &tile_h_min, &tile_h_max, &tile_w_min, &tile_w_max);
    counts[idx] = static_cast<int64_t>(tile_h_max - tile_h_min + 1) * static_cast<int64_t>(tile_w_max - tile_w_min + 1);
}

__global__ void fill_tile_entries_kernel(
    const float* means,
    const float* support_extent,
    const int32_t* active_ids,
    const int64_t* entry_offsets,
    int64_t num_active,
    int dim_h,
    int dim_w,
    float spacing_h,
    float spacing_w,
    int origin_h,
    int origin_w,
    int patch_h,
    int patch_w,
    int tile_size,
    int tile_w_count,
    int32_t* tile_ids,
    int32_t* tile_gaussian_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active) {
        return;
    }
    int32_t gaussian_idx = active_ids[idx];
    const float* mean = means + gaussian_idx * 3;
    const float* extent = support_extent + gaussian_idx * 3;
    int tile_h_min = 0;
    int tile_h_max = 0;
    int tile_w_min = 0;
    int tile_w_max = 0;
    project_patch_ranges(mean, extent, dim_h, dim_w, spacing_h, spacing_w, origin_h, origin_w, patch_h, patch_w, tile_size, &tile_h_min, &tile_h_max, &tile_w_min, &tile_w_max);
    int64_t write_offset = entry_offsets[idx];
    for (int tile_h = tile_h_min; tile_h <= tile_h_max; ++tile_h) {
        for (int tile_w = tile_w_min; tile_w <= tile_w_max; ++tile_w) {
            tile_ids[write_offset] = tile_h * tile_w_count + tile_w;
            tile_gaussian_ids[write_offset] = gaussian_idx;
            ++write_offset;
        }
    }
}

__device__ __forceinline__ void slice_world_point(
    int axis_index,
    int slice_idx,
    int origin_h,
    int origin_w,
    int patch_w,
    float spacing_z,
    float spacing_y,
    float spacing_x,
    int pixel_index,
    float* x,
    float* y,
    float* z) {
    int row = pixel_index / patch_w;
    int col = pixel_index % patch_w;
    if (axis_index == 0) {
        *x = (static_cast<float>(origin_w + col) + 0.5f) * spacing_x;
        *y = (static_cast<float>(origin_h + row) + 0.5f) * spacing_y;
        *z = (static_cast<float>(slice_idx) + 0.5f) * spacing_z;
    } else if (axis_index == 1) {
        *x = (static_cast<float>(origin_w + col) + 0.5f) * spacing_x;
        *y = (static_cast<float>(slice_idx) + 0.5f) * spacing_y;
        *z = (static_cast<float>(origin_h + row) + 0.5f) * spacing_z;
    } else {
        *x = (static_cast<float>(slice_idx) + 0.5f) * spacing_x;
        *y = (static_cast<float>(origin_w + col) + 0.5f) * spacing_y;
        *z = (static_cast<float>(origin_h + row) + 0.5f) * spacing_z;
    }
}

__global__ void render_slice_patch_local_forward_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const int64_t* tile_offsets,
    const int32_t* tile_gaussian_ids,
    int axis_index,
    int slice_idx,
    int origin_h,
    int origin_w,
    int patch_h,
    int patch_w,
    float spacing_z,
    float spacing_y,
    float spacing_x,
    int tile_size,
    int tile_w_count,
    float* output) {
    int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = patch_h * patch_w;
    if (pixel_index >= total_pixels) {
        return;
    }

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    slice_world_point(axis_index, slice_idx, origin_h, origin_w, patch_w, spacing_z, spacing_y, spacing_x, pixel_index, &x, &y, &z);
    int row = pixel_index / patch_w;
    int col = pixel_index % patch_w;
    int tile_h = row / tile_size;
    int tile_w = col / tile_size;
    int tile_id = tile_h * tile_w_count + tile_w;

    float accum = 0.0f;
    int64_t start = tile_offsets[tile_id];
    int64_t end = tile_offsets[tile_id + 1];
    for (int64_t cursor = start; cursor < end; ++cursor) {
        int32_t gaussian_idx = tile_gaussian_ids[cursor];
        const float* mean = means + gaussian_idx * 3;
        const float* rotation = rotations + gaussian_idx * 9;
        const float* scale = scales + gaussian_idx * 3;

        float dx = x - mean[0];
        float dy = y - mean[1];
        float dz = z - mean[2];
        float local_x = dx * rotation[0] + dy * rotation[3] + dz * rotation[6];
        float local_y = dx * rotation[1] + dy * rotation[4] + dz * rotation[7];
        float local_z = dx * rotation[2] + dy * rotation[5] + dz * rotation[8];

        float nx = local_x / scale[0];
        float ny = local_y / scale[1];
        float nz = local_z / scale[2];
        float exponent = -0.5f * (nx * nx + ny * ny + nz * nz);
        accum += expf(exponent) * opacity[gaussian_idx];
    }
    output[pixel_index] = accum;
}

__global__ void render_slice_patch_local_backward_kernel(
    const float* means,
    const float* rotations,
    const float* scales,
    const float* opacity,
    const int64_t* tile_offsets,
    const int32_t* tile_gaussian_ids,
    const float* grad_output,
    int axis_index,
    int slice_idx,
    int origin_h,
    int origin_w,
    int patch_h,
    int patch_w,
    float spacing_z,
    float spacing_y,
    float spacing_x,
    int tile_size,
    int tile_w_count,
    float* grad_means,
    float* grad_rotations,
    float* grad_scales,
    float* grad_opacity) {
    int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = patch_h * patch_w;
    if (pixel_index >= total_pixels) {
        return;
    }

    float upstream = grad_output[pixel_index];
    if (upstream == 0.0f) {
        return;
    }

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    slice_world_point(axis_index, slice_idx, origin_h, origin_w, patch_w, spacing_z, spacing_y, spacing_x, pixel_index, &x, &y, &z);
    int row = pixel_index / patch_w;
    int col = pixel_index % patch_w;
    int tile_h = row / tile_size;
    int tile_w = col / tile_size;
    int tile_id = tile_h * tile_w_count + tile_w;

    int64_t start = tile_offsets[tile_id];
    int64_t end = tile_offsets[tile_id + 1];
    for (int64_t cursor = start; cursor < end; ++cursor) {
        int32_t gaussian_idx = tile_gaussian_ids[cursor];
        const float* mean = means + gaussian_idx * 3;
        const float* rotation = rotations + gaussian_idx * 9;
        const float* scale = scales + gaussian_idx * 3;

        float dx = x - mean[0];
        float dy = y - mean[1];
        float dz = z - mean[2];

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

std::vector<at::Tensor> build_slice_tile_lists(
    at::Tensor means,
    at::Tensor support_extent,
    at::Tensor active_ids,
    int dim_h,
    int dim_w,
    float spacing_h,
    float spacing_w,
    int origin_h,
    int origin_w,
    int patch_h,
    int patch_w,
    int tile_size) {
    auto options_long = at::TensorOptions().device(means.device()).dtype(at::kLong);
    auto options_int = at::TensorOptions().device(means.device()).dtype(at::kInt);
    const int tile_h_count = (patch_h + tile_size - 1) / tile_size;
    const int tile_w_count = (patch_w + tile_size - 1) / tile_size;
    const int num_tiles = tile_h_count * tile_w_count;
    auto tile_offsets = at::zeros({num_tiles + 1}, options_long);
    if (active_ids.numel() == 0) {
        return {tile_offsets, at::empty({0}, options_int)};
    }

    auto counts = at::zeros({active_ids.size(0)}, options_long);
    int blocks = static_cast<int>((active_ids.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    count_tile_entries_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        support_extent.contiguous().data_ptr<float>(),
        active_ids.contiguous().data_ptr<int32_t>(),
        active_ids.size(0),
        dim_h,
        dim_w,
        spacing_h,
        spacing_w,
        origin_h,
        origin_w,
        patch_h,
        patch_w,
        tile_size,
        counts.data_ptr<int64_t>());
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "count_tile_entries kernel launch failed: ", cudaGetErrorString(error));

    auto entry_offsets = at::zeros({active_ids.size(0) + 1}, options_long);
    entry_offsets.slice(0, 1).copy_(at::cumsum(counts, 0));
    int64_t total_entries = entry_offsets[-1].item<int64_t>();
    auto tile_gaussian_ids = at::empty({total_entries}, options_int);
    if (total_entries == 0) {
        return {tile_offsets, tile_gaussian_ids};
    }

    auto unsorted_tile_ids = at::empty({total_entries}, options_int);
    fill_tile_entries_kernel<<<blocks, kThreads, 0, stream>>>(
        means.contiguous().data_ptr<float>(),
        support_extent.contiguous().data_ptr<float>(),
        active_ids.contiguous().data_ptr<int32_t>(),
        entry_offsets.contiguous().data_ptr<int64_t>(),
        active_ids.size(0),
        dim_h,
        dim_w,
        spacing_h,
        spacing_w,
        origin_h,
        origin_w,
        patch_h,
        patch_w,
        tile_size,
        tile_w_count,
        unsorted_tile_ids.data_ptr<int32_t>(),
        tile_gaussian_ids.data_ptr<int32_t>());
    error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "fill_tile_entries kernel launch failed: ", cudaGetErrorString(error));

    auto sort_result = at::sort(unsorted_tile_ids, 0, false);
    auto sorted_tile_ids = std::get<0>(sort_result);
    auto sort_indices = std::get<1>(sort_result);
    auto sorted_gaussian_ids = tile_gaussian_ids.index_select(0, sort_indices);
    auto counts_per_tile = at::bincount(sorted_tile_ids.toType(at::kLong), {}, num_tiles);
    tile_offsets.slice(0, 1).copy_(at::cumsum(counts_per_tile, 0));
    return {tile_offsets, sorted_gaussian_ids};
}

}  // namespace
