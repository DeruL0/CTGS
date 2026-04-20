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

void check_int32(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32.");
}

__global__ void count_grid_entries_kernel(
    const int32_t* cell_min,
    const int32_t* cell_max,
    int64_t num_gaussians,
    int64_t* counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) {
        return;
    }

    const int32_t* min_ptr = cell_min + idx * 3;
    const int32_t* max_ptr = cell_max + idx * 3;
    int64_t dx = static_cast<int64_t>(max_ptr[0] - min_ptr[0] + 1);
    int64_t dy = static_cast<int64_t>(max_ptr[1] - min_ptr[1] + 1);
    int64_t dz = static_cast<int64_t>(max_ptr[2] - min_ptr[2] + 1);
    dx = dx > 0 ? dx : 0;
    dy = dy > 0 ? dy : 0;
    dz = dz > 0 ? dz : 0;
    counts[idx] = dx * dy * dz;
}

__global__ void fill_grid_entries_kernel(
    const int32_t* cell_min,
    const int32_t* cell_max,
    const int64_t* offsets,
    int64_t num_gaussians,
    int32_t dim_x,
    int32_t dim_y,
    int32_t dim_z,
    int32_t* cell_ids,
    int32_t* gaussian_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) {
        return;
    }

    const int32_t* min_ptr = cell_min + idx * 3;
    const int32_t* max_ptr = cell_max + idx * 3;
    int64_t write_offset = offsets[idx];
    for (int32_t z = min_ptr[2]; z <= max_ptr[2]; ++z) {
        for (int32_t y = min_ptr[1]; y <= max_ptr[1]; ++y) {
            for (int32_t x = min_ptr[0]; x <= max_ptr[0]; ++x) {
                int32_t flat = x + dim_x * (y + dim_y * z);
                cell_ids[write_offset] = flat;
                gaussian_ids[write_offset] = idx;
                ++write_offset;
            }
        }
    }
}

}  // namespace

std::vector<at::Tensor> build_uniform_grid_cuda(
    at::Tensor cell_min,
    at::Tensor cell_max,
    at::Tensor grid_dims) {
    check_cuda(cell_min, "cell_min");
    check_cuda(cell_max, "cell_max");
    check_cuda(grid_dims, "grid_dims");
    check_int32(cell_min, "cell_min");
    check_int32(cell_max, "cell_max");
    check_int32(grid_dims, "grid_dims");
    TORCH_CHECK(cell_min.dim() == 2 && cell_min.size(1) == 3, "cell_min must have shape (N, 3).");
    TORCH_CHECK(cell_max.dim() == 2 && cell_max.size(1) == 3, "cell_max must have shape (N, 3).");
    TORCH_CHECK(grid_dims.dim() == 1 && grid_dims.size(0) == 3, "grid_dims must have shape (3,).");

    c10::cuda::CUDAGuard device_guard(cell_min.device());
    auto options_long = at::TensorOptions().device(cell_min.device()).dtype(at::kLong);
    auto options_int = at::TensorOptions().device(cell_min.device()).dtype(at::kInt);

    const auto dim_x = grid_dims[0].item<int32_t>();
    const auto dim_y = grid_dims[1].item<int32_t>();
    const auto dim_z = grid_dims[2].item<int32_t>();
    const int64_t num_cells = static_cast<int64_t>(dim_x) * static_cast<int64_t>(dim_y) * static_cast<int64_t>(dim_z);

    auto empty_offsets = at::zeros({num_cells + 1}, options_long);
    if (cell_min.size(0) == 0 || num_cells <= 0) {
        return {empty_offsets, at::empty({0}, options_int)};
    }

    auto counts = at::zeros({cell_min.size(0)}, options_long);
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((cell_min.size(0) + kThreads - 1) / kThreads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();

    count_grid_entries_kernel<<<blocks, kThreads, 0, stream>>>(
        cell_min.contiguous().data_ptr<int32_t>(),
        cell_max.contiguous().data_ptr<int32_t>(),
        cell_min.size(0),
        counts.data_ptr<int64_t>());

    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "count_grid_entries kernel launch failed: ", cudaGetErrorString(error));

    auto entry_offsets = at::zeros({cell_min.size(0) + 1}, options_long);
    entry_offsets.slice(0, 1).copy_(at::cumsum(counts, 0));
    const int64_t total_entries = entry_offsets[-1].item<int64_t>();
    if (total_entries == 0) {
        return {empty_offsets, at::empty({0}, options_int)};
    }

    auto unsorted_cell_ids = at::empty({total_entries}, options_int);
    auto unsorted_gaussian_ids = at::empty({total_entries}, options_int);

    fill_grid_entries_kernel<<<blocks, kThreads, 0, stream>>>(
        cell_min.contiguous().data_ptr<int32_t>(),
        cell_max.contiguous().data_ptr<int32_t>(),
        entry_offsets.contiguous().data_ptr<int64_t>(),
        cell_min.size(0),
        dim_x,
        dim_y,
        dim_z,
        unsorted_cell_ids.data_ptr<int32_t>(),
        unsorted_gaussian_ids.data_ptr<int32_t>());

    error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "fill_grid_entries kernel launch failed: ", cudaGetErrorString(error));

    auto sort_result = at::sort(unsorted_cell_ids, 0, false);
    auto sorted_cell_ids = std::get<0>(sort_result);
    auto sort_indices = std::get<1>(sort_result);
    auto sorted_gaussian_ids = unsorted_gaussian_ids.index_select(0, sort_indices);

    auto counts_per_cell = at::bincount(sorted_cell_ids.toType(at::kLong), {}, num_cells);
    auto cell_offsets = at::zeros({num_cells + 1}, options_long);
    cell_offsets.slice(0, 1).copy_(at::cumsum(counts_per_cell, 0));
    return {cell_offsets, sorted_gaussian_ids};
}
