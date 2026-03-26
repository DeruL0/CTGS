#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxSupportedK = 32;

void check_cuda(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
}

void check_float(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32.");
}

__device__ inline bool better_candidate(float dist, int64_t idx, float current_dist, int64_t current_idx) {
    return (dist < current_dist) || ((dist == current_dist) && (idx < current_idx));
}

template <int K_MAX>
__global__ void build_neighbor_index_kernel(
    const float* xyz,
    int64_t num_points,
    int64_t effective_k,
    int64_t tile_size,
    int64_t* output) {
    extern __shared__ float shared_xyz[];
    float* shared_x = shared_xyz;
    float* shared_y = shared_xyz + tile_size;
    float* shared_z = shared_xyz + 2 * tile_size;

    int64_t query_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query_index >= num_points) {
        return;
    }

    float query_x = xyz[query_index * 3 + 0];
    float query_y = xyz[query_index * 3 + 1];
    float query_z = xyz[query_index * 3 + 2];

    float best_dist[K_MAX];
    int64_t best_idx[K_MAX];
#pragma unroll
    for (int i = 0; i < K_MAX; ++i) {
        best_dist[i] = std::numeric_limits<float>::infinity();
        best_idx[i] = -1;
    }

    for (int64_t tile_start = 0; tile_start < num_points; tile_start += tile_size) {
        int64_t current_tile = tile_size < (num_points - tile_start) ? tile_size : (num_points - tile_start);
        for (int64_t load_idx = threadIdx.x; load_idx < current_tile; load_idx += blockDim.x) {
            int64_t point_idx = tile_start + load_idx;
            shared_x[load_idx] = xyz[point_idx * 3 + 0];
            shared_y[load_idx] = xyz[point_idx * 3 + 1];
            shared_z[load_idx] = xyz[point_idx * 3 + 2];
        }
        __syncthreads();

        for (int64_t local_idx = 0; local_idx < current_tile; ++local_idx) {
            int64_t candidate_idx = tile_start + local_idx;
            if (candidate_idx == query_index) {
                continue;
            }

            float dx = query_x - shared_x[local_idx];
            float dy = query_y - shared_y[local_idx];
            float dz = query_z - shared_z[local_idx];
            float dist = dx * dx + dy * dy + dz * dz;

            int insert_pos = -1;
            for (int rank = 0; rank < effective_k; ++rank) {
                if (better_candidate(dist, candidate_idx, best_dist[rank], best_idx[rank])) {
                    insert_pos = rank;
                    break;
                }
            }
            if (insert_pos < 0) {
                continue;
            }

            for (int rank = static_cast<int>(effective_k) - 1; rank > insert_pos; --rank) {
                best_dist[rank] = best_dist[rank - 1];
                best_idx[rank] = best_idx[rank - 1];
            }
            best_dist[insert_pos] = dist;
            best_idx[insert_pos] = candidate_idx;
        }
        __syncthreads();
    }

    for (int rank = 0; rank < effective_k; ++rank) {
        output[query_index * effective_k + rank] = best_idx[rank];
    }
}

}  // namespace

at::Tensor build_neighbor_index_cuda(
    at::Tensor xyz,
    int64_t k,
    int64_t tile_size) {
    check_cuda(xyz, "xyz");
    check_float(xyz, "xyz");
    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must have shape (N, 3).");
    TORCH_CHECK(k >= 0, "k must be >= 0.");
    TORCH_CHECK(tile_size >= 1, "tile_size must be >= 1.");

    auto long_options = at::TensorOptions().device(xyz.device()).dtype(at::kLong);
    if (xyz.size(0) == 0 || k == 0) {
        return at::empty({xyz.size(0), 0}, long_options);
    }
    if (xyz.size(0) == 1) {
        return at::empty({1, 0}, long_options);
    }

    int64_t effective_k = std::min<int64_t>(k, xyz.size(0) - 1);
    TORCH_CHECK(effective_k <= kMaxSupportedK, "build_neighbor_index_cuda currently supports k <= ", kMaxSupportedK, ".");

    c10::cuda::CUDAGuard device_guard(xyz.device());
    auto output = at::empty({xyz.size(0), effective_k}, long_options);
    int64_t effective_tile_size = std::max<int64_t>(1, std::min<int64_t>(tile_size, xyz.size(0)));
    dim3 blocks(static_cast<unsigned int>((xyz.size(0) + kThreads - 1) / kThreads));
    size_t shared_bytes = static_cast<size_t>(effective_tile_size) * 3 * sizeof(float);
    auto stream = at::cuda::getDefaultCUDAStream().stream();

    build_neighbor_index_kernel<kMaxSupportedK><<<blocks, kThreads, shared_bytes, stream>>>(
        xyz.contiguous().data_ptr<float>(),
        xyz.size(0),
        effective_k,
        effective_tile_size,
        output.data_ptr<int64_t>());

    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "build_neighbor_index kernel launch failed: ", cudaGetErrorString(error));
    return output;
}
