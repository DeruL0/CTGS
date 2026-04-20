#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

void check_cuda(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
}

void check_mask(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.scalar_type() == at::kBool || tensor.scalar_type() == at::kByte,
        name,
        " must be bool or uint8.");
}

template <typename scalar_t>
__device__ inline bool read_mask(const scalar_t* mask, int64_t index) {
    return mask[index] != static_cast<scalar_t>(0);
}

template <>
__device__ inline bool read_mask<bool>(const bool* mask, int64_t index) {
    return mask[index];
}

template <typename scalar_t>
__global__ void build_boundary_mask_kernel(
    const scalar_t* material_mask,
    int64_t depth,
    int64_t height,
    int64_t width,
    uint8_t* boundary_mask) {
    const int64_t linear_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t voxel_count = depth * height * width;
    if (linear_index >= voxel_count) {
        return;
    }

    const int64_t z = linear_index / (height * width);
    const int64_t rem = linear_index - z * height * width;
    const int64_t y = rem / width;
    const int64_t x = rem - y * width;

    const bool current = read_mask(material_mask, linear_index);
    bool is_boundary = false;

    if (x > 0) {
        is_boundary |= read_mask(material_mask, linear_index - 1) != current;
    }
    if (!is_boundary && x + 1 < width) {
        is_boundary |= read_mask(material_mask, linear_index + 1) != current;
    }
    if (!is_boundary && y > 0) {
        is_boundary |= read_mask(material_mask, linear_index - width) != current;
    }
    if (!is_boundary && y + 1 < height) {
        is_boundary |= read_mask(material_mask, linear_index + width) != current;
    }
    if (!is_boundary && z > 0) {
        is_boundary |= read_mask(material_mask, linear_index - height * width) != current;
    }
    if (!is_boundary && z + 1 < depth) {
        is_boundary |= read_mask(material_mask, linear_index + height * width) != current;
    }

    boundary_mask[linear_index] = is_boundary ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
}

template <typename scalar_t>
__global__ void build_signed_field_kernel(
    const scalar_t* material_mask,
    const uint8_t* boundary_mask,
    int64_t depth,
    int64_t height,
    int64_t width,
    int band_voxels,
    float* signed_field) {
    const int64_t linear_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t voxel_count = depth * height * width;
    if (linear_index >= voxel_count) {
        return;
    }

    const int64_t z = linear_index / (height * width);
    const int64_t rem = linear_index - z * height * width;
    const int64_t y = rem / width;
    const int64_t x = rem - y * width;

    if (boundary_mask[linear_index] != 0) {
        signed_field[linear_index] = 0.0f;
        return;
    }

    float min_distance_sq = static_cast<float>((band_voxels + 1) * (band_voxels + 1) * 3);
    bool found = false;

    for (int dz = -band_voxels; dz <= band_voxels; ++dz) {
        const int64_t nz = z + dz;
        if (nz < 0 || nz >= depth) {
            continue;
        }
        for (int dy = -band_voxels; dy <= band_voxels; ++dy) {
            const int64_t ny = y + dy;
            if (ny < 0 || ny >= height) {
                continue;
            }
            for (int dx = -band_voxels; dx <= band_voxels; ++dx) {
                const int64_t nx = x + dx;
                if (nx < 0 || nx >= width) {
                    continue;
                }
                const int64_t neighbor_index = (nz * height + ny) * width + nx;
                if (boundary_mask[neighbor_index] == 0) {
                    continue;
                }

                const float distance_sq = static_cast<float>(dx * dx + dy * dy + dz * dz);
                if (!found || distance_sq < min_distance_sq) {
                    min_distance_sq = distance_sq;
                    found = true;
                }
            }
        }
    }

    float distance = found ? sqrtf(min_distance_sq) : static_cast<float>(band_voxels);
    distance = fminf(distance, static_cast<float>(band_voxels));
    signed_field[linear_index] = read_mask(material_mask, linear_index) ? distance : -distance;
}

}  // namespace

at::Tensor build_signed_field_cuda(
    at::Tensor material_mask,
    int64_t band_voxels) {
    check_cuda(material_mask, "material_mask");
    check_mask(material_mask, "material_mask");
    TORCH_CHECK(material_mask.dim() == 3, "material_mask must have shape (D, H, W).");
    TORCH_CHECK(band_voxels >= 1, "band_voxels must be >= 1.");

    c10::cuda::CUDAGuard device_guard(material_mask.device());
    auto contiguous_mask = material_mask.contiguous();
    const auto depth = contiguous_mask.size(0);
    const auto height = contiguous_mask.size(1);
    const auto width = contiguous_mask.size(2);
    const auto voxel_count = depth * height * width;

    auto boundary_mask = at::zeros(
        {depth, height, width},
        at::TensorOptions().device(contiguous_mask.device()).dtype(at::kByte));
    auto signed_field = at::zeros(
        {depth, height, width},
        at::TensorOptions().device(contiguous_mask.device()).dtype(at::kFloat));
    if (voxel_count == 0) {
        return signed_field;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((voxel_count + threads - 1) / threads);

    if (contiguous_mask.scalar_type() == at::kBool) {
        build_boundary_mask_kernel<bool><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            contiguous_mask.data_ptr<bool>(),
            depth,
            height,
            width,
            boundary_mask.data_ptr<uint8_t>());
        build_signed_field_kernel<bool><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            contiguous_mask.data_ptr<bool>(),
            boundary_mask.data_ptr<uint8_t>(),
            depth,
            height,
            width,
            static_cast<int>(band_voxels),
            signed_field.data_ptr<float>());
    } else {
        auto byte_mask = contiguous_mask.to(at::kByte);
        build_boundary_mask_kernel<uint8_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            byte_mask.data_ptr<uint8_t>(),
            depth,
            height,
            width,
            boundary_mask.data_ptr<uint8_t>());
        build_signed_field_kernel<uint8_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            byte_mask.data_ptr<uint8_t>(),
            boundary_mask.data_ptr<uint8_t>(),
            depth,
            height,
            width,
            static_cast<int>(band_voxels),
            signed_field.data_ptr<float>());
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return signed_field;
}
