#include <torch/extension.h>

#include <vector>

torch::Tensor render_slice_patch_forward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    int64_t axis_index,
    int64_t slice_idx,
    int64_t origin_h,
    int64_t origin_w,
    int64_t patch_h,
    int64_t patch_w,
    double spacing_z,
    double spacing_y,
    double spacing_x);

std::vector<torch::Tensor> render_slice_patch_backward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor grad_output,
    int64_t axis_index,
    int64_t slice_idx,
    int64_t origin_h,
    int64_t origin_w,
    int64_t patch_h,
    int64_t patch_w,
    double spacing_z,
    double spacing_y,
    double spacing_x);

torch::Tensor query_density_forward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor query_points);

std::vector<torch::Tensor> query_density_backward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor query_points,
    torch::Tensor grad_output);

std::vector<torch::Tensor> build_plane_targets_cuda(
    torch::Tensor xyz,
    torch::Tensor normals,
    torch::Tensor planarity,
    torch::Tensor material_ids,
    torch::Tensor planar_mask,
    torch::Tensor neighbor_index);

torch::Tensor build_neighbor_index_cuda(
    torch::Tensor xyz,
    int64_t k,
    int64_t tile_size);

torch::Tensor point_to_plane_loss_cuda(
    torch::Tensor xyz,
    torch::Tensor active_indices,
    torch::Tensor centroids,
    torch::Tensor fitted_normals,
    torch::Tensor weights);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "render_slice_patch_forward",
        &render_slice_patch_forward_cuda,
        "CT native slice patch render forward (CUDA)");
    m.def(
        "render_slice_patch_backward",
        &render_slice_patch_backward_cuda,
        "CT native slice patch render backward (CUDA)");
    m.def(
        "query_density_forward",
        &query_density_forward_cuda,
        "CT native density query forward (CUDA)");
    m.def(
        "query_density_backward",
        &query_density_backward_cuda,
        "CT native density query backward (CUDA)");
    m.def(
        "build_plane_targets_cuda",
        &build_plane_targets_cuda,
        "Build detached point-to-plane targets on CUDA");
    m.def(
        "build_neighbor_index_cuda",
        &build_neighbor_index_cuda,
        "Build exact KNN neighbor indices on CUDA");
    m.def(
        "point_to_plane_loss_cuda",
        &point_to_plane_loss_cuda,
        "Compute point-to-plane loss from cached plane targets on CUDA");
}
