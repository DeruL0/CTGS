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

std::vector<torch::Tensor> render_slice_patch_local_forward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor support_extent,
    torch::Tensor grid_world_min,
    torch::Tensor grid_dims,
    double cell_size,
    torch::Tensor cell_offsets,
    torch::Tensor cell_gaussian_ids,
    int64_t axis_index,
    int64_t slice_idx,
    int64_t origin_h,
    int64_t origin_w,
    int64_t patch_h,
    int64_t patch_w,
    double spacing_z,
    double spacing_y,
    double spacing_x,
    int64_t tile_size);

std::vector<torch::Tensor> render_slice_patch_local_backward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor tile_offsets,
    torch::Tensor tile_gaussian_ids,
    torch::Tensor grad_output,
    int64_t axis_index,
    int64_t slice_idx,
    int64_t origin_h,
    int64_t origin_w,
    int64_t patch_h,
    int64_t patch_w,
    double spacing_z,
    double spacing_y,
    double spacing_x,
    int64_t tile_size);

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

std::vector<torch::Tensor> query_density_local_forward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor support_extent,
    torch::Tensor query_points,
    torch::Tensor grid_world_min,
    torch::Tensor grid_dims,
    double cell_size,
    torch::Tensor cell_offsets,
    torch::Tensor cell_gaussian_ids);

std::vector<torch::Tensor> query_density_local_backward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor query_points,
    torch::Tensor query_offsets,
    torch::Tensor query_gaussian_ids,
    torch::Tensor grad_output);

std::vector<torch::Tensor> query_density_qcut_local_forward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor support_extent,
    torch::Tensor query_points,
    torch::Tensor grid_world_min,
    torch::Tensor grid_dims,
    double cell_size,
    torch::Tensor cell_offsets,
    torch::Tensor cell_gaussian_ids,
    double q_cut);

std::vector<torch::Tensor> query_density_qcut_local_backward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor query_points,
    torch::Tensor query_offsets,
    torch::Tensor query_gaussian_ids,
    torch::Tensor grad_output);

std::vector<torch::Tensor> query_bulk_intensity_local_forward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor attenuation,
    torch::Tensor center_sdf,
    torch::Tensor center_normals,
    torch::Tensor material_membership,
    torch::Tensor support_extent,
    torch::Tensor query_points,
    torch::Tensor grid_world_min,
    torch::Tensor grid_dims,
    double cell_size,
    torch::Tensor cell_offsets,
    torch::Tensor cell_gaussian_ids,
    double q_cut,
    double tau,
    double skip_depth,
    bool apply_gate,
    bool has_membership);

std::vector<torch::Tensor> query_bulk_intensity_local_backward_cuda(
    torch::Tensor means,
    torch::Tensor rotations,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor attenuation,
    torch::Tensor center_sdf,
    torch::Tensor center_normals,
    torch::Tensor material_membership,
    torch::Tensor query_points,
    torch::Tensor query_offsets,
    torch::Tensor query_gaussian_ids,
    torch::Tensor grad_raw,
    torch::Tensor grad_den,
    double q_cut,
    double tau,
    double skip_depth,
    bool apply_gate,
    bool has_membership);

std::vector<torch::Tensor> build_uniform_grid_cuda(
    torch::Tensor cell_min,
    torch::Tensor cell_max,
    torch::Tensor grid_dims);

std::vector<torch::Tensor> sample_boundary_field_forward_cuda(
    torch::Tensor strength_volume,
    torch::Tensor normal_volume,
    torch::Tensor query_points,
    double spacing_z,
    double spacing_y,
    double spacing_x);

torch::Tensor sample_boundary_field_backward_cuda(
    torch::Tensor strength_volume,
    torch::Tensor normal_volume,
    torch::Tensor query_points,
    torch::Tensor grad_strength,
    torch::Tensor grad_normals,
    double spacing_z,
    double spacing_y,
    double spacing_x);

torch::Tensor surface_thickness_loss_forward_cuda(
    torch::Tensor raw_scaling,
    torch::Tensor rotation_mats,
    torch::Tensor normals,
    double max_thickness);

std::vector<torch::Tensor> surface_thickness_loss_backward_cuda(
    torch::Tensor raw_scaling,
    torch::Tensor rotation_mats,
    torch::Tensor normals,
    double max_thickness,
    torch::Tensor grad_output);

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
        "render_slice_patch_local_forward",
        &render_slice_patch_local_forward_cuda,
        "CT native local slice patch render forward (CUDA)");
    m.def(
        "render_slice_patch_local_backward",
        &render_slice_patch_local_backward_cuda,
        "CT native local slice patch render backward (CUDA)");
    m.def(
        "query_density_forward",
        &query_density_forward_cuda,
        "CT native density query forward (CUDA)");
    m.def(
        "query_density_backward",
        &query_density_backward_cuda,
        "CT native density query backward (CUDA)");
    m.def(
        "query_density_local_forward",
        &query_density_local_forward_cuda,
        "CT native local density query forward (CUDA)");
    m.def(
        "query_density_local_backward",
        &query_density_local_backward_cuda,
        "CT native local density query backward (CUDA)");
    m.def(
        "query_density_qcut_local_forward",
        &query_density_qcut_local_forward_cuda,
        "CT native local q-cut density query forward (CUDA)");
    m.def(
        "query_density_qcut_local_backward",
        &query_density_qcut_local_backward_cuda,
        "CT native local q-cut density query backward (CUDA)");
    m.def(
        "query_bulk_intensity_local_forward",
        &query_bulk_intensity_local_forward_cuda,
        "CT native local bulk intensity query forward (CUDA)");
    m.def(
        "query_bulk_intensity_local_backward",
        &query_bulk_intensity_local_backward_cuda,
        "CT native local bulk intensity query backward (CUDA)");
    m.def(
        "build_uniform_grid_cuda",
        &build_uniform_grid_cuda,
        "Build a CT native uniform grid over Gaussian support AABBs (CUDA)");
    m.def(
        "sample_boundary_field_forward",
        &sample_boundary_field_forward_cuda,
        "CT native boundary field sampling forward (CUDA)");
    m.def(
        "sample_boundary_field_backward",
        &sample_boundary_field_backward_cuda,
        "CT native boundary field sampling backward (CUDA)");
    m.def(
        "surface_thickness_loss_forward",
        &surface_thickness_loss_forward_cuda,
        "CT surface thickness loss forward");
    m.def(
        "surface_thickness_loss_backward",
        &surface_thickness_loss_backward_cuda,
        "CT surface thickness loss backward");
}
