/////////////////////////////////////////////////////////////////////////////
/// \file ball_query.cu
///
/// \brief Implementation of the CUDA operations to compute a ball query.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#include "ball_query.cuh"
#include "compute_keys.cuh"
#include "build_grid_ds.cuh"
#include "find_ranges_grid_ds.cuh"
#include "count_neighbors.cuh"
#include "store_neighbors.cuh"

#include <iostream>

std::vector<torch::Tensor> ball_query(
    torch::Tensor& p_pt_src,
    torch::Tensor& p_pt_dest,
    torch::Tensor& p_batch_ids_src,
    torch::Tensor& p_batch_ids_dest,
    torch::Tensor& p_min_pt,
    torch::Tensor& p_num_cells,
    torch::Tensor& p_radius,
    int p_max_neighbors)
{
    // Compute grid keys for the source point cloud.
    auto pts_keys = compute_keys(
        p_pt_src, 
        p_batch_ids_src, 
        p_min_pt, 
        p_num_cells,
        p_radius);
    auto sorted_ids = torch::argsort(pts_keys);
    auto sorted_pts_keys = torch::index_select(
        pts_keys, 0, sorted_ids.to(torch::kInt64));
    auto sorted_pts = torch::index_select(
        p_pt_src, 0, sorted_ids.to(torch::kInt64));

    // Build search grid for the source point cloud.
    int max_batch_id = torch::amax(p_batch_ids_src, 0).item<int>()+1;
    std::vector<int64_t> out_shape = {
        max_batch_id, 
        p_num_cells[0].item<int>(),
        p_num_cells[1].item<int>(),
        2};
    auto grid_ds = build_grid_ds(
        sorted_pts_keys,
        p_num_cells,
        out_shape);
    
    // Compute grid keys for the samples.
    auto sample_keys = compute_keys(
        p_pt_dest, 
        p_batch_ids_dest, 
        p_min_pt, 
        p_num_cells,
        p_radius);

    // Find the neighbors.
    // Get the number of dimensions and points.
    int num_dims = p_num_cells.size(0);
    int num_pts = p_pt_src.size(0);
    int num_samples = p_pt_dest.size(0);

    // Compute the number of offsets to used in the search.
    std::vector<int> comb_offsets;
    unsigned int num_offsets = compute_total_num_offsets(
        num_dims, 1, comb_offsets);

    // Upload offsets to gpu.
    auto tensor_options = torch::TensorOptions().dtype(torch::kInt32);
    torch::Device cuda_device(p_pt_src.device().type(), p_pt_src.device().index());
    auto offset_tensor = torch::from_blob(&comb_offsets[0], 
        {num_offsets, num_dims}, tensor_options).to(cuda_device);

    // Find ranges.
    auto out_ranges = find_ranges_grid_ds_gpu(
        1, offset_tensor, sample_keys, sorted_pts_keys,
        p_num_cells, grid_ds);

    // Count neighbors.
    auto inv_radius = torch::reciprocal(p_radius);
    auto counts = count_neighbors(p_pt_dest, sorted_pts, out_ranges, inv_radius);

    // Store neighbors and return result.
    auto results_store_op = store_neighbors(
        p_max_neighbors, p_pt_dest, sorted_pts,
        out_ranges, inv_radius, counts);

    // Transform to unsorted ids.
    auto unsorted_neigh_ids = torch::index_select(
        sorted_ids, 0, results_store_op[0].index({"...", 0}).to(torch::kInt64));
    auto final_neighbors = torch::cat({
        results_store_op[0].index({"...", 1}).reshape({-1, 1}),
        unsorted_neigh_ids.reshape({-1, 1})}, -1);

    return {final_neighbors, results_store_op[1]};
}