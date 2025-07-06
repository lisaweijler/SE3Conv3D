/////////////////////////////////////////////////////////////////////////////
/// \file knn_query.cu
///
/// \brief Implementation of the CUDA operations to compute a knn query.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#include "knn_query_utils.cuh"
#include "knn_query.cuh"

#include <iostream>

template <typename scalar_t, int D>
__global__ void fast_knn_kernel(
    const scalar_t *__restrict__ pts,
    const int *__restrict__ batch_ids, 
    int *__restrict__ out_ids,
    const int k, 
    const int num_pts,
    const int sort_d) {

    extern __shared__ unsigned char sharedMemory[];
    scalar_t* sharedMemoryBuff = reinterpret_cast<scalar_t*>(sharedMemory);

    scalar_t* cur_coords = &sharedMemoryBuff[(D + 128)*threadIdx.x];
    scalar_t* best_dist = &sharedMemoryBuff[D + (D + 128)*threadIdx.x];
    int* best_idx = reinterpret_cast<int*>(&sharedMemoryBuff[(D + 64) + (D + 128)*threadIdx.x]);

    int init_sample_index = compute_global_index_gpu_funct();
    int total_threads = compute_total_threads_gpu_funct();
    
    for(int curIter = init_sample_index; 
        curIter < num_pts; 
        curIter += total_threads)
    {   
        const int n_y = curIter;

        const int cur_batch_id = batch_ids[n_y];

        for (int e = 0; e < k; e++) {
            best_dist[e] = 1e10;
            best_idx[e] = -1;
        }

    #pragma unroll
        for (int d = 0; d < D; d++)
            cur_coords[d] = pts[n_y * D + d];

        // Iterate positive increment.
        bool stop_iter = false;
        int cur_incr = 0;
        while (!stop_iter) {
            int n_x = n_y + cur_incr;

            if(n_x >= num_pts || cur_batch_id != batch_ids[n_x]){
                stop_iter = true;
            }else{

                scalar_t tmp_dist = 0;
    #pragma unroll
                for (int d = 0; d < D; d++) {
                    scalar_t diff_coord = pts[n_x * D + d] - cur_coords[d];
                    tmp_dist += diff_coord*diff_coord;
                }

                for (int e1 = 0; e1 < k; e1++) {
                    if (best_dist[e1] > tmp_dist) {
                        for (int e2 = k - 1; e2 > e1; e2--) {
                            best_dist[e2] = best_dist[e2 - 1];
                            best_idx[e2] = best_idx[e2 - 1];
                        }
                        best_dist[e1] = tmp_dist;
                        best_idx[e1] = n_x;
                        break;
                    }
                }

                scalar_t sort_dist = pts[n_x * D + sort_d] - cur_coords[sort_d];
                if(best_dist[k-1] < sort_dist*sort_dist)
                    stop_iter = true;

                cur_incr += 1;
            }
        }

        // Iterate negative increment.
        stop_iter = false;
        cur_incr = 1;
        while (!stop_iter) {
            int n_x = n_y - cur_incr;

            if(n_x < 0 || cur_batch_id != batch_ids[n_x]){
                stop_iter = true;
            }else{

                scalar_t tmp_dist = 0;
    #pragma unroll
                for (int d = 0; d < D; d++) {
                    scalar_t diff_coord = pts[n_x * D + d] - cur_coords[d];
                    tmp_dist += diff_coord*diff_coord;
                }
                
                for (int e1 = 0; e1 < k; e1++) {
                    if (best_dist[e1] > tmp_dist) {
                        for (int e2 = k - 1; e2 > e1; e2--) {
                            best_dist[e2] = best_dist[e2 - 1];
                            best_idx[e2] = best_idx[e2 - 1];
                        }
                        best_dist[e1] = tmp_dist;
                        best_idx[e1] = n_x;
                        break;
                    }
                }

                scalar_t sort_dist = pts[n_x * D + sort_d] - cur_coords[sort_d];
                if(best_dist[k-1] < sort_dist*sort_dist)
                    stop_iter = true;

                cur_incr += 1;
            }
        }

        for (int e = 0; e < k; e++) {
            out_ids[n_y * k + e] = best_idx[e];
        }
    }
}


torch::Tensor knn_query(
    torch::Tensor& p_pt_src,
    torch::Tensor& p_batch_ids_src,
    int p_k)
{
    cudaDeviceProp props = get_cuda_device_properties();

    int num_dims = p_pt_src.size(1);
    int num_pts = p_pt_src.size(0);

    auto max_coord = torch::amax(p_pt_src, 0);
    auto min_coord = torch::amin(p_pt_src, 0);

    auto coord_diff = max_coord - min_coord;
    int sort_dim = torch::argmax(coord_diff).item().toInt();

    auto sorted_ids = torch::argsort(p_pt_src.index({"...", sort_dim}) - min_coord.index({sort_dim})
        + p_batch_ids_src*coord_diff.index({sort_dim})).to(torch::kInt32);

    auto sorted_pts = p_pt_src.index({sorted_ids}).contiguous();
    auto sorted_batch_ids = p_batch_ids_src.index({sorted_ids}).contiguous();

    auto tensor_options = torch::TensorOptions().dtype(torch::kInt32).
        device(sorted_pts.device().type(), sorted_pts.device().index());
    auto out_tensor = torch::zeros({sorted_pts.size(0), p_k}, tensor_options)-1;

    void* func_ptr = nullptr;
    KNN_FUNCT_PTR(num_dims, sorted_pts.scalar_type(), fast_knn_kernel, func_ptr);

    // Calculate the ideal number of blocks for the selected block size.
    unsigned int num_MP = props.multiProcessorCount;
    unsigned int block_size = props.warpSize*2;
    unsigned int sharedMemSize = block_size*torch::elementSize(sorted_pts.scalar_type())*(num_dims+128);
    unsigned int num_blocks = get_max_active_block_x_sm(
        block_size, func_ptr, sharedMemSize);

    // Calculate the total number of blocks to execute.
    unsigned int exec_blocks = num_pts/block_size;
    exec_blocks += (num_pts%block_size != 0)?1:0;
    unsigned int total_num_blocks = num_MP*num_blocks;
    total_num_blocks = (total_num_blocks > exec_blocks)?exec_blocks:total_num_blocks;
     
    KNN_FUNCT_CALL( num_dims, 
                    sorted_pts.scalar_type(),
                    total_num_blocks, block_size, sharedMemSize,
                    "fast_knn_kernel", fast_knn_kernel,
                    (const scalar_t*)sorted_pts.data_ptr(),
                    (const int*)sorted_batch_ids.data_ptr(), 
                    (int*)out_tensor.data_ptr(),
                    (int)p_k, 
                    (int)num_pts,
                    (int)sort_dim); 

    auto new_out_tensor = sorted_ids.index({out_tensor});
    out_tensor = (out_tensor < 0)*-1 + (out_tensor >= 0)*new_out_tensor;
    
    auto out_tensor_2 = torch::zeros({sorted_pts.size(0), p_k}, tensor_options)-1;
    out_tensor_2.scatter_(0, 
        sorted_ids.to(torch::kInt64).reshape({-1, 1}).repeat({1, p_k}), 
        out_tensor.to(torch::kInt32));

    return out_tensor_2;
}