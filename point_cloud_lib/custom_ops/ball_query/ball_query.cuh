/////////////////////////////////////////////////////////////////////////////
/// \file ball_query.cuh
///
/// \brief Declaraion of the CUDA operations to compute a ball query.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#ifndef _BALL_QUERY_CUH_
#define _BALL_QUERY_CUH_

#include <torch/extension.h>
    
/**
 *  Method to compute the point projection into an mlp basis.
 *  @param  p_pt_src            Source point cloud.
 *  @param  p_pt_dest           Destination point cloud.
 *  @param  p_batch_ids_src     Source batch ids.
 *  @param  p_batch_ids_dest    Destination batch ids.
 *  @param  p_min_pt            Minimum point of point cloud.
 *  @param  p_num_cells         Number of cells.
 *  @param  p_radius            Radius.
 *  @param  p_max_neighbors     Maximum neighbors.
 *  @return     Tensor with neighbor list and start ids.
 */
std::vector<torch::Tensor> ball_query(
    torch::Tensor& p_pt_src,
    torch::Tensor& p_pt_dest,
    torch::Tensor& p_batch_ids_src,
    torch::Tensor& p_batch_ids_dest,
    torch::Tensor& p_min_pt,
    torch::Tensor& p_num_cells,
    torch::Tensor& p_radius,
    int p_max_neighbors);
#endif