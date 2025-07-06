/////////////////////////////////////////////////////////////////////////////
/// \file knn_query.cuh
///
/// \brief Declaraion of the CUDA operations to compute a knn query.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#ifndef _KNN_QUERY_CUH_
#define _KNN_QUERY_CUH_

#include <torch/extension.h>
    
/**
 *  Method to compute the point projection into an mlp basis.
 *  @param  p_pt_src            Source point cloud.
 *  @param  p_batch_ids_src     Source batch ids.
 *  @param  p_k                 K in knn.
 *  @return     Tensor with neighbor list.
 */
torch::Tensor knn_query(
    torch::Tensor& p_pt_src,
    torch::Tensor& p_batch_ids_src,
    int p_k);

#endif