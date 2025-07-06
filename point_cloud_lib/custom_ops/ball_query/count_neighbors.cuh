/////////////////////////////////////////////////////////////////////////////
/// \file count_neighbors.cuh
///
/// \brief Declaraion of the CUDA operations to count the neighbors for each
///         point.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#ifndef _COUNT_NEIGHBORS_CUH_
#define _COUNT_NEIGHBORS_CUH_

#include "ball_query_utils.cuh"

/**
 *  Method to count the number of neighbors.
 *  @param  pSamples    Samples tensor.
 *  @param  pPts        Points tensor.
 *  @param  pRanges     Ranges tensor.
 *  @param  pInvRadii   Inverse radii tensor.
 *  @return Number of neighbors per sample tensor.
 */
torch::Tensor count_neighbors(
    const torch::Tensor& pSamples,
    const torch::Tensor& pPts,
    const torch::Tensor& pRanges,
    const torch::Tensor& pInvRadii);
        
#endif