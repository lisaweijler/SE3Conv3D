/////////////////////////////////////////////////////////////////////////////
/// \file store_neighbors.cuh
///
/// \brief Declaraion of the CUDA operations to store the neighbors for each
///         point.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#ifndef _STORE_NEIGHBORS_CUH_
#define _STORE_NEIGHBORS_CUH_

#include "ball_query_utils.cuh"

/**
 *  Method to store the number of neighbors.
 *  @param  pMaxNeighbors       Maximum number of neighbors. If zero or less, 
 *      there is not limit.
 *  @param  pNumRanges          Number of ranges per point.
 *  @param  pSamples            Tensor with the input samples.
 *  @param  pPts                Tensor with the input points..
 *  @param  pRanges             Input tensor with the search ranges.
 *  @param  pInvRadii           Inverse of the radius used on the 
 *      search of neighbors in each dimension.
 *  @param  pCounts             Input/Output tensor with the number 
 *      of neighbors for each sample without the limit of pMaxNeighbors.
 *  @return     Output tensor with the output neighbors and tensor with the
 *      starting ids.
 */
std::vector<torch::Tensor> store_neighbors(
    const int pMaxNeighbors,
    const torch::Tensor& pSamples,
    const torch::Tensor& pPts,
    const torch::Tensor& pRanges,
    const torch::Tensor& pInvRadii,
    torch::Tensor& pCounts);    

#endif