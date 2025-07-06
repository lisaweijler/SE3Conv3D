/////////////////////////////////////////////////////////////////////////////
/// \file find_ranges_grid_ds.cuh
///
/// \brief Declaraion of the CUDA operations to find the ranges in the list
///     of points for a grid cell and its 26 neighbors.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#ifndef _BUILD_GRID_DS_CUH_
#define _BUILD_GRID_DS_CUH_

#include "ball_query_utils.cuh"

/**
 *  Method to find the ranges in the list of points for a grid cell 
 *  and its N neighbors.
 *  @param  pLastDOffsets    Number of displacement in the last
 *      dimension in the positive and negative axis.
 *  @param  pOffsets         Tensor with the offsets.
 *  @param  pSampleKeys      Tensor with the sample keys.
 *  @param  pPtKeys          Tensor with the point keys.
 *  @param  pGridSize        Tensor with the grid size.
 *  @param  pGridDS          Tensor with the grid data structure.
 *  @return Output tensor to the array containing
 *      the search ranges for each sample. 
 */

torch::Tensor find_ranges_grid_ds_gpu(
    const unsigned int pLastDOffsets,
    const torch::Tensor& pOffsets,
    const torch::Tensor& pSampleKeys,
    const torch::Tensor& pPtKeys,
    const torch::Tensor& pGridSize,
    const torch::Tensor& pGridDS);

/**
 *  Method to compute the total number of offsets
 *  to apply for each range search.
 *  @param  pNumDimensions  Number of dimensions.
 *  @param  pAxisOffset     Offset apply to each axis.
 *  @param  pOutVector      Output parameter with the 
 *      displacements applied to each axis.
 */
unsigned int compute_total_num_offsets(
    const unsigned int pNumDimensions,
    const unsigned int pAxisOffset,
    std::vector<int>& pOutVector);

#endif