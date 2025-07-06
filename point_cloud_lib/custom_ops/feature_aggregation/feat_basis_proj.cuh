/////////////////////////////////////////////////////////////////////////////
/// \file feat_basis_proj.cuh
///
/// \brief Declaraion of the CUDA operations to compute the projection of
///     the features into the basis.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#ifndef _FEAT_BASIS_PROJ_CUH_
#define _FEAT_BASIS_PROJ_CUH_

#include <torch/extension.h>
    
/**
 *  Method to compute the point projection into an mlp basis.
 *  @param  p_pt_basis      Point basis tensor.
 *  @param  p_pt_features   Point feature tensor.
 *  @param  p_neighbors     Neighbor tensor.
 *  @param  p_start_ids     Start ids tensor.
 *  @return     Tensor with the feature projection.
 */
torch::Tensor feat_basis_proj(
    torch::Tensor& p_pt_basis,
    torch::Tensor& p_pt_features,    
    torch::Tensor& p_neighbors,
    torch::Tensor& p_start_ids);

#endif