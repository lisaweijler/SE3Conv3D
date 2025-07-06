/////////////////////////////////////////////////////////////////////////////
/// \file ops_list.cpp
///
/// \brief Declaration of all operations in the module. 
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#include "./feature_aggregation/feat_basis_proj.cuh"
#include "./feature_aggregation/feat_basis_proj_grads.cuh"
#include "./ball_query/ball_query.cuh"
#include "./knn_query/knn_query.cuh"
#include "./ball_query/compute_keys.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("feat_basis_proj", &feat_basis_proj, "Project features into basis functions");
    m.def("feat_basis_proj_grad", &feat_basis_proj_grads, "Gradients of the project features into basis functions");
    m.def("ball_query", &ball_query, "Compute a ball query");
    m.def("knn_query", &knn_query, "Compute a knn query");
    m.def("compute_keys", &compute_keys, "Compute keys voxel");
}