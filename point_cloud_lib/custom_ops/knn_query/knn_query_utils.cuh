/////////////////////////////////////////////////////////////////////////////
/// \file knn_query_utils.h
///
/// \brief Definitions.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _KNN_QUERY_UTILS_H_
#define _KNN_QUERY_UTILS_H_

#include <memory>
#include <torch/torch.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include "../utils.cuh"

#define KNN_FUNCT_PTR_CASE(Dim, Type, Func, FunctPtr)       \
    case Dim:                                               \
        if(Type==torch::ScalarType::Double)                 \
            FunctPtr = (void*)Func<double, Dim>;            \
        else if(Type==torch::ScalarType::Float)             \
            FunctPtr = (void*)Func<float, Dim>;             \
        else if(Type==torch::ScalarType::Half)              \
            FunctPtr = (void*)Func<torch::Half, Dim>;       \
        break;                                              \

#define KNN_FUNCT_PTR(Dim, Type, Func, FunctPtr)        \
    switch(Dim){                                        \
        KNN_FUNCT_PTR_CASE(2, Type, Func, FunctPtr)     \
        KNN_FUNCT_PTR_CASE(3, Type, Func, FunctPtr)     \
    };

#define KNN_FUNCT_CALL(Dim, Type, totalNumBlocks, blockSize, sharedMemSize, FunctNameStr, FunctName, ...)                               \
    switch(Dim){                                                                                                                        \
        case 2:                                                                                                                         \
            AT_DISPATCH_FLOATING_TYPES(Type, FunctNameStr, ([&] {                                                                       \
                FunctName<scalar_t, 2><<<totalNumBlocks, blockSize, sharedMemSize, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__);    \
                }));                                                                                                                    \
            break;                                                                                                                      \
        case 3:                                                                                                                         \
            AT_DISPATCH_FLOATING_TYPES(Type, FunctNameStr, ([&] {                                                                       \
                FunctName<scalar_t, 3><<<totalNumBlocks, blockSize, sharedMemSize, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__);    \
                }));                                                                                                                    \
            break;                                                                                                                      \
    };

#endif