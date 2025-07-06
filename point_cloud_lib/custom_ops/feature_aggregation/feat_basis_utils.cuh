/////////////////////////////////////////////////////////////////////////////
/// \file feat_basis_utils.h
///
/// \brief Definitions.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _FEAT_BASIS_UTILS_H_
#define _FEAT_BASIS_UTILS_H_

#include <memory>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include "../utils.cuh"

#define FEAT_FUNCT_PTR_CASE(Basis, Type, Func, FunctPtr)    \
    case Basis:                                             \
        if(Type==torch::ScalarType::Double)                 \
            FunctPtr = (void*)Func<Basis, double>;          \
        else if(Type==torch::ScalarType::Float)             \
            FunctPtr = (void*)Func<Basis, float>;           \
        else if(Type==torch::ScalarType::Half)              \
            FunctPtr = (void*)Func<Basis, torch::Half>;     \
        break;                                              \

#define FEAT_FUNCT_PTR(Basis, Type, Func, FunctPtr)     \
    switch(Basis){                                      \
        FEAT_FUNCT_PTR_CASE(8, Type, Func, FunctPtr)    \
        FEAT_FUNCT_PTR_CASE(16, Type, Func, FunctPtr)   \
        FEAT_FUNCT_PTR_CASE(32, Type, Func, FunctPtr)   \
        FEAT_FUNCT_PTR_CASE(64, Type, Func, FunctPtr)   \
    };

#define FEAT_FUNCT_CALL(Basis, Type, totalNumBlocks, blockSize, sharedMemSize, FunctNameStr, FunctName, ...)                            \
    switch(Basis){                                                                                                                      \
        case 8:                                                                                                                         \
            AT_DISPATCH_FLOATING_TYPES(Type, FunctNameStr, ([&] {                                                                       \
                FunctName<8, scalar_t><<<totalNumBlocks, blockSize, sharedMemSize, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__);    \
                }));                                                                                                                    \
            break;                                                                                                                      \
        case 16:                                                                                                                        \
            AT_DISPATCH_FLOATING_TYPES(Type, FunctNameStr, ([&] {                                                                       \
                FunctName<16, scalar_t><<<totalNumBlocks, blockSize, sharedMemSize, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__);   \
                }));                                                                                                                    \
            break;                                                                                                                      \
        case 32:                                                                                                                        \
            AT_DISPATCH_FLOATING_TYPES(Type, FunctNameStr, ([&] {                                                                       \
                FunctName<32, scalar_t><<<totalNumBlocks, blockSize, sharedMemSize, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__);   \
                }));                                                                                                                    \
            break;                                                                                                                      \
        case 64:                                                                                                                        \
            AT_DISPATCH_FLOATING_TYPES(Type, FunctNameStr, ([&] {                                                                       \
                FunctName<64, scalar_t><<<totalNumBlocks, blockSize, sharedMemSize, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__);   \
                }));                                                                                                                    \
            break;                                                                                                                      \
    };

#endif