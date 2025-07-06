/////////////////////////////////////////////////////////////////////////////
/// \file defines.h
///
/// \brief Definitions.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef BALL_QUERY_UTILS_H_
#define BALL_QUERY_UTILS_H_

#include <memory>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

//Definition of the minimum and maximum number of dimensions.
#define MIN_DIMENSIONS 2
#define MAX_DIMENSIONS 6                                           

#define DIMENSION_CASE_SWITCH(Dim, Func, NumBlocks, NumThreads, SharedMem, ...)                         \
    case Dim:                                                                                           \
        Func<Dim><<<NumBlocks, NumThreads, SharedMem, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        break;

#define DIMENSION_SWITCH_CALL(Dims, Func, NumBlocks, NumThreads, SharedMem, ...)        \
    switch(Dims){                                                                       \
        DIMENSION_CASE_SWITCH(2, Func, NumBlocks, NumThreads, SharedMem, __VA_ARGS__)   \
        DIMENSION_CASE_SWITCH(3, Func, NumBlocks, NumThreads, SharedMem, __VA_ARGS__)   \
        DIMENSION_CASE_SWITCH(4, Func, NumBlocks, NumThreads, SharedMem, __VA_ARGS__)   \
        DIMENSION_CASE_SWITCH(5, Func, NumBlocks, NumThreads, SharedMem, __VA_ARGS__)   \
        DIMENSION_CASE_SWITCH(6, Func, NumBlocks, NumThreads, SharedMem, __VA_ARGS__)   \
    }  

#define DIMENSION_FUNCT_PTR(Dims, Func, FunctPtr)   \
    switch(Dims){                                   \
        case 2:                                     \
            FunctPtr = (void*)Func<2>;              \
            break;                                  \
        case 3:                                     \
            FunctPtr = (void*)Func<3>;              \
            break;                                  \
        case 4:                                     \
            FunctPtr = (void*)Func<4>;              \
            break;                                  \
        case 5:                                     \
            FunctPtr = (void*)Func<5>;              \
            break;                                  \
        case 6:                                     \
            FunctPtr = (void*)Func<6>;              \
            break;                                  \
    };

//Definition of the min and max operation for cuda code.
#define PC_MAX(a, b) (a < b) ? b : a;
#define PC_MIN(a, b) (a > b) ? b : a;

//Definition of the int 64 bit.
typedef long long int64_m;

#endif