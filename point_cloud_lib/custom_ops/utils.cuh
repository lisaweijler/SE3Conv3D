/////////////////////////////////////////////////////////////////////////////
/// \file utils.cuh
///
/// \brief Declaraion of the utils CUDA operations.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#ifndef _UTILS_CUH_
#define _UTILS_CUH_

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include "cuda_runtime.h"
#include "cuda_fp16.h"

//Definition of the min and max operation for cuda code.
#define MAX_OP(a, b) (a < b) ? b : a;
#define MIN_OP(a, b) (a > b) ? b : a;

/**
 *  Method to get the properties of the device.
 *  @return Cuda device properties.
 */
__forceinline__ cudaDeviceProp get_cuda_device_properties()
{
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    return prop;
};


/**
 *  Get the maximum number of active blocks per sm.
 *  @param  p_block_size            Size of the block.
 *  @param  p_kernel                Kernel.
 *  @param  p_shared_mem_x_block    Dynamic shared memory per block.
 */
__forceinline__ int get_max_active_block_x_sm(
        const unsigned int p_block_size, 
        const void* p_kernel,
        const size_t p_shared_mem_x_block)
{
    int outputNumBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor ( 
        &outputNumBlocks, p_kernel, p_block_size, p_shared_mem_x_block);
    return outputNumBlocks;
};

/**
*  Function to compute the global index of the current thread.
*  @return   Current thread index.
*/
__device__ __forceinline__ unsigned long long int compute_global_index_gpu_funct()
{
    return threadIdx.x + blockDim.x*blockIdx.x;
};

/**
*  Function to compute the total number of threads in execution..
*  @return   Total number of threads.
*/
__device__ __forceinline__ unsigned long long int compute_total_threads_gpu_funct()
{
    return gridDim.x*blockDim.x;
};

/**
*  Function to do an atomic max operation on floats.
*  @param  pAddress    Address in which we want to perform the atomic operation.
*  @param  pVal        Value we want to input.
*  @return Stored value.
*/
__device__ static float atomicMax(float* pAddress, const float pVal)
{
    int* address_as_i = (int*) pAddress;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(pVal, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
};

#endif