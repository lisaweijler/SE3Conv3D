/////////////////////////////////////////////////////////////////////////////
/// \file count_neighbors.cu
///
/// \brief Implementation of the CUDA operations to count the neighbors for 
///         each point.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#include "../utils.cuh"
#include "ball_query_utils.cuh"
#include "math_helper.cuh"

#include "count_neighbors.cuh"

#define NUM_THREADS_X_RANGE 16

///////////////////////// GPU

/**
 *  GPU kernel to count the number of neighbors for each sample.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumRanges      Number of ranges per sample.
 *  @param  pSamples        3D coordinates of each sample.
 *  @param  pPts            3D coordinates of each point.
 *  @param  pRanges         Search ranges for each sample.
 *  @param  pInvRadii       Inverse of the radius used on the 
 *      search of neighbors in each dimension.
 *  @param  pOutNumNeighs   Number of neighbors for each sample.
 *  @tparam D               Number of dimensions
 */
 template<int D>
 __global__ void count_neighbors_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumRanges,
    const float* __restrict__ pSamples,
    const float* __restrict__ pPts,
    const int2* __restrict__ pRanges,
    const float* __restrict__ pInvRadii,
    int* __restrict__ pOutNumNeighs)
{
    //Create the pointers.
    const fpoint<D>* samplerPtr = (const fpoint<D>*)pSamples;
    const fpoint<D>* ptsPtr = (const fpoint<D>*)pPts;
    fpoint<D> invRadii(pInvRadii);

    //Declare shared memory.
    extern __shared__ int localCounter[];

    //Get the global thread index.
    int initSampleIndex = compute_global_index_gpu_funct();
    int totalThreads = compute_total_threads_gpu_funct();

    for(long long curIter = initSampleIndex; 
        curIter < pNumSamples*pNumRanges*NUM_THREADS_X_RANGE; 
        curIter += totalThreads)
    {        
        //Get the point id.
        int sampleIndex = curIter/(pNumRanges*NUM_THREADS_X_RANGE);
        
        //Initialize the shared memory
        localCounter[threadIdx.x] = sampleIndex;
        localCounter[blockDim.x + threadIdx.x] = 0;

        __syncthreads();

        //Get the offset and index of the local counter.
        int localIndex = curIter%NUM_THREADS_X_RANGE;
        int offsetCounter = sampleIndex-localCounter[0];

        //Get the current sample coordinates and the search range.
        fpoint<D> curSampleCoords = samplerPtr[sampleIndex];
        int2 curRange = pRanges[curIter/NUM_THREADS_X_RANGE];

        //Iterate over the points.
        for(int curPtIter = curRange.x+localIndex; 
            curPtIter < curRange.y; curPtIter+=NUM_THREADS_X_RANGE)
        {
            //Check if the point is closer than the selected radius.
            fpoint<D> curPtCoords = ptsPtr[curPtIter];

            if(length((curSampleCoords - curPtCoords)*invRadii) < 1.0f){
                //Increment the shared counters.
                atomicAdd(&localCounter[blockDim.x+offsetCounter], 1);
            }
        }

        __syncthreads();

        //Update the global counters.
        if(threadIdx.x == 0){
            atomicAdd(&pOutNumNeighs[sampleIndex], localCounter[blockDim.x]);
        }else if(sampleIndex != localCounter[threadIdx.x-1]){
            atomicAdd(&pOutNumNeighs[sampleIndex], localCounter[blockDim.x+offsetCounter]);
        }

        __syncthreads();
    }
}

///////////////////////// CPU

torch::Tensor count_neighbors(
    const torch::Tensor& pSamples,
    const torch::Tensor& pPts,
    const torch::Tensor& pRanges,
    const torch::Tensor& pInvRadii)
{
    // Get the number of dimensions and points.
    int numDims = pSamples.size(1);
    int numRanges = pRanges.size(1);
    int numSamples = pSamples.size(0);
    int numPts = pPts.size(0);

    // Get device properties.
    cudaDeviceProp props = get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    DIMENSION_FUNCT_PTR(numDims, count_neighbors_gpu_kernel, funcPtr);

    // Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;
    unsigned int numBlocks = get_max_active_block_x_sm(
        blockSize, funcPtr, blockSize*2*sizeof(int));

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = (numSamples*numRanges*NUM_THREADS_X_RANGE)/blockSize;
    execBlocks += ((numSamples*numRanges*NUM_THREADS_X_RANGE)%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    // Generate count neighbors tensor.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kInt32)
        .device(pPts.device().type(), pPts.device().index());
    auto countTensor = torch::zeros({numSamples}, tensorOptions);

    // Call the cuda kernel.
    DIMENSION_SWITCH_CALL(numDims, count_neighbors_gpu_kernel,
        totalNumBlocks, blockSize, blockSize*2*sizeof(int),
        numSamples, numRanges,
        (const float*)pSamples.data_ptr(),
        (const float*)pPts.data_ptr(),
        (const int2*)pRanges.data_ptr(),
        (const float*)pInvRadii.data_ptr(),
        (int*)countTensor.data_ptr());

    // Return result.
    return countTensor;
}
