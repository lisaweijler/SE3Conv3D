/////////////////////////////////////////////////////////////////////////////
/// \file store_neighbors.cu
///
/// \brief Implementation of the CUDA operations to store the neighbors for 
///         each point.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#include "ball_query_utils.cuh"
#include "math_helper.cuh"
#include "../utils.cuh"
#include "grid_utils.cuh"
#include "rnd_utils.cuh"

#include "store_neighbors.cuh"

#include <iostream>

#define NUM_THREADS_X_RANGE 16

///////////////////////// GPU

/**
 *  GPU kernel to store the number of neighbors for each sample.
 *  @param  pSeed           Seed used to initialize the random state.
 *  @param  pMaxNeighbors   Maximum neighbors allowed per sample.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumRanges      Number of ranges per sample.
 *  @param  pSamples        3D coordinates of each sample.
 *  @param  pPts            3D coordinates of each point.
 *  @param  pRanges         Search ranges for each sample.
 *  @param  pInvRadii       Inverse of the radius used on the 
 *      search of neighbors in each dimension.
 *  @param  pOutNumNeighsU  Number of neighbors for each sample
 *      without the limit imposed by pMaxNeighbors.
 *  @param  pAuxCounter     Auxiliar counter.
 *  @param  pOutNumNeighs   Number of neighbors for each sample.
 *  @param  pOutNeighs      Final beighbors.
 *  @tparam D               Number of dimensions.
 */
 template<int D>
 __global__ void store_neighbors_limited_gpu_kernel(
    const int pSeed,
    const int pMaxNeighbors,
    const unsigned int pNumSamples,
    const unsigned int pNumRanges,
    const float* __restrict__ pSamples,
    const float* __restrict__ pPts,
    const int2* __restrict__ pRanges,
    const float* __restrict__ pInvRadii,
    const int* __restrict__ pOutNumNeighsU,
    int* __restrict__ pAuxCounter,
    int* __restrict__ pOutNumNeighs,
    int2* __restrict__ pOutNeighs)
{
    // Get the pointers.
    const fpoint<D>* samples = (const fpoint<D>*)pSamples;
    const fpoint<D>* pts = (const fpoint<D>*)pPts;
    fpoint<D> invRadii(pInvRadii);

    // Get the global thread index.
    int initSampleIndex = compute_global_index_gpu_funct();
    int totalThreads = compute_total_threads_gpu_funct();

    // Initialize the random seed generator.
    int curSeed = wang_hash(pSeed+initSampleIndex);

    for(long long curIter = initSampleIndex; 
        curIter < pNumSamples*pNumRanges*NUM_THREADS_X_RANGE; 
        curIter += totalThreads)
    {        
        // Get the point id.
        int sampleIndex = curIter/(pNumRanges*NUM_THREADS_X_RANGE);

        // Get the offset and index of the local counter.
        int localIndex = curIter%NUM_THREADS_X_RANGE;

        // Get the current sample coordinates and the search range.
        fpoint<D> curSampleCoords = samples[sampleIndex];
        int2 curRange = pRanges[curIter/NUM_THREADS_X_RANGE];

        // Iterate over the points.
        for(int curPtIter = curRange.x+localIndex; 
            curPtIter < curRange.y; curPtIter+=NUM_THREADS_X_RANGE)
        {
            // Check if the point is closer than the selected radius.
            fpoint<D> curPtCoors = pts[curPtIter];
            if(length((curSampleCoords - curPtCoors)*invRadii) < 1.0f){
                // Increment the shared counters.
                int neighIndex = atomicAdd(&pAuxCounter[sampleIndex], 1);
                if(neighIndex < pMaxNeighbors){
                    neighIndex = pOutNumNeighs[sampleIndex] - neighIndex - 1;
                    pOutNeighs[neighIndex] = make_int2(curPtIter, sampleIndex);
                }else{
                    float ratio = (float)pMaxNeighbors/(float)pOutNumNeighsU[sampleIndex];
                    curSeed = rand_xorshift(curSeed);
                    float uniformVal = seed_to_float(curSeed);
                    if(uniformVal < ratio){
                        curSeed = rand_xorshift(curSeed);
                        uniformVal = seed_to_float(curSeed);
                        neighIndex = (int)floorf(uniformVal*((float)pMaxNeighbors))+1;
                        pOutNeighs[pOutNumNeighs[sampleIndex] - neighIndex] = 
                            make_int2(curPtIter, sampleIndex);
                    }
                }
            }
        }
    }
}

/**
 *  GPU kernel to store the number of neighbors for each sample.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumRanges      Number of ranges per sample.
 *  @param  pSamples        3D coordinates of each sample.
 *  @param  pPts            3D coordinates of each point.
 *  @param  pRanges         Search ranges for each sample.
 *  @param  pInvRadii       Inverse of the radius used on the 
 *      search of neighbors in each dimension.
 *  @param  pOutNumNeighs   Number of neighbors for each sample.
 *  @param  pOutNeighs      Final beighbors.
 *  @tparam D               Number of dimensions.
 */
 template<int D>
 __global__ void store_neighbors_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumRanges,
    const float* __restrict__ pSamples,
    const float* __restrict__ pPts,
    const int2* __restrict__ pRanges,
    const float* __restrict__ pInvRadii,
    int* __restrict__ pOutNumNeighs,
    int2* __restrict__ pOutNeighs)
{
    // Get the pointers.
    const fpoint<D>* samples = (const fpoint<D>*)pSamples;
    const fpoint<D>* pts = (const fpoint<D>*)pPts;
    fpoint<D> invRadii(pInvRadii);

    // Get the global thread index.
    int initSampleIndex = compute_global_index_gpu_funct();
    int totalThreads = compute_total_threads_gpu_funct();

    for(long long curIter = initSampleIndex; 
        curIter < pNumSamples*pNumRanges*NUM_THREADS_X_RANGE; 
        curIter += totalThreads)
    {        
        // Get the point id.
        int sampleIndex = curIter/(pNumRanges*NUM_THREADS_X_RANGE);

        // Get the offset and index of the local counter.
        int localIndex = curIter%NUM_THREADS_X_RANGE;

        // Get the current sample coordinates and the search range.
        fpoint<D> curSampleCoords = samples[sampleIndex];
        int2 curRange = pRanges[curIter/NUM_THREADS_X_RANGE];

        // Iterate over the points.
        for(int curPtIter = curRange.x+localIndex; 
            curPtIter < curRange.y; curPtIter+=NUM_THREADS_X_RANGE)
        {
            // Check if the point is closer than the selected radius.
            fpoint<D> curPtCoors = pts[curPtIter];
            if(length((curSampleCoords - curPtCoors)*invRadii) < 1.0f){
                // Increment the shared counters.
                int neighIndex = atomicAdd(&pOutNumNeighs[sampleIndex], 1);
                pOutNeighs[neighIndex] = make_int2(curPtIter, sampleIndex);
            }
        }
    }
}

///////////////////////// CPU

std::vector<torch::Tensor> store_neighbors(
    const int pMaxNeighbors,
    const torch::Tensor& pSamples,
    const torch::Tensor& pPts,
    const torch::Tensor& pRanges,
    const torch::Tensor& pInvRadii,
    torch::Tensor& pCounts)
{
    // Get device properties.
    cudaDeviceProp props = get_cuda_device_properties();

    // Get the number of dimensions and points.
    int numDims = pSamples.size(1);
    int numPts = pPts.size(0);
    int numSamples = pSamples.size(0);
    int numRanges = pRanges.size(1);

    // Get the function pointer.
    void* funcPtr = nullptr;
    if(pMaxNeighbors > 0){
        DIMENSION_FUNCT_PTR(numDims, store_neighbors_limited_gpu_kernel, funcPtr);
    }else{
        DIMENSION_FUNCT_PTR(numDims, store_neighbors_gpu_kernel, funcPtr);  
    }

    // Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;
    unsigned int numBlocks = get_max_active_block_x_sm(
        blockSize, funcPtr, 0);

    // Calculate the total number of blocks to execute.
    unsigned int execBlocks = (numSamples*numRanges*NUM_THREADS_X_RANGE)/blockSize;
    execBlocks += ((numSamples*numRanges*NUM_THREADS_X_RANGE)%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Prepare output.
    std::vector<torch::Tensor> output;

    if(pMaxNeighbors > 0){

        // Max number of neighbors
        torch::Tensor clampCounts = torch::clamp(pCounts, 0, (int)pMaxNeighbors);
        
        // Scan counts.
        torch::Tensor accumCounts = torch::cumsum(clampCounts, 0, torch::kInt32);
        int oldTotalNumNeighs = accumCounts[numSamples-1].item<int>(); //TODO -Making it slow in multigpu?
        int totalNumNeighs = pMaxNeighbors*numSamples;
        
        // Create the output tensor.
        auto tensorOptions = torch::TensorOptions().dtype(torch::kInt32).
            device(pSamples.device().type(), pSamples.device().index());
        auto outTensor = torch::ones({totalNumNeighs, 2}, tensorOptions)*-1;

        // Create a temporal tensor.
        auto tmpTensor = torch::zeros({numSamples}, tensorOptions);        

        // Execute the cuda kernel.
        DIMENSION_SWITCH_CALL(numDims, store_neighbors_limited_gpu_kernel,
            totalNumBlocks, blockSize, 0,
            time(NULL), pMaxNeighbors, numSamples, numRanges,
            (const float*)pSamples.data_ptr(),
            (const float*)pPts.data_ptr(),
            (const int2*)pRanges.data_ptr(),
            (const float*)pInvRadii.data_ptr(),
            (int*)pCounts.data_ptr(),
            (int*)tmpTensor.data_ptr(),
            (int*)accumCounts.data_ptr(),
            (int2*)outTensor.data_ptr());

        // Mask not valid neighbors.
        auto mask = outTensor >= 0;
        mask = torch::logical_and(mask.index({"...", 0}), mask.index({"...", 1}));
        outTensor = outTensor.index({mask, "..."});

        // Save the output.
        output.push_back(outTensor);        
        output.push_back(accumCounts);
            
    }else{

        // Scan counts.
        torch::Tensor accumCounts = torch::cumsum(pCounts, 0, torch::kInt32);
        int totalNumNeighs = accumCounts[numSamples-1].item<int>();//TODO -Making it slow in multigpu?
        accumCounts = accumCounts - pCounts;

        // Create the output tensor.
        auto tensorOptions = torch::TensorOptions().dtype(torch::kInt32).
            device(pSamples.device().type(), pSamples.device().index());
        auto outTensor = torch::zeros({totalNumNeighs, 2}, tensorOptions);

        // Execute the cuda kernel.
        DIMENSION_SWITCH_CALL(numDims, store_neighbors_gpu_kernel,
            totalNumBlocks, blockSize, 0,
            numSamples, numRanges,
            (const float*)pSamples.data_ptr(),
            (const float*)pPts.data_ptr(),
            (const int2*)pRanges.data_ptr(),
            (const float*)pInvRadii.data_ptr(),
            (int*)accumCounts.data_ptr(),
            (int2*)outTensor.data_ptr());

        // Save the output.
        output.push_back(outTensor);        
        output.push_back(accumCounts);
    }

    return output;
}
