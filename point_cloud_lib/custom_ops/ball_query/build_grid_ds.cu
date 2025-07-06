/////////////////////////////////////////////////////////////////////////////
/// \file build_grid_ds.cu
///
/// \brief Implementation of the CUDA operations to build the data structure to 
///     access the sparse regular grid. 
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#include "ball_query_utils.cuh"
#include "math_helper.cuh"
#include "grid_utils.cuh"
#include "../utils.cuh"

#include "build_grid_ds.cuh"

///////////////////////// GPU
 
/**
 *  GPU kernel to compute the grid data structure.
 *  @param  pNumPts         Number of points.
 *  @param  pKeys           Array of keys.
 *  @param  pNumCells       Number of cells.
 *  @param  pOutDS          Output array with the data structure.
 *  @paramT D                       Number of dimensions.
 */
 template<int D>
 __global__ void build_grid_gpu_kernel(
    const unsigned int pNumPts,
    const int64_m* __restrict__ pKeys,
    const int* __restrict__ pNumCells,
    int2* __restrict__ pOutDS)
{
    const ipoint<D>* numCellsNewPtr = (const ipoint<D>*)pNumCells;

    int initPtIndex = compute_global_index_gpu_funct();
    int totalThreads = compute_total_threads_gpu_funct();

    for(int curPtIndex = initPtIndex; curPtIndex < pNumPts; curPtIndex += totalThreads)
    {
        //Get the key and compute the index into the ds.
        int64_m curKey = pKeys[curPtIndex];
        int dsIndex = compute_ds_index_from_key_gpu_funct(curKey, numCellsNewPtr[0]);
        
        //Check if it is the first point in the ds cell.
        int prevPtIndex = curPtIndex-1;
        if(prevPtIndex >= 0){
            if(dsIndex != 
                compute_ds_index_from_key_gpu_funct(pKeys[prevPtIndex], numCellsNewPtr[0])){
                    pOutDS[dsIndex].x = curPtIndex;
            }
        }

        //Check if it is the last point in the ds cell.
        int nextPtIndex = curPtIndex+1;
        if(nextPtIndex == pNumPts){
            pOutDS[dsIndex].y = pNumPts;
        }else if(dsIndex != 
            compute_ds_index_from_key_gpu_funct(pKeys[nextPtIndex], numCellsNewPtr[0])){
            pOutDS[dsIndex].y = nextPtIndex;
        }
    }
}

///////////////////////// CPU

torch::Tensor build_grid_ds(
    torch::Tensor pKeys,
    torch::Tensor pGridSize,
    std::vector<int64_t>& pOutShape) {

    // Get the number of dimensions and points.
    int numDims = pGridSize.size(0);
    int numPts = pKeys.size(0);

    // Get device properties.
    cudaDeviceProp props = get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    DIMENSION_FUNCT_PTR(numDims, build_grid_gpu_kernel, funcPtr);

    // Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;
    unsigned int numBlocks = get_max_active_block_x_sm(
        blockSize, funcPtr, 0);

    // Calculate the total number of blocks to execute.
    unsigned int execBlocks = numPts/blockSize;
    execBlocks += (numPts%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    // Create output.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kInt32).
        device(pKeys.device().type(), pKeys.device().index());
    auto outTensor = torch::zeros(pOutShape, tensorOptions);

    // Call the cuda kernel.
    DIMENSION_SWITCH_CALL(numDims, build_grid_gpu_kernel, totalNumBlocks, blockSize, 0,
        numPts, 
        (const int64_m*)pKeys.data_ptr(), 
        (const int*)pGridSize.data_ptr(), 
        (int2*)outTensor.data_ptr());

    // Return result.
    return outTensor;
}
