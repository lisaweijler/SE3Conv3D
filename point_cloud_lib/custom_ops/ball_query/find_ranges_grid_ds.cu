/////////////////////////////////////////////////////////////////////////////
/// \file compute_keys.cu
///
/// \brief Implementation of the CUDA operations to compute the keys indices 
///     of a point cloud into a regular grid. 
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#include "ball_query_utils.cuh"
#include "../utils.cuh"
#include "math_helper.cuh"
#include "grid_utils.cuh"

#include "find_ranges_grid_ds.cuh"

///////////////////////// GPU

/**
 *  GPU kernel to find the ranges in the list of points for a grid cell 
 *  and its 26 neighbors.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumPts         Number of points.
 *  @param  pLastDOffsets   Number of displacement in the last
 *      dimension in the positive and negative axis.
 *  @param  pNumOffsets     Number of offsets applied to  the 
 *      keys.
 *  @param  pOffsets        List of offsets to apply.
 *  @param  pSampleKeys     Array of keys for each sample.
 *  @param  pPtKeys         Array of keys for each point.
 *  @param  pGridDs         Grid data structure.
 *  @param  pNumCells       Number of cells.
 *  @param  pOutDS          Output array with the ranges.
 *  @paramT D               Number of dimensions.
 */
template<int D>
__global__ void find_ranges_grid_ds_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumPts,
    const unsigned int pLastDOffsets,
    const unsigned int pNumOffsets,
    const int* __restrict__ pOffsets,
    const int64_m* __restrict__ pSampleKeys,
    const int64_m* __restrict__ pPtKeys,
    const int2* __restrict__ pGridDs,
    const int* __restrict__ pNumCells,
    int2* __restrict__ pOutRanges)
{
    // Pointers.
    const ipoint<D>* offsetPtr = (const ipoint<D>*)pOffsets;
    const ipoint<D> numCellsPtr(pNumCells);

    // Thread indices.
    int initPtIndex = compute_global_index_gpu_funct();
    int totalThreads = compute_total_threads_gpu_funct();

    // Calculate the total number of cells.
    int64_m totalCells = compute_total_num_cells_gpu_funct(numCellsPtr);

    for(int curIter = initPtIndex; curIter < pNumSamples*pNumOffsets; curIter += totalThreads)
    {
        // Calculate the point and offset index.
        int curPtIndex = curIter/pNumOffsets;
        int curOffset = curIter%pNumOffsets;

        // Get the current offset.
        ipoint<D> cellOffset = offsetPtr[curOffset];

        // Get the key of the current point.
        int64_m curKey = pSampleKeys[curPtIndex];

        // Get the new cell with the offset.
        ipoint<D+1> cellIndex = 
            compute_cell_from_key_gpu_funct(curKey, numCellsPtr);
#pragma unroll
        for(int i=0; i < D; ++i)
            cellIndex[i+1] += cellOffset[i];

        // Check if we are out of the bounding box.
        bool inside = true;
#pragma unroll
        for(int i=0; i < D-1; ++i)
            inside = inside && cellIndex[i+1] >= 0 && cellIndex[i+1] < numCellsPtr[i];
        if(inside)
        {
            // Get the range of pts to check in the data structure.
            int curDsIndex = compute_ds_index_from_cell_gpu_funct(cellIndex, numCellsPtr);
            int2 dsRange = pGridDs[curDsIndex];
            int rangeSize = dsRange.y-dsRange.x-1;

            // Tube has at least 1 element.
            if(rangeSize >= 0){

                // Compute max key of the range.
                ipoint<D> auxCell(&cellIndex[1]);
                int64_m auxKey = compute_key_gpu_funct(auxCell, numCellsPtr, cellIndex[0]);
                int64_m maxKey = auxKey+pLastDOffsets;
                int64_m minKey = auxKey-pLastDOffsets;
                maxKey = (auxKey/totalCells == maxKey/totalCells)?maxKey:((auxKey/totalCells)*totalCells) + totalCells - 1;
                minKey = (auxKey/totalCells == minKey/totalCells)?minKey:((auxKey/totalCells)*totalCells);

                // Declare iterators and auxiliar variables.
                int2 curMinRange = make_int2(0, rangeSize);
                int2 curMaxRange = make_int2(0, rangeSize);

                // Search for the range.
                bool stopMinRange = rangeSize <= 1;
                bool stopMaxRange = stopMinRange;
                while(!stopMinRange || !stopMaxRange){
                    
                    // Compute the pivots.
                    int minPivot = (curMinRange.y + curMinRange.x)/2;
                    int maxPivot = (curMaxRange.y + curMaxRange.x)/2;

                    // Check the minimum range.
                    if(!stopMinRange){
                        int64_m curMinKey = pPtKeys[minPivot+dsRange.x];
                        if(curMinKey < minKey) curMinRange.x = minPivot;
                        else curMinRange.y = minPivot;
                    }

                    // Check the maximum range.
                    if(!stopMaxRange){
                        int64_m curMaxKey = pPtKeys[maxPivot+dsRange.x];
                        if(curMaxKey > maxKey) curMaxRange.y = maxPivot;
                        else curMaxRange.x = maxPivot;
                    }

                    // Check the stopping condition.
                    stopMinRange = (curMinRange.y - curMinRange.x) <= 1;
                    stopMaxRange = (curMaxRange.y - curMaxRange.x) <= 1;
                }

                int2 resultingRange = make_int2(0, 0);

                // Get the values of the keys.
                int64_m lastMinKey1 = pPtKeys[curMinRange.x+dsRange.x];
                int64_m lastMinKey2 = pPtKeys[curMinRange.y+dsRange.x];

                // Test for the init of the range.
                if(lastMinKey1 >= minKey && lastMinKey1 <= maxKey){
                    resultingRange.x = curMinRange.x+dsRange.x;
                }else if(lastMinKey2 >= minKey && lastMinKey2 <= maxKey){
                    resultingRange.x = curMinRange.y+dsRange.x;
                }

                int64_m lastMaxKey1 = pPtKeys[curMaxRange.x+dsRange.x];
                int64_m lastMaxKey2 = pPtKeys[curMaxRange.y+dsRange.x];

                // Test for the end of the range.
                if(lastMaxKey2 >= minKey && lastMaxKey2 <= maxKey){
                    resultingRange.y = curMaxRange.y+dsRange.x+1;
                }else if(lastMaxKey1 >= minKey && lastMaxKey1 <= maxKey){
                    resultingRange.y = curMaxRange.x+dsRange.x+1;
                }

                // Store in memory the resulting range.
                pOutRanges[curIter] = resultingRange;
            }
        }
    }
}

/**
 *  GPU kernel for the 2D case.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumPts         Number of points.
 *  @param  pNumOffsets     Number of offsets applied to  the 
 *      keys.
 *  @param  pOffsets        List of offsets to apply.
 *  @param  pSampleKeys     Array of keys for each sample.
 *  @param  pGridDs         Grid data structure.
 *  @param  pNumCells       Number of cells.
 *  @param  pOutDS          Output array with the ranges.
 *  @paramT D               Number of dimensions.
 */
template<int D>
__global__ void find_ranges_grid_ds_2d_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumPts,
    const unsigned int pNumOffsets,
    const int* __restrict__ pOffsets,
    const int64_m* __restrict__ pSampleKeys,
    const int2* __restrict__ pGridDs,
    const int* __restrict__ pNumCells,
    int2* __restrict__ pOutRanges)
{
    // Pointers.
    const ipoint<D>* offsetPtr = (const ipoint<D>*)pOffsets;
    const ipoint<D> numCellsPtr(pNumCells);

    int initPtIndex = compute_global_index_gpu_funct();
    int totalThreads = compute_total_threads_gpu_funct();
 
    //Calculate the total number of cells.
    int64_m totalCells = compute_total_num_cells_gpu_funct(numCellsPtr);
 
    for(int curIter = initPtIndex; curIter < pNumSamples*pNumOffsets; curIter += totalThreads)
    {
        //Calculate the point and offset index.
        int curPtIndex = curIter/pNumOffsets;
        int curOffset = curIter%pNumOffsets;
 
        //Get the current offset.
        ipoint<D> cellOffset = offsetPtr[curOffset];
 
        //Get the key of the current point.
        int64_m curKey = pSampleKeys[curPtIndex];
 
        //Get the new cell with the offset.
        ipoint<D+1> cellIndex = 
            compute_cell_from_key_gpu_funct(curKey, numCellsPtr);
#pragma unroll
        for(int i=0; i < D; ++i)
            cellIndex[i+1] += cellOffset[i];
 
        //Check if we are out of the bounding box.
        bool inside = true;
#pragma unroll
        for(int i=0; i < D; ++i)
            inside = inside && cellIndex[i+1] >= 0 && cellIndex[i+1] < numCellsPtr[i];
        if(inside)
        {
            //Get the range of pts to check in the data structure.
            int curDsIndex = compute_ds_index_from_cell_gpu_funct(cellIndex, numCellsPtr);
            int2 dsRange = pGridDs[curDsIndex];
 
            //Store in memory the resulting range.
            pOutRanges[curIter] = dsRange;
        }
    }
}

///////////////////////// CPU

torch::Tensor find_ranges_grid_ds_gpu(
    const unsigned int pLastDOffsets,
    const torch::Tensor& pOffsets,
    const torch::Tensor& pSampleKeys,
    const torch::Tensor& pPtKeys,
    const torch::Tensor& pGridSize,
    const torch::Tensor& pGridDS)
{
    // Get the number of dimensions and points.
    int numDims = pGridSize.size(0);
    int numOffsets = pOffsets.size(0);
    int numSamples = pSampleKeys.size(0);
    int numPts = pPtKeys.size(0);

    // Get device properties.
    cudaDeviceProp props = get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    if(numDims > 2){
        DIMENSION_FUNCT_PTR(numDims, find_ranges_grid_ds_gpu_kernel, funcPtr);
    }else{
        DIMENSION_FUNCT_PTR(numDims, find_ranges_grid_ds_2d_gpu_kernel, funcPtr);
    }    

    // Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;
    unsigned int numBlocks = get_max_active_block_x_sm(
        blockSize, funcPtr, 0);

    // Calculate the total number of blocks to execute.
    unsigned int execBlocks = (numSamples*numOffsets)/blockSize;
    execBlocks += ((numSamples*numOffsets)%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    // Generate range tensor.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kInt32)
        .device(pPtKeys.device().type(), pPtKeys.device().index());
    auto outRanges = torch::zeros({numSamples, numOffsets, 2}, tensorOptions);

    // Call the cuda kernel.
    if(numDims > 2){
        DIMENSION_SWITCH_CALL(numDims, find_ranges_grid_ds_gpu_kernel, totalNumBlocks, blockSize, 0,
            numSamples, numPts, pLastDOffsets, numOffsets,
            (const int*)pOffsets.data_ptr(), 
            (const int64_m*)pSampleKeys.data_ptr(), 
            (const int64_m*)pPtKeys.data_ptr(),
            (const int2*)pGridDS.data_ptr(),
            (const int*)pGridSize.data_ptr(),
            (int2*)outRanges.data_ptr());
    }else{
        DIMENSION_SWITCH_CALL(numDims, find_ranges_grid_ds_2d_gpu_kernel, totalNumBlocks, blockSize, 0,
            numSamples, numPts, numOffsets,
            (const int*)pOffsets.data_ptr(), 
            (const int64_m*)pSampleKeys.data_ptr(), 
            (const int2*)pGridDS.data_ptr(),
            (const int*)pGridSize.data_ptr(),
            (int2*)outRanges.data_ptr());
    }

    return outRanges;
}

unsigned int compute_total_num_offsets(
    const unsigned int pNumDimensions,
    const unsigned int pAxisOffset,
    std::vector<int>& pOutVector)
{
    //Calculate the total number of offsets.
    unsigned int cellsXAxis = pAxisOffset*2 + 1;
    unsigned int numOffsets = cellsXAxis;
    for(int i = 0 ; i < std::max((int)pNumDimensions-2, 1); ++i)
        numOffsets *= cellsXAxis;

    //Calculate each offset.
    pOutVector.clear();
    std::vector<int> curOffset(pNumDimensions, 0);
    for(int i = 0; i < numOffsets; ++i)
    {
        int auxInt = i;
        
        for(int j = std::max((int)pNumDimensions-2, 1); j >= 0 ; --j) 
        {
            int auxInt2 = auxInt%cellsXAxis;
            auxInt2 = auxInt2-pAxisOffset;
            curOffset[j] = auxInt2;
            auxInt = auxInt/cellsXAxis;
        }  
        if(pNumDimensions != 2) curOffset[pNumDimensions-1] = 0;

        for(int j = 0; j < pNumDimensions; ++j)
            pOutVector.push_back(curOffset[j]);
    }

    return numOffsets;
}