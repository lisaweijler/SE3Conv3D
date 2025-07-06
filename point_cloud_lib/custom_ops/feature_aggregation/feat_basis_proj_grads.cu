/////////////////////////////////////////////////////////////////////////////
/// \file feat_basis_proj_grads.cuh
///
/// \brief Implementation of the CUDA operations to compute the projection of
///     the features into the basis.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#include "feat_basis_utils.cuh"
#include "feat_basis_proj_grads.cuh"

#include <THC/THCAtomics.cuh>
    
///////////////////////// GPU

//Definition of the number of which the number of features should be 
// multiple of.
#define MULTIPLE_IN_FEATURES 8

//WARNING - Group features should be equal or smaller than K.
template<int K, typename scalar_t>
__global__ void compute_grads_weighted_in_features(
    const unsigned int pGroupFeatures,
    const unsigned int pNumSamples,       
    const unsigned int pNumInFeatures,
    const scalar_t* __restrict__ pInFeaturesGPUPtr,
    const float* __restrict__ pInPtProjBasisGPUPtr,
    const int2* __restrict__ pInNeighborsGPUPtr,
    const int* __restrict__ pSampleNeighIdsGPUPtr,
    const scalar_t* __restrict__ pInFeatGradsGPUPts,
    scalar_t* __restrict__ pOutFeatGradsGPUPtr,
    float* __restrict__ pOutProjBasisGradsGPUPtr)
{
    extern __shared__ unsigned int sharedMemory[];
    scalar_t* sharedMemoryBuff = reinterpret_cast<scalar_t*>(sharedMemory);

    //Compute the total number of blocks executed and other
    //useful indices.
    unsigned int numGroupsXBlock = blockDim.x/K;
    unsigned int numFeatureBlocks = pNumInFeatures/pGroupFeatures;
    unsigned int localId = threadIdx.x%K;
    unsigned int groupId = threadIdx.x/K;
    unsigned int totalBlocks = pNumSamples*numFeatureBlocks;

    //Get the pointers to shared memory.
    scalar_t* accumFeatGrads = sharedMemoryBuff;
    scalar_t* features = &sharedMemoryBuff[blockDim.x*pGroupFeatures];
    scalar_t* inFeatGrads = &sharedMemoryBuff[blockDim.x*pGroupFeatures 
        + numGroupsXBlock*pGroupFeatures];

    for(int curIter = blockIdx.x; 
        curIter < totalBlocks; 
        curIter += gridDim.x)
    {
        //Get the sample id and the feature offset.
        int sampleId = curIter/numFeatureBlocks;
        int featureOffset = (curIter%numFeatureBlocks)*pGroupFeatures;

        //Get the range of points for this receptive field.
        int2 rangePts;
        rangePts.x = (sampleId > 0)?pSampleNeighIdsGPUPtr[sampleId-1]:0;
        rangePts.y = pSampleNeighIdsGPUPtr[sampleId];
        int numNeighbors = rangePts.y - rangePts.x;
        numNeighbors += numGroupsXBlock-(numNeighbors%numGroupsXBlock);

        //Initialize shared memory with the gradients of the point.
        for(int auxIter = threadIdx.x; 
            auxIter < K*pGroupFeatures; 
            auxIter += blockDim.x)
            inFeatGrads[auxIter] = pInFeatGradsGPUPts[
                sampleId*pNumInFeatures*K + featureOffset*K + auxIter];

        __syncthreads();

        //Iterate over the neighbors.
        for(int curNeighIter = groupId; 
            curNeighIter < numNeighbors; 
            curNeighIter += numGroupsXBlock)
        {
            int neighIndex = curNeighIter+rangePts.x;
            float curWeight = 0.0f; 
            float curWeightGrad = 0.0f;
            float curWeightAccumError = 0.0f;
            int2 neighAndSampleIndices;

            if(neighIndex < rangePts.y){
                //Get the neighbor index.
                neighAndSampleIndices = pInNeighborsGPUPtr[neighIndex];

                //Save the weights in shared memory.
                curWeight = pInPtProjBasisGPUPtr[neighIndex*K + localId];                

                //Save the features in shared memory.
                if(localId < pGroupFeatures)
                    features[groupId*pGroupFeatures + localId] = pInFeaturesGPUPtr[
                        neighAndSampleIndices.y*pNumInFeatures 
                        + featureOffset + localId];
            }

            __syncthreads();

            //Iterate over the feature gradients.
            for(int featIter = 0; featIter < pGroupFeatures; ++featIter)
            {
                accumFeatGrads[featIter*blockDim.x + threadIdx.x] = 
                    inFeatGrads[featIter*K + localId]*curWeight;

                //(Kahan summation algorithm for numerical stabitility).
                float auxVar1 = inFeatGrads[featIter*K + localId]*
                    features[groupId*pGroupFeatures + featIter] - 
                    curWeightAccumError;
                float auxVar2 = curWeightGrad + auxVar1;
                curWeightAccumError = (auxVar2 - curWeightGrad) - auxVar1;
                curWeightGrad = auxVar2;
            }

            __syncthreads();

            //Accumulate the contribution of each K and store in memory.
            if(neighIndex < rangePts.y){
                gpuAtomicAdd(&pOutProjBasisGradsGPUPtr[neighIndex*K + localId], curWeightGrad);

                if(localId < pGroupFeatures){
                    //(Kahan summation algorithm for numerical stabitility).
                    scalar_t accum = 0.0;
                    scalar_t accumError = 0.0;
#pragma unroll
                    for(int kIter = 0; kIter < K; ++kIter){
                        scalar_t auxVar1 = accumFeatGrads[localId*blockDim.x + groupId*K + kIter] - accumError;
                        scalar_t auxVar2 = accum + auxVar1;
                        accumError = (auxVar2 - accum) - auxVar1;
                        accum = auxVar2;
                    }

                    gpuAtomicAdd(&pOutFeatGradsGPUPtr[neighAndSampleIndices.y*pNumInFeatures +
                        featureOffset + localId], accum);
                }
            }

            __syncthreads();
        }
    }
}

///////////////////////// CPU

std::vector<torch::Tensor> feat_basis_proj_grads(
    torch::Tensor& p_pt_basis,
    torch::Tensor& p_pt_features,    
    torch::Tensor& p_neighbors,
    torch::Tensor& p_start_ids,
    torch::Tensor& p_in_gradients)
{
    // Get the number of dimensions and points.
    int numBasis = p_pt_basis.size(1);
    int numPts = p_pt_features.size(0);
    int numSamples = p_start_ids.size(0);
    int numNeighbors = p_neighbors.size(0);
    int numFeatures = p_pt_features.size(1);

    //Determine the group of features.
    unsigned int groupFeatSize = min(MULTIPLE_IN_FEATURES, numFeatures);

    // Get device properties.
    cudaDeviceProp props = get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    FEAT_FUNCT_PTR(numBasis, p_pt_features.scalar_type(), 
        compute_grads_weighted_in_features, funcPtr);
        
    //Calculate the block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;

    //Calculate the shared memory needed.
    unsigned int sharedMemSize = groupFeatSize*(blockSize + blockSize/numBasis + numBasis)*
        torch::elementSize(p_pt_features.scalar_type());

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numBlocks = get_max_active_block_x_sm(
        blockSize, funcPtr, sharedMemSize);

    //Calculate the total number of blocks to execute.
    unsigned int numFeatureBlocks = numFeatures/groupFeatSize;
    unsigned int execBlocks = numPts*numFeatureBlocks;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    // Create output.
    auto tensorOptions = torch::TensorOptions().dtype(p_pt_features.scalar_type()).
        device(p_pt_features.device().type(), p_pt_features.device().index());
    auto outFeatGradTensor = torch::zeros({numPts, numFeatures}, tensorOptions);
    tensorOptions = torch::TensorOptions().dtype(torch::ScalarType::Float).
        device(p_pt_features.device().type(), p_pt_features.device().index());
    auto outProjBasisGradTensor = torch::zeros({numNeighbors, numBasis}, tensorOptions);

    // Call the function.
    FEAT_FUNCT_CALL(numBasis, p_pt_features.scalar_type(), totalNumBlocks, blockSize, sharedMemSize,
        "compute_grads_weighted_in_features", compute_grads_weighted_in_features, 
            groupFeatSize, numSamples, numFeatures,
            (const scalar_t*)p_pt_features.data_ptr(),
            (const float*)p_pt_basis.data_ptr(), 
            (const int2*)p_neighbors.data_ptr(), 
            (const int*)p_start_ids.data_ptr(), 
            (const scalar_t*)p_in_gradients.data_ptr(), 
            (scalar_t*)outFeatGradTensor.data_ptr(),
            (float*)outProjBasisGradTensor.data_ptr()
        )

    return {outFeatGradTensor, outProjBasisGradTensor};
}
