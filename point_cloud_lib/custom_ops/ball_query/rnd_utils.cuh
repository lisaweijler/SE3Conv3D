/////////////////////////////////////////////////////////////////////////////
/// \file rnd_utils.cuh
///
/// \brief Implementation of the CUDA operations to create pseudo-random
///     numbers
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#ifndef _RND_UTILS_CUH_
#define _RND_UTILS_CUH_

/**
    *  Pseudo random number generator: http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
    */
__device__ __forceinline__ unsigned int wang_hash(unsigned int pSeed)
{
    pSeed = (pSeed ^ 61) ^ (pSeed >> 16);
    pSeed *= 9;
    pSeed = pSeed ^ (pSeed >> 4);
    pSeed *= 0x27d4eb2d;
    pSeed = pSeed ^ (pSeed >> 15);
    return pSeed;
}

__device__ __forceinline__ unsigned int rand_xorshift(unsigned int pSeed)
{
    pSeed ^= (pSeed << 13);
    pSeed ^= (pSeed >> 17);
    pSeed ^= (pSeed << 5);
    return pSeed;
}

__device__ __forceinline__ unsigned int seed_to_float(unsigned int pSeed)
{
    return __saturatef(float(pSeed) * (1.0 / 4294967296.0));
}

#endif