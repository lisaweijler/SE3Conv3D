/////////////////////////////////////////////////////////////////////////////
/// \file math_helper.cuh
///
/// \brief Definition of math helper functions.
///
/// \copyright Copyright (c) 2023 Pedro Hermosilla, TU-Wien, Austria  
///            See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (phermosilla@cvl.tuwien.ac.at)
/////////////////////////////////////////////////////////////////////////////

#ifndef MATH_HELPER_H_
#define MATH_HELPER_H_

#include <math.h>
#include <algorithm>
#include "../utils.cuh"
#include "ball_query_utils.cuh"
#include "cuda_runtime.h"
#include "cuda_fp16.h"

//Definition of a point with D dimensions.
template<class T, int D>
struct point{
    
    /**Position of the point.*/
    T   pos_[D];

    ///////////////// CONSTRUCTORS

    __host__ __device__ point(){
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] = 0;
    }

    __host__ __device__ point(T* pPtr){
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] = pPtr[i];
    }

    __host__ __device__ point(const T* pPtr){
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] = pPtr[i];
    }
    
    __host__ __device__ point(T pVal){
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] = pVal;
    }

    __host__ __device__ point(const point<T, D>& pPt){
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] = pPt[i];
    }
    
    ///////////////// OPERATORS

    ///////// ACCESS

    __host__ __device__ __forceinline__ T& 
    operator[](int pIndex)
    {
        return pos_[pIndex];
    }

    __host__ __device__ __forceinline__ const T& 
    operator[](int pIndex) const
    {
        return pos_[pIndex];
    }

    ///////// EQUAL

    __host__ __device__ __forceinline__ point<T, D>& 
    operator=(const point<T, D>& pPt1)
    {
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] = pPt1[i];
        return *this;
    }

    ///////// CONVERSION

    template<class T2>
    __host__ __device__ __forceinline__ operator 
    point<T2, D>() const
    {
        point<T2, D> result;
#pragma unroll
        for(int i=0; i < D; ++i)
            result[i] = static_cast<T2>(pos_[i]);
        return result;
    }

    ///////// SUMATION       

    __host__ __device__ __forceinline__ point<T, D>& 
    operator+=(const point<T, D>& pPt1)
    {
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] += pPt1[i];
        return *this;
    }

    __host__ __device__ __forceinline__ point<T, D>& 
    operator+=(const T pVal)
    {
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] += pVal;
        return *this;
    }

    ///////// SUBSTRACTION      

    __host__ __device__ __forceinline__ point<T, D>& 
    operator-=(const point<T, D>& pPt1)
    {
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] -= pPt1[i];
        return *this;
    }

    __host__ __device__ __forceinline__ point<T, D>& 
    operator-=(const T pVal)
    {
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] -= pVal;
        return *this;
    }

    ///////// MULTIPLICATION      

    __host__ __device__ __forceinline__ point<T, D>& 
    operator*=(const point<T, D>& pPt1)
    {
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] *= pPt1[i];
        return *this;
    }

    __host__ __device__ __forceinline__ point<T, D>& 
    operator*=(const T pVal)
    {
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] *= pVal;
        return *this;
    }

    ///////// DIVISION      

    __host__ __device__ __forceinline__ point<T, D>& 
    operator/=(const point<T, D>& pPt1)
    {
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] /= pPt1[i];
        return *this;
    }

    __host__ __device__ __forceinline__ point<T, D>& 
    operator/=(const T pVal)
    {
#pragma unroll
        for(int i=0; i < D; ++i)
            pos_[i] /= pVal;
        return *this;
    }
};

template<int D>
using fpoint = point<float, D>;

template<int D>
using hpoint = point<half, D>;

template<int D>
using ipoint = point<int, D>;

///////////////// OPERATORS

///////// SUMATION

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> operator+(
    const point<T, D>& pPt1, const point<T, D>& pPt2)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = pPt1[i] + pPt2[i];
    return result;
}

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> operator+(
    const point<T, D>& pPt1, const T pVal)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = pPt1[i] + pVal;
    return result;
}

///////// SUBTRACTION

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> operator-(
    const point<T, D>& pPt1, const point<T, D>& pPt2)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = pPt1[i] - pPt2[i];
    return result;
}

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> operator-(
    const point<T, D>& pPt1, const T pVal)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = pPt1[i] - pVal;
    return result;
}

///////// MULTIPLICATION

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> operator*(
    const point<T, D>& pPt1, const point<T, D>& pPt2)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = pPt1[i] * pPt2[i];
    return result;
}

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> operator*(
    const point<T, D>& pPt1, const T pVal)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = pPt1[i] * pVal;
    return result;
}

///////// DIVISION

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> operator/(
    const point<T, D>& pPt1, const point<T, D>& pPt2)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = pPt1[i] / pPt2[i];
    return result;
}

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> operator/(
    const point<T, D>& pPt1, const T pVal)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = pPt1[i] / pVal;
    return result;
}


///////////////// MATH

template<class T>
__host__ __device__ __forceinline__ point<T, 3> cross(
    const point<T, 3>& pPt1, const point<T, 3>& pPt2)
{
    point<T, 3> result;
    result[0] = pPt1[1]*pPt2[2] - pPt1[2]*pPt2[1];
    result[1] = pPt1[2]*pPt2[0] - pPt1[0]*pPt2[2];
    result[2] = pPt1[0]*pPt2[1] - pPt1[1]*pPt2[0];
    return result;
}

template<class T, int D>
__host__ __device__ __forceinline__ T dot(
    const point<T, D>& pPt1, const point<T, D>& pPt2)
{
    T result = (T)0;
#pragma unroll
    for(int i=0; i < D; ++i)
        result += pPt1[i] * pPt2[i];
    return result;
}

template<class T, int D>
__host__ __device__ __forceinline__ float length(
    const point<T, D>& pPt)
{
    return sqrt(dot(pPt, pPt));
}

template<class T, int D>
__host__ __device__ __forceinline__ void normalize(point<T, D>& pPt)
{
    float curInvLength = 1.0f/length(pPt);
    pPt *= curInvLength;
}

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> normalize(point<T, D> pPt)
{
    float curInvLength = 1.0f/length(pPt);
    pPt *= curInvLength;
    return pPt;
}

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> floorp(
    const point<T, D>& pPt)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = floorf(pPt[i]);
    return result;
}

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> maxp(
    const point<T, D>& pPt1, const point<T, D>& pPt2)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = PC_MAX(pPt1[i], pPt2[i]);
    return result;
}

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> minp(
    const point<T, D>& pPt1, const point<T, D>& pPt2)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = PC_MIN(pPt1[i], pPt2[i]);
    return result;
}

template<class T, int D>
__host__ __device__ __forceinline__ point<T, D> expf(
    const point<T, D>& pPt)
{
    point<T, D> result;
#pragma unroll
    for(int i=0; i < D; ++i)
        result[i] = exp(pPt[i]);
    return result;
}

#endif