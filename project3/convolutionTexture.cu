/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef OLD
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#define BLKSIZE 32
#include "convolutionTexture_common.h"

////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Use unrolled innermost convolution loop
#define UNROLL_INNER 1

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}



////////////////////////////////////////////////////////////////////////////////
// Convolution kernel and input array storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[128];

extern "C" void setConvolutionKernel(float *h_Kernel,int kernel_length)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, kernel_length * sizeof(float));
}

texture<float, 2, cudaReadModeElementType> texSrc;

extern "C" void setInputArray(cudaArray *a_Src)
{
}

extern "C" void detachInputArray(void)
{
}



////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float x, float y)
{
    return
        tex2D(texSrc, x + (float)(KERNEL_RADIUS - i), y) * c_Kernel[i]
        + convolutionRow<i - 1>(x, y);
}

template<> __device__ float convolutionRow<-1>(float x, float y)
{
    return 0;
}

template<int i> __device__ float convolutionColumn(float x, float y)
{
    return
        tex2D(texSrc, x, y + (float)(KERNEL_RADIUS - i)) * c_Kernel[i]
        + convolutionColumn<i - 1>(x, y);
}

template<> __device__ float convolutionColumn<-1>(float x, float y)
{
    return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(
    #ifdef GB
    float *d_Input,
    #endif
    float *d_Dst,
    int imageW,
    int imageH,
     int kernel_size
)
{
    __shared__  float s[BLKSIZE];
    
    /*
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;
    */
    int radium=(kernel_size-1)>>1;
    int base_x=blockIdx.x*(blockDim.x-2*radium)-radium;//important!
    const int base_y=blockIdx.y*blockDim.y+threadIdx.y;
    #ifdef GB
    if(base_x +threadIdx.x<0)
    s[threadIdx.x]=d_Input[base_y*imageW];
    else if(base_x +threadIdx.x>=imageW)
    {
       s[threadIdx.x]=d_Input[base_y*imageW+imageW-1];
    }else
        s[threadIdx.x]=d_Input[base_y*imageW+base_x+threadIdx.x];
    #else
    s[threadIdx.x]=tex2D(texSrc, base_x+threadIdx.x+0.5 , base_y+0.5);
    #endif
    __syncthreads();
    int ix=base_x+threadIdx.x;
    const int iy=base_y;

    if (ix >= imageW )
    {
        return;
    }
    float sum=0;
    if(threadIdx.x>=radium && threadIdx.x <= blockDim.x-radium-1){//only middle thread attend compute
        for(int i=-radium;i<=radium;i++){
            sum += s[threadIdx.x+i] * c_Kernel[radium + i];//why it is -i from samples?
        }
    }
    

    

/*
#if(UNROLL_INNER)
    sum = convolutionRow<2 *KERNEL_RADIUS>(x, y);
#else

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D(texSrc, x + (float)k, y) * c_Kernel[KERNEL_RADIUS - k];
    }

#endif
*/

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}


extern "C" void convolutionRowsGPU(
    #ifdef GB
    float *d_Input,
    #endif
    float *d_Dst,
    #ifndef GB
    cudaArray *a_Src,
    #endif
    int imageW,
    int imageH,
    int kernel_length
)
{
    dim3 threads(BLKSIZE);
    dim3 blocks(iDivUp(imageW, threads.x-kernel_length+1), iDivUp(imageH, threads.y));
    #ifndef GB
    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    #endif
    convolutionRowsKernel<<<blocks, threads>>>(
        #ifdef GB//that is global memory
        d_Input,
        #endif
        d_Dst,
        imageW,
        imageH,
        kernel_length
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");
    #ifndef GB
    checkCudaErrors(cudaUnbindTexture(texSrc));
    #endif

}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernel(
    #ifdef GB
    float *d_Input,
    #endif
    float *d_Dst,
    int imageW,
    int imageH,
    int kernel_length
)
{
    int radius=(kernel_length-1)>>1;
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)//divergency
    {
        return;
    }

    float sum = 0;



    for (int k = -radius; k <= radius; k++)
    {
        #ifdef GB
        int toY=iy+k;
        if(toY<0) toY=0;
        if(toY>=imageH) toY=imageH-1;
        sum+= d_Input[toY*imageW+ix]* c_Kernel[radius + k];
        #else
        sum += tex2D(texSrc, x, y + (float)k) * c_Kernel[radius + k];
        #endif
    }



    d_Dst[IMAD(iy, imageW, ix)] = sum;
}

extern "C" void convolutionColumnsGPU(
    #ifdef GB
    float *d_Input,
    #endif
    float *d_Dst,
    #ifndef GB
    cudaArray *a_Src,
    #endif
    int imageW,
    int imageH,
    int kernel_length
)
{
    dim3 threads(1,256);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));
    #ifndef GB
    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    #endif
    convolutionColumnsKernel<<<blocks, threads>>>(
        #ifdef GB
        d_Input,
        #endif
        d_Dst,
        imageW,
        imageH,
        kernel_length

    );
    
    getLastCudaError("convolutionColumnsKernel() execution failed\n");
    
#ifndef GB
    checkCudaErrors(cudaUnbindTexture(texSrc));
    #endif
}


#endif
#ifdef OLD
/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>

#include "convolutionTexture_common.h"

////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Use unrolled innermost convolution loop
#define UNROLL_INNER 1

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}



////////////////////////////////////////////////////////////////////////////////
// Convolution kernel and input array storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}

texture<float, 2, cudaReadModeElementType> texSrc;

extern "C" void setInputArray(cudaArray *a_Src)
{
}

extern "C" void detachInputArray(void)
{
}



////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float x, float y)
{
    return
        tex2D(texSrc, x + (float)(KERNEL_RADIUS - i), y) * c_Kernel[i]
        + convolutionRow<i - 1>(x, y);
}

template<> __device__ float convolutionRow<-1>(float x, float y)
{
    return 0;
}

template<int i> __device__ float convolutionColumn(float x, float y)
{
    return
        tex2D(texSrc, x, y + (float)(KERNEL_RADIUS - i)) * c_Kernel[i]
        + convolutionColumn<i - 1>(x, y);
}

template<> __device__ float convolutionColumn<-1>(float x, float y)
{
    return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)
    {
        return;
    }

    float sum = 0;

#if(UNROLL_INNER)
    sum = convolutionRow<2 *KERNEL_RADIUS>(x, y);
#else

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D(texSrc, x + (float)k, y) * c_Kernel[KERNEL_RADIUS - k];
    }

#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}


extern "C" void convolutionRowsGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    convolutionRowsKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernel(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)
    {
        return;
    }

    float sum = 0;

#if(UNROLL_INNER)
    sum = convolutionColumn<2 *KERNEL_RADIUS>(x, y);
#else

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D(texSrc, x, y + (float)k) * c_Kernel[KERNEL_RADIUS - k];
    }

#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    convolutionColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionColumnsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}

#endif