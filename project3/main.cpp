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


/*
 * This sample implements the same algorithm as the convolutionSeparable
 * CUDA Sample, but without using the shared memory at all.
 * Instead, it uses textures in exactly the same way an OpenGL-based
 * implementation would do.
 * Refer to the "Performance" section of convolutionSeparable whitepaper.
 */




#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include<log.hpp>
#include<parseOprand.hpp>
#include "convolutionTexture_common.h"



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

    int x,y,z;
    if(0!=parseOpt(argc,const_cast<const char**>( argv),x,y,z)){
        QERROR("parse error,exiting");
        return -1;
    }
    
    

    float
    *h_Kernel,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;

    cudaArray
    *a_Src;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    float
    *d_Input,
    *d_Output;

    float
    gpuTime;

    StopWatchInterface *hTimer = NULL;

    StopWatchInterface *total = NULL;
    sdkCreateTimer(&total);
    sdkStartTimer(&total);
    int imageW = x;
    int imageH = y;
    unsigned int iterations = 10;

    printf("[%s] - Starting...\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);
    unsigned kernel_length=z;
    printf("Initializing data...\n");
    h_Kernel    = (float *)malloc(kernel_length * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
    #ifdef GB
    checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
    #else
    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));
    #endif
    
    checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));

    srand(2009);

    for (unsigned int i = 0; i < kernel_length; i++)
    {
        h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned int i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (float)(rand() % 16);
    }

    setConvolutionKernel(h_Kernel
    #ifndef OLD
    ,
    kernel_length
    #endif
    );
    #ifdef GB
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    #else
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    #endif


    printf("Running GPU rows convolution (%u identical iterations)...\n", iterations);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (unsigned int i = 0; i < iterations; i++)
    {
        convolutionRowsGPU(
            #ifdef GB
            d_Input,
            #endif
            d_Output,
            #ifndef GB
            a_Src,
            #endif
            imageW,
            imageH
            #ifndef OLD
            ,
            kernel_length
            #endif
        );
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
    printf("Average convolutionRowsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

    //While CUDA kernels can't write to textures directly, this copy is inevitable
    printf("Copying convolutionRowGPU() output back to the texture...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    #ifdef GB
    checkCudaErrors(cudaMemcpy(d_Input, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    #else
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    #endif
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("cudaMemcpyToArray() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

    printf("Running GPU columns convolution (%i iterations)\n", iterations);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int i = 0; i < iterations; i++)
    {
        convolutionColumnsGPU(
            #ifdef GB
            d_Input,
            #endif
            d_Output,
            #ifndef GB
            a_Src,
            #endif
            imageW,
            imageH
            #ifndef OLD
            ,
            kernel_length
            #endif
        );
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
    printf("Average convolutionColumnsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

    printf("Reading back GPU results...\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    sdkStopTimer(&total);
    auto totoalTime=sdkGetTimerValue(&total);
    printf("total time = %f: ",totoalTime);

    printf("Checking the results...\n");
    printf("...running convolutionRowsCPU()\n");
    convolutionRowsCPU(
        h_Buffer,
        h_Input,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    printf("...running convolutionColumnsCPU()\n");
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    convolutionColumnsCPU(
        h_OutputCPU,
        h_Buffer,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("cpu time: %f,%f Mpix/s",gpuTime,imageW * imageH * 1e-6 / (0.001 * gpuTime));

    double delta = 0;
    double sum = 0;

    for (unsigned int i = 0; i < imageW * imageH; i++)
    {
        sum += h_OutputCPU[i] * h_OutputCPU[i];
        delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
    }

    double L2norm = sqrt(delta / sum);
    printf("Relative L2 norm: %E\n", L2norm);
    printf("Shutting down...\n");

    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFreeArray(a_Src));
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

    sdkDeleteTimer(&hTimer);

    if (L2norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
