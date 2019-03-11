#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include<cstdlib>
#include<cstdio>
#include<iostream>
using namespace std;
// project include
#include "histogram_common.h"
#include<parseOprand.hpp>
#include<log.hpp>
__global__ void naiveKernel64(unsigned int *histo,unsigned char *data,int dim){
    
    int allThreads=gridDim.x*blockDim.x;
    int index=threadIdx.x+blockIdx.x*blockDim.x;
    
    if(index<64){
        histo[index]=0;
    }
    int numbers=(dim+allThreads-1)/allThreads;
    int base=index*numbers;
    for(int i=0;i<numbers;i++){
        int curr=base+i;
        if(curr<dim){
            atomicAdd(histo+data[curr],1);
        }
    }

}

void histogram64(unsigned int *d_Histogram,unsigned char *d_Data,unsigned int byteCount){
    int blockSize=6*32;//6 warp per block
    int gridSize=240;
    naiveKernel64<<<gridSize,blockSize>>>(d_Histogram,d_Data,byteCount);
}

int main(int argc,char**argv){
    int dim=0;
    if(0!=parseOpt(argc,(const char**)argv,dim)){
        QERROR("can not parse oprand,exit");
        return -1;
    }
    uchar *h_Data;
    uint  *h_HistogramCPU, *h_HistogramGPU;
    uchar *d_Data;
    uint  *d_Histogram;
    StopWatchInterface *hTimer = nullptr;
    int PassFailFlag = 1;
    uint byteCount = dim;//modified,the byteCount can be arbitary number
    uint uiSizeMult = 1;

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_Data         = new uchar[byteCount];
    h_HistogramCPU = new uint[HISTOGRAM256_BIN_COUNT];
    h_HistogramGPU = new uint[HISTOGRAM256_BIN_COUNT];
    printf("...generating input data\n");
    srand(2009);

    for (uint i = 0; i < byteCount; i++)
    {
        h_Data[i] = rand() % 256;
    }
    for(uint i=0;i<64;i++){
        h_HistogramGPU[i]=0;
    }

    printf("...allocating GPU memory and copying input data\n\n");
    checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
    checkCudaErrors(cudaMalloc((void **)&d_Histogram, 64 * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_Histogram, h_HistogramGPU, 64*sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    
    
    //start the kernel

    
    histogram64(d_Histogram,d_Data,byteCount);//warm up

    cudaDeviceSynchronize();
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for(int i=0;i<16;i++)//run test for 16 times
        histogram64(d_Histogram,d_Data,byteCount);
    cudaDeviceSynchronize();

    sdkStopTimer(&hTimer);
    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)1;
    printf("histogram64() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf("histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM64_THREADBLOCK_SIZE);

    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
    checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

    printf(" ...histogram64CPU()\n");
    histogram64CPU(
        h_HistogramCPU,
        h_Data,
        byteCount
    );

    printf(" ...comparing the results...\n");

    for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i])
        {
            PassFailFlag = 0;
        }

    printf(PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n");

    printf("Shutting down 64-bin histogram...\n\n\n");


}