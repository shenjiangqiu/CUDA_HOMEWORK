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

#define MERGE_THREADBLOCK_SIZE 256
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
            unsigned char t_data=data[curr];

            unsigned int pos=(t_data>>2)&0x3FU;

            atomicAdd(histo+pos,1);
        }
    }

}

__global__ void baseKernel64(unsigned int *histo,unsigned char *d_data,int dim){
    unsigned allThreads=gridDim.x*blockDim.x;
    unsigned index=threadIdx.x+blockIdx.x*blockDim.x;
    
    if(index<64){
        histo[index]=0;
    }//reset to 0;
    int i=0;
    while(index<dim){
        unsigned char t_data=d_data[index];

        unsigned int pos=(t_data>>2)&0x3FU;

        atomicAdd(histo+pos,1);
        index+=allThreads;
    }
}
__global__ void baseKernel64_share(unsigned int *partial_histo,unsigned char *d_data,int dim){
    __shared__ unsigned share_hist[64];
    if(threadIdx.x<64){
        share_hist[threadIdx.x]=0;//init
    }
    unsigned allThreads=gridDim.x*blockDim.x;
    unsigned index=threadIdx.x+blockIdx.x*blockDim.x;
    

    int i=0;
    while(index<dim){
        unsigned char t_data=d_data[index];

        unsigned int pos=(t_data>>2)&0x3FU;

        atomicAdd(share_hist+pos,1);
        index+=allThreads;
    }
    __syncthreads(); 
    if(threadIdx.x<HISTOGRAM64_BIN_COUNT)
        partial_histo[blockIdx.x * HISTOGRAM64_BIN_COUNT + threadIdx.x] = share_hist[threadIdx.x];
    

}

__global__ void histogram64Kernel_private(uint *d_PartialHistograms, uchar *d_Data, uint dataCount)
{
    // Handle to thread block group
    
    //Per-warp subhistogram storage
    unsigned all_threads=gridDim.x*blockDim.x;
    __shared__ uint s_Hist[6*64];//every warp have a private histogram bin
    uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM64_BIN_COUNT;//the warps start point

    //Clear shared memory storage for current threadblock before processing
#pragma unroll

    int curr_index=threadIdx.x;
    while(curr_index<6*64){//totoal size of shared memory
        s_Hist[curr_index]=0;
        curr_index+=blockDim.x;
    }



    __syncthreads();

    curr_index=threadIdx.x;
    while(curr_index<dataCount){
        unsigned char data=d_Data[curr_index];
        atomicAdd(s_WarpHist+((data>>2)&0x3FU),1);
        curr_index+=all_threads;
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();
    

    for (uint bin = threadIdx.x; bin < 64; bin += 6*32)
    {
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++)
        {
            sum += s_Hist[bin + i * 64] ;
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}


__global__ void mergeHistogram64Kernel(
    uint *d_Histogram,
    uint *d_PartialHistograms,
    uint histogramCount
)
{
    //in this kernel ,each block culculate one byte in bin,we need 64 blocks,and 256 threads,each thread read seperate

    // Handle to thread block group
    //cg::thread_block cta = cg::this_thread_block();

    uint sum = 0;

    for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)//this read is not coalesed
    {
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM64_BIN_COUNT];
    }

    __shared__ uint data[MERGE_THREADBLOCK_SIZE];
    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        if (threadIdx.x < stride)
        {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0)
    {
        d_Histogram[blockIdx.x] = data[0];
    }
}


void histogram64(unsigned int *d_Histogram,unsigned char *d_Data,unsigned int byteCount,unsigned* partial_histo=nullptr){
    const int blockSize=6*32;//6 warp per block
    const int gridSize=240;
    #ifdef K1
    QDEBUG("enter naive");
    naiveKernel64<<<gridSize,blockSize>>>(d_Histogram,d_Data,byteCount);
    return;
    #endif

    #ifdef K2
    QDEBUG("enter base");
    baseKernel64<<<gridSize,blockSize>>>(d_Histogram,d_Data,byteCount);
    //mergeHistogram64Kernel<<<64,MERGE_THREADBLOCK_SIZE>>>(d_Histogram,partial_histo,gridSize);
    return;
    #endif

    #ifdef K3
    QDEBUG("enter base_share");
    baseKernel64_share<<<gridSize,blockSize>>>(partial_histo,d_Data,byteCount);
    
    mergeHistogram64Kernel<<<64,MERGE_THREADBLOCK_SIZE>>>(d_Histogram,partial_histo,gridSize);
    return;
    #endif
    #ifdef K4
    QDEBUG("enter private kernel")
    histogram64Kernel_private<<<gridSize,blockSize>>>(partial_histo,d_Data,byteCount);
    
    mergeHistogram64Kernel<<<64,MERGE_THREADBLOCK_SIZE>>>(d_Histogram,partial_histo,gridSize);
    return;
    #endif

    QERROR("NO Kernel selected!");
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
    uint  *d_Histogram,*d_partial_histo;
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
    #ifdef K3
    checkCudaErrors(cudaMalloc((void **)&d_partial_histo,240*64*sizeof(uint)));
    #endif
    checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_Histogram, h_HistogramGPU, 64*sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    
    
    //start the kernel

    
    histogram64(d_Histogram,d_Data,byteCount,d_partial_histo);//warm up

    cudaDeviceSynchronize();
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for(int i=0;i<16;i++)//run test for 16 times
        histogram64(d_Histogram,d_Data,byteCount,d_partial_histo);
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