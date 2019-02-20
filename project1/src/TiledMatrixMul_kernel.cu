
#ifndef __TILED_KERNEL__
#define __TILED_KERNEL__
#include<cuda_runtime.h>
__global__ void normal(int row,float* input1,float* input2,float *output){
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int sum=0;
    if(bx*blockDim.x+tx<row && by*blockDim.y+ty<row){
        for(int i=0;i<row;i++){
            sum+=input1[ty*row+i]*input2[i*row+tx];
        }
        output[bx*blockDim.x+tx+(by*blockDim.y+ty)*row]=sum;
    }
}
template <int blockSize>
__global__ void tile(int row,float* input1,float* input2,float *output){
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int targetx=bx*blockSize+tx;
    int targety=by*blockSize+ty;

    
    __shared__ float smemA[blockSize][blockSize];
    __shared__ float smemB[blockSize][blockSize];

    int totalPhase=(blockSize+row-1)/blockSize;
    int startA=by*blockSize*row;
    int startB=bx*blockSize;
    int stepA=blockSize;
    int stepB=blockSize*row;


    float sum=0;
    for(int i=0;i<totalPhase;i++){
        //start phase,load share memory
        if((i*blockSize+tx<row)&&(by*blockSize+ty<row))
            smemA[ty][tx]=input1[startA+tx+row*ty];
        else
        {
            smemA[ty][tx]=0;
        }
        if((bx*blockSize+tx<row)&&(i*blockSize+ty<row))
            smemB[ty][tx]=input2[startB+tx+row*ty];
        else
        {
            smemB[ty][tx]=0;
        }
        startA+=stepA;
        startB+=stepB;
        
        __syncthreads();

        for(int j=0;j<blockSize;j++){
            sum+=smemA[ty][j]*smemB[j][tx];
        }
        __syncthreads();
        



    }
    
    if(targetx<row&&targety<row){
        output[targetx+targety*row]=sum;
    }
}
#endif
