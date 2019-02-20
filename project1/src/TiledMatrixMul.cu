#include<cuda_runtime.h>

#include<iostream>
#include<fstream>
#include<string>
#include<sstream>

#include<parseOprand.hpp>
#include<spdlog/spdlog.h>

#include<TiledMatrixMul_kernel.cu>

using namespace std;
#ifndef blockSize
#define blockSize 32//real size is blockSize*blockSize
#endif

int main(int argc, char const *argv[])
{
    int row;
    spdlog::set_pattern("[%c] [%s] [%^-%L-%$] %v");
    
    if(0!=parseOpt(argc,argv,row)){
        SPDLOG_ERROR("parseOpt false");
        return -1;
    }
    SPDLOG_DEBUG("the row={} ",row);

    auto martrix_A=new float[row*row];
    auto martrix_B=new float[row*row];
    auto martrix_output=new float[row*row];


    dim3 blockDim(blockSize,blockSize);
    int gridsize=(row+blockSize-1)/blockSize;
    dim3 gridDim(gridsize,gridsize);
    SPDLOG_DEBUG("grid size = {}",gridsize);


    //init martrix
    for(int i=0;i<row*row;i++){
        martrix_A[i]=1.0;
        martrix_B[i]=1.0;
    }

    float *d_a;
    float *d_b;
    float *d_out;
    if(cudaSuccess!=cudaMalloc(&d_a,row*row*sizeof(float))){
        SPDLOG_ERROR("cannot allocate device memory");
        return -1;
    }
    if(cudaSuccess!=cudaMalloc(&d_b,row*row*sizeof(float))){
        SPDLOG_ERROR("cannot allocate device memory");
        return -1;
    }
    if(cudaSuccess!=cudaMalloc(&d_out,row*row*sizeof(float))){
        SPDLOG_ERROR("cannot allocate device memory");
        return -1;
    }
    if(cudaSuccess!=cudaMemcpy(d_a,martrix_A,sizeof(float)*row*row,cudaMemcpyHostToDevice)){
        SPDLOG_ERROR("cannot move memory");
        return -1;
    }
    if(cudaSuccess!=cudaMemcpy(d_b,martrix_B,sizeof(float)*row*row,cudaMemcpyHostToDevice)){
        SPDLOG_ERROR("cannot move memory");
        return -1;
    }
    SPDLOG_INFO("start to lauch kernel ,wait");
    cudaEvent_t start[3], stop[3];
    for(int i=0;i<3;i++){
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
    } 
    cudaEventRecord(start[0]);
    tile<blockSize><<<gridDim,blockDim>>>(row,d_a,d_b,d_out);
    //normal<<<gridDim,blockDim>>>(row,d_a,d_b,d_out);
    cudaEventRecord(stop[0]);

    
    
    if(cudaGetLastError()!=cudaSuccess){
        SPDLOG_ERROR("cannot launch kernel");
        return -1;
    }
    if(cudaSuccess!=cudaMemcpy(martrix_output,d_out,sizeof(float)*row*row,cudaMemcpyDeviceToHost)){
        SPDLOG_ERROR("cannot retrive data");
        return -1;
    }

    SPDLOG_INFO("finish excute,start varify");
    for(int i=0;i<row*row;i++){
        if(martrix_output[i]!=row){
            SPDLOG_DEBUG("answer not correct");
            SPDLOG_DEBUG("output dumped");
            ofstream out("dump.txt");
            for(int j=0;j<row;j++){
                for(int k=0;k<row;k++){
                    out<<martrix_output[j*row+i]<<" ";
                }
                out<<endl;
            }
            return -1;
        }

    }

    SPDLOG_INFO("finished varify,all passed");
    cudaEventSynchronize(stop[0]);
    float time=0;
    cudaEventElapsedTime(&time,start[0],stop[0]);
    SPDLOG_INFO("the elapsed time of TILED with block size {} is {}",blockSize,time);

   
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    delete[] martrix_A;
    delete[] martrix_B;
    delete[] martrix_output;








    

    
    
}
