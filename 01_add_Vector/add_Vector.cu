#include <cuda_runtime.h>
#include <stdio.h>
#include <utils.h>

using namespace std;

void addVectorCPU(float* a, float* b, float* c, const int length)
{
    for(int i=0; i<length; i++){
        c[i] = a[i] + b[i];
    }
}

__global__ void addVectorGPU(float* a, float* b, float* c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        c[i] = a[i] + b[i];
}

int main(int argc, char** argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    int nElems = 1<<24;
    int nBytes = sizeof(float) * nElems;
    float* a_host = (float*)malloc(nBytes);
    float* b_host = (float*)malloc(nBytes);
    float* c_host = (float*)malloc(nBytes);
    float* c_from_dev_host = (float*)malloc(nBytes);

    //初始化
    initialData(a_host, nElems);
    initialData(b_host, nElems);
    memset(c_host, 0, nBytes);
    memset(c_from_dev_host, 0, nBytes);

    double iStart, iElaps;

    //GPU
    //定义设备内存
    float *a_dev, *b_dev, *c_dev;
    CHECK(cudaMalloc((float**)&a_dev, nBytes));
    CHECK(cudaMalloc((float**)&b_dev, nBytes));
    CHECK(cudaMalloc((float**)&c_dev, nBytes));

    CHECK(cudaMemcpy(a_dev, a_host, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_dev, b_host, nBytes, cudaMemcpyHostToDevice));

    dim3 block(512);
    dim3 grid((nElems-1)/block.x+1);
    iStart = cpuSecond();
    addVectorGPU<<<grid, block>>>(a_dev, b_dev, c_dev, nElems);
    iElaps = cpuSecond() - iStart;
    printf("<<<%d, %d>>>, Time elapsed %f sec\n", block.x, grid.x, iElaps);

    CHECK(cudaMemcpy(c_from_dev_host, c_dev, nBytes, cudaMemcpyDeviceToHost));

    //CPU 
    iStart = cpuSecond();
    addVectorCPU(a_host, b_host, c_host, nElems);
    iElaps = cpuSecond() - iStart;
    printf("Time elapsed %f sec\n", block.x, grid.x, iElaps);

    checkResult(c_host, c_from_dev_host, nElems);

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    free(a_host);
    free(b_host);
    free(c_host);
    free(c_from_dev_host);

    return 0;
}