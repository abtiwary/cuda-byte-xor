#include "cuda_functions.h"

__global__
void xor_kernel(unsigned char* c, unsigned char* a, unsigned char* b, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        c[i] = a[i] ^ b[i];
    }    
}


void xor_gpu(unsigned char* c, unsigned char* a, unsigned char* b, size_t size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    unsigned char* device_c = NULL;
    unsigned char* device_a = NULL;
    unsigned char* device_b = NULL;
    
    cudaMalloc((void**)&device_c, size * sizeof(unsigned char));
    cudaMalloc((void**)&device_a, size * sizeof(unsigned char));
    cudaMalloc((void**)&device_b, size * sizeof(unsigned char));

    cudaMemcpy(device_a, a, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
                    
    xor_kernel<<<blocks_per_grid, threads_per_block>>>(device_c, device_a, device_b, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c, device_c, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(device_c);
    cudaFree(device_a);
    cudaFree(device_b);

    cudaDeviceReset();
}


