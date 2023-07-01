#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <random>


auto xor_vectors(unsigned char* __restrict__ a, unsigned char* __restrict__ b, int size) -> void {
    for (auto i=0; i < size; i++) {
        a[i] = a[i] ^ b[i];
    }
}


__global__ void xorKernel(unsigned char* c, unsigned char* a, unsigned char* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        c[i] = a[i] ^ b[i];
    }
}


int main(int argc, char** argv) {
    constexpr int size = 8 * 1024;
    constexpr int threadsPerBlock = 256;
    constexpr int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, 255);
    
    std::vector<unsigned char> random_bytes_1(size);
    std::vector<unsigned char> random_bytes_2(size);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    for (unsigned char& c : random_bytes_1) {
        c = static_cast<unsigned char>(dist(rd) & 0xFF);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto delta = end_time - start_time;

    std::cout << 
        "\n\nbytes generated in: " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count() << 
        " nanoseconds\n";
    
    /*
    for (auto i : random_bytes_1) {
        printf("%02X ", i);
    }
    */

    start_time = std::chrono::high_resolution_clock::now();
    for (unsigned char& c : random_bytes_2) {
        c = static_cast<unsigned char>(dist(rd) & 0xFF);
    }
    end_time = std::chrono::high_resolution_clock::now();
    delta = end_time - start_time;

    std::cout << 
        "\n\nbytes generated in: " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count() << 
        " nanoseconds\n";
    
    /*
    for (auto i : random_bytes_2) {
        printf("%02X ", i);
    }
    */

    start_time = std::chrono::high_resolution_clock::now();
    xor_vectors(random_bytes_1.data(), random_bytes_2.data(), random_bytes_1.size());
    end_time = std::chrono::high_resolution_clock::now();
    delta = end_time - start_time;
    
    std::cout << 
        "\n\nCPU vector xor in: " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count() << 
        " nanoseconds\n";

    /*
    std::cout << "\n\n\n\n";
    for (auto i : random_bytes_1) {
        printf("%02X ", i);
    }
    */

    // CUDA bit
    unsigned char* device_c = nullptr;
    unsigned char* device_a = nullptr;
    unsigned char* device_b = nullptr;

    cudaMalloc((void**)&device_c, random_bytes_1.size() * sizeof(unsigned char));
    cudaMalloc((void**)&device_a, random_bytes_1.size() * sizeof(unsigned char));
    cudaMalloc((void**)&device_b, random_bytes_1.size() * sizeof(unsigned char));

    //start_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(device_a, random_bytes_1.data(), random_bytes_1.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, random_bytes_2.data(), random_bytes_2.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    start_time = std::chrono::high_resolution_clock::now();
    xorKernel<<<blocksPerGrid, threadsPerBlock>>>(device_c, device_a, device_b, random_bytes_1.size());
    end_time = std::chrono::high_resolution_clock::now();

    cudaDeviceSynchronize();

    unsigned char c[random_bytes_1.size()] = {0};
    cudaMemcpy(c, device_c, random_bytes_1.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //end_time = std::chrono::high_resolution_clock::now();
    delta = end_time - start_time;

    cudaFree(device_c);
    cudaFree(device_a);
    cudaFree(device_b);

    cudaDeviceReset();

    std::cout << 
        "\n\nGPU vector xor in: " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count() << 
        " nanoseconds\n";

    
    std::cout << "\n\n\n\n";
    for (auto i=0; i < random_bytes_1.size(); i++) {
        printf("%02X ", c[i]);
    }
    

    return 0;
}
