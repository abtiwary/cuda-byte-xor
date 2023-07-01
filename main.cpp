#include <algorithm>
#include <chrono>
#include <climits>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "cuda/cuda_functions.h"

using random_bytes_engine = std::independent_bits_engine<
        std::default_random_engine,
        CHAR_BIT,
        unsigned char>;

template<typename T>
void print_vector_contents_hex(const std::vector<T>& vec) {
    for (auto b : vec) {
        printf("%02x ", b);
    }
    std::cout << "\n";
};

template<typename T>
auto xor_vectors(T* __restrict__ a, T* __restrict__ b, int size) -> void {
    for (auto i=0; i < size; i++) {
        a[i] = a[i] ^ b[i];
    }
}

auto get_vec_random_bytes_of_size(size_t size) -> std::vector<unsigned char> {
    random_bytes_engine rbe;
    rbe.seed(std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<unsigned char> rbytes(size);
    std::generate(begin(rbytes), end(rbytes), std::ref(rbe));
    return std::move(rbytes);
}

int main(int argc, char** argv) {
    auto first_vec = get_vec_random_bytes_of_size(1024);
    //print_vector_contents_hex<unsigned char>(first_vec);

    auto second_vec = get_vec_random_bytes_of_size(1024);
    //print_vector_contents_hex<unsigned char>(second_vec);
    
    auto start_time_xor = std::chrono::high_resolution_clock::now();
    xor_vectors(first_vec.data(), second_vec.data(), first_vec.size());
    auto end_time_xor = std::chrono::high_resolution_clock::now();
    //print_vector_contents_hex(first_vec);
    std::cout << 
        "\nCPU XOR in " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_xor - start_time_xor).count() << 
        " nanoseconds\n";
    
    auto start_time_xor_gpu = std::chrono::high_resolution_clock::now();
    xor_gpu(first_vec.data(), first_vec.data(), second_vec.data(), first_vec.size());
    auto end_time_xor_gpu = std::chrono::high_resolution_clock::now();
    std::cout << 
        "\nGPU XOR in " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_xor_gpu - start_time_xor_gpu).count() << 
        " nanoseconds\n";
    //print_vector_contents_hex(first_vec);

    return 0;
}

