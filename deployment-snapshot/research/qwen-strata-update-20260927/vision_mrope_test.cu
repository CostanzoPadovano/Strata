#include "strata/kernels/mrope.hpp"

#include <cuda_runtime.h>

#include <cstdio>

__global__ void read_positions(const int32_t* table, int pos, int32_t* out) {
    const int pair = (int) threadIdx.x;
    if (pair < 32) out[pair] = strata::kernels::mrope_pos(table, pos, pair);
}

int main() {
    constexpr int cells = 8;
    int32_t host[cells * 3];
    for (int c = 0; c < cells; ++c) host[c * 3] = host[c * 3 + 1] = host[c * 3 + 2] = c;
    int32_t* table = nullptr;
    int32_t* output = nullptr;
    int32_t got[32];
    bool ok = cudaMalloc(&table, sizeof host) == cudaSuccess &&
              cudaMalloc(&output, sizeof got) == cudaSuccess &&
              cudaMemcpy(table, host, sizeof host, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        read_positions<<<1, 32>>>(nullptr, 5, output);
        ok = cudaMemcpy(got, output, sizeof got, cudaMemcpyDeviceToHost) == cudaSuccess;
        for (int pair = 0; pair < 32; ++pair) ok = ok && got[pair] == 5;
    }
    if (ok) {
        read_positions<<<1, 32>>>(table, 5, output);
        ok = cudaMemcpy(got, output, sizeof got, cudaMemcpyDeviceToHost) == cudaSuccess;
        for (int pair = 0; pair < 32; ++pair) ok = ok && got[pair] == 5;
    }
    if (ok) {
        const int32_t image[3] = {7, 11, 13};
        ok = cudaMemcpy(table + 15, image, sizeof image, cudaMemcpyHostToDevice) == cudaSuccess;
        read_positions<<<1, 32>>>(table, 5, output);
        ok = ok && cudaMemcpy(got, output, sizeof got, cudaMemcpyDeviceToHost) == cudaSuccess;
        for (int pair = 0; pair < 32; ++pair) ok = ok && got[pair] == image[pair % 3];
    }
    if (table) cudaFree(table);
    if (output) cudaFree(output);
    if (!ok) {
        std::fprintf(stderr, "vision M-RoPE synthetic gate failed\n");
        return 1;
    }
    std::printf("vision M-RoPE synthetic gate passed: null, text identity and 32-pair image mapping\n");
    return 0;
}
