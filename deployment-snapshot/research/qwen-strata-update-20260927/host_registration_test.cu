// No GGUF: test a bounded CUDA-registered prefix and strictly resident anonymous suffix with both contexts.
#include "strata/core/pinned.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void cu(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
__global__ void gather(const uint32_t* source, uint32_t* output) {
    const int i = threadIdx.x;
    output[i] = source[i] ^ 0x12345678u;
}
int main() {
    try {
        int count = 0;
        cu(cudaGetDeviceCount(&count));
        require(count >= 2, "two visible CUDA devices required");
        for (int device : {0, 1}) {
            cu(cudaSetDevice(device));
            cu(cudaFree(nullptr));
            cu(cudaDeviceSynchronize());
        }
        cu(cudaSetDevice(0));
        constexpr uint64_t MIB = 1ull << 20;
        const std::vector<uint64_t> bounds{0, 8*MIB, 16*MIB, 24*MIB, 32*MIB, 40*MIB, 48*MIB, 56*MIB, 64*MIB};
        for (int rep = 0; rep < 3; ++rep) {
            strata::core::PinnedArena arena(64*MIB, bounds, 16*MIB);
            require(arena.valid(), "arena allocation failed: " + arena.note);
            require(arena.registered_bytes == 16*MIB && arena.registered_slices == 2,
                    "registered prefix differs from finite budget: " + arena.note);
            require(arena.registered_bytes + arena.locked_bytes == arena.capacity,
                    "anonymous suffix is not fully resident: " + arena.note);
            auto* words = reinterpret_cast<uint32_t*>(arena.data());
            for (int i = 0; i < 32; ++i) words[i] = 0x3f800000u + uint32_t(rep*32+i);
            auto* tail = reinterpret_cast<uint32_t*>(arena.data() + arena.capacity - 128);
            for (int i = 0; i < 32; ++i) tail[i] = words[i] ^ 0xffffffffu;
            for (int device : {0, 1}) {
                cu(cudaSetDevice(device));
                void* alias = nullptr;
                cu(cudaHostGetDevicePointer(&alias, arena.base, 0));
                uint32_t* out = nullptr;
                cu(cudaMalloc(reinterpret_cast<void**>(&out), 128));
                gather<<<1,32>>>(reinterpret_cast<const uint32_t*>(alias), out);
                cu(cudaGetLastError());
                uint32_t back[32];
                cu(cudaMemcpy(back, out, sizeof(back), cudaMemcpyDeviceToHost));
                cu(cudaFree(out));
                for (int i = 0; i < 32; ++i)
                    require(back[i] == (words[i] ^ 0x12345678u), "mapped prefix bit mismatch");
                size_t free_b = 0, total_b = 0;
                cu(cudaMemGetInfo(&free_b, &total_b));
                require(free_b > 0 && total_b > free_b, "invalid CUDA memory-info result");
            }
            for (int i = 0; i < 32; ++i)
                require(tail[i] == (words[i] ^ 0xffffffffu), "locked suffix bit mismatch");
            cu(cudaSetDevice(0));
            std::printf("registration rep=%d registered=%llu locked=%llu cap=%llu: %s\n", rep,
                        (unsigned long long)arena.registered_bytes, (unsigned long long)arena.locked_bytes,
                        (unsigned long long)(16*MIB), arena.note.c_str());
        }
        cu(cudaGetLastError());
        {
            strata::core::PinnedArena invalid(32*MIB, std::vector<uint64_t>{0, 16*MIB, 8*MIB}, 16*MIB);
            require(!invalid.valid(), "invalid nonmonotonic bounds admitted");
        }
        {
            strata::core::PinnedArena legacy(1*MIB);
            require(legacy.valid() && legacy.registered_bytes == legacy.capacity, "default tiny arena changed");
        }
        cu(cudaGetLastError());
        std::puts("PASS bounded registration: 3 arenas, 6 device reads, locked suffix and cleanup; no GGUF");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL bounded registration: %s\n", error.what());
        return 1;
    }
}
