// Bounded CPU-only wake/burst/phase regression, using the native portable GGML path.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
namespace cpu = strata::kernels::cpu;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
uint64_t process_cpu_100ns() {
#if defined(_WIN32)
    FILETIME created, exited, kernel, user;
    require(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user), "process times");
    ULARGE_INTEGER k, u;
    k.LowPart=kernel.dwLowDateTime; k.HighPart=kernel.dwHighDateTime;
    u.LowPart=user.dwLowDateTime; u.HighPart=user.dwHighDateTime;
    return k.QuadPart + u.QuadPart;
#else
    return 0;
#endif
}
int main() {
    try {
        // The old canonical planar Q2 API requires AVX-512. Our native model uses GGML AVX2 instead;
        // exercise the actual multi-native pool and both its gate/up and down publication phases.
        cpu::NativeFmt fmt;
        std::string error;
        require(cpu::native_fmt(2,2,cpu::H,cpu::FF,fmt,error), "native Q4_0 fixture format");
        constexpr int ne = 3;
        std::vector<std::vector<uint8_t>> blobs(ne, std::vector<uint8_t>(fmt.bytes));
        for (int e = 0; e < ne; ++e) {
            for (size_t i = 0; i < fmt.bytes; ++i) blobs[e][i] = (uint8_t)(i*37 + e*11);
            for (size_t i = 0; i < fmt.bytes; i += 18) { // Q4_0: FP16 scale, 16 packed bytes
                uint16_t scale = (uint16_t)(0x1419 + e*64); // finite, small FP16
                std::memcpy(blobs[e].data() + i, &scale, 2);
            }
        }
        std::vector<float> x(cpu::H), ref(ne*cpu::H), got(ne*cpu::H);
        for (int i=0; i<cpu::H; ++i) x[i] = .02f * std::sin(i*.011f);
        std::vector<uint8_t> act(fmt.act_bytes), hidden(fmt.h_bytes);
        std::vector<float> ff(cpu::FF);
        cpu::native_quant_act(fmt,x.data(),act.data());
        const void* acts[]{act.data()}, *hqs[]{hidden.data()};
        float* ffs[]{ff.data()};
        cpu::ExpertJobMulti jobs[ne]{};
        for (int e=0; e<ne; ++e) {
            float* outs[]{ref.data()+e*cpu::H};
            cpu::native_gu_rows(fmt,blobs[e].data(),acts,1,ffs,0,cpu::FF);
            cpu::native_quant_h(fmt,ff.data(),hidden.data());
            cpu::native_down_rows(fmt,blobs[e].data(),hqs,1,outs,0,cpu::H);
            jobs[e].blob=blobs[e].data(); jobs[e].nt=1;
            jobs[e].nact[0]=act.data(); jobs[e].out[0]=got.data()+e*cpu::H;
            for (int i=0; i<cpu::H; ++i) require(std::isfinite(ref[e*cpu::H+i]), "non-finite fixture");
        }
        for (bool host : {false, true}) {
            cpu::ExpertPool pool(4, false, host);
            for (int turn=0; turn<180; ++turn) {
                if (turn%60 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(35));
                const int n=1+turn%ne;
                std::fill(got.begin(), got.end(), -1.2345e33f);
                pool.run_split_multi_native(fmt,jobs,n);
                require(std::memcmp(got.data(),ref.data(),n*cpu::H*sizeof(float)) == 0,
                        "wake/burst/phase output differs from serial");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            const auto t0=process_cpu_100ns();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const double cpu_ms=(process_cpu_100ns()-t0)/10000.0;
            require(cpu_ms < 100.0, "workers still busy-spinning during idle");
            std::printf("pool host_works=%d: 180 wake/burst batches passed, idle CPU %.3f ms/250ms\n",host,cpu_ms);
        }
        for (int i=0; i<8; ++i) {
            cpu::ExpertPool pool(4,false,i%2 != 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(30)); // destroy while asleep
        }
        std::puts("pool_idle_test OK (CPU component test; not model throughput)");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"pool idle test: %s\n",e.what()); return 1; }
}
