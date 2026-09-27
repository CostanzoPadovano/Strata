// Bounded graph parity/timing gate for the verifier's optional profiled flag wait. No model is loaded.
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using Counter = strata::kernels::VerifyWaitCounter;

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error(why); }
void ck(cudaError_t s, const char* what) {
    if (s != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(s));
}

__global__ void mark_entered(volatile uint32_t* entered) {
    *entered = 1;
    __threadfence_system();
}

__global__ void copy_payload(const volatile uint32_t* payload, uint32_t* out) { *out = *payload; }

struct Fixture {
    int device = -1;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t original = nullptr;
    cudaGraphExec_t profiled = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    uint32_t *h_flag = nullptr, *d_flag = nullptr;
    uint32_t *h_payload = nullptr, *d_payload = nullptr;
    uint32_t *h_entered = nullptr, *d_entered = nullptr;
    uint32_t *d_original_out = nullptr, *d_profiled_out = nullptr;
    Counter* d_counters = nullptr;

    explicit Fixture(int d) : device(d) {
        try {
        ck(cudaSetDevice(device), "select fixture device");
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create test stream");
        ck(cudaEventCreate(&begin), "create begin event");
        ck(cudaEventCreate(&end), "create end event");
        mapped(&h_flag, &d_flag, "flag");
        mapped(&h_payload, &d_payload, "payload");
        mapped(&h_entered, &d_entered, "entry sentinel");
        ck(cudaMalloc(&d_original_out, sizeof(uint32_t)), "allocate original output");
        ck(cudaMalloc(&d_profiled_out, sizeof(uint32_t)), "allocate profiled output");
        ck(cudaMalloc(&d_counters, 3 * sizeof(Counter)), "allocate bounded counter guards");
        const Counter initial[3] = {{0x1111111111111111ull, 0x2222222222222222ull, 0x3333333333333333ull},
                                    {},
                                    {0xaaaaaaaaaaaaaaaaull, 0xbbbbbbbbbbbbbbbbull, 0xccccccccccccccccull}};
        ck(cudaMemcpy(d_counters, initial, sizeof(initial), cudaMemcpyHostToDevice), "initialize counter guards");
        original = capture(false);
        profiled = capture(true);
        ck(cudaGraphUpload(original, stream), "upload original graph");
        ck(cudaGraphUpload(profiled, stream), "upload profiled graph");
        ck(cudaStreamSynchronize(stream), "finish graph upload");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Fixture() { cleanup(); }

    void cleanup() noexcept {
        if (device >= 0) (void) cudaSetDevice(device);
        // A failure after launch must release the mapped wait before synchronizing.
        // This also covers a missing entry sentinel and partially built fixtures.
        if (h_flag) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flag = 1u;
        }
        if (stream) (void) cudaStreamSynchronize(stream);
        if (original) (void) cudaGraphExecDestroy(original);
        if (profiled) (void) cudaGraphExecDestroy(profiled);
        if (begin) (void) cudaEventDestroy(begin);
        if (end) (void) cudaEventDestroy(end);
        if (d_original_out) (void) cudaFree(d_original_out);
        if (d_profiled_out) (void) cudaFree(d_profiled_out);
        if (d_counters) (void) cudaFree(d_counters);
        if (h_flag) (void) cudaFreeHost(h_flag);
        if (h_payload) (void) cudaFreeHost(h_payload);
        if (h_entered) (void) cudaFreeHost(h_entered);
        if (stream) (void) cudaStreamDestroy(stream);
    }

    void mapped(uint32_t** host, uint32_t** alias, const char* what) {
        ck(cudaHostAlloc((void**) host, sizeof(uint32_t), cudaHostAllocMapped | cudaHostAllocPortable), what);
        const cudaError_t s = cudaHostGetDevicePointer((void**) alias, *host, 0);
        if (s != cudaSuccess) {
            (void) cudaFreeHost(*host);
            *host = nullptr;
            fail(std::string("obtain mapped device alias: ") + cudaGetErrorString(s));
        }
        **host = 0;
    }

    cudaGraphExec_t capture(bool profile) {
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin wait graph capture");
        mark_entered<<<1, 1, 0, stream>>>(d_entered);
        if (profile)
            strata::kernels::wait_flag_ge_profiled(d_flag, 1, d_counters + 1, stream);
        else
            strata::kernels::wait_flag_ge(d_flag, 1, stream);
        copy_payload<<<1, 1, 0, stream>>>(d_payload, profile ? d_profiled_out : d_original_out);
        cudaGraph_t graph = nullptr;
        const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
        if (ended != cudaSuccess || !graph) fail(std::string("end wait graph capture: ") + cudaGetErrorString(ended));
        cudaGraphExec_t exec = nullptr;
        const cudaError_t made = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        const cudaError_t destroyed = cudaGraphDestroy(graph);
        if (made != cudaSuccess || destroyed != cudaSuccess) {
            if (exec) (void) cudaGraphExecDestroy(exec);
            fail(std::string("instantiate wait graph: ") + cudaGetErrorString(made != cudaSuccess ? made : destroyed));
        }
        return exec;
    }

    float replay(bool profile, bool delayed, uint32_t payload) {
        *(volatile uint32_t*) h_flag = delayed ? 0u : 1u;
        *(volatile uint32_t*) h_payload = payload;
        *(volatile uint32_t*) h_entered = 0u;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ck(cudaEventRecord(begin, stream), "record replay begin");
        ck(cudaGraphLaunch(profile ? profiled : original, stream), "launch wait graph");
        ck(cudaEventRecord(end, stream), "record replay end");
        if (delayed) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (*(volatile uint32_t*) h_entered == 0u) {
                if (std::chrono::steady_clock::now() > deadline) fail("wait graph never published its entry sentinel");
                std::this_thread::yield();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flag = 1u;
        }
        ck(cudaEventSynchronize(end), "synchronize replay end");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, begin, end), "measure wait replay");
        uint32_t out = 0;
        ck(cudaMemcpy(&out, profile ? d_profiled_out : d_original_out, sizeof(out), cudaMemcpyDeviceToHost),
           "read wait graph output");
        if (out != payload) fail("wait graph changed or reordered the mapped payload");
        return ms;
    }
};

void run_device(int device) {
    Fixture f(device);
    double profiled_delayed_event_ms = 0.0;
    int original_replays = 0, profiled_replays = 0;
    for (int replay = 0; replay < 4; ++replay) {
        const bool delayed = replay >= 2;
        const uint32_t payload = 0x5a000000u | (uint32_t) (device << 8) | (uint32_t) replay;
        (void) f.replay(false, delayed, payload);
        ++original_replays;
        const float ms = f.replay(true, delayed, payload);
        if (delayed) profiled_delayed_event_ms += ms;
        ++profiled_replays;
        if (replay == 1) {
            Counter ready{};
            ck(cudaMemcpy(&ready, f.d_counters + 1, sizeof(ready), cudaMemcpyDeviceToHost),
               "read already-ready wait counter");
            if (ready.calls != 2 || ready.waited_calls != 0 || ready.blocked_ns != 0)
                fail("already-ready profiled waits reported blocked time");
        }
    }
    Counter counters[3]{};
    ck(cudaMemcpy(counters, f.d_counters, sizeof(counters), cudaMemcpyDeviceToHost), "read wait counters");
    const Counter lo{0x1111111111111111ull, 0x2222222222222222ull, 0x3333333333333333ull};
    const Counter hi{0xaaaaaaaaaaaaaaaaull, 0xbbbbbbbbbbbbbbbbull, 0xccccccccccccccccull};
    if (std::memcmp(&counters[0], &lo, sizeof(lo)) != 0 || std::memcmp(&counters[2], &hi, sizeof(hi)) != 0)
        fail("profiled wait wrote outside its assigned counter row");
    if (original_replays != 4 || profiled_replays != 4 || counters[1].calls != 4 ||
        counters[1].waited_calls != 2 || counters[1].blocked_ns == 0)
        fail("profiled wait replay counters are inconsistent");
    const double blocked_ms = (double) counters[1].blocked_ns / 1.0e6;
    const double ratio = blocked_ms / profiled_delayed_event_ms;
    // Event spans include marker/copy kernels and host scheduling; this only detects a gross timer-unit error.
    if (!(profiled_delayed_event_ms > 0.0 && ratio >= 0.25 && ratio <= 1.25))
        fail("%globaltimer delta is inconsistent with loose CUDA-event calibration");
    std::printf("VERIFY_WAIT device=%d calls=4 waited=2 blocked_ns=%llu delayed_event_ms=%.3f ratio=%.3f\n",
                device, (unsigned long long) counters[1].blocked_ns, profiled_delayed_event_ms, ratio);
}

int run() {
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "query CUDA devices");
    if (devices < 2) fail("verify_wait_test requires two CUDA devices");
    run_device(0);
    ck(cudaSetDevice(0), "restore CUDA device 0 after GPU0 fixture");
    run_device(1);
    ck(cudaSetDevice(0), "restore CUDA device 0 after GPU1 fixture");
    std::printf("PASS verify_wait devices=2 replays=4 ready=2 delayed=2 timer=globaltimer_ns outputs=exact\n");
    return 0;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL verify_wait_test: %s\n", e.what());
        return 1;
    }
}
