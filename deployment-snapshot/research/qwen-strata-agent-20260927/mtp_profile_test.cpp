// Synthetic one-layer MTP transport profiler. This is not a full-model throughput benchmark.

#include "strata/core/mtp.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using strata::core::ModelGeometry;
using strata::core::MtpDrafter;
using strata::core::NativeEmbed;
using strata::core::NativeHead;
using strata::core::SessionState;
using strata::core::WeightTable;
using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }

void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(status));
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        if (!ptr_) return;
        int old = -1;
        (void) cudaGetDevice(&old);
        if (cudaSetDevice(device_) == cudaSuccess) (void) cudaFree(ptr_);
        if (old >= 0) (void) cudaSetDevice(old);
    }
    void allocate(int device, size_t bytes, const char* what) {
        if (bytes == 0) fail(std::string(what) + ": zero-byte allocation");
        cuda_check(cudaSetDevice(device), "select allocation device");
        const cudaError_t status = cudaMalloc(&ptr_, bytes);
        if (status != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(status));
        device_ = device;
        bytes_ = bytes;
    }
    void upload(const void* src, size_t bytes, const char* what) {
        if (!ptr_ || !src || bytes > bytes_) fail(std::string(what) + ": invalid upload bounds");
        cuda_check(cudaSetDevice(device_), "select upload device");
        cuda_check(cudaMemcpy(ptr_, src, bytes, cudaMemcpyHostToDevice), what);
    }
    void* get() const { return ptr_; }
    template <class T> T* as() const { return static_cast<T*>(ptr_); }

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    int device_ = -1;
};

struct SessionCleanup {
    SessionState* ss = nullptr;
    const ModelGeometry* g = nullptr;
    ~SessionCleanup() {
        if (!ss || !g || !ss->qsa_states) return;
        for (int64_t i = 0; i < g->n_qsa_layers(); ++i) {
            if (ss->qsa_states[i].host_step) (void) cudaFreeHost(ss->qsa_states[i].host_step);
            if (ss->qsa_states[i].host_pos) (void) cudaFreeHost(ss->qsa_states[i].host_pos);
        }
        delete[] ss->qsa_states;
        ss->qsa_states = nullptr;
    }
};

std::set<std::string> all_index_names(const std::string& pack) {
    std::ifstream f(pack + "/index.txt");
    if (!f) fail("cannot open " + pack + "/index.txt");
    std::set<std::string> names;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t end = line.find_first_of(" \t\r\n");
        const std::string name = line.substr(0, end);
        if (name.empty()) fail("empty tensor name in index.txt");
        names.insert(name);
    }
    if (names.empty()) fail("index.txt has no tensor rows");
    return names;
}

struct Args { std::string pack, native, mtp; int gap_ms = 0; bool chain_batch = false; };

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if ((k == "--pack" || k == "--native" || k == "--mtp") && i + 1 < argc) {
            std::string& dst = k == "--pack" ? a.pack : k == "--native" ? a.native : a.mtp;
            dst = argv[++i];
        } else if (k == "--gap-ms" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "0") a.gap_ms = 0;
            else if (v == "40") a.gap_ms = 40;
            else fail("--gap-ms must be 0 or 40");
        } else if (k == "--chain-batch") {
            a.chain_batch = true;
        } else {
            fail("usage: mtp_profile_test --pack PATH --native GGUF --mtp PATH [--gap-ms 0|40] [--chain-batch]");
        }
    }
    if (a.pack.empty() || a.native.empty() || a.mtp.empty())
        fail("usage: mtp_profile_test --pack PATH --native GGUF --mtp PATH [--gap-ms 0|40] [--chain-batch]");
    return a;
}

struct DraftResult {
    int n = 0;
    std::array<int32_t, 3> ids{{-1, -1, -1}};
    std::array<float, 3> probs{{-1.0f, -1.0f, -1.0f}};
};

uint32_t float_bits(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

void validate_result(const char* mode, int iteration, const DraftResult& r, int64_t n_vocab) {
    if (r.n != 3) fail(std::string(mode) + " iteration " + std::to_string(iteration) + " did not produce 3 drafts");
    for (int i = 0; i < 3; ++i) {
        if (r.ids[(size_t) i] < 0 || (int64_t) r.ids[(size_t) i] >= n_vocab)
            fail(std::string(mode) + " iteration " + std::to_string(iteration) + " produced an out-of-range id");
        if (!std::isfinite(r.probs[(size_t) i]))
            fail(std::string(mode) + " iteration " + std::to_string(iteration) + " produced a non-finite probability");
    }
}

void validate_stable(const char* mode, int iteration, const DraftResult& reference, const DraftResult& got) {
    if (got.n != reference.n) fail(std::string(mode) + " changed chain length at iteration " + std::to_string(iteration));
    for (int i = 0; i < 3; ++i)
        if (got.ids[(size_t) i] != reference.ids[(size_t) i] ||
            float_bits(got.probs[(size_t) i]) != float_bits(reference.probs[(size_t) i]))
            fail(std::string(mode) + " changed its fixed-path id/probability bits at iteration " +
                 std::to_string(iteration));
}

int run(const Args& args) {
    constexpr int64_t kContext = 256;
    constexpr int64_t kExperts = 10;
    constexpr int kMaxT = 4;
    constexpr int64_t kVocab = 248320;
    constexpr int kWarm = 8;
    constexpr int kIterations = 64;

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) fail("MTP profile requires visible CUDA devices 0 and 1");
    for (int device : {0, 1}) {
        cuda_check(cudaSetDevice(device), "initialize CUDA device");
        cuda_check(cudaFree(nullptr), "initialize CUDA context");
    }
    cuda_check(cudaSetDevice(0), "restore target CUDA device");

    const ModelGeometry g;
    const int64_t hcn = g.hc * g.n_embd;
    if (hcn != 10240) fail("unexpected model geometry: hc*n_embd is not 10240");

    const std::set<std::string> skip = all_index_names(args.pack);
    uint64_t pool = 0;
    std::string err;
    if (!WeightTable::pool_bytes(args.pack, pool, err, &skip)) fail(err);
    if (pool != 0) fail("metadata-only WeightTable unexpectedly needs a nonzero pool");
    DeviceBuffer dummy;
    dummy.allocate(0, 256, "metadata dummy arena");
    WeightTable wt;
    if (!wt.load(args.pack, dummy.get(), 256, err, &skip)) fail(err);
    const strata::core::WeightRef* output = wt.find("output.weight");
    if (!output || output->ne1 != kVocab) fail("output.weight metadata does not report n_vocab=248320");
    const int64_t n_vocab = output->ne1;

    // The portable host table and its format metadata outlive every resident MTP copy/captured graph.
    NativeEmbed embed;
    if (!embed.load(args.native, g.n_embd, n_vocab, err)) fail(err);
    strata::core::set_native_embed(&embed);

    std::vector<float> prefill((size_t) 17 * (size_t) hcn);
    std::vector<float> window((size_t) kMaxT * (size_t) hcn);
    for (size_t i = 0; i < prefill.size(); ++i)
        prefill[i] = (float) ((int) (i % 257) - 128) / 256.0f;
    for (size_t i = 0; i < window.size(); ++i)
        window[i] = (float) ((int) ((i * 13 + 7) % 509) - 254) / 512.0f;
    std::array<int32_t, 17> prefill_tokens{};
    for (int i = 0; i < 17; ++i) prefill_tokens[(size_t) i] = 1000 + 17 * i;
    const std::array<int32_t, 4> draft_tokens{{3101, 3102, 3103, 3104}};

    struct Mode { const char* name; int target; int draft; bool mapped; };
    const std::array<Mode, 4> modes{{
        {"resident-local-gpu0", 0, 0, false},
        {"resident-remote-gpu1-staged", 0, 1, false},
        {"resident-remote-gpu1-mapped", 0, 1, true},
        {"resident-local-gpu1", 1, 1, false},
    }};

    const uint64_t residual_bytes = (uint64_t) kMaxT * (uint64_t) hcn * sizeof(float);
    for (const Mode& mode : modes) {
        // Everything whose pointer is captured or device-owned is constructed on this mode's target and dies
        // after its MTP owner, before the next mode starts.
        cuda_check(cudaSetDevice(mode.target), "select profile target device");
        NativeHead head;
        if (!head.load(args.native, g.n_embd, n_vocab, err)) fail(std::string(mode.name) + " head: " + err);
        DeviceBuffer session_mem;
        session_mem.allocate(mode.target, (size_t) strata::core::session_bytes(g, kContext, kExperts), "mode session");
        SessionState ss;
        SessionCleanup session_cleanup{&ss, &g};
        if (strata::core::session_init(g, kContext, kExperts, session_mem.get(), ss) == 0)
            fail(std::string(mode.name) + " session_init failed");
        strata::core::session_zero(ss, g, nullptr, nullptr);
        cuda_check(cudaDeviceSynchronize(), "initialize mode session state");

        DeviceBuffer d_prefill, d_window;
        d_prefill.allocate(mode.target, prefill.size() * sizeof(float), "prefill residuals");
        d_window.allocate(mode.target, window.size() * sizeof(float), "window residuals");
        d_prefill.upload(prefill.data(), prefill.size() * sizeof(float), "upload prefill residuals");
        d_window.upload(window.data(), window.size() * sizeof(float), "reset profile window");
        cuda_check(cudaDeviceSynchronize(), "finish mode fixture uploads");

        MtpDrafter d;
        cuda_check(cudaSetDevice(mode.target), "select target before MTP load");
        d.set_prompt_len(17);
        d.set_resident_embedding(true);
        d.set_mapped_residuals(mode.mapped);
        d.set_chain_batch(args.chain_batch);
        if (!d.load(args.mtp, g, ss, kMaxT, err, kContext, mode.draft))
            fail(std::string(mode.name) + " load: " + err);
        if (!d.bind(wt, &head, d_window.as<float>(), err))
            fail(std::string(mode.name) + " bind: " + err);
        if (!d.prefill(d_prefill.as<float>(), prefill_tokens.data(), 17, 0, err))
            fail(std::string(mode.name) + " prefill: " + err);

        auto one = [&](int iteration) {
            DraftResult r;
            if (!d.draft(4, draft_tokens.data(), 120, 3, r.ids.data(), err, r.probs.data(), 0.0f, &r.n))
                fail(std::string(mode.name) + " draft: " + err);
            validate_result(mode.name, iteration, r, n_vocab);
            return r;
        };
        DraftResult stable = one(-kWarm);
        for (int i = 1; i < kWarm; ++i) {
            const DraftResult got = one(-kWarm + i);
            validate_stable(mode.name, -kWarm + i, stable, got);
        }

        const double draft0 = d.ms_draft;
        const int64_t rounds0 = d.rounds;
        const int64_t transfers0 = d.transfer_count();
        const uint64_t bytes0 = d.transfer_bytes();
        const int64_t mapped0 = d.mapped_residual_count();
        const uint64_t mapped_bytes0 = d.mapped_residual_bytes();
        const int64_t batched0 = d.batched_rounds();
        const int64_t extra0 = d.extra_unused_computed_drafts();
        const Clock::time_point t0 = Clock::now();
        double active_wall_ms = 0.0;
        for (int i = 0; i < kIterations; ++i) {
            const Clock::time_point active0 = Clock::now();
            const DraftResult got = one(i);
            validate_stable(mode.name, i, stable, got);
            active_wall_ms += std::chrono::duration<double, std::milli>(Clock::now() - active0).count();
            // Warm-up is intentionally ungapped. Timed mode inserts exactly 63 requested sleeps BETWEEN its
            // 64 calls; these count in outer wall_ms but not in the sum of active call durations above.
            if (args.gap_ms > 0 && i + 1 < kIterations)
                std::this_thread::sleep_for(std::chrono::milliseconds(args.gap_ms));
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        const double draft_ms = d.ms_draft - draft0;
        const int64_t rounds = d.rounds - rounds0;
        const int64_t transfers = d.transfer_count() - transfers0;
        const uint64_t bytes = d.transfer_bytes() - bytes0;
        const int64_t mapped_reads = d.mapped_residual_count() - mapped0;
        const uint64_t mapped_bytes = d.mapped_residual_bytes() - mapped_bytes0;
        const int64_t batched_rounds = d.batched_rounds() - batched0;
        const int64_t unused_drafts = d.extra_unused_computed_drafts() - extra0;

        if (rounds != kIterations) fail(std::string(mode.name) + " MTP round counter mismatch");
        const int64_t expected_batched = args.chain_batch ? kIterations : 0;
        if (batched_rounds != expected_batched)
            fail(std::string(mode.name) + " chain-batch round counter mismatch");
        if (unused_drafts != 0)
            fail(std::string(mode.name) + " unexpectedly computed unused drafts at minP=0");
        const bool remote = mode.target != mode.draft;
        const int64_t expected_transfers = !remote ? 0 : mode.mapped ? kIterations : 2 * kIterations;
        const uint64_t expected_bytes = !remote ? 0 : (uint64_t) expected_transfers * residual_bytes;
        const int64_t expected_mapped = remote && mode.mapped ? kIterations : 0;
        const uint64_t expected_mapped_bytes = (uint64_t) expected_mapped * residual_bytes;
        if (transfers != expected_transfers || bytes != expected_bytes || mapped_reads != expected_mapped ||
            mapped_bytes != expected_mapped_bytes)
            fail(std::string(mode.name) + " transport counters do not match the fixed-round formula");

        std::printf("PROFILE mode=%s iterations=%d rounds=%lld chain=%d chain_batch=%d batched_rounds=%lld "
                    "unused_drafts=%lld gap_ms=%d wall_ms=%.3f "
                    "active_wall_ms=%.3f mtp_draft_ms=%.3f "
                    "transfers=%lld bytes=%llu mapped_reads=%lld mapped_bytes=%llu "
                    "ids=%d,%d,%d probs=%.9g,%.9g,%.9g\n",
                    mode.name, kIterations, (long long) rounds, stable.n, args.chain_batch ? 1 : 0,
                    (long long) batched_rounds, (long long) unused_drafts, args.gap_ms, wall_ms, active_wall_ms,
                    draft_ms, (long long) transfers, (unsigned long long) bytes, (long long) mapped_reads,
                    (unsigned long long) mapped_bytes, stable.ids[0], stable.ids[1], stable.ids[2], stable.probs[0],
                    stable.probs[1], stable.probs[2]);
        cuda_check(cudaSetDevice(mode.target), "restore target before profile MTP cleanup");
    }  // each MTP owner is destroyed before the next mode and before the shared owners below

    strata::core::set_native_embed(nullptr);
    cuda_check(cudaSetDevice(0), "restore target after profile");
    std::printf("PASS mtp_profile modes=4 iterations=64 chain=3\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mtp_profile_test: %s\n", e.what());
        return 1;
    }
}
