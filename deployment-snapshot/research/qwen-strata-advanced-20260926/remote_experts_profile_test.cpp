// Diagnostic burst profiler for the bounded resident expert service. This is not a full-model throughput claim.
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t H = 2560;
constexpr int64_t NE = 512;
constexpr int K = 10;
constexpr int LAYERS = 4;
constexpr int WARM_BURSTS = 8;
constexpr int BURSTS = 64;
constexpr size_t INPUT_BYTES = (size_t) 4 * H * sizeof(float);
constexpr size_t OUTPUT_BYTES = (size_t) 4 * K * H * sizeof(float);
constexpr size_t REFERENCE_BYTES = (size_t) BURSTS * 10 * K * H * sizeof(float);  // 62.5 MiB

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }
void ck(cudaError_t s, const char* what) { if (s != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(s)); }

void require_device0(const char* what) {
    int device = -1;
    ck(cudaGetDevice(&device), "query current CUDA device");
    if (device != 0) fail(std::string(what) + " did not restore CUDA device 0");
}

struct PortableBuffers {
    float* input = nullptr;
    float* output = nullptr;
    PortableBuffers() {
        ck(cudaSetDevice(0), "select target for portable profile buffers");
        ck(cudaHostAlloc((void**) &input, INPUT_BYTES, cudaHostAllocPortable), "allocate portable profile input");
        const cudaError_t status = cudaHostAlloc((void**) &output, OUTPUT_BYTES, cudaHostAllocPortable);
        if (status != cudaSuccess) {
            const cudaError_t cleanup = cudaFreeHost(input);
            input = nullptr;
            std::string why = std::string("allocate portable profile output: ") + cudaGetErrorString(status);
            if (cleanup != cudaSuccess) why += std::string("; input cleanup: ") + cudaGetErrorString(cleanup);
            fail(why);
        }
    }
    ~PortableBuffers() { if (input) (void) cudaFreeHost(input); if (output) (void) cudaFreeHost(output); }
};

uint64_t hash_bytes(uint64_t h, const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

void fill_call(PortableBuffers& b, int burst, int layer, int T, int k_count, std::vector<int32_t>& ids) {
    for (int64_t i = 0; i < (int64_t) T * H; ++i)
        b.input[i] = (float) ((int) ((i * 43 + burst * 17 + layer * 29) % 607) - 303) / 283.0f;
    ids.resize((size_t) T * k_count);
    int base = (burst * 13 + layer * 67) % (int) NE;
    if (base < 0) base += (int) NE;
    for (int t = 0; t < T; ++t) for (int k = 0; k < k_count; ++k)
        ids[(size_t) t * k_count + k] = (base + 41 * k + 7 * (t & 1)) % (int) NE;
}

struct Reference {
    std::vector<uint32_t> words;
    uint64_t digest = 0;
    bool ready = false;
};

std::vector<int32_t> parse_layers(const std::string& text) {
    std::vector<int32_t> out;
    size_t at = 0;
    while (at < text.size()) {
        const size_t comma = text.find(',', at);
        const std::string item = text.substr(at, comma == std::string::npos ? text.size() - at : comma - at);
        size_t used = 0;
        int value = 0;
        try { value = std::stoi(item, &used); } catch (...) { fail("invalid --layer-ids list"); }
        if (item.empty() || used != item.size()) fail("invalid --layer-ids list");
        out.push_back(value);
        if (comma == std::string::npos) break;
        at = comma + 1;
        if (at == text.size()) fail("invalid --layer-ids list");
    }
    if (out.size() != LAYERS) fail("--layer-ids must contain exactly four IDs");
    return out;
}

std::string join_layers(const std::vector<int32_t>& layers) {
    std::string out;
    for (int32_t layer : layers) { if (!out.empty()) out += ','; out += std::to_string(layer); }
    return out;
}

void run_device(int device, const std::string& pack, const std::string& native, PortableBuffers& buffers,
                Reference& reference, std::vector<uint32_t>& fallback_reference,
                const std::vector<int32_t>& layers, bool selected) {
    strata::core::RemoteExperts remote;
    std::string err;
    ck(cudaSetDevice(0), "select target before profile service load");
    if (selected) {
        if (!remote.load(pack, layers, device, err, native)) fail(err);
    } else if (!remote.load(pack, LAYERS, device, err, native)) fail(err);
    require_device0("profile service load");
    if (remote.layer_ids() != layers) fail("profile service returned different global layer IDs");
    const std::string layer_text = join_layers(remote.layer_ids());

    const auto initial_mode = remote.decode_mode();
    const uint64_t initial_vram = remote.vram_bytes();
    const int64_t initial_graph_calls = remote.decode_graph_calls();
    if (initial_mode != strata::core::RemoteDecodeMode::Original ||
        remote.set_decode_mode(static_cast<strata::core::RemoteDecodeMode>(99), err) ||
        remote.decode_mode() != initial_mode || remote.vram_bytes() != initial_vram ||
        remote.decode_graph_calls() != initial_graph_calls)
        fail("invalid decode mode changed service state");
    require_device0("invalid decode mode rejection");

    const strata::core::RemoteDecodeMode modes[] = {strata::core::RemoteDecodeMode::Original,
                                                    strata::core::RemoteDecodeMode::Packed,
                                                    strata::core::RemoteDecodeMode::Graphs};
    const char* mode_names[] = {"original", "packed", "graphs"};
    for (int mode_i = 0; mode_i < 3; ++mode_i) {
        if (!remote.set_decode_mode(modes[mode_i], err)) fail(err);
        require_device0("decode mode toggle");
        if (remote.decode_mode() != modes[mode_i]) fail("decode mode getter mismatch");
        if (!remote.set_decode_profiling(false, err)) fail(err);
        const int64_t fallback_graph_before = remote.decode_graph_calls();
        std::vector<int32_t> ids;
        size_t fallback_at = 0;
        const bool store_fallback = fallback_reference.empty();
        for (int layer : layers) {
            fill_call(buffers, -1000, layer, 1, 4, ids);
            if (!remote.decode(layer, buffers.input, ids.data(), 1, 4, buffers.output, err)) fail(err);
            require_device0("K4 fallback decode");
            const size_t words = (size_t) 4 * H;
            if (store_fallback) {
                const auto* begin = reinterpret_cast<const uint32_t*>(buffers.output);
                fallback_reference.insert(fallback_reference.end(), begin, begin + words);
            } else {
                if (fallback_at + words > fallback_reference.size() ||
                    std::memcmp(fallback_reference.data() + fallback_at, buffers.output, words * sizeof(float)) != 0)
                    fail("K4 packed/graph fallback differs from Original");
                fallback_at += words;
            }
        }
        if (!store_fallback && fallback_at != fallback_reference.size())
            fail("K4 fallback reference extent mismatch");
        if (remote.decode_graph_calls() != fallback_graph_before)
            fail("K4 graph mode did not use Packed fallback");

    for (int profile_i = 0; profile_i < 2; ++profile_i) for (int gap_i = 0; gap_i < 2; ++gap_i) {
        const bool profiling = profile_i != 0;
        const int gap_ms = gap_i ? 40 : 0;
        if (!remote.set_decode_profiling(profiling, err)) fail(err);
        require_device0("profiling toggle");
        for (int warm = -WARM_BURSTS; warm < 0; ++warm) {
            const int T = ((warm % 4) + 4) % 4 + 1;
            for (int layer : layers) {
                fill_call(buffers, warm, layer, T, K, ids);
                if (!remote.decode(layer, buffers.input, ids.data(), T, K, buffers.output, err)) fail(err);
                require_device0("warm decode");
            }
        }
        if (!remote.reset_decode_profile(err)) fail(err);
        const int64_t graph_before = remote.decode_graph_calls();

        std::vector<uint32_t> condition;
        if (!reference.ready) condition.reserve(REFERENCE_BYTES / sizeof(uint32_t));
        size_t reference_at = 0;
        uint64_t digest = 1469598103934665603ull;
        double active_ms = 0.0;
        for (int burst = 0; burst < BURSTS; ++burst) {
            const int T = burst % 4 + 1;
            for (int layer : layers) {
                fill_call(buffers, burst, layer, T, K, ids);
                const Clock::time_point active0 = Clock::now();
                if (!remote.decode(layer, buffers.input, ids.data(), T, K, buffers.output, err)) fail(err);
                active_ms += std::chrono::duration<double, std::milli>(Clock::now() - active0).count();
                require_device0("measured decode");
                const size_t words = (size_t) T * K * H;
                for (size_t i = 0; i < words; ++i)
                    if (!std::isfinite(buffers.output[i])) fail("profile produced a non-finite expert output");
                digest = hash_bytes(digest, buffers.output, words * sizeof(float));
                if (!reference.ready) {
                    const auto* begin = reinterpret_cast<const uint32_t*>(buffers.output);
                    condition.insert(condition.end(), begin, begin + words);
                } else {
                    if (reference_at + words > reference.words.size() ||
                        std::memcmp(reference.words.data() + reference_at, buffers.output, words * sizeof(float)) != 0)
                        fail("profile output bits differ across mode/device");
                    reference_at += words;
                }
            }
            if (gap_ms && burst + 1 < BURSTS) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
        }
        if (!reference.ready) {
            reference.words = std::move(condition);
            if (reference.words.size() * sizeof(uint32_t) != REFERENCE_BYTES)
                fail("profile reference exceeded its fixed extent");
            reference.digest = digest;
            reference.ready = true;
        } else if (reference_at != reference.words.size() || digest != reference.digest) {
            fail("profile exact reference extent/digest mismatch");
        }

        strata::core::RemoteDecodeProfile stats;
        if (!remote.decode_profile_snapshot(stats, err)) fail(err);
        require_device0("profile snapshot");
        const int64_t graph_delta = remote.decode_graph_calls() - graph_before;
        if (graph_delta != (mode_i == 2 ? BURSTS * LAYERS : 0))
            fail("decode graph replay count does not match selected mode");
        const bool finite_times = std::isfinite(active_ms) && std::isfinite(stats.cpu_prepare_ms) &&
                                  std::isfinite(stats.cpu_submit_ms) && std::isfinite(stats.cpu_wait_ms) &&
                                  std::isfinite(stats.cpu_total_ms) && std::isfinite(stats.gpu_upload_ms) &&
                                  std::isfinite(stats.gpu_kernel_ms) && std::isfinite(stats.gpu_download_ms);
        if (!finite_times || active_ms <= 0.0) fail("profile reported a non-finite/non-positive active time");
        if (profiling) {
            if (!stats.enabled || stats.calls != (uint64_t) BURSTS * LAYERS || stats.cpu_prepare_ms <= 0.0 ||
                stats.cpu_submit_ms <= 0.0 || stats.cpu_wait_ms <= 0.0 || stats.cpu_total_ms <= 0.0 ||
                stats.gpu_upload_ms <= 0.0 || stats.gpu_kernel_ms <= 0.0 || stats.gpu_download_ms <= 0.0)
                fail("enabled decode profile has missing counters/times");
        } else if (stats.enabled || stats.calls != 0 || stats.cpu_prepare_ms != 0.0 || stats.cpu_submit_ms != 0.0 ||
                   stats.cpu_wait_ms != 0.0 || stats.cpu_total_ms != 0.0 || stats.gpu_upload_ms != 0.0 ||
                   stats.gpu_kernel_ms != 0.0 || stats.gpu_download_ms != 0.0) {
            fail("disabled decode profile collected data");
        }
        std::printf("{\"device\":%d,\"layer_ids\":\"%s\",\"payload_bytes\":%llu,\"mode\":\"%s\",\"profiling\":%s,\"gap_ms\":%d,\"bursts\":%d,\"calls\":%d,"
                    "\"graph_replays\":%lld,"
                    "\"profile_calls\":%llu,\"active_decode_wall_ms\":%.3f,"
                    "\"timing_scope\":\"decode_calls_only_checks_and_pauses_excluded\",\"cpu_prepare_ms\":%.3f,"
                    "\"cpu_api_submit_ms\":%.3f,\"cpu_wait_ms\":%.3f,\"cpu_total_ms\":%.3f,"
                    "\"gpu_upload_stream_ms\":%.3f,\"gpu_kernel_stream_ms\":%.3f,"
                    "\"gpu_download_stream_ms\":%.3f,\"input_bytes\":%zu,"
                    "\"output_bytes\":%zu,\"reference_bytes\":%zu,\"digest\":\"%016llx\"}\n",
                    device, layer_text.c_str(), (unsigned long long) remote.payload_bytes(), mode_names[mode_i],
                    profiling ? "true" : "false", gap_ms, BURSTS,
                    BURSTS * LAYERS,
                    (long long) graph_delta,
                    (unsigned long long) stats.calls, active_ms, stats.cpu_prepare_ms, stats.cpu_submit_ms,
                    stats.cpu_wait_ms, stats.cpu_total_ms, stats.gpu_upload_ms, stats.gpu_kernel_ms,
                    stats.gpu_download_ms, INPUT_BYTES, OUTPUT_BYTES, REFERENCE_BYTES,
                    (unsigned long long) digest);
        std::fflush(stdout);
    }
    }
    if (!remote.set_decode_profiling(false, err)) fail(err);
    require_device0("profile event cleanup");
}

int run(const std::string& pack, const std::string& native, const std::vector<int32_t>& layers, bool selected) {
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "query CUDA devices");
    if (devices < 2) fail("remote_experts_profile_test requires two CUDA devices");
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(pack, 48, NE, err)) fail(err);
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.n_layers != 48 || lay.n_expert != NE) fail("profile requires the guarded native pack");
    for (int layer : layers) if (layer < 0 || layer >= lay.n_layers) fail("profile layer ID is out of range");
    PortableBuffers buffers;
    Reference reference;
    std::vector<uint32_t> fallback_reference;
    run_device(0, pack, native, buffers, reference, fallback_reference, layers, selected);
    require_device0("GPU0 profile owner destruction");
    run_device(1, pack, native, buffers, reference, fallback_reference, layers, selected);
    require_device0("GPU1 profile owner destruction");
    if (selected)
        std::printf("PASS remote_experts_profile bursts=64 layers=4 modes=3 conditions=24 ids=%s outputs=exact\n",
                    join_layers(layers).c_str());
    else
        std::printf("PASS remote_experts_profile bursts=64 layers=4 modes=3 outputs=exact\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string pack, native, layer_text;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--pack" && i + 1 < argc) pack = argv[++i];
            else if (arg == "--native" && i + 1 < argc) native = argv[++i];
            else if (arg == "--layer-ids" && i + 1 < argc) layer_text = argv[++i];
            else fail("usage: remote_experts_profile_test --pack PATH --native GGUF [--layer-ids 22,28,30,34]");
        }
        if (pack.empty() || native.empty())
            fail("usage: remote_experts_profile_test --pack PATH --native GGUF [--layer-ids 22,28,30,34]");
        const bool selected = !layer_text.empty();
        const std::vector<int32_t> layers = selected ? parse_layers(layer_text) : std::vector<int32_t>{0, 1, 2, 3};
        return run(pack, native, layers, selected);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL remote_experts_profile_test: %s\n", e.what());
        return 1;
    }
}
