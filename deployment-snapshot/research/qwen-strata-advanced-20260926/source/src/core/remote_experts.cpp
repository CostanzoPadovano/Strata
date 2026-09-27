#include "strata/core/remote_experts.hpp"

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

namespace strata::core {
namespace {

constexpr int64_t H = 2560;
constexpr int64_t FF = 640;
constexpr int64_t NE = 512;
constexpr int64_t MAX_T = 2048;
constexpr int64_t MAX_K = 10;
constexpr int64_t MAX_DECODE_T = 4;
constexpr int64_t MAX_GLOBAL_LAYERS = 48;
constexpr int64_t MAX_ENTRIES = MAX_T * MAX_K;
constexpr int64_t MAX_DECODE_ENTRIES = MAX_DECODE_T * MAX_K;
constexpr uint64_t VRAM_CAP = 8ull << 30;
constexpr size_t STAGING_BYTES = 32u << 20;
constexpr size_t GEMM_WORKSPACE_BYTES = 32u << 20;
constexpr size_t Q8_ROW_BYTES = (size_t) (H / 32) * 36;
constexpr size_t align_up(size_t n, size_t a) { return (n + a - 1) / a * a; }
constexpr size_t PACKET_X = 0;
constexpr size_t PACKET_PTR = align_up(PACKET_X + MAX_DECODE_T * H * sizeof(float), 16);
constexpr size_t PACKET_START = align_up(PACKET_PTR + MAX_DECODE_ENTRIES * sizeof(unsigned long long), 16);
constexpr size_t PACKET_N_GROUPS = align_up(PACKET_START + (MAX_DECODE_ENTRIES + 1) * sizeof(int32_t), 16);
constexpr size_t PACKET_DST = align_up(PACKET_N_GROUPS + sizeof(int32_t), 16);
constexpr size_t PACKET_TOK = align_up(PACKET_DST + MAX_DECODE_ENTRIES * sizeof(int32_t), 16);
constexpr size_t PACKET_BYTES = align_up(PACKET_TOK + MAX_DECODE_ENTRIES * sizeof(int32_t), 256);
static_assert(PACKET_BYTES <= 64u << 10, "decode packet must remain bounded to 64 KiB");
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string cuda_error(const char* what, cudaError_t status) {
    return std::string("remote experts: ") + what + ": " + cudaGetErrorString(status);
}

class DeviceScope {
public:
    bool enter(int device, std::string& err) {
        cudaError_t s = cudaGetDevice(&old_);
        if (s == cudaSuccess) s = cudaSetDevice(device);
        if (s != cudaSuccess) { err = cuda_error("select device", s); return false; }
        active_ = true;
        return true;
    }
    bool leave(std::string& err) {
        if (!active_) return true;
        active_ = false;
        const cudaError_t s = cudaSetDevice(old_);
        if (s != cudaSuccess) {
            const std::string why = cuda_error("restore device", s);
            if (err.empty()) err = why; else err += "; " + why;
            return false;
        }
        return true;
    }
    ~DeviceScope() { if (active_) (void) cudaSetDevice(old_); }
private:
    int old_ = -1;
    bool active_ = false;
};

class BusyGuard {
public:
    explicit BusyGuard(std::atomic<bool>& busy) : busy_(busy) {
        bool expected = false;
        acquired_ = busy_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    }
    ~BusyGuard() { if (acquired_) busy_.store(false, std::memory_order_release); }
    bool acquired() const { return acquired_; }
private:
    std::atomic<bool>& busy_;
    bool acquired_ = false;
};

bool seek_file(std::FILE* f, uint64_t off) {
#if defined(_WIN32)
    return off <= (uint64_t) std::numeric_limits<__int64>::max() && _fseeki64(f, (__int64) off, SEEK_SET) == 0;
#else
    return off <= (uint64_t) std::numeric_limits<off_t>::max() && fseeko(f, (off_t) off, SEEK_SET) == 0;
#endif
}

}  // namespace

struct RemoteExperts::Impl {
    int device = -1;
    int layers = 0;
    bool attempted = false;
    bool loaded = false;
    std::atomic<bool> busy{false};
    uint64_t vram = 0;
    uint64_t payload = 0;
    int64_t decode_calls = 0;
    int64_t masked_decode_calls = 0;
    int64_t masked_decode_entries = 0;
    int64_t masked_all_hit_calls = 0;
    int64_t prefill_calls = 0;
    int64_t decode_graph_calls = 0;
    RemoteDecodeMode decode_mode = RemoteDecodeMode::Original;
    bool decode_profile_enabled = false;
    std::array<cudaEvent_t, 4> decode_profile_events{};
    RemoteDecodeProfile decode_profile{};
    cudaStream_t stream = nullptr;
    std::unique_ptr<strata::prefill::Gemm> gemm;
    std::array<void*, 8> layer_data{};
    std::array<uint64_t, 8> layer_bytes{};
    std::array<strata::kernels::cpu::NativeFmt, 8> fmt{};
    std::vector<int32_t> global_layers;
    std::array<int32_t, MAX_GLOBAL_LAYERS> slot_by_global{};
    void* host_stage = nullptr;
    void* d_decode_packet = nullptr;
    std::array<std::array<cudaGraphExec_t, MAX_DECODE_T>, 8> decode_graphs{};

    float* d_decode_x = nullptr;
    void* d_decode_q8 = nullptr;
    float* d_decode_out = nullptr;
    void* d_decode_scratch = nullptr;
    unsigned long long* d_group_ptr = nullptr;
    int32_t* d_group_start = nullptr;
    int32_t* d_n_groups = nullptr;
    int32_t* d_ent_dst = nullptr;
    int32_t* d_ent_tok = nullptr;

    uint16_t* d_mixed = nullptr;
    int32_t* d_src = nullptr;
    uint16_t* d_xs = nullptr;
    float* d_gu = nullptr;
    uint16_t* d_h = nullptr;
    float* d_dm = nullptr;
    uint16_t* d_dq_gu = nullptr;
    uint16_t* d_dq_down = nullptr;
    void* d_gemm_workspace = nullptr;

    ~Impl() {
        if (device < 0) return;
        int old = -1;
        const bool restore = cudaGetDevice(&old) == cudaSuccess && old != device;
        (void) cudaSetDevice(device);
        if (stream) (void) cudaStreamSynchronize(stream);
        for (auto& layer : decode_graphs)
            for (cudaGraphExec_t exec : layer) if (exec) (void) cudaGraphExecDestroy(exec);
        for (cudaEvent_t event : decode_profile_events) if (event) (void) cudaEventDestroy(event);
        gemm.reset();
        for (void* p : layer_data) if (p) (void) cudaFree(p);
        void* ptrs[] = {d_decode_x, d_decode_q8, d_decode_out, d_decode_scratch, d_group_ptr, d_group_start,
                        d_n_groups, d_ent_dst, d_ent_tok, d_mixed, d_src, d_xs, d_gu, d_h, d_dm, d_dq_gu,
                        d_dq_down, d_gemm_workspace, d_decode_packet};
        for (void* p : ptrs) if (p) (void) cudaFree(p);
        if (stream) (void) cudaStreamDestroy(stream);
        if (host_stage) (void) cudaFreeHost(host_stage);
        if (restore) (void) cudaSetDevice(old);
    }

    bool alloc(void** dst, uint64_t bytes, const char* what, std::string& err) {
        if (bytes > VRAM_CAP - vram) { err = std::string("remote experts: 8 GiB cap exceeded by ") + what; return false; }
        const cudaError_t s = cudaMalloc(dst, (size_t) bytes);
        if (s != cudaSuccess) {
            size_t free_b = 0, total_b = 0;
            std::ostringstream os;
            os << cuda_error(what, s) << " (requested " << bytes << " bytes";
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) os << ", free " << free_b << " of " << total_b;
            os << ")";
            err = os.str();
            return false;
        }
        vram += bytes;
        return true;
    }

    bool sync(const char* what, std::string& err) {
        const cudaError_t s = cudaStreamSynchronize(stream);
        if (s != cudaSuccess) { err = cuda_error(what, s); return false; }
        return true;
    }

    int slot(int64_t global_layer) const {
        if (!loaded || global_layer < 0 || global_layer >= MAX_GLOBAL_LAYERS) return -1;
        return slot_by_global[(size_t) global_layer];
    }

    bool copy_from_target(int target, cudaStream_t producer, void* dst, const void* src, uint64_t bytes,
                          std::string& err) {
        cudaError_t s = cudaSetDevice(target);
        if (s == cudaSuccess) s = cudaStreamSynchronize(producer);
        if (s != cudaSuccess) { err = cuda_error("synchronize prefill producer", s); return false; }
        if (target == device) {
            s = cudaMemcpyAsync(dst, src, (size_t) bytes, cudaMemcpyDeviceToDevice, stream);
            if (s != cudaSuccess) { err = cuda_error("copy local prefill input", s); return false; }
            return true;
        }
        auto* d = static_cast<uint8_t*>(dst);
        const auto* q = static_cast<const uint8_t*>(src);
        for (uint64_t off = 0; off < bytes;) {
            const size_t n = (size_t) std::min<uint64_t>(STAGING_BYTES, bytes - off);
            s = cudaSetDevice(target);
            if (s == cudaSuccess) s = cudaMemcpy(host_stage, q + off, n, cudaMemcpyDeviceToHost);
            if (s == cudaSuccess) s = cudaSetDevice(device);
            if (s == cudaSuccess) s = cudaMemcpyAsync(d + off, host_stage, n, cudaMemcpyHostToDevice, stream);
            if (s == cudaSuccess) s = cudaStreamSynchronize(stream);  // staging cannot be reused before H2D completes
            if (s != cudaSuccess) { err = cuda_error("stage prefill input", s); return false; }
            off += n;
        }
        return true;
    }

    bool copy_to_target(int target, cudaStream_t consumer, void* dst, const void* src, uint64_t bytes,
                        std::string& err) {
        cudaError_t s = cudaSuccess;
        if (target == device) {
            s = cudaMemcpyAsync(dst, src, (size_t) bytes, cudaMemcpyDeviceToDevice, consumer);
            if (s == cudaSuccess) s = cudaStreamSynchronize(consumer);
            if (s != cudaSuccess) { err = cuda_error("copy local prefill output", s); return false; }
            return true;
        }
        auto* d = static_cast<uint8_t*>(dst);
        const auto* q = static_cast<const uint8_t*>(src);
        for (uint64_t off = 0; off < bytes;) {
            const size_t n = (size_t) std::min<uint64_t>(STAGING_BYTES, bytes - off);
            s = cudaSetDevice(device);
            if (s == cudaSuccess) s = cudaMemcpy(host_stage, q + off, n, cudaMemcpyDeviceToHost);
            if (s == cudaSuccess) s = cudaSetDevice(target);
            if (s == cudaSuccess) s = cudaMemcpyAsync(d + off, host_stage, n, cudaMemcpyHostToDevice, consumer);
            if (s == cudaSuccess) s = cudaStreamSynchronize(consumer);  // staging cannot be reused before target H2D
            if (s != cudaSuccess) { err = cuda_error("stage prefill output", s); return false; }
            off += n;
        }
        return true;
    }
};

RemoteExperts::RemoteExperts() : p_(new Impl) {}
RemoteExperts::~RemoteExperts() = default;

bool RemoteExperts::load(const std::string& pack_dir, int layer_count, int device, std::string& err,
                         const std::string& native_shard) {
    if (layer_count < 0 || layer_count > 8) {
        if (p_->attempted) err = "remote experts: load called twice";
        else { p_->attempted = true; err = "remote experts: layer_count must be in [0,8]"; }
        return false;
    }
    std::vector<int32_t> ids((size_t) layer_count);
    std::iota(ids.begin(), ids.end(), 0);
    return load(pack_dir, ids, device, err, native_shard);
}

bool RemoteExperts::load(const std::string& pack_dir, const std::vector<int32_t>& layer_ids, int device,
                         std::string& err, const std::string& native_shard) {
    if (p_->attempted) { err = "remote experts: load called twice"; return false; }
    p_->attempted = true;
    if (layer_ids.size() > 8) { err = "remote experts: at most eight layer IDs may be resident"; return false; }
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.n_expert != NE || lay.n_layers < 0 || lay.n_layers > MAX_GLOBAL_LAYERS) {
        err = "remote experts: process-wide native ExpertLayout is absent or incompatible";
        return false;
    }
    for (size_t slot = 0; slot < layer_ids.size(); ++slot) {
        const int32_t global = layer_ids[slot];
        if (global < 0 || global >= lay.n_layers) {
            err = "remote experts: global layer ID is out of range";
            return false;
        }
        if (slot != 0 && layer_ids[slot - 1] >= global) {
            err = "remote experts: global layer IDs must be strictly increasing and unique";
            return false;
        }
        if ((size_t) global >= lay.fmt.size() || (size_t) global >= lay.bytes.size() ||
            (size_t) global >= lay.offset.size()) {
            err = "remote experts: process-wide native ExpertLayout is incomplete for selected layer " +
                  std::to_string(global);
            return false;
        }
        const auto& f = lay.fmt[(size_t) global];
        if (f.n_embd != H || f.n_ff != FF || f.bytes != lay.blob_bytes(global) || f.up_off == 0 ||
            f.up_off > std::numeric_limits<size_t>::max() / 2 || f.down_off != 2 * f.up_off ||
            f.down_off >= f.bytes) {
            err = "remote experts: native layer geometry/blob size mismatch at layer " + std::to_string(global);
            return false;
        }
    }
    int ndev = 0;
    cudaError_t query = cudaGetDeviceCount(&ndev);
    if (query != cudaSuccess || device < 0 || device >= ndev) {
        err = query == cudaSuccess ? "remote experts: CUDA device ordinal out of range" : cuda_error("device query", query);
        return false;
    }
    p_->device = device;
    p_->layers = (int) layer_ids.size();
    p_->global_layers = layer_ids;
    p_->slot_by_global.fill(-1);
    for (size_t slot = 0; slot < layer_ids.size(); ++slot)
        p_->slot_by_global[(size_t) layer_ids[slot]] = (int32_t) slot;
    if (layer_ids.empty()) { p_->loaded = true; return true; }
    std::error_code ec;
    const std::filesystem::path file_path = std::filesystem::path(pack_dir) / "experts.bin";
    const bool packed_exists = std::filesystem::exists(file_path, ec);
    if (ec) { err = "remote experts: cannot inspect experts.bin"; return false; }
    std::array<std::filesystem::path, 8> layer_source{};
    if (packed_exists) {
        const uint64_t file_bytes = std::filesystem::file_size(file_path, ec);
        if (ec || file_bytes != lay.total) {
            err = "remote experts: experts.bin size does not match the immutable ExpertLayout";
            return false;
        }
    } else {
        if (native_shard.empty()) {
            err = "remote experts: experts.bin is absent; an explicit native GGUF with complete offsets is required";
            return false;
        }
        std::filesystem::path primary = std::filesystem::weakly_canonical(native_shard, ec);
        if (ec || !std::filesystem::is_regular_file(primary, ec) || ec) {
            err = "remote experts: explicit native GGUF is not a readable regular file";
            return false;
        }
        for (int slot = 0; slot < p_->layers; ++slot) {
            const int global = p_->global_layers[(size_t) slot];
            if (lay.gguf_off.size() <= (size_t) (3 * global + 2) ||
                (!lay.gguf_file.empty() && lay.gguf_file.size() <= (size_t) global)) {
                err = "remote experts: native layout has incomplete GGUF metadata for layer " +
                      std::to_string(global);
                return false;
            }
            std::filesystem::path source = primary;
            if (!lay.gguf_file.empty() && !lay.gguf_file[(size_t) global].empty()) {
                const std::filesystem::path sibling(lay.gguf_file[(size_t) global]);
                if (sibling.empty() || sibling == "." || sibling == ".." || sibling.is_absolute() ||
                    sibling.has_parent_path() || sibling.filename() != sibling) {
                    err = "remote experts: per-layer GGUF shard name is not a safe sibling filename";
                    return false;
                }
                source = primary.parent_path() / sibling;
            }
            source = std::filesystem::weakly_canonical(source, ec);
            if (ec || !std::filesystem::is_regular_file(source, ec) || ec || source.parent_path() != primary.parent_path()) {
                err = "remote experts: required per-layer GGUF shard is absent or outside the explicit shard directory";
                return false;
            }
            const uint64_t source_bytes = std::filesystem::file_size(source, ec);
            if (ec) { err = "remote experts: cannot size a required GGUF shard"; return false; }
            const auto& fm = lay.fmt[(size_t) global];
            const uint64_t blob = lay.blob_bytes(global);
            const uint64_t per[3] = {(uint64_t) fm.up_off, (uint64_t) fm.up_off, blob - (uint64_t) fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const uint64_t src = lay.gguf_off[(size_t) (3 * global + r)];
                if (per[r] == 0 || src == 0 || src > source_bytes || per[r] > (source_bytes - src) / NE) {
                    err = "remote experts: GGUF expert tensor slice is outside its shard at layer " +
                          std::to_string(global);
                    return false;
                }
            }
            layer_source[(size_t) slot] = source;
        }
    }

    uint64_t payload = 0;
    for (int global : p_->global_layers) {
        const uint64_t b = lay.blob_bytes(global);
        if (b > (VRAM_CAP - payload) / NE) { err = "remote experts: resident payload exceeds 8 GiB"; return false; }
        payload += b * NE;
    }
    const uint64_t fixed =
        MAX_DECODE_T * H * sizeof(float) + MAX_DECODE_T * Q8_ROW_BYTES + MAX_DECODE_ENTRIES * H * sizeof(float) +
        strata::kernels::native_expert_scratch_bytes(MAX_DECODE_ENTRIES, FF) +
        MAX_DECODE_ENTRIES * (sizeof(unsigned long long) + 3 * sizeof(int32_t)) +
        (MAX_DECODE_ENTRIES + 1) * sizeof(int32_t) +
        MAX_T * H * sizeof(uint16_t) + MAX_ENTRIES * sizeof(int32_t) +
        MAX_ENTRIES * H * sizeof(uint16_t) + MAX_ENTRIES * 1280ull * sizeof(float) +
        MAX_ENTRIES * FF * sizeof(uint16_t) + MAX_ENTRIES * H * sizeof(float) +
        1280ull * H * sizeof(uint16_t) + H * FF * sizeof(uint16_t) + GEMM_WORKSPACE_BYTES;
    if (payload > VRAM_CAP - fixed) { err = "remote experts: payload plus fixed workspace exceeds 8 GiB"; return false; }

    DeviceScope scope;
    if (!scope.enter(device, err)) return false;
    const bool ok = [&]() -> bool {
        if (cudaStreamCreateWithFlags(&p_->stream, cudaStreamNonBlocking) != cudaSuccess) {
            err = "remote experts: nonblocking stream allocation failed";
            return false;
        }
        if (cudaHostAlloc(&p_->host_stage, STAGING_BYTES, cudaHostAllocPortable) != cudaSuccess) {
            err = "remote experts: 32 MiB portable staging allocation failed";
            return false;
        }
        bool read_ok = true;
        std::FILE* packed = packed_exists ? std::fopen(file_path.string().c_str(), "rb") : nullptr;
        if (packed_exists && !packed) { err = "remote experts: cannot open experts.bin"; return false; }
        for (int slot = 0; slot < p_->layers && read_ok; ++slot) {
            const int global = p_->global_layers[(size_t) slot];
            const auto& fm = lay.fmt[(size_t) global];
            p_->fmt[(size_t) slot] = fm;
            p_->layer_bytes[(size_t) slot] = lay.blob_bytes(global);
            const uint64_t blob = p_->layer_bytes[(size_t) slot];
            const uint64_t bytes = blob * NE;
            if (!p_->alloc(&p_->layer_data[(size_t) slot], bytes, "resident layer payload", err)) {
                read_ok = false;
                break;
            }
            p_->payload += bytes;
            if (packed_exists) {
                if (!seek_file(packed, lay.layer_offset(global))) { err = "remote experts: experts.bin seek failed"; read_ok = false; break; }
                for (uint64_t off = 0; off < bytes;) {
                    const size_t n = (size_t) std::min<uint64_t>(STAGING_BYTES, bytes - off);
                    if (std::fread(p_->host_stage, 1, n, packed) != n ||
                        cudaMemcpyAsync((uint8_t*) p_->layer_data[(size_t) slot] + off, p_->host_stage, n,
                                        cudaMemcpyHostToDevice, p_->stream) != cudaSuccess ||
                        cudaStreamSynchronize(p_->stream) != cudaSuccess) {
                        err = "remote experts: bounded experts.bin upload failed";
                        read_ok = false;
                        break;
                    }
                    off += n;
                }
                continue;
            }

            std::FILE* shard = std::fopen(layer_source[(size_t) slot].string().c_str(), "rb");
            if (!shard) { err = "remote experts: cannot open required GGUF shard"; read_ok = false; break; }
            const uint64_t per[3] = {(uint64_t) fm.up_off, (uint64_t) fm.up_off, blob - (uint64_t) fm.down_off};
            const uint64_t at[3] = {0, (uint64_t) fm.up_off, (uint64_t) fm.down_off};
            for (int r = 0; r < 3 && read_ok; ++r) {
                const uint64_t src = lay.gguf_off[(size_t) (3 * global + r)];
                const uint64_t cap_experts = STAGING_BYTES / per[r];
                if (cap_experts == 0) { err = "remote experts: one GGUF expert slice exceeds staging"; read_ok = false; break; }
                for (uint64_t e0 = 0; e0 < NE;) {
                    const uint64_t count = std::min<uint64_t>(cap_experts, NE - e0);
                    const size_t n = (size_t) (count * per[r]);
                    if (!seek_file(shard, src + e0 * per[r]) || std::fread(p_->host_stage, 1, n, shard) != n) {
                        err = "remote experts: bounded GGUF expert read failed";
                        read_ok = false;
                        break;
                    }
                    for (uint64_t q = 0; q < count; ++q) {
                        const cudaError_t s = cudaMemcpyAsync((uint8_t*) p_->layer_data[(size_t) slot] +
                                                                  (e0 + q) * blob + at[r],
                                                              (uint8_t*) p_->host_stage + q * per[r], (size_t) per[r],
                                                              cudaMemcpyHostToDevice, p_->stream);
                        if (s != cudaSuccess) { err = cuda_error("enqueue GGUF expert slice", s); read_ok = false; break; }
                    }
                    if (!read_ok) break;
                    if (cudaStreamSynchronize(p_->stream) != cudaSuccess) {
                        err = "remote experts: bounded GGUF expert upload failed";
                        read_ok = false;
                        break;
                    }
                    e0 += count;
                }
            }
            std::fclose(shard);
        }
        if (packed) std::fclose(packed);
        if (!read_ok) return false;

        auto A = [&](auto** q, uint64_t n, const char* what) { return p_->alloc((void**) q, n, what, err); };
        if (!A(&p_->d_decode_x, MAX_DECODE_T * H * sizeof(float), "decode activation") ||
            !A(&p_->d_decode_q8, MAX_DECODE_T * Q8_ROW_BYTES, "decode q8") ||
            !A(&p_->d_decode_out, MAX_DECODE_ENTRIES * H * sizeof(float), "decode output") ||
            !A(&p_->d_decode_scratch, strata::kernels::native_expert_scratch_bytes(MAX_DECODE_ENTRIES, FF), "decode scratch") ||
            !A(&p_->d_group_ptr, MAX_DECODE_ENTRIES * sizeof(unsigned long long), "decode group pointers") ||
            !A(&p_->d_group_start, (MAX_DECODE_ENTRIES + 1) * sizeof(int32_t), "decode group offsets") ||
            !A(&p_->d_n_groups, sizeof(int32_t), "decode group count") ||
            !A(&p_->d_ent_dst, MAX_DECODE_ENTRIES * sizeof(int32_t), "decode destinations") ||
            !A(&p_->d_ent_tok, MAX_DECODE_ENTRIES * sizeof(int32_t), "decode tokens") ||
            !A(&p_->d_mixed, MAX_T * H * sizeof(uint16_t), "prefill mixed") ||
            !A(&p_->d_src, MAX_ENTRIES * sizeof(int32_t), "prefill source rows") ||
            !A(&p_->d_xs, MAX_ENTRIES * H * sizeof(uint16_t), "prefill gathered rows") ||
            !A(&p_->d_gu, MAX_ENTRIES * 1280ull * sizeof(float), "prefill gate/up") ||
            !A(&p_->d_h, MAX_ENTRIES * FF * sizeof(uint16_t), "prefill hidden") ||
            !A(&p_->d_dm, MAX_ENTRIES * H * sizeof(float), "prefill output") ||
            !A(&p_->d_dq_gu, 1280ull * H * sizeof(uint16_t), "prefill gate/up dequant") ||
            !A(&p_->d_dq_down, H * FF * sizeof(uint16_t), "prefill down dequant") ||
            !A(&p_->d_gemm_workspace, GEMM_WORKSPACE_BYTES, "cuBLAS workspace")) return false;
        p_->gemm.reset(new strata::prefill::Gemm);
        if (!p_->gemm->init_external(p_->stream, nullptr, 0, p_->d_gemm_workspace, GEMM_WORKSPACE_BYTES, err)) return false;
        p_->loaded = true;
        return true;
    }();
    const bool restored = scope.leave(err);
    if (!restored) p_->loaded = false;
    return ok && restored;
}

bool RemoteExperts::owns(int64_t layer) const { return p_->slot(layer) >= 0; }

bool RemoteExperts::decode(int64_t layer, const float* x_f_host, const int32_t* ids_host, int64_t n_tok, int64_t k,
                           float* out_host, std::string& err) {
    BusyGuard busy(p_->busy);
    if (!busy.acquired()) { err = "remote experts: overlapping request"; return false; }
    const int layer_slot = p_->slot(layer);
    if (layer_slot < 0 || !x_f_host || !ids_host || !out_host || n_tok < 1 || n_tok > MAX_DECODE_T || k < 1 || k > MAX_K) {
        err = "remote experts: invalid decode request";
        return false;
    }
    const bool profile = p_->decode_profile_enabled;
    Clock::time_point call_start{}, prepare_done{}, submit_done{}, wait_done{};
    double cpu_prepare = 0.0, cpu_submit = 0.0, cpu_wait = 0.0;
    double gpu_upload = 0.0, gpu_kernel = 0.0, gpu_download = 0.0;
    bool replayed_graph = false;
    if (profile) call_start = Clock::now();
    DeviceScope scope;
    if (!scope.enter(p_->device, err)) return false;
    const bool ok = [&]() -> bool {
        const int entries = (int) (n_tok * k);
        std::array<int32_t, NE> cnt{};
        for (int i = 0; i < entries; ++i) {
            const int32_t e = ids_host[i];
            if (e < 0 || e >= NE) { err = "remote experts: decode expert id out of range"; return false; }
            ++cnt[(size_t) e];
        }
        std::vector<unsigned long long> ptr;
        std::vector<int32_t> start(1, 0), dst, tok;
        ptr.reserve((size_t) entries); dst.reserve((size_t) entries); tok.reserve((size_t) entries);
        for (int e = 0; e < NE; ++e) if (cnt[(size_t) e]) {
            ptr.push_back((unsigned long long) ((uint8_t*) p_->layer_data[(size_t) layer_slot] +
                                               (uint64_t) e * p_->layer_bytes[(size_t) layer_slot]));
            for (int i = 0; i < entries; ++i) if (ids_host[i] == e) { dst.push_back(i); tok.push_back(i / (int) k); }
            start.push_back((int32_t) dst.size());
        }
        const int32_t ng = (int32_t) ptr.size();
        const bool packed = p_->decode_mode != RemoteDecodeMode::Original;
        const bool graph = p_->decode_mode == RemoteDecodeMode::Graphs && k == MAX_K;
        float* decode_x = p_->d_decode_x;
        auto* group_ptr = p_->d_group_ptr;
        auto* group_start = p_->d_group_start;
        auto* n_groups = p_->d_n_groups;
        auto* ent_dst = p_->d_ent_dst;
        auto* ent_tok = p_->d_ent_tok;
        if (packed) {
            auto* hp = static_cast<uint8_t*>(p_->host_stage);
            std::memset(hp, 0, PACKET_BYTES);
            std::memcpy(hp + PACKET_X, x_f_host, (size_t) n_tok * H * sizeof(float));
            std::memcpy(hp + PACKET_PTR, ptr.data(), ptr.size() * sizeof(ptr[0]));
            std::memcpy(hp + PACKET_START, start.data(), start.size() * sizeof(start[0]));
            std::memcpy(hp + PACKET_N_GROUPS, &ng, sizeof(ng));
            std::memcpy(hp + PACKET_DST, dst.data(), dst.size() * sizeof(dst[0]));
            std::memcpy(hp + PACKET_TOK, tok.data(), tok.size() * sizeof(tok[0]));
            auto* dp = static_cast<uint8_t*>(p_->d_decode_packet);
            decode_x = reinterpret_cast<float*>(dp + PACKET_X);
            group_ptr = reinterpret_cast<unsigned long long*>(dp + PACKET_PTR);
            group_start = reinterpret_cast<int32_t*>(dp + PACKET_START);
            n_groups = reinterpret_cast<int32_t*>(dp + PACKET_N_GROUPS);
            ent_dst = reinterpret_cast<int32_t*>(dp + PACKET_DST);
            ent_tok = reinterpret_cast<int32_t*>(dp + PACKET_TOK);
        }
        cudaError_t s = cudaSuccess;
        if (profile) {
            prepare_done = Clock::now();
            s = cudaEventRecord(p_->decode_profile_events[0], p_->stream);
        }
        if (packed) {
            if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_decode_packet, p_->host_stage, PACKET_BYTES,
                                                       cudaMemcpyHostToDevice, p_->stream);
        } else {
            if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_decode_x, x_f_host, (size_t) n_tok * H * sizeof(float),
                                                       cudaMemcpyHostToDevice, p_->stream);
            if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_group_ptr, ptr.data(), ptr.size() * sizeof(ptr[0]), cudaMemcpyHostToDevice, p_->stream);
            if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_group_start, start.data(), start.size() * sizeof(start[0]), cudaMemcpyHostToDevice, p_->stream);
            if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_n_groups, &ng, sizeof(ng), cudaMemcpyHostToDevice, p_->stream);
            if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_ent_dst, dst.data(), dst.size() * sizeof(dst[0]), cudaMemcpyHostToDevice, p_->stream);
            if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_ent_tok, tok.data(), tok.size() * sizeof(tok[0]), cudaMemcpyHostToDevice, p_->stream);
        }
        if (profile && s == cudaSuccess) s = cudaEventRecord(p_->decode_profile_events[1], p_->stream);
        if (s != cudaSuccess) { err = cuda_error("upload decode request", s); return false; }
        if (graph) {
            s = cudaGraphLaunch(p_->decode_graphs[(size_t) layer_slot][(size_t) n_tok - 1], p_->stream);
            if (s != cudaSuccess) { err = cuda_error("launch decode graph", s); return false; }
            replayed_graph = true;
        } else {
            strata::kernels::quantize_q8_1_rows(decode_x, n_tok, H, p_->d_decode_q8, p_->stream);
            const auto& f = p_->fmt[(size_t) layer_slot];
            const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
            strata::kernels::native_expert_grouped(L, group_ptr, group_start, n_groups, ent_dst, ent_tok,
                                                   entries, entries, p_->d_decode_q8, p_->d_decode_scratch,
                                                   p_->d_decode_out, p_->stream);
        }
        if (profile) {
            s = cudaEventRecord(p_->decode_profile_events[2], p_->stream);
            if (s != cudaSuccess) { err = cuda_error("record decode kernel timing", s); return false; }
        }
        s = cudaMemcpyAsync(out_host, p_->d_decode_out, (size_t) entries * H * sizeof(float),
                            cudaMemcpyDeviceToHost, p_->stream);
        if (profile && s == cudaSuccess) s = cudaEventRecord(p_->decode_profile_events[3], p_->stream);
        if (s != cudaSuccess) { err = cuda_error("download decode output", s); return false; }
        if (profile) submit_done = Clock::now();
        if (!p_->sync("decode", err)) return false;
        if (profile) {
            wait_done = Clock::now();
            float ms_upload = 0.0f, ms_kernel = 0.0f, ms_download = 0.0f;
            s = cudaEventElapsedTime(&ms_upload, p_->decode_profile_events[0], p_->decode_profile_events[1]);
            if (s == cudaSuccess)
                s = cudaEventElapsedTime(&ms_kernel, p_->decode_profile_events[1], p_->decode_profile_events[2]);
            if (s == cudaSuccess)
                s = cudaEventElapsedTime(&ms_download, p_->decode_profile_events[2], p_->decode_profile_events[3]);
            if (s != cudaSuccess) { err = cuda_error("read decode timing events", s); return false; }
            cpu_prepare = elapsed_ms(call_start, prepare_done);
            cpu_submit = elapsed_ms(prepare_done, submit_done);
            cpu_wait = elapsed_ms(submit_done, wait_done);
            gpu_upload = ms_upload;
            gpu_kernel = ms_kernel;
            gpu_download = ms_download;
        }
        ++p_->decode_calls;
        return true;
    }();
    const bool restored = scope.leave(err);
    if (ok && restored && replayed_graph) ++p_->decode_graph_calls;
    if (ok && restored && profile) {
        ++p_->decode_profile.calls;
        p_->decode_profile.cpu_prepare_ms += cpu_prepare;
        p_->decode_profile.cpu_submit_ms += cpu_submit;
        p_->decode_profile.cpu_wait_ms += cpu_wait;
        p_->decode_profile.cpu_total_ms += elapsed_ms(call_start, Clock::now());
        p_->decode_profile.gpu_upload_ms += gpu_upload;
        p_->decode_profile.gpu_kernel_ms += gpu_kernel;
        p_->decode_profile.gpu_download_ms += gpu_download;
    }
    return ok && restored;
}

bool RemoteExperts::decode_masked(int64_t layer, const float* x_f_host, const int32_t* ids_host,
                                  const uint8_t* compute_mask, int64_t n_tok, int64_t k,
                                  float* out_host, std::string& err) {
    const int layer_slot = p_->slot(layer);
    if (layer_slot < 0 || !x_f_host || !ids_host || !compute_mask || !out_host ||
        n_tok < 1 || n_tok > MAX_DECODE_T || k < 1 || k > MAX_K) {
        err = "remote experts: invalid masked decode request";
        return false;
    }
    const int entries = (int) (n_tok * k);
    int compute_entries = 0;
    std::array<int32_t, NE> cnt{};
    for (int i = 0; i < entries; ++i) {
        if (ids_host[i] < 0 || ids_host[i] >= NE) {
            err = "remote experts: masked decode expert id out of range";
            return false;
        }
        if (compute_mask[i] > 1) {
            err = "remote experts: masked decode mask must contain only zero or one";
            return false;
        }
        if (compute_mask[i]) { ++cnt[(size_t) ids_host[i]]; ++compute_entries; }
    }
    BusyGuard busy(p_->busy);
    if (!busy.acquired()) { err = "remote experts: overlapping request"; return false; }
    if (p_->decode_mode != RemoteDecodeMode::Original) {
        err = "remote experts: masked decode requires Original mode";
        return false;
    }
    const size_t output_bytes = (size_t) entries * H * sizeof(float);
    if (compute_entries == 0) {
        std::memset(out_host, 0, output_bytes);
        ++p_->masked_decode_calls;
        ++p_->masked_all_hit_calls;
        return true;
    }

    std::vector<unsigned long long> ptr;
    std::vector<int32_t> start(1, 0), dst, tok;
    ptr.reserve((size_t) compute_entries);
    dst.reserve((size_t) compute_entries);
    tok.reserve((size_t) compute_entries);
    for (int e = 0; e < NE; ++e) if (cnt[(size_t) e]) {
        ptr.push_back((unsigned long long) ((uint8_t*) p_->layer_data[(size_t) layer_slot] +
                                           (uint64_t) e * p_->layer_bytes[(size_t) layer_slot]));
        for (int i = 0; i < entries; ++i) if (compute_mask[i] && ids_host[i] == e) {
            dst.push_back(i);
            tok.push_back(i / (int) k);
        }
        start.push_back((int32_t) dst.size());
    }
    const int32_t ng = (int32_t) ptr.size();
    DeviceScope scope;
    if (!scope.enter(p_->device, err)) return false;
    const bool ok = [&]() -> bool {
        cudaError_t s = cudaMemcpyAsync(p_->d_decode_x, x_f_host, (size_t) n_tok * H * sizeof(float),
                                        cudaMemcpyHostToDevice, p_->stream);
        if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_group_ptr, ptr.data(), ptr.size() * sizeof(ptr[0]), cudaMemcpyHostToDevice, p_->stream);
        if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_group_start, start.data(), start.size() * sizeof(start[0]), cudaMemcpyHostToDevice, p_->stream);
        if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_n_groups, &ng, sizeof(ng), cudaMemcpyHostToDevice, p_->stream);
        if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_ent_dst, dst.data(), dst.size() * sizeof(dst[0]), cudaMemcpyHostToDevice, p_->stream);
        if (s == cudaSuccess) s = cudaMemcpyAsync(p_->d_ent_tok, tok.data(), tok.size() * sizeof(tok[0]), cudaMemcpyHostToDevice, p_->stream);
        if (s == cudaSuccess) s = cudaMemsetAsync(p_->d_decode_out, 0, output_bytes, p_->stream);
        if (s != cudaSuccess) { err = cuda_error("upload masked decode request", s); return false; }
        strata::kernels::quantize_q8_1_rows(p_->d_decode_x, n_tok, H, p_->d_decode_q8, p_->stream);
        const auto& f = p_->fmt[(size_t) layer_slot];
        const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
        strata::kernels::native_expert_grouped(L, p_->d_group_ptr, p_->d_group_start, p_->d_n_groups,
                                               p_->d_ent_dst, p_->d_ent_tok, ng, compute_entries,
                                               p_->d_decode_q8, p_->d_decode_scratch, p_->d_decode_out, p_->stream);
        s = cudaMemcpyAsync(out_host, p_->d_decode_out, output_bytes, cudaMemcpyDeviceToHost, p_->stream);
        if (s != cudaSuccess) { err = cuda_error("download masked decode output", s); return false; }
        return p_->sync("masked decode", err);
    }();
    const bool restored = scope.leave(err);
    if (ok && restored) {
        ++p_->decode_calls;
        ++p_->masked_decode_calls;
        p_->masked_decode_entries += compute_entries;
    }
    return ok && restored;
}

bool RemoteExperts::readback_blob(int64_t layer, int64_t expert, uint8_t* out_host, size_t bytes,
                                  std::string& err) {
    BusyGuard busy(p_->busy);
    if (!busy.acquired()) { err = "remote experts: overlapping request"; return false; }
    const int layer_slot = p_->slot(layer);
    if (layer_slot < 0 || expert < 0 || expert >= NE || out_host == nullptr ||
        bytes != p_->layer_bytes[(size_t) layer_slot]) {
        err = "remote experts: invalid blob readback request";
        return false;
    }
    DeviceScope scope;
    if (!scope.enter(p_->device, err)) return false;
    const bool ok = [&]() -> bool {
        const uint8_t* src = (const uint8_t*) p_->layer_data[(size_t) layer_slot] + (uint64_t) expert * bytes;
        const cudaError_t s = cudaMemcpyAsync(out_host, src, bytes, cudaMemcpyDeviceToHost, p_->stream);
        if (s != cudaSuccess) { err = cuda_error("blob readback", s); return false; }
        return p_->sync("blob readback", err);
    }();
    const bool restored = scope.leave(err);
    return ok && restored;
}

bool RemoteExperts::set_decode_profiling(bool enabled, std::string& err) {
    BusyGuard busy(p_->busy);
    if (!busy.acquired()) { err = "remote experts: overlapping request"; return false; }
    if (!p_->loaded || p_->layers == 0) { err = "remote experts: profiling requires a loaded service"; return false; }
    bool have_events = false;
    for (cudaEvent_t event : p_->decode_profile_events) have_events = have_events || event != nullptr;
    // A failed enable can leave owned events pending destructor cleanup if device restoration itself failed.
    // A later explicit disable must still enter the cleanup path even though collection never became enabled.
    if (enabled == p_->decode_profile_enabled && (enabled || !have_events)) return true;
    DeviceScope scope;
    if (!scope.enter(p_->device, err)) return false;
    bool ok = true;
    if (enabled) {
        for (cudaEvent_t event : p_->decode_profile_events)
            if (event != nullptr) { err = "remote experts: stale decode timing event"; ok = false; break; }
        for (size_t i = 0; ok && i < p_->decode_profile_events.size(); ++i) {
            const cudaError_t s = cudaEventCreateWithFlags(&p_->decode_profile_events[i], cudaEventDefault);
            if (s != cudaSuccess) { err = cuda_error("create decode timing event", s); ok = false; }
        }
        if (!ok) {
            for (cudaEvent_t& event : p_->decode_profile_events) {
                if (!event) continue;
                const cudaError_t s = cudaEventDestroy(event);
                if (s == cudaSuccess) event = nullptr;
                else err += "; " + cuda_error("clean up decode timing event", s);
            }
        }
    } else {
        const cudaError_t sync = cudaStreamSynchronize(p_->stream);
        if (sync != cudaSuccess) { err = cuda_error("synchronize before timing-event cleanup", sync); ok = false; }
        for (cudaEvent_t& event : p_->decode_profile_events) {
            if (event) {
                const cudaError_t s = cudaEventDestroy(event);
                if (s != cudaSuccess) {
                    const std::string why = cuda_error("destroy decode timing event", s);
                    if (ok) err = why; else err += "; " + why;
                    ok = false;
                }
                if (s == cudaSuccess) event = nullptr;
            }
        }
        p_->decode_profile_enabled = false;
        p_->decode_profile = {};
    }
    const bool restored = scope.leave(err);
    if (enabled && ok && restored) {
        p_->decode_profile = {};
        p_->decode_profile_enabled = true;
    }
    return ok && restored;
}

bool RemoteExperts::reset_decode_profile(std::string& err) {
    BusyGuard busy(p_->busy);
    if (!busy.acquired()) { err = "remote experts: overlapping request"; return false; }
    if (!p_->loaded) { err = "remote experts: profiling reset requires a loaded service"; return false; }
    p_->decode_profile = {};
    return true;
}

bool RemoteExperts::decode_profile_snapshot(RemoteDecodeProfile& out, std::string& err) {
    BusyGuard busy(p_->busy);
    if (!busy.acquired()) { err = "remote experts: overlapping request"; return false; }
    if (!p_->loaded) { err = "remote experts: profiling snapshot requires a loaded service"; return false; }
    out = p_->decode_profile;
    out.enabled = p_->decode_profile_enabled;
    return true;
}

bool RemoteExperts::set_decode_mode(RemoteDecodeMode mode, std::string& err) {
    if (mode != RemoteDecodeMode::Original && mode != RemoteDecodeMode::Packed && mode != RemoteDecodeMode::Graphs) {
        err = "remote experts: invalid decode mode";
        return false;
    }
    BusyGuard busy(p_->busy);
    if (!busy.acquired()) { err = "remote experts: overlapping request"; return false; }
    if (!p_->loaded || p_->layers == 0) { err = "remote experts: decode mode requires a loaded service"; return false; }
    if (mode == p_->decode_mode) return true;
    if (mode == RemoteDecodeMode::Original) {
        p_->decode_mode = mode;  // Retain bounded optional resources for later reuse.
        return true;
    }

    DeviceScope scope;
    if (!scope.enter(p_->device, err)) return false;
    const bool packet_new = p_->d_decode_packet == nullptr;
    bool ok = true;
    if (packet_new) ok = p_->alloc(&p_->d_decode_packet, PACKET_BYTES, "packed decode request", err);

    if (ok && mode == RemoteDecodeMode::Graphs) {
        auto* dp = static_cast<uint8_t*>(p_->d_decode_packet);
        auto* x = reinterpret_cast<float*>(dp + PACKET_X);
        auto* ptr = reinterpret_cast<unsigned long long*>(dp + PACKET_PTR);
        auto* start = reinterpret_cast<int32_t*>(dp + PACKET_START);
        auto* ng = reinterpret_cast<int32_t*>(dp + PACKET_N_GROUPS);
        auto* dst = reinterpret_cast<int32_t*>(dp + PACKET_DST);
        auto* tok = reinterpret_cast<int32_t*>(dp + PACKET_TOK);
        for (int layer = 0; ok && layer < p_->layers; ++layer) {
            const auto& f = p_->fmt[(size_t) layer];
            const auto layout = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
            for (int t = 1; ok && t <= MAX_DECODE_T; ++t) {
                cudaGraphExec_t& exec = p_->decode_graphs[(size_t) layer][(size_t) t - 1];
                if (exec) continue;
                (void) cudaGetLastError();
                cudaError_t s = cudaStreamBeginCapture(p_->stream, cudaStreamCaptureModeThreadLocal);
                if (s != cudaSuccess) { err = cuda_error("begin decode graph capture", s); ok = false; break; }
                strata::kernels::quantize_q8_1_rows(x, t, H, p_->d_decode_q8, p_->stream);
                const int entries = t * (int) MAX_K;
                strata::kernels::native_expert_grouped(layout, ptr, start, ng, dst, tok, entries, entries,
                                                       p_->d_decode_q8, p_->d_decode_scratch,
                                                       p_->d_decode_out, p_->stream);
                const cudaError_t launch = cudaPeekAtLastError();
                cudaGraph_t graph = nullptr;
                const cudaError_t ended = cudaStreamEndCapture(p_->stream, &graph);
                if (launch != cudaSuccess || ended != cudaSuccess || graph == nullptr) {
                    if (graph) (void) cudaGraphDestroy(graph);
                    if (launch != cudaSuccess) err = cuda_error("capture decode graph launch", launch);
                    else if (ended != cudaSuccess) err = cuda_error("end decode graph capture", ended);
                    else err = "remote experts: decode graph capture returned no graph";
                    ok = false;
                    break;
                }
                s = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                const cudaError_t destroyed = cudaGraphDestroy(graph);
                if (s != cudaSuccess || destroyed != cudaSuccess) {
                    err = cuda_error(s != cudaSuccess ? "instantiate decode graph" : "destroy captured decode graph",
                                     s != cudaSuccess ? s : destroyed);
                    if (exec) {
                        const cudaError_t cleaned = cudaGraphExecDestroy(exec);
                        if (cleaned == cudaSuccess) exec = nullptr;
                        else err += "; " + cuda_error("clean up instantiated decode graph", cleaned);
                    }
                    ok = false;
                }
            }
        }
    }

    if (!ok) {
        bool graphs_clean = true;
        for (auto& layer : p_->decode_graphs) for (cudaGraphExec_t& exec : layer) {
            if (!exec) continue;
            const cudaError_t s = cudaGraphExecDestroy(exec);
            if (s == cudaSuccess) exec = nullptr;
            else { err += "; " + cuda_error("clean up decode graph", s); graphs_clean = false; }
        }
        if (packet_new && graphs_clean && p_->d_decode_packet) {
            const cudaError_t s = cudaFree(p_->d_decode_packet);
            if (s == cudaSuccess) { p_->d_decode_packet = nullptr; p_->vram -= PACKET_BYTES; }
            else err += "; " + cuda_error("clean up packed decode request", s);
        }
    }
    const bool restored = scope.leave(err);
    if (ok && restored) p_->decode_mode = mode;
    return ok && restored;
}

RemoteDecodeMode RemoteExperts::decode_mode() const { return p_->decode_mode; }
int64_t RemoteExperts::decode_graph_calls() const { return p_->decode_graph_calls; }

bool RemoteExperts::prefill(int64_t layer, int target_device, void* target_stream,
                            const uint16_t* mixed_f16_device, const int32_t* src_host, const int32_t* off_host,
                            const int32_t* cnt_host, int64_t T, int64_t k, float* ordered_Dm_device,
                            std::string& err) {
    BusyGuard busy(p_->busy);
    if (!busy.acquired()) { err = "remote experts: overlapping request"; return false; }
    const int layer_slot = p_->slot(layer);
    if (layer_slot < 0 || !mixed_f16_device || !src_host || !off_host || !cnt_host || !ordered_Dm_device ||
        T < 1 || T > MAX_T || k < 1 || k > MAX_K || target_stream == nullptr) {
        err = "remote experts: invalid prefill request";
        return false;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || target_device < 0 || target_device >= ndev) {
        err = "remote experts: invalid prefill target device";
        return false;
    }
    const int64_t entries = T * k;
    if (off_host[0] != 0 || off_host[NE] != entries) { err = "remote experts: invalid prefill group extent"; return false; }
    for (int e = 0; e < NE; ++e) {
        if (cnt_host[e] < 0 || off_host[e] < 0 || off_host[e + 1] < off_host[e] ||
            off_host[e + 1] - off_host[e] != cnt_host[e]) {
            err = "remote experts: inconsistent prefill grouping";
            return false;
        }
    }
    for (int64_t i = 0; i < entries; ++i)
        if (src_host[i] < 0 || src_host[i] >= T) { err = "remote experts: prefill source row out of range"; return false; }

    DeviceScope scope;
    if (!scope.enter(p_->device, err)) return false;
    const cudaStream_t target_cs = (cudaStream_t) target_stream;
    const bool ok = [&]() -> bool {
        if (!p_->copy_from_target(target_device, target_cs, p_->d_mixed, mixed_f16_device,
                                  (uint64_t) T * H * sizeof(uint16_t), err)) return false;
        if (cudaSetDevice(p_->device) != cudaSuccess) { err = "remote experts: cannot reselect service device"; return false; }
        cudaError_t s = cudaMemcpyAsync(p_->d_src, src_host, (size_t) entries * sizeof(int32_t),
                                        cudaMemcpyHostToDevice, p_->stream);
        if (s != cudaSuccess) { err = cuda_error("upload prefill grouping", s); return false; }
        strata::prefill::gather_rows16(p_->d_mixed, p_->d_src, p_->d_xs, entries, H, p_->stream);
        const auto& f = p_->fmt[(size_t) layer_slot];
        auto* base = (uint8_t*) p_->layer_data[(size_t) layer_slot];
        const uint64_t blob = p_->layer_bytes[(size_t) layer_slot];
        for (int e = 0; e < NE; ++e) {
            const int64_t ne = cnt_host[e];
            if (ne == 0) continue;
            const uint8_t* b = base + (uint64_t) e * blob;
            strata::kernels::iq_dequant_gu_f16(f.gu_type, b, b + f.up_off, FF, H, p_->d_dq_gu, p_->stream);
            strata::kernels::iq_dequant_f16(f.d_type, b + f.down_off, H * FF, p_->d_dq_down, p_->stream);
            const int64_t o = off_host[e];
            p_->gemm->f16(p_->d_xs + o * H, p_->d_dq_gu, p_->d_gu + o * 1280, ne, 1280, H);
            strata::prefill::swiglu_interleaved(p_->d_gu + o * 1280, p_->d_h + o * FF, ne, p_->stream);
            p_->gemm->f16(p_->d_h + o * FF, p_->d_dq_down, p_->d_dm + o * H, ne, H, FF);
        }
        if (!p_->sync("prefill compute", err)) return false;
        if (!p_->copy_to_target(target_device, target_cs, ordered_Dm_device, p_->d_dm,
                                (uint64_t) entries * H * sizeof(float), err)) return false;
        ++p_->prefill_calls;
        return true;
    }();
    const bool restored = scope.leave(err);
    return ok && restored;
}

int RemoteExperts::device() const { return p_->device; }
int RemoteExperts::layer_count() const { return p_->loaded ? p_->layers : 0; }
std::vector<int32_t> RemoteExperts::layer_ids() const { return p_->loaded ? p_->global_layers : std::vector<int32_t>{}; }
uint64_t RemoteExperts::payload_bytes() const { return p_->loaded ? p_->payload : 0; }
uint64_t RemoteExperts::vram_bytes() const { return p_->vram; }
int64_t RemoteExperts::decode_calls() const { return p_->decode_calls; }
int64_t RemoteExperts::masked_decode_calls() const { return p_->masked_decode_calls; }
int64_t RemoteExperts::masked_decode_entries() const { return p_->masked_decode_entries; }
int64_t RemoteExperts::masked_all_hit_calls() const { return p_->masked_all_hit_calls; }
int64_t RemoteExperts::prefill_calls() const { return p_->prefill_calls; }

}  // namespace strata::core
