// src/core/mtp.cpp - see include/strata/core/mtp.hpp.
#include "strata/core/mtp.hpp"

#include "strata/core/native_head.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        if (cudaGetDevice(&previous_) != cudaSuccess) previous_ = -1;
        if (device >= 0 && device != previous_) changed_ = cudaSetDevice(device) == cudaSuccess;
    }
    ~DeviceGuard() { if (changed_ && previous_ >= 0) (void) cudaSetDevice(previous_); }
    bool active(int device) const { int now = -1; return cudaGetDevice(&now) == cudaSuccess && now == device; }
private:
    int previous_ = -1;
    bool changed_ = false;
};

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    out.resize((size_t) n);
    return (bool) f.read((char*) out.data(), n);
}

std::string allocation_failure(const char* what, uint64_t requested, cudaError_t status) {
    size_t free_b = 0, total_b = 0;
    const cudaError_t info_status = cudaMemGetInfo(&free_b, &total_b);
    int device = -1;
    const cudaError_t device_status = cudaGetDevice(&device);
    std::ostringstream os;
    os << "mtp: " << what << " cudaMalloc(" << requested << " bytes) failed: " << cudaGetErrorString(status);
    if (device_status == cudaSuccess) os << " on CUDA device " << device;
    if (info_status == cudaSuccess)
        os << "; " << free_b << " bytes free of " << total_b << " total";
    else
        os << "; cudaMemGetInfo failed: " << cudaGetErrorString(info_status);
    return os.str();
}

}  // namespace

bool mtp_copy_device_staged(void* dst, int dst_device, const void* src, int src_device, void* pinned, size_t bytes,
                            cudaStream_t dst_stream, std::string& err) {
    if (bytes == 0) return true;
    if (dst == nullptr || src == nullptr || pinned == nullptr || dst_device < 0 || src_device < 0) {
        err = "mtp staged copy: invalid arguments";
        return false;
    }
    int original = -1, count = 0;
    cudaError_t status = cudaGetDevice(&original);
    if (status == cudaSuccess) status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || src_device >= count || dst_device >= count) {
        err = status == cudaSuccess ? "mtp staged copy: device ordinal out of range"
                                    : std::string("mtp staged copy device query: ") + cudaGetErrorString(status);
        return false;
    }
    status = cudaSetDevice(src_device);
    if (status == cudaSuccess) status = cudaMemcpy(pinned, src, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        const std::string why = std::string("mtp staged copy readback: ") + cudaGetErrorString(status);
        const cudaError_t restore = cudaSetDevice(original);
        err = restore == cudaSuccess ? why : why + "; device restore: " + cudaGetErrorString(restore);
        return false;
    }
    status = cudaSetDevice(dst_device);
    if (status == cudaSuccess) status = cudaMemcpyAsync(dst, pinned, bytes, cudaMemcpyHostToDevice, dst_stream);
    const cudaError_t restore = cudaSetDevice(original);
    if (status != cudaSuccess || restore != cudaSuccess) {
        err = status != cudaSuccess ? std::string("mtp staged copy upload: ") + cudaGetErrorString(status)
                                    : std::string("mtp staged copy device restore: ") + cudaGetErrorString(restore);
        return false;
    }
    return true;
}

MtpDrafter::~MtpDrafter() {
    DeviceGuard guard(device_);
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& e : prefill_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : round_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_) if (e) cudaGraphExecDestroy(e);
    if (cs_) cudaStreamDestroy(cs_);
    if (dense_) cudaFree(dense_);
    if (experts_) cudaFree(experts_);
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (dhead_) cudaFree(dhead_);
    if (dvocab_) cudaFree(dvocab_);
    if (resident_embedding_table_) cudaFree(resident_embedding_table_);
    if (target_tok_ || target_embedding_) {
        DeviceGuard target(target_device_);
        if (target_tok_) cudaFree(target_tok_);
        if (target_embedding_) cudaFree(target_embedding_);
    }
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_, h_residual_, h_embedding_,
                     st_.host_step, st_.host_pos};
    for (void* h : hosts) if (h) cudaFreeHost(h);
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "bf16") return (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}

bool MtpDrafter::load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
                      int64_t window, int device) {
    if (cudaGetDevice(&target_device_) != cudaSuccess) { err = "mtp: cannot query the target CUDA device"; return false; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) { err = "mtp: no CUDA device"; return false; }
    device_ = device < 0 ? target_device_ : device;
    if (device_ < 0 || device_ >= ndev) { err = "mtp: --mtp-device is outside the visible CUDA device range"; return false; }
    remote_ = device_ != target_device_;
    DeviceGuard guard(device_);
    if (!guard.active(device_)) { err = "mtp: cannot select the draft CUDA device"; return false; }
    g_ = &g;
    ss_ = &ss;
    max_t_ = max_t;
    rt_dir_ = rt_dir;
    if (max_t < 1 || max_t > strata::kernels::kVerifyMaxT) { err = "mtp: max_t out of range"; return false; }
    if (chain_batch_enabled_ && !resident_embedding_enabled_) {
        err = "mtp: chain batching requires resident embeddings";
        return false;
    }
    // ---- the index and the dense weights
    {
        std::ifstream idx(rt_dir + "/dense.txt");
        if (!idx) { err = "mtp: cannot open " + rt_dir + "/dense.txt (run tools/mtp_rt.py)"; return false; }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) { err = "mtp: malformed dense.txt line: " + line; return false; }
            tensors_.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(rt_dir + "/dense.bin", blob)) { err = "mtp: cannot read dense.bin"; return false; }
        const cudaError_t alloc_status = cudaMalloc((void**) &dense_, blob.size());
        if (alloc_status != cudaSuccess) {
            err = allocation_failure("dense weights", (uint64_t) blob.size(), alloc_status);
            return false;
        }
        cudaMemcpy(dense_, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        vram_ += blob.size();
    }
    // ---- the 512 routed experts, one blob each
    {
        const uint64_t bytes = (uint64_t) g.n_expert * strata::kernels::cpu::BLOB;
        std::ifstream f(rt_dir + "/experts.bin", std::ios::binary);
        if (!f) { err = "mtp: cannot open experts.bin"; return false; }
        const cudaError_t alloc_status = cudaMalloc((void**) &experts_, bytes);
        if (alloc_status != cudaSuccess) {
            err = allocation_failure("512 experts", bytes, alloc_status);
            return false;
        }
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (!f.read((char*) chunk.data(), (std::streamsize) n)) { err = "mtp: experts.bin is truncated"; return false; }
            cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice);
            off += n;
        }
        vram_ += bytes;
    }
    const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }

    // ---- the layer's own K/V (dense attention: no indexer state is read)
    const strata::kernels::QsaShapes s = shapes_of(g);
    const int64_t max_cells = ss.qsa_states[0].max_cells;
    // A remote drafter cannot borrow the target QSA state's device RoPE tables.
    const uint64_t sb = qsa_state_bytes(g, max_cells, remote_);
    if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
    if (qsa_state_init(g, max_cells, state_arena_, st_, remote_ ? nullptr : &ss.qsa_states[0]) == 0) {
        err = "mtp: state init failed";
        return false;
    }
    qsa_state_zero(st_, g, nullptr);
    cudaDeviceSynchronize();
    vram_ += sb;

    // ---- buffers
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_);
    if (!ok) { err = "mtp: mapped staging failed"; return false; }
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(R2 * 4); pos_ = b.take<int32_t>(R2 * NH); row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N); R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N); en_ = b.take<float>(T * N); e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N); h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
        xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        hit_scratch_ = b.take<uint8_t>(strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
        sh_scratch_ = (float*) b.take<uint8_t>(strata::kernels::shared_expert_scratch_bytes(g.n_ff));
        x_bf16_ = b.take<uint16_t>(N);
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        dummy_inj_ = b.take<float>(HC);
        if (remote_) remote_gr_mem_ = b.take<uint8_t>(strata::kernels::gr_workspace_bytes({g.n_embd, g.hc, g.hc_lr}));
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) { err = "mtp: buffers do not fit"; return false; }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    if (remote_) {
        const strata::kernels::GrShapes gs{g.n_embd, g.hc, g.hc_lr};
        strata::kernels::gr_workspace_init(gs, remote_gr_mem_, remote_gr_);
    }
    vram_ += count.used;
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        cudaMemcpy(ident_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) { err = "mtp: stream"; return false; }
    if (remote_) {
        const size_t rb = (size_t) T * HC * N * sizeof(float);
        const unsigned residual_flags = cudaHostAllocPortable |
                                        (mapped_residuals_enabled_ ? cudaHostAllocMapped : 0u);
        cudaError_t status = cudaHostAlloc((void**) &h_residual_, rb, residual_flags);
        if (status != cudaSuccess) {
            err = std::string("mtp: bounded remote residual staging allocation failed: ") + cudaGetErrorString(status);
            return false;
        }
        if (mapped_residuals_enabled_) {
            status = cudaHostGetDevicePointer((void**) &m_residual_, h_residual_, 0);
            if (status != cudaSuccess || m_residual_ == nullptr) {
                err = status == cudaSuccess ? "mtp: mapped residual alias is null"
                                            : std::string("mtp: mapped residual alias failed: ") +
                                                  cudaGetErrorString(status);
                return false;
            }
            std::fprintf(stderr, "strata mtp: remote residual transport uses a %.1f KiB mapped graph input\n",
                         (double) rb / 1024.0);
        }
        if (!resident_embedding_enabled_) {
            const size_t eb = (size_t) T * N * sizeof(float);
            if (cudaHostAlloc((void**) &h_embedding_, eb, cudaHostAllocPortable) != cudaSuccess) {
                err = "mtp: bounded remote embedding staging allocation failed";
                return false;
            }
            DeviceGuard target(target_device_);
            if (!target.active(target_device_) ||
                cudaMalloc((void**) &target_tok_, (size_t) T * sizeof(int32_t)) != cudaSuccess ||
                cudaMalloc((void**) &target_embedding_, eb) != cudaSuccess) {
                err = "mtp: target-side embedding staging allocation failed";
                return false;
            }
        }
    }
    std::fprintf(stderr, "strata mtp: draft layer loaded on CUDA device %d%s, %.0f MiB of VRAM (experts %.0f, dense %.0f)\n",
                 device_, remote_ ? " (remote from target)" : "",
                 (double) vram_ / 1048576.0, (double) g.n_expert * strata::kernels::cpu::BLOB / 1048576.0,
                 (double) tensors_.back().off / 1048576.0);
    return true;
}

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err) {
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head (--native)"; return false; }
    DeviceGuard draft_device(device_);
    if (!draft_device.active(device_)) { err = "mtp: cannot select the draft device while binding"; return false; }

    if (resident_embedding_enabled_) {
        const NativeEmbed* ne = native_embed();
        const uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
        if (ne == nullptr || n_vocab_ <= 0 || g_->n_embd <= 0 || ne->host_data() == nullptr ||
            ne->bytes() == 0 || ne->bytes() > (512ull << 20) || ne->row_bytes() == 0 || ne->n_embd() != g_->n_embd ||
            ne->n_vocab() != n_vocab_ || (uint64_t) ne->row_bytes() > max_u64 / (uint64_t) n_vocab_ ||
            ne->bytes() != (uint64_t) ne->row_bytes() * (uint64_t) n_vocab_) {
            err = "mtp: --mtp-resident-embedding needs an exact native embedding table matching model geometry";
            return false;
        }
        if (resident_embedding_table_ == nullptr) {
            cudaError_t status = cudaMalloc(&resident_embedding_table_, (size_t) ne->bytes());
            if (status != cudaSuccess) {
                err = allocation_failure("resident native embedding", ne->bytes(), status);
                return false;
            }
            status = cudaMemcpy(resident_embedding_table_, ne->host_data(), (size_t) ne->bytes(), cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                err = std::string("mtp: resident native embedding upload failed: ") + cudaGetErrorString(status);
                return false;
            }
            embed_meta_ = ne;
            vram_ += ne->bytes();
            std::fprintf(stderr, "strata mtp: resident native embedding on CUDA device %d (%.1f MiB)\n", device_,
                         (double) ne->bytes() / 1048576.0);
        } else if (embed_meta_ != ne) {
            err = "mtp: native embedding owner changed after the resident copy was bound";
            return false;
        }
    }

    // The draft head's token subset. A remote drafter requires it: copying the full main head would defeat the
    // bounded second-device design, and dereferencing `head->weights()` from the other device is invalid.
    if (dhead_ == nullptr) {
        std::vector<uint8_t> raw;
        const bool have_subset = read_file(rt_dir_ + "/draft_vocab.bin", raw) && raw.size() >= 4 && raw.size() % 4 == 0;
        if (remote_ && !have_subset) { err = "mtp: remote mode requires draft_vocab.bin"; return false; }
        if (have_subset) {
            n_dvocab_ = (int64_t) (raw.size() / 4);
            const int64_t row_bytes = (int64_t) head->row_bytes();
            if (n_dvocab_ <= 0 || n_dvocab_ > n_vocab_ || row_bytes <= 0 ||
                (uint64_t) n_dvocab_ > head->weight_bytes() / (uint64_t) row_bytes) {
                err = "mtp: draft_vocab.bin or native head dimensions are invalid";
                return false;
            }
            for (int64_t i = 0; i < n_dvocab_; ++i) {
                int32_t id = 0;
                std::memcpy(&id, raw.data() + (size_t) i * sizeof(id), sizeof(id));
                if (id < 0 || (int64_t) id >= n_vocab_) {
                    err = "mtp: draft_vocab.bin token " + std::to_string((long long) id) + " at row " +
                          std::to_string((long long) i) + " is outside the vocabulary";
                    return false;
                }
            }
            const size_t head_bytes = (size_t) n_dvocab_ * (size_t) row_bytes;
            if (cudaMalloc((void**) &dvocab_, raw.size()) != cudaSuccess ||
                cudaMalloc((void**) &dhead_, head_bytes) != cudaSuccess) {
                err = "mtp: the draft head does not fit";
                return false;
            }
            if (cudaMemcpy(dvocab_, raw.data(), raw.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "mtp: draft vocabulary upload failed";
                return false;
            }
            if (!remote_) {
                strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_, nullptr);
                if (cudaDeviceSynchronize() != cudaSuccess) { err = "mtp: draft-head gather failed"; return false; }
            } else {
                // Gather fixed-size chunks on the target device, then relay them through one portable pinned buffer.
                int32_t* target_ids = nullptr;
                uint8_t* target_head = nullptr;
                const size_t rows_cap = std::max<size_t>(1, (64u << 20) / (size_t) row_bytes);
                const size_t chunk_cap = std::min<size_t>(head_bytes, rows_cap * (size_t) row_bytes);
                {
                    DeviceGuard target(target_device_);
                    if (!target.active(target_device_) || cudaMalloc((void**) &target_ids, rows_cap * sizeof(int32_t)) != cudaSuccess ||
                        cudaMalloc((void**) &target_head, chunk_cap) != cudaSuccess) {
                        if (target_ids) cudaFree(target_ids);
                        if (target_head) cudaFree(target_head);
                        err = "mtp: target-side draft-head staging does not fit";
                        return false;
                    }
                }
                void* stage = nullptr;
                if (cudaHostAlloc(&stage, chunk_cap, cudaHostAllocPortable) != cudaSuccess) {
                    DeviceGuard target(target_device_); cudaFree(target_ids); cudaFree(target_head);
                    err = "mtp: draft-head transfer staging failed";
                    return false;
                }
                for (size_t row0 = 0; row0 < (size_t) n_dvocab_; row0 += rows_cap) {
                    const size_t nr = std::min(rows_cap, (size_t) n_dvocab_ - row0);
                    const size_t n = nr * (size_t) row_bytes;
                    {
                        DeviceGuard target(target_device_);
                        if (cudaMemcpy(target_ids, raw.data() + row0 * sizeof(int32_t), nr * sizeof(int32_t),
                                       cudaMemcpyHostToDevice) != cudaSuccess) {
                            cudaFreeHost(stage); cudaFree(target_ids); cudaFree(target_head);
                            err = "mtp: target-side draft vocabulary upload failed";
                            return false;
                        }
                        strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, target_ids, (int64_t) nr,
                                                     target_head, nullptr);
                        if (cudaMemcpy(stage, target_head, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
                            cudaFreeHost(stage); cudaFree(target_ids); cudaFree(target_head);
                            err = "mtp: target-side draft-head gather failed";
                            return false;
                        }
                    }
                    {
                        DeviceGuard remote(device_);
                        if (cudaMemcpy(dhead_ + row0 * (size_t) row_bytes, stage, n, cudaMemcpyHostToDevice) != cudaSuccess) {
                            cudaFreeHost(stage);
                            { DeviceGuard target(target_device_); cudaFree(target_ids); cudaFree(target_head); }
                            err = "mtp: remote draft-head upload failed";
                            return false;
                        }
                    }
                    transfer_bytes_ += 2 * n;
                    transfer_count_ += 2;
                }
                cudaFreeHost(stage);
                { DeviceGuard target(target_device_); cudaFree(target_ids); cudaFree(target_head); }
            }
            vram_ += (uint64_t) head_bytes + raw.size();
            std::fprintf(stderr, "strata mtp: draft head over %lld tokens (%.1f MiB)\n", (long long) n_dvocab_,
                         (double) head_bytes / 1048576.0);
        }
    }
    const int64_t head_rows = dhead_ ? n_dvocab_ : n_vocab_;
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) head_rows * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    return true;
}

bool MtpDrafter::stage_residuals(const float* src, int rows, std::string& err) {
    if (!remote_) return true;
    if (src == nullptr || rows < 1 || rows > max_t_) { err = "mtp: invalid remote residual transfer"; return false; }
    const size_t bytes = (size_t) rows * (size_t) g_->hc * (size_t) g_->n_embd * sizeof(float);
    if (!mapped_residuals_enabled_) {
        if (!mtp_copy_device_staged(Rin_, device_, src, target_device_, h_residual_, bytes, cs_, err)) return false;
        transfer_bytes_ += 2 * bytes;
        transfer_count_ += 2;
        return true;
    }

    // The target producer is complete before this callback (prefill and verify both synchronize their source
    // stream). Copy only GPU0 -> host. GPU1 consumes the fixed mapped alias inside its captured graph; there is
    // intentionally no H2D API copy in this mode.
    int original = -1;
    cudaError_t status = cudaGetDevice(&original);
    if (status == cudaSuccess) status = cudaSetDevice(target_device_);
    if (status == cudaSuccess) status = cudaMemcpy(h_residual_, src, bytes, cudaMemcpyDeviceToHost);
    const cudaError_t restore = original >= 0 ? cudaSetDevice(original) : cudaErrorInvalidDevice;
    if (status != cudaSuccess || restore != cudaSuccess) {
        if (status != cudaSuccess) {
            err = std::string("mtp: mapped residual readback failed: ") + cudaGetErrorString(status);
            if (restore != cudaSuccess)
                err += std::string("; device restore failed: ") + cudaGetErrorString(restore);
        } else {
            err = std::string("mtp: mapped residual device restore failed: ") + cudaGetErrorString(restore);
        }
        return false;
    }
    transfer_bytes_ += bytes;       // one explicit D2H API transfer
    transfer_count_ += 1;
    mapped_residual_bytes_ += bytes;  // one graph kernel read through the mapped GPU1 alias
    mapped_residual_count_ += 1;
    return true;
}

bool MtpDrafter::stage_embeddings(const int32_t* tokens, int rows, std::string& err) {
    if (resident_embedding_enabled_) {
        if (tokens == nullptr || rows < 1 || rows > max_t_) { err = "mtp: invalid resident embedding request"; return false; }
        return true;
    }
    if (!remote_) return true;
    if (tokens == nullptr || rows < 1 || rows > max_t_ || native_embed() == nullptr) {
        err = "mtp: remote mode requires bounded native-embedding staging";
        return false;
    }
    const size_t bytes = (size_t) rows * (size_t) g_->n_embd * sizeof(float);
    {
        DeviceGuard target(target_device_);
        if (!target.active(target_device_) ||
            cudaMemcpy(target_tok_, tokens, (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "mtp: target embedding token upload failed";
            return false;
        }
        native_embed()->gather_dev(target_tok_, rows, target_embedding_, nullptr);
        if (cudaMemcpy(h_embedding_, target_embedding_, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "mtp: target embedding readback failed";
            return false;
        }
    }
    {
        DeviceGuard remote(device_);
        if (!remote.active(device_) || cudaMemcpyAsync(emb_, h_embedding_, bytes, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "mtp: remote embedding upload failed";
            return false;
        }
    }
    transfer_bytes_ += 2 * bytes;
    transfer_count_ += 2;
    return true;
}

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    const int row0 = full ? step_row0 : -1 - step_row0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    try {
        // ---- the two input branches
        if (resident_embedding_enabled_) {
            if (embed_meta_ == nullptr || resident_embedding_table_ == nullptr || native_embed() != embed_meta_) {
                err = "mtp: resident embedding metadata is absent or no longer bound";
                return false;
            }
            embed_meta_->gather_from(resident_embedding_table_, tok_, T, emb_, cs);
        } else if (!remote_) {
            const WeightRef* we = wt_->find("token_embd.weight");
            if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
            if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table
                ne->gather_dev(tok_, T, emb_, cs);
            } else {
                const auto* codes = (const uint8_t*) we->data;
                const auto* scales = (const float*) (codes + we->codes_bytes);
                const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
                embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                     (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
            }
        }
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        native_mmvq(GGML_Q8_0, q8("fc_embedding.weight"), xq_, e2_, (int) N, (int) N, T, cs);
        native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
        for (int c0 = 0; c0 < T * HC; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, T * HC - c0);
            native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
            native_mmvq(GGML_Q8_0, q8("fc_hidden.weight"), xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
        }
        add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
        // ---- the attention hyper-connection
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- attention: K/V into the layer's own cache, then (full) dense attention over every cell
        auto norm_rope = [&](float* data, const float* gamma, int rows, int cols, const int32_t* p) {
            native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
            if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, (float) qsa_freq_base(), p, cs);
            else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st_.cos_tab, st_.sin_tab, p, cs);
        };
        native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.k_proj.weight"), xq_, kcur_, (int) N, (int) (NKV * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.v_proj.weight"), xq_, vcur_, (int) N, (int) (NKV * HD), T, cs);
        for (int t = 0; t < T; ++t) {
            norm_rope(kcur_ + t * NKV * HD, f32("self_attn.k_norm.weight"), (int) NKV, (int) HD, pos + t * NH);
            if (st_.kv_int8)
                kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                  kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs);
            else
                kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                               vcur_ + t * NKV * HD, s, cs);
        }
        if (!full) return true;
        native_mmvq(GGML_Q8_0, q8("self_attn.q_proj.weight"), xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        for (int t = 0; t < T; ++t) {
            float* qc = qcur_ + t * NH * HD;
            if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4, (size_t) HD * 4,
                                  (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qc, f32("self_attn.q_norm.weight"), (int) NH, (int) HD, pos + t * NH);
        }
        QsaAttnPools pools;
        pools.page_table = st_.page_table;
        if (st_.kv_int8) { pools.k_q = st_.k_q; pools.v_q = st_.v_q; pools.k_scale = st_.k_scale; pools.v_scale = st_.v_scale; }
        else { pools.k_pool = st_.k_pool; pools.v_pool = st_.v_pool; }
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
        for (int t = 0; t < T; ++t)
            native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD, (int) NH, (int) HD, cs);
        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.o_proj.weight"), xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        // ---- the MLP hyper-connection (the attention write folded in)
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- MoE: router, the 512 resident experts, the shared expert, the combine, the write
        for (int t = 0; t < T; ++t) {
            bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N, (int) g.n_expert, cs);
            if (native_router_enabled()) native_router_top10(logits_ + t * g.n_expert, ids_ + t * K, w_ + t * K, cs);
            else router_top10(logits_ + t * g.n_expert, 1, (int) g.n_expert, (int) K, ids_ + t * K, w_ + t * K, cs);
        }
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) strata::kernels::cpu::BLOB, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                       hit_xs_, hit_scratch_, parts_, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = GGML_Q8_0; nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
        nsw.up_type = GGML_Q8_0; nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
        nsw.down_type = GGML_Q8_0; nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
        nsw.q8_1 = xq_;
        const SForm none{};
        for (int t = 0; t < T; ++t) {
            f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, cs);
            shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
                          nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_, shared_ + t * N,
                          N, g.n_ff, 32, cs, mixed_ + t * N, &nsw);
            if (native_moe_combine_enabled())
                native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            else
                moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
        // ---- the final mixer and the main model's head
        for (int t = 0; t < T; ++t)
            gr_read(R_ + (size_t) t * HC * N, f32("hyper_connection_mixer.hc_norm.weight"),
                    bf16("hyper_connection_mixer.input_mix_weight_down.weight"),
                    bf16("hyper_connection_mixer.input_mix_weight_up.weight"), nullptr, EPS, gs,
                    remote_ ? remote_gr_ : ss.block.gr,
                    sample_ + t * N, dummy_inj_, cs);
        native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dhead_ != nullptr;
        const int64_t nv = sub ? n_dvocab_ : n_vocab_;
        native_mmvq(head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) nv, nullptr, 0, sp, out_ids_, cs);
        row_top_prob(head_logits_, T, (int) nv, out_ids_, probs_, cs);
        if (sub) map_ids(out_ids_, dvocab_, T, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp)
    cudaGraphUpload(exec, cs);
    cudaStreamSynchronize(cs);
    return true;
}
}  // namespace

bool MtpDrafter::capture_prefill(int T, std::string& err) {
    DeviceGuard guard(device_);
    if (!guard.active(device_)) { err = "mtp: cannot select the draft device for prefill capture"; return false; }
    if (prefill_exec_[T]) return true;
    using namespace strata::kernels;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * g_->n_head, cs_);
    if (remote_ && mapped_residuals_enabled_)
        copy_from_mapped(Rin_, m_residual_, (int64_t) T * g_->hc * g_->n_embd, cs_);
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T)
    return finish_capture(cs_, ok, prefill_exec_[T], "prefill", err);
}

bool MtpDrafter::capture_round(int T, std::string& err) {
    DeviceGuard guard(device_);
    if (!guard.active(device_)) { err = "mtp: cannot select the draft device for round capture"; return false; }
    if (round_exec_[T]) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    bool ok = true;
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) 2 * T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) 2 * T * g_->n_head, cs_);
    copy_i32_from_mapped(row_, m_row_, 2, cs_);
    if (!remote_) copy_from_mapped(Rin_, window_R_, (int64_t) T * HCN, cs_);
    else if (mapped_residuals_enabled_) copy_from_mapped(Rin_, m_residual_, (int64_t) T * HCN, cs_);
    // the catch-up: K/V for the window's T cells, then the full layer for row a only (its cell's K/V is written
    // again, identically), staged by the host in step row 2*max_t - 1; the draft chain is one graph per step
    // (`capture_step`) so the host can stop it when a draft is unlikely
    const int ra = 2 * max_t_ - 1;
    ok = record_forward(T, -1, cs_, err);
    if (ok) {
        // Staged remote rows already contain T gathered embeddings, so select row a in-place. A resident table
        // instead mirrors the local path: select only tok_, then the full-row graph gathers that selected token.
        if (remote_ && !resident_embedding_enabled_)
            mtp_select(emb_, g_->n_embd, tok_, row_, emb_, tok_, nullptr, 0, cs_);
        mtp_select(Rin_, HCN, tok_, row_, Rin_, tok_, nullptr, 0, cs_);
    }
    if (ok) {
        copy_i32_from_mapped(step_ + ra * 4, m_step_ + ra * 4, 4, cs_);
        copy_i32_from_mapped(pos_ + ra * g_->n_head, m_pos_ + ra * g_->n_head, g_->n_head, cs_);
        ok = record_forward(1, ra, cs_, err);
    }
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_);
    return finish_capture(cs_, ok, round_exec_[T], "round", err);
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in Rin_[0] / tok_[0] by mtp_select); draft j and its probability to the mapped outputs.
bool MtpDrafter::capture_step(int j, std::string& err) {
    DeviceGuard guard(device_);
    if (!guard.active(device_)) { err = "mtp: cannot select the draft device for step capture"; return false; }
    if (step_exec_[j]) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int row = max_t_ + j - 1;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head, cs_);
    bool ok = record_forward(1, row, cs_, err);
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_);
    return finish_capture(cs_, ok, step_exec_[j], "step", err);
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    DeviceGuard guard(device_);
    if (!guard.active(device_)) { err = "mtp: cannot select the draft device for prefill"; return false; }
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // cells the window can never reach again need no K/V
    const int64_t first_needed = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    for (int64_t c = 0; c < n; c += max_t_) {
        const int T = (int) std::min<int64_t>(max_t_, n - c);
        if (cell0 + c + T <= first_needed) continue;
        if (!capture_prefill(T, err)) return false;
        if (!stage_residuals(R_rows + (size_t) c * HCN, T, err) ||
            !stage_embeddings(next_tokens + c, T, err)) return false;
        for (int t = 0; t < T; ++t) {
            const int64_t cell = cell0 + c + t;
            h_tok_[t] = next_tokens[c + t];
            h_step_[t * 4 + 0] = (int32_t) cell;
            h_step_[t * 4 + 1] = (int32_t) (cell + 1);
            h_step_[t * 4 + 2] = (int32_t) ((cell + 1) / 4);
            h_step_[t * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < g_->n_head; ++h) h_pos_[t * g_->n_head + h] = (int32_t) cell;
        }
        if ((!remote_ && cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float),
                                         cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) ||
            cudaGraphLaunch(prefill_exec_[T], cs_) != cudaSuccess ||
            cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    ms_prefill += ms_since(t0);
    return true;
}

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, float min_p, int* n_drafts) {
    DeviceGuard guard(device_);
    if (!guard.active(device_)) { err = "mtp: cannot select the draft device"; return false; }
    if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
    if (p < 0 || p > st_.max_cells || T > st_.max_cells - p) {
        err = "mtp: draft catch-up cells exceed the allocated context";
        return false;
    }
    if (!capture_round(T, err)) return false;
    if ((remote_ && window_R_ == nullptr) || !stage_residuals(window_R_, T, err) || !stage_embeddings(tokens, T, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, p + a);
    h_row_[0] = a;
    h_row_[1] = 0;
    // A batched chain needs no host-produced embeddings: every step consumes the previous graph's device token
    // and residual. Capture first, then publish every mapped step record before any launch. Future speculative
    // K/V cells are safe: attention masks cells beyond each step's contiguous width, and a later catch-up append
    // overwrites them before they become live. There is no ring alias because every written cell is in max_cells.
    // Express both bounds as differences so a hostile/invalid p cannot overflow while deciding whether to batch.
    const bool catchup_fits = p >= 0 && p <= st_.max_cells && T <= st_.max_cells - p;
    const int64_t chain_offset = (int64_t) a + (max_t_ - 2);
    const bool chain_fits = p >= 0 && p < st_.max_cells && chain_offset <= st_.max_cells - 1 - p;
    const bool batch = chain_batch_enabled_ && catchup_fits && chain_fits;
    if (batch) {
        for (int j = 1; j < max_t_ - 1; ++j)
            if (!capture_step(j, err)) return false;
        for (int j = 1; j < max_t_ - 1; ++j) put(max_t_ + j - 1, p + a + j);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (batch) {
        if (cudaGraphLaunch(round_exec_[T], cs_) != cudaSuccess) {
            err = std::string("mtp batched draft round: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        for (int j = 1; j < max_t_ - 1; ++j)
            if (cudaGraphLaunch(step_exec_[j], cs_) != cudaSuccess) {
                err = std::string("mtp batched draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp batched draft sync: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pj = ((volatile float*) h_prob_)[0];
        if (probs) probs[0] = pj;
        int n = 1;
        // Match the sequential prefix contract exactly: pj gates COMPUTING the following draft there, so the
        // already-computed draft j is returned iff draft j-1 met min_p.
        for (int j = 1; j < max_t_ - 1 && pj >= min_p; ++j) {
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
            if (probs) probs[j] = pj;
            ++n;
        }
        for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
        if (n_drafts) *n_drafts = n;
        ++batched_rounds_;
        extra_unused_computed_drafts_ += (max_t_ - 1) - n;
    } else {
        if (cudaGraphLaunch(round_exec_[T], cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pj = ((volatile float*) h_prob_)[0];
        if (probs) probs[0] = pj;
        int n = 1;
        // the chain continues while the last draft is likely enough to be verified
        for (int j = 1; j < max_t_ - 1 && pj >= min_p; ++j) {
            // The batch path already proves its furthest cell fits. Its sequential fallback must make the same
            // guarantee one step at a time: near the context edge, return the valid prefix without publishing
            // metadata or launching a graph for an out-of-range future cell. Use differences to avoid overflow.
            const int64_t step_offset = (int64_t) a + j;
            if (p >= st_.max_cells || step_offset > st_.max_cells - 1 - p) break;
            if (!capture_step(j, err)) return false;
            put(max_t_ + j - 1, p + a + j);
            if (!stage_embeddings(drafts + j - 1, 1, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (cudaGraphLaunch(step_exec_[j], cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) {
                err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
            if (probs) probs[j] = pj;
            ++n;
        }
        for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
        if (n_drafts) *n_drafts = n;
    }
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, float min_p, int* n_drafts) {
    DeviceGuard target(target_device_);
    if (!target.active(target_device_)) { err = "mtp: cannot select the target device for the first draft"; return false; }
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row, (size_t) HCN * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
}

}  // namespace strata::core
