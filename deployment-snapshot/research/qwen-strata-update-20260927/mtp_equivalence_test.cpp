// Bounded real-checkpoint parity gate for MTP embedding and residual transport.
//
// This loads only index metadata from the main pack (every indexed tensor is skipped) plus the native
// embedding/head and one sequential MTP-layer owner. NativeEmbed is constructed outside the MTP scope:
// captured graphs retain its immutable format metadata, so it must outlive every MtpDrafter.

#include "strata/core/mtp.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strata::core::ModelGeometry;
using strata::core::MtpDrafter;
using strata::core::NativeEmbed;
using strata::core::NativeHead;
using strata::core::SessionState;
using strata::core::WeightTable;

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
        cudaError_t status = cudaMalloc(&ptr_, bytes);
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

uint32_t float_bits(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

struct Result {
    int n = 0;
    std::array<int32_t, 3> ids{{-1, -1, -1}};
    std::array<float, 3> probs{{-1.0f, -1.0f, -1.0f}};
};

void compare(const Result& want, const Result& got, const char* mode, const char* fixture, int T, int a) {
    if (got.n != want.n) {
        std::fprintf(stderr, "FAIL mode=%s fixture=%s T=%d a=%d index=-1 field=n_drafts expected=%d got=%d\n",
                     mode, fixture, T, a, want.n, got.n);
        fail("MTP n_drafts mismatch");
    }
    for (int i = 0; i < 3; ++i) {
        if (got.ids[(size_t) i] != want.ids[(size_t) i]) {
            std::fprintf(stderr, "FAIL mode=%s fixture=%s T=%d a=%d index=%d field=id expected=%d got=%d\n",
                         mode, fixture, T, a, i, want.ids[(size_t) i], got.ids[(size_t) i]);
            fail("MTP draft-id mismatch");
        }
        const uint32_t wb = float_bits(want.probs[(size_t) i]);
        const uint32_t gb = float_bits(got.probs[(size_t) i]);
        if (gb != wb) {
            std::fprintf(stderr,
                         "FAIL mode=%s fixture=%s T=%d a=%d index=%d field=prob_bits expected=0x%08x got=0x%08x\n",
                         mode, fixture, T, a, i, wb, gb);
            fail("MTP probability-bit mismatch");
        }
    }
}

Result run_draft(MtpDrafter& d, int T, const int32_t* tokens, int64_t p, int a, float min_p, const char* mode) {
    Result r;
    std::string err;
    if (!d.draft(T, tokens, p, a, r.ids.data(), err, r.probs.data(), min_p, &r.n))
        fail(std::string(mode) + " draft: " + err);
    return r;
}

Result run_first(MtpDrafter& d, const float* row, int32_t token, float min_p, const char* mode) {
    Result r;
    std::string err;
    if (!d.draft_first(4, row, token, 16, r.ids.data(), err, r.probs.data(), min_p, &r.n))
        fail(std::string(mode) + " draft_first: " + err);
    return r;
}

struct Args { std::string pack, native, mtp; };

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if ((k == "--pack" || k == "--native" || k == "--mtp") && i + 1 < argc) {
            std::string& dst = k == "--pack" ? a.pack : k == "--native" ? a.native : a.mtp;
            dst = argv[++i];
        } else {
            fail("usage: mtp_equivalence_test --pack PATH --native GGUF --mtp PATH");
        }
    }
    if (a.pack.empty() || a.native.empty() || a.mtp.empty())
        fail("usage: mtp_equivalence_test --pack PATH --native GGUF --mtp PATH");
    return a;
}

int run(const Args& args) {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) fail("MTP equivalence requires visible CUDA devices 0 and 1");
    for (int device : {0, 1}) {
        cuda_check(cudaSetDevice(device), "initialize CUDA device");
        cuda_check(cudaFree(nullptr), "initialize CUDA context");
    }
    cuda_check(cudaSetDevice(0), "restore target CUDA device");

    const ModelGeometry g;
    constexpr int64_t kContext = 256;
    constexpr int64_t kExperts = 10;
    constexpr int kMaxT = 4;
    constexpr int64_t kVocab = 248320;
    const int64_t hcn = g.hc * g.n_embd;
    if (hcn != 10240) fail("unexpected model geometry: hc*n_embd is not 10240");

    // Skip every indexed tensor: WeightTable retains exact shapes/metadata but reads no pack binary.
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

    // These owners are deliberately outside the MTP scope below.
    NativeEmbed embed;
    if (!embed.load(args.native, g.n_embd, n_vocab, err)) fail(err);
    strata::core::set_native_embed(&embed);
    NativeHead head;
    if (!head.load(args.native, g.n_embd, n_vocab, err)) fail(err);

    DeviceBuffer session_mem;
    session_mem.allocate(0, (size_t) strata::core::session_bytes(g, kContext, kExperts), "shared session");
    SessionState ss;
    SessionCleanup session_cleanup{&ss, &g};
    if (strata::core::session_init(g, kContext, kExperts, session_mem.get(), ss) == 0)
        fail("session_init failed");
    strata::core::session_zero(ss, g, nullptr, nullptr);
    cuda_check(cudaDeviceSynchronize(), "initialize shared session state");

    std::vector<float> prefill((size_t) 17 * (size_t) hcn);
    std::vector<float> window((size_t) kMaxT * (size_t) hcn);
    for (size_t i = 0; i < prefill.size(); ++i)
        prefill[i] = (float) ((int) (i % 257) - 128) / 256.0f;
    for (size_t i = 0; i < window.size(); ++i)
        window[i] = (float) ((int) ((i * 13 + 7) % 509) - 254) / 512.0f;
    std::array<int32_t, 17> prefill_tokens{};
    for (int i = 0; i < 17; ++i) prefill_tokens[(size_t) i] = 1000 + 17 * i;

    DeviceBuffer d_prefill, d_window;
    d_prefill.allocate(0, prefill.size() * sizeof(float), "prefill residuals");
    d_window.allocate(0, window.size() * sizeof(float), "window residuals");
    d_prefill.upload(prefill.data(), prefill.size() * sizeof(float), "upload prefill residuals");
    d_window.upload(window.data(), window.size() * sizeof(float), "upload window residuals");
    cuda_check(cudaDeviceSynchronize(), "finish fixture uploads");

    {
        struct Mode { const char* name; int device; bool resident; bool mapped; bool batch; };
        const std::array<Mode, 9> modes{{
            {"local-relay", 0, false, false, false},
            {"local-resident", 0, true, false, false},
            {"remote-relay", 1, false, false, false},
            {"remote-resident", 1, true, false, false},
            {"remote-mapped-relay", 1, false, true, false},
            {"remote-mapped-resident", 1, true, true, false},
            {"local-resident-batch", 0, true, false, true},
            {"remote-resident-batch", 1, true, false, true},
            {"remote-mapped-resident-batch", 1, true, true, true},
        }};
        const char* policy_name[3] = {"minp0", "minp1", "alternating"};
        std::array<std::vector<Result>, 3> references;
        std::array<Result, 3> boundary_references{};
        std::array<bool, 3> have_boundary_reference{{false, false, false}};

        auto min_for = [](int policy, size_t fixture) -> float {
            if (policy == 0) return 0.0f;
            if (policy == 1) return 1.0f;
            return (fixture & 1u) ? 1.0f : 0.0f;
        };
        auto require_minp1 = [](float min_p, const Result& r, const char* mode) {
            if (min_p == 1.0f && r.n != 1) fail(std::string(mode) + " minP=1 fixture did not return one draft");
        };

        // Each policy receives a fresh owner/KV state. Contiguous p advances make the following fixture observe
        // any unsafe future-KV side effect from the preceding batched round.
        for (size_t mode_i = 0; mode_i < modes.size(); ++mode_i) {
            const Mode& m = modes[mode_i];
            for (int policy = 0; policy < 3; ++policy) {
                d_window.upload(window.data(), window.size() * sizeof(float), "reset policy window");
                cuda_check(cudaDeviceSynchronize(), "finish policy window reset");
                MtpDrafter d;
                cuda_check(cudaSetDevice(0), "select target before MTP load");
                d.set_prompt_len(17);
                d.set_resident_embedding(m.resident);
                d.set_mapped_residuals(m.mapped);
                d.set_chain_batch(m.batch);
                if (!d.load(args.mtp, g, ss, kMaxT, err, 16, m.device))
                    fail(std::string(m.name) + "/" + policy_name[policy] + " load: " + err);
                if (!d.bind(wt, &head, d_window.as<float>(), err))
                    fail(std::string(m.name) + "/" + policy_name[policy] + " bind: " + err);
                const int64_t copies0 = d.transfer_count();
                const uint64_t bytes0 = d.transfer_bytes();
                if (!d.prefill(d_prefill.as<float>(), prefill_tokens.data(), 17, 0, err))
                    fail(std::string(m.name) + "/" + policy_name[policy] + " prefill: " + err);

                size_t fixture_i = 0;
                int64_t expected_unused = 0;
                float min_p = min_for(policy, fixture_i);
                Result got = run_first(d, d_prefill.as<float>() + 16 * hcn, prefill_tokens[16], min_p, m.name);
                require_minp1(min_p, got, m.name);
                expected_unused += 3 - got.n;
                if (mode_i == 0) references[(size_t) policy].push_back(got);
                else compare(references[(size_t) policy][fixture_i], got, m.name, policy_name[policy], 4, 0);
                ++fixture_i;

                d_window.upload(window.data(), window.size() * sizeof(float), "restore catch-up window");
                cuda_check(cudaDeviceSynchronize(), "finish catch-up window restore");
                int64_t p = 17;
                int fixture_number = 1;
                for (int T = 1; T <= 4; ++T) {
                    for (int a = 0; a < T; ++a) {
                        std::array<int32_t, 4> tokens{};
                        for (int t = 0; t < T; ++t) tokens[(size_t) t] = 2000 + fixture_number * 19 + t;
                        min_p = min_for(policy, fixture_i);
                        got = run_draft(d, T, tokens.data(), p, a, min_p, m.name);
                        require_minp1(min_p, got, m.name);
                        expected_unused += 3 - got.n;
                        if (mode_i == 0) references[(size_t) policy].push_back(got);
                        else compare(references[(size_t) policy][fixture_i], got, m.name, policy_name[policy], T, a);
                        ++fixture_i;
                        p += a + 1;
                        ++fixture_number;
                    }
                }
                if (fixture_i != 11 || fixture_number != 11 || p >= kContext)
                    fail("internal policy fixture-count/context invariant failed");
                if (m.batch) {
                    if (d.batched_rounds() != 11 || d.extra_unused_computed_drafts() != expected_unused)
                        fail(std::string(m.name) + " did not batch every safe policy fixture exactly");
                } else if (d.batched_rounds() != 0 || d.extra_unused_computed_drafts() != 0) {
                    fail(std::string(m.name) + " unexpectedly reported chain batching");
                }

                const uint64_t expected_mapped_bytes = 51ull * (uint64_t) hcn * sizeof(float);
                if (m.mapped) {
                    if (!d.mapped_residuals() || d.mapped_residual_count() != 16 ||
                        d.mapped_residual_bytes() != expected_mapped_bytes)
                        fail(std::string(m.name) + " did not exercise the per-policy mapped residual profile");
                    if (m.resident && (d.transfer_count() - copies0 != 16 ||
                                       d.transfer_bytes() - bytes0 != expected_mapped_bytes))
                        fail(std::string(m.name) + " residuals did not use the per-policy D2H-only profile");
                } else if (d.mapped_residuals() || d.mapped_residual_count() != 0 ||
                           d.mapped_residual_bytes() != 0) {
                    fail(std::string(m.name) + " unexpectedly exercised mapped residual transport");
                }
                cuda_check(cudaSetDevice(0), "restore target before policy cleanup");
            }

            struct BoundaryCase { int64_t p; float min_p; int expected_n; const char* name; };
            const std::array<BoundaryCase, 3> boundary_cases{{
                {255, 1.0f, 1, "p255-minp1"},
                {255, 0.0f, 1, "p255-minp0"},
                {254, 0.0f, 2, "p254-minp0"},
            }};
            for (size_t boundary_i = 0; boundary_i < boundary_cases.size(); ++boundary_i) {
                const BoundaryCase& bc = boundary_cases[boundary_i];
                // Every edge fixture owns fresh K/V. The minP=0 cases prove the sequential fallback truncates
                // before an invalid graph: p255 returns only cell 255; p254 may add cell 255 but never cell 256.
                d_window.upload(window.data(), window.size() * sizeof(float), "reset boundary window");
                cuda_check(cudaDeviceSynchronize(), "finish boundary window reset");
                MtpDrafter boundary;
                cuda_check(cudaSetDevice(0), "select target before boundary MTP load");
                boundary.set_prompt_len(17);
                boundary.set_resident_embedding(m.resident);
                boundary.set_mapped_residuals(m.mapped);
                boundary.set_chain_batch(m.batch);
                if (!boundary.load(args.mtp, g, ss, kMaxT, err, 16, m.device))
                    fail(std::string(m.name) + "/" + bc.name + " boundary load: " + err);
                if (!boundary.bind(wt, &head, d_window.as<float>(), err))
                    fail(std::string(m.name) + "/" + bc.name + " boundary bind: " + err);
                const int64_t boundary_copies0 = boundary.transfer_count();
                const uint64_t boundary_bytes0 = boundary.transfer_bytes();
                if (!boundary.prefill(d_prefill.as<float>(), prefill_tokens.data(), 17, 0, err))
                    fail(std::string(m.name) + "/" + bc.name + " boundary prefill: " + err);
                const int32_t boundary_token = 4096;
                Result edge = run_draft(boundary, 1, &boundary_token, bc.p, 0, bc.min_p, m.name);
                if (edge.n != bc.expected_n)
                    fail(std::string(m.name) + "/" + bc.name + " returned the wrong valid prefix length");
                if (mode_i == 0) {
                    boundary_references[boundary_i] = edge;
                    have_boundary_reference[boundary_i] = true;
                } else {
                    compare(boundary_references[boundary_i], edge, m.name, bc.name, 1, 0);
                }
                if (boundary.batched_rounds() != 0 || boundary.extra_unused_computed_drafts() != 0)
                    fail(std::string(m.name) + "/" + bc.name + " did not use sequential fallback");
                const uint64_t boundary_mapped_bytes = 18ull * (uint64_t) hcn * sizeof(float);
                if (m.mapped) {
                    if (boundary.mapped_residual_count() != 6 ||
                        boundary.mapped_residual_bytes() != boundary_mapped_bytes)
                        fail(std::string(m.name) + "/" + bc.name + " mapped profile is not 6 calls/18 rows");
                    if (m.resident && (boundary.transfer_count() - boundary_copies0 != 6 ||
                                       boundary.transfer_bytes() - boundary_bytes0 != boundary_mapped_bytes))
                        fail(std::string(m.name) + "/" + bc.name + " residual profile is not D2H-only");
                } else if (boundary.mapped_residuals() || boundary.mapped_residual_count() != 0 ||
                           boundary.mapped_residual_bytes() != 0) {
                    fail(std::string(m.name) + "/" + bc.name + " unexpectedly used mapped residual transport");
                }
                cuda_check(cudaSetDevice(0), "restore target before boundary cleanup");
            }
        }
        for (const auto& r : references) if (r.size() != 11) fail("reference policy does not have 11 fixtures");
        for (bool have : have_boundary_reference) if (!have) fail("boundary reference was not recorded");
    }  // every MTP owner dies before NativeEmbed/session/head/target buffers

    strata::core::set_native_embed(nullptr);
    cuda_check(cudaSetDevice(0), "restore target after MTP cleanup");
    std::printf("PASS mtp_equivalence fixtures=36 modes=9 probability_bits=exact\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mtp_equivalence_test: %s\n", e.what());
        return 1;
    }
}
