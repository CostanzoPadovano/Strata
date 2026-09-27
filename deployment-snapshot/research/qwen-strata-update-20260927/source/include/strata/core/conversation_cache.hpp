#pragma once

#include "strata/core/session.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

// A speculative window may verify more tokens than the visible reply emits. Retain
// only the input rows responsible for that reply, so its live prefix never includes
// hidden tokens past EOS or the requested output budget. The verifier can commit a
// prefix of any accepted window; its final emitted output remains the next input.
inline int cache_terminal_keep(const int32_t* outputs, int accepted_inputs, int64_t remaining,
                               const std::vector<int64_t>& eos_ids) {
    if (!outputs || accepted_inputs < 1 || remaining < 1) return 0;
    int keep = (int) std::min<int64_t>(accepted_inputs, remaining);
    for (int i = 0; i < keep; ++i) {
        if (std::find(eos_ids.begin(), eos_ids.end(), (int64_t) outputs[i]) != eos_ids.end()) {
            keep = i + 1;
            break;
        }
    }
    return keep;
}

struct ImgKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return start == o.start && hash == o.hash; }
};

struct ConvCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ImgKey> imgs;
    std::vector<uint8_t> gdn, ple, tails;
    // Cell L-1 in the MTP cache depends on token L, outside this checkpoint's prefix.
    // Preserve the target residual so a new continuation can repair that single cell.
    std::vector<float> r_last;
};

inline uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// Only consume rows returned by the bounded SVE parser. Adjacent images may share a pad span;
// hashing that whole span is conservative for partial-image checkpoints, never less restrictive.
inline bool cache_image_keys(const std::vector<const float*>& rows, const std::vector<int32_t>& positions,
                             int64_t n, int64_t width, std::vector<ImgKey>& keys) {
    keys.clear();
    if (n < 0 || width < 1 || (uint64_t) n > rows.size() || (uint64_t) n > positions.size() / 3) return false;
    for (int64_t i = 0; i < n;) {
        if (!rows[(size_t) i]) { ++i; continue; }
        const int64_t start = i;
        uint64_t h = 1469598103934665603ull;
        while (i < n && rows[(size_t) i]) {
            h = fnv1a(rows[(size_t) i], (size_t) width * sizeof(float), h);
            h = fnv1a(positions.data() + (size_t) i * 3, 3 * sizeof(int32_t), h);
            ++i;
        }
        keys.push_back({start, h});
    }
    return true;
}

inline std::vector<ImgKey> images_below(const std::vector<ImgKey>& all, int64_t L) {
    std::vector<ImgKey> result;
    for (const ImgKey& k : all) if (k.start < L) result.push_back(k);
    return result;
}

inline bool cache_prefix_matches(const std::vector<int32_t>& prefix, const std::vector<ImgKey>& images,
                                 const std::vector<int64_t>& prompt, const std::vector<ImgKey>& request_images) {
    if (prefix.empty() || prefix.size() >= prompt.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i) if (prefix[i] != prompt[i]) return false;
    return images_below(request_images, (int64_t) prefix.size()) == images;
}

struct ConvStateSizes { size_t gdn = 0, ple = 0, tail = 0; };

inline ConvStateSizes conv_state_sizes(const ModelGeometry& g) {
    ConvStateSizes z;
    z.gdn = (size_t) g.n_gdn_layers() *
            ((size_t) g.ssm_state_size * (size_t) g.ssm_v_heads * (size_t) g.ssm_state_size +
             (size_t) g.ssm_conv_channels * (size_t) (g.ssm_d_conv - 1)) * sizeof(float);
    z.ple = (size_t) kernels::NG_HIST * (size_t) kernels::NG_HC_DIM * sizeof(float);
    z.tail = (size_t) (kernels::qsa_real_shapes().idx_block - 1) * (size_t) g.idx_key_dim * sizeof(float);
    return z;
}

// Positional KV is deliberately not copied. The caller must retain only compatible-prefix checkpoints,
// synchronize the target device and keep its ownership selected during these running-state copies.
inline bool checkpoint_save(ConvCheckpoint& c, const SessionState& ss, const ModelGeometry& g) {
    const ConvStateSizes z = conv_state_sizes(g);
    if (c.ids.empty() || c.r_last.size() != (size_t) g.hc_dim()) return false;
    c.gdn.resize(z.gdn);
    c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(z.tail * (size_t) g.n_qsa_layers());
    if (cudaMemcpy(c.gdn.data(), ss.gdn_state, z.gdn, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (!c.ple.empty() && cudaMemcpy(c.ple.data(), ss.ple_hist, z.ple, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i)
        if (cudaMemcpy(c.tails.data() + (size_t) i * z.tail, ss.qsa_states[i].idx_tail,
                       z.tail, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

inline bool checkpoint_restore(const ConvCheckpoint& c, SessionState& ss, const ModelGeometry& g) {
    const ConvStateSizes z = conv_state_sizes(g);
    if (c.ids.empty() || c.r_last.size() != (size_t) g.hc_dim() || c.gdn.size() != z.gdn ||
        c.ple.size() != (ss.ple_hist ? z.ple : 0) || c.tails.size() != z.tail * (size_t) g.n_qsa_layers()) return false;
    if (cudaMemcpy(ss.gdn_state, c.gdn.data(), z.gdn, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (!c.ple.empty() && cudaMemcpy(ss.ple_hist, c.ple.data(), z.ple, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i)
        if (cudaMemcpy(ss.qsa_states[i].idx_tail, c.tails.data() + (size_t) i * z.tail,
                       z.tail, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    const size_t L = c.ids.size();
    ss.ple_prev[0] = L >= 2 ? c.ids[L - 2] : -1;
    ss.ple_prev[1] = c.ids[L - 1];
    return cudaDeviceSynchronize() == cudaSuccess;
}

} // namespace strata::core
