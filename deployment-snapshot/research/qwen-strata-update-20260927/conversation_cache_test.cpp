// Synthetic running-state/cache-key gate. No weights, tokenizer, server or full model.
#include "strata/core/conversation_cache.hpp"
#include <cstdio>
#include <stdexcept>
#include <cstring>

using namespace strata::core;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void cu(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { cu(cudaMalloc(&p, n)); }
    ~Buffer() { cudaFree(p); }
};

int main() {
    try {
        const int32_t outputs[]{11, 99, 22, 33};
        require(cache_terminal_keep(outputs, 4, 8, {99}) == 2, "hidden accepted rows past EOS retained");
        require(cache_terminal_keep(outputs, 4, 1, {99}) == 1, "hidden accepted rows past budget retained");
        require(cache_terminal_keep(outputs, 4, 8, {11}) == 1, "first-row EOS");
        require(cache_terminal_keep(outputs, 4, 8, {33}) == 4, "last-row EOS");
        require(cache_terminal_keep(outputs, 4, 8, {}) == 4, "nonterminal accepted prefix truncated");
        require(cache_terminal_keep(outputs, 2, 8, {22}) == 2, "read beyond accepted outputs");
        require(cache_terminal_keep(nullptr, 4, 8, {}) == 0 &&
                cache_terminal_keep(outputs, 0, 8, {}) == 0 &&
                cache_terminal_keep(outputs, 4, 0, {}) == 0, "invalid terminal bounds");
        const std::vector<int32_t> pre{1, 2, 3};
        require(cache_prefix_matches(pre, {}, {1, 2, 3, 4}, {}), "valid prefix");
        require(!cache_prefix_matches(pre, {}, {1, 9, 3, 4}, {}), "divergent prefix admitted");
        require(!cache_prefix_matches(pre, {}, {1, 2, 3}, {}), "last prompt token must remain fresh");
        require(!cache_prefix_matches({}, {}, {1, 2}, {}), "empty prefix admitted");
        float a[4]{1, 2, 3, 4}, b[4]{5, 6, 7, 8};
        std::vector<const float*> rows{nullptr, a, b, nullptr};
        std::vector<int32_t> pos{0,0,0, 1,1,1, 1,1,2, 2,2,2};
        std::vector<ImgKey> keys, changed;
        require(cache_image_keys(rows, pos, 4, 4, keys) && keys.size() == 1 && keys[0].start == 1, "image span");
        require(cache_prefix_matches(pre, keys, {1,2,3,4}, keys), "same image rejected");
        b[0] += 1;
        require(cache_image_keys(rows, pos, 4, 4, changed), "changed image parse");
        require(!cache_prefix_matches(pre, keys, {1,2,3,4}, changed), "changed pixels admitted");
        b[0] -= 1;
        pos[8] += 1;
        require(cache_image_keys(rows, pos, 4, 4, changed) && changed != keys, "changed grid admitted");
        require(!cache_image_keys(rows, pos, 5, 4, changed), "image bounds not enforced");

        ModelGeometry g;
        g.n_layers = 4; g.qsa_interval = 2; g.ssm_state_size = 2; g.ssm_v_heads = 2;
        g.ssm_conv_channels = 8; g.ssm_d_conv = 2; g.idx_key_dim = 4; g.n_embd = 4; g.hc = 2;
        const auto z = conv_state_sizes(g);
        int devices = 0; cu(cudaGetDeviceCount(&devices)); require(devices >= 2, "dual GPU gate needs two devices");
        for (int device = 0; device < 2; ++device) {
            cu(cudaSetDevice(device));
            Buffer gd(z.gdn), pl(z.ple), tail(z.tail * 2);
            QsaState qs[2]{};
            for (int i = 0; i < 2; ++i) qs[i].idx_tail = (float*)((uint8_t*) tail.p + z.tail * i);
            SessionState ss;
            ss.gdn_state = (float*) gd.p; ss.ple_hist = (float*) pl.p; ss.qsa_states = qs;
            cu(cudaMemset(gd.p, 0x21 + device, z.gdn)); cu(cudaMemset(pl.p, 0x42, z.ple));
            cu(cudaMemset(tail.p, 0x63, z.tail * 2)); cu(cudaDeviceSynchronize());
            ConvCheckpoint c;
            // Deliberately not aligned to a KV page/indexer block.
            c.ids.resize(257);
            for (size_t i = 0; i < c.ids.size(); ++i) c.ids[i] = (int32_t) i + 7;
            c.r_last.assign((size_t) g.hc_dim(), 0.125f);
            require(checkpoint_save(c, ss, g), "state save");
            cu(cudaMemset(gd.p, 0, z.gdn)); cu(cudaMemset(pl.p, 0, z.ple)); cu(cudaMemset(tail.p, 0, z.tail * 2));
            require(checkpoint_restore(c, ss, g), "state restore");
            ConvCheckpoint readback = c;
            require(checkpoint_save(readback, ss, g), "restored state save");
            require(readback.gdn == c.gdn && readback.ple == c.ple && readback.tails == c.tails,
                    "running state roundtrip mismatch");
            require(ss.ple_prev[0] == 262 && ss.ple_prev[1] == 263, "PLE token boundary");
            c.ple.pop_back(); require(!checkpoint_restore(c, ss, g), "truncated PLE accepted");
            c = readback; c.r_last.clear(); require(!checkpoint_restore(c, ss, g), "missing MTP boundary accepted");
            c = readback; c.ids = {19}; require(checkpoint_restore(c, ss, g), "one-token checkpoint");
            require(ss.ple_prev[0] == -1 && ss.ple_prev[1] == 19, "one-token PLE boundary");
            int current = -1; cu(cudaGetDevice(&current)); require(current == device, "device ownership changed");
            std::printf("cache running-state roundtrip CUDA %d passed, %zu bytes\n", device, z.gdn + z.ple + 2*z.tail);
        }
        std::puts("conversation_cache_test OK (synthetic; not full-model equivalence)");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "cache test: %s\n", e.what()); return 1; }
}
