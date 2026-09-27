// Bounded resident-expert service for a second CUDA device.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

enum class RemoteDecodeMode : int {
    Original = 0,
    Packed = 1,
    Graphs = 2,
};

struct RemoteDecodeProfile {
    bool enabled = false;
    uint64_t calls = 0;
    double cpu_prepare_ms = 0.0;
    double cpu_submit_ms = 0.0;   ///< CUDA API submission interval, including any implicit host-side waits
    double cpu_wait_ms = 0.0;
    double cpu_total_ms = 0.0;
    double gpu_upload_ms = 0.0;    ///< service-stream interval; not pure wire bandwidth
    double gpu_kernel_ms = 0.0;    ///< service-stream interval; may include stream idle/host launch gaps
    double gpu_download_ms = 0.0;  ///< service-stream interval; not pure wire bandwidth
};

class RemoteExperts {
public:
    RemoteExperts();
    ~RemoteExperts();
    RemoteExperts(const RemoteExperts&) = delete;
    RemoteExperts& operator=(const RemoteExperts&) = delete;

    /// Load the contiguous layer prefix [0, layer_count) from its verified packed/GGUF source onto `device`. The process-wide
    /// ExpertLayout must already describe the complete native pack. Zero layers is the explicit disabled mode;
    /// enabled services accept at most twelve layers in Original mode and never map or retain a host copy of
    /// experts.bin. Packed and Graphs modes retain their eight-layer implementation bound.
    /// When experts.bin is absent, `native_shard` must explicitly name the already-bound GGUF shard; v3 sibling
    /// shard names from the layout are then resolved beside it. No implicit disk/model fallback is attempted.
    bool load(const std::string& pack_dir, int layer_count, int device, std::string& err,
              const std::string& native_shard = {});
    /// Load up to twelve strictly increasing, unique global layer IDs into compact resident slots.
    bool load(const std::string& pack_dir, const std::vector<int32_t>& layer_ids, int device, std::string& err,
              const std::string& native_shard = {});
    bool owns(int64_t layer) const;

    /// Synchronous decode mailbox. Inputs and output are host FP32; all CUDA work occurs on the service device.
    bool decode(int64_t layer, const float* x_f_host, const int32_t* ids_host, int64_t n_tok, int64_t k,
                float* out_host, std::string& err);

    /// Synchronous sparse decode mailbox. compute_mask has n_tok*k bytes: one computes that routed entry,
    /// zero returns an exact-zero output row. Masked decode currently requires Original decode mode.
    bool decode_masked(int64_t layer, const float* x_f_host, const int32_t* ids_host,
                       const uint8_t* compute_mask, int64_t n_tok, int64_t k,
                       float* out_host, std::string& err);

    /// Bounded exact cold-cache resident blob readback. `bytes` must equal this layer's immutable blob size.
    /// It does not affect service counters and never rereads the source file.
    bool readback_blob(int64_t layer, int64_t expert, uint8_t* out_host, size_t bytes, std::string& err);

    /// Diagnostic-only decode instrumentation. Disabled is allocation-free and is the default. Enabling owns
    /// exactly four CUDA events on the service device; reset retains those events and zeros successful-call totals.
    bool set_decode_profiling(bool enabled, std::string& err);
    bool reset_decode_profile(std::string& err);
    bool decode_profile_snapshot(RemoteDecodeProfile& out, std::string& err);

    /// Select the decode transport/launch implementation. Original is the default. Packed combines the
    /// bounded request metadata and activations into one H2D packet. Graphs additionally replays a fixed
    /// kernel-only graph for K=10; smaller K requests deliberately use the Packed path.
    bool set_decode_mode(RemoteDecodeMode mode, std::string& err);
    RemoteDecodeMode decode_mode() const;
    int64_t decode_graph_calls() const;

    /// Synchronous bounded prompt service. `mixed_f16_device` and `ordered_Dm_device` belong to target_device.
    /// src/off/cnt are the caller's expert-id-ordered grouping (off has 513 entries, cnt 512, src T*k).
    bool prefill(int64_t layer, int target_device, void* target_stream, const uint16_t* mixed_f16_device,
                 const int32_t* src_host, const int32_t* off_host, const int32_t* cnt_host, int64_t T, int64_t k,
                 float* ordered_Dm_device, std::string& err);

    int device() const;
    int layer_count() const;
    std::vector<int32_t> layer_ids() const;
    uint64_t payload_bytes() const;
    uint64_t vram_bytes() const;
    int64_t decode_calls() const;
    int64_t masked_decode_calls() const;
    int64_t masked_decode_entries() const;
    int64_t masked_all_hit_calls() const;
    int64_t prefill_calls() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::core
