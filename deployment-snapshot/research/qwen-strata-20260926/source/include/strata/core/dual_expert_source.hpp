#pragma once
#include "strata/core/expert_source.hpp"
#include "strata/core/pinned.hpp"
#include "strata/platform/direct_file.hpp"
#include <memory>
#include <map>

namespace strata::core {
// Fixed whole-layer ownership: GPU1 blobs are never duplicated in the host arena.
// Prefill's existing bounded staging ring can request a synchronous bounce copy.
class DualExpertSource final : public ExpertSource {
public:
    ~DualExpertSource() override;
    bool open(const std::string& gguf, uint64_t gpu_budget, uint64_t host_budget, std::string& err,
              const std::string& profile = {}, uint64_t primary_cache_budget = 0);
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool pinned(int64_t layer, int64_t expert) const override;
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;
    bool remote_layer(int64_t layer) const override;
    bool stage_remote(int64_t layer, int64_t expert, void* dst, void* stream, std::string& err) override;
    bool compute_remote(int64_t layer, const float* x, const int32_t* ids, int64_t nt, int64_t k,
                        float* out, std::string& err) override;
    uint64_t host_bytes() const { return host_bytes_; }
    uint64_t gpu_bytes() const { return gpu_bytes_; }
    uint64_t disk_bytes() const { return disk_bytes_; }
private:
    bool read_blob(int64_t layer, int64_t expert, uint8_t* dst, std::string& err);
    std::unique_ptr<PinnedArena> host_;
    std::vector<int64_t> host_off_;
    std::vector<uint64_t> gpu_off_;
    std::vector<bool> remote_;
    std::vector<uint8_t> bounce_;
    std::vector<uint8_t> cold_;
    std::vector<int64_t> cold_keys_;
    std::vector<std::unique_ptr<strata::platform::DirectFile>> files_;
    std::vector<size_t> file_of_;
    void* io_buffer_=nullptr;
    size_t cold_next_=0;
    bool io_failed_=false;
    uint64_t host_bytes_=0, gpu_bytes_=0, disk_bytes_=0;
    void *gpu_=nullptr, *stream_=nullptr, *x_=nullptr, *xq_=nullptr, *scratch_=nullptr, *y_=nullptr;
    void *ptr_=nullptr, *start_=nullptr, *count_=nullptr, *dst_=nullptr, *tok_=nullptr;
    void *hx_=nullptr,*hp_=nullptr,*hb_=nullptr,*hc_=nullptr,*hd_=nullptr,*ht_=nullptr,*hy_=nullptr;
    std::map<uint32_t,void*> graphs_; // at most 8 windows × 7 gate formats × 2 down formats
    int device_=1;
};
}
