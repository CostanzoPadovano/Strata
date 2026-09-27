#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

inline constexpr int32_t kVisionMagic = 0x31455653;
inline constexpr int64_t kVisionImagePad = 248056;
inline constexpr int32_t kVisionWidth = 2560;
inline constexpr int32_t kVisionMaxRecords = 8;
inline constexpr int32_t kVisionMaxRowsPerImage = 1024;
inline constexpr int32_t kVisionMaxTotalRows = 8192;

struct VisionRequest {
    std::vector<float> rows;
    std::vector<const float*> row_ptr;
    std::vector<int32_t> mrope;
};

bool validate_request_ids(const std::vector<int64_t>& ids, int64_t max_new, int64_t max_context,
                          int64_t n_vocab, std::string& err);

bool load_vision_request(const std::string& path, const std::vector<int64_t>& ids, int64_t max_new,
                         int64_t max_context, int64_t n_vocab, VisionRequest& out, std::string& err);

}  // namespace strata::core
