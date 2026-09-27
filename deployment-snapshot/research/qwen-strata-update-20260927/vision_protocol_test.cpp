#include "strata/core/vision_request.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

using strata::core::VisionRequest;
constexpr int32_t W = strata::core::kVisionWidth;

struct TempFile {
    std::filesystem::path path;
    TempFile(const char* name) : path(std::filesystem::temp_directory_path() / name) {}
    ~TempFile() { std::error_code ec; std::filesystem::remove(path, ec); }
};

bool append_record(const std::filesystem::path& path, int32_t n, int32_t nx, int32_t ny, int32_t width,
                   float base, bool append = false, size_t floats = SIZE_MAX) {
    std::FILE* f = std::fopen(path.string().c_str(), append ? "ab" : "wb");
    if (!f) return false;
    const int32_t hdr[5] = {strata::core::kVisionMagic, n, nx, ny, width};
    bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1;
    const size_t count = floats == SIZE_MAX ? (size_t) std::max(0, n) * (size_t) std::max(0, width) : floats;
    std::vector<float> row(count);
    for (size_t i = 0; i < count; ++i) row[i] = base + (float) (i / (size_t) std::max(1, width));
    ok = ok && std::fwrite(row.data(), sizeof(float), row.size(), f) == row.size();
    std::fclose(f);
    return ok;
}

bool expect_fail(const std::filesystem::path& path, const std::vector<int64_t>& ids,
                 int64_t max_new, int64_t max_context, int64_t vocab, const char* label) {
    VisionRequest req;
    std::string err;
    if (strata::core::load_vision_request(path.string(), ids, max_new, max_context, vocab, req, err)) {
        std::fprintf(stderr, "%s unexpectedly passed\n", label);
        return false;
    }
    if (!req.rows.empty() || !req.row_ptr.empty() || !req.mrope.empty()) {
        std::fprintf(stderr, "%s retained output after failure\n", label);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    using namespace strata::core;
    const int64_t P = kVisionImagePad;
    bool ok = true;

    TempFile valid("strata-vision-protocol-valid.sve");
    ok = ok && append_record(valid.path, 4, 2, 2, W, 10.0f);
    const std::vector<int64_t> ids = {1, P, P, P, P, 2};
    VisionRequest req;
    std::string err;
    ok = ok && load_vision_request(valid.path.string(), ids, 8, 64, 300000, req, err);
    ok = ok && req.rows.size() == (size_t) 4 * W && req.row_ptr.size() == ids.size();
    ok = ok && req.row_ptr[0] == nullptr && req.row_ptr[5] == nullptr;
    for (int i = 0; i < 4; ++i) ok = ok && req.row_ptr[(size_t) i + 1] && req.row_ptr[(size_t) i + 1][0] == 10.0f + i;
    const int32_t expected[][3] = {{0,0,0},{1,1,1},{1,1,2},{1,2,1},{1,2,2},{3,3,3},{4,4,4}};
    for (size_t i = 0; i < 7; ++i) for (int d = 0; d < 3; ++d) ok = ok && req.mrope[i * 3 + d] == expected[i][d];

    ok = ok && expect_fail(valid.path, {}, 1, 64, 300000, "empty prompt");
    ok = ok && expect_fail(valid.path, ids, 60, 64, 300000, "oversized context");
    ok = ok && expect_fail(valid.path, {300000, 1}, 1, 64, 300000, "vocabulary bound");
    ok = ok && expect_fail(valid.path, {1, 2}, 1, 64, 300000, "extra image");
    ok = ok && expect_fail(valid.path, {1, P, P, 2}, 1, 64, 300000, "short image run");

    TempFile bad_header("strata-vision-protocol-header.sve");
    ok = ok && append_record(bad_header.path, 1, 1, 1, 4096, 0.0f);
    ok = ok && expect_fail(bad_header.path, {1, P, 2}, 1, 64, 300000, "wrong width");
    ok = ok && append_record(bad_header.path, 1025, 1025, 1, W, 0.0f);
    ok = ok && expect_fail(bad_header.path, {1, P, 2}, 1, 64, 300000, "oversized image");

    TempFile records("strata-vision-protocol-records.sve");
    for (int i = 0; i < 9; ++i) ok = ok && append_record(records.path, 1, 1, 1, W, (float) i, i != 0);
    ok = ok && expect_fail(records.path, {1, P, 2}, 1, 64, 300000, "record cap");

    TempFile truncated("strata-vision-protocol-truncated.sve");
    ok = ok && append_record(truncated.path, 4, 2, 2, W, 0.0f, false, 3);
    ok = ok && expect_fail(truncated.path, ids, 1, 64, 300000, "truncated payload");

    TempFile nan_file("strata-vision-protocol-nan.sve");
    ok = ok && append_record(nan_file.path, 1, 1, 1, W, 0.0f);
    {
        std::FILE* f = std::fopen(nan_file.path.string().c_str(), "r+b");
        float nan = std::numeric_limits<float>::quiet_NaN();
        std::fseek(f, 20, SEEK_SET);
        ok = ok && std::fwrite(&nan, sizeof nan, 1, f) == 1;
        std::fclose(f);
    }
    ok = ok && expect_fail(nan_file.path, {1, P, 2}, 1, 64, 300000, "non-finite payload");

    TempFile trailing("strata-vision-protocol-trailing.sve");
    ok = ok && append_record(trailing.path, 1, 1, 1, W, 0.0f);
    {
        std::FILE* f = std::fopen(trailing.path.string().c_str(), "ab");
        const unsigned char byte = 7;
        ok = ok && std::fwrite(&byte, 1, 1, f) == 1;
        std::fclose(f);
    }
    ok = ok && expect_fail(trailing.path, {1, P, 2}, 1, 64, 300000, "trailing payload");

    if (!ok) {
        std::fprintf(stderr, "vision protocol synthetic gate failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("vision protocol synthetic gate passed: bounded headers, payloads, positions and image-row substitution\n");
    return 0;
}
