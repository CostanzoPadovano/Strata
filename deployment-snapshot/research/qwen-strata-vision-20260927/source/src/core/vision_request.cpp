#include "strata/core/vision_request.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace strata::core {

bool validate_request_ids(const std::vector<int64_t>& ids, int64_t max_new, int64_t max_context,
                          int64_t n_vocab, std::string& err) {
    if (ids.empty()) {
        err = "token list was empty";
        return false;
    }
    if (max_new < 1 || max_context < 9 || (int64_t) ids.size() > max_context - 8 ||
        max_new > max_context - 8 - (int64_t) ids.size()) {
        err = "prompt plus max_new exceeds the context";
        return false;
    }
    for (int64_t id : ids) {
        if (id < 0 || id >= n_vocab) {
            err = "a token id is outside the vocabulary";
            return false;
        }
    }
    return true;
}

bool load_vision_request(const std::string& path, const std::vector<int64_t>& ids, int64_t max_new,
                         int64_t max_context, int64_t n_vocab, VisionRequest& out, std::string& err) {
    out = {};
    if (!validate_request_ids(ids, max_new, max_context, n_vocab, err)) return false;

    struct Image {
        int32_t n;
        int32_t nx;
        int32_t ny;
        size_t off;
    };
    std::vector<Image> images;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    while (err.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, 1, sizeof hdr, f);
        if (got == 0) break;
        if (got != sizeof hdr || hdr[0] != kVisionMagic || hdr[1] < 1 || hdr[1] > kVisionMaxRowsPerImage ||
            hdr[2] < 1 || hdr[3] < 1 || (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != kVisionWidth) {
            err = "bad embeddings header";
            break;
        }
        if ((int) images.size() >= kVisionMaxRecords ||
            (int64_t) out.rows.size() / kVisionWidth + hdr[1] > kVisionMaxTotalRows) {
            err = "embeddings file exceeds the image or row limit";
            break;
        }
        const size_t off = out.rows.size();
        const size_t count = (size_t) hdr[1] * kVisionWidth;
        out.rows.resize(off + count);
        if (std::fread(out.rows.data() + off, sizeof(float), count, f) != count) {
            err = "short embeddings payload";
            break;
        }
        for (size_t i = off; i < off + count; ++i) {
            if (!std::isfinite(out.rows[i])) {
                err = "embeddings payload contains a non-finite value";
                break;
            }
        }
        if (err.empty()) images.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    std::fclose(f);
    if (!err.empty()) {
        out = {};
        return false;
    }

    const int64_t cells = max_context + 64;
    out.row_ptr.assign(ids.size(), nullptr);
    out.mrope.resize((size_t) cells * 3);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        out.mrope[(size_t) c * 3] = (int32_t) t;
        out.mrope[(size_t) c * 3 + 1] = (int32_t) h;
        out.mrope[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0;
    size_t i = 0;
    size_t k = 0;
    while (i < ids.size()) {
        if (ids[i] != kVisionImagePad) {
            put((int64_t) i, p, p, p);
            ++p;
            ++i;
            continue;
        }
        if (k >= images.size()) {
            err = "the prompt has more images than the embeddings file";
            break;
        }
        const Image& im = images[k++];
        for (int64_t j = 0; j < im.n; ++j) {
            if (i + (size_t) j >= ids.size() || ids[i + (size_t) j] != kVisionImagePad) {
                err = "an image has fewer image-pad tokens than embedding rows";
                break;
            }
        }
        for (int64_t j = 0; err.empty() && j < im.n; ++j) {
            const int64_t y = j / im.nx;
            const int64_t x = j % im.nx;
            put((int64_t) i + j, p, p + y, p + x);
            out.row_ptr[i + (size_t) j] = out.rows.data() + im.off + (size_t) j * kVisionWidth;
        }
        i += (size_t) im.n;
        p += std::max(im.nx, im.ny);
    }
    if (err.empty() && k != images.size()) err = "the embeddings file has more images than the prompt";
    if (err.empty() && ids.back() == kVisionImagePad) err = "the prompt cannot end in an image";
    for (int64_t c = (int64_t) ids.size(); err.empty() && c < cells; ++c) {
        put(c, p + c - (int64_t) ids.size(), p + c - (int64_t) ids.size(), p + c - (int64_t) ids.size());
    }
    if (!err.empty()) {
        out = {};
        return false;
    }
    return true;
}

}  // namespace strata::core
