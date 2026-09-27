// Tiny immutable-per-run fixture: source offsets stay logical, destination offsets exclude the GPU prefix.
#include "strata/core/pinned.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using strata::core::ExpertHostLayout;
using strata::kernels::cpu::ExpertLayout;
void cu(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
void write_fixture(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    require(!std::filesystem::exists(path), "fixture already exists");
    std::ofstream file(path, std::ios::binary);
    require((bool) file, "cannot create fixture");
    file.write((const char*) bytes.data(), (std::streamsize) bytes.size());
    require((bool) file, "fixture write failed");
}
ExpertLayout geometry(uint64_t scale = 1) {
    ExpertLayout lay;
    lay.native = true; lay.n_layers = 5; lay.n_expert = 2;
    lay.bytes = {8, 16, 24, 16, 8}; lay.offset = {0, 16, 48, 96, 128};
    lay.total = 144 * scale; lay.max_blob = 24 * scale;
    lay.fmt.resize(5);
    for (size_t i = 0; i < 5; ++i) {
        lay.bytes[i] *= scale; lay.offset[i] *= scale;
        lay.fmt[i].up_off = (size_t) lay.bytes[i] / 4;
        lay.fmt[i].down_off = (size_t) lay.bytes[i] / 2;
        lay.fmt[i].bytes = (size_t) lay.bytes[i];
    }
    return lay;
}
ExpertLayout geometry_six(uint64_t scale = 1) {
    ExpertLayout lay;
    lay.native = true; lay.n_layers = 12; lay.n_expert = 2;
    lay.max_blob = 0;
    lay.bytes.resize(12); lay.offset.resize(12); lay.fmt.resize(12);
    uint64_t off = 0;
    for (int l = 0; l < 12; ++l) {
        lay.bytes[(size_t) l] = (uint64_t) (8 + 8 * (l % 3)) * scale;
        lay.offset[(size_t) l] = off;
        off += 2 * lay.bytes[(size_t) l];
        lay.max_blob = std::max(lay.max_blob, lay.bytes[(size_t) l]);
        lay.fmt[(size_t) l].up_off = (size_t) lay.bytes[(size_t) l] / 4;
        lay.fmt[(size_t) l].down_off = (size_t) lay.bytes[(size_t) l] / 2;
        lay.fmt[(size_t) l].bytes = (size_t) lay.bytes[(size_t) l];
    }
    lay.total = off;
    return lay;
}
ExpertLayout geometry_twelve(uint64_t scale = 1) {
    ExpertLayout lay;
    lay.native = true; lay.n_layers = 24; lay.n_expert = 2;
    lay.max_blob = 0;
    lay.bytes.resize(24); lay.offset.resize(24); lay.fmt.resize(24);
    uint64_t off = 0;
    for (int l = 0; l < 24; ++l) {
        lay.bytes[(size_t) l] = (uint64_t) (8 + 8 * (l % 3)) * scale;
        lay.offset[(size_t) l] = off;
        off += 2 * lay.bytes[(size_t) l];
        lay.max_blob = std::max(lay.max_blob, lay.bytes[(size_t) l]);
        lay.fmt[(size_t) l].up_off = (size_t) lay.bytes[(size_t) l] / 4;
        lay.fmt[(size_t) l].down_off = (size_t) lay.bytes[(size_t) l] / 2;
        lay.fmt[(size_t) l].bytes = (size_t) lay.bytes[(size_t) l];
    }
    lay.total = off;
    return lay;
}
void selected_maps_and_scatter(const std::filesystem::path& base) {
    auto lay = geometry();
    ExpertHostLayout host;
    std::string err;
    require(host.build(lay, {1, 3}, err), "selected map rejected");
    require(host.host_bytes == 80 && host.excluded_bytes == 64 &&
            host.offset == std::vector<uint64_t>({0, UINT64_MAX, 16, UINT64_MAX, 64}) &&
            host.slice == std::vector<int32_t>({0, -1, 1, -1, 2}) &&
            host.bounds == std::vector<uint64_t>({0, 16, 64, 80}) &&
            host.source_off == std::vector<uint64_t>({0, 48, 128}) &&
            host.destination_off == std::vector<uint64_t>({0, 16, 64}) &&
            host.layer_bytes == std::vector<uint64_t>({16, 48, 16}), "global/compact map differs");
    uint64_t off = 0;
    for (int l = 0; l < 5; ++l) for (int e = 0; e < 2; ++e) {
        if (l == 1 || l == 3) require(!host.blob_offset(l, e, off), "remote host blob admitted");
        else require(host.blob_offset(l, e, off) && off == host.offset[l] + e * lay.bytes[l],
                     "first/last compact expert offset differs");
    }
    require(!host.blob_offset(-1, 0, off) && !host.blob_offset(5, 0, off) &&
            !host.blob_offset(0, -1, off) && !host.blob_offset(0, 2, off), "invalid axes admitted");
    for (auto ids : {std::vector<int32_t>{}, {0, 1, 2, 3}, {0, 4}}) {
        ExpertHostLayout other;
        require(other.build(lay, ids, err) && other.host_bytes + other.excluded_bytes == lay.total,
                "identity/prefix/edge compaction failed");
    }
    for (auto ids : {std::vector<int32_t>{1, 1}, {3, 1}, {-1}, {5}, {0, 1, 2, 3, 4}})
        require(!host.build(lay, ids, err), "invalid excluded IDs admitted");
    require(host.host_bytes == 80, "failed build mutated accepted map");
    auto malformed = lay;
    malformed.offset[2]++;
    require(!host.build(malformed, {1, 3}, err), "logical packing hole accepted");
    malformed = lay; malformed.total++;
    require(!host.build(malformed, {}, err), "wrong logical total accepted");
    malformed = lay; malformed.bytes[0] = malformed.max_blob = UINT64_MAX;
    require(!host.build(malformed, {}, err), "expert product overflow accepted");
    auto malformed_host = host;
    malformed_host.blob_bytes.clear();
    require(!malformed_host.blob_offset(0, 0, off), "short public shape accepted");
    malformed_host = host; malformed_host.offset[0] = UINT64_MAX - 1;
    require(!malformed_host.blob_offset(0, 0, off), "public offset overflow accepted");

    std::vector<uint8_t> packed(144);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = (uint8_t) ((i * 43 + 29) % 251);
    const auto packed_path = std::filesystem::path(base.string() + ".packed");
    write_fixture(packed_path, packed);
    std::vector<uint8_t> dst(96, 0xfe), expected(80);
    for (size_t i = 0; i < host.source_off.size(); ++i)
        std::copy_n(packed.data() + host.source_off[i], (size_t) host.layer_bytes[i],
                    expected.data() + host.destination_off[i]);
    const auto stats = strata::core::load_experts_scatter(packed_path.string(), dst.data() + 8, 80,
        host.source_off, host.destination_off, host.layer_bytes, 2, 7);
    require(stats.seconds >= 0 && stats.bytes == 80 && stats.layers == 3 &&
            std::equal(expected.begin(), expected.end(), dst.begin() + 8), "selected packed scatter differs");
    require(std::all_of(dst.begin(), dst.begin() + 8, [](uint8_t b){return b == 0xfe;}) &&
            std::all_of(dst.end() - 8, dst.end(), [](uint8_t b){return b == 0xfe;}), "selected guards overwritten");
    const auto untouched = dst;
    auto reject = [&](std::vector<uint64_t> src, std::vector<uint64_t> dest,
                      std::vector<uint64_t> bytes, uint64_t capacity = 80, int threads = 2, uint64_t chunk = 7) {
        require(strata::core::load_experts_scatter(packed_path.string(), dst.data() + 8, capacity,
                    src, dest, bytes, threads, chunk).seconds < 0 && dst == untouched,
                "invalid selected scatter accepted or wrote destination");
    };
    reject({0}, {}, {1}); reject({0, 1}, {0, 1}, {2, 1}); reject({144}, {0}, {1});
    reject({0}, {79}, {2}); reject({0}, {UINT64_MAX}, {1}); reject({0}, {0}, {0});
    reject({0}, {0}, {1}, 80, 0); reject({0}, {0}, {1}, 80, 65);
    reject({0}, {0}, {1}, 80, 2, 0); reject({0}, {0}, {1}, 80, 2, (32u << 20) + 1);

    // Two tiny GGUF-like shards contain role-major expert tensors. Scatter must restore expert-major blobs.
    std::vector<uint8_t> shards[2];
    lay.gguf_off.resize(15); lay.gguf_file.resize(5);
    const auto shard0 = std::filesystem::path(base.string() + ".roles0");
    const auto shard1 = std::filesystem::path(base.string() + ".roles1");
    for (int l = 0; l < 5; ++l) {
        auto& shard = shards[l == 4 ? 1 : 0];
        if (l == 4) lay.gguf_file[l] = shard1.filename().string();
        const uint64_t role_bytes[3] = {lay.fmt[l].up_off, lay.fmt[l].up_off, lay.bytes[l] - lay.fmt[l].down_off};
        const uint64_t role_at[3] = {0, lay.fmt[l].up_off, lay.fmt[l].down_off};
        for (int r = 0; r < 3; ++r) {
            lay.gguf_off[3*l+r] = shard.size();
            for (int e = 0; e < 2; ++e) {
                const size_t start = (size_t) (lay.offset[l] + e * lay.bytes[l] + role_at[r]);
                shard.insert(shard.end(), packed.begin() + start, packed.begin() + start + role_bytes[r]);
            }
        }
    }
    write_fixture(shard0, shards[0]); write_fixture(shard1, shards[1]);
    const auto native_stats = strata::core::load_experts_gguf_compact(shard0.string(), dst.data() + 8, lay, host, 2);
    require(native_stats.seconds >= 0 && native_stats.bytes == 80 && native_stats.layers == 3 &&
            dst == untouched, "global shard/role compact scatter differs");
    malformed = lay; malformed.fmt[2].down_off++;
    require(strata::core::load_experts_gguf_compact(shard0.string(), dst.data() + 8, malformed, host, 2).seconds < 0 &&
            dst == untouched, "invalid role geometry wrote destination");
    malformed = lay; malformed.bytes.clear();
    require(strata::core::load_experts_gguf_compact(shard0.string(), dst.data() + 8, malformed, host, 2).seconds < 0,
            "short native geometry accepted");
    malformed_host = host; malformed_host.destination_off[1] = 0;
    require(strata::core::load_experts_gguf_compact(shard0.string(), dst.data() + 8, lay, malformed_host, 2).seconds < 0 &&
            dst == untouched, "forged host shape accepted");
}
void selected_device_aliases() {
    auto lay = geometry(1024);
    ExpertHostLayout host;
    std::string err;
    require(host.build(lay, {1, 3}, err), "scaled alias map rejected");
    int count = 0; cu(cudaGetDeviceCount(&count)); require(count >= 2, "two CUDA devices required");
    cu(cudaSetDevice(0)); cu(cudaFree(nullptr));
    strata::core::PinnedArena arena(host.host_bytes + lay.max_blob, host.bounds, 64u << 10);
    require(arena.valid() && arena.registered_bytes == (64u << 10) && arena.registered_slices == 2 &&
            arena.registered_bytes + arena.locked_bytes == arena.capacity, "scaled alias residency incomplete");
    for (uint64_t i = 0; i < arena.capacity; ++i) arena.data()[i] = (uint8_t) ((i * 23 + 31) % 251);
    for (int device : {0, 1}) {
        cu(cudaSetDevice(device)); cu(cudaFree(nullptr));
        std::vector<const uint8_t*> aliases;
        for (uint64_t start : arena.slice_starts) {
            void* alias = nullptr;
            cu(cudaHostGetDevicePointer(&alias, arena.data() + start, 0));
            aliases.push_back((const uint8_t*) alias);
        }
        for (int l : {1, 3, 4}) for (int e = 0; e < 2; ++e)
            require(host.device_alias(l, e, arena.registered_bytes, aliases, true) == nullptr,
                    "remote/unregistered layer obtained device alias");
        uint8_t* output = nullptr; cu(cudaMalloc((void**) &output, 32));
        for (int l : {0, 2}) for (int e = 0; e < 2; ++e) {
            const auto* alias = host.device_alias(l, e, arena.registered_bytes, aliases, true);
            uint64_t off = 0;
            require(alias && host.blob_offset(l, e, off), "retained compact alias missing");
            cu(cudaMemcpy(output, alias, 32, cudaMemcpyDeviceToDevice));
            uint8_t back[32]; cu(cudaMemcpy(back, output, 32, cudaMemcpyDeviceToHost));
            require(std::equal(back, back + 32, arena.data() + off), "mapped compact device bytes differ");
        }
        cu(cudaFree(output));
        auto short_aliases = aliases; short_aliases.pop_back();
        require(!host.device_alias(2, 0, arena.registered_bytes, short_aliases, true), "short alias list accepted");
        require(!host.device_alias(0, 0, 1, aliases, false), "partially registered expert admitted");
    }
    cu(cudaSetDevice(0));
}
void selected_six_hole_layout(const std::filesystem::path& base) {
    const std::vector<int32_t> excluded{1,3,5,7,9,11};
    auto lay = geometry_six();
    ExpertHostLayout host;
    std::string err;
    require(host.build(lay, excluded, err), "six-hole selected map rejected");
    uint64_t retained_bytes = 0, excluded_bytes = 0;
    std::vector<uint64_t> want_src, want_dst, want_bytes, want_bounds{0};
    for (int l = 0; l < 12; ++l) {
        const uint64_t layer_bytes = 2 * lay.bytes[(size_t) l];
        if (l & 1) {
            excluded_bytes += layer_bytes;
            require(host.offset[(size_t) l] == UINT64_MAX && host.slice[(size_t) l] == -1,
                    "six-hole excluded layer retained");
        } else {
            require(host.offset[(size_t) l] == retained_bytes && host.slice[(size_t) l] >= 0,
                    "six-hole retained compact offset differs");
            want_src.push_back(lay.offset[(size_t) l]);
            want_dst.push_back(retained_bytes);
            want_bytes.push_back(layer_bytes);
            retained_bytes += layer_bytes;
            want_bounds.push_back(retained_bytes);
        }
    }
    require(host.host_bytes == retained_bytes && host.excluded_bytes == excluded_bytes &&
            retained_bytes + excluded_bytes == lay.total && host.source_off == want_src &&
            host.destination_off == want_dst && host.layer_bytes == want_bytes && host.bounds == want_bounds,
            "six-hole compact shape differs");
    uint64_t blob_off = 0;
    for (int l = 0; l < 12; ++l) for (int e = 0; e < 2; ++e) {
        if (l & 1) require(!host.blob_offset(l, e, blob_off), "six-hole remote blob admitted");
        else require(host.blob_offset(l, e, blob_off) &&
                     blob_off == host.offset[(size_t) l] + (uint64_t) e * lay.bytes[(size_t) l],
                     "six-hole retained blob offset differs");
    }
    ExpertHostLayout rejected = host;
    require(!rejected.build(lay, {1,3,5,7,9,9}, err) &&
            !rejected.build(lay, {1,3,5,9,7,11}, err) &&
            !rejected.build(lay, {1,3,5,7,9,12}, err), "invalid six-hole IDs admitted");

    std::vector<uint8_t> packed((size_t) lay.total);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = (uint8_t) ((i * 37 + 19) % 251);
    const auto packed_path = std::filesystem::path(base.string() + ".sixpacked");
    write_fixture(packed_path, packed);
    std::vector<uint8_t> dst((size_t) retained_bytes + 16, 0xfd), expected((size_t) retained_bytes);
    for (size_t i = 0; i < want_src.size(); ++i)
        std::copy_n(packed.data() + want_src[i], (size_t) want_bytes[i], expected.data() + want_dst[i]);
    auto stats = strata::core::load_experts_scatter(packed_path.string(), dst.data() + 8, retained_bytes,
                                                     want_src, want_dst, want_bytes, 2, 11);
    require(stats.seconds >= 0 && stats.bytes == retained_bytes && stats.layers == 6 &&
            std::equal(expected.begin(), expected.end(), dst.begin() + 8) &&
            std::all_of(dst.begin(), dst.begin() + 8, [](uint8_t b){return b == 0xfd;}) &&
            std::all_of(dst.end() - 8, dst.end(), [](uint8_t b){return b == 0xfd;}),
            "six-hole guarded packed scatter differs");

    std::vector<uint8_t> shards[2];
    lay.gguf_off.resize(36); lay.gguf_file.resize(12);
    const auto shard0 = std::filesystem::path(base.string() + ".sixroles0");
    const auto shard1 = std::filesystem::path(base.string() + ".sixroles1");
    for (int l = 0; l < 12; ++l) {
        auto& shard = shards[l >= 6 ? 1 : 0];
        if (l >= 6) lay.gguf_file[(size_t) l] = shard1.filename().string();
        const uint64_t role_bytes[3] = {lay.fmt[(size_t) l].up_off, lay.fmt[(size_t) l].up_off,
                                        lay.bytes[(size_t) l] - lay.fmt[(size_t) l].down_off};
        const uint64_t role_at[3] = {0, lay.fmt[(size_t) l].up_off, lay.fmt[(size_t) l].down_off};
        for (int r = 0; r < 3; ++r) {
            lay.gguf_off[(size_t) (3*l+r)] = shard.size();
            for (int e = 0; e < 2; ++e) {
                const size_t start = (size_t) (lay.offset[(size_t) l] + (uint64_t) e * lay.bytes[(size_t) l] + role_at[r]);
                shard.insert(shard.end(), packed.begin() + start, packed.begin() + start + role_bytes[r]);
            }
        }
    }
    write_fixture(shard0, shards[0]); write_fixture(shard1, shards[1]);
    std::fill(dst.begin(), dst.end(), 0xfd);
    stats = strata::core::load_experts_gguf_compact(shard0.string(), dst.data() + 8, lay, host, 2);
    require(stats.seconds >= 0 && stats.bytes == retained_bytes && stats.layers == 6 &&
            std::equal(expected.begin(), expected.end(), dst.begin() + 8) &&
            std::all_of(dst.begin(), dst.begin() + 8, [](uint8_t b){return b == 0xfd;}) &&
            std::all_of(dst.end() - 8, dst.end(), [](uint8_t b){return b == 0xfd;}),
            "six-hole guarded native-role scatter differs");

    auto scaled = geometry_six(1024);
    ExpertHostLayout scaled_host;
    require(scaled_host.build(scaled, excluded, err), "scaled six-hole map rejected");
    const uint64_t capacity = scaled_host.host_bytes + scaled.max_blob;
    strata::core::PinnedArena arena(capacity, scaled_host.bounds, capacity);
    require(arena.valid() && arena.registered_bytes == scaled_host.host_bytes &&
            arena.registered_bytes + arena.locked_bytes == capacity && arena.registered_slices == 6,
            "six-hole retained slices are not fully registered");
    for (uint64_t i = 0; i < capacity; ++i) arena.data()[i] = (uint8_t) ((i * 29 + 13) % 251);
    int count = 0; cu(cudaGetDeviceCount(&count)); require(count >= 2, "two CUDA devices required");
    for (int device : {0,1}) {
        cu(cudaSetDevice(device)); cu(cudaFree(nullptr));
        std::vector<const uint8_t*> aliases;
        for (uint64_t start : arena.slice_starts) {
            void* alias = nullptr; cu(cudaHostGetDevicePointer(&alias, arena.data() + start, 0));
            aliases.push_back((const uint8_t*) alias);
        }
        uint8_t* output = nullptr; cu(cudaMalloc((void**) &output, 32));
        for (int l = 0; l < 12; ++l) for (int e = 0; e < 2; ++e) {
            const uint8_t* alias = scaled_host.device_alias(l, e, arena.registered_bytes, aliases, true);
            if (l & 1) require(alias == nullptr, "six-hole excluded layer obtained device alias");
            else {
                uint64_t off = 0;
                require(alias && scaled_host.blob_offset(l, e, off), "six-hole retained alias missing");
                cu(cudaMemcpy(output, alias, 32, cudaMemcpyDeviceToDevice));
                uint8_t back[32]; cu(cudaMemcpy(back, output, 32, cudaMemcpyDeviceToHost));
                require(std::equal(back, back + 32, arena.data() + off), "six-hole mapped alias bytes differ");
            }
        }
        cu(cudaFree(output));
        auto short_aliases = aliases; short_aliases.pop_back();
        require(!scaled_host.device_alias(10, 0, arena.registered_bytes, short_aliases, true),
                "six-hole short alias list accepted");
    }
    cu(cudaSetDevice(0));
}
void selected_twelve_hole_layout(const std::filesystem::path& base) {
    const std::vector<int32_t> excluded{1,3,5,7,9,11,13,15,17,19,21,23};
    auto lay = geometry_twelve();
    ExpertHostLayout host;
    std::string err;
    require(host.build(lay, excluded, err), "twelve-hole selected map rejected");
    uint64_t retained_bytes = 0, excluded_bytes = 0;
    std::vector<uint64_t> want_src, want_dst, want_bytes, want_bounds{0};
    for (int l = 0; l < 24; ++l) {
        const uint64_t layer_bytes = 2 * lay.bytes[(size_t) l];
        if (l & 1) {
            excluded_bytes += layer_bytes;
            require(host.offset[(size_t) l] == UINT64_MAX && host.slice[(size_t) l] == -1,
                    "twelve-hole excluded layer retained");
        } else {
            require(host.offset[(size_t) l] == retained_bytes && host.slice[(size_t) l] >= 0,
                    "twelve-hole retained compact offset differs");
            want_src.push_back(lay.offset[(size_t) l]);
            want_dst.push_back(retained_bytes);
            want_bytes.push_back(layer_bytes);
            retained_bytes += layer_bytes;
            want_bounds.push_back(retained_bytes);
        }
    }
    require(host.host_bytes == retained_bytes && host.excluded_bytes == excluded_bytes &&
            retained_bytes + excluded_bytes == lay.total && host.source_off == want_src &&
            host.destination_off == want_dst && host.layer_bytes == want_bytes && host.bounds == want_bounds,
            "twelve-hole compact shape differs");
    uint64_t blob_off = 0;
    for (int l = 0; l < 24; ++l) for (int e = 0; e < 2; ++e) {
        if (l & 1) require(!host.blob_offset(l, e, blob_off), "twelve-hole remote blob admitted");
        else require(host.blob_offset(l, e, blob_off) &&
                     blob_off == host.offset[(size_t) l] + (uint64_t) e * lay.bytes[(size_t) l],
                     "twelve-hole retained blob offset differs");
    }
    ExpertHostLayout rejected = host;
    require(!rejected.build(lay, {1,3,5,7,9,11,13,15,17,19,21,21}, err) &&
            !rejected.build(lay, {1,3,5,7,9,11,13,15,17,21,19,23}, err) &&
            !rejected.build(lay, {1,3,5,7,9,11,13,15,17,19,21,24}, err),
            "invalid twelve-hole IDs admitted");

    std::vector<uint8_t> packed((size_t) lay.total);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = (uint8_t) ((i * 41 + 23) % 251);
    const auto packed_path = std::filesystem::path(base.string() + ".twelvepacked");
    write_fixture(packed_path, packed);
    std::vector<uint8_t> dst((size_t) retained_bytes + 16, 0xfc), expected((size_t) retained_bytes);
    for (size_t i = 0; i < want_src.size(); ++i)
        std::copy_n(packed.data() + want_src[i], (size_t) want_bytes[i], expected.data() + want_dst[i]);
    const auto stats = strata::core::load_experts_scatter(packed_path.string(), dst.data() + 8, retained_bytes,
                                                           want_src, want_dst, want_bytes, 2, 13);
    require(stats.seconds >= 0 && stats.bytes == retained_bytes && stats.layers == 12 &&
            std::equal(expected.begin(), expected.end(), dst.begin() + 8) &&
            std::all_of(dst.begin(), dst.begin() + 8, [](uint8_t b){return b == 0xfc;}) &&
            std::all_of(dst.end() - 8, dst.end(), [](uint8_t b){return b == 0xfc;}),
            "twelve-hole guarded packed scatter differs");
    const auto unchanged = dst;
    require(strata::core::load_experts_scatter(packed_path.string(), dst.data() + 8, retained_bytes,
                                                {0}, {retained_bytes}, {1}, 1, 13).seconds < 0 &&
            dst == unchanged, "twelve-hole invalid scatter wrote destination");

    auto scaled = geometry_twelve(1024);
    ExpertHostLayout scaled_host;
    require(scaled_host.build(scaled, excluded, err), "scaled twelve-hole map rejected");
    const uint64_t capacity = scaled_host.host_bytes + scaled.max_blob;
    strata::core::PinnedArena arena(capacity, scaled_host.bounds, capacity);
    require(arena.valid() && arena.registered_bytes == scaled_host.host_bytes &&
            arena.registered_bytes + arena.locked_bytes == capacity && arena.registered_slices == 12,
            "twelve-hole retained slices are not fully registered");
    for (uint64_t i = 0; i < capacity; ++i) arena.data()[i] = (uint8_t) ((i * 31 + 17) % 251);
    int count = 0; cu(cudaGetDeviceCount(&count)); require(count >= 2, "two CUDA devices required");
    for (int device : {0,1}) {
        cu(cudaSetDevice(device)); cu(cudaFree(nullptr));
        std::vector<const uint8_t*> aliases;
        for (uint64_t start : arena.slice_starts) {
            void* alias = nullptr; cu(cudaHostGetDevicePointer(&alias, arena.data() + start, 0));
            aliases.push_back((const uint8_t*) alias);
        }
        uint8_t* output = nullptr; cu(cudaMalloc((void**) &output, 32));
        for (int l = 0; l < 24; ++l) for (int e = 0; e < 2; ++e) {
            const uint8_t* alias = scaled_host.device_alias(l, e, arena.registered_bytes, aliases, true);
            if (l & 1) require(alias == nullptr, "twelve-hole excluded layer obtained device alias");
            else {
                uint64_t off = 0;
                require(alias && scaled_host.blob_offset(l, e, off), "twelve-hole retained alias missing");
                cu(cudaMemcpy(output, alias, 32, cudaMemcpyDeviceToDevice));
                uint8_t back[32]; cu(cudaMemcpy(back, output, 32, cudaMemcpyDeviceToHost));
                require(std::equal(back, back + 32, arena.data() + off), "twelve-hole mapped alias bytes differ");
            }
        }
        cu(cudaFree(output));
        auto short_aliases = aliases; short_aliases.pop_back();
        require(!scaled_host.device_alias(22, 0, arena.registered_bytes, short_aliases, true),
                "twelve-hole short alias list accepted");
    }
    cu(cudaSetDevice(0));
}
void selected_pair_hole_layout(const std::filesystem::path& base) {
    const std::vector<int32_t> remote{3,9};
    const std::vector<std::pair<int32_t,int32_t>> cached{{0,0},{2,1},{4,0},{4,1},{6,0},{8,1}};
    auto lay = geometry_six();
    ExpertHostLayout host;
    std::string err;
    require(host.build(lay, remote, cached, err), "mixed pair-hole map rejected");
    uint64_t remote_bytes = 0, cached_bytes = 0, retained_bytes = 0;
    std::vector<uint8_t> packed((size_t) lay.total);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = (uint8_t) ((i * 43 + 29) % 251);
    std::vector<uint8_t> expected;
    for (int l = 0; l < lay.n_layers; ++l) {
        const bool whole = std::binary_search(remote.begin(), remote.end(), l);
        for (int e = 0; e < lay.n_expert; ++e) {
            const std::pair<int32_t,int32_t> pair{l,e};
            const bool pair_hole = std::binary_search(cached.begin(), cached.end(), pair);
            uint64_t off = 0;
            if (whole) {
                remote_bytes += lay.bytes[(size_t) l];
                require(!host.blob_offset(l,e,off), "remote pair retained in mixed map");
            } else if (pair_hole) {
                cached_bytes += lay.bytes[(size_t) l];
                require(!host.blob_offset(l,e,off), "cached pair retained in mixed map");
            } else {
                require(host.blob_offset(l,e,off) && off == retained_bytes,
                        "mixed retained pair offset differs");
                const size_t src = (size_t) (lay.offset[(size_t) l] + (uint64_t) e * lay.bytes[(size_t) l]);
                expected.insert(expected.end(), packed.begin() + src,
                                packed.begin() + src + (size_t) lay.bytes[(size_t) l]);
                retained_bytes += lay.bytes[(size_t) l];
            }
        }
    }
    require(host.remote_excluded_bytes == remote_bytes && host.cached_excluded_bytes == cached_bytes &&
            host.excluded_bytes == remote_bytes + cached_bytes && host.host_bytes == retained_bytes &&
            retained_bytes + remote_bytes + cached_bytes == lay.total && host.excluded_layers == remote &&
            host.excluded_cached_pairs == cached && host.pair_offset.size() == 24,
            "mixed pair-hole accounting differs");
    require(host.offset[0] == 0 && host.slice[0] >= 0 && host.offset[4] == UINT64_MAX && host.slice[4] == -1,
            "expert-zero/full-cached-layer map markers differ");

    ExpertHostLayout rejected;
    require(!rejected.build(lay, remote, {{0,0},{0,0}}, err) &&
            !rejected.build(lay, remote, {{2,1},{0,0}}, err) &&
            !rejected.build(lay, remote, {{0,2}}, err) &&
            !rejected.build(lay, remote, {{3,0}}, err) &&
            !rejected.build(lay, {3,3}, cached, err) &&
            !rejected.build(lay, {9,3}, cached, err), "invalid mixed exclusions admitted");
    auto excessive_geometry=lay; excessive_geometry.n_layers=4096; excessive_geometry.n_expert=7;
    require(!rejected.build(excessive_geometry, {}, {}, err), "unbounded pair table geometry admitted");

    const auto packed_path = std::filesystem::path(base.string() + ".pairpacked");
    write_fixture(packed_path, packed);
    std::vector<uint8_t> dst((size_t) retained_bytes + 16, 0xfb);
    auto stats = strata::core::load_experts_scatter(packed_path.string(), dst.data() + 8, retained_bytes,
                                                     host.source_off, host.destination_off, host.layer_bytes, 2, 9);
    require(stats.seconds >= 0 && stats.bytes == retained_bytes &&
            std::equal(expected.begin(), expected.end(), dst.begin() + 8) &&
            std::all_of(dst.begin(), dst.begin()+8, [](uint8_t b){return b==0xfb;}) &&
            std::all_of(dst.end()-8, dst.end(), [](uint8_t b){return b==0xfb;}),
            "mixed pair-hole packed scatter differs");

    std::vector<uint8_t> shards[2];
    lay.gguf_off.resize(36); lay.gguf_file.resize(12);
    const auto shard0 = std::filesystem::path(base.string() + ".pairroles0");
    const auto shard1 = std::filesystem::path(base.string() + ".pairroles1");
    for (int l = 0; l < 12; ++l) {
        auto& shard = shards[l >= 6 ? 1 : 0];
        if (l >= 6) lay.gguf_file[(size_t) l] = shard1.filename().string();
        const uint64_t role_bytes[3] = {lay.fmt[(size_t) l].up_off, lay.fmt[(size_t) l].up_off,
                                        lay.bytes[(size_t) l] - lay.fmt[(size_t) l].down_off};
        const uint64_t role_at[3] = {0, lay.fmt[(size_t) l].up_off, lay.fmt[(size_t) l].down_off};
        for (int r = 0; r < 3; ++r) {
            lay.gguf_off[(size_t) (3*l+r)] = shard.size();
            for (int e = 0; e < 2; ++e) {
                const size_t start = (size_t) (lay.offset[(size_t) l] + (uint64_t) e * lay.bytes[(size_t) l] + role_at[r]);
                shard.insert(shard.end(), packed.begin()+start, packed.begin()+start+role_bytes[r]);
            }
        }
    }
    write_fixture(shard0, shards[0]); write_fixture(shard1, shards[1]);
    std::fill(dst.begin(), dst.end(), 0xfb);
    stats = strata::core::load_experts_gguf_compact(shard0.string(), dst.data()+8, lay, host, 2);
    require(stats.seconds >= 0 && stats.bytes == retained_bytes && stats.layers == 9 &&
            std::equal(expected.begin(), expected.end(), dst.begin()+8),
            "mixed pair-hole native-role scatter differs");
    ExpertHostLayout corrupt = host; ++corrupt.cached_excluded_bytes;
    const auto before_corrupt = dst;
    require(strata::core::load_experts_gguf_compact(shard0.string(), dst.data()+8, lay, corrupt, 2).seconds < 0 &&
            dst == before_corrupt, "corrupt pair accounting reached destination");

    // Pair holes make retained-layer endpoints deliberately non-page-aligned. Registering each layer could make
    // CUDA round adjacent registrations onto the same page, so static pair mode uses exactly one 64-KiB prefix.
    auto scaled = geometry_six(257);
    ExpertHostLayout scaled_host;
    require(scaled_host.build(scaled, remote, cached, err), "scaled mixed pair-hole map rejected");
    const uint64_t capacity = scaled_host.host_bytes + scaled.max_blob;
    const uint64_t prefix = 64u << 10;
    require(capacity > prefix && prefix < scaled_host.host_bytes &&
            std::any_of(scaled_host.bounds.begin(), scaled_host.bounds.end(),
                        [](uint64_t v){ return (v & 4095) != 0; }),
            "tiny pair fixture did not create unaligned/cross-prefix geometry");
    strata::core::PinnedArena arena(capacity, std::vector<uint64_t>{0,prefix}, prefix);
    require(arena.valid() && arena.registered_bytes == prefix &&
            arena.registered_bytes + arena.locked_bytes == capacity && arena.registered_slices == 1 &&
            arena.slice_starts == std::vector<uint64_t>{0},
            "mixed pair arena did not obtain one complete prefix plus locked suffix");
    for (uint64_t i=0;i<capacity;++i) arena.data()[i]=(uint8_t)((i*47+7)%251);
    int count=0; cu(cudaGetDeviceCount(&count)); require(count>=2,"two CUDA devices required");
    for (int device : {0,1}) {
        cu(cudaSetDevice(device)); cu(cudaFree(nullptr));
        void* alias=nullptr; cu(cudaHostGetDevicePointer(&alias,arena.data(),0));
        const std::vector<const uint8_t*> aliases{(const uint8_t*)alias};
        const uint8_t* retained_alias=scaled_host.device_alias(0,1,arena.registered_bytes,aliases,false);
        require(!scaled_host.device_alias(0,0,arena.registered_bytes,aliases,false) && retained_alias &&
                !scaled_host.device_alias(4,0,arena.registered_bytes,aliases,false) &&
                !scaled_host.device_alias(4,1,arena.registered_bytes,aliases,false),
                "mixed pair aliases admitted absent expert-zero/full layer");
        bool saw_crossing=false;
        for(int l=0;l<scaled.n_layers;++l) for(int e=0;e<scaled.n_expert;++e) {
            uint64_t off=0;
            if(scaled_host.blob_offset(l,e,off) && off < prefix && scaled.bytes[(size_t)l] > prefix-off) {
                saw_crossing=true;
                require(!scaled_host.device_alias(l,e,prefix,aliases,false),
                        "expert crossing registered prefix obtained an alias");
            }
        }
        require(saw_crossing,"tiny pair fixture has no prefix-crossing expert");
        uint64_t retained_off=0; require(scaled_host.blob_offset(0,1,retained_off),"retained pair offset missing");
        uint8_t* output=nullptr; cu(cudaMalloc((void**)&output,32));
        cu(cudaMemcpy(output,retained_alias,32,cudaMemcpyDeviceToDevice));
        uint8_t back[32]; cu(cudaMemcpy(back,output,32,cudaMemcpyDeviceToHost)); cu(cudaFree(output));
        require(std::equal(back,back+32,arena.data()+retained_off),
                "partial-layer alias ignored actual compact pair offset");
    }
    cu(cudaSetDevice(0));

    const int cold_layer=2,cold_expert=0;
    const size_t cold_bytes=(size_t)lay.bytes[(size_t)cold_layer];
    const size_t cold_src=(size_t)(lay.offset[(size_t)cold_layer]+(uint64_t)cold_expert*lay.bytes[(size_t)cold_layer]);
    std::vector<uint8_t> cold(cold_bytes,0), cold2(cold_bytes,0);
    require(strata::core::read_expert_gguf_blob(shard0.string(),lay,cold_layer,cold_expert,
                                                cold.data(),cold.size(),err) &&
            strata::core::read_expert_gguf_blob(shard0.string(),lay,cold_layer,cold_expert,
                                                cold2.data(),cold2.size(),err) &&
            std::equal(cold.begin(),cold.end(),packed.begin()+cold_src) && cold2==cold,
            "repeated cold expert reads differ from independent role bytes");
    require(!strata::core::read_expert_gguf_blob(shard0.string(),lay,cold_layer,cold_expert,
                                                 cold.data(),cold.size()-1,err) &&
            !strata::core::read_expert_gguf_blob(shard0.string(),lay,-1,cold_expert,
                                                 cold.data(),cold.size(),err) &&
            !strata::core::read_expert_gguf_blob(shard0.string(),lay,cold_layer,2,
                                                 cold.data(),cold.size(),err) &&
            !strata::core::read_expert_gguf_blob(shard0.string(),lay,cold_layer,cold_expert,
                                                 nullptr,cold.size(),err), "cold reader API negative accepted");
    auto unsafe=lay; unsafe.gguf_file[(size_t)cold_layer]="../outside.gguf";
    require(!strata::core::read_expert_gguf_blob(shard0.string(),unsafe,cold_layer,cold_expert,
                                                 cold.data(),cold.size(),err), "unsafe sibling admitted");
    auto truncated_lay=lay;
    const auto truncated=std::filesystem::path(base.string()+".pairtruncated");
    auto truncated_bytes=shards[1]; truncated_bytes.pop_back(); write_fixture(truncated,truncated_bytes);
    for(int l=6;l<12;++l) truncated_lay.gguf_file[(size_t)l]=truncated.filename().string();
    std::vector<uint8_t> sentinel((size_t)lay.bytes[11],0xa7), unchanged=sentinel;
    require(!strata::core::read_expert_gguf_blob(shard0.string(),truncated_lay,11,1,
                                                 sentinel.data(),sentinel.size(),err) && sentinel==unchanged,
            "truncated cold read succeeded or partially filled destination");
}
int run(const std::filesystem::path& path) {
    require(!std::filesystem::exists(path), "fixture already exists");
    std::vector<uint8_t> file(1088);
    for (size_t i = 0; i < file.size(); ++i) file[i] = (uint8_t) ((i * 29 + 17) % 251);
    { std::ofstream f(path, std::ios::binary); require((bool) f, "cannot create fixture");
      f.write((const char*) file.data(), (std::streamsize) file.size()); require((bool) f, "fixture write failed"); }
    constexpr uint64_t bias = 320, payload = 768;
    std::vector<uint8_t> dst(payload + 16, 0xfe);
    const std::vector<uint64_t> off{320, 832}, bytes{512, 256};
    auto st = strata::core::load_experts_ranges(path.string(), dst.data() + 8, off, bytes, 2, 31, bias);
    require(st.seconds >= 0 && st.bytes == payload && st.layers == 2, "compacted load counters failed");
    require(std::equal(file.begin() + bias, file.end(), dst.begin() + 8), "compacted suffix bytes differ");
    require(std::all_of(dst.begin(), dst.begin() + 8, [](uint8_t b){return b == 0xfe;}) &&
            std::all_of(dst.end() - 8, dst.end(), [](uint8_t b){return b == 0xfe;}), "destination guard overwritten");
    const auto unchanged = dst;
    require(strata::core::load_experts_ranges(path.string(), dst.data() + 8, {319}, {1}, 1, 31, bias).seconds < 0,
            "bias underflow accepted");
    require(strata::core::load_experts_ranges(path.string(), dst.data() + 8, off, {512}, 1, 31, bias).seconds < 0,
            "mismatched ranges accepted");
    require(strata::core::load_experts_ranges(path.string(), dst.data() + 8, off, bytes, 1, 0, bias).seconds < 0,
            "zero read chunk accepted");
    require(dst == unchanged, "invalid ranges wrote destination bytes");
    require(strata::core::load_experts_ranges(path.string(), dst.data() + 8, {1088}, {1}, 1, 31, bias).seconds < 0,
            "short read was not rejected");
    std::printf("PASS compacted_loader bytes=768 layers=2 guards=exact negative_cases=4\n");
    selected_maps_and_scatter(path);
    selected_device_aliases();
    std::printf("PASS selected_host_layout payload_bytes=80 layers=3 scatter=exact native_roles=exact aliases=exact negatives=exact\n");
    selected_six_hole_layout(path);
    std::printf("PASS selected_host_layout6 ids=1,3,5,7,9,11 holes=6 retained=6 scatter=exact native_roles=exact aliases=exact negatives=exact\n");
    selected_twelve_hole_layout(path);
    std::printf("PASS selected_host_layout12 ids=1,3,5,7,9,11,13,15,17,19,21,23 holes=12 retained=12 scatter=exact aliases=exact negatives=duplicate,unsorted,out_of_range,scatter_bounds,short_alias\n");
    selected_pair_hole_layout(path);
    std::printf("PASS selected_host_pair_holes remote_layers=2 cached_pairs=6 fully_cached_layers=1 expert0_hole=exact accounting=exact packed_scatter=exact native_roles=exact registration=single_prefix_locked_suffix crossing_alias=reject cold_reads=repeat_exact truncated=reject_unchanged negatives=overlap,duplicate,unsorted,out_of_range,corrupt_accounting,unsafe_sibling\n");
    for (int device : {0, 1}) {
        cu(cudaSetDevice(device));
        strata::core::ExpertCache cache;
        std::string err;
        require(cache.open(1, 1, 1, 32, err), "tiny cache allocation failed");
        require(!cache.verify_slot(0, nullptr, err, 32) &&
                err == "ExpertCache::verify_slot: null host source", "null cache source was not cleanly refused");
        const std::vector<uint8_t> source(32, 73);
        require(cache.fill_slot_blocking(0, source.data(), err, 32) &&
                cache.verify_slot(0, source.data(), err, 32), "valid cache check changed after null refusal");
    }
    cu(cudaSetDevice(0));
    std::printf("PASS expert_cache_null_source devices=2 refusal=clean valid_after_refusal=exact\n");
    return 0;
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 3 && std::string(argv[1]) == "--fixture", "usage: compacted_loader_test --fixture PATH");
        return run(std::filesystem::path(argv[2]));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL compacted_loader_test: %s\n", e.what());
        return 1;
    }
}




