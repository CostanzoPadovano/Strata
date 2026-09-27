// Bounded real-pack parity gate for the resident expert service. It never opens the main model or a host arena.
#include "strata/core/remote_experts.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int64_t H = 2560;
constexpr int64_t NE = 512;
constexpr int kLayers = 4;
constexpr int kSmallK = 4;
constexpr int kMaxK = 10;
constexpr int kMaxT = 2048;
constexpr size_t kQ8Row = (size_t) (H / 32) * 36;

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }
void ck(cudaError_t s, const char* what) { if (s != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(s)); }

void require_device0(const char* what) {
    int device = -1;
    ck(cudaGetDevice(&device), "query restored device");
    if (device != 0) fail(std::string(what) + " did not restore CUDA device 0");
}

struct DeviceBuffer {
    int device = -1;
    void* ptr = nullptr;
    ~DeviceBuffer() {
        if (!ptr) return;
        int old = -1;
        if (cudaGetDevice(&old) != cudaSuccess) return;
        if (cudaSetDevice(device) == cudaSuccess) (void) cudaFree(ptr);
        (void) cudaSetDevice(old);
    }
    void alloc(int d, size_t bytes) {
        device = d;
        ck(cudaSetDevice(d), "select allocation device");
        ck(cudaMalloc(&ptr, bytes), "allocate test buffer");
    }
};

std::vector<uint8_t> quant_on(int device, const std::vector<float>& x, int64_t rows) {
    ck(cudaSetDevice(device), "select quant device");
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create quant stream");
    void *dx = nullptr, *dq = nullptr;
    ck(cudaMalloc(&dx, x.size() * sizeof(float)), "allocate quant input");
    ck(cudaMalloc(&dq, (size_t) rows * kQ8Row), "allocate quant output");
    ck(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "upload quant input");
    strata::kernels::quantize_q8_1_rows((const float*) dx, rows, H, dq, stream);
    std::vector<uint8_t> out((size_t) rows * kQ8Row);
    ck(cudaMemcpyAsync(out.data(), dq, out.size(), cudaMemcpyDeviceToHost, stream), "download quant bytes");
    ck(cudaStreamSynchronize(stream), "finish quant parity");
    ck(cudaFree(dx), "free quant input");
    ck(cudaFree(dq), "free quant output");
    ck(cudaStreamDestroy(stream), "destroy quant stream");
    return out;
}

struct FixtureResult {
    std::vector<std::vector<float>> decode;
    std::vector<std::vector<float>> prefill;
    std::vector<std::vector<uint8_t>> quant;
    bool large_prefill_compared = false;
    uint64_t large_prefill_compared_bytes = 0;
};

struct HybridPlanStorage {
    int32_t counts[3]{};
    int32_t start[41]{}, start2[41]{}, dst[40]{}, tok[40]{};
    unsigned long long ptr[40]{}, ptr2[40]{};
};

struct HybridSignals {
    int publish = 0, fetch = 0;
    bool bad_fetch = false;
};

void hybrid_publish(void* ctx) { ++static_cast<HybridSignals*>(ctx)->publish; }
void hybrid_fetch(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    auto& s = *static_cast<HybridSignals*>(ctx);
    ++s.fetch;
    if (src != nullptr || n != 0 || bytes != 0) s.bad_fetch = true;
}

struct MappedHybridPlan {
    HybridPlanStorage* host = nullptr;
    HybridPlanStorage* device = nullptr;
    MappedHybridPlan() {
        ck(cudaHostAlloc((void**) &host, sizeof(*host), cudaHostAllocPortable | cudaHostAllocMapped),
           "allocate mapped hybrid plan");
        ck(cudaHostGetDevicePointer((void**) &device, host, 0), "alias mapped hybrid plan");
    }
    ~MappedHybridPlan() { if (host) (void) cudaFreeHost(host); }
};

void exact(const std::vector<float>& want, const std::vector<float>& got, const char* what) {
    if (want.size() != got.size()) fail(std::string(what) + " size mismatch");
    if (std::memcmp(want.data(), got.data(), want.size() * sizeof(float)) != 0) {
        size_t i = 0;
        while (i < want.size() && std::memcmp(&want[i], &got[i], sizeof(float)) == 0) ++i;
        fail(std::string(what) + " bit mismatch at float " + std::to_string(i));
    }
}

void grouping(const std::vector<int32_t>& ids, int T, int K, std::vector<int32_t>& src,
              std::array<int32_t, NE + 1>& off, std::array<int32_t, NE>& cnt) {
    if ((int64_t) ids.size() != (int64_t) T * K) fail("grouping input extent mismatch");
    cnt.fill(0);
    for (int32_t e : ids) { if (e < 0 || e >= NE) fail("test id out of range"); ++cnt[(size_t) e]; }
    off[0] = 0;
    for (int e = 0; e < NE; ++e) off[(size_t) e + 1] = off[(size_t) e] + cnt[(size_t) e];
    std::array<int32_t, NE> fill{};
    std::copy(off.begin(), off.begin() + NE, fill.begin());
    src.assign(ids.size(), -1);
    for (size_t i = 0; i < ids.size(); ++i) src[(size_t) fill[(size_t) ids[i]]++] = (int32_t) (i / K);
    if (off[NE] != T * K) fail("grouping extent mismatch");
}

void run_hybrid_dispatch_fixture(strata::core::RemoteExperts& remote, int layer, uint64_t blob_bytes) {
    using strata::core::ExpertCache;
    using strata::core::ExpertDispatch;
    using strata::core::GpuPlanSink;
    constexpr int T = 2, K = kMaxK, entries = T * K;
    std::string err;
    const int64_t decode0 = remote.decode_calls(), masked0 = remote.masked_decode_calls();
    const int64_t masked_entries0 = remote.masked_decode_entries(), all_hit0 = remote.masked_all_hit_calls();
    ck(cudaSetDevice(0), "select hybrid cache device");
    ExpertCache cache;
    if (!cache.open_sized({(int64_t) blob_bytes, (int64_t) blob_bytes}, 48, NE, err)) fail(err);
    const int32_t resident_a = 7, resident_b = 3;
    const int32_t slot_a = cache.admit(layer, resident_a), slot_b = cache.admit(layer, resident_b);
    if (slot_a < 0 || slot_b < 0 || slot_a == slot_b) fail("hybrid cache admission failed");
    std::vector<uint8_t> blob_a((size_t) blob_bytes), blob_b((size_t) blob_bytes), blob_c((size_t) blob_bytes);
    if (!remote.readback_blob(layer, resident_a, blob_a.data(), blob_a.size(), err) ||
        !remote.readback_blob(layer, resident_b, blob_b.data(), blob_b.size(), err) ||
        !remote.readback_blob(layer, 511, blob_c.data(), blob_c.size(), err)) fail(err);
    require_device0("hybrid cold blob readback");
    if (!cache.fill_slot_blocking(slot_a, blob_a.data(), err, (int64_t) blob_bytes) ||
        !cache.fill_slot_blocking(slot_b, blob_b.data(), err, (int64_t) blob_bytes) ||
        !cache.verify_slot(slot_a, blob_a.data(), err, (int64_t) blob_bytes) ||
        !cache.verify_slot(slot_b, blob_b.data(), err, (int64_t) blob_bytes)) fail(err);

    std::vector<int32_t> residency((size_t) 48 * NE, strata::core::kNotResident);
    residency[(size_t) layer * NE + resident_a] = slot_a;
    residency[(size_t) layer * NE + resident_b] = slot_b;
    MappedHybridPlan mapped;
    HybridSignals signals;
    GpuPlanSink plan{};
    plan.counts = mapped.host->counts; plan.start = mapped.host->start; plan.start2 = mapped.host->start2;
    plan.dst = mapped.host->dst; plan.tok = mapped.host->tok; plan.ptr = mapped.host->ptr;
    plan.ptr2 = mapped.host->ptr2; plan.cap = 40; plan.publish = hybrid_publish;
    plan.fetch = hybrid_fetch; plan.ctx = &signals;

    ExpertDispatch d{};
    d.remote = &remote; d.remote_hybrid = true; d.n_expert = NE; d.cache = &cache;
    d.cache_base = cache.device_slot(0); d.cache_slots = cache.slots(); d.cache_bytes = (uint64_t) cache.bytes();
    d.cache_blob = (int64_t) blob_bytes; d.cache_slot_off = cache.slot_offsets();
    d.host_res = residency.data(); d.plan = &plan;

    DeviceBuffer dx, dq, dscratch, dhit, dparts;
    dx.alloc(0, (size_t) T * H * sizeof(float));
    dq.alloc(0, (size_t) T * kQ8Row);
    dscratch.alloc(0, strata::kernels::native_expert_scratch_bytes(entries, 640));
    dhit.alloc(0, (size_t) entries * H * sizeof(float));
    dparts.alloc(0, (size_t) entries * H * sizeof(float));
    const auto& layout = strata::kernels::cpu::expert_layout();
    const auto& fmt = layout.fmt[(size_t) layer];
    const auto native_layout = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, H, 640);

    int64_t gpu0_entries = 0, gpu1_entries = 0;
    auto run_case = [&](const std::vector<int32_t>& ids, int64_t expected_hits, const char* name) {
        std::vector<float> x((size_t) T * H);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = (float) ((int) ((i * 43 + ids[0] * 17) % 557) - 278) / 269.0f;
        std::vector<float> reference((size_t) entries * H), combined(reference.size(), -19.0f);
        if (!remote.decode(layer, x.data(), ids.data(), T, K, reference.data(), err)) fail(err);
        d.layers = layer; d.failed = false; d.fail = nullptr; d.fail_layer = -1;
        const int pub0 = signals.publish, fetch0 = signals.fetch;
        strata::core::expert_pool_dispatch_multi(d, x.data(), ids.data(), T, K, combined.data());
        require_device0(name);
        if (d.failed || signals.publish != pub0 + 1 || signals.fetch != fetch0 + 1 || signals.bad_fetch)
            fail(std::string(name) + " dispatch/handshake failed");
        if (mapped.host->counts[1] != expected_hits || mapped.host->counts[2] != 0 ||
            mapped.host->start2[0] != expected_hits) fail(std::string(name) + " plan counts mismatch");
        std::array<uint8_t, entries> seen{};
        for (int g = 0; g < mapped.host->counts[0]; ++g) {
            if (mapped.host->start[g] < 0 || mapped.host->start[g] >= mapped.host->start[g + 1] ||
                mapped.host->start[g + 1] > expected_hits) fail(std::string(name) + " group bounds mismatch");
            const int first = mapped.host->dst[mapped.host->start[g]];
            const int32_t slot = residency[(size_t) layer * NE + (size_t) ids[(size_t) first]];
            if (slot < 0 || mapped.host->ptr[g] != (unsigned long long) cache.device_slot(slot))
                fail(std::string(name) + " group pointer mismatch");
        }
        for (int j = 0; j < expected_hits; ++j) {
            const int dst = mapped.host->dst[j];
            if (dst < 0 || dst >= entries || seen[(size_t) dst]++ || mapped.host->tok[j] != dst / K ||
                residency[(size_t) layer * NE + (size_t) ids[(size_t) dst]] < 0)
                fail(std::string(name) + " dst/token identity mismatch");
        }
        int resident_routes = 0;
        for (int i = 0; i < entries; ++i)
            resident_routes += residency[(size_t) layer * NE + (size_t) ids[(size_t) i]] >= 0;
        if (resident_routes != expected_hits) fail(std::string(name) + " missing/duplicate hit row");
        ck(cudaMemcpy(dx.ptr, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload hybrid activation");
        ck(cudaMemcpy(dparts.ptr, combined.data(), combined.size() * sizeof(float), cudaMemcpyHostToDevice),
           "upload hybrid remote parts");
        ck(cudaMemset(dhit.ptr, 0, (size_t) entries * H * sizeof(float)), "zero hybrid hit output");
        strata::kernels::quantize_q8_1_rows((const float*) dx.ptr, T, H, dq.ptr, nullptr);
        strata::kernels::native_expert_grouped(native_layout, mapped.device->ptr, mapped.device->start,
            mapped.device->counts, mapped.device->dst, mapped.device->tok, entries, entries,
            dq.ptr, dscratch.ptr, (float*) dhit.ptr, nullptr);
        strata::kernels::moe_hit_add((float*) dparts.ptr, (const float*) dhit.ptr, mapped.device->dst,
                                    mapped.device->counts + 1, entries, H, nullptr);
        ck(cudaMemcpy(combined.data(), dparts.ptr, combined.size() * sizeof(float), cudaMemcpyDeviceToHost),
           "download hybrid combined parts");
        exact(reference, combined, name);
        gpu0_entries += expected_hits; gpu1_entries += entries - expected_hits;
    };

    std::vector<int32_t> all_hit(entries), all_miss(entries), mixed(entries);
    const int32_t misses[4] = {511,4,9,300};
    for (int i = 0; i < entries; ++i) {
        all_hit[(size_t) i] = (i & 1) ? resident_a : resident_b;
        all_miss[(size_t) i] = misses[i % 4];
        mixed[(size_t) i] = (i & 1) ? misses[(i / 2) % 4] : ((i / 2) & 1 ? resident_a : resident_b);
    }
    run_case(all_hit, 20, "hybrid all-hit");
    run_case(all_miss, 0, "hybrid all-miss");
    run_case(mixed, 10, "hybrid mixed");
    // Borrow a slot for a formerly remote expert, route with the changed table, then cold-restore its owner.
    if (!cache.fill_slot_blocking(slot_a, blob_c.data(), err, (int64_t) blob_bytes) ||
        !cache.verify_slot(slot_a, blob_c.data(), err, (int64_t) blob_bytes)) fail(err);
    residency[(size_t) layer * NE + resident_a] = strata::core::kNotResident;
    residency[(size_t) layer * NE + 511] = slot_a;
    run_case(mixed, 8, "hybrid changed residency");
    residency[(size_t) layer * NE + 511] = strata::core::kNotResident;
    residency[(size_t) layer * NE + resident_a] = slot_a;
    std::fill(blob_a.begin(), blob_a.end(), UINT8_C(0xa5));
    if (!remote.readback_blob(layer, resident_a, blob_a.data(), blob_a.size(), err)) fail(err);
    require_device0("hybrid cold restore readback");
    if (!cache.fill_slot_blocking(slot_a, blob_a.data(), err, (int64_t) blob_bytes) ||
        !cache.verify_slot(slot_a, blob_a.data(), err, (int64_t) blob_bytes)) fail(err);
    run_case(mixed, 10, "hybrid restored mixed");
    if (d.remote_hybrid_calls != 5 || d.remote_hybrid_hits != 48 || d.remote_hybrid_misses != 52 ||
        gpu0_entries != 48 || gpu1_entries != 52) fail("hybrid aggregate counters mismatch");

    int negative_assertions = 0;
    auto expect_plan_failure = [&](std::vector<int32_t> ids, const char* name) {
        d.layers = layer; d.failed = false; d.fail = nullptr;
        const int pub0 = signals.publish, fetch0 = signals.fetch;
        std::vector<float> x((size_t) T * H, 0.25f), out((size_t) entries * H, -4.0f);
        strata::core::expert_pool_dispatch_multi(d, x.data(), ids.data(), T, K, out.data());
        require_device0(name);
        if (!d.failed || signals.publish != pub0 || signals.fetch != fetch0)
            fail(std::string(name) + " published an invalid plan");
        ++negative_assertions;
    };
    residency[(size_t) layer * NE + resident_a] = (int32_t) cache.slots();
    expect_plan_failure(mixed, "hybrid invalid slot");
    residency[(size_t) layer * NE + resident_a] = slot_a;
    const uint64_t saved_bytes = d.cache_bytes; d.cache_bytes = blob_bytes - 1;
    expect_plan_failure(mixed, "hybrid short extent");
    d.cache_bytes = saved_bytes;
    auto bad_ids = mixed; bad_ids[5] = 512;
    expect_plan_failure(bad_ids, "hybrid invalid id");

    if (!remote.set_decode_mode(strata::core::RemoteDecodeMode::Packed, err)) fail(err);
    d.layers = layer; d.failed = false; d.fail = nullptr;
    const int pub0 = signals.publish, fetch0 = signals.fetch;
    std::vector<float> fail_x((size_t) T * H, 0.125f), fail_out((size_t) entries * H, 6.0f);
    strata::core::expert_pool_dispatch_multi(d, fail_x.data(), mixed.data(), T, K, fail_out.data());
    require_device0("hybrid remote failure");
    if (!d.failed || signals.publish != pub0 + 1 || signals.fetch != fetch0 + 1)
        fail("hybrid remote failure did not release both callback signals");
    exact(std::vector<float>(fail_out.size(), 0.0f), fail_out, "hybrid remote failure zero rows");
    ++negative_assertions;
    if (!remote.set_decode_mode(strata::core::RemoteDecodeMode::Original, err)) fail(err);
    require_device0("hybrid mode restoration");
    if (signals.publish != 6 || signals.fetch != 6 || signals.bad_fetch)
        fail("hybrid callback totals mismatch");
    if (negative_assertions != 4 || remote.decode_calls() != decode0 + 9 ||
        remote.masked_decode_calls() != masked0 + 5 ||
        remote.masked_decode_entries() != masked_entries0 + 52 ||
        remote.masked_all_hit_calls() != all_hit0 + 1)
        fail("hybrid service/negative counters mismatch");
}

bool seek_file(std::FILE* f, uint64_t off) {
#if defined(_WIN32)
    return off <= (uint64_t) std::numeric_limits<__int64>::max() && _fseeki64(f, (__int64) off, SEEK_SET) == 0;
#else
    return off <= (uint64_t) std::numeric_limits<off_t>::max() && fseeko(f, (off_t) off, SEEK_SET) == 0;
#endif
}

std::filesystem::path shard_for(const std::string& native, int layer) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::error_code ec;
    std::filesystem::path primary = std::filesystem::weakly_canonical(native, ec);
    if (ec) fail("cannot canonicalize independent native-shard reference");
    if (lay.gguf_file.empty() || lay.gguf_file[(size_t) layer].empty()) return primary;
    const std::filesystem::path leaf(lay.gguf_file[(size_t) layer]);
    if (leaf.empty() || leaf == "." || leaf == ".." || leaf.is_absolute() || leaf.has_parent_path() ||
        leaf.filename() != leaf)
        fail("independent reference rejected unsafe sibling shard name");
    const std::filesystem::path result = std::filesystem::weakly_canonical(primary.parent_path() / leaf, ec);
    if (ec || result.parent_path() != primary.parent_path()) fail("independent sibling shard resolution failed");
    return result;
}

std::vector<uint8_t> expected_blob(const std::string& native, int layer, int expert) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.blob_bytes(layer);
    const uint64_t per[3] = {(uint64_t) fm.up_off, (uint64_t) fm.up_off, blob - (uint64_t) fm.down_off};
    const uint64_t at[3] = {0, (uint64_t) fm.up_off, (uint64_t) fm.down_off};
    const std::filesystem::path shard = shard_for(native, layer);
    std::FILE* f = std::fopen(shard.string().c_str(), "rb");
    if (!f) fail("cannot open independent GGUF shard reference");
    std::vector<uint8_t> out((size_t) blob);
    for (int r = 0; r < 3; ++r) {
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + (uint64_t) expert * per[r];
        if (!seek_file(f, src) || std::fread(out.data() + at[r], 1, (size_t) per[r], f) != per[r]) {
            std::fclose(f);
            fail("independent GGUF expert slice read failed");
        }
    }
    std::fclose(f);
    return out;
}

FixtureResult run_service(int service_device, const std::string& pack, const std::string& native, cudaStream_t target_stream,
                          const uint16_t* d_mixed, float* d_out, uint64_t expected_payload,
                          const std::vector<int32_t>& layers, bool selected) {
    FixtureResult result;
    strata::core::RemoteExperts remote;
    std::string err;
    ck(cudaSetDevice(0), "select target before service load");
    if (selected) {
        if (!remote.load(pack, layers, service_device, err, native)) fail(err);
    } else if (!remote.load(pack, kLayers, service_device, err, native)) fail(err);
    require_device0("service load");
    if (remote.device() != service_device || remote.layer_count() != kLayers ||
        remote.layer_ids() != layers || remote.payload_bytes() != expected_payload || remote.vram_bytes() > (8ull << 30))
        fail("resident service accounting mismatch");

    // Independent source reads prove that both service devices hold the right byte scatter, not merely the
    // same scatter. Only two experts per selected layer are read; the test never materializes a tensor/file.
    for (int layer : layers) for (int expert : {0, 511}) {
        const std::vector<uint8_t> want = expected_blob(native, layer, expert);
        std::vector<uint8_t> got(want.size());
        if (!remote.readback_blob(layer, expert, got.data(), got.size(), err)) fail(err);
        require_device0("blob readback");
        if (got != want) fail("resident blob bytes differ from independent GGUF slices");
        if (remote.decode_calls() != 0 || remote.prefill_calls() != 0) fail("blob readback changed service counters");
    }

    // Every rejection is pre-write and must leave counters and the caller's output sentinel untouched.
    std::vector<float> bad_x((size_t) 5 * H, 0.25f);
    std::vector<int32_t> bad_ids((size_t) 5 * kMaxK, 7);
    std::vector<float> bad_out((size_t) 5 * kMaxK * H, -123.5f);
    const std::vector<float> bad_out_sentinel = bad_out;
    auto rejected_decode = [&](int64_t layer, int64_t T, int64_t K, const char* name) {
        err.clear();
        if (remote.decode(layer, bad_x.data(), bad_ids.data(), T, K, bad_out.data(), err))
            fail(std::string(name) + " unexpectedly succeeded");
        require_device0(name);
        exact(bad_out_sentinel, bad_out, name);
        if (remote.decode_calls() != 0) fail(std::string(name) + " changed the decode counter");
    };
    int missing_layer = 0;
    while (std::find(layers.begin(), layers.end(), missing_layer) != layers.end()) ++missing_layer;
    const int first_layer = layers.front(), middle_layer = layers[2], last_layer = layers.back();
    rejected_decode(missing_layer, 1, kMaxK, "missing-layer decode");
    rejected_decode(first_layer, 5, kMaxK, "T5 decode");
    rejected_decode(first_layer, 1, 11, "K11 decode");
    bad_ids[3] = 512;
    rejected_decode(first_layer, 1, kMaxK, "bad-id decode");
    bad_ids[3] = 7;
    for (int layer = 0; layer < 48; ++layer)
        if (remote.owns(layer) != (std::find(layers.begin(), layers.end(), layer) != layers.end()))
            fail("owns() selected-layer map mismatch");
    err.clear();
    const bool repeated = selected ? remote.load(pack, layers, service_device, err, native)
                                   : remote.load(pack, kLayers, service_device, err, native);
    if (repeated) fail("repeated load unexpectedly succeeded");
    require_device0("repeated load");
    if (remote.layer_count() != kLayers || remote.decode_calls() != 0 || remote.prefill_calls() != 0)
        fail("repeated load changed completed service state");

    std::array<int32_t, NE + 1> bad_off{};
    std::array<int32_t, NE> bad_cnt{};
    std::array<int32_t, kMaxK> bad_src{};
    bad_cnt[7] = kMaxK;
    for (int e = 8; e <= NE; ++e) bad_off[(size_t) e] = kMaxK;
    ck(cudaMemsetAsync(d_out, 0xa5, (size_t) kMaxK * H * sizeof(float), target_stream), "set prefill sentinel");
    ck(cudaStreamSynchronize(target_stream), "finish prefill sentinel");
    std::vector<uint8_t> device_sentinel((size_t) kMaxK * H * sizeof(float));
    ck(cudaMemcpy(device_sentinel.data(), d_out, device_sentinel.size(), cudaMemcpyDeviceToHost), "read prefill sentinel");
    auto rejected_prefill = [&](const int32_t* src, const int32_t* off, const int32_t* cnt, const char* name) {
        err.clear();
        if (remote.prefill(first_layer, 0, target_stream, d_mixed, src, off, cnt, 1, kMaxK, d_out, err))
            fail(std::string(name) + " unexpectedly succeeded");
        require_device0(name);
        std::vector<uint8_t> got(device_sentinel.size());
        ck(cudaMemcpy(got.data(), d_out, got.size(), cudaMemcpyDeviceToHost), "read rejected prefill output");
        if (got != device_sentinel) fail(std::string(name) + " wrote caller output");
        if (remote.prefill_calls() != 0) fail(std::string(name) + " changed the prefill counter");
    };
    auto inconsistent_off = bad_off;
    inconsistent_off[8] = kMaxK - 1;
    rejected_prefill(bad_src.data(), inconsistent_off.data(), bad_cnt.data(), "bad-offset prefill");
    auto invalid_src = bad_src;
    invalid_src[0] = 1;
    rejected_prefill(invalid_src.data(), bad_off.data(), bad_cnt.data(), "bad-source prefill");

    // Retain the original layer-edge fixtures, then cover every supported decode T at the real K=10.
    std::vector<float> masked_x, masked_reference;
    std::vector<int32_t> masked_ids;
    for (int layer : {first_layer, last_layer}) {
        constexpr int T = 4;
        std::vector<float> x((size_t) T * H);
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float) ((int) ((i * 37 + layer * 11) % 521) - 260) / 257.0f;
        const std::array<int32_t, T * kSmallK> ids{{7, 3, 7, 511, 4, 3, 8, 4, 511, 8, 7, 3, 9, 9, 4, 3}};
        std::vector<float> out((size_t) T * kSmallK * H);
        if (!remote.decode(layer, x.data(), ids.data(), T, kSmallK, out.data(), err)) fail(err);
        require_device0("edge decode");
        if (layer == first_layer) {
            masked_x = x;
            masked_ids.assign(ids.begin(), ids.end());
            masked_reference = out;
        }
        result.decode.push_back(std::move(out));
    }
    for (int T = 1; T <= 4; ++T) {
        const int layer = T < 3 ? first_layer : last_layer;
        std::vector<float> x((size_t) T * H);
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float) ((int) ((i * 41 + T * 13) % 547) - 273) / 263.0f;
        std::vector<int32_t> ids((size_t) T * kMaxK);
        const int32_t pattern[13] = {511, 2, 17, 2, 300, 9, 17, 4, 511, 9, 3, 300, 2};
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = pattern[(i * 7 + (size_t) T) % 13];
        if (T == 1) {
            std::array<int, NE> counts{};
            for (int32_t e : ids) ++counts[(size_t) e];
            bool has_one = false, has_two = false;
            for (int n : counts) { has_one = has_one || n == 1; has_two = has_two || n == 2; }
            if (!has_one || !has_two) fail("K10 fixture did not exercise one- and two-entry groups");
        }
        std::vector<float> out((size_t) T * kMaxK * H);
        if (!remote.decode(layer, x.data(), ids.data(), T, kMaxK, out.data(), err)) fail(err);
        require_device0("K10 decode");
        result.decode.push_back(std::move(out));
    }

    // Invalid masked requests are fully validated before output/counters/device state change.
    constexpr int masked_T = 4, masked_K = kSmallK, masked_entries = masked_T * masked_K;
    std::vector<uint8_t> mask((size_t) masked_entries, 0);
    std::vector<float> masked_out((size_t) masked_entries * H, -91.25f);
    const std::vector<float> masked_sentinel = masked_out;
    auto rejected_masked = [&](int64_t layer, const int32_t* ids, const uint8_t* bits,
                               int64_t T, int64_t K, const char* name) {
        err.clear();
        if (remote.decode_masked(layer, masked_x.data(), ids, bits, T, K, masked_out.data(), err))
            fail(std::string(name) + " unexpectedly succeeded");
        require_device0(name);
        exact(masked_sentinel, masked_out, name);
        if (remote.decode_calls() != 6 || remote.masked_decode_calls() != 0 ||
            remote.masked_decode_entries() != 0 || remote.masked_all_hit_calls() != 0)
            fail(std::string(name) + " changed masked counters");
    };
    auto invalid_mask = mask;
    invalid_mask[3] = 2;
    rejected_masked(first_layer, masked_ids.data(), invalid_mask.data(), masked_T, masked_K, "bad masked value");
    auto invalid_mask_id = masked_ids;
    invalid_mask_id[2] = 512;
    rejected_masked(first_layer, invalid_mask_id.data(), mask.data(), masked_T, masked_K, "bad masked id");
    rejected_masked(missing_layer, masked_ids.data(), mask.data(), masked_T, masked_K, "missing masked layer");
    rejected_masked(first_layer, masked_ids.data(), nullptr, masked_T, masked_K, "null mask");
    rejected_masked(first_layer, masked_ids.data(), mask.data(), 5, masked_K, "T5 masked decode");
    rejected_masked(first_layer, masked_ids.data(), mask.data(), masked_T, 11, "K11 masked decode");
    if (!remote.set_decode_mode(strata::core::RemoteDecodeMode::Packed, err)) fail(err);
    rejected_masked(first_layer, masked_ids.data(), mask.data(), masked_T, masked_K, "packed masked decode");
    if (!remote.set_decode_mode(strata::core::RemoteDecodeMode::Original, err)) fail(err);
    require_device0("restore Original masked mode");

    // All-hit avoids CUDA entirely; all-one and complementary masks prove exact scatter and stale-row clearing.
    if (!remote.decode_masked(first_layer, masked_x.data(), masked_ids.data(), mask.data(),
                              masked_T, masked_K, masked_out.data(), err)) fail(err);
    require_device0("all-hit masked decode");
    exact(std::vector<float>(masked_out.size(), 0.0f), masked_out, "all-hit masked zero rows");
    if (remote.decode_calls() != 6 || remote.masked_decode_calls() != 1 ||
        remote.masked_decode_entries() != 0 || remote.masked_all_hit_calls() != 1)
        fail("all-hit masked counters mismatch");

    std::fill(mask.begin(), mask.end(), 1);
    if (!remote.decode_masked(first_layer, masked_x.data(), masked_ids.data(), mask.data(),
                              masked_T, masked_K, masked_out.data(), err)) fail(err);
    require_device0("all-one masked decode");
    exact(masked_reference, masked_out, "all-one masked reference");

    auto check_sparse = [&](int parity, const char* name) {
        std::vector<float> want(masked_reference.size(), 0.0f);
        for (int i = 0; i < masked_entries; ++i) {
            mask[(size_t) i] = (uint8_t) ((i & 1) == parity);
            if (mask[(size_t) i])
                std::memcpy(want.data() + (size_t) i * H,
                            masked_reference.data() + (size_t) i * H, (size_t) H * sizeof(float));
        }
        std::fill(masked_out.begin(), masked_out.end(), -73.5f);
        if (!remote.decode_masked(first_layer, masked_x.data(), masked_ids.data(), mask.data(),
                                  masked_T, masked_K, masked_out.data(), err)) fail(err);
        require_device0(name);
        exact(want, masked_out, name);
    };
    check_sparse(0, "mixed even masked decode");
    check_sparse(1, "changing odd masked decode");
    if (remote.decode_calls() != 9 || remote.masked_decode_calls() != 4 ||
        remote.masked_decode_entries() != 32 || remote.masked_all_hit_calls() != 1)
        fail("masked service counters mismatch");

    static constexpr uint16_t half_pattern[] = {0x0000, 0x3c00, 0xbc00, 0x3800, 0xb800, 0x3400};
    for (int T = 1; T <= 4; ++T) {
        std::vector<uint16_t> mixed((size_t) T * H);
        for (size_t i = 0; i < mixed.size(); ++i) mixed[i] = half_pattern[(i * 5 + (size_t) T) % 6];
        ck(cudaSetDevice(0), "select prefill target");
        ck(cudaMemcpyAsync((void*) d_mixed, mixed.data(), mixed.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice, target_stream), "upload prefill fixture");
        std::vector<int32_t> ids((size_t) T * kSmallK);
        const int32_t pattern[8] = {17, 2, 17, 300, 2, 9, 300, 17};
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = pattern[(i + (size_t) T) % 8];
        std::vector<int32_t> src;
        std::array<int32_t, NE + 1> off{};
        std::array<int32_t, NE> cnt{};
        grouping(ids, T, kSmallK, src, off, cnt);
        if (!remote.prefill(middle_layer, 0, target_stream, d_mixed, src.data(), off.data(), cnt.data(), T, kSmallK, d_out, err))
            fail(err);
        require_device0("small prefill");
        std::vector<float> out((size_t) T * kSmallK * H);
        ck(cudaSetDevice(0), "select prefill result device");
        ck(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost), "download prefill result");
        result.prefill.push_back(std::move(out));
    }
    {
        const int T = kMaxT, K = kMaxK;
        const uint64_t output_bytes = (uint64_t) T * K * H * sizeof(float);
        if (output_bytes <= (32ull << 20)) fail("large prefill does not cross the transport chunk boundary");
        std::vector<uint16_t> mixed((size_t) T * H);
        for (size_t i = 0; i < mixed.size(); ++i) mixed[i] = half_pattern[(i * 11 + 3) % 6];
        ck(cudaSetDevice(0), "select large prefill target");
        ck(cudaMemcpyAsync((void*) d_mixed, mixed.data(), mixed.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice, target_stream), "upload large prefill fixture");
        std::vector<int32_t> ids((size_t) T * K);
        const int32_t sparse[5] = {2, 17, 300, 511, 9};
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = sparse[(i * 3 + i / K) % 5];
        std::vector<int32_t> src;
        std::array<int32_t, NE + 1> off{};
        std::array<int32_t, NE> cnt{};
        grouping(ids, T, K, src, off, cnt);
        int active = 0;
        for (int32_t n : cnt) active += n > 0;
        if (active < 3) fail("large prefill did not create sparse expert groups");
        if (!remote.prefill(last_layer, 0, target_stream, d_mixed, src.data(), off.data(), cnt.data(), T, K, d_out, err))
            fail(err);
        require_device0("large prefill");
        std::vector<float> out((size_t) T * K * H);
        ck(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost),
           "download multi-chunk prefill result");
        result.prefill.push_back(std::move(out));
    }
    if (remote.decode_calls() != 9 || remote.prefill_calls() != 5 || remote.masked_decode_calls() != 4 ||
        remote.masked_decode_entries() != 32 || remote.masked_all_hit_calls() != 1)
        fail("service call counters mismatch");
    if (service_device == 1)
        run_hybrid_dispatch_fixture(remote, first_layer,
                                    strata::kernels::cpu::expert_layout().blob_bytes(first_layer));
    return result;
}

FixtureResult run_service_six(int service_device, const std::string& pack, const std::string& native,
                              cudaStream_t target_stream, const uint16_t* d_mixed, float* d_out,
                              const std::vector<int32_t>& layers, const FixtureResult* reference) {
    constexpr uint64_t expected_payload = 6448742400ull;
    constexpr uint64_t expected_allocation = 6949138728ull;
    FixtureResult result;
    strata::core::RemoteExperts remote;
    std::string err;
    std::fprintf(stderr, "SIX_STAGE owner=%d begin\n", service_device);
    ck(cudaSetDevice(0), "select target before six-layer service load");
    if (!remote.load(pack, layers, service_device, err, native)) fail(err);
    std::fprintf(stderr, "SIX_STAGE owner=%d load\n", service_device);
    require_device0("six-layer service load");
    if (remote.device() != service_device || remote.layer_count() != 6 || remote.layer_ids() != layers ||
        remote.payload_bytes() != expected_payload || remote.vram_bytes() != expected_allocation ||
        remote.decode_mode() != strata::core::RemoteDecodeMode::Original)
        fail("six-layer resident service accounting/mode mismatch");

    int blob_cases = 0;
    for (int layer : layers) for (int expert : {0, 511}) {
        const std::vector<uint8_t> want = expected_blob(native, layer, expert);
        std::vector<uint8_t> got(want.size());
        if (!remote.readback_blob(layer, expert, got.data(), got.size(), err)) fail(err);
        require_device0("six-layer blob readback");
        if (got != want) fail("six-layer resident blob differs from independent GGUF roles");
        ++blob_cases;
    }
    if (blob_cases != 12 || remote.decode_calls() != 0 || remote.prefill_calls() != 0)
        fail("six-layer blob coverage/counters mismatch");
    std::fprintf(stderr, "SIX_STAGE owner=%d bytesdone cases=12\n", service_device);

    for (size_t li = 0; li < layers.size(); ++li) {
        const int layer = layers[li], T = (int) (li % 4) + 1, K = kMaxK;
        std::vector<float> x((size_t) T * H);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = (float) ((int) ((i * 47 + li * 31 + layer * 7) % 563) - 281) / 271.0f;
        std::vector<int32_t> ids((size_t) T * K);
        const int32_t pattern[11] = {511,2,17,300,9,4,17,3,511,300,2};
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = pattern[(i * 5 + li) % 11];
        result.quant.push_back(quant_on(service_device, x, T));
        ck(cudaSetDevice(0), "restore target after six-layer quant fixture");
        std::vector<float> out((size_t) T * K * H);
        if (!remote.decode(layer, x.data(), ids.data(), T, K, out.data(), err)) fail(err);
        require_device0("six-layer decode");
        result.decode.push_back(std::move(out));
    }
    std::fprintf(stderr, "SIX_STAGE owner=%d decode calls=6\n", service_device);

    static constexpr uint16_t half_pattern[] = {0x0000,0x3c00,0xbc00,0x3800,0xb800,0x3400};
    for (size_t li = 0; li < layers.size(); ++li) {
        const int layer = layers[li], T = (int) (li % 4) + 1, K = kSmallK;
        std::vector<uint16_t> mixed((size_t) T * H);
        for (size_t i = 0; i < mixed.size(); ++i) mixed[i] = half_pattern[(i * 7 + li) % 6];
        ck(cudaSetDevice(0), "select six-layer prefill target");
        ck(cudaMemcpyAsync((void*) d_mixed, mixed.data(), mixed.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice, target_stream), "upload six-layer prefill fixture");
        std::vector<int32_t> ids((size_t) T * K);
        const int32_t pattern[8] = {17,2,17,300,2,9,300,17};
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = pattern[(i + li) % 8];
        std::vector<int32_t> src;
        std::array<int32_t, NE + 1> off{};
        std::array<int32_t, NE> cnt{};
        grouping(ids, T, K, src, off, cnt);
        if (!remote.prefill(layer, 0, target_stream, d_mixed, src.data(), off.data(), cnt.data(), T, K, d_out, err))
            fail(err);
        require_device0("six-layer prefill");
        std::vector<float> out((size_t) T * K * H);
        ck(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost),
           "download six-layer prefill result");
        result.prefill.push_back(std::move(out));
    }
    {
        const int T = kMaxT, K = kMaxK, layer = layers.back();
        constexpr uint64_t output_bytes = (uint64_t) kMaxT * kMaxK * H * sizeof(float);
        static_assert(output_bytes == 209715200ull);
        std::vector<uint16_t> mixed((size_t) T * H);
        for (size_t i = 0; i < mixed.size(); ++i) mixed[i] = half_pattern[(i * 11 + 5) % 6];
        ck(cudaSetDevice(0), "select six-layer large prefill target");
        ck(cudaMemcpyAsync((void*) d_mixed, mixed.data(), mixed.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice, target_stream), "upload six-layer large prefill fixture");
        std::vector<int32_t> ids((size_t) T * K);
        const int32_t sparse[5] = {2,17,300,511,9};
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = sparse[(i * 3 + i / K) % 5];
        std::vector<int32_t> src;
        std::array<int32_t, NE + 1> off{};
        std::array<int32_t, NE> cnt{};
        grouping(ids, T, K, src, off, cnt);
        std::fprintf(stderr, "SIX_STAGE owner=%d large_native_begin bytes=%llu\n", service_device,
                     (unsigned long long) output_bytes);
        if (!remote.prefill(layer, 0, target_stream, d_mixed, src.data(), off.data(), cnt.data(), T, K, d_out, err))
            fail(err);
        require_device0("six-layer large prefill");
        std::fprintf(stderr, "SIX_STAGE owner=%d large_native_done bytes=%llu\n", service_device,
                     (unsigned long long) output_bytes);
        ck(cudaSetDevice(0), "select six-layer large prefill result device");
        ck(cudaStreamSynchronize(target_stream), "finish six-layer large prefill target stream");
        std::fprintf(stderr, "SIX_STAGE owner=%d large_host_begin mode=%s\n", service_device,
                     reference ? "chunked_compare" : "full_reference");
        if (!reference) {
            std::vector<float> out((size_t) T * K * H);
            ck(cudaMemcpy(out.data(), d_out, (size_t) output_bytes, cudaMemcpyDeviceToHost),
               "download six-layer large prefill reference");
            result.prefill.push_back(std::move(out));
        } else {
            if (reference->prefill.size() != 7 ||
                reference->prefill.back().size() != (size_t) T * K * H ||
                reference->prefill.back().size() * sizeof(float) != output_bytes)
                fail("six-layer large prefill reference extent mismatch");
            constexpr uint64_t chunk_bytes = 16ull << 20;
            std::vector<float> chunk((size_t) (chunk_bytes / sizeof(float)));
            const uint8_t* want = (const uint8_t*) reference->prefill.back().data();
            uint64_t compared = 0;
            while (compared < output_bytes) {
                const uint64_t n = std::min(chunk_bytes, output_bytes - compared);
                if (!n || n % sizeof(float) || compared > output_bytes - n) fail("six-layer chunk bounds invalid");
                ck(cudaMemcpy(chunk.data(), (const uint8_t*) d_out + compared, (size_t) n,
                              cudaMemcpyDeviceToHost), "download six-layer large prefill chunk");
                if (std::memcmp(chunk.data(), want + compared, (size_t) n) != 0)
                    fail("six-layer large prefill chunk bit mismatch");
                compared += n;
            }
            if (compared != output_bytes) fail("six-layer large prefill compared byte count mismatch");
            result.prefill.emplace_back();
            result.large_prefill_compared_bytes = compared;
            result.large_prefill_compared = true;
            std::fprintf(stderr, "SIX_STAGE owner=%d large_compare_done bytes=%llu\n", service_device,
                         (unsigned long long) compared);
        }
    }
    std::fprintf(stderr, "SIX_STAGE owner=%d prefill calls=7\n", service_device);
    if (result.quant.size() != 6 || result.decode.size() != 6 || result.prefill.size() != 7 ||
        remote.decode_calls() != 6 || remote.prefill_calls() != 7 || remote.decode_graph_calls() != 0 ||
        remote.masked_decode_calls() != 0 || result.large_prefill_compared != (reference != nullptr) ||
        result.large_prefill_compared_bytes != (reference ? 209715200ull : 0ull))
        fail("six-layer numerical coverage/counters mismatch");
    return result;
}

std::vector<int32_t> parse_layers(const std::string& text) {
    std::vector<int32_t> out;
    size_t at = 0;
    while (at < text.size()) {
        const size_t comma = text.find(',', at);
        const std::string item = text.substr(at, comma == std::string::npos ? text.size() - at : comma - at);
        int value = 0;
        if (item.empty()) fail("invalid --layer-ids list");
        for (char ch : item) {
            if (ch < '0' || ch > '9') fail("invalid --layer-ids list");
            value = value * 10 + (ch - '0');
            if (value > 999) fail("invalid --layer-ids list");
        }
        out.push_back(value);
        if (comma == std::string::npos) break;
        at = comma + 1;
        if (at == text.size()) fail("invalid --layer-ids list");
    }
    if (out.size() != 4 && out.size() != 6) fail("--layer-ids must contain exactly four or six IDs");
    if (!std::is_sorted(out.begin(), out.end()) || std::adjacent_find(out.begin(), out.end()) != out.end() ||
        out.front() < 0 || out.back() >= 48)
        fail("--layer-ids must be sorted, unique and in 0..47");
    return out;
}

std::string join_layers(const std::vector<int32_t>& layers) {
    std::string out;
    for (int32_t layer : layers) { if (!out.empty()) out += ','; out += std::to_string(layer); }
    return out;
}

int run(const std::string& pack, const std::string& native, const std::vector<int32_t>& layers, bool selected) {
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "query devices");
    if (devices < 2) fail("remote_experts_test requires two CUDA devices");
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(pack, 48, NE, err)) fail(err);
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.n_layers != 48 || lay.n_expert != NE) fail("test requires the 48-layer native pack");
    uint64_t payload = 0;
    for (int layer : layers) {
        if (layer < 0 || layer >= lay.n_layers) fail("selected test layer is out of range");
        payload += lay.blob_bytes(layer) * NE;
    }
    if (!selected && payload != 3250585600ull) fail("four-layer native payload no longer matches the guarded artifact");

    // Invalid selections must fail before device allocation or device switching.
    auto rejected_selection = [&](const std::vector<int32_t>& ids, const char* name) {
        strata::core::RemoteExperts rejected;
        err.clear();
        ck(cudaSetDevice(0), "select target before rejected layer selection");
        if (rejected.load(pack, ids, 1, err, native)) fail(std::string(name) + " unexpectedly loaded");
        require_device0(name);
        if (err.empty() || rejected.device() != -1 || rejected.layer_count() != 0 || !rejected.layer_ids().empty() ||
            rejected.payload_bytes() != 0 ||
            rejected.vram_bytes() != 0) fail(std::string(name) + " allocated or exposed resident state");
    };
    rejected_selection({22, 22}, "duplicate-layer selection");
    rejected_selection({28, 22}, "unsorted-layer selection");
    rejected_selection({-1}, "negative-layer selection");
    rejected_selection({48}, "out-of-range-layer selection");
    rejected_selection({0, 1, 2, 3, 4, 5, 6, 7, 8}, "nine-layer selection");

    ck(cudaSetDevice(0), "select target before disabled service");
    {
        strata::core::RemoteExperts disabled;
        if (!disabled.load(pack, 0, 1, err, native)) fail(err);
        if (disabled.layer_count() != 0 || disabled.payload_bytes() != 0 || disabled.vram_bytes() != 0 ||
            disabled.owns(0)) fail("zero-layer service is not strictly disabled");
    }
    require_device0("disabled service destruction");

    std::vector<float> qx((size_t) 4 * H);
    for (size_t i = 0; i < qx.size(); ++i) qx[i] = (float) ((int) ((i * 29 + 3) % 509) - 254) / 251.0f;
    const auto q0 = quant_on(0, qx, 4);
    const auto q1 = quant_on(1, qx, 4);
    if (q0 != q1) fail("GPU0/GPU1 q8_1 quantization bytes differ");

    ck(cudaSetDevice(0), "select target device");
    cudaStream_t target_stream = nullptr;
    ck(cudaStreamCreateWithFlags(&target_stream, cudaStreamNonBlocking), "create target stream");
    DeviceBuffer mixed, output;
    mixed.alloc(0, (size_t) kMaxT * H * sizeof(uint16_t));
    output.alloc(0, (size_t) kMaxT * kMaxK * H * sizeof(float));
    // Owners are deliberately sequential: four payload layers plus workspace are never duplicated on one GPU.
    FixtureResult gpu0 = run_service(0, pack, native, target_stream, (const uint16_t*) mixed.ptr,
                                     (float*) output.ptr, payload, layers, selected);
    require_device0("GPU0 service destruction");
    FixtureResult gpu1 = run_service(1, pack, native, target_stream, (const uint16_t*) mixed.ptr,
                                     (float*) output.ptr, payload, layers, selected);
    require_device0("GPU1 service destruction");
    for (size_t i = 0; i < gpu0.decode.size(); ++i) exact(gpu0.decode[i], gpu1.decode[i], "decode");
    for (size_t i = 0; i < gpu0.prefill.size(); ++i) exact(gpu0.prefill[i], gpu1.prefill[i], "prefill");
    ck(cudaSetDevice(0), "restore target before stream cleanup");
    ck(cudaStreamDestroy(target_stream), "destroy target stream");
    if (selected)
        std::printf("PASS remote_experts_selected layers=4 ids=%s payload_bytes=%llu owns=exact decode_calls=6 prefill_calls=5 blob_bytes=exact quant_bytes=exact outputs=exact\n",
                    join_layers(layers).c_str(), (unsigned long long) payload);
    else
        std::printf("PASS remote_experts layers=4 decode_calls=6 prefill_calls=5 blob_bytes=exact quant_bytes=exact outputs=exact\n");
    std::printf("PASS remote_experts_masked calls=4 entries=32 all_hit_calls=1 all_zero_bypass=exact mixed_rows=exact changing_masks=exact negatives=exact\n");
    std::printf("PASS hybrid_expert_dispatch devices=2 outputs=exact cold_restore=exact failures=exact\n");
    return 0;
}

int run_six(const std::string& pack, const std::string& native, const std::vector<int32_t>& layers) {
    static const std::vector<int32_t> required{22,24,28,30,33,34};
    if (layers != required) fail("six-layer fixture requires IDs 22,24,28,30,33,34");
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "query six-layer devices");
    if (devices < 2) fail("six-layer remote_experts_test requires two CUDA devices");
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(pack, 48, NE, err)) fail(err);
    const auto& lay = strata::kernels::cpu::expert_layout();
    uint64_t payload = 0;
    for (int layer : layers) payload += lay.blob_bytes(layer) * NE;
    if (!lay.native || lay.n_layers != 48 || lay.n_expert != NE || payload != 6448742400ull)
        fail("six-layer native geometry/payload differs from bound fixture");

    auto rejected = [&](const std::vector<int32_t>& ids, const char* name) {
        strata::core::RemoteExperts service;
        err.clear();
        ck(cudaSetDevice(0), "select target before rejected six-layer selection");
        if (service.load(pack, ids, 1, err, native)) fail(std::string(name) + " unexpectedly loaded");
        require_device0(name);
        if (err.empty() || service.device() != -1 || service.layer_count() != 0 ||
            !service.layer_ids().empty() || service.payload_bytes() != 0 || service.vram_bytes() != 0)
            fail(std::string(name) + " allocated or exposed service state");
    };
    rejected({22,24,28,30,33,33}, "duplicate six-layer selection");
    rejected({22,24,30,28,33,34}, "unsorted six-layer selection");
    rejected({22,24,28,30,33,48}, "out-of-range six-layer selection");
    rejected({0,1,2,3,4,5,6,7,8}, "nine-layer selection");

    auto run_owner = [&](int service_device, const FixtureResult* reference) {
        ck(cudaSetDevice(0), "select six-layer target device");
        cudaStream_t target_stream = nullptr;
        ck(cudaStreamCreateWithFlags(&target_stream, cudaStreamNonBlocking), "create six-layer target stream");
        FixtureResult result;
        {
            DeviceBuffer mixed, output;
            mixed.alloc(0, (size_t) kMaxT * H * sizeof(uint16_t));
            output.alloc(0, (size_t) kMaxT * kMaxK * H * sizeof(float));
            result = run_service_six(service_device, pack, native, target_stream,
                                     (const uint16_t*) mixed.ptr, (float*) output.ptr, layers,
                                     reference);
            require_device0("six-layer service destruction");
        }
        ck(cudaStreamDestroy(target_stream), "destroy six-layer target stream");
        return result;
    };
    FixtureResult gpu0 = run_owner(0, nullptr);
    ck(cudaSetDevice(0), "select completed reference device for reset");
    ck(cudaDeviceReset(), "release completed GPU0 reference context");
    FixtureResult gpu1 = run_owner(1, &gpu0);
    if (gpu0.quant.size() != 6 || gpu0.decode.size() != 6 || gpu0.prefill.size() != 7 ||
        gpu1.quant.size() != 6 || gpu1.decode.size() != 6 || gpu1.prefill.size() != 7)
        fail("six-layer comparison extent mismatch");
    for (size_t i = 0; i < gpu0.quant.size(); ++i)
        if (gpu0.quant[i] != gpu1.quant[i]) fail("six-layer target quantization bytes differ");
    for (size_t i = 0; i < gpu0.decode.size(); ++i) exact(gpu0.decode[i], gpu1.decode[i], "six-layer decode");
    if (gpu0.prefill.back().size() * sizeof(float) != 209715200ull || gpu0.large_prefill_compared ||
        gpu0.large_prefill_compared_bytes != 0 || !gpu1.large_prefill_compared ||
        gpu1.large_prefill_compared_bytes != 209715200ull ||
        !gpu1.prefill.back().empty()) fail("six-layer chunked comparison proof mismatch");
    for (size_t i = 0; i < 6; ++i) exact(gpu0.prefill[i], gpu1.prefill[i], "six-layer prefill");
    require_device0("six-layer final owner cleanup");
    std::printf("PASS remote_experts_selected6 layers=6 ids=22,24,28,30,33,34 payload_bytes=6448742400 allocation_bytes=6949138728 blob_cases=12 decode_calls=6 prefill_calls=7 blob_bytes=exact quant_bytes=exact outputs=exact layer_coverage=all mode=original\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string pack, native, layer_text;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--pack" && i + 1 < argc) pack = argv[++i];
            else if (arg == "--native" && i + 1 < argc) native = argv[++i];
            else if (arg == "--layer-ids" && i + 1 < argc) layer_text = argv[++i];
            else fail("usage: remote_experts_test --pack PATH --native GGUF [--layer-ids IDS]");
        }
        if (pack.empty() || native.empty())
            fail("usage: remote_experts_test --pack PATH --native GGUF [--layer-ids IDS]");
        const bool selected = !layer_text.empty();
        const std::vector<int32_t> layers = selected ? parse_layers(layer_text) : std::vector<int32_t>{0, 1, 2, 3};
        return layers.size() == 6 ? run_six(pack, native, layers) : run(pack, native, layers, selected);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL remote_experts_test: %s\n", e.what());
        return 1;
    }
}
