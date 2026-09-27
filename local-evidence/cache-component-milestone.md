# Strata cache/idle update: component milestone, 2026-09-27

User: "perfetto quindi possiamo passare aall'aggiornamento?"
Working F107 build, normal BAT, global Pi and published fork remain unchanged.
Separate candidate: research/qwen-strata-update-20260927.

Official latest GitHub API response: v0.1.8, published2026-09-27T10:26:14Z,
89ba2dc54daf3210b052e542605445a55953a1b4. Selective v0.1.3 feature port,
54cf7b6f0700f4442ffa8bc8b5af257d5db91cf4, not whole latest-rebase.
Copied frozen working source, normalized touched CRLF/LF for mechanical merge,
resolved four generate.cpp/one mtp.hpp conflict while preserving dedup-aware
refill, bounded vision parser, selected12 and MTP transport/telemetry.

Read-only architecture review found MTP boundary and pool epoch completion
hazards; candidate saves40KiB target residual per prefix, repairs final MTP cell,
populates complete MTP prefix, catches up terminal rows and acknowledges every
CPU phase. Cache capped6, additional reservation1GiB, KV remains GPU, short-read0.
No host-KV/Q4/sampling/prompt-lookup port.

Build Jobs2 passed; candidate SHA256
7d406a5478968bd346b67edc580df379a3cbcc20b432d7d489b384b69439707d.
Fresh native-gates.json binds nine small component gates plus real MTP and
selected12 numerical gates. Cache state roundtrip passed on both GPUs at a
non-block-aligned257-token prefix; changed image pixels/grid rejected. Pool:
180 native AVX2 burst batches each host_works false/true, idle CPU0.000ms per
250ms sample, idle teardown8 times. Not a model throughput benchmark.

First pool fixture wrongly required canonical AVX512 on this CPU; failed before
work. Replaced only test with actual native/GGML AVX2 path; fresh tagged rerun
passed. Initial Python execution hit sandbox Temp permissions, then Windows
paths under bash; separate native Windows34tests/WSL8tests now pass. Installed
Pi0.87.1 in-memory same-ID text/vision overlay also passed; other model/config
bytes unchanged. Source-gates and serve-only offline derivative created, no
model/encoder launched yet. Full-model/cache/Pi live admission still pending.

Ignore only this duplicate source/upstream/build snapshot to avoid1464-file
background Git diffs. Keep source-only reproducible patch, scripts and numeric
evidence visible. No commit/push/PR or source/BAT/Pi promotion.
