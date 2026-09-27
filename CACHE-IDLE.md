# Frozen working Windows cache/idle build — September 27, 2026

The owner reports that the build works very well during normal use, **including
Pi `/compact`**, and requests no further tuning/tests. This is a user-reported
operational confirmation, not a new instrumented speed, RAM or context benchmark.
Publication changes only the recovery archive and documentation. Active BAT,
Pi integration, model, engine binary and safety limits remain unchanged.

Tag: `local-dual5060ti-cache-idle-20260927`.
Native SHA256: `2e630a0b125a0b4f2be300678216d8aca2e086d1e5c2d9cb54855b55685c4c77`.
Source patch SHA256: `3b116921455ac861152ad4d057647a06c54b5c3f0e937d3a28f2a87e142e95da`.
Original base: `6da1f667e86558b152ab128edf3ebf77a80a9e57` (v0.1.2).
Selective v0.1.3 cache/CPU-idle backport, **not the full v0.1.8 release**.
Original author and upstream: [Niko1221/Strata](https://github.com/Niko1221/Strata).

## Preserved setup

Hardware: Intel Core Ultra 7 265KF, two RTX 5060 Ti 16-GB GPUs, 64-GiB DDR5
configured at 4800 MT/s, Windows 11 Pro; CUDA 13.3.73 / SM120 / MSVC 19.51.
See [LOCAL-PC.md](LOCAL-PC.md) for exact inventory and historical workload lanes.
ISTA-DASLab IQ3_XXS checkpoint and quantization unchanged; no model weights here.

- GPU0: exact3500 expert cache with host dedup, MTP/spec4, dense projections/KV.
  GPU1: static expert ownership of layers 28,29,30,32,34,35,37,42,43,44,46,47.
- Context capacity 98,304. Output uses remaining rendered context minus 8 tokens,
  including thinking; no artificial 1,024/4,096/8,192 output cap.
- Conversation cache: at most six host recurrent-state/PLE/QSA checkpoints,
  interval 16,384, short-read 0. Target/MTP KV stays on GPU. Cache is volatile,
  one active conversation, not a persistent full-KV RAM cache for every chat.
  Compaction or incompatible prefixes may require partial/full prefill again.
- MTP boundary repair and terminal speculative-commit bounds; CPU workers park
  with epoch acknowledgements. Existing dual-GPU ownership/vision are retained.
- CPU vision on demand. Same Desktop `1d` BAT: server only, **zero startup tests**,
  native prefill/decode/cache console timings. Debug `1e` uses the same engine.
- Global WSL Pi: `local-qwen38/qwen3.8-flash-next-local`, xhigh default; other
  providers and server-off startup remain independent. No global settings/auth
  file or conversation is exported. Authenticated private bridge, no LAN API.
- Guards unchanged: RAM free >=8 GiB, commit free >=4 GiB, Job <=60 GiB,
  GPU <80 C, finite 4-hour lifetime, queues, mutex and kill-on-close. Free-VRAM
  watchdog floor remains 0; conservative additional cache reservation 1 GiB.
  Commit is not resident RAM or SSD-write volume. No Windows/pagefile changes.

## Already completed qualification (not repeated for publication)

Native components, bounded MTP/selected12 numerical checks, Python/WSL adapter
checks, CPU encoder, fresh 4K/98K load/text/vision/cache passed. Finite greedy
cache/fresh outputs match in checkpoint/live/EOS/length/cancellation/rewind cases.
Longest real prompt: 34,817 tokens, not a populated 98K history certificate.
The long-to-short 16,417-token case reused 16,384 tokens, with identical output:
66,861.0 ms cold prefill vs 715.1 ms warm. This is a cache-latency comparison,
not evidence that all 16K tokens were recomputed at the warm apparent rate.

Two successful actual Pi image/file tool requests are archived as
[numeric-only evidence](local-evidence/cache-pi-trial.json), with endpoint/native
identity attestation and unchanged model/settings checks. They used warm cache
and retries: not a matched benchmark against F107. Decode was 39.93 and 40.44
tok/s; processed prefill excludes reused tokens. Historical 188.609/38.138
and separate 346.080/50.920 figures in LOCAL-PC.md belong to older workloads.

Codex stayed open during qualification; build/Git/archive jobs were not run
alongside those measurements. Pi run: free physical RAM min 22.037 GiB,
free commit min 4.572 GiB, owned Job peak 52.480 GiB; no resource-stop event.
SSD/pagefile writes were not measured. User compaction success adds an
operational observation, not organic 98K/4-hour/model-quality certification.
The experimental upstream cache/numerical quality caveat remains disclosed.

## Recovery and frozen-state policy

[RESTORE.md](RESTORE.md) describes dependencies, paths, model hashes, fresh-gate
requirements and rollback. Exact candidate source, scripts, tests, configs,
launcher backups, patch and proof bindings live under
`deployment-snapshot/research/qwen-strata-update-20260927/`.
The tagged release saves exact native/test binaries and selected proof closure
in `windows-cache-runtime.zip`, with a SHA256 and per-member manifest.
No GGUF, packed tensors, projector weights, credentials or conversations.
Archive integrity is verified; a new-machine restore is **not** live-tested.

The earlier F107 tag/release is preserved. Freeze the current working setup.
Any future upstream update is a separate, user-authorized, qualified change;
no automatic updater, recurring monitor or pending runtime modification.
