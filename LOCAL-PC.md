# Windows dual-RTX-5060-Ti experiment (2026-09-27)

This is an **experimental local fork**, not a new Qwen model or a claim of
upstream performance/quality equivalence. Native source is based on Strata
`6da1f667e86558b152ab128edf3ebf77a80a9e57`. The root source is the tested local
cache/idle/vision/static-owner snapshot with a selective v0.1.3 backport, not a
full v0.1.8 update. See [CACHE-IDLE.md](CACHE-IDLE.md) for current evidence and
the owner's successful manual `/compact` report. Original author: [Niko1221/Strata](https://github.com/Niko1221/Strata).

## Tested hardware and software

| Component | Measured/inventoried configuration |
| --- | --- |
| CPU | Intel Core Ultra 7 265KF, 20 cores / 20 logical processors |
| Motherboard | ASUS PRIME Z890-P WIFI |
| System memory | 64 GiB, 4 x 16 GiB DDR5, configured 4800 MT/s (Windows CIM) |
| GPUs | 2 x NVIDIA GeForce RTX 5060 Ti, advertised 16 GB each; driver reports 16311 MiB each |
| Driver / OS | NVIDIA 610.88; Windows 11 Pro, 10.0.26200 |
| Build | CUDA 13.3.73, SM120, MSVC 14.51.36231, C++20 |
| Client | Pi 0.87.1, Node 22.23.0, Ubuntu-24.04 under WSL2 |
| Model | ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, two GGUF shards |
| Frozen cache/vision executable SHA256 | `2e630a0b125a0b4f2be300678216d8aca2e086d1e5c2d9cb54855b55685c4c77` |
| llama.cpp/ggml dependency | `3cf03257f219afbe7334045ff7c6a06ac68c627d` |

PCIe is asymmetric: GPU0 reported Gen5/x8, GPU1 Gen4/x4; CUDA peer access was
unavailable both ways. This is not a unified 96-GB memory pool.

## Results: different workloads, not interchangeable

| Workload | Prefill tok/s | Decode tok/s | Interpretation |
| --- | ---: | ---: | --- |
| Real manual Pi, xhigh, 10 requests, prompts 3394-20639 tokens | **188.609** | **38.138** | Weighted native throughput, thinking included |
| Synthetic occupied 94601-95738-token task, 4 long turns | 177.885-178.800 | 33.067-43.911 | Functional tool/edit/retrieval test; short outputs, not a speed benchmark |
| Separate experimental 4K six-layer dual-GPU profile | 346.080 | 50.920 | Pooled per-request means over two paired repetitions; NOT the daily 98K profile |
| Matched single-GPU control for that 4K experiment | 338.537 | 48.131 | +2.228% prefill / +5.796% decode for the paired experiment; no significance claim |

Manual trial formula: `sum(tokens) * 1000 / sum(native milliseconds)`.
Prefill: 122987 / 652.0731 s. Decode: 4261 / 111.7253 s.
[Numeric-only request rows](local-evidence/pi-trial.json) exclude the four
startup probes. No prompt text, request bodies or Pi sessions are published.
That historical F107 trial predates conversation caching: entire prompts were
processed again each turn. The new bounded cache has separate evidence in
[CACHE-IDLE.md](CACHE-IDLE.md); these rates are not its benchmark.
Native times exclude cold model loading, image encoding, tools and client overhead.

The manual trial lasted about 24 minutes: sampled Strata physical working set
peaked at 22.760 GiB, total Windows physical RAM at 40.753 GiB, Windows commit
at 73.204 / 80.530 GiB, Job commit at 52.304 GiB, and current pagefile usage
was 0.395-0.423 GiB. GPU peaks: 14.862 / 13.366 GiB, temperatures 65 / 53 C.
Commit is not resident RAM or SSD bytes written. Sampling can miss peaks;
WSL VM and WDDM shared-memory attribution were incomplete. The outer launcher's
final result was missing: **clean exit is not certified for this manual trial**.

One of ten responses reached the historical 1024-output cap (`finish=length`).
The current adapter removes that artificial cap and budgets
`98304 - exact rendered prompt tokens - 8`, including thinking and images.
Actual generation beyond 1024 tokens under the new policy has **not** been
measured. Smaller explicit client limits remain respected.

The separate text-only 95K synthetic campaign completed tool edits, a webapp,
browser checks, real compaction and follow-up with native exit 0 (~56 minutes).
It used earlier text engine `59ef6d14...`, not the current vision build, and
synthetically inserted history rather than organic long-running agent work.
Neither configured 98304 capacity nor that test certifies a 98K organic vision
session, four-hour stability, model quality or scientific validity.

## Local changes and safety

- GPU0: main projections, MTP, KV and bounded expert cache. GPU1: static
  original expert ownership at layers `28,29,30,32,34,35,37,42,43,44,46,47`.
- Compact CPU expert payload: 23817318400 bytes (22.182 GiB), separate from
  total process RAM, Windows commit and the GGUF's disk size.
- Final normal BAT: serve-only, no arithmetic/OCR startup requests, CPU vision
  encoder starts only for images, authentic per-request prefill/decode display.
- Same global Pi identity `local-qwen38/qwen3.8-flash-next-local`; optional
  authenticated runtime overlay. Pi still opens when Strata is off and supports
  other providers. Private WSL bridge only; no unauthenticated LAN listener.
- Retained limits: free RAM >=8 GiB, free commit >=4 GiB, owned Job <=60 GiB,
  GPU temperature <80 C, finite queues/body/cache/lifetime (4h). No free-VRAM
  watchdog floor. Internal 700-MiB cache sizing reserve is not a stop threshold.
- Fresh images additionally need 10 GiB RAM / 6 GiB commit free, including
  normal floors. Enter requests orderly stop; closing the owner kills its Job.

The original F107 serve-only publication had offline checks only after removing
startup probes. The current cache build subsequently passed fresh components,
4K/98K load/cache/vision and actual Pi checks; the owner then reported successful
normal use including compaction. This supersedes that startup-only limitation,
not the scientific/quality/organic-98K limitations below.

**Open risk:** upstream/cache-on versus CPU-miss numerical divergence is not
fully retired. Kernel tests and functional outputs do not prove full-model
quality equivalence. Do not treat this fork as production-ready or a validated
bioinformatics inference engine.

## Archive and restore

Read [RESTORE.md](RESTORE.md). `deployment-snapshot/` preserves exact local
source, guards, tests, BATs, configs and historical proof metadata, including
the other experimental lanes. Root source is the final vision lane; older
performance results must not be attributed to it. Upstream install scripts and
upstream speed claims below the README banner describe the original engine,
not an automatic installation of this local profile.

Patches do not change the checkpoint. Attribution, upstream notices and
dependency licenses are retained; no new licensing claim is made for third-
party code or model weights. Model files, raw prompts, credentials and virtual
environments are intentionally excluded from GitHub.
