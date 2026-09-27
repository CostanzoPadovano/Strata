# Strata per Core Ultra 7 265KF / 2× RTX 5060 Ti 16 GB

Experimental implementation, separate from the existing ISTA launcher. No measured speed or promotion is implied by a build.

- Strata source `6da1f667e86558b152ab128edf3ebf77a80a9e57`; ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`; CUDA13.3.73/sm120. AVX2 fallback on this non-AVX512 CPU.
- Reuse the two installed ISTA IQ3_XXS GGUF shards, SHA256 verification before admission. No duplicate full checkpoint or expert pack.
- GPU0: dense model, MTP, INT8 KV, fixed expert-cache budget of2304×largest-blob bytes (about5GiB, variable-sized slots). GPU1: fixed whole expert layers0–15,11.426GiB, no corresponding host copy. CPU: profile-ranked compact anonymous hot set≤14GiB, excluding GPU0 cache payloads, VirtualLock with explicit backing report and no huge CUDA registration. About9.54GiB of expert payloads are not resident in RAM/VRAM. SSD fallback additionally includes the GPU0 cached payloads when lent to prefill. It uses positioned, unbuffered reads, one in-flight request, aligned scratch≤largest-blob+8192bytes, and80fixed cold slots. No runtime cache growth. GPU0 cache excludes GPU1 layers; adaptation and PCIe miss sharing are disabled.
- Secondary-lane activations/results transfer through fixed small buffers. Prefill uses its existing eight-slot staging ring and CUDA cross-device copies; no peer-access/NVLink claim. The direct, positioned PLE reader has32 in-flight requests and262144 cache rows. Expert mmap is forbidden.
- CPU:8workers within12-core Job affinity, leaving8cores outside. This is not a global CPU utilization cap.
- Windows guardian: Job kill-on-close,48GiB job committed-memory cap, exclusive existing model mutex, actual availableRAM≥12GiB, availablecommit≥16GiB, freeVRAM≥2GiB/device, temperatures<80C. No trims, OS/pagefile changes, app closures or waived floors. Startup RAM requires calculated host payload+4GiB headroom+12GiB reserve. Startup commit requires host+24GiB estimated GPU/driver/MTP/cache/IO overhead+16GiB reserve, and14GiB free/device. The24GiB figure is an estimate informed by the first load's measured~16GiB pre-MTP/cache Job overhead, not a guarantee.
- Loopback API port8035; authenticated dedicated WSL bridge if validated, no changes to global Pi settings. At most4HTTP handlers,8MiB body,60s socket timeout,16384output tokens,1024queued engine token lines.

## Admission ladder

1. Build, bounded synthetic component tests on each GPU, independent scalar/dequant float expert oracle, cross-GPU bit identity, duplicate routes, changed activations, T1/2/3/4/8, compact host bytes and prefill transfers. Mock HTTP tooling/streaming/auth/bounds and actual xhigh template.
2. Shard hashes, header geometry/offset validation from existing pack; MTP fixed-revision ranges/hashes; source/runtime manifest; preflight.
3. Guarded4096capacity load-only then semantic/tool smoke; collect actual RAM/commit/VRAM/temperature and exit cause.
4. Same-server Pi multi-tool/multi-turn checks and diversified prompt/decode repeats. Explicitly record no prefix reuse in original Strata; do not pretend this is equivalent to llama.cpp slot persistence.
5. Larger contexts only after preceding tier passes and recorded admission.98K/128K stability and a speed improvement remain unproven until measured.

GPU1 ownership changes which expert evaluations use CUDA Q8_1 rather than the CPU activation quantizer. Float-oracle tolerance is2% relativeL1 in synthetic tests; cross-GPU kernel results must be bit-identical. This does not prove full-model equivalence to BF16 or llama.cpp. MTP is speculative verification, not a license to ignore correctness. Published5070 results are not measurements of this PC.

## Recorded failed approaches (not superseded evidence)

- `runs/load4k-first`: portable CUDA registration of the full28.54GiB CPU arena; GPU1 reports OOM before its first completed layer. Job cleanup frees the GPUs. No inference.
- `runs/load4k-bounded-pin`: no giant CUDA registration; all28.54GiB CPU pages VirtualLocked, GPU1 loading succeeds. Guardian stops at15.15GiB availablecommit during CPU layer loading; pre-MTP/cache Job peak44.716GiB. This does not prove external-process causation: WDDM/system commitment differs from Job accounting.
- The20GiB hot set/direct-read candidate passed4K load and8HTTP smoke checks but measured only~8decode/~108prefill token/s. Adding GPU1 CUDA graphs did not yield an appreciable improvement in its subsequent screen.
- The current14GiB hot set/cache5GiB candidate passed fresh synthetic gates and4K load/8HTTP smoke checks. Its operational-only4K admission is bound to the current binary/manifest; it does not establish quality equivalence or a speed improvement.
- `runs/smoke32k-pi`:32K-capacity loading and the first6short requests complete; the guardian stops during the first3107-token prefill at15.68GiB available commit. Peak Job commitment remains about37.65GiB at the final samples, while system commitment grows. That discrepancy does not establish an external-process cause. The actual30K-token request and Pi agent loop were not reached; no32K admission.
