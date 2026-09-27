# Strata agent >=90K guarded protocol

Goal: establish a working WSL2 Pi agent at context 98,304 with the isolated
Strata engine, minimizing RAM and using at most two GPUs. Speed is not an
acceptance criterion.

## Fixed profile

- Context ladder: `4096, 8192, 16384, 32768, 65536, 98304`; no skipped tier.
- Remote Original layers on GPU1: `28,29,30,32,34,35,37,42,43,44,46,47`.
- Expected remote payload/allocation: `13,526,630,400 / 14,027,026,728` bytes.
- GPU0: exact full cache (initially 3500 slots), adaptation off, no prefill
  borrowing, host copies deduplicated; MTP GPU0, spec 2..4, pin 16384 MiB,
  prefill chunk 512, PCIe fraction 0.0. No mmap or hybrid dispatch.
- Hard guards: Job 60 GiB with kill-on-close, RAM free >=8 GiB, available
  commit >=4 GiB, GPU temperature <80 C, zero VRAM floors, global mutex,
  finite timeout, and post-READY resource sampling.

## Admission order

1. Build with `powershell -File .\build.ps1`; the binding freezes all source
   files, nine test fixtures, executables, CUDA/SM and ggml revision.
2. Run `python .\prepare_evidence.py --tag TAG`. This performs fresh transport,
   registration, embedding and compacted-loader tests. It does not load a GGUF.
3. Run `python .\numerical_guard.py mtp --tag TAG`, then
   `python .\numerical_guard.py remote12 --tag TAG`. Both are real native tests;
   the latter compares all 12 selected layers and a 2048x10 prefill in 16 MiB
   chunks under a finite <=24 GiB numerical Job.
4. For each tier, create/validate the exact config from the admitted 4K template
   with `python .\prepare_evidence.py --config-only --context N`.
   Run a fresh
   `agent_guard.py load` with that config. Only then run smoke/agent validation.
   A tier above 4096 also requires the immediately previous tier admission.
5. At 98,304, exercise actual isolated Pi in WSL2 with real file/tool/browser
   work. Record original-prefill cost, resource minima/maxima and clean exit.

Never describe an unrun gate as passed. Old 18-component results are inherited
baseline evidence only. Size/mtime revalidation may reuse the prior full hashes
of the two 75.8 GB GGUF shards; do not download or rehash them. Report compact
host arena arithmetic as a lower bound and global commit as a separate measured
counter, never as inferred savings.

No OS/pagefile/app closure/global-Pi/ISTA changes and no commit/push/PR are in
scope. A resource stop fails the tier; it is not permission to relax a floor.
