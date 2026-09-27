# Strata official updates and cross-request cache audit — 2026-09-27

## Scope and outcome

Read-only investigation requested by the user: identify official Strata updates
useful to this PC and determine whether the active local build reuses context
between Pi requests. No native model, vision encoder, or benchmark was started;
no launcher, runtime, guard, global Pi setting, or GitHub artifact was changed.

Latest release verified with the live GitHub API is **v0.1.8**, published
`2026-09-27T10:26:14Z`. Its tag resolves to
`89ba2dc54daf3210b052e542605445a55953a1b4`. API comparison against the local
upstream base `6da1f667e86558b152ab128edf3ebf77a80a9e57` shows 35 commits ahead.
Cached web release listings were older; they were not used to declare latest.

## Active local boundary: no cross-request context reuse

- Base is v0.1.2 with local dual-GPU/host-compaction/vision adaptations.
- Native executable: `research/qwen-strata-vision-20260927/build/strata.exe`.
- Observed SHA256 remains
  `f1072687e794aa5958f121c8967983947b5ea1322fa336c60688cfdc7736742f`.
- In `research/qwen-strata-vision-20260927/source/src/program/generate.cpp`,
  lines 2275–2277 explicitly describe an empty sequence per request, and line
  2560 unconditionally calls `session_zero(ss, g, nullptr, main_cs)`.
- Local `serve/server.py` sends the complete rendered token sequence in each
  `GEN` request. Native `DONE` does not report reused prefix tokens.

Attention KV and recurrent state exist within a generation, and model/expert
storage remains resident between requests. The missing feature is reuse of that
conversation state across API requests, including requests after a Pi tool
result. Resetting means full prompt processing again; it does **not** mean a full
model reload from disk or a full prompt pass for each generated token.

## Official changes relevant to this deployment

### v0.1.3: conversation reuse — first priority

[Release](https://github.com/Niko1221/Strata/releases/tag/v0.1.3) introduces
continuation from a compatible live token prefix or the longest matching saved
checkpoint. Default `--prompt-cache 6` keeps up to six checkpoints;
`--prompt-cache 0` disables it. Author reports about 118 MB host RAM per checkpoint
(about 708 MB for six), **not a measurement on this PC**. Author's 7.7K follow-up
example falls from 12.9 s to 0.3 s; this is not a forecast for our 98K setup.

The [v0.1.8 source](https://github.com/Niko1221/Strata/blob/v0.1.8/src/program/generate.cpp)
confirms CPU byte vectors store the 36 GDN recurrent/convolution states, PLE
history and QSA indexer tails. Positional KV stays in its existing storage; it is
not copied in full per checkpoint. Default periodic checkpoint spacing is 16,384
fresh prompt tokens. Images participate in prefix identity via content keys.

This is finite, in-process prefix/state reuse, not a durable multi-session
archive. History changes, incompatible conversations or compaction may invalidate
reuse or require processing from an earlier checkpoint; restart loses live state.
The update also lets idle CPU expert workers sleep rather than spin. The local
CPU pool still uses spinning waits.

### v0.1.5/v0.1.6: host KV streaming — separate RAM/VRAM tradeoff

[v0.1.5](https://github.com/Niko1221/Strata/releases/tag/v0.1.5) can back QSA KV
with pinned system RAM and retain only an attention working window in VRAM
(`--kv-resident`). It frees VRAM but consumes more host/pinned RAM and PCIe
traffic. [v0.1.6](https://github.com/Niko1221/Strata/releases/tag/v0.1.6) fixes an
MTP startup failure when pinned memory is unavailable by retaining draft KV in
VRAM. Neither feature is necessary for conversation prefix reuse. Given earlier
desktop responsiveness problems, do not automatically enable host KV here.

### v0.1.7/v0.1.8: additional useful options

- [v0.1.7](https://github.com/Niko1221/Strata/releases/tag/v0.1.7): prompt-lookup
  drafts for repeated code, request sampling controls, and removal of the old
  1,024-token default cap. Our own remaining-context output policy already
  supersedes that cap; prompt lookup is not the conversation checkpoint cache.
- [v0.1.8](https://github.com/Niko1221/Strata/releases/tag/v0.1.8): Web UI monitor
  and `/metrics`, plus optional Hadamard Q4 KV. Author reports 0.83 versus
  1.52 GiB KV at 128K, with a documented perplexity tradeoff; default 8-bit KV is
  unchanged. Neither the memory saving nor quality effect has been tested here.
- v0.1.8 also replaces the previous expert-cache quality warning with a reported
  2,557-token cache-on/off comparison. This narrows upstream's previous claim;
  it does not certify our modified dual-GPU cache or agent/scientific quality.

## Integration constraints and recommendation

Literal inspection of official v0.1.8 finds none of our custom flags
`--remote-expert-layer-ids`, `--gpu-host-dedup`, or `--expert-cache-exact`.
Replacing the executable directly would lose required local functionality and
break hash-bound launch admission. Official model storage also does not establish
the same compact host expert payload as our local 22.182 GiB layout.

Recommended next implementation, **not performed or newly authorized by this
audit**: port conversation checkpoints and idle-worker sleeping into the local
dual-GPU fork, keep current 8-bit KV placement initially, then qualify numerical
state restoration, image identity, cancellation, Pi tool turns, queue/resource
bounds and memory use. Update native protocol parsing and timing accounting for
`RESUME`/`REUSED`/new `DONE` fields. Never divide the complete cached prompt token
count by suffix-only processing time and label that as fresh prefill throughput.

For comparison, [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
describes common-prefix KV reuse through `cache_prompt`. KV location can be GPU
or host depending on offload configuration; "cache in RAM" and "reuse previous
context" are separate properties there too.

## Evidence acquisition

GitHub API reads: `releases/latest`, `releases?per_page=5`, tag v0.1.8, comparison
of local base to main, commits, and `contents/src/program/generate.cpp?ref=v0.1.8`.
The official C++ source was decoded in memory and inspected without a checkout
or local upstream-file write. Local source, adapter, CPU pool and native SHA256
were inspected. No new performance, memory or correctness test of upstream ran.
