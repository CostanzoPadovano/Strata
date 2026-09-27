# Separate CPU vision integration — September27

User requests the omitted vision in Desktop Strata and the SAME global WSL Pi
ISTA profile. Parent frozen static-owner text source/build/guards remain intact.
New source snapshot changes only the dedup/vision compatibility exclusion and
bounded GENI/SVE parsing; no model, expert placement/cache kernel or MTP changes.
Inherited numerical-cache warning remains; functionality is not equivalence.

Baseline: text engine59ef6d14, static GPU0exact3500 slots, GPU1twelve immutable
layers, pin16384MiB, prefill512/spec4/MTP0, context98304. Vision factor: CPU-only
official encoder98ffde98, existing Q8 projector616703104bytes/b2e9b5e4,
8threads,1024image tokens. New native M-RoPE position table ~1.126MiB per
host/GPU at98K is arithmetic, not measured overall memory.

Stopping/acceptance: native synthetic parser/position/ownership/component
gates pass with fresh binary binding; eleven Python input/SVE/context/lifecycle tests;
encoder-only owned4GiB Job warms/encodes1024tokens, finite2560-wide exact SVE,
contrasting images differ and repeatcache hash matches. Fresh4Kload then
two-image exact OCR (ORCHID-7314 vs COBALT-4428, no code in prompt), ordinary
text42 after GENI; repeat at98304 configured capacity. Actual global Pi0.87.1
must read both image files via tool, preserve tool-text marker, use same
provider/model, return both visual codes, normal stop, with authenticated image
capability. Pi uses a new red fixture EMBER-9056, absent from startup cache,
to exercise cold-start on demand. No synthetic95K rerun or new speed/
organic-long-vision claims.

Measure encoder/Job/private/working-set and actual global RAM/commit separately;
native timings report actual expanded prompt counts. Model weights remain
75.84GB split GGUF; no rehash/download. Existing floors/temperature/kill-on-close
remain, CPU vision adds conservative2GiB startup reservation until measured.
Encoder is lazily started only on an uncached image; after combined SVE files
are prepared, QUIT/wait/join/close releases the encoder's committed working
buffers. Cache hits do not restart it. Cold start requires6GiB free commit and
10GiB free physical RAM, including unchanged4/8GiB floors. Refusal is per-request,
not a latched encoder failure; text/cache continue. Resident-encoder global Pi
runs01 and02 stopped at3.946/3.980GiB free commit, without GPU OOM. Retain both
failed measurements; on-demand is a new intervention, not proof of stability.
URLs/arbitrary files forbidden, bounded images/pixels/cache/output/body/queues.
Do not install a new provider or alter global models/settings/sessions.

Completed final gates: encoder03,4Kload04/smoke02,98Kload02/smoke02 and actual
globalPi03 all pass. Pi reads uncachedEMBER-9056, cachedCOBALT-4428 and text
ORCHID-7314, exactJSON/finalstop; native/encoder/observer/guardian exits0.
DesktopBAT--check passes; global models/settings bytes and frozen text binary
unchanged. A scoped generated-tensor/build ignore follows observed CodexGit
diff proliferation; no files/apps removed. Initial failed runs are retained.
Evidence: wiki/raw/2026-09-27-strata-vision-global-pi-final.md. Configured98K with
short images is not an organic98K image-history/4h soak or numerical promotion.
