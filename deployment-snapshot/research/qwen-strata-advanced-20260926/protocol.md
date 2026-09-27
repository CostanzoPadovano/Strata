# Advanced dual-GPU Strata experiment — 2026-09-26

## Authorization and boundary

The user explicitly requests designing, implementing and testing a two-GPU
engine, authorizes experimental strategies, and asks to stop and document once
an equal or better significant result is demonstrated. The active goal records
the original request. Work is confined to this isolated campaign; the official
Strata control, historical prototype, ISTA/Pi, Windows/pagefile, drivers, clocks,
other applications and existing checkpoints must not be changed or closed.

Pinned upstream: `6da1f667e86558b152ab128edf3ebf77a80a9e57`. Existing ISTA
IQ3_XXS GGUF, pack and MTP artifacts are reused by hash, not downloaded again.
The clean clone is the implementation boundary. No commit/push/PR is authorized.

## Measured starting point and topology

Official GPU1/4K/8-worker stock screen: decode mean33.486tok/s (27.333,
36.715,36.411); prefill296.651tok/s on3107 actual prompt tokens. These are
small screens, not128K benchmarks or validated code quality. First measure the
same official executable onGPU0; do not attribute a faster PCIe slot to dualGPU.

CUDA peer-access is0 in both directions. GPU0 reports PCIe5/maxx8;
GPU1 PCIe4/currentx4 at idle on this PRIME Z890-P WIFI. Loaded bandwidth is
not measured. No direct peer pointer access is permitted. The old prototype
contains nonresident SSD experts and is not a useful speed control.

## Candidate architecture

First candidate: GPU0 target verification/dense/KV/prefill and its ordinary
bounded expert cache; GPU1 resident MTP draft weights/KV/head. Only small
residual/embedding data crosses via explicitly bounded pinned host staging.
No expert SSD fallback, second full model, unified-VRAM claim, or cross-device
pointer embedded in a CUDA graph. Default single-GPU behavior must be retained.
Both GPUs must perform real inference, not merely reserve VRAM.

If this candidate does not match the control, record the failure and refine the
placement. Merely keeping two devices busy is not an acceptance criterion.
The first matched3500-slot candidate is3.78% lower decode/1.46% lower prefill,
so it is not selected. Next opt-in `--mtp-resident-embedding` copies the native
embedding table to the selected drafter device (hard cap512MiB, current260MiB)
and gathers within its graphs, eliminating per-draft embedding host relays.
The same flag must be used for the local single control; defaults stay false.
No expert mmap or model re-quantization. Require new64 bounded synthetic native
IQ3 embedding mapped/resident graph bit checks before the full load.

Resident embedding second paired screen reaches dual46.928/324.100 versus
single48.279/333.794 tok/s; still not selected. Next opt-in
`--mtp-mapped-residuals` preserves the GPU0 working residual buffer and stages
only its completed snapshot to a fixed max_t*HCN portable mapped host buffer.
GPU1's graph copies this into its own Rin_ instead of queuing a separate H2D
copy-engine upload. The producer must be complete and the consumer stream
must finish before snapshot reuse. No peer pointer may enter a remote graph.
This is a transport hypothesis, not a demonstrated speed improvement.

The mapped candidate's matched pair remains slower:46.299decode/327.819prefill
versus single47.773/333.282. Isolated continuous MTP is equal-speed on both
GPUs; sparse40ms jobs reproduce GPU1-local slowdown even without transfers or
the main arena. Clock/WDDM causes are not individually measured. Next opt-in
`--mtp-chain-batch` requires resident embeddings, launches bounded round/chain
graphs on one stream and synchronizes once, then returns the original min-p
prefix. Extra speculative KV cells must stay masked until overwritten;
near-context-boundary execution retains the sequential path. Defaults remain
false. Extra unused work is a cost, not a speedup assumption.

The first sparse batch screen gives no clear gain. Next strategy is opt-in
`--remote-expert-layers 4`: MTP stays onGPU0, GPU1 owns four complete native
expert layers, and their exact3,250,585,600B prefix is physically excluded from
the anonymous host arena. Original GGUF gate/up/down roles are cold-loaded
through one32MiB staging buffer before READY; this pack has no experts.bin.
Only explicitly bound GGUF shards/metadata may be consumed. No runtime expert
file reads, mmap, implicit CPU/SSD fallback or duplicated full host payload.
GPU0 keeps a3500-slot cache of ranked non-remote pairs: capacity/workers/model
match the control, but the pair list necessarily differs and is reported.

Decode uses the existing exact FP32 doorbell mailbox. GPU1 recomputes the same
pure GPU activation quantizer, writes unweighted ordered expert vectors back,
and GPU0 retains combination. Empty GPU0 plans and flagA/B are published before
remote work; no CUDA0 copy may wait behind the blocked verifier graph. Prefill
preserves FP16 rows, grouping/expert order and full GEMM dimensions. Transfers,
not GEMMs, are chunked. One request, fixed<=2048token/10expert workspace and
payload+workspace<=8GiB service cap; full guardian limits unchanged.
Fresh8GiB numerical admission requires independent complete-blob GGUF byte
checks, exact GPU0/GPU1 quantized bytes,6 ordered decode/5 grouped prefill
fixtures including2048x10 multi-chunk output, negative bounds and device
restoration. This does not prove CPU/GPU upstream arithmetic identity.

The first four-layer matched screen passes all semantic/tool/table checks but
is slower (45.165decode/330.789prefill versus47.895/335.054). One of eight
output contents differs; exact full-generation equivalence is not claimed.
Before extending residency, use an isolated four-layer burst diagnostic with
64 measured bursts after warm-up, continuous and40ms inter-burst gap modes,
instrumentation on/off and sequential GPU0/GPU1 ownership. Actual float bytes
must match across modes/devices, not merely digest values. Reference outputs
are bounded at100MiB; fixed input/output mailboxes and four timing events are
diagnostic-only. Disabled profiling allocates no events and collects no clocks.
GPU event intervals include possible host submission gaps; CPU submission can
include implicit waits in pageable-memory CUDA calls. Uninstrumented active
wall time is the comparison metric; stage intervals are diagnostic, not pure
wire bandwidth or instruction time. No main host expert arena/PLE/generation;
verified8GiB Job,160s native/180s guardian, commit4GiB/80C/mutex remain.
This diagnostic cannot issue a full-model speed/quality admission.

Next opt-in `--remote-expert-mode packed|graphs` retains Original as default.
Packed sends one aligned41,984B pinned request packet rather than six small
uploads. Graphs additionally caches kernel-only quantizer/grouped-expert graphs
per owned layer/T1..4/K10:16 for prefix4, API maximum32 for prefix8. Upload,
download and optional timing events remain outside capture; smaller K uses
Packed fallback. One BusyGuard also protects the existing32MiB host stage.
Device buffers count against the8GiB cap; driver-internal graph storage is not
included in `vram_bytes()`, so that field is not total physical VRAM accounting.
No CUDA0 pointer enters those graphs. Selection happens before READY; successful
graph replay counters are logged per full-model request. Single control always
uses Original. Fresh numerical admission requires all24 conditions (two GPUs,
three modes, profile on/off, gap0/40), exact65,536,000B reference comparisons,
K4 fallback parity and256 actual graph replays per graph condition. Its bound
admission permits an optimized full-model trial, not a speed/quality promotion.

Next diagnostic `--verify-wait-profile` measures the verifier's existing A/B/CPU
flag waits on its owning GPU. Default-off preserves the original wait kernels
and allocates nothing. The sink is fixed48layers x2groups x3stages x24B=6912B,
selected after initialization and before graph capture. Snapshot/reset reject
active requests and require an idle stream; snapshots include only completed
verification windows. Counter initialization/reset is enqueued on the owning
nonblocking verifier stream and checked to completion before graph reuse;
default-stream memset ordering is not assumed. No CUDA0 API call is added
inside the CPU pool callback.
The already-ready path counts calls without timer reads; the blocked path uses
two ordered device-local `%globaltimer` reads, not a CPU/GPU timestamp subtraction.
This timer is target-specific; the tiny two-GPU gate checks exact mapped payloads,
ready/delayed/guard counters, original/profiled graph replays and a loose CUDA
event unit calibration before any full-model diagnostic.

The CLI emits48 ordered layer rows per completed request, with A/B/CPU call,
waited-call and blocked-nanosecond counters. Validate layer extent/stage call
consistency/completed window counts; zero or incomplete rows cannot rank layers.
Snapshot/reporting are after native decode timing, but profiled kernels themselves
perturb that timing. Require same-build diagnostic-off/on content/count checks
and measure overhead before interpreting wait share. Exposed CPU wait after
GPU cache/PCIe work is useful for placement; raw CPU compute duration can overlap
useful GPU work and is not the selection metric. An on-only diagnostic may guide
the next bounded strategy, never demonstrate a speedup or numerical equivalence
of the full model. Preserve fresh configuration/source/numerical/load bindings.

## Resource and process gates

Next opt-in `--remote-expert-hybrid` keeps the SAME ranked GPU0 cache pairs as
the single control, including selected-layer pairs. GPU0 executes its cached
hit groups while GPU1 computes only the missing routed entries. The bounded
40-entry plan validates all IDs, slot byte extents and original destination/
token indices before publishing; A/B are published before synchronous remote
work, and no CUDA0 call is added in the blocked verifier callback. Original
decode only; other modes are rejected explicitly. Sparse GPU1 output zeros
all active rows, and an all-hit call writes zero host rows without GPU1 launch.
Defaults and the existing whole-layer remote candidate remain unchanged.

Selected payloads remain physically excluded from the host arena. One cold
cache blob (max_blob, never above32MiB) may be read back from resident GPU1 for
startup, verification, adaptive refill and EVERY prefill-borrow/restore path.
The GPU0 copy completes before reusing that single stage; nonremote arena
copies retain their original asynchronous behavior. No runtime expert file
read/mmap/full host copy is introduced. The same3500-slot profile now implies
the same byte budget and pair list on GPU0, not merely equal slot counts.

New masked/service and actual dispatch fixtures must pass before full load:
all-hit/all-miss/mixed/changing masks, repeated experts/token groups, complete
bit-exact GPU0-hit/GPU1-miss outputs, actual moe_hit_add, verified cold restore,
invalid plan no-publish and remote-error release/zero rows. Evidence binds
the opt-in and matching IDs/payload/build; old complete-layer admissions do
not admit hybrid inference. Positive per-request hit/miss counters prove
device work division, not a measured overlap duration or general CPU equality.
Build3597b105/staged23/MTP17/Remote10 and both fresh4K loads now pass. First
matched full screen is slower:41.697decode/344.087prefill dual versus47.198/
337.215 single, -11.656%/+2.038%; no selection, historicalbest48.279055 kept.
New opt-in `--remote-expert-freeze-cache` freezes adaptation ONLY for owned
remote layers; initial cache pairs, other-layer adaptation, borrowing and
every fresh bounded restore remain unchanged. Candidate-stage exclusion in
both loops and a fail-closed refill backstop must prevent selected adaptive
transfers. Per-request adaptive/restore readbacks, payload bytes and complete
round-trip wall time are separate; wall time may overlap MTP, not unoverlapped
decode cost. Fresh source/numeric/load gates remain required. Completed
hybrid trials reject missing, repeated, inconsistent or nonzero frozen
adaptive counters. Fixed selected pairs may increase misses; do not assume
an improvement until matched repeated screens pass.
First complete frozen33ca616a pair fails selection:43.131/342.243 dual versus
46.344/335.501 single (-6.932%decode/+2.010%prefill), no stop/crash. All8
requests have zero selected adaptive transfers and83fresh remote restores,
but selected hit coverage is lower. Next bind the same binary to an
instrumented unfrozen comparison; renew numeric/load gates after config
changes. Instrumented unfrozen33ca616a now completes42.243decode/340.854prefill;
351–445ms/code selected adaptive wall may overlapMTP, not exposed cost.
Next six-ID Original whole-resident candidate22,24,28,30,33,34 now compiles
f7d5bf01/staged26/MTP20; payload6448742400/total6949138728B below8GiB service.
Native launcher accepts4|6 explicit IDs, legacy0|4 remains; generic API0..8
is unchanged. Six campaign rejects hybrid/freeze/Packed/Graphs. Actual bound
six-hole host/alias fixture and all-six independent blob/quant/decode/prefill
gate (12blobs,6decode,7prefill including2048x10 perdevice) are required.
Firstremote13 refuses withbad allocation near its8GiB privateJob during
secondowner. Release reference-only CUDA0 contexts after all owner/target
buffers/streams are destroyed, preserve full host reference bytes, add phase
labels and renew gates; do not widen limits. No fullmodel admission until
exact same-candidate numerical proof passes. Historicalbest48.279055 remains.
Test-only referencecontext teardown alone still failsremote14 inprefill.
Finalgate preserves all209715200B GPU0 reference, executes unchangedGPU1large
prefill, then compares its completed target output in16MiB chunks plus8MiBtail
instead of allocating a second200MiB host vector. Explicit completion/bytecount
and six other full comparisons preserve7nativeprefill calls. staged28/MTP22/
Remote15 and27regressions nowpass, numericalJobpeak7.809189GiB. The exact
six-layer candidate may proceed to fresh4K load/matchedscreen, not promotion.

The exposed-wait screen selects the next provisional opt-in global IDs
`--remote-expert-layer-ids 22,28,30,34`: exactly four sorted unique IDs0..47,
mutually exclusive with the legacy prefix flag, which retains0|4 behavior.
This is a placement hypothesis, not a measured speedup. Original decode remains
the comparison mode and profiling is off in matched speed runs. The selected
payload is4,535,091,200B, below the unchanged8GiB service cap including bounded
workspace. Each service translates global IDs to compact GPU slots; original
global native format, layer and GGUF role/shard offsets are never renumbered.

The host map separately translates global IDs to physical host offsets and
registration-slice indices. Excluded IDs get an invalid offset/slice and no host
blob, pinned status, device alias or GPU0 cache pair. Non-excluded layers are
packed contiguously, with independent source/destination scatter ranges and the
largest-blob tail still allocated/locked. Destination overlap, dimensional and
integer overflow errors are rejected before cold copies. Native cold scatter
validates its compact map and keeps role-major source/expert-major destination
semantics, including multiple original shards. Defaults use the identity map.
Small gates cover disjoint/prefix/edge placements, complete tiny native roles,
guards, invalid geometry and actual mapped-device reads on both GPUs. Fresh
real complete-blob/quant/decode/prefill evidence must name the same selected IDs
and exact payload before loading the full candidate. Prefix evidence cannot
admit an arbitrary selected-ID run. Graph-mode evidence likewise binds IDs.

The user's experimental-memory and VRAM-floor exceptions are extended only to
this advanced campaign: 4096context, RAM reserve0GiB, available-commit emergency
floor4GiB, Job committed-memory cap60GiB, startup/runtime freeVRAM floors0.
Temperature80C, finite allocations/transfers/graphs, bounded600–1800s trials,
exclusive model mutex, verified Windows Job attachment before load, fresh READY
resource approval, kill-on-close and clean owned-process shutdown remain.
Cache sizing is a finite allocation budget, not a free-VRAM shutdown threshold.
Matched candidate/control may fix the same3072/3500/4000/4359/4903/5500target
slots instead of relying on changing free-memory readings in `auto`. Each
selection needs fresh bound configs/gates/load;3500 is the first memory screen,
not a claimed optimum. The opt-in `--expert-cache-exact` is required: upstream
numeric cache budgets can otherwise expand into a different actual slot count.
The next screen caps expert CUDA registration at16384MiB and requires complete
resident backing of the anonymous suffix through VirtualLock; its synthetic
counterpart uses64MiB total/16MiB registration. No whole-arena registration
attempt is permitted in this opt-in mode. CUDA queries/device identity/sync and
native-head allocation stages are checked rather than inferred from nvidia-smi.
The first late111MiB MTP allocation failed; early initialization then loaded799MiB
but25GiB registration was followed by zero cache/head OOM. These are failed
loads, not benchmarks. The cap is a pressure hypothesis, not a proven fix.
No automatic trimming or pagefile modification to rescue a failed run.

No concurrent GPU benchmark, real-GGUF comparator or model process is permitted
while measuring a control/candidate. Read-only review may run alongside; builds
are serialized with all model speed runs after a telemetry timeout overlapped
an earlier build.

After an incomplete graphscreen stops at commit3.814GiB, add current owned
private counters alongside unchanged global RAM/commit and Job peaks. Read only
PIDs listed in the guardian Job, max16, verify membership before counters and
close every handle. Exit/PID-reuse races mark the sample incomplete. Do not
interpret constant peaks as constant current use, or globalcommit minus owned
private as another application's usage: shared/kernel/driver allocations and
sample timing also contribute. No Windows settings/guard-floor change.
Full-model load needs source/build/checkpoint binding and numerical/transport
gates first. A fresh matching load-only admission precedes inference for every
binary/config/device placement. Changing MTP device or cache invalidates it.

## Correctness and measurement

- Reuse both pristine-source18/18 CUDA component suites, labeled CUDA13.3
  source-level evidence, not tests inside the official CUDA13.0 binary.
- Independently test no-peer host staging, device restoration, buffer bounds
  and cleanup before using remote MTP with the GGUF.
- Full trials additionally require a fresh bound real one-layer MTP gate:
  36 deterministic prefill/first/catch-up/chain fixtures across nine modes,
  exact IDs, draft counts and float probability bits. The test-only Windows
  Job has a verified8GiB kill-on-close cap, no main expert arena or generation;
  local/remote relay/resident, remote mapped-residual and resident batched
  modes are compared. Three sequences use min-p0, min-p1 and alternating
  thresholds with contiguous p+=a+1, a16-cell attention window; three fresh
  near-context-end fixtures test sequential fallback and min-p0 truncation
  before any out-of-range launch. Batched execution
  and mapped transfer counters must be positively asserted, not inferred.
  This does not replace full target/corrupt-draft quality gates.
- Audit native cache arithmetic. CPU/GPU activation formats and reduction
  order differ in upstream; do not call greedy text identity a general proof
  of CPU equivalence or claim a flag repairs that difference.
- Preserve target verification. Test remote-vs-local draft residual transfer,
  emitted semantic/tool follow-up results and controlled continuation parity.
  Corrupted-draft/exact-target checks are required before quality promotion.
- Match model, context, prompt IDs, output counts, workers, cache budget,
  sampling and build/toolchain for the final single/dual comparison. Record
  effective device placement, real native counts/times, transfer/draft metrics,
  all failures, resources, warm-up and repetitions.
- Minimum screen:3semantic/tool requests,3x256-token decodes,2x3107-token
  prefills. Repeat the paired control/candidate at least once before selecting;
  report spread and cold-vs-warm behavior, not just the fastest request.

## Stop and handoff

Six Original whole-resident IDs22,24,28,30,33,34 now satisfy the isolated
performance stopping rule after a complete BAAB experiment. Same-engine
dual50.729674/51.110510 versus controls47.961769/48.299353 decode gives
+5.771%/+5.820%; pooled50.920092 versus48.130561 (+5.796%). Pooled prefill
346.079621 versus338.537074 (+2.228%). Both dual means exceed historicalbest
48.279055; the slightly higher new control48.299353 is also retained.
All eight fresh loads/benchmarks pass/nativeexit0/no stop, both GPUs compute,
and exact native expert/MTP/tiny-host gates plus27Python regressions pass.
Stop tuning and document/select this experimental 4K performance profile.

This is not quality or production promotion: semantic/tool/table checks pass,
counts8/8, but paired payloads5/8 and dual-repeat7/8, code probes truncated256
tokens. Actual cachebytes5.56GiB dual/5.71GiB single and pair lists differ.
Serve ignores spec-corrupt; true diagnostic corruption/window counters and
same-placement target-only token traces are still required before any quality
claim. No changes to ordinary launchers/Pi/ISTA/Windows/driver/pagefile.
All eight artifacts are checksum-snapshotted; no model process remains.
Evidence: wiki/raw/2026-09-27-strata-six-layer-repeated-screen-and-stop.md.

Earlier selected-four BAAB4K failed repeatable selection: first near-equal48.144 versus
48.082 decode is followed by42.682 versus48.003. Pooled dual45.413 does not
replace the historicalbest48.279055 threshold. Keep failed repetitions and
full payload/count evidence, not only the fastest result. Next source-reviewed
now numerically admitted hypothesis overlaps GPU0 cached hits with GPU1 selected-layer
misses. This requires exact sparse row/zero-hit/failure-handshake tests and
bounded GPU1 blob readback for every cold cache restore/refill path, outside
blocked callbacks. No inherited admission for a new sparse service or cache
policy; defaults and all resource caps remain. No cause-specific clock/PCIe/
WDDM claim is inferred from the two profiler-off runs.

Select only if both GPUs demonstrably compute and the matched dual-GPU result
has no observed correctness regression and matches/exceeds the best single-GPU
control (also report against33.486decode/296.651prefill historical stock).
Prefer a repeatable >=5% improvement, but an equal-speed result with genuinely
useful extra GPU capacity may satisfy the user's stated equal-or-better goal;
label equality/uncertainty honestly. Larger context, Pi/production launchers and
general agentic quality require separate evidence and are not silently promoted.
Keep immutable raw evidence, update short wiki synthesis/index/log, and release
all model processes when the stopping criteria are met or a trial is refused.
