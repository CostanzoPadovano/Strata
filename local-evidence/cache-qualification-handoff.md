# Cache/idle selective backport: local handoff, 2026-09-27

User authorizes update, elects to finish qualification, then asks for the same
Desktop BAT so they can test themselves. Candidate promoted locally, not pushed.

Final engine SHA256:
`2e630a0b125a0b4f2be300678216d8aca2e086d1e5c2d9cb54855b55685c4c77`.
Source-only7-file patch SHA256:
`3b116921455ac861152ad4d057647a06c54b5c3f0e937d3a28f2a87e142e95da`.
Source-gates SHA256:
`bdcde0c94faec4ead733fd92ebb3d19a8ae469d7c1a3feca9ee2ab68b6532e82`.
Selective upstream v0.1.3 on local v0.1.2 dual-GPU/vision, NOT all of v0.1.8.

`cache-idle04`: 9native components plus MTP/selected12 numerical gates pass.
Final Windows Python gate45tests:37pass/8POSIX-specific skipped (actual WSL
wrapper8tests and installedPi in-memory overlay had separately passed).
`update-encoder03` passes. `update-r2-{4k,98k}-{load,smoke,cache}` all pass;
native clean exit0. Cache comparisons include EOS live exact prefix29,
checkpoint3066, live-length3073, cancel recovery, long→short16384.
Longest real prompt34817, not populated98304-history or four-hour acceptance.
Rewind-short16417: cold66861.0ms, warm715.1ms, same output-token SHA256.

Serve-only `update-r2-pi` starts without any timings file/generation/encoder.
Single GetProcess idle sample:5.0191413s, CPU delta0.0s at counter resolution,
native working set22.233711GiB. Actual staged Pi test with images/file passes
all checks, synthetic output EMBER-9056/COBALT-4428/ORCHID-7314. Effective
WSL gateway8038 health/SHA checked before/after and per request; payload marker
and native input/output counts correlated. No actual conversations or API keys
printed. Staged successful2calls decode39.93/40.44tok/s, not a paired benchmark.

Failed helper evidence retained: inlineNode gateway quoting expanded a backtick
before Pi; standalone .mjs/directargv fixes that. Windows text stdin CRLF then
failed a hash preflight, corrected to UTF8 bytes. First actual Pi test read both
images correctly but the synthetic text file had not been copied; model honestly
reported ENOENT. File added; immutable suffixfixture02 test passes. Do not label
these as model crashes, token mismatch, or benchmark failures.

Measured minimum free RAM/commit and Jobpeak (GiB, whole-run samples):

| Run | RAM free min | Commit free min | Job peak |
| --- | ---: | ---: | ---: |
| 4K load | 26.634 | 14.269 | 47.641 |
| 4K smoke | 20.932 | 7.631 | 49.888 |
| 4K cache | 26.367 | 13.585 | 48.162 |
| 98K load | 26.696 | 11.741 | 50.169 |
| 98K smoke | 22.845 | 7.081 | 52.430 |
| 98K cache | 24.349 | 9.135 | 50.949 |
| Pi server | 22.037 | 4.572 | 52.480 |

No resource-floor stop; hottest sampled GPU66C acrossfinalruns. RAM/commit
global pressure is not process attribution; Jobpeak is NOT physical RAM or
pagefile usage. Pagefile writes were not measured by this campaign. Codex stayed
open; no builds/Git/packaging during measured model runs. No OS/pagefile edit.

Normal Desktop1d delegates to root BAT now SHA256
`926d9bc79484ec8e0fc4271f8796e0245dafeb352e8034d53ce65e8606ca9f57`.
Actual Desktop--check0: gates/resources current, startup_self_test=false,
automatic_test_runs=false, model_started=false. Debug uses the SAME candidate
serve-only guardian;32offline debug tests pass, sampler/protections unchanged.
Backup rootBAT before-cache-20260927 remains executable with frozen F107.

WSL global Pi wrapper fe8a548aa40e8aa6bcf95875fdeaabc4c6ef2b7899a2fff5983963abd1d8a347
matches the tested staged file; backup8ae336c66ff1b76c0f2617ab079fd06500f4ba99f7784277874be261362cc4a8.
Models668b6490e7e59549915dbdfdff0b097cdc27b98789a9b85559210e7c74b4d7f1,
settings8dcb6df71babc993c91c6acd05d965711f04674b0de2d4895ad652c4c573fd50
unchanged. GlobalPi--version outputs0.87.1 with server stopped.

Test server stopped cleanly; no model left running by this task. Weights,
previous runtime, guards and publication preserved. Normal output policy
remaining-context-v1 retained; no1024/4096/8192 hard cap. Use same DesktopBAT,
wait PRONTO, reopen globalPi. This is experimental/manual-use admission, not
scientific validation or universal agent-quality certification.

Evidence boundary: `research/qwen-strata-update-20260927/`: native-gates.json,
source-gates.json, manual/serve-only-gates.json, encoder/load/vision/cache
admissions, gate-runs/cache-idle04, runs/update-r2-*, especially
pi-staged-vision-fixture02-result.json/-requests.jsonl in runs/update-r2-pi.
