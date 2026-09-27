# Recovery guide, not a one-click portable installer

The goal of this archive is to preserve this PC's implementation and evidence.
Absolute-path configs and hash-bound admissions are intentionally retained
unchanged. **They are historical proof metadata, not permission to skip fresh
validation on another PC, relocated paths, new binaries, drivers or weights.**

## 1. Retrieve the pinned source and binaries

Clone `https://github.com/CostanzoPadovano/Strata.git`, then check out branch
`codex/windows-dual5060ti-98k` or tag `local-dual5060ti-20260927`.
Run `python archive-tools/verify_archive.py` before using the snapshot.
The GitHub release with that tag contains `windows-runtime.zip` and its SHA256:
the original Windows engine/test binaries and selected bound proof files,
**not** model weights or a virtual environment. The accompanying asset manifest
enumerates every byte/checksum. Never run executables from an unverified copy.

## 2. Restore dependencies, without overwriting an existing project blindly

The original project root is `C:\Users\costa\Documents\Project_ANTIREZ`.
Review/copy only the named `deployment-snapshot/` files into an empty recovery
directory first; preserve any newer work. Runtime ZIP paths are relative to
the original project root. The archive does not alter Windows/pagefile/WSL.

Required external software: NVIDIA driver compatible with CUDA13.3/SM120,
Python3 with `numpy`, `Pillow` and `tokenizers`, Node22.23.0, Pi0.87.1,
Ubuntu-24.04 WSL2. Install dependencies in a fresh virtual environment at the
location expected by the BAT (`research/qwen-strata-20260926/.venv`).
The scripts were validated with the already installed environment, not a newly
provisioned one; fresh-machine recovery has not been tested.

If rebuilding: MSVC14.51.36231 / Visual Studio18 Community, Ninja/CMake and
CUDA13.3.73. Local build scripts retain exact paths (`E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit`)
and pinned ggml revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
Restore/clone the dependency at `research/qwen-strata-stock-20260926/tests-build/_deps/strata_llamacpp-src`.
`research/qwen-strata-vision-20260927/build.ps1` owns the current build;
`research/qwen-strata-agent-20260927/build.ps1` owns the older text build.
Do not compile the root fork and then substitute its new executable into old
admissions. Byte-identical binaries are not guaranteed across rebuilds.

## 3. Restore model assets locally (not hosted in this fork)

Obtain both IQ3_XXS shards from
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):

| Shard | Bytes | SHA256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | 47039860096 | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` | 28800138432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

Total disk size 75839998528 bytes (75.840 decimal GB), NOT resident RAM.
Original paths are in the archived configs under the user's LM Studio models
directory. Vision projector is `models/qwen38-vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf`,
616703104 bytes, SHA256
`b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad`.

Generated pack (`pack-iq3/dense.bin`, index and tokenizer) and MTP runtime
(`mtp/rt/dense.bin`, experts and vocab) are model-derived weights, excluded from
the archive. Restore an existing matching local backup or regenerate with the
**archived** tools, then compare against `runtime-manifest.json`. Run the
commands below from `deployment-snapshot/research/qwen-strata-20260926/source`,
whose MTP fetcher supports the pinned-repository environment override (the
unmodified upstream fetcher at fork root does not):

```
python tools/iq_pack.py --gguf <first-GGUF-shard> --out <pack-iq3>
python tools/mtp_fetch.py fetch --out <mtp-download-directory>
python tools/mtp_pack.py --src <mtp-download-directory> --experts q2_0 --out <mtp-q2_0.gguf>
python tools/mtp_rt.py --gguf <mtp-q2_0.gguf> --out <mtp/rt>
```

Use official Qwen revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` for MTP
(see archived `tools/mtp_fetch.py`), not a mutable main checkpoint.
Before `mtp_fetch.py`, set `STRATA_MTP_REPO` to
`https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/`.
Copy upstream `data/draft_vocab.bin` into `mtp/rt/` as setup.py does.
GGUF tools need the pinned dependency's `gguf-py` available via
`STRATA_GGUF_PY`. These are preparation steps, not an instruction to run the
full model before gates pass. Preserve model-license obligations.

## 4. Admissions and client recovery

Do not edit an old JSON `passed` flag or rebind old test results to new paths.
The archived component/numerical/encoder/4K/98K gate scripts are the recovery
workflow; read the archived campaign protocols before rerunning them with the
same finite memory/thermal bounds. Missing assets or changed identity must
stop. The normal BAT never automatically runs qualification tests.

The WSL wrapper source is
`research/qwen-strata-agent-20260927/manual/pi-global-installed.sh`.
It also depends on the pre-existing `~/.local/bin/pi-qwen` wrapper and other
TEST_QWEN provider helpers. The release archives the wrapper source, not
`~/.pi/agent/models.json`, settings, auth or sessions. Review the paths and
merge only the Strata integration into a newly installed Pi; do not replace
other providers or run old hash-conditioned installers blindly. The archived
ThinkingCap configurator preserves the independent wrapper.
The current `pi-qwen` source is also saved at
`deployment-snapshot/external/wsl/pi-qwen.sh`; other providers' engines/helpers
are outside this Strata fork, and must be restored from their own projects.

After fresh/current gates pass, use `run_qwen38_98k_ista_strata_server.bat --check`
first. The normal BAT starts only the server and prints real per-request timing.
Debug BAT is separate and may perform qualification prompts. Wait for PRONTO,
then reopen Pi; same identity `local-qwen38/qwen3.8-flash-next-local`, xhigh.
Always keep the memory/commit/thermal/lifetime guards.

## Exclusions and coverage

Included: upstream history, final native changes, historical lane sources,
BATs, guards, adapter tests/configs/proof metadata, numeric benchmark evidence,
checksums, selected exact native binaries and proof closure in the release.
Excluded: all GGUF/generated model tensors, toolchains/drivers, Python/npm
installations, auth/keys, personal conversations/requests, global settings and
unrelated projects. Therefore this is a reproducible implementation archive,
**not a claim that a clean clone alone immediately serves the model**.
