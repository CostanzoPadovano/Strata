"""Run bounded source-built Strata components and bind reproducible evidence.

This intentionally does not load a GGUF, expert arena, or official release
binary. Results apply only to this clean-source CUDA 13.3 rebuild.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
BUILD = ROOT / "tests-build"
LOGS = ROOT / "source-gates-logs"
MANIFEST = ROOT / "source-gates.json"
REVISION = "6da1f667e86558b152ab128edf3ebf77a80a9e57"
OFFICIAL_EXE_SHA256 = "e3684cc5c6ff51cfff3c4ebc864be6fa6c7179ae98176319f3da042cc1dd0c97"
GPU = 1
TIMEOUT_SECONDS = 120
TARGETS = (
    "strata-device", "dequant_s2_parity", "s2_gemv_parity", "shared_expert_parity",
    "gr_parity", "gdn_parity", "s2_gemv_q8_parity", "sampler_parity", "rope_parity",
    "quantize_act_parity", "router_top10_parity", "s_gemv_parity", "elementwise_parity",
    "bf16_gemv_parity", "s_gemv_q8k_parity", "qsa_parity", "ple_reader_test", "kv_q8_parity",
)
OMISSIONS = (
    "Published tests/ and bench/micro/ are absent; Catch2 and bench/micro oracle targets were not built or run.",
    "ple_parity was not run because its required bench/micro/ple_in.bin and ple_out.bin fixtures are absent.",
    "expert_arena_load and all GGUF/native-arena/full-model tests were not run.",
    "These are not tests embedded in or executed by the official CUDA 13.0 release binary.",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def output(args: list[str], **kwargs: object) -> str:
    return subprocess.check_output(args, text=True, encoding="utf-8", errors="replace", **kwargs).strip()


def temperature() -> int:
    raw = output([
        "nvidia-smi", f"--id={GPU}", "--query-gpu=temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    return int(raw.splitlines()[0].strip())


def tracked_hashes() -> list[dict[str, object]]:
    raw = subprocess.check_output(["git", "-C", str(SOURCE), "ls-files", "-z"])
    paths = [p.decode("utf-8") for p in raw.split(b"\0") if p]
    rows: list[dict[str, object]] = []
    for rel in sorted(paths):
        path = SOURCE / rel
        index = output(["git", "-C", str(SOURCE), "ls-files", "-s", "--", rel]).split()
        row: dict[str, object] = {
            "path": rel.replace("\\", "/"), "git_index_mode": index[0], "git_object": index[1],
        }
        if path.is_file():
            row.update({"type": "file", "bytes": path.stat().st_size, "sha256": sha256(path)})
        else:
            # The published checkout records third_party/llama.cpp as an
            # unpopulated gitlink. Bind its exact index object without trying
            # to hash a directory or silently treating it as a source file.
            row.update({"type": "gitlink", "checkout_present": path.exists()})
        rows.append(row)
    return rows


def main() -> int:
    global GPU, LOGS, MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=[0, 1], default=1)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    if args.label and any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.label):
        raise ValueError('Invalid evidence label')
    GPU = args.gpu
    if args.label:
        LOGS = ROOT / f'source-gates-logs-{args.label}'
        MANIFEST = ROOT / f'source-gates-{args.label}.json'
        if MANIFEST.exists():
            raise RuntimeError('Named gate evidence already exists; use a fresh label')
    LOGS.mkdir(exist_ok=True)
    revision = output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"])
    changes = output(["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no"])
    if revision != REVISION or changes:
        raise RuntimeError("Official source revision is wrong or has tracked changes")

    official_exe = ROOT / "official-engine" / "strata.exe"
    if sha256(official_exe) != OFFICIAL_EXE_SHA256:
        raise RuntimeError("Official release executable identity mismatch")

    env = os.environ.copy()
    cuda_bin = Path(r"E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit\bin")
    env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(GPU)
    env["STRATA_FORCE_AVX2"] = "1"
    rows: list[dict[str, object]] = []
    failed = False
    for name in TARGETS:
        exe = BUILD / f"{name}.exe"
        if not exe.is_file():
            raise RuntimeError(f"Missing explicit component executable: {exe}")
        before = temperature()
        if before >= 80:
            raise RuntimeError(f"GPU{GPU} is {before} C before {name}; refusing")
        args = [str(exe)] + ([] if name == "kv_q8_parity" else ["--selftest"])
        started = time.monotonic()
        try:
            run = subprocess.run(
                args, cwd=SOURCE, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=TIMEOUT_SECONDS,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            run = None
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        elapsed = time.monotonic() - started
        after = temperature()
        if run is not None:
            stdout, stderr, returncode = run.stdout, run.stderr, run.returncode
        else:
            returncode = None
        log = LOGS / f"{name}.log"
        log.write_text(
            f"command={subprocess.list2cmdline(args)}\n"
            f"CUDA_VISIBLE_DEVICES={GPU}\nSTRATA_FORCE_AVX2=1\n"
            f"timeout_seconds={TIMEOUT_SECONDS}\nelapsed_seconds={elapsed:.6f}\n"
            f"temperature_before_c={before}\ntemperature_after_c={after}\n"
            f"returncode={returncode}\ntimed_out={str(timed_out).lower()}\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
            encoding="utf-8",
        )
        passed = returncode == 0 and not timed_out and after < 80
        rows.append({
            "name": name,
            "command": args,
            "sha256": sha256(exe),
            "bytes": exe.stat().st_size,
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_seconds": round(elapsed, 6),
            "temperature_before_c": before,
            "temperature_after_c": after,
            "passed": passed,
            "log": str(log.relative_to(ROOT)).replace("\\", "/"),
            "log_sha256": sha256(log),
        })
        if not passed:
            failed = True
            break

    cache = BUILD / "CMakeCache.txt"
    manifest = {
        "schema": "antirez-strata-source-gates/v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": not failed and len(rows) == len(TARGETS),
        "source_revision": revision,
        "source_tracked_changes": changes,
        "source_files": tracked_hashes(),
        "cmake_cache_sha256": sha256(cache),
        "official_exe_sha256": OFFICIAL_EXE_SHA256,
        "official_exe_executed": False,
        "build_environment": {
            "cmake": output(["cmake", "--version"]).splitlines()[0],
            "nvcc": output([r"E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit\bin\nvcc.exe", "--version"]).splitlines()[-1],
            "cuda_toolkit": "13.3.1",
            "compiler": "MSVC 19.51 (VS 18 toolset 14.51.36231)",
            "cuda_architecture": 120,
            "configuration": "Release; STRATA_BUILD_TESTS=OFF; STRATA_PORTABLE=ON",
        },
        "execution": {
            "gpu_physical_index": GPU,
            "cuda_visible_devices": str(GPU),
            "strata_force_avx2": "1",
            "sequential": True,
            "timeout_seconds_per_test": TIMEOUT_SECONDS,
            "full_gguf_loaded": False,
            "native_expert_arena_loaded": False,
        },
        "component_gates": rows,
        "omissions": list(OMISSIONS),
        "evidence_scope": (
            "Source-level synthetic numerical evidence from a clean revision rebuilt with CUDA 13.3.1. "
            "It is not a test inside the official CUDA 13.0 release binary, does not prove that binary's "
            "runtime numerical correctness, and does not establish full-model or GGUF equivalence."
        ),
        "next_gate": "Subsequent guarded semantic load of the hash-pinned official release binary.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": manifest["passed"], "tests_run": len(rows), "tests_expected": len(TARGETS),
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
    }, indent=2))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
