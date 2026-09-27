"""Run and bind real bounded MTP or selected-12 numerical evidence."""
from __future__ import annotations
import argparse
import ctypes as C
from ctypes import wintypes as W
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'qwen-strata-agent-20260927'))
import agent_guard as guard

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
IDS = ",".join(map(str, guard.REMOTE_IDS))
MTP_MARKER = "PASS mtp_equivalence fixtures=36 modes=9 probability_bits=exact"
REMOTE_MARKER = ("PASS remote_experts_selected12 layers=12 ids=" + IDS +
 " payload_bytes=13526630400 allocation_bytes=14027026728 blob_cases=24 decode_calls=12 "
 "prefill_calls=13 blob_bytes=exact quant_bytes=exact outputs=exact large_prefill_bytes=209715200 "
 "chunk_bytes=16777216 owner_teardown=exact optimized_over8=rejected layer_coverage=all mode=original")


def load_binding() -> tuple[dict, Path]:
    path = BUILD / "build-binding.json"
    binding = json.loads(path.read_text(encoding="utf-8-sig"))
    for row in binding.get("source_files", []):
        candidate = ROOT / "source" / row["path"]
        if not candidate.is_file() or guard.digest(candidate) != row["sha256"]:
            raise RuntimeError(f"Source differs from build binding: {row['path']}")
    fixture_keys = {
        "transport_test.cu": "transport", "host_registration_test.cu": "registration",
        "embedding_cache_test.cu": "embedding", "mtp_equivalence_test.cpp": "mtp_equivalence",
        "mtp_profile_test.cpp": "mtp_profile", "remote_experts_test.cpp": "remote_experts",
        "compacted_loader_test.cpp": "compacted_loader", "remote_experts_profile_test.cpp": "remote_experts_profile",
        "verify_wait_test.cu": "verify_wait",
        "vision_protocol_test.cpp": "vision_protocol", "vision_mrope_test.cu": "vision_mrope",
        "conversation_cache_test.cpp": "conversation_cache", "pool_idle_test.cpp": "pool_idle",
    }
    for filename, key in fixture_keys.items():
        source = ROOT / filename
        if not source.is_file() or guard.digest(source) != binding.get(key + "_source_sha256"):
            raise RuntimeError(f"Fixture differs from build binding: {source.name}")
    executables = {
        "strata.exe": "executable_sha256", "mtp_transport_test.exe": "transport_sha256",
        "host_registration_test.exe": "registration_sha256", "embedding_cache_test.exe": "embedding_sha256",
        "mtp_equivalence_test.exe": "mtp_equivalence_sha256", "mtp_profile_test.exe": "mtp_profile_sha256",
        "remote_experts_test.exe": "remote_experts_sha256", "compacted_loader_test.exe": "compacted_loader_sha256",
        "remote_experts_profile_test.exe": "remote_experts_profile_sha256", "verify_wait_test.exe": "verify_wait_sha256",
        "vision_protocol_test.exe": "vision_protocol_sha256", "vision_mrope_test.exe": "vision_mrope_sha256",
        "conversation_cache_test.exe": "conversation_cache_sha256", "pool_idle_test.exe": "pool_idle_sha256",
    }
    for name, key in executables.items():
        if guard.digest(BUILD / name) != binding.get(key):
            raise RuntimeError(f"Executable differs from build binding: {name}")
    return binding, path


def _make_job(cap_gib: int):
    if cap_gib < 8 or cap_gib > 24:
        raise ValueError("Numerical Job cap must be within 8..24 GiB")
    s = guard.shared
    s.k32.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
    s.k32.CreateJobObjectW.restype = W.HANDLE
    s.k32.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
    s.k32.QueryInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.POINTER(W.DWORD)]
    s.k32.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
    s.k32.IsProcessInJob.argtypes = [W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)]
    job = s.k32.CreateJobObjectW(None, None); s.wincheck(job)
    try:
        limits = s.Limits(); limits.basic.flags = 0x2000 | 0x200; limits.job_limit = cap_gib * guard.GIB
        s.wincheck(s.k32.SetInformationJobObject(job, 9, C.byref(limits), C.sizeof(limits)))
        actual = s.Limits()
        s.wincheck(s.k32.QueryInformationJobObject(job, 9, C.byref(actual), C.sizeof(actual), None))
        if actual.job_limit != limits.job_limit or not (actual.basic.flags & 0x2000) or not (actual.basic.flags & 0x200):
            raise RuntimeError("Numerical Job limits were not verified")
        return job
    except BaseException:
        s.k32.CloseHandle(job)
        raise


def _resource_check(job=None) -> dict:
    memory = guard.shared.counters(); gpu = guard.shared.gpu_counters()
    if memory["ram_free"] < 8 * guard.GIB: raise RuntimeError("Free RAM below 8 GiB")
    if memory["commit_free"] < 4 * guard.GIB: raise RuntimeError("Available commit below 4 GiB")
    if len(gpu) != 2 or any(row[2] >= 80 for row in gpu): raise RuntimeError("GPU count/temperature guard failed")
    row = {**memory, "gpu": gpu}
    if job is not None: row.update(guard.measured_job_counters(job))
    return row


def wait_for_job_attachment(run: Path, timeout_seconds: float = 30) -> None:
    deadline = time.monotonic() + timeout_seconds
    marker = run / "job-attached"
    while not marker.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Parent did not attach the numerical observer Job")
        time.sleep(.05)


def child_run(run: Path) -> int:
    """Wait for attachment, then start the native child inside inherited Job."""
    wait_for_job_attachment(run)
    s = guard.shared
    s.k32.GetCurrentProcess.argtypes = []
    s.k32.GetCurrentProcess.restype = W.HANDLE
    s.k32.IsProcessInJob.argtypes = [W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)]
    s.k32.IsProcessInJob.restype = W.BOOL
    member = W.BOOL(); s.wincheck(s.k32.IsProcessInJob(s.k32.GetCurrentProcess(), None, C.byref(member)))
    if not member.value: raise RuntimeError("Numerical observer is not in a Job")
    command = json.loads((run / "command.json").read_text())["command"]
    with (run / "native.log").open("x", encoding="utf-8") as log:
        native = subprocess.Popen(command, cwd=ROOT / "source", stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        native_member = W.BOOL()
        s.wincheck(s.k32.IsProcessInJob(int(native._handle), None, C.byref(native_member)))
        if not native_member.value:
            native.kill(); native.wait(timeout=10)
            raise RuntimeError("Native numerical child did not inherit the Job")
        exit_code = native.wait()
    (run / "child-result.json").write_text(json.dumps({"native_exit_code": exit_code,
                                                        "native_job_member": True}))
    return exit_code


def run_native(command: list[str], run: Path, marker: str, timeout: int, cap_gib: int = 24,
               *, exact_marker: bool = True, required_markers: tuple[str, ...] = ()) -> dict:
    """Execute one native gate under mutex, kill-on-close Job and live floors."""
    run.mkdir(parents=True, exist_ok=False)
    (run / "command.json").write_text(json.dumps({"command": command, "marker": marker,
                                                    "timeout_seconds": timeout, "job_cap_gib": cap_gib}, indent=2))
    s = guard.shared; mutex = job = process = None; owned = False
    s.k32.CreateMutexW.argtypes = [C.c_void_p, W.BOOL, W.LPCWSTR]
    s.k32.CreateMutexW.restype = W.HANDLE
    s.k32.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
    s.k32.WaitForSingleObject.restype = W.DWORD
    s.k32.ReleaseMutex.argtypes = [W.HANDLE]
    s.k32.ReleaseMutex.restype = W.BOOL
    s.k32.CloseHandle.argtypes = [W.HANDLE]
    s.k32.CloseHandle.restype = W.BOOL
    try:
        mutex = s.k32.CreateMutexW(None, False, "Local\\QwenBioagentGpuExclusive"); s.wincheck(mutex)
        if s.k32.WaitForSingleObject(mutex, 0) not in (0, 0x80):
            raise RuntimeError("Another guarded model owns the GPUs")
        owned = True; before = _resource_check(); job = _make_job(cap_gib)
        env = dict(os.environ)
        for name in list(env):
            if name.startswith("STRATA_"): env.pop(name)
        env["STRATA_FORCE_AVX2"] = "1"; env["CUDA_VISIBLE_DEVICES"] = "0,1"
        env["PATH"] = r"E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit\bin" + os.pathsep + env["PATH"]
        log_path = run / "native.log"
        with (run / "observer.log").open("x", encoding="utf-8") as log, (run / "telemetry.jsonl").open("x") as telemetry:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child-run", str(run)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            s.wincheck(s.k32.AssignProcessToJobObject(job, int(process._handle)))
            member = W.BOOL(); s.wincheck(s.k32.IsProcessInJob(int(process._handle), job, C.byref(member)))
            if not member.value: raise RuntimeError("Numerical process is not in the Job")
            (run / "job-attached").write_text("owned\n")
            started = time.monotonic()
            while process.poll() is None:
                sample = _resource_check(job); sample["elapsed"] = time.monotonic() - started
                telemetry.write(json.dumps(sample) + "\n"); telemetry.flush()
                if sample["elapsed"] > timeout: raise RuntimeError("Numerical gate timeout")
                time.sleep(.5)
        child_result = json.loads((run / "child-result.json").read_text())
        text = log_path.read_text(encoding="utf-8", errors="replace")
        marker_ok = text.splitlines().count(marker) == 1 if exact_marker else sum(marker in line for line in text.splitlines()) == 1
        if (process.returncode != 0 or child_result.get("native_exit_code") != 0
                or not child_result.get("native_job_member") or not marker_ok
                or any(item not in text.splitlines() for item in required_markers)):
            raise RuntimeError("Native numerical gate failed or exact marker is absent")
        after = _resource_check(job)
        artifact_paths = [log_path, run / "command.json", run / "observer.log",
                          run / "telemetry.jsonl", run / "child-result.json", run / "job-attached"]
        bound_evidence = [{"path": str(path), "sha256": guard.digest(path)} for path in artifact_paths]
        return {"passed": True, "exit_code": 0, "before": before, "after": after,
                "log": {"path": str(log_path), "sha256": guard.digest(log_path)},
                "command": {"path": str(run / "command.json"), "sha256": guard.digest(run / "command.json")},
                "bound_evidence": bound_evidence}
    finally:
        if job: s.k32.CloseHandle(job)  # verified kill-on-close
        if process and process.poll() is None:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        if owned: s.k32.ReleaseMutex(mutex)
        if mutex: s.k32.CloseHandle(mutex)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--child-run":
        raise SystemExit(child_run(Path(sys.argv[2]).resolve()))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("purpose", choices=("mtp", "remote12")); p.add_argument("--tag", required=True)
    p.add_argument("--timeout", type=int, default=1800); p.add_argument("--job-cap-gib", type=int, default=24)
    a = p.parse_args()
    if not a.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in a.tag):
        raise ValueError("Invalid evidence tag")
    if not 30 <= a.timeout <= 3600: raise ValueError("Timeout outside bounded range")
    binding, binding_path = load_binding()
    gates_path = ROOT / "source-gates.json"; gates = json.loads(gates_path.read_text())
    if not gates.get("passed") or gates.get("build_binding_sha256") != guard.digest(binding_path):
        raise RuntimeError("Fresh matching source gates must pass first")
    if a.purpose == "remote12":
        mtp_path = ROOT / "mtp-admission.json"
        mtp = json.loads(mtp_path.read_text())
        if (not mtp.get("passed") or mtp.get("source_gates_sha256") != guard.digest(gates_path)
                or mtp.get("build_binding_sha256") != guard.digest(binding_path)):
            raise RuntimeError("Fresh matching MTP admission must pass before remote12")
    cfg = json.loads((ROOT / "configs" / "tier4096.json").read_text())
    guard.validate_config(cfg, 4096)
    args = cfg["args"]; pack = args[args.index("--pack") + 1]
    native = args[args.index("--native") + 1]; mtp = args[args.index("--mtp") + 1]
    if a.purpose == "mtp":
        exe = BUILD / "mtp_equivalence_test.exe"; marker = MTP_MARKER
        command = [str(exe), "--pack", pack, "--native", native, "--mtp", mtp]
        admission_name = "mtp-admission.json"
    else:
        exe = BUILD / "remote_experts_test.exe"; marker = REMOTE_MARKER
        command = [str(exe), "--pack", pack, "--native", native, "--layer-ids", IDS]
        admission_name = "remote12-admission.json"
    (ROOT / admission_name).write_text(json.dumps({"passed": False, "invalidated_by": a.tag,
                                                   "source_gates_sha256": guard.digest(gates_path)}, indent=2))
    evidence = run_native(command, ROOT / "numerical-runs" / a.tag, marker, a.timeout, a.job_cap_gib)
    admission = {**evidence, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "purpose": a.purpose, "marker": marker,
                 "engine_executable_sha256": binding["executable_sha256"],
                 "test_executable_sha256": guard.digest(exe),
                 "build_binding_sha256": guard.digest(binding_path),
                 "source_gates_sha256": guard.digest(gates_path),
                 "bound_evidence": [*evidence["bound_evidence"],
                                    {"path": str(binding_path), "sha256": guard.digest(binding_path)},
                                    {"path": str(exe), "sha256": guard.digest(exe)}]}
    (ROOT / admission_name).write_text(json.dumps(admission, indent=2))
    print(json.dumps({"passed": True, "admission": str(ROOT / admission_name)}))


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"STRATA NUMERICAL STOP: {error}", file=sys.stderr, flush=True); raise SystemExit(1)
