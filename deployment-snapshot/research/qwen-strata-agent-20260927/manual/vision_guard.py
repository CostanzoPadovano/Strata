"""Separate protected encoder/4K/98K vision admissions, then manual serving."""
from __future__ import annotations
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from vision_campaign import (ROOT, MANUAL, TEXT, admitted, artifact_checks, bound, check_bound,
                             digest, encoder_config, validate_config)
from manual_guard import report_and_stop, stop_on_enter


def source_checks():
    gate = json.loads((ROOT / "source-gates.json").read_text())
    if not gate.get("passed") or gate.get("parent_gates_sha256") != digest(TEXT / "source-gates.json"):
        raise RuntimeError("Python/vision identity gate missing/stale")
    for row in gate["bound_files"]: check_bound(row)
    admitted.artifact_checks(admitted.PROFILE)


def encoder_trial(tag, timeout):
    source_checks()
    shared = admitted.shared
    limits = admitted.policy(admitted.PROFILE)
    shared.check_resources(shared.counters(), shared.gpu_counters(), limits)
    run = ROOT / "runs" / tag
    run.mkdir(parents=True, exist_ok=False)
    (run / "encoder-config.json").write_text(json.dumps({"vision": encoder_config(), "fixtures": str(ROOT / "fixtures")}))
    result = {"passed": False, "scope": "encoder-only owned4GiB Job; not a full-model load"}
    job = process = mutex = None
    owned = False
    try:
        shared.k32.CreateMutexW.argtypes = [C.c_void_p, shared.W.BOOL, shared.W.LPCWSTR]
        shared.k32.CreateMutexW.restype = shared.W.HANDLE
        mutex = shared.k32.CreateMutexW(None, False, "Local\\QwenBioagentGpuExclusive")
        shared.wincheck(mutex)
        shared.k32.WaitForSingleObject.argtypes = [shared.W.HANDLE, shared.W.DWORD]
        if shared.k32.WaitForSingleObject(mutex, 0) not in (0, 0x80): raise RuntimeError("Another model owns GPUs")
        owned = True
        job = shared.make_job(60)
        cap = shared.Limits()
        cap.basic.flags = 0x2000 | 0x200
        cap.job_limit = 4 * (1 << 30)
        shared.wincheck(shared.k32.SetInformationJobObject(job, 9, C.byref(cap), C.sizeof(cap)))
        with (run / "observer.log").open("x") as log, (run / "telemetry.jsonl").open("x") as telemetry:
            process = subprocess.Popen([sys.executable, str(MANUAL / "vision_encoder_probe.py"), str(run)],
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            shared.wincheck(shared.k32.AssignProcessToJobObject(job, int(process._handle)))
            member = shared.W.BOOL()
            shared.wincheck(shared.k32.IsProcessInJob(int(process._handle), job, C.byref(member)))
            if not member.value: raise RuntimeError("Encoder Job attachment failed")
            (run / "job-attached").write_text("owned\n")
            started = time.monotonic()
            while process.poll() is None:
                memory, gpu = shared.counters(), shared.gpu_counters()
                telemetry.write(json.dumps({"elapsed": time.monotonic() - started, **memory, "gpu": gpu,
                                           **admitted.measured_job_counters(job)}) + "\n")
                telemetry.flush()
                shared.check_resources(memory, gpu, limits)
                if time.monotonic() - started > timeout: raise TimeoutError("Encoder gate lifetime exceeded")
                time.sleep(.5)
            result["observer_exit_code"] = process.returncode
            probe = json.loads((run / "probe.json").read_text())
            if process.returncode or not probe.get("passed") or probe.get("encoder_exit_code") != 0:
                raise RuntimeError("Encoder gate failed; see " + str(run))
            result["passed"] = True
    except BaseException as error:
        result["failure"] = str(error)
        raise
    finally:
        if job: shared.k32.CloseHandle(job)
        if process and process.poll() is None: process.wait(timeout=10)
        if owned:
            shared.k32.ReleaseMutex.argtypes = [shared.W.HANDLE]
            shared.k32.ReleaseMutex(mutex)
        if mutex: shared.k32.CloseHandle(mutex)
        (run / "result.json").write_text(json.dumps(result, indent=2))
    proof = {"passed": True, "source_gates_sha256": digest(ROOT / "source-gates.json"),
             "bound_files": [bound(run / name) for name in ("result.json", "probe.json", "encoder.log", "telemetry.jsonl")]}
    (ROOT / "encoder-admission.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps({"encoder_passed": True, "run": str(run)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("encoder", "load", "smoke", "manual", "check"))
    parser.add_argument("--context", type=int, choices=(4096, 98304), default=98304)
    parser.add_argument("--tag")
    parser.add_argument("--timeout", type=int, default=14400)
    opts = parser.parse_args()
    if not 60 <= opts.timeout <= 14400: raise ValueError("Finite lifetime must be60..14400s")
    if opts.mode == "manual" and opts.context != 98304:
        raise ValueError("Manual vision launcher is fixed at98304")
    tag = opts.tag or ("vision-manual-" + time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1000000}")
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in tag):
        raise ValueError("Invalid evidence tag")
    if opts.mode == "encoder": return encoder_trial(tag, min(opts.timeout, 360))
    config_file = ROOT / f"vision{opts.context}.json"
    validate_config(json.loads(config_file.read_text()), opts.context)
    report, gates = artifact_checks(admitted.PROFILE)
    if opts.context == 98304 or opts.mode == "manual":
        proof = json.loads((ROOT / "vision-admission-4096.json").read_text())
        if not proof.get("passed") or proof.get("source_gates_sha256") != digest(ROOT / "source-gates.json"):
            raise RuntimeError("Fresh4K image/text smoke must pass first")
        for row in proof["bound_files"]: check_bound(row)
    if opts.mode in ("manual", "check") and opts.context == 98304:
        proof = json.loads((ROOT / "vision-admission-98304.json").read_text())
        if not proof.get("passed") or proof.get("source_gates_sha256") != digest(ROOT / "source-gates.json"):
            raise RuntimeError("Fresh98K image/text smoke must pass first")
        for row in proof["bound_files"]: check_bound(row)
    if opts.mode == "check":
        print(json.dumps({"gates_passed": True, "resources_passed": report["minimum_checks_passed"],
                          "memory": report["measured_memory"], "failures": report["failures"]}))
        if report["failures"]: raise RuntimeError("Resource admission failed")
        return
    shared = admitted.shared
    shared.ROOT = ROOT
    shared.EXE_HASH = digest(ROOT / "build/strata.exe")
    shared.policy = admitted.policy
    shared.artifact_checks = artifact_checks
    shared.job_counters = admitted.measured_job_counters
    mode = "bench" if opts.mode == "manual" else opts.mode
    done = threading.Event()
    if opts.mode == "manual":
        print("ISTA / Strata VISION CPU:98K, stesso Pi globale, xhigh/output fino al contesto residuo. Encoder8thread/1024token/immagine.", flush=True)
        print("RAMfree8/commitfree4/Job60GiB, VRAM0, GPU<80C. Cache numerica sperimentale; INVIO arresta.", flush=True)
        threading.Thread(target=stop_on_enter, args=(ROOT / "runs" / tag, done), daemon=True).start()
    threading.Thread(target=report_and_stop, args=(ROOT / "runs" / tag, done, opts.timeout), daemon=True).start()
    try:
        ns = argparse.Namespace(mode=mode, profile=admitted.PROFILE, gpu=0, pool_workers=8,
                                vram_reserve_mib=None, tag=tag, timeout=opts.timeout)
        if mode == "load":
            (ROOT / "load-admission.json").write_text(json.dumps({"passed": False, "invalidated_by": tag}))
        shared.run_trial(ns, config_file=config_file, probe_file=MANUAL / "manual_server.py", device_visibility="0,1")
    finally: done.set()
    run = ROOT / "runs" / tag
    if mode == "smoke":
        proof = {"passed": True, "context": opts.context, "source_gates_sha256": digest(ROOT / "source-gates.json"),
                 "config_sha256": digest(config_file),
                 "bound_files": [bound(run / name) for name in ("result.json", "probe.json", "timings.jsonl")]}
        (ROOT / f"vision-admission-{opts.context}.json").write_text(json.dumps(proof, indent=2))


if __name__ == "__main__":
    try: main()
    except (Exception, KeyboardInterrupt) as error:
        print(f"STRATA VISION ARRESTATO: {str(error) or type(error).__name__}", file=sys.stderr, flush=True)
        raise SystemExit(1)
