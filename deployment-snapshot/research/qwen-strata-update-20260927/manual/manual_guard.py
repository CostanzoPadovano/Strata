"""User-authorized manual Strata launcher; reuses, never edits, frozen gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import threading
import time

MANUAL = Path(__file__).resolve().parent
CAMPAIGN = MANUAL.parent
sys.path.insert(0, str(CAMPAIGN.parent / 'qwen-strata-agent-20260927'))
import agent_guard as admitted


def report_and_stop(run: Path, done: threading.Event, timeout: int) -> None:
    last = ""
    deadline = time.monotonic() + timeout - 30
    while not done.wait(1):
        try:
            if time.monotonic() >= deadline and run.exists():
                (run / "stop-server").touch(exist_ok=True)
                print("[Strata] Durata massima raggiunta: arresto ordinato.", flush=True)
                deadline = float("inf")
            phase = json.loads((run / "phase.json").read_text())["phase"]
            if phase != last:
                print(f"[Strata] {phase}", flush=True)
                last = phase
            # Only show progress, never request content or credentials.
            lines = (run / "observer.log").read_text(encoding="utf-8", errors="replace").splitlines()
            progress = next((s for s in reversed(lines) if s.startswith("[strata]")), "")
            if progress and progress != getattr(report_and_stop, "progress", ""):
                print(progress, flush=True)
                report_and_stop.progress = progress
        except (OSError, ValueError, KeyError):
            pass


def stop_on_enter(run: Path, done: threading.Event) -> None:
    try:
        line = sys.stdin.readline()
        if line:
            deadline = time.monotonic() + 30
            while not run.exists() and not done.wait(0.1) and time.monotonic() < deadline:
                pass
            if not done.is_set() and run.exists():
                (run / "stop-server").touch(exist_ok=False)
    except (OSError, FileExistsError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Gates/resources only; never loads the model")
    parser.add_argument("--tag")
    parser.add_argument("--timeout", type=int, default=14400)
    opts = parser.parse_args()
    if not 60 <= opts.timeout <= 14400:
        raise ValueError("Manual lifetime must be 60..14400 seconds")
    config_file = CAMPAIGN / "configs" / "tier98304.json"
    config = json.loads(config_file.read_text())
    admitted.validate_config(config, 98304)
    admitted._previous_tier_admission(98304)
    # The completed 98K functional test must remain bound to the same source/config.
    proof = json.loads((CAMPAIGN / "tier-admission-98304.json").read_text())
    if (not proof.get("passed") or proof.get("context") != 98304
            or proof.get("config_sha256") != admitted.digest(config_file)
            or proof.get("source_gates_sha256") != admitted.digest(CAMPAIGN / "source-gates.json")):
        raise RuntimeError("Completed 98K functional admission is missing or stale")
    admitted._validate_bound_file(proof["result"])
    report, _ = admitted.artifact_checks(admitted.PROFILE)
    if opts.check:
        print(json.dumps({"gates_passed": True, "resources_passed": report["minimum_checks_passed"],
                          "memory": report["measured_memory"], "failures": report["failures"]}))
        if report["failures"]:
            raise RuntimeError("Resource admission failed")
        return
    tag = opts.tag or ("manual-" + time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1000000}")
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in tag):
        raise ValueError("Invalid manual evidence tag")
    run = CAMPAIGN / "runs" / tag
    if run.exists():
        raise FileExistsError("Evidence already exists; choose a fresh tag")
    shared = admitted.shared
    shared.ROOT = CAMPAIGN
    shared.EXE_HASH = admitted.digest(CAMPAIGN / "build" / "strata.exe")
    shared.policy = admitted.policy
    shared.artifact_checks = admitted.artifact_checks
    shared.job_counters = admitted.measured_job_counters
    print("ISTA / Strata sperimentale: 98304 context, xhigh, solo testo, output fino al contesto residuo.", flush=True)
    print("RAM libera >=8 GiB; commit libero >=4 GiB; Job <=60 GiB; GPU <80 C; nessuna soglia VRAM.", flush=True)
    print("Qualita numerica cache non certificata. Non avviare altri server sulla stessa macchina.", flush=True)
    print(f"Durata massima {opts.timeout // 60} minuti. INVIO arresta il server; chiudere la finestra termina il Job.", flush=True)
    print(f"Log: {run}", flush=True)
    done = threading.Event()
    threading.Thread(target=report_and_stop, args=(run, done, opts.timeout), daemon=True).start()
    threading.Thread(target=stop_on_enter, args=(run, done), daemon=True).start()
    try:
        args = argparse.Namespace(mode="bench", profile=admitted.PROFILE, gpu=0,
                                  pool_workers=8, vram_reserve_mib=None, tag=tag, timeout=opts.timeout)
        # Shared guardian retains fresh post-READY checks, mutex, Job kill-on-close,
        # all continuous floors and exact same-config load admission.
        shared.run_trial(args, config_file=config_file, probe_file=MANUAL / "manual_server.py",
                         device_visibility="0,1")
    finally:
        done.set()


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f"STRATA ARRESTATO: {str(error) or type(error).__name__}", file=sys.stderr, flush=True)
        raise SystemExit(1)
