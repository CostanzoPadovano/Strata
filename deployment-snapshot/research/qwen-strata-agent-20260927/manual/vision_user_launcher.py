"""User-initiated startup; complete stale98K gates on first launch, never bypass."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from vision_campaign import ROOT, MANUAL, admitted, artifact_checks, check_bound, digest, validate_config


def current_smoke(context):
    path = ROOT / f"vision-admission-{context}.json"
    if not path.is_file():
        return False
    proof = json.loads(path.read_text())
    if (not proof.get("passed") or proof.get("context") != context
            or proof.get("source_gates_sha256") != digest(ROOT / "source-gates.json")
            or proof.get("config_sha256") != digest(ROOT / f"vision{context}.json")):
        return False
    for row in proof["bound_files"]:
        check_bound(row)
    return True


def current_load():
    path = ROOT / "load-admission.json"
    if not path.is_file():
        return False
    proof = json.loads(path.read_text())
    return (proof.get("passed") is True and proof.get("ready_state_resource_check") is True
            and proof.get("official_exe_sha256") == digest(ROOT / "build/strata.exe")
            and proof.get("source_gates_sha256") == digest(ROOT / "source-gates.json")
            and proof.get("base_config_sha256") == digest(ROOT / "vision98304.json")
            and proof.get("profile") == admitted.PROFILE and proof.get("gpu") == 0
            and proof.get("pool_workers") == 8 and proof.get("device_visibility") == "0,1"
            and proof.get("gpu_start_floor_mib") == 0 and proof.get("gpu_floor_mib") == 0
            and proof.get("vram_reserve_mib") == 700)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("manual", "check"))
    parser.add_argument("--tag")
    parser.add_argument("--timeout", type=int, default=14400)
    opts = parser.parse_args()
    if not 60 <= opts.timeout <= 14400:
        raise ValueError("User startup+server lifetime must be60..14400s")
    tag = opts.tag or ("vision-manual-" + time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns()%1000000}")
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in tag):
        raise ValueError("Invalid evidence tag")
    validate_config(json.loads((ROOT / "vision98304.json").read_text()), 98304)
    report, _ = artifact_checks(admitted.PROFILE)  # Python/native/encoder gates and unchanged resource floors.
    if not current_smoke(4096):
        raise RuntimeError("Fresh4K image/text smoke is required; no model started")
    ready = current_load() and current_smoke(98304)
    if opts.mode == "check":
        print(json.dumps({"component_gates_passed": True, "resources_passed": report["minimum_checks_passed"],
                          "manual98k_admission_current": ready, "first_user_start_requires98k_checks": not ready,
                          "output_policy": "remaining-context-v1", "memory": report["measured_memory"],
                          "failures": report["failures"], "model_started": False}))
        return 0 if report["minimum_checks_passed"] else 1
    if not report["minimum_checks_passed"]:
        raise RuntimeError("Resource admission failed; no model started")
    deadline = time.monotonic() + opts.timeout

    def launch(mode, run_tag, maximum):
        remaining = int(deadline - time.monotonic())
        if remaining < 60:
            raise TimeoutError("Finite startup/server lifetime exhausted")
        command = [sys.executable, str(MANUAL / "vision_guard.py"), mode, "--context", "98304",
                   "--tag", run_tag, "--timeout", str(min(maximum, remaining))]
        # Same original guardian: strict source/load/smoke chain, exclusive
        # mutex, actual postREADY counters, owned Job/kill-on-close and floors.
        completed = subprocess.run(command)
        if completed.returncode:
            raise RuntimeError(f"Protected {mode} stopped with code{completed.returncode}; no gate bypass")

    if not ready:
        print("Primo avvio dopo modifica output: verifica protetta98K, poi server. Nessun gate ignorato.", flush=True)
        launch("load", tag + "-output98k-load", 600)
        launch("smoke", tag + "-output98k-smoke", 600)
        if not current_load() or not current_smoke(98304):
            raise RuntimeError("Fresh98K gates did not complete; manual API not started")
    launch("manual", tag, 14400)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        print("STRATA ARRESTATO: " + (str(error) or type(error).__name__), file=sys.stderr, flush=True)
        raise SystemExit(1)
