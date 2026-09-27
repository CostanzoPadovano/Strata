"""Bounded helper-death probe: NO model, no external process termination."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import debug_launcher as launcher
from debug_metrics import ParentWatcher


def main():
    tag = "probe-parent-exit-" + time.strftime("%Y%m%d-%H%M%S")
    run = launcher.HERE / "runs" / tag
    run.mkdir(parents=True, exist_ok=False)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    parent = subprocess.Popen([sys.executable, "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(90)"],
                              stdout=subprocess.PIPE, creationflags=flags)
    sampler = None
    actual_parent = None
    result = {"passed": False, "scope": "Own sleeping-helper exit, NO model/guardian"}
    try:
        actual_parent_pid = int(parent.stdout.readline(64))
        actual_parent = ParentWatcher(actual_parent_pid)
        result["parent_wrapper_pid"] = parent.pid
        result["parent_interpreter_pid"] = actual_parent_pid
        sampler = subprocess.Popen([
            sys.executable, str(launcher.SAMPLER), "--output", str(run),
            "--stop-file", str(run / "stop-monitor"), "--duration", "60",
            "--interval", "2", "--parent-pid", str(parent.pid)], creationflags=flags)
        launcher.wait_for_sampler(sampler, run / "ready.json")
        triggered = time.monotonic()
        parent.terminate()  # Exact newly-created sleeping helper only.
        parent.wait(timeout=5)
        # Windows closes the redirector Job asynchronously after its exit.
        interpreter_deadline = time.monotonic() + 5
        while not actual_parent.exited() and time.monotonic() < interpreter_deadline:
            time.sleep(.05)
        result["parent_interpreter_exited"] = actual_parent.exited()
        sampler.wait(timeout=30)
        result["tail_wall_seconds"] = time.monotonic() - triggered
        summary = launcher.small_json(run / "summary.json")
        result["reason"] = summary.get("reason")
        result["sample_count"] = summary.get("sample_count")
        result["sampler_exit_code"] = sampler.returncode
        result["passed"] = (sampler.returncode == 0 and result["reason"] == "parent_exit"
                            and result["parent_interpreter_exited"]
                            and 19 <= result["tail_wall_seconds"] <= 29)
    finally:
        for child in (sampler, parent):
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
        if actual_parent is not None:
            actual_parent.close()
        if parent.stdout is not None:
            parent.stdout.close()
        launcher.write_record(run / "probe-result.json", result)
    print(json.dumps({"run": str(run), **result}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
