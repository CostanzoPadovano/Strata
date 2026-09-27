"""Read-only bounded diagnostics around the unchanged, gated vision guardian."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
VISION = PROJECT / "research/qwen-strata-vision-20260927"
GUARD = PROJECT / "research/qwen-strata-agent-20260927/manual/vision_user_launcher.py"
SAMPLER = HERE / "debug_metrics.py"
MAX_RECORD = 64 * 1024
GUARD_SECONDS = 14400


def write_record(path: Path, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=True, indent=2)
    if len(data.encode("utf-8")) > MAX_RECORD:
        raise ValueError("Diagnostic record exceeds finite cap")
    path.write_text(data + "\n", encoding="utf-8")


def small_json(path: Path) -> dict:
    with path.open("rb") as source:
        data = source.read(MAX_RECORD + 1)
    if len(data) > MAX_RECORD:
        raise ValueError("Diagnostic input exceeds finite cap")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("Diagnostic input must be an object")
    return result


def small_hash(path: Path) -> str:
    with path.open("rb") as source:
        data = source.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Only small configuration/source identities are hashed")
    return hashlib.sha256(data).hexdigest()


def guard_command(tag: str, check: bool = False) -> list[str]:
    if not tag.startswith("debug-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in tag):
        raise ValueError("Invalid debug evidence tag")
    return [sys.executable, str(GUARD), "check" if check else "manual",
            "--tag", tag, "--timeout", str(GUARD_SECONDS)]


def guardian_summary(run: Path) -> dict:
    """Stream only known numeric counters, never copy observer/prompt content."""
    report = {"scope": "Actual owned guardian Job commit, NOT physical RAM or pagefile",
              "job_peak_bytes": None, "ram_free_min_bytes": None,
              "commit_free_min_bytes": None, "rows": 0, "partial": False}
    telemetry = run / "telemetry.jsonl"
    if telemetry.exists():
        scanned = 0
        with telemetry.open("rb") as source:
            while report["rows"] < 100000:
                line = source.readline(MAX_RECORD + 1)
                if not line:
                    break
                scanned += len(line)
                if len(line) > MAX_RECORD or scanned > 64 * 1024 * 1024:
                    report["partial"] = True
                    break
                try:
                    row = json.loads(line)
                except ValueError:
                    report["partial"] = True
                    continue
                report["rows"] += 1
                for key, source_key, operation in (
                    ("job_peak_bytes", "peak_job_bytes", max),
                    ("ram_free_min_bytes", "ram_free", min),
                    ("commit_free_min_bytes", "commit_free", min),
                ):
                    value = row.get(source_key)
                    if isinstance(value, (int, float)) and value >= 0:
                        previous = report[key]
                        report[key] = value if previous is None else operation(previous, value)
            else:
                report["partial"] = True
    if (run / "result.json").exists():
        try:
            result = small_json(run / "result.json")
            report["guard_passed"] = result.get("passed") is True
            report["model_started"] = result.get("model_started") is True
            report["observer_exit_code"] = result.get("observer_exit_code")
        except (OSError, ValueError):
            report["partial"] = True
    return report


def request_guard_stop(run: Path, guard: subprocess.Popen) -> None:
    """Only the exact child started here; guardian retains Job kill-on-close."""
    if run.is_dir():
        (run / "stop-server").touch(exist_ok=True)
    try:
        guard.wait(timeout=35)
    except subprocess.TimeoutExpired:
        # Terminate this owned guardian handle, never enumerate/kill model names.
        guard.terminate()
        guard.wait(timeout=10)


def wait_for_sampler(sampler: subprocess.Popen, ready: Path, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sampler.poll() is not None:
            raise RuntimeError("Monitor failed before launch; model NOT started")
        if ready.exists():
            handshake = small_json(ready)
            owned_pid = (handshake.get("pid") == sampler.pid
                         or handshake.get("parent_pid") == sampler.pid)
            if handshake.get("ready") is not True or not owned_pid:
                raise RuntimeError("Invalid monitor identity; model NOT started")
            with (ready.parent / "samples.jsonl").open("rb") as source:
                first_line = source.readline(MAX_RECORD + 1)
            if len(first_line) > MAX_RECORD:
                raise RuntimeError("Monitor startup record too large; model NOT started")
            memory = json.loads(first_line).get("memory")
            if not isinstance(memory, dict) or any(
                type(memory.get(key)) not in (int, float) or memory[key] < 0
                for key in ("ram_total_bytes", "ram_available_bytes",
                            "windows_commit_limit_bytes", "windows_commit_available_bytes")
            ):
                raise RuntimeError("Essential RAM/commit metrics missing; model NOT started")
            if memory["ram_total_bytes"] <= 0 or memory["windows_commit_limit_bytes"] <= 0:
                raise RuntimeError("Invalid RAM/commit capacity; model NOT started")
            return
        time.sleep(.1)
    raise TimeoutError("Monitor handshake timed out; model NOT started")


def monitored_guard(tag: str, guard_run: Path, sampler: subprocess.Popen, check: bool) -> int:
    guard = subprocess.Popen(guard_command(tag, check), cwd=str(PROJECT))
    try:
        while guard.poll() is None:
            if sampler.poll() is not None:
                request_guard_stop(guard_run, guard)
                raise RuntimeError("Monitor ended unexpectedly; owned server stopped")
            time.sleep(.25)
        return guard.returncode
    except BaseException:
        if guard.poll() is None:
            request_guard_stop(guard_run, guard)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Metrics and original gates only; NO model")
    parser.add_argument("--monitor-only", action="store_true", help="Metrics only; NO model or guardian")
    parser.add_argument("--seconds", type=int, default=60, help="Monitor-only duration,10..600s")
    parser.add_argument("--baseline-seconds", type=int, default=15, help="Before guardian,0..60s")
    parser.add_argument("--interval", type=int, default=5, help="Light sampler interval,2..30s")
    opts = parser.parse_args()
    if opts.check and opts.monitor_only:
        parser.error("Choose --check OR --monitor-only")
    if not 10 <= opts.seconds <= 600 or not 0 <= opts.baseline_seconds <= 60 or not 2 <= opts.interval <= 30:
        parser.error("Parameters exceed finite diagnostic bounds")
    if os.name != "nt":
        parser.error("Windows native diagnostic launcher required")
    if not SAMPLER.is_file() or (not opts.monitor_only and not GUARD.is_file()):
        raise FileNotFoundError("Required diagnostic/guardian script missing; model NOT started")

    tag = "debug-" + time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1000000:06d}"
    run = HERE / "runs" / tag
    run.mkdir(parents=True, exist_ok=False)
    guard_run = VISION / "runs" / tag
    mode = "monitor-only" if opts.monitor_only else "check" if opts.check else "manual"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
                "guard_run": str(guard_run), "debug_run": str(run), "interval_seconds": opts.interval,
                "baseline_seconds": opts.baseline_seconds, "guardian_max_seconds": GUARD_SECONDS,
                "python_version": sys.version.split()[0], "logical_cpus": os.cpu_count(),
                "model": "existing ISTA GSQ-RCO IQ3_XXS", "context_capacity": 98304,
                "protections_unchanged": {"ram_free_floor_gib": 8, "commit_free_floor_gib": 4,
                                          "job_commit_cap_gib": 60, "gpu_temperature_limit_c": 80,
                                          "gpu_free_floor_mib": 0},
                "privacy": "Metrics/names only; no commandlines, API keys, prompts or images",
                "purpose": "Diagnostic timeline; not speed/quality/daily-use acceptance"}
    manifest["debug_source_sha256"] = {path.name: small_hash(path) for path in (Path(__file__), SAMPLER)}
    if not opts.monitor_only:
        manifest["identities"] = {str(path.relative_to(PROJECT)): small_hash(path) for path in (
            GUARD, PROJECT / "run_qwen38_98k_ista_strata_server.bat",
            VISION / "vision98304.json", VISION / "source-gates.json",
            VISION / "vision-admission-98304.json")}
        manifest["engine_sha256"] = small_json(VISION / "vision98304.json")["vision_engine_sha256"]
    write_record(run / "manifest.json", manifest)
    print(f"DEBUG: log in {run}", flush=True)
    print("RAM/commit/paging/CPU/disco ogni5s circa; GPU10s, processi15s. Nessun prompt registrato.", flush=True)
    stop_file = run / "stop-monitor"
    duration = opts.seconds if opts.monitor_only else GUARD_SECONDS + opts.baseline_seconds + 60
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    sampler = subprocess.Popen(
        [sys.executable, str(SAMPLER), "--output", str(run), "--stop-file", str(stop_file),
         "--duration", str(duration), "--interval", str(opts.interval),
         "--parent-pid", str(os.getpid()), "--observe-run", str(guard_run)],
        cwd=str(HERE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    result = {"mode": mode, "guard_started": False, "guard_exit_code": None,
              "monitor_exit_code": None, "passed": False}
    try:
        wait_for_sampler(sampler, run / "ready.json")
        if opts.monitor_only:
            sampler.wait(timeout=opts.seconds + 30)
        else:
            print(f"Baseline: {opts.baseline_seconds}s senza modello. Lascia il PC a riposo.", flush=True)
            deadline = time.monotonic() + opts.baseline_seconds
            while time.monotonic() < deadline:
                if sampler.poll() is not None:
                    raise RuntimeError("Monitor failed during baseline; model NOT started")
                time.sleep(.25)
            result["guard_started"] = True
            result["guard_exit_code"] = monitored_guard(tag, guard_run, sampler, opts.check)
        result["passed"] = result["guard_exit_code"] in (None, 0)
    except BaseException as error:
        result["failure"] = type(error).__name__ + ": " + str(error)[:512]
        print("DEBUG ARRESTATO: " + result["failure"], file=sys.stderr, flush=True)
    finally:
        stop_file.touch(exist_ok=True)
        try:
            sampler.wait(timeout=30)
        except subprocess.TimeoutExpired:
            sampler.terminate()  # Exact owned diagnostic helper, never another application.
            sampler.wait(timeout=10)
            result["passed"] = False
            result["monitor_forced_stop"] = True
        result["monitor_exit_code"] = sampler.returncode
        result["passed"] = result["passed"] and sampler.returncode == 0
        result["guardian_metrics"] = guardian_summary(guard_run)
        write_record(run / "launcher-result.json", result)
        print(f"Log salvati: {run}\nsamples.jsonl, summary.json, manifest.json, launcher-result.json", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
