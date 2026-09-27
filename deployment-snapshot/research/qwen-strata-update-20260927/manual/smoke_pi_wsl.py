"""Capture bounded actual global Pi tool smoke; never reads auth or changes config."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
opts = parser.parse_args()
run = opts.run.resolve()
campaign = Path(__file__).resolve().parents[1]
if not run.is_relative_to(campaign / "runs") or not (run / "ready-accepted.json").exists():
    raise ValueError("An active admitted manual run is required")
script = ("cd /mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-agent-20260927/manual; "
          "/home/costapad/.local/bin/pi --mode json --no-session --tools read --no-skills --no-prompt-templates -p "
          "'Usa read per leggere smoke-fixture.txt nella directory corrente. Rispondi solo con il valore di "
          "STRATA_GLOBAL_PI_OK. Non eseguire altre operazioni.'")
trace = run / "pi-global-smoke.jsonl"
if trace.exists():
    raise FileExistsError("Immutable smoke evidence already exists")
process = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "costapad", "--", "bash", "-lc", script],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                           creationflags=subprocess.CREATE_NO_WINDOW)
try:
    out, err = process.communicate(timeout=120)
except subprocess.TimeoutExpired:
    process.kill()
    process.communicate(timeout=10)
    raise
if len(out.encode()) > 256 * 1024 or len(err.encode()) > 16 * 1024:
    raise ValueError("Smoke trace exceeds256KiB/16KiB evidence bounds")
trace.write_text(out, encoding="utf-8")
(run / "pi-global-smoke.stderr.log").write_text(err, encoding="utf-8")
events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
messages = [e["message"] for e in events if e.get("type") == "message_end"]
assistants = [m for m in messages if m.get("role") == "assistant"]
tools = [m for m in messages if m.get("role") == "toolResult"]
final = "".join(c.get("text", "") for c in assistants[-1].get("content", []) if c.get("type") == "text") if assistants else ""
checks = {"exit0": process.returncode == 0,
          "same_provider": bool(assistants) and all(m.get("provider") == "local-qwen38" for m in assistants),
          "same_model": bool(assistants) and all(m.get("model") == "qwen3.8-flash-next-local" for m in assistants),
          "actual_read": any(m.get("toolName") == "read" and not m.get("isError") for m in tools),
          "final_value": final.strip() == "ORCHID-7314",
          "final_stop": bool(assistants) and assistants[-1].get("stopReason") == "stop",
          "no_api_error": all(m.get("stopReason") != "error" for m in assistants)}
result = {"passed": all(checks.values()), "checks": checks, "final": final,
          "calls": [{"usage": m.get("usage"), "stopReason": m.get("stopReason")} for m in assistants]}
(run / "pi-global-smoke-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
if not result["passed"]:
    raise SystemExit(1)
