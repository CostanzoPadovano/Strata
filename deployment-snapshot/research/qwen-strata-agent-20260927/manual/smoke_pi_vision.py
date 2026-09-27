"""Real global WSL Pi read-image + tool/text retention, no sessions/config writes."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
opts = parser.parse_args()
run = opts.run.resolve()
vision_root = Path(__file__).resolve().parents[1].parent / "qwen-strata-vision-20260927"
if not run.is_relative_to(vision_root / "runs") or not (run / "ready-accepted.json").exists():
    raise ValueError("Active separately admitted vision run required")
script = ("cd /mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-agent-20260927/manual; "
          "/home/costapad/.local/bin/pi --mode json --no-session --tools read --no-skills --no-prompt-templates -p "
          "'Usa read per aprire ../../qwen-strata-vision-20260927/fixtures/alt-red.png e ../../qwen-strata-vision-20260927/fixtures/blue.png. "
          "Trascrivi il codice grande al centro di ciascuna immagine. Usa anche read per leggere smoke-fixture.txt. "
          "Alla fine rispondi soltanto con JSON {red:codice immagine rossa,blue:codice immagine blu,text:valore STRATA_GLOBAL_PI_OK}. "
          "Non eseguire altre operazioni.'")
trace = run / "pi-vision.jsonl"
if trace.exists(): raise FileExistsError("Vision smoke evidence already exists")
process = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "costapad", "--", "bash", "-lc", script],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                           creationflags=subprocess.CREATE_NO_WINDOW)
try:
    out, err = process.communicate(timeout=240)
except subprocess.TimeoutExpired:
    process.kill()
    process.communicate(timeout=10)
    raise
if len(out.encode()) > 1024 * 1024 or len(err.encode()) > 16 * 1024:
    raise ValueError("Vision smoke evidence exceeds1MiB/16KiB")
trace.write_text(out, encoding="utf-8")
(run / "pi-vision.stderr.log").write_text(err, encoding="utf-8")
events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
messages = [e["message"] for e in events if e.get("type") == "message_end"]
assistants = [m for m in messages if m.get("role") == "assistant"]
tools = [m for m in messages if m.get("role") == "toolResult"]
final = "".join(c.get("text", "") for c in assistants[-1].get("content", []) if c.get("type") == "text") if assistants else ""
try: payload = json.loads(final.strip().removeprefix("```json").removesuffix("```").strip())
except ValueError: payload = None
images = [c for m in tools for c in m.get("content", []) if c.get("type") == "image"]
checks = {"exit0": process.returncode == 0,
          "same_provider_model": bool(assistants) and all(m.get("provider") == "local-qwen38" and m.get("model") == "qwen3.8-flash-next-local" for m in assistants),
          "actual_read_images": len(images) == 2,
          "actual_read_text": any(m.get("toolName") == "read" and not m.get("isError") and any(c.get("type") == "text" and "STRATA_GLOBAL_PI_OK" in c.get("text", "") for c in m.get("content", [])) for m in tools),
          "correct_visual_text": payload == {"red": "EMBER-9056", "blue": "COBALT-4428", "text": "ORCHID-7314"},
          "final_stop": bool(assistants) and assistants[-1].get("stopReason") == "stop",
          "no_api_error": all(m.get("stopReason") != "error" for m in assistants)}
result = {"passed": all(checks.values()), "checks": checks, "final": final,
          "calls": [{"usage": m.get("usage"), "stopReason": m.get("stopReason")} for m in assistants]}
(run / "pi-vision-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
if not result["passed"]: raise SystemExit(1)
