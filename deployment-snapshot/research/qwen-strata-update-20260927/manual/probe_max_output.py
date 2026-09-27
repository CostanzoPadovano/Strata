"""Bounded live API probes: actual >1024 output, exact budget, cancellation."""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1].parent / "qwen-strata-vision-20260927/runs"
run = args.run.resolve()
if not run.is_relative_to(root) or not (run / "ready-accepted.json").exists():
    raise ValueError("An active owned vision run is required")
result_path = run / "output-budget-probe.json"
if result_path.exists():
    raise FileExistsError("Immutable probe evidence already exists")
key = Path(r"C:\llama_official\router\api-key.txt").read_text().strip()
headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
base = "http://127.0.0.1:8037"
model = "qwen3.8-flash-next-local"


def get(path):
    with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers), timeout=10) as response:
        return json.loads(response.read(8192))


def stream(prompt, cancel_after_content=False):
    body = {"model": model, "stream": True, "reasoning_effort": "minimal", "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    text, finish, usage, seen_bytes = "", None, None, 0
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        while True:
            if time.monotonic() - started > 180:
                raise TimeoutError("Finite API probe exceeded180s")
            line = response.readline(65537)
            if not line:
                break
            seen_bytes += len(line)
            if len(line) > 65536 or seen_bytes > 2 * (1 << 20):
                raise ValueError("Probe exceeds64KiB/line or2MiB response")
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                break
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text += choice.get("delta", {}).get("content", "")
                finish = choice.get("finish_reason") or finish
            if cancel_after_content and len(text) >= 16:
                return {"client_closed_after_content": True, "received_chars": len(text),
                        "wall_seconds": time.monotonic() - started}
    return {"content": text, "finish": finish, "usage": usage, "wall_seconds": time.monotonic() - started}


result = {"passed": False, "scope": "New output policy API only; not full98K-generation/quality certification"}
try:
    health = get("/health")
    assert health["max_output_tokens"] == 98296 and health["output_policy"] == "remaining-context-v1"
    short = stream("Rispondi soltanto BUDGET_OK senza altro testo.")
    result["short"] = short
    assert short["finish"] == "stop" and short["content"].strip() == "BUDGET_OK"
    long = stream("Scrivi tutti i numeri interi da 0 a 599 inclusi, in ordine crescente, separati solo da virgole. "
                  "Non usare ellissi, abbreviazioni, blocchi di codice, commenti o spiegazioni. Scrivi tutti i600numeri e poi fermati.")
    numbers = [int(x) for x in re.findall(r"\d+", long["content"])]
    result["long"] = {k: v for k, v in long.items() if k != "content"}
    result["long"].update(csv_correct=numbers == list(range(600)), content_chars=len(long["content"]))
    assert long["finish"] == "stop" and numbers == list(range(600))
    assert long["usage"]["completion_tokens"] > 1024
    result["cancel"] = stream("Conta da0 in avanti fino a100000, separando tutti i numeri con virgole. "
                              "Non usare abbreviazioni o ellissi e non fermarti prima.", cancel_after_content=True)
    deadline = time.monotonic() + 15
    while get("/status").get("busy"):
        if time.monotonic() >= deadline:
            raise TimeoutError("Native generation did not stop after client disconnect")
        time.sleep(.2)
    result["cancel"]["server_idle_after_disconnect"] = True
    after = stream("Rispondi soltanto STOP_OK senza altro testo.")
    result["after_cancel"] = after
    assert after["finish"] == "stop" and after["content"].strip() == "STOP_OK"
    requests = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
    for row in requests[-4:]:
        assert row["requested_max_tokens"] == 98296
        assert row["max_tokens"] == 98304 - row["prompt_tokens"] - 8
    result["exact_budget_rows"] = requests[-4:]
    result["passed"] = True
except BaseException as error:
    result["failure"] = type(error).__name__ + ": " + str(error)[:512]
    raise
finally:
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "failure": result.get("failure"),
                      "long": result.get("long"), "evidence": str(result_path)}))
