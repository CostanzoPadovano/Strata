"""Encoder-only evidence under a separate 4GiB Job; opens text vocab, not weights."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

from bounded_vision import BoundedVision, validate_sve

run = Path(sys.argv[1]).resolve()
deadline = time.monotonic() + 30
while not (run / "job-attached").exists():
    if time.monotonic() > deadline: raise TimeoutError("No owned encoder Job")
    time.sleep(.05)
cfg = json.loads((run / "encoder-config.json").read_text())
result = {"passed": False, "scope": "CPU encoder/vocab-only, not full LLM"}
vision = None
with (run / "encoder.log").open("x", encoding="utf-8") as log:
    try:
        vision = BoundedVision(cfg["vision"], run / "scratch", log)
        rows = []
        result["encoder_pids"] = []
        for name in ("red.png", "blue.png", "max.png", "red.png"):
            file = Path(cfg["fixtures"]) / name
            source = "data:image/png;base64," + base64.b64encode(file.read_bytes()).decode()
            path, n = vision.encode(source)
            if vision.proc is not None:
                result["encoder_pids"].append(vision.proc.pid)
            rows.append({"fixture": name, "tokens": n, "grid": validate_sve(path),
                         "sve_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
            vision.park()
            if vision.proc is not None or vision.failed or vision.last_exit_code != 0:
                raise AssertionError("Encoder did not park cleanly between images")
        if rows[0]["sve_sha256"] != rows[-1]["sve_sha256"] or rows[0]["sve_sha256"] == rows[1]["sve_sha256"]:
            raise AssertionError("Encoder cache/contrast failed")
        if rows[2]["tokens"] != 1024:
            raise AssertionError("Largest-image allocation was not exercised at1024tokens")
        if vision.stats["starts"] != 3 or vision.stats["hits"] != 1:
            raise AssertionError("Cold restarts/cache-hit lifecycle not exercised")
        result.update(rows=rows, stats=vision.stats)
        result["passed"] = True
    except BaseException as error:
        result["failure"] = str(error)
        raise
    finally:
        if vision:
            vision.close()
            result["encoder_exit_code"] = vision.last_exit_code
            result["passed"] &= vision.last_exit_code == 0
        (run / "probe.json").write_text(json.dumps(result, indent=2))
