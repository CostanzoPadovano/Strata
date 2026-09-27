"""Generate deterministic fixtures and immutable Python/encoder gate identity."""
import json
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont
from vision_campaign import (ROOT, MANUAL, TEXT, ENCODER, ENCODER_SHA, PROJECTOR, PROJECTOR_SHA,
                             bound, digest, encoder_config, make_config)

ROOT.mkdir(exist_ok=True)
native = json.loads((ROOT / "native-gates.json").read_text())
if not native.get("passed") or any(not native.get("numerical_gates", {}).get(name, {}).get("passed") for name in ("mtp", "remote12")):
    raise RuntimeError("Complete fresh native/component gates first")
fixtures = ROOT / "fixtures"
fixtures.mkdir(exist_ok=True)
font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 110)
for name, color, code in (("red.png", "#ff3333", "ORCHID-7314"), ("blue.png", "#2255ff", "COBALT-4428"),
                          ("alt-red.png", "#ff3333", "EMBER-9056")):
    target = fixtures / name
    if not target.exists():
        img = Image.new("RGB", (1024, 512), color)
        draw = ImageDraw.Draw(img)
        draw.rectangle((30, 120, 994, 390), fill="white")
        draw.text((512, 256), code, fill="black", font=font, anchor="mm")
        img.save(target, "PNG")
if not (fixtures / "max.png").exists():
    Image.new("RGB", (2048, 2048), "#cccccc").save(fixtures / "max.png", "PNG")
if digest(ENCODER) != ENCODER_SHA or PROJECTOR.stat().st_size != 616703104 or digest(PROJECTOR) != PROJECTOR_SHA:
    raise RuntimeError("Existing official encoder/projector binding mismatch")
result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(MANUAL), "-p", "test_*.py"],
                        capture_output=True, text=True, timeout=30)
log = ROOT / f"python-gates-{time.time_ns()}.log"
log.write_text(result.stdout + result.stderr, encoding="utf-8")
if result.returncode: raise RuntimeError("Vision Python gates failed; see " + str(log))
for context in (4096, 98304):
    config = ROOT / f"vision{context}.json"
    config.write_text(json.dumps(make_config(context), indent=2), encoding="utf-8")
sources = [MANUAL / name for name in ("bounded_vision.py", "test_vision.py", "test_manual.py", "test_output_tokenizer.py", "test_output_lifecycle.py", "manual_server.py",
           "manual_bridge.mjs", "vision_campaign.py", "vision_guard.py", "vision_encoder_probe.py", "prepare_vision.py",
           "pi_route.mjs", "pi_strata.mjs", "test_pi_strata.mjs", "test_pi_route.mjs", "test_pi_overlay_wsl.mjs", "test_pi_output_wsl.mjs",
           "smoke_pi_vision.py", "stop_pi_vision.py", "capture_vision_memory.ps1", "verify_api.py")]
files = sources + [ENCODER, PROJECTOR, log, ROOT / "native-gates.json", ROOT / "vision4096.json", ROOT / "vision98304.json"] + sorted(fixtures.glob("*.png"))
gate = {"passed": True, "scope": "Bounded CPU vision adapter/unit tests; parent text gates retained separately",
        "parent_gates_sha256": digest(TEXT / "source-gates.json"), "bound_files": [bound(path) for path in files]}
(ROOT / "source-gates.json").write_text(json.dumps(gate, indent=2))
print(json.dumps({"python_gates_passed": True, "encoder_sha256": ENCODER_SHA,
                  "projector_sha256": PROJECTOR_SHA, "source_gates_sha256": digest(ROOT / "source-gates.json")}))
