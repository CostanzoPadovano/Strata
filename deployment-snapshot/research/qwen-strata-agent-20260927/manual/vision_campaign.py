"""Separate vision identity/admission. Frozen text campaign remains untouched."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

MANUAL = Path(__file__).resolve().parent
TEXT = MANUAL.parent
ROOT = TEXT.parent / "qwen-strata-vision-20260927"
sys.path.insert(0, str(TEXT))
import agent_guard as admitted

ENCODER = TEXT.parent / "qwen-strata-stock-20260926/official-engine/strata-vision.exe"
ENCODER_SHA = "98ffde98f5388b518d8581b2983973d785f20f00a21e5e82393b38f4188e094c"
PROJECTOR = TEXT.parents[1] / "models/qwen38-vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"
PROJECTOR_SHA = "b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad"


def digest(path): return admitted.digest(Path(path))


def bound(path): return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def check_bound(row):
    if not Path(row["path"]).is_file() or digest(row["path"]) != row["sha256"]:
        raise RuntimeError(f"Vision bound file changed: {row['path']}")


def encoder_config():
    base = json.loads((TEXT / "configs/tier98304.json").read_text())
    return {"exe": str(ENCODER), "mmproj": str(PROJECTOR),
            "model": admitted._one_value(base["args"], "--native"),
            "gpu": False, "threads": 8, "max_tokens": 1024}


def make_config(context):
    if context not in (4096, 98304): raise ValueError("Only4K/98K vision context admitted")
    base = json.loads((TEXT / f"configs/tier{context}.json").read_text())
    admitted.validate_config(base, context)
    base["exe"] = str(ROOT / "build/strata.exe")
    base["cwd"] = str(ROOT / "source")
    # Profiles deliberately reuse the immutable exact same expert profile.
    base["args"] += ["--vision"]
    base["vision"] = encoder_config()
    base["vision_fixtures"] = str(ROOT / "fixtures")
    base["vision_engine_sha256"] = digest(ROOT / "build/strata.exe")
    return base


def validate_config(config, context):
    if config != make_config(context): raise ValueError("Vision config differs from exact bound profile")


def artifact_checks(profile):
    report, parent_gates = admitted.artifact_checks(profile)
    path = ROOT / "source-gates.json"
    gates = json.loads(path.read_text())
    if not gates.get("passed") or gates.get("parent_gates_sha256") != digest(TEXT / "source-gates.json"):
        raise RuntimeError("Vision source gates missing/stale")
    for row in gates["bound_files"]: check_bound(row)
    # New source/native tests are independently bound, never attributed to the
    # old executable. The parent gates still validate unchanged shared assets.
    native = json.loads((ROOT / "native-gates.json").read_text())
    if not native.get("passed") or native["engine_sha256"] != digest(ROOT / "build/strata.exe"):
        raise RuntimeError("Fresh vision native gates missing/stale")
    if any(not native.get("numerical_gates", {}).get(name, {}).get("passed") for name in ("mtp", "remote12")):
        raise RuntimeError("Fresh bounded MTP/selected12 component numerical gates must pass")
    for row in native["bound_files"]: check_bound(row)
    encoder = json.loads((ROOT / "encoder-admission.json").read_text())
    if not encoder.get("passed") or encoder.get("source_gates_sha256") != digest(path):
        raise RuntimeError("Fresh CPU encoder gate missing/stale")
    for row in encoder["bound_files"]: check_bound(row)
    # Conservative additional CPU encoder/scratch reservation on top of exact
    # compact expert arena. Measured peaks are recorded separately, not inferred.
    report["expert_arena_bytes"] += 2 * (1 << 30)
    report["executable_kind"] = "separately-bound-vision-static-owner"
    report["failures"] = []
    for key, floor in (("ram_free", 8), ("commit_free", 4)):
        if report["measured_memory"][key] < report["expert_arena_bytes"] + floor * (1 << 30):
            report["failures"].append(f"Insufficient {key} for expert arena+2GiB vision+floor")
    report["minimum_checks_passed"] = not report["failures"]
    return report, gates
