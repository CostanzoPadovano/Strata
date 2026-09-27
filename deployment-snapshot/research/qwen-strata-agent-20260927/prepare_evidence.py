"""Create fresh source-gate evidence or validate one generated tier config."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import agent_guard as guard
import numerical_guard as numerical

ROOT = guard.ROOT
BUILD = ROOT / "build"
OLD_RUNTIME = ROOT.parent / "qwen-strata-20260926"
STOCK = ROOT.parent / "qwen-strata-stock-20260926"

SOURCE_SPECS = {
    "transport": ("mtp_transport_test.exe", "PASS 64 staged copies", 120, False, ()),
    "registration": ("host_registration_test.exe",
                     "PASS bounded registration: 3 arenas, 6 device reads, locked suffix and cleanup; no GGUF",
                     60, True, ()),
    "embedding": ("embedding_cache_test.exe", "PASS 64 embedding graph bit checks", 60, False, ()),
    "compacted_loader": ("compacted_loader_test.exe",
                         "PASS compacted_loader bytes=768 layers=2 guards=exact negative_cases=4", 60, True,
                         ("PASS selected_host_layout payload_bytes=80 layers=3 scatter=exact native_roles=exact aliases=exact negatives=exact",
                          "PASS selected_host_layout6 ids=1,3,5,7,9,11 holes=6 retained=6 scatter=exact native_roles=exact aliases=exact negatives=exact",
                          "PASS selected_host_layout12 ids=1,3,5,7,9,11,13,15,17,19,21,23 holes=12 retained=12 scatter=exact aliases=exact negatives=duplicate,unsorted,out_of_range,scatter_bounds,short_alias",
                          "PASS selected_host_pair_holes remote_layers=2 cached_pairs=6 fully_cached_layers=1 expert0_hole=exact accounting=exact packed_scatter=exact native_roles=exact registration=single_prefix_locked_suffix crossing_alias=reject cold_reads=repeat_exact truncated=reject_unchanged negatives=overlap,duplicate,unsorted,out_of_range,corrupt_accounting,unsafe_sibling",
                          "PASS expert_cache_null_source devices=2 refusal=clean valid_after_refusal=exact")),
}


def _runtime_bindings() -> list[dict]:
    """Freeze the reused pack/MTP runtime without hashing either huge GGUF."""
    manifest_path = OLD_RUNTIME / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [{"path": str(manifest_path), "sha256": guard.digest(manifest_path)}]
    for row in manifest["runtime_files"]:
        normalized = row["path"].replace("\\", "/")
        if normalized.startswith(("pack-iq3/", "mtp/rt/")):
            path = OLD_RUNTIME / row["path"]
            if guard.digest(path) != row["sha256"]:
                raise RuntimeError(f"Reused runtime identity mismatch: {path}")
            rows.append({"path": str(path), "sha256": row["sha256"]})
    profile = ROOT / "source" / "data" / "expert-profile.bin"
    rows.append({"path": str(profile), "sha256": guard.digest(profile)})
    inherited = ROOT / "inherited-performance-binding.json"
    rows.append({"path": str(inherited), "sha256": guard.digest(inherited)})
    for name in ("agent_guard.py", "numerical_guard.py", "prepare_evidence.py",
                 "owned_memory.py", "native_probe.py", "memory-authorization.json", "protocol.md"):
        path = ROOT / name
        rows.append({"path": str(path), "sha256": guard.digest(path)})
    # The earlier component suites are baseline only, never substitutes for
    # the four fresh gates above.
    for name in ("source-gates-gpu0-advanced01.json", "source-gates.json"):
        path = STOCK / name
        proof = json.loads(path.read_text())
        if not proof.get("passed"):
            raise RuntimeError("Inherited component baseline is not passing")
        rows.append({"path": str(path), "sha256": guard.digest(path)})
    return rows


def ensure_tier_config(context: int) -> dict:
    path = ROOT / "configs" / f"tier{context}.json"
    if not path.exists():
        base_path = ROOT / "configs" / "tier4096.json"
        base = json.loads(base_path.read_text())
        guard.validate_config(base, 4096)
        args = list(base["args"]); args[args.index("--max-context") + 1] = str(context)
        base["args"] = args
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(base, stream, indent=2)
    config = json.loads(path.read_text())
    guard.validate_config(config, context)
    return {"passed": True, "context": context, "path": str(path), "sha256": guard.digest(path)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag"); p.add_argument("--config-only", action="store_true")
    p.add_argument("--context", type=int, choices=guard.LADDER, default=4096)
    a = p.parse_args()
    if a.config_only:
        print(json.dumps(ensure_tier_config(a.context))); return
    if not a.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in a.tag):
        raise ValueError("A valid unique --tag is required")
    binding, binding_path = numerical.load_binding()
    config_proof = ensure_tier_config(4096)
    gate_root = ROOT / "gate-runs" / a.tag
    if gate_root.exists(): raise FileExistsError(gate_root)
    (ROOT / "source-gates.json").write_text(json.dumps({"passed": False,
        "invalidated_by": a.tag, "build_binding_sha256": guard.digest(binding_path)}, indent=2))
    results = {}
    for name, (exe_name, marker, timeout, exact, required) in SOURCE_SPECS.items():
        command = [str(BUILD / exe_name)]
        fixture = None
        if name == "compacted_loader":
            fixture = gate_root / name / "loader-fixture.bin"
            command += ["--fixture", str(fixture)]
        result = numerical.run_native(command, gate_root / name, marker, timeout, 8,
                                      exact_marker=exact, required_markers=required)
        if fixture is not None:
            result["fixture"] = {"path": str(fixture), "sha256": guard.digest(fixture)}
        results[name] = result
    bound = _runtime_bindings()
    bound += [{"path": str(binding_path), "sha256": guard.digest(binding_path)}, config_proof]
    for path in (ROOT / "agent_guard.py", ROOT / "numerical_guard.py", ROOT / "prepare_evidence.py",
                 ROOT / "native_probe.py", ROOT / "owned_memory.py", ROOT / "build.ps1",
                 ROOT / "memory-authorization.json", ROOT / "protocol.md"):
        if not path.is_file(): raise RuntimeError(f"Required campaign file is missing: {path.name}")
        bound.append({"path": str(path), "sha256": guard.digest(path)})
    for name, result in results.items():
        bound.extend(result["bound_evidence"])
        if "fixture" in result: bound.append(result["fixture"])
    proof = {"passed": True, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "source_revision": binding["source_base"], "ggml_revision": binding["ggml_revision"],
             "cuda": binding["cuda"], "architecture": binding["architecture"],
             "executable_sha256": binding["executable_sha256"],
             "build_binding_sha256": guard.digest(binding_path),
             "fresh_native_gates": results, "bound_files": bound,
             "reused_component_results_are_baseline_only": True,
             "full_gguf_loaded": False,
             "evidence_scope": "Fresh two-GPU transport, bounded registration, embedding and compacted 12-hole loader gates; runtime/profile identity. No GGUF load, generation or MTP/expert numerical claim."}
    (gate_root / "gates.json").write_text(json.dumps(proof, indent=2))
    (ROOT / "source-gates.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps({"passed": True, "source_gates": str(ROOT / "source-gates.json")}))


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"STRATA EVIDENCE STOP: {error}", file=sys.stderr, flush=True); raise SystemExit(1)
