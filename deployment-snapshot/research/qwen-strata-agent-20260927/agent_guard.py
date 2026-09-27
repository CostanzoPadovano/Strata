"""Fail-closed guardian for the isolated Strata >=90K context ladder."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STOCK = ROOT.parent / "qwen-strata-stock-20260926"
OLD = ROOT.parent / "qwen-strata-advanced-20260926"
sys.path.insert(0, str(STOCK))
import stock_guard as shared
from owned_memory import current_owned_memory

_stock_job_counters = shared.job_counters


def measured_job_counters(job):
    # Current private memory is additive telemetry, not a replacement for global commit or Job peaks.
    return {**_stock_job_counters(job), **current_owned_memory(job)}

GIB = 1 << 30
PROFILE = "agent-90k-experimental"
LADDER = (4096, 8192, 16384, 32768, 65536, 98304)
REMOTE_IDS = (28, 29, 30, 32, 34, 35, 37, 42, 43, 44, 46, 47)
REMOTE_PAYLOAD = 13_526_630_400
REMOTE_ALLOCATION = 14_027_026_728
TOTAL_EXPERT_PAYLOAD = 42_912_972_800
MAX_BLOB = 2_329_600
GPU0_CACHED_PAYLOAD = 5_569_024_000
COMPACT_HOST_PAYLOAD = 23_817_318_400
COMPACT_ARENA_LOWER_BOUND = COMPACT_HOST_PAYLOAD + MAX_BLOB
SOURCE_GATE_NAMES = ("transport", "registration", "embedding", "compacted_loader")


def digest(path: Path) -> str:
    return shared.digest(Path(path))


def policy(name: str, vram_reserve_mib=None) -> dict:
    auth = json.loads((ROOT / "memory-authorization.json").read_text())
    if name != PROFILE or auth.get("authorized_profile") != PROFILE:
        raise ValueError("Only the explicitly authorized agent profile is supported")
    if vram_reserve_mib is not None:
        raise ValueError("VRAM reserve overrides are not part of this fixed profile")
    expected = {"ram_floor_gib": 8, "commit_floor_gib": 4, "job_cap_gib": 60,
                "gpu_start_floor_mib": 0, "gpu_floor_mib": 0}
    if any(auth.get("limits", {}).get(k) != v for k, v in expected.items()):
        raise RuntimeError("Memory authorization was changed or relaxed")
    return {**expected, "vram_reserve_mib": 700}


def _positions(args: list[str], flag: str) -> list[int]:
    return [i for i, value in enumerate(args) if value == flag]


def _one_value(args: list[str], flag: str) -> str:
    pos = _positions(args, flag)
    if len(pos) != 1 or pos[0] + 1 >= len(args) or args[pos[0] + 1].startswith("--"):
        raise ValueError(f"{flag} must occur exactly once with a value")
    return args[pos[0] + 1]


def validate_config(config: dict, expected_context: int | None = None) -> int:
    """Reject drift, duplicate flags, wrong engine and unsupported contexts."""
    expected_exe = (ROOT / "build" / "strata.exe").resolve()
    expected_cwd = (ROOT / "source").resolve()
    if Path(config.get("exe", "")).resolve() != expected_exe:
        raise ValueError("Configuration does not select the bound Strata engine")
    if Path(config.get("cwd", "")).resolve() != expected_cwd:
        raise ValueError("Configuration has the wrong source/runtime directory")
    if config.get("strategy") != "static-gpu-ownership":
        raise ValueError("Configuration has the wrong execution strategy")
    if config.get("lib_dirs") != [r"E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit\bin"]:
        raise ValueError("Configuration has an unbound runtime library search path")
    if Path(config.get("tokenizer", "")).resolve() != (ROOT.parent / "qwen-strata-20260926/pack-iq3/tokenizer").resolve():
        raise ValueError("Configuration has an unbound tokenizer")
    args = config.get("args")
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError("Configuration args must be a string list")
    flags = [x for x in args if x.startswith("--")]
    allowed_flags = {
        "--pack", "--native", "--ple-gguf", "--expert-profile", "--expert-cache",
        "--prefill", "--spec", "--spec-min-p", "--mtp", "--max-context",
        "--mtp-device", "--expert-cache-exact", "--expert-pin-mib", "--no-prefill-borrow",
        "--gpu-host-dedup", "--mtp-resident-embedding", "--mtp-mapped-residuals",
        "--remote-expert-layer-ids", "--remote-expert-mode", "--adapt-every", "--pcie-frac",
    }
    unknown = sorted(set(flags) - allowed_flags)
    if unknown:
        raise ValueError("Unknown or unbound effective flags: " + ",".join(unknown))
    duplicates = sorted({f for f in flags if flags.count(f) > 1})
    if duplicates:
        raise ValueError("Duplicate effective flags: " + ",".join(duplicates))
    context = int(_one_value(args, "--max-context"))
    if context not in LADDER or (expected_context is not None and context != expected_context):
        raise ValueError("Unsupported or mismatched context tier")
    values = {
        "--remote-expert-layer-ids": ",".join(map(str, REMOTE_IDS)),
        "--remote-expert-mode": "original", "--mtp-device": "0",
        "--expert-pin-mib": "16384", "--prefill": "512",
        "--adapt-every": "0", "--pcie-frac": "0.0", "--expert-cache": "3500",
        "--spec-min-p": "0.5",
    }
    for flag, expected in values.items():
        if _one_value(args, flag) != expected:
            raise ValueError(f"{flag} differs from the fixed profile")
    spec = int(_one_value(args, "--spec"))
    if spec not in (2, 3, 4):
        raise ValueError("Speculation window must be 2..4")
    expected_paths = {
        "--pack": ROOT.parent / "qwen-strata-20260926" / "pack-iq3",
        "--mtp": ROOT.parent / "qwen-strata-20260926" / "mtp" / "rt",
        "--expert-profile": ROOT / "source" / "data" / "expert-profile.bin",
    }
    model_proof = json.loads((ROOT.parent / "qwen-strata-20260926" / "model-proof.json").read_text())
    expected_paths["--native"] = Path(model_proof["files"][0]["path"])
    expected_paths["--ple-gguf"] = Path(model_proof["files"][1]["path"])
    for flag, expected in expected_paths.items():
        if Path(_one_value(args, flag)).resolve() != expected.resolve():
            raise ValueError(f"{flag} points outside the bound runtime/model")
    # Both allowed observer implementations add exactly one --serve before
    # these config args; putting it in the config would duplicate the flag.
    required = {"--native", "--expert-cache-exact", "--no-prefill-borrow",
                "--gpu-host-dedup", "--mtp-resident-embedding", "--mtp-mapped-residuals"}
    forbidden = {"--mmap-experts", "--remote-expert-hybrid", "--remote-expert-freeze-cache",
                 "--verify-wait-profile", "--mtp-chain-batch", "--prefill-borrow",
                 "--pool-workers", "--vram-reserve-mib", "--vision", "--spec-corrupt",
                 "--tokens", "--tokens-file", "--gpu-stages", "--graph-only", "--gpu-only-full"}
    if not required.issubset(flags) or forbidden.intersection(flags):
        raise ValueError("Required fixed flags are missing or an unsupported mode is present")
    return context


def _validate_bound_file(row: dict, base: Path = ROOT) -> None:
    path = Path(row["path"])
    if not path.is_absolute():
        path = base / path
    if not path.is_file() or digest(path) != row.get("sha256"):
        raise RuntimeError(f"Bound evidence identity mismatch: {path}")


def artifact_checks(profile: str):
    policy(profile)
    # Lazy import avoids the numerical_guard -> agent_guard module cycle while
    # forcing every model admission to hash current source, all nine fixtures,
    # and all ten executables against the completed binding.
    import numerical_guard
    current_binding, current_binding_path = numerical_guard.load_binding()
    gates_path = ROOT / "source-gates.json"
    gates = json.loads(gates_path.read_text())
    binding_path = ROOT / "build" / "build-binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    if current_binding != binding or current_binding_path.resolve() != binding_path.resolve():
        raise RuntimeError("Completed build binding changed during admission")
    exe = ROOT / "build" / "strata.exe"
    if (not gates.get("passed") or gates.get("build_binding_sha256") != digest(binding_path)
            or gates.get("executable_sha256") != digest(exe)
            or binding.get("executable_sha256") != digest(exe)):
        raise RuntimeError("Source gates are missing, failed, or stale")
    if set(gates.get("fresh_native_gates", {})) != set(SOURCE_GATE_NAMES):
        raise RuntimeError("Fresh source-gate set is incomplete")
    for gate in gates["fresh_native_gates"].values():
        if not gate.get("passed"):
            raise RuntimeError("A source gate did not pass")
        _validate_bound_file(gate["log"])
    for row in gates.get("bound_files", []):
        _validate_bound_file(row)
    for name in ("mtp-admission.json", "remote12-admission.json"):
        admission_path = ROOT / name
        admission = json.loads(admission_path.read_text())
        if (not admission.get("passed") or admission.get("source_gates_sha256") != digest(gates_path)
                or admission.get("engine_executable_sha256") != digest(exe)
                or admission.get("build_binding_sha256") != digest(binding_path)):
            raise RuntimeError(f"Numerical admission is missing, failed, or stale: {name}")
        for row in admission.get("bound_evidence", []):
            _validate_bound_file(row)

    # Reuse the prior full-hash proof without rehashing 75.8 GB. collect_report
    # revalidates both GGUF size/mtime and the stock/runtime identity.
    report = shared.collect_report(allow_no_ram_reserve=True,
                                   allow_experimental_memory=True,
                                   allow_no_gpu_floor=True)
    report.update({
        "executable_kind": "source-built-bound-agent-campaign",
        "full_expert_payload_bytes": TOTAL_EXPERT_PAYLOAD,
        "remote_gpu1_payload_bytes": REMOTE_PAYLOAD,
        "remote_gpu1_allocation_bytes": REMOTE_ALLOCATION,
        "gpu0_cached_payload_bytes": GPU0_CACHED_PAYLOAD,
        "compact_host_payload_bytes": COMPACT_HOST_PAYLOAD,
        "profile_storage_lower_bound_bytes": COMPACT_ARENA_LOWER_BOUND + GPU0_CACHED_PAYLOAD + REMOTE_ALLOCATION,
        "expert_arena_bytes": COMPACT_ARENA_LOWER_BOUND,
        "compact_host_arena_lower_bound_bytes": COMPACT_ARENA_LOWER_BOUND,
        "arena_accounting": "exact profile/pack byte arithmetic plus fixed remote workspace; lower bound, not total process/global commit",
        "global_commit_accounting": "commit_used/commit_limit/commit_free are measured Windows counters, not inferred savings",
        "ram_reserve_gib": 8, "commit_reserve_gib": 4,
    })
    report["failures"] = []
    memory = report["measured_memory"]
    if memory["ram_free"] < COMPACT_ARENA_LOWER_BOUND + 8 * GIB:
        report["failures"].append("Available RAM is below compact arena plus 8 GiB floor")
    if memory["commit_free"] < COMPACT_ARENA_LOWER_BOUND + 4 * GIB:
        report["failures"].append("Available commit is below compact arena plus 4 GiB floor")
    report["minimum_checks_passed"] = not report["failures"]
    return report, gates


def _previous_tier_admission(context: int) -> None:
    if context == LADDER[0]:
        return
    previous = LADDER[LADDER.index(context) - 1]
    path = ROOT / f"tier-admission-{previous}.json"
    proof = json.loads(path.read_text())
    if (not proof.get("passed") or proof.get("context") != previous
            or proof.get("source_gates_sha256") != digest(ROOT / "source-gates.json")):
        raise RuntimeError("The immediately previous context tier is not admitted")
    _validate_bound_file(proof["result"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("load", "smoke", "bench"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe", type=Path,
                        help="Campaign native probe or the one isolated TEST_QWEN Pi observer")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    opts = parser.parse_args()
    if not 30 <= opts.timeout <= 7200:
        raise ValueError("Timeout outside the bounded range")
    config = json.loads(opts.config.read_text())
    context = validate_config(config)
    _previous_tier_admission(context)
    probe = (opts.probe or (ROOT / "native_probe.py")).resolve()
    allowed_probes = {(ROOT / "native_probe.py").resolve(),
                      Path(r"C:\MYPROJECT\TEST_QWEN\strata-agent-eval-20260927\server_observer.py").resolve()}
    if probe not in allowed_probes or not probe.is_file():
        raise ValueError("Probe is not one of the two scoped campaign observers")
    # shared.run_trial enforces a same-config load admission for every non-load
    # mode. Patching only these scoped entry points preserves the old campaign.
    shared.ROOT = ROOT
    shared.EXE_HASH = digest(ROOT / "build" / "strata.exe")
    shared.policy = policy
    shared.artifact_checks = artifact_checks
    shared.job_counters = measured_job_counters
    if opts.mode == "load":
        # A requested fresh load must fail closed; a failed attempt cannot
        # leave an older matching load admission available to smoke/bench.
        (ROOT / "load-admission.json").write_text(json.dumps({"passed": False,
            "invalidated_by": opts.tag, "config_sha256": digest(opts.config)}, indent=2))
    ns = argparse.Namespace(mode=opts.mode, profile=PROFILE, gpu=0, pool_workers=8,
                            vram_reserve_mib=None, tag=opts.tag, timeout=opts.timeout)
    shared.run_trial(ns, config_file=opts.config, probe_file=probe,
                     device_visibility="0,1")
    result = ROOT / "runs" / opts.tag / "result.json"
    evidence = json.loads(result.read_text())
    if not evidence.get("passed"):
        raise RuntimeError("Guarded tier trial did not pass")
    if opts.mode != "load":
        out = ROOT / f"tier-admission-{context}.json"
        out.write_text(json.dumps({"passed": True, "context": context, "mode": opts.mode,
                                   "config_sha256": digest(opts.config),
                                   "source_gates_sha256": digest(ROOT / "source-gates.json"),
                                   "result": {"path": str(result), "sha256": digest(result)}}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STRATA AGENT STOP: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
