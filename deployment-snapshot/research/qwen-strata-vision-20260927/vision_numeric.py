"""Run and bind the isolated vision build's bounded numerical prerequisites."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
TEXT_ROOT = ROOT.parent / "qwen-strata-agent-20260927"
sys.path.insert(0, str(TEXT_ROOT))

import numerical_guard as frozen  # noqa: E402


GATE_PATH = ROOT / "native-gates.json"
RESULT_PATH = ROOT / "numerical-runs" / "results.json"
BUILD = ROOT / "build"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_rows(rows: list[dict]) -> None:
    failures = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            failures.append(f"missing: {path}")
        elif digest(path) != row["sha256"]:
            failures.append(f"drift: {path}")
    if failures:
        raise RuntimeError("Bound artifact validation failed:\n" + "\n".join(failures))


def append_unique(rows: list[dict], additions: list[dict]) -> list[dict]:
    by_path = {row["path"]: row for row in rows}
    for row in additions:
        previous = by_path.get(row["path"])
        if previous is not None and previous != row:
            raise RuntimeError(f"Conflicting bound artifact: {row['path']}")
        by_path[row["path"]] = row
    return list(by_path.values())


def main() -> None:
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError("Native gate is not passed")
    original_rows = gate["bound_files"]
    validate_rows(original_rows)

    mtp_exe = (BUILD / "mtp_equivalence_test.exe").resolve()
    remote_exe = (BUILD / "remote_experts_test.exe").resolve()
    for executable in (mtp_exe, remote_exe):
        matches = [row for row in original_rows if Path(row["path"]) == executable]
        if len(matches) != 1 or digest(executable) != matches[0]["sha256"]:
            raise RuntimeError(f"Numerical binary is not uniquely bound: {executable}")

    config = json.loads((TEXT_ROOT / "configs" / "tier4096.json").read_text(encoding="utf-8"))
    args = config["args"]
    pack = args[args.index("--pack") + 1]
    native = args[args.index("--native") + 1]
    mtp = args[args.index("--mtp") + 1]

    mtp_command = [str(mtp_exe), "--pack", pack, "--native", native, "--mtp", mtp]
    remote_command = [str(remote_exe), "--pack", pack, "--native", native,
                      "--layer-ids", frozen.IDS]
    mtp_result = frozen.run_native(
        mtp_command, ROOT / "numerical-runs" / "vision-mtp-fresh36x9",
        frozen.MTP_MARKER, timeout=900, cap_gib=8,
    )
    remote_result = frozen.run_native(
        remote_command, ROOT / "numerical-runs" / "vision-remote-selected12",
        frozen.REMOTE_MARKER, timeout=900, cap_gib=24,
    )

    validate_rows(original_rows)
    results = {
        "passed": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prior_native_gates_sha256": digest(GATE_PATH),
        "mtp": {**mtp_result, "purpose": "fresh36x9", "marker": frozen.MTP_MARKER,
                "test_executable_sha256": digest(mtp_exe)},
        "remote12": {**remote_result, "purpose": "selected12", "marker": frozen.REMOTE_MARKER,
                     "test_executable_sha256": digest(remote_exe)},
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    additions = [*mtp_result["bound_evidence"], *remote_result["bound_evidence"],
                 {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).resolve())},
                 {"path": str(RESULT_PATH.resolve()), "sha256": digest(RESULT_PATH)}]
    gate["numerical_gates"] = {"mtp": results["mtp"], "remote12": results["remote12"]}
    gate["bound_files"] = append_unique(original_rows, additions)
    validate_rows(gate["bound_files"])
    GATE_PATH.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "native_gates": str(GATE_PATH),
                      "native_gates_sha256": digest(GATE_PATH),
                      "bound_files": len(gate["bound_files"])}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STRATA VISION NUMERICAL STOP: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
