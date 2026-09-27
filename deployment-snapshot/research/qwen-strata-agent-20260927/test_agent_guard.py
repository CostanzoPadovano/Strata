from __future__ import annotations
import json
import unittest
from unittest import mock

import agent_guard as guard
import numerical_guard


def valid_config(context=4096):
    proof = json.loads((guard.ROOT.parent / "qwen-strata-20260926" / "model-proof.json").read_text())
    args = [
        "--pack", str(guard.ROOT.parent / "qwen-strata-20260926" / "pack-iq3"),
        "--native", proof["files"][0]["path"], "--ple-gguf", proof["files"][1]["path"],
        "--expert-profile", str(guard.ROOT / "source/data/expert-profile.bin"),
        "--expert-cache", "3500", "--prefill", "512", "--spec", "4",
        "--spec-min-p", "0.5", "--mtp", str(guard.ROOT.parent / "qwen-strata-20260926/mtp/rt"),
        "--max-context", str(context), "--mtp-device", "0",
        "--expert-cache-exact", "--expert-pin-mib", "16384", "--no-prefill-borrow",
        "--gpu-host-dedup", "--mtp-resident-embedding", "--mtp-mapped-residuals",
        "--remote-expert-layer-ids", ",".join(map(str, guard.REMOTE_IDS)),
        "--remote-expert-mode", "original", "--adapt-every", "0", "--pcie-frac", "0.0",
    ]
    return {"exe": str(guard.ROOT / "build/strata.exe"), "cwd": str(guard.ROOT / "source"),
            "tokenizer": str(guard.ROOT.parent / "qwen-strata-20260926/pack-iq3/tokenizer"),
            "lib_dirs": [r"E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit\bin"],
            "strategy": "static-gpu-ownership", "args": args}


class ConfigTests(unittest.TestCase):
    def test_all_authorized_tiers_validate(self):
        for tier in guard.LADDER:
            self.assertEqual(guard.validate_config(valid_config(tier), tier), tier)

    def test_wrong_engine_and_strategy_refused(self):
        cfg = valid_config(); cfg["exe"] = str(guard.ROOT / "other.exe")
        with self.assertRaisesRegex(ValueError, "bound Strata engine"): guard.validate_config(cfg)
        cfg = valid_config(); cfg["strategy"] = "resident-experts"
        with self.assertRaisesRegex(ValueError, "execution strategy"): guard.validate_config(cfg)

    def test_duplicate_and_unknown_flags_refused(self):
        cfg = valid_config(); cfg["args"] += ["--spec", "4"]
        with self.assertRaisesRegex(ValueError, "Duplicate"): guard.validate_config(cfg)
        cfg = valid_config(); cfg["args"].append("--relax-floor")
        with self.assertRaisesRegex(ValueError, "Unknown"): guard.validate_config(cfg)

    def test_unsupported_context_and_profile_drift_refused(self):
        cfg = valid_config(); cfg["args"][cfg["args"].index("--max-context") + 1] = "90000"
        with self.assertRaisesRegex(ValueError, "context"): guard.validate_config(cfg)
        cfg = valid_config(); cfg["args"][cfg["args"].index("--remote-expert-layer-ids") + 1] = "0,1,2,3"
        with self.assertRaisesRegex(ValueError, "fixed profile"): guard.validate_config(cfg)

    def test_forbidden_runtime_override_refused(self):
        for flag in ("--mmap-experts", "--vram-reserve-mib", "--pool-workers", "--vision"):
            cfg = valid_config(); cfg["args"].append(flag)
            with self.subTest(flag=flag), self.assertRaises(ValueError): guard.validate_config(cfg)

    def test_wrong_model_path_refused(self):
        cfg = valid_config(); cfg["args"][cfg["args"].index("--native") + 1] = "wrong.gguf"
        with self.assertRaisesRegex(ValueError, "bound runtime/model"): guard.validate_config(cfg)


class AdmissionTests(unittest.TestCase):
    def test_previous_tier_binds_result_bytes(self):
        proof = {"passed": True, "context": 4096,
                 "source_gates_sha256": "gates",
                 "result": {"path": "bound-result.json", "sha256": "a" * 64}}
        with mock.patch("pathlib.Path.read_text", return_value=json.dumps(proof)), \
             mock.patch.object(guard, "digest", return_value="gates"), \
             mock.patch.object(guard, "_validate_bound_file") as check:
            guard._previous_tier_admission(8192)
            check.assert_called_once_with(proof["result"])
        with mock.patch("pathlib.Path.read_text", return_value=json.dumps(proof)), \
             mock.patch.object(guard, "digest", return_value="gates"), \
             mock.patch.object(guard, "_validate_bound_file", side_effect=RuntimeError("stale")), \
             self.assertRaisesRegex(RuntimeError, "stale"):
            guard._previous_tier_admission(8192)

    def test_4096_load_has_no_previous_tier(self):
        guard._previous_tier_admission(4096)

    def test_policy_refuses_wrong_profile_and_override(self):
        with self.assertRaises(ValueError): guard.policy("experimental-no-vram-floor")
        with self.assertRaises(ValueError): guard.policy(guard.PROFILE, 0)

    def test_numerical_markers_bind_required_extent(self):
        self.assertIn("fixtures=36 modes=9", numerical_guard.MTP_MARKER)
        for token in ("layers=12", "blob_cases=24", "decode_calls=12", "prefill_calls=13",
                      "large_prefill_bytes=209715200", "chunk_bytes=16777216", "quant_bytes=exact"):
            self.assertIn(token, numerical_guard.REMOTE_MARKER)
        self.assertLessEqual(24, 24)

    def test_actual_build_revalidation_precedes_artifact_admission(self):
        with mock.patch.object(numerical_guard, "load_binding", side_effect=RuntimeError("stale actual source")) as check, \
             self.assertRaisesRegex(RuntimeError, "stale actual source"):
            guard.artifact_checks(guard.PROFILE)
        check.assert_called_once_with()

    def test_child_wait_refuses_without_attachment_marker(self):
        with self.assertRaisesRegex(TimeoutError, "did not attach"):
            numerical_guard.wait_for_job_attachment(guard.ROOT / "definitely-no-such-run", 0)


if __name__ == "__main__": unittest.main()
