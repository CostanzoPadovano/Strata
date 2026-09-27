import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import debug_launcher as target


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = target.HERE / "runs" / "test-tmp"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_same_guard_and_no_unsafe_flags(self):
        normal = target.guard_command("debug-fixture")
        self.assertEqual(normal[1], str(target.GUARD))
        self.assertEqual(normal[2], "manual")
        self.assertEqual(normal[-2:], ["--timeout", "14400"])
        self.assertEqual(target.guard_command("debug-fixture", True)[2], "check")

    def test_no_tag_escape(self):
        for tag in ("../other", "debug-../other", "debug-a/b", "vision-manual-x"):
            with self.assertRaises(ValueError):
                target.guard_command(tag)

    def test_bounded_json(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(target.HERE.resolve()))
            path = Path(folder) / "bad.json"
            path.write_bytes(b" " * (target.MAX_RECORD + 1))
            with self.assertRaises(ValueError):
                target.small_json(path)
            with self.assertRaises(ValueError):
                target.write_record(path, {"too_large": "x" * target.MAX_RECORD})

    def test_stream_guard_numeric_only(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(target.HERE.resolve()))
            root = Path(folder)
            rows = [{"peak_job_bytes": 10, "ram_free": 30, "commit_free": 40, "prompt": "SECRET"},
                    {"peak_job_bytes": 20, "ram_free": 25, "commit_free": 35}]
            (root / "telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            (root / "result.json").write_text('{"passed":true,"model_started":true,"observer_exit_code":0}')
            result = target.guardian_summary(root)
            self.assertEqual(result["job_peak_bytes"], 20)
            self.assertEqual(result["ram_free_min_bytes"], 25)
            self.assertEqual(result["commit_free_min_bytes"], 35)
            self.assertNotIn("SECRET", json.dumps(result))

    def test_monitor_failure_stops_only_owned_guard(self):
        guard, sampler = Mock(), Mock()
        guard.poll.side_effect = [None, 0]
        sampler.poll.return_value = 1
        with patch.object(target.subprocess, "Popen", return_value=guard), patch.object(target, "request_guard_stop") as stop:
            with self.assertRaises(RuntimeError):
                target.monitored_guard("debug-fixture", Path("exact-owned-run"), sampler, False)
        stop.assert_called_once_with(Path("exact-owned-run"), guard)

    def test_stop_marker_and_owned_timeout(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(target.HERE.resolve()))
            guard = Mock()
            guard.wait.side_effect = [subprocess.TimeoutExpired("owned", 35), 0]
            target.request_guard_stop(Path(folder), guard)
            self.assertTrue((Path(folder) / "stop-server").exists())
            guard.terminate.assert_called_once()

    def test_sampler_failure_prevents_model(self):
        sampler = Mock()
        sampler.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            target.wait_for_sampler(sampler, Path("nonexistent-ready"))

    def test_sampler_missing_ram_prevents_model(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(target.HERE.resolve()))
            root = Path(folder)
            sampler = Mock(pid=123)
            sampler.poll.return_value = None
            (root / "ready.json").write_text('{"ready":true,"pid":123}')
            (root / "samples.jsonl").write_text('{"memory":null}\n')
            with self.assertRaisesRegex(RuntimeError, "Essential"):
                target.wait_for_sampler(sampler, root / "ready.json")

    def test_sampler_valid_startup(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(target.HERE.resolve()))
            root = Path(folder)
            sampler = Mock(pid=123)
            sampler.poll.return_value = None
            (root / "ready.json").write_text('{"ready":true,"pid":123}')
            (root / "samples.jsonl").write_text(json.dumps({"memory": {
                "ram_total_bytes": 100, "ram_available_bytes": 90,
                "windows_commit_limit_bytes": 200, "windows_commit_available_bytes": 180}}) + "\n")
            target.wait_for_sampler(sampler, root / "ready.json")

    def test_sampler_venv_redirector_startup(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(target.HERE.resolve()))
            root = Path(folder)
            sampler = Mock(pid=123)
            sampler.poll.return_value = None
            (root / "ready.json").write_text('{"ready":true,"pid":456,"parent_pid":123}')
            (root / "samples.jsonl").write_text(json.dumps({"memory": {
                "ram_total_bytes": 100, "ram_available_bytes": 90,
                "windows_commit_limit_bytes": 200, "windows_commit_available_bytes": 180}}) + "\n")
            target.wait_for_sampler(sampler, root / "ready.json")


if __name__ == "__main__":
    unittest.main()
