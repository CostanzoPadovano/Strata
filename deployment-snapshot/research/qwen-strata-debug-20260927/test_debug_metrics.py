import ctypes
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import debug_metrics as dm

TEST_DIR = Path(__file__).resolve().parent


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FixedCollector:
    def __init__(self, value=None, failure=None):
        self.value = value
        self.failure = failure
        self.closed = False

    def collect(self):
        if self.failure:
            raise self.failure
        return self.value

    def close(self):
        self.closed = True


class DebugMetricsTests(unittest.TestCase):
    def test_error_tracker_throttles_and_caps(self):
        errors = dm.ErrorTracker(maximum=2)
        errors.record("pdh", "same", 1)
        errors.record("pdh", "same", 2)
        errors.record("gpu", "different", 3)
        errors.record("process", "overflow", 4)
        self.assertEqual(errors.snapshot()[0]["count"], 2)
        self.assertEqual(errors.dropped, 1)

    def test_summary_min_max_availability_and_intervals(self):
        summary = dm.SummaryAccumulator()
        summary.add({"wall_interval_s": None, "metric": 8, "nested": {"x": None}})
        summary.add({"wall_interval_s": 5, "metric": 3, "nested": {"x": 4}})
        result = summary.result()
        self.assertEqual(result["metrics"]["metric"]["min"], 3)
        self.assertEqual(result["metrics"]["metric"]["max"], 8)
        self.assertEqual(result["metrics"]["nested.x"]["availability_fraction"], 0.5)
        self.assertEqual(result["wall_intervals"]["mean_s"], 5)

    def test_summary_tracks_gpu_min_max_by_index(self):
        summary = dm.SummaryAccumulator()
        summary.add({"gpus": [{"memory_used_mib": 10}]})
        summary.add({"gpus": [{"memory_used_mib": 14}]})
        metric = summary.result()["metrics"]["gpus.gpu0.memory_used_mib"]
        self.assertEqual((metric["min"], metric["max"]), (10, 14))

    def test_pdh_first_sample_is_unavailable(self):
        class Dll:
            def PdhCollectQueryData(self, _query):
                return 0

            def PdhGetFormattedCounterValue(self, _handle, _fmt, _kind, output):
                output._obj.CStatus = 0
                output._obj.value.doubleValue = 12.5
                return 0

        pdh = dm.PdhCollector.__new__(dm.PdhCollector)
        pdh._dll = Dll()
        pdh._query = ctypes.c_void_p(1)
        pdh._handles = {"cpu_total_percent": ctypes.c_void_p(2)}
        pdh.add_errors = {}
        pdh._first = True
        first = pdh.collect()
        second = pdh.collect()
        self.assertTrue(all(value is None for value in first.values()))
        self.assertEqual(second["cpu_total_percent"], 12.5)

    def test_partial_counter_failure_is_null_with_bounded_error(self):
        collector = FixedCollector({"counter": None})
        collector.last_errors = {"counter": "unavailable"}
        errors = dm.ErrorTracker()
        self.assertEqual(dm._try_collect("pdh", collector, errors), {"counter": None})
        self.assertEqual(errors.snapshot()[0]["source"], "pdh.counter")

    def test_nvidia_smi_timeout_and_parsing_contract(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, "0, GPU, 1.2, 100, 20, 80, 50, 60, 123.5\n", "")

        rows = dm.query_nvidia_smi(runner=runner)
        self.assertEqual(calls[0][1]["timeout"], 3.0)
        self.assertEqual(rows[0]["memory_used_mib"], 20)
        self.assertEqual(rows[0]["power_w"], 123.5)

    def test_nvidia_smi_rejects_failure_without_zero_fallback(self):
        def runner(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        with self.assertRaises(subprocess.TimeoutExpired):
            dm.query_nvidia_smi(runner=runner)

    def test_nvidia_smi_live_reader_retains_at_most_64k(self):
        class FakeProcess:
            def __init__(self):
                self.stdout = io.BytesIO(b"x" * (64 * 1024 + 2))

            def wait(self, timeout):
                return 0

            def kill(self):
                raise AssertionError("completed process must not be killed")

        with self.assertRaisesRegex(OSError, "64 KiB"):
            dm.query_nvidia_smi(popen_factory=lambda *args, **kwargs: FakeProcess())

    def test_process_groups_are_name_only_and_exact_sums_can_overlap(self):
        groups = dm.ProcessCollector._groups
        self.assertEqual(groups("git.exe"), ["git"])
        self.assertIn("codex", groups("codex.exe"))
        self.assertIn("vmmem_wsl", groups("vmmemWSL.exe"))
        self.assertIn("native", groups("strata-engine.exe"))
        self.assertIn("encoder", groups("strata-vision-encoder.exe"))

    def test_inaccessible_processes_are_counted_without_false_zero_memory(self):
        result = dm.summarize_processes(
            [(1, "vmmemWSL.exe"), (2, "git.exe")],
            [{"pid": 2, "name": "git.exe", "working_set_bytes": 20,
              "private_bytes": 10, "cpu_percent_one_core": None, "cpu_time_s": 1}],
            truncated=False, scan_duration_ms=2.0, monitor_pid=2,
        )
        self.assertEqual(result["enumerated_count"], 2)
        self.assertEqual(result["memory_read_count"], 1)
        self.assertEqual(result["inaccessible_count"], 1)
        self.assertFalse(result["memory_complete"])
        self.assertEqual(result["groups"]["vmmem_wsl"]["count"], 1)
        self.assertIsNone(result["groups"]["vmmem_wsl"]["working_set_bytes"])
        self.assertFalse(result["groups"]["vmmem_wsl"]["memory_complete"])
        self.assertEqual(result["groups"]["git"]["working_set_bytes"], 20)

    def test_pagefile_peak_sum_is_explicitly_non_simultaneous(self):
        collector = dm.PagefileCollector.__new__(dm.PagefileCollector)
        collector.page_size = 4096
        collector._callback_type = ctypes.WINFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p,
            ctypes.POINTER(dm.PagefileCollector.PAGEFILEINFO), ctypes.c_wchar_p,
        )

        def enum_pagefiles(callback, _context):
            for name, peak in ((r"C:\pagefile.sys", 3), (r"D:\pagefile.sys", 7)):
                info = dm.PagefileCollector.PAGEFILEINFO()
                info.total_pages, info.used_pages, info.peak_pages = 20, 2, peak
                callback(None, ctypes.pointer(info), name)
            return True

        collector._fn = enum_pagefiles
        result = collector.collect()
        self.assertEqual(result["peak_used_bytes"], 10 * 4096)
        self.assertEqual(result["peak_aggregation_scope"], "sum_of_per_file_peaks_not_simultaneous_global_peak")
        self.assertTrue(all(row["peak_scope"] == "per_file_lifetime_peak" for row in result["files"]))

    def test_observe_run_reads_only_status_fields_and_bounds_file(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as root:
            run = Path(root) / "debug-case"
            run.mkdir()
            (run / "phase.json").write_text(json.dumps({
                "phase": "loading", "status": "ok", "prompt": "secret", "config": {"x": 1}
            }), encoding="utf-8")
            (run / "result.json").write_text("{}", encoding="utf-8")
            observed = dm.observe_run(run)
            self.assertEqual(observed["observed_phase"], {"phase": "loading", "status": "ok"})
            self.assertNotIn("prompt", json.dumps(observed))
            self.assertTrue(observed["result_exists"])
            (run / "phase.json").write_bytes(b"x" * (16 * 1024 + 1))
            self.assertIn("16 KiB", dm.observe_run(run)["observation_error"])

    def test_jsonl_cap_leaves_complete_partial_rows(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as root:
            writer = dm.JsonlWriter(Path(root) / "samples.jsonl", maximum_bytes=16)
            self.assertTrue(writer.write({"x": 1}))
            self.assertFalse(writer.write({"long": "0123456789"}))
            writer.close()
            rows = (Path(root) / "samples.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(row) for row in rows], [{"x": 1}])
            self.assertTrue(writer.capped)

    def test_sampler_flushes_ready_records_failures_and_cleans_up(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as root:
            clock = FakeClock()
            collectors = {
                "memory": FixedCollector({"ram_used_bytes": 10}),
                "pagefile": FixedCollector(failure=OSError("blocked")),
                "pdh": FixedCollector({"cpu_total_percent": None}),
                "process": FixedCollector({"groups": {}}),
            }
            factories = {name: (lambda collector=collector: collector) for name, collector in collectors.items()}
            summary = dm.run_sampler(
                Path(root), Path(root) / "stop", 4.1, 2.0,
                clock=clock, sleeper=clock.sleep, factories=factories, gpu_query=lambda: [],
            )
            self.assertTrue((Path(root) / "ready.json").is_file())
            sample_lines = (Path(root) / "samples.jsonl").read_text().splitlines()
            self.assertEqual(len(sample_lines), 3)
            self.assertIsNone(json.loads(sample_lines[0])["pagefiles"])
            rows = [json.loads(line) for line in sample_lines]
            self.assertEqual([row["gpu_sample_age_s"] for row in rows], [0, 2, 4])
            self.assertEqual([row["process_sample_age_s"] for row in rows], [0, 2, 4])
            self.assertTrue(any(e["source"] == "pagefile" for e in summary["errors"]))
            self.assertTrue(all(collector.closed for collector in collectors.values()))

    def test_parent_exit_starts_bounded_tail_and_never_kills(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as root:
            clock = FakeClock()

            class Parent:
                def __init__(self, _pid):
                    self.closed = False

                def exited(self):
                    return clock.value >= 102.0

                def close(self):
                    self.closed = True

            factories = {
                name: (lambda: FixedCollector({}))
                for name in ("memory", "pagefile", "pdh", "process")
            }
            factories["parent"] = Parent
            summary = dm.run_sampler(
                Path(root), Path(root) / "stop", 100, 2,
                parent_pid=123, clock=clock, sleeper=clock.sleep,
                factories=factories, gpu_query=lambda: [], post_stop_seconds=4,
            )
            self.assertEqual(summary["reason"], "parent_exit")
            self.assertGreaterEqual(summary["elapsed_s"], 6)
            self.assertLess(summary["elapsed_s"], 8.1)

    def test_parent_watcher_has_no_process_control_primitive(self):
        source = Path(dm.__file__).read_text(encoding="utf-8")
        self.assertNotIn("TerminateProcess", source)
        self.assertEqual(dm.ParentWatcher.SYNCHRONIZE, 0x00100000)

    def test_sampler_log_cap_finishes_with_summary(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as root:
            clock = FakeClock()
            factories = {name: (lambda: FixedCollector({"large": "x" * 80}))
                         for name in ("memory", "pagefile", "pdh", "process")}
            summary = dm.run_sampler(
                Path(root), Path(root) / "stop", 10, 2,
                clock=clock, sleeper=clock.sleep, factories=factories,
                gpu_query=lambda: [], max_log_bytes=32,
            )
            self.assertEqual(summary["reason"], "log_cap")
            self.assertTrue(summary["log_capped"])
            self.assertTrue((Path(root) / "summary.json").is_file())
            self.assertFalse((Path(root) / "ready.json").exists())

    def test_cli_validation(self):
        with self.assertRaises(SystemExit):
            dm.main(["--output", "x", "--stop-file", "y", "--duration", "14521"])
        with self.assertRaises(SystemExit):
            dm.main(["--output", "x", "--stop-file", "y", "--duration", "1", "--interval", "1"])


if __name__ == "__main__":
    unittest.main()
