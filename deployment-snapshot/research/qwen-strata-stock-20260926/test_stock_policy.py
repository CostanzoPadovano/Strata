"""Boundary tests; no model load, no upstream modifications."""
import unittest
import ctypes as C
from ctypes import wintypes as W
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from check_stock import GIB, minimum_failures
from stock_guard import check_resources, policy, make_job, accept_ready, k32, wincheck


class StockPolicyTests(unittest.TestCase):
    def setUp(self):
        self.arena = 42915302400
        self.gpu = [[0, 4096, 40], [1, 4096, 40]]

    def test_default_still_requires_twelve_gib(self):
        memory = {'ram_free': self.arena + 6 * GIB, 'commit_free': self.arena + 16 * GIB}
        self.assertTrue(minimum_failures(memory, self.gpu, self.arena))
        self.assertEqual(minimum_failures(memory, self.gpu, self.arena, ram_reserve_bytes=0), [])

    def test_zero_reserve_is_not_a_memory_capacity_bypass(self):
        memory = {'ram_free': self.arena - 1, 'commit_free': self.arena + 16 * GIB}
        self.assertTrue(minimum_failures(memory, self.gpu, self.arena, ram_reserve_bytes=0))
        memory['ram_free'] += 1
        self.assertEqual(minimum_failures(memory, self.gpu, self.arena, ram_reserve_bytes=0), [])

    def test_commit_floor_not_relaxed(self):
        memory = {'ram_free': self.arena + GIB, 'commit_free': self.arena + 16 * GIB - 1}
        self.assertTrue(minimum_failures(memory, self.gpu, self.arena, ram_reserve_bytes=0))

    def test_gpu_floor_and_temperature_not_relaxed(self):
        memory = {'ram_free': self.arena + GIB, 'commit_free': self.arena + 16 * GIB}
        for gpu in ([[0, 2047, 40], [1, 4096, 40]], [[0, 4096, 40], [1, 4096, 80]]):
            self.assertTrue(minimum_failures(memory, gpu, self.arena, ram_reserve_bytes=0))

    def test_unknown_reserve_rejected(self):
        with self.assertRaises(ValueError):
            minimum_failures({}, [], self.arena, ram_reserve_bytes=GIB)

    def test_isolated_runtime_ram_exception(self):
        limits = policy('no-ram-reserve')
        self.assertEqual({k: limits[k] for k in ('ram_floor_gib', 'commit_floor_gib', 'job_cap_gib')},
                         {'ram_floor_gib': 0, 'commit_floor_gib': 16, 'job_cap_gib': 48})
        self.assertEqual((limits['gpu_start_floor_mib'], limits['gpu_floor_mib']), (14336, 2048))
        check_resources({'ram_free': 1, 'commit_free': 16 * GIB}, self.gpu, limits)
        with self.assertRaises(RuntimeError):
            check_resources({'ram_free': 1, 'commit_free': 16 * GIB - 1}, self.gpu, limits)
        with self.assertRaises(RuntimeError):
            check_resources({'ram_free': self.arena - 1, 'commit_free': self.arena + 16 * GIB},
                            [[0, 14336, 40], [1, 14336, 40]], limits, start=True, arena=self.arena)

    def test_default_runtime_remains_twelve_gib(self):
        limits = policy('default')
        with self.assertRaises(RuntimeError):
            check_resources({'ram_free': 11 * GIB, 'commit_free': 16 * GIB}, self.gpu, limits)

    def test_confirmed_experimental_profile_is_finite(self):
        limits = policy('experimental-memory')
        self.assertEqual({k: limits[k] for k in ('ram_floor_gib', 'commit_floor_gib', 'job_cap_gib')},
                         {'ram_floor_gib': 0, 'commit_floor_gib': 4, 'job_cap_gib': 60})
        check_resources({'ram_free': 0, 'commit_free': 4 * GIB}, self.gpu, limits)
        with self.assertRaises(RuntimeError):
            check_resources({'ram_free': 0, 'commit_free': 4 * GIB - 1}, self.gpu, limits)
        memory = {'ram_free': self.arena, 'commit_free': self.arena + 4 * GIB}
        self.assertEqual(minimum_failures(memory, self.gpu, self.arena,
                         ram_reserve_bytes=0, commit_reserve_bytes=4 * GIB), [])
        with self.assertRaises(ValueError):
            make_job(64)

    def test_confirmed_vram_exception_keeps_temperature_commit_and_capacity(self):
        limits = policy('experimental-no-vram-floor')
        self.assertEqual((limits['gpu_start_floor_mib'], limits['gpu_floor_mib']), (0, 0))
        self.assertEqual(limits['vram_reserve_mib'], 700)
        gpu = [[0, 0, 40], [1, 1, 79]]
        check_resources({'ram_free': 0, 'commit_free': 4 * GIB}, gpu, limits)
        check_resources({'ram_free': self.arena, 'commit_free': self.arena + 4 * GIB},
                        gpu, limits, start=True, arena=self.arena)
        self.assertEqual(minimum_failures({'ram_free': self.arena, 'commit_free': self.arena + 4 * GIB},
                         gpu, self.arena, ram_reserve_bytes=0, commit_reserve_bytes=4 * GIB, gpu_floor_mib=0), [])
        for memory, devices, start in (
                ({'ram_free': 0, 'commit_free': 4 * GIB - 1}, gpu, False),
                ({'ram_free': self.arena - 1, 'commit_free': self.arena + 4 * GIB}, gpu, True),
                ({'ram_free': 0, 'commit_free': 4 * GIB}, [[0, 0, 80], [1, 0, 40]], False)):
            with self.assertRaises(RuntimeError):
                check_resources(memory, devices, limits, start=start, arena=self.arena)
        with self.assertRaises(ValueError):
            minimum_failures({}, [], self.arena, gpu_floor_mib=1)

    def test_default_gpu_stops_remain(self):
        for profile in ('default', 'no-ram-reserve', 'experimental-memory'):
            limits = policy(profile)
            self.assertEqual(limits['vram_reserve_mib'], 3072)
            memory = {'ram_free': 12 * GIB, 'commit_free': 16 * GIB}
            with self.assertRaises(RuntimeError):
                check_resources(memory, [[0, 2047, 40], [1, 4096, 40]], limits)

    def test_cache_sizing_is_not_a_gpu_stop(self):
        limits = policy('experimental-no-vram-floor', 1536)
        self.assertEqual(limits['vram_reserve_mib'], 1536)
        check_resources({'ram_free': 0, 'commit_free': 4 * GIB}, [[0, 0, 40], [1, 0, 40]], limits)
        for profile, reserve in (('default', 1536), ('experimental-memory', 700),
                                 ('experimental-no-vram-floor', 0)):
            with self.assertRaises(ValueError):
                policy(profile, reserve)

    def test_vram_exception_cannot_be_used_outside_confirmed_memory_profile(self):
        from check_stock import collect_report
        with self.assertRaises(ValueError):
            collect_report(allow_no_gpu_floor=True)
        with patch('stock_guard.Path.read_text', return_value='{"authorized_profiles":["experimental-memory"]}'):
            with self.assertRaises(RuntimeError):
                policy('experimental-no-vram-floor')

    def test_ready_snapshot_must_pass_before_observer_is_released(self):
        limits = policy('experimental-memory')
        memory = {'ram_free': 3 * GIB, 'commit_free': 5 * GIB}
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix='strata-ready-test-', dir=root) as directory:
            run = Path(directory)
            self.assertTrue(run.resolve().is_relative_to(root))
            with patch('stock_guard.counters', return_value=memory), patch('stock_guard.gpu_counters', return_value=[[0, 4000, 40], [1, 1944, 40]]):
                with self.assertRaises(RuntimeError):
                    accept_ready(run, limits)
            self.assertFalse((run / 'ready-accepted.json').exists())
            with patch('stock_guard.counters', return_value=memory), patch('stock_guard.gpu_counters', return_value=[[0, 4000, 40], [1, 2048, 40]]):
                accept_ready(run, limits)
            self.assertTrue((run / 'ready-accepted.json').exists())

    def test_job_close_kills_attached_child(self):
        job = make_job(48)
        process = None
        try:
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)'],
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            wincheck(k32.AssignProcessToJobObject(job, int(process._handle)))
            member = W.BOOL()
            wincheck(k32.IsProcessInJob(int(process._handle), job, C.byref(member)))
            self.assertTrue(member.value)
            wincheck(k32.CloseHandle(job))
            job = None
            process.wait(timeout=10)
            self.assertIsNotNone(process.returncode)
        finally:
            if job:
                k32.CloseHandle(job)
            if process and process.poll() is None:
                process.kill()
                process.wait(timeout=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
