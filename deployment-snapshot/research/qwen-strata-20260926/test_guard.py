import ctypes as C
from ctypes import wintypes as W
import subprocess
import sys
import time
import unittest
from guard import counters,resources_ok,make_job,k32,wincheck,GiB

class GuardTests(unittest.TestCase):
    def test_counters(self):
        c=counters();self.assertGreater(c['ram_free'],0);self.assertGreater(c['commit_free'],0)
    def test_floors(self):
        for ram,commit,gpu,temp in [(11,20,4096,40),(20,15,4096,40),(20,20,2047,40),(20,20,4096,80)]:
            with self.assertRaises(RuntimeError):resources_ok({'ram_free':ram*GiB,'commit_free':commit*GiB},[[0,gpu,temp],[1,4096,40]])
        resources_ok({'ram_free':12*GiB,'commit_free':16*GiB},[[0,2048,79],[1,2048,79]])
    def test_job_close_kills_child(self):
        job=make_job();p=None
        try:
            p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],creationflags=subprocess.CREATE_NO_WINDOW)
            wincheck(k32.AssignProcessToJobObject(job,int(p._handle)))
            member=W.BOOL();wincheck(k32.IsProcessInJob(int(p._handle),job,C.byref(member)))
            self.assertTrue(member.value)
            wincheck(k32.CloseHandle(job));job=None
            p.wait(timeout=10);self.assertIsNotNone(p.returncode)
        finally:
            if job:k32.CloseHandle(job)
            if p and p.poll() is None:p.kill()
if __name__=='__main__':unittest.main(verbosity=2)
