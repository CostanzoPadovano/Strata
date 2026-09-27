"""Bounded no-GPU checks of current owned-private telemetry and handle cleanup."""
import ctypes as C
from ctypes import wintypes as W
import subprocess
import sys
import queue
import threading
import unittest
from unittest.mock import patch
import owned_memory as m


class MemoryTests(unittest.TestCase):
    def test_counts_ownership_cleanup_bounds_and_failed_reads(self):
        pids=[101,102]
        count=2
        member=True
        readable=True
        exited=False
        closed=[]
        def query(job,kind,ptr,size,out):
            self.assertEqual(kind,3)
            value=C.cast(ptr,C.POINTER(m.ProcessIds)).contents
            value.assigned=value.listed=count
            for i,pid in enumerate(pids):value.ids[i]=pid
            return True
        def membership(handle,job,ptr):
            C.cast(ptr,C.POINTER(W.BOOL)).contents.value=member
            return True
        def memory(handle,ptr,size):
            value=C.cast(ptr,C.POINTER(m.ProcessMemory)).contents
            self.assertEqual(value.cb,C.sizeof(m.ProcessMemory))
            value.private=handle*1000
            return readable
        def exit_code(handle,ptr):
            C.cast(ptr,C.POINTER(W.DWORD)).contents.value=0 if exited else 259
            return True
        with patch.object(m.k32,'QueryInformationJobObject',side_effect=query), \
             patch.object(m.k32,'OpenProcess',side_effect=lambda access,inherit,pid:pid), \
             patch.object(m.k32,'IsProcessInJob',side_effect=membership), \
             patch.object(m.k32,'CloseHandle',side_effect=lambda handle:closed.append(handle) or True), \
             patch.object(m.k32,'GetExitCodeProcess',side_effect=exit_code), \
             patch.object(m.psapi,'GetProcessMemoryInfo',side_effect=memory), \
             patch.object(m.C,'get_last_error',return_value=5):
            row=m.current_owned_memory(1)
            self.assertEqual(row['owned_private_bytes_current'],203000)
            self.assertTrue(row['owned_private_sample_complete'])
            self.assertEqual(closed,[101,102])
            member=False
            row=m.current_owned_memory(1)
            self.assertFalse(row['owned_private_sample_complete'])
            self.assertEqual(row['owned_private_bytes_current'],0)
            member=True
            readable=False
            with self.assertRaises(OSError):m.current_owned_memory(1)
            self.assertEqual(closed[-1],101)
            exited=True
            row=m.current_owned_memory(1)
            self.assertFalse(row['owned_private_sample_complete'])
            self.assertEqual(row['owned_private_bytes_current'],0)
            self.assertEqual(closed[-2:],[101,102])
            readable=True
            with patch.object(m.k32,'OpenProcess',return_value=0), \
                 patch.object(m.C,'get_last_error',return_value=87):
                row=m.current_owned_memory(1)
                self.assertFalse(row['owned_private_sample_complete'])
                self.assertEqual(row['owned_private_processes'],[])
            count=m.MAX_PROCESSES+1
            with self.assertRaises(RuntimeError):m.current_owned_memory(1)

    def test_real_attached_child_private_usage(self):
        from run_mtp_equivalence import make_mtp_job
        from advanced_guard import shared
        job=make_mtp_job()
        child=None
        try:
            child=subprocess.Popen([sys.executable,'-c',
                  "import sys; data=bytearray(16*1024**2); print('READY',flush=True); sys.stdin.readline()"],
                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            shared.wincheck(shared.k32.AssignProcessToJobObject(job,int(child._handle)))
            ready=queue.Queue(maxsize=1)
            reader=threading.Thread(target=lambda:ready.put(child.stdout.readline()),daemon=True)
            reader.start()
            self.assertEqual(ready.get(timeout=10).strip(),'READY')
            reader.join(timeout=1)
            row=m.current_owned_memory(job)
            self.assertTrue(row['owned_private_sample_complete'])
            # A Windows venv launcher can create interpreter descendants; the Job
            # owns those too, and they must be included rather than discarded.
            self.assertIn(child.pid,[r['pid'] for r in row['owned_private_processes']])
            self.assertLessEqual(len(row['owned_private_processes']),m.MAX_PROCESSES)
            self.assertEqual(row['owned_private_bytes_current'],sum(r['private_bytes'] for r in row['owned_private_processes']))
            self.assertGreaterEqual(row['owned_private_bytes_current'],16*1024**2)
            child.communicate('\n',timeout=10)
            self.assertEqual(child.returncode,0)
        finally:
            shared.k32.CloseHandle(job)
            if child and child.poll() is None:
                child.kill();child.wait(timeout=10)


if __name__=='__main__':unittest.main(verbosity=2)
