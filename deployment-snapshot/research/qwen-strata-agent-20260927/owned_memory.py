"""Read current private memory only for members of the guardian's existing Job.

PrivateUsage is not total Windows commit or complete driver/shared memory.
The sample is bounded to 16 processes and is not atomic with global counters.
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W

MAX_PROCESSES=16
k32=C.WinDLL('kernel32',use_last_error=True)
psapi=C.WinDLL('psapi',use_last_error=True)


class ProcessIds(C.Structure):
    _fields_=[('assigned',W.DWORD),('listed',W.DWORD),('ids',C.c_size_t*MAX_PROCESSES)]


class ProcessMemory(C.Structure):
    _fields_=[('cb',W.DWORD),('page_faults',W.DWORD)]+[(name,C.c_size_t) for name in
              ('peak_working_set','working_set','quota_peak_paged','quota_paged','quota_peak_nonpaged',
               'quota_nonpaged','pagefile','peak_pagefile','private')]


k32.QueryInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD,C.c_void_p]
k32.QueryInformationJobObject.restype=W.BOOL
k32.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD]
k32.OpenProcess.restype=W.HANDLE
k32.IsProcessInJob.argtypes=[W.HANDLE,W.HANDLE,C.POINTER(W.BOOL)]
k32.IsProcessInJob.restype=W.BOOL
k32.GetExitCodeProcess.argtypes=[W.HANDLE,C.POINTER(W.DWORD)]
k32.GetExitCodeProcess.restype=W.BOOL
k32.CloseHandle.argtypes=[W.HANDLE]
k32.CloseHandle.restype=W.BOOL
psapi.GetProcessMemoryInfo.argtypes=[W.HANDLE,C.POINTER(ProcessMemory),W.DWORD]
psapi.GetProcessMemoryInfo.restype=W.BOOL


def _check(ok):
    if not ok:raise C.WinError(C.get_last_error())


def current_owned_memory(job):
    ids=ProcessIds()
    _check(k32.QueryInformationJobObject(job,3,C.byref(ids),C.sizeof(ids),None))
    if ids.assigned>MAX_PROCESSES or ids.listed>MAX_PROCESSES or ids.listed!=ids.assigned:
        raise RuntimeError('Owned-memory process list exceeds the complete bounded sample')
    rows=[]
    complete=True
    for pid in ids.ids[:ids.listed]:
        if not pid:raise RuntimeError('Invalid PID in owned Job process list')
        handle=k32.OpenProcess(0x1000|0x10,False,pid)  # query-limited + VM-read; no mutation rights
        if not handle:
            if C.get_last_error()==87:  # process exited between the Job query and OpenProcess
                complete=False
                continue
            _check(False)
        try:
            member=W.BOOL()
            _check(k32.IsProcessInJob(handle,job,C.byref(member)))
            if not member.value:  # never read memory counters of a reused non-owned PID
                complete=False
                continue
            memory=ProcessMemory()
            memory.cb=C.sizeof(memory)
            if not psapi.GetProcessMemoryInfo(handle,C.byref(memory),C.sizeof(memory)):
                failed=C.get_last_error()
                exit_code=W.DWORD()
                if k32.GetExitCodeProcess(handle,C.byref(exit_code)) and exit_code.value!=259:
                    complete=False
                    continue
                raise C.WinError(failed)
            rows.append({'pid':int(pid),'private_bytes':int(memory.private)})
        finally:
            _check(k32.CloseHandle(handle))
    return {'owned_private_bytes_current':sum(row['private_bytes'] for row in rows),
            'owned_private_sample_complete':complete,'owned_private_processes':rows}
