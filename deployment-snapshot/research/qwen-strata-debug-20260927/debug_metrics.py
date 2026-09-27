"""Bounded, observational Windows diagnostics for manual Strata runs.

The collector deliberately avoids command lines, environment data, process
control, CIM/WMI, and PowerShell.  Missing counters are represented as null;
they are never silently converted to zero.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import mmap
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping


MAX_DURATION_SECONDS = 14_520.0
MIN_INTERVAL_SECONDS = 2.0
MAX_LOG_BYTES = 64 * 1024 * 1024
GPU_INTERVAL_SECONDS = 10.0
PROCESS_INTERVAL_SECONDS = 15.0
GPU_TIMEOUT_SECONDS = 3.0
PROCESS_SCAN_BUDGET_SECONDS = 0.300
POST_STOP_SECONDS = 20.0
MAX_PROCESSES = 8_192
MAX_ERRORS = 32
TOP_PROCESS_COUNT = 12


def _safe_number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return int(result) if result.is_integer() else result


class ErrorTracker:
    """Keeps a finite, throttled record of sensor failures."""

    def __init__(self, maximum: int = MAX_ERRORS) -> None:
        self.maximum = maximum
        self._items: dict[tuple[str, str], dict[str, Any]] = {}
        self.dropped = 0

    def record(self, source: str, exc: BaseException | str, now: float | None = None) -> None:
        message = str(exc).replace("\r", " ").replace("\n", " ")[:240]
        key = (source[:64], message)
        timestamp = time.time() if now is None else now
        if key in self._items:
            item = self._items[key]
            item["count"] += 1
            item["last_unix_s"] = timestamp
        elif len(self._items) < self.maximum:
            self._items[key] = {
                "source": key[0], "message": key[1], "count": 1,
                "first_unix_s": timestamp, "last_unix_s": timestamp,
            }
        else:
            self.dropped += 1

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._items.values()]


class MemoryCollector:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("GlobalMemoryStatusEx is Windows-only")
        self._fn = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        self._fn.argtypes = [ctypes.POINTER(self.MEMORYSTATUSEX)]
        self._fn.restype = wintypes.BOOL

    def collect(self) -> dict[str, int]:
        status = self.MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not self._fn(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Windows calls the committed-address-space limit "page file" here.
        return {
            "ram_total_bytes": int(status.ullTotalPhys),
            "ram_available_bytes": int(status.ullAvailPhys),
            "ram_used_bytes": int(status.ullTotalPhys - status.ullAvailPhys),
            "memory_load_percent": int(status.dwMemoryLoad),
            "windows_commit_limit_bytes": int(status.ullTotalPageFile),
            "windows_commit_available_bytes": int(status.ullAvailPageFile),
            "windows_commit_used_bytes": int(status.ullTotalPageFile - status.ullAvailPageFile),
        }


class PagefileCollector:
    class PAGEFILEINFO(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("reserved", wintypes.DWORD),
            ("total_pages", ctypes.c_size_t), ("used_pages", ctypes.c_size_t),
            ("peak_pages", ctypes.c_size_t),
        ]

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("EnumPageFilesW is Windows-only")
        self.page_size = mmap.PAGESIZE
        self._callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, ctypes.c_void_p, ctypes.POINTER(self.PAGEFILEINFO), wintypes.LPCWSTR
        )
        self._fn = ctypes.WinDLL("psapi", use_last_error=True).EnumPageFilesW
        self._fn.argtypes = [self._callback_type, ctypes.c_void_p]
        self._fn.restype = wintypes.BOOL

    def collect(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []

        @self._callback_type
        def callback(_context: int, info_ptr: Any, filename: str) -> bool:
            info = info_ptr.contents
            rows.append({
                "name": Path(filename).name[:128],
                "allocated_bytes": int(info.total_pages) * self.page_size,
                "current_used_bytes": int(info.used_pages) * self.page_size,
                "peak_used_bytes": int(info.peak_pages) * self.page_size,
                "peak_scope": "per_file_lifetime_peak",
            })
            return True

        if not self._fn(callback, None):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "files": rows,
            "allocated_bytes": sum(x["allocated_bytes"] for x in rows),
            "current_used_bytes": sum(x["current_used_bytes"] for x in rows),
            "peak_used_bytes": sum(x["peak_used_bytes"] for x in rows),
            "peak_aggregation_scope": "sum_of_per_file_peaks_not_simultaneous_global_peak",
        }


PDH_COUNTERS = {
    "cpu_total_percent": r"\Processor(_Total)\% Processor Time",
    "page_reads_per_s": r"\Memory\Page Reads/sec",
    "page_writes_per_s": r"\Memory\Page Writes/sec",
    "pages_input_per_s": r"\Memory\Pages Input/sec",
    "pages_output_per_s": r"\Memory\Pages Output/sec",
    "modified_page_list_bytes": r"\Memory\Modified Page List Bytes",
    "pool_paged_bytes": r"\Memory\Pool Paged Bytes",
    "pool_nonpaged_bytes": r"\Memory\Pool Nonpaged Bytes",
    "disk_c_read_bytes_per_s": r"\LogicalDisk(C:)\Disk Read Bytes/sec",
    "disk_c_write_bytes_per_s": r"\LogicalDisk(C:)\Disk Write Bytes/sec",
    "disk_c_queue_length": r"\LogicalDisk(C:)\Current Disk Queue Length",
    "disk_c_read_latency_s": r"\LogicalDisk(C:)\Avg. Disk sec/Read",
    "disk_c_write_latency_s": r"\LogicalDisk(C:)\Avg. Disk sec/Write",
}


class _PdhValueUnion(ctypes.Union):
    _fields_ = [("longValue", wintypes.LONG), ("doubleValue", ctypes.c_double),
                ("largeValue", ctypes.c_longlong), ("ansiStringValue", ctypes.c_char_p),
                ("wideStringValue", wintypes.LPWSTR)]


class _PdhFormattedValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("value", _PdhValueUnion)]


class PdhCollector:
    PDH_FMT_DOUBLE = 0x00000200
    ERROR_SUCCESS = 0

    FORMATTED_VALUE = _PdhFormattedValue

    def __init__(self, counters: Mapping[str, str] = PDH_COUNTERS) -> None:
        if os.name != "nt":
            raise OSError("PDH is Windows-only")
        self._dll = ctypes.WinDLL("pdh", use_last_error=True)
        self._dll.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self._dll.PdhOpenQueryW.restype = wintypes.DWORD
        self._dll.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self._dll.PdhAddEnglishCounterW.restype = wintypes.DWORD
        self._dll.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self._dll.PdhCollectQueryData.restype = wintypes.DWORD
        self._dll.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(self.FORMATTED_VALUE)]
        self._dll.PdhGetFormattedCounterValue.restype = wintypes.DWORD
        self._dll.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        self._dll.PdhCloseQuery.restype = wintypes.DWORD
        self._query = ctypes.c_void_p()
        status = self._dll.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
        if status != self.ERROR_SUCCESS:
            raise OSError(f"PdhOpenQueryW status 0x{status:08x}")
        self._handles: dict[str, ctypes.c_void_p] = {}
        self.add_errors: dict[str, str] = {}
        self.last_errors: dict[str, str] = {}
        for name, path in counters.items():
            handle = ctypes.c_void_p()
            status = self._dll.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(handle))
            if status == self.ERROR_SUCCESS:
                self._handles[name] = handle
            else:
                self.add_errors[name] = f"PDH status 0x{status:08x}"
        status = self._dll.PdhCollectQueryData(self._query)
        if status != self.ERROR_SUCCESS:
            self.close()
            raise OSError(f"PdhCollectQueryData status 0x{status:08x}")
        self._first = True

    def collect(self) -> dict[str, float | None]:
        self.last_errors = dict(self.add_errors)
        status = self._dll.PdhCollectQueryData(self._query)
        if status != self.ERROR_SUCCESS:
            raise OSError(f"PdhCollectQueryData status 0x{status:08x}")
        first = self._first
        self._first = False
        result: dict[str, float | None] = {name: None for name in PDH_COUNTERS}
        if first:
            return result
        for name, handle in self._handles.items():
            value = self.FORMATTED_VALUE()
            status = self._dll.PdhGetFormattedCounterValue(
                handle, self.PDH_FMT_DOUBLE, None, ctypes.byref(value)
            )
            if status == self.ERROR_SUCCESS and value.CStatus in (0, 1):
                number = _safe_number(value.value.doubleValue)
                result[name] = float(number) if number is not None else None
            else:
                self.last_errors[name] = f"PDH format status 0x{status:08x}, data 0x{value.CStatus:08x}"
        return result

    def close(self) -> None:
        query, self._query = getattr(self, "_query", None), ctypes.c_void_p()
        if query and query.value:
            self._dll.PdhCloseQuery(query)


def query_nvidia_smi(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    timeout: float = GPU_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    fields = (
        "index,name,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu,power.draw"
    )
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    if runner is not None:  # deterministic injection seam for unit tests
        completed = runner(
            command, capture_output=True, text=True, timeout=timeout, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = completed.stdout
        returncode = completed.returncode
        error_text = completed.stderr
        if len(output.encode("utf-8", errors="replace")) > 64 * 1024:
            raise OSError("nvidia-smi output exceeded 64 KiB")
    else:
        process = popen_factory(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), bufsize=0,
        )
        captured = bytearray()
        overflow = [False]

        def drain() -> None:
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                remaining = 64 * 1024 + 1 - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
                if len(captured) > 64 * 1024:
                    overflow[0] = True

        reader = threading.Thread(target=drain, name="nvidia-smi-reader", daemon=True)
        reader.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()  # only the child created above; never a sampled process
            process.wait(timeout=1.0)
            reader.join(timeout=1.0)
            raise
        reader.join(timeout=1.0)
        if reader.is_alive():
            process.kill()
            raise OSError("nvidia-smi output reader did not finish")
        if overflow[0]:
            raise OSError("nvidia-smi output exceeded 64 KiB")
        output = captured.decode("utf-8", errors="replace")
        error_text = output
    if returncode != 0:
        raise OSError(f"nvidia-smi exit {returncode}: {error_text[:160]}")
    rows = []
    for line in output.splitlines()[:16]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9:
            continue
        rows.append({
            "index": _safe_number(parts[0]), "name": parts[1][:128],
            "driver_version": parts[2][:64],
            "memory_total_mib": _safe_number(parts[3]), "memory_used_mib": _safe_number(parts[4]),
            "memory_free_mib": _safe_number(parts[5]), "utilization_percent": _safe_number(parts[6]),
            "temperature_c": _safe_number(parts[7]), "power_w": _safe_number(parts[8]),
        })
    if not rows:
        raise OSError("nvidia-smi returned no parseable GPUs")
    return rows


class ProcessCollector:
    """Native Toolhelp/process-memory scan; never reads command lines."""

    TH32CS_SNAPPROCESS = 0x00000002
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260),
        ]

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("process scan is Windows-only")
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self._k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self._k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(self.PROCESSENTRY32W)]
        self._k32.Process32FirstW.restype = wintypes.BOOL
        self._k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(self.PROCESSENTRY32W)]
        self._k32.Process32NextW.restype = wintypes.BOOL
        self._k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._k32.OpenProcess.restype = wintypes.HANDLE
        self._k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self._k32.GetProcessTimes.restype = wintypes.BOOL
        self._k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k32.CloseHandle.restype = wintypes.BOOL
        self._psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(self.PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
        self._psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self._previous_cpu: dict[int, tuple[int, float]] = {}

    @staticmethod
    def _groups(name: str) -> list[str]:
        lowered = name.casefold()
        groups = []
        if "git" in lowered:
            groups.append("git")
        if "codex" in lowered:
            groups.append("codex")
        if lowered in {"vmmem", "vmmemwsl", "wsl.exe", "wslhost.exe"} or "vmmemwsl" in lowered:
            groups.append("vmmem_wsl")
        if any(token in lowered for token in ("llama", "strata", "qwen")):
            groups.append("native")
        if any(token in lowered for token in ("clip", "encoder", "vision")):
            groups.append("encoder")
        return groups

    @staticmethod
    def _filetime_value(value: wintypes.FILETIME) -> int:
        return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

    def collect(self, budget: float = PROCESS_SCAN_BUDGET_SECONDS) -> dict[str, Any]:
        started = time.monotonic()
        snapshot = self._k32.CreateToolhelp32Snapshot(self.TH32CS_SNAPPROCESS, 0)
        if snapshot == self.INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())
        rows: list[dict[str, Any]] = []
        seen: list[tuple[int, str]] = []
        truncated = False
        current_cpu: dict[int, tuple[int, float]] = {}
        try:
            entry = self.PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            more = bool(self._k32.Process32FirstW(snapshot, ctypes.byref(entry)))
            while more:
                if len(seen) >= MAX_PROCESSES or time.monotonic() - started >= budget:
                    truncated = True
                    break
                pid, name = int(entry.th32ProcessID), str(entry.szExeFile)[:128]
                seen.append((pid, name))
                access = self.PROCESS_QUERY_LIMITED_INFORMATION | self.PROCESS_VM_READ
                handle = self._k32.OpenProcess(access, False, pid)
                if not handle:
                    handle = self._k32.OpenProcess(self.PROCESS_QUERY_INFORMATION | self.PROCESS_VM_READ, False, pid)
                if handle:
                    try:
                        memory = self.PROCESS_MEMORY_COUNTERS_EX()
                        memory.cb = ctypes.sizeof(memory)
                        if self._psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
                            cpu_percent = None
                            cpu_time_s = None
                            if self._k32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
                                ticks = self._filetime_value(kernel) + self._filetime_value(user)
                                cpu_time_s = ticks / 10_000_000.0
                                wall = time.monotonic()
                                old = self._previous_cpu.get(pid)
                                current_cpu[pid] = (ticks, wall)
                                if old and wall > old[1] and ticks >= old[0]:
                                    cpu_percent = (ticks - old[0]) / 10_000_000 / (wall - old[1]) * 100.0
                            rows.append({
                                "pid": pid, "name": name,
                                "working_set_bytes": int(memory.WorkingSetSize),
                                "private_bytes": int(memory.PrivateUsage),
                                "cpu_percent_one_core": _safe_number(cpu_percent),
                                "cpu_time_s": _safe_number(cpu_time_s),
                            })
                    finally:
                        self._k32.CloseHandle(handle)
                more = bool(self._k32.Process32NextW(snapshot, ctypes.byref(entry)))
        finally:
            self._k32.CloseHandle(snapshot)
        self._previous_cpu = current_cpu
        return summarize_processes(
            seen, rows, truncated=truncated,
            scan_duration_ms=round((time.monotonic() - started) * 1000, 3),
            monitor_pid=os.getpid(), group_classifier=self._groups,
        )


def summarize_processes(
    seen: Iterable[tuple[int, str]], readable_rows: Iterable[Mapping[str, Any]], *,
    truncated: bool, scan_duration_ms: float, monitor_pid: int,
    group_classifier: Callable[[str], list[str]] = ProcessCollector._groups,
) -> dict[str, Any]:
    """Separate name enumeration completeness from optional memory access."""
    seen_rows = list(seen)
    readable = [dict(row) for row in readable_rows]
    sums: dict[str, dict[str, Any]] = {
        name: {
            "count": 0, "memory_read_count": 0,
            "working_set_bytes": None, "private_bytes": None,
            "memory_complete": True,
        }
        for name in ("git", "codex", "vmmem_wsl", "native", "encoder")
    }
    for _pid, name in seen_rows:
        for group in group_classifier(name):
            sums[group]["count"] += 1
    for row in readable:
        for group in group_classifier(str(row["name"])):
            item = sums[group]
            item["memory_read_count"] += 1
            item["working_set_bytes"] = (item["working_set_bytes"] or 0) + int(row["working_set_bytes"])
            item["private_bytes"] = (item["private_bytes"] or 0) + int(row["private_bytes"])
    for item in sums.values():
        item["memory_complete"] = item["memory_read_count"] == item["count"] and not truncated
    top = sorted(readable, key=lambda row: row["working_set_bytes"], reverse=True)[:TOP_PROCESS_COUNT]
    self_row = next((row for row in readable if row["pid"] == monitor_pid), None)
    enumerated_count = len(seen_rows)
    memory_read_count = len(readable)
    return {
        "scanned_count": enumerated_count, "enumerated_count": enumerated_count,
        "memory_read_count": memory_read_count,
        "inaccessible_count": enumerated_count - memory_read_count,
        "memory_complete": memory_read_count == enumerated_count and not truncated,
        "truncated": truncated, "scan_duration_ms": scan_duration_ms,
        "groups": sums, "top_by_working_set": top, "monitor_process": self_row,
    }


class ParentWatcher:
    """A wait-only parent handle; it can neither terminate nor modify a process."""

    SYNCHRONIZE = 0x00100000
    WAIT_TIMEOUT = 0x00000102

    def __init__(self, pid: int) -> None:
        if os.name != "nt":
            raise OSError("parent watching is Windows-only")
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._k32.OpenProcess.restype = wintypes.HANDLE
        self._k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._k32.WaitForSingleObject.restype = wintypes.DWORD
        self._k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k32.CloseHandle.restype = wintypes.BOOL
        self._handle = self._k32.OpenProcess(self.SYNCHRONIZE, False, pid)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def exited(self) -> bool:
        return self._k32.WaitForSingleObject(self._handle, 0) != self.WAIT_TIMEOUT

    def close(self) -> None:
        handle, self._handle = getattr(self, "_handle", None), None
        if handle:
            self._k32.CloseHandle(handle)


def observe_run(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    run_path = Path(path)
    result = {"run_name": run_path.name[:128], "observed_phase": None, "result_exists": False}
    if not run_path.name.startswith("debug-"):
        result["observation_error"] = "run directory name must start with debug-"
        return result
    result["result_exists"] = (run_path / "result.json").is_file()
    phase_path = run_path / "phase.json"
    try:
        if not phase_path.is_file():
            return result
        if phase_path.stat().st_size > 16 * 1024:
            result["observation_error"] = "phase.json exceeds 16 KiB"
            return result
        raw = json.loads(phase_path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            # Deliberately allow only status-like fields, never prompts/configuration.
            phase = {key: str(raw[key])[:160] for key in ("phase", "status", "state", "stage") if key in raw}
            result["observed_phase"] = phase or None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["observation_error"] = str(exc)[:160]
    return result


class SummaryAccumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.metrics: dict[str, dict[str, float | int]] = {}
        self.intervals: list[float] = []

    def add(self, row: Mapping[str, Any]) -> None:
        self.samples += 1
        interval = _safe_number(row.get("wall_interval_s"))
        if interval is not None:
            self.intervals.append(float(interval))
        self._walk("", row)

    def _walk(self, prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                self._walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, list) and prefix == "gpus":
            for index, child in enumerate(value[:16]):
                self._walk(f"gpus.gpu{index}", child)
        elif not isinstance(value, (list, bool)):
            number = _safe_number(value)
            if number is not None:
                item = self.metrics.setdefault(prefix, {"min": number, "max": number, "available": 0})
                item["min"] = min(item["min"], number)
                item["max"] = max(item["max"], number)
                item["available"] += 1

    def result(self) -> dict[str, Any]:
        interval = None
        if self.intervals:
            interval = {
                "min_s": min(self.intervals), "max_s": max(self.intervals),
                "mean_s": sum(self.intervals) / len(self.intervals),
            }
        metrics = {
            name: {**values, "availability_fraction": values["available"] / self.samples}
            for name, values in self.metrics.items()
            if not name.endswith(("unix_s", "monotonic_s", "wall_interval_s"))
        }
        return {"sample_count": self.samples, "wall_intervals": interval, "metrics": metrics}


class JsonlWriter:
    def __init__(self, path: Path, maximum_bytes: int = MAX_LOG_BYTES) -> None:
        self.path = path
        self.maximum_bytes = maximum_bytes
        self.bytes_written = 0
        self.capped = False
        # A debug run must use a fresh output directory; never blend evidence.
        self._file = path.open("xb", buffering=0)

    def write(self, row: Mapping[str, Any]) -> bool:
        payload = (json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        if self.bytes_written + len(payload) > self.maximum_bytes:
            self.capped = True
            return False
        self._file.write(payload)
        self._file.flush()
        self.bytes_written += len(payload)
        return True

    def close(self) -> None:
        self._file.close()


def _try_collect(source: str, collector: Any, errors: ErrorTracker) -> Any:
    if collector is None:
        return None
    try:
        result = collector.collect()
        for detail, message in getattr(collector, "last_errors", {}).items():
            errors.record(f"{source}.{detail}", message)
        return result
    except Exception as exc:
        errors.record(source, exc)
        return None


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run_sampler(
    output: Path | str,
    stop_file: Path | str,
    duration: float,
    interval: float,
    *,
    parent_pid: int | None = None,
    observe_run_path: Path | str | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    factories: Mapping[str, Callable[[], Any]] | None = None,
    gpu_query: Callable[[], list[dict[str, Any]]] = query_nvidia_smi,
    max_log_bytes: int = MAX_LOG_BYTES,
    post_stop_seconds: float = POST_STOP_SECONDS,
) -> dict[str, Any]:
    if not MIN_INTERVAL_SECONDS <= interval:
        raise ValueError(f"interval must be >= {MIN_INTERVAL_SECONDS}")
    if not 0 < duration <= MAX_DURATION_SECONDS:
        raise ValueError(f"duration must be in (0, {MAX_DURATION_SECONDS}]")
    output_path, stop_path = Path(output), Path(stop_file)
    output_path.mkdir(parents=True, exist_ok=True)
    errors, accumulator = ErrorTracker(), SummaryAccumulator()
    default_factories = {
        "memory": MemoryCollector, "pagefile": PagefileCollector,
        "pdh": PdhCollector, "process": ProcessCollector,
    }
    default_factories.update({key: value for key, value in (factories or {}).items() if key != "parent"})
    collectors: dict[str, Any] = {}
    for name, factory in default_factories.items():
        try:
            collectors[name] = factory()
        except Exception as exc:
            errors.record(name + "_init", exc)
            collectors[name] = None
    parent = None
    if parent_pid is not None:
        try:
            parent_factory = (factories or {}).get("parent", ParentWatcher)
            parent = parent_factory(parent_pid)
        except Exception as exc:
            errors.record("parent_init", exc)
    if parent_pid is not None and parent is None:
        for collector in collectors.values():
            close = getattr(collector, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass
        raise RuntimeError("could not open the requested parent with wait-only access")

    writer = JsonlWriter(output_path / "samples.jsonl", max_log_bytes)
    started = clock()
    last_sample = None
    next_sample = started
    next_gpu = started
    next_process = started
    latest_gpu = None
    latest_process = None
    gpu_sample_elapsed = None
    process_sample_elapsed = None
    reason = "duration"
    tail_deadline = None
    first_row_written = False
    try:
        while True:
            now = clock()
            if now - started >= duration:
                break
            trigger = None
            if stop_path.exists():
                trigger = "stop_file"
            elif parent is not None:
                try:
                    if parent.exited():
                        trigger = "parent_exit"
                except Exception as exc:
                    errors.record("parent_wait", exc)
                    trigger = "parent_wait_error"
            if trigger is not None and tail_deadline is None:
                reason = trigger
                tail_deadline = min(started + duration, now + max(0.0, post_stop_seconds))
            if tail_deadline is not None and now >= tail_deadline:
                break
            if now < next_sample:
                sleeper(min(next_sample - now, 0.5))
                continue
            collection_started = clock()
            if now >= next_gpu:
                gpu_sample_elapsed = now - started
                try:
                    latest_gpu = gpu_query()
                except Exception as exc:
                    errors.record("nvidia_smi", exc)
                    latest_gpu = None
                next_gpu = now + GPU_INTERVAL_SECONDS
            if now >= next_process:
                process_sample_elapsed = now - started
                latest_process = _try_collect("process", collectors["process"], errors)
                next_process = now + PROCESS_INTERVAL_SECONDS
            row = {
                "schema_version": 1, "unix_s": time.time(), "monotonic_s": now,
                "elapsed_s": now - started,
                "wall_interval_s": None if last_sample is None else now - last_sample,
                "memory": _try_collect("memory", collectors["memory"], errors),
                "pagefiles": _try_collect("pagefile", collectors["pagefile"], errors),
                "pdh": _try_collect("pdh", collectors["pdh"], errors),
                "gpus": latest_gpu, "processes": latest_process,
                "gpu_sample_elapsed_s": gpu_sample_elapsed,
                "gpu_sample_age_s": None if gpu_sample_elapsed is None else (now - started - gpu_sample_elapsed),
                "process_sample_elapsed_s": process_sample_elapsed,
                "process_sample_age_s": None if process_sample_elapsed is None else (now - started - process_sample_elapsed),
                "observed_run": observe_run(observe_run_path),
                "errors_seen": sum(item["count"] for item in errors.snapshot()) + errors.dropped,
            }
            row["collection_duration_ms"] = max(0.0, (clock() - collection_started) * 1000.0)
            accumulator.add(row)
            if not writer.write(row):
                reason = "log_cap"
                break
            if not first_row_written:
                first_row_written = True
                _atomic_json(output_path / "ready.json", {
                    "schema_version": 1, "ready": True, "pid": os.getpid(),
                    "parent_pid": os.getppid(),
                    "first_sample_unix_s": row["unix_s"],
                })
            last_sample = now
            next_sample += interval
            if next_sample <= now:
                next_sample = now + interval
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    finally:
        writer.close()
        for collector in collectors.values():
            close = getattr(collector, "close", None)
            if close:
                try:
                    close()
                except Exception as exc:
                    errors.record("cleanup", exc)
        if parent is not None:
            try:
                parent.close()
            except Exception as exc:
                errors.record("parent_cleanup", exc)
    summary = {
        "schema_version": 1, "reason": reason,
        "requested_duration_s": duration, "requested_interval_s": interval,
        "elapsed_s": clock() - started, "samples_jsonl_bytes": writer.bytes_written,
        "log_capped": writer.capped,
        "ready_written": first_row_written,
        "counter_scope": {
            "ram_commit": "system-wide; Windows commit, not actual pagefile use",
            "pagefiles": "system-wide EnumPageFilesW; peak is sum of per-file peaks, not a simultaneous global peak",
            "pdh": "system-wide CPU/memory and LogicalDisk C:",
            "gpu": "all GPUs returned by nvidia-smi",
            "process": "bounded name-only native snapshot; CPU is one-core percentage",
            "observed_run": "phase/status/state/stage only (phase.json <=16 KiB); result existence only",
            "wddm_shared_gpu_memory": "unavailable (not sampled)",
        },
        **accumulator.result(), "errors": errors.snapshot(), "dropped_errors": errors.dropped,
    }
    _atomic_json(output_path / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop-file", required=True, type=Path)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--interval", default=5.0, type=float)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--observe-run", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval < MIN_INTERVAL_SECONDS:
        raise SystemExit(f"--interval must be >= {MIN_INTERVAL_SECONDS}")
    if not 0 < args.duration <= MAX_DURATION_SECONDS:
        raise SystemExit(f"--duration must be in (0, {MAX_DURATION_SECONDS}]")
    if args.parent_pid is not None and args.parent_pid <= 0:
        raise SystemExit("--parent-pid must be positive")
    run_sampler(
        args.output, args.stop_file, args.duration, args.interval,
        parent_pid=args.parent_pid, observe_run_path=args.observe_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
