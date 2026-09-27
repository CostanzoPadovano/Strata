"""Non-invasive request I/O/CPU diagnostic of the already-gated native engine.

Does not modify the engine, its configuration, resource floors or admissions.
Win32 read counters include PLE and any other engine reads, not just experts.
"""
from __future__ import annotations

import argparse
import ctypes as C
from ctypes import wintypes as W
import json
from pathlib import Path
import time
import urllib.request

import guard
from probe import call

ROOT = Path(__file__).resolve().parent
k32 = guard.k32


class ProcessEntry(C.Structure):
    _fields_ = [
        ('size', W.DWORD), ('usage', W.DWORD), ('pid', W.DWORD),
        ('heap', C.c_size_t), ('module', W.DWORD), ('threads', W.DWORD),
        ('parent', W.DWORD), ('priority', W.LONG), ('flags', W.DWORD),
        ('exe', W.WCHAR * 260),
    ]


def engine_handle():
    k32.CreateToolhelp32Snapshot.argtypes = [W.DWORD, W.DWORD]
    k32.CreateToolhelp32Snapshot.restype = W.HANDLE
    k32.Process32FirstW.argtypes = [W.HANDLE, C.POINTER(ProcessEntry)]
    k32.Process32NextW.argtypes = [W.HANDLE, C.POINTER(ProcessEntry)]
    k32.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k32.OpenProcess.restype = W.HANDLE
    k32.QueryFullProcessImageNameW.argtypes = [W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)]
    k32.CloseHandle.argtypes = [W.HANDLE]
    snapshot = k32.CreateToolhelp32Snapshot(2, 0)
    if snapshot in (None, C.c_void_p(-1).value):
        raise C.WinError(C.get_last_error())
    matches = []
    entry = ProcessEntry()
    entry.size = C.sizeof(entry)
    expected = (ROOT / 'build-native/strata.exe').resolve()
    try:
        ok = k32.Process32FirstW(snapshot, C.byref(entry))
        while ok:
            if entry.exe.lower() == 'strata.exe':
                handle = k32.OpenProcess(0x1000, False, entry.pid)
                if not handle:
                    raise C.WinError(C.get_last_error())
                path = C.create_unicode_buffer(32768)
                length = W.DWORD(len(path))
                if not k32.QueryFullProcessImageNameW(handle, 0, path, C.byref(length)):
                    k32.CloseHandle(handle)
                    raise C.WinError(C.get_last_error())
                if Path(path.value).resolve() == expected:
                    matches.append((int(entry.pid), handle))
                else:
                    k32.CloseHandle(handle)
            ok = k32.Process32NextW(snapshot, C.byref(entry))
    finally:
        k32.CloseHandle(snapshot)
    if len(matches) != 1:
        for _, handle in matches:
            k32.CloseHandle(handle)
        raise RuntimeError(f'Expected one engine at the verified local path, got {len(matches)}')
    return matches[0]


def sample(handle):
    k32.GetProcessIoCounters.argtypes = [W.HANDLE, C.POINTER(guard.IO)]
    k32.GetProcessTimes.argtypes = [W.HANDLE] + [C.POINTER(C.c_ulonglong)] * 4
    io = guard.IO()
    guard.wincheck(k32.GetProcessIoCounters(handle, C.byref(io)))
    created, exited, kernel, user = [C.c_ulonglong() for _ in range(4)]
    guard.wincheck(k32.GetProcessTimes(handle, C.byref(created), C.byref(exited), C.byref(kernel), C.byref(user)))
    return {
        'read_ops': int(io.read_ops), 'read_bytes': int(io.read),
        'write_ops': int(io.write_ops), 'write_bytes': int(io.write),
        'cpu_seconds': (kernel.value + user.value) / 1e7,
        'clock': time.monotonic(), **guard.counters(),
    }


def measure(handle, name, messages, maximum):
    before = sample(handle)
    response = call(messages, maximum)
    after = sample(handle)
    elapsed = after['clock'] - before['clock']
    generated = response['usage']['completion_tokens']
    row = {
        'name': name, 'usage': response['usage'], 'timings': response['timings'],
        'measured_rates': response['measured_rates'], 'wall_seconds': elapsed,
        'read_ops': after['read_ops'] - before['read_ops'],
        'read_bytes': after['read_bytes'] - before['read_bytes'],
        'cpu_seconds': after['cpu_seconds'] - before['cpu_seconds'],
        'memory_before': {key: before[key] for key in ('ram_free', 'commit_free')},
        'memory_after': {key: after[key] for key in ('ram_free', 'commit_free')},
    }
    row['average_busy_cpu_cores'] = row['cpu_seconds'] / elapsed
    row['read_mib_per_output_token'] = row['read_bytes'] / (1 << 20) / generated if generated else None
    row['read_mib_per_second'] = row['read_bytes'] / (1 << 20) / elapsed
    print(json.dumps(row), flush=True)
    return row, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    if not args.tag or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for ch in args.tag):
        raise ValueError('Invalid run tag')
    run = ROOT / 'runs' / args.tag
    if not run.is_dir() or not (run / 'job-attached').exists():
        raise RuntimeError('The guardian must create and own this run first')
    result = {'passed': False, 'kind': 'io-diagnostic-not-throughput-or-quality-admission', 'rows': []}
    handle = None
    try:
        deadline = time.monotonic() + 180
        while True:
            if (run / 'failure.json').exists():
                raise RuntimeError('Guardian has stopped the run')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8035/health', timeout=2) as request:
                    if json.load(request).get('status') == 'ok':
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Engine was not ready within the diagnostic deadline')
            time.sleep(1)
        pid, handle = engine_handle()
        config = json.loads((run / 'config.json').read_text())
        if config['args'][config['args'].index('--max-context') + 1] != '4096':
            raise RuntimeError('This diagnostic is restricted to the gated 4K context')
        result['engine_pid'] = pid
        result['runtime_manifest_sha256'] = guard.digest(ROOT / 'runtime-manifest.json')
        row, response = measure(handle, 'warmup_math', [{'role': 'user', 'content': 'Quanto fa 17 + 25? Rispondi solo con il numero.'}], 64)
        result['rows'].append(row)
        assert response['choices'][0]['message']['content'].strip() == '42'
        row, response = measure(handle, 'decode256', [{'role': 'user', 'content': 'Scrivi un esempio Python completo e commentato per leggere record FASTQ e calcolare il GC senza usare librerie esterne. Variante 1.'}], 256)
        result['rows'].append(row)
        assert response['usage']['completion_tokens'] == 256
        context = '\n'.join(f'read_{i:04d}\tACGTACGT\tIIIIIIII' for i in range(180))
        row, response = measure(handle, 'prefill3107', [{'role': 'user', 'content': context + '\nQuesta è una tabella dimostrativa, non un FASTQ reale. Qual è la sequenza della prima riga? Rispondi solo ACGTACGT.'}], 32)
        result['rows'].append(row)
        assert response['usage']['prompt_tokens'] == 3107
        assert response['choices'][0]['message']['content'].strip() == 'ACGTACGT'
        result['passed'] = True
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        if handle:
            k32.CloseHandle(handle)
        with (run / 'io-profile.json').open('x', encoding='utf-8') as report:
            json.dump(result, report, indent=2)
        if args.stop:
            (run / 'stop-requested').write_text('io diagnostic finished\n')


if __name__ == '__main__':
    main()
