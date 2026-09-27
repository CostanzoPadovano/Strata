"""Scoped official-Strata trial guardian; never changes upstream or ordinary launchers."""
from __future__ import annotations
import argparse
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'qwen-strata-20260926'))
from guard import Limits, GiB, k32, wincheck, counters, gpu_counters, job_counters, digest
from check_stock import collect_report, EXE_HASH, REVISION


def policy(name, vram_reserve_mib=None):
    choices = {
        'default': {'ram_floor_gib': 12, 'commit_floor_gib': 16, 'job_cap_gib': 48},
        'no-ram-reserve': {'ram_floor_gib': 0, 'commit_floor_gib': 16, 'job_cap_gib': 48},
        'experimental-memory': {'ram_floor_gib': 0, 'commit_floor_gib': 4, 'job_cap_gib': 60},
        'experimental-no-vram-floor': {'ram_floor_gib': 0, 'commit_floor_gib': 4, 'job_cap_gib': 60},
    }
    if name not in choices:
        raise ValueError('Unknown memory policy')
    if vram_reserve_mib is not None and (name != 'experimental-no-vram-floor'
            or vram_reserve_mib not in (700, 1024, 1536, 2048, 2560, 3072)):
        raise ValueError('Cache-sizing sweeps are restricted to the isolated no-VRAM-floor profile')
    if name != 'default':
        authorization = json.loads((ROOT / 'memory-authorization.json').read_text())
        if name not in authorization['authorized_profiles']:
            raise RuntimeError('This memory exception has not been authorized by the user')
    return {**choices[name], 'gpu_start_floor_mib': 0 if name == 'experimental-no-vram-floor' else 14336,
            'gpu_floor_mib': 0 if name == 'experimental-no-vram-floor' else 2048,
            'vram_reserve_mib': (vram_reserve_mib if vram_reserve_mib is not None else 700)
                               if name == 'experimental-no-vram-floor' else 3072}


def check_resources(memory, gpu, limits, *, start=False, arena=0):
    ram = limits['ram_floor_gib'] * GiB + (arena if start else 0)
    commit = limits['commit_floor_gib'] * GiB + (arena if start else 0)
    if memory['ram_free'] < ram:
        raise RuntimeError(f"Available RAM {memory['ram_free'] / GiB:.2f} GiB < {ram / GiB:.2f} GiB")
    if memory['commit_free'] < commit:
        raise RuntimeError(f"Available commit {memory['commit_free'] / GiB:.2f} GiB < {commit / GiB:.2f} GiB")
    for index, free, temperature in gpu:
        floor = limits['gpu_start_floor_mib' if start else 'gpu_floor_mib']
        if floor and free < floor:
            raise RuntimeError(f'GPU{index}: only {free} MiB free')
        if temperature >= 80:
            raise RuntimeError(f'GPU{index}: temperature {temperature} C')


def make_job(cap):
    if cap not in (48, 60):
        raise ValueError('Unsupported Job cap')
    k32.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
    k32.CreateJobObjectW.restype = W.HANDLE
    k32.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
    k32.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
    k32.IsProcessInJob.argtypes = [W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)]
    k32.CloseHandle.argtypes = [W.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    wincheck(job)
    limits = Limits()
    limits.basic.flags = 0x2000 | 0x200  # kill-on-close and finite job memory; stock CPU scheduling
    limits.job_limit = cap * GiB
    try:
        wincheck(k32.SetInformationJobObject(job, 9, C.byref(limits), C.sizeof(limits)))
    except BaseException:
        k32.CloseHandle(job)
        raise
    return job


def accept_ready(run, limits):
    """Never release the observer using a pre-READY/stale GPU sample."""
    memory, gpu = counters(), gpu_counters()
    check_resources(memory, gpu, limits)
    (run / 'ready-accepted.json').write_text(json.dumps({'memory': memory, 'gpu': gpu}))


def artifact_checks(profile):
    report = collect_report(allow_no_ram_reserve=profile != 'default',
                            allow_experimental_memory=profile in ('experimental-memory', 'experimental-no-vram-floor'),
                            allow_no_gpu_floor=profile == 'experimental-no-vram-floor')
    gates = json.loads((ROOT / 'source-gates.json').read_text())
    if not gates.get('passed') or gates.get('source_revision') != REVISION or gates.get('official_exe_sha256') != EXE_HASH:
        raise RuntimeError('Pristine-source component gates missing, failed or stale')
    manifest = json.loads((ROOT.parent / 'qwen-strata-20260926/runtime-manifest.json').read_text())
    for row in manifest['runtime_files']:
        if row['path'].startswith(('pack-iq3\\', 'mtp\\rt\\', 'pack-iq3/', 'mtp/rt/')):
            path = ROOT.parent / 'qwen-strata-20260926' / row['path']
            if digest(path) != row['sha256']:
                raise RuntimeError(f'Reused pack/MTP identity mismatch: {path.name}')
    return report, gates


def run_trial(args, *, config_file=None, probe_file=None, device_visibility=None):
    # Explicit library entry points for the separate advanced campaign. The
    # stock CLI/defaults continue to expose only the selected single device.
    config_file = Path(config_file) if config_file else ROOT / 'stock-config.json'
    probe_file = Path(probe_file) if probe_file else ROOT / 'native_probe.py'
    visibility = device_visibility if device_visibility is not None else str(args.gpu)
    if visibility not in ('0', '1', '0,1'):
        raise ValueError('Unsupported bounded device visibility')
    base_config_hash = digest(config_file)
    limits = policy(args.profile, args.vram_reserve_mib)
    report, gates = artifact_checks(args.profile)
    check_resources(report['measured_memory'], report['gpu'], limits, start=True, arena=report['expert_arena_bytes'])
    if args.mode != 'load':
        admission = json.loads((ROOT / 'load-admission.json').read_text())
        if (not admission.get('passed') or admission.get('official_exe_sha256') != EXE_HASH
                or admission.get('profile') != args.profile or admission.get('gpu') != args.gpu
                or not admission.get('ready_state_resource_check')
                or admission.get('vram_reserve_mib') != limits['vram_reserve_mib']
                or admission.get('gpu_start_floor_mib') != limits['gpu_start_floor_mib']
                or admission.get('gpu_floor_mib') != limits['gpu_floor_mib']
                or admission.get('pool_workers') != args.pool_workers
                or admission.get('device_visibility', str(args.gpu)) != visibility
                or (admission.get('base_config_sha256') is not None
                    and admission.get('base_config_sha256') != base_config_hash)
                or admission.get('source_gates_sha256') != digest(ROOT / 'source-gates.json')):
            raise RuntimeError('Matching load-only gate must pass first')
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.tag):
        raise ValueError('Invalid isolated run tag')
    run = ROOT / 'runs' / args.tag
    run.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_file.read_text())
    # Normal upstream configuration, not a patch. The no-floor experiment uses
    # stock's 700MiB CUDA-buffer reserve; other profiles retain the 2GiB watchdog.
    config['args'] += ['--vram-reserve-mib', str(limits['vram_reserve_mib'])]
    if args.pool_workers is not None:
        config['args'] += ['--pool-workers', str(args.pool_workers)]
    config['log'] = str(run / 'engine.log')
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    evidence = {'profile': args.profile, **limits, 'gpu': args.gpu,
                'device_visibility': visibility, 'base_config_sha256': base_config_hash,
                'executable_kind': report.get('executable_kind', 'untouched-official-release'),
                'official_exe_sha256': EXE_HASH, 'source_gates_sha256': digest(ROOT / 'source-gates.json'),
                'source_gate_scope': gates.get('evidence_scope'), 'config_sha256': digest(run / 'config.json'),
                'preflight': report, 'passed': False, 'model_started': False}
    (run / 'policy.json').write_text(json.dumps(evidence, indent=2))
    job = process = mutex = None
    owned = False
    k32.CreateMutexW.argtypes = [C.c_void_p, W.BOOL, W.LPCWSTR]
    k32.CreateMutexW.restype = W.HANDLE
    k32.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
    k32.ReleaseMutex.argtypes = [W.HANDLE]
    try:
        mutex = k32.CreateMutexW(None, False, 'Local\\QwenBioagentGpuExclusive')
        wincheck(mutex)
        if k32.WaitForSingleObject(mutex, 0) not in (0, 0x80):
            raise RuntimeError('Another guarded model owns the GPUs')
        owned = True
        # Re-read under the mutex: preflight is not a reservation against races.
        check_resources(counters(), gpu_counters(), limits, start=True, arena=report['expert_arena_bytes'])
        job = make_job(limits['job_cap_gib'])
        env = dict(os.environ)
        for name in list(env):
            if name.startswith('STRATA_'):
                env.pop(name)
        env['STRATA_FORCE_AVX2'] = '1'
        env['CUDA_VISIBLE_DEVICES'] = visibility
        env['PATH'] = os.pathsep.join(config['lib_dirs']) + os.pathsep + env['PATH']
        with (run / 'observer.log').open('x', encoding='utf-8') as log, (run / 'telemetry.jsonl').open('x') as telemetry:
            process = subprocess.Popen([sys.executable, str(probe_file), '--run', str(run), '--mode', args.mode],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            wincheck(k32.AssignProcessToJobObject(job, int(process._handle)))
            member = W.BOOL()
            wincheck(k32.IsProcessInJob(int(process._handle), job, C.byref(member)))
            if not member.value:
                raise RuntimeError('Job attachment verification failed')
            (run / 'job-attached').write_text('owned\n')
            evidence['model_started'] = True
            started = time.monotonic()
            while process.poll() is None:
                memory, gpu = counters(), gpu_counters()
                phase = 'starting'
                try:
                    phase = json.loads((run / 'phase.json').read_text())['phase']
                except (OSError, ValueError):
                    pass
                telemetry.write(json.dumps({'elapsed': time.monotonic() - started, 'phase': phase,
                                             **memory, 'gpu': gpu, **job_counters(job)}) + '\n')
                telemetry.flush()
                check_resources(memory, gpu, limits)
                if (run / 'ready.json').exists() and not (run / 'ready-accepted.json').exists():
                    # READY may appear after the sampled GPU query. Re-sample AFTER
                    # observing it and hold the child before exit or generation.
                    accept_ready(run, limits)
                if time.monotonic() - started > args.timeout:
                    raise RuntimeError('Bounded trial timeout')
                time.sleep(0.5)
            evidence['observer_exit_code'] = process.returncode
            if process.returncode:
                raise RuntimeError(f'Native observer exited {process.returncode}; see engine.log and observer.log')
            result = json.loads((run / 'probe.json').read_text())
            if not result.get('passed') or result.get('native_exit_code') != 0:
                raise RuntimeError('Trial/clean native exit failed')
            evidence['passed'] = True
            if args.mode == 'load':
                (ROOT / 'load-admission.json').write_text(json.dumps({'passed': True, 'profile': args.profile, 'gpu': args.gpu,
                      'official_exe_sha256': EXE_HASH, 'source_gates_sha256': evidence['source_gates_sha256'],
                      'ready_state_resource_check': True, 'vram_reserve_mib': limits['vram_reserve_mib'],
                      'gpu_start_floor_mib': limits['gpu_start_floor_mib'], 'gpu_floor_mib': limits['gpu_floor_mib'],
                      'pool_workers': args.pool_workers,
                      'device_visibility': visibility, 'base_config_sha256': base_config_hash,
                      'run': str(run)}, indent=2))
            print(json.dumps({'passed': True, 'mode': args.mode, 'run': str(run), 'profile': args.profile}), flush=True)
    except BaseException as error:
        evidence['failure'] = str(error)
        raise
    finally:
        if job:
            k32.CloseHandle(job)
        if process and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if owned:
            k32.ReleaseMutex(mutex)
        if mutex:
            k32.CloseHandle(mutex)
        (run / 'result.json').write_text(json.dumps(evidence, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['load', 'smoke', 'bench'])
    parser.add_argument('--profile', choices=['default', 'no-ram-reserve', 'experimental-memory', 'experimental-no-vram-floor'], default='default')
    parser.add_argument('--gpu', type=int, choices=[0, 1], default=1)
    parser.add_argument('--pool-workers', type=int, choices=[4, 6, 8, 10, 12, 16, 19])
    parser.add_argument('--vram-reserve-mib', type=int, choices=[700, 1024, 1536, 2048, 2560, 3072],
                        help='Upstream cache-sizing option only; not a guardian VRAM stop')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if not 30 <= args.timeout <= 1800:
        raise ValueError('Timeout outside bounded range')
    run_trial(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'STOCK STRATA STOP: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
