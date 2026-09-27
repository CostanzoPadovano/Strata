"""Normal Strata server: validated existing admissions, ZERO startup prompts."""
from __future__ import annotations
import argparse
import json
import threading
import time
from pathlib import Path
from vision_campaign import ROOT, MANUAL, admitted, artifact_checks, check_bound, digest, validate_config
from vision_user_launcher import current_load, current_smoke
from manual_guard import stop_on_enter
from serve_only_derivation import derive


def source_checks():
    proof = json.loads((MANUAL / 'serve-only-gates.json').read_text())
    if not proof.get('passed') or proof.get('qualification') != 'offline-startup-only-derivative':
        raise RuntimeError('Serve-only component proof missing')
    if proof['parent_source_gates_sha256'] != digest(ROOT / 'source-gates.json'):
        raise RuntimeError('Serve-only parent proof changed; no automatic tests')
    for row in proof['bound_files']: check_bound(row)
    if (MANUAL / 'serve_only_server.py').read_text() != derive((MANUAL / 'manual_server.py').read_text()):
        raise RuntimeError('Serve-only observer differs beyond the qualified startup/log change')


def console(run, done, timeout):
    position, partial, last = 0, '', ''
    deadline = time.monotonic() + timeout - 30
    while True:
        try:
            phase = json.loads((run / 'phase.json').read_text())['phase']
            if phase != last:
                print('[Strata] ' + phase, flush=True); last = phase
            with (run / 'observer.log').open(encoding='utf-8', errors='replace') as log:
                log.seek(position); chunk = log.read(65536); position = log.tell()
            rows = (partial + chunk).split('\n'); partial = rows.pop()[-4096:]
            for row in rows:
                if row.startswith(('[timings]', '[strata]')): print(row, flush=True)
            if time.monotonic() >= deadline and run.exists():
                (run / 'stop-server').touch(exist_ok=True)
                print('[Strata] Durata massima: arresto ordinato.', flush=True)
                deadline = float('inf')
        except (OSError, ValueError, KeyError): pass
        if done.wait(.25): return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('manual','check'))
    parser.add_argument('--tag')
    parser.add_argument('--timeout', type=int, default=14400)
    opts = parser.parse_args()
    if not 60 <= opts.timeout <= 14400: raise ValueError('Finite lifetime must be60..14400s')
    tag = opts.tag or ('serve-only-' + time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns()%1000000}')
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in tag):
        raise ValueError('Invalid run tag')
    source_checks()
    config_file = ROOT / 'vision98304.json'
    validate_config(json.loads(config_file.read_text()), 98304)
    report, _ = artifact_checks(admitted.PROFILE)
    if not current_load() or not current_smoke(4096) or not current_smoke(98304):
        raise RuntimeError('Existing load/vision admissions stale: stop, do not run tests automatically')
    if opts.mode == 'check':
        print(json.dumps({'component_gates_passed':True,'manual98k_admission_current':True,
            'resources_passed':report['minimum_checks_passed'],'failures':report['failures'],
            'startup_self_test':False,'automatic_test_runs':False,'model_started':False}))
        return 0 if report['minimum_checks_passed'] else 1
    if not report['minimum_checks_passed']: raise RuntimeError('Insufficient startup resources')
    shared = admitted.shared
    shared.ROOT = ROOT
    shared.EXE_HASH = digest(ROOT / 'build/strata.exe')
    shared.policy = admitted.policy
    shared.artifact_checks = artifact_checks
    shared.job_counters = admitted.measured_job_counters
    done = threading.Event(); run = ROOT / 'runs' / tag
    print('Strata 98K - solo server, testo/vision CPU. Nessun prompt o test automatico.', flush=True)
    print('RAMfree8/commitfree4/Job60GiB/GPU<80C; INVIO arresta. Pi indipendente.', flush=True)
    threading.Thread(target=stop_on_enter,args=(run,done),daemon=True).start()
    reporter = threading.Thread(target=console,args=(run,done,opts.timeout),daemon=True); reporter.start()
    try:
        shared.run_trial(argparse.Namespace(mode='bench',profile=admitted.PROFILE,gpu=0,pool_workers=8,
            vram_reserve_mib=None,tag=tag,timeout=opts.timeout),config_file=config_file,
            probe_file=MANUAL / 'serve_only_server.py',device_visibility='0,1')
    finally:
        done.set(); reporter.join(timeout=2)
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as error:
        print('STRATA ARRESTATO: ' + (str(error) or type(error).__name__), flush=True)
        raise SystemExit(1)
