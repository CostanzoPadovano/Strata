"""Bounded CLI admission checks that deliberately stop BEFORE any GGUF/CUDA load."""
import json
import os
from pathlib import Path
import subprocess
import sys
import agent_guard as guard

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / 'configs/tier4096.json').read_text())
guard.validate_config(config, 4096)
missing = ROOT / 'nonexistent-cli-pack'
if missing.exists():
    raise RuntimeError('CLI sentinel unexpectedly exists')
base = config['args'].copy()
base[base.index('--pack') + 1] = str(missing)
env = dict(os.environ)
env['STRATA_FORCE_AVX2'] = '1'
env['PATH'] = os.pathsep.join(config['lib_dirs']) + os.pathsep + env['PATH']
rows = []

def check(name, arguments, marker):
    completed = subprocess.run([config['exe'], '--serve', *arguments], env=env,
        capture_output=True, text=True, encoding='utf-8', timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW)
    passed = completed.returncode == 2 and marker in completed.stderr and not completed.stdout
    rows.append({'name': name, 'passed': passed, 'exit_code': completed.returncode,
                 'stderr': completed.stderr, 'context': arguments[arguments.index('--max-context') + 1]})
    if not passed:
        raise AssertionError(json.dumps(rows[-1]))

try:
    for context in guard.LADDER:
        args = base.copy(); args[args.index('--max-context') + 1] = str(context)
        check(f'admitted tier {context}, no model', args, '--expert-cache-exact requires a native expert pack')
    for flag, value in [('--max-context', '12288'), ('--spec', '1'), ('--mtp-device', '1'),
                        ('--prefill', '0'), ('--prefill', '2049'), ('--adapt-every', '4'),
                        ('--pcie-frac', '0.55'), ('--remote-expert-mode', 'graphs'),
                        ('--remote-expert-mode', 'packed')]:
        args = base.copy(); args[args.index(flag) + 1] = value
        marker = 'remote expert layers require native resident admitted-context' if flag == '--max-context' else '--gpu-host-dedup requires'
        check(f'refuse {flag}={value}', args, marker)
    for flag in ('--no-prefill-borrow', '--expert-cache-exact'):
        args = base.copy(); args.remove(flag)
        check(f'refuse missing {flag}', args, '--gpu-host-dedup requires')
    for flag in ('--mmap-experts', '--no-capture', '--vision'):
        check(f'refuse {flag}', [*base, flag], '--gpu-host-dedup requires')
    check('duplicate ownership opt-in', [*base, '--gpu-host-dedup'], 'duplicate --gpu-host-dedup')
    args = base.copy(); args.remove('--gpu-host-dedup')
    args[args.index('--remote-expert-layer-ids') + 1] = '22,24,28,30,33,34'
    args[args.index('--max-context') + 1] = '8192'
    check('old six layers not admitted at8K', args, 'remote expert layers require native resident admitted-context')
finally:
    evidence = {'passed': bool(rows) and all(row['passed'] for row in rows), 'cases': rows,
                'exe_sha256': guard.digest(Path(config['exe'])), 'gguf_loaded': False,
                'scope': 'CLI parser/refusal only; missing pack forces accepted tiers to exit before native GGUF/CUDA allocation'}
    (ROOT / 'cli-admission.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print(f'PASS {len(rows)} native CLI admission/refusal cases; no GGUF load')
