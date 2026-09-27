"""Explicit serial 4K -> 98K admission campaign. Never called by the normal BAT."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.tag):
        raise ValueError('Invalid immutable evidence tag')
    for context, short in ((4096, '4k'), (98304, '98k')):
        for mode, seconds in (('load', 180), ('smoke', 300), ('cache', 900)):
            tag = f'{args.tag}-{short}-{mode}'
            if (ROOT / 'runs' / tag).exists():
                raise FileExistsError('Evidence tag already exists: ' + tag)
            if mode == 'cache':
                command = [sys.executable, str(ROOT / 'manual/cache_guard.py'), '--context', str(context)]
            else:
                command = [sys.executable, str(ROOT / 'manual/vision_guard.py'), mode, '--context', str(context)]
            command += ['--tag', tag, '--timeout', str(seconds)]
            print(f'QUALIFICATION {short} {mode}: bounded serial run', flush=True)
            child = subprocess.Popen(command)
            try:
                code = child.wait(timeout=seconds + 45)
            except BaseException:
                # Closing the guardian process closes its kill-on-close Job; never
                # stop unrelated models, apps, or WSL work by process name.
                child.kill()
                child.wait(timeout=10)
                raise
            if code:
                raise SystemExit(code)
    print('QUALIFICATION PASSED: separate 4K/98K load, vision/text and cache proofs', flush=True)

if __name__ == '__main__':
    main()
