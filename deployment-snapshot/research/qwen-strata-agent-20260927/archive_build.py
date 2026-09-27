"""Recoverable immutable snapshot of a completed build; never overwrites evidence."""
import argparse
import json
from pathlib import Path
import shutil
import numerical_guard as numerical
import agent_guard as guard

parser = argparse.ArgumentParser()
parser.add_argument('--tag', required=True)
args = parser.parse_args()
if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.tag):
    raise ValueError('Invalid archive tag')
binding, path = numerical.load_binding()
target = guard.ROOT / 'build-snapshots' / args.tag
target.mkdir(parents=True, exist_ok=False)
copied = []

def copy(source, relative):
    dest = target / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    before = guard.digest(source)
    shutil.copy2(source, dest)
    if guard.digest(dest) != before or guard.digest(source) != before:
        raise RuntimeError('Artifact changed while being archived')
    copied.append({'path': str(relative), 'sha256': before})

copy(path, Path('build-binding.json'))
for row in binding['source_files']:
    copy(guard.ROOT / 'source' / row['path'], Path('source') / row['path'])
for name in ('strata.exe', 'mtp_transport_test.exe', 'host_registration_test.exe', 'embedding_cache_test.exe',
             'mtp_equivalence_test.exe', 'mtp_profile_test.exe', 'remote_experts_test.exe',
             'compacted_loader_test.exe', 'remote_experts_profile_test.exe', 'verify_wait_test.exe'):
    copy(guard.ROOT / 'build' / name, Path('build') / name)
for name in ('transport_test.cu', 'host_registration_test.cu', 'embedding_cache_test.cu',
             'mtp_equivalence_test.cpp', 'mtp_profile_test.cpp', 'remote_experts_test.cpp',
             'compacted_loader_test.cpp', 'remote_experts_profile_test.cpp', 'verify_wait_test.cu',
             'agent_guard.py', 'numerical_guard.py', 'prepare_evidence.py', 'native_probe.py',
             'owned_memory.py', 'memory-authorization.json', 'protocol.md'):
    copy(guard.ROOT / name, Path(name))
(target / 'manifest.json').write_text(json.dumps({'executable_sha256': binding['executable_sha256'],
    'build_binding_sha256': guard.digest(path), 'files': copied}, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'snapshot': str(target), 'files': len(copied)}))
