"""Read-only minimum resource check for the untouched official Strata release.

Never starts the model or grants a runtime admission. Reports lower bounds,
not an estimate of total runtime memory. Explicit user-authorized exceptions
affect this isolated stock trial only; all ordinary profiles remain unchanged.
"""
from __future__ import annotations
import datetime
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'qwen-strata-20260926'
sys.path.insert(0, str(PREVIOUS))
from guard import counters, gpu_counters

GIB = 1 << 30
REVISION = '6da1f667e86558b152ab128edf3ebf77a80a9e57'
ZIP_HASH = '34ad72b75836ce737a5acd0b2cceee0c03a4b1b4420d286f939705b9bb8e60be'
EXE_HASH = 'e3684cc5c6ff51cfff3c4ebc864be6fa6c7179ae98176319f3da042cc1dd0c97'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            result.update(block)
    return result.hexdigest()


def minimum_failures(memory, gpu, arena, *, ram_reserve_bytes=12 * GIB, commit_reserve_bytes=16 * GIB, gpu_floor_mib=2048):
    failures = []
    # Both bounds ignore auxiliary CPU allocations and WDDM/GPU backing:
    # exceeding either is already sufficient to refuse the full load.
    if ram_reserve_bytes not in (0, 12 * GIB):
        raise ValueError('Only the default or explicitly authorized zero RAM reserve is supported')
    if commit_reserve_bytes not in (4 * GIB, 16 * GIB):
        raise ValueError('Only the default or explicitly authorized 4GiB commit reserve is supported')
    if gpu_floor_mib not in (0, 2048):
        raise ValueError('Only the default or explicitly authorized zero VRAM floor is supported')
    if memory['ram_free'] < arena + ram_reserve_bytes:
        failures.append(f'Available RAM cannot hold the resident expert arena with the {ram_reserve_bytes / GIB:g} GiB RAM reserve')
    if memory['commit_free'] < arena + commit_reserve_bytes:
        failures.append(f'Available commit cannot hold even the expert arena while retaining the {commit_reserve_bytes / GIB:g} GiB floor')
    for index, free, temperature in gpu:
        if gpu_floor_mib and free < gpu_floor_mib:
            failures.append(f'GPU{index} has less than {gpu_floor_mib} MiB free')
        if temperature >= 80:
            failures.append(f'GPU{index} is at least 80 C')
    return failures


def collect_report(*, allow_no_ram_reserve=False, allow_experimental_memory=False, allow_no_gpu_floor=False):
    if allow_no_gpu_floor:
        if not allow_experimental_memory:
            raise ValueError('The zero VRAM floor is restricted to the confirmed experimental memory profile')
        authorization = json.loads((ROOT / 'memory-authorization.json').read_text())
        if 'experimental-no-vram-floor' not in authorization['authorized_profiles']:
            raise RuntimeError('Removing VRAM floors requires separate user confirmation')
    if allow_experimental_memory:
        authorization = json.loads((ROOT / 'memory-authorization.json').read_text())
        if 'experimental-memory' not in authorization['authorized_profiles']:
            raise RuntimeError('Experimental memory profile requires explicit user confirmation')
        allow_no_ram_reserve = True
    source = ROOT / 'source'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != REVISION or changes.strip():
        raise RuntimeError('The stock source must remain at the downloaded revision with no tracked changes')
    archive = ROOT / 'strata-windows-x64-v0.1.2.zip'
    exe = ROOT / 'official-engine/strata.exe'
    if archive.stat().st_size != 80296747 or digest(archive) != ZIP_HASH or digest(exe) != EXE_HASH:
        raise RuntimeError('Official release identity mismatch')
    metadata = json.loads((ROOT / 'official-engine/BUILD.json').read_text())
    if metadata.get('version') != '0.1.2' or metadata.get('source') != 'release' or 120 not in metadata.get('archs', []):
        raise RuntimeError('Unexpected official release metadata')
    proof = json.loads((PREVIOUS / 'model-proof.json').read_text())
    for row in proof['files']:
        info = Path(row['path']).stat()
        if info.st_size != row['bytes'] or info.st_mtime_ns != row['mtime_ns']:
            raise RuntimeError('GGUF file changed since the recorded full SHA256 verification')
    rows = []
    for line in (PREVIOUS / 'pack-iq3/native_experts.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            fields = line.split()
            rows.append((int(fields[0]), int(fields[4])))
    if [layer for layer, _ in rows] != list(range(48)):
        raise RuntimeError('Unexpected expert layout')
    payload = sum(size for _, size in rows) * 512
    largest = max(size for _, size in rows)
    if payload != 42912972800 or largest != 2329600:
        raise RuntimeError('Unexpected expert payload sizes')
    arena = payload + largest  # ArenaExpertSource reserves one extra maximum blob.
    memory = counters()
    gpu = gpu_counters()
    reserve = 0 if allow_no_ram_reserve else 12 * GIB
    commit_reserve = (4 if allow_experimental_memory else 16) * GIB
    gpu_floor = 0 if allow_no_gpu_floor else 2048
    failures = minimum_failures(memory, gpu, arena, ram_reserve_bytes=reserve,
                                commit_reserve_bytes=commit_reserve, gpu_floor_mib=gpu_floor)
    report = {
        'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_revision': revision, 'source_tracked_changes': changes,
        'official_release': metadata, 'archive_sha256': ZIP_HASH, 'exe_sha256': EXE_HASH,
        'model': 'existing ISTA GSQ-RCO IQ3_XXS',
        'model_hash_evidence': str(PREVIOUS / 'model-proof.json'),
        'model_size_mtime_revalidated': True,
        'expert_payload_bytes': payload, 'expert_arena_bytes': arena,
        'ram_reserve_gib': reserve / GIB,
        'user_authorized_no_ram_reserve': allow_no_ram_reserve,
        'commit_reserve_gib': commit_reserve / GIB,
        'user_authorized_experimental_memory': allow_experimental_memory,
        'user_authorized_no_gpu_floor': allow_no_gpu_floor,
        'gpu_minimum_free_mib': gpu_floor,
        'gpu_temperature_limit_c': 80,
        'available_ram_gib': memory['ram_free'] / GIB,
        'available_commit_gib': memory['commit_free'] / GIB,
        'minimum_ram_free_gib_before_auxiliary_allocations': (arena + reserve) / GIB,
        'minimum_commit_free_gib_before_auxiliary_allocations': (arena + commit_reserve) / GIB,
        'ram_shortfall_to_minimum_gib': max(0, arena + reserve - memory['ram_free']) / GIB,
        'measured_memory': memory, 'gpu': gpu,
        'minimum_checks_passed': not failures, 'failures': failures,
        'model_started': False, 'runtime_admission': False,
        'note': 'These are necessary lower bounds, not complete admission or a full-model benchmark. Auxiliary memory, synthetic gates and guarded load remain separate.',
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-no-ram-reserve', action='store_true',
                        help='Explicit September26 user-authorized opt-in; remove only the stock trial RAM reserve')
    parser.add_argument('--allow-experimental-memory', action='store_true',
                        help='Separately confirmed 4K experiment: zero RAM reserve and 4GiB commit reserve')
    parser.add_argument('--allow-no-gpu-floor', action='store_true',
                        help='Separately confirmed VRAM-floor removal; requires --allow-experimental-memory')
    args = parser.parse_args()
    report = collect_report(allow_no_ram_reserve=args.allow_no_ram_reserve,
                            allow_experimental_memory=args.allow_experimental_memory,
                            allow_no_gpu_floor=args.allow_no_gpu_floor)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    path = ROOT / f'preflight-{stamp}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print(f'Report: {path}', flush=True)
    return 2 if report['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
