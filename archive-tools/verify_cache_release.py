"""Streaming integrity/privacy checks for the frozen archive; never inference."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = '2e630a0b125a0b4f2be300678216d8aca2e086d1e5c2d9cb54855b55685c4c77'
PATCH = '3b116921455ac861152ad4d057647a06c54b5c3f0e937d3a28f2a87e142e95da'
ENCODER_PATH = 'research/qwen-strata-stock-20260926/official-engine/strata-vision.exe'
ENCODER_SHA = '98ffde98f5388b518d8581b2983973d785f20f00a21e5e82393b38f4188e094c'
SECRET = re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b')
PRIVATE = re.compile(r'^(?:api[-.]?key(?:\..*)?|auth\.json|settings\.json|models\.json|requests.*\.jsonl|session.*\.jsonl)$', re.I)


def check_stream(stream):
    digest = hashlib.sha256()
    tail = b''
    size = 0
    while chunk := stream.read(1 << 20):
        if SECRET.search(tail + chunk):
            raise RuntimeError('Potential credential detected; do not publish')
        digest.update(chunk)
        tail = chunk[-256:]
        size += len(chunk)
    return digest.hexdigest(), size


def verify_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or PRIVATE.match(path.name):
        raise RuntimeError('Unsafe/private archive path: ' + name)
    model_prefixes = ('models/', 'research/qwen-strata-20260926/mtp/', 'research/qwen-strata-20260926/pack-iq3/')
    if path.suffix.lower() == '.gguf' or name.startswith(model_prefixes) or any(p in {'.venv', '.git'} for p in path.parts):
        raise RuntimeError('Excluded model/environment data in archive: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zip', type=Path)
    parser.add_argument('--index', action='store_true', help='Check staged Git bytes and every changed file for privacy')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'local-evidence/source-snapshot-manifest.json').read_text())
    if manifest['engine_sha256'] != ENGINE:
        raise RuntimeError('Wrong frozen source identity')
    for row in manifest['files']:
        verify_path(row['path'])
        file = (ROOT / row['path']).resolve()
        if not file.is_relative_to(ROOT):
            raise RuntimeError('Source path escaped archive')
        with file.open('rb') as stream:
            digest, size = check_stream(stream)
        if (digest, size) != (row['sha256'], row['bytes']):
            raise RuntimeError('Source hash/length mismatch: ' + row['path'])
    patch = ROOT / 'deployment-snapshot/research/qwen-strata-update-20260927/cache-idle-source.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != PATCH:
        raise RuntimeError('Frozen source patch differs')
    evidence = json.loads((ROOT / 'local-evidence/cache-pi-trial.json').read_text())
    expected_keys = {'generated', 'prompt_tokens', 'prompt_ms', 'decode_ms', 'finish',
                     'draft_accepted', 'draft_offered', 'prompt_reused',
                     'fresh_prompt_tokens', 'prefill_processed_tokens'}
    if evidence['engine_sha256'] != ENGINE or evidence['contains_prompts'] or len(evidence['rows']) != 2:
        raise RuntimeError('Unexpected numeric-only Pi evidence')
    if any(set(row) != expected_keys for row in evidence['rows']) or not all(evidence['checks'].values()):
        raise RuntimeError('Numeric evidence/checks changed')
    asset_manifest = json.loads((ROOT / 'local-evidence/runtime-assets-manifest.json').read_text())
    rows = {row['path']: row for row in asset_manifest['files']}
    if len(rows) != len(asset_manifest['files']) or asset_manifest['engine_sha256'] != ENGINE:
        raise RuntimeError('Duplicate runtime member/wrong engine')
    with zipfile.ZipFile(args.zip) as archive:
        members = [info.filename for info in archive.infolist() if not info.is_dir()]
        if len(members) != len(set(members)) or set(members) != set(rows) | {'runtime-assets-manifest.json'}:
            raise RuntimeError('Extra/missing/duplicate ZIP member')
        if json.loads(archive.read('runtime-assets-manifest.json')) != asset_manifest:
            raise RuntimeError('ZIP manifest differs from committed manifest')
        for name, row in rows.items():
            verify_path(name)
            exact_encoder = name == ENCODER_PATH and row['sha256'] == ENCODER_SHA and row['bytes'] == 90725376
            if archive.getinfo(name).file_size > 50 << 20 and not exact_encoder:
                raise RuntimeError('Oversized runtime member')
            with archive.open(name) as stream:
                digest, size = check_stream(stream)
            if (digest, size) != (row['sha256'], row['bytes']):
                raise RuntimeError('Runtime hash/length mismatch: ' + name)
        if rows['research/qwen-strata-update-20260927/build/strata.exe']['sha256'] != ENGINE:
            raise RuntimeError('Exact candidate executable missing')
        if rows.get(ENCODER_PATH, {}).get('sha256') != ENCODER_SHA:
            raise RuntimeError('Exact vision encoder missing')
    with args.zip.open('rb') as stream:
        zip_sha, _ = check_stream(stream)
    if args.zip.with_suffix('.zip.sha256').read_text().split()[0] != zip_sha:
        raise RuntimeError('ZIP sidecar mismatch')
    staged_count = None
    if args.index:
        index = {}
        for record in subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT).split(b'\0'):
            if record:
                header, name = record.split(b'\t', 1)
                mode, oid, stage = header.split()
                if stage != b'0':
                    raise RuntimeError('Unmerged publication index')
                index[name.decode()] = oid.decode()
        for row in manifest['files']:
            data = (ROOT / row['path']).read_bytes()
            git_oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if row['path'] not in index:
                raise RuntimeError('Source missing from index: ' + row['path'])
            if index[row['path']] != git_oid:
                # Archived nested attributes deliberately restore .bat as CRLF.
                # Verify exact checkout bytes, not falsely demand raw Git blobs.
                checkout = subprocess.check_output(['git', 'cat-file', '--filters', ':' + row['path']], cwd=ROOT)
                if (hashlib.sha256(checkout).hexdigest(), len(checkout)) != (row['sha256'], row['bytes']):
                    raise RuntimeError('Staged checkout bytes differ: ' + row['path'])
        names = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z'], cwd=ROOT).split(b'\0')
        names = [name.decode() for name in names if name]
        for name in names:
            verify_path(name)
            with (ROOT / name).open('rb') as stream:
                check_stream(stream)
        staged_count = len(names)
    print(json.dumps({'source_files_verified': len(manifest['files']), 'runtime_members_verified': len(rows),
                      'zip_sha256': zip_sha, 'engine_sha256': ENGINE, 'pi_requests': 2,
                      'pi_decode_tok_s': [round(r['generated'] * 1000 / r['decode_ms'], 3) for r in evidence['rows']],
                      'credential_pattern_matches': 0, 'staged_files_checked': staged_count, 'model_started': False}))


if __name__ == '__main__':
    main()
