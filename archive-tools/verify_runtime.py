"""Verify the release ZIP without extraction or executable/model execution."""
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'local-evidence/runtime-assets-manifest.json').read_text())
zip_path = Path(sys.argv[1]).resolve()
with zipfile.ZipFile(zip_path) as archive:
    entries = {name.replace('\\','/'):name for name in archive.namelist()}
    for row in manifest['files']:
        name = row['path']
        if name not in entries or '..' in name.split('/') or name.startswith('/'):
            raise RuntimeError('Missing/invalid runtime member: '+name)
        digest = hashlib.sha256()
        size = 0
        with archive.open(entries[name]) as stream:
            for block in iter(lambda:stream.read(1<<20),b''):
                size += len(block)
                digest.update(block)
        if size != row['bytes'] or digest.hexdigest() != row['sha256']:
            raise RuntimeError('Runtime checksum mismatch: '+name)
    expected = {row['path'] for row in manifest['files']} | {'runtime-assets-manifest.json'}
    extra = [name for name in entries if not name.endswith('/') and name not in expected]
    if extra:
        raise RuntimeError('Unexpected runtime ZIP members: '+str(extra))
    secret = re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{30,})\b')
    for name, original in entries.items():
        if name.endswith(('.json','.log','.py','.mjs','.sh','.ps1','.bat','.md','.txt')):
            if secret.search(archive.read(original)):
                raise RuntimeError('Possible credential in ZIP: '+name)
print(json.dumps({'runtime_files_verified':len(manifest['files']),
                  'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                  'unexpected_members':0,'credential_pattern_matches':0,'model_started':False}))
