"""Read-only source archive and numeric benchmark verification; no inference."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'local-evidence/source-snapshot-manifest.json').read_text())
secret = re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{30,})\b')
for row in manifest['files']:
    path = (root / row['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError('Missing/out-of-bound archived file: ' + row['path'])
    if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise RuntimeError('Archived file checksum mismatch: ' + row['path'])
    if secret.search(path.read_bytes()):
        raise RuntimeError('Possible credential in archive: '+row['path'])
trial = json.loads((root / 'local-evidence/pi-trial.json').read_text())
rows = trial['rows']
if len(rows) != 10 or any(set(row) != {'prompt_tokens','generated','prompt_ms','decode_ms','finish','requested_max_new'} for row in rows):
    raise RuntimeError('Unexpected numeric evidence shape')
prefill = sum(row['prompt_tokens'] for row in rows)*1000/sum(row['prompt_ms'] for row in rows)
decode = sum(row['generated'] for row in rows)*1000/sum(row['decode_ms'] for row in rows)
if abs(prefill-188.609) > .001 or abs(decode-38.138) > .001:
    raise RuntimeError('Published benchmark does not match evidence')
screen = json.loads((root/'local-evidence/4k-screens.json').read_text())
screen_rates = {}
for variant in ('dual','single'):
    samples = [row for group in screen['screens'] if group['label'].startswith(variant) for row in group['rows']]
    p = [row['prompt_tokens']*1000/row['prompt_ms'] for row in samples if row['name'].startswith('prefill')]
    d = [row['generated']*1000/row['decode_ms'] for row in samples if row['name'].startswith('decode')]
    screen_rates[variant] = {'prefill':sum(p)/len(p),'decode':sum(d)/len(d)}
if abs(screen_rates['dual']['prefill']-346.079621)>.001 or abs(screen_rates['dual']['decode']-50.920092)>.001:
    raise RuntimeError('Published separate 4K screen does not match evidence')
print(json.dumps({'source_files_verified':len(manifest['files']), 'pi_requests':len(rows),
                  'prefill_tok_s':prefill,'decode_tok_s':decode,'separate_4k_rates':screen_rates,'model_started':False}))
