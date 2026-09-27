"""Operational admission only from current, completed and resource-safe measured evidence."""
import argparse
import json
from pathlib import Path
from guard import ROOT,digest,identity,resources_ok
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--context',type=int,choices=[4096,32768],required=True);a=p.parse_args()
    manifest=json.loads((ROOT/'runtime-manifest.json').read_text());identity(manifest)
    run=ROOT/'runs'/a.tag;result=json.loads((run/'result.json').read_text());probe=json.loads((run/'probe.json').read_text())
    mh=digest(ROOT/'runtime-manifest.json')
    assert result.get('load_passed') and result['context']==a.context and result['runtime_manifest_sha256']==mh
    assert result['strata_sha256']==manifest['strata_sha256'] and probe.get('passed') is True
    assert not (run/'failure.json').exists()
    names={x['name'] for x in probe['rows']}
    assert {'math','tool_call','tool_followup',*(f'decode_screen_{i}' for i in range(3)),*(f'prefill_screen_{i}' for i in range(2))}<=names
    if a.context==32768:
        assert 'context32k' in names
        pi=json.loads((run/'pi-probe.json').read_text());assert pi['passed'] and pi['native_xhigh'] and pi['tools']==['bash','read','write']
    telemetry=[json.loads(line) for line in (run/'telemetry.jsonl').read_text().splitlines()]
    assert len(telemetry)>=3
    for t in telemetry:resources_ok(t,t['gpu']);assert t['peak_job_bytes']<=48*(1<<30)
    admission={'passed':True,'kind':'experimental-operational-not-quality-equivalence','context':a.context,
        'strata_sha256':manifest['strata_sha256'],'runtime_manifest_sha256':mh,'evidence':str(run.relative_to(ROOT)),
        'min_ram_free_gib':min(t['ram_free'] for t in telemetry)/(1<<30),
        'min_commit_free_gib':min(t['commit_free'] for t in telemetry)/(1<<30),
        'peak_job_gib':max(t['peak_job_bytes'] for t in telemetry)/(1<<30),
        'probe_sha256':digest(run/'probe.json'),'telemetry_sha256':digest(run/'telemetry.jsonl')}
    path=ROOT/f'admission-{a.context}.json'
    if path.exists():raise RuntimeError('Admission exists: do not overwrite measured evidence')
    path.write_text(json.dumps(admission,indent=2));print(json.dumps(admission,indent=2))
if __name__=='__main__':main()
