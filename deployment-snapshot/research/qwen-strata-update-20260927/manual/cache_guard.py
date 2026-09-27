"""Explicit bounded cache qualification, separate from the ordinary server launcher."""
import argparse
import json
from pathlib import Path
from vision_campaign import ROOT,MANUAL,admitted,artifact_checks,bound,digest,validate_config
from vision_user_launcher import current_smoke

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',type=int,choices=(4096,98304),required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--timeout',type=int,default=600)
    args=parser.parse_args()
    if not 60<=args.timeout<=900: raise ValueError('Bounded cache timeout60..900')
    if not current_smoke(4096): raise RuntimeError('Fresh4K vision/text smoke required before cache probe')
    config=ROOT/f'vision{args.context}.json'
    validate_config(json.loads(config.read_text()),args.context)
    artifact_checks(admitted.PROFILE)
    shared=admitted.shared
    shared.ROOT=ROOT; shared.EXE_HASH=digest(ROOT/'build/strata.exe')
    shared.policy=admitted.policy; shared.artifact_checks=artifact_checks
    shared.job_counters=admitted.measured_job_counters
    ns=argparse.Namespace(mode='bench',profile=admitted.PROFILE,gpu=0,pool_workers=8,
                         vram_reserve_mib=None,tag=args.tag,timeout=args.timeout)
    shared.run_trial(ns,config_file=config,probe_file=MANUAL/'cache_probe.py',device_visibility='0,1')
    run=ROOT/'runs'/args.tag
    proof={'passed':True,'context':args.context,'engine_sha256':digest(ROOT/'build/strata.exe'),
           'source_gates_sha256':digest(ROOT/'source-gates.json'),'config_sha256':digest(config),
           'scope':'finite cache/fresh greedy branches, EOS/length/cancel; not universal numeric/quality proof',
           'bound_files':[bound(run/name) for name in ('result.json','probe.json','timings.jsonl','telemetry.jsonl')]
                         +[bound(Path(__file__)),bound(MANUAL/'cache_probe.py'),bound(MANUAL/'cache_protocol.py')]}
    (ROOT/f'cache-admission-{args.context}.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({'cache_admission_passed':True,'context':args.context}),flush=True)

if __name__=='__main__': main()
