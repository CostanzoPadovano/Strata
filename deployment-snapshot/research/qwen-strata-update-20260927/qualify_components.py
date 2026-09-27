"""Fresh bounded component qualification; never starts strata --serve or the full model."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import native_guard as native

ROOT = Path(__file__).resolve().parent
def bound(path): return {'path':str(path.resolve()),'sha256':native.guard.digest(path)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--include-numerical',action='store_true')
    opts=parser.parse_args()
    if not opts.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in opts.tag):
        raise ValueError('Invalid finite evidence tag')
    binding,binding_path=native.load_binding()
    report={'passed':False,'scope':binding['scope'],'full_model_started':False,
            'engine_sha256':binding['executable_sha256'],'build_binding_sha256':native.guard.digest(binding_path),
            'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'gates':{},'numerical_gates':{}}
    root=ROOT/'gate-runs'/opts.tag
    cases=[('cache','conversation_cache_test.exe',[], 'conversation_cache_test OK (synthetic; not full-model equivalence)'),
           ('pool_idle','pool_idle_test.exe',[], 'pool_idle_test OK (CPU component test; not model throughput)'),
           ('vision_protocol','vision_protocol_test.exe',[], 'vision protocol synthetic gate passed'),
           ('vision_mrope','vision_mrope_test.exe',[], 'vision M-RoPE synthetic gate passed'),
           ('transport','mtp_transport_test.exe',[], 'PASS 64 staged copies'),
           ('registration','host_registration_test.exe',[], 'PASS bounded registration'),
           ('embedding','embedding_cache_test.exe',[], 'PASS 64 embedding graph bit checks'),
           ('compacted_loader','compacted_loader_test.exe',['--fixture',str(root/'loader-fixture.bin')],'PASS compacted_loader'),
           ('verify_wait','verify_wait_test.exe',[], 'PASS verify_wait')]
    files=[ROOT/'build.ps1',ROOT/'native_guard.py',Path(__file__),binding_path]
    for name,exe,args,marker in cases:
        result=native.run_native([str(ROOT/'build'/exe),*args],root/name,marker,90,8,exact_marker=False)
        report['gates'][name]=result
        files+=[ROOT/'build'/exe]
        files += [Path(row['path']) for row in result['bound_evidence']]
        print(json.dumps({'component':name,'passed':True}),flush=True)
    if opts.include_numerical:
        cfg=json.loads((native.guard.ROOT/'configs/tier4096.json').read_text())
        native.guard.validate_config(cfg,4096)
        def arg(flag): return cfg['args'][cfg['args'].index(flag)+1]
        common=['--pack',arg('--pack'),'--native',arg('--native')]
        for name,exe,args,marker in [('mtp','mtp_equivalence_test.exe',common+['--mtp',arg('--mtp')],native.MTP_MARKER),
                                    ('remote12','remote_experts_test.exe',common+['--layer-ids',native.IDS],native.REMOTE_MARKER)]:
            result=native.run_native([str(ROOT/'build'/exe),*args],root/name,marker,1200,24)
            report['numerical_gates'][name]=result
            files += [ROOT/'build'/exe] + [Path(row['path']) for row in result['bound_evidence']]
            print(json.dumps({'component':name,'passed':True}),flush=True)
    # Detect changes during execution; no older proof is relabeled as this candidate.
    native.load_binding()
    report['passed']=True
    report['bound_files']=[bound(path) for path in dict.fromkeys(files)]
    (ROOT/'native-gates.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'fresh_native_components_passed':True,'full_model_started':False,
                      'numerical_complete':opts.include_numerical}),flush=True)

if __name__=='__main__': main()
