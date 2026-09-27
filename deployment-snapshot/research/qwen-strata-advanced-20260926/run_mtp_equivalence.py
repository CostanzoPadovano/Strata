"""Bounded MTP/expert parity and diagnostics (no main host expert arena or generation)."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from advanced_guard import (ROOT, shared, source_hashes, MTP_FIXTURES, MTP_MODES, PARITY_SCOPE,
                            REMOTE_SCOPE, REMOTE_MARKER, DECODE_SCOPE, DECODE_MARKER,enable_owned_memory_telemetry,
                            remote_placement, MASKED_MARKER, HYBRID_MARKER, validate_hybrid_log,
                            SIX_REMOTE_LAYERS, REMOTE_SIX_SCOPE, REMOTE_SIX_MARKER,
                            SIX_EXPERT_PROOF, validate_six_expert_log)


def test_spec(purpose):
    """Executable prefix, exact marker, prerequisites and intentionally narrow scope."""
    specs={
        'equivalence':('mtp_equivalence',f'PASS mtp_equivalence fixtures={MTP_FIXTURES} modes={MTP_MODES} probability_bits=exact',
                       False,False,PARITY_SCOPE),
        'profile':('mtp_profile','PASS mtp_profile modes=4 iterations=64 chain=3',True,False,
                   'Isolated one-layer synthetic MTP timing; no full main experts/PLE/generation'),
        'remote-experts':('remote_experts',REMOTE_MARKER,True,False,REMOTE_SCOPE),
        'expert-profile':('remote_experts_profile',DECODE_MARKER,True,True,DECODE_SCOPE),
    }
    if purpose not in specs:
        raise ValueError('Unknown isolated diagnostic purpose')
    return specs[purpose]


def validate_decode_profile(text,expected_layer_ids,expected_payload,expected_marker):
    rows=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
    expected={(device,mode,profiling,gap) for device in (0,1) for mode in ('original','packed','graphs')
              for profiling in (False,True) for gap in (0,40)}
    keys=[(row['device'],row['mode'],row['profiling'],row['gap_ms']) for row in rows]
    if len(rows)!=24 or len(set(keys))!=24 or set(keys)!=expected or expected_marker not in text.splitlines():
        raise RuntimeError('Decode diagnostic must cover all 24 distinct conditions')
    if len({row['digest'] for row in rows})!=1:
        raise RuntimeError('Decode diagnostic output digests differ')
    for row in rows:
        if (row['bursts']!=64 or row['calls']!=256 or row['reference_bytes']!=65536000
                or row.get('layer_ids')!=','.join(map(str,expected_layer_ids))
                or row.get('payload_bytes')!=expected_payload
                or row['input_bytes']!=40960 or row['output_bytes']!=409600
                or row['graph_replays']!=(256 if row['mode']=='graphs' else 0)
                or row['profile_calls']!=(256 if row['profiling'] else 0)
                or row['timing_scope']!='decode_calls_only_checks_and_pauses_excluded'
                or not math.isfinite(row['active_decode_wall_ms']) or row['active_decode_wall_ms']<=0):
            raise RuntimeError('Decode diagnostic counts, extents or active timing are invalid')
    return rows


def diagnostic_placement(config,gates,purpose):
    """Separate full-runtime placement from the standalone numerical fixture."""
    runtime_ids,runtime_selected=remote_placement(config['args'])
    bound_ids=gates.get('remote_expert_layer_ids_for_dual',[])
    bound_payload=gates.get('remote_expert_payload_bytes_for_dual',0)
    expected_runtime=bound_ids if config.get('strategy')=='resident-experts' else []
    if runtime_ids!=expected_runtime:
        raise RuntimeError('Diagnostic config remote layer IDs differ from bound placement')
    if purpose in ('remote-experts','expert-profile'):
        if len(bound_ids)==6 and purpose=='expert-profile':
            raise ValueError('Six-layer candidate is Original only; four-layer optimized profile is unsupported')
        selected=gates.get('remote_expert_layer_ids_explicit_for_dual',False)
        if (selected and config.get('strategy')!='resident-experts') or (not selected and bound_ids!=[0,1,2,3]):
            raise RuntimeError('Bound numerical expert fixture placement is inconsistent')
        return bound_ids,bound_payload,selected
    return runtime_ids,0,runtime_selected


def make_mtp_job():
    """Use a stricter test-only cap without widening the stock cap whitelist."""
    k32,C,W=shared.k32,shared.C,shared.W
    k32.CreateJobObjectW.argtypes=[C.c_void_p,W.LPCWSTR]
    k32.CreateJobObjectW.restype=W.HANDLE
    k32.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD]
    k32.QueryInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD,C.POINTER(W.DWORD)]
    k32.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE]
    k32.IsProcessInJob.argtypes=[W.HANDLE,W.HANDLE,C.POINTER(W.BOOL)]
    k32.CloseHandle.argtypes=[W.HANDLE]
    job=k32.CreateJobObjectW(None,None)
    shared.wincheck(job)
    limits=shared.Limits()
    limits.basic.flags=0x2000|0x200
    limits.job_limit=8*shared.GiB
    try:
        shared.wincheck(k32.SetInformationJobObject(job,9,C.byref(limits),C.sizeof(limits)))
        actual=shared.Limits()
        shared.wincheck(k32.QueryInformationJobObject(job,9,C.byref(actual),C.sizeof(actual),None))
        if actual.job_limit!=limits.job_limit or actual.basic.flags & limits.basic.flags!=limits.basic.flags:
            raise RuntimeError('Test-only 8 GiB kill-on-close Job limits not verified')
    except BaseException:
        k32.CloseHandle(job)
        raise
    return job


def child(run):
    run=Path(run).resolve()
    if not run.is_relative_to(ROOT/'parity-runs'):
        raise ValueError('Child run outside campaign')
    deadline=time.monotonic()+30
    while not (run/'job-attached').exists():
        if time.monotonic()>deadline:
            raise RuntimeError('Parent did not attach the Job')
        time.sleep(0.05)
    config=json.loads((run/'command.json').read_text())
    with (run/'native.log').open('x',encoding='utf-8') as log:
        probe=subprocess.run(config['command'],cwd=ROOT/'source',stdout=log,stderr=subprocess.STDOUT,
                             timeout=160,creationflags=subprocess.CREATE_NO_WINDOW)
    text=(run/'native.log').read_text(errors='replace')
    marker=config['expected_marker']
    passed=(probe.returncode==0 and marker in text.splitlines()
            and all(item in text.splitlines() for item in config.get('required_markers',[])))
    (run/'probe.json').write_text(json.dumps({'passed':passed,'native_exit_code':probe.returncode}))
    return 0 if passed else 1


def parent(tag,purpose='equivalence',profile_gap_ms=0,profile_chain_batch=False):
    prefix,_,require_mtp,require_remote,scope=test_spec(purpose)
    enable_owned_memory_telemetry()
    if profile_gap_ms not in (0,40) or (purpose!='profile' and (profile_gap_ms or profile_chain_batch)):
        raise ValueError('Inter-round gaps are restricted to the isolated profile')
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in tag):
        raise ValueError('Invalid run tag')
    run=ROOT/'parity-runs'/tag
    run.mkdir(parents=True,exist_ok=False)
    binding=json.loads((ROOT/'build/build-binding.json').read_text(encoding='utf-8-sig'))
    if source_hashes()!={r['path']:r['sha256'] for r in binding['source_files']}:
        raise RuntimeError('Current source differs from completed build')
    exe=ROOT/f'build/{prefix}_test.exe'
    if shared.digest(exe)!=binding[f'{prefix}_sha256']:
        raise RuntimeError('MTP test executable changed after build')
    if shared.digest(ROOT/f'{prefix}_test.cpp')!=binding[f'{prefix}_source_sha256']:
        raise RuntimeError('MTP test source changed after build')
    # Bind current synthetic gates/config/model/checkpoints before loading the isolated layer.
    shared.ROOT=ROOT
    shared.EXE_HASH=shared.digest(ROOT/'build/strata.exe')
    from advanced_guard import artifact_checks
    report,gates=artifact_checks('experimental-no-vram-floor',require_mtp_equivalence=require_mtp,
                                require_remote_experts=require_remote,require_remote_decode_equivalence=False)
    limits=shared.policy('experimental-no-vram-floor',3072)
    limits['job_cap_gib']=8
    config=json.loads((ROOT/'dual-config.json').read_text())
    args=config['args']
    layer_ids,expected_payload,selected=diagnostic_placement(config,gates,purpose)
    command=[str(exe),'--pack',args[args.index('--pack')+1],'--native',args[args.index('--native')+1]]
    if purpose in ('equivalence','profile'):
        command+=['--mtp',args[args.index('--mtp')+1]]
    if purpose=='profile':command+=['--gap-ms',str(profile_gap_ms)]
    if purpose=='profile' and profile_chain_batch:command+=['--chain-batch']
    if purpose in ('remote-experts','expert-profile') and selected:
        command+=['--layer-ids',','.join(map(str,layer_ids))]
    if purpose=='remote-experts' and selected:
        if len(layer_ids)==6:
            marker,scope=REMOTE_SIX_MARKER,REMOTE_SIX_SCOPE
        else:
            marker=(f'PASS remote_experts_selected layers=4 ids={",".join(map(str,layer_ids))} '
                    f'payload_bytes={expected_payload} owns=exact decode_calls=6 prefill_calls=5 '
                    'blob_bytes=exact quant_bytes=exact outputs=exact')
    elif purpose=='expert-profile' and selected:
        marker=(f'PASS remote_experts_profile bursts=64 layers=4 modes=3 conditions=24 '
                f'ids={",".join(map(str,layer_ids))} outputs=exact')
    else:
        marker=test_spec(purpose)[1]
    (run/'command.json').write_text(json.dumps({'command':command,'purpose':purpose,'expected_marker':marker,
                                                'required_markers':[MASKED_MARKER,HYBRID_MARKER]
                                                  if purpose=='remote-experts' and gates.get('remote_hybrid_for_dual',False) else [],
                                                'layer_ids':layer_ids,'payload_bytes':expected_payload},indent=2))
    evidence={'passed':False,'purpose':purpose,'scope':scope,
              'profile_gap_ms':profile_gap_ms if purpose=='profile' else None,
              'profile_chain_batch':profile_chain_batch if purpose=='profile' else None,
              'limits':limits,'executable_sha256':shared.digest(exe),
              'engine_executable_sha256':binding['executable_sha256'],
              'gates_sha256':shared.digest(ROOT/'source-gates.json')}
    evidence['layer_ids']=layer_ids
    evidence['payload_bytes']=expected_payload
    mutex=job=process=None
    owned=False
    k32,C,W=shared.k32,shared.C,shared.W
    k32.CreateMutexW.argtypes=[C.c_void_p,W.BOOL,W.LPCWSTR]
    k32.CreateMutexW.restype=W.HANDLE
    k32.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD]
    k32.ReleaseMutex.argtypes=[W.HANDLE]
    try:
        mutex=k32.CreateMutexW(None,False,'Local\\QwenBioagentGpuExclusive')
        shared.wincheck(mutex)
        if k32.WaitForSingleObject(mutex,0) not in (0,0x80):
            raise RuntimeError('Another owned model/test uses the GPUs')
        owned=True
        shared.check_resources(shared.counters(),shared.gpu_counters(),limits)
        job=make_mtp_job()
        env=dict(os.environ)
        for name in list(env):
            if name.startswith('STRATA_'):env.pop(name)
        env['CUDA_VISIBLE_DEVICES']='0,1'
        env['STRATA_FORCE_AVX2']='1'
        env['PATH']=os.pathsep.join(config['lib_dirs'])+os.pathsep+env['PATH']
        with (run/'observer.log').open('x',encoding='utf-8') as log,(run/'telemetry.jsonl').open('x') as telemetry:
            process=subprocess.Popen([sys.executable,__file__,'--child',str(run)],cwd=ROOT,env=env,
                                     stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            shared.wincheck(k32.AssignProcessToJobObject(job,int(process._handle)))
            member=W.BOOL()
            shared.wincheck(k32.IsProcessInJob(int(process._handle),job,C.byref(member)))
            if not member.value:raise RuntimeError('Job membership not verified')
            (run/'job-attached').write_text('owned\n')
            started=time.monotonic()
            while process.poll() is None:
                memory,gpu=shared.counters(),shared.gpu_counters()
                telemetry.write(json.dumps({'elapsed':time.monotonic()-started,**memory,'gpu':gpu,
                                            **shared.job_counters(job)})+'\n')
                telemetry.flush()
                shared.check_resources(memory,gpu,limits)
                if time.monotonic()-started>180:raise RuntimeError('Bounded MTP test timeout')
                time.sleep(0.5)
            if process.returncode:raise RuntimeError('Native diagnostic observer failed; see native/observer log')
            probe=json.loads((run/'probe.json').read_text())
            if not probe['passed'] or probe['native_exit_code']!=0:
                raise RuntimeError('MTP parity or clean exit failed')
            if purpose=='expert-profile':
                validate_decode_profile((run/'native.log').read_text(),layer_ids,expected_payload,marker)
            if purpose=='remote-experts' and len(layer_ids)==6:
                validate_six_expert_log((run/'native.log').read_text())
                evidence['six_layer_coverage']=SIX_EXPERT_PROOF
            if purpose=='remote-experts' and gates.get('remote_hybrid_for_dual',False):
                validate_hybrid_log((run/'native.log').read_text())
                evidence['masked_decode']={'devices':2,'calls_per_device':4,'entries_per_device':32,
                                          'all_hit_calls_per_device':1,'outputs':'exact','negatives':'exact'}
                evidence['hybrid_dispatch']={'devices':2,'outputs':'exact','cold_restore':'exact','failures':'exact'}
            evidence['passed']=True
            print(json.dumps({'passed':True,'run':str(run),'scope':evidence['scope']}))
    except BaseException as error:
        evidence['failure']=str(error)
        raise
    finally:
        if job:k32.CloseHandle(job)
        if process and process.poll() is None:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        if owned:k32.ReleaseMutex(mutex)
        if mutex:k32.CloseHandle(mutex)
        (run/'result.json').write_text(json.dumps(evidence,indent=2))
    if evidence['passed'] and purpose=='equivalence':
        admission={**evidence,'fixtures':MTP_FIXTURES,'modes':MTP_MODES,'probability_bits':'exact',
                   'bound_evidence':[{'path':str(path),'sha256':shared.digest(path)}
                                     for path in (run/'result.json',run/'native.log',run/'probe.json',run/'command.json')]}
        (ROOT/'mtp-parity-admission.json').write_text(json.dumps(admission,indent=2))
    if evidence['passed'] and purpose=='remote-experts':
        admission={**evidence,'layers':len(layer_ids),'decode_calls':6,'prefill_calls':7 if len(layer_ids)==6 else 5,
                   'quant_bytes':'exact','outputs':'exact','blob_bytes':'exact',
                   'bound_evidence':[{'path':str(path),'sha256':shared.digest(path)}
                                    for path in (run/'result.json',run/'native.log',run/'probe.json',run/'command.json')]}
        (ROOT/'remote-experts-admission.json').write_text(json.dumps(admission,indent=2))
    if evidence['passed'] and purpose=='expert-profile':
        admission={**evidence,'layers':4,'bursts':64,'modes':3,'devices':2,'conditions':24,
                   'outputs':'exact','reference_bytes':65536000,'k4_fallback_exact':True,
                   'graph_replays_per_graph_condition':256,
                   'bound_evidence':[{'path':str(path),'sha256':shared.digest(path)}
                                    for path in (run/'result.json',run/'native.log',run/'probe.json',run/'command.json')]}
        (ROOT/'remote-decode-admission.json').write_text(json.dumps(admission,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag')
    parser.add_argument('--purpose',choices=['equivalence','profile','remote-experts','expert-profile'],default='equivalence')
    parser.add_argument('--profile-gap-ms',type=int,choices=[0,40],default=0)
    parser.add_argument('--profile-chain-batch',action='store_true')
    parser.add_argument('--child',type=Path)
    opts=parser.parse_args()
    if opts.child:sys.exit(child(opts.child))
    if not opts.tag:parser.error('--tag required')
    parent(opts.tag,opts.purpose,opts.profile_gap_ms,opts.profile_chain_batch)
