"""Bind an isolated build and bounded no-model transport gates to fresh evidence."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from advanced_guard import (ROOT, STOCK, shared, source_hashes, parse_remote_layer_ids,
                            native_expert_payload, PREFIX_REMOTE_LAYERS,SIX_REMOTE_LAYERS,
                            SIX_HOST_MARKER,SIX_HOST_PROOF)

ALLOWED = {'CMakeLists.txt', 'include/strata/core/mtp.hpp', 'src/core/mtp.cpp', 'src/program/generate.cpp',
           'include/strata/core/pinned.hpp', 'src/core/pinned.cu',
           'include/strata/core/expert_source.hpp', 'src/core/expert_source.cpp',
           'include/strata/core/native_head.hpp', 'src/core/native_head.cpp','src/prefill/prefill.cpp',
           'include/strata/core/verify.hpp','src/core/verify.cpp',
           'include/strata/kernels/verify_kernels.hpp','src/kernels/cuda/verify_kernels.cu'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--strategy',choices=['mtp','resident-experts'],default='mtp')
    parser.add_argument('--remote-decode-mode',choices=['original','packed','graphs'],default='original')
    parser.add_argument('--remote-layer-ids',help='Four or six sorted unique global IDs, resident-experts only')
    parser.add_argument('--remote-hybrid',action='store_true',help='Keep GPU0 cache hits, compute remote misses only')
    parser.add_argument('--remote-cache-frozen',action='store_true',help='Freeze only remote-owned hybrid cache pairs')
    parser.add_argument('--verify-wait-profile',action='store_true')
    parser.add_argument('--mtp-chain-batch',action='store_true')
    parser.add_argument('--cache-slots', type=int, choices=[3072,3500,4000,4359,4903,5500], default=3500,
                        help='Fixed equal target cache for matched single/dual screens')
    args = parser.parse_args()
    if args.remote_decode_mode!='original' and args.strategy!='resident-experts':
        raise ValueError('Optimized remote decode requires the resident-experts strategy')
    if args.remote_layer_ids is not None and args.strategy!='resident-experts':
        raise ValueError('Selected remote layers require the resident-experts strategy')
    if args.remote_hybrid and (args.strategy!='resident-experts' or args.remote_decode_mode!='original'):
        raise ValueError('Hybrid experts require resident-experts strategy and original mode')
    if args.remote_cache_frozen and not args.remote_hybrid:
        raise ValueError('Frozen remote cache requires the hybrid strategy')
    remote_ids=(parse_remote_layer_ids(args.remote_layer_ids) if args.remote_layer_ids is not None
                else PREFIX_REMOTE_LAYERS.copy()) if args.strategy=='resident-experts' else []
    remote_ids_explicit=args.remote_layer_ids is not None
    if len(remote_ids)==6 and (remote_ids!=SIX_REMOTE_LAYERS or args.remote_hybrid or args.remote_decode_mode!='original'):
        raise ValueError('Six-layer campaign requires the exact Original whole-resident candidate')
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.tag):
        raise ValueError('Invalid evidence tag')
    run = ROOT / 'gate-runs' / args.tag
    run.mkdir(parents=True, exist_ok=False)
    build = ROOT / 'build'
    binding = json.loads((build / 'build-binding.json').read_text(encoding='utf-8-sig'))
    hashes = source_hashes()
    if hashes != {r['path']: r['sha256'] for r in binding['source_files']}:
        raise RuntimeError('Source differs from the completed build binding')
    if shared.digest(ROOT/'transport_test.cu')!=binding['transport_source_sha256']:
        raise RuntimeError('Transport source differs from the completed build binding')
    if shared.digest(ROOT/'host_registration_test.cu')!=binding['registration_source_sha256']:
        raise RuntimeError('Registration source differs from the completed build binding')
    if shared.digest(ROOT/'embedding_cache_test.cu')!=binding['embedding_source_sha256']:
        raise RuntimeError('Embedding source differs from the completed build binding')
    if shared.digest(ROOT/'mtp_equivalence_test.cpp')!=binding['mtp_equivalence_source_sha256']:
        raise RuntimeError('MTP equivalence source differs from the completed build binding')
    if shared.digest(ROOT/'mtp_profile_test.cpp')!=binding['mtp_profile_source_sha256']:
        raise RuntimeError('MTP profile source differs from the completed build binding')
    if shared.digest(ROOT/'verify_wait_test.cu')!=binding['verify_wait_source_sha256']:
        raise RuntimeError('Verifier wait source differs from the completed build binding')
    for prefix in ('remote_experts','compacted_loader','remote_experts_profile'):
        if shared.digest(ROOT/f'{prefix}_test.cpp')!=binding[f'{prefix}_source_sha256']:
            raise RuntimeError(f'{prefix} source differs from the completed build binding')
    changes = subprocess.check_output(['git', '-C', str(ROOT/'source'), 'diff', '--name-only'], text=True).splitlines()
    if set(changes) - ALLOWED:
        raise RuntimeError('Changes outside the scoped remote-MTP implementation')
    for file, key in (('strata.exe', 'executable_sha256'), ('mtp_transport_test.exe', 'transport_sha256'),
                      ('host_registration_test.exe', 'registration_sha256'),('embedding_cache_test.exe','embedding_sha256'),
                      ('mtp_equivalence_test.exe','mtp_equivalence_sha256'),('mtp_profile_test.exe','mtp_profile_sha256'),
                      ('remote_experts_test.exe','remote_experts_sha256'),('compacted_loader_test.exe','compacted_loader_sha256'),
                      ('remote_experts_profile_test.exe','remote_experts_profile_sha256'),('verify_wait_test.exe','verify_wait_sha256')):
        if shared.digest(build/file) != binding[key]:
            raise RuntimeError('Compiled executable changed after build')
    reused = []
    for gpu, file in ((0, 'source-gates-gpu0-advanced01.json'), (1, 'source-gates.json')):
        path = STOCK / file
        proof = json.loads(path.read_text())
        if not proof.get('passed') or proof['source_revision'] != binding['source_base']:
            raise RuntimeError('Pristine unchanged component evidence failed/stale')
        reused.append({'gpu': gpu, 'path': str(path), 'sha256': shared.digest(path),
                       'scope': proof.get('evidence_scope')})
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES']='0,1'
    env['STRATA_FORCE_AVX2']='1'
    env['PATH']=r'E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit\bin'+os.pathsep+env['PATH']
    before=shared.gpu_counters()
    if any(row[2]>=80 for row in before):
        raise RuntimeError('Temperature gate before transport')
    started=time.monotonic()
    probe=subprocess.run([str(build/'mtp_transport_test.exe')], cwd=ROOT/'source', env=env,
                         capture_output=True, text=True, timeout=120)
    (run/'transport.log').write_text(probe.stdout+'\n'+probe.stderr,encoding='utf-8')
    if probe.returncode or 'PASS 64 staged copies' not in probe.stdout:
        raise RuntimeError('Bounded staged-transport test failed; see transport.log')
    registration=subprocess.run([str(build/'host_registration_test.exe')], cwd=ROOT/'source', env=env,
                                capture_output=True, text=True, timeout=60)
    (run/'registration.log').write_text(registration.stdout+'\n'+registration.stderr,encoding='utf-8')
    if registration.returncode or 'PASS bounded registration: 3 arenas, 6 device reads' not in registration.stdout:
        raise RuntimeError('Bounded registration/residency test failed; see registration.log')
    embedding=subprocess.run([str(build/'embedding_cache_test.exe')],cwd=ROOT/'source',env=env,
                             capture_output=True,text=True,timeout=60)
    (run/'embedding.log').write_text(embedding.stdout+'\n'+embedding.stderr,encoding='utf-8')
    if embedding.returncode or 'PASS 64 embedding graph bit checks' not in embedding.stdout:
        raise RuntimeError('Embedding mapped/resident bit checks failed; see embedding.log')
    compacted=subprocess.run([str(build/'compacted_loader_test.exe'),'--fixture',str(run/'loader-fixture.bin')],
                              cwd=ROOT/'source',env=env,capture_output=True,text=True,timeout=30)
    (run/'compacted-loader.log').write_text(compacted.stdout+'\n'+compacted.stderr,encoding='utf-8')
    if (compacted.returncode or 'PASS compacted_loader bytes=768 layers=2 guards=exact negative_cases=4' not in compacted.stdout
            or 'PASS selected_host_layout payload_bytes=80 layers=3 scatter=exact native_roles=exact aliases=exact negatives=exact' not in compacted.stdout
            or compacted.stdout.splitlines().count(SIX_HOST_MARKER)!=1):
        raise RuntimeError('Compacted source/destination loader gate failed; see compacted-loader.log')
    verify_wait=subprocess.run([str(build/'verify_wait_test.exe')],cwd=ROOT/'source',env=env,
                               capture_output=True,text=True,timeout=30)
    (run/'verify-wait.log').write_text(verify_wait.stdout+'\n'+verify_wait.stderr,encoding='utf-8')
    if verify_wait.returncode or 'PASS verify_wait devices=2 replays=4 ready=2 delayed=2 timer=globaltimer_ns outputs=exact' not in verify_wait.stdout:
        raise RuntimeError('Verifier wait/timer/graph gate failed; see verify-wait.log')
    help_probe=subprocess.run([str(build/'strata.exe'),'--help'],env=env,capture_output=True,text=True,timeout=30)
    (run/'help.log').write_text(help_probe.stdout+'\n'+help_probe.stderr,encoding='utf-8')
    if help_probe.returncode or not all(flag in help_probe.stdout+help_probe.stderr
                                       for flag in ('--mtp-device','--expert-pin-mib','--expert-cache-exact',
                                                    '--mtp-resident-embedding','--mtp-mapped-residuals','--mtp-chain-batch',
                                                    '--remote-expert-layers','--remote-expert-layer-ids',
                                                    '--remote-expert-mode','--remote-expert-hybrid',
                                                    '--remote-expert-freeze-cache','--verify-wait-profile')):
        raise RuntimeError('Compiled CLI/placement option proof failed')
    invalid_ids=['','22,28,30','22,28,30,34,35','22,28,30,34,','22,28,28,34',
                 '28,22,30,34','-1,22,30,34','22,28,30,48','22, 28,30,34','22x,28,30,34',
                 '22,24,28,30,33,34,35','22,24,28,30,33,34,35,36']
    cli_cases=[(['--remote-expert-layer-ids',value],
                '--remote-expert-layer-ids requires four or six sorted unique IDs 0..47') for value in invalid_ids]
    for flags in (['--remote-expert-layers','4','--remote-expert-layer-ids','22,28,30,34'],
                  ['--remote-expert-layer-ids','22,28,30,34','--remote-expert-layers','0'],
                  ['--remote-expert-layer-ids','22,28,30,34','--remote-expert-layer-ids','22,28,30,34'],
                  ['--remote-expert-layers','4','--remote-expert-layers','4']):
        cli_cases.append((flags,'supply one remote expert placement flag only'))
    cli_cases.append((['--remote-expert-layers','4junk'],'--remote-expert-layers must be 0 or 4'))
    for flags in (['--remote-expert-layer-ids','22,28,30,34'],['--remote-expert-layers','4']):
        cli_cases.append((['--tokens','248045','--no-ple']+flags,'remote expert layers require native resident 4K serve'))
    for flags in (['--remote-expert-hybrid'],
                  ['--remote-expert-layers','4','--remote-expert-mode','packed','--remote-expert-hybrid'],
                  ['--remote-expert-layers','4','--remote-expert-mode','graphs','--remote-expert-hybrid']):
        cli_cases.append((['--tokens','248045','--no-ple']+flags,
                          '--remote-expert-hybrid requires four remote layers and original mode'))
    cli_cases.append((['--tokens','248045','--no-ple','--remote-expert-layers','4','--remote-expert-hybrid'],
                      'remote expert layers require native resident 4K serve'))
    cli_cases.append((['--tokens','248045','--no-ple','--remote-expert-freeze-cache'],
                      '--remote-expert-freeze-cache requires --remote-expert-hybrid'))
    cli_cases.append((['--remote-expert-freeze-cache','--remote-expert-freeze-cache'],
                      'duplicate --remote-expert-freeze-cache'))
    cli_cases.append((['--tokens','248045','--no-ple','--remote-expert-layers','4',
                       '--remote-expert-hybrid','--remote-expert-freeze-cache'],
                      'remote expert layers require native resident 4K serve'))
    six_flag=['--remote-expert-layer-ids',','.join(map(str,SIX_REMOTE_LAYERS))]
    for mode in ('packed','graphs'):
        cli_cases.append((['--tokens','248045','--no-ple']+six_flag+['--remote-expert-mode',mode],
                          'optimized remote expert mode requires four remote expert layers'))
    cli_cases.append((['--tokens','248045','--no-ple']+six_flag+['--remote-expert-hybrid'],
                      '--remote-expert-hybrid requires four remote layers and original mode'))
    cli_cases.append((['--remote-expert-layers','6'],'--remote-expert-layers must be 0 or 4'))
    cli_cases.append((['--tokens','248045','--no-ple']+six_flag,'remote expert layers require native resident 4K serve'))
    cli_results=[]
    for flags,expected in cli_cases:
        invalid=subprocess.run([str(build/'strata.exe')]+flags,env=env,capture_output=True,text=True,timeout=5)
        cli_results.append({'args':flags,'exit_code':invalid.returncode,'stderr':invalid.stderr})
        (run/'cli-boundary.json').write_text(json.dumps(cli_results,indent=2))
        if invalid.returncode!=2 or expected not in invalid.stderr:
            raise RuntimeError('Native selected-placement invalid/unsupported CLI gate failed')
    (run/'cli-boundary.json').write_text(json.dumps(cli_results,indent=2))
    after=shared.gpu_counters()
    if any(row[2]>=80 for row in after):
        raise RuntimeError('Temperature gate after transport')
    base=json.loads((STOCK/'stock-config.json').read_text())
    pack=Path(base['args'][base['args'].index('--pack')+1])
    numerical_ids=remote_ids if args.strategy=='resident-experts' else PREFIX_REMOTE_LAYERS.copy()
    remote_payload=native_expert_payload(pack,numerical_ids)
    for variant, device in (('single','0'),('dual','1')):
        if args.strategy=='resident-experts':device='0'
        remote_active=args.strategy=='resident-experts' and variant=='dual'
        remote_mode=args.remote_decode_mode if remote_active else 'original'
        cfg={**base, 'args':list(base['args'])+['--mtp-device',device,'--expert-cache-exact',
                                             '--expert-pin-mib','16384','--mtp-resident-embedding','--mtp-mapped-residuals',
                                             '--remote-expert-mode',remote_mode],
             'strategy':args.strategy,
             'exe':str(build/'strata.exe'), 'cwd':str(ROOT/'source'),
             'model_name':'qwen3.8-flash-next-iq3_xxs-advanced', 'log':str(ROOT/'unused-default.log')}
        if remote_active and remote_ids_explicit:
            cfg['args']+=['--remote-expert-layer-ids',','.join(map(str,remote_ids))]
        else:
            cfg['args']+=['--remote-expert-layers','4' if remote_active else '0']
        if args.mtp_chain_batch:cfg['args'].append('--mtp-chain-batch')
        if args.verify_wait_profile:cfg['args'].append('--verify-wait-profile')
        if args.remote_hybrid and remote_active:cfg['args'].append('--remote-expert-hybrid')
        if args.remote_cache_frozen and remote_active:cfg['args'].append('--remote-expert-freeze-cache')
        idx=cfg['args'].index('--expert-profile')
        cfg['args'][idx+1]=str(ROOT/'source/data/expert-profile.bin')
        cfg['args'][cfg['args'].index('--expert-cache')+1]=str(args.cache_slots)
        (ROOT/f'{variant}-config.json').write_text(json.dumps(cfg,indent=2))
    bound=[]
    old=ROOT.parent/'qwen-strata-20260926'
    manifest=json.loads((old/'runtime-manifest.json').read_text())
    for row in manifest['runtime_files']:
        if row['path'].replace('\\','/').startswith(('pack-iq3/','mtp/rt/')):
            path=old/row['path']
            if shared.digest(path)!=row['sha256']:
                raise RuntimeError('Reused pack/MTP checksum mismatch')
            bound.append({'path':str(path),'sha256':row['sha256']})
    for path in [build/'strata.exe',build/'mtp_transport_test.exe',build/'host_registration_test.exe',
                 build/'embedding_cache_test.exe',build/'build-binding.json',
                 build/'mtp_equivalence_test.exe',ROOT/'mtp_equivalence_test.cpp',ROOT/'run_mtp_equivalence.py',
                 build/'mtp_profile_test.exe',ROOT/'mtp_profile_test.cpp',
                 build/'remote_experts_test.exe',ROOT/'remote_experts_test.cpp',
                 build/'remote_experts_profile_test.exe',ROOT/'remote_experts_profile_test.cpp',
                 build/'compacted_loader_test.exe',ROOT/'compacted_loader_test.cpp',
                 build/'verify_wait_test.exe',ROOT/'verify_wait_test.cu',
                 ROOT/'transport_test.cu',ROOT/'host_registration_test.cu',ROOT/'embedding_cache_test.cu',
                 ROOT/'advanced_guard.py',ROOT/'native_probe.py',
                 ROOT/'owned_memory.py',
                 ROOT/'prepare_evidence.py',ROOT/'build.ps1',
                 ROOT/'single-config.json',ROOT/'dual-config.json',run/'verify-wait.log',run/'cli-boundary.json',
                 run/'compacted-loader.log',
                 STOCK/'stock_guard.py',STOCK/'native_probe.py']:
        bound.append({'path':str(path),'sha256':shared.digest(path)})
    for row in reused:
        bound.append({'path':row['path'],'sha256':row['sha256']})
    proof={'passed':True,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
           'source_hashes':hashes,'source_revision':binding['source_base'],
           'executable_sha256':binding['executable_sha256'],'bound_files':bound,
           'transport':{'cases':64,'wall_seconds':time.monotonic()-started,'before':before,'after':after,
                        'log':str(run/'transport.log')},'reused_components':reused,
           'registration':{'arenas':3,'device_reads':6,'arena_bytes':64*1024**2,'registration_cap_bytes':16*1024**2,
                           'log':str(run/'registration.log')},
           'embedding':{'cases':64,'type':21,'n_embd':2560,'synthetic_vocab':128,'log':str(run/'embedding.log')},
           'evidence_scope':'CUDA13.3 source build; unchanged primitive suites reused on both GPUs; same-helper staged-copy/graph bit checks T1..8, bounded registration/locked-suffix device-bit checks and native IQ3 mapped/resident embedding graph bit checks on both GPUs. No GGUF load, full MTP numerical equality or full-model correctness claim.'}
    proof['matched_target_cache_slots']=args.cache_slots
    proof['matched_expert_pin_mib']=16384
    proof['matched_mtp_resident_embedding']=True
    proof['matched_mtp_mapped_residuals']=True
    proof['matched_mtp_chain_batch']=args.mtp_chain_batch
    proof['strategy']=args.strategy
    proof['remote_decode_mode_for_dual']=args.remote_decode_mode
    proof['remote_hybrid_for_dual']=args.remote_hybrid
    proof['remote_cache_frozen_for_dual']=args.remote_cache_frozen
    proof['remote_expert_layer_ids_for_dual']=numerical_ids
    proof['remote_expert_layer_ids_explicit_for_dual']=remote_ids_explicit
    proof['remote_expert_payload_bytes_for_dual']=remote_payload
    proof['matched_verify_wait_profile']=args.verify_wait_profile
    proof['verify_wait']={'devices':2,'profiled_replays_per_device':4,'ready_per_device':2,
                          'delayed_per_device':2,'timer_unit':'globaltimer_ns','outputs':'exact',
                          'log':str(run/'verify-wait.log')}
    proof['compacted_loader']={'payload_bytes':768,'layers':2,'negative_cases':4,
                               'selected6_host_layout':SIX_HOST_PROOF,
                               'selected_host_layout':{'payload_bytes':80,'layers':3,'scatter':'exact',
                                                       'native_roles':'exact','aliases':'exact','negatives':'exact'},
                               'log':str(run/'compacted-loader.log')}
    proof['remote_expert_equivalence_required_for_full_trial']=True
    proof['remote_cli_boundary_cases']=len(cli_results)
    proof['mtp_equivalence_required_for_full_trial']=True
    (run/'gates.json').write_text(json.dumps(proof,indent=2))
    (ROOT/'source-gates.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({'passed':True,'gates':str(run/'gates.json'),'executable_sha256':binding['executable_sha256']}))


if __name__=='__main__':
    main()
