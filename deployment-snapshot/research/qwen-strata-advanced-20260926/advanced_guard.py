"""Isolated advanced trial; same verified Job/READY/resource machinery as stock.

The historical 'official_exe_sha256' evidence key is a shared-schema legacy
name; executable_kind explicitly labels this binary as a source-built candidate.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MTP_FIXTURES=36
MTP_MODES=9
PARITY_SCOPE='Real MTP-layer parity, synthetic residuals; no full main experts/PLE/generation'
REMOTE_SCOPE='Resident native expert kernel parity, bounded four-layer GPU owners; no main host arena/PLE/generation'
REMOTE_MARKER='PASS remote_experts layers=4 decode_calls=6 prefill_calls=5 blob_bytes=exact quant_bytes=exact outputs=exact'
MASKED_MARKER='PASS remote_experts_masked calls=4 entries=32 all_hit_calls=1 all_zero_bypass=exact mixed_rows=exact changing_masks=exact negatives=exact'
HYBRID_MARKER='PASS hybrid_expert_dispatch devices=2 outputs=exact cold_restore=exact failures=exact'
DECODE_SCOPE='Resident expert decode mode parity and burst timing; no main host expert arena/PLE/generation'
DECODE_MARKER='PASS remote_experts_profile bursts=64 layers=4 modes=3 outputs=exact'
PREFIX_REMOTE_LAYERS=[0,1,2,3]
SIX_REMOTE_LAYERS=[22,24,28,30,33,34]
SIX_REMOTE_PAYLOAD=6448742400
SIX_REMOTE_ALLOCATION=6949138728
REMOTE_SIX_SCOPE='Resident native expert kernel parity, bounded six-layer GPU owners; no main host arena/PLE/generation'
REMOTE_SIX_MARKER=('PASS remote_experts_selected6 layers=6 ids=22,24,28,30,33,34 '
                   'payload_bytes=6448742400 allocation_bytes=6949138728 blob_cases=12 '
                   'decode_calls=6 prefill_calls=7 blob_bytes=exact quant_bytes=exact outputs=exact '
                   'layer_coverage=all mode=original')
SIX_HOST_MARKER=('PASS selected_host_layout6 ids=1,3,5,7,9,11 holes=6 retained=6 '
                 'scatter=exact native_roles=exact aliases=exact negatives=exact')
SIX_HOST_PROOF={'logical_layers':12,'ids':[1,3,5,7,9,11],'holes':6,'retained':6,
                'devices':2,'scatter':'exact','native_roles':'exact','aliases':'exact','negatives':'exact'}
SIX_EXPERT_PROOF={'devices':2,'layer_ids':SIX_REMOTE_LAYERS,'payload_bytes':SIX_REMOTE_PAYLOAD,
                  'allocation_bytes':SIX_REMOTE_ALLOCATION,'blob_cases_per_device':12,
                  'decode_calls_per_device':6,'prefill_calls_per_device':7,
                  'all_layer_coverage':'exact','quant_bytes':'exact','outputs':'exact','mode':'original',
                  'max_prefill_t':2048,'max_prefill_k':10}
STOCK = ROOT.parent / 'qwen-strata-stock-20260926'
sys.path.insert(0, str(STOCK))
import stock_guard as shared
from owned_memory import current_owned_memory

_original_job_counters=shared.job_counters


def parse_remote_layer_ids(value):
    """Strict four/six-ID placement; unsupported intermediate/larger counts fail closed."""
    parts=value.split(',') if isinstance(value,str) else None
    if (parts is None or len(parts) not in (4,6)
            or any(not part or any(ch<'0' or ch>'9' for ch in part) for part in parts)):
        raise ValueError('Remote layer IDs must be exactly four or six comma-separated integers')
    ids=[int(part) for part in parts]
    if ids!=sorted(set(ids)) or any(layer<0 or layer>47 for layer in ids):
        raise ValueError('Remote layer IDs must be sorted, unique and in [0,47]')
    return ids


def native_expert_payload(pack,layer_ids):
    """Derive selected resident bytes from the bound native_experts.txt metadata."""
    rows={}
    for line in (Path(pack)/'native_experts.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        fields=line.split()
        if len(fields)<5:raise RuntimeError('Malformed native expert metadata')
        layer,blob=int(fields[0]),int(fields[4])
        if layer in rows:raise RuntimeError('Duplicate native expert metadata layer')
        rows[layer]=blob
    if any(layer not in rows for layer in layer_ids):
        raise RuntimeError('Selected layer absent from native expert metadata')
    return sum(rows[layer]*512 for layer in layer_ids)


def remote_placement(args):
    """Return (global IDs, explicit-selected flag), rejecting ambiguous placement."""
    legacy=[i for i,arg in enumerate(args) if arg=='--remote-expert-layers']
    selected=[i for i,arg in enumerate(args) if arg=='--remote-expert-layer-ids']
    if len(legacy)+len(selected)!=1 or (legacy and selected):
        raise ValueError('Remote expert placement must use exactly one placement flag')
    if selected:
        i=selected[0]
        if i+1>=len(args):raise ValueError('Remote expert layer IDs are missing')
        return parse_remote_layer_ids(args[i+1]),True
    i=legacy[0]
    if i+1>=len(args) or args[i+1] not in ('0','4'):
        raise ValueError('Legacy remote expert layer count must be 0 or 4')
    return (PREFIX_REMOTE_LAYERS.copy() if args[i+1]=='4' else []),False


def hybrid_placement(args,expected):
    """Reject repeated or unbound hybrid opt-ins before full native startup."""
    count=args.count('--remote-expert-hybrid')
    if type(expected) is not bool or count not in (0,1) or bool(count)!=expected:
        raise ValueError('Hybrid expert placement differs from the bound variant')
    return bool(count)


def validate_hybrid_log(text):
    lines=text.splitlines()
    if lines.count(MASKED_MARKER)!=1 or lines.count(HYBRID_MARKER)!=1:
        raise RuntimeError('Masked dispatch, cold restore and failure gates must all pass exactly once')


def validate_six_expert_log(text):
    lines=text.splitlines()
    if lines.count(REMOTE_SIX_MARKER)!=1 or any(line in lines for line in (MASKED_MARKER,HYBRID_MARKER)):
        raise RuntimeError('Six-layer all-format coverage must pass once without hybrid/optimized scope')


def frozen_remote_cache_placement(args,expected):
    count=args.count('--remote-expert-freeze-cache')
    if type(expected) is not bool or count not in (0,1) or bool(count)!=expected:
        raise ValueError('Frozen remote cache policy differs from the bound variant')
    if count and args.count('--remote-expert-hybrid')!=1:
        raise ValueError('Frozen remote cache requires exactly one hybrid opt-in')
    return bool(count)


def validate_hybrid_cache_log(text,requests,frozen):
    """Completed-request round-trip counters; wall time is not unoverlapped decode cost."""
    if type(requests) is not int or requests<=0 or type(frozen) is not bool:
        raise ValueError('Invalid hybrid cache validation scope')
    stats=[line for line in text.splitlines() if line.startswith('strata hybrid cache request:')]
    hits=[line for line in text.splitlines() if line.startswith('strata hybrid experts request:')]
    if len(stats)!=requests or len(hits)!=requests:
        raise RuntimeError('Hybrid cache counters missing, repeated or outside completed request scope')
    pattern=(r'strata hybrid cache request: frozen ([01]); adaptive_reads (\d+) bytes (\d+) '
             r'wall_ms ([\d.]+); restore_reads (\d+) bytes (\d+) wall_ms ([\d.]+)')
    hit_pattern=(r'strata hybrid experts request: delta (\d+) layers, GPU0 (\d+) hit entries, '
                 r'GPU1 (\d+) miss entries, (\d+) cold cache reads')
    rows=[]
    for stat,hit in zip(stats,hits):
        match=re.fullmatch(pattern,stat);hm=re.fullmatch(hit_pattern,hit)
        if match is None or hm is None:raise RuntimeError('Malformed hybrid cache counters')
        flag,ar,ab,am,rr,rb,rm=match.groups()
        ar,ab,rr,rb=map(int,(ar,ab,rr,rb));am,rm=map(float,(am,rm))
        if bool(int(flag))!=frozen or not all(math.isfinite(v) and v>=0 for v in (am,rm)):
            raise RuntimeError('Hybrid cache policy/timing differs from the bound scope')
        for reads,byte_count,wall in ((ar,ab,am),(rr,rb,rm)):
            if reads==0 and (byte_count!=0 or wall!=0) or reads>0 and not reads<=byte_count<=reads*(32<<20):
                raise RuntimeError('Hybrid cache transfer counters exceed bounded blob policy')
        if frozen and (ar or ab or am):raise RuntimeError('Frozen remote layer performed adaptive transfers')
        if ar+rr!=int(hm.group(4)):
            raise RuntimeError('Cold remote reads are not accounted for by adaptation/restoration')
        rows.append({'frozen':frozen,'adaptive_reads':ar,'adaptive_payload_bytes':ab,'adaptive_wall_ms':am,
                     'restore_reads':rr,'restore_payload_bytes':rb,'restore_wall_ms':rm,
                     'layers':int(hm.group(1)),'hit_entries':int(hm.group(2)),'miss_entries':int(hm.group(3))})
    if not any(row['restore_reads'] for row in rows):
        raise RuntimeError('No completed remote cache restoration was measured')
    return rows


def enable_owned_memory_telemetry():
    def measured(job):
        return {**_original_job_counters(job),**current_owned_memory(job)}
    shared.job_counters=measured


def source_hashes():
    source = ROOT / 'source'
    files = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).split(b'\0')
    names={rel.decode('utf-8') for rel in files if rel}
    names.update(('include/strata/core/remote_experts.hpp','src/core/remote_experts.cpp'))
    return {rel:shared.digest(source/rel) for rel in names if (source/rel).is_file()}


def artifact_checks(profile, *, require_mtp_equivalence=True, require_remote_experts=True,
                    require_remote_decode_equivalence=True):
    if profile != 'experimental-no-vram-floor':
        raise ValueError('Only the explicitly authorized experimental profile is supported')
    # The stock checker independently binds the pristine reference and existing
    # GGUF identity. It does not claim that this advanced source is pristine.
    report = shared.collect_report(allow_no_ram_reserve=True,
                                  allow_experimental_memory=True, allow_no_gpu_floor=True)
    gates = json.loads((ROOT / 'source-gates.json').read_text())
    if not gates.get('passed') or gates['executable_sha256'] != shared.EXE_HASH:
        raise RuntimeError('Advanced build/transport gates missing, failed or stale')
    if source_hashes() != gates['source_hashes']:
        raise RuntimeError('Advanced source changed after build/gates')
    for row in gates['bound_files']:
        if shared.digest(Path(row['path'])) != row['sha256']:
            raise RuntimeError(f"Advanced artifact mismatch: {row['path']}")
    decode_mode=gates.get('remote_decode_mode_for_dual','original')
    if decode_mode not in ('original','packed','graphs'):
        raise RuntimeError('Unknown bound remote decode mode')
    expected_ids=gates.get('remote_expert_layer_ids_for_dual')
    expected_payload=gates.get('remote_expert_payload_bytes_for_dual')
    expected_explicit=gates.get('remote_expert_layer_ids_explicit_for_dual')
    if (not isinstance(expected_ids,list) or len(expected_ids) not in (0,4,6)
            or (expected_ids and expected_ids!=sorted(set(expected_ids)))
            or any(type(layer) is not int or layer<0 or layer>47 for layer in expected_ids)
            or type(expected_payload) is not int or expected_payload<0
            or type(expected_explicit) is not bool or (expected_explicit and len(expected_ids) not in (4,6))
            or (expected_ids and not expected_explicit and expected_ids!=PREFIX_REMOTE_LAYERS)
            or bool(expected_ids)!=(expected_payload>0)):
        raise RuntimeError('Bound remote expert placement proof is invalid')
    six=len(expected_ids)==6
    if six:
        if expected_ids!=SIX_REMOTE_LAYERS or expected_payload!=SIX_REMOTE_PAYLOAD or decode_mode!='original':
            raise RuntimeError('Six-layer campaign is restricted to the exact Original candidate')
        compact=gates.get('compacted_loader',{})
        if compact.get('selected6_host_layout')!=SIX_HOST_PROOF:
            raise RuntimeError('Six-hole host/alias numerical gate is missing')
        path=Path(compact.get('log','')).resolve()
        row=next((row for row in gates['bound_files'] if Path(row['path']).resolve()==path),None)
        if (not path.is_relative_to(ROOT/'gate-runs') or path.name!='compacted-loader.log' or row is None
                or path.read_text().splitlines().count(SIX_HOST_MARKER)!=1):
            raise RuntimeError('Six-hole compaction evidence must bind its actual passing log')
    expected_hybrid=gates.get('remote_hybrid_for_dual',False)
    if type(expected_hybrid) is not bool or (expected_hybrid and (len(expected_ids)!=4 or decode_mode!='original')):
        raise RuntimeError('Bound hybrid expert placement proof is invalid')
    expected_frozen=gates.get('remote_cache_frozen_for_dual',False)
    if type(expected_frozen) is not bool or (expected_frozen and not expected_hybrid):
        raise RuntimeError('Bound frozen remote cache proof is invalid')
    wait_profiling=gates.get('matched_verify_wait_profile',False)
    if type(wait_profiling) is not bool:
        raise RuntimeError('Verifier wait profiling selection is not boolean')
    if wait_profiling:
        wait=gates.get('verify_wait',{})
        if (wait.get('devices')!=2 or wait.get('profiled_replays_per_device')!=4
                or wait.get('ready_per_device')!=2 or wait.get('delayed_per_device')!=2
                or wait.get('timer_unit')!='globaltimer_ns' or wait.get('outputs')!='exact'):
            raise RuntimeError('Verifier wait/timer numerical admission must pass before profiled full trials')
    if require_mtp_equivalence:
        parity=json.loads((ROOT/'mtp-parity-admission.json').read_text())
        if (not parity.get('passed') or parity.get('engine_executable_sha256')!=shared.EXE_HASH
                or parity.get('gates_sha256')!=shared.digest(ROOT/'source-gates.json')
                or parity.get('fixtures')!=MTP_FIXTURES or parity.get('modes')!=MTP_MODES
                or parity.get('probability_bits')!='exact'
                or parity.get('scope')!=PARITY_SCOPE):
            raise RuntimeError('Matching real MTP-layer equivalence gate must pass before full trials')
        evidence=parity.get('bound_evidence',[])
        expected={'result.json','native.log','probe.json','command.json'}
        paths=[Path(row['path']).resolve() for row in evidence]
        if (len(paths)!=4 or {path.name for path in paths}!=expected
                or len({path.parent for path in paths})!=1):
            raise RuntimeError('MTP admission must bind the four artifacts of one real parity run')
        for row,path in zip(evidence,paths):
            if not path.is_relative_to(ROOT/'parity-runs') or shared.digest(path)!=row['sha256']:
                raise RuntimeError('MTP equivalence evidence changed/outside campaign')
    if require_remote_experts:
        parity=json.loads((ROOT/'remote-experts-admission.json').read_text())
        if (not parity.get('passed') or parity.get('engine_executable_sha256')!=shared.EXE_HASH
                or parity.get('gates_sha256')!=shared.digest(ROOT/'source-gates.json')
                or parity.get('layers')!=(6 if six else 4) or parity.get('decode_calls')!=6
                or parity.get('prefill_calls')!=(7 if six else 5)
                or parity.get('quant_bytes')!='exact' or parity.get('outputs')!='exact' or parity.get('blob_bytes')!='exact'
                or parity.get('layer_ids')!=expected_ids or parity.get('payload_bytes')!=expected_payload
                or parity.get('scope')!=(REMOTE_SIX_SCOPE if six else REMOTE_SCOPE)):
            raise RuntimeError('Matching resident expert kernel equivalence must pass before full trials')
        evidence=parity.get('bound_evidence',[])
        paths=[Path(row['path']).resolve() for row in evidence]
        if (len(paths)!=4 or {path.name for path in paths}!={'result.json','native.log','probe.json','command.json'}
                or len({path.parent for path in paths})!=1):
            raise RuntimeError('Resident expert admission must bind four artifacts of one real run')
        for row,path in zip(evidence,paths):
            if not path.is_relative_to(ROOT/'parity-runs') or shared.digest(path)!=row['sha256']:
                raise RuntimeError('Resident expert equivalence evidence changed/outside campaign')
        if six:
            if parity.get('six_layer_coverage')!=SIX_EXPERT_PROOF:
                raise RuntimeError('Every six-layer format must have independent bytes/quant/output coverage')
            native_log=next(path for path in paths if path.name=='native.log')
            validate_six_expert_log(native_log.read_text())
        if expected_hybrid:
            if parity.get('masked_decode')!={'devices':2,'calls_per_device':4,'entries_per_device':32,
                                            'all_hit_calls_per_device':1,'outputs':'exact','negatives':'exact'} or \
               parity.get('hybrid_dispatch')!={'devices':2,'outputs':'exact','cold_restore':'exact','failures':'exact'}:
                raise RuntimeError('Hybrid dispatch/restore numerical admission missing or incomplete')
            native_log=next(path for path in paths if path.name=='native.log')
            validate_hybrid_log(native_log.read_text())
    decode_checked=require_remote_decode_equivalence and decode_mode!='original'
    if decode_checked:
        parity=json.loads((ROOT/'remote-decode-admission.json').read_text())
        if (not parity.get('passed') or parity.get('engine_executable_sha256')!=shared.EXE_HASH
                or parity.get('gates_sha256')!=shared.digest(ROOT/'source-gates.json')
                or parity.get('layers')!=4 or parity.get('bursts')!=64 or parity.get('modes')!=3
                or parity.get('devices')!=2 or parity.get('conditions')!=24
                or parity.get('reference_bytes')!=65536000 or parity.get('outputs')!='exact'
                or parity.get('layer_ids')!=expected_ids or parity.get('payload_bytes')!=expected_payload
                or parity.get('k4_fallback_exact') is not True
                or parity.get('graph_replays_per_graph_condition')!=256 or parity.get('scope')!=DECODE_SCOPE):
            raise RuntimeError('Matching packed/graph decode equivalence must pass before full trials')
        evidence=parity.get('bound_evidence',[])
        paths=[Path(row['path']).resolve() for row in evidence]
        if (len(paths)!=4 or {path.name for path in paths}!={'result.json','native.log','probe.json','command.json'}
                or len({path.parent for path in paths})!=1):
            raise RuntimeError('Decode mode admission must bind four artifacts of one real run')
        for row,path in zip(evidence,paths):
            if not path.is_relative_to(ROOT/'parity-runs') or shared.digest(path)!=row['sha256']:
                raise RuntimeError('Decode mode equivalence evidence changed/outside campaign')
    report.update(executable_kind='source-built-advanced-candidate',
                  advanced_executable_sha256=shared.EXE_HASH,
                  mtp_layer_equivalence_checked=require_mtp_equivalence,
                  remote_expert_equivalence_checked=require_remote_experts,
                  six_layer_expert_equivalence_checked=require_remote_experts and six,
                  hybrid_expert_equivalence_checked=require_remote_experts and expected_hybrid,
                  remote_cache_frozen_for_dual=expected_frozen,
                  remote_decode_equivalence_checked=decode_checked,
                  verifier_wait_profiling=wait_profiling,
                  remote_expert_layer_ids=expected_ids,
                  remote_expert_layer_ids_explicit=expected_explicit,
                  remote_expert_payload_bytes=expected_payload,
                  advanced_gates_scope=gates['evidence_scope'])
    return report, gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['load', 'smoke', 'bench'])
    parser.add_argument('--variant', choices=['single', 'dual'], required=True)
    parser.add_argument('--pool-workers', type=int, choices=[4,6,8,10,12,16,19], default=8)
    parser.add_argument('--vram-reserve-mib', type=int, choices=[700,1024,1536,2048,2560,3072], default=3072)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if not 30 <= args.timeout <= 1800:
        raise ValueError('Timeout outside bounded range')
    config = ROOT / f'{args.variant}-config.json'
    cfg = json.loads(config.read_text())
    exe = Path(cfg['exe'])
    if exe.resolve() != (ROOT / 'build/strata.exe').resolve():
        raise ValueError('Executable outside isolated build boundary')
    strategy=cfg.get('strategy','mtp')
    if strategy not in ('mtp','resident-experts'):
        raise ValueError('Unknown isolated strategy')
    expected_device = '0' if args.variant == 'single' or strategy=='resident-experts' else '1'
    idx = cfg['args'].index('--mtp-device')
    if cfg['args'][idx + 1] != expected_device:
        raise ValueError('MTP placement does not match selected variant')
    gates=json.loads((ROOT/'source-gates.json').read_text())
    remote_ids,remote_explicit=remote_placement(cfg['args'])
    expected_ids=gates.get('remote_expert_layer_ids_for_dual',[]) if strategy=='resident-experts' and args.variant=='dual' else []
    expected_explicit=gates.get('remote_expert_layer_ids_explicit_for_dual',False) if expected_ids else False
    if remote_ids!=expected_ids or remote_explicit!=expected_explicit:
        raise ValueError('Resident expert placement does not match selected strategy/variant')
    decode_mode=cfg['args'][cfg['args'].index('--remote-expert-mode')+1]
    expected_mode=gates.get('remote_decode_mode_for_dual','original') if remote_ids else 'original'
    if decode_mode not in ('original','packed','graphs') or decode_mode!=expected_mode:
        raise ValueError('Remote decode mode does not match bound strategy/variant')
    hybrid_placement(cfg['args'],bool(remote_ids) and gates.get('remote_hybrid_for_dual',False))
    frozen_remote_cache_placement(cfg['args'],bool(remote_ids) and gates.get('remote_cache_frozen_for_dual',False))
    if ('--verify-wait-profile' in cfg['args'])!=gates.get('matched_verify_wait_profile',False):
        raise ValueError('Verifier wait profiling does not match the bound configuration')
    if cfg['args'][cfg['args'].index('--max-context') + 1] != '4096':
        raise ValueError('Only the initial 4K tier is authorized')
    args.profile, args.gpu = 'experimental-no-vram-floor', 0
    shared.ROOT = ROOT
    shared.EXE_HASH = shared.digest(exe)
    shared.artifact_checks = artifact_checks
    enable_owned_memory_telemetry()
    shared.run_trial(args, config_file=config, probe_file=ROOT / 'native_probe.py', device_visibility='0,1')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'ADVANCED STRATA STOP: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
