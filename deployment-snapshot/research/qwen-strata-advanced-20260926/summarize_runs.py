"""Read-only compact native timing/resource/placement summary of named campaign runs."""
import argparse
import json
from pathlib import Path
import re
import statistics
from verify_wait_profile import validate_frames,rank_decode_cpu_wait

ROOT = Path(__file__).resolve().parent


def summarize(tag):
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in tag):
        raise ValueError('Invalid run tag')
    run = ROOT/'runs'/tag
    result = json.loads((run/'result.json').read_text())
    probe = json.loads((run/'probe.json').read_text()) if (run/'probe.json').exists() else {}
    log = (run/'engine.log').read_text(errors='replace') if (run/'engine.log').exists() else ''
    telemetry = [json.loads(line) for line in (run/'telemetry.jsonl').read_text().splitlines() if line.strip()]
    decode = [row['decode_tok_s'] for row in probe.get('rows',[]) if row['name'].startswith('decode_screen_')]
    prefill = [row['prompt_tok_s'] for row in probe.get('rows',[]) if row['name'].startswith('prefill_screen_')]
    cache = re.findall(r'expert cache (\d+) slots, ([\d.]+) GiB',log)
    pin = re.findall(r'bounded CUDA registration (\d+) MiB / (\d+) MiB cap \((\d+) slices\)',log)
    metrics = re.findall(r'strata mtp request: target CUDA (\d+) -> draft CUDA (\d+), delta (\d+) rounds, '
                         r'([\d.]+) ms prefill, ([\d.]+) ms draft, (\d+) (?:staged copies|explicit CUDA copies), (\d+) bytes',log)
    mapped = re.findall(r'; (\d+) mapped graph residual reads, (\d+) bytes',log)
    batched = re.findall(r'; (\d+) batched rounds, (\d+) unused drafts computed',log)
    remote = re.findall(r'strata remote experts request: CUDA (\d+), delta (\d+) decode layers, '
                        r'(\d+) prefill layers',log)
    graph_replays = re.findall(r'strata remote experts graph request: delta (\d+) kernel replays',log)
    remote_modes = re.findall(r'strata remote experts mode: (original|packed|graphs);',log)
    remote_placement = re.findall(r'strata remote experts: CUDA (\d+) owns (\d+) whole layers, '
                                  r'(\d+) payload bytes, (\d+) total device bytes; MTP remains on CUDA (\d+)',log)
    host_payload = re.findall(r'strata generate: loaded ([\d.]+) GiB at ([\d.]+) GiB/s',log)
    wait_rows=[json.loads(line.removeprefix('VERIFY_WAIT_PROFILE ')) for line in log.splitlines()
               if line.startswith('VERIFY_WAIT_PROFILE ')]
    wait_frames=None
    wait_rank=None
    if wait_rows:
        wait_frames=validate_frames(wait_rows,[row['name'] for row in probe.get('rows',[])])
        if result['passed'] and decode:
            wait_rank=rank_decode_cpu_wait(wait_frames)
    return {'tag':tag,'passed':result['passed'],'failure':result.get('failure'),
            'native_exit_code':probe.get('native_exit_code'),'load_seconds':probe.get('load_wall_seconds'),
            'semantic_smoke':probe.get('semantic_smoke_passed'),
            'cache':cache,'pin':pin,'request_device_metrics':metrics,'request_mapped_residual_metrics':mapped,
            'request_chain_batch_metrics':batched,
            'request_remote_expert_metrics':remote,'remote_expert_placement':remote_placement,
            'request_remote_graph_replays':[int(value) for value in graph_replays],
            'remote_decode_modes':remote_modes,
            'verify_wait_rows':wait_rows,
            'verify_wait_frames':wait_frames,'decode_cpu_wait_ranking':wait_rank,
            'host_expert_payload_gib_and_load_rate':host_payload,
            'decode':decode,'decode_mean':statistics.mean(decode) if decode else None,
            'prefill':prefill,'prefill_mean':statistics.mean(prefill) if prefill else None,
            'native_counts':[{'name':r['name'],**r['native_timings']} for r in probe.get('rows',[])],
            'ram_min_gib':min((t['ram_free']/2**30 for t in telemetry),default=None),
            'commit_min_gib':min((t['commit_free']/2**30 for t in telemetry),default=None),
            'job_peak_gib':max((t['peak_job_bytes']/2**30 for t in telemetry),default=None),
            'owned_private_peak_sample_gib':max((t['owned_private_bytes_current']/2**30 for t in telemetry
                 if t.get('owned_private_sample_complete')),default=None),
            'max_temperature_c':max((gpu[2] for t in telemetry for gpu in t['gpu']),default=None)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags',nargs='+')
    args=parser.parse_args()
    print(json.dumps([summarize(tag) for tag in args.tags],indent=2))
