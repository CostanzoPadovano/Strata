"""Validate complete bounded GPU-wait frames before ranking exposed CPU waits.

These are diagnostic GPU time deltas, not raw CPU work, link bandwidth or a
speedup. In particular, a remote layer's CPU-stage wait also includes its remote
GPU service. Placement should start from a single-GPU diagnostic.
"""
from __future__ import annotations

MAX_LAYERS=48
MAX_REQUESTS=16
UINT64_MAX=2**64-1


def _uint(value):
    return type(value) is int and 0<=value<=UINT64_MAX


def validate_frames(rows,names,layers=MAX_LAYERS):
    if type(layers) is not int or not 1<=layers<=MAX_LAYERS or len(names)>MAX_REQUESTS:
        raise ValueError('GPU-wait frame dimensions exceed diagnostic bounds')
    if len(rows)!=len(names)*layers or len(set(names))!=len(names):
        raise ValueError('GPU-wait frames must match every completed unique request')
    frames=[]
    for index,name in enumerate(names):
        frame=rows[index*layers:(index+1)*layers]
        windows=frame[0].get('windows')
        if not _uint(windows) or not 1<=windows<=UINT64_MAX//2:
            raise ValueError('GPU-wait frame has no bounded completed windows')
        checked=[]
        for layer,row in enumerate(frame):
            if (not _uint(row.get('layer')) or not _uint(row.get('windows')) or
                    row.get('layer')!=layer or row.get('windows')!=windows):
                raise ValueError('GPU-wait layers/order/window counts are inconsistent')
            arrays=[row.get(key) for key in ('calls','waited','blocked_ns')]
            if any(not isinstance(values,list) or len(values)!=3 or
                   not all(_uint(value) for value in values) for values in arrays):
                raise ValueError('GPU-wait counters require three uint64 stages')
            calls,waited,blocked=arrays
            if len(set(calls))!=1 or not windows<=calls[0]<=2*windows:
                raise ValueError('GPU-wait stage calls do not match completed windows')
            if any(w>c or (w==0 and ns!=0) for c,w,ns in zip(calls,waited,blocked)):
                raise ValueError('GPU-wait blocked/ready counters are inconsistent')
            checked.append({'layer':layer,'cpu_blocked_ns':blocked[2],
                            'plan_blocked_ns':blocked[0],'pcie_blocked_ns':blocked[1],
                            'cpu_waited_calls':waited[2],'calls':calls[0]})
        frames.append({'name':name,'windows':windows,'layers':checked})
    return frames


def rank_decode_cpu_wait(frames):
    decodes=[frame for frame in frames if frame['name'].startswith('decode_screen_')]
    if len(decodes)!=3:
        raise ValueError('Layer ranking requires all three completed decode requests')
    layers=len(decodes[0]['layers'])
    if any(len(frame['layers'])!=layers for frame in decodes):
        raise ValueError('Layer ranking requires matching dimensions')
    windows=sum(frame['windows'] for frame in decodes)
    ranking=[]
    for layer in range(layers):
        ns=sum(frame['layers'][layer]['cpu_blocked_ns'] for frame in decodes)
        ranking.append({'layer':layer,'cpu_blocked_ms':ns/1e6,'cpu_blocked_us_per_window':ns/windows/1e3,
                        'cpu_waited_calls':sum(frame['layers'][layer]['cpu_waited_calls'] for frame in decodes)})
    ranking.sort(key=lambda row:(-row['cpu_blocked_ms'],row['layer']))
    return {'decode_requests':3,'completed_windows':windows,'ranking':ranking,
            'cpu_blocked_ms_total':sum(row['cpu_blocked_ms'] for row in ranking),
            'scope':'instrumented GPU exposed CPU-stage waits; not CPU runtime or a speedup'}
