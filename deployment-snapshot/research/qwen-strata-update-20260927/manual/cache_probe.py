"""Finite native cache/fresh equivalence probe inside the inherited, monitored model Job."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import queue
import struct
import sys
import threading
import time
from cache_protocol import parse_cache_done

ROOT=Path(__file__).resolve().parent.parent

def padded_prompt(base, target, start_token, end_token, cycle):
    if not base or not cycle or not len(base)<=target<=40000: raise ValueError('Invalid finite prompt plan')
    turn=max(i for i,t in enumerate(base) if t==start_token)
    end=max(i for i,t in enumerate(base[:turn]) if t==end_token)
    extra=target-len(base)
    padded=base[:end]+[cycle[i%len(cycle)] for i in range(extra)]+base[end:]
    return padded,turn+extra

def token_hash(tokens):
    return hashlib.sha256(b''.join(struct.pack('<i',t) for t in tokens)).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--mode',choices=('bench',),required=True)
    opts=parser.parse_args()
    run=opts.run.resolve()
    if not run.is_relative_to(ROOT/'runs'): raise ValueError('Evidence outside candidate')
    deadline=time.monotonic()+30
    def wait(name):
        while not (run/name).exists():
            if time.monotonic()>deadline: raise TimeoutError('No owned Job/READY acceptance')
            time.sleep(.05)
    wait('job-attached')
    sys.path[:0]=[str(ROOT/'source'),str(ROOT/'source/tools')]
    from serve import server
    from serve.frontend import ChatTemplate
    from strata_tokenizer import Tokenizer
    cfg=json.loads((run/'config.json').read_text())
    context=int(cfg['args'][cfg['args'].index('--max-context')+1])
    if context not in (4096,98304): raise ValueError('Unqualified cache context')
    result={'passed':False,'scope':'finite greedy cache equivalence, not overall quality/throughput',
            'context':context,'rows':[],'comparisons':[]}
    engine=None
    def save(): (run/'probe.json').write_text(json.dumps(result,indent=2))
    def phase(name):
        (run/'phase.json').write_text(json.dumps({'phase':name,'time':time.time()}))
        print(name,flush=True)
    class Engine(server.StrataEngine):
        def _pump(self):
            self.lines=queue.Queue(maxsize=2048)
            return super()._pump()
        def _parse_done(self,line): self.last=parse_cache_done(line)
    try:
        phase('cache probe loading')
        engine=Engine(cfg['exe'],cfg['args'],cwd=cfg['cwd'],log=cfg['log'])
        if engine.max_context!=context: raise ValueError('Native context mismatch')
        (run/'ready.json').write_text(json.dumps({'native_pid':engine.proc.pid,'context':context}))
        deadline=time.monotonic()+30; wait('ready-accepted.json')
        directory=Path(cfg['tokenizer'])
        vocab=json.loads((directory/'vocab.json').read_text())
        tokens=[None]*len(vocab)
        for token,index in vocab.items(): tokens[index]=token
        tok=Tokenizer(tokens,(directory/'merges.txt').read_text().split('\n'),json.loads((directory/'token_type.json').read_text()))
        svc=server.Service(engine,tok,ChatTemplate(directory/'chat_template.jinja'),model_name='qwen3.8-flash-next-local')
        base,_=svc.prepare([{'role':'user','content':'Test controllato. Rispondi solo con il numero 42.'}],None,
                           {'enable_thinking':False},24)
        start=tok.encode('<|im_start|>',parse_special=True)[0]
        end=tok.encode('<|im_end|>',parse_special=True)[0]
        cycle=tok.encode(' Alfa beta gamma delta: 13, 21, 34. Contesto sintetico per il test della cache.\n')
        a,L=padded_prompt(base,3073,start,end,cycle)
        alternate=tok.encode(' NOTE')[0]
        if alternate==a[L]: alternate=1
        b=list(a); b[L]=alternate
        c=a+[alternate,1,2]
        def call(name,ids,maximum=24,cancel_kind=None):
            if not ids or len(ids)+maximum+8>context or maximum>128: raise ValueError('Probe outside finite limits')
            if any(type(t)!=int or not 0<=t<len(vocab) for t in ids): raise ValueError('Invalid planned token')
            phase(name)
            cancel=threading.Event(); generated=[]; engine.last={}
            gen=engine.generate(ids,maximum,{},cancel)
            started=time.monotonic()
            try:
                for t in gen:
                    if t is not None: generated.append(t)
                    if (cancel_kind=='prefill' and t is None) or (cancel_kind=='decode' and t is not None):
                        cancel.set(); break
            finally: gen.close()
            if not engine.last: raise ValueError('No authentic DONE')
            if engine.last['prompt_tokens']!=len(ids) or engine.last['generated']>maximum: raise ValueError('Count mismatch')
            if cancel_kind and engine.last['finish']!='cancel': raise ValueError('Cancellation did not reach native DONE cancel')
            row={'name':name,'input_tokens':len(ids),'observed_tokens':len(generated),
                 'input_sha256':token_hash(ids),'output_sha256':token_hash(generated),
                 'native_timings':dict(engine.last),'wall_seconds':time.monotonic()-started}
            result['rows'].append(row)
            with (run/'timings.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
            save(); return generated,dict(engine.last)
        def reset(): call('reset',[0,1],2)
        def same(name,expected,actual,reused,floor):
            if expected!=actual: raise AssertionError(name+' cache/fresh output-token mismatch')
            if reused<floor: raise AssertionError(name+f' missing reuse: {reused} < {floor}')
            result['comparisons'].append({'name':name,'tokens_exact':True,'reused':reused,'minimum_reuse':floor})
            save()
        # Checkpoint restore at a non-block-aligned last-turn boundary, including changed next token.
        expected_b,_=call('checkpoint-branch-cold',b)
        reset(); call('checkpoint-seed',a,1)
        actual_b,timing=call('checkpoint-branch-warm',b)
        same('checkpoint branch',expected_b,actual_b,timing['prompt_reused'],L)
        # Live prefix, length-terminal catch-up and changed MTP continuation token.
        reset(); expected_c,_=call('live-branch-cold',c)
        reset(); call('live-seed-length1',a,1)
        actual_c,timing=call('live-branch-warm',c)
        same('live branch after length',expected_c,actual_c,timing['prompt_reused'],len(a))
        # EOS termination followed by a live continuation, plus cancellation at both phases.
        reset(); eos_tokens,eos_timing=call('eos-seed',base,48)
        if eos_timing['finish']!='stop': raise AssertionError('EOS fixture did not reach stop')
        continuation=base+eos_tokens+[alternate,1]
        warm_eos,timing=call('eos-live-warm',continuation)
        eos_live_length=len(base)+len(eos_tokens)-1
        if timing['prompt_reused']!=eos_live_length:
            raise AssertionError(f'EOS live boundary: {timing["prompt_reused"]} != {eos_live_length}')
        reset(); cold_eos,_=call('eos-live-cold',continuation)
        same('live after EOS',cold_eos,warm_eos,timing['prompt_reused'],eos_live_length)
        reset(); call('cancel-prefill',a,128,'prefill')
        after_cancel,_=call('after-prefill-cancel',base,48)
        reset(); cold_base,_=call('after-cancel-cold-control',base,48)
        same('after prefill cancel',cold_base,after_cancel,0,0)
        counting,_=svc.prepare([{'role':'user','content':'Conta da 1 a 1000, una riga per numero, senza fermarti.'}],None,
                               {'enable_thinking':False},128)
        call('cancel-decode',counting,128,'decode')
        after_decode,_=call('after-decode-cancel',base,48)
        same('after decode cancel',cold_base,after_decode,0,0)
        if context==98304:
            # A formerly long prompt cannot leave skipped early draft KV on a shorter rewind.
            long_a,_=padded_prompt(base,34817,start,end,cycle)
            short=long_a[:16417]
            short[16384]=alternate
            reset(); cold_short,_=call('rewind-short-cold',short)
            reset(); call('rewind-long-seed',long_a,1)
            warm_short,timing=call('rewind-short-warm',short)
            same('long-to-short checkpoint rewind',cold_short,warm_short,timing['prompt_reused'],16384)
        result['passed']=True
    except BaseException as error:
        result['failure']=str(error); raise
    finally:
        if engine:
            engine.close()
            engine.proc.wait(timeout=15)
            result['native_exit_code']=engine.proc.returncode
            if engine.proc.returncode: result['passed']=False
        save()

if __name__=='__main__': main()
