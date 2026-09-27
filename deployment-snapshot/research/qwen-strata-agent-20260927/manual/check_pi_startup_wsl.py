"""Read-only Pi RPC startup/model-switch probes. Never send a prompt."""
import hashlib
import json
import os
from pathlib import Path
import signal
import selectors
import subprocess
import time

BASE=Path('/home/costapad/.pi/agent')
FILES=[BASE/'models.json',BASE/'settings.json']
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES]
wrapper='/home/costapad/.local/bin/pi'
document=json.loads(FILES[0].read_text())
settings=json.loads(FILES[1].read_text())
other=next((p,m['id']) for p,c in document['providers'].items() for m in c.get('models',[])
           if p not in ('local-qwen38','local-thinkingcap-qwen38','local-thinkingcap-qwen38-vision'))
cases=[('default',[]),('explicit-other',['--provider',other[0],'--model',other[1]]),
       ('explicit-thinkingcap',['--provider','local-thinkingcap-qwen38','--model',settings['defaultModel']])]
if '--before' in os.sys.argv:
    cases=cases[:1]
report=[]
for label,args in cases:
    commands=[{'id':'state1','type':'get_state'},
              {'id':'switch','type':'set_model','provider':other[0],'modelId':other[1]},
              {'id':'state2','type':'get_state'}]
    started=time.monotonic()
    process=subprocess.Popen([wrapper,'--mode','rpc','--no-session',*args],stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    messages=[]
    try:
        selector=selectors.DefaultSelector(); selector.register(process.stdout,selectors.EVENT_READ)
        for command in commands:
            process.stdin.write(json.dumps(command)+'\n'); process.stdin.flush()
            while True:
                remaining=16-(time.monotonic()-started)
                if remaining<=0 or not selector.select(remaining): raise TimeoutError('RPC response timeout')
                line=process.stdout.readline()
                if not line: break
                try: row=json.loads(line)
                except ValueError: continue
                if row.get('type')=='response':
                    messages.append(row)
                    if row.get('id')==command['id']: break
            if process.poll() is not None: break
        selector.close()
        process.stdin.close(); process.stdin=None
        stdout,stderr=process.communicate(timeout=3)
    except (TimeoutError,subprocess.TimeoutExpired,BrokenPipeError):
        os.killpg(process.pid,signal.SIGTERM)
        if process.stdin:
            process.stdin.close(); process.stdin=None
        stdout,stderr=process.communicate(timeout=3)
    for line in stdout.splitlines():
        try:
            row=json.loads(line)
            if row.get('type')=='response': messages.append(row)
        except ValueError: pass
    state=next((r.get('data',{}) for r in messages if r.get('id')=='state1'),{})
    switched=next((r for r in messages if r.get('id')=='switch'),{})
    final=next((r.get('data',{}) for r in messages if r.get('id')=='state2'),{})
    report.append({'case':label,'exit':process.returncode,'elapsed_s':round(time.monotonic()-started,3),
                   'rpc_responses':len(messages),'initial_provider':state.get('model',{}).get('provider'),
                   'initial_model':state.get('model',{}).get('id'),'switch_success':switched.get('success'),
                   'final_provider':final.get('model',{}).get('provider'),
                   'strata_warning':'Strata' in stderr,'prompt_sent':False})
after=[hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES]
result={'probes':report,'models_settings_unchanged':before==after,'hashes':after,'inference_started':False}
print(json.dumps(result,indent=2))
if '--before' not in os.sys.argv:
    assert before==after
    assert all(r['exit']==0 and r['rpc_responses']==3 and r['switch_success'] is True for r in report)
    assert all(r['final_provider']==other[0] for r in report)
