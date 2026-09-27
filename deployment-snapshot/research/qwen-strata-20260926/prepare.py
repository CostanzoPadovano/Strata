"""Reproducible synthetic gates, model proofs and runtime binding; no model inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import struct
import threading
import urllib.request
import urllib.error
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
BUILD=ROOT/'build-native'
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(ROOT))
from guard import digest
MODEL=Path(r'C:\Users\costa\.lmstudio\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\IQ3_XXS')
SHARDS=[('Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf',47039860096,'219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15'),
        ('Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf',28800138432,'316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113')]
def http_tests():
    from serve.server import ByteTokenizer,MockEngine,Service,serve
    from serve.frontend import ChatTemplate,openai_to_messages
    tok=ByteTokenizer();tpl=ChatTemplate(SOURCE/'serve/chat_template.jinja')
    tools=[{'type':'function','function':{'name':'bash','parameters':{'type':'object','properties':{'command':{'type':'string'}}}}}]
    script='<tool_call>\n<function=bash>\n<parameter=command>printf STRATA_OK</parameter>\n</function>\n</tool_call>'
    svc=Service(MockEngine(tok,script),tok,tpl,model_name='strata-test');svc.api_key='synthetic-test-key'
    httpd=serve(svc,port=0);url=f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
    def request(body,key='synthetic-test-key'):
        req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        try:
            with urllib.request.urlopen(req,timeout=10) as r:return r.status,r.read().decode()
        except urllib.error.HTTPError as e:return e.code,e.read().decode()
    try:
        body={'messages':[{'role':'user','content':'Use bash.'}],'tools':tools,'reasoning_effort':'none','max_tokens':256}
        code,out=request(body);assert code==200,(code,out)
        call=json.loads(out)['choices'][0]['message']['tool_calls'][0]
        assert call['function']['name']=='bash' and json.loads(call['function']['arguments'])=={'command':'printf STRATA_OK'}
        code,out=request(dict(body,stream=True));assert code==200 and 'tool_calls' in out and '[DONE]' in out
        assert request(body,key='wrong')[0]==401
        assert request(dict(body,max_tokens=-1))[0]==400
        assert request(dict(body,max_tokens=16385))[0]==400
        assert request(dict(body,reasoning_effort='nonsense'))[0]==400
        _,_,kw=openai_to_messages(dict(body,reasoning_effort='xhigh'))
        assert kw=={'reasoning_effort':'xhigh'} and 'Reasoning effort is set to xhigh' in tpl.render(body['messages'],**kw)
        golden=json.loads((SOURCE/'serve/chat_golden.json').read_text())
        checked=0
        for row in golden:
            if row.get('error'):continue
            kwargs={k:row[k] for k in ('add_generation_prompt','enable_thinking','reasoning_effort') if k in row}
            rendered=tpl.render(row['messages'],tools=row.get('tools'),**kwargs)
            assert rendered==row['rendered'],row['name'];checked+=1
        return {'passed':True,'golden_templates':checked,'http_checks':7,'native_xhigh':True}
    finally:httpd.shutdown();httpd.server_close()
def validate():
    env=dict(os.environ);env['PATH']=r'E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit\bin'+os.pathsep+env['PATH']
    names=['strata-device','dequant_s2_parity','s2_gemv_parity','shared_expert_parity','gr_parity','gdn_parity',
        's2_gemv_q8_parity','sampler_parity','rope_parity','quantize_act_parity','router_top10_parity','s_gemv_parity',
        'elementwise_parity','bf16_gemv_parity','s_gemv_q8k_parity','qsa_parity','ple_reader_test','kv_q8_parity']
    rows=[]
    guard_test=subprocess.run([sys.executable,str(ROOT/'test_guard.py')],cwd=ROOT,capture_output=True,text=True,timeout=30)
    (ROOT/'guard-test.log').write_text(guard_test.stdout+guard_test.stderr)
    for gpu in (0,1):
        env['CUDA_VISIBLE_DEVICES']=str(gpu)
        for name in names:
            args=[str(BUILD/(name+'.exe'))]+([] if name=='kv_q8_parity' else ['--selftest'])
            r=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
            (ROOT/f'gate-gpu{gpu}-{name}.log').write_text(r.stdout+r.stderr,encoding='utf-8')
            rows.append({'gpu':gpu,'name':name,'passed':r.returncode==0});print(rows[-1],flush=True)
    env.pop('CUDA_VISIBLE_DEVICES',None)
    r=subprocess.run([str(BUILD/'dual_expert_test.exe'),str(ROOT/'synthetic-dual-final')],cwd=ROOT,env=env,
                      capture_output=True,text=True,timeout=180)
    (ROOT/'dual-test-final.log').write_text(r.stdout+r.stderr)
    dual=r.returncode==0
    frontend=http_tests()
    result={'strata_sha256':digest(BUILD/'strata.exe'),'passed':all(x['passed'] for x in rows) and dual and frontend['passed'] and guard_test.returncode==0,
            'component_gates':rows,'dual_gpu':dual,'frontend':frontend,'guardian':guard_test.returncode==0}
    (ROOT/'gates.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise RuntimeError('Synthetic admission failed')
def prove_model():
    rows=[]
    for name,size,expected in SHARDS:
        p=MODEL/name;before=p.stat()
        assert before.st_size==size,'Shard length mismatch'
        print('SHA256 '+name,flush=True);h=digest(p);after=p.stat()
        assert h==expected and after.st_mtime_ns==before.st_mtime_ns,'Shard identity mismatch'
        rows.append({'path':str(p),'bytes':size,'sha256':h,'mtime_ns':after.st_mtime_ns})
    (ROOT/'model-proof.json').write_text(json.dumps({'files':rows},indent=2))
def bind():
    gates=json.loads((ROOT/'gates.json').read_text());assert gates['passed']
    cpu_total=0;host=0;gpu=0;layers=[];layer_bytes={}
    for line in (ROOT/'pack-iq3/native_experts.txt').read_text().splitlines():
        if line.startswith('#') or not line:continue
        fields=line.split();size=int(fields[4])*512
        layer_bytes[int(fields[0])]=int(fields[4])
        if gpu+size<=12*(1<<30):gpu+=size;layers.append(int(fields[0]))
        else:cpu_total+=size
    raw=(SOURCE/'data/expert-profile.bin').read_bytes();header=struct.unpack('<4s5I',raw[:24])
    assert header[:4]==(b'STRP',1,48,512),header
    rank=list(struct.iter_unpack('<HH',raw[24:24+header[5]*4]));reserved=set();primary=0
    primary_budget=2304*max(layer_bytes.values())
    for l,e in rank:
        if l in layers:continue
        size=(layer_bytes[l]+255)//256*256
        if primary+size>primary_budget:break
        primary+=size;reserved.add((l,e))
    resident=set()
    def retain(l,e):
        nonlocal host
        if l not in layers and (l,e) not in reserved and (l,e) not in resident and host+layer_bytes[l]<=14*(1<<30):
            resident.add((l,e));host+=layer_bytes[l]
    for l,e in rank:retain(l,e)
    for l in range(48):
        for e in range(512):retain(l,e)
    mtp=ROOT/'mtp/rt'
    assert (mtp/'experts.bin').stat().st_size>0
    config={'exe':str(BUILD/'strata.exe'),'cwd':str(SOURCE),'tokenizer':str(ROOT/'pack-iq3/tokenizer'),
        'model_name':'qwen38-strata-dual','api_key_file':r'C:\llama_official\router\api-key.txt',
        'lib_dirs':[r'E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit\bin'],
        'args':['--pack',str(ROOT/'pack-iq3'),'--native',str(MODEL/SHARDS[0][0]),'--ple-gguf',str(MODEL/SHARDS[1][0]),
                '--expert-profile',str(SOURCE/'data/expert-profile.bin'),'--expert-cache','2304','--expert-gpu1-mib','12288',
                '--expert-host-mib','14336',
                '--adapt-every','0','--pcie-frac','0','--vram-reserve-mib','2560','--pool-workers','8',
                '--ple-io','direct','--ple-inflight','32','--ple-row-cache','262144',
                '--prefill','1024','--spec','4','--spec-min-p','0.5','--mtp',str(mtp),'--max-context','4096','--kv','int8']}
    (ROOT/'server-config.json').write_text(json.dumps(config,indent=2))
    paths=[BUILD/'strata.exe',ROOT/'guard.py',ROOT/'server-config.json']
    paths+=list((SOURCE/'serve').glob('*.py'))+list((SOURCE/'serve').glob('*.jinja'))
    paths+=[SOURCE/'tools/strata_tokenizer.py',SOURCE/'data/expert-profile.bin']
    paths+=list((ROOT/'pack-iq3').glob('*.bin'))+list((ROOT/'pack-iq3').glob('*.txt'))+list((ROOT/'pack-iq3/tokenizer').glob('*'))
    paths+=list(mtp.glob('*'))
    paths+=list((SOURCE/'include/strata/core').glob('dual*'))+[SOURCE/'include/strata/core/expert_source.hpp',SOURCE/'include/strata/core/pinned.hpp',SOURCE/'src/core/pinned.cu',SOURCE/'CMakeLists.txt',SOURCE/'src/core/dual_expert_source.cpp',SOURCE/'src/core/expert_source.cpp',SOURCE/'src/program/generate.cpp',SOURCE/'src/prefill/prefill.cpp']
    paths+=[SOURCE/'include/strata/platform/direct_file.hpp',SOURCE/'src/platform/direct_file.cpp']
    files=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in paths if p.is_file()]
    manifest={'schema':'antirez-strata/v1','source_commit':'6da1f667e86558b152ab128edf3ebf77a80a9e57',
        'ggml_commit':'3cf03257f219afbe7334045ff7c6a06ac68c627d','mtp_commit':'de4b8e4d43b917e7706784d8bb445c9af86a3540',
        'cuda':'13.3.73','gpu_arch':120,'cpu':'AVX2','gpu1_layers':layers,'host_experts_bytes':host,'gpu1_experts_bytes':gpu,
        'disk_experts_bytes':cpu_total-host,'gpu0_experts_reserved_bytes':primary,'gpu0_cache_budget_bytes':primary_budget,
        'startup_commit_overhead_estimate_bytes':24*(1<<30),
        'strata_sha256':digest(BUILD/'strata.exe'),'runtime_files':files}
    (ROOT/'runtime-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='runtime_files'},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['validate','prove-model','bind']);a=ap.parse_args()
    {'validate':validate,'prove-model':prove_model,'bind':bind}[a.action]()
