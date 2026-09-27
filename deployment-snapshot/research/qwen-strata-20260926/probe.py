"""External semantic/tool/performance probe; guardian continues sampling during every request."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
ROOT=Path(__file__).resolve().parent
def call(messages,max_tokens=256,tools=None,effort='none',timeout=300):
    key=Path(r'C:\llama_official\router\api-key.txt').read_text().strip()
    body={'model':'qwen38-strata-dual','messages':messages,'max_tokens':max_tokens,'reasoning_effort':effort,'stream':False}
    if tools:body['tools']=tools
    req=urllib.request.Request('http://127.0.0.1:8035/v1/chat/completions',data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    t=time.monotonic()
    with urllib.request.urlopen(req,timeout=timeout) as r:result=json.load(r)
    result['wall_seconds']=time.monotonic()-t
    timing=result.get('timings',{})
    result['measured_rates']={
        'prompt_tok_s':timing['prompt_tokens']*1000/timing['prompt_ms'] if timing.get('prompt_ms',0)>0 else None,
        'decode_tok_s':timing['generated']*1000/timing['decode_ms'] if timing.get('decode_ms',0)>0 else None}
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stop',action='store_true');p.add_argument('--context-check',action='store_true');a=p.parse_args()
    run=ROOT/'runs'/a.tag;deadline=time.monotonic()+360
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8035/health',timeout=2) as r:
                if json.load(r).get('status')=='ok':break
        except OSError:pass
        if time.monotonic()>deadline:raise RuntimeError('Server did not become ready')
        time.sleep(1)
    rows=[]
    try:
        math=call([{'role':'user','content':'Quanto fa 17 + 25? Rispondi solo con il numero.'}],64)
        rows.append({'name':'math','response':math});print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
        assert math['choices'][0]['message']['content'].strip()=='42','Arithmetic smoke failed'
        tools=[{'type':'function','function':{'name':'bash','description':'Run a shell command.',
            'parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}]
        tool=call([{'role':'user','content':'Chiama lo strumento bash con il comando esatto printf STRATA_OK. Non rispondere con testo.'}],256,tools)
        rows.append({'name':'tool_call','response':tool});print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
        msg=tool['choices'][0]['message'];calls=msg.get('tool_calls',[])
        assert len(calls)==1 and calls[0]['function']['name']=='bash','Tool name smoke failed'
        assert json.loads(calls[0]['function']['arguments'])['command']=='printf STRATA_OK','Tool arguments smoke failed'
        follow=call([{'role':'user','content':'Chiama bash: printf STRATA_OK.'},msg,
                     {'role':'tool','tool_call_id':calls[0]['id'],'content':'STRATA_OK'},
                     {'role':'user','content':'Quale stringa ha restituito lo strumento? Solo la stringa.'}],64,tools)
        rows.append({'name':'tool_followup','response':follow});print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
        assert follow['choices'][0]['message']['content'].strip()=='STRATA_OK','Tool result follow-up failed'
        for i in range(3):
            r=call([{'role':'user','content':f'Scrivi un esempio Python completo e commentato per leggere record FASTQ e calcolare il GC senza usare librerie esterne. Variante {i+1}.'}],256)
            rows.append({'name':f'decode_screen_{i}','response':r});print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
        context='\n'.join(f'read_{i:04d}\tACGTACGT\tIIIIIIII' for i in range(180))
        for i in range(2):
            r=call([{'role':'user','content':context+'\nQuesta è una tabella dimostrativa, non un FASTQ reale. Qual è la sequenza della prima riga? Rispondi solo ACGTACGT.'}],32)
            rows.append({'name':f'prefill_screen_{i}','response':r});print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
            assert r['choices'][0]['message']['content'].strip()=='ACGTACGT','Prefill semantic check failed'
        if a.context_check:
            import sys
            sys.path.insert(0,str(ROOT/'source/tools'))
            from strata_tokenizer import Tokenizer
            vocab=json.loads((ROOT/'pack-iq3/tokenizer/vocab.json').read_text(encoding='utf-8'))
            tokens=[None]*len(vocab)
            for text,index in vocab.items():tokens[index]=text
            tok=Tokenizer(tokens,(ROOT/'pack-iq3/tokenizer/merges.txt').read_text(encoding='utf-8').split('\n'),
                          json.loads((ROOT/'pack-iq3/tokenizer/token_type.json').read_text()))
            n=1800
            while True:
                data='PRIMA_STRINGA=ALFA_7391\n'+'\n'.join(f'record_{i:04d}\tACGTACGT\tIIIIIIII' for i in range(n))
                data+='\nFINE_STRINGA=OMEGA_5821\nRispondi solo con JSON contenente i valori di PRIMA_STRINGA e FINE_STRINGA nelle chiavi prima e ultima.'
                length=len(tok.encode(data))
                if 30000<=length<=31500:break
                n+=25 if length<30000 else -25
                if not 1000<=n<=2500:raise RuntimeError('Cannot construct bounded long prompt')
            r=call([{'role':'user','content':data}],64,timeout=600)
            rows.append({'name':'context32k','response':r});print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
            assert r['usage']['prompt_tokens']>=30000,'Long context request was too short'
            content=r['choices'][0]['message']['content'].strip()
            if content.startswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
            assert json.loads(content)=={'prima':'ALFA_7391','ultima':'OMEGA_5821'},'Long context retrieval failed'
        result={'passed':True,'rows':rows}
    except BaseException:
        result={'passed':False,'rows':rows};raise
    finally:
        (run/'probe.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        if a.stop:(run/'stop-requested').write_text('probe finished\n')
if __name__=='__main__':main()
