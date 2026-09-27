"""Bounded Windows guardian for the isolated Strata runtime. Never trims other processes."""
from __future__ import annotations
import argparse
import ctypes as C
from ctypes import wintypes as W
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
GiB=1<<30
CUDA=Path(r'E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit\bin')
k32=C.WinDLL('kernel32',use_last_error=True) if os.name=='nt' else None
psapi=C.WinDLL('psapi',use_last_error=True) if os.name=='nt' else None
class Memory(C.Structure):
    _fields_=[('length',W.DWORD),('load',W.DWORD)]+[(n,C.c_ulonglong) for n in
        ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
class Performance(C.Structure):
    _fields_=[('cb',W.DWORD)]+[(n,C.c_size_t) for n in
        ('commit','limit','peak','physical','available','cache','kernel','paged','nonpaged','page_size')]+[
        ('handles',W.DWORD),('processes',W.DWORD),('threads',W.DWORD)]
class Basic(C.Structure):
    _fields_=[('process_time',C.c_longlong),('job_time',C.c_longlong),('flags',W.DWORD),
        ('min_ws',C.c_size_t),('max_ws',C.c_size_t),('active',W.DWORD),('affinity',C.c_size_t),
        ('priority',W.DWORD),('scheduling',W.DWORD)]
class IO(C.Structure):
    _fields_=[(n,C.c_ulonglong) for n in ('read_ops','write_ops','other_ops','read','write','other')]
class Limits(C.Structure):
    _fields_=[('basic',Basic),('io',IO),('process_limit',C.c_size_t),('job_limit',C.c_size_t),
        ('peak_process',C.c_size_t),('peak_job',C.c_size_t)]
def wincheck(ok):
    if not ok: raise C.WinError(C.get_last_error())
def counters():
    m=Memory();m.length=C.sizeof(m);wincheck(k32.GlobalMemoryStatusEx(C.byref(m)))
    p=Performance();p.cb=C.sizeof(p);wincheck(psapi.GetPerformanceInfo(C.byref(p),p.cb))
    return {'ram_free':int(m.available),'commit_free':int((p.limit-p.commit)*p.page_size),
            'commit_used':int(p.commit*p.page_size),'commit_limit':int(p.limit*p.page_size),
            'system_cache':int(p.cache*p.page_size)}
def gpu_counters():
    cmd=['nvidia-smi','--query-gpu=index,memory.free,temperature.gpu','--format=csv,noheader,nounits']
    text=subprocess.check_output(cmd,text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW)
    rows=[[int(x.strip()) for x in row.split(',')] for row in text.strip().splitlines()]
    if len(rows)!=2 or [r[0] for r in rows]!=[0,1]: raise RuntimeError('Expected exactly the two inventoried GPUs')
    return rows
def resources_ok(c,g,start=False,host_bytes=0):
    ram=(host_bytes+4*GiB+12*GiB) if start else 12*GiB
    # Estimate, not a measurement: GPU backing + dense/MTP/cache/IO + driver headroom.
    # The first load measured ~16 GiB extra Job commitment before MTP/cache.
    commit=(host_bytes+24*GiB+16*GiB) if start else 16*GiB
    if c['ram_free']<ram: raise RuntimeError(f"Available RAM {c['ram_free']/GiB:.2f} GiB < {ram/GiB:.2f} GiB")
    if c['commit_free']<commit: raise RuntimeError(f"Available commit {c['commit_free']/GiB:.2f} GiB < {commit/GiB:.2f} GiB")
    for i,free,temp in g:
        if free<(14336 if start else 2048): raise RuntimeError(f'GPU{i}: only {free} MiB free')
        if temp>=80: raise RuntimeError(f'GPU{i}: temperature {temp} C')
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return h.hexdigest()
def identity(manifest):
    if manifest.get('schema')!='antirez-strata/v1':raise RuntimeError('Wrong manifest')
    for row in manifest['runtime_files']:
        p=ROOT/row['path']
        if not p.resolve().is_relative_to(ROOT):raise RuntimeError('Runtime path escapes isolated directory')
        if digest(p)!=row['sha256']:raise RuntimeError(f'Runtime identity mismatch: {p.name}')
    proof=json.loads((ROOT/'model-proof.json').read_text())
    for row in proof['files']:
        p=Path(row['path']);st=p.stat()
        if st.st_size!=row['bytes'] or st.st_mtime_ns!=row['mtime_ns']:
            raise RuntimeError('Model changed since SHA256 verification')
    gates=json.loads((ROOT/'gates.json').read_text())
    if gates.get('passed') is not True or gates['strata_sha256']!=digest(ROOT/'build-native/strata.exe'):
        raise RuntimeError('Synthetic gates absent/failed/stale')
def make_job():
    k32.CreateJobObjectW.argtypes=[C.c_void_p,W.LPCWSTR];k32.CreateJobObjectW.restype=W.HANDLE
    k32.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD]
    k32.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE]
    k32.IsProcessInJob.argtypes=[W.HANDLE,W.HANDLE,C.POINTER(W.BOOL)]
    k32.CloseHandle.argtypes=[W.HANDLE]
    job=k32.CreateJobObjectW(None,None);wincheck(job)
    lim=Limits();lim.basic.flags=0x2000|0x200|0x10 # kill-on-close, job memory, affinity
    lim.basic.affinity=0xFFF;lim.job_limit=48*GiB
    wincheck(k32.SetInformationJobObject(job,9,C.byref(lim),C.sizeof(lim)))
    return job
def job_counters(job):
    k32.QueryInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD,C.c_void_p]
    lim=Limits();wincheck(k32.QueryInformationJobObject(job,9,C.byref(lim),C.sizeof(lim),None))
    return {'peak_process_bytes':int(lim.peak_process),'peak_job_bytes':int(lim.peak_job)}
def bootstrap(marker,config):
    deadline=time.monotonic()+30
    while not marker.exists():
        if time.monotonic()>deadline:raise RuntimeError('Guardian did not attach job')
        time.sleep(.05)
    os.chdir(ROOT/'source');sys.path.insert(0,str(ROOT/'source'))
    sys.argv=['serve.server','--engine','strata','--config',str(config),'--host','127.0.0.1','--port','8035']
    from serve.server import main
    main()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['preflight','load','smoke','server','bootstrap'])
    ap.add_argument('--context',type=int,choices=[4096,32768,98304],default=4096)
    ap.add_argument('--tag',default=time.strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--marker',type=Path);ap.add_argument('--config',type=Path)
    ap.add_argument('--timeout',type=int,default=1800)
    a=ap.parse_args()
    if a.mode=='bootstrap':return bootstrap(a.marker,a.config)
    if not k32:raise RuntimeError('This guardian is Windows-only')
    if not 1<=a.timeout<=86400:raise ValueError('Timeout out of range')
    manifest=json.loads((ROOT/'runtime-manifest.json').read_text())
    identity(manifest)
    if a.mode=='server':
        admission_path=ROOT/f'admission-{a.context}.json'
        if not admission_path.exists():raise RuntimeError(f'Contesto {a.context} non ancora validato: avvio rifiutato, launcher ordinario invariato')
        admission=json.loads(admission_path.read_text())
        if not admission.get('passed') or admission['strata_sha256']!=manifest['strata_sha256'] or admission.get('runtime_manifest_sha256')!=digest(ROOT/'runtime-manifest.json'):
            raise RuntimeError('This context tier is not admitted; ordinary server remains unchanged')
    c=counters();g=gpu_counters();resources_ok(c,g,True,manifest['host_experts_bytes'])
    print(json.dumps({'preflight':'passed','ram_free_gib':c['ram_free']/GiB,'commit_free_gib':c['commit_free']/GiB,
          'host_experts_gib':manifest['host_experts_bytes']/GiB,'gpu1_experts_gib':manifest['gpu1_experts_bytes']/GiB,'gpu':g}),flush=True)
    if a.mode=='preflight':return
    if not a.tag or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for ch in a.tag):
        raise ValueError('Invalid run tag')
    run=ROOT/'runs'/a.tag;run.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'server-config.json').read_text())
    args=config['args'];args[args.index('--max-context')+1]=str(a.context)
    config['log']=str(run/'engine.log');config_path=run/'config.json';config_path.write_text(json.dumps(config,indent=2))
    marker=run/'job-attached';job=None;proc=None;mutex=None;owned=False
    k32.CreateMutexW.argtypes=[C.c_void_p,W.BOOL,W.LPCWSTR];k32.CreateMutexW.restype=W.HANDLE
    k32.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD];k32.ReleaseMutex.argtypes=[W.HANDLE]
    try:
        mutex=k32.CreateMutexW(None,False,'Local\\QwenBioagentGpuExclusive');wincheck(mutex)
        wait=k32.WaitForSingleObject(mutex,0)
        if wait not in (0,0x80):raise RuntimeError('Another guarded model owns the GPUs')
        owned=True
        with socket.socket() as probe:probe.bind(('127.0.0.1',8035))
        job=make_job()
        env=dict(os.environ);env['PATH']=str(CUDA)+os.pathsep+env['PATH']
        env['STRATA_FORCE_AVX2']='1';env.pop('CUDA_VISIBLE_DEVICES',None)
        env['STRATA_API_KEY']=Path(config['api_key_file']).read_text().strip()
        if not env['STRATA_API_KEY'] or '\n' in env['STRATA_API_KEY']:raise RuntimeError('Invalid local API key file')
        with open(run/'server.log','w',encoding='utf-8') as log,open(run/'telemetry.jsonl','w') as telemetry:
            proc=subprocess.Popen([sys.executable,str(__file__),'bootstrap','--marker',str(marker),'--config',str(config_path)],
                stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            wincheck(k32.AssignProcessToJobObject(job,int(proc._handle)))
            member=W.BOOL();wincheck(k32.IsProcessInJob(int(proc._handle),job,C.byref(member)))
            if not member.value:raise RuntimeError('Job assignment verification failed')
            marker.write_text('owned\n');start=time.monotonic();ready=False;result={}
            while proc.poll() is None:
                c=counters();g=gpu_counters()
                telemetry.write(json.dumps({'elapsed':time.monotonic()-start,**c,'gpu':g,**job_counters(job)})+'\n');telemetry.flush()
                resources_ok(c,g)
                if time.monotonic()-start>a.timeout:raise RuntimeError('Bounded run timeout')
                if not ready:
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8035/health',timeout=1) as r:
                            ready=json.load(r).get('status')=='ok'
                    except OSError:pass
                    if ready:
                        print(f'Strata pronto su http://127.0.0.1:8035/v1 (context {a.context})',flush=True)
                        result={'load_passed':True,'context':a.context,'strata_sha256':manifest['strata_sha256'],
                                'runtime_manifest_sha256':digest(ROOT/'runtime-manifest.json')}
                        if a.mode=='load':break
                        if a.mode=='smoke':
                            req={'model':'qwen38-strata-dual','messages':[{'role':'user','content':'Quanto fa 17 + 25? Rispondi solo con il numero.'}],
                                 'max_tokens':64,'reasoning_effort':'none','stream':False}
                            data=json.dumps(req).encode()
                            # Inference must be probed by a separate harness so the guardian never stops sampling.
                            result['smoke_request']=req
                            print('Smoke harness must query the live endpoint; use probe.py.',flush=True)
                if (run/'stop-requested').exists():break
                time.sleep(1)
            if proc.poll() is not None and proc.returncode:raise RuntimeError(f'Strata child exited {proc.returncode}; see {run}')
            (run/'result.json').write_text(json.dumps(result,indent=2))
    except BaseException as e:
        (run/'failure.json').write_text(json.dumps({'passed':False,'reason':str(e),'context':a.context,
            'strata_sha256':manifest['strata_sha256'],'runtime_manifest_sha256':digest(ROOT/'runtime-manifest.json')},indent=2))
        raise
    finally:
        if job:k32.CloseHandle(job)
        if proc:
            if proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill()
        if owned:k32.ReleaseMutex(mutex)
        if mutex:k32.CloseHandle(mutex)
if __name__=='__main__':
    try:main()
    except Exception as e:print(f'STRATA STOP: {e}',file=sys.stderr,flush=True);sys.exit(1)
