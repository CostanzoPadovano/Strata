"""Qualify only the startup/log derivative offline; NEVER load a model."""
import json
from pathlib import Path
import subprocess
import sys
import time
from vision_campaign import ROOT,MANUAL,bound,digest
from serve_only_derivation import derive

if (MANUAL/'serve_only_server.py').read_text()!=derive((MANUAL/'manual_server.py').read_text()):
    raise RuntimeError('Unexpected observer delta')
log=MANUAL/f'serve-only-tests-{time.time_ns()}.log'
with log.open('x',encoding='utf-8') as output:
    result=subprocess.run([sys.executable,'-m','unittest','test_serve_only','-v'],cwd=MANUAL,
                          stdout=output,stderr=subprocess.STDOUT,timeout=30)
if result.returncode: raise RuntimeError('Offline startup checks failed: '+str(log))
files=['serve_only_server.py','serve_only_guard.py','serve_only_derivation.py','test_serve_only.py','prepare_serve_only.py',
       'vision_user_launcher.py','manual_guard.py']
proof={'passed':True,'qualification':'offline-startup-only-derivative','model_started':False,
       'parent_source_gates_sha256':digest(ROOT/'source-gates.json'),
       'bound_files':[bound(MANUAL/name) for name in files]+[bound(log)]}
(MANUAL/'serve-only-gates.json').write_text(json.dumps(proof,indent=2))
print(json.dumps({'offline_passed':True,'model_started':False,'log':str(log)}))
