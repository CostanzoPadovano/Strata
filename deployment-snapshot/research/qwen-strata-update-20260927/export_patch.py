"""Generate a reproducible source-only artifact against the frozen, working F107 snapshot."""
from pathlib import Path
import difflib
import hashlib
import json

root=Path(__file__).resolve().parent
base=root.parent/'qwen-strata-vision-20260927/source'
names=['CMakeLists.txt','include/strata/core/mtp.hpp','include/strata/core/conversation_cache.hpp',
       'include/strata/kernels/cpu/pool.hpp','src/kernels/cpu/pool.cpp','src/program/generate.cpp','serve/server.py']
parts=[]
for name in names:
    old=base/name; new=root/'source'/name
    before=old.read_text(encoding='utf-8').splitlines(keepends=True) if old.exists() else []
    after=new.read_text(encoding='utf-8').splitlines(keepends=True)
    parts.extend(difflib.unified_diff(before,after,fromfile='a/'+name if old.exists() else '/dev/null',tofile='b/'+name))
target=root/'cache-idle-source.patch'
target.write_text(''.join(parts),encoding='utf-8',newline='\n')
print(json.dumps({'patch':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                  'base':'qwen-strata-vision-20260927/source F107','files':len(names)}))
