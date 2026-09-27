"""Execute actual shell dispatch against isolated helpers; no real providers."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

STUB = r'''#!/usr/bin/python3
import json,os,sys
from pathlib import Path
a=sys.argv[1:]; scene=os.environ['SCENE']; name=Path(sys.argv[0]).name
with open(os.environ['CALLS'],'a') as f: f.write(json.dumps([name,*a])+'\n')
if name=='pi-qwen':
 print(json.dumps({'args':a,'base':os.environ.get('ISTA_STRATA_BASE_URL'),
                   'vision':os.environ.get('ISTA_STRATA_VISION')})); sys.exit(0)
if name=='powershell': sys.exit(3)
if a and a[0].endswith('pi_global_dispatch.mjs'):
 args=a[2:]; explicit=next((args[i+1] for i,x in enumerate(args[:-1]) if x=='--model'),None)
 explicit=explicit or next((x[8:] for x in args if x.startswith('--model=')),None)
 if a[1]=='legacy-selected': sys.exit(0 if explicit and 'flash-next-local' in explicit else 1)
 if scene=='timeout': sys.exit(124)
 if scene=='invalid': print('invalid\n1\n1'); sys.exit(0)
 if scene!='ready': sys.exit(2 if scene=='identity' else 1)
 disabled='--no-extensions' in args
 if disabled: sys.exit(1)
 other=explicit and 'flash-next-local' not in explicit
 other=other or ('--provider' in args and args[args.index('--provider')+1]!='local-qwen38')
 resumed=any(x in args for x in ['--continue','--resume','--session','--models'])
 force=not other and (not resumed or explicit)
 print('http://172.19.48.1:8038/v1\n1\n'+str(int(bool(force)))); sys.exit(0)
if '--is-selected' in a:
 selected=os.environ.get('SELECTED','none')
 sys.exit(0 if (selected=='thinkingcap' and 'thinkingcap' in a[0])
              or (selected=='gsq' and 'configure_pi_gsq.py' in a[0])
              or (selected=='sc5' and 'pi-sc5-config' in a[0]) else 1)
if '--refresh' in a: sys.exit(3)
sys.exit(1)
'''

class Wrapper(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pi-independent-')
        self.root = Path(self.temp.name)
        self.bin = self.root / '.local/bin'; self.bin.mkdir(parents=True)
        self.calls = self.root / 'calls.jsonl'
        for name in ['node','python3','pi-qwen','powershell']:
            p = self.bin / name; p.write_text(STUB); p.chmod(0o700)
        bridge = self.root / 'ensure_thinkingcap_wsl_bridge.sh'
        bridge.write_text('#!/bin/bash\nexit 1\n'); bridge.chmod(0o700)
        source = Path(__file__).with_name('pi-global-installed.sh').read_text()
        source = source.replace('/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-agent-20260927/manual',str(self.root))
        source = source.replace('/home/costapad/.nvm/versions/node/v22.23.0/bin/node',str(self.bin/'node'))
        source = source.replace('/mnt/c/MYPROJECT/TEST_QWEN/scripts',str(self.root))
        source = source.replace('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',str(self.bin/'powershell'))
        self.script=self.root/'pi'; self.script.write_text(source)
        self.env={**os.environ,'HOME':str(self.root),'PATH':f'{self.bin}:/usr/bin:/bin','CALLS':str(self.calls),
                  'ISTA_STRATA_BASE_URL':'stale','ISTA_STRATA_VISION':'1'}

    def tearDown(self): self.temp.cleanup()

    def run_pi(self,scene,args=(),selected='none'):
        result=subprocess.run(['bash',str(self.script),*args],env={**self.env,'SCENE':scene,'SELECTED':selected},
                              capture_output=True,text=True,timeout=4)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_all_unavailable_states_open_pi(self):
        for scene in ['refused','identity','timeout','invalid']:
            for args in [[],['--model','other/model']]:
                with self.subTest(scene=scene,args=args):
                    r=self.run_pi(scene,args)
                    self.assertEqual(r['args'],['--pi-default',*args]); self.assertIsNone(r['base'])

    def test_other_model_ready_switchable_overlay(self):
        args=['--model','other/model','--thinking','low','hello with spaces']
        r=self.run_pi('ready',args)
        self.assertEqual(r['args'],['--pi-default','-e',str(self.root/'pi_strata.mjs'),*args])
        self.assertEqual(r['base'],'http://172.19.48.1:8038/v1')
        self.assertEqual(r['vision'],'1')

    def test_default_ready_selects_strata(self):
        r=self.run_pi('ready',['--thinking','low'])
        self.assertEqual(r['args'][1:5],['--model','local-qwen38/qwen3.8-flash-next-local','--thinking','xhigh'])
        self.assertEqual(r['args'][-2:],['--thinking','low'])

    def test_resume_not_hijacked(self):
        r=self.run_pi('ready',['--continue'])
        self.assertNotIn('--model',r['args']); self.assertIn('-e',r['args'])

    def test_maintenance_bypasses_all_helpers(self):
        for args in [['--version'],['--list-models'],['install','pkg']]:
            self.calls.unlink(missing_ok=True)
            r=self.run_pi('identity',args)
            self.assertEqual(r['args'],['--pi-default',*args])
            self.assertEqual(len(self.calls.read_text().splitlines()),1)

    def test_provider_helper_failures_are_nonfatal(self):
        for selected in ['thinkingcap','gsq','sc5']:
            r=self.run_pi('identity',selected=selected)
            self.assertEqual(r['args'][0],'--pi-default')

    def test_only_selected_flash_starts_legacy_proxy(self):
        self.run_pi('identity',['--model','other/model'])
        calls=self.calls.read_text(); self.assertNotIn('powershell',calls)
        self.run_pi('identity',['--model','local-qwen38/qwen3.8-flash-next-local'])
        self.assertIn('powershell',self.calls.read_text())

    def test_no_extensions_does_not_select_strata(self):
        r=self.run_pi('ready',['--no-extensions'])
        self.assertNotIn('-e',r['args']); self.assertNotIn('--model',r['args'])

if __name__=='__main__': unittest.main()
