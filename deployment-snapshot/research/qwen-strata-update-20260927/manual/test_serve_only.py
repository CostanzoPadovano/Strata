"""No native model: exact derivative + actual startup against failing GEN stubs."""
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import serve_only_server as observer
from serve_only_derivation import derive


class Startup(unittest.TestCase):
    def test_exact_delta(self):
        root=Path(__file__).resolve().parent
        self.assertEqual((root/'serve_only_server.py').read_text(),derive((root/'manual_server.py').read_text()))
        self.assertNotIn('Fresh semantic smoke failed',(root/'serve_only_server.py').read_text())

    def test_real_native_timing_display(self):
        self.assertEqual(observer.timing_summary({'prompt_tokens':1000,'generated':100,
            'prompt_ms':5000,'decode_ms':2000,'finish':'stop'}),
            '[timings] prefill 199.80 tok/s (999 nuovi, 0 riusati, 1000 contesto) | decode 50.00 tok/s (100 token) | stop')
        self.assertIn('n/a',observer.timing_summary({'prompt_tokens':0,'generated':0,
            'prompt_ms':0,'decode_ms':0,'finish':'stop'}))

    def test_reuse_not_counted_as_prefill_throughput(self):
        row={'prompt_tokens':20001,'prompt_reused':19000,'generated':40,'prompt_ms':2000,'decode_ms':1000,'finish':'stop'}
        self.assertEqual(observer.timing_summary(row),
            '[timings] prefill 500.00 tok/s (1000 nuovi, 19000 riusati, 20001 contesto) | decode 40.00 tok/s (40 token) | stop')

    def test_actual_main_never_generates_or_encodes(self):
        class Process:
            pid=123; returncode=None
            def poll(self): return self.returncode
            def terminate(self): self.returncode=0
            kill=terminate
            def wait(self,timeout=None): self.returncode=0; return 0
        class Engine:
            def __init__(self,*a,**kw):
                self.proc=Process(); self.max_context=98304; self.last={}; self._pump()
            def _pump(self): pass
            def generate(self,*a,**kw): raise AssertionError('Startup sent GEN')
            def close(self): self.proc.terminate()
        class Service:
            def __init__(self,*a,**kw): pass
            def prepare(self,*a,**kw): raise AssertionError('Startup prepared a prompt/image')
        class Server:
            def __init__(self,*a,**kw): pass
            def serve_forever(self): pass
            def shutdown(self): pass
            def server_close(self): pass
        class Vision:
            proc=None; last_exit_code=None; parking=False; failed=False
            def __init__(self,*a,**kw): self.stats={'starts':0,'encoded':0}
            def close(self): pass
        def unexpected(*a,**kw): raise AssertionError('Startup called inference/frontend conversion')
        engine_module=types.ModuleType('serve.server')
        engine_module.StrataEngine=Engine; engine_module.Service=Service; engine_module.Server=Server
        engine_module.make_handler=lambda svc: object
        engine_module.openai_collect=unexpected; engine_module.openai_chunks=unexpected
        package=types.ModuleType('serve');package.server=engine_module
        frontend=types.ModuleType('serve.frontend');frontend.ChatTemplate=lambda *a:None;frontend.openai_to_messages=unexpected
        tokenizer=types.ModuleType('strata_tokenizer');tokenizer.Tokenizer=lambda *a:None
        vision=types.ModuleType('bounded_vision');vision.BoundedVision=Vision;vision.MAX_BODY=16*1024*1024
        vision.validate_sve=unexpected;vision.vision_service_class=lambda server:Service
        with tempfile.TemporaryDirectory(prefix='serve-only-test-') as temporary:
            base=Path(temporary);run=base/'runs/run';run.mkdir(parents=True)
            tok=base/'tokenizer';tok.mkdir()
            for name,data in [('vocab.json','{"a":0}'),('merges.txt',''),('token_type.json','{}')]:
                (tok/name).write_text(data)
            key=base/'key';key.write_text('isolated-test-key')
            (run/'config.json').write_text(json.dumps({'args':['--max-context','98304'],
                'exe':'fake-never-executed','cwd':str(base),'log':str(run/'engine.log'),'tokenizer':str(tok),'vision':{}}))
            for name in ['job-attached','ready-accepted.json','stop-server']: (run/name).touch()
            with patch.dict(sys.modules,{'serve':package,'serve.server':engine_module,'serve.frontend':frontend,
                    'strata_tokenizer':tokenizer,'bounded_vision':vision}),\
                 patch.object(observer,'VISION_CAMPAIGN',base),patch.object(observer,'KEYFILE',key),\
                 patch.object(observer.subprocess,'Popen',return_value=Process()),\
                 patch.object(sys,'argv',['observer','--run',str(run),'--mode','bench']),\
                 patch.object(observer.time,'sleep'),patch('sys.stdout',new_callable=io.StringIO):
                observer.main()
            result=json.loads((run/'probe.json').read_text())
            self.assertTrue(result['api_ready']);self.assertTrue(result['passed'])
            self.assertEqual(result['startup_self_test'],'disabled')
            self.assertEqual(result['vision_stats']['starts'],0)
            self.assertFalse((run/'timings.jsonl').exists())
            self.assertNotIn('semantic_smoke_passed',result)


if __name__=='__main__': unittest.main()
