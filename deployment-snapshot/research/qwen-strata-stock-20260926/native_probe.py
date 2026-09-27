"""External observer using untouched upstream frontend and native engine.

No HTTP listener, tool execution, engine patch, or production configuration.
The parent must attach the Windows kill-on-close Job before releasing us.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mode', choices=['load', 'smoke', 'bench'], required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / 'runs'):
        raise ValueError('Run path outside isolated campaign')
    deadline = time.monotonic() + 30
    while not (run / 'job-attached').exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('Guardian did not attach Job')
        time.sleep(0.05)
    source = ROOT / 'source'
    sys.path[:0] = [str(source), str(source / 'tools')]
    from serve.server import StrataEngine, Service, openai_chunks, openai_collect
    from serve.frontend import ChatTemplate, openai_to_messages
    from strata_tokenizer import Tokenizer
    config = json.loads((run / 'config.json').read_text())
    tok_path = Path(config['tokenizer'])
    vocab = json.loads((tok_path / 'vocab.json').read_text(encoding='utf-8'))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = Tokenizer(tokens, (tok_path / 'merges.txt').read_text(encoding='utf-8').split('\n'),
                    json.loads((tok_path / 'token_type.json').read_text()))
    rows = []
    engine = None
    result = {'passed': False, 'mode': args.mode, 'rows': rows}

    def save():
        (run / 'probe.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    def phase(name):
        (run / 'phase.json').write_text(json.dumps({'phase': name, 'time': time.time()}))
        print(name, flush=True)

    try:
        phase('loading')
        started = time.monotonic()
        engine = StrataEngine(config['exe'], config['args'], cwd=config['cwd'], log=config['log'])
        result.update(load_passed=True, load_wall_seconds=time.monotonic() - started,
                      native_pid=engine.proc.pid, context=engine.max_context)
        save()
        (run / 'ready.json').write_text(json.dumps({'native_pid': engine.proc.pid, 'context': engine.max_context}))
        phase('waiting-ready-resource-check')
        deadline = time.monotonic() + 30
        while not (run / 'ready-accepted.json').exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Guardian did not approve actual ready-state resources')
            time.sleep(0.05)
        phase('ready')
        if args.mode == 'load':
            result['passed'] = True
            return
        svc = Service(engine, tok, ChatTemplate(tok_path / 'chat_template.jinja'), model_name=config['model_name'])

        def call(name, messages, maximum=64, tools=None):
            if not 1 <= maximum <= 256:
                raise ValueError('Probe output bound exceeded')
            req = {'model': config['model_name'], 'messages': messages, 'reasoning_effort': 'none',
                   'max_tokens': maximum, 'stream': False}
            if tools:
                req['tools'] = tools
            normalized, flat_tools, kwargs = openai_to_messages(req)
            ids, thinking = svc.prepare(normalized, flat_tools, kwargs, maximum)
            (run / (name + '-request.json')).write_text(json.dumps(req, ensure_ascii=False), encoding='utf-8')
            phase(name)
            started = time.monotonic()
            response = openai_collect(openai_chunks(svc, req, ids, thinking, flat_tools, maximum, threading.Event()))
            timing = dict(engine.last)
            if not timing or timing.get('prompt_tokens') != len(ids):
                raise RuntimeError('Native DONE timing missing or prompt count mismatch')
            row = {'name': name, 'wall_seconds': time.monotonic() - started, 'response': response,
                   'native_timings': timing,
                   'prompt_tok_s': timing['prompt_tokens'] * 1000 / timing['prompt_ms'] if timing['prompt_ms'] else None,
                   'decode_tok_s': timing['generated'] * 1000 / timing['decode_ms'] if timing['decode_ms'] else None}
            rows.append(row)
            save()
            print(json.dumps({k: row[k] for k in ('name', 'prompt_tok_s', 'decode_tok_s', 'native_timings')}) , flush=True)
            return response['choices'][0]['message']

        math = call('math', [{'role': 'user', 'content': 'Quanto fa 17 + 25? Rispondi solo con il numero.'}])
        if (math.get('content') or '').strip() != '42':
            raise AssertionError('Arithmetic semantic smoke failed')
        tools = [{'type': 'function', 'function': {'name': 'bash', 'description': 'Run a shell command.',
                 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
        tool_message = call('tool_call', [{'role': 'user', 'content':
                            'Chiama lo strumento bash con il comando esatto printf STRATA_OK. Non rispondere con testo.'}], 256, tools)
        calls = tool_message.get('tool_calls', [])
        if len(calls) != 1 or calls[0]['function']['name'] != 'bash' or json.loads(calls[0]['function']['arguments']) != {'command': 'printf STRATA_OK'}:
            raise AssertionError('Tool-call semantic smoke failed; no command was executed')
        follow = call('tool_followup', [{'role': 'user', 'content': 'Chiama bash: printf STRATA_OK.'}, tool_message,
                       {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': 'STRATA_OK'},
                       {'role': 'user', 'content': 'Quale stringa ha restituito lo strumento? Solo la stringa.'}], 64, tools)
        if (follow.get('content') or '').strip() != 'STRATA_OK':
            raise AssertionError('Tool-result follow-up smoke failed')
        result['semantic_smoke_passed'] = True
        save()
        if args.mode == 'bench':
            for index in range(3):
                call(f'decode_screen_{index}', [{'role': 'user', 'content':
                    f'Scrivi un esempio Python completo e commentato per leggere record FASTQ e calcolare il GC senza usare librerie esterne. Variante {index + 1}.'}], 256)
            context = '\n'.join(f'read_{index:04d}\tACGTACGT\tIIIIIIII' for index in range(180))
            for index in range(2):
                answer = call(f'prefill_screen_{index}', [{'role': 'user', 'content': context +
                    '\nQuesta è una tabella dimostrativa, non un FASTQ reale. Qual è la sequenza della prima riga? Rispondi solo ACGTACGT.'}], 32)
                if (answer.get('content') or '').strip() != 'ACGTACGT':
                    raise AssertionError('Prefill semantic smoke failed')
        result['passed'] = True
    except BaseException as error:
        result['failure'] = str(error)
        raise
    finally:
        phase('closing')
        if engine is not None:
            engine.close()
            result['native_exit_code'] = engine.proc.returncode
            if engine.proc.returncode != 0:
                result['passed'] = False
        save()


if __name__ == '__main__':
    main()
