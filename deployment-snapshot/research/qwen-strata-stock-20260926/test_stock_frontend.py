"""Untouched frontend/tokenizer checks without model or GPU execution."""
import json
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
sys.path[:0] = [str(SOURCE), str(SOURCE / 'tools')]
from serve.frontend import ChatTemplate, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service, openai_chunks, openai_collect
from strata_tokenizer import Tokenizer


class FrontendTests(unittest.TestCase):
    def test_original_golden_templates(self):
        template = ChatTemplate(SOURCE / 'serve/chat_template.jinja')
        count = 0
        for case in json.loads((SOURCE / 'serve/chat_golden.json').read_text()):
            if case.get('error'):
                continue
            keywords = {key: case[key] for key in ('add_generation_prompt', 'enable_thinking', 'reasoning_effort') if key in case}
            self.assertEqual(template.render(case['messages'], tools=case.get('tools'), **keywords), case['rendered'])
            count += 1
        self.assertEqual(count, 10)

    def test_non_thinking_observer_collection(self):
        tokenizer = ByteTokenizer()
        engine = MockEngine(tokenizer, '42', max_context=4096)
        service = Service(engine, tokenizer, ChatTemplate(SOURCE / 'serve/chat_template.jinja'))
        request = {'messages': [{'role': 'user', 'content': 'Quanto fa 17 + 25?'}], 'reasoning_effort': 'none', 'max_tokens': 64}
        messages, tools, kwargs = openai_to_messages(request)
        ids, thinking = service.prepare(messages, tools, kwargs, 64)
        response = openai_collect(openai_chunks(service, request, ids, thinking, tools, 64, threading.Event()))
        self.assertEqual(response['choices'][0]['message']['content'], '42')
        self.assertEqual(response['usage']['prompt_tokens'], len(ids))

    def test_exact_installed_tokenizer_roundtrip(self):
        path = ROOT.parent / 'qwen-strata-20260926/pack-iq3/tokenizer'
        vocab = json.loads((path / 'vocab.json').read_text(encoding='utf-8'))
        tokens = [None] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
        tokenizer = Tokenizer(tokens, (path / 'merges.txt').read_text(encoding='utf-8').split('\n'),
                              json.loads((path / 'token_type.json').read_text()))
        for text in ('ACGT\nIIII', 'Bioinformatica: qualità, αβ e 🧬.', '\tA\r\nG  C', 'e\u0301\n'):
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
