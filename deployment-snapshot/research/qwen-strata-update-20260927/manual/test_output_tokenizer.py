"""Actual installed template/tokenizer accounting; no engine/model is started."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

MANUAL = Path(__file__).resolve().parent
SOURCE = MANUAL.parent / "source"
sys.path[:0] = [str(SOURCE), str(SOURCE / "tools")]
from manual_server import MAX_OUTPUT, prepare_output_request
from serve.server import Service
from serve.frontend import ChatTemplate, openai_to_messages
from strata_tokenizer import Tokenizer


class ActualTokenBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((MANUAL.parent / "configs/tier98304.json").read_text())
        directory = Path(config["tokenizer"])
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
        cls.tok = Tokenizer(tokens, (directory / "merges.txt").read_text(encoding="utf-8").split("\n"),
                            json.loads((directory / "token_type.json").read_text()))
        cls.template = ChatTemplate(directory / "chat_template.jinja")

    def service(self, context):
        return Service(SimpleNamespace(max_context=context), self.tok, self.template)

    def test_real_template_budget_above1024_and_no_input_mutation(self):
        req = {"model": "qwen3.8-flash-next-local", "stream": True, "reasoning_effort": "xhigh",
               "messages": [{"role": "user", "content": "x " * 1200}]}
        original = json.dumps(req, sort_keys=True)
        ids, thinking, tools, requested, effective = prepare_output_request(self.service(4096), req, openai_to_messages)
        self.assertTrue(thinking)
        self.assertGreater(effective, 1024)
        self.assertEqual(requested, MAX_OUTPUT)
        self.assertEqual(len(ids) + effective + 8, 4096)
        self.assertEqual(json.dumps(req, sort_keys=True), original)
        self.assertEqual(ids, self.tok.encode(self.template.render(*openai_to_messages(req)[:1],
                         tools=tools, **openai_to_messages(req)[2]), parse_special=True))

    def test_real_long_prompt_remaining_budget_without_native_prefill(self):
        req = {"model": "qwen3.8-flash-next-local", "stream": True, "reasoning_effort": "minimal",
               "messages": [{"role": "user", "content": "x " * 95000}]}
        ids, thinking, _, _, effective = prepare_output_request(self.service(98304), req, openai_to_messages)
        self.assertGreater(len(ids), 94000)
        self.assertLess(effective, 4304)
        self.assertEqual(len(ids) + effective + 8, 98304)
        self.assertFalse(thinking)

    def test_real_template_overflow_rejected_not_silently_cut(self):
        req = {"model": "qwen3.8-flash-next-local", "stream": True, "messages": [{"role": "user", "content": "x " * 4200}]}
        with self.assertRaises(ValueError):
            prepare_output_request(self.service(4096), req, openai_to_messages)


if __name__ == "__main__":
    unittest.main()
