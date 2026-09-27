"""SSE header/initial-chunk/active disconnect and owned SVE cleanup, no model."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import uuid

from manual_server import serve_budgeted_openai, cleanup_prepared_embedding, MODEL, MAX_OUTPUT


class OutputLifecycle(unittest.TestCase):
    def setUp(self):
        self.run = Path(__file__).resolve().parent / ("test-output-" + uuid.uuid4().hex)
        self.run.mkdir()
        self.embedding = self.run / "req-test.sve"
        self.embedding.write_bytes(b"test fixture, never sent to native")
        self.svc = SimpleNamespace(engine=SimpleNamespace(max_context=4096), vision=SimpleNamespace(dir=self.run),
                                   embeddings=SimpleNamespace(path=self.embedding), prepare=Mock(return_value=([1] * 32, True)))
        self.req = {"model": MODEL, "stream": True}
        self.handler = SimpleNamespace(_sse=Mock(), _json=Mock(), wfile=io.BytesIO())
        self.events, self.cancel = [], None

        def chunks(*args):
            self.cancel = args[-1]
            try:
                self.events.append("role")
                yield {"choices": [{"delta": {"role": "assistant"}}]}
                self.events.append("native")
                yield {"choices": [{"delta": {"content": "OK"}}]}
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            finally:
                self.events.append("closed")

        self.server = SimpleNamespace(openai_to_messages=lambda req: ([], None, {}), openai_chunks=chunks,
                                      openai_collect=lambda chunks: list(chunks))

    def tearDown(self):
        # Exact generated paths only; never recursive/broad cleanup.
        for name in ("req-test.sve", "requests.jsonl", "wrong.sve"):
            (self.run / name).unlink(missing_ok=True)
        self.run.rmdir()

    def test_normal_stream_done_budget_log_and_cleanup(self):
        serve_budgeted_openai(self.handler, self.server, self.svc, self.req, self.run)
        self.assertTrue(self.handler.wfile.getvalue().endswith(b"data: [DONE]\n\n"))
        self.assertFalse(self.embedding.exists())
        self.assertIsNone(self.svc.embeddings.path)
        self.assertFalse(self.cancel.is_set())
        row = json.loads((self.run / "requests.jsonl").read_text())
        self.assertEqual(row["requested_max_tokens"], MAX_OUTPUT)
        self.assertEqual(row["max_tokens"], 4056)
        self.assertEqual(row["budget_limited_by"], "context")

    def test_header_disconnect_cleans_unstarted_generator(self):
        self.handler._sse.side_effect = BrokenPipeError("synthetic disconnect")
        serve_budgeted_openai(self.handler, self.server, self.svc, self.req, self.run)
        self.assertFalse(self.embedding.exists())
        self.assertEqual(self.events, [])

    def test_first_role_disconnect_cleans_before_native(self):
        self.handler.wfile = Mock()
        self.handler.wfile.write.side_effect = BrokenPipeError("synthetic disconnect")
        serve_budgeted_openai(self.handler, self.server, self.svc, self.req, self.run)
        self.assertTrue(self.cancel.is_set())
        self.assertEqual(self.events, ["role", "closed"])
        self.assertFalse(self.embedding.exists())

    def test_active_stream_disconnect_cancels_and_closes(self):
        self.handler.wfile = Mock()
        self.handler.wfile.write.side_effect = [1, BrokenPipeError("synthetic disconnect")]
        serve_budgeted_openai(self.handler, self.server, self.svc, self.req, self.run)
        self.assertTrue(self.cancel.is_set())
        self.assertEqual(self.events, ["role", "native", "closed"])
        self.assertFalse(self.embedding.exists())

    def test_log_failure_cleans_before_generation(self):
        with self.assertRaises(OSError):
            serve_budgeted_openai(self.handler, self.server, self.svc, self.req, self.run / "not-created")
        self.assertEqual(self.events, [])
        self.assertFalse(self.embedding.exists())

    def test_small_nonstream_retained_large_nonstream_refused(self):
        serve_budgeted_openai(self.handler, self.server, self.svc,
                              {"model": MODEL, "max_tokens": 32}, self.run)
        self.handler._json.assert_called_once()
        self.assertFalse(self.embedding.exists())
        self.svc.prepare.reset_mock()
        with self.assertRaises(ValueError):
            serve_budgeted_openai(self.handler, self.server, self.svc, {"model": MODEL}, self.run)
        self.svc.prepare.assert_not_called()

    def test_unowned_path_never_deleted(self):
        bad = self.run / "wrong.sve"
        bad.write_bytes(b"not owned")
        self.svc.embeddings.path = bad
        with self.assertRaises(ValueError):
            cleanup_prepared_embedding(self.svc)
        self.assertTrue(bad.exists())


if __name__ == "__main__":
    unittest.main()
