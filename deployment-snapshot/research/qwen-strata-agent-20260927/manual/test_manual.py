import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location("manual_server", Path(__file__).with_name("manual_server.py"))
manual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manual)


class ManualBounds(unittest.TestCase):
    def test_default_and_boundaries(self):
        for value in (1, 32, 1024, 1025, 8192, manual.MAX_OUTPUT):
            self.assertEqual(manual.request_cap({"model": manual.MODEL, "max_tokens": value}), value)
        self.assertEqual(manual.request_cap({"model": manual.MODEL}), manual.MAX_OUTPUT)

    def test_reject_output_types_and_unbounded(self):
        for value in (0, -1, manual.MAX_OUTPUT + 1, 2**40, None, True, "1024", 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                manual.request_cap({"model": manual.MODEL, "max_tokens": value})

    def test_alias_and_conflicting_limits(self):
        for request in ({}, [], {"model": "strata-eval"},
                        {"model": manual.MODEL, "max_tokens": 1024, "max_completion_tokens": 32}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                manual.request_cap(request)
        self.assertEqual(manual.request_cap({"model": manual.MODEL, "max_tokens": 32,
                                           "max_completion_tokens": 32}), 32)

    def test_exact_remaining_context_and_explicit_cap(self):
        self.assertEqual(manual.output_budget(manual.MAX_OUTPUT, 19231, 98304), 79065)
        self.assertEqual(manual.output_budget(manual.MAX_OUTPUT, 95000, 98304), 3296)
        self.assertEqual(manual.output_budget(32, 95000, 98304), 32)
        self.assertEqual(manual.output_budget(manual.MAX_OUTPUT, 1, 98304), 98295)
        self.assertEqual(manual.output_budget(manual.MAX_OUTPUT, 4087, 4096), 1)

    def test_exact_context_edges_invalid_counts(self):
        for prompt in (4088, 4096, 8192):
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, "Context exhausted"):
                manual.output_budget(manual.MAX_OUTPUT, prompt, 4096)
        for requested, prompt, context in ((True, 1, 4096), (0, 1, 4096), (manual.MAX_OUTPUT + 1, 1, 4096),
                                          (1, 0, 4096), (1, True, 4096), (1, 1.5, 4096),
                                          (1, 1, 0), (1, 1, 131072)):
            with self.subTest(values=(requested, prompt, context)), self.assertRaises(ValueError):
                manual.output_budget(requested, prompt, context)

    def test_prepare_once_after_image_expansion_without_trimming(self):
        ids = [248056] * 1024 + [1] * 3000
        svc = Mock(engine=Mock(max_context=4096))
        svc.prepare.return_value = (ids, True)
        convert = Mock(return_value=([{"role": "user"}], None, {"enable_thinking": True}))
        result = manual.prepare_output_request(svc, {"model": manual.MODEL, "stream": True}, convert)
        self.assertIs(result[0], ids)
        self.assertEqual(result[3:], (manual.MAX_OUTPUT, 64))
        svc.prepare.assert_called_once_with(convert.return_value[0], None, convert.return_value[2], 1)

    def test_prepare_rejects_oversized_budget_before_tokenization(self):
        svc, convert = Mock(), Mock()
        with self.assertRaises(ValueError):
            manual.prepare_output_request(svc, {"model": manual.MODEL, "max_tokens": 2**40}, convert)
        svc.prepare.assert_not_called()
        convert.assert_not_called()

    def test_large_nonstream_rejected_before_prepare(self):
        for stream in (None, False, "true", 1):
            svc, convert = Mock(), Mock()
            with self.subTest(stream=stream), self.assertRaises(ValueError):
                manual.prepare_output_request(svc, {"model": manual.MODEL, "stream": stream}, convert)
            svc.prepare.assert_not_called()

    def test_small_request_stream_type_is_boolean(self):
        for stream in ("false", 1, None):
            with self.subTest(stream=stream), self.assertRaisesRegex(ValueError, "stream must be a boolean"):
                manual.prepare_output_request(Mock(), {"model": manual.MODEL, "max_tokens": 32, "stream": stream}, Mock())


if __name__ == "__main__":
    unittest.main()
