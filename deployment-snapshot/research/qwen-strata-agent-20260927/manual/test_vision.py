import base64
import io
from pathlib import Path
import struct
import uuid
import unittest
from unittest.mock import Mock, patch
from collections import OrderedDict

from PIL import Image
from bounded_vision import (MAGIC, WIDTH, inline_image, normalize, validate_sources, validate_sve,
                            vision_service_class, BoundedVision)


def picture(mode="RGB", size=(32, 32)):
    out = io.BytesIO()
    Image.new(mode, size, "red").save(out, "PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


class VisionBounds(unittest.TestCase):
    def test_lazy_construction_close_without_native_start(self):
        tmp = Path(__file__).resolve().parent / ("test-vision-" + uuid.uuid4().hex)
        with patch.object(BoundedVision, "_start") as start:
            vision = BoundedVision({"gpu": False, "max_tokens": 1024, "threads": 8}, tmp, io.StringIO())
            self.assertIsNone(vision.proc)
            self.assertEqual(vision.stats["starts"], 0)
            start.assert_not_called()
            vision.close()
        self.assertFalse(tmp.exists())

    def test_cold_memory_refusal_is_not_latched_and_cache_remains(self):
        import hashlib
        from vision_campaign import admitted
        tmp = Path(__file__).resolve().parent / ("test-vision-" + uuid.uuid4().hex)
        vision = BoundedVision({"gpu": False, "max_tokens": 1024, "threads": 8}, tmp, io.StringIO())
        try:
            with patch.object(admitted.shared, "counters", return_value={"commit_free": 5 << 30, "ram_free": 12 << 30}), \
                    patch("bounded_vision.subprocess.Popen") as launch:
                with self.assertRaisesRegex(ValueError, "cached images/text remain usable"):
                    vision.encode(picture())
                launch.assert_not_called()
                self.assertFalse(vision.failed)
                self.assertIsNone(vision.proc)
                key = hashlib.sha256(normalize(inline_image(picture()))).hexdigest()
                vision.cache[key] = (tmp / "cached.sve", 1)
                self.assertEqual(vision.encode(picture()), (tmp / "cached.sve", 1))
                self.assertEqual(vision.stats["hits"], 1)
                launch.assert_not_called()
        finally:
            vision.close()

    def test_clean_park_retains_cache_closes_pipes_and_joins_reader(self):
        vision = BoundedVision.__new__(BoundedVision)
        stdin, stdout = io.StringIO(), io.StringIO()
        proc = Mock(stdin=stdin, stdout=stdout, returncode=0)
        proc.poll.return_value = None
        vision.proc, vision.reader = proc, Mock()
        vision.reader.is_alive.return_value = False
        vision.parking, vision.failed = False, False
        vision.cache = OrderedDict([("image", (Path("cached.sve"), 512))])
        vision.park()
        proc.wait.assert_called_once_with(timeout=10)
        proc.kill.assert_not_called()
        self.assertTrue(stdin.closed and stdout.closed)
        self.assertIsNone(vision.proc)
        self.assertFalse(vision.failed or vision.parking)
        self.assertEqual(vision.last_exit_code, 0)
        self.assertEqual(len(vision.cache), 1)
        vision.park()
        proc.wait.assert_called_once()

    def test_park_cleanup_failure_latches_and_resets_transition(self):
        vision = BoundedVision.__new__(BoundedVision)
        proc = Mock(returncode=0)
        proc.poll.return_value = 0
        proc.stdin.close.side_effect = OSError("test pipe close failure")
        vision.proc, vision.reader = proc, None
        vision.parking, vision.failed = False, False
        with self.assertRaises(OSError):
            vision.park()
        self.assertIsNone(vision.proc)
        self.assertTrue(vision.failed)
        self.assertFalse(vision.parking)
        proc.stdout.close.assert_called_once()

    def test_inline_only_and_invalid(self):
        self.assertTrue(normalize(inline_image(picture())))
        for source in ("https://example.com/x.png", "file:///c:/secret", "c:/secret", "data:image/png;base64,", "data:image/svg+xml;base64,YQ==", True, "data:image/png;base64,AAAA="):
            with self.subTest(source=source), self.assertRaises(ValueError):
                raw = inline_image(source)
                normalize(raw)

    def test_images_and_pixels_bounded(self):
        with self.assertRaises(ValueError):
            validate_sources([picture()] * 9)
        with patch("bounded_vision.MAX_PIXELS", 16):
            with self.assertRaises(ValueError):
                normalize(inline_image(picture()))

    def test_sve_structural_and_float_bounds(self):
        tmp = Path(__file__).resolve().parent / ("test-sve-" + uuid.uuid4().hex)
        tmp.mkdir()
        path = tmp / "x.sve"
        try:
            good = struct.pack("<5i", MAGIC, 1, 1, 1, WIDTH) + struct.pack("<f", 0.125) * WIDTH
            path.write_bytes(good)
            self.assertEqual(validate_sve(path, expected_tokens=1), [(1, 1, 1)])
            bad = [good[:-1], good[:19], good + b"x", good * 9,
                   struct.pack("<5i", MAGIC, 2**30, 2**30, 1, WIDTH),
                   struct.pack("<5i", MAGIC, 1, 1, 1, WIDTH + 1) + good[20:],
                   struct.pack("<5i", MAGIC, 1, 2, 1, WIDTH) + good[20:],
                   good[:20] + struct.pack("<f", float("nan")) * WIDTH,
                   good[:20] + struct.pack("<f", float("inf")) * WIDTH]
            for raw in bad:
                path.write_bytes(raw)
                with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                    validate_sve(path)
            path.write_bytes(good)
            with self.assertRaises(ValueError):
                validate_sve(path, expected_tokens=2)
        finally:
            path.unlink(missing_ok=True)
            tmp.rmdir()

    def test_context_rejected_before_encoder_or_native(self):
        class FakeServer:
            @staticmethod
            def images_of(messages): return []
            class Service:
                def __init__(self):
                    self.engine = type("E", (), {"max_context": 32})()
                    self.tok = type("T", (), {"encode": lambda *a, **kw: [1] * 24})()
                    self.template = type("P", (), {"render": lambda *a, **kw: "text"})()
                def prepare(self, *a): raise AssertionError("Native/encoder must not be reached")
        svc = vision_service_class(FakeServer)()
        with self.assertRaises(ValueError):
            svc.prepare([], None, {}, 1)


if __name__ == "__main__": unittest.main()
