"""CPU-only image encoder with finite inputs, cache, files and pipe deadlines.

No HTTP/file URLs: Pi must send inline images. No expert/model tensor mapping.
"""
from __future__ import annotations
from array import array
import base64
from collections import OrderedDict
import hashlib
import io
import math
from pathlib import Path
import queue
import re
import struct
import subprocess
import threading
import time

from PIL import Image, ImageOps

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_TOTAL = 8 * 1024 * 1024
MAX_BODY = 16 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
MAX_IMAGES = 8
MAX_TOKENS = 1024
WIDTH = 2560
MAX_SVE = MAX_IMAGES * (20 + MAX_TOKENS * WIDTH * 4)
MAGIC = 0x31455653
DATA = re.compile(r"data:image/(?:png|jpeg|jpg|webp|bmp|gif);base64,([A-Za-z0-9+/=]+)\Z", re.I)


def inline_image(source: str) -> bytes:
    if not isinstance(source, str) or len(source) > 64 + 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("Image exceeds4MiB")
    match = DATA.fullmatch(source)
    if not match:
        raise ValueError("Vision accepts inline base64 PNG/JPEG/WebP/BMP/GIF only; no URLs or file paths")
    try:
        raw = base64.b64decode(match[1], validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("Invalid image base64") from error
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is empty or exceeds4MiB")
    return raw


def normalize(raw: bytes) -> bytes:
    # Header/dimension/frame checks BEFORE decoding pixels, unlike upstream.
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.format not in {"PNG", "JPEG", "WEBP", "BMP", "GIF"}:
                raise ValueError("Unsupported image format")
            if img.width * img.height > MAX_PIXELS or max(img.size) > 8192 or getattr(img, "n_frames", 1) != 1:
                raise ValueError("Image exceeds16M pixels/8192px or is animated")
            img.load()
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA") or "transparency" in img.info:
                rgba = img.convert("RGBA")
                rgb = Image.new("RGB", img.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
            else:
                rgb = img.convert("RGB")
            # Bound the materialized file as well as encoder work. Images larger
            # than this are reduced before native stb/mtmd sees them.
            rgb.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            rgb.save(out, format="PNG")
            if out.tell() > 16 * 1024 * 1024:
                raise ValueError("Normalized image exceeds16MiB")
            return out.getvalue()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Invalid/oversized image") from error


def validate_sve(path: Path, *, expected_tokens: int | None = None) -> list[tuple[int, int, int]]:
    size = path.stat().st_size
    if not 20 < size <= MAX_SVE:
        raise ValueError("Embeddings file size outside bound")
    rows, count = [], 0
    with path.open("rb") as stream:
        while stream.tell() < size:
            hdr = stream.read(20)
            if len(hdr) != 20:
                raise ValueError("Truncated SVE header")
            magic, n, nx, ny, width = struct.unpack("<5i", hdr)
            if magic != MAGIC or not 1 <= n <= MAX_TOKENS or nx < 1 or ny < 1 or nx * ny != n or width != WIDTH:
                raise ValueError("Invalid SVE magic/grid/tokens/width")
            byte_count = n * width * 4
            if byte_count > size - stream.tell():
                raise ValueError("Truncated SVE payload")
            vals = array("f")
            vals.frombytes(stream.read(byte_count))
            if not all(math.isfinite(v) for v in vals):
                raise ValueError("Non-finite image embeddings")
            rows.append((n, nx, ny))
            count += n
            if len(rows) > MAX_IMAGES or count > MAX_IMAGES * MAX_TOKENS:
                raise ValueError("Too many images/embedding rows")
    if expected_tokens is not None and count != expected_tokens:
        raise ValueError("SVE token count differs from encoder/prompt")
    return rows


class BoundedVision:
    def __init__(self, cfg: dict, directory: Path, log, *, startup_timeout=180, encode_timeout=120):
        if cfg.get("gpu") is not False or cfg.get("max_tokens") != MAX_TOKENS or cfg.get("threads") != 8:
            raise ValueError("Only CPU/8threads/1024 image tokens admitted")
        self.dir = directory.resolve()
        if any(c.isspace() for c in str(self.dir)):
            raise ValueError("Native embedding protocol requires scratch paths without spaces")
        self.dir.mkdir(exist_ok=False)
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.lines = queue.Queue(maxsize=16)
        self.encode_timeout = encode_timeout
        self.failed = False
        self.parking = False
        self.cfg, self.log, self.startup_timeout = cfg, log, startup_timeout
        self.proc, self.reader, self.last_exit_code = None, None, None
        self.stats = {"encoded": 0, "hits": 0, "max_tokens": 0, "encode_seconds": [], "warmup_seconds": [], "starts": 0}

    def _start(self):
        # Measured encoder Job peak1.696GiB; reserve2GiB PLUS unchanged global
        # floors before a cold restart. If unavailable, reject just this image
        # request: never borrow the emergency4GiB or kill the text service.
        from vision_campaign import admitted
        memory = admitted.shared.counters()
        if memory["commit_free"] < 6 * (1 << 30) or memory["ram_free"] < 10 * (1 << 30):
            raise ValueError("New image needs6GiB free commit/10GiB free RAM including safety floors; cached images/text remain usable")
        cfg = self.cfg
        args = [cfg["exe"], "--mmproj", cfg["mmproj"], "--model", cfg["model"],
                "--threads", "8", "--max-tokens", str(MAX_TOKENS)]
        self.lines = queue.Queue(maxsize=16)
        started = time.monotonic()
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, encoding="utf-8", bufsize=1,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        self.reader = threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True)
        self.reader.start()
        try:
            if self._line(self.startup_timeout) != f"READY {WIDTH}":
                raise RuntimeError("Vision encoder did not report READY2560")
        except BaseException:
            self.failed = True
            self.park()
            raise
        self.stats["starts"] += 1
        self.stats["warmup_seconds"].append(round(time.monotonic() - started, 3))
        self.stats["warmup_seconds"] = self.stats["warmup_seconds"][-16:]

    def _pump(self, proc, lines):
        for line in proc.stdout:
            if len(line) > 4096:
                self.failed = True
                break
            try:
                lines.put(line.strip(), timeout=2)
            except queue.Full:
                self.failed = True
                break
        try:
            lines.put(None, timeout=2)
        except queue.Full:
            self.failed = True

    def _line(self, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as error:
            self.failed = True
            self.proc.kill()
            self.proc.wait(timeout=10)
            raise TimeoutError("Vision encoder deadline exceeded") from error
        if line is None or self.failed:
            self.failed = True
            raise RuntimeError("Vision encoder ended/failed")
        return line

    def encode(self, source):
        raw = normalize(inline_image(source))
        key = hashlib.sha256(raw).hexdigest()
        with self.lock:
            if self.failed:
                raise RuntimeError("Vision encoder failed; restart the server")
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return self.cache[key]
            if self.proc is None:
                self._start()
            # Evict BEFORE allocation so all cached+incoming embeddings fit8.
            if len(self.cache) >= MAX_IMAGES:
                _, (old, _) = self.cache.popitem(last=False)
                old.unlink()
            img, out = self.dir / f"{key}.png", self.dir / f"{key}.sve"
            started = time.monotonic()
            img.write_bytes(raw)
            try:
                self.proc.stdin.write(f"ENC {img} {out}\n")
                self.proc.stdin.flush()
                reply = self._line(self.encode_timeout).split()
                if len(reply) != 5 or reply[0] != "OK":
                    self.failed = True
                    raise RuntimeError("Vision encoder rejected image")
                n, nx, ny = map(int, reply[1:4])
                if validate_sve(out, expected_tokens=n) != [(n, nx, ny)]:
                    self.failed = True
                    raise RuntimeError("Encoder grid mismatch")
                self.stats["encoded"] += 1
                self.stats["max_tokens"] = max(n, self.stats["max_tokens"])
                self.stats["encode_seconds"].append(round(time.monotonic() - started, 3))
                self.stats["encode_seconds"] = self.stats["encode_seconds"][-16:]
                self.cache[key] = (out, n)
                return out, n
            finally:
                img.unlink(missing_ok=True)
                if key not in self.cache:
                    out.unlink(missing_ok=True)

    def park(self):
        """Release encoder RAM/CUDA context, retain the finite on-disk cache."""
        proc = self.proc
        if proc is None:
            return
        self.parking = True
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=10)
            self.last_exit_code = proc.returncode
            self.proc = None  # Confirmed exit, even if later pipe/thread cleanup fails.
            if self.reader:
                self.reader.join(timeout=2)
                if self.reader.is_alive():
                    raise RuntimeError("Encoder reader did not finish after owned process exit")
            try:
                proc.stdin.close()
            finally:
                proc.stdout.close()
            if self.last_exit_code != 0:
                self.failed = True
        except BaseException:
            self.failed = True
            raise
        finally:
            self.parking = False

    def close(self):
        self.park()
        # Exact owned scratch directory, no broad/recursive filesystem deletion.
        for path in self.dir.iterdir():
            if path.is_file() and path.suffix in {".sve", ".png"}:
                path.unlink()
        self.dir.rmdir()


def validate_sources(sources):
    if len(sources) > MAX_IMAGES:
        raise ValueError("At most8 images per request/history; compact the conversation")
    if sum(len(inline_image(src)) for src in sources) > MAX_IMAGE_TOTAL:
        raise ValueError("Images exceed8MiB total compressed input")


def vision_service_class(server):
    class BoundedService(server.Service):
        def prepare(self, messages, tools, kwargs, max_new):
            images = server.images_of(messages)
            validate_sources(images)
            # Fail BEFORE any encoder/native allocations for an oversized prompt.
            base_ids = self.tok.encode(self.template.render(messages, tools=tools, **kwargs), parse_special=True)
            if len(base_ids) + max_new + 8 > self.engine.max_context:
                raise ValueError("Prompt+output+8 exceeds context; requests are never truncated")
            # The upstream preparation retains each path after releasing its
            # encoder lock. Serialize this whole operation to prevent another
            # request evicting an embedding before the combined copy is made.
            with self.prepare_lock:
                try:
                    ids, thinking = super().prepare(messages, tools, kwargs, max_new)
                    if len(ids) + max_new + 8 > self.engine.max_context:
                        raise ValueError("Image-expanded prompt+output+8 exceeds context")
                    if self.embeddings.path:
                        validate_sve(Path(self.embeddings.path), expected_tokens=ids.count(248056))
                    return ids, thinking
                except BaseException:
                    path = getattr(self.embeddings, "path", None)
                    if path:
                        Path(path).unlink(missing_ok=True)
                    self.embeddings.path = None
                    raise
                finally:
                    # Complete encode+copy before freeing CPU work buffers;
                    # retained SVE files serve all subsequent cache hits.
                    if self.vision is not None:
                        self.vision.park()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prepare_lock = threading.Lock()
    return BoundedService
