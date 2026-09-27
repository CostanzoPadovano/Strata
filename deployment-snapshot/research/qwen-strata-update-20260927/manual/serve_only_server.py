"""Bounded authenticated API, inside an owned Job; separately admitted CPU vision."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from cache_protocol import parse_cache_done

MANUAL = Path(__file__).resolve().parent
CAMPAIGN = MANUAL.parent
MODEL = "qwen3.8-flash-next-local"  # SAME historical global Pi identity.
ENGINE_SHA = "59ef6d14bd5c1c032b84e2ca08ec11aab11bd8c24cca51a156eb45ee0a20fea6"
KEYFILE = Path(r"C:\llama_official\router\api-key.txt")
NODE = r"C:\Users\costa\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
CONTEXT_CAPACITY = 98304
CONTEXT_RESERVE = 8  # Keep the native position/checkpoint safety margin.
MAX_OUTPUT = CONTEXT_CAPACITY - CONTEXT_RESERVE
OUTPUT_POLICY = "remaining-context-v1"
MAX_BODY = 1024 * 1024
VISION_CAMPAIGN = CAMPAIGN


def request_cap(req: dict) -> int:
    if not isinstance(req, dict) or req.get("model") != MODEL:
        raise ValueError("Wrong model; select local-qwen38/qwen3.8-flash-next-local")
    value = req.get("max_completion_tokens", req.get("max_tokens", MAX_OUTPUT))
    if type(value) is not int or not 1 <= value <= MAX_OUTPUT:
        raise ValueError(f"Output must be 1..{MAX_OUTPUT} tokens including thinking")
    if "max_tokens" in req and "max_completion_tokens" in req and req["max_tokens"] != value:
        raise ValueError("Conflicting output limits")
    return value


def output_budget(requested: int, prompt_tokens: int, context: int) -> int:
    """Fit generation after REAL template/image tokenization; never trim input."""
    if type(requested) is not int or not 1 <= requested <= MAX_OUTPUT:
        raise ValueError("Invalid finite output budget")
    if (type(prompt_tokens) is not int or prompt_tokens <= 0 or type(context) is not int
            or not 1 <= context <= CONTEXT_CAPACITY):
        raise ValueError("Invalid prompt/context token counts")
    available = context - prompt_tokens - CONTEXT_RESERVE
    if available < 1:
        raise ValueError("Context exhausted: compact the conversation; input is never truncated")
    return min(requested, available)


def prepare_output_request(service, request, convert):
    requested = request_cap(request)
    if "stream" in request and type(request["stream"]) is not bool:
        raise ValueError("stream must be a boolean")
    if request.get("stream", False) is not True and requested > 1024:
        raise ValueError("Output above1024 tokens requires stream:true so client disconnect can cancel generation")
    messages, tools, kwargs = convert(request)
    # Prepare ONCE, with the minimum output. In vision mode this includes the
    # actual expanded image pads/SVE and the existing +8 admission checks.
    ids, thinking = service.prepare(messages, tools, kwargs, 1)
    maximum = output_budget(requested, len(ids), service.engine.max_context)
    return ids, thinking, tools, requested, maximum


def cleanup_prepared_embedding(service):
    path = getattr(service.embeddings, "path", None)
    if path is None:
        return
    path = Path(path).resolve()
    vision = getattr(service, "vision", None)
    if (vision is None or path.parent != Path(vision.dir).resolve()
            or not path.name.startswith("req-") or path.suffix != ".sve"):
        raise ValueError("Refusing cleanup of unowned embedding file")
    try:
        path.unlink(missing_ok=True)
    finally:
        service.embeddings.path = None


def serve_budgeted_openai(handler, server, service, req, run, body_bytes=0):
    ids, thinking, tools, requested, maximum = prepare_output_request(service, req, server.openai_to_messages)
    cancel, chunks = threading.Event(), None
    def client_write(action):
        # Only SOCKET writes count as disconnects; never swallow log/native IO.
        try:
            action()
            return True
        except OSError:
            cancel.set()
            return False
    try:
        with (run / "requests.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"body_bytes": body_bytes, "max_tokens": maximum,
                      "requested_max_tokens": requested, "prompt_tokens": len(ids),
                      "context_remaining_tokens": service.engine.max_context - len(ids) - CONTEXT_RESERVE,
                      "output_policy": OUTPUT_POLICY,
                      "budget_limited_by": "context" if maximum < requested else "request",
                      "reasoning_effort": req.get("reasoning_effort"), "time": time.time()}) + "\n")
        chunks = server.openai_chunks(service, req, ids, thinking, tools, maximum, cancel)
        if not req.get("stream"):
            # Retain only the old, small <=1024 non-stream contract.
            reply = server.openai_collect(chunks)
            client_write(lambda: handler._json(200, reply))
            return
        if not client_write(handler._sse):
            return
        for chunk in chunks:
            payload = (b": keep-alive\n\n" if chunk is None else
                       b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")
            if not client_write(lambda: handler.wfile.write(payload)) or not client_write(handler.wfile.flush):
                return
        if client_write(lambda: handler.wfile.write(b"data: [DONE]\n\n")):
            client_write(handler.wfile.flush)
    finally:
        try:
            if chunks is not None:
                chunks.close()  # Active native generation sends STOP/drains on close.
        finally:
            # Also handle failures BEFORE the first yield (log/header/initial
            # role chunk): Service.run's finally has not started at that point.
            cleanup_prepared_embedding(service)


def timing_summary(timings):
    def rate(count, milliseconds):
        return f"{count * 1000 / milliseconds:.2f}" if milliseconds > 0 else "n/a"
    prompt = timings["prompt_tokens"]
    reused = timings.get("prompt_reused", 0)
    fresh = prompt - reused
    processed = max(0, fresh - 1) # Last prompt token is the first decode window, not prefill.
    generated = timings["generated"]
    return (f"[timings] prefill {rate(processed, timings['prompt_ms'])} tok/s "
            f"({processed} nuovi, {reused} riusati, {prompt} contesto) | decode {rate(generated, timings['decode_ms'])} tok/s "
            f"({generated} token) | {timings['finish']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=["bench"], required=True)
    opts = parser.parse_args()
    run = opts.run.resolve()
    if not (run.is_relative_to(CAMPAIGN / "runs") or run.is_relative_to(VISION_CAMPAIGN / "runs")):
        raise ValueError("Evidence outside admitted campaign")
    result = {"passed": False, "scope": "user-requested experimental global Pi integration; NOT quality promotion"}
    engine = httpd = bridge = vision = None
    bridge_log = vision_log = None

    def save():
        (run / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def phase(name):
        (run / "phase.json").write_text(json.dumps({"phase": name, "time": time.time()}))
        print(name, flush=True)

    def wait_marker(path, seconds):
        deadline = time.monotonic() + seconds
        while not path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timeout waiting for {path.name}")
            time.sleep(0.1)

    try:
        wait_marker(run / "job-attached", 30)
        source = CAMPAIGN / "source"
        sys.path[:0] = [str(source), str(source / "tools")]
        import serve.server as server
        from serve.frontend import ChatTemplate, openai_to_messages
        from strata_tokenizer import Tokenizer
        config = json.loads((run / "config.json").read_text())
        context = int(config["args"][config["args"].index("--max-context") + 1])
        vision_enabled = "vision" in config
        if vision_enabled:
            from bounded_vision import BoundedVision, MAX_BODY as VISION_BODY, validate_sve, vision_service_class
            phase("Vision CPU su richiesta (8 thread); nessun encoder residente")
            vision_log = (run / "vision.log").open("x", encoding="utf-8")
            vision = BoundedVision(config["vision"], run / "vision-scratch", vision_log)
            result.update(vision_enabled=True, encoder_pid=None)
        tok_path = Path(config["tokenizer"])
        vocab = json.loads((tok_path / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
        tok = Tokenizer(tokens, (tok_path / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tok_path / "token_type.json").read_text()))

        class BoundedEngine(server.StrataEngine):
            def _parse_done(self, line):
                self.last = parse_cache_done(line)

            def __init__(self, *positional, **named):
                self.pump_ready = threading.Event()
                super().__init__(*positional, **named)
                if not self.pump_ready.wait(2):
                    raise TimeoutError("Bounded native reader did not initialize")

            def _pump(self):
                self.lines = queue.Queue(maxsize=2048)
                self.pump_ready.set()
                return super()._pump()

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                if not 1 <= max_new <= MAX_OUTPUT or not ids or len(ids) + max_new + 8 > self.max_context:
                    raise ValueError("Prompt+output+8 outside admitted context/output")
                if any(type(t) is not int or not 0 <= t < len(vocab) for t in ids):
                    raise ValueError("Invalid native token id")
                if embeddings is not None:
                    if not vision_enabled:
                        raise ValueError("Images are not admitted in text-only mode")
                    path = Path(embeddings).resolve()
                    if path.parent != vision.dir or not path.name.startswith("req-") or path.suffix != ".sve":
                        raise ValueError("Unowned image embeddings path")
                    validate_sve(path, expected_tokens=ids.count(248056))
                self.last = {}
                started = time.monotonic()
                try:
                    yield from super().generate(ids, max_new, sampling, cancel, embeddings=embeddings)
                except Exception as error:
                    result.setdefault("runtime_failure", str(error) or type(error).__name__)
                    save()
                    raise
                finally:
                    if self.last:
                        if (self.last.get("prompt_tokens") != len(ids)
                                or not 0 <= self.last.get("generated", -1) <= max_new):
                            result.setdefault("runtime_failure", "Authentic native prompt/output count mismatch")
                            save()
                            raise RuntimeError(result["runtime_failure"])
                        row = {"native_timings": self.last, "authentic_prompt_tokens": len(ids),
                               "requested_max_new": max_new, "wall_seconds": time.monotonic() - started}
                        with (run / "timings.jsonl").open("a", encoding="utf-8") as log:
                            log.write(json.dumps(row) + "\n")
                        print(timing_summary(self.last), flush=True)

        phase("Caricamento modello; attendi PRONTO")
        engine = BoundedEngine(config["exe"], config["args"], cwd=config["cwd"], log=config["log"])
        if engine.max_context != context:
            raise RuntimeError("Native context differs from admitted configuration")
        result.update(load_passed=True, native_pid=engine.proc.pid, context=engine.max_context)
        save()
        (run / "ready.json").write_text(json.dumps({"native_pid": engine.proc.pid, "context": engine.max_context}))
        wait_marker(run / "ready-accepted.json", 30)
        service = vision_service_class(server) if vision_enabled else server.Service
        svc = service(engine, tok, ChatTemplate(tok_path / "chat_template.jinja"), model_name=MODEL, vision=vision)
        svc.api_key = KEYFILE.read_text().strip()
        if not svc.api_key:
            raise ValueError("Missing local authentication")
        result["startup_self_test"] = "disabled"
        save()
        if context != 98304:
            raise ValueError("Manual API is only admitted at98K")
        base = server.make_handler(svc)

        class Handler(base):
            def setup(self):
                super().setup()
                self.connection.settimeout(1200)

            def do_GET(self):
                if not self._authorized():
                    return
                if self.path == "/health":
                    encoder_process = vision.proc if vision_enabled else None
                    return self._json(200, {"status": "ok", "runtime": "ista-strata-vision-v1" if vision_enabled else "ista-strata-manual-v1",
                        "max_context": 98304, "max_output_tokens": MAX_OUTPUT, "model": MODEL,
                        "output_policy": OUTPUT_POLICY, "output_context_reserve_tokens": CONTEXT_RESERVE,
                        "engine_sha256": config.get("vision_engine_sha256", ENGINE_SHA), "images": vision_enabled,
                        "conversation_cache": {"enabled": True, "max_checkpoints": 6,
                                               "interval_tokens": 16384, "short_read": 0,
                                               "kv_location": "gpu", "host_reservation_bytes": 1 << 30},
                        **({"vision_cpu": True, "max_image_tokens": 1024, "max_images": 8,
                             "encoder_sha256": "98ffde98f5388b518d8581b2983973d785f20f00a21e5e82393b38f4188e094c",
                             "projector_sha256": "b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad",
                             "encoder_residency": "on-demand",
                             "encoder_active": encoder_process is not None and encoder_process.poll() is None,
                             "encoder_pid": encoder_process.pid if encoder_process is not None else None} if vision_enabled else {}),
                        "experimental": True})
                if self.path not in ("/status", "/v1/models"):
                    return self._json(404, {"error": "API endpoint only"})
                return super().do_GET()

            def do_POST(self):
                if not self._authorized():
                    return
                if self.path != "/v1/chat/completions" or self.headers.get("Transfer-Encoding"):
                    return self._json(400, {"error": {"message": "Unsupported request"}})
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= (VISION_BODY if vision_enabled else MAX_BODY):
                        raise ValueError("Request body exceeds bounded limit or is empty")
                    req = json.loads(self.rfile.read(size))
                    request_cap(req)
                    return self._openai(req, body_bytes=size)
                except ValueError as error:
                    return self._json(400, {"error": {"message": str(error), "type": "invalid_request_error"}})

            def _openai(self, req, body_bytes=0):
                try:
                    return serve_budgeted_openai(self, server, svc, req, run, body_bytes)
                except OSError as error:
                    result.setdefault("runtime_failure", "Request evidence/native IO failed: " + type(error).__name__)
                    raise  # Main service loop now stops instead of losing evidence silently.

        class BoundedServer(server.Server):
            request_queue_size = 4
            slots = threading.BoundedSemaphore(4)

            def process_request(self, request, address):
                if not self.slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, address)
                except BaseException:
                    self.slots.release()
                    raise

            def process_request_thread(self, request, address):
                try:
                    super().process_request_thread(request, address)
                finally:
                    self.slots.release()

        httpd = BoundedServer(("127.0.0.1", 8037), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        # Discover WSL's private gateway at each launch; never bind LAN/all-address.
        bridge_log = (run / "bridge.log").open("x", encoding="utf-8")
        bridge = subprocess.Popen([NODE, str(MANUAL / "manual_bridge.mjs")], stdout=bridge_log,
                                  stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(1)
        if bridge.poll() is not None:
            raise RuntimeError("Private WSL bridge failed; inspect bridge.log")
        phase("PRONTO - in WSL usa pi (ISTA / Strata); INVIO per arrestare")
        result["api_ready"] = True
        save()
        while not (run / "stop-server").exists():
            encoder_process = vision.proc if vision else None
            if (engine.proc.poll() is not None or bridge.poll() is not None or result.get("runtime_failure")
                    or (vision and (vision.failed or (not vision.parking and encoder_process is not None
                                                      and encoder_process.poll() is not None
                                                      and vision.proc is encoder_process)))):
                raise RuntimeError("Native engine/bridge failed or native counts differed")
            time.sleep(0.25)
        result["passed"] = not result.get("runtime_failure")
    finally:
        phase("Arresto server")
        def cleanup(name, action):
            try:
                action()
            except Exception as error:
                result.setdefault("cleanup_failures", []).append({"component": name, "error": str(error)})
                result["passed"] = False

        def close_bridge():
            bridge.terminate()
            try:
                bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                bridge.kill()
                bridge.wait(timeout=10)

        if bridge is not None:
            cleanup("bridge", close_bridge)
        if bridge_log is not None:
            cleanup("bridge_log", bridge_log.close)
        if httpd is not None:
            cleanup("httpd", httpd.shutdown)
            cleanup("httpd_sockets", httpd.server_close)
        if engine is not None:
            cleanup("engine", engine.close)
            if engine.proc.poll() is None:
                cleanup("engine_wait", lambda: engine.proc.wait(timeout=10))
            result["native_exit_code"] = engine.proc.poll()
            if engine.proc.returncode != 0:
                result["passed"] = False
        if vision is not None:
            result["vision_stats"] = vision.stats
            cleanup("vision", vision.close)
            result["encoder_exit_code"] = vision.last_exit_code
            if vision.last_exit_code not in (None, 0) or (vision.stats["starts"] and vision.last_exit_code != 0):
                result["passed"] = False
        if vision_log is not None:
            cleanup("vision_log", vision_log.close)
        save()


if __name__ == "__main__":
    main()
