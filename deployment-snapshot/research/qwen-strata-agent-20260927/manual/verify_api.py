"""Small read-only HTTP boundary checks against an active admitted manual server."""
import json
from pathlib import Path
import urllib.error
import urllib.request

key = Path(r"C:\llama_official\router\api-key.txt").read_text().strip()
base = "http://127.0.0.1:8037"
model = "qwen3.8-flash-next-local"


def request(path, body=None, authenticated=True):
    headers = {"Authorization": "Bearer " + (key if authenticated else "invalid")}
    data = None if body is None else json.dumps(body).encode()
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read(8192))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read(8192))


code, health = request("/health")
assert code == 200 and health["max_context"] == 98304
assert health["max_output_tokens"] == 98296 and health["output_policy"] == "remaining-context-v1"
assert health["output_context_reserve_tokens"] == 8
assert health["model"] == model
assert health["runtime"] == ("ista-strata-vision-v1" if health["images"] else "ista-strata-manual-v1")
assert request("/health", authenticated=False)[0] == 401
for body in ({"model": "wrong", "messages": []}, {"model": model, "messages": [], "max_tokens": 98297},
             {"model": model, "messages": [{"role": "user", "content": "x " * 100000}], "max_tokens": 1024},
             {"model": model, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}], "max_tokens": 32}):
    code, payload = request("/v1/chat/completions", body)
    assert code == 400, (code, payload)
if health["images"]:
    assert health["vision_cpu"] and health["max_images"] == 8 and health["max_image_tokens"] == 1024
    for src in ("http://127.0.0.1:8037/health", "file:///C:/Windows/win.ini"):
        body = {"model": model, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": src}}]}], "max_tokens": 32}
        assert request("/v1/chat/completions", body)[0] == 400
print("Live API: authenticated identity/401/wrong-model/output-cap/context-overflow/invalid-image/remote-file rejection passed; no generation requested")
