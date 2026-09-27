"""Stop only this exact disposable Pi image test; never other Pi/WSL work."""
import os
from pathlib import Path
import signal
import time

EXPECTED_CWD = "/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-agent-20260927/manual"
NEEDLE = b"Usa read per aprire ../../qwen-strata-vision-20260927/fixtures/alt-red.png"


def ours(pid):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        return (NEEDLE in cmd and b"--no-session\x00" in cmd and b"--tools\x00read\x00" in cmd
                and os.readlink(f"/proc/{pid}/cwd") == EXPECTED_CWD)
    except (OSError, ValueError): return False


targets = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit() and ours(p.name)]
for pid in targets:
    if ours(pid): os.kill(pid, signal.SIGTERM)
deadline = time.monotonic() + 10
while any(ours(pid) for pid in targets) and time.monotonic() < deadline: time.sleep(.1)
for pid in targets:
    if ours(pid): os.kill(pid, signal.SIGKILL)
print({"stopped_exact_disposable_pi_pids": targets})
