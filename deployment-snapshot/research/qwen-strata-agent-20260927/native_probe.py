"""Fresh load/smoke evidence through the established observer and bounded reader."""
from pathlib import Path
import queue
import sys
import threading

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'source/tools'),
               str(ROOT.parent / 'qwen-strata-stock-20260926')]
import native_probe as stock_probe
import serve.server as server


class BoundedEngine(server.StrataEngine):
    def __init__(self, *args, **kwargs):
        self.reader_ready = threading.Event()
        super().__init__(*args, **kwargs)
        if not self.reader_ready.wait(2):
            raise TimeoutError('Bounded native reader not initialized')

    def _pump(self):
        self.lines = queue.Queue(maxsize=2048)
        self.reader_ready.set()
        return super()._pump()


if __name__ == '__main__':
    server.StrataEngine = BoundedEngine
    stock_probe.ROOT = ROOT
    stock_probe.main()
