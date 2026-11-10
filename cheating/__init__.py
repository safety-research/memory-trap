"""The cheating setting: test-repair formation in a terminal, the broken-safeguard set, and ImpossibleBench.

Paths: inputs are in cheating/data/, outputs go to outputs/<model>/cheating/ (see memtrap.paths).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ROOT = os.path.dirname(HERE)
IMPOSSIBLEBENCH = os.environ.get("MEMTRAP_IMPOSSIBLEBENCH") or os.path.join(ROOT, "bench", "impossiblebench")


def data(name):
    return os.path.join(DATA, name)
