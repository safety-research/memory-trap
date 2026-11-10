"""Paths and run settings.

Inputs live in data/, everything a run writes goes to outputs/<model>/. `run.py` sets the environment variables
below before it starts an experiment, and every module reads them when it is imported.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUTPUTS = os.environ.get("MEMTRAP_OUTPUTS") or os.path.join(ROOT, "outputs")
GRADINGBENCH = os.environ.get("MEMTRAP_GRADINGBENCH") or os.path.join(
    ROOT, "bench", "superhuman", "imobench", "gradingbench.csv")

MODEL = os.environ.get("MEMTRAP_MODEL", "")             # the model under test
WORKERS = int(os.environ.get("MEMTRAP_WORKERS", "32"))  # concurrent API requests
FORCE = os.environ.get("MEMTRAP_FORCE") == "1"          # start a step over instead of resuming
JUDGE = "claude-sonnet-4-6"                             # the fixed judge of every setting


def out(*parts, model=None):
    """Path of an output file under outputs/<model>/, creating its folder."""
    path = os.path.join(OUTPUTS, model or MODEL, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def data(*parts):
    return os.path.join(DATA, *parts)


def need(path, step):
    """Exit with a hint if an input written by an earlier step (or the benchmark) is missing."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run `{step}` first (see README)")
    return path
