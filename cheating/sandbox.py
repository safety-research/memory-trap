"""Adapter from the sandbox module `rh_dock_sandbox` to `cheating.terminal.Terminal`.

It does two things:

  * maps a record from `cheating/data/` onto the arguments that module's builders take, and
  * forwards `Terminal.run` / `final_test` / `close` to that module's functions.

The module runs a session in a container or directly on the host, the two terminals `cheating/terminal.py`
describes, and `open_repo` below selects one per session.

Environment:

    MEMTRAP_RHFORM_DIR=~/rhform3/code        directory holding rh_dock_sandbox.py (default ~/rhform3/code)
    MEMTRAP_RHFORM_SRC=~/rhform3/src         its sibling src/, which holds memento (default ../src)
    MEMTRAP_DOCK_POOL=48                     containers to pre-start (default 48)
    MEMTRAP_DOCK_PREFIX=rhdock_p             container name prefix; one per parallel run
    MEMTRAP_DOCK_MODULE=rh_dock_sandbox      module name, if yours is called something else

To use another sandbox, replace this file; `cheating/terminal.py` describes the interface.
"""
import importlib
import os
import sys
import threading

from cheating import terminal

CODE_DIR = os.environ.get("MEMTRAP_RHFORM_DIR") or os.path.expanduser("~/rhform3/code")
# The sandbox module imports `memento` from the checkout's sibling `src/` directory.
SRC_DIR = os.environ.get("MEMTRAP_RHFORM_SRC") or os.path.join(os.path.dirname(CODE_DIR.rstrip("/")), "src")
MODULE = os.environ.get("MEMTRAP_DOCK_MODULE") or "rh_dock_sandbox"
POOL = int(os.environ.get("MEMTRAP_DOCK_POOL", "48"))

_DK = None
_POOL_UP = False
_LOCK = threading.Lock()


def _use(dk, where):
    """Select the terminal in the calling thread.

    `rh_dock_sandbox` keeps the choice in a `threading.local()` whose default is the host, and sessions run in a
    thread pool, so it is set on each entry into the module.
    """
    dk.set_mode(where)


def backend():
    """Import the sandbox module, once per process."""
    global _DK
    if _DK is not None:
        return _DK
    with _LOCK:
        if _DK is None:
            for d in (SRC_DIR, CODE_DIR):
                if os.path.isdir(d) and d not in sys.path:
                    sys.path.insert(0, d)
            try:
                dk = importlib.import_module(MODULE)
            except ImportError as e:
                raise SystemExit(
                    f"could not import {MODULE} from {CODE_DIR}.\n"
                    "Set MEMTRAP_RHFORM_DIR to the directory holding it, or replace cheating/sandbox.py with your "
                    f"own runner (see cheating/README.md).\n\n({e})")
            prefix = os.environ.get("MEMTRAP_DOCK_PREFIX")
            if prefix:
                dk.NAME_PREFIX = prefix
            _DK = dk
    return _DK


def _start_containers(dk):
    """Build the image and start the container pool, the first time a session needs a container.

    The builders lease their containers from the pool.
    """
    global _POOL_UP
    if _POOL_UP:
        return
    with _LOCK:
        if not _POOL_UP:
            _use(dk, terminal.DOCKER)
            dk.ensure_image()
            dk.ensure_pool(POOL)
            _POOL_UP = True


def _formation_fsc(task):
    """A `cheating/data/formation_tasks.json` record -> the dict the formation builder reads.

    It takes the two commits as `old_impl` (commit 1, with `name` as that commit's message) and `new_impl`
    (commit 2, with `commit_msg`), plus the test to commit alongside the first.
    """
    return {"name": task["name"],
            "old_impl": task["initial_solution"],
            "new_impl": task["solution"],
            "canon_test": task["test"],
            "commit_msg": task["commit_message"]}


class SandboxTerminal(terminal.Terminal):
    """Forwards the Terminal interface to the sandbox module."""

    def __init__(self, path, canonical, dk, where):
        self._path = path
        self._canonical = canonical
        self._dk = dk
        self.where = where

    def run(self, command):
        # The module applies its own output limit and timeout; TIMEOUT is passed so the two agree.
        _use(self._dk, self.where)
        return self._dk.run_bash(self._path, command, self._canonical, timeout=terminal.TIMEOUT)["output"]

    def final_test(self):
        try:
            with open(os.path.join(self._path, "test_solution.py")) as f:
                return f.read()
        except OSError:
            return ""

    def close(self):
        try:
            _use(self._dk, self.where)
            self._dk.cleanup_sandbox(self._path)
        except Exception:
            pass


def open_repo(kind, record, workdir=None, where=terminal.DOCKER):
    """A fresh sandbox for one formation task or one broken-safeguard case.

    A formation session runs in `where`; a safeguard session runs in the container.
    """
    dk = backend()
    if kind != terminal.FORMATION:
        where = terminal.DOCKER
    if where == terminal.DOCKER:
        _start_containers(dk)
    _use(dk, where)
    try:
        if kind == terminal.FORMATION:
            path = dk.make_form_sandbox_frozen_git(_formation_fsc(record))
        else:
            path = dk.make_sandbox(record["solution"], record["test"])
    except TypeError as e:
        # The module resolves its builders at import time in a try/except, so a failed import surfaces here.
        if "NoneType" in str(e):
            raise SystemExit(
                f"{MODULE} imported, but its sandbox builders are unavailable: the harness modules it wraps "
                f"failed to import.\nUsually `memento` is not on the path -- expected in {SRC_DIR}.\n"
                "Set MEMTRAP_RHFORM_SRC to the directory that contains it.") from e
        raise
    # Bytes: the module compares the file against this with a binary read.
    # `get_mode()` is the terminal the module built the session in.
    return SandboxTerminal(path, record["solution"].encode(), dk, dk.get_mode())
