"""The model under test at deployment: its client, one completion, and how memory is delivered.

Memory arrives as a user message before the task (MEMTRAP_MEMORY_FORMAT):
  note              the note, framed as the model's own MEMORY.md
  transcript        the formation transcript quoted inside one user message
  transcript_turns  the transcript's turns as separate messages after a preamble
An empty memory gives the plain no-memory session.
"""

import json
import os
import random
import time

from memtrap import paths
from memtrap.clients import make_client
from memtrap.clients.messages import Message
from memtrap.harness.prompts import NOTE_HEADER, TRANSCRIPT_PREAMBLE

SUBJ_MODEL = paths.MODEL
_TIMEOUT = float(os.environ["MEMTRAP_CLIENT_TIMEOUT"]) if os.environ.get("MEMTRAP_CLIENT_TIMEOUT") else None
SUBJ = make_client("auto", model=SUBJ_MODEL, max_retries=8, timeout=_TIMEOUT)
MEMORY_FORMAT = os.environ.get("MEMTRAP_MEMORY_FORMAT", "note")
assert MEMORY_FORMAT in ("note", "transcript", "transcript_turns"), MEMORY_FORMAT


def _turns(mem):
    """A transcript pool entry is a JSON list of {role, text} turns."""
    try:
        return json.loads(mem)
    except (ValueError, TypeError):
        return [{"role": "user", "content": mem}]


def deliver_msgs(mem, task):
    """The first user turn(s) of a session: the task alone, or the memory and then the task."""
    if not mem:
        return [Message("user", task)]
    if MEMORY_FORMAT == "transcript":
        text = "\n\n".join(f"{str(x.get('role', 'user')).capitalize()}: {x.get('text', x.get('content', ''))}"
                           for x in _turns(mem) if x.get("text", x.get("content", "")))
        return [Message("user", f"{TRANSCRIPT_PREAMBLE}\n\n{text}"), Message("user", task)]
    if MEMORY_FORMAT == "transcript_turns":
        return ([Message("user", TRANSCRIPT_PREAMBLE)]
                + [Message(t.get("role", "user"), t.get("text", t.get("content", "")))
                   for t in _turns(mem) if t.get("text") or t.get("content")]
                + [Message("user", task)])
    return [Message("user", NOTE_HEADER + mem), Message("user", task)]


def complete(system, messages, mt, temp):
    """One completion as (text, stop_reason), retried with backoff; ("", None) if every attempt fails."""
    for i in range(16):
        try:
            r = SUBJ.complete(system=system, messages=messages, max_tokens=mt, temperature=temp)
            return r.text, r.stop_reason
        except Exception as e:
            if i == 15:
                print(f"[comp] giving up after 16 tries: {type(e).__name__}", flush=True)
                return "", None
            time.sleep(min(5 * (2 ** i), 90) + random.uniform(0, 5))
    return "", None


def comp(system, messages, mt, temp):
    """One completion's text ("" if every attempt fails)."""
    return complete(system, messages, mt, temp)[0]
