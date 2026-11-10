"""Shared pieces of the proof graders: the memory arms and a resumable grade log."""
import csv
import json
import os
import random
import threading

from memtrap import paths
from memtrap.sycophancy.feedback import bank_slots

KNOTES = 10
OUTPUT_LIMITS = [16000, 32000, 64000, 128000]   # max_tokens of a grading, in order
TRIES = 7                                       # gradings per output limit
BENCH_HINT = "git clone https://github.com/google-deepmind/superhuman bench/superhuman"


def note_arms():
    """{"baseline": [""], "syco": the empathy note slots, "overcrit": the rigor note slots}; a missing note is ''."""
    hint = lambda e: f"python run.py {e} --model {paths.MODEL}"
    return {"baseline": [""],
            "syco": bank_slots(paths.need(paths.out("sycophancy", "empathy_notes.json"), hint("empathy")), KNOTES),
            "overcrit": bank_slots(paths.need(paths.out("sycophancy", "rigor_notes.json"), hint("rigor")), KNOTES)}


def grade_record(key, field, value, text, stop, mt):
    """One grading's log record. A reply cut off by the output limit carries no grade: its parse is kept as
    <field>_truncated_parse and the grade is null."""
    rec = {**key, field: value, "mt": mt, "stop_reason": stop}
    if stop == "max_tokens" and value is not None:
        rec[field + "_truncated_parse"] = value
        rec[field] = None
    if rec[field] is None:
        rec["t"] = (text or "")[-4000:]   # the reply's tail, for checking
    return rec


def grade(key, field, parse, complete):
    """One grading as a log record. `complete(max_tokens)` returns (text, stop_reason). A reply without a grade is
    asked again, at the next output limit if it was cut off."""
    for mt in OUTPUT_LIMITS:
        for _ in range(TRIES):
            text, stop = complete(mt)
            rec = grade_record(key, field, parse(text), text, stop, mt)
            if rec[field] is not None:
                return rec
            if stop == "max_tokens" or not (rec.get("t") or "").strip():
                break
    return rec


def sample():
    """The 1,000 rows, shuffled within each reference grade (seed 42), grades in order of first appearance."""
    rows = list(csv.DictReader(open(paths.need(paths.GRADINGBENCH, BENCH_HINT))))
    by_rew = {}
    for r in rows:
        by_rew.setdefault(r["Reward"], []).append(r)
    rng = random.Random(42); out = []
    for rew, rs in by_rew.items():
        rng.shuffle(rs); out += rs
    return out


class GradeCache:
    """Append-only JSONL log of grades keyed by (arm, note index, item); later lines win."""

    def __init__(self, path, keys, field):
        self.keys, self.field = keys, field
        self.done = {}
        if os.path.exists(path):
            for l in open(path):
                try:
                    x = json.loads(l)
                    self.done[tuple(x[k] for k in keys)] = x[field]
                except Exception:
                    pass
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a")

    def add(self, rec):
        with self._lock:
            self._f.write(json.dumps(rec) + "\n"); self._f.flush()
