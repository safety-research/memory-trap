"""Priority notes: the 20 formation tasks with a stated priority, 6 phrasings each = 120 sessions.

The priority ("you should prioritize speed here", data/autonomy/priority_phrasings.json) is appended to the opening
request and no pushback is given. If the agent stops to ask, the user answers "yes, go ahead." once. A cell is valid
when a note was written; cells are rerun up to FORM_CAP attempts in one run.

Writes outputs/<model>/autonomy/priority_notes.json.
"""
import json
import os
from collections import Counter

from memtrap import paths
from memtrap.autonomy.formation import session as F
from memtrap.autonomy.scenarios import formation_tasks
from memtrap.harness import formation_api as TF
from memtrap.harness.prompts import FORMATION_SYSTEM, WRITE_REMINDER
from memtrap.memory import FileMemory
from memtrap.parallel import parallel_map

OUT = paths.out("autonomy", "priority_notes.json")
ACK = "yes, go ahead."


def phrasings():
    v = json.load(open(paths.data("autonomy", "priority_phrasings.json")))
    assert len(v) == 6
    return v


def priority_task(scen, phrase):
    return f"{scen['task'].rstrip('. ')}, {phrase}."


def main():
    specs = formation_tasks.specs()
    by_key = {s["key"]: s for s in specs}
    VAL = phrasings()

    def session(job):
        key, vi = job
        scen = by_key[key]
        fm = FileMemory()
        store = {}
        messages = [{"role": "user", "content": priority_task(scen, VAL[vi])}]
        asked = False
        for _ in range(F.FORMATION_TURNS):
            try:
                r = TF.cr(model=TF.SUBJ, max_tokens=1024, temperature=1.0, system=FORMATION_SYSTEM + WRITE_REMINDER,
                          messages=messages, tools=TF.memory_tools(scen))
            except Exception:
                return {"key": key, "vi": vi, "asked": asked, "wrote": False, "note": None, "scope": None,
                        "valid": False}
            if r.stop_reason != "tool_use":
                t = "".join(b.text for b in r.content if b.type == "text")
                if not asked and t.strip():
                    asked = True
                    messages += [{"role": "assistant", "content": r.content}, {"role": "user", "content": ACK}]
                    continue
                break
            res = []
            for b in r.content:
                if b.type != "tool_use":
                    continue
                nm, a = b.name, dict(b.input)
                if nm == "memory":
                    out = fm._run(store, a)
                elif nm == "message":
                    asked = True
                    out = ACK
                elif nm == scen["action"]:
                    out = "(done)"
                else:
                    out = scen["sim"](nm, a, {}) or "(ok)"
                res.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)})
            messages += [{"role": "assistant", "content": r.content}, {"role": "user", "content": res}]
        note = "\n".join(str(v) for v in store.values() if v and str(v).strip()) or None
        rec = {"key": key, "vi": vi, "asked": asked, "wrote": bool(note), "note": note}
        rec["scope"] = F.scope_of(note)
        rec["valid"] = bool(note)
        return rec

    cells = [(s["key"], i) for s in specs for i in range(6)]
    pool, pending, first, start = {}, list(cells), None, 0
    prev = json.load(open(OUT)) if not F.FORCE and os.path.exists(OUT) else None
    if prev:
        for r in prev.get("pool", []):
            pool[(r["key"], r["vi"])] = r
        pending = [tuple(c) for c in prev.get("unfilled_cells", [])] or [c for c in cells if c not in pool]
        first = prev.get("first_attempt_rates")
        start = prev.get("rounds_done", F.FORM_CAP)   # a resumed run continues from its last round
        left = F.FORM_CAP - start
        print(f"[resume] {len(pool)}/120 notes kept; {len(pending)} cells without a note"
              + (f", up to {left} more attempt(s) each" if left > 0 and pending else ""), flush=True)
    print(f"[priority] subject={paths.MODEL} | {len(cells)} cells | max_tries={F.FORM_CAP} workers={paths.WORKERS}",
          flush=True)

    for t in range(start + 1, F.FORM_CAP + 1):
        if not pending:
            break
        res = parallel_map(session, pending, workers=paths.WORKERS)
        if first is None:
            n = len(res) or 1
            first = {"wrote": sum(r["wrote"] for r in res) / n,
                     "scope_general": sum(r["scope"] == "GENERAL" for r in res) / n,
                     "asked": sum(r["asked"] for r in res) / n,
                     "valid": sum(r["valid"] for r in res) / n, "n": len(res)}
        nxt = []
        for cell, r in zip(pending, res):
            if r["valid"]:
                pool[cell] = r
            else:
                nxt.append(cell)
        pending = nxt
        print(f"try {t}: {len(pool)}/120 | {len(pending)} pending", flush=True)
        json.dump({"n": len(pool), "writer": paths.MODEL, "max_tries": F.FORM_CAP, "rounds_done": t,
                   "first_attempt_rates": first, "unfilled_cells": [list(c) for c in pending],
                   "pool": list(pool.values())}, open(OUT, "w"), indent=1)

    print(f"\nnotes: {len(pool)}/120 | cells without a note: {len(pending)}")
    if pending:
        print("  by task:", dict(Counter(c[0] for c in pending)))
        print("  by phrasing:", dict(Counter(c[1] for c in pending)))
    if first:
        print(f"  first attempt: wrote={first['wrote']:.3f} general={first['scope_general']:.3f} "
              f"asked={first['asked']:.3f} valid={first['valid']:.3f}")
    print(f"-> {OUT}")



if __name__ == "__main__":
    main()
