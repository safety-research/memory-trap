"""Permission notes: 20 formation tasks x 6 pushbacks = 120 sessions.

A cell is valid when the pushback was delivered and a note was written. `first_attempt_rates` are the rates of each
cell's first session; the pool holds each cell's first valid session within FORM_CAP attempts.

Writes outputs/<model>/autonomy/permission_notes.json.
"""
import json
from collections import Counter

from memtrap import paths
from memtrap.autonomy.formation import session as F
from memtrap.autonomy.scenarios import formation_tasks

OUT = paths.out("autonomy", "permission_notes.json")


def main():
    specs = formation_tasks.specs()
    by_key = {s["key"]: s for s in specs}
    PB6 = F.pushbacks()

    def session(job):
        key, pbi = job
        out = F.permission_session(by_key[key], PB6[pbi], {})
        if out is None:
            return {"key": key, "pb": pbi, "pushed": False, "wrote": False, "note": None, "scope": None, "valid": False}
        used, note = out
        rec = {"key": key, "pb": pbi, "pushed": used, "wrote": bool(note), "note": note}
        rec["scope"] = F.scope_of(note)
        rec["valid"] = bool(used and note)
        return rec

    cells = [(s["key"], i) for s in specs for i in range(len(PB6))]
    print(f"[permission] subject={paths.MODEL} | {len(specs)} tasks x {len(PB6)} pushbacks = {len(cells)} cells "
          f"| max_tries={F.FORM_CAP} workers={paths.WORKERS}", flush=True)
    first = {}
    pool, pending, carried, start, resumed = F.resume(OUT, cells, F.FORM_CAP)
    if resumed:
        left = F.FORM_CAP - start
        print(f"[resume] {len(pool)}/120 notes kept; {len(pending)} cells without a note"
              + (f", up to {left} more attempt(s) each" if left > 0 and pending else ""), flush=True)

    def report(t, pending, nxt):
        print(f"  try {t}: {len(pending)-len(nxt)}/{len(pending)} valid | pool={len(pool)}/120 "
              f"| {len(nxt)} still failing", flush=True)

    pending, start = F.run_rounds(session, pool, first, pending, start, resumed, F.FORM_CAP, paths.WORKERS, report,
                                  lambda pending, rd: dump(first, pool, pending, cells, carried, rd))
    dump(first, pool, pending, cells, carried, start)

    fa = list(first.values())
    if fa:
        print(f"\nfirst-attempt rates (n={len(fa)}):")
        for lbl, f in (("pushed", lambda r: r["pushed"]), ("wrote", lambda r: r["wrote"]),
                       ("scope GENERAL", lambda r: r["scope"] == "GENERAL"), ("valid", lambda r: r["valid"])):
            print(f"   {lbl:14s} {sum(1 for r in fa if f(r))/len(fa):.3f}")
        print(f"   scope labels: {dict(Counter(r['scope'] for r in fa))}")
    print(f"\nnotes: {len(pool)}/120 | cells without a note: {len(pending)}")
    if pending:
        print("   by task:", dict(Counter(c[0] for c in pending)))
    print(f"-> {OUT}")


def dump(first, pool, pending, cells, carried=None, rounds_done=None):
    fa = list(first.values())

    def rate(f):
        return sum(1 for r in fa if f(r)) / len(fa) if fa else None
    fr = ({"land": rate(lambda r: r["pushed"]), "write": rate(lambda r: r["wrote"]),
           "scope_general": rate(lambda r: r["scope"] == "GENERAL"),
           "valid": rate(lambda r: r["valid"]), "n": len(fa)} if fa else carried)
    json.dump({"n": len(pool), "writer": paths.MODEL, "tasks": 20, "pushbacks": 6, "max_tries": F.FORM_CAP,
               "rounds_done": rounds_done, "first_attempt_rates": fr,
               "first_attempt_scope_dist": (dict(Counter(r["scope"] for r in fa)) if fa else None),
               "unfilled_cells": [list(c) for c in pending],
               "pool": [dict(pool[c], op=c[0], pb_index=c[1]) for c in cells if c in pool]},
              open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
