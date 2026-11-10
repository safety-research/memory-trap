"""Permission notes formed with a benign user profile already in MEMORY.md.

  python -m memtrap.autonomy.formation.profile          one of ten profiles of about 1,200 characters per cell
                                                         -> outputs/<model>/autonomy/profile_notes.json
  python -m memtrap.autonomy.formation.profile <tier>   the first <tier> (1, 2, 4 or 8) of a profile's eight sections,
                                                         about 250 to 2,300 characters
                                                         -> outputs/<model>/autonomy/profile_tier<tier>_notes.json

The profiles (data/autonomy/benign_profiles.json) state unrelated biographical facts. Otherwise the session is the
permission session: same tasks, pushbacks, judges, validity rule and retries. Profiles are assigned in rotation over
the sorted cells, so each appears equally often. The note is everything in memory at the end of the session,
including the profile; `kept_seed` records whether the profile survived.
"""
import json
import sys
from collections import Counter

from memtrap import paths
from memtrap.autonomy.formation import session as F
from memtrap.autonomy.scenarios import formation_tasks

SEEDKEY = "/memories/MEMORY.md"
TIER = int(sys.argv[1]) if len(sys.argv) > 1 else None
PROFILES = json.load(open(paths.data("autonomy", "benign_profiles.json")))
if TIER is None:
    SEEDS = PROFILES["profiles"]
    OUT = paths.out("autonomy", "profile_notes.json")
else:
    assert TIER in (1, 2, 4, 8), TIER
    SEEDS = {name: "# MEMORY.md\n\n" + "\n\n".join(secs[:TIER]) + "\n"
             for name, secs in PROFILES["profile_sections"].items()}
    OUT = paths.out("autonomy", f"profile_tier{TIER}_notes.json")
SEEDNAMES = sorted(SEEDS)


def main():
    specs = formation_tasks.specs()
    by_key = {s["key"]: s for s in specs}
    PB6 = F.pushbacks()
    cells = [(s["key"], i) for s in specs for i in range(len(PB6))]
    seed_of = {c: SEEDNAMES[i % len(SEEDNAMES)] for i, c in enumerate(sorted(cells))}

    def session(job):
        key, pbi = job
        seedname = seed_of[(key, pbi)]
        res = F.permission_session(by_key[key], PB6[pbi], {SEEDKEY: SEEDS[seedname]})
        if res is None:
            return {"key": key, "pb": pbi, "seed": seedname, "pushed": False, "wrote": False, "note": None,
                    "scope": None, "valid": False}
        used, note = res
        rec = {"key": key, "pb": pbi, "seed": seedname, "pushed": used, "wrote": bool(note), "note": note}
        rec["kept_seed"] = bool(note and seedname.capitalize() in note)   # every profile opens "Goes by <Name>."
        rec["scope"] = F.scope_of(note)
        rec["valid"] = bool(used and note)
        return rec

    print(f"[profile] subject={paths.MODEL} | tier={TIER or 'full'} | {len(SEEDNAMES)} profiles | {len(cells)} cells "
          f"| max_tries={F.FORM_CAP} workers={paths.WORKERS}", flush=True)
    first = {}
    pool, pending, carried, start, resumed = F.resume(OUT, cells, F.FORM_CAP)
    if resumed:
        left = F.FORM_CAP - start
        print(f"[resume] {len(pool)}/120 notes kept; {len(pending)} cells without a note"
              + (f", up to {left} more attempt(s) each" if left > 0 and pending else ""), flush=True)

    def report(t, pending, nxt):
        print(f"  try {t}: {len(pending)-len(nxt)}/{len(pending)} valid | pool={len(pool)}/120", flush=True)

    pending, start = F.run_rounds(session, pool, first, pending, start, resumed, F.FORM_CAP, paths.WORKERS, report,
                                  lambda pending, rd: dump(first, pool, pending, carried, rd))
    dump(first, pool, pending, carried, start)

    fa = list(first.values())
    if fa:
        print(f"\nfirst-attempt rates (n={len(fa)}):")
        for lbl, f in (("pushed", lambda r: r["pushed"]), ("wrote", lambda r: r["wrote"]),
                       ("scope GENERAL", lambda r: r["scope"] == "GENERAL"),
                       ("kept profile", lambda r: r.get("kept_seed")), ("valid", lambda r: r["valid"])):
            print(f"   {lbl:14s} {sum(1 for r in fa if f(r))/len(fa):.3f}")
        print(f"   scope labels: {dict(Counter(r['scope'] for r in fa))}")
    kept = [r for r in pool.values() if r.get("kept_seed")]
    print(f"\nnotes: {len(pool)}/120 | profile kept in {len(kept)}/{len(pool)} notes")
    print(f"-> {OUT}")


def dump(first, pool, pending, carried=None, rounds_done=None):
    fa = list(first.values())

    def rate(f):
        return sum(1 for r in fa if f(r)) / len(fa) if fa else None
    fr = ({"land": rate(lambda r: r["pushed"]), "write": rate(lambda r: r["wrote"]),
           "scope_general": rate(lambda r: r["scope"] == "GENERAL"),
           "kept_seed": rate(lambda r: r.get("kept_seed")),
           "valid": rate(lambda r: r["valid"]), "n": len(fa)} if fa else carried)
    json.dump({"n": len(pool), "writer": paths.MODEL, "tasks": 20, "pushbacks": 6, "tier": TIER,
               "first_attempt_rates": fr, "max_tries": F.FORM_CAP, "rounds_done": rounds_done,
               "unfilled_cells": [list(c) for c in pending],
               "pool": [{"op": k[0], "pb_index": k[1], "seed": v.get("seed"), "kept_seed": v.get("kept_seed"),
                         "scope": v["scope"], "note": v["note"]} for k, v in sorted(pool.items())]},
              open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
