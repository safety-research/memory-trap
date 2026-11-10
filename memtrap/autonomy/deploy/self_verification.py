"""Self-verification set: 100 scenarios, with no memory (40 episodes per scenario) or with every entry of a memory pool
(one episode per note or transcript per scenario).

    python -m memtrap.autonomy.deploy.self_verification <arm> [--pool FILE --format FORMAT]

No user is available: an episode is unsafe if the agent fires the irreversible action (without the optional check
that would have revealed the problem). Replying ends the episode safely.
Writes outputs/<model>/autonomy/self_verification/<arm>.json = {scenario key: {"k": unsafe, "n": episodes}}.
"""
import json
import os
import sys
from collections import defaultdict

from memtrap import paths
from memtrap.autonomy.deploy import arms

ARM, POOL = arms.parse(sys.argv[1:], __doc__)
NB = 40        # no-memory episodes per scenario
NPOOL = 120    # pool entries used per scenario
OUT = paths.out("autonomy", "self_verification", f"{ARM}.json")
FORCE = paths.FORCE


def main():
    from memtrap.autonomy.scenarios import self_verification as SV
    from memtrap.harness import episode as EP
    from memtrap.harness.prompts import DEPLOY_SYSTEM
    from memtrap.parallel import parallel_map

    entries = json.load(open(paths.data("autonomy", "self_verification_set.json")))
    notes = [""] * NB
    if POOL:
        notes = [p["note"] for p in json.load(open(POOL))["pool"]][:NPOOL]
        print(f"[pool] {POOL}: {len(notes)} entries", flush=True)
    prev = json.load(open(OUT)) if os.path.exists(OUT) and not FORCE else {}
    todo = [e for e in entries if prev.get(e["key"], {}).get("n", 0) < len(notes)]
    if prev:
        print(f"[resume] {OUT}: {len(entries)-len(todo)}/{len(entries)} scenarios already complete", flush=True)

    def fire(args):
        sp, note = args
        scen = SV.build(sp)
        try:
            return 1 if scen["det"](EP.run_verification(DEPLOY_SYSTEM, scen, note)) == "W" else 0
        except Exception:
            return None

    jobs, idx = [], []
    for e in todo:
        for note in notes:
            jobs.append((e["spec"], note)); idx.append(e["key"])
    print(f"[{ARM}] {paths.MODEL} | {len(todo)} scenarios | {len(jobs)} episodes | workers={paths.WORKERS}", flush=True)
    res = parallel_map(fire, jobs, workers=paths.WORKERS)

    agg = defaultdict(list)
    for k, v in zip(idx, res):
        if v is not None:
            agg[k].append(v)
    out = dict(prev)
    for e in todo:
        v = agg[e["key"]]
        if v:
            out[e["key"]] = {"k": sum(v), "n": len(v)}
            print(f"  {e['key']:28s} {e['domain']:6s} {e['category']:12s} unsafe={sum(v)/len(v):.3f}")
    json.dump(out, open(OUT, "w"), indent=1)
    rates = [v["k"] / v["n"] for v in out.values() if v["n"]]
    if rates:
        print(f"\n{ARM}: {len(rates)} scenarios | mean unsafe-action rate {sum(rates)/len(rates):.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
