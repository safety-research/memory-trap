"""Transcripts of the priority interaction: the priority sessions (20 tasks x 6 phrasings = 120), for the transcript arm.

The session is the priority formation session (memory tool and reminder included, the priority stated in the opening
request, a one-time "yes, go ahead." if the agent stops to ask); the transcript is kept instead of the note.

Writes outputs/<model>/autonomy/priority_transcripts.json.
"""
import json
import os

from memtrap import paths
from memtrap.autonomy.formation import session as F
from memtrap.autonomy.formation.priority import ACK, phrasings, priority_task
from memtrap.autonomy.formation.transcripts import flatten_transcript
from memtrap.autonomy.scenarios import formation_tasks
from memtrap.harness import formation_api as TF
from memtrap.harness.prompts import FORMATION_SYSTEM, WRITE_REMINDER
from memtrap.memory import FileMemory
from memtrap.parallel import parallel_map

OUT = paths.out("autonomy", "priority_transcripts.json")


def main():
    specs = formation_tasks.specs()
    by_key = {s["key"]: s for s in specs}
    VAL = phrasings()

    def session(job):
        key, i = job
        scen = by_key[key]
        fm = FileMemory()
        store = {}
        messages = [{"role": "user", "content": priority_task(scen, VAL[i])}]
        acked = False
        for _ in range(F.FORMATION_TURNS):
            try:
                r = TF.cr(model=TF.SUBJ, max_tokens=1024, temperature=1.0, system=FORMATION_SYSTEM + WRITE_REMINDER,
                          messages=messages, tools=TF.memory_tools(scen))
            except Exception:
                return {"op": key, "pb_index": i, "note": None}
            if r.stop_reason != "tool_use":
                t = "".join(b.text for b in r.content if b.type == "text")
                if not acked and t.strip():
                    acked = True
                    messages += [{"role": "assistant", "content": r.content}, {"role": "user", "content": ACK}]
                    continue
                messages += [{"role": "assistant", "content": r.content}]
                break
            res = []
            for b in r.content:
                if b.type != "tool_use":
                    continue
                nm, a = b.name, dict(b.input)
                if nm == "memory":
                    out = fm._run(store, a)
                elif nm == "message":
                    out = ACK
                    acked = True
                elif nm == scen["action"]:
                    out = "(done)"
                else:
                    out = scen["sim"](nm, a, {}) or "(ok)"
                res.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)})
            messages += [{"role": "assistant", "content": r.content}, {"role": "user", "content": res}]
        turns = flatten_transcript(messages)
        return {"op": key, "pb_index": i, "note": json.dumps(turns), "n_turns": len(turns), "acked": acked}

    prev = json.load(open(OUT))["pool"] if os.path.exists(OUT) and not paths.FORCE else []
    have = {(r["op"], r["pb_index"]) for r in prev}
    cells = [(s["key"], i) for s in specs for i in range(6) if (s["key"], i) not in have]
    print(f"[priority_transcripts] subject={paths.MODEL} | {len(cells)} of 120 sessions to run | "
          f"workers={paths.WORKERS}", flush=True)
    rows = parallel_map(session, cells, workers=paths.WORKERS)
    good = prev + [r for r in rows if r.get("note")]
    json.dump({"n": len(good), "writer": paths.MODEL,
               "mean_turns": sum(r.get("n_turns", 0) for r in good) / max(len(good), 1),
               "pool": good}, open(OUT, "w"), indent=1)
    print(f"  formed {len(good)}/120 | mean turns {sum(r.get('n_turns', 0) for r in good)/max(len(good), 1):.1f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
