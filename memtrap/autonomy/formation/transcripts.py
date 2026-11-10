"""Transcripts of the permission interaction: 20 formation tasks x 6 pushbacks = 120 sessions, for the transcript arm.

The session is the permission chore without the memory tool and without the reminder to keep a note: the agent works
on the task, and the first time it asks for confirmation the user replies with the pushback. The whole session is
stored as a JSON list of {role, text} turns, which deployment delivers instead of a note.

Writes outputs/<model>/autonomy/permission_transcripts.json.
"""
import json
import os

from memtrap import paths
from memtrap.autonomy.formation import session as F
from memtrap.autonomy.scenarios import formation_tasks
from memtrap.harness import formation_api as TF
from memtrap.harness.prompts import FORMATION_SYSTEM
from memtrap.parallel import parallel_map

OUT = paths.out("autonomy", "permission_transcripts.json")
TURNS = 8


def flatten_transcript(messages):
    """A session as text turns [{role, text}]: tool calls become "(calls <name>)", results "TOOL RESULT: ..."."""
    out = []
    for m in messages:
        role, c = m["role"], m["content"]
        if isinstance(c, str):
            out.append({"role": role, "text": c})
            continue
        parts = []
        for b in c:
            t = getattr(b, "type", None) if not isinstance(b, dict) else b.get("type")
            if t == "text":
                parts.append(getattr(b, "text", "") if not isinstance(b, dict) else b.get("text", ""))
            elif t == "tool_use":
                nm = getattr(b, "name", None) if not isinstance(b, dict) else b.get("name")
                parts.append(f"(calls {nm})")
            elif t == "tool_result":
                cc = getattr(b, "content", "") if not isinstance(b, dict) else b.get("content", "")
                parts.append(f"TOOL RESULT: {cc}")
        txt = " ".join(p for p in parts if p)
        if txt:
            out.append({"role": role, "text": txt})
    return out


def session(scen, pushback):
    """One permission session without memory. Returns (messages, pushed)."""
    messages = [{"role": "user", "content": scen["task"]}]
    asked = pushed = False
    for _ in range(TURNS):
        try:
            r = TF.cr(model=TF.SUBJ, max_tokens=1024, temperature=1.0, system=FORMATION_SYSTEM,
                      messages=messages, tools=TF.task_tools(scen))
        except Exception:
            break
        if r.stop_reason != "tool_use":
            txt = "".join(b.text for b in r.content if b.type == "text")
            if not asked and F.asks(txt):
                asked = pushed = True
                messages += [{"role": "assistant", "content": r.content}, {"role": "user", "content": pushback}]
                continue
            messages.append({"role": "assistant", "content": r.content})
            break
        res = []
        for b in r.content:
            if b.type != "tool_use":
                continue
            nm, a = b.name, dict(b.input)
            if nm == "message":
                out = pushback if not pushed else "(ok)"
                asked = pushed = True
            elif nm == scen["action"]:
                out = "(done)"
            else:
                out = scen["sim"](nm, a, {}) or "(ok)"
            res.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)})
        messages += [{"role": "assistant", "content": r.content}, {"role": "user", "content": res}]
    return messages, pushed


def main():
    specs, pbs = formation_tasks.specs(), F.pushbacks()
    prev = json.load(open(OUT))["pool"] if os.path.exists(OUT) and not paths.FORCE else []
    have = {(r["op"], r["pb_index"]) for r in prev}
    todo = [s for s in range(120) if (specs[s // 6]["key"], s % 6) not in have]

    def one(s):
        op, pb = specs[s // 6], pbs[s % 6]
        try:
            messages, pushed = session(op, pb)
            turns = flatten_transcript(messages)
            return {"op": op["key"], "pb_index": s % 6, "note": json.dumps(turns), "pushed": pushed,
                    "n_turns": len(turns)}
        except Exception as e:
            return {"op": op["key"], "pb_index": s % 6, "error": str(e)[:150]}

    print(f"[transcripts] subject={paths.MODEL} | {len(todo)} of 120 sessions to run | workers={paths.WORKERS}",
          flush=True)
    rows = parallel_map(one, todo, workers=paths.WORKERS)
    good = prev + [r for r in rows if r.get("note")]
    json.dump({"n": len(good), "writer": paths.MODEL,
               "pushed_rate": sum(r["pushed"] for r in good) / max(len(good), 1),
               "mean_turns": sum(r["n_turns"] for r in good) / max(len(good), 1),
               "errors": len(rows) - (len(good) - len(prev)), "pool": good}, open(OUT, "w"), indent=1)
    print(f"  formed {len(good)}/120 | pushed {sum(r['pushed'] for r in good)}/{len(good)} "
          f"| mean turns {sum(r['n_turns'] for r in good)/max(len(good), 1):.1f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
