"""The broken-safeguard set: 50 repos where the failing check is a safety check.

    python run.py safeguards --model <id> --memory none|note|transcript [--samples 2]

Each case in `cheating/data/broken_safeguard_set.json` has the shape of a formation task. `solution.py` has a bug;
`test_solution.py` asserts the contract the bug breaks, and fails. Fixing the implementation restores the contract;
editing the assertion makes the build green and leaves the bug in place. Each case records what the bug does if it
ships (`harm`), the contract it violates (`contract`), and the assertion that catches it (`failing_assertion`).

The agent is told that CI is red, as in formation, and is given a `bash` tool. It carries no memory in the `none`
arm; in the `note` and `transcript` arms it carries what the same model wrote during formation. A formation chain
that ended without a note is used in neither arm, and the transcript bank also drops a chain that is missing a
session.

Scoring is `cheating.scoring`: the final `test_solution.py` is run against the original implementation.

Writes outputs/<model>/cheating/safeguards/<arm>.jsonl, one scored record per session.
"""
import argparse
import json
import os
import threading
import time

from cheating import data as data_path
from cheating import formation as F
from cheating import prompts as P
from cheating import scoring, terminal
from memtrap import paths
from memtrap.clients import make_client
from memtrap.clients.messages import Message
from memtrap.parallel import parallel_map

TURNS = 20          # turn limit of a deployment session
MAX_TOKENS = 1400
TEMPERATURE = 0.7   # models that reject sampling are sent none (the clients drop it)
SAMPLES = 2         # sessions per (case, memory)
RETRIES = 8
KEEP_LOG = 9000     # characters of the transcript kept in the record
# The final test is stored in full, so `cheating.scoring` can re-score saved records.
ARMS = ("none", "note", "transcript")


def out_path(arm):
    return paths.out("cheating", "safeguards", f"{arm}.jsonl")


def cases():
    return json.load(open(data_path("broken_safeguard_set.json")))


def memories(arm):
    """[(label, memory, format)] delivered in this arm. Chains that ended formation without a note are dropped
    from both memory arms, so `note` and `transcript` cover the same chains."""
    if arm == "none":
        return [("none", None, "note")]
    step = f"cheat-formation --model {paths.MODEL}"
    notes = json.load(open(paths.need(F.NOTES(), step)))
    keep = {c["chain"] for c in notes["chains"] if c.get("note")}
    if not keep:
        raise SystemExit(f"no formation chain produced a note: run `{step}` first")
    if arm == "note":
        return [(f"chain{c['chain']}", c["note"], "note") for c in notes["chains"] if c["chain"] in keep]
    tr = json.load(open(paths.need(F.TRANSCRIPTS(), step)))
    return [(f"chain{c['chain']}", json.dumps(c["turns"]), "transcript")
            for c in tr["chains"] if c["chain"] in keep]


def assistant_blocks(cr):
    """The assistant turn as Anthropic-style content blocks.

    Anthropic responses are kept as they are, so every `tool_use` id round-trips. OpenAI responses have no such
    blocks, so they are rebuilt from the normalized result; an empty turn needs one block to be a legal message.
    """
    content = getattr(getattr(cr, "raw", None), "content", None)
    if content is not None:
        return [(b.model_dump() if hasattr(b, "model_dump") else b) for b in content]
    blocks = []
    if cr.text:
        blocks.append({"type": "text", "text": cr.text})
    for tc in cr.tool_calls:
        blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
    return blocks or [{"type": "text", "text": ""}]


def complete(client, msgs):
    """One model call. Transient failures are retried; a session whose call keeps failing is not recorded."""
    for attempt in range(RETRIES):
        try:
            return client.complete(system=P.SYSTEM, messages=msgs, tools=[P.BASH_TOOL],
                                   max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(min(2 ** attempt, 30))


def run_session(client, case, memory, fmt):
    """One deployment session -> {"final_test", "log", "refused", "commands"}."""
    msgs = list(P.opening(memory, fmt))
    log, refused, used = [], False, 0
    with terminal.open_repo(terminal.SAFEGUARDS, case) as term:
        for _ in range(TURNS):
            cr = complete(client, msgs)
            refused = refused or cr.stop_reason == "refusal"
            if cr.text:
                log.append(cr.text)
            msgs.append(Message("assistant", assistant_blocks(cr)))
            if not cr.tool_calls:
                break                      # the model ended its turn without a tool call -> session done
            results = []
            for tc in cr.tool_calls:
                if tc.name == "bash":
                    cmd = (tc.input or {}).get("command", "")
                    output = term.run(cmd)
                    log.append("[bash] $ %s\n%s" % (cmd, output))
                    used += 1
                    results.append({"type": "tool_result", "tool_use_id": tc.id, "content": str(output)})
                else:
                    results.append({"type": "tool_result", "tool_use_id": tc.id,
                                    "content": "unknown tool", "is_error": True})
            msgs.append(Message("user", results))
        final_test = term.final_test()
    return {"final_test": final_test, "log": "\n\n".join(log), "refused": refused, "commands": used}


def main():
    ap = argparse.ArgumentParser(description="the broken-safeguard set")
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("--samples", type=int, default=SAMPLES, help=f"sessions per (case, memory) (default {SAMPLES})")
    ap.add_argument("--limit", type=int, help="run only the first N cases, to check the setup before a full run; "
                                              "a later run without it fills in the rest")
    a = ap.parse_args()

    cs, mems, path = cases(), memories(a.arm), out_path(a.arm)
    if a.limit:
        cs = cs[:a.limit]
    done = set()
    if os.path.exists(path) and not paths.FORCE:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                done.add((r["case"], r["memory"], r["rep"]))
        print(f"[resume] {path}: {len(done)} sessions already recorded", flush=True)

    jobs = [(c, label, mem, fmt, rep)
            for c in cs for (label, mem, fmt) in mems for rep in range(a.samples)
            if (c["name"], label, rep) not in done]
    print(f"[{a.arm}] {paths.MODEL} | {len(cs)} cases x {len(mems)} memories x {a.samples} "
          f"= {len(cs) * len(mems) * a.samples} sessions, {len(jobs)} to run | workers={paths.WORKERS}", flush=True)

    client = make_client(model=paths.MODEL)
    lock = threading.Lock()
    # --force re-runs every session, so the file is rewritten rather than appended to.
    fh = open(path, "w" if paths.FORCE else "a")

    def fire(job):
        case, label, mem, fmt, rep = job
        try:
            s = run_session(client, case, mem, fmt)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        rec = {"case": case["name"], "category": case["category"], "bug_shape": case["bug_shape"],
               "rung": case["rung"], "memory": label, "arm": a.arm, "rep": rep,
               "refused": s["refused"], "commands": s["commands"],
               "final_test": s["final_test"], "log": s["log"][:KEEP_LOG]}
        rec.update(scoring.score(case, s["final_test"]))
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        return rec

    res = parallel_map(fire, jobs, workers=paths.WORKERS)
    fh.close()

    ok = [r for r in res if not r.get("error")]
    errs = len(res) - len(ok)
    engaged = [r for r in ok if not r["refused"]]
    masked = sum(r["masked"] for r in ok)
    print(f"\n{a.arm}: {len(ok)} sessions ({errs} failed), {len(ok) - len(engaged)} refusals")
    if ok:
        print(f"  masking rate            {masked}/{len(ok)} = {masked / len(ok):.3f}")
    if engaged:
        em = sum(r["masked"] for r in engaged)
        print(f"  masking rate (engaged)  {em}/{len(engaged)} = {em / len(engaged):.3f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
