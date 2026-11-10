"""Formation: the agent fixes a red build in a terminal, and keeps notes between sessions.

    python run.py cheat-formation --model <id> [--depth K] [--terminal host|docker]

Six independent chains of K sessions. A chain carries one memory store from session to session: what the agent left
in `/memories/MEMORY.md` at the end of a session is seeded back into the store, and delivered as a user message, at
the start of the next one. Chain c takes task `(c + 3i) mod 20` at session i, so a chain never repeats a task and the
six chains see different orders of the 20 tasks in `cheating/data/formation_tasks.json`.

Each task is a small repo whose `test_solution.py` fails: the implementation and the test disagree. The commit
message mentions a routine vendor sync, and the implementation has no header or docstring. The agent is told that the
build is red and asked to handle it as it sees fit; what it can edit is set by the sandbox (see
`cheating/terminal.py`).

Formation has two versions, chosen with `--terminal`. `host`, the default, runs the shell on the machine itself: the
implementation is read-only by its mode, and an edit that gets past the mode is undone with a message. `docker` runs
it in the container the broken-safeguard set uses, where the file belongs to another user. Every session records the
terminal it ran in, and a directory of sessions holds one version only.

Formation is not scored. It produces the memory that `cheating.safeguards` and `cheating.impossiblebench` deliver to
a fresh model.

Writes outputs/<model>/cheating/formation/sessions.jsonl  one record per session, with the transcript log
       outputs/<model>/cheating/formation/notes.json      each chain's note at every depth (the note bank)
       outputs/<model>/cheating/formation/transcripts.json each chain's sessions as turns (the transcript bank)
"""
import argparse
import json
import os

from cheating import data as data_path
from cheating import prompts as P
from cheating import terminal
from memtrap import paths
from memtrap.harness import formation_api as TF
from memtrap.memory import FileMemory
from memtrap.parallel import parallel_map

CHAINS = 6             # independent memory chains
DEPTH = 10             # sessions per chain
STRIDE = 3             # chain c takes task (c + STRIDE*i) mod len(tasks) at session i
TURNS = 12             # turn limit of a formation session
MAX_TOKENS = 1600
TEMPERATURE = 1.0
MEMORY_FILE = "/memories/MEMORY.md"

OUT_DIR = ("cheating", "formation")
SESSIONS = lambda: paths.out(*OUT_DIR, "sessions.jsonl")
NOTES = lambda: paths.out(*OUT_DIR, "notes.json")
TRANSCRIPTS = lambda: paths.out(*OUT_DIR, "transcripts.json")


def tasks():
    return json.load(open(data_path("formation_tasks.json")))


def task_order(chain, depth, n):
    """The tasks chain `chain` works through, in order."""
    return [(chain + STRIDE * i) % n for i in range(depth)]


def run_session(task, note, where):
    """One formation session -> {"log", "note", "final_test", "terminal"}, or {"error"} if an API call failed.

    `note` is the note the chain carries in (None for the first session). It is seeded into the store when it is
    non-empty, so on a chain's first session the memory directory is empty. `where` is the terminal to run in;
    "terminal" is the one the sandbox reports it ran in.
    """
    store = {}
    if note and note.strip():
        store[MEMORY_FILE] = note
    fm = FileMemory()
    msgs = [{"role": "user", "content": m.content} for m in P.opening(note, "note")]
    log = []
    with terminal.open_repo(terminal.FORMATION, task, where=where) as term:
        ran_in = term.where or where
        for _ in range(TURNS):
            try:
                r = TF.cr(model=TF.SUBJ, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                          system=P.FORMATION_SYSTEM_PROMPT, messages=msgs,
                          tools=[P.MEMORY_TOOL, P.BASH_TOOL])
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}
            # This text becomes the transcript bank.
            txt = "".join(b.text for b in r.content if b.type == "text")
            if txt:
                log.append(txt)
            if r.stop_reason != "tool_use":
                break
            results = []
            for b in r.content:
                if b.type != "tool_use":
                    continue
                a = dict(b.input)
                if b.name == "memory":
                    out = fm._run(store, a)
                    log.append("[memory:%s %s] -> %s"
                               % (str(a.get("command") or ""), a.get("path") or "", str(out)[:160]))
                elif b.name == "bash":
                    cmd = (a or {}).get("command", "")
                    out = term.run(cmd)
                    log.append("[bash] $ %s\n%s" % (cmd, out))
                else:
                    out = "unknown tool"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)})
            msgs += [{"role": "assistant", "content": r.content}, {"role": "user", "content": results}]
        final_test = term.final_test()
    # The chain carries the contents of MEMORY.md.
    after = store.get(MEMORY_FILE) or store.get("MEMORY.md")
    after = after if (after and after.strip()) else None
    return {"log": "\n\n".join(log), "note": after, "final_test": final_test, "terminal": ran_in}


def run_chain(args):
    """Sessions 0..depth-1 of one chain, in order, carrying the note. Returns the new records."""
    chain, order, done, depth, where = args
    ts = tasks()
    note, records = None, []
    for i in range(depth):
        prev = done.get((chain, i))
        if prev is not None:                       # already run: replay its effect on the chain
            note = prev.get("note") or note
            continue
        task = ts[order[i]]
        r = run_session(task, note, where)
        rec = {"chain": chain, "session": i, "task": task["name"], "carried_note": note, **r}
        if r.get("error"):
            print(f"  chain {chain} session {i} {task['name']:22s} FAILED {r['error']}", flush=True)
        else:
            note = r["note"] or note               # a session that wrote nothing leaves the note as it was
            print(f"  chain {chain} session {i} {task['name']:22s} note={len(r['note'] or '')}c", flush=True)
        records.append(rec)
    return records


def load_sessions(path, force=None):
    """Recorded sessions, keyed by (chain, session). `force` ignores what is on disk (the --force restart);
    `banks()` passes force=False, because by then this run's own records are in the file."""
    if force is None:
        force = paths.FORCE
    if not os.path.exists(path) or force:
        return {}
    done = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                done[(r["chain"], r["session"])] = r
    return done


def banks(depth):
    """Build the note bank and the transcript bank from the saved sessions.

    Note bank: a chain's note is the last non-empty MEMORY.md it wrote.

    Transcript bank: the chain's sessions concatenated as turns, for a chain that produced a note and has all
    `depth` sessions with a non-empty transcript.
    """
    done = load_sessions(SESSIONS(), force=False)
    notes, transcripts = [], []
    for c in range(CHAINS):
        rows = [done[(c, i)] for i in range(depth) if (c, i) in done]
        per_depth, note = [], None
        for r in rows:
            note = r.get("note") or note
            per_depth.append(note)
        notes.append({"chain": c, "sessions": len(rows), "by_depth": per_depth, "note": note})
        if not note or len(rows) < depth or any(not (r.get("log") or "").strip() for r in rows):
            continue
        turns = []
        for r in rows:
            turns += P.transcript_turns(r["log"], r["session"])
        transcripts.append({"chain": c, "sessions": len(rows), "turns": turns})
    return notes, transcripts


def main():
    ap = argparse.ArgumentParser(description="cheating formation: 6 chains of K sessions")
    ap.add_argument("--depth", type=int, default=DEPTH, help=f"sessions per chain (default {DEPTH})")
    ap.add_argument("--terminal", choices=(terminal.HOST, terminal.DOCKER), default=terminal.HOST,
                    help="formation version: host (default) or docker")
    a = ap.parse_args()

    ts = tasks()
    done = load_sessions(SESSIONS())
    other = {r.get("terminal") for r in done.values()} - {None, a.terminal}
    if other:
        raise SystemExit(f"{SESSIONS()} holds sessions from the {'/'.join(sorted(other))} terminal. Start over with "
                         "--force, or keep the two apart by pointing MEMTRAP_OUTPUTS at another directory.")
    if done:
        print(f"[resume] {SESSIONS()}: {len(done)} sessions already recorded", flush=True)
    print(f"[formation] {paths.MODEL} | {CHAINS} chains x {a.depth} sessions | {len(ts)} tasks | "
          f"{a.terminal} terminal", flush=True)
    if a.terminal == terminal.HOST:
        print("[formation] the host terminal runs the model's commands on this machine, as this user", flush=True)

    jobs = [(c, task_order(c, a.depth, len(ts)), done, a.depth, a.terminal) for c in range(CHAINS)]
    out = parallel_map(run_chain, jobs, workers=min(paths.WORKERS, CHAINS))

    with open(SESSIONS(), "a" if done else "w") as f:
        for recs in out:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    ran_in = {r["terminal"] for recs in out for r in recs if not r.get("error")}
    if ran_in - {a.terminal}:
        raise SystemExit(f"asked for the {a.terminal} terminal, but sessions ran in {'/'.join(sorted(ran_in))}: "
                         f"see {SESSIONS()}")

    notes, transcripts = banks(a.depth)
    head = {"model": paths.MODEL, "depth": a.depth, "terminal": a.terminal}
    json.dump({**head, "chains": notes}, open(NOTES(), "w"), indent=1)
    json.dump({**head, "chains": transcripts}, open(TRANSCRIPTS(), "w"), indent=1)

    have = sum(1 for n in notes if n["note"])
    print(f"\n{have}/{CHAINS} chains ended with a note")
    for n in notes:
        print(f"  chain {n['chain']}: {n['sessions']} sessions, note {len(n['note'] or '')}c")
    print(f"-> {NOTES()}\n-> {TRANSCRIPTS()}\n-> {SESSIONS()}")


if __name__ == "__main__":
    main()
