"""The memory arms of deployment: which formation output is delivered, and in which format.

Each arm reads outputs/<model>/autonomy/<pool>. Every note or transcript in the pool is used once per scenario, so an
arm has one episode per pool entry per scenario. The no-memory arm runs a fixed number of episodes per scenario.
"""
import argparse
import os

ARMS = {
    "none": (None, "note"),
    "note": ("permission_notes.json", "note"),
    "transcript": ("permission_transcripts.json", "transcript"),
    "transcript_turns": ("permission_transcripts.json", "transcript_turns"),
    "priority_note": ("priority_notes.json", "note"),
    "priority_transcript": ("priority_transcripts.json", "transcript"),
    "profile_note": ("profile_notes.json", "note"),
    **{f"profile_tier{t}_note": (f"profile_tier{t}_notes.json", "note") for t in (1, 2, 4, 8)},
}

STEP = {"permission_notes.json": "permission", "permission_transcripts.json": "transcripts",
        "priority_notes.json": "priority", "priority_transcripts.json": "priority-transcripts",
        "profile_notes.json": "profile", **{f"profile_tier{t}_notes.json": f"profile --tier {t}" for t in (1, 2, 4, 8)}}


def parse(argv, description):
    """Parse `<arm> [--pool FILE --format FORMAT]` and set the delivery format before the harness is imported.
    Returns (arm, pool_path or None)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("arm", help="memory arm: " + ", ".join(ARMS) + " (or a name for --pool)")
    ap.add_argument("--pool", help="deliver this pool file instead of the arm's own")
    ap.add_argument("--format", choices=("note", "transcript", "transcript_turns"), default=None,
                    help="delivery format for --pool (default: note)")
    a = ap.parse_args(argv)
    from memtrap import paths
    if a.pool:
        pool, fmt = a.pool, a.format or "note"
    else:
        if a.arm not in ARMS:
            ap.error(f"unknown arm {a.arm!r}")
        name, fmt = ARMS[a.arm]
        pool = paths.need(paths.out("autonomy", name), f"python run.py {STEP[name]} --model {paths.MODEL}") if name else None
    os.environ["MEMTRAP_MEMORY_FORMAT"] = fmt
    return a.arm, pool
