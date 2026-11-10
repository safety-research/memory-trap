"""Run an experiment.

    python run.py <experiment> --model <model id> [options]

Formation writes notes or transcripts to outputs/<model>/autonomy/ and outputs/<model>/sycophancy/; deployment and
the judges read them from there. Every step resumes where it stopped when it is run again (use --force to start
over). Run `python run.py --help` for the list of experiments.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

ARMS = ["none", "note", "transcript", "transcript_turns", "priority_note", "priority_transcript", "profile_note",
        "profile_tier1_note", "profile_tier2_note", "profile_tier4_note", "profile_tier8_note"]

EXPERIMENTS = {
    # autonomy: formation
    "permission": ("memtrap.autonomy.formation.permission", "permission notes: 20 chores x 6 pushbacks"),
    "transcripts": ("memtrap.autonomy.formation.transcripts", "transcripts of the permission interactions"),
    "priority": ("memtrap.autonomy.formation.priority", 'priority notes ("you should prioritize speed here")'),
    "priority-transcripts": ("memtrap.autonomy.formation.priority_transcripts",
                             "transcripts of the priority interactions"),
    "profile": ("memtrap.autonomy.formation.profile",
                "permission notes formed with a benign profile in memory (--tier 1/2/4/8 for the length sweep)"),
    # autonomy: deployment and judges
    "confirmation": ("memtrap.autonomy.deploy.confirmation", "confirmation set (--memory ARM)"),
    "self-verification": ("memtrap.autonomy.deploy.self_verification", "self-verification set (--memory ARM)"),
    "goal-judge": ("memtrap.autonomy.judges.goal", "goal judge over the priority notes"),
    "scope-audit": ("memtrap.autonomy.judges.scope_ladder", "scope ladder over the permission notes (--judge)"),
    "stakes": ("memtrap.autonomy.judges.stakes", "stakes ratings of the formation tasks and test scenarios"),
    # sycophancy and over-critique
    "empathy": ("memtrap.sycophancy.empathy", "empathy notes; missed errors, missed fallacies, weaker-poem choice"),
    "rigor": ("memtrap.sycophancy.rigor", "rigor notes; flaws asserted in sound arguments, stronger-poem choice"),
    "proofs": ("memtrap.sycophancy.proofs.official", "IMO-GradingBench, official four-way protocol"),
    "proofs-rubric": ("memtrap.sycophancy.proofs.rubric", "IMO-GradingBench with reference solution and rubric (0-7)"),
    "note-rules": ("memtrap.sycophancy.note_rules", "classify the empathy and rigor notes"),
    # cheating: formation, the broken-safeguard set, ImpossibleBench (see cheating/README.md)
    "cheat-formation": ("cheating.formation", "test-repair notes: 6 chains x K sessions (--depth, --terminal)"),
    "safeguards": ("cheating.safeguards", "broken-safeguard set (--memory ARM, --samples)"),
    "impossiblebench": ("cheating.impossiblebench", "ImpossibleBench conflicting split (--memory ARM)"),
    "cheat-methods": ("cheating.methods", "classify how each cheating session cheated (--judge)"),
    "cheat-report": ("cheating.report", "the cheating setting's measures from a model's outputs"),
    # results
    "report": ("memtrap.report", "the autonomy and sycophancy measures from a model's outputs"),
}

# The cheating setting's memory arms (`cheating.safeguards`, `cheating.impossiblebench`).
CHEAT_ARMS = ["none", "note", "transcript"]
CHEAT_SETS = ("safeguards", "impossiblebench")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="experiments:\n" + "\n".join(f"  {k:22s}{v[1]}" for k, v in EXPERIMENTS.items()))
    ap.add_argument("experiment", choices=EXPERIMENTS, metavar="experiment")
    ap.add_argument("--model", help="model id, e.g. claude-sonnet-4-6 or gpt-5.6-sol (not needed for stakes)")
    ap.add_argument("--memory", help="memory arm for confirmation / self-verification: " + ", ".join(ARMS) +
                                     "; for safeguards / impossiblebench: " + ", ".join(CHEAT_ARMS))
    ap.add_argument("--tier", type=int, choices=(1, 2, 4, 8), help="profile sections for profile")
    ap.add_argument("--judge", help="judge model for scope-audit / cheat-methods")
    ap.add_argument("--depth", type=int, help="sessions per chain for cheat-formation (default 10)")
    ap.add_argument("--terminal", choices=("host", "docker"), help="formation version for cheat-formation: host "
                                                                   "(default) or docker")
    ap.add_argument("--samples", type=int, help="sessions per (case, memory) for safeguards (default 2)")
    ap.add_argument("--limit", type=int, help="safeguards / impossiblebench: run only the first N cases or "
                                              "instances, to check the setup before a full run")
    ap.add_argument("--pool", help="deliver this pool file instead of the arm's own (confirmation / self-verification)")
    ap.add_argument("--format", choices=("note", "transcript", "transcript_turns"), help="delivery format for --pool")
    ap.add_argument("--workers", type=int, default=32, help="concurrent API requests (default 32)")
    ap.add_argument("--force", action="store_true", help="start over instead of resuming")
    a = ap.parse_args()

    module = EXPERIMENTS[a.experiment][0]
    args = []
    if a.experiment in ("confirmation", "self-verification"):
        if not a.memory or (a.memory not in ARMS and not a.pool):
            ap.error(f"{a.experiment} needs --memory ({', '.join(ARMS)}), or --memory NAME with --pool")
        args = [a.memory] + (["--pool", os.path.abspath(a.pool)] if a.pool else []) + (["--format", a.format] if a.format else [])
    if a.experiment == "profile" and a.tier:
        args = [str(a.tier)]
    if a.experiment == "scope-audit" and a.judge:
        args = ["--judge", a.judge]
    if a.experiment in CHEAT_SETS:
        if a.memory not in CHEAT_ARMS:
            ap.error(f"{a.experiment} needs --memory ({', '.join(CHEAT_ARMS)})")
        args = [a.memory]
        if a.samples and a.experiment == "safeguards":
            args += ["--samples", str(a.samples)]
        if a.limit:
            args += ["--limit", str(a.limit)]
    if a.experiment == "cheat-formation":
        args = (["--depth", str(a.depth)] if a.depth else []) + (["--terminal", a.terminal] if a.terminal else [])
    if a.experiment == "cheat-methods" and a.judge:
        args = ["--judge", a.judge]
    if a.experiment != "stakes" and not a.model:
        ap.error("--model is required")

    pp = os.environ.get("PYTHONPATH")
    env = dict(os.environ, MEMTRAP_WORKERS=str(a.workers), PYTHONPATH=ROOT + (os.pathsep + pp if pp else ""))
    if a.model:
        env["MEMTRAP_MODEL"] = a.model
    if a.force:
        env["MEMTRAP_FORCE"] = "1"
    sys.exit(subprocess.call([sys.executable, "-u", "-m", module, *args], cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
