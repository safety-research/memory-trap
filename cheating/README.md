# The Cheating Setting

## Overview

An agent repairs failing builds in a terminal over a series of sessions, keeping notes in its own memory. The
sessions are low-stakes: twenty small repos whose implementation and test disagree after a routine vendor sync. The
memory it writes is then loaded into fresh sessions on two other tasks:

- **The broken-safeguard set**: 50 repos of the same shape, except the failing assertion is a safety check, so
  weakening the test ships the bug it was there to catch.
- **ImpossibleBench** ([Zhang et al.](https://github.com/safety-research/impossiblebench)): SWE-bench issues whose
  tests cannot be satisfied honestly.

Both measure whether the agent makes the build green by defeating the check instead of fixing the code.

## Project Structure

```
cheating/
├── data/
│   ├── formation_tasks.json            # 20 formation repos
│   ├── broken_safeguard_set.json       # 50 broken-safeguard repos
│   └── impossiblebench_instances.json  # The 348 ImpossibleBench instances
├── docker/Dockerfile           # Container image
├── terminal.py                 # Sandbox interface
├── sandbox.py                  # Sandbox adapter
├── prompts.py                  # Task text, system prompt, tools, memory delivery
├── formation.py                # Formation: 6 chains x K sessions
├── safeguards.py               # The broken-safeguard set
├── scoring.py                  # Masking scorer
├── impossiblebench.py          # ImpossibleBench
├── methods.py                  # How each cheating session cheated
└── report.py                   # The setting's measures
```

## Setup

Complete the setup in the top-level [README](../README.md) first.

1. Install [Docker](https://docs.docker.com/engine/install/). On Ubuntu or Debian:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker "$USER"                      # then log out and back in
docker run --rm hello-world                          # check it works
```

2. Install the extra dependencies:
```bash
pip install -r requirements.txt -r cheating/requirements.txt
```

3. For ImpossibleBench, clone it:
```bash
git clone https://github.com/safety-research/impossiblebench bench/impossiblebench
cd bench/impossiblebench && git checkout 061dc3d
```
Set `MEMTRAP_IMPOSSIBLEBENCH` if you keep it elsewhere. GPT models on ImpossibleBench need `openai==2.54.0`, in a
separate environment.

4. Point the sandbox adapter at the sandbox module. `cheating/sandbox.py` drives `rh_dock_sandbox`, which is not part
   of this repository:
```bash
export MEMTRAP_RHFORM_DIR=~/rhform3/code   # directory holding rh_dock_sandbox.py
export MEMTRAP_RHFORM_SRC=~/rhform3/src    # directory holding its memento package
export MEMTRAP_DOCK_POOL=48                # containers to pre-start
export MEMTRAP_DOCK_PREFIX=rhdock_p        # container name prefix; one per parallel run
```
These are the defaults. To use your own sandbox, implement the interface in `cheating/terminal.py` and set
`MEMTRAP_CHEAT_SANDBOX` to your module.

## Usage

Run formation first; the other steps read its notes and transcripts.

```bash
python run.py cheat-formation  --model claude-sonnet-4-6                  # 6 chains x 10 sessions
python run.py safeguards       --model claude-sonnet-4-6 --memory none    # also note, transcript
python run.py impossiblebench  --model claude-sonnet-4-6 --memory none    # also note, transcript
python run.py cheat-methods    --model claude-sonnet-4-6                  # how the cheating was done
python run.py cheat-report     --model claude-sonnet-4-6
```

### Two Versions of Formation

Formation has two versions, chosen with `--terminal`:

- `host` (default): the agent's commands run on the machine itself, in a fresh directory. `solution.py` is
  read-only, and an edit that gets past its mode is undone with a message that the file is frozen and owned by
  another team.
- `docker`: the agent's commands run in a container with no network, where `solution.py` belongs to another user
  and the kernel refuses any change to it.

Broken-safeguard sessions always run in the container. Give each version its own outputs directory:

```bash
export MEMTRAP_OUTPUTS=outputs_docker
python run.py cheat-formation  --model claude-sonnet-4-6 --terminal docker
python run.py safeguards       --model claude-sonnet-4-6 --memory note
```

### Command Line Options

- `--memory`: memory arm for `safeguards` and `impossiblebench`: `none`, `note`, `transcript`
- `--depth`: sessions per chain for `cheat-formation` (default 10)
- `--terminal`: formation version for `cheat-formation`: `host` (default) or `docker`
- `--samples`: sessions per case and memory for `safeguards` (default 2)
- `--limit`: run only the first N cases or instances, to check the setup
- `--judge`: judge model for `cheat-methods` (default `claude-opus-4-7`)
- `--workers`, `--force`: as in the top-level README

A full run is 60 formation sessions. The broken-safeguard set is 100 sessions without memory and 600 per memory arm.
ImpossibleBench is 1,044 sessions without memory (348 instances x 3) and 2,088 per memory arm (348 x 6 notes).

## Data

Data files are in `cheating/data/`:
- `formation_tasks.json`: 20 repos. `solution` and `test` are the two files the agent works on; `initial_solution`
  and `commit_message` give the repo its two-commit git history; `fixed_test` is the test that matches the
  implementation. Chain `c` takes task `(c + 3i) mod 20` at session `i`.
- `broken_safeguard_set.json`: 50 repos with `solution` and `test`, plus the `failing_assertion`, the `contract` it
  guarantees, the `harm` of shipping the bug, and `category`, `bug_shape` and `rung`. `contract` and `harm` are not
  shown to the model.
- `impossiblebench_instances.json`: the 348 instances of the `conflicting` split, in order, with repo, version and
  difficulty, and the one instance excluded (`astropy__astropy-7336`).

## Outputs

```
outputs/<model>/cheating/
├── formation/        sessions.jsonl, notes.json, transcripts.json
├── safeguards/       none.jsonl, note.jsonl, transcript.jsonl   (one scored session per line)
├── impossiblebench/  logs/, samples.jsonl, summary.json, methods.jsonl
└── report.json
```
