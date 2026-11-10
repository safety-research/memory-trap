"""The JSON tool-call protocol of the test episodes.

The system prompt lists the scenario's tools and asks for one JSON object per turn,
{"tool": "<name>", "args": {...}}.
"""

import json
import re
import time

from memtrap.harness import subject as M

CONFIRM_TOOLS = {"ask_user", "confirm"}   # removed from a scenario's tools before they are offered


def crobust(system, msgs, mt=400, temp=0.7):
    """One completion, retried while the reply is empty."""
    r = ""
    for i in range(5):
        r = M.comp(system, msgs, mt, temp)
        if r and r.strip():
            return r
        time.sleep(min(8 * (i + 1), 40))
    return r


def parse_call(text):
    """The first JSON object in `text` that names a "tool", or None."""
    for m in re.finditer(r"\{", text or ""):
        d = 0
        for j in range(m.start(), len(text)):
            if text[j] == "{":
                d += 1
            elif text[j] == "}":
                d -= 1
                if d == 0:
                    try:
                        o = json.loads(text[m.start():j + 1])
                        if isinstance(o, dict) and "tool" in o:
                            return o
                    except Exception:
                        pass
                    break
    return None
