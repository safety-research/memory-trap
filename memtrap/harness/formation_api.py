"""The Messages API call of the autonomy formation sessions and judges.

`cr(**kw)` takes `messages.create` keyword arguments. GPT models are served through `gpt_shim` in the same response
shape. Rate-limit and overload errors are retried patiently (honouring Retry-After); other errors are raised at once,
and the caller counts the session as failed.
"""

import random
import time

from memtrap import paths
from memtrap.clients.anthropic_client import sdk_client
from memtrap.harness.routing import uses_openai

SUBJ = paths.MODEL
MAX_ATTEMPTS = 14
MAX_SLEEP = 30.0

cl = sdk_client()


def _is_rate_limit(e):
    if type(e).__name__ in ("RateLimitError", "OverloadedError", "APIStatusError",
                            "InternalServerError", "APIConnectionError", "APITimeoutError"):
        return True
    s = str(e)
    return "rate_limit" in s or "429" in s or "529" in s or "overloaded" in s.lower()


def _retry_after(e):
    try:
        h = getattr(getattr(e, "response", None), "headers", None) or {}
        v = h.get("retry-after") or h.get("Retry-After")
        if v:
            return min(float(v), 60.0)
    except Exception:
        pass
    return None


def cr(**kw):
    if uses_openai(kw.get("model", "")):
        from memtrap.harness import gpt_shim
        return gpt_shim.cr_openai(**kw)
    for i in range(MAX_ATTEMPTS):
        try:
            return cl.messages.create(**kw)
        except Exception as e:
            if not _is_rate_limit(e) or i == MAX_ATTEMPTS - 1:
                raise
            d = _retry_after(e)
            time.sleep(d if d is not None else min(1.5 * (1.7 ** i), MAX_SLEEP) + random.uniform(0, 2.0))


def memory_tools(scen):
    """The API-native memory tool plus the scenario's tools (free-form arguments)."""
    return [{"type": "memory_20250818", "name": "memory"}] + task_tools(scen)


def task_tools(scen):
    return [{"name": n, "description": d, "input_schema": {"type": "object", "properties": {}, "additionalProperties": True}}
            for n, d in scen["tools"].items()]
