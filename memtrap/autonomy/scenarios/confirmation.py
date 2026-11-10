"""Confirmation set: 200 scenarios (data/autonomy/confirmation_set.json) in which a user asks for an action and
inspecting the environment reveals a reason to check back with that user first.

Each record's `mode` names its shape:
  M1, M5, M8   open -> checks -> action -> posts -> finish   (M1 and M8 are the reviewed-batch scenarios)
  M8n          the same without posts
  M4           open -> checks -> action                      (a personal chore; no finish tool)
  M2, M3       inspect -> [benign steps] -> action -> finish
  M10          as M2, with an action that takes a target; unsafe only on the harmful target
  stale, scope inspect(s) -> [benign steps] -> action -> finish, records with a `harm_tool`
Every scenario offers at most one check and one post. The `harm` field describes the consequence and is not shown to
the model.
"""

from memtrap.harness.detectors import mk_det, mk_det_target


def _pipe(spec, with_posts=True, with_finish=True):
    checks = spec.get("checks", [])[:1]
    posts = spec.get("posts", [])[:1] if with_posts else []
    oname, odesc = spec["open"]
    aname, adesc = spec["action"]

    tools = {oname: odesc}
    for n, d, _ in checks:
        tools[n] = d
    tools[aname] = adesc
    for n, d, _ in posts:
        tools[n] = d
    if with_finish:
        tools["finish"] = "finish(message): end the task"

    results = {oname: spec["view"]}
    results.update({n: r for n, _, r in checks})
    results.update({n: r for n, _, r in posts})
    ares = spec["action_res"]

    def sim(n, x, st):
        if n in results:
            return results[n]
        if n == aname:
            return ares
        if n == "finish":
            return "(done)"
        return "(no result)"

    return dict(key=spec["key"], task=spec["task"], tools=tools, sim=sim, destructive=aname, det=mk_det(aname))


def _delegate(spec, det=None):
    iname, idesc = spec["inspect"]
    hname, hdesc = spec["action"]
    benign = spec.get("benign", [])

    tools = {iname: idesc}
    for n, d, _ in benign:
        tools[n] = d
    tools[hname] = hdesc
    tools["finish"] = "finish(message): end the task"

    results = {iname: spec["state"]}
    results.update({n: r for n, _, r in benign})
    hres = spec["action_res"]

    def sim(n, x, st):
        if n in results:
            return results[n]
        if n == hname:
            return hres
        if n == "finish":
            return "(done)"
        return "(no result)"

    return dict(key=spec["key"], task=spec["task"], tools=tools, sim=sim, destructive=hname,
                det=det or mk_det(hname))


def _target(spec):
    return _delegate(spec, det=mk_det_target(spec["action"][0], spec.get("target")))


def _harm_tool(spec):
    tools, results = {}, {}
    for n, d, r in spec.get("inspects", []):
        tools[n] = d
        results[n] = r
    for n, d, r in spec.get("benign", []):
        tools[n] = d
        results[n] = r
    hname, hdesc = spec["harm_tool"]
    tools[hname] = hdesc
    tools["finish"] = "finish(message): end the task"

    def sim(n, x, st):
        if n in results:
            return results[n]
        if n == hname:
            return spec["harm_res"]
        if n == "finish":
            return "(done)"
        return "(no result)"

    return dict(key=spec["key"], task=spec["task"], tools=tools, sim=sim, destructive=hname, det=mk_det(hname))


BUILDERS = {
    "M1": _pipe, "M5": _pipe, "M8": _pipe,
    "M8n": lambda s: _pipe(s, with_posts=False),
    "M4": lambda s: _pipe(s, with_posts=False, with_finish=False),
    "M2": _delegate, "M3": _delegate,
    "M10": _target,
}


def build(spec):
    if "harm_tool" in spec:
        return _harm_tool(spec)
    return BUILDERS[spec["mode"]](spec)
