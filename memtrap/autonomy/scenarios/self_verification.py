"""Self-verification set: 100 scenarios (data/autonomy/self_verification_set.json) in which no user is available.

The agent opens an item, runs routine checks that pass, and may run one optional, slower check before an irreversible
action; only that check reveals the problem (`reveal`). A hold tool, and in some scenarios a post step, are also
offered. Firing the action is unsafe.
"""

from memtrap.harness.detectors import mk_det


def build(sp):
    oname, odesc = sp["open"]
    vname, vdesc = sp["verify"]
    aname, adesc = sp["action"]
    hname, hdesc, hres = sp["hold"]
    cheap = sp["cheap"]
    posts = sp.get("posts", [])

    tools = {oname: odesc}
    for n, d, _ in cheap:
        tools[n] = d
    tools[vname] = vdesc
    tools[aname] = adesc
    for n, d, _ in posts:
        tools[n] = d
    tools[hname] = hdesc
    tools["finish"] = "finish(message): end the task"

    results = {oname: sp["view"], vname: sp["reveal"], hname: hres}
    results.update({n: r for n, _, r in cheap})
    results.update({n: r for n, _, r in posts})
    ares = sp["action_res"]

    def sim(n, x, st):
        if n == aname:
            return ares
        if n in results:
            return results[n]
        return "(done)" if n == "finish" else "(ok)"

    return dict(key=sp["key"], task=sp["task"], tools=tools, sim=sim, destructive=aname, det=mk_det(aname))
