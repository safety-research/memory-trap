"""GPT models for the Messages-API formation loops.

The formation loops call `cr(model=..., system=..., messages=..., tools=...)` and read the Anthropic response shape
(`stop_reason`, and `content` blocks with type / text / id / name / input that are appended back into `messages`).
For GPT models `cr_openai` makes the same call through `OpenAIClient` (which turns the memory tool into an explicit
function) and returns a response with that shape.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from memtrap.clients.openai_client import OpenAIClient


@dataclass
class Block:
    """An Anthropic content block (text or tool_use)."""
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        if self.type == "text":
            return {"type": "text", "text": self.text}
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": dict(self.input)}


@dataclass
class Resp:
    """An Anthropic Message: `content` blocks and `stop_reason`."""
    content: list
    stop_reason: str
    raw: Any = None


_clients: dict[str, OpenAIClient] = {}


def client_for(model: str) -> OpenAIClient:
    if model not in _clients:
        _clients[model] = OpenAIClient(model=model, max_retries=8)
    return _clients[model]


def cr_openai(**kw) -> Resp:
    """Messages-API keyword arguments -> GPT call -> Anthropic-shaped response."""
    system = kw.get("system") or ""
    if isinstance(system, list):
        system = "\n\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    res = client_for(kw["model"]).complete(system=system, messages=kw["messages"], tools=kw.get("tools"),
                                           max_tokens=kw.get("max_tokens", 1024),
                                           temperature=kw.get("temperature", 1.0))
    blocks: list[Block] = []
    if res.text:
        blocks.append(Block("text", text=res.text))
    for c in res.tool_calls:
        blocks.append(Block("tool_use", id=c.id or f"call_{uuid.uuid4().hex[:12]}", name=c.name, input=dict(c.input)))
    if not blocks:
        blocks.append(Block("text", text=""))
    return Resp(content=blocks, stop_reason=res.stop_reason or "end_turn", raw=res.raw)
