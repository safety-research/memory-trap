"""OpenAI Responses API client that speaks the code's Anthropic-style message format.

GPT models have no API-native memory tool. A request that carries the memory tool gets the six memory commands as a
function (`MEMORY_FUNCTION_TOOL`) and the protocol text that the Anthropic API adds to the system prompt
(`MEMORY_PROTOCOL`); `memtrap.memory.FileMemory` executes the commands.
"""

import json
import os
import re
import time
import uuid
from typing import Any, Optional

from memtrap.clients.base import ModelClient
from memtrap.clients.messages import CompletionResult, Message, ToolCall

_OPENAI_RE = re.compile(r"^(gpt-|o\d(?:-|$)|chatgpt-)", re.I)
_NO_SAMPLING_RE = re.compile(r"^(gpt-[5-9]|gpt-\d{2}|o\d)", re.I)   # reasoning models reject `temperature`


def is_openai_model(model: str) -> bool:
    return bool(_OPENAI_RE.match(model or ""))


def _rejects_sampling(model: str) -> bool:
    return bool(_NO_SAMPLING_RE.match(model or ""))


def _is_memory_tool(tool: dict) -> bool:
    return tool.get("name") == "memory" or str(tool.get("type", "")).startswith("memory")


# The memory tool's commands and result strings follow Anthropic's documented memory_20250818 tool.
MEMORY_FUNCTION_TOOL: dict = {
    "type": "function",
    "name": "memory",
    "description": (
        "Your persistent memory: a directory of files under /memories that is kept across "
        "sessions and loaded at the start of every session. Commands: "
        "view (path[, view_range]) — show a directory listing or a file's contents with line "
        "numbers; create (path, file_text) — create or overwrite a file; "
        "str_replace (path, old_str, new_str) — replace exactly one occurrence of old_str; "
        "insert (path, insert_line, insert_text) — insert text after line insert_line (0 = top); "
        "delete (path) — delete a file or directory; rename (old_path, new_path) — move or "
        "rename. All paths must stay under /memories."
    ),
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "delete", "rename"]},
            "path": {"type": "string", "description": "absolute path under /memories"},
            "view_range": {"type": "array", "items": {"type": "integer"},
                           "minItems": 2, "maxItems": 2,
                           "description": "optional [start_line, end_line] for view"},
            "file_text": {"type": "string", "description": "full contents for create"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "insert_line": {"type": "integer"},
            "insert_text": {"type": "string"},
            "old_path": {"type": "string"},
            "new_path": {"type": "string"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}

# From Anthropic's memory-tool documentation (prompting guidance).
MEMORY_PROTOCOL = (
    "IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE.\n"
    "MEMORY PROTOCOL:\n"
    "1. Use the `view` command of your `memory` tool to check for earlier progress.\n"
    "2. ... (work on the task) ...\n"
    "     - As you make progress, record status / progress / thoughts etc in your memory.\n"
    "ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk "
    "losing any progress that is not recorded in your memory directory."
)


def _bget(b: Any, key: str, default=None):
    if isinstance(b, dict):
        return b.get(key, default)
    return getattr(b, key, default)


def _text_of(content: Any) -> str:
    """Flatten a string, a block list or a nested tool_result content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for b in content:
        t = _bget(b, "type")
        if t == "text":
            out.append(_bget(b, "text", "") or "")
        elif t == "tool_result":
            out.append(_text_of(_bget(b, "content", "")))
        elif isinstance(b, str):
            out.append(b)
    return "\n".join(out)


class OpenAIClient(ModelClient):
    def __init__(self, model: Optional[str] = None, max_retries: int = 8, timeout: Optional[float] = None):
        import openai
        self._openai = openai
        ckw: dict = {"max_retries": max_retries}
        if timeout:
            ckw["timeout"] = timeout
        if os.environ.get("OPENAI_BASE_URL"):
            ckw["base_url"] = os.environ["OPENAI_BASE_URL"]
        self.client = openai.OpenAI(**ckw)
        self.model = model or ""

    @staticmethod
    def convert_tools(tools: Optional[list[dict]]) -> list[dict]:
        """Anthropic tool descriptors -> Responses function tools; the memory tool -> MEMORY_FUNCTION_TOOL."""
        out = []
        for t in tools or []:
            if _is_memory_tool(t):
                out.append(MEMORY_FUNCTION_TOOL)
                continue
            if t.get("type") == "function" and "parameters" in t:
                out.append(t)
                continue
            params = dict(t.get("input_schema") or {"type": "object", "properties": {}})
            params.setdefault("type", "object")
            params.setdefault("properties", {})
            out.append({"type": "function", "name": t["name"], "description": t.get("description", ""),
                        "parameters": params, "strict": False})
        return out

    @staticmethod
    def convert_messages(messages: list) -> list[dict]:
        """Anthropic-style turns -> Responses input items (tool_use -> function_call, tool_result ->
        function_call_output). Accepts Message objects or {"role", "content"} dicts."""
        items: list[dict] = []
        for m in messages:
            role, content = _bget(m, "role"), _bget(m, "content")
            if isinstance(content, str):
                items.append({"role": role, "content": content})
                continue
            text_parts = []
            for b in content or []:
                t = _bget(b, "type")
                if t == "text":
                    text_parts.append(_bget(b, "text", "") or "")
                elif t == "tool_use":
                    if text_parts:
                        items.append({"role": role, "content": "\n".join(text_parts)}); text_parts = []
                    items.append({"type": "function_call", "call_id": _bget(b, "id"), "name": _bget(b, "name"),
                                  "arguments": json.dumps(dict(_bget(b, "input") or {}))})
                elif t == "tool_result":
                    if text_parts:
                        items.append({"role": role, "content": "\n".join(text_parts)}); text_parts = []
                    items.append({"type": "function_call_output", "call_id": _bget(b, "tool_use_id"),
                                  "output": _text_of(_bget(b, "content", "")) or "(no result)"})
                elif isinstance(b, str):
                    text_parts.append(b)
            if text_parts:
                items.append({"role": role, "content": "\n".join(text_parts)})
        return items

    def _build_request(self, system, messages, tools, max_tokens, temperature):
        instructions = system or ""
        if tools and any(_is_memory_tool(t) for t in tools):
            instructions = (instructions + "\n\n" if instructions else "") + MEMORY_PROTOCOL
        effort = getattr(self, "_effort", None) or "none"
        floor = 16 if effort == "none" else 4096   # keeps a short visible answer when reasoning is on
        params: dict = {"model": self.model, "input": self.convert_messages(messages),
                        "max_output_tokens": max(int(max_tokens), floor)}
        if _rejects_sampling(self.model):
            params["reasoning"] = {"effort": effort}
        if instructions:
            params["instructions"] = instructions
        if tools:
            params["tools"] = self.convert_tools(tools)
        if temperature is not None and not _rejects_sampling(self.model):
            params["temperature"] = temperature
        return params

    def _create(self, params: dict):
        openai = self._openai
        for i in range(6):
            try:
                return self.client.responses.create(**params)
            except openai.BadRequestError as e:
                msg = str(e)
                if "temperature" in params and "temperature" in msg:
                    params = {k: v for k, v in params.items() if k != "temperature"}
                    continue
                if "reasoning" in params and "reasoning" in msg.lower():
                    # move to the lowest supported effort, for this client's later calls too
                    nxt = {"none": "minimal", "minimal": "low"}.get(params["reasoning"].get("effort"))
                    if nxt:
                        self._effort = nxt
                        params = {**params, "reasoning": {"effort": nxt},
                                  "max_output_tokens": max(params["max_output_tokens"], 4096)}
                        continue
                raise
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError):
                if i == 5:
                    raise
                time.sleep(min(3 * 2 ** i, 40))
        raise RuntimeError("unreachable")

    def complete(self, *, system: str, messages: list[Message], tools: Optional[list[dict]] = None,
                 max_tokens: int = 1024, temperature: float = 1.0, **kwargs) -> CompletionResult:
        return self._normalize(self._create(self._build_request(system, messages, tools, max_tokens, temperature)))

    @staticmethod
    def _normalize(resp) -> CompletionResult:
        text_parts, calls = [], []
        for item in getattr(resp, "output", None) or []:
            t = getattr(item, "type", None)
            if t == "message":
                for c in getattr(item, "content", None) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        text_parts.append(getattr(c, "text", "") or "")
            elif t == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": item.arguments}
                calls.append(ToolCall(id=item.call_id or f"call_{uuid.uuid4().hex[:12]}",
                                      name=item.name, input=args if isinstance(args, dict) else {}))
        if calls:
            stop = "tool_use"
        elif getattr(resp, "status", None) == "incomplete":
            stop = "max_tokens"
        else:
            stop = "end_turn"
        return CompletionResult(text="".join(text_parts), tool_calls=calls, stop_reason=stop, raw=resp)
