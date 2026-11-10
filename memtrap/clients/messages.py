"""Message and result types shared by the clients.

A message's content is a string or a list of Anthropic-style content blocks (text / tool_use / tool_result).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

Content = Union[str, list[dict]]


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: Content

    def to_api(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class CompletionResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: Optional[str] = None
    raw: Any = None

    @property
    def used_tool(self) -> bool:
        return len(self.tool_calls) > 0
