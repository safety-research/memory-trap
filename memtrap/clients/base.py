"""The interface every model client implements."""

from abc import ABC, abstractmethod
from typing import Optional

from memtrap.clients.messages import CompletionResult, Message


class ModelClient(ABC):
    """A chat-completion client. `complete` must not mutate `messages`."""

    model: str = "abstract"

    @abstractmethod
    def complete(self, *, system: str, messages: list[Message], tools: Optional[list[dict]] = None,
                 max_tokens: int = 1024, temperature: float = 1.0, **kwargs) -> CompletionResult:
        """Return the model's response to `messages` under `system`."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
