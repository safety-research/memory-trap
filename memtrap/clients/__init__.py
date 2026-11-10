"""Model clients: one `complete()` interface over the Anthropic and OpenAI APIs."""

from memtrap.clients.base import ModelClient


def make_client(backend: str = "auto", **kwargs) -> ModelClient:
    """Return a client for `kwargs["model"]`. `auto` sends gpt-* and o-series ids to OpenAI, others to Anthropic."""
    if backend == "auto":
        from memtrap.clients.openai_client import is_openai_model
        backend = "openai" if is_openai_model(kwargs.get("model") or "") else "anthropic"
    if backend == "anthropic":
        from memtrap.clients.anthropic_client import AnthropicClient
        return AnthropicClient(**kwargs)
    if backend == "openai":
        from memtrap.clients.openai_client import OpenAIClient
        return OpenAIClient(**kwargs)
    raise ValueError(f"unknown backend {backend!r}")
