"""Anthropic Messages API client.

Requests that carry the memory tool go to the beta endpoint with the context-management beta. The system prompt is
sent with prompt caching.
"""

import os
import re
from typing import Optional

from memtrap.clients.base import ModelClient
from memtrap.clients.messages import CompletionResult, Message, ToolCall

MEMORY_BETA = "context-management-2025-06-27"
LONG_OUTPUT = 16000            # max_tokens above this get LONG_OUTPUT_TIMEOUT seconds
LONG_OUTPUT_TIMEOUT = 3600.0

# Models whose API rejects `temperature`: Opus 4.7 and later, and every 5-series model.
_NO_SAMPLING = ("opus-4-7", "opus-4-8", "fable-5")
_NO_SAMPLING_RE = re.compile(r"-(?:opus|sonnet|haiku|fable)-[5-9](?![\d-])")


def _rejects_sampling(model: str) -> bool:
    return any(s in model for s in _NO_SAMPLING) or bool(_NO_SAMPLING_RE.search(model))


def sdk_client(**kw):
    """An `anthropic.Anthropic` client; every Anthropic client in the code is made here.

    By default the SDK reads ANTHROPIC_API_KEY. With WIF_ORG set, the client instead authenticates through
    Workload Identity Federation with the IDs in ~/.anthropic/wif-<WIF_ORG>.env (or the file named by WIF_IDS_FILE).
    """
    import anthropic
    org = os.environ.get("WIF_ORG")
    if org:
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):   # a set key would override federation
            os.environ.pop(k, None)
        kw.pop("api_key", None)
        kw.pop("auth_token", None)
        kw["credentials"] = _wif_credentials(org)
    return anthropic.Anthropic(**kw)


def _wif_credentials(org):
    import google.auth.transport.requests
    import google.oauth2.id_token
    from anthropic import WorkloadIdentityCredentials
    path = os.environ.get("WIF_IDS_FILE") or os.path.expanduser(f"~/.anthropic/wif-{org}.env")
    if not os.path.isfile(path):
        raise SystemExit(f"WIF_ORG={org} is set, but there is no federation ids file at {path}. "
                         "Set WIF_IDS_FILE to one, or unset WIF_ORG to authenticate with an API key.")
    ids = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                ids[k] = v
    return WorkloadIdentityCredentials(
        identity_token_provider=lambda: google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), "https://api.anthropic.com"),
        federation_rule_id=_need(ids, "ANTHROPIC_FEDERATION_RULE_ID_GCP", path),
        organization_id=_need(ids, "ANTHROPIC_ORGANIZATION_ID", path),
        service_account_id=_need(ids, "ANTHROPIC_SERVICE_ACCOUNT_ID", path),
        workspace_id=_need(ids, "ANTHROPIC_WORKSPACE_ID", path))


def _need(ids, key, path):
    if key not in ids:
        raise SystemExit(f"{path} has no {key}=")
    return ids[key]


def _is_memory_tool(tool: dict) -> bool:
    return tool.get("name") == "memory" or str(tool.get("type", "")).startswith("memory")


class AnthropicClient(ModelClient):
    def __init__(self, model: Optional[str] = None, cache_system: bool = True, max_retries: int = 8,
                 timeout: Optional[float] = None):
        ckw = {"max_retries": max_retries}
        if timeout:
            ckw["timeout"] = timeout
        self.client = sdk_client(**ckw)
        self.model = model or ""
        self.cache_system = cache_system

    def _system_param(self, system: str):
        if not system:
            return None
        if self.cache_system:
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return system

    def _build_request(self, system, messages, tools, max_tokens, temperature, kwargs):
        params: dict = {"model": self.model, "max_tokens": max_tokens,
                        "messages": [m.to_api() for m in messages]}
        sys_param = self._system_param(system)
        if sys_param is not None:
            params["system"] = sys_param
        if tools:
            params["tools"] = tools
        if temperature is not None and not _rejects_sampling(self.model):
            params["temperature"] = temperature
        if max_tokens > LONG_OUTPUT:
            params["timeout"] = LONG_OUTPUT_TIMEOUT
        params.update(kwargs)
        betas = [MEMORY_BETA] if tools and any(_is_memory_tool(t) for t in tools) else []
        return params, betas

    def complete(self, *, system: str, messages: list[Message], tools: Optional[list[dict]] = None,
                 max_tokens: int = 1024, temperature: float = 1.0, **kwargs) -> CompletionResult:
        params, betas = self._build_request(system, messages, tools, max_tokens, temperature, kwargs)
        if betas:
            resp = self.client.beta.messages.create(betas=betas, **params)
        else:
            resp = self.client.messages.create(**params)
        return self._normalize(resp)

    @staticmethod
    def _normalize(resp) -> CompletionResult:
        text_parts, tool_calls = [], []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        return CompletionResult(text="".join(text_parts), tool_calls=tool_calls,
                                stop_reason=getattr(resp, "stop_reason", None), raw=resp)
