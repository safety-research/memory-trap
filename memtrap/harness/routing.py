"""Which API serves a model id: gpt-*, o-series and chatgpt-* ids go to OpenAI, all others to Anthropic."""

OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def uses_openai(model):
    return str(model).startswith(OPENAI_PREFIXES)
