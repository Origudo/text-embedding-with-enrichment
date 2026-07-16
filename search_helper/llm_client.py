"""OpenAI-compatible LLM client for RAG answer generation.

Respects ``settings.llm_provider`` to pick the right backend
(LM Studio or Ollama) and reads temperature / max tokens from config.
"""

from openai import OpenAI

from settings import PROVIDERS, settings


# ── Lazy client (created once per provider) ─────────────────────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Build (or reuse) the OpenAI client for the configured provider."""
    global _client
    if _client is not None:
        return _client

    provider = settings.llm_provider
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from {list(PROVIDERS)}")

    chat_url = PROVIDERS[provider]["chat_url"]
    # The OpenAI client expects the base URL without the endpoint suffix
    base_url = chat_url.replace("/chat/completions", "").replace("/api/chat", "")

    _client = OpenAI(
        base_url=base_url,
        api_key="not-needed",  # Local backends don't require a key
        timeout=settings.llm_timeout,
    )
    return _client


_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on the provided context. "
    "Use only the context below to answer. "
    "If the context lacks enough information, say so honestly."
)


def _build_prompt(question: str, chunks: list) -> str:
    """Combine retrieved chunks into a context block."""
    context_parts = [
        f"[Source {i}]\n{c.text}" for i, c in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based solely on the context above."


def generate_answer(question: str, chunks: list) -> str:
    """Send question + retrieved chunks to the LLM and return the answer."""
    if not chunks:
        return "No relevant context found."

    client = _get_client()
    user_prompt = _build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    return response.choices[0].message.content
