"""
Application settings — uses Pydantic-Settings
All values can be overridden via environment variables or a .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration loaded from .env / environment variables."""

    # ── Embedding ──────────────────────────────────────────────
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_cache_folder: str | None = None

    # ── LLM (document analysis + RAG answer generation) ────────
    llm_provider: str = "lmstudio"  # "lmstudio" or "ollama"
    llm_model: str = "default"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    llm_timeout: int = 60
    llm_chat_url: str | None = None   # override default chat URL per provider
    llm_health_url: str | None = None  # override default health URL per provider

    # ── Text chunking ──────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Vector database (ChromaDB) ────────────────────────────
    chroma_db_path: str = "./chroma_db"
    chroma_collection_name: str = "documents"
    chroma_top_k: int = 5

    # ── Analysis prompt (use {text} as placeholder) ────────────
    analysis_prompt: str = (
        'Analyze the following document and return ONLY valid JSON '
        'with keys "document_topic", "document_type", "keywords". '
        'document_topic: A concise title for the document. '
        'document_type: The type of content (e.g. Article, Report, Essay, Technical Guide). '
        'keywords: 3-8 key terms or phrases.\n\n'
        'Document:\n{text}'
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ── Provider endpoint defaults (can be overridden via settings above) ────
_PROVIDER_DEFAULTS = {
    "lmstudio": {
        "chat_url": "http://localhost:1234/v1/chat/completions",
        "health_url": "http://localhost:1234/v1/models",
    },
    "ollama": {
        "chat_url": "http://localhost:11434/api/chat",
        "health_url": "http://localhost:11434/api/tags",
    },
}


def _build_providers(s: Settings) -> dict:
    """Merge provider defaults with any user overrides from settings."""
    providers = {}
    for name, urls in _PROVIDER_DEFAULTS.items():
        providers[name] = {
            "chat_url": s.llm_chat_url or urls["chat_url"],
            "health_url": s.llm_health_url or urls["health_url"],
        }
    return providers


# Single global instance — import `settings` anywhere to get config.
settings = Settings()
PROVIDERS = _build_providers(settings)
