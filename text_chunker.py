"""
Text Chunker

Splits documents into token-based chunks using the same embedding model's tokenizer.
Each chunk includes its index, text, and token count.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single chunk of text with metadata."""

    index: int
    text: str
    token_count: int


# ── Lazy-loaded tokenizer & splitter (shared across calls) ──

_tokenizer = None
_splitter = None


def _ensure_tokenizer():
    """Load the tokenizer from the embedding model (cached)."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer for '{settings.embedding_model_name}'...")
    _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model_name)
    logger.info("Tokenizer loaded.")
    return _tokenizer


def _ensure_splitter():
    """Create the text splitter using the model's tokenizer (cached)."""
    global _splitter
    if _splitter is not None:
        return _splitter

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    tokenizer = _ensure_tokenizer()
    _splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=True,
        strip_whitespace=True,
    )
    return _splitter


# ── Public API ──


def chunk_text(text: str) -> list[TextChunk]:
    """
    Split text into token-aware chunks.

    Args:
        text: The full document text to chunk.

    Returns:
        A list of ``TextChunk`` objects, each with ``index``, ``text``,
        and ``token_count``.
    """
    if not text or text.isspace():
        return []

    splitter = _ensure_splitter()
    tokenizer = _ensure_tokenizer()

    raw_chunks = splitter.split_text(text)

    return [
        TextChunk(
            index=idx,
            text=chunk,
            token_count=len(tokenizer.encode(chunk, add_special_tokens=False)),
        )
        for idx, chunk in enumerate(raw_chunks)
    ]


def chunk_file(file_path: str) -> list[TextChunk]:
    """
    Read a text file and split it into token-aware chunks.

    Args:
        file_path: Path to a text file.

    Returns:
        A list of ``TextChunk`` objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    return chunk_text(text)


# ── CLI ──

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python text_chunker.py <file_path>")
        sys.exit(1)

    chunks = chunk_file(sys.argv[1])
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"[{c.index}] tokens={c.token_count} | {c.text[:120]}...")
