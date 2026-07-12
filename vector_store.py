"""
Vector Store

Orchestrates the full pipeline:
  1. Analyze a file via DocumentAnalyzer (topic, type, keywords)
  2. Split the file into chunks via TextChunker
  3. Append enrichment to each chunk's text
  4. Embed each enriched chunk via EmbeddingProcessor
  5. Gather file metadata
  6. Store everything in ChromaDB with UUID-based IDs
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from document_analyzer import DocumentAnalyzer
from embedding_processor import EmbeddingProcessor
from settings import PROVIDERS, settings
from text_chunker import TextChunk, chunk_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Data models
# ======================================================================


@dataclass
class SearchResult:
    """A single result from a vector similarity search."""

    id: str
    text: str
    index: int
    token_count: int
    distance: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Enrichment:
    """Enrichment data produced by DocumentAnalyzer."""

    document_topic: str
    document_type: str
    keywords: list[str]


@dataclass
class FileMetadata:
    """File-system metadata for a processed document."""

    file_name: str
    file_path: str
    file_size: int
    created_at: str
    modified_at: str


# ======================================================================
# ChromaDB client (singleton)
# ======================================================================

_chroma_client = None
_collection = None


def _get_collection():
    """Get or create the ChromaDB collection (lazy singleton)."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(
        path=settings.chroma_db_path,
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )

    _collection = _chroma_client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        f"ChromaDB ready at '{settings.chroma_db_path}' "
        f"(collection: '{settings.chroma_collection_name}')"
    )
    return _collection


# ======================================================================
# Helpers
# ======================================================================


def _get_file_metadata(file_path: str) -> FileMetadata:
    """Extract file-system metadata from a file."""
    path = Path(file_path)
    stat = path.stat()
    return FileMetadata(
        file_name=path.name,
        file_path=str(path.resolve()),
        file_size=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    )


def _build_enriched_text(chunk_text: str, enrichment: Enrichment) -> str:
    """Append enrichment metadata to a chunk's text as a header block."""
    header = (
        f"[Document Topic: {enrichment.document_topic}]\n"
        f"[Document Type: {enrichment.document_type}]\n"
        f"[Keywords: {', '.join(enrichment.keywords)}]\n\n"
    )
    return header + chunk_text


# ======================================================================
# Public API
# ======================================================================


def process_and_store(
    file_path: str,
    analyzer: Optional[DocumentAnalyzer] = None,
    embedder: Optional[EmbeddingProcessor] = None,
) -> list[str]:
    """
    Full pipeline: analyze → chunk → enrich → embed → store.

    Args:
        file_path: Path to the text file to process.
        analyzer:  Reusable DocumentAnalyzer instance (created fresh if None).
        embedder:  Reusable EmbeddingProcessor instance (created fresh if None).

    Returns:
        List of UUIDs assigned to the stored chunks.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # ── 1. Read file content ──────────────────────────────────
    text = path.read_text(encoding="utf-8")
    logger.info(f"Processing: {path.name}")

    # ── 2. Analyze with DocumentAnalyzer ──────────────────────
    if analyzer is None:
        analyzer = DocumentAnalyzer()
    enrichment_data = analyzer.analyze(file_path)
    enrichment = Enrichment(
        document_topic=enrichment_data["document_topic"],
        document_type=enrichment_data["document_type"],
        keywords=enrichment_data["keywords"],
    )
    logger.info(f"  Topic: {enrichment.document_topic}")

    # ── 3. Chunk the text ─────────────────────────────────────
    chunks = chunk_file(file_path)
    logger.info(f"  Chunks: {len(chunks)}")

    if not chunks:
        logger.warning("  No chunks produced — skipping.")
        return []

    # ── 4. Build enriched text for each chunk ─────────────────
    if embedder is None:
        embedder = EmbeddingProcessor()

    enriched_texts = [_build_enriched_text(c.text, enrichment) for c in chunks]
    embeddings = embedder.embed_batch(enriched_texts)
    logger.info(f"  Embeddings: {len(embeddings)} x {len(embeddings[0])}d")

    # ── 5. Gather file metadata ───────────────────────────────
    file_meta = _get_file_metadata(file_path)

    # ── 6. Store in ChromaDB ──────────────────────────────────
    collection = _get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for chunk, enriched, emb in zip(chunks, enriched_texts, embeddings):
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)
        documents.append(enriched)
        metadatas.append({
            "chunk_index": chunk.index,
            "token_count": chunk.token_count,
            "document_topic": enrichment.document_topic,
            "document_type": enrichment.document_type,
            "keywords": json.dumps(enrichment.keywords),
            "file_name": file_meta.file_name,
            "file_path": file_meta.file_path,
            "file_size": file_meta.file_size,
            "created_at": file_meta.created_at,
            "modified_at": file_meta.modified_at,
        })

    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    logger.info(f"  Stored {len(ids)} chunks in ChromaDB.")
    return ids


def search(
    query_embedding: list[float],
    top_k: Optional[int] = None,
) -> list[SearchResult]:
    """
    Search ChromaDB for chunks most similar to a query embedding.

    Args:
        query_embedding: The embedding vector of the query.
        top_k:           Number of results (defaults to ``chroma_top_k`` from settings).

    Returns:
        A list of ``SearchResult`` objects ordered by relevance.
    """
    collection = _get_collection()
    k = top_k or settings.chroma_top_k

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    search_results: list[SearchResult] = []
    for id_, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        search_results.append(
            SearchResult(
                id=id_,
                text=doc,
                index=meta.get("chunk_index", -1),
                token_count=meta.get("token_count", 0),
                distance=dist,
                metadata=dict(meta),
            )
        )

    return search_results


def reset_database() -> None:
    """Delete all data from the ChromaDB collection."""
    collection = _get_collection()
    count = collection.count()
    if count > 0:
        all_ids = collection.get()["ids"]
        collection.delete(ids=all_ids)
    logger.info(f"Deleted {count} records from ChromaDB.")


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vector_store.py process <file_path>")
        print("  python vector_store.py search <query_text>")
        print("  python vector_store.py reset")
        sys.exit(1)

    command = sys.argv[1]

    if command == "process":
        if len(sys.argv) < 3:
            print("Usage: python vector_store.py process <file_path>")
            sys.exit(1)
        ids = process_and_store(sys.argv[2])
        print(f"\nStored chunk IDs: {ids}")

    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python vector_store.py search <query_text>")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        embedder = EmbeddingProcessor()
        query_vec = embedder.embed(query)
        results = search(query_vec)
        print(f"\nTop {len(results)} results for: \"{query}\"\n")
        for r in results:
            print(f"  [{r.distance:.4f}] {r.metadata.get('file_name', '?')} "
                  f"(chunk {r.index})")
            print(f"       {r.text[:120]}...\n")

    elif command == "reset":
        reset_database()
        print("Database reset.")
