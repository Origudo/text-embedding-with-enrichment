"""RAG processor — embed query → retrieve chunks → generate answer."""

from dataclasses import dataclass

from embedding_processor import EmbeddingProcessor
from settings import settings
from vector_store import SearchResult, search

from .llm_client import generate_answer


@dataclass
class RagResult:
    """The output of a RAG query."""

    answer: str
    chunks: list[SearchResult]


class RagProcessor:
    """Orchestrates the RAG loop: embed → retrieve → generate."""

    def __init__(self):
        self._embedder = EmbeddingProcessor()

    def query(self, question: str, top_k: int | None = None) -> RagResult:
        """Run the full RAG pipeline for a single question.

        Args:
            question: The user's natural language query.
            top_k:    Number of chunks to retrieve (default from settings).

        Returns:
            A ``RagResult`` with the generated answer and retrieved chunks.
        """
        # 1. Embed the question
        query_vec = self._embedder.embed(question)

        # 2. Retrieve relevant chunks
        chunks = search(query_vec, top_k=top_k or settings.chroma_top_k)

        # 3. Generate answer from context
        answer = generate_answer(question, chunks)

        return RagResult(answer=answer, chunks=chunks)
