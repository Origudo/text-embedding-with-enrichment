"""
Embedding Processor

Generates text embeddings using sentence-transformers models.
Configuration is managed via `settings.py` / .env.
"""

import logging

from settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Handles text embedding using a configurable sentence-transformers model."""

    def __init__(self, model_name: str | None = None, cache_folder: str | None = None):
        """
        Args:
            model_name: Override the default model from settings.
            cache_folder: Override the default cache folder from settings.
        """
        self.model_name = model_name or settings.embedding_model_name
        self.cache_folder = cache_folder or settings.embedding_cache_folder
        self._model = None

    @property
    def model(self):
        """Lazy-load the model, checking cache first before downloading."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {"model_name_or_path": self.model_name}
            if self.cache_folder:
                kwargs["cache_folder"] = self.cache_folder

            # sentence_transformers automatically checks the local cache
            # (under ~/.cache/huggingface/hub/) and only downloads if missing.
            logger.info(f"Loading model '{self.model_name}' (will use cached copy if available)...")
            self._model = SentenceTransformer(**kwargs)
            logger.info(f"Model '{self.model_name}' loaded successfully.")
        return self._model

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string and return the embedding vector.

        Args:
            text: The input sentence or message to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: A list of sentences to embed.

        Returns:
            A list of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "This is a sample sentence to embed."

    processor = EmbeddingProcessor()
    vec = processor.embed(text)
    print(f"Text: {text}")
    print(f"Embedding dimension: {len(vec)}")
    print(f"First 10 values: {vec[:10]}")
