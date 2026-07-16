"""Initial integration test for search_helper."""
import sys
from pathlib import Path

# Ensure project root is on sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

print("=" * 50)
print("TEST: ChromaDB health")
print("=" * 50)
from vector_store import _get_collection
collection = _get_collection()
count = collection.count()
print(f"  Records in ChromaDB: {count}")

print("\n" + "=" * 50)
print("TEST: Embedding")
print("=" * 50)
from embedding_processor import EmbeddingProcessor
ep = EmbeddingProcessor()
vec = ep.embed("What is machine learning?")
print(f"  Embedding dimension: {len(vec)}")

print("\n" + "=" * 50)
print("TEST: Vector search")
print("=" * 50)
from vector_store import search
results = search(vec, top_k=3)
print(f"  Results found: {len(results)}")
for r in results:
    fname = r.metadata.get("file_name", "?")
    topic = r.metadata.get("document_topic", "?")
    print(f"    [{r.distance:.4f}] {fname} — {topic}")

print("\n" + "=" * 50)
print("TEST: RAG processor (needs LM Studio running)")
print("=" * 50)
import requests
from settings import PROVIDERS
try:
    r = requests.get(PROVIDERS["lmstudio"]["health_url"], timeout=3)
    lm_ok = r.status_code == 200
    print(f"  LM Studio reachable: {lm_ok}")
except Exception:
    lm_ok = False
    print(f"  LM Studio reachable: No")

if lm_ok:
    from search_helper.rag_processor import RagProcessor
    processor = RagProcessor()
    result = processor.query("What is consciousness?", top_k=2)
    print(f"  Chunks retrieved: {len(result.chunks)}")
    print(f"  Answer preview: {result.answer[:200]}...")
else:
    print("  Skipping RAG test — start LM Studio to test full pipeline.")

print("\n" + "=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)
