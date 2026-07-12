"""Quick test script — run with: .venv\Scripts\python test_all.py"""

from document_analyzer import DocumentAnalyzer
from embedding_processor import EmbeddingProcessor

# ── Test Document Analyzer ────────────────────────────────
print("=" * 60)
print("DOCUMENT ANALYZER")
print("=" * 60)

da = DocumentAnalyzer()
print(f"LLM running: {da.is_running()}\n")

for name in [
    "technology_ai.txt",
    "personal_journal.txt",
    "financial_report.txt",
    "meeting_notes.txt",
    "recipe_notes.txt",
    "project_proposal.txt",
]:
    path = f"sample_files/{name}"
    try:
        r = da.analyze(path)
        print(f"  File: {name}")
        print(f"  Topic: {r['document_topic']}")
        print(f"  Type:  {r['document_type']}")
        print(f"  Tags:  {', '.join(r['keywords'][:4])}")
        print()
    except Exception as e:
        print(f"  File: {name} — ERROR: {e}\n")

# ── Test Embedding Processor ──────────────────────────────
print("=" * 60)
print("EMBEDDING PROCESSOR")
print("=" * 60)

ep = EmbeddingProcessor()
text = "Machine learning is transforming the world."
vec = ep.embed(text)
print(f"  Text: \"{text}\"")
print(f"  Embedding dimension: {len(vec)}")
print(f"  First 5 values: {vec[:5]}")
print()

# Batch test
texts = ["Hello world", "AI is cool", "Python is great"]
vecs = ep.embed_batch(texts)
print(f"  Batch of {len(texts)} texts")
print(f"  Embedding dimensions: {[len(v) for v in vecs]}")
