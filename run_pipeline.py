"""Run the full vector_store pipeline on a test file."""
from document_analyzer import DocumentAnalyzer
from vector_store import process_and_store, search, reset_database
from embedding_processor import EmbeddingProcessor

reset_database()

ids = process_and_store(
    "sample_files/philosophy_consciousness.txt",
    analyzer=DocumentAnalyzer(model="google/gemma-4-e4b"),
)
print(f"\nStored {len(ids)} chunks\n")

# Search
embedder = EmbeddingProcessor()
query_vec = embedder.embed("What is the hard problem of consciousness?")
results = search(query_vec, top_k=2)

print("Search results:")
for r in results:
    print(f"  Distance: {r.distance:.4f}")
    print(f"  File: {r.metadata.get('file_name')}")
    print(f"  Topic: {r.metadata.get('document_topic')}")
    print(f"  Text: {r.text[:150]}...\n")
