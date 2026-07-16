"""Interactive CLI — prompt for a query, run RAG, print the answer."""

from .rag_processor import RagProcessor


def _print_results(results) -> None:
    """Pretty-print search results."""
    if not results:
        print("No results found.")
        return
    print(f"\nFound {len(results)} result(s):\n")
    for i, r in enumerate(results, start=1):
        meta = r.metadata
        topic = meta.get("document_topic", "?")
        fname = meta.get("file_name", "?")
        print(f"  {i}. [{topic}] ({fname}) — distance: {r.distance:.4f}")


def main() -> None:
    """Loop: prompt → embed → retrieve → generate → print."""
    processor = RagProcessor()
    print("RAG Query Interface (type 'exit' or 'quit' to stop)\n")

    while True:
        query = input("Enter your search query: ").strip()

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not query:
            print("Query cannot be empty.\n")
            continue

        print("Searching and generating answer...")
        result = processor.query(query)

        _print_results(result.chunks)

        print(f"\n{'=' * 60}")
        print("ANSWER")
        print(f"{'=' * 60}")
        print(result.answer)
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
