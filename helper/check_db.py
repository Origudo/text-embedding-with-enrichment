"""
Helper to inspect the contents of ChromaDB.

Usage:
    python helper/check_db.py              # List all records
    python helper/check_db.py --count       # Just show count
    python helper/check_db.py --id <uuid>   # Show specific record
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so we can import from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_store import _get_collection


def list_all(show_full_text: bool = False):
    """Print all records stored in ChromaDB."""
    collection = _get_collection()
    data = collection.get(include=["documents", "metadatas", "embeddings"])

    if not data["ids"]:
        print("ChromaDB is empty.")
        return

    print(f"\n{'='*60}")
    print(f"Total records: {len(data['ids'])}")
    print(f"{'='*60}\n")

    for i, (id_, doc, meta, emb) in enumerate(
        zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"])
    ):
        print(f"--- Record {i+1} ---")
        print(f"  ID:         {id_}")
        print(f"  File:       {meta.get('file_name', '?')}")
        print(f"  Path:       {meta.get('file_path', '?')}")
        print(f"  Chunk:      {meta.get('chunk_index', '?')}")
        print(f"  Tokens:     {meta.get('token_count', '?')}")
        print(f"  Topic:      {meta.get('document_topic', '?')}")
        print(f"  Type:       {meta.get('document_type', '?')}")
        print(f"  Keywords:   {meta.get('keywords', '?')}")
        print(f"  Created:    {meta.get('created_at', '?')}")
        print(f"  Modified:   {meta.get('modified_at', '?')}")
        print(f"  File Size:  {meta.get('file_size', '?')} bytes")
        print(f"  Embedding:  {len(emb)}-dimensional vector")
        if show_full_text:
            print(f"  Text:\n{doc}\n")
        else:
            print(f"  Text (first 200 chars): {doc[:200]}...\n")


def show_count():
    """Print only the record count."""
    collection = _get_collection()
    print(f"Total records in ChromaDB: {collection.count()}")


def show_by_id(record_id: str):
    """Show a single record by its UUID."""
    collection = _get_collection()
    data = collection.get(
        ids=[record_id],
        include=["documents", "metadatas", "embeddings"],
    )

    if not data["ids"]:
        print(f"No record found with ID: {record_id}")
        return

    id_ = data["ids"][0]
    doc = data["documents"][0]
    meta = data["metadatas"][0]
    emb = data["embeddings"][0]

    print(f"\n--- Record ---")
    print(f"  ID:         {id_}")
    print(f"  File:       {meta.get('file_name', '?')}")
    print(f"  Path:       {meta.get('file_path', '?')}")
    print(f"  Chunk:      {meta.get('chunk_index', '?')}")
    print(f"  Tokens:     {meta.get('token_count', '?')}")
    print(f"  Topic:      {meta.get('document_topic', '?')}")
    print(f"  Type:       {meta.get('document_type', '?')}")
    print(f"  Keywords:   {meta.get('keywords', '?')}")
    print(f"  Created:    {meta.get('created_at', '?')}")
    print(f"  Modified:   {meta.get('modified_at', '?')}")
    print(f"  File Size:  {meta.get('file_size', '?')} bytes")
    print(f"  Embedding:  {len(emb)}-dimensional vector")
    print(f"  Text:\n{doc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB contents")
    parser.add_argument("--count", action="store_true", help="Show only record count")
    parser.add_argument("--id", type=str, help="Show a specific record by UUID")
    parser.add_argument("--full-text", action="store_true", help="Show full text of each record")
    args = parser.parse_args()

    if args.count:
        show_count()
    elif args.id:
        show_by_id(args.id)
    else:
        list_all(show_full_text=args.full_text)
