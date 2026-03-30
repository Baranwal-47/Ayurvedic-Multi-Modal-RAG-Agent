"""Delete Qdrant points for one document from rebuilt collections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db.qdrant_client import QdrantManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete Qdrant points by doc_id or source_file")
    parser.add_argument("--doc-id", type=str, default="", help="Exact doc_id to delete")
    parser.add_argument("--source-file", type=str, default="", help="Exact source_file to delete")
    parser.add_argument(
        "--collection",
        type=str,
        default="all",
        choices=["text_chunks", "image_chunks", "all"],
        help="Which collection(s) to target",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    qdrant = QdrantManager()

    if args.doc_id:
        deleted = qdrant.delete_by_doc_id(args.doc_id, None if args.collection == "all" else args.collection)
        print(f"Deleted {deleted} points for doc_id={args.doc_id}")
        return 0

    if args.source_file:
        deleted = qdrant.delete_by_source(args.source_file, None if args.collection == "all" else args.collection)
        print(f"Deleted {deleted} points for source_file={args.source_file}")
        return 0

    raise ValueError("Provide --doc-id or --source-file")


if __name__ == "__main__":
    raise SystemExit(main())
