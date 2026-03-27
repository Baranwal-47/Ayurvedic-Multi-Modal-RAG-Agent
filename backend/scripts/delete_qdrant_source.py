"""Delete Qdrant points for a single source_file from text/image collections.

Examples:
    python scripts/delete_qdrant_source.py --source-file "93aa6-145-rasa-shastra.pdf"
    python scripts/delete_qdrant_source.py --source-file "93aa6-145-rasa-shastra.pdf" --collection all
    python scripts/delete_qdrant_source.py --source-file "some.pdf" --collection image_chunks --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchValue

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db.qdrant_client import QdrantManager


def _build_filter(source_file: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="source_file",
                match=MatchValue(value=source_file),
            )
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete Qdrant points by payload.source_file"
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default="93aa6-145-rasa-shastra.pdf",
        help="Exact payload.source_file value to delete",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="text_chunks",
        choices=["text_chunks", "image_chunks", "all"],
        help="Which collection(s) to target",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matching point counts; do not delete",
    )
    return parser.parse_args()


def _resolve_targets(qdrant: QdrantManager, collection: str) -> list[str]:
    if collection == "all":
        return [qdrant.text_collection, qdrant.image_collection]
    return [qdrant._resolve_collection_name(collection)]


def main() -> int:
    args = _parse_args()
    source_file = str(args.source_file or "").strip()
    if not source_file:
        raise ValueError("--source-file cannot be empty")

    qdrant = QdrantManager()
    delete_filter = _build_filter(source_file)
    targets = _resolve_targets(qdrant, args.collection)

    total_matches = 0
    print(f"[DELETE] source_file={source_file}")
    print(f"[DELETE] collection_mode={args.collection}")

    for collection_name in targets:
        match_count = int(
            qdrant.client.count(collection_name=collection_name, count_filter=delete_filter).count
        )
        total_matches += match_count
        print(f"[DELETE] matches in {collection_name}: {match_count}")

    if args.dry_run:
        print("[DELETE] dry-run enabled; no points deleted")
        return 0

    total_deleted = 0
    for collection_name in targets:
        total_deleted += int(qdrant.delete_by_source(source_file, collection_name))

    print(f"[DELETE] total_deleted={total_deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
