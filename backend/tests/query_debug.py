from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.query_engine import QueryEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug the retrieval pipeline for one or many queries.")
    parser.add_argument("query", nargs="?", help="User query to run through retrieval, rerank, and context assembly.")
    parser.add_argument("--doc-id", dest="doc_id")
    parser.add_argument("--page-start", dest="page_start", type=int)
    parser.add_argument("--page-end", dest="page_end", type=int)
    parser.add_argument("--languages", nargs="*", default=[])
    parser.add_argument("--scripts", nargs="*", default=[])
    parser.add_argument("--chunk-types", nargs="*", default=[])
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep QueryEngine and models loaded, then accept multiple queries in one process.",
    )
    parser.add_argument(
        "--prewarm",
        action="store_true",
        help="Load the embedder and reranker before the first query so startup cost is paid once.",
    )
    return parser


def _run_query(engine: QueryEngine, *, query: str, args) -> None:
    query_started = time.perf_counter()
    result = engine.debug_query(
        query=query,
        doc_id=args.doc_id,
        page_start=args.page_start,
        page_end=args.page_end,
        languages=args.languages,
        scripts=args.scripts,
        chunk_types=args.chunk_types,
    )
    query_elapsed = time.perf_counter() - query_started

    print(f"\n=== QUERY ===\n{query}")

    print("\n=== QUERY BUNDLE ===")
    print(json.dumps(result["query_bundle"], indent=2, ensure_ascii=False))

    print("\n=== RETRIEVED CHUNKS ===")
    print(json.dumps(result["retrieved_candidates"], indent=2, ensure_ascii=False))

    print("\n=== RERANKED CHUNKS ===")
    print(json.dumps(result["reranked_candidates"], indent=2, ensure_ascii=False))

    print("\n=== FINAL CONTEXT ===")
    print(json.dumps(result["final_context"], indent=2, ensure_ascii=False))

    print("\n=== TIMINGS ===")
    print(json.dumps(result["timings"], indent=2, ensure_ascii=False))
    print(f"\n=== QUERY ELAPSED ===\n{query_elapsed:.2f}s")


def _interactive_loop(engine: QueryEngine, args) -> int:
    print("\nInteractive mode ready. Enter a query, or type `exit` to quit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            return 0
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting interactive mode.")
            return 0
        _run_query(engine, query=query, args=args)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    startup_started = time.perf_counter()
    engine = QueryEngine()
    if args.prewarm:
        engine.prewarm(load_reranker=True)
    startup_elapsed = time.perf_counter() - startup_started
    print(f"QueryEngine startup complete in {startup_elapsed:.2f}s")

    if args.interactive or not args.query:
        if args.query:
            _run_query(engine, query=args.query, args=args)
        return _interactive_loop(engine, args)

    _run_query(engine, query=args.query, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
