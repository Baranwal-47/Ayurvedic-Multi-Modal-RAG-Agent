"""Smoke-test Google Vision OCR on selected PDF pages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.ocr_pipeline import OCRPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Google Vision on PDF pages")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to PDF")
    parser.add_argument("--pages", type=int, nargs="+", required=True, help="1-based page numbers")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ocr = OCRPipeline()
    for page_number in args.pages:
        result = ocr.process_page(args.pdf, page_number, route_reason="forced")
        text_preview = str(result.get("text") or "").strip().replace("\n", " ")[:300]
        print(
            f"page={page_number} engine={result.get('engine_used')} "
            f"confidence={result.get('confidence')} paragraphs={len(result.get('text_units') or [])}"
        )
        print(text_preview)
        print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
