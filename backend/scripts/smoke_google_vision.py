"""Smoke-test Google Vision OCR on selected PDF pages."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--json-out-dir",
        type=Path,
        default=None,
        help="Optional directory to write per-page OCR JSON output",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ocr = OCRPipeline()
    if args.json_out_dir is not None:
        args.json_out_dir.mkdir(parents=True, exist_ok=True)

    for page_number in args.pages:
        result = ocr.process_page(args.pdf, page_number, route_reason="forced")
        text_preview = str(result.get("text") or "").strip().replace("\n", " ")[:300]
        safe_preview = text_preview.encode("cp1252", errors="replace").decode("cp1252", errors="replace")
        print(
            f"page={page_number} engine={result.get('engine_used')} "
            f"confidence={result.get('confidence')} paragraphs={len(result.get('text_units') or [])}"
        )
        print(safe_preview)
        print("-" * 60)

        if args.json_out_dir is not None:
            output = {
                "page_number": page_number,
                "engine_used": result.get("engine_used"),
                "confidence": result.get("confidence"),
                "text": result.get("text"),
                "text_units": result.get("text_units") or [],
                "word_units": result.get("word_units") or [],
            }
            output_path = args.json_out_dir / f"{args.pdf.stem}_page_{page_number}.json"
            output_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"json={output_path}")
            print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
