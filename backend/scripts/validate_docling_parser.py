"""Validation script for ingestion.docling_parser.DoclingParser.

Run examples:
    python scripts/validate_docling_parser.py --pdf "data/pdfs/file.pdf"
    python scripts/validate_docling_parser.py --glob "data/pdfs/*.pdf" --max-files 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fitz

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingestion.docling_parser import DoclingParser
from ingestion.ocr_routing import classify_page_for_ocr

REQUIRED_KEYS = {"text", "block_type", "page_number", "heading_context"}
ALLOWED_TYPES = {"paragraph", "heading", "table", "figure_caption"}


def summarize_pdf(pdf_path: Path, parser: DoclingParser) -> dict[str, Any]:
    blocks = parser.parse(pdf_path)

    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        by_page[int(block.get("page_number", 0))].append(block)

    block_types = Counter(str(b.get("block_type", "")) for b in blocks)

    issues: list[str] = []
    warnings: list[str] = []

    if not blocks:
        issues.append("No blocks extracted")

    for idx, block in enumerate(blocks):
        missing = REQUIRED_KEYS.difference(block.keys())
        if missing:
            issues.append(f"Block {idx} missing keys: {sorted(missing)}")

        btype = str(block.get("block_type", ""))
        if btype not in ALLOWED_TYPES:
            issues.append(f"Block {idx} has unknown block_type: {btype}")

        text = str(block.get("text", "")).strip()
        if not text:
            issues.append(f"Block {idx} has empty text")

    pages_with_no_blocks = []
    pages_flagged_scanned = []

    for page_no in range(1, total_pages + 1):
        page_blocks = by_page.get(page_no, [])
        if not page_blocks:
            pages_with_no_blocks.append(page_no)
        decision = classify_page_for_ocr(
            page_number=page_no,
            page_blocks=page_blocks,
            parser=parser,
            pdf_path=pdf_path,
        )
        if decision.scanned:
            pages_flagged_scanned.append(page_no)

    if pages_with_no_blocks:
        warnings.append(
            f"Pages with no extracted blocks: {pages_with_no_blocks[:20]}"
        )

    if "table" not in block_types:
        warnings.append("No table blocks found (may be expected for this PDF)")

    if "figure_caption" not in block_types:
        warnings.append("No figure_caption blocks found (may be expected for this PDF)")

    if not any(
        b.get("block_type") == "paragraph" and str(b.get("heading_context", "")).strip()
        for b in blocks
    ):
        warnings.append("No paragraph with heading_context found")

    return {
        "pdf": str(pdf_path),
        "total_pages": total_pages,
        "total_blocks": len(blocks),
        "block_type_counts": dict(block_types),
        "pages_with_no_blocks": pages_with_no_blocks,
        "pages_flagged_scanned": pages_flagged_scanned,
        "issues": issues,
        "warnings": warnings,
        "ok": len(issues) == 0,
    }


def iter_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.pdf:
        return [Path(args.pdf)]

    matches = sorted(Path().glob(args.glob))
    if args.max_files > 0:
        matches = matches[: args.max_files]
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DoclingParser extraction quality")
    parser.add_argument("--pdf", type=str, help="Single PDF path to validate")
    parser.add_argument(
        "--glob",
        type=str,
        default="data/pdfs/*.pdf",
        help="Glob pattern when --pdf is not provided",
    )
    parser.add_argument("--max-files", type=int, default=1, help="Limit matched files")
    parser.add_argument(
        "--json", action="store_true", help="Output full JSON report for each file"
    )
    args = parser.parse_args()

    pdfs = iter_pdfs(args)
    if not pdfs:
        print("No PDF files found for validation")
        return 2

    parser_impl = DoclingParser()
    overall_ok = True

    for pdf in pdfs:
        if not pdf.exists():
            print(f"[FAIL] Missing file: {pdf}")
            overall_ok = False
            continue

        report = summarize_pdf(pdf, parser_impl)

        status = "PASS" if report["ok"] else "FAIL"
        print(f"\n[{status}] {report['pdf']}")
        print(f"  pages={report['total_pages']} blocks={report['total_blocks']}")
        print(f"  block_type_counts={report['block_type_counts']}")
        print(f"  pages_flagged_scanned={report['pages_flagged_scanned']}")

        for issue in report["issues"]:
            print(f"  ISSUE: {issue}")

        for warning in report["warnings"]:
            print(f"  WARN: {warning}")

        if args.json:
            print("  JSON:")
            print(json.dumps(report, ensure_ascii=False, indent=2))

        overall_ok = overall_ok and report["ok"]

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
