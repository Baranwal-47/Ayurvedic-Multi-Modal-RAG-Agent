"""End-to-end ingestion CLI for PDF parsing, OCR fallback, image captions, and chunking."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from ingestion.chunker import Chunker
from ingestion.docling_parser import DoclingParser
from ingestion.image_captioner import build_image_caption
from ingestion.image_extractor import ImageExtractor
from ingestion.ocr_pipeline import OCRPipeline, warmup_ocr
from normalization.diacritic_normalizer import DiacriticNormalizer


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Ingest a PDF end-to-end into chunk records")
	parser.add_argument("pdf", type=Path, help="Input PDF path")
	parser.add_argument(
		"--image-output-dir",
		type=Path,
		default=Path("data/images/_ingest"),
		help="Directory where extracted figure images are saved",
	)
	parser.add_argument(
		"--force-ocr-all-pages",
		action="store_true",
		help="Run OCR on every page regardless of parser scanned detection",
	)
	parser.add_argument(
		"--save-json",
		type=Path,
		default=None,
		help="Optional path to save full output payload (summary/chunks/images)",
	)
	parser.add_argument(
		"--max-print-chunks",
		type=int,
		default=3,
		help="How many sample chunks to print",
	)
	return parser.parse_args()


def _group_blocks_by_page(blocks: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
	grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
	for block in blocks:
		page = int(block.get("page_number") or 1)
		grouped[page].append(block)
	return grouped


def _inject_ocr_for_scanned_pages(
	pdf_path: Path,
	base_blocks: list[dict[str, Any]],
	parser: DoclingParser,
	ocr: OCRPipeline,
	force_ocr_all_pages: bool,
) -> tuple[list[dict[str, Any]], list[int], list[int], list[int], Counter, Counter]:
	with fitz.open(pdf_path) as doc:
		page_count = doc.page_count

	grouped = _group_blocks_by_page(base_blocks)
	scanned_pages: list[int] = []
	garbled_pages: list[int] = []
	non_latin_pages: list[int] = []
	ocr_engine_counts: Counter = Counter()
	ocr_reason_counts: Counter = Counter()
	rebuilt: list[dict[str, Any]] = []

	for page in range(1, page_count + 1):
		page_blocks = grouped.get(page, [])
		is_scanned = parser.is_page_scanned(page_blocks)
		is_garbled = parser.is_page_garbled(page_blocks)
		is_non_latin = parser.is_page_non_latin(page_blocks)
		should_ocr = force_ocr_all_pages or is_scanned or is_garbled or is_non_latin

		if not should_ocr:
			rebuilt.extend(page_blocks)
			continue

		reason = ""
		if force_ocr_all_pages:
			reason = "forced"
		elif is_scanned:
			reason = "scanned"
		elif is_garbled:
			reason = "garbled"
		elif is_non_latin:
			reason = "non_latin"
		ocr_reason_counts[reason or "unknown"] += 1

		if is_scanned:
			scanned_pages.append(page)
		if is_garbled:
			garbled_pages.append(page)
		if is_non_latin:
			non_latin_pages.append(page)

		ocr_result = ocr.process_page(pdf_path, page)
		engine = str(ocr_result.get("engine_used") or "unknown")
		ocr_engine_counts[engine] += 1

		text = str(ocr_result.get("text") or "").strip()
		if text:
			heading_context = ""
			for b in page_blocks:
				if str(b.get("block_type", "")) == "heading" and str(b.get("text", "")).strip():
					heading_context = str(b.get("text", "")).strip()
					break

			if is_garbled and not is_scanned and not is_non_latin:
				# Preserve cleaner blocks and replace only blocks likely affected by glyph-decoding corruption.
				for b in page_blocks:
					if not parser.is_text_garbled(str(b.get("text") or "")):
						rebuilt.append(b)
			else:
				# Scanned/non-latin pages are represented by OCR text for consistent Unicode output.
				pass

			rebuilt.append(
				{
					"text": text,
					"block_type": "paragraph",
					"page_number": page,
					"source_file": pdf_path.name,
					"heading_context": heading_context,
				}
			)
		else:
			rebuilt.extend(page_blocks)

	return rebuilt, scanned_pages, garbled_pages, non_latin_pages, ocr_engine_counts, ocr_reason_counts


def _build_image_caption_blocks(pdf_path: Path, image_output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	images = ImageExtractor().extract(pdf_path, image_output_dir)
	image_blocks: list[dict[str, Any]] = []

	for row in images:
		caption = build_image_caption(row)
		row["caption"] = caption

		if not str(caption).strip():
			continue

		image_blocks.append(
			{
				"text": caption,
				"block_type": "figure_caption",
				"page_number": int(row.get("page_number") or 1),
				"source_file": pdf_path.name,
				"heading_context": str(row.get("nearest_heading") or ""),
			}
		)

	return images, image_blocks


def run_ingestion(args: argparse.Namespace) -> dict[str, Any]:
	# Warm OCR in background while parsing/other setup work runs.
	warmup_ocr()

	pdf_path = args.pdf
	if not pdf_path.exists():
		raise FileNotFoundError(f"PDF not found: {pdf_path}")

	image_output_dir = args.image_output_dir / pdf_path.stem
	image_output_dir.mkdir(parents=True, exist_ok=True)

	parser = DoclingParser()
	ocr = OCRPipeline()
	normalizer = DiacriticNormalizer()
	chunker = Chunker()

	parsed_blocks = parser.parse(pdf_path)
	parsed_blocks, scanned_pages, garbled_pages, non_latin_pages, ocr_engine_counts, ocr_reason_counts = _inject_ocr_for_scanned_pages(
		pdf_path=pdf_path,
		base_blocks=parsed_blocks,
		parser=parser,
		ocr=ocr,
		force_ocr_all_pages=bool(args.force_ocr_all_pages),
	)

	images, image_caption_blocks = _build_image_caption_blocks(pdf_path, image_output_dir)

	# Keep this touchpoint explicit in Step 7 flow: normalizer is part of pipeline prep.
	for b in parsed_blocks:
		b["text"] = str(b.get("text") or "")
		b["normalized_text"] = normalizer.normalize(b["text"])

	ordered: list[tuple[int, int, dict[str, Any]]] = []
	for idx, block in enumerate(parsed_blocks):
		ordered.append((int(block.get("page_number") or 1), idx, block))

	base_idx = len(parsed_blocks)
	for j, block in enumerate(image_caption_blocks):
		# Keep image captions at the end of each page while preserving parser block flow.
		ordered.append((int(block.get("page_number") or 1), base_idx + j, block))

	ordered.sort(key=lambda x: (x[0], x[1]))
	all_blocks = [item[2] for item in ordered]

	chunks = chunker.chunk(all_blocks)

	with fitz.open(pdf_path) as doc:
		page_count = doc.page_count

	source_ok = all(str(c.get("source_file") or "") == pdf_path.name for c in chunks)
	chunk_reasonable = len(chunks) > 0 and len(chunks) <= (page_count * 10)

	summary = {
		"pdf": str(pdf_path),
		"source_file": pdf_path.name,
		"pages_parsed": page_count,
		"parser_blocks": len(parsed_blocks),
		"image_caption_blocks": len(image_caption_blocks),
		"chunks_created": len(chunks),
		"images_extracted": len(images),
		"scanned_pages_hit": len(scanned_pages),
		"scanned_page_numbers": scanned_pages,
		"garbled_pages_hit": len(garbled_pages),
		"garbled_page_numbers": garbled_pages,
		"non_latin_pages_hit": len(non_latin_pages),
		"non_latin_page_numbers": non_latin_pages,
		"ocr_engine_breakdown": dict(ocr_engine_counts),
		"ocr_routing_breakdown": dict(ocr_reason_counts),
		"source_file_all_chunks_ok": source_ok,
		"chunk_count_reasonable": chunk_reasonable,
	}

	return {
		"summary": summary,
		"chunks": chunks,
		"images": images,
	}


def main() -> int:
	args = _parse_args()
	result = run_ingestion(args)

	print("Ingestion Summary:")
	print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

	max_show = max(0, int(args.max_print_chunks))
	if max_show:
		print("\nSample chunks:")
		for i, ch in enumerate(result["chunks"][:max_show], start=1):
			print(f"[{i}] page={ch.get('page_number')} type={ch.get('block_type')} lang={ch.get('language')} shloka={ch.get('shloka_number')}")
			print(f"    text={str(ch.get('original_text') or '')[:220]}")

	if args.save_json:
		args.save_json.parent.mkdir(parents=True, exist_ok=True)
		with args.save_json.open("w", encoding="utf-8") as f:
			json.dump(result, f, ensure_ascii=False, indent=2)
		print(f"\nSaved JSON payload to: {args.save_json}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
