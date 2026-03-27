"""Shared OCR routing and merge logic for ingestion pipelines."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import time
from typing import Any

import fitz


@dataclass(frozen=True)
class OCRRoutingDecision:
	page_number: int
	scanned: bool
	garbled: bool
	non_latin: bool
	use_ocr: bool
	reason: str


def _group_blocks_by_page(blocks: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
	grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
	for block in blocks:
		page = int(block.get("page_number") or 1)
		grouped[page].append(block)
	return grouped


def _combined_page_text(page_blocks: list[dict[str, Any]]) -> str:
	parts = [str((b or {}).get("text") or "").strip() for b in page_blocks]
	return "\n".join(p for p in parts if p)


def _largest_image_ratio_for_page(pdf_path: Path, page_number: int) -> float:
	"""Return max image area ratio on a page using embedded image placement metadata."""
	try:
		with fitz.open(pdf_path) as doc:
			if page_number < 1 or page_number > doc.page_count:
				return 0.0
			page = doc.load_page(page_number - 1)
			page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
			largest = 0.0
			for info in page.get_image_info(xrefs=True):
				bbox = info.get("bbox")
				if not bbox:
					continue
				rect = fitz.Rect(bbox)
				area = max(0.0, float(rect.width)) * max(0.0, float(rect.height))
				largest = max(largest, area / page_area)
			return float(largest)
	except Exception:
		return 0.0


def detect_scanned_page(
	blocks: list[dict[str, Any]],
	largest_image_ratio: float,
	used_scanned_fallback: bool,
) -> bool:
	"""Robust scanned-page detection combining text, image coverage, and fallback usage signals."""
	text_len = sum(len(str((b or {}).get("text") or "")) for b in blocks)
	return bool(
		text_len < 50
		or float(largest_image_ratio) > 0.8
		or bool(used_scanned_fallback)
	)


def _is_garbled_table_candidate(page_blocks: list[dict[str, Any]], parser: Any) -> bool:
	"""Return True when garbled content is likely table-like and needs stronger OCR preprocessing."""
	if not page_blocks:
		return False

	has_table_block = any(str((b or {}).get("block_type") or "") == "table" for b in page_blocks)
	if has_table_block:
		return True

	garbled_count = 0
	pipe_count = 0
	for block in page_blocks:
		text = str((block or {}).get("text") or "")
		if parser.is_text_garbled(text):
			garbled_count += 1
		if "|" in text:
			pipe_count += 1

	return garbled_count >= 1 and pipe_count >= 1


def classify_page_for_ocr(
	page_number: int,
	page_blocks: list[dict[str, Any]],
	parser: Any,
	pdf_path: Path,
	used_scanned_fallback: bool = False,
) -> OCRRoutingDecision:
	"""Classify a page and emit structured OCR routing logs."""
	page_text = _combined_page_text(page_blocks)
	largest_image_ratio = _largest_image_ratio_for_page(pdf_path, page_number)
	scanned = detect_scanned_page(page_blocks, largest_image_ratio, used_scanned_fallback)
	garbled = parser.is_page_garbled(page_blocks)
	non_latin = parser.is_page_non_latin(page_blocks)
	# Deterministic routing:
	# - scanned pages always use OCR
	# - digitized pages only use OCR when parser text is clearly garbled
	use_ocr = bool(scanned or garbled)

	reason = "none"
	if scanned:
		reason = "scanned"
	elif garbled:
		reason = "garbled"

	print(
		f"[OCR ROUTING] page={page_number}, scanned={scanned}, garbled={garbled}, "
		f"non_latin={non_latin}, image_ratio={largest_image_ratio:.3f}, "
		f"fallback_scanned={bool(used_scanned_fallback)}, use_ocr={use_ocr}"
	)

	return OCRRoutingDecision(
		page_number=page_number,
		scanned=scanned,
		garbled=garbled,
		non_latin=non_latin,
		use_ocr=use_ocr,
		reason=reason,
	)


def _split_text_to_slots(text: str, slots: int) -> list[str]:
	if slots <= 0:
		return []

	lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
	if not lines:
		return [str(text or "").strip()] + [""] * (slots - 1)

	if len(lines) <= slots:
		return lines + [""] * (slots - len(lines))

	# Distribute line chunks across slots while preserving order.
	base = len(lines) // slots
	extra = len(lines) % slots
	out: list[str] = []
	idx = 0
	for i in range(slots):
		take = base + (1 if i < extra else 0)
		chunk = "\n".join(lines[idx : idx + take]).strip()
		out.append(chunk)
		idx += take
	return out


def _merge_text_into_block_subset(
	page_blocks: list[dict[str, Any]],
	target_indexes: list[int],
	ocr_text: str,
) -> list[dict[str, Any]]:
	if not page_blocks:
		return []
	if not target_indexes:
		return page_blocks

	segments = _split_text_to_slots(ocr_text, len(target_indexes))
	updated = [dict(b) for b in page_blocks]

	for idx, seg in zip(target_indexes, segments):
		updated[idx]["text"] = seg

	return updated


def attach_ocr_to_blocks(
	parsed_blocks: list[dict[str, Any]],
	page_number: int,
	ocr_result: dict[str, Any],
	*,
	scanned: bool,
	garbled: bool,
	parser: Any,
	source_file: str,
) -> list[dict[str, Any]]:
	"""
	Merge OCR output into parsed page blocks.

	Rules:
	- If scanned: full replacement intent, while preserving block metadata shape when present.
	- If garbled: preserve structure and overwrite only garbled block text.
	- Always preserve non-text metadata keys.
	"""
	ocr_text = str((ocr_result or {}).get("text") or "").strip()
	if not ocr_text and not garbled:
		return parsed_blocks
	if garbled and not ocr_text:
		ocr_text = str((ocr_result or {}).get("raw_text") or "").strip()

	if scanned:
		if not parsed_blocks:
			return [
				{
					"text": ocr_text,
					"block_type": "paragraph",
					"page_number": page_number,
					"source_file": source_file,
					"heading_context": "",
				}
			]
		all_indexes = list(range(len(parsed_blocks)))
		return _merge_text_into_block_subset(parsed_blocks, all_indexes, ocr_text)

	if garbled:
		target_indexes = []
		for i, block in enumerate(parsed_blocks):
			text = str((block or {}).get("text") or "")
			if parser.is_text_garbled(text):
				target_indexes.append(i)

		if not target_indexes:
			target_indexes = [i for i, block in enumerate(parsed_blocks) if str((block or {}).get("text") or "").strip()]

		return _merge_text_into_block_subset(parsed_blocks, target_indexes, ocr_text)

	return parsed_blocks


def _resolve_ocr_batch_size(page_count: int) -> int:
	"""Resolve OCR routing batch size with always-on batching defaults."""
	raw = str(os.getenv("OCR_PAGE_BATCH_SIZE", "")).strip()
	if raw:
		try:
			return max(1, int(raw))
		except Exception:
			print(f"[OCR ROUTING] Invalid OCR_PAGE_BATCH_SIZE='{raw}', using auto mode")

	auto_batch = max(1, int(os.getenv("OCR_AUTO_BATCH_SIZE", "4")))
	return min(page_count, auto_batch)


def _iter_page_batches(page_count: int, batch_size: int):
	start = 1
	while start <= page_count:
		end = min(page_count, start + batch_size - 1)
		yield start, end
		start = end + 1


def apply_ocr_routing_to_document(
	pdf_path: str | Path,
	base_blocks: list[dict[str, Any]],
	parser: Any,
	ocr: Any,
	force_ocr_all_pages: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	"""Apply shared page-wise OCR routing and return rebuilt blocks plus routing stats."""
	path = Path(pdf_path)
	with fitz.open(path) as doc:
		page_count = doc.page_count

	batch_size = _resolve_ocr_batch_size(page_count)
	batch_cooldown_sec = float(os.getenv("OCR_BATCH_COOLDOWN_SEC", "0"))
	print(
		f"[OCR ROUTING] page_count={page_count}, batch_size={batch_size}, "
		f"cooldown={batch_cooldown_sec:.2f}s"
	)

	grouped = _group_blocks_by_page(base_blocks)
	rebuilt: list[dict[str, Any]] = []
	scanned_pages: list[int] = []
	garbled_pages: list[int] = []
	non_latin_pages: list[int] = []
	ocr_engine_counts: Counter = Counter()
	ocr_reason_counts: Counter = Counter()

	for batch_start, batch_end in _iter_page_batches(page_count, batch_size):
		print(f"[OCR ROUTING] Processing batch pages {batch_start}-{batch_end}")

		for page in range(batch_start, batch_end + 1):
			page_blocks = grouped.get(page, [])
			decision = classify_page_for_ocr(page, page_blocks, parser, path)
			use_ocr = bool(force_ocr_all_pages) or decision.use_ocr
			reason = "forced" if force_ocr_all_pages else decision.reason

			if decision.scanned:
				scanned_pages.append(page)
			if decision.garbled:
				garbled_pages.append(page)
			if decision.non_latin:
				non_latin_pages.append(page)

			if not use_ocr:
				rebuilt.extend(page_blocks)
				continue

			ocr_reason_counts[reason or "unknown"] += 1
			ocr_profile = "garbled_table" if (decision.garbled and _is_garbled_table_candidate(page_blocks, parser)) else "default"
			ocr_result = ocr.process_page(path, page, route_reason=reason, ocr_profile=ocr_profile)
			engine = str((ocr_result or {}).get("engine_used") or "unknown")
			print(
				f"[OCR ENGINE] page={page}, scanned_page={decision.scanned}, "
				f"garbled_page={decision.garbled}, reason={reason}, profile={ocr_profile}, engine={engine}"
			)
			ocr_engine_counts[engine] += 1

			updated = attach_ocr_to_blocks(
				parsed_blocks=page_blocks,
				page_number=page,
				ocr_result=ocr_result,
				scanned=decision.scanned or bool(force_ocr_all_pages),
				garbled=decision.garbled and not decision.scanned,
				parser=parser,
				source_file=path.name,
			)
			rebuilt.extend(updated)

		gc.collect()
		if batch_cooldown_sec > 0:
			time.sleep(batch_cooldown_sec)

	stats = {
		"scanned_pages": scanned_pages,
		"garbled_pages": garbled_pages,
		"non_latin_pages": non_latin_pages,
		"ocr_engine_counts": dict(ocr_engine_counts),
		"ocr_reason_counts": dict(ocr_reason_counts),
	}
	return rebuilt, stats
