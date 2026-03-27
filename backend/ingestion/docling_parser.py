"""Docling-first PDF parser for structured ingestion blocks."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
import unicodedata

import fitz


class DoclingParser:
	"""Parse PDFs into normalized block dictionaries for downstream chunking."""

	# Typical symbols seen when PDF glyph encodings are decoded without proper cmap.
	_MOJIBAKE_HINT_CHARS = set("¶¤¦µ¿¡¢£¬÷×ØÐÞþðøåæçèéêëìíîïñòóôõöùúûüýÿ")

	def __init__(self) -> None:
		self._docling_converter: Any | None = None

	def _get_docling_converter(self):
		if self._docling_converter is None:
			from docling.document_converter import DocumentConverter

			self._docling_converter = DocumentConverter()
		return self._docling_converter

	def parse(self, pdf_path: str | Path) -> list[dict[str, Any]]:
		"""
		Parse a PDF and return blocks with keys:
		text, block_type, page_number, heading_context.
		"""
		path = Path(pdf_path)
		if not path.exists():
			raise FileNotFoundError(f"PDF not found: {path}")

		auto_batch_size = max(1, int(os.getenv("DOCLING_AUTO_BATCH_SIZE", "4")))
		explicit_batch_size = int(os.getenv("DOCLING_PAGE_BATCH_SIZE", "4"))

		with fitz.open(path) as doc:
			total_pages = int(doc.page_count)

		batch_size = explicit_batch_size if explicit_batch_size > 0 else auto_batch_size

		try:
			if batch_size > 0:
				print(
					f"[DoclingParser] Using Docling batched mode "
					f"(pages={total_pages}, batch_size={batch_size})"
				)
				blocks = self._parse_with_docling_batched(path, batch_size=batch_size)
			else:
				blocks = self._parse_with_docling(path)
			if blocks:
				return blocks
			print("[DoclingParser] WARNING: zero blocks extracted; using PyMuPDF fallback")
			return self._parse_with_pymupdf(path)
		except Exception as exc:
			print(f"[DoclingParser] Docling parse failed ({exc}); using PyMuPDF fallback")
			return self._parse_with_pymupdf(path)

	def _parse_with_docling_batched(self, pdf_path: Path, batch_size: int) -> list[dict[str, Any]]:
		"""Parse large PDFs in small page windows to reduce peak memory pressure."""
		if batch_size <= 0:
			return self._parse_with_docling(pdf_path)

		with fitz.open(pdf_path) as src_doc:
			total_pages = int(src_doc.page_count)

		all_blocks: list[dict[str, Any]] = []
		with tempfile.TemporaryDirectory(prefix="docling_batch_") as tmp_dir:
			tmp_root = Path(tmp_dir)

			for start_page in range(1, total_pages + 1, batch_size):
				end_page = min(total_pages, start_page + batch_size - 1)
				batch_pdf = tmp_root / f"batch_{start_page:04d}_{end_page:04d}.pdf"

				with fitz.open(pdf_path) as src_doc:
					with fitz.open() as dst_doc:
						dst_doc.insert_pdf(src_doc, from_page=start_page - 1, to_page=end_page - 1)
						dst_doc.save(batch_pdf)

				try:
					batch_blocks = self._parse_with_docling(batch_pdf, source_file=pdf_path.name)
				except Exception as exc:
					print(
						f"[DoclingParser] Batch {start_page}-{end_page} failed with Docling "
						f"({exc}); falling back to PyMuPDF for this batch"
					)
					batch_blocks = self._parse_with_pymupdf(
						pdf_path,
						source_file=pdf_path.name,
						start_page=start_page,
						end_page=end_page,
					)

				for block in batch_blocks:
					local_page = int(block.get("page_number") or 1)
					global_page = (start_page - 1) + local_page
					if global_page < start_page:
						global_page = start_page
					if global_page > end_page:
						global_page = end_page
					block["page_number"] = global_page
					block["source_file"] = pdf_path.name

				all_blocks.extend(batch_blocks)

		return all_blocks

	def is_page_scanned(self, page_blocks: list[dict[str, Any]]) -> bool:
		"""Return True when page text is empty or near-empty (OCR candidate)."""
		if not page_blocks:
			return True

		joined = " ".join((b.get("text") or "").strip() for b in page_blocks)
		text_len = len(joined.strip())
		alnum_count = sum(ch.isalnum() for ch in joined)

		return text_len < 40 or alnum_count < 20

	def is_page_garbled(self, page_blocks: list[dict[str, Any]]) -> bool:
		"""Return True when extracted text appears mojibake-garbled and OCR should be preferred."""
		if not page_blocks:
			return False

		garbled_blocks = 0
		substantial_blocks = 0
		for block in page_blocks:
			text = str(block.get("text") or "").strip()
			if len(text) < 40:
				continue
			substantial_blocks += 1
			if self.is_text_garbled(text):
				garbled_blocks += 1

		if substantial_blocks == 0:
			return False

		return garbled_blocks >= 1 and (garbled_blocks / substantial_blocks) >= 0.30

	def is_page_non_latin(self, page_blocks: list[dict[str, Any]]) -> bool:
		"""Return True when a page contains non-Latin text and should be OCR processed."""
		if not page_blocks:
			return False

		for block in page_blocks:
			text = str(block.get("text") or "").strip()
			if self.is_text_non_latin(text):
				return True

		return False

	def is_text_garbled(self, text: str) -> bool:
		"""Heuristic for broken glyph decoding from embedded non-Unicode PDF fonts."""
		if not text or len(text.strip()) < 5:
			return False

		total = len(text)
		if total == 0:
			return False

		weird = sum(1 for c in text if not c.isalnum() and c not in " \n")
		devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097F")

		# High symbol ratio usually indicates glyph decoding corruption.
		if (weird / total) > 0.3:
			return True

		# Common mojibake remnants in legacy encoded Indic text.
		if devanagari == 0 and any(x in text for x in ["É", "Ç", "Æ", "Ð", "ñ"]):
			return True

		return False

	def is_text_non_latin(self, text: str) -> bool:
		"""Return True when text contains substantial non-Latin script letters."""
		if not text:
			return False

		joined = " ".join(text.split())
		if len(joined) < 5:
			return False

		letter_count = 0
		non_latin_count = 0
		for ch in joined:
			if not ch.isalpha():
				continue
			letter_count += 1
			if self._is_non_latin_script_char(ch):
				non_latin_count += 1

		if letter_count == 0:
			return False

		return (non_latin_count / letter_count) >= 0.20

	@staticmethod
	def _is_non_latin_script_char(ch: str) -> bool:
		code = ord(ch)

		# Indic blocks (Devanagari through Malayalam and additional Indic ranges)
		if 0x0900 <= code <= 0x0DFF:
			return True
		if 0x1CD0 <= code <= 0x1CFF:
			return True
		if 0xA8E0 <= code <= 0xA8FF:
			return True

		# Arabic and related blocks
		if 0x0600 <= code <= 0x06FF:
			return True
		if 0x0750 <= code <= 0x077F:
			return True
		if 0x08A0 <= code <= 0x08FF:
			return True

		# CJK + Japanese + Korean
		if 0x4E00 <= code <= 0x9FFF:
			return True
		if 0x3040 <= code <= 0x30FF:
			return True
		if 0xAC00 <= code <= 0xD7AF:
			return True

		# Greek, Cyrillic, Hebrew
		if 0x0370 <= code <= 0x03FF:
			return True
		if 0x0400 <= code <= 0x04FF:
			return True
		if 0x0590 <= code <= 0x05FF:
			return True

		return False

	def _parse_with_docling(self, pdf_path: Path, source_file: str | None = None) -> list[dict[str, Any]]:
		converter = self._get_docling_converter()
		conversion = converter.convert(str(pdf_path))
		doc = conversion.document
		file_name = source_file or pdf_path.name

		blocks: list[dict[str, Any]] = []
		heading_by_page: dict[int, str] = {}

		for item, _level in doc.iterate_items():
			block_type = self._map_docling_item_type(item)
			if block_type is None:
				continue

			page_number = self._extract_page_number(item)
			text = self._extract_docling_text(item, doc)
			if not text:
				continue

			if block_type == "heading":
				heading_by_page[page_number] = text
				heading_context = text
			else:
				heading_context = heading_by_page.get(page_number, "")

			blocks.append(
				{
					"text": text,
					"block_type": block_type,
					"page_number": page_number,
					"source_file": file_name,
					"heading_context": heading_context,
				}
			)

		return blocks

	def _parse_with_pymupdf(
		self,
		pdf_path: Path,
		source_file: str | None = None,
		start_page: int | None = None,
		end_page: int | None = None,
	) -> list[dict[str, Any]]:
		blocks: list[dict[str, Any]] = []
		heading_by_page: dict[int, str] = {}
		file_name = source_file or pdf_path.name
		range_start = max(1, int(start_page or 1))

		with fitz.open(pdf_path) as doc:
			range_end = min(int(end_page or doc.page_count), int(doc.page_count))
			if range_end < range_start:
				return []

			for page_idx, page in enumerate(doc, start=1):
				if page_idx < range_start or page_idx > range_end:
					continue

				page_dict = page.get_text("dict")
				for block in page_dict.get("blocks", []):
					if block.get("type") != 0:
						continue

					text = self._extract_text_from_fitz_block(block)
					if not text:
						continue

					if self._looks_like_heading(text):
						block_type = "heading"
						heading_by_page[page_idx] = text
						heading_context = text
					else:
						block_type = "paragraph"
						heading_context = heading_by_page.get(page_idx, "")

					blocks.append(
						{
							"text": text,
							"block_type": block_type,
							"page_number": page_idx,
							"source_file": file_name,
							"heading_context": heading_context,
						}
					)

		return blocks

	@staticmethod
	def _map_docling_item_type(item: Any) -> str | None:
		label = str(getattr(item, "label", "")).lower()
		item_type = type(item).__name__.lower()

		if "table" in label or "table" in item_type:
			return "table"
		if "section_header" in label or "header" in label:
			return "heading"
		if "caption" in label:
			return "figure_caption"
		if any(k in label for k in ["text", "list_item", "footnote", "formula"]):
			return "paragraph"
		if "text" in item_type or "list" in item_type:
			return "paragraph"

		return None

	@staticmethod
	def _extract_page_number(item: Any) -> int:
		prov = getattr(item, "prov", None) or []
		if prov:
			page_no = getattr(prov[0], "page_no", None)
			if isinstance(page_no, int) and page_no > 0:
				return page_no
		return 1

	@staticmethod
	def _extract_docling_text(item: Any, doc: Any = None) -> str:
		if hasattr(item, "text") and isinstance(item.text, str):
			return item.text.strip()

		if hasattr(item, "export_to_markdown"):
			try:
				md = item.export_to_markdown(doc=doc) if doc is not None else item.export_to_markdown()
				if isinstance(md, str):
					return md.strip()
			except Exception:
				pass

		if hasattr(item, "export_to_html"):
			try:
				html = item.export_to_html()
				if isinstance(html, str):
					return html.strip()
			except Exception:
				pass

		return ""

	@staticmethod
	def _extract_text_from_fitz_block(block: dict[str, Any]) -> str:
		lines = []
		for line in block.get("lines", []):
			spans = line.get("spans", [])
			text = "".join(span.get("text", "") for span in spans).strip()
			if text:
				lines.append(text)
		return " ".join(lines).strip()

	@staticmethod
	def _looks_like_heading(text: str) -> bool:
		compact = " ".join(text.split())
		if not compact:
			return False

		if len(compact) <= 80 and compact.isupper():
			return True

		if len(compact) <= 120 and compact.endswith(":"):
			return True

		words = compact.split()
		if len(words) <= 10 and all(w[:1].isupper() for w in words if w and w[0].isalpha()):
			return True

		return False
