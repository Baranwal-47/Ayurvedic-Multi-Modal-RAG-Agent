"""Docling-first PDF parser for structured ingestion blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import unicodedata

import fitz


class DoclingParser:
	"""Parse PDFs into normalized block dictionaries for downstream chunking."""

	# Typical symbols seen when PDF glyph encodings are decoded without proper cmap.
	_MOJIBAKE_HINT_CHARS = set("¶¤¦µ¿¡¢£¬÷×ØÐÞþðøåæçèéêëìíîïñòóôõöùúûüýÿ")

	def parse(self, pdf_path: str | Path) -> list[dict[str, Any]]:
		"""
		Parse a PDF and return blocks with keys:
		text, block_type, page_number, heading_context.
		"""
		path = Path(pdf_path)
		if not path.exists():
			raise FileNotFoundError(f"PDF not found: {path}")

		try:
			blocks = self._parse_with_docling(path)
			if blocks:
				return blocks
			print("[DoclingParser] WARNING: zero blocks extracted; using PyMuPDF fallback")
			return self._parse_with_pymupdf(path)
		except Exception as exc:
			print(f"[DoclingParser] Docling parse failed ({exc}); using PyMuPDF fallback")
			return self._parse_with_pymupdf(path)

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

	def _parse_with_docling(self, pdf_path: Path) -> list[dict[str, Any]]:
		from docling.document_converter import DocumentConverter

		converter = DocumentConverter()
		conversion = converter.convert(str(pdf_path))
		doc = conversion.document

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
					"source_file": pdf_path.name,
					"heading_context": heading_context,
				}
			)

		return blocks

	def _parse_with_pymupdf(self, pdf_path: Path) -> list[dict[str, Any]]:
		blocks: list[dict[str, Any]] = []
		heading_by_page: dict[int, str] = {}

		with fitz.open(pdf_path) as doc:
			for page_idx, page in enumerate(doc, start=1):
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
							"source_file": pdf_path.name,
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
