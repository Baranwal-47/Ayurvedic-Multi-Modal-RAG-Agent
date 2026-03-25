"""Token-aware chunking for parsed document blocks."""

from __future__ import annotations

import re
from typing import Any

import tiktoken
from langdetect import detect

from normalization.diacritic_normalizer import DiacriticNormalizer


class Chunker:
	"""Build normalized retrieval chunks from parser output blocks."""

	def __init__(self, max_tokens: int = 400, overlap_tokens: int = 50) -> None:
		self.max_tokens = max_tokens
		self.overlap_tokens = overlap_tokens
		self.encoding = tiktoken.get_encoding("cl100k_base")
		self.normalizer = DiacriticNormalizer()

	def chunk(self, blocks: list) -> list[dict[str, Any]]:
		"""Create chunk records with a stable schema for ingestion."""
		if not blocks:
			return []

		cleaned = [b for b in blocks if str((b or {}).get("text", "")).strip()]
		if not cleaned:
			return []

		chunks: list[dict[str, Any]] = []
		i = 0
		while i < len(cleaned):
			current = cleaned[i]
			current_text = self._clean_text(current.get("text", ""))
			if not current_text:
				i += 1
				continue

			block_type = str(current.get("block_type", "paragraph") or "paragraph")
			if self._is_atomic_block(block_type):
				chunks.append(self._make_chunk(current_text, current))
				i += 1
				continue

			merge_count = 1
			merged_text = current_text

			if i + 1 < len(cleaned):
				nxt = cleaned[i + 1]
				next_text = self._clean_text(nxt.get("text", ""))
				if next_text and not self._is_atomic_block(str(nxt.get("block_type", "paragraph") or "paragraph")):
					if self._should_merge_consecutive(current, nxt, current_text, next_text):
						combined = f"{current_text}\n\n{next_text}".strip()
						if self._token_count(combined) <= self.max_tokens:
							merged_text = combined
							merge_count = 2

			parts = self._split_with_overlap(merged_text)
			for part in parts:
				chunks.append(self._make_chunk(part, current))

			i += merge_count

		return chunks

	@staticmethod
	def _clean_text(text: Any) -> str:
		return " ".join(str(text or "").split()).strip()

	@staticmethod
	def _is_atomic_block(block_type: str) -> bool:
		return block_type in {"table", "figure_caption"}

	def _token_count(self, text: str) -> int:
		return len(self.encoding.encode(text or ""))

	def _split_with_overlap(self, text: str) -> list[str]:
		token_ids = self.encoding.encode(text or "")
		if len(token_ids) <= self.max_tokens:
			return [text]

		step = max(1, self.max_tokens - self.overlap_tokens)
		parts: list[str] = []
		start = 0
		while start < len(token_ids):
			end = min(start + self.max_tokens, len(token_ids))
			part_ids = token_ids[start:end]
			part_text = self.encoding.decode(part_ids)
			if part_text.strip():
				parts.append(part_text)
			if end >= len(token_ids):
				break
			start += step

		return parts if parts else [text]

	def _should_merge_consecutive(
		self,
		current: dict[str, Any],
		nxt: dict[str, Any],
		current_text: str,
		next_text: str,
	) -> bool:
		curr_type = str(current.get("block_type", "paragraph") or "paragraph")
		next_type = str(nxt.get("block_type", "paragraph") or "paragraph")

		if curr_type == "heading" and next_type == "paragraph":
			return True

		# Sanskrit verse + commentary heuristic: numbered verse followed by plain paragraph.
		curr_shloka = self._extract_shloka_number(current_text)
		next_shloka = self._extract_shloka_number(next_text)
		if curr_type == "paragraph" and next_type == "paragraph" and curr_shloka and not next_shloka:
			return True

		return False

	def _make_chunk(self, original_text: str, source_block: dict[str, Any]) -> dict[str, Any]:
		normalized_text = self.normalizer.normalize(original_text)
		language = self._detect_language(original_text)
		shloka_number = self._extract_shloka_number(original_text)

		return {
			"original_text": original_text,
			"normalized_text": normalized_text,
			"page_number": source_block.get("page_number"),
			"source_file": source_block.get("source_file", ""),
			"block_type": str(source_block.get("block_type", "paragraph") or "paragraph"),
			"language": language,
			"heading_context": source_block.get("heading_context", ""),
			"shloka_number": shloka_number,
		}

	def _detect_language(self, text: str) -> str:
		script = self.normalizer.detect_script(text)
		if script == "devanagari":
			try:
				lang = detect(text)
				if lang == "hi":
					return "hindi"
				# langdetect is weak for Sanskrit; default Devanagari non-Hindi to Sanskrit.
				return "sanskrit"
			except Exception:
				return "devanagari"

		if script != "latin":
			return script

		try:
			lang = detect(text)
		except Exception:
			return "unknown"

		mapping = {
			"hi": "hindi",
			"en": "english",
			"sa": "sanskrit",
		}
		return mapping.get(lang, "unknown")

	@staticmethod
	def _extract_shloka_number(text: str) -> str | None:
		if not text:
			return None

		m = re.match(r"^\s*(\d+\.\d+)\b", text)
		if m:
			return m.group(1)

		m = re.match(r"^\s*([IVXLCDM]+\.)\s*", text, flags=re.IGNORECASE)
		if m:
			return m.group(1)

		return None
