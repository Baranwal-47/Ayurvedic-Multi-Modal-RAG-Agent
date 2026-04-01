"""Semantic chunking for normalized PageModel documents."""

from __future__ import annotations

import re
from typing import Any

import tiktoken
from langdetect import detect

from normalization.diacritic_normalizer import DiacriticNormalizer


class Chunker:
    """Build Qdrant-ready text chunk payloads from normalized page models."""

    def __init__(self, target_words: int = 180, hard_cap_words: int = 300, overlap_words: int = 40) -> None:
        self.target_words = int(target_words)
        self.hard_cap_words = int(hard_cap_words)
        self.overlap_words = int(overlap_words)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.normalizer = DiacriticNormalizer()

    def chunk_document(self, page_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pages = sorted(page_models, key=lambda item: int(item.get("page_number", 0)))
        chunks: list[dict[str, Any]] = []
        chunk_index = 0

        for page_model in pages:
            page_chunks = self._chunk_page(page_model, start_index=chunk_index)
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        bridge_chunks: list[dict[str, Any]] = []
        for previous, current in zip(chunks, chunks[1:]):
            if self._should_bridge(previous, current):
                bridge_chunks.append(self._make_bridge_chunk(previous, current, chunk_index))
                chunk_index += 1

        chunks.extend(bridge_chunks)
        return sorted(chunks, key=lambda item: (item["page_start"], item["chunk_id"]))

    def _chunk_page(self, page_model: dict[str, Any], *, start_index: int) -> list[dict[str, Any]]:
        page_number = int(page_model.get("page_number", 0))
        units = [
            unit
            for unit in sorted(page_model.get("text_units", []), key=lambda item: int(item.get("reading_order", 0)))
            if unit.get("kind") not in {"noise", "label"}
        ]
        if not units:
            return []

        chunks: list[dict[str, Any]] = []
        buffer_units: list[dict[str, Any]] = []
        chunk_counter = start_index

        for unit in units:
            kind = str(unit.get("kind") or "paragraph")
            text = " ".join(str(unit.get("text") or "").split()).strip()
            if not text:
                continue

            if kind in {"caption", "shloka"}:
                if buffer_units:
                    chunks.extend(self._flush_buffer(page_model, buffer_units, chunk_counter))
                    chunk_counter = start_index + len(chunks)
                    buffer_units = []
                chunks.append(self._make_atomic_chunk(page_model, unit, chunk_counter))
                chunk_counter += 1
                continue

            if buffer_units:
                previous_kind = str(buffer_units[-1].get("kind") or "paragraph")
                boundary_crossed = (kind == "table_row" and previous_kind != "table_row") or (
                    kind != "table_row" and previous_kind == "table_row"
                )
                if boundary_crossed:
                    chunks.extend(self._flush_buffer(page_model, buffer_units, chunk_counter))
                    chunk_counter = start_index + len(chunks)
                    buffer_units = []

            candidate = [*buffer_units, unit]
            if self._word_count(" ".join(str(item.get("text") or "") for item in candidate)) > self.hard_cap_words:
                chunks.extend(self._flush_buffer(page_model, buffer_units, chunk_counter))
                chunk_counter = start_index + len(chunks)
                buffer_units = [unit]
            else:
                buffer_units.append(unit)
                if self._word_count(" ".join(str(item.get("text") or "") for item in buffer_units)) >= self.target_words:
                    chunks.extend(self._flush_buffer(page_model, buffer_units, chunk_counter))
                    chunk_counter = start_index + len(chunks)
                    buffer_units = []

        if buffer_units:
            chunks.extend(self._flush_buffer(page_model, buffer_units, chunk_counter))

        return chunks

    def _flush_buffer(self, page_model: dict[str, Any], units: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
        if not units:
            return []
        text = "\n\n".join(" ".join(str(unit.get("text") or "").split()) for unit in units).strip()
        parts = self._split_words_with_overlap(text)
        chunks: list[dict[str, Any]] = []
        for offset, part in enumerate(parts):
            chunks.append(self._make_chunk(page_model, units, part, start_index + offset))
        return chunks

    def _make_atomic_chunk(self, page_model: dict[str, Any], unit: dict[str, Any], chunk_index: int) -> dict[str, Any]:
        text = " ".join(str(unit.get("text") or "").split()).strip()
        chunk_type = "image_caption" if unit.get("kind") == "caption" else "shloka"
        return self._make_chunk(page_model, [unit], text, chunk_index, force_chunk_type=chunk_type)

    def _make_chunk(
        self,
        page_model: dict[str, Any],
        units: list[dict[str, Any]],
        text: str,
        chunk_index: int,
        *,
        force_chunk_type: str | None = None,
    ) -> dict[str, Any]:
        doc_id = str(page_model.get("doc_id") or "unknown")
        source_file = str(page_model.get("source_file") or "")
        page_number = int(page_model.get("page_number") or 0)
        section_path = list(units[-1].get("section_path") or page_model.get("section_path") or [])
        heading_text = section_path[-1] if section_path else None
        source_unit_ids = [str(unit.get("unit_id")) for unit in units if unit.get("unit_id")]
        languages = self._merge_unique(units, "languages")
        scripts = self._merge_unique(units, "scripts")
        image_ids = self._image_ids_for_units(page_model, source_unit_ids)
        chunk_type = force_chunk_type or self._infer_chunk_type(units, text)
        normalized_text = self.normalizer.normalize(text)
        chunk_id = f"{doc_id}:p{page_number}-{page_number}:{chunk_type}:{chunk_index}"
        table_id = f"{doc_id}:p{page_number}:table:{chunk_index}" if chunk_type == "table_text" else None
        table_rows = self._extract_table_rows(units) if chunk_type == "table_text" else []
        table_caption = self._extract_table_caption(table_rows) if table_rows else None
        table_markdown = self._table_rows_to_markdown(table_rows) if table_rows else None

        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_file": source_file,
            "page_start": page_number,
            "page_end": page_number,
            "page_numbers": [page_number],
            "chunk_type": chunk_type,
            "text": text,
            "text_for_embedding": text,
            "normalized_text": normalized_text,
            "section_path": section_path,
            "heading_text": heading_text,
            "languages": languages,
            "scripts": scripts,
            "layout_type": str(page_model.get("layout_type") or "single"),
            "route": str(page_model.get("route") or "digitized"),
            "ocr_source": self._resolve_ocr_source(page_model, units),
            "ocr_confidence": page_model.get("quality", {}).get("ocr_confidence"),
            "is_shloka": chunk_type == "shloka",
            "shloka_id": chunk_id if chunk_type == "shloka" else None,
            "is_multilingual": len(languages) > 1 or len(scripts) > 1,
            "has_image_context": bool(image_ids),
            "image_ids": image_ids,
            "source_unit_ids": source_unit_ids,
            "bridge_source_chunk_ids": [],
            "shloka_number": self._extract_shloka_number(text) if chunk_type == "shloka" else None,
            "table_id": table_id,
            "table_rows": table_rows if chunk_type == "table_text" else None,
            "table_caption": table_caption,
            "table_markdown": table_markdown,
        }

    @staticmethod
    def _resolve_ocr_source(page_model: dict[str, Any], units: list[dict[str, Any]]) -> str | None:
        route = str(page_model.get("route") or "")
        if route in {"ocr", "ocr_fallback", "scanned"}:
            return "vision"

        if bool(page_model.get("quality", {}).get("hybrid_repair_used_ocr")):
            return "vision"

        for unit in units:
            source_engine = str(unit.get("source_engine") or "").strip().lower()
            if source_engine in {"vision", "ocr"}:
                return "vision"

        return None

    def _make_bridge_chunk(self, previous: dict[str, Any], current: dict[str, Any], chunk_index: int) -> dict[str, Any]:
        merged_text = f"{previous['text']}\n\n{current['text']}".strip()
        truncated = " ".join(merged_text.split()[: self.hard_cap_words]).strip()
        doc_id = previous["doc_id"]
        chunk_id = f"{doc_id}:p{previous['page_start']}-{current['page_end']}:page_bridge:{chunk_index}"
        languages = sorted(set(previous.get("languages", [])) | set(current.get("languages", [])))
        scripts = sorted(set(previous.get("scripts", [])) | set(current.get("scripts", [])))
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_file": previous["source_file"],
            "page_start": previous["page_start"],
            "page_end": current["page_end"],
            "page_numbers": [previous["page_start"], current["page_end"]],
            "chunk_type": "page_bridge",
            "text": truncated,
            "text_for_embedding": truncated,
            "normalized_text": self.normalizer.normalize(truncated),
            "section_path": previous.get("section_path") or current.get("section_path") or [],
            "heading_text": previous.get("heading_text") or current.get("heading_text"),
            "languages": languages,
            "scripts": scripts,
            "layout_type": previous.get("layout_type") or current.get("layout_type"),
            "route": current.get("route") or previous.get("route"),
            "ocr_source": current.get("ocr_source") or previous.get("ocr_source"),
            "ocr_confidence": current.get("ocr_confidence") or previous.get("ocr_confidence"),
            "is_shloka": False,
            "shloka_id": None,
            "is_multilingual": len(languages) > 1 or len(scripts) > 1,
            "has_image_context": bool(previous.get("image_ids") or current.get("image_ids")),
            "image_ids": sorted(set(previous.get("image_ids", [])) | set(current.get("image_ids", []))),
            "source_unit_ids": list(previous.get("source_unit_ids", [])) + list(current.get("source_unit_ids", [])),
            "bridge_source_chunk_ids": [previous["chunk_id"], current["chunk_id"]],
            "shloka_number": None,
            "table_id": None,
        }

    def _image_ids_for_units(self, page_model: dict[str, Any], source_unit_ids: list[str]) -> list[str]:
        image_ids: list[str] = []
        for image in page_model.get("images", []):
            caption_unit_ids = set(image.get("caption_unit_ids", []))
            if caption_unit_ids.intersection(source_unit_ids):
                image_ids.append(str(image["image_id"]))
        return image_ids

    @staticmethod
    def _merge_unique(units: list[dict[str, Any]], key: str) -> list[str]:
        values: list[str] = []
        for unit in units:
            for value in unit.get(key, []) or []:
                if value not in values:
                    values.append(str(value))
        return values or ["unknown"]

    @classmethod
    def _infer_chunk_type(cls, units: list[dict[str, Any]], text: str) -> str:
        first_kind = str(units[0].get("kind") or "paragraph")
        if first_kind == "caption":
            return "image_caption"
        if first_kind == "shloka":
            return "shloka"
        if cls._looks_like_table_text(units, text):
            return "table_text"
        if first_kind == "heading":
            return "section_intro"
        return "paragraph"

    @classmethod
    def _looks_like_table_text(cls, units: list[dict[str, Any]], text: str) -> bool:
        lower = " ".join(str(text or "").split()).lower()
        if lower.startswith("table "):
            return True

        table_row_units = sum(1 for unit in units if str(unit.get("kind") or "") == "table_row")
        if table_row_units >= max(1, len(units) // 2):
            return True
        rowish_units = sum(1 for unit in units if cls._looks_like_table_row(str(unit.get("text") or "")))
        return rowish_units >= 2 and rowish_units >= max(2, len(units) // 2)

    @staticmethod
    def _looks_like_table_row(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False
        if compact.lower().startswith(("table ", "fig", "figure ")):
            return True
        if compact.lower().startswith(("s.no", "sample no")):
            return True
        digit_count = sum(ch.isdigit() for ch in compact)
        word_count = len(compact.split())
        numeric_tokens = len(re.findall(r"\d+(?:\.\d+)?", compact))
        has_serial_prefix = bool(re.match(r"^\d{1,3}[\.)]?\s+", compact))
        if has_serial_prefix and word_count <= 4:
            return True
        lower = compact.lower()
        if any(token in lower.split() for token in {"nil", "na", "n/a", "nd"}):
            return word_count <= 12
        if "%" in compact or "°" in compact:
            return word_count <= 12 and (numeric_tokens >= 1 or any(token.isalpha() for token in compact.split()))
        if numeric_tokens >= 2 and word_count <= 12:
            return True
        return digit_count >= 6 and word_count <= 12

    @staticmethod
    def _extract_table_rows(units: list[dict[str, Any]]) -> list[list[str]]:
        rows: list[list[str]] = []
        for unit in units:
            text = " ".join(str(unit.get("text") or "").split()).strip()
            if str(unit.get("kind") or "") != "table_row" and not Chunker._looks_like_table_row(text):
                continue

            raw_cells = unit.get("table_cells")
            if isinstance(raw_cells, list) and raw_cells:
                cells = [" ".join(str(cell or "").split()).strip() for cell in raw_cells]
                cells = [cell for cell in cells if cell]
            else:
                text = " ".join(str(unit.get("text") or "").split()).strip()
                if "|" in text:
                    cells = [part.strip() for part in text.split("|") if part.strip()]
                elif text:
                    cells = [text]
                else:
                    cells = []

            if not cells:
                continue

            if rows and rows[-1] == cells:
                continue
            rows.append(cells)

        return rows

    @staticmethod
    def _extract_table_caption(table_rows: list[list[str]]) -> str | None:
        if not table_rows:
            return None
        first = table_rows[0]
        if len(first) == 1 and first[0].lower().startswith("table"):
            return first[0]
        return None

    @staticmethod
    def _table_rows_to_markdown(table_rows: list[list[str]]) -> str | None:
        if not table_rows:
            return None

        caption = None
        start_index = 0
        if len(table_rows[0]) == 1 and table_rows[0][0].lower().startswith("table"):
            caption = table_rows[0][0]
            start_index = 1

        rows = table_rows[start_index:]
        if not rows:
            return caption

        max_cols = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]

        lines: list[str] = []
        if caption:
            lines.append(caption)

        if max_cols == 1:
            lines.extend(f"- {row[0]}" for row in normalized_rows)
            return "\n".join(lines)

        header = normalized_rows[0]
        body = normalized_rows[1:]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _split_words_with_overlap(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.hard_cap_words:
            return [text]

        step = max(1, self.hard_cap_words - self.overlap_words)
        parts: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.hard_cap_words, len(words))
            part = " ".join(words[start:end]).strip()
            if part:
                parts.append(part)
            if end >= len(words):
                break
            start += step
        return parts

    @staticmethod
    def _word_count(text: str) -> int:
        return len(str(text or "").split())

    def _should_bridge(self, previous: dict[str, Any], current: dict[str, Any]) -> bool:
        if previous["doc_id"] != current["doc_id"]:
            return False
        if current["page_start"] - previous["page_end"] != 1:
            return False
        if previous["chunk_type"] in {"image_caption", "table_text"} or current["chunk_type"] in {"image_caption", "table_text", "section_intro"}:
            return False
        if previous.get("heading_text") and current.get("heading_text") and previous["heading_text"] != current["heading_text"]:
            return False
        prev_text = str(previous.get("text") or "").strip()
        curr_text = str(current.get("text") or "").strip()
        if not prev_text or not curr_text:
            return False
        if prev_text.endswith((".", ":", "?", "!")):
            return False
        return True

    def _detect_language(self, text: str) -> str:
        script = self.normalizer.detect_script(text)
        if script == "devanagari":
            try:
                lang = detect(text)
                if lang == "hi":
                    return "hi"
                return "sa"
            except Exception:
                return "sa"
        if script == "telugu":
            return "te"
        if script == "arabic":
            return "ur"
        return "en"

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    @staticmethod
    def _extract_shloka_number(text: str) -> str | None:
        if not text:
            return None
        match = re.match(r"^\s*(\d+(\.\d+)?)\b", text)
        return match.group(1) if match else None
