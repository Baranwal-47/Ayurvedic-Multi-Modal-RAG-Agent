"""Build normalized PageModel dictionaries from native or OCR units."""

from __future__ import annotations

import re
from typing import Any

from normalization.diacritic_normalizer import DiacriticNormalizer


class PageModelBuilder:
    """Normalize page-level units into one shared PageModel structure."""

    def __init__(self) -> None:
        self.normalizer = DiacriticNormalizer()
        self._table_row_labels = (
            "appearance",
            "colour",
            "color",
            "consistency",
            "temperature",
            "total loss",
            "loss",
            "odour",
            "odor",
            "taste",
            "hardness",
            "breakable",
            "soft",
            "weight",
            "ph",
        )

    def build(
        self,
        *,
        doc_id: str,
        source_file: str,
        page_number: int,
        route: str,
        native_units: list[dict[str, Any]] | None = None,
        ocr_units: list[dict[str, Any]] | None = None,
        ocr_confidence: float | None = None,
        language_hints: list[str] | None = None,
    ) -> dict[str, Any]:
        base_units = list(ocr_units or native_units or [])
        text_units: list[dict[str, Any]] = []

        for unit in base_units:
            text = str(unit.get("text") or "").strip()
            if not text:
                continue
            bbox = unit.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            scripts = list(unit.get("scripts") or self._detect_scripts(text))
            languages = list(unit.get("languages") or self._detect_languages_from_scripts(scripts))
            kind = str(unit.get("kind") or unit.get("block_type") or "paragraph")
            block_type = str(unit.get("block_type") or unit.get("kind") or "paragraph")

            if kind == "paragraph" and self._looks_like_table_row(text):
                kind = "table_row"
                block_type = "table_row"

            built_unit = {
                "unit_id": unit.get("unit_id"),
                "kind": kind,
                "block_type": block_type,
                "text": text,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "column_id": unit.get("column_id"),
                "reading_order": int(unit.get("reading_order", len(text_units))),
                "confidence": unit.get("confidence"),
                "languages": languages,
                "scripts": scripts,
                "source_engine": str(unit.get("source_engine") or ("vision" if route != "digitized" else "pymupdf")),
                "section_path": list(unit.get("section_path") or []),
            }
            for key in ("table_cells", "table_id", "table_row_index"):
                if unit.get(key) is not None:
                    built_unit[key] = unit.get(key)

            text_units.append(built_unit)

        text_units = self._promote_contextual_table_rows(text_units)

        return {
            "doc_id": doc_id,
            "source_file": source_file,
            "page_number": int(page_number),
            "route": "ocr" if route == "scanned" else route,
            "layout_type": "single",
            "language_hints": list(language_hints or []),
            "text_units": text_units,
            "images": [],
            "quality": {
                "native_text_ok": route == "digitized",
                "ocr_confidence": ocr_confidence,
                "garbled_text": route == "ocr_fallback",
                "image_heavy": route == "scanned",
            },
        }

    def _promote_contextual_table_rows(self, text_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        table_context_remaining = 0

        for unit in text_units:
            text = " ".join(str(unit.get("text") or "").split()).strip()
            if not text:
                continue

            if self._contains_table_anchor(text):
                table_context_remaining = max(table_context_remaining, 12)
                continue

            if table_context_remaining <= 0:
                continue

            kind = str(unit.get("kind") or "paragraph")
            if kind in {"paragraph", "table_row"} and self._looks_like_contextual_table_row(text):
                unit["kind"] = "table_row"
                unit["block_type"] = "table_row"

            if self._looks_like_context_break(text):
                table_context_remaining = 0
            else:
                table_context_remaining -= 1

        return text_units

    @staticmethod
    def _contains_table_anchor(text: str) -> bool:
        lower = text.lower()
        return bool(re.search(r"\btable\s*(no\.?\s*\d+|\d+)\b", lower))

    def _looks_like_contextual_table_row(self, text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False

        if self._looks_like_table_row(compact):
            return True

        lower = compact.lower()
        words = compact.split()
        word_count = len(words)
        numeric_tokens = len(re.findall(r"\d+(?:\.\d+)?", compact))

        if word_count > 18:
            return False
        if re.search(r"[.!?]\s*$", compact) and word_count >= 8:
            return False
        if any(lower.startswith(label) for label in self._table_row_labels):
            return True
        if numeric_tokens >= 1 and word_count <= 18:
            return True

        prose_markers = re.findall(r"\b(the|and|with|were|was|is|are|this|that|from|into|over|under)\b", lower)
        if word_count <= 6 and not prose_markers:
            return True
        return False

    @staticmethod
    def _looks_like_context_break(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False
        lower = compact.lower()
        words = compact.split()
        if len(words) < 16:
            return False
        if not re.search(r"[.!?;]", compact):
            return False
        return bool(re.search(r"\b(method|observations|prepared|adopting|objective|analysis|discussion|conclusion)\b", lower))

    def _detect_scripts(self, text: str) -> list[str]:
        mapping = {
            "devanagari": ["Deva"],
            "telugu": ["Telu"],
            "arabic": ["Arab"],
            "latin": ["Latn"],
            "tamil": ["Taml"],
            "bengali": ["Beng"],
        }
        script = self.normalizer.detect_script(text)
        return mapping.get(script, ["Zyyy"])

    @staticmethod
    def _detect_languages_from_scripts(scripts: list[str]) -> list[str]:
        if "Deva" in scripts:
            return ["hi", "sa"]
        if "Telu" in scripts:
            return ["te"]
        if "Arab" in scripts:
            return ["ur"]
        if "Latn" in scripts:
            return ["en"]
        return ["unknown"]

    @staticmethod
    def _looks_like_table_row(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False

        lower = compact.lower()
        word_count = len(compact.split())
        numeric_tokens = len(re.findall(r"\d+(?:\.\d+)?", compact))
        has_serial_prefix = bool(re.match(r"^\d{1,3}[\.)]?\s+", compact))

        if lower.startswith(("table ", "table no", "s.no", "sample no")):
            return True
        if has_serial_prefix and word_count <= 4:
            return True
        if has_serial_prefix and numeric_tokens >= 2 and word_count <= 12:
            return True
        if numeric_tokens >= 3 and word_count <= 12:
            return True
        return False
