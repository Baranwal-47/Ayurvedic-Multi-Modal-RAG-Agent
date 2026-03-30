"""Detect verse-like text units without contaminating prose."""

from __future__ import annotations

import re
from typing import Any


class ShlokaDetector:
    """Mark likely shloka units using deliberately strict heuristics."""

    _NUMBERED_SHLOKA = re.compile(r"^\s*([0-9\u0966-\u096F\u0C66-\u0C6F]+([\.\)-][0-9\u0966-\u096F\u0C66-\u0C6F]+)?)\b")
    _DANDA_MARKS = ("।", "॥", "à¥¤", "à¥¥", "|")

    def apply(self, page_model: dict[str, Any]) -> dict[str, Any]:
        for unit in page_model.get("text_units", []):
            if unit.get("kind") in {"noise", "caption", "label", "heading"}:
                continue
            text = str(unit.get("text") or "").strip()
            if self._looks_like_shloka(text, unit):
                unit["kind"] = "shloka"
                unit["block_type"] = "shloka"
        return page_model

    def _looks_like_shloka(self, text: str, unit: dict[str, Any]) -> bool:
        if not text:
            return False
        if self._looks_like_table_or_reference(text):
            return False

        scripts = {str(s) for s in (unit.get("scripts") or [])}
        has_deva = "Deva" in scripts or self._has_devanagari_text(text)
        has_telu = "Telu" in scripts or self._has_telugu_text(text)
        if not (has_deva or has_telu):
            return False
        if scripts == {"Latn"}:
            return False

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        has_danda = any(mark in text for mark in self._DANDA_MARKS)
        numbered = bool(self._NUMBERED_SHLOKA.match(text))
        compact_multiline = (
            2 <= len(lines) <= 4 and
            all(1 <= len(line.split()) <= 14 for line in lines)
        )
        low_prose_endings = self._has_low_prose_endings(lines)

        signal_count = int(has_danda) + int(numbered) + int(compact_multiline) + int(low_prose_endings)
        return signal_count >= 2 and (compact_multiline or has_danda)

    @staticmethod
    def _has_devanagari_text(text: str) -> bool:
        return any("\u0900" <= ch <= "\u097F" for ch in text)

    @staticmethod
    def _has_telugu_text(text: str) -> bool:
        return any("\u0C00" <= ch <= "\u0C7F" for ch in text)

    @staticmethod
    def _has_low_prose_endings(lines: list[str]) -> bool:
        prose_endings = sum(1 for line in lines if line.endswith((".", "?", "!")))
        return prose_endings <= max(0, len(lines) // 3)

    @staticmethod
    def _looks_like_table_or_reference(text: str) -> bool:
        compact = " ".join(text.split())
        lower = compact.lower()
        if not compact:
            return True
        if lower.startswith(("table ", "table no", "fig", "figure ", "references", "reference", "chapter ")):
            return True
        if "www." in lower or "issn" in lower:
            return True
        if ShlokaDetector._looks_like_table_row(compact):
            return True
        if compact.count("%") >= 1 or compact.count("°") >= 1:
            return True
        if sum(ch.isdigit() for ch in compact) >= 5:
            return True
        if len(compact.split()) >= 10 and compact.endswith((".", "?", "!")):
            return True
        return False

    @staticmethod
    def _looks_like_table_row(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False
        word_count = len(compact.split())
        numeric_tokens = len(re.findall(r"\d+(?:\.\d+)?", compact))
        has_serial_prefix = bool(re.match(r"^\d{1,3}[\.)]?\s+", compact))
        if compact.lower().startswith(("s.no", "sample no", "parameter", "inference")):
            return True
        if has_serial_prefix and word_count <= 8:
            return True
        return numeric_tokens >= 3 and word_count <= 12
