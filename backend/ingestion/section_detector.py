"""Assign heading kinds and running section paths."""

from __future__ import annotations

import re
from typing import Any


class SectionDetector:
    """Promote heading-like text units and maintain a shallow section path."""

    _NUMBERED_HEADING = re.compile(r"^\s*(\d+(\.\d+)?|[IVXLCDM]+)\b", re.IGNORECASE)
    _PAGE_FURNITURE_KEYWORDS = ("issn", "www.", "http", "volume", "issue", "journal")
    _NUMBERED_SECTION_PREFIXES = (
        "introduction",
        "materials",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "references",
        "abstract",
    )

    def apply(self, page_model: dict[str, Any], previous_path: list[str] | None = None) -> tuple[dict[str, Any], list[str]]:
        current_path = list(previous_path or [])
        table_heavy_page = self._is_table_heavy_page(page_model)
        for unit in sorted(page_model.get("text_units", []), key=lambda item: int(item.get("reading_order", 0))):
            if unit.get("kind") in {"noise", "table_row"}:
                unit["section_path"] = list(current_path)
                continue
            text = " ".join(str(unit.get("text") or "").split()).strip()
            if self._is_heading(unit, text, table_heavy=table_heavy_page):
                unit["kind"] = "heading"
                unit["block_type"] = "heading"
                current_path = [text][:2]
            elif str(unit.get("kind") or "") == "heading":
                # Parser may over-tag table cells as headings; demote unless section rules still accept.
                unit["kind"] = "paragraph"
                unit["block_type"] = "paragraph"
            unit["section_path"] = list(current_path)

        page_model["section_path"] = list(current_path)
        return page_model, current_path

    def _is_heading(self, unit: dict[str, Any], text: str, *, table_heavy: bool) -> bool:
        lower = text.lower()
        if not text or len(text) > 120:
            return False
        if any(keyword in lower for keyword in self._PAGE_FURNITURE_KEYWORDS):
            return False
        if str(unit.get("block_type") or "") == "table_row" or str(unit.get("kind") or "") == "table_row":
            return False
        if self._looks_like_table_row(text):
            return False
        if table_heavy and self._looks_like_table_cell_text(text):
            return False
        if re.fullmatch(r"[\(\[]?\d{1,4}[\)\]]?", text):
            return False
        if unit.get("block_type") == "heading":
            return True
        numbered = self._NUMBERED_HEADING.match(text)
        if numbered:
            remainder = re.sub(r"^\s*(\d+(\.\d+)?|[IVXLCDM]+)[\.)\-: ]*", "", lower).strip()
            if any(remainder.startswith(prefix) for prefix in self._NUMBERED_SECTION_PREFIXES):
                return True
            return text.endswith(":")
        return len(text.split()) >= 2 and len(text.split()) <= 10 and (text == text.title() or text.endswith(":"))

    @staticmethod
    def _is_table_heavy_page(page_model: dict[str, Any]) -> bool:
        table_row_count = sum(1 for unit in page_model.get("text_units", []) if str(unit.get("kind") or "") == "table_row")
        return table_row_count >= 5

    def _looks_like_table_cell_text(self, text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False
        lower = compact.lower()
        if any(lower.startswith(prefix) for prefix in self._NUMBERED_SECTION_PREFIXES):
            return False
        if compact.endswith(":"):
            return False
        return len(compact.split()) <= 6

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
