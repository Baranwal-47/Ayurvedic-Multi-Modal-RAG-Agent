"""Assign heading kinds and running section paths."""

from __future__ import annotations

import re
from typing import Any


class SectionDetector:
    """Promote heading-like text units and maintain a shallow section path."""

    _NUMBERED_HEADING = re.compile(r"^\s*(\d+(\.\d+)?|[IVXLCDM]+)\b", re.IGNORECASE)

    def apply(self, page_model: dict[str, Any], previous_path: list[str] | None = None) -> tuple[dict[str, Any], list[str]]:
        current_path = list(previous_path or [])
        for unit in sorted(page_model.get("text_units", []), key=lambda item: int(item.get("reading_order", 0))):
            if unit.get("kind") == "noise":
                continue
            text = " ".join(str(unit.get("text") or "").split()).strip()
            if self._is_heading(unit, text):
                unit["kind"] = "heading"
                unit["block_type"] = "heading"
                current_path = [text][:2]
            unit["section_path"] = list(current_path)

        page_model["section_path"] = list(current_path)
        return page_model, current_path

    def _is_heading(self, unit: dict[str, Any], text: str) -> bool:
        if unit.get("block_type") == "heading":
            return True
        if not text or len(text) > 120:
            return False
        if self._NUMBERED_HEADING.match(text):
            return True
        return len(text.split()) <= 10 and text == text.title()
