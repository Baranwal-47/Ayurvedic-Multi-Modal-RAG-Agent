"""Detect verse-like text units without contaminating prose."""

from __future__ import annotations

import re
from typing import Any


class ShlokaDetector:
    """Mark likely shloka units using conservative heuristics."""

    _NUMBERED_SHLOKA = re.compile(r"^\s*(\d+(\.\d+)?)\b")

    def apply(self, page_model: dict[str, Any]) -> dict[str, Any]:
        for unit in page_model.get("text_units", []):
            if unit.get("kind") in {"noise", "caption", "label"}:
                continue
            text = str(unit.get("text") or "").strip()
            if self._looks_like_shloka(text, unit):
                unit["kind"] = "shloka"
                unit["block_type"] = "shloka"
        return page_model

    def _looks_like_shloka(self, text: str, unit: dict[str, Any]) -> bool:
        if not text:
            return False
        if "।" in text or "॥" in text:
            return True
        scripts = set(unit.get("scripts") or [])
        if "Deva" in scripts and self._NUMBERED_SHLOKA.match(text):
            return True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return len(lines) >= 2 and all(len(line.split()) <= 12 for line in lines)
