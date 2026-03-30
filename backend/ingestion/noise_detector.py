"""Detect repeated header/footer noise across a document."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any


class NoiseDetector:
    """Mark repeated page furniture as noise before chunking."""

    _KEYWORDS = ("issn", "www.", "http", "volume", "issue", "journal")

    def mark_document_noise(self, page_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        strong_counter: Counter[str] = Counter()
        generic_counter: Counter[str] = Counter()

        for page_model in page_models:
            page_height = self._page_height(page_model)
            strong_seen_on_page: set[str] = set()
            generic_seen_on_page: set[str] = set()
            for unit in page_model.get("text_units", []):
                text = self._clean_text(unit.get("text"))
                bbox = unit.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                normalized = self._normalize_repeated_text(text)
                candidate_type = self._noise_candidate_type(
                    text=text,
                    normalized=normalized,
                    bbox=bbox,
                    page_height=page_height,
                )
                if candidate_type == "strong":
                    strong_seen_on_page.add(normalized)
                elif candidate_type == "generic":
                    generic_seen_on_page.add(normalized)
            strong_counter.update(strong_seen_on_page)
            generic_counter.update(generic_seen_on_page)

        repeated = {text for text, count in strong_counter.items() if count >= 2}
        repeated.update({text for text, count in generic_counter.items() if count >= 3})
        for page_model in page_models:
            page_height = self._page_height(page_model)
            for unit in page_model.get("text_units", []):
                text = self._clean_text(unit.get("text"))
                bbox = unit.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                normalized = self._normalize_repeated_text(text)
                if (
                    normalized in repeated
                    or self._is_page_number_only(text=text, bbox=bbox, page_height=page_height)
                ):
                    unit["kind"] = "noise"
                    unit["block_type"] = "noise"
        return page_models

    @staticmethod
    def _clean_text(value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _normalize_repeated_text(self, text: str) -> str:
        if not text:
            return ""
        normalized = text.lower()
        normalized = re.sub(r"^\s*\d{1,4}\s*", "", normalized)
        normalized = re.sub(r"\s*\d{1,4}\s*$", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip(" |.-")

    def _noise_candidate_type(self, *, text: str, normalized: str, bbox: list[float], page_height: float) -> str | None:
        if not normalized or len(text) > 140:
            return None

        y0 = float(bbox[1] or 0.0)
        y1 = float(bbox[3] or 0.0)
        top_zone = y1 <= page_height * 0.14
        bottom_zone = y0 >= page_height * 0.84
        if not (top_zone or bottom_zone):
            return None
        if any(keyword in normalized for keyword in self._KEYWORDS):
            return "strong"
        if self._is_page_number_only(text=text, bbox=bbox, page_height=page_height):
            return "strong"

        word_count = len(normalized.split())
        if top_zone and word_count <= 12 and (":" in text or re.search(r"\bet\s*al\b|etal", normalized)):
            return "generic"
        if bottom_zone and word_count <= 8 and len(normalized) <= 70:
            return "generic"
        return None

    @staticmethod
    def _is_page_number_only(*, text: str, bbox: list[float], page_height: float) -> bool:
        compact = text.strip()
        if not re.fullmatch(r"[\(\[]?\d{1,4}[\)\]]?", compact):
            return False
        y0 = float(bbox[1] or 0.0)
        y1 = float(bbox[3] or 0.0)
        return y1 <= page_height * 0.14 or y0 >= page_height * 0.84

    @staticmethod
    def _page_height(page_model: dict[str, Any]) -> float:
        max_y = 0.0
        for unit in page_model.get("text_units", []):
            bbox = unit.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            max_y = max(max_y, float(bbox[3] or 0.0))
        return max(max_y, 1.0)
