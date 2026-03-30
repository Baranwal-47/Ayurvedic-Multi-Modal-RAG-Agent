"""Detect repeated header/footer noise across a document."""

from __future__ import annotations

from collections import Counter
from typing import Any


class NoiseDetector:
    """Mark repeated short lines as noise before chunking."""

    def mark_document_noise(self, page_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counter: Counter[str] = Counter()
        for page_model in page_models:
            seen_on_page: set[str] = set()
            for unit in page_model.get("text_units", []):
                text = " ".join(str(unit.get("text") or "").split()).strip()
                if 1 <= len(text) <= 80:
                    seen_on_page.add(text)
            counter.update(seen_on_page)

        repeated = {text for text, count in counter.items() if count >= 3}
        for page_model in page_models:
            for unit in page_model.get("text_units", []):
                text = " ".join(str(unit.get("text") or "").split()).strip()
                if text in repeated:
                    unit["kind"] = "noise"
                    unit["block_type"] = "noise"
        return page_models
