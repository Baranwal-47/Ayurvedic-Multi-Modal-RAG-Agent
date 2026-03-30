"""Deterministic page classification for native vs OCR routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


@dataclass(frozen=True)
class PageClassification:
    page_number: int
    page_type: str
    reason: str
    native_text_ok: bool
    image_heavy: bool


class PageClassifier:
    """Classify pages as digitized, scanned, or OCR fallback."""

    def __init__(self, min_native_chars: int = 40, scanned_image_ratio: float = 0.75) -> None:
        self.min_native_chars = int(min_native_chars)
        self.scanned_image_ratio = float(scanned_image_ratio)

    def classify_page(
        self,
        *,
        pdf_path: str | Path,
        page_number: int,
        native_units: list[dict[str, Any]],
        parser: Any,
    ) -> PageClassification:
        native_text = "\n".join(str(unit.get("text") or "").strip() for unit in native_units if str(unit.get("text") or "").strip())
        native_text_ok = bool(native_text) and not parser.is_text_garbled(native_text)
        image_ratio = self._largest_image_ratio_for_page(Path(pdf_path), page_number)
        image_heavy = image_ratio >= self.scanned_image_ratio

        if len(native_text.strip()) < self.min_native_chars and image_heavy:
            return PageClassification(page_number, "scanned", "low_text_high_image", False, True)
        if native_text_ok:
            return PageClassification(page_number, "digitized", "native_good", True, image_heavy)
        if native_text.strip():
            return PageClassification(page_number, "ocr_fallback", "garbled_native", False, image_heavy)
        return PageClassification(page_number, "scanned", "empty_native", False, image_heavy)

    @staticmethod
    def _largest_image_ratio_for_page(pdf_path: Path, page_number: int) -> float:
        try:
            with fitz.open(pdf_path) as doc:
                if page_number < 1 or page_number > doc.page_count:
                    return 0.0
                page = doc.load_page(page_number - 1)
                page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
                largest = 0.0
                for info in page.get_image_info(xrefs=True):
                    bbox = info.get("bbox")
                    if not bbox:
                        continue
                    rect = fitz.Rect(bbox)
                    area = max(0.0, float(rect.width)) * max(0.0, float(rect.height))
                    largest = max(largest, area / page_area)
                return float(largest)
        except Exception:
            return 0.0
