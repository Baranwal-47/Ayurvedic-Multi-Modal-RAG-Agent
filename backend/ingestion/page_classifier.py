"""Deterministic page classification for native vs OCR routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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

    def __init__(
        self,
        min_native_chars: int = 40,
        scanned_image_ratio: float = 0.75,
        min_meaningful_word_ratio: float = 0.45,
        max_symbol_ratio: float = 0.30,
        max_mojibake_ratio: float = 0.08,
        max_non_ascii_ratio: float = 0.55,
    ) -> None:
        self.min_native_chars = int(min_native_chars)
        self.scanned_image_ratio = float(scanned_image_ratio)
        self.min_meaningful_word_ratio = float(min_meaningful_word_ratio)
        self.max_symbol_ratio = float(max_symbol_ratio)
        self.max_mojibake_ratio = float(max_mojibake_ratio)
        self.max_non_ascii_ratio = float(max_non_ascii_ratio)

    def classify_page(
        self,
        *,
        pdf_path: str | Path,
        page_number: int,
        native_units: list[dict[str, Any]],
        parser: Any,
    ) -> PageClassification:
        native_text = "\n".join(str(unit.get("text") or "").strip() for unit in native_units if str(unit.get("text") or "").strip())
        image_ratio = self._largest_image_ratio_for_page(Path(pdf_path), page_number)
        image_heavy = image_ratio >= self.scanned_image_ratio
        use_ocr, ocr_reason = self.should_use_ocr(native_text, parser)
        native_text_ok = bool(native_text) and not use_ocr

        if len(native_text.strip()) < self.min_native_chars and image_heavy:
            return PageClassification(page_number, "scanned", "low_text_high_image", False, True)
        if native_text_ok:
            return PageClassification(page_number, "digitized", "native_good", True, image_heavy)
        if native_text.strip():
            return PageClassification(page_number, "ocr_fallback", ocr_reason, False, image_heavy)
        return PageClassification(page_number, "scanned", "empty_native", False, image_heavy)

    def should_use_ocr(self, native_text: str, parser: Any) -> tuple[bool, str]:
        text = str(native_text or "").strip()
        if not text:
            return True, "empty_native"

        if parser.is_text_garbled(text):
            return True, "parser_garbled"

        quality = self._analyze_text_quality(text)
        reasons: list[str] = []

        if quality["too_short"]:
            reasons.append("too_short")
        if quality["mojibake_ratio"] > self.max_mojibake_ratio:
            reasons.append("mojibake_like")
        if quality["non_ascii_ratio"] > self.max_non_ascii_ratio and quality["indic_ratio"] < 0.10:
            reasons.append("high_non_ascii_ratio")
        if quality["meaningful_ratio"] < self.min_meaningful_word_ratio:
            reasons.append("low_meaningful_words")
        if quality["symbol_ratio"] > self.max_symbol_ratio:
            reasons.append("high_symbol_ratio")
        if quality["symbol_regex_hit"] and quality["symbol_ratio"] > (self.max_symbol_ratio * 0.5):
            reasons.append("symbol_regex_hit")

        if reasons:
            return True, "quality:" + ",".join(reasons)
        return False, "native_good"

    def _analyze_text_quality(self, text: str) -> dict[str, float | bool]:
        compact = " ".join(str(text or "").split())
        total_chars = max(1, len(compact))
        words = re.findall(r"\S+", compact)
        total_words = max(1, len(words))

        weird_symbol_chars = sum(1 for ch in compact if self._is_weird_symbol(ch))
        non_ascii_chars = sum(1 for ch in compact if ord(ch) > 127)
        indic_chars = sum(1 for ch in compact if self._is_indic_char(ch))
        mojibake_markers = len(re.findall(r"(?:Ã.|Â.|ï.|à¤|à¥|à¦|à¨|àª|à³)", compact))
        meaningful_words = sum(1 for word in words if self._is_meaningful_token(word))
        symbol_regex_hit = bool(re.search(r"[^\w\s.,]", compact))

        return {
            "symbol_ratio": weird_symbol_chars / total_chars,
            "non_ascii_ratio": non_ascii_chars / total_chars,
            "indic_ratio": indic_chars / total_chars,
            "mojibake_ratio": mojibake_markers / total_chars,
            "meaningful_ratio": meaningful_words / total_words,
            "symbol_regex_hit": symbol_regex_hit,
            "too_short": len(compact) < self.min_native_chars,
        }

    @staticmethod
    def _is_weird_symbol(ch: str) -> bool:
        if ch.isalnum() or ch.isspace():
            return False
        if ch in ".,:;!?()[]{}'\"/-_%+&*#°₹$=|":
            return False
        return True

    @staticmethod
    def _is_meaningful_token(token: str) -> bool:
        normalized = re.sub(r"^[^\w]+|[^\w]+$", "", str(token or ""), flags=re.UNICODE)
        if len(normalized) < 2:
            return False
        if not any(ch.isalpha() for ch in normalized):
            return False
        lower = normalized.lower()
        if re.search(r"(?:Ã.|Â.|ï.)", normalized):
            return False
        if re.search(r"(?:à¤|à¥|à¦|à¨|àª|à³)", lower):
            return False
        return True

    @staticmethod
    def _is_indic_char(ch: str) -> bool:
        code = ord(ch)
        return (
            0x0900 <= code <= 0x097F or
            0x0980 <= code <= 0x09FF or
            0x0A00 <= code <= 0x0A7F or
            0x0A80 <= code <= 0x0AFF or
            0x0B00 <= code <= 0x0B7F or
            0x0B80 <= code <= 0x0BFF or
            0x0C00 <= code <= 0x0C7F or
            0x0C80 <= code <= 0x0CFF or
            0x0D00 <= code <= 0x0D7F
        )

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
