"""Deterministic page classification for native vs OCR routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
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
        page: fitz.Page | None = None,
    ) -> PageClassification:
        native_text = "\n".join(str(unit.get("text") or "").strip() for unit in native_units if str(unit.get("text") or "").strip())
        if page is not None:
            image_ratio = self._largest_image_ratio_for_page_from_page(page)
        else:
            image_ratio = self._largest_image_ratio_for_page(Path(pdf_path), page_number)
        image_heavy = image_ratio >= self.scanned_image_ratio
        use_ocr, ocr_reason = self.should_use_ocr(native_text, parser)
        native_text_ok = bool(native_text) and not use_ocr

        if len(native_text.strip()) < self.min_native_chars and image_heavy:
            return PageClassification(page_number, "scanned", "low_text_high_image", False, True)
        if native_text_ok:
            return PageClassification(page_number, "digitized", "native_good", True, image_heavy)
        if native_text.strip() and not image_heavy and self._looks_like_index_page(native_text):
            return PageClassification(page_number, "digitized", "index_like_native", True, image_heavy)
        if native_text.strip():
            return PageClassification(page_number, "ocr_fallback", ocr_reason, False, image_heavy)
        return PageClassification(page_number, "scanned", "empty_native", False, image_heavy)

    def should_use_ocr(self, native_text: str, parser: Any) -> tuple[bool, str]:
        text = str(native_text or "").strip()
        if not text:
            return True, "empty_native"

        if self._looks_like_index_page(text):
            return False, "index_like_native"

        if parser.is_text_garbled(text):
            return True, "parser_garbled"

        quality = self._analyze_text_quality(text)
        reasons: list[str] = []

        if quality["too_short"]:
            reasons.append("too_short")
        if quality["mojibake_ratio"] > self.max_mojibake_ratio:
            reasons.append("mojibake_like")
        if quality["legacy_mixed_ratio"] > 0.10:
            reasons.append("legacy_mixed_tokens")
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

    def should_use_docling(self, native_units: list[dict[str, Any]]) -> tuple[bool, str]:
        """Return True when native page content is likely table-heavy or structurally complex."""
        rows = [" ".join(str((unit or {}).get("text") or "").split()).strip() for unit in native_units]
        rows = [row for row in rows if row]
        if not rows:
            return False, "no_native_text"

        table_anchor_hits = sum(1 for row in rows if self._has_table_anchor(row))
        row_pattern_hits = sum(1 for row in rows if self._looks_like_table_row_text(row))
        numeric_dense_hits = sum(1 for row in rows if self._is_numeric_dense_row(row))
        two_columnish = self._is_two_columnish(native_units)

        if table_anchor_hits >= 1 and row_pattern_hits >= 1:
            return True, "table_anchor"
        if row_pattern_hits >= 3:
            return True, "table_rows"
        if two_columnish and (row_pattern_hits >= 1 or numeric_dense_hits >= 2):
            return True, "complex_layout"
        return False, "simple_layout"

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
        legacy_mixed_tokens = sum(1 for word in words if self._looks_like_legacy_mixed_token(word))
        symbol_regex_hit = any(self._is_weird_symbol(ch) for ch in compact)

        return {
            "symbol_ratio": weird_symbol_chars / total_chars,
            "non_ascii_ratio": non_ascii_chars / total_chars,
            "indic_ratio": indic_chars / total_chars,
            "mojibake_ratio": mojibake_markers / total_chars,
            "legacy_mixed_ratio": legacy_mixed_tokens / total_words,
            "meaningful_ratio": meaningful_words / total_words,
            "symbol_regex_hit": symbol_regex_hit,
            "too_short": len(compact) < self.min_native_chars,
        }

    def _looks_like_legacy_mixed_token(self, token: str) -> bool:
        raw = str(token or "").strip()
        if not raw:
            return False

        has_ascii = any("a" <= ch.lower() <= "z" for ch in raw)
        has_non_ascii = any(ord(ch) > 127 for ch in raw)
        has_indic = any(self._is_indic_char(ch) for ch in raw)

        if not (has_ascii and has_non_ascii):
            return False
        if has_indic:
            return False
        return True

    @staticmethod
    def _has_table_anchor(text: str) -> bool:
        lower = str(text or "").lower()
        return bool(
            re.search(r"\btable\b", lower)
            or re.search(r"\bs\.?\s*no\b", lower)
            or "sample no" in lower
            or "parameter" in lower
        )

    @staticmethod
    def _looks_like_table_row_text(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False
        if re.match(r"^\d{1,3}[\.)]?\s+", compact):
            return True
        if "|" in compact:
            return True
        numeric_tokens = len(re.findall(r"\d+(?:\.\d+)?", compact))
        word_count = len(compact.split())
        return numeric_tokens >= 2 and word_count <= 12

    @staticmethod
    def _is_numeric_dense_row(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False
        letters = sum(1 for ch in compact if ch.isalpha())
        digits = sum(1 for ch in compact if ch.isdigit())
        if letters + digits == 0:
            return False
        ratio = digits / (letters + digits)
        return ratio >= 0.35 and len(compact.split()) <= 14

    @staticmethod
    def _is_two_columnish(native_units: list[dict[str, Any]]) -> bool:
        centers: list[float] = []
        for unit in native_units:
            bbox = (unit or {}).get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x0 = float(bbox[0])
                x1 = float(bbox[2])
            except Exception:
                continue
            centers.append((x0 + x1) / 2.0)

        if len(centers) < 6:
            return False

        ordered = sorted(centers)
        midpoint = ordered[len(ordered) // 2]
        left = sum(1 for x in centers if x <= midpoint)
        right = len(centers) - left
        spread = max(centers) - min(centers)

        if spread < 120.0:
            return False
        min_side = max(2, int(len(centers) * 0.25))
        return left >= min_side and right >= min_side

    @staticmethod
    def _is_weird_symbol(ch: str) -> bool:
        if ch.isalnum() or ch.isspace():
            return False
        if ch in ".,:;!?()[]{}'\"/-_%+&*#°₹$=|।॥":
            return False
        if unicodedata.category(ch).startswith("M"):
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

    @staticmethod
    def _largest_image_ratio_for_page_from_page(page: fitz.Page) -> float:
        try:
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

    @staticmethod
    def _looks_like_index_page(native_text: str) -> bool:
        lines = [line.strip() for line in str(native_text or "").splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        return PageClassifier._has_index_entry_pattern(lines)

    @staticmethod
    def _has_index_entry_pattern(lines: list[str]) -> bool:
        entry_lines = 0
        page_range_lines = 0

        for line in lines:
            compact = " ".join(line.split())
            if not compact:
                continue

            if re.search(r"\b\d{1,4}\s*-\s*\d{1,4}\b", compact):
                page_range_lines += 1
            if re.match(r"^\d{1,3}\b", compact) and len(compact.split()) <= 8:
                entry_lines += 1

        return entry_lines >= 2 and page_range_lines >= 2
