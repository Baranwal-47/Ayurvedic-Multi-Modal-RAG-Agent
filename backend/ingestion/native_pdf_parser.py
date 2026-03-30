"""PyMuPDF-first parser for digitized PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import fitz

from normalization.diacritic_normalizer import DiacriticNormalizer


@dataclass(frozen=True)
class NativePageParse:
    page_number: int
    text_units: list[dict[str, object]]
    raw_text: str


class NativePDFParser:
    """Parse native PDF text into simple block records for chunking and OCR routing."""

    _MOJIBAKE_HINTS = ("Ãƒ", "Ã‚", "ï¿½", "\ufffd")
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

    def __init__(self) -> None:
        self.normalizer = DiacriticNormalizer()

    def parse(self, pdf_path: str | Path) -> list[dict[str, object]]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        blocks: list[dict[str, object]] = []
        with fitz.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                page_parse = self.parse_page(page_number=page_number, page=page, source_file=path.name)
                blocks.extend(page_parse.text_units)

        return blocks

    def parse_page(self, *, page_number: int, page: fitz.Page, source_file: str) -> NativePageParse:
        heading_context = ""
        text_units: list[dict[str, object]] = []
        raw_parts: list[str] = []

        for index, item in enumerate(page.get_text("blocks", sort=True), start=1):
            x0, y0, x1, y1, text, *_rest = item
            cleaned = self._clean_text(text)
            if not cleaned:
                continue

            rect = fitz.Rect(x0, y0, x1, y1)
            block_type = self._classify_block(text=cleaned, bbox=rect, page_rect=page.rect)
            if block_type == "heading":
                heading_context = cleaned

            raw_parts.append(cleaned)
            text_units.append(
                {
                    "unit_id": f"{Path(source_file).stem}:p{page_number}:native:{index}",
                    "text": cleaned,
                    "block_type": block_type,
                    "kind": block_type,
                    "page_number": page_number,
                    "source_file": source_file,
                    "heading_context": heading_context,
                    "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                    "reading_order": len(text_units),
                    "column_id": None,
                    "languages": self._detect_languages(cleaned),
                    "scripts": self._detect_scripts(cleaned),
                    "confidence": None,
                    "source_engine": "pymupdf",
                }
            )

        return NativePageParse(
            page_number=page_number,
            text_units=text_units,
            raw_text="\n".join(raw_parts).strip(),
        )

    def is_page_scanned(self, page_blocks: list[dict[str, object]]) -> bool:
        if not page_blocks:
            return True

        text = " ".join(str(block.get("text") or "") for block in page_blocks).strip()
        text_len = len(text)
        alnum_count = sum(ch.isalnum() for ch in text)
        return text_len < 40 or alnum_count < 20

    def is_page_garbled(self, page_blocks: list[dict[str, object]]) -> bool:
        if not page_blocks:
            return False

        substantial = 0
        garbled = 0
        for block in page_blocks:
            text = str(block.get("text") or "").strip()
            if len(text) < 40:
                continue
            substantial += 1
            if self.is_text_garbled(text):
                garbled += 1

        if substantial == 0:
            return False
        return garbled >= 1 and (garbled / substantial) >= 0.30

    def is_page_non_latin(self, page_blocks: list[dict[str, object]]) -> bool:
        return any(self.is_text_non_latin(str(block.get("text") or "")) for block in page_blocks)

    def is_text_garbled(self, text: str) -> bool:
        cleaned = str(text or "").strip()
        if len(cleaned) < 5:
            return False

        total = len(cleaned)
        if total == 0:
            return False

        weird = sum(1 for ch in cleaned if not ch.isalnum() and ch not in " \n|.,;:!?-()[]{}'\"/")
        devanagari = sum(1 for ch in cleaned if "\u0900" <= ch <= "\u097F")
        arabic = sum(1 for ch in cleaned if "\u0600" <= ch <= "\u06FF")
        telugu = sum(1 for ch in cleaned if "\u0C00" <= ch <= "\u0C7F")

        if weird / total > 0.30:
            return True
        if devanagari == 0 and arabic == 0 and telugu == 0 and any(hint in cleaned for hint in self._MOJIBAKE_HINTS):
            return True
        return False

    def is_text_non_latin(self, text: str) -> bool:
        joined = " ".join(str(text or "").split())
        if len(joined) < 5:
            return False

        letter_count = 0
        non_latin_count = 0
        for ch in joined:
            if not ch.isalpha():
                continue
            letter_count += 1
            if self._is_non_latin_script_char(ch):
                non_latin_count += 1

        if letter_count == 0:
            return False
        return (non_latin_count / letter_count) >= 0.20

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned_lines: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = " ".join(raw_line.split()).strip()
            if not line:
                continue
            lower = line.lower()

            # Remove recurring journal footer lines that are frequently merged into body blocks.
            if re.search(r"\bwww\.ijaar\.in\b", lower) and "volume" in lower and "issue" in lower:
                continue
            if re.match(r"^\d{1,4}\s+www\.ijaar\.in\b", lower):
                continue

            # Drop boilerplate metadata lines that should not become retrieval chunks.
            if "conflict of interest" in lower or "source of support" in lower or "financial support" in lower:
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _detect_languages(self, text: str) -> list[str]:
        script = self.normalizer.detect_script(text)
        mapping = {
            "devanagari": ["hi", "sa"],
            "telugu": ["te"],
            "arabic": ["ur"],
            "latin": ["en"],
            "tamil": ["ta"],
            "bengali": ["bn"],
        }
        return mapping.get(script, ["unknown"])

    def _detect_scripts(self, text: str) -> list[str]:
        script = self.normalizer.detect_script(text)
        mapping = {
            "devanagari": ["Deva"],
            "telugu": ["Telu"],
            "arabic": ["Arab"],
            "latin": ["Latn"],
            "tamil": ["Taml"],
            "bengali": ["Beng"],
        }
        return mapping.get(script, ["Zyyy"])

    @staticmethod
    def _classify_block(*, text: str, bbox: fitz.Rect, page_rect: fitz.Rect) -> str:
        single_line = " ".join(text.splitlines()).strip()
        line_count = max(1, len([line for line in text.splitlines() if line.strip()]))
        word_count = len(single_line.split())
        width_ratio = float(bbox.width) / max(float(page_rect.width), 1.0)
        centered = abs(((bbox.x0 + bbox.x1) / 2.0) - (page_rect.width / 2.0)) <= (page_rect.width * 0.12)
        numbered = single_line[:6].strip().replace(".", "").isdigit()
        title_like = single_line == unicodedata.normalize("NFKC", single_line).title()
        uppercase_like = bool(single_line) and single_line.upper() == single_line
        ends_with_colon = single_line.endswith(":")
        normalized = re.sub(r"\s+", " ", single_line).strip().lower()
        top_zone = float(bbox.y1) <= float(page_rect.height) * 0.14
        bottom_zone = float(bbox.y0) >= float(page_rect.height) * 0.86

        if NativePDFParser._looks_like_page_furniture(
            text=single_line,
            normalized=normalized,
            bbox=bbox,
            page_rect=page_rect,
            top_zone=top_zone,
            bottom_zone=bottom_zone,
        ):
            return "noise"

        if NativePDFParser._looks_like_table_row(single_line):
            return "table_row"

        if line_count <= 2 and word_count <= 16 and (
            (uppercase_like and width_ratio < 0.90) or
            (numbered and word_count >= 2) or
            (ends_with_colon and word_count <= 12 and width_ratio < 0.90) or
            (centered and width_ratio < 0.65 and word_count >= 2 and (uppercase_like or ends_with_colon)) or
            (title_like and 2 <= word_count <= 10 and width_ratio < 0.60)
        ):
            return "heading"
        return "paragraph"

    @staticmethod
    def _looks_like_page_furniture(
        *,
        text: str,
        normalized: str,
        bbox: fitz.Rect,
        page_rect: fitz.Rect,
        top_zone: bool,
        bottom_zone: bool,
    ) -> bool:
        if not normalized:
            return False

        in_margin_zone = top_zone or bottom_zone
        corner_zone = float(bbox.x0) <= float(page_rect.width) * 0.15 or float(bbox.x1) >= float(page_rect.width) * 0.85

        if in_margin_zone and re.fullmatch(r"[\(\[]?\d{1,4}[\)\]]?", normalized):
            return True
        if not in_margin_zone:
            return False
        if any(keyword in normalized for keyword in NativePDFParser._PAGE_FURNITURE_KEYWORDS):
            return True
        if top_zone and ":" in text and re.search(r"\bet\s*al\b|etal", normalized):
            return True
        if corner_zone and re.match(r"^\d{1,4}\s+[a-z]", normalized):
            return True
        if len(text) <= 6 and text.strip(". ").isdigit():
            return True
        return False

    @staticmethod
    def _looks_like_table_row(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return False

        lower = compact.lower()
        words = compact.split()
        word_count = len(words)
        numeric_tokens = len(re.findall(r"\d+(?:\.\d+)?", compact))
        has_serial_prefix = bool(re.match(r"^\d{1,3}[\.)]?\s+", compact))
        after_serial = re.sub(r"^\d{1,3}[\.)]?\s+", "", lower).strip()

        if lower.startswith(("table ", "table no", "s.no", "sample no")):
            return True

        if has_serial_prefix and any(after_serial.startswith(prefix) for prefix in NativePDFParser._NUMBERED_SECTION_PREFIXES):
            return False

        if has_serial_prefix and word_count <= 4:
            return True
        if has_serial_prefix and numeric_tokens >= 2 and word_count <= 12:
            return True
        if numeric_tokens >= 3 and word_count <= 12:
            return True

        return False

    @staticmethod
    def _is_non_latin_script_char(ch: str) -> bool:
        code = ord(ch)
        if 0x0900 <= code <= 0x0DFF:
            return True
        if 0x1CD0 <= code <= 0x1CFF:
            return True
        if 0xA8E0 <= code <= 0xA8FF:
            return True
        if 0x0600 <= code <= 0x06FF:
            return True
        if 0x0750 <= code <= 0x077F:
            return True
        if 0x08A0 <= code <= 0x08FF:
            return True
        return False
