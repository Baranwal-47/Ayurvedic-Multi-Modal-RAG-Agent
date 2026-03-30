"""PyMuPDF-first parser for digitized PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

    _MOJIBAKE_HINTS = ("Ã", "Â", "�", "\ufffd")

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
        lines = [" ".join(line.split()) for line in str(text or "").splitlines()]
        return "\n".join(line for line in lines if line).strip()

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

        if line_count <= 2 and word_count <= 16 and (centered or numbered or (title_like and width_ratio < 0.75)):
            return "heading"
        return "paragraph"

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
