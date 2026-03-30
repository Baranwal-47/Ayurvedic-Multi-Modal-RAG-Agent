"""Build normalized PageModel dictionaries from native or OCR units."""

from __future__ import annotations

from typing import Any

from normalization.diacritic_normalizer import DiacriticNormalizer


class PageModelBuilder:
    """Normalize page-level units into one shared PageModel structure."""

    def __init__(self) -> None:
        self.normalizer = DiacriticNormalizer()

    def build(
        self,
        *,
        doc_id: str,
        source_file: str,
        page_number: int,
        route: str,
        native_units: list[dict[str, Any]] | None = None,
        ocr_units: list[dict[str, Any]] | None = None,
        ocr_confidence: float | None = None,
        language_hints: list[str] | None = None,
    ) -> dict[str, Any]:
        base_units = list(ocr_units or native_units or [])
        text_units: list[dict[str, Any]] = []

        for unit in base_units:
            text = str(unit.get("text") or "").strip()
            if not text:
                continue
            bbox = unit.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            scripts = list(unit.get("scripts") or self._detect_scripts(text))
            languages = list(unit.get("languages") or self._detect_languages_from_scripts(scripts))
            text_units.append(
                {
                    "unit_id": unit.get("unit_id"),
                    "kind": str(unit.get("kind") or unit.get("block_type") or "paragraph"),
                    "block_type": str(unit.get("block_type") or unit.get("kind") or "paragraph"),
                    "text": text,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "column_id": unit.get("column_id"),
                    "reading_order": int(unit.get("reading_order", len(text_units))),
                    "confidence": unit.get("confidence"),
                    "languages": languages,
                    "scripts": scripts,
                    "source_engine": str(unit.get("source_engine") or ("vision" if route != "digitized" else "pymupdf")),
                    "section_path": list(unit.get("section_path") or []),
                }
            )

        return {
            "doc_id": doc_id,
            "source_file": source_file,
            "page_number": int(page_number),
            "route": "ocr" if route == "scanned" else route,
            "layout_type": "single",
            "language_hints": list(language_hints or []),
            "text_units": text_units,
            "images": [],
            "quality": {
                "native_text_ok": route == "digitized",
                "ocr_confidence": ocr_confidence,
                "garbled_text": route == "ocr_fallback",
                "image_heavy": route == "scanned",
            },
        }

    def _detect_scripts(self, text: str) -> list[str]:
        mapping = {
            "devanagari": ["Deva"],
            "telugu": ["Telu"],
            "arabic": ["Arab"],
            "latin": ["Latn"],
            "tamil": ["Taml"],
            "bengali": ["Beng"],
        }
        script = self.normalizer.detect_script(text)
        return mapping.get(script, ["Zyyy"])

    @staticmethod
    def _detect_languages_from_scripts(scripts: list[str]) -> list[str]:
        if "Deva" in scripts:
            return ["hi", "sa"]
        if "Telu" in scripts:
            return ["te"]
        if "Arab" in scripts:
            return ["ur"]
        if "Latn" in scripts:
            return ["en"]
        return ["unknown"]
