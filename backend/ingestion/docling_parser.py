"""Docling-based parser for digitized PDF pages.

This parser is intentionally used only for native-good pages.
It converts Docling items into the same text-unit shape used by PageModelBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from normalization.diacritic_normalizer import DiacriticNormalizer


@dataclass(frozen=True)
class DoclingPageParse:
    page_number: int
    text_units: list[dict[str, Any]]
    raw_text: str
    table_count: int


class DoclingPDFParser:
    """Extract page-level text units from Docling, including structured table rows."""

    def __init__(self) -> None:
        self.normalizer = DiacriticNormalizer()
        self._converter: Any | None = None
        self._doc_cache: dict[str, Any] = {}
        self._available = self._detect_availability()

    @property
    def available(self) -> bool:
        return bool(self._available)

    @staticmethod
    def _detect_availability() -> bool:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401

            return True
        except Exception:
            return False

    def parse_page(self, *, pdf_path: str | Path, page_number: int, source_file: str) -> DoclingPageParse:
        if not self.available:
            raise RuntimeError("Docling is not available in the current environment")

        path = Path(pdf_path)
        doc = self._get_document(path)
        stem = Path(source_file).stem
        page_height = self._page_height(doc, page_number)

        text_units: list[dict[str, Any]] = []
        raw_parts: list[str] = []
        table_count = 0
        seen_table_captions: set[str] = set()

        for item, _level in doc.iterate_items():
            prov = self._get_page_prov(item, page_number)
            if prov is None:
                continue

            bbox = self._bbox_from_prov(prov, page_height=page_height)
            label = str(getattr(item, "label", "") or "").lower().strip()

            if label == "table":
                table_rows = self._table_rows_from_item(item, doc)
                if not table_rows:
                    continue

                table_count += 1
                table_id = f"{stem}:p{page_number}:table:{table_count}"
                for row_index, row in enumerate(table_rows):
                    cleaned = self._clean_text(str(row.get("text") or ""))
                    if not cleaned:
                        continue

                    if bool(row.get("is_caption")):
                        seen_table_captions.add(self._normalize_table_caption(cleaned))

                    cells = [self._clean_text(cell) for cell in list(row.get("cells") or [])]
                    cells = [cell for cell in cells if cell]
                    raw_parts.append(cleaned)
                    text_units.append(
                        self._make_unit(
                            stem=stem,
                            page_number=page_number,
                            index=len(text_units) + 1,
                            text=cleaned,
                            kind="table_row",
                            bbox=bbox,
                            extra_fields={
                                "table_id": table_id,
                                "table_row_index": int(row_index),
                                "table_cells": cells,
                            },
                        )
                    )
                continue

            text = self._extract_item_text(item, doc)
            cleaned = self._clean_text(text)
            if not cleaned:
                continue

            if label == "caption" and cleaned.lower().startswith("table"):
                caption_key = self._normalize_table_caption(cleaned)
                if caption_key in seen_table_captions:
                    continue

            kind = self._map_label_to_kind(label=label, text=cleaned)
            raw_parts.append(cleaned)
            text_units.append(
                self._make_unit(
                    stem=stem,
                    page_number=page_number,
                    index=len(text_units) + 1,
                    text=cleaned,
                    kind=kind,
                    bbox=bbox,
                )
            )

        return DoclingPageParse(
            page_number=page_number,
            text_units=text_units,
            raw_text="\n".join(raw_parts).strip(),
            table_count=table_count,
        )

    def _get_document(self, pdf_path: Path) -> Any:
        key = str(pdf_path.resolve())
        cached = self._doc_cache.get(key)
        if cached is not None:
            return cached

        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()

        result = self._converter.convert(str(pdf_path))
        document = result.document
        self._doc_cache[key] = document
        return document

    @staticmethod
    def _get_page_prov(item: Any, page_number: int) -> Any | None:
        prov_list = list(getattr(item, "prov", []) or [])
        if not prov_list:
            return None
        for prov in prov_list:
            if int(getattr(prov, "page_no", 0) or 0) == int(page_number):
                return prov
        return None

    @staticmethod
    def _bbox_from_prov(prov: Any, *, page_height: float) -> list[float]:
        bbox = getattr(prov, "bbox", None)
        if bbox is None:
            return [0.0, 0.0, 0.0, 0.0]

        left = float(getattr(bbox, "l", 0.0) or 0.0)
        right = float(getattr(bbox, "r", 0.0) or 0.0)
        top = float(getattr(bbox, "t", 0.0) or 0.0)
        bottom = float(getattr(bbox, "b", 0.0) or 0.0)

        x0, x1 = sorted((left, right))
        if page_height > 0:
            # Docling provenance uses a bottom-left origin; convert to top-left origin.
            y0 = float(page_height) - max(top, bottom)
            y1 = float(page_height) - min(top, bottom)
        else:
            y0, y1 = sorted((top, bottom))
        return [x0, y0, x1, y1]

    @staticmethod
    def _page_height(doc: Any, page_number: int) -> float:
        pages = getattr(doc, "pages", None)
        if pages is None:
            return 0.0

        page = None
        if isinstance(pages, dict):
            page = pages.get(int(page_number))
        elif isinstance(pages, list):
            index = int(page_number) - 1
            if 0 <= index < len(pages):
                page = pages[index]
        else:
            try:
                page = pages[int(page_number)]
            except Exception:
                page = None

        if page is None:
            return 0.0

        size = getattr(page, "size", None)
        if size is None:
            return 0.0
        return float(getattr(size, "height", 0.0) or 0.0)

    @staticmethod
    def _extract_item_text(item: Any, doc: Any) -> str:
        text = str(getattr(item, "text", "") or "").strip()
        if text:
            return text

        export_text = getattr(item, "export_to_text", None)
        if callable(export_text):
            try:
                return str(export_text(doc=doc) or "").strip()
            except Exception:
                return ""
        return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return ""
        cleaned = re.sub(r"\s*\|\s*", " | ", cleaned)
        return " ".join(cleaned.split()).strip()

    @staticmethod
    def _map_label_to_kind(*, label: str, text: str) -> str:
        if label in {"section_header", "title"}:
            return "heading"
        if label == "caption":
            lower = text.lower()
            if lower.startswith("table"):
                return "table_row"
            if lower.startswith(("figure", "fig", "image", "photo", "plate", "diagram", "chart")):
                return "caption"
            return "paragraph"
        return "paragraph"

    @staticmethod
    def _normalize_table_caption(text: str) -> str:
        lower = " ".join(str(text or "").split()).strip().lower()
        lower = re.sub(r"\s+", " ", lower)
        return lower.strip(" .;,-")

    def _table_rows_from_item(self, table_item: Any, doc: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        caption_fn = getattr(table_item, "caption_text", None)
        if callable(caption_fn):
            try:
                caption_text = str(caption_fn(doc=doc) or "").strip()
            except Exception:
                caption_text = ""
            if caption_text:
                cleaned_caption = self._clean_text(caption_text)
                if cleaned_caption:
                    rows.append(
                        {
                            "text": cleaned_caption,
                            "cells": [cleaned_caption],
                            "is_caption": True,
                        }
                    )

        df_export = getattr(table_item, "export_to_dataframe", None)
        if callable(df_export):
            try:
                dataframe = df_export(doc=doc)
            except Exception:
                dataframe = None
            if dataframe is not None and not getattr(dataframe, "empty", True):
                headers = [self._clean_cell(col) for col in list(getattr(dataframe, "columns", []))]
                if any(cell for cell in headers):
                    header_cells = [cell for cell in headers if cell]
                    rows.append(
                        {
                            "text": " | ".join(header_cells),
                            "cells": header_cells,
                            "is_caption": False,
                        }
                    )
                for values in dataframe.itertuples(index=False):
                    cells = [self._clean_cell(value) for value in values]
                    cleaned_cells = [cell for cell in cells if cell]
                    if cleaned_cells:
                        rows.append(
                            {
                                "text": " | ".join(cleaned_cells),
                                "cells": cleaned_cells,
                                "is_caption": False,
                            }
                        )

        if rows:
            return rows

        markdown_export = getattr(table_item, "export_to_markdown", None)
        if callable(markdown_export):
            try:
                markdown = str(markdown_export(doc=doc) or "")
            except Exception:
                markdown = ""
            for line in markdown.splitlines():
                stripped = line.strip()
                if not stripped.startswith("|"):
                    continue
                if re.fullmatch(r"\|?\s*[-:| ]+\s*\|?", stripped):
                    continue
                cells = [self._clean_text(part) for part in stripped.strip("|").split("|")]
                cleaned_cells = [cell for cell in cells if cell]
                if cleaned_cells:
                    rows.append(
                        {
                            "text": " | ".join(cleaned_cells),
                            "cells": cleaned_cells,
                            "is_caption": False,
                        }
                    )

        return rows

    @staticmethod
    def _clean_cell(value: Any) -> str:
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            return ""
        return " ".join(text.split())

    def _make_unit(
        self,
        *,
        stem: str,
        page_number: int,
        index: int,
        text: str,
        kind: str,
        bbox: list[float],
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scripts = self._detect_scripts(text)
        languages = self._detect_languages_from_scripts(scripts)
        unit = {
            "unit_id": f"{stem}:p{page_number}:docling:{index}",
            "text": text,
            "block_type": kind,
            "kind": kind,
            "page_number": int(page_number),
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "reading_order": int(index - 1),
            "column_id": None,
            "languages": languages,
            "scripts": scripts,
            "confidence": None,
            "source_engine": "docling",
        }
        if extra_fields:
            unit.update(extra_fields)
        return unit

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