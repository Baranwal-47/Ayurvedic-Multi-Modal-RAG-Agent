"""Docling-based parser for digitized PDF pages.

This parser is intentionally used only for native-good pages.
It converts Docling items into the same text-unit shape used by PageModelBuilder.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
import time
from collections.abc import Iterable
from typing import Any

import fitz

from normalization.diacritic_normalizer import DiacriticNormalizer


@dataclass(frozen=True)
class DoclingPageParse:
    page_number: int
    text_units: list[dict[str, Any]]
    raw_text: str
    table_count: int


def plan_page_windows(page_numbers: Iterable[int], batch_size: int) -> list[tuple[int, int]]:
    """Group requested page numbers into contiguous windows with a maximum size."""
    numbers = sorted({int(page) for page in page_numbers if int(page) > 0})
    if not numbers:
        return []

    batch_size = max(1, int(batch_size))
    windows: list[tuple[int, int]] = []

    run_start = numbers[0]
    run_end = numbers[0]
    for page_number in numbers[1:]:
        if page_number == run_end + 1:
            run_end = page_number
            continue

        windows.extend(_split_window(run_start, run_end, batch_size))
        run_start = run_end = page_number

    windows.extend(_split_window(run_start, run_end, batch_size))
    return windows


def _split_window(start_page: int, end_page: int, batch_size: int) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    page = int(start_page)
    while page <= int(end_page):
        window_end = min(int(end_page), page + batch_size - 1)
        windows.append((page, window_end))
        page = window_end + 1
    return windows


class DoclingPDFParser:
    """Extract page-level text units from Docling, including structured table rows."""

    def __init__(self) -> None:
        self.normalizer = DiacriticNormalizer()
        self._converter: Any | None = None
        self._page_cache: dict[str, dict[int, DoclingPageParse]] = {}
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

    def prime_document(self, pdf_path: str | Path, page_numbers: Iterable[int] | None = None) -> dict[int, DoclingPageParse]:
        if not self.available:
            raise RuntimeError("Docling is not available in the current environment")

        path = Path(pdf_path)
        key = str(path.resolve())
        requested_pages = self._normalize_requested_pages(path, page_numbers)
        cached_pages = self._page_cache.setdefault(key, {})
        missing_pages = [page_number for page_number in requested_pages if page_number not in cached_pages]
        if not missing_pages:
            return cached_pages

        total_pages = self._page_count(path)
        batch_size = self._resolve_batch_size(total_pages)
        windows = plan_page_windows(missing_pages, batch_size)
        print(
            f"[DOCLING] Priming {path.name}: requested={len(requested_pages)}, missing={len(missing_pages)}, "
            f"windows={len(windows)}, batch_size={batch_size}"
        )

        with tempfile.TemporaryDirectory(prefix="docling_batch_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            for start_page, end_page in windows:
                batch_pdf = tmp_root / f"batch_{start_page:04d}_{end_page:04d}.pdf"
                with fitz.open(path) as src_doc:
                    with fitz.open() as dst_doc:
                        dst_doc.insert_pdf(src_doc, from_page=start_page - 1, to_page=end_page - 1)
                        dst_doc.save(batch_pdf)

                batch_started = time.perf_counter()
                try:
                    batch_pages = self._parse_document_pages(batch_pdf, source_file=path.name)
                except Exception as exc:
                    print(
                        f"[DOCLING] Batch {start_page}-{end_page} failed with Docling ({exc}); using PyMuPDF fallback"
                    )
                    batch_pages = self._parse_with_pymupdf_pages(
                        batch_pdf,
                        source_file=path.name,
                    )
                finally:
                    gc.collect()

                for local_page, parsed_page in batch_pages.items():
                    global_page = start_page + int(local_page) - 1
                    if global_page < start_page or global_page > end_page:
                        continue
                    remapped_units: list[dict[str, Any]] = []
                    for unit in parsed_page.text_units:
                        remapped_unit = dict(unit)
                        remapped_unit["page_number"] = global_page
                        unit_id = str(remapped_unit.get("unit_id") or "")
                        if unit_id:
                            remapped_unit["unit_id"] = re.sub(
                                r":p\d+:",
                                f":p{global_page}:",
                                unit_id,
                                count=1,
                            )
                        remapped_units.append(remapped_unit)
                    cached_pages[global_page] = DoclingPageParse(
                        page_number=global_page,
                        text_units=remapped_units,
                        raw_text=parsed_page.raw_text,
                        table_count=parsed_page.table_count,
                    )

                for global_page in range(start_page, end_page + 1):
                    cached_pages.setdefault(
                        global_page,
                        DoclingPageParse(page_number=global_page, text_units=[], raw_text="", table_count=0),
                    )

                print(
                    f"[DOCLING] Batch {start_page}-{end_page} primed in {time.perf_counter() - batch_started:.2f}s"
                )

        return cached_pages

    def clear_document(self, pdf_path: str | Path) -> None:
        key = str(Path(pdf_path).resolve())
        self._page_cache.pop(key, None)

    def clear_cache(self) -> None:
        self._page_cache.clear()

    def parse_page(self, *, pdf_path: str | Path, page_number: int, source_file: str) -> DoclingPageParse:
        if not self.available:
            raise RuntimeError("Docling is not available in the current environment")

        path = Path(pdf_path)
        cache = self.prime_document(path, [page_number])
        parsed = cache.get(int(page_number))
        if parsed is not None:
            return parsed
        return DoclingPageParse(page_number=int(page_number), text_units=[], raw_text="", table_count=0)

    def _get_document(self, pdf_path: Path) -> Any:
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()

        result = self._converter.convert(str(pdf_path))
        return result.document

    def _normalize_requested_pages(self, pdf_path: Path, page_numbers: Iterable[int] | None) -> list[int]:
        total_pages = self._page_count(pdf_path)
        if page_numbers is None:
            return list(range(1, total_pages + 1))

        requested = sorted({int(page) for page in page_numbers if int(page) > 0})
        for page_number in requested:
            if page_number > total_pages:
                raise ValueError(f"page {page_number} out of range 1..{total_pages}")
        return requested

    @staticmethod
    def _page_count(pdf_path: Path) -> int:
        with fitz.open(pdf_path) as doc:
            return int(doc.page_count)

    @staticmethod
    def _resolve_batch_size(total_pages: int) -> int:
        explicit = str(os.getenv("DOCLING_PAGE_BATCH_SIZE", "")).strip()
        if explicit:
            try:
                return max(1, int(explicit))
            except Exception:
                print(f"[DOCLING] Invalid DOCLING_PAGE_BATCH_SIZE='{explicit}', using auto mode")

        # Keep the default window conservative because Docling OCR rendering can spike memory on large PDFs.
        auto_min_pages = max(1, int(os.getenv("DOCLING_AUTO_BATCH_MIN_PAGES", "12")))
        auto_batch_size = max(1, int(os.getenv("DOCLING_AUTO_BATCH_SIZE", "8")))
        if total_pages < auto_min_pages:
            return max(1, total_pages)
        return min(max(1, total_pages), auto_batch_size)

    def _parse_document_pages(self, pdf_path: Path, source_file: str | None = None) -> dict[int, DoclingPageParse]:
        converter = self._get_document(pdf_path)
        doc = converter
        file_name = source_file or pdf_path.name

        page_state: dict[int, dict[str, Any]] = {}
        for item, _level in doc.iterate_items():
            page_number = self._extract_page_number(item)
            if page_number < 1:
                continue

            state = page_state.setdefault(
                page_number,
                {
                    "text_units": [],
                    "raw_parts": [],
                    "table_count": 0,
                    "seen_table_captions": set(),
                },
            )

            page_height = self._page_height(doc, page_number)
            prov = self._get_page_prov(item, page_number)
            bbox = self._bbox_from_prov(prov, page_height=page_height)
            label = str(getattr(item, "label", "") or "").lower().strip()

            if label == "table":
                table_rows = self._table_rows_from_item(item, doc)
                if not table_rows:
                    continue

                state["table_count"] = int(state["table_count"]) + 1
                table_id = f"{Path(file_name).stem}:p{page_number}:table:{state['table_count']}"
                for row_index, row in enumerate(table_rows):
                    cleaned = self._clean_text(str(row.get("text") or ""))
                    if not cleaned:
                        continue

                    if bool(row.get("is_caption")):
                        state["seen_table_captions"].add(self._normalize_table_caption(cleaned))

                    cells = [self._clean_text(cell) for cell in list(row.get("cells") or [])]
                    cells = [cell for cell in cells if cell]
                    state["raw_parts"].append(cleaned)
                    state["text_units"].append(
                        self._make_unit(
                            stem=Path(file_name).stem,
                            page_number=page_number,
                            index=len(state["text_units"]) + 1,
                            text=cleaned,
                            kind="table_row",
                            bbox=bbox,
                            source_engine="docling",
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
                if caption_key in state["seen_table_captions"]:
                    continue

            kind = self._map_label_to_kind(label=label, text=cleaned)
            state["raw_parts"].append(cleaned)
            state["text_units"].append(
                self._make_unit(
                    stem=Path(file_name).stem,
                    page_number=page_number,
                    index=len(state["text_units"]) + 1,
                    text=cleaned,
                    kind=kind,
                    bbox=bbox,
                    source_engine="docling",
                )
            )

        parsed_pages: dict[int, DoclingPageParse] = {}
        for page_number, state in page_state.items():
            parsed_pages[page_number] = DoclingPageParse(
                page_number=page_number,
                text_units=list(state["text_units"]),
                raw_text="\n".join(str(part) for part in state["raw_parts"] if str(part).strip()).strip(),
                table_count=int(state["table_count"]),
            )

        return parsed_pages

    def _parse_with_pymupdf_pages(self, pdf_path: Path, source_file: str | None = None) -> dict[int, DoclingPageParse]:
        parsed_pages: dict[int, DoclingPageParse] = {}
        file_name = source_file or pdf_path.name

        with fitz.open(pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                text_units: list[dict[str, Any]] = []
                raw_parts: list[str] = []

                for index, block in enumerate(page.get_text("blocks", sort=True), start=1):
                    x0, y0, x1, y1, text, *_rest = block
                    cleaned = self._clean_text(text)
                    if not cleaned:
                        continue

                    block_type = "heading" if self._looks_like_heading(cleaned) else "paragraph"
                    raw_parts.append(cleaned)
                    text_units.append(
                        self._make_unit(
                            stem=Path(file_name).stem,
                            page_number=page_index,
                            index=len(text_units) + 1,
                            text=cleaned,
                            kind=block_type,
                            bbox=[float(x0), float(y0), float(x1), float(y1)],
                            source_engine="pymupdf",
                        )
                    )

                parsed_pages[page_index] = DoclingPageParse(
                    page_number=page_index,
                    text_units=text_units,
                    raw_text="\n".join(raw_parts).strip(),
                    table_count=0,
                )

        return parsed_pages

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
    def _extract_page_number(item: Any) -> int:
        prov_list = list(getattr(item, "prov", []) or [])
        if prov_list:
            page_no = getattr(prov_list[0], "page_no", None)
            if isinstance(page_no, int) and page_no > 0:
                return page_no
        return 1

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
        source_engine: str = "docling",
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
            "source_engine": source_engine,
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

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        compact = " ".join(str(text or "").split()).strip()
        if not compact:
            return False

        if len(compact) <= 80 and compact.isupper():
            return True

        if len(compact) <= 120 and compact.endswith(":"):
            return True

        words = compact.split()
        if len(words) <= 10 and all(word[:1].isupper() for word in words if word and word[0].isalpha()):
            return True

        return False