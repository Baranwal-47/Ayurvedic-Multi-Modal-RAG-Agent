"""Inspect one or more PDF pages through the rebuilt ingestion pipeline.

This script does not upsert to Qdrant or upload to Cloudinary.
It writes stage-by-stage JSON so the final pre-upsert artifacts can be reviewed.

The command is python scripts/inspect_pipeline_page.py --pdf data/pdfs/93aa6-145-rasa-shastra.pdf --pages 19 --output-dir data/images/pipeline_inspect --with-embeddings
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import re
import time
import sys
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embeddings.image_embedder import ImageEmbedder
from embeddings.text_embedder import TextEmbedder
from ingestion.chunker import Chunker
from ingestion.docling_parser import DoclingPDFParser
from ingestion.image_extractor import ImageExtractor
from ingestion.image_text_linker import ImageTextLinker
from ingestion.native_pdf_parser import NativePDFParser
from ingestion.noise_detector import NoiseDetector
from ingestion.ocr_pipeline import OCRPipeline
from ingestion.page_classifier import PageClassifier
from ingestion.page_layout import PageLayout
from ingestion.page_model_builder import PageModelBuilder
from ingestion.qdrant_mapper import QdrantMapper
from ingestion.section_detector import SectionDetector
from ingestion.shloka_detector import ShlokaDetector


def _stable_doc_id(pdf_path: Path) -> str:
    raw = f"{pdf_path.name}|{pdf_path.stat().st_size}|{pdf_path.resolve()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect rebuilt ingestion pipeline on selected pages")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to PDF")
    parser.add_argument("--pages", type=int, nargs="+", required=True, help="1-based page numbers")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/images/pipeline_inspect"),
        help="Directory where JSON and extracted page images are written",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Also build Qdrant-ready points with embeddings",
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Keep full stage payloads (default writes compact JSON to reduce redundancy)",
    )
    return parser.parse_args()


def _flatten_page_blocks(page_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page_model in page_models:
        heading_context = ""
        for unit in sorted(page_model.get("text_units", []), key=lambda item: int(item.get("reading_order", 0))):
            if unit.get("kind") == "noise":
                continue
            if unit.get("kind") == "heading":
                heading_context = str(unit.get("text") or "").strip()
            rows.append(
                {
                    "text": unit.get("text") or "",
                    "block_type": unit.get("kind") or "paragraph",
                    "page_number": page_model["page_number"],
                    "source_file": page_model["source_file"],
                    "heading_context": heading_context,
                    "bbox": unit.get("bbox"),
                    "column_id": unit.get("column_id"),
                    "reading_order": unit.get("reading_order"),
                }
            )
    return rows


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _count_kinds(units: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        kind = str(unit.get("kind") or unit.get("block_type") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _log_stage(label: str, started_at: float) -> None:
    print(f"[TIMING] {label} took {time.perf_counter() - started_at:.2f}s")


def _compact_stage_outputs(page_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for stage in page_artifacts:
        native_parse = dict(stage.get("native_parse") or {})
        native_units = list(native_parse.get("text_units") or [])

        ocr_result = dict(stage.get("ocr_result") or {}) if stage.get("ocr_result") is not None else None
        if ocr_result is not None:
            ocr_units = list(ocr_result.get("text_units") or [])
            ocr_line_units = list(ocr_result.get("line_units") or [])
            ocr_effective_units = list(ocr_result.get("effective_units") or [])
            ocr_words = list(ocr_result.get("word_units") or [])
            effective_units = ocr_effective_units or ocr_line_units or ocr_units
            ocr_result = {
                "engine_used": ocr_result.get("engine_used"),
                "confidence": ocr_result.get("confidence"),
                "text": ocr_result.get("text"),
                "text_unit_count": len(ocr_units),
                "line_unit_count": len(ocr_line_units),
                "word_unit_count": len(ocr_words),
                "effective_unit_count": len(effective_units),
                "effective_unit_granularity": str(
                    ocr_result.get("effective_unit_granularity")
                    or ("line" if ocr_line_units else "paragraph")
                ),
            }

        repair_result = dict(stage.get("repair_result") or {}) if stage.get("repair_result") is not None else None
        if repair_result is not None:
            repair_result = {
                "used_ocr": repair_result.get("used_ocr"),
                "ocr_profile": repair_result.get("ocr_profile"),
                "repaired_unit_indexes": repair_result.get("repaired_unit_indexes"),
                "legacy_repaired_unit_indexes": repair_result.get("legacy_repaired_unit_indexes"),
            }

        pre_noise = dict(stage.get("page_model_pre_noise") or {})
        pre_noise_units = list(pre_noise.get("text_units") or [])

        compact.append(
            {
                "page_number": stage.get("page_number"),
                "router_decision": stage.get("router_decision"),
                "classification": stage.get("classification"),
                "native_parse": {
                    "raw_text": native_parse.get("raw_text"),
                    "text_unit_count": len(native_units),
                    "kind_counts": _count_kinds(native_units),
                },
                "docling_result": stage.get("docling_result"),
                "repair_result": repair_result,
                "ocr_result": ocr_result,
                "page_model_pre_noise_summary": {
                    "route": pre_noise.get("route"),
                    "layout_type": pre_noise.get("layout_type"),
                    "text_unit_count": len(pre_noise_units),
                    "kind_counts": _count_kinds(pre_noise_units),
                },
            }
        )

    return compact


def _compact_page_models(page_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = copy.deepcopy(page_models)
    for page_model in compact:
        page_section = list(page_model.get("section_path") or [])
        for unit in page_model.get("text_units", []):
            if list(unit.get("section_path") or []) == page_section:
                unit.pop("section_path", None)
    return compact


def _compact_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = copy.deepcopy(chunks)
    for chunk in compact:
        if chunk.get("text_for_embedding") == chunk.get("text"):
            chunk.pop("text_for_embedding", None)
        for key in [
            "ocr_source",
            "ocr_confidence",
            "shloka_id",
            "shloka_number",
            "table_rows",
            "table_caption",
            "table_markdown",
        ]:
            if chunk.get(key) is None:
                chunk.pop(key, None)
    return compact


def _chunk_fragmentation_stats(chunks: list[dict[str, Any]], page_models: list[dict[str, Any]]) -> tuple[float, str | None]:
    page_count = max(1, len(page_models))
    avg_chunks_per_page = float(len(chunks)) / float(page_count)
    if avg_chunks_per_page >= 40.0:
        return avg_chunks_per_page, "critical_fragmentation"
    if avg_chunks_per_page >= 30.0:
        return avg_chunks_per_page, "high_fragmentation"
    return avg_chunks_per_page, None


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", str(value or "")).strip().strip(" .")
    return cleaned or "untitled"


def main() -> int:
    args = _parse_args()
    pdf_path = args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    parser = NativePDFParser()
    docling = DoclingPDFParser()
    classifier = PageClassifier()
    builder = PageModelBuilder()
    layout = PageLayout()
    shloka = ShlokaDetector()
    section = SectionDetector()
    noise = NoiseDetector()
    linker = ImageTextLinker()
    chunker = Chunker()
    mapper = QdrantMapper()

    ocr: OCRPipeline | None = None
    docling_ready = False
    docling_targets: set[int] = set()
    decision_rows: list[dict[str, Any]] = []

    doc_id = _stable_doc_id(pdf_path)
    output_dir = args.output_dir / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    selected_pages = sorted({int(page) for page in args.pages})
    page_artifacts: list[dict[str, Any]] = []
    page_models: list[dict[str, Any]] = []
    scanned_pages: set[int] = set()

    with fitz.open(pdf_path) as doc:
        for page_number in selected_pages:
            if page_number < 1 or page_number > doc.page_count:
                raise ValueError(f"page {page_number} out of range 1..{doc.page_count}")

            page = doc.load_page(page_number - 1)
            native_started = time.perf_counter()
            native_page = parser.parse_page(page_number=page_number, page=page, source_file=pdf_path.name)
            _log_stage(f"native parse page {page_number}", native_started)

            classify_started = time.perf_counter()
            classification = classifier.classify_page(
                pdf_path=pdf_path,
                page_number=page_number,
                native_units=native_page.text_units,
                parser=parser,
                page=page,
            )
            _log_stage(f"classify page {page_number}", classify_started)

            native_text = "\n".join(
                str(unit.get("text") or "").strip() for unit in native_page.text_units if str(unit.get("text") or "").strip()
            )
            use_ocr, ocr_reason = classifier.should_use_ocr(native_text, parser)

            route = "native"
            route_reason = "native_clean"
            if not native_text.strip():
                route = "ocr"
                route_reason = "empty_native"
            elif use_ocr:
                route = "ocr"
                route_reason = ocr_reason
            elif classification.page_type != "digitized":
                route = "ocr"
                route_reason = classification.reason
            else:
                use_docling, docling_reason = classifier.should_use_docling(native_page.text_units)
                if use_docling:
                    route = "docling"
                    route_reason = docling_reason
                    docling_targets.add(page_number)

            if route == "ocr":
                scanned_pages.add(page_number)

            print(f"[ROUTER] page {page_number}: route={route} reason={route_reason}")
            decision_rows.append(
                {
                    "page_number": page_number,
                    "native_page": native_page,
                    "classification": classification,
                    "route": route,
                    "route_reason": route_reason,
                }
            )

        if docling.available and docling_targets:
            try:
                priming_started = time.perf_counter()
                docling.prime_document(pdf_path, page_numbers=sorted(docling_targets))
                docling_ready = True
                _log_stage(f"docling prime selective {pdf_path.name}", priming_started)
            except Exception as exc:
                docling.clear_document(pdf_path)
                docling_ready = False
                print(f"[DOCLING] selective prime failed for {pdf_path.name}; falling back to native parser: {exc}")

    for decision in decision_rows:
        page_number = int(decision["page_number"])
        native_page = decision["native_page"]
        classification = decision["classification"]
        route = str(decision["route"])
        route_reason = str(decision["route_reason"])

        ocr_result: dict[str, Any] | None = None
        docling_result: dict[str, Any] | None = None

        if route == "ocr":
            if ocr is None:
                ocr = OCRPipeline()
            ocr_started = time.perf_counter()
            ocr_result = ocr.process_page(
                pdf_path=pdf_path,
                page_number=page_number,
                route_reason=classification.page_type if classification.page_type in {"scanned", "ocr_fallback"} else "ocr_fallback",
                ocr_profile="default",
            )
            _log_stage(f"ocr page {page_number}", ocr_started)
            raw_line_units = list(ocr_result.get("line_units") or [])
            if raw_line_units:
                merged_line_units = OCRPipeline.merge_line_units(raw_line_units)
                ocr_units = merged_line_units or raw_line_units
                effective_granularity = "line_merged" if len(ocr_units) < len(raw_line_units) else "line"
            else:
                ocr_units = list(ocr_result.get("text_units") or [])
                effective_granularity = "paragraph"
            if not ocr_units and native_page.text_units:
                raise RuntimeError(f"OCR failed on page {page_number}")

            ocr_route = classification.page_type if classification.page_type in {"scanned", "ocr_fallback"} else "ocr_fallback"
            page_model = builder.build(
                doc_id=doc_id,
                source_file=pdf_path.name,
                page_number=page_number,
                route=ocr_route,
                ocr_units=ocr_units,
                ocr_confidence=ocr_result.get("confidence"),
            )
            quality = page_model.setdefault("quality", {})
            quality["structure_engine"] = "vision"
            quality["route_reason"] = route_reason
            quality["ocr_unit_granularity"] = effective_granularity
            quality["ocr_line_unit_count_raw"] = len(raw_line_units)
            quality["ocr_unit_count_effective"] = len(ocr_units)
            docling_result = {"used": False, "reason": "route_ocr"}
        else:
            selected_units = native_page.text_units
            structure_engine = "pymupdf"

            if route == "docling" and docling_ready:
                try:
                    docling_started = time.perf_counter()
                    docling_page = docling.parse_page(
                        pdf_path=pdf_path,
                        page_number=page_number,
                        source_file=pdf_path.name,
                    )
                    _log_stage(f"docling page {page_number}", docling_started)
                    if docling_page.text_units:
                        selected_units = docling_page.text_units
                        structure_engine = "docling"
                    docling_result = {
                        "used": structure_engine == "docling",
                        "table_count": int(docling_page.table_count),
                        "text_unit_count": len(docling_page.text_units),
                    }
                except Exception as exc:
                    docling_result = {"used": False, "error": str(exc)}
            elif route == "docling":
                docling_result = {"used": False, "error": "docling_not_primed"}
            else:
                docling_result = {"used": False, "reason": "native_selected"}

            page_model = builder.build(
                doc_id=doc_id,
                source_file=pdf_path.name,
                page_number=page_number,
                route="digitized",
                native_units=selected_units,
            )
            quality = page_model.setdefault("quality", {})
            quality["structure_engine"] = structure_engine
            quality["route_reason"] = route_reason

        page_model = layout.apply(page_model)
        page_model = shloka.apply(page_model)

        page_artifacts.append(
            {
                "page_number": page_number,
                "router_decision": {
                    "route": route,
                    "reason": route_reason,
                },
                "classification": {
                    "page_type": classification.page_type,
                    "reason": classification.reason,
                    "native_text_ok": classification.native_text_ok,
                    "image_heavy": classification.image_heavy,
                },
                "native_parse": {
                    "raw_text": native_page.raw_text,
                    "text_units": native_page.text_units,
                },
                "docling_result": docling_result,
                "repair_result": None,
                "ocr_result": {
                    "engine_used": (ocr_result or {}).get("engine_used"),
                    "confidence": (ocr_result or {}).get("confidence"),
                    "text": (ocr_result or {}).get("text"),
                    "text_units": (ocr_result or {}).get("text_units") or [],
                    "line_units": (ocr_result or {}).get("line_units") or [],
                    "effective_units": ocr_units,
                    "effective_unit_granularity": quality.get("ocr_unit_granularity"),
                    "word_units": (ocr_result or {}).get("word_units") or [],
                }
                if ocr_result is not None
                else None,
                "page_model_pre_noise": page_model,
            }
        )
        page_models.append(page_model)

        del native_page
        gc.collect()

    if docling.available:
        docling.clear_document(pdf_path)
    gc.collect()

    page_models = noise.mark_document_noise(page_models)
    section_path: list[str] = []
    for index, page_model in enumerate(page_models):
        page_models[index], section_path = section.apply(page_model, section_path)

    image_rows = ImageExtractor().extract(
        pdf_path=pdf_path,
        output_dir=image_output_dir,
        scanned_pages=scanned_pages,
        page_blocks=_flatten_page_blocks(page_models),
        allowed_pages=set(selected_pages),
    )

    for page_model in page_models:
        linker.apply(page_model, image_rows)

    chunks = chunker.chunk_document(page_models)
    avg_chunks_per_page, chunk_fragmentation_flag = _chunk_fragmentation_stats(chunks, page_models)

    for page_model in page_models:
        for image in page_model.get("images", []):
            image.setdefault("doc_id", doc_id)
            image.setdefault("source_file", pdf_path.name)
            image.setdefault("page_number", int(page_model.get("page_number") or 0))
            image.setdefault("languages", ["unknown"])
            image.setdefault("scripts", ["Zyyy"])
            image.setdefault("linked_chunk_ids", [])
            image.setdefault("cloudinary_public_id", None)
            image.setdefault("image_url", None)

    for page_model in page_models:
        for image in page_model.get("images", []):
            image["linked_chunk_ids"] = [
                chunk["chunk_id"]
                for chunk in chunks
                if image.get("image_id") in (chunk.get("image_ids") or [])
            ]

    text_points = []
    image_points = []
    if args.with_embeddings:
        text_embedder = TextEmbedder()
        image_embedder = ImageEmbedder(model=text_embedder.model)
        text_vectors = text_embedder.embed([chunk["text_for_embedding"] for chunk in chunks])
        image_texts = [
            image.get("caption") or image.get("surrounding_text") or image["image_id"]
            for page_model in page_models
            for image in page_model.get("images", [])
        ]
        image_vectors = image_embedder.embed(image_texts)
        all_images = [image for page_model in page_models for image in page_model.get("images", [])]
        text_points = mapper.map_text_points(chunks, text_vectors)
        image_points = mapper.map_image_points(all_images, image_vectors)

    output_stage = page_artifacts if args.full_output else _compact_stage_outputs(page_artifacts)
    output_models = page_models if args.full_output else _compact_page_models(page_models)
    output_chunks = chunks if args.full_output else _compact_chunks(chunks)

    final_payload = {
        "doc_id": doc_id,
        "source_file": pdf_path.name,
        "pages": selected_pages,
        "chunk_stats": {
            "chunk_count": len(chunks),
            "page_count": len(page_models),
            "avg_chunks_per_page": round(avg_chunks_per_page, 2),
            "fragmentation_flag": chunk_fragmentation_flag,
        },
        "page_stage_outputs": output_stage,
        "page_models_final": output_models,
        "image_rows": image_rows,
        "chunks": output_chunks,
        "qdrant_text_points": text_points,
        "qdrant_image_points": image_points,
    }

    output_json = output_dir / f"{_safe_path_component(pdf_path.stem)}_pages_{'_'.join(str(page) for page in selected_pages)}.json"
    output_json.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"output_json={output_json}")
    print(f"image_output_dir={image_output_dir}")
    print(f"pages={selected_pages}")
    print(f"chunks={len(chunks)}")
    print(f"avg_chunks_per_page={avg_chunks_per_page:.2f}")
    if chunk_fragmentation_flag is not None:
        print(
            f"[CHUNKING][WARN] fragmentation={chunk_fragmentation_flag} "
            f"(avg_chunks_per_page={avg_chunks_per_page:.2f}; expected < 30)"
        )
    print(f"images={len(image_rows)}")
    print("Inspect `page_stage_outputs`, `page_models_final`, `chunks`, and optional `qdrant_*_points` in the JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
