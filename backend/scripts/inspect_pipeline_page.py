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


def _compact_stage_outputs(page_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for stage in page_artifacts:
        native_parse = dict(stage.get("native_parse") or {})
        native_units = list(native_parse.get("text_units") or [])

        ocr_result = dict(stage.get("ocr_result") or {}) if stage.get("ocr_result") is not None else None
        if ocr_result is not None:
            ocr_units = list(ocr_result.get("text_units") or [])
            ocr_words = list(ocr_result.get("word_units") or [])
            ocr_result = {
                "engine_used": ocr_result.get("engine_used"),
                "confidence": ocr_result.get("confidence"),
                "text": ocr_result.get("text"),
                "text_unit_count": len(ocr_units),
                "word_unit_count": len(ocr_words),
            }

        pre_noise = dict(stage.get("page_model_pre_noise") or {})
        pre_noise_units = list(pre_noise.get("text_units") or [])

        compact.append(
            {
                "page_number": stage.get("page_number"),
                "classification": stage.get("classification"),
                "native_parse": {
                    "raw_text": native_parse.get("raw_text"),
                    "text_unit_count": len(native_units),
                    "kind_counts": _count_kinds(native_units),
                },
                "docling_result": stage.get("docling_result"),
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


def main() -> int:
    args = _parse_args()
    pdf_path = args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = args.output_dir / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    parser = NativePDFParser()
    docling = DoclingPDFParser()
    classifier = PageClassifier()
    ocr = OCRPipeline()
    builder = PageModelBuilder()
    layout = PageLayout()
    shloka = ShlokaDetector()
    section = SectionDetector()
    noise = NoiseDetector()
    linker = ImageTextLinker()
    chunker = Chunker()
    mapper = QdrantMapper()

    doc_id = _stable_doc_id(pdf_path)
    selected_pages = sorted({int(page) for page in args.pages})
    page_artifacts: list[dict[str, Any]] = []
    page_models: list[dict[str, Any]] = []
    scanned_pages: set[int] = set()

    with fitz.open(pdf_path) as doc:
        for page_number in selected_pages:
            if page_number < 1 or page_number > doc.page_count:
                raise ValueError(f"page {page_number} out of range 1..{doc.page_count}")

            page = doc.load_page(page_number - 1)
            native_page = parser.parse_page(page_number=page_number, page=page, source_file=pdf_path.name)
            classification = classifier.classify_page(
                pdf_path=pdf_path,
                page_number=page_number,
                native_units=native_page.text_units,
                parser=parser,
            )

            ocr_result: dict[str, Any] | None = None
            docling_result: dict[str, Any] | None = None
            if classification.page_type == "digitized":
                selected_units = native_page.text_units
                structure_engine = "pymupdf"

                if docling.available:
                    try:
                        docling_page = docling.parse_page(
                            pdf_path=pdf_path,
                            page_number=page_number,
                            source_file=pdf_path.name,
                        )
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
                else:
                    docling_result = {"used": False, "error": "docling_unavailable"}

                page_model = builder.build(
                    doc_id=doc_id,
                    source_file=pdf_path.name,
                    page_number=page_number,
                    route="digitized",
                    native_units=selected_units,
                )
                page_model.setdefault("quality", {})["structure_engine"] = structure_engine
            else:
                scanned_pages.add(page_number)
                ocr_result = ocr.process_page(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    route_reason=classification.page_type,
                )
                page_model = builder.build(
                    doc_id=doc_id,
                    source_file=pdf_path.name,
                    page_number=page_number,
                    route=classification.page_type,
                    ocr_units=ocr_result.get("text_units") or [],
                    ocr_confidence=ocr_result.get("confidence"),
                )

            page_model = layout.apply(page_model)
            page_model = shloka.apply(page_model)

            page_artifacts.append(
                {
                    "page_number": page_number,
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
                    "ocr_result": {
                        "engine_used": (ocr_result or {}).get("engine_used"),
                        "confidence": (ocr_result or {}).get("confidence"),
                        "text": (ocr_result or {}).get("text"),
                        "text_units": (ocr_result or {}).get("text_units") or [],
                        "word_units": (ocr_result or {}).get("word_units") or [],
                    } if ocr_result is not None else None,
                    "page_model_pre_noise": page_model,
                }
            )
            page_models.append(page_model)

            del native_page
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
        "page_stage_outputs": output_stage,
        "page_models_final": output_models,
        "image_rows": image_rows,
        "chunks": output_chunks,
        "qdrant_text_points": text_points,
        "qdrant_image_points": image_points,
    }

    output_json = output_dir / f"{pdf_path.stem}_pages_{'_'.join(str(page) for page in selected_pages)}.json"
    output_json.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"output_json={output_json}")
    print(f"image_output_dir={image_output_dir}")
    print(f"pages={selected_pages}")
    print(f"chunks={len(chunks)}")
    print(f"images={len(image_rows)}")
    print("Inspect `page_stage_outputs`, `page_models_final`, `chunks`, and optional `qdrant_*_points` in the JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
