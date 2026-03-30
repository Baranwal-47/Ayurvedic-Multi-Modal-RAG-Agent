"""Ingest multilingual PDFs into Qdrant using the rebuilt page-model pipeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embeddings.image_embedder import ImageEmbedder
from embeddings.text_embedder import TextEmbedder
from ingestion.chunker import Chunker
from ingestion.cloudinary_uploader import CloudinaryUploader
from ingestion.image_extractor import ImageExtractor
from ingestion.image_text_linker import ImageTextLinker
from ingestion.native_pdf_parser import NativePDFParser
from ingestion.noise_detector import NoiseDetector
from ingestion.ocr_pipeline import OCRPipeline, warmup_ocr
from ingestion.page_classifier import PageClassifier
from ingestion.page_layout import PageLayout
from ingestion.page_model_builder import PageModelBuilder
from ingestion.qdrant_mapper import QdrantMapper
from ingestion.run_state import RunState
from ingestion.section_detector import SectionDetector
from ingestion.shloka_detector import ShlokaDetector
from vector_db.qdrant_client import QdrantManager


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}")


def _run_with_retries(*, label: str, fn, retries: int, wait_sec: float):
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(float(wait_sec) * attempt)


def _stable_doc_id(pdf_path: Path) -> str:
    raw = f"{pdf_path.name}|{pdf_path.stat().st_size}|{pdf_path.resolve()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _build_pdf_list(pdf_path: Path | None, pdf_dir: Path | None) -> list[Path]:
    if pdf_path and pdf_dir:
        raise ValueError("Use either --pdf or --dir, not both")
    if not pdf_path and not pdf_dir:
        raise ValueError("Provide --pdf or --dir")

    if pdf_path:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return [pdf_path]

    assert pdf_dir is not None
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    files = sorted(pdf_dir.rglob("*.pdf"))
    if not files:
        raise FileNotFoundError(f"No PDFs found in directory: {pdf_dir}")
    return files


def _validate_required_env() -> None:
    required = [
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
    ]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing required env vars: {', '.join(missing)}")


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
                }
            )
    return rows


def _collect_page_models(pdf_path: Path, doc_id: str) -> tuple[list[dict[str, Any]], set[int]]:
    parser = NativePDFParser()
    classifier = PageClassifier()
    ocr = OCRPipeline()
    builder = PageModelBuilder()
    layout = PageLayout()
    shloka = ShlokaDetector()

    page_models: list[dict[str, Any]] = []
    scanned_pages: set[int] = set()

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            native_page = parser.parse_page(page_number=page_number, page=page, source_file=pdf_path.name)
            classification = classifier.classify_page(
                pdf_path=pdf_path,
                page_number=page_number,
                native_units=native_page.text_units,
                parser=parser,
            )

            if classification.page_type == "digitized":
                page_model = builder.build(
                    doc_id=doc_id,
                    source_file=pdf_path.name,
                    page_number=page_number,
                    route="digitized",
                    native_units=native_page.text_units,
                )
            else:
                scanned_pages.add(page_number)
                ocr_result = ocr.process_page(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    route_reason=classification.page_type,
                )
                if not ocr_result.get("text_units") and native_page.text_units:
                    raise RuntimeError(f"OCR failed on page {page_number}")
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
            page_models.append(page_model)

            del native_page
            gc.collect()

    page_models = NoiseDetector().mark_document_noise(page_models)
    section_detector = SectionDetector()
    section_path: list[str] = []
    for index, page_model in enumerate(page_models):
        page_models[index], section_path = section_detector.apply(page_model, section_path)

    return page_models, scanned_pages


def _attach_images(
    *,
    pdf_path: Path,
    image_output_dir: Path,
    page_models: list[dict[str, Any]],
    scanned_pages: set[int],
) -> list[dict[str, Any]]:
    extractor = ImageExtractor()
    linker = ImageTextLinker()
    image_rows = extractor.extract(
        pdf_path=pdf_path,
        output_dir=image_output_dir,
        scanned_pages=scanned_pages,
        page_blocks=_flatten_page_blocks(page_models),
    )
    for page_model in page_models:
        linker.apply(page_model, image_rows)
    return image_rows


def _upload_images(
    *,
    page_models: list[dict[str, Any]],
    uploader: CloudinaryUploader,
) -> list[dict[str, Any]]:
    uploaded_images: list[dict[str, Any]] = []
    for page_model in page_models:
        for image in page_model.get("images", []):
            local_path = str(image.get("image_path") or "").strip()
            if not local_path:
                continue
            public_id = uploader.build_public_id(
                source_file=page_model["source_file"],
                page_number=page_model["page_number"],
                figure_index=int(image.get("figure_index") or 1),
            )
            cloudinary_public_id, image_url = uploader.upload_image(local_path, public_id)
            if not image_url:
                raise RuntimeError(f"Cloudinary upload returned empty URL for {local_path}")
            image["cloudinary_public_id"] = cloudinary_public_id
            image["image_url"] = image_url
            image["doc_id"] = page_model["doc_id"]
            image["source_file"] = page_model["source_file"]
            image["page_number"] = page_model["page_number"]
            image["languages"] = ["unknown"]
            image["scripts"] = ["Zyyy"]
            image["linked_chunk_ids"] = []
            uploaded_images.append(image)
            Path(local_path).unlink(missing_ok=True)
    return uploaded_images


def ingest_pdf(
    *,
    pdf_path: Path,
    qdrant: QdrantManager,
    text_embedder: TextEmbedder,
    image_embedder: ImageEmbedder,
    cloudinary_uploader: CloudinaryUploader,
    image_output_root: Path,
    run_state: RunState,
) -> dict[str, Any]:
    doc_id = _stable_doc_id(pdf_path)
    state = run_state.load(doc_id)
    image_output_dir = image_output_root / pdf_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    page_models, scanned_pages = _collect_page_models(pdf_path, doc_id)
    _attach_images(
        pdf_path=pdf_path,
        image_output_dir=image_output_dir,
        page_models=page_models,
        scanned_pages=scanned_pages,
    )

    chunks = Chunker().chunk_document(page_models)
    images = _upload_images(page_models=page_models, uploader=cloudinary_uploader)

    for image in images:
        linked_chunk_ids = [chunk["chunk_id"] for chunk in chunks if image["image_id"] in chunk.get("image_ids", [])]
        image["linked_chunk_ids"] = linked_chunk_ids

    text_vectors = text_embedder.embed([chunk["text_for_embedding"] for chunk in chunks])
    image_vectors = image_embedder.embed([image.get("caption") or image.get("surrounding_text") or image["image_id"] for image in images])

    mapper = QdrantMapper()
    text_points = mapper.map_text_points(chunks, text_vectors)
    image_points = mapper.map_image_points(images, image_vectors)

    qdrant.delete_by_doc_id(doc_id)
    try:
        qdrant.upsert_text_chunks(text_points)
        qdrant.upsert_image_chunks(image_points)
    except Exception:
        qdrant.delete_by_doc_id(doc_id)
        raise

    for page_model in page_models:
        run_state.mark_page_completed(doc_id, int(page_model["page_number"]))
    run_state.mark_document_complete(doc_id)

    state = run_state.load(doc_id)
    return {
        "doc_id": doc_id,
        "source_file": pdf_path.name,
        "page_count": len(page_models),
        "chunks_stored": len(text_points),
        "images_stored": len(image_points),
        "failed_pages": state.get("failed_pages", {}),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant using the rebuilt pipeline")
    parser.add_argument("--pdf", type=Path, default=None, help="Single PDF path")
    parser.add_argument("--dir", type=Path, default=None, help="Directory of PDFs")
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=Path(os.getenv("IMAGE_STORAGE_PATH", "data/images")) / "_ingest",
        help="Local directory for extracted images before Cloudinary upload",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    _validate_required_env()
    args = _parse_args()
    pdfs = _build_pdf_list(args.pdf, args.dir)
    warmup_ocr()

    qdrant = QdrantManager()
    qdrant.create_collections()
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder(model=text_embedder.model)
    uploader = CloudinaryUploader.from_env()
    run_state = RunState(ROOT / "data" / "ingestion_runs")

    summaries: list[dict[str, Any]] = []
    for pdf in pdfs:
        _log(f"[INGEST] Starting {pdf.name}")
        summaries.append(
            _run_with_retries(
                label=pdf.name,
                fn=lambda p=pdf: ingest_pdf(
                    pdf_path=p,
                    qdrant=qdrant,
                    text_embedder=text_embedder,
                    image_embedder=image_embedder,
                    cloudinary_uploader=uploader,
                    image_output_root=args.image_output_dir,
                    run_state=run_state,
                ),
                retries=int(os.getenv("QDRANT_UPSERT_RETRIES", "3")),
                wait_sec=float(os.getenv("QDRANT_UPSERT_RETRY_WAIT_SEC", "2")),
            )
        )
        _log(f"[INGEST] Completed {pdf.name}")

    print("\n=== Ingestion Summary ===")
    for summary in summaries:
        print(
            f"doc_id={summary['doc_id']} source={summary['source_file']} "
            f"pages={summary['page_count']} chunks={summary['chunks_stored']} images={summary['images_stored']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
