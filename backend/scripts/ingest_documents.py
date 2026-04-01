"""Ingest multilingual PDFs into Qdrant using the rebuilt page-model pipeline."""

from __future__ import annotations

import argparse
import contextlib
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
from ingestion.run_state import RunState
from ingestion.section_detector import SectionDetector
from ingestion.shloka_detector import ShlokaDetector
from vector_db.qdrant_client import QdrantManager


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}")


def _log_stage(label: str, started_at: float) -> None:
    _log(f"[TIMING] {label} took {time.perf_counter() - started_at:.2f}s")


class _TeeStream:
    def __init__(self, terminal_stream) -> None:
        self.terminal_stream = terminal_stream
        self.log_stream = None

    def set_log_stream(self, stream) -> None:
        self.log_stream = stream

    def clear_log_stream(self) -> None:
        self.log_stream = None

    def write(self, data: str) -> int:
        self.terminal_stream.write(data)
        log_stream = self.log_stream
        if log_stream is not None and not getattr(log_stream, "closed", False):
            log_stream.write(data)
        return len(data)

    def flush(self) -> None:
        self.terminal_stream.flush()
        log_stream = self.log_stream
        if log_stream is not None and not getattr(log_stream, "closed", False):
            log_stream.flush()


_STDOUT_ROUTER = _TeeStream(sys.stdout)
_STDERR_ROUTER = _TeeStream(sys.stderr)


def _document_log_path(run_state: RunState, doc_id: str) -> Path:
    return run_state.root_dir / f"{doc_id}.log"


def _cleanup_document_artifacts(*, doc_id: str, pdf_path: Path, qdrant: QdrantManager, cloudinary_uploader: CloudinaryUploader) -> None:
    qdrant.delete_by_doc_id(doc_id)
    cloudinary_uploader.delete_document_assets(pdf_path.name)


def _run_with_retries(*, label: str, fn, retries: int, wait_sec: float):
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt < attempts:
                _log(f"[RETRY] {label} attempt {attempt}/{attempts} failed: {exc}")
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
                    "bbox": unit.get("bbox"),
                    "column_id": unit.get("column_id"),
                    "reading_order": unit.get("reading_order"),
                }
            )
    return rows


def _chunk_fragmentation_stats(chunks: list[dict[str, Any]], page_models: list[dict[str, Any]]) -> tuple[float, str | None]:
    page_count = max(1, len(page_models))
    avg_chunks_per_page = float(len(chunks)) / float(page_count)
    if avg_chunks_per_page >= 40.0:
        return avg_chunks_per_page, "critical_fragmentation"
    if avg_chunks_per_page >= 30.0:
        return avg_chunks_per_page, "high_fragmentation"
    return avg_chunks_per_page, None


def _collect_page_models(pdf_path: Path, doc_id: str, run_state: RunState) -> tuple[list[dict[str, Any]], set[int]]:
    parser = NativePDFParser()
    docling = DoclingPDFParser()
    classifier = PageClassifier()
    builder = PageModelBuilder()
    layout = PageLayout()
    shloka = ShlokaDetector()

    ocr: OCRPipeline | None = None
    docling_ready = False
    docling_targets: set[int] = set()
    decision_rows: list[dict[str, Any]] = []

    page_models: list[dict[str, Any]] = []
    scanned_pages: set[int] = set()

    try:
        with fitz.open(pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
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

                _log(f"[ROUTER] page {page_number}: route={route} reason={route_reason}")
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
                    _log(f"[DOCLING] selective prime failed for {pdf_path.name}; falling back to native parser: {exc}")

        for decision in decision_rows:
            page_number = int(decision["page_number"])
            native_page = decision["native_page"]
            classification = decision["classification"]
            route = str(decision["route"])
            route_reason = str(decision["route_reason"])

            try:
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
                    else:
                        ocr_units = list(ocr_result.get("text_units") or [])
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
                    if raw_line_units:
                        quality["ocr_unit_granularity"] = "line_merged" if len(ocr_units) < len(raw_line_units) else "line"
                    else:
                        quality["ocr_unit_granularity"] = "paragraph"
                    quality["ocr_line_unit_count_raw"] = len(raw_line_units)
                    quality["ocr_unit_count_effective"] = len(ocr_units)
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
                        except Exception as exc:
                            _log(f"[DOCLING] page {page_number} fallback to native parser: {exc}")

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
                page_models.append(page_model)
            except Exception as exc:
                run_state.mark_page_failed(doc_id, page_number, str(exc))
                raise
            finally:
                gc.collect()
    finally:
        if docling.available:
            docling.clear_document(pdf_path)
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
    run_state: RunState,
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
            try:
                cloudinary_public_id, image_url = uploader.upload_image(local_path, public_id)
            except Exception as exc:
                run_state.mark_page_failed(
                    str(page_model.get("doc_id") or ""),
                    int(page_model.get("page_number") or 0),
                    f"image_upload_failed: {exc}",
                )
                raise
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

    with fitz.open(pdf_path) as doc:
        total_pages = int(doc.page_count)

    completed_pages = set(int(page) for page in state.get("completed_pages", []))
    if state.get("document_complete") and len(completed_pages) >= total_pages and not state.get("failed_pages"):
        _log(f"[INGEST] Skipping {pdf_path.name}; all pages already completed in run_state")
        return {
            "doc_id": doc_id,
            "source_file": pdf_path.name,
            "page_count": total_pages,
            "chunks_stored": 0,
            "images_stored": 0,
            "failed_pages": {},
            "skipped": True,
        }

    _cleanup_document_artifacts(doc_id=doc_id, pdf_path=pdf_path, qdrant=qdrant, cloudinary_uploader=cloudinary_uploader)

    run_state.start_document(doc_id)

    image_output_dir = image_output_root / pdf_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        page_models, scanned_pages = _collect_page_models(pdf_path, doc_id, run_state)
        _attach_images(
            pdf_path=pdf_path,
            image_output_dir=image_output_dir,
            page_models=page_models,
            scanned_pages=scanned_pages,
        )

        chunks = Chunker().chunk_document(page_models)
        avg_chunks_per_page, chunk_fragmentation_flag = _chunk_fragmentation_stats(chunks, page_models)
        _log(f"[CHUNKING] pages={len(page_models)} chunks={len(chunks)} avg_chunks_per_page={avg_chunks_per_page:.2f}")
        if chunk_fragmentation_flag is not None:
            _log(
                f"[CHUNKING][WARN] fragmentation={chunk_fragmentation_flag} "
                f"(avg_chunks_per_page={avg_chunks_per_page:.2f}; expected < 30)"
            )
        images = _upload_images(page_models=page_models, uploader=cloudinary_uploader, run_state=run_state)

        for image in images:
            linked_chunk_ids = [chunk["chunk_id"] for chunk in chunks if image["image_id"] in chunk.get("image_ids", [])]
            image["linked_chunk_ids"] = linked_chunk_ids

        text_vectors = text_embedder.embed([chunk["text_for_embedding"] for chunk in chunks])
        image_vectors = image_embedder.embed([image.get("caption") or image.get("surrounding_text") or image["image_id"] for image in images])

        mapper = QdrantMapper()
        text_points = mapper.map_text_points(chunks, text_vectors)
        image_points = mapper.map_image_points(images, image_vectors)

        _run_with_retries(
            label=f"qdrant upsert for {pdf_path.name}",
            fn=lambda: (qdrant.upsert_text_chunks(text_points), qdrant.upsert_image_chunks(image_points)),
            retries=int(os.getenv("QDRANT_UPSERT_RETRIES", "3")),
            wait_sec=float(os.getenv("QDRANT_UPSERT_RETRY_WAIT_SEC", "2")),
        )
    except Exception:
        qdrant.delete_by_doc_id(doc_id)
        try:
            cloudinary_uploader.delete_document_assets(pdf_path.name)
        except Exception as cleanup_exc:
            _log(f"[CLEANUP] Cloudinary cleanup failed for {pdf_path.name}: {cleanup_exc}")
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
        "avg_chunks_per_page": round(avg_chunks_per_page, 2),
        "chunk_fragmentation_flag": chunk_fragmentation_flag,
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

    qdrant = QdrantManager()
    qdrant.create_collections()
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder(model=text_embedder.model)
    uploader = CloudinaryUploader.from_env()
    run_state = RunState(ROOT / "data" / "ingestion_runs")

    summaries: list[dict[str, Any]] = []
    for pdf in pdfs:
        doc_id = _stable_doc_id(pdf)
        log_path = _document_log_path(run_state, doc_id)
        with log_path.open("w", encoding="utf-8") as log_file:
            _STDOUT_ROUTER.set_log_stream(log_file)
            _STDERR_ROUTER.set_log_stream(log_file)
            try:
                with contextlib.redirect_stdout(_STDOUT_ROUTER), contextlib.redirect_stderr(_STDERR_ROUTER):
                    _log(f"[INGEST] Starting {pdf.name}")
                    _log(f"[INGEST] Log file: {log_path}")
                    summaries.append(
                        ingest_pdf(
                            pdf_path=pdf,
                            qdrant=qdrant,
                            text_embedder=text_embedder,
                            image_embedder=image_embedder,
                            cloudinary_uploader=uploader,
                            image_output_root=args.image_output_dir,
                            run_state=run_state,
                        )
                    )
                    _log(f"[INGEST] Completed {pdf.name}")
            finally:
                _STDOUT_ROUTER.clear_log_stream()
                _STDERR_ROUTER.clear_log_stream()

    print("\n=== Ingestion Summary ===")
    for summary in summaries:
        print(
            f"doc_id={summary['doc_id']} source={summary['source_file']} "
            f"pages={summary['page_count']} chunks={summary['chunks_stored']} "
            f"avg_chunks_per_page={summary.get('avg_chunks_per_page')} "
            f"images={summary['images_stored']} fragmentation={summary.get('chunk_fragmentation_flag')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
