"""
scripts/ingest_documents.py

Requirements:
    pip install python-dotenv FlagEmbedding qdrant-client pymupdf cloudinary
  pip install docling paddleocr pdf2image opencv-python pytesseract tiktoken langdetect
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embeddings.image_embedder import ImageEmbedder
from embeddings.text_embedder import TextEmbedder
from ingestion.chunker import Chunker
from ingestion.cloudinary_uploader import CloudinaryUploader
from ingestion.docling_parser import DoclingParser
from ingestion.image_captioner import build_image_caption
from ingestion.image_extractor import ImageExtractor
from ingestion.ocr_pipeline import OCRPipeline, warmup_ocr
from ingestion.ocr_routing import apply_ocr_routing_to_document
from normalization.diacritic_normalizer import DiacriticNormalizer
from vector_db.qdrant_client import QdrantManager


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}")


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    batch_size = max(1, int(size))
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _run_with_retries(
    *,
    run_label: str,
    action,
    max_retries: int,
    retry_wait_sec: float,
) -> None:
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        try:
            action()
            return
        except Exception as exc:
            if attempt >= attempts:
                raise
            _log(
                f"[RETRY] {run_label} failed on attempt {attempt}/{attempts}: {exc}. "
                f"Retrying in {retry_wait_sec:.1f}s"
            )
            time.sleep(float(retry_wait_sec))


def _hash_to_int(raw: str) -> int:
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest(), 16) % (10**12)


def _text_point_id(chunk: dict[str, Any]) -> int:
    source_file = str(chunk.get("source_file") or "")
    page_number = int(chunk.get("page_number") or 1)
    normalized_text = str(chunk.get("normalized_text") or "")
    block_type = str(chunk.get("block_type") or "paragraph")
    raw = f"{source_file}_{page_number}_{block_type}_{normalized_text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _image_point_id(image_row: dict[str, Any]) -> int:
    source_file = str(image_row.get("source_file") or "")
    page_number = int(image_row.get("page_number") or 1)
    figure_index = int(image_row.get("figure_index") or 1)
    raw = f"{source_file}_{page_number}_{figure_index}"
    return _hash_to_int(raw)


def _validate_cloudinary_env() -> None:
    required = [
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
    ]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required Cloudinary env vars: {joined}")


def _build_pdf_list(pdf_path: Path | None, pdf_dir: Path | None) -> list[Path]:
    if pdf_path and pdf_dir:
        raise ValueError("Use either --pdf or --dir, not both")
    if not pdf_path and not pdf_dir:
        raise ValueError("Provide one input mode: --pdf <file> or --dir <folder>")

    if pdf_path:
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return [pdf_path]

    assert pdf_dir is not None
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    files = sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No PDFs found in directory: {pdf_dir}")
    return files


def _build_image_caption_blocks(
    pdf_path: Path,
    image_output_dir: Path,
    scanned_pages: set[int] | None = None,
    page_blocks: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images = ImageExtractor().extract(
        pdf_path,
        image_output_dir,
        scanned_pages=scanned_pages,
        page_blocks=page_blocks,
    )
    caption_blocks: list[dict[str, Any]] = []

    for row in images:
        caption = build_image_caption(row)
        row["caption"] = caption

        if not str(caption).strip():
            continue

        caption_blocks.append(
            {
                "text": caption,
                "block_type": "figure_caption",
                "page_number": int(row.get("page_number") or 1),
                "source_file": pdf_path.name,
                "heading_context": str(row.get("nearest_heading") or ""),
            }
        )

    return images, caption_blocks


def _run_pipeline_for_pdf(pdf_path: Path, image_output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _log(f"[PIPELINE] Starting parse/OCR/image/chunk pipeline for: {pdf_path.name}")
    t0 = time.perf_counter()

    parser = DoclingParser()
    ocr = OCRPipeline()
    normalizer = DiacriticNormalizer()
    chunker = Chunker()

    t_parse = time.perf_counter()
    parsed_blocks = parser.parse(pdf_path)
    _log(
        f"[PIPELINE] Parsed blocks: {len(parsed_blocks)} in {time.perf_counter() - t_parse:.1f}s"
    )

    t_ocr = time.perf_counter()
    parsed_blocks, ocr_stats = apply_ocr_routing_to_document(
        pdf_path=pdf_path,
        base_blocks=parsed_blocks,
        parser=parser,
        ocr=ocr,
        force_ocr_all_pages=False,
    )
    _log(
        "[PIPELINE] OCR routing complete "
        f"(scanned={len(ocr_stats.get('scanned_pages') or [])}, "
        f"garbled={len(ocr_stats.get('garbled_pages') or [])}) "
        f"in {time.perf_counter() - t_ocr:.1f}s"
    )

    t_images = time.perf_counter()
    images, image_caption_blocks = _build_image_caption_blocks(
        pdf_path=pdf_path,
        image_output_dir=image_output_dir,
        scanned_pages=set(ocr_stats.get("scanned_pages") or []),
        page_blocks=parsed_blocks,
    )
    _log(
        f"[PIPELINE] Images extracted: {len(images)}, caption blocks: {len(image_caption_blocks)} "
        f"in {time.perf_counter() - t_images:.1f}s"
    )

    t_norm = time.perf_counter()
    ordered: list[tuple[int, int, dict[str, Any]]] = []
    for idx, block in enumerate(parsed_blocks):
        block["text"] = str(block.get("text") or "")
        block["normalized_text"] = normalizer.normalize(block["text"])
        ordered.append((int(block.get("page_number") or 1), idx, block))

    base_idx = len(parsed_blocks)
    for j, block in enumerate(image_caption_blocks):
        ordered.append((int(block.get("page_number") or 1), base_idx + j, block))

    ordered.sort(key=lambda x: (x[0], x[1]))
    all_blocks = [item[2] for item in ordered]
    _log(f"[PIPELINE] Normalization/order complete in {time.perf_counter() - t_norm:.1f}s")

    t_chunk = time.perf_counter()
    chunks = chunker.chunk(all_blocks)
    _log(f"[PIPELINE] Chunks produced: {len(chunks)} in {time.perf_counter() - t_chunk:.1f}s")
    _log(f"[PIPELINE] Total pipeline time for {pdf_path.name}: {time.perf_counter() - t0:.1f}s")

    return chunks, images


def _upsert_text_chunks(
    qdrant: QdrantManager,
    text_embedder: TextEmbedder,
    chunks: list[dict[str, Any]],
) -> int:
    if not chunks:
        return 0

    texts = [str(c.get("normalized_text") or "") for c in chunks]
    t_embed = time.perf_counter()
    vectors = text_embedder.embed(texts)
    _log(
        f"[EMBED] Text vectors generated: {len(vectors)} "
        f"in {time.perf_counter() - t_embed:.1f}s"
    )

    points: list[dict[str, Any]] = []
    for chunk, vec in zip(chunks, vectors):
        payload = {
            "original_text": str(chunk.get("original_text") or ""),
            "normalized_text": str(chunk.get("normalized_text") or ""),
            "language": str(chunk.get("language") or ""),
            "page_number": int(chunk.get("page_number") or 1),
            "source_file": str(chunk.get("source_file") or ""),
            "block_type": str(chunk.get("block_type") or "paragraph"),
            "shloka_number": str(chunk.get("shloka_number") or ""),
            "heading_context": str(chunk.get("heading_context") or ""),
        }

        points.append(
            {
                "id": _text_point_id(chunk),
                "dense_vector": vec["dense_vector"],
                "sparse_indices": vec["sparse_indices"],
                "sparse_values": vec["sparse_values"],
                "payload": payload,
            }
        )

    text_batch_size = int(os.getenv("QDRANT_TEXT_UPSERT_BATCH", "24"))
    upsert_retries = int(os.getenv("QDRANT_UPSERT_RETRIES", "3"))
    retry_wait_sec = float(os.getenv("QDRANT_UPSERT_RETRY_WAIT_SEC", "2"))

    t_upsert = time.perf_counter()
    batches = _chunked(points, text_batch_size)
    for idx, batch in enumerate(batches, start=1):
        _run_with_retries(
            run_label=f"text batch {idx}/{len(batches)}",
            action=lambda b=batch: qdrant.upsert_text_chunks(b),
            max_retries=upsert_retries,
            retry_wait_sec=retry_wait_sec,
        )
        _log(f"[QDRANT] Text batch upserted: {idx}/{len(batches)} ({len(batch)} points)")

    _log(f"[QDRANT] Text upserted total: {len(points)} in {time.perf_counter() - t_upsert:.1f}s")
    return len(points)


def _upsert_image_chunks(
    qdrant: QdrantManager,
    image_embedder: ImageEmbedder,
    cloudinary_uploader: CloudinaryUploader,
    images: list[dict[str, Any]],
) -> int:
    rows = [img for img in images if str(img.get("caption") or "").strip()]
    if not rows:
        return 0

    captions = [str(img.get("caption") or "") for img in rows]
    t_embed = time.perf_counter()
    dense_vectors = image_embedder.embed(captions)
    _log(
        f"[EMBED] Image caption vectors generated: {len(dense_vectors)} "
        f"in {time.perf_counter() - t_embed:.1f}s"
    )

    points: list[dict[str, Any]] = []
    for row, dense in zip(rows, dense_vectors):
        source_file = str(row.get("source_file") or "")
        page_number = int(row.get("page_number") or 1)
        figure_index = int(row.get("figure_index") or 1)
        local_image_path = str(row.get("image_path") or "")

        public_id = cloudinary_uploader.build_public_id(
            source_file=source_file,
            page_number=page_number,
            figure_index=figure_index,
        )
        t_upload = time.perf_counter()
        uploaded_public_id, image_url = cloudinary_uploader.upload_image(
            file_path=local_image_path,
            public_id=public_id,
        )
        _log(
            f"[CLOUDINARY] Uploaded {Path(local_image_path).name} -> {uploaded_public_id} "
            f"in {time.perf_counter() - t_upload:.1f}s"
        )

        try:
            Path(local_image_path).unlink(missing_ok=True)
            _log(f"[CLOUDINARY] Deleted local image after upload: {local_image_path}")
        except Exception as cleanup_exc:
            _log(f"[WARN] Local image cleanup failed ({local_image_path}): {cleanup_exc}")

        payload = {
            "image_caption": str(row.get("caption") or ""),
            "caption": str(row.get("caption") or ""),
            "image_path": image_url,
            "image_url": image_url,
            "public_id": uploaded_public_id,
            "page_number": page_number,
            "source_file": source_file,
            "content_type": str(row.get("content_type") or "figure"),
            "nearest_heading": str(row.get("nearest_heading") or ""),
        }

        points.append(
            {
                "id": _image_point_id(row),
                "dense_vector": dense,
                "payload": payload,
            }
        )

    image_batch_size = int(os.getenv("QDRANT_IMAGE_UPSERT_BATCH", "24"))
    upsert_retries = int(os.getenv("QDRANT_UPSERT_RETRIES", "3"))
    retry_wait_sec = float(os.getenv("QDRANT_UPSERT_RETRY_WAIT_SEC", "2"))

    t_upsert = time.perf_counter()
    batches = _chunked(points, image_batch_size)
    for idx, batch in enumerate(batches, start=1):
        _run_with_retries(
            run_label=f"image batch {idx}/{len(batches)}",
            action=lambda b=batch: qdrant.upsert_image_chunks(b),
            max_retries=upsert_retries,
            retry_wait_sec=retry_wait_sec,
        )
        _log(f"[QDRANT] Image batch upserted: {idx}/{len(batches)} ({len(batch)} points)")

    _log(f"[QDRANT] Image upserted total: {len(points)} in {time.perf_counter() - t_upsert:.1f}s")
    return len(points)


def ingest_pdf(
    pdf_path: Path,
    qdrant: QdrantManager,
    text_embedder: TextEmbedder,
    image_embedder: ImageEmbedder,
    cloudinary_uploader: CloudinaryUploader,
    image_output_root: Path,
) -> dict[str, Any]:
    t_file = time.perf_counter()
    source_file = pdf_path.name
    image_output_dir = image_output_root / pdf_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    chunks, images = _run_pipeline_for_pdf(pdf_path=pdf_path, image_output_dir=image_output_dir)

    deleted_text = qdrant.delete_by_source(source_file, "text_chunks")
    deleted_images = qdrant.delete_by_source(source_file, "image_chunks")
    _log(f"[DEDUP] source={source_file} deleted_text={deleted_text} deleted_images={deleted_images}")

    text_stored = _upsert_text_chunks(qdrant=qdrant, text_embedder=text_embedder, chunks=chunks)
    image_stored = _upsert_image_chunks(
        qdrant=qdrant,
        image_embedder=image_embedder,
        cloudinary_uploader=cloudinary_uploader,
        images=images,
    )

    summary = {
        "source_file": source_file,
        "chunks_stored": text_stored,
        "images_stored": image_stored,
        "deleted_text_points": deleted_text,
        "deleted_image_points": deleted_images,
    }
    _log(
        f"[INGEST] source={source_file} chunks_stored={text_stored} "
        f"images_stored={image_stored} total_time={time.perf_counter() - t_file:.1f}s"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant (text_chunks + image_chunks)")
    parser.add_argument("--pdf", type=Path, default=None, help="Single PDF path")
    parser.add_argument("--dir", type=Path, default=None, help="Directory to ingest all PDFs recursively")
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=Path(os.getenv("IMAGE_STORAGE_PATH", "data/images")) / "_ingest",
        help="Directory where extracted figure images are saved",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch ingestion immediately on the first file failure",
    )
    return parser.parse_args()


def main() -> int:
    t_all = time.perf_counter()
    load_dotenv()
    _validate_cloudinary_env()
    args = _parse_args()

    pdfs = _build_pdf_list(args.pdf, args.dir)
    _log(f"Discovered {len(pdfs)} PDF files for ingestion")
    warmup_ocr()

    qdrant = QdrantManager()
    qdrant.create_collections()

    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder(model=text_embedder.model)
    cloudinary_uploader = CloudinaryUploader.from_env()
    _log("Embedding models initialized (TextEmbedder + shared ImageEmbedder model)")
    _log("Cloudinary uploader initialized (mandatory mode)")

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    total = len(pdfs)
    for idx, pdf in enumerate(pdfs, start=1):
        _log(f"[PROGRESS] {idx}/{total} Starting: {pdf}")
        try:
            summaries.append(
                ingest_pdf(
                    pdf_path=pdf,
                    qdrant=qdrant,
                    text_embedder=text_embedder,
                    image_embedder=image_embedder,
                    cloudinary_uploader=cloudinary_uploader,
                    image_output_root=args.image_output_dir,
                )
            )
            _log(f"[PROGRESS] {idx}/{total} Completed: {pdf.name}")
        except Exception as exc:
            failures.append({"source_file": pdf.name, "error": str(exc)})
            _log(f"[ERROR] {idx}/{total} Failed: {pdf.name} | {exc}")
            if args.stop_on_error:
                _log("Stopping batch due to --stop-on-error")
                break

    print("\n=== Ingestion Summary ===")
    for row in summaries:
        print(
            f"source={row['source_file']} chunks={row['chunks_stored']} images={row['images_stored']} "
            f"deleted_text={row['deleted_text_points']} deleted_images={row['deleted_image_points']}"
        )

    success_count = len(summaries)
    attempted = success_count + len(failures)
    print("\n=== Ingestion Verification ===")
    print(f"attempted={attempted}/{total} success={success_count}/{total} failed={len(failures)}")
    if success_count == total:
        print(f"ALL FILES INGESTED: {success_count}/{total}")
    else:
        print(f"INCOMPLETE INGEST: {success_count}/{total}")

    if failures:
        print("\nFailed files:")
        for f in failures:
            print(f"- {f['source_file']}: {f['error']}")

    print(f"\nTotal elapsed time: {time.perf_counter() - t_all:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
