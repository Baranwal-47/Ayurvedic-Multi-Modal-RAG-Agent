"""
scripts/ingest_documents.py

Requirements:
  pip install python-dotenv FlagEmbedding qdrant-client pymupdf
  pip install docling paddleocr pdf2image opencv-python pytesseract tiktoken langdetect
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embeddings.image_embedder import ImageEmbedder
from embeddings.text_embedder import TextEmbedder
from ingestion.chunker import Chunker
from ingestion.docling_parser import DoclingParser
from ingestion.image_captioner import build_image_caption
from ingestion.image_extractor import ImageExtractor
from ingestion.ocr_pipeline import OCRPipeline, warmup_ocr
from ingestion.ocr_routing import apply_ocr_routing_to_document
from normalization.diacritic_normalizer import DiacriticNormalizer
from vector_db.qdrant_client import QdrantManager


def _hash_to_int(raw: str) -> int:
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest(), 16) % (10**12)


def _text_point_id(chunk: dict[str, Any]) -> int:
    source_file = str(chunk.get("source_file") or "")
    page_number = int(chunk.get("page_number") or 1)
    normalized_text = str(chunk.get("normalized_text") or "")
    raw = f"{source_file}_{page_number}_{normalized_text[:60]}"
    return _hash_to_int(raw)


def _image_point_id(image_row: dict[str, Any]) -> int:
    image_path = str(image_row.get("image_path") or "")
    return _hash_to_int(image_path)


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images = ImageExtractor().extract(pdf_path, image_output_dir, scanned_pages=scanned_pages)
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
    parser = DoclingParser()
    ocr = OCRPipeline()
    normalizer = DiacriticNormalizer()
    chunker = Chunker()

    parsed_blocks = parser.parse(pdf_path)
    parsed_blocks, ocr_stats = apply_ocr_routing_to_document(
        pdf_path=pdf_path,
        base_blocks=parsed_blocks,
        parser=parser,
        ocr=ocr,
        force_ocr_all_pages=False,
    )

    images, image_caption_blocks = _build_image_caption_blocks(
        pdf_path=pdf_path,
        image_output_dir=image_output_dir,
        scanned_pages=set(ocr_stats.get("scanned_pages") or []),
    )

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
    chunks = chunker.chunk(all_blocks)

    return chunks, images


def _upsert_text_chunks(
    qdrant: QdrantManager,
    text_embedder: TextEmbedder,
    chunks: list[dict[str, Any]],
) -> int:
    if not chunks:
        return 0

    texts = [str(c.get("normalized_text") or "") for c in chunks]
    vectors = text_embedder.embed(texts)

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

    qdrant.upsert_text_chunks(points)
    return len(points)


def _upsert_image_chunks(
    qdrant: QdrantManager,
    image_embedder: ImageEmbedder,
    images: list[dict[str, Any]],
) -> int:
    rows = [img for img in images if str(img.get("caption") or "").strip()]
    if not rows:
        return 0

    captions = [str(img.get("caption") or "") for img in rows]
    dense_vectors = image_embedder.embed(captions)

    points: list[dict[str, Any]] = []
    for row, dense in zip(rows, dense_vectors):
        payload = {
            "image_caption": str(row.get("caption") or ""),
            "caption": str(row.get("caption") or ""),
            "image_path": str(row.get("image_path") or ""),
            "page_number": int(row.get("page_number") or 1),
            "source_file": str(row.get("source_file") or ""),
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

    qdrant.upsert_image_chunks(points)
    return len(points)


def ingest_pdf(
    pdf_path: Path,
    qdrant: QdrantManager,
    text_embedder: TextEmbedder,
    image_embedder: ImageEmbedder,
    image_output_root: Path,
) -> dict[str, Any]:
    source_file = pdf_path.name
    image_output_dir = image_output_root / pdf_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    chunks, images = _run_pipeline_for_pdf(pdf_path=pdf_path, image_output_dir=image_output_dir)

    deleted_text = qdrant.delete_by_source(source_file, "text_chunks")
    deleted_images = qdrant.delete_by_source(source_file, "image_chunks")
    print(
        f"[DEDUP] source={source_file} deleted_text={deleted_text} deleted_images={deleted_images}"
    )

    text_stored = _upsert_text_chunks(qdrant=qdrant, text_embedder=text_embedder, chunks=chunks)
    image_stored = _upsert_image_chunks(qdrant=qdrant, image_embedder=image_embedder, images=images)

    summary = {
        "source_file": source_file,
        "chunks_stored": text_stored,
        "images_stored": image_stored,
        "deleted_text_points": deleted_text,
        "deleted_image_points": deleted_images,
    }
    print(
        f"[INGEST] source={source_file} chunks_stored={text_stored} images_stored={image_stored}"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant (text_chunks + image_chunks)")
    parser.add_argument("--pdf", type=Path, default=None, help="Single PDF path")
    parser.add_argument("--dir", type=Path, default=None, help="Directory to ingest all PDFs recursively")
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=Path("data/images/_ingest"),
        help="Directory where extracted figure images are saved",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = _parse_args()

    pdfs = _build_pdf_list(args.pdf, args.dir)
    warmup_ocr()

    qdrant = QdrantManager()
    qdrant.create_collections()

    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder(model=text_embedder.model)

    summaries: list[dict[str, Any]] = []
    for pdf in pdfs:
        summaries.append(
            ingest_pdf(
                pdf_path=pdf,
                qdrant=qdrant,
                text_embedder=text_embedder,
                image_embedder=image_embedder,
                image_output_root=args.image_output_dir,
            )
        )

    print("\n=== Ingestion Summary ===")
    for row in summaries:
        print(
            f"source={row['source_file']} chunks={row['chunks_stored']} images={row['images_stored']} "
            f"deleted_text={row['deleted_text_points']} deleted_images={row['deleted_image_points']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
