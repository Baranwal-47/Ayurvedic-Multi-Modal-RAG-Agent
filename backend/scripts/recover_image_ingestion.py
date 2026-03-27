"""Recover image-only ingestion for a single PDF.

This script is intentionally scoped to image recovery only:
- extracts images
- reuses existing Cloudinary assets when present
- uploads only missing images
- builds captions + image embeddings
- upserts into Qdrant image collection

It does NOT process text chunks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import cloudinary
import cloudinary.api
import cloudinary.uploader
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embeddings.image_embedder import ImageEmbedder
from ingestion.image_captioner import build_image_caption
from ingestion.image_extractor import ImageExtractor
from vector_db.qdrant_client import QdrantManager


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}")


def _retry(action_name: str, fn, retries: int = 3, base_wait_sec: float = 1.0):
    attempts = max(1, int(retries))
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i >= attempts - 1:
                raise
            wait = float(base_wait_sec) * (2 ** i)
            _log(
                f"[RETRY] {action_name} failed ({exc}); "
                f"retry {i + 1}/{attempts - 1} in {wait:.1f}s"
            )
            time.sleep(wait)


def _point_id(pdf_stem: str, page_number: int, figure_index: int) -> str:
    return f"{pdf_stem}*page*{page_number}*figure*{figure_index}"


def _build_public_id(pdf_stem: str, page_number: int, figure_index: int) -> str:
    # Required format: ayurveda-images/{pdf_name}/page_{page_number}/figure_{figure_index}
    return f"ayurveda-images/{pdf_stem}/page_{page_number}/figure_{figure_index}"


def _manual_delivery_url(cloud_name: str, public_id: str) -> str:
    return f"https://res.cloudinary.com/{cloud_name}/image/upload/{public_id}"


def _cloudinary_exists(
    public_id: str,
    *,
    retries: int,
    retry_base_sec: float,
    connect_timeout_sec: int,
    read_timeout_sec: int,
) -> dict[str, Any] | None:
    def _lookup():
        return cloudinary.api.resource(
            public_id,
            resource_type="image",
            timeout=(int(connect_timeout_sec), int(read_timeout_sec)),
        )

    try:
        result = _retry(
            action_name=f"cloudinary.resource({public_id})",
            fn=_lookup,
            retries=retries,
            base_wait_sec=retry_base_sec,
        )
        if isinstance(result, dict):
            return result
        return None
    except Exception as exc:
        http_code = getattr(exc, "http_code", None)
        text = str(exc or "").lower()
        if http_code == 404 or ("not found" in text and "resource" in text):
            return None
        raise


def _upload_cloudinary(
    image_path: Path,
    public_id: str,
    *,
    retries: int,
    retry_base_sec: float,
    connect_timeout_sec: int,
    read_timeout_sec: int,
) -> tuple[str, str]:
    parts = [p for p in str(public_id).strip("/").split("/") if p]
    if not parts:
        raise ValueError("public_id cannot be empty")

    upload_public_id = parts[-1]
    upload_folder = "/".join(parts[:-1])

    def _do_upload():
        return cloudinary.uploader.upload(
            str(image_path),
            public_id=upload_public_id,
            folder=upload_folder,
            resource_type="image",
            overwrite=True,
            invalidate=True,
            timeout=(int(connect_timeout_sec), int(read_timeout_sec)),
        )

    result = _retry(
        action_name=f"cloudinary.upload({public_id})",
        fn=_do_upload,
        retries=retries,
        base_wait_sec=retry_base_sec,
    )

    uploaded_public_id = str((result or {}).get("public_id") or "").strip()
    secure_url = str((result or {}).get("secure_url") or "").strip()
    if not uploaded_public_id:
        raise RuntimeError(f"Cloudinary upload returned empty public_id: {image_path}")
    if not secure_url:
        raise RuntimeError(f"Cloudinary upload returned empty secure_url: {uploaded_public_id}")

    return uploaded_public_id, secure_url


def _delete_local_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        _log(f"[WARN] Local file cleanup failed: {path} | {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover image-only ingestion for one PDF")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(
            "D:/Docs/Computer/ayurveda-rag/backend/data/pdfs/93aa6-145-rasa-shastra.pdf"
        ),
        help="Absolute path to target PDF",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=Path(os.getenv("IMAGE_STORAGE_PATH", "data/images")) / "_recover_ingest",
        help="Directory for temporary extracted images",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=24,
        help="Qdrant image upsert batch size",
    )
    return parser.parse_args()


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    batch_size = max(1, int(size))
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> int:
    load_dotenv()
    args = _parse_args()

    pdf_path = args.pdf
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cloud_name = str(os.getenv("CLOUDINARY_CLOUD_NAME") or "").strip()
    api_key = str(os.getenv("CLOUDINARY_API_KEY") or "").strip()
    api_secret = str(os.getenv("CLOUDINARY_API_SECRET") or "").strip()
    if not cloud_name or not api_key or not api_secret:
        raise ValueError(
            "Missing Cloudinary env vars: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET"
        )

    retries = int(os.getenv("CLOUDINARY_UPLOAD_RETRIES", "3"))
    retry_base_sec = float(os.getenv("CLOUDINARY_RETRY_BASE_SEC", "1.0"))
    connect_timeout_sec = int(os.getenv("CLOUDINARY_CONNECT_TIMEOUT_SEC", "10"))
    read_timeout_sec = int(os.getenv("CLOUDINARY_READ_TIMEOUT_SEC", "300"))

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )

    image_output_dir = args.image_output_dir / pdf_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[RECOVER] Start image recovery for: {pdf_path.name}")

    extractor = ImageExtractor()
    rows = extractor.extract(pdf_path, image_output_dir)
    if not rows:
        _log("[RECOVER] No images extracted; nothing to recover")
        return 0

    rows.sort(key=lambda r: (int(r.get("page_number") or 1), int(r.get("figure_index") or 1)))

    embedder = ImageEmbedder()
    qdrant = QdrantManager()
    qdrant.create_collections()

    # Keep exact PDF stem to match existing Cloudinary public_id paths.
    pdf_stem = pdf_path.stem
    total = len(rows)
    points: list[dict[str, Any]] = []
    skipped_uploads = 0
    successful_uploads = 0
    current_page = None

    for i, row in enumerate(rows, start=1):
        page_number = int(row.get("page_number") or 1)
        figure_index = int(row.get("figure_index") or i)
        local_image_path = Path(str(row.get("image_path") or ""))

        if current_page != page_number:
            current_page = page_number
            _log(f"[PAGE] Processing page {page_number}")

        public_id = _build_public_id(pdf_stem, page_number, figure_index)

        existing = _cloudinary_exists(
            public_id,
            retries=retries,
            retry_base_sec=retry_base_sec,
            connect_timeout_sec=connect_timeout_sec,
            read_timeout_sec=read_timeout_sec,
        )

        if existing is not None:
            image_url = _manual_delivery_url(cloud_name=cloud_name, public_id=public_id)
            uploaded_public_id = str(existing.get("public_id") or public_id)
            skipped_uploads += 1
            _log(
                f"[CLOUDINARY][SKIP] {pdf_path.name} page={page_number} figure={figure_index} "
                f"public_id={uploaded_public_id}"
            )
            _delete_local_file(local_image_path)
        else:
            uploaded_public_id, image_url = _upload_cloudinary(
                image_path=local_image_path,
                public_id=public_id,
                retries=retries,
                retry_base_sec=retry_base_sec,
                connect_timeout_sec=connect_timeout_sec,
                read_timeout_sec=read_timeout_sec,
            )
            successful_uploads += 1
            _log(
                f"[CLOUDINARY][UPLOAD] {pdf_path.name} page={page_number} figure={figure_index} "
                f"public_id={uploaded_public_id}"
            )
            _delete_local_file(local_image_path)

        caption = build_image_caption(row)
        dense = embedder.embed([caption])[0]

        point_id = _point_id(pdf_stem, page_number, figure_index)
        payload = {
            "image_caption": caption,
            "caption": caption,
            "image_url": image_url,
            "image_path": image_url,
            "public_id": uploaded_public_id,
            "page_number": page_number,
            "source_file": pdf_path.name,
            "content_type": "figure",
            "nearest_heading": str(row.get("nearest_heading") or ""),
        }

        points.append(
            {
                "id": point_id,
                "dense_vector": dense,
                "payload": payload,
            }
        )

        _log(
            f"[PROGRESS] {i}/{total} prepared point_id={point_id} "
            f"(qdrant upsert pending)"
        )

    batches = _chunked(points, args.upsert_batch_size)
    for idx, batch in enumerate(batches, start=1):
        qdrant.upsert_image_chunks(batch)
        _log(f"[QDRANT][UPSERT] batch {idx}/{len(batches)} inserted {len(batch)} image points")

    _log("[RECOVER] Completed")
    _log(f"[RECOVER] uploads_skipped_existing={skipped_uploads}")
    _log(f"[RECOVER] uploads_successful={successful_uploads}")
    _log(f"[RECOVER] qdrant_points_upserted={len(points)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
