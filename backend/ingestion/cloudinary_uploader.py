"""Cloudinary upload helper for extracted ingestion images."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Callable, TypeVar

import cloudinary
import cloudinary.api
import cloudinary.uploader
from cloudinary.utils import cloudinary_url

T = TypeVar("T")


class CloudinaryUploader:
    """Uploads local images to Cloudinary and returns deterministic delivery metadata."""

    def __init__(
        self,
        cloud_name: str,
        api_key: str,
        api_secret: str,
        upload_folder: str = "ayurveda-images",
        upload_retries: int = 3,
        retry_base_sec: float = 1.0,
        connect_timeout_sec: int = 10,
        read_timeout_sec: int = 300,
    ) -> None:
        if not cloud_name or not api_key or not api_secret:
            raise ValueError(
                "Cloudinary configuration missing. Required: "
                "CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET"
            )

        self.upload_folder = self._sanitize_folder(upload_folder or "ayurveda_images")
        self.upload_retries = max(1, int(upload_retries))
        self.retry_base_sec = max(0.1, float(retry_base_sec))
        self.connect_timeout_sec = max(1, int(connect_timeout_sec))
        self.read_timeout_sec = max(1, int(read_timeout_sec))

        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,
        )

    def _timeout_sec(self) -> int:
        """
        Cloudinary Python SDK expects a single numeric timeout value.
        Use the larger configured timeout so long reads are not cut short.
        """
        return max(int(self.connect_timeout_sec), int(self.read_timeout_sec))

    @classmethod
    def from_env(cls) -> "CloudinaryUploader":
        return cls(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "").strip(),
            api_key=os.getenv("CLOUDINARY_API_KEY", "").strip(),
            api_secret=os.getenv("CLOUDINARY_API_SECRET", "").strip(),
            upload_folder=os.getenv("CLOUDINARY_UPLOAD_FOLDER", "ayurveda_images").strip(),
            upload_retries=int(os.getenv("CLOUDINARY_UPLOAD_RETRIES", "3")),
            retry_base_sec=float(os.getenv("CLOUDINARY_RETRY_BASE_SEC", "1.0")),
            connect_timeout_sec=int(os.getenv("CLOUDINARY_CONNECT_TIMEOUT_SEC", "10")),
            read_timeout_sec=int(os.getenv("CLOUDINARY_READ_TIMEOUT_SEC", "300")),
        )

    def build_public_id(self, source_file: str, page_number: int, figure_index: int) -> str:
        source_stem = Path(str(source_file or "")).stem
        safe_stem = self._slug(source_stem) or "unknown_source"
        page = max(1, int(page_number or 1))
        figure = max(1, int(figure_index or 1))
        return f"{self.upload_folder}/{safe_stem}/page_{page}/figure_{figure}"

    def upload_image(self, file_path: str | Path, public_id: str) -> tuple[str, str]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found for Cloudinary upload: {path}")

        raw_public_id = str(public_id or "").strip("/")
        pieces = [p for p in raw_public_id.split("/") if p]
        if not pieces:
            raise ValueError("public_id cannot be empty")

        upload_public_id = pieces[-1]
        upload_folder = "/".join(pieces[:-1])

        existing = self._get_existing_asset(raw_public_id)
        if existing is not None:
            existing_public_id = str(existing.get("public_id") or raw_public_id).strip()
            existing_url = str(existing.get("secure_url") or "").strip()
            if not existing_url:
                existing_url = cloudinary_url(existing_public_id, secure=True)[0]
            if not existing_url:
                raise RuntimeError(
                    f"Cloudinary existing asset found but URL generation failed for: {existing_public_id}"
                )
            return existing_public_id, str(existing_url)

        result = self._retry_call(
            operation_label=f"upload public_id={raw_public_id}",
            fn=lambda: cloudinary.uploader.upload(
                str(path),
                public_id=upload_public_id,
                folder=upload_folder,
                resource_type="image",
                overwrite=True,
                invalidate=True,
                timeout=self._timeout_sec(),
            ),
        )

        uploaded_public_id = str(result.get("public_id") or "").strip()
        if not uploaded_public_id:
            raise RuntimeError(f"Cloudinary upload returned empty public_id for: {path}")

        delivery_url = str(result.get("secure_url") or "").strip()
        if not delivery_url:
            delivery_url = cloudinary_url(uploaded_public_id, secure=True)[0]
        if not delivery_url:
            raise RuntimeError(f"Cloudinary URL generation failed for public_id: {uploaded_public_id}")

        return uploaded_public_id, str(delivery_url)

    def _get_existing_asset(self, public_id: str) -> dict | None:
        """Return Cloudinary resource metadata when asset already exists, else None."""
        try:
            result = self._retry_call(
                operation_label=f"resource lookup public_id={public_id}",
                fn=lambda: cloudinary.api.resource(
                    public_id,
                    resource_type="image",
                    timeout=self._timeout_sec(),
                ),
            )
            return result if isinstance(result, dict) else None
        except Exception as exc:
            if self._is_not_found_error(exc):
                return None
            raise

    def _retry_call(self, operation_label: str, fn: Callable[[], T]) -> T:
        last_error: Exception | None = None
        for attempt in range(self.upload_retries):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if attempt >= self.upload_retries - 1:
                    break
                wait_sec = self.retry_base_sec * (2 ** attempt)
                print(
                    f"[CloudinaryUploader] Retry {attempt + 1}/{self.upload_retries} "
                    f"for {operation_label} after error: {exc}. Sleeping {wait_sec:.1f}s"
                )
                time.sleep(wait_sec)

        assert last_error is not None
        raise last_error

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        http_code = getattr(exc, "http_code", None)
        if http_code == 404:
            return True
        text = str(exc or "").lower()
        return "not found" in text and "resource" in text

    @staticmethod
    def _slug(value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-")
        return text

    @staticmethod
    def _sanitize_folder(folder: str) -> str:
        pieces = [p for p in str(folder or "").split("/") if p]
        safe = [CloudinaryUploader._slug(piece) for piece in pieces]
        safe = [s for s in safe if s]
        return "/".join(safe) if safe else "ayurveda-images"
