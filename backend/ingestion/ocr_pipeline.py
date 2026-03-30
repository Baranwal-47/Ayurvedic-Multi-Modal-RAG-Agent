"""Google Vision OCR pipeline for scanned or garbled PDF pages."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


_VISION_CLIENT: Any | None = None


@dataclass(frozen=True)
class OCRPageResult:
    page_number: int
    text: str
    confidence: float
    text_units: list[dict[str, Any]]
    word_units: list[dict[str, Any]]
    image_size: tuple[int, int]
    page_size: tuple[float, float]


def _build_vision_client() -> Any:
    global _VISION_CLIENT
    if _VISION_CLIENT is not None:
        return _VISION_CLIENT

    try:
        from google.cloud import vision
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-vision is required for OCR. Add it to the environment before ingestion."
        ) from exc

    _VISION_CLIENT = vision.ImageAnnotatorClient()
    return _VISION_CLIENT


def warmup_ocr() -> None:
    """Initialize the Vision client early when credentials are available."""
    try:
        _build_vision_client()
    except Exception as exc:
        print(f"[OCRPipeline] Vision warmup skipped: {exc}")


def wait_for_ocr_ready() -> None:
    """Compatibility wrapper for existing callers."""
    warmup_ocr()


class OCRPipeline:
    """OCR adapter that renders PDF pages and sends them to Google Vision."""

    def __init__(self, dpi: int = 220, max_side: int = 2600) -> None:
        self.dpi = int(dpi)
        self.max_side = int(max_side)

    def process_page(
        self,
        pdf_path: str | Path,
        page_number: int,
        route_reason: str | None = None,
        ocr_profile: str = "default",
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "page_number": int(page_number),
            "text": "",
            "raw_text": "",
            "confidence": 0.0,
            "engine_used": "none",
        }

        if route_reason not in {"scanned", "garbled", "forced"}:
            return response

        try:
            image_bytes, image_size, page_size = self._render_page_as_png(
                pdf_path,
                page_number,
                ocr_profile=ocr_profile,
            )
            page_result = self._run_google_vision(
                page_number=page_number,
                image_bytes=image_bytes,
                image_size=image_size,
                page_size=page_size,
                source_file=Path(pdf_path).name,
            )
            response.update(
                {
                    "text": page_result.text,
                    "raw_text": page_result.text,
                    "confidence": float(page_result.confidence),
                    "engine_used": "google_vision",
                    "text_units": page_result.text_units,
                    "word_units": page_result.word_units,
                }
            )
            return response
        except Exception as exc:
            print(f"[OCRPipeline] Vision OCR failed on page {page_number}: {exc}")
            return response

    def _render_page_as_png(
        self,
        pdf_path: str | Path,
        page_number: int,
        *,
        ocr_profile: str,
    ) -> tuple[bytes, tuple[int, int], tuple[float, float]]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if page_number < 1:
            raise ValueError("page_number must be 1-based and >= 1")

        dpi = 260 if ocr_profile == "garbled_table" else self.dpi
        scale = max(float(dpi) / 72.0, 1.0)

        with fitz.open(path) as doc:
            if page_number > doc.page_count:
                raise ValueError(f"page_number {page_number} out of range (1..{doc.page_count})")
            page = doc.load_page(page_number - 1)
            page_size = (float(page.rect.width), float(page.rect.height))
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

        image_mode = "RGB" if pix.n >= 3 else "L"
        image = Image.frombytes(image_mode, (pix.width, pix.height), pix.samples)
        image = self._resize_image(image, max_side=self.max_side if ocr_profile != "garbled_table" else 3200)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), image.size, page_size

    @staticmethod
    def _resize_image(image: Image.Image, *, max_side: int) -> Image.Image:
        width, height = image.size
        longest = max(width, height)
        if longest <= max_side:
            return image

        scale = float(max_side) / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(new_size)

    @staticmethod
    def _average_word_confidence(annotation: Any) -> float:
        page_scores: list[float] = []
        for page in getattr(annotation, "pages", []) or []:
            for block in getattr(page, "blocks", []) or []:
                for paragraph in getattr(block, "paragraphs", []) or []:
                    for word in getattr(paragraph, "words", []) or []:
                        confidence = getattr(word, "confidence", None)
                        if confidence is not None:
                            page_scores.append(float(confidence))
        if not page_scores:
            return 0.0
        return sum(page_scores) / len(page_scores)

    @staticmethod
    def _bbox_to_page_space(vertices: list[Any], *, image_size: tuple[int, int], page_size: tuple[float, float]) -> list[float]:
        image_width = max(1, int(image_size[0]))
        image_height = max(1, int(image_size[1]))
        page_width = max(1.0, float(page_size[0]))
        page_height = max(1.0, float(page_size[1]))

        xs = [max(0, min(image_width, int(getattr(v, "x", 0) or 0))) for v in vertices]
        ys = [max(0, min(image_height, int(getattr(v, "y", 0) or 0))) for v in vertices]
        if not xs or not ys:
            return [0.0, 0.0, 0.0, 0.0]

        x_scale = page_width / image_width
        y_scale = page_height / image_height
        return [
            min(xs) * x_scale,
            min(ys) * y_scale,
            max(xs) * x_scale,
            max(ys) * y_scale,
        ]

    def _run_google_vision(
        self,
        *,
        page_number: int,
        image_bytes: bytes,
        image_size: tuple[int, int],
        page_size: tuple[float, float],
        source_file: str,
    ) -> OCRPageResult:
        client = _build_vision_client()
        from google.cloud import vision

        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
        if response.error.message:
            raise RuntimeError(response.error.message)

        annotation = response.full_text_annotation
        if annotation is None:
            return OCRPageResult(
                page_number=page_number,
                text="",
                confidence=0.0,
                text_units=[],
                word_units=[],
                image_size=image_size,
                page_size=page_size,
            )

        text = str(getattr(annotation, "text", "") or "").strip()
        confidence = self._average_word_confidence(annotation)
        paragraph_units: list[dict[str, Any]] = []
        word_units: list[dict[str, Any]] = []
        paragraph_index = 0
        word_index = 0

        for page in getattr(annotation, "pages", []) or []:
            for block in getattr(page, "blocks", []) or []:
                for paragraph in getattr(block, "paragraphs", []) or []:
                    paragraph_index += 1
                    paragraph_words: list[str] = []
                    for word in getattr(paragraph, "words", []) or []:
                        word_index += 1
                        symbols = getattr(word, "symbols", []) or []
                        word_text = "".join(getattr(symbol, "text", "") for symbol in symbols).strip()
                        if word_text:
                            paragraph_words.append(word_text)
                        word_bbox = self._bbox_to_page_space(
                            list(getattr(getattr(word, "bounding_box", None), "vertices", []) or []),
                            image_size=image_size,
                            page_size=page_size,
                        )
                        word_units.append(
                            {
                                "unit_id": f"{Path(source_file).stem}:p{page_number}:ocr-word:{word_index}",
                                "text": word_text,
                                "page_number": page_number,
                                "bbox": word_bbox,
                                "confidence": float(getattr(word, "confidence", 0.0) or 0.0),
                                "source_engine": "vision",
                            }
                        )

                    paragraph_text = " ".join(part for part in paragraph_words if part).strip()
                    if not paragraph_text:
                        continue

                    paragraph_bbox = self._bbox_to_page_space(
                        list(getattr(getattr(paragraph, "bounding_box", None), "vertices", []) or []),
                        image_size=image_size,
                        page_size=page_size,
                    )
                    paragraph_units.append(
                        {
                            "unit_id": f"{Path(source_file).stem}:p{page_number}:ocr-paragraph:{paragraph_index}",
                            "text": paragraph_text,
                            "block_type": "paragraph",
                            "kind": "paragraph",
                            "page_number": page_number,
                            "source_file": source_file,
                            "heading_context": "",
                            "bbox": paragraph_bbox,
                            "reading_order": len(paragraph_units),
                            "column_id": None,
                            "languages": ["unknown"],
                            "scripts": ["Zyyy"],
                            "confidence": float(getattr(paragraph, "confidence", 0.0) or 0.0),
                            "source_engine": "vision",
                        }
                    )

        return OCRPageResult(
            page_number=page_number,
            text=text,
            confidence=confidence,
            text_units=paragraph_units,
            word_units=word_units,
            image_size=image_size,
            page_size=page_size,
        )
