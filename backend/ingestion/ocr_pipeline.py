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
    line_units: list[dict[str, Any]]
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

        if route_reason not in {"scanned", "garbled", "ocr_fallback", "forced"}:
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
                    "line_units": page_result.line_units,
                    "word_units": page_result.word_units,
                }
            )
            return response
        except Exception as exc:
            print(f"[OCRPipeline] Vision OCR failed on page {page_number}: {exc}")
            return response

    @classmethod
    def merge_line_units(
        cls,
        line_units: list[dict[str, Any]],
        *,
        min_line_chars: int = 28,
        max_vertical_gap_abs: float = 8.0,
        max_vertical_gap_ratio: float = 0.85,
        min_x_overlap_ratio: float = 0.25,
        max_merge_lines: int = 6,
        max_merge_chars: int = 420,
    ) -> list[dict[str, Any]]:
        """Merge adjacent OCR lines into paragraph-like units for stable chunking."""
        if not line_units:
            return []

        ordered = sorted(
            [dict(unit) for unit in line_units],
            key=lambda unit: (
                int(unit.get("reading_order", 0)),
                float((unit.get("bbox") or [0.0, 0.0, 0.0, 0.0])[1]),
                float((unit.get("bbox") or [0.0, 0.0, 0.0, 0.0])[0]),
            ),
        )

        merged: list[dict[str, Any]] = []
        buffer: list[dict[str, Any]] = []

        def _flush_buffer() -> None:
            nonlocal buffer
            if not buffer:
                return

            first = buffer[0]
            texts = [str(unit.get("text") or "").strip() for unit in buffer if str(unit.get("text") or "").strip()]
            if not texts:
                buffer = []
                return

            confidences = [float(unit.get("confidence") or 0.0) for unit in buffer if unit.get("confidence") is not None]
            merged_unit: dict[str, Any] = {
                "unit_id": cls._merged_unit_id(first, len(merged)),
                "text": "\n".join(texts),
                "block_type": "paragraph",
                "kind": "paragraph",
                "page_number": int(first.get("page_number") or 0),
                "source_file": first.get("source_file"),
                "heading_context": "",
                "bbox": cls._union_bboxes([list(unit.get("bbox") or [0.0, 0.0, 0.0, 0.0]) for unit in buffer]),
                "reading_order": len(merged),
                "column_id": first.get("column_id"),
                "languages": list(first.get("languages") or ["unknown"]),
                "scripts": list(first.get("scripts") or ["Zyyy"]),
                "confidence": (sum(confidences) / len(confidences)) if confidences else None,
                "source_engine": str(first.get("source_engine") or "vision"),
                "merged_line_count": len(buffer),
            }
            merged.append(merged_unit)
            buffer = []

        for unit in ordered:
            text = str(unit.get("text") or "").strip()
            if not text:
                continue
            if not buffer:
                buffer = [unit]
                continue

            prev = buffer[-1]
            should_merge = cls._should_merge_lines(
                prev=prev,
                current=unit,
                min_line_chars=min_line_chars,
                max_vertical_gap_abs=max_vertical_gap_abs,
                max_vertical_gap_ratio=max_vertical_gap_ratio,
                min_x_overlap_ratio=min_x_overlap_ratio,
            )
            if should_merge:
                projected_chars = sum(len(str(item.get("text") or "").strip()) for item in buffer) + len(text)
                if len(buffer) >= max_merge_lines or projected_chars >= max_merge_chars:
                    _flush_buffer()
                    buffer = [unit]
                else:
                    buffer.append(unit)
            else:
                _flush_buffer()
                buffer = [unit]

        _flush_buffer()
        return merged

    @staticmethod
    def _merged_unit_id(first_unit: dict[str, Any], index: int) -> str:
        base = str(first_unit.get("unit_id") or "ocr-line")
        if ":ocr-line:" in base:
            return base.replace(":ocr-line:", ":ocr-merged:")
        return f"{base}:merged:{index + 1}"

    @classmethod
    def _should_merge_lines(
        cls,
        *,
        prev: dict[str, Any],
        current: dict[str, Any],
        min_line_chars: int,
        max_vertical_gap_abs: float,
        max_vertical_gap_ratio: float,
        min_x_overlap_ratio: float,
    ) -> bool:
        prev_text = str(prev.get("text") or "").strip()
        current_text = str(current.get("text") or "").strip()
        if not prev_text or not current_text:
            return False

        prev_bbox = list(prev.get("bbox") or [0.0, 0.0, 0.0, 0.0])
        curr_bbox = list(current.get("bbox") or [0.0, 0.0, 0.0, 0.0])

        prev_height = max(1.0, float(prev_bbox[3]) - float(prev_bbox[1]))
        curr_height = max(1.0, float(curr_bbox[3]) - float(curr_bbox[1]))
        gap = float(curr_bbox[1]) - float(prev_bbox[3])
        gap_limit = max(float(max_vertical_gap_abs), min(prev_height, curr_height) * float(max_vertical_gap_ratio))

        x_overlap_ratio = cls._x_overlap_ratio(prev_bbox, curr_bbox)
        same_column = x_overlap_ratio >= float(min_x_overlap_ratio) or abs(float(curr_bbox[0]) - float(prev_bbox[0])) <= 12.0

        if not same_column:
            return False

        prev_short = len(prev_text) < int(min_line_chars)
        curr_short = len(current_text) < max(10, int(min_line_chars) // 2)
        small_gap = gap <= gap_limit

        if small_gap:
            return True
        if prev_short and gap <= (gap_limit * 2.0):
            return True
        if curr_short and gap <= (gap_limit * 1.25):
            return True
        return False

    @staticmethod
    def _x_overlap_ratio(a: list[float], b: list[float]) -> float:
        ax0, _, ax1, _ = [float(v) for v in a]
        bx0, _, bx1, _ = [float(v) for v in b]
        overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        min_width = max(1.0, min(ax1 - ax0, bx1 - bx0))
        return overlap / min_width

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
                line_units=[],
                word_units=[],
                image_size=image_size,
                page_size=page_size,
            )

        text = str(getattr(annotation, "text", "") or "").strip()
        confidence = self._average_word_confidence(annotation)
        paragraph_units: list[dict[str, Any]] = []
        line_units: list[dict[str, Any]] = []
        word_units: list[dict[str, Any]] = []
        paragraph_index = 0
        line_index = 0
        word_index = 0

        for page in getattr(annotation, "pages", []) or []:
            for block in getattr(page, "blocks", []) or []:
                for paragraph in getattr(block, "paragraphs", []) or []:
                    paragraph_index += 1
                    paragraph_words: list[str] = []
                    paragraph_word_rows: list[dict[str, Any]] = []
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
                        if word_text:
                            paragraph_word_rows.append(
                                {
                                    "text": word_text,
                                    "bbox": word_bbox,
                                    "confidence": float(getattr(word, "confidence", 0.0) or 0.0),
                                }
                            )

                    paragraph_line_rows = self._group_words_into_lines(paragraph_word_rows)
                    line_text_parts: list[str] = []
                    for line_words in paragraph_line_rows:
                        line_text = " ".join(str(row.get("text") or "").strip() for row in line_words if str(row.get("text") or "").strip()).strip()
                        if not line_text:
                            continue
                        line_index += 1
                        line_bbox = self._union_bboxes([list(row.get("bbox") or [0.0, 0.0, 0.0, 0.0]) for row in line_words])
                        line_confidences = [float(row.get("confidence") or 0.0) for row in line_words if row.get("confidence") is not None]
                        line_confidence = (sum(line_confidences) / len(line_confidences)) if line_confidences else 0.0
                        line_units.append(
                            {
                                "unit_id": f"{Path(source_file).stem}:p{page_number}:ocr-line:{line_index}",
                                "text": line_text,
                                "block_type": "paragraph",
                                "kind": "paragraph",
                                "page_number": page_number,
                                "source_file": source_file,
                                "heading_context": "",
                                "bbox": line_bbox,
                                "reading_order": len(line_units),
                                "column_id": None,
                                "languages": ["unknown"],
                                "scripts": ["Zyyy"],
                                "confidence": float(line_confidence),
                                "source_engine": "vision",
                            }
                        )
                        line_text_parts.append(line_text)

                    paragraph_text = "\n".join(part for part in line_text_parts if part).strip()
                    if not paragraph_text:
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
            line_units=line_units,
            word_units=word_units,
            image_size=image_size,
            page_size=page_size,
        )

    @staticmethod
    def _group_words_into_lines(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not words:
            return []

        def _y_center(row: dict[str, Any]) -> float:
            bbox = list(row.get("bbox") or [0.0, 0.0, 0.0, 0.0])
            return (float(bbox[1]) + float(bbox[3])) / 2.0

        def _x0(row: dict[str, Any]) -> float:
            bbox = list(row.get("bbox") or [0.0, 0.0, 0.0, 0.0])
            return float(bbox[0])

        heights = []
        for row in words:
            bbox = list(row.get("bbox") or [0.0, 0.0, 0.0, 0.0])
            heights.append(max(1.0, float(bbox[3]) - float(bbox[1])))
        sorted_heights = sorted(heights)
        median_height = sorted_heights[len(sorted_heights) // 2] if sorted_heights else 12.0
        y_threshold = max(4.0, float(median_height) * 0.60)

        lines: list[dict[str, Any]] = []
        for row in sorted(words, key=lambda item: (_y_center(item), _x0(item))):
            row_center = _y_center(row)
            target_idx: int | None = None
            best_delta = None

            for idx, line in enumerate(lines):
                delta = abs(float(line["y_center"]) - row_center)
                if delta <= y_threshold and (best_delta is None or delta < best_delta):
                    target_idx = idx
                    best_delta = delta

            if target_idx is None:
                lines.append({"y_center": row_center, "words": [row]})
            else:
                lines[target_idx]["words"].append(row)
                centers = [_y_center(item) for item in lines[target_idx]["words"]]
                lines[target_idx]["y_center"] = sum(centers) / max(1, len(centers))

        ordered_lines = sorted(lines, key=lambda line: float(line["y_center"]))
        return [sorted(list(line["words"]), key=_x0) for line in ordered_lines]

    @staticmethod
    def _union_bboxes(bboxes: list[list[float]]) -> list[float]:
        if not bboxes:
            return [0.0, 0.0, 0.0, 0.0]

        xs0 = [float(bbox[0]) for bbox in bboxes]
        ys0 = [float(bbox[1]) for bbox in bboxes]
        xs1 = [float(bbox[2]) for bbox in bboxes]
        ys1 = [float(bbox[3]) for bbox in bboxes]
        return [min(xs0), min(ys0), max(xs1), max(ys1)]
