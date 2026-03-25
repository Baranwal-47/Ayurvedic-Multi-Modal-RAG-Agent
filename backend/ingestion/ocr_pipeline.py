"""OCR pipeline for scanned PDF pages with PaddleOCR primary and Tesseract fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
from pdf2image import convert_from_path


class OCRPipeline:
	"""Extract text from scanned PDF pages with confidence-aware fallback."""

	def __init__(self, paddle_langs: list[str] | None = None) -> None:
		self.paddle_langs = paddle_langs or ["hi", "en"]
		self._paddle_model: PaddleOCR | None = None

	def process_page(self, pdf_path: str | Path, page_number: int) -> dict[str, Any]:
		"""
		Process a single PDF page and return:
		{page_number, text, confidence, engine_used}
		"""
		response: dict[str, Any] = {
			"page_number": int(page_number),
			"text": "",
			"confidence": 0.0,
			"engine_used": "tesseract",
		}

		try:
			page_image = self._pdf_page_to_image(pdf_path, page_number)
			processed = self._preprocess_image(page_image)

			paddle_text, paddle_conf = self._run_paddleocr(processed)
			if paddle_text and paddle_conf >= 0.80 and self._is_unicode_valid(paddle_text):
				response.update(
					{
						"text": paddle_text,
						"confidence": float(paddle_conf),
						"engine_used": "paddleocr",
					}
				)
				return response

			tess_lang = self._detect_tesseract_langs(processed)
			tess_text, tess_conf = self._run_tesseract(processed, tess_lang)

			response.update(
				{
					"text": tess_text,
					"confidence": float(tess_conf),
					"engine_used": "tesseract",
				}
			)
			return response

		except Exception as exc:
			print(f"[OCRPipeline] process_page failed on page {page_number}: {exc}")
			return response

	def _pdf_page_to_image(self, pdf_path: str | Path, page_number: int) -> np.ndarray:
		path = Path(pdf_path)
		if not path.exists():
			raise FileNotFoundError(f"PDF not found: {path}")
		if page_number < 1:
			raise ValueError("page_number must be 1-based and >= 1")

		try:
			pil_pages = convert_from_path(
				str(path),
				dpi=300,
				first_page=page_number,
				last_page=page_number,
			)
			if pil_pages:
				rgb = np.array(pil_pages[0])
				return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
		except Exception as exc:
			print(f"[OCRPipeline] pdf2image failed, fallback to PyMuPDF render: {exc}")

		# Fallback path when Poppler is unavailable.
		with fitz.open(path) as doc:
			if page_number > doc.page_count:
				raise ValueError(
					f"page_number {page_number} out of range (1..{doc.page_count})"
				)
			page = doc.load_page(page_number - 1)
			matrix = fitz.Matrix(2.0, 2.0)
			pix = page.get_pixmap(matrix=matrix, alpha=True)
			arr = np.frombuffer(pix.samples, dtype=np.uint8)
			img = arr.reshape(pix.height, pix.width, pix.n)

			# Handle common channel layouts from PyMuPDF robustly.
			if pix.n == 4:
				return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
			if pix.n == 3:
				return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			if pix.n == 1:
				return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

			# Conservative fallback if uncommon channel count appears.
			if img.shape[-1] > 3:
				return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
			return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	def _preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
		gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
		denoised = cv2.GaussianBlur(gray, (5, 5), 0)
		binary = cv2.adaptiveThreshold(
			denoised,
			255,
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			cv2.THRESH_BINARY,
			31,
			11,
		)
		return self._deskew(binary)

	def _deskew(self, binary_img: np.ndarray) -> np.ndarray:
		coords = np.column_stack(np.where(binary_img < 255))
		if coords.size == 0:
			return binary_img

		angle = cv2.minAreaRect(coords)[-1]
		if angle < -45:
			angle = 90 + angle
		angle = -angle

		h, w = binary_img.shape[:2]
		center = (w // 2, h // 2)
		matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

		return cv2.warpAffine(
			binary_img,
			matrix,
			(w, h),
			flags=cv2.INTER_CUBIC,
			borderMode=cv2.BORDER_REPLICATE,
		)

	def _get_paddle(self) -> PaddleOCR:
		if self._paddle_model is None:
			# PaddleOCR accepts a single language model; use Hindi model for Devanagari-first pages.
			# If unavailable in local setup, it falls back to English model.
			try:
				self._paddle_model = PaddleOCR(use_angle_cls=True, lang="hi")
			except Exception:
				self._paddle_model = PaddleOCR(use_angle_cls=True, lang="en")
		return self._paddle_model

	def _run_paddleocr(self, processed_img: np.ndarray) -> tuple[str, float]:
		model = self._get_paddle()
		result = None

		# PaddleOCR API differs across versions; try common call signatures.
		try:
			result = model.ocr(processed_img, cls=True)
		except TypeError:
			try:
				result = model.ocr(processed_img)
			except Exception:
				result = None
		except Exception:
			result = None

		if result is None:
			try:
				result = model.predict(processed_img)
			except Exception:
				return "", 0.0

		lines: list[str] = []
		confs: list[float] = []

		if not result:
			return "", 0.0

		page_result = result[0] if isinstance(result, list) else result
		if not page_result:
			return "", 0.0

		for item in page_result:
			if not item or len(item) < 2:
				continue
			payload = item[1]
			if not payload or len(payload) < 2:
				continue

			text = str(payload[0]).strip()
			try:
				conf = float(payload[1])
			except Exception:
				conf = 0.0

			if text:
				lines.append(text)
				confs.append(conf)

		text = "\n".join(lines).strip()
		confidence = (sum(confs) / len(confs)) if confs else 0.0
		return text, confidence

	def _detect_tesseract_langs(self, processed_img: np.ndarray) -> str:
		default_lang = "san+hin+eng"
		try:
			osd = pytesseract.image_to_osd(processed_img, timeout=10)
			osd_lower = osd.lower()

			if "script: arabic" in osd_lower:
				return "ara+urd+eng"
			if "script: devanagari" in osd_lower:
				return "san+hin+eng"
			if "script: tamil" in osd_lower:
				return "tam+eng"
			if "script: telugu" in osd_lower:
				return "tel+eng"
			if "script: bengali" in osd_lower:
				# Assamese and Bengali share script ranges; try both packs if available.
				return "asm+ben+eng"
			if "script: latin" in osd_lower:
				return "eng"

			return default_lang
		except Exception:
			return default_lang

	def _run_tesseract(self, processed_img: np.ndarray, lang: str) -> tuple[str, float]:
		try:
			data = pytesseract.image_to_data(
				processed_img,
				lang=lang,
				output_type=pytesseract.Output.DICT,
				config="--oem 1 --psm 6",
				timeout=20,
			)

			text_chunks: list[str] = []
			confs: list[float] = []

			for txt, conf_str in zip(data.get("text", []), data.get("conf", [])):
				token = str(txt).strip()
				if not token:
					continue

				try:
					conf = float(conf_str)
				except Exception:
					conf = -1.0

				text_chunks.append(token)
				if conf >= 0:
					confs.append(conf / 100.0)

			text = " ".join(text_chunks).strip()
			confidence = (sum(confs) / len(confs)) if confs else 0.0

			if not text:
				# Fallback in case image_to_data yields empty tokens.
				text = pytesseract.image_to_string(
					processed_img,
					lang=lang,
					config="--oem 1 --psm 6",
					timeout=20,
				).strip()

			if not self._is_unicode_valid(text):
				text = text.replace("\ufffd", "")

			return text, confidence
		except Exception as exc:
			print(f"[OCRPipeline] Tesseract failed ({lang}): {exc}")
			return "", 0.0

	@staticmethod
	def _is_unicode_valid(text: str) -> bool:
		return "\ufffd" not in text
