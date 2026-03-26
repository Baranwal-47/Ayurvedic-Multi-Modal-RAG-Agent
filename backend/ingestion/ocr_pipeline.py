"""OCR pipeline for scanned PDF pages with PaddleOCR primary and Tesseract fallback."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING

import cv2
import fitz
import numpy as np
import pytesseract
from pdf2image import convert_from_path

if TYPE_CHECKING:
	from paddleocr import PaddleOCR


_PADDLE_SINGLETON: Any | None = None
_PADDLE_LOCK = threading.Lock()
_PADDLE_WARMUP_THREAD: threading.Thread | None = None


def _load_paddle_model() -> None:
	"""Load PaddleOCR once per process in a thread-safe manner."""
	global _PADDLE_SINGLETON
	with _PADDLE_LOCK:
		if _PADDLE_SINGLETON is not None:
			return

		print("[OCRPipeline] Loading PaddleOCR model into memory...")
		try:
			from paddleocr import PaddleOCR

			_PADDLE_SINGLETON = PaddleOCR(use_angle_cls=True, lang="hi")
		except Exception:
			from paddleocr import PaddleOCR

			_PADDLE_SINGLETON = PaddleOCR(use_angle_cls=True, lang="en")
		print("[OCRPipeline] PaddleOCR model ready.")


def warmup_ocr() -> None:
	"""Start asynchronous OCR model warmup; safe to call repeatedly."""
	global _PADDLE_WARMUP_THREAD
	if _PADDLE_SINGLETON is not None:
		return
	if _PADDLE_WARMUP_THREAD is not None and _PADDLE_WARMUP_THREAD.is_alive():
		return

	_PADDLE_WARMUP_THREAD = threading.Thread(
		target=_load_paddle_model,
		daemon=True,
		name="paddle-warmup",
	)
	_PADDLE_WARMUP_THREAD.start()
	print("[OCRPipeline] PaddleOCR warming up in background thread...")


def wait_for_ocr_ready() -> None:
	"""Block until OCR warmup is complete; loads synchronously if needed."""
	if _PADDLE_SINGLETON is not None:
		return
	if _PADDLE_WARMUP_THREAD is not None:
		_PADDLE_WARMUP_THREAD.join()
		return
	_load_paddle_model()


def _get_paddle() -> Any:
	"""Return the singleton OCR model, waiting for warmup if necessary."""
	if _PADDLE_SINGLETON is not None:
		return _PADDLE_SINGLETON
	wait_for_ocr_ready()
	return _PADDLE_SINGLETON


# Begin model loading early so first OCR call usually avoids cold start.
warmup_ocr()


class OCRPipeline:
	"""Extract text from scanned PDF pages with confidence-aware fallback."""

	def __init__(self, paddle_langs: list[str] | None = None) -> None:
		self.paddle_langs = paddle_langs or ["hi", "en"]

	def process_page(
		self,
		pdf_path: str | Path,
		page_number: int,
		route_reason: str | None = None,
		ocr_profile: str = "default",
	) -> dict[str, Any]:
		"""
		Process a single PDF page and return:
		{page_number, text, confidence, engine_used}
		"""
		response: dict[str, Any] = {
			"page_number": int(page_number),
			"text": "",
			"raw_text": "",
			"confidence": 0.0,
			"engine_used": "paddleocr",
		}

		try:
			dpi = 600 if ocr_profile == "garbled_table" else 300
			page_image = self._pdf_page_to_image(pdf_path, page_number, dpi=dpi)
			processed = self._preprocess_image(page_image, profile=ocr_profile)

			paddle_text, paddle_conf = self._run_paddleocr(processed)
			response["raw_text"] = paddle_text
			paddle_valid = bool(paddle_text) and self._is_unicode_valid(paddle_text)
			if ocr_profile == "garbled_table" and paddle_text:
				response.update(
					{
						"text": paddle_text,
						"confidence": float(paddle_conf),
						"engine_used": "paddleocr",
					}
				)
				return response

			if paddle_valid and (paddle_conf >= 0.80 or route_reason in {"scanned", "garbled", "non_latin"}):
				response.update(
					{
						"text": paddle_text,
						"confidence": float(paddle_conf),
						"engine_used": "paddleocr",
					}
				)
				return response

			tess_lang = self._detect_tesseract_langs(processed)
			if tess_lang != "eng" and ocr_profile != "garbled_table":
				if paddle_valid:
					response.update(
						{
							"text": paddle_text,
							"confidence": float(paddle_conf),
							"engine_used": "paddleocr",
						}
					)
				return response
			if ocr_profile == "garbled_table" and tess_lang == "eng":
				tess_lang = "san+hin+eng"

			tess_text, tess_conf = self._run_tesseract(processed, tess_lang)
			if ocr_profile == "garbled_table" and tess_text:
				response.update(
					{
						"text": tess_text,
						"confidence": float(tess_conf),
						"engine_used": "tesseract",
					}
				)
				return response

			if paddle_valid and paddle_conf >= tess_conf:
				response.update(
					{
						"text": paddle_text,
						"confidence": float(paddle_conf),
						"engine_used": "paddleocr",
					}
				)
				return response

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

	def _pdf_page_to_image(self, pdf_path: str | Path, page_number: int, dpi: int = 300) -> np.ndarray:
		path = Path(pdf_path)
		if not path.exists():
			raise FileNotFoundError(f"PDF not found: {path}")
		if page_number < 1:
			raise ValueError("page_number must be 1-based and >= 1")

		try:
			pil_pages = convert_from_path(
				str(path),
				dpi=int(dpi),
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

	def _preprocess_image(self, image_bgr: np.ndarray, profile: str = "default") -> np.ndarray:
		if profile == "garbled_table":
			return self._preprocess_table_image(image_bgr)

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

	def _preprocess_table_image(self, image_bgr: np.ndarray) -> np.ndarray:
		"""Table-friendly preprocessing to preserve fine glyphs and reduce gridline interference."""
		gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

		# Upscale to improve OCR on tightly spaced transliteration tables.
		height, width = gray.shape[:2]
		scaled = cv2.resize(gray, (int(width * 1.5), int(height * 1.5)), interpolation=cv2.INTER_CUBIC)

		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		enhanced = clahe.apply(scaled)

		# Sharpen thin strokes before binarization.
		sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
		sharpened = cv2.filter2D(enhanced, ddepth=-1, kernel=sharpen_kernel)

		_, otsu = cv2.threshold(
			sharpened,
			0,
			255,
			cv2.THRESH_BINARY + cv2.THRESH_OTSU,
		)

		binary = cv2.adaptiveThreshold(
			sharpened,
			255,
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			cv2.THRESH_BINARY,
			31,
			7,
		)

		# Blend adaptive and Otsu maps to keep both fine and strong strokes.
		binary = cv2.bitwise_and(binary, otsu)
		dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
		binary = cv2.dilate(binary, dilate_kernel, iterations=1)

		# Remove strong table gridlines while keeping character strokes.
		inv = 255 - binary
		h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, scaled.shape[1] // 30), 1))
		v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, scaled.shape[0] // 30)))
		h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)
		v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)
		grid = cv2.bitwise_or(h_lines, v_lines)
		inv_wo_grid = cv2.subtract(inv, grid)
		clean = 255 - inv_wo_grid
		clean = cv2.medianBlur(clean, 3)

		return self._deskew(clean)

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

	def _run_paddleocr(self, processed_img: np.ndarray) -> tuple[str, float]:
		model = _get_paddle()
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
