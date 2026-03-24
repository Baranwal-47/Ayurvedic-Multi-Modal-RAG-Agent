"""Embedded image extraction with page-level text context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz


class ImageExtractor:
	"""Extract embedded PDF images and attach local textual context metadata."""

	def extract(self, pdf_path: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
		"""
		Extract images from PDF and return metadata rows:
		{image_path, page_number, source_file, surrounding_text, nearest_heading}
		"""
		path = Path(pdf_path)
		if not path.exists():
			raise FileNotFoundError(f"PDF not found: {path}")

		out_dir = Path(output_dir)
		out_dir.mkdir(parents=True, exist_ok=True)

		source_stem = path.stem
		source_file = path.name
		rows: list[dict[str, Any]] = []

		with fitz.open(path) as doc:
			for page_number, page in enumerate(doc, start=1):
				text_blocks = self._extract_text_blocks(page)
				heading_blocks = [b for b in text_blocks if self._looks_like_heading(b["text"])]

				page_images = page.get_images(full=True)
				for image_index, image_info in enumerate(page_images, start=1):
					xref = image_info[0]
					image_name = f"{source_stem}_page{page_number}_{image_index}.png"
					image_path = out_dir / image_name

					pix = fitz.Pixmap(doc, xref)
					try:
						if pix.n - pix.alpha >= 4:
							pix = fitz.Pixmap(fitz.csRGB, pix)
						pix.save(image_path)
					finally:
						pix = None

					image_rects = page.get_image_rects(xref)
					image_rect = image_rects[0] if image_rects else None

					surrounding_text = self._get_surrounding_text(text_blocks, image_rect)
					nearest_heading = self._get_nearest_heading(heading_blocks, image_rect)

					rows.append(
						{
							"image_path": str(image_path),
							"page_number": page_number,
							"source_file": source_file,
							"surrounding_text": surrounding_text,
							"nearest_heading": nearest_heading,
						}
					)

		return rows

	@staticmethod
	def _extract_text_blocks(page: fitz.Page) -> list[dict[str, Any]]:
		blocks: list[dict[str, Any]] = []
		page_dict = page.get_text("dict")

		for block in page_dict.get("blocks", []):
			if block.get("type") != 0:
				continue

			lines = []
			for line in block.get("lines", []):
				text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
				if text:
					lines.append(text)

			merged = " ".join(lines).strip()
			if not merged:
				continue

			blocks.append(
				{
					"text": merged,
					"rect": fitz.Rect(block.get("bbox", [0, 0, 0, 0])),
				}
			)

		return blocks

	@staticmethod
	def _get_surrounding_text(text_blocks: list[dict[str, Any]], image_rect: fitz.Rect | None) -> str:
		if not text_blocks:
			return ""

		if image_rect is None:
			joined = " ".join(block["text"] for block in text_blocks)
			return joined[:200].strip()

		# Prefer text blocks spatially near the image rectangle.
		expanded = fitz.Rect(image_rect)
		expanded.x0 -= 80
		expanded.y0 -= 80
		expanded.x1 += 80
		expanded.y1 += 80

		nearby = []
		for block in text_blocks:
			rect = block["rect"]
			if rect.intersects(expanded):
				nearby.append(block)

		if not nearby:
			def dist2(block: dict[str, Any]) -> float:
				r = block["rect"]
				cx, cy = (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0
				ix, iy = (image_rect.x0 + image_rect.x1) / 2.0, (image_rect.y0 + image_rect.y1) / 2.0
				return (cx - ix) ** 2 + (cy - iy) ** 2

			nearby = sorted(text_blocks, key=dist2)[:3]

		nearby_sorted = sorted(nearby, key=lambda b: (b["rect"].y0, b["rect"].x0))
		joined = " ".join(block["text"] for block in nearby_sorted)
		return joined[:200].strip()

	@staticmethod
	def _get_nearest_heading(heading_blocks: list[dict[str, Any]], image_rect: fitz.Rect | None) -> str:
		if not heading_blocks:
			return ""

		if image_rect is None:
			return heading_blocks[0]["text"]

		above = [h for h in heading_blocks if h["rect"].y1 <= image_rect.y0]
		if above:
			above_sorted = sorted(above, key=lambda h: image_rect.y0 - h["rect"].y1)
			return above_sorted[0]["text"]

		nearest = min(
			heading_blocks,
			key=lambda h: abs(((h["rect"].y0 + h["rect"].y1) / 2.0) - ((image_rect.y0 + image_rect.y1) / 2.0)),
		)
		return nearest["text"]

	@staticmethod
	def _looks_like_heading(text: str) -> bool:
		compact = " ".join(text.split())
		if not compact:
			return False

		if len(compact) <= 90 and compact.isupper():
			return True

		if len(compact) <= 120 and compact.endswith(":"):
			return True

		words = compact.split()
		if len(words) <= 10 and all(w[:1].isupper() for w in words if w and w[0].isalpha()):
			return True

		return False
