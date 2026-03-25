"""Embedded image extraction with page-level text context."""

from __future__ import annotations

import re
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

				placements = self._extract_image_placements(page)
				groups = self._group_placements_into_figures(placements)
				page_text = self._build_page_text(text_blocks)
				caption_candidates = self._extract_figure_caption_candidates(page_text)
				caption_by_group_index = self._assign_captions_to_groups(groups, caption_candidates)

				for figure_index, group in enumerate(groups, start=1):
					group_rects = [p["rect"] for p in group]
					figure_rect = self._union_rect(group_rects)
					if figure_rect is None:
						continue

					image_name = f"{source_stem}_page{page_number}_figure{figure_index}.png"
					image_path = out_dir / image_name
					self._save_page_clip(page, figure_rect, image_path)

					surrounding_text = self._get_surrounding_text(text_blocks, figure_rect)
					nearest_heading = self._get_nearest_heading(heading_blocks, figure_rect)
					explicit_caption = caption_by_group_index.get(figure_index - 1, "")
					resolved_caption = self._resolve_caption_for_figure(
						text_blocks=text_blocks,
						figure_rect=figure_rect,
						explicit_caption=explicit_caption,
					)

					rows.append(
						{
							"image_path": str(image_path),
							"page_number": page_number,
							"source_file": source_file,
							"figure_caption": resolved_caption,
							"surrounding_text": surrounding_text,
							"nearest_heading": nearest_heading,
							"figure_index": figure_index,
							"figure_bbox": [figure_rect.x0, figure_rect.y0, figure_rect.x1, figure_rect.y1],
							"part_count": len(group),
							"source_xrefs": sorted({int(p.get("xref", 0) or 0) for p in group if p.get("xref")}),
						}
					)

		return rows

	@staticmethod
	def _build_page_text(text_blocks: list[dict[str, Any]]) -> str:
		if not text_blocks:
			return ""

		sorted_blocks = sorted(text_blocks, key=lambda b: (b["rect"].y0, b["rect"].x0))
		parts = [" ".join(str(b.get("text", "")).split()) for b in sorted_blocks if str(b.get("text", "")).strip()]
		return " ".join(parts).strip()

	@staticmethod
	def _extract_figure_caption_candidates(page_text: str) -> list[str]:
		if not page_text:
			return []

		compact = " ".join(page_text.split())
		pattern = re.compile(
			r"(Figure\s*\d+\s*\([a-z]\)\s*:\s*.*?)(?=\s*Figure\s*\d+\s*\([a-z]\)\s*:|$)",
			flags=re.IGNORECASE,
		)

		candidates = []
		for m in pattern.finditer(compact):
			text = " ".join(m.group(1).split()).strip(" .;,-")
			text = ImageExtractor._clean_figure_caption(text)
			if len(text) >= 12:
				candidates.append(text)

		return candidates

	@staticmethod
	def _assign_captions_to_groups(groups: list[list[dict[str, Any]]], captions: list[str]) -> dict[int, str]:
		if not groups or not captions:
			return {}

		mapping: dict[int, str] = {}
		limit = min(len(groups), len(captions))
		for i in range(limit):
			mapping[i] = captions[i]

		return mapping

	@staticmethod
	def _resolve_caption_for_figure(
		text_blocks: list[dict[str, Any]],
		figure_rect: fitz.Rect,
		explicit_caption: str,
	) -> str:
		block_caption = ImageExtractor._extract_nearby_caption_text(text_blocks, figure_rect)

		# Prefer explicit Figure/label captures when available, otherwise use geometric fallback.
		if explicit_caption:
			return ImageExtractor._clean_figure_caption(explicit_caption)
		if block_caption:
			return ImageExtractor._clean_figure_caption(block_caption)
		return ""

	@staticmethod
	def _extract_nearby_caption_text(text_blocks: list[dict[str, Any]], image_rect: fitz.Rect) -> str:
		if not text_blocks:
			return ""

		candidates: list[tuple[float, str]] = []
		for block in text_blocks:
			text = " ".join(str(block.get("text", "")).split()).strip()
			rect = block.get("rect")
			if not text or rect is None:
				continue

			if not isinstance(rect, fitz.Rect):
				rect = fitz.Rect(rect)

			# Captions are commonly just below an image and horizontally overlapping.
			vertical_gap = rect.y0 - image_rect.y1
			if vertical_gap < -10 or vertical_gap > 180:
				continue

			overlap_w = max(0.0, min(rect.x1, image_rect.x1) - max(rect.x0, image_rect.x0))
			min_w = min(max(1.0, rect.width), max(1.0, image_rect.width))
			overlap_ratio = overlap_w / min_w
			if overlap_ratio < 0.2:
				continue

			score = 0.0
			lower = text.lower()
			if re.match(r"^(figure|fig\.?|image|plate|photo|graph|chart|diagram|scheme|illustration|panel)\b", lower):
				score += 5.0
			if re.search(r"\([a-z]\)", lower):
				score += 1.0
			if ":" in text:
				score += 0.5
			if 10 <= len(text) <= 260:
				score += 1.0
			if ImageExtractor._looks_like_heading(text):
				score -= 2.0

			# Prefer closer blocks below the image.
			score -= min(vertical_gap, 120.0) / 120.0
			candidates.append((score, text))

		if not candidates:
			return ""

		candidates.sort(key=lambda x: x[0], reverse=True)
		best_score, best_text = candidates[0]
		if best_score < 1.0:
			return ""
		return best_text

	@staticmethod
	def _clean_figure_caption(text: str) -> str:
		cleaned = " ".join(text.split())

		# Trim common journal/footer bleed that may appear after the true figure caption.
		for marker in (" Annals of ", " Vol-", " Issue-", " Jan-", " Feb-", " Mar-", " Apr-", " May-", " Jun-"):
			idx = cleaned.find(marker)
			if idx > 0:
				cleaned = cleaned[:idx].strip(" .;,-")

		return cleaned

	@staticmethod
	def _save_page_clip(page: fitz.Page, rect: fitz.Rect, image_path: Path) -> None:
		clip = rect & page.rect
		if clip.is_empty or clip.width <= 1 or clip.height <= 1:
			return

		pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
		pix.save(image_path)

	@staticmethod
	def _extract_image_placements(page: fitz.Page) -> list[dict[str, Any]]:
		placements: list[dict[str, Any]] = []

		for info in page.get_image_info(xrefs=True):
			bbox = info.get("bbox")
			if not bbox:
				continue

			rect = fitz.Rect(bbox)
			if rect.width <= 2 or rect.height <= 2:
				continue

			placements.append({"xref": info.get("xref"), "rect": rect})

		# Fallback for PDFs where get_image_info is sparse.
		if not placements:
			for image_info in page.get_images(full=True):
				xref = image_info[0]
				for rect in page.get_image_rects(xref):
					if rect.width <= 2 or rect.height <= 2:
						continue
					placements.append({"xref": xref, "rect": fitz.Rect(rect)})

		return ImageExtractor._dedupe_placements(placements)

	@staticmethod
	def _dedupe_placements(placements: list[dict[str, Any]]) -> list[dict[str, Any]]:
		unique: list[dict[str, Any]] = []
		for item in placements:
			rect = item["rect"]
			xref = item.get("xref")
			already = False
			for existing in unique:
				er = existing["rect"]
				same_xref = existing.get("xref") == xref
				close_rect = (
					abs(er.x0 - rect.x0) <= 0.5
					and abs(er.y0 - rect.y0) <= 0.5
					and abs(er.x1 - rect.x1) <= 0.5
					and abs(er.y1 - rect.y1) <= 0.5
				)
				if same_xref and close_rect:
					already = True
					break
			if not already:
				unique.append(item)

		return unique

	@staticmethod
	def _group_placements_into_figures(placements: list[dict[str, Any]], gap: float = 10.0) -> list[list[dict[str, Any]]]:
		if not placements:
			return []

		groups: list[list[dict[str, Any]]] = []
		visited = [False] * len(placements)

		for i in range(len(placements)):
			if visited[i]:
				continue

			stack = [i]
			visited[i] = True
			group: list[dict[str, Any]] = []

			while stack:
				idx = stack.pop()
				group.append(placements[idx])

				for j in range(len(placements)):
					if visited[j]:
						continue
					if ImageExtractor._rects_related(placements[idx]["rect"], placements[j]["rect"], gap):
						visited[j] = True
						stack.append(j)

			groups.append(group)

		def reading_key(group: list[dict[str, Any]]) -> tuple[int, float]:
			y0 = min(p["rect"].y0 for p in group)
			x0 = min(p["rect"].x0 for p in group)
			# Quantize y to avoid microscopic float differences flipping left-right order.
			return (int((y0 + 10.0) // 20.0), x0)

		groups.sort(key=reading_key)
		return groups

	@staticmethod
	def _rects_related(a: fitz.Rect, b: fitz.Rect, gap: float) -> bool:
		expanded = fitz.Rect(a.x0 - gap, a.y0 - gap, a.x1 + gap, a.y1 + gap)
		return expanded.intersects(b)

	@staticmethod
	def _union_rect(rects: list[fitz.Rect]) -> fitz.Rect | None:
		if not rects:
			return None

		union = fitz.Rect(rects[0])
		for rect in rects[1:]:
			union.include_rect(rect)

		if union.width <= 2 or union.height <= 2:
			return None

		return union

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
