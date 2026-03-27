"""Embedded image extraction with page-level text context."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np


class ImageExtractor:
	"""Extract embedded PDF images and attach local textual context metadata."""

	def extract(
		self,
		pdf_path: str | Path,
		output_dir: str | Path,
		scanned_pages: set[int] | None = None,
		page_blocks: list[dict[str, Any]] | None = None,
	) -> list[dict[str, Any]]:
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
		scanned_page_set = set(scanned_pages or set())
		document_blocks_by_page = self._group_document_blocks_by_page(page_blocks or [])

		with fitz.open(path) as doc:
			for page_number, page in enumerate(doc, start=1):
				page_width = float(page.rect.width or 0.0)
				page_height = float(page.rect.height or 0.0)
				raw_text_blocks = self._extract_text_blocks(page)
				document_text_blocks = self._convert_document_blocks_to_text_blocks(
					document_blocks_by_page.get(page_number, []),
					page.rect,
				)
				text_blocks = self._select_text_blocks_for_page(
					raw_text_blocks=raw_text_blocks,
					document_text_blocks=document_text_blocks,
					prefer_document_text=(page_number in scanned_page_set),
				)
				heading_blocks = self._extract_heading_blocks(text_blocks)

				placements = self._extract_image_placements(page)
				embedded_groups = self._group_placements_into_figures(placements, gap=10.0)
				scanned_groups: list[list[dict[str, Any]]] = []
				used_scanned_fallback = False
				has_non_full_embedded = any(
					not self.is_full_page_image(p["rect"], page_width, page_height)
					for p in placements
				)
				if not has_non_full_embedded:
					scanned_candidates = self._detect_scanned_region_placements(page, raw_text_blocks)
					if scanned_candidates:
						used_scanned_fallback = True
						print(
							f"[IMAGE DETECT] page={page_number}, scanned_candidates={len(scanned_candidates)}"
						)
						# Keep scanned candidates as individual groups to prevent over-merge into full-page boxes.
						scanned_groups = [[cand] for cand in scanned_candidates]

				groups = [*embedded_groups, *scanned_groups]
				page_text = self._build_page_text(text_blocks)
				caption_candidates = self._extract_figure_caption_candidates(page_text)
				caption_by_group_index = self._assign_captions_to_groups(groups, caption_candidates)

				for figure_index, group in enumerate(groups, start=1):
					group_rects = [p["rect"] for p in group]
					figure_rect = self._union_rect(group_rects)
					if figure_rect is None:
						continue

					surrounding_text = self._get_surrounding_text(text_blocks, figure_rect)
					nearest_heading = self._get_nearest_heading(heading_blocks, figure_rect)
					explicit_caption = caption_by_group_index.get(figure_index - 1, "")
					resolved_caption = self._resolve_caption_for_figure(
						text_blocks=text_blocks,
						figure_rect=figure_rect,
						explicit_caption=explicit_caption,
					)

					area_ratio = self._compute_area_ratio(figure_rect, page_width, page_height)
					is_full = self.is_full_page_image(figure_rect, page_width, page_height)
					page_has_meaningful_text = self.has_meaningful_text(page_text)
					is_scanned_page = (page_number in scanned_page_set) or used_scanned_fallback
					if is_scanned_page and document_text_blocks:
						text_blocks = document_text_blocks
						heading_blocks = self._extract_heading_blocks(text_blocks)
						page_text = self._build_page_text(text_blocks)
						page_has_meaningful_text = self.has_meaningful_text(page_text)
						surrounding_text = self._get_surrounding_text(text_blocks, figure_rect)
						nearest_heading = self._get_nearest_heading(heading_blocks, figure_rect)
						explicit_caption = caption_by_group_index.get(figure_index - 1, "")
						resolved_caption = self._resolve_caption_for_figure(
							text_blocks=text_blocks,
							figure_rect=figure_rect,
							explicit_caption=explicit_caption,
						)
					has_caption = self.has_caption_nearby(text_blocks, figure_rect) or bool(str(resolved_caption or "").strip())
					is_table_like = self.looks_like_table(
						text_blocks=text_blocks,
						image_bbox=figure_rect,
						resolved_caption=resolved_caption,
						surrounding_text=surrounding_text,
					)
					print(
						f"[IMAGE CANDIDATE] page={page_number}, "
						f"bbox=({figure_rect.x0:.1f},{figure_rect.y0:.1f},{figure_rect.x1:.1f},{figure_rect.y1:.1f}), "
						f"area_ratio={area_ratio:.3f}"
					)
					keep = self._should_keep_image_candidate(
						is_full=is_full,
						area_ratio=area_ratio,
						has_caption=has_caption,
						page_has_meaningful_text=page_has_meaningful_text,
						is_table_like=is_table_like,
						is_scanned_page=is_scanned_page,
					)
					print(
						f"[IMAGE FILTER] page={page_number}, area_ratio={area_ratio:.2f}, "
						f"full_page={is_full}, scanned_page={is_scanned_page}, kept={keep}"
					)
					if not keep:
						continue

					image_name = f"{source_stem}_page{page_number}_figure{figure_index}.png"
					image_path = out_dir / image_name
					self._save_page_clip(page, figure_rect, image_path)

					if is_table_like:
						content_type = "table"
					elif is_full and not page_has_meaningful_text:
						content_type = "figure"
					else:
						content_type = "figure"

					rows.append(
						{
							"image_path": str(image_path),
							"page_number": page_number,
							"source_file": source_file,
							"content_type": content_type,
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
	def _group_document_blocks_by_page(
		page_blocks: list[dict[str, Any]],
	) -> dict[int, list[dict[str, Any]]]:
		grouped: dict[int, list[dict[str, Any]]] = {}
		for block in page_blocks:
			page_number = int((block or {}).get("page_number") or 1)
			grouped.setdefault(page_number, []).append(block)
		return grouped

	@staticmethod
	def _convert_document_blocks_to_text_blocks(
		page_blocks: list[dict[str, Any]],
		page_rect: fitz.Rect,
	) -> list[dict[str, Any]]:
		filtered = [b for b in page_blocks if str((b or {}).get("text") or "").strip()]
		if not filtered:
			return []

		height = max(1.0, float(page_rect.height))
		band_height = max(18.0, height / max(1, len(filtered)))
		text_blocks: list[dict[str, Any]] = []

		for idx, block in enumerate(filtered):
			y0 = min(page_rect.y1 - 1.0, page_rect.y0 + (idx * band_height))
			y1 = min(page_rect.y1, max(y0 + 18.0, y0 + band_height))
			text_blocks.append(
				{
					"text": str(block.get("text") or "").strip(),
					"rect": fitz.Rect(page_rect.x0, y0, page_rect.x1, y1),
					"block_type": str(block.get("block_type") or "paragraph"),
					"heading_context": str(block.get("heading_context") or "").strip(),
				}
			)

		return text_blocks

	@staticmethod
	def _select_text_blocks_for_page(
		raw_text_blocks: list[dict[str, Any]],
		document_text_blocks: list[dict[str, Any]],
		prefer_document_text: bool,
	) -> list[dict[str, Any]]:
		if prefer_document_text and document_text_blocks:
			return document_text_blocks
		if raw_text_blocks:
			return raw_text_blocks
		if document_text_blocks:
			return document_text_blocks
		return raw_text_blocks

	@staticmethod
	def _extract_heading_blocks(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
		headings: list[dict[str, Any]] = []
		seen: set[tuple[str, int, int]] = set()
		for block in text_blocks:
			text = str(block.get("text") or "").strip()
			rect = block.get("rect")
			if not text or rect is None:
				continue

			if str(block.get("block_type") or "") == "heading" or ImageExtractor._looks_like_heading(text):
				key = (text, int(rect.x0), int(rect.y0))
				if key not in seen:
					headings.append({"text": text, "rect": rect})
					seen.add(key)

			heading_context = str(block.get("heading_context") or "").strip()
			if heading_context:
				key = (heading_context, int(rect.x0), int(rect.y0))
				if key not in seen:
					headings.append({"text": heading_context, "rect": rect})
					seen.add(key)

		return headings

	@staticmethod
	def _compute_area_ratio(img_bbox: fitz.Rect, page_width: float, page_height: float) -> float:
		if page_width <= 0 or page_height <= 0:
			return 0.0

		img_area = max(0.0, float(img_bbox.width)) * max(0.0, float(img_bbox.height))
		page_area = page_width * page_height
		if page_area <= 0:
			return 0.0
		return img_area / page_area

	@staticmethod
	def is_full_page_image(img_bbox: fitz.Rect, page_width: float, page_height: float) -> bool:
		"""Return True when image occupies most of a page (likely scanned page background)."""
		img_area = max(0.0, float(img_bbox.width)) * max(0.0, float(img_bbox.height))
		page_area = max(0.0, page_width) * max(0.0, page_height)
		if page_area <= 0:
			return False
		return (img_area / page_area) > 0.85

	@staticmethod
	def has_meaningful_text(page_text: str, min_chars: int = 50) -> bool:
		"""Return True when parser-extracted page text is substantial."""
		return len(str(page_text or "").strip()) > int(min_chars)

	@staticmethod
	def has_caption_nearby(blocks: list[dict[str, Any]], image_bbox: fitz.Rect) -> bool:
		"""Return True when a likely caption appears immediately below/near an image."""
		caption_text = ImageExtractor._extract_nearby_caption_text(blocks, image_bbox)
		return bool(str(caption_text or "").strip())

	@staticmethod
	def looks_like_table(
		text_blocks: list[dict[str, Any]],
		image_bbox: fitz.Rect,
		resolved_caption: str,
		surrounding_text: str,
	) -> bool:
		"""Heuristic table detector from nearby text/caption cues."""
		caption = str(resolved_caption or "").lower()
		surround = str(surrounding_text or "").lower()
		if "table" in caption or "table" in surround:
			return True

		near_caption = ImageExtractor._extract_nearby_caption_text(text_blocks, image_bbox).lower()
		if "table" in near_caption:
			return True

		# Numeric-heavy nearby text often accompanies tabular figures.
		num_ratio = 0.0
		letters = sum(1 for c in surround if c.isalpha())
		digits = sum(1 for c in surround if c.isdigit())
		if letters + digits > 0:
			num_ratio = digits / (letters + digits)
		return num_ratio > 0.30 and len(surround.strip()) > 30

	@staticmethod
	def _should_keep_image_candidate(
		is_full: bool,
		area_ratio: float,
		has_caption: bool,
		page_has_meaningful_text: bool,
		is_table_like: bool,
		is_scanned_page: bool,
	) -> bool:
		"""
		Classify image candidates to reduce scanned-page false positives.

		Rules:
		- Full-page and near-full-page images are treated as page snapshots and ignored.
		- Medium-large no-caption images are retained when table-like, else suppressed only if near-full-page.
		- Smaller images are retained as figure candidates.
		"""
		if is_full and not is_scanned_page:
			return False

		# Ignore near-full-page captures to avoid ingesting scanned page backgrounds.
		if area_ratio >= 0.85 and not is_scanned_page:
			return False

		if area_ratio >= 0.70 and not has_caption and not is_table_like:
			return False

		if area_ratio > 0.30 and not has_caption:
			# Keep medium-size no-caption images to preserve table/diagram content.
			return True

		return True

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
	def _detect_scanned_region_placements(
		page: fitz.Page,
		text_blocks: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Detect diagram-like regions from scanned page rendering when embedded image extraction is insufficient."""
		pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
		arr = np.frombuffer(pix.samples, dtype=np.uint8)
		img = arr.reshape(pix.height, pix.width, pix.n)

		if pix.n == 1:
			gray = img
		else:
			gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)

		# Invert-binary map for connected component extraction.
		_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

		contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			return []

		img_h, img_w = gray.shape[:2]
		if img_h <= 0 or img_w <= 0:
			return []

		scale_x = float(page.rect.width) / float(img_w)
		scale_y = float(page.rect.height) / float(img_h)
		text_rects = [b.get("rect") for b in text_blocks if isinstance(b.get("rect"), fitz.Rect)]

		candidates: list[dict[str, Any]] = []
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			if w <= 0 or h <= 0:
				continue

			area_ratio = (w * h) / float(img_w * img_h)
			w_ratio = w / float(img_w)
			h_ratio = h / float(img_h)

			# Ignore tiny, thin, and near-full-page regions.
			if area_ratio < 0.015 or area_ratio >= 0.70:
				continue
			if w_ratio < 0.10 or h_ratio < 0.08:
				continue

			rect = fitz.Rect(
				x * scale_x,
				y * scale_y,
				(x + w) * scale_x,
				(y + h) * scale_y,
			)

			overlap_ratio = ImageExtractor._text_overlap_ratio(rect, text_rects)
			if overlap_ratio > 0.60:
				continue

			# Prefer line-rich candidates (diagrams) over plain text patches.
			edges = cv2.Canny(gray[y : y + h, x : x + w], 80, 180)
			edge_density = float(np.count_nonzero(edges)) / float(max(1, w * h))
			if edge_density < 0.012 and overlap_ratio > 0.30:
				continue

			candidates.append(
				{
					"xref": None,
					"rect": rect,
					"score": edge_density + (0.5 - overlap_ratio),
				}
			)

		# NMS-like suppression for overlapping candidates.
		candidates.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
		picked: list[dict[str, Any]] = []
		for cand in candidates:
			if any(ImageExtractor._iou(cand["rect"], p["rect"]) >= 0.60 for p in picked):
				continue
			picked.append({"xref": None, "rect": cand["rect"]})

		return picked

	@staticmethod
	def _text_overlap_ratio(rect: fitz.Rect, text_rects: list[fitz.Rect]) -> float:
		rect_area = max(1.0, float(rect.width) * float(rect.height))
		overlap_area = 0.0
		for tr in text_rects:
			inter = rect & tr
			if inter.is_empty:
				continue
			overlap_area += max(0.0, float(inter.width)) * max(0.0, float(inter.height))
		return min(1.0, overlap_area / rect_area)

	@staticmethod
	def _iou(a: fitz.Rect, b: fitz.Rect) -> float:
		inter = a & b
		if inter.is_empty:
			return 0.0
		inter_area = max(0.0, float(inter.width)) * max(0.0, float(inter.height))
		a_area = max(0.0, float(a.width)) * max(0.0, float(a.height))
		b_area = max(0.0, float(b.width)) * max(0.0, float(b.height))
		den = a_area + b_area - inter_area
		if den <= 0:
			return 0.0
		return inter_area / den

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
