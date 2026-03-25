"""Caption assembly for extracted image metadata."""

from __future__ import annotations


def build_image_caption(image_metadata: dict) -> str:
	"""
	Build a searchable caption from image metadata using priority:
	figure_caption -> nearest_heading -> surrounding_text[:300] -> Diagram from page {N}
	"""
	metadata = image_metadata or {}

	figure_caption = str(metadata.get("figure_caption", "") or "").strip()
	if figure_caption:
		return figure_caption

	nearest_heading = str(metadata.get("nearest_heading", "") or "").strip()
	if nearest_heading and not _is_generic_heading(nearest_heading):
		return nearest_heading

	surrounding_text = str(metadata.get("surrounding_text", "") or "").strip()
	if surrounding_text:
		return surrounding_text[:300]

	page_number = metadata.get("page_number")
	if page_number is None or str(page_number).strip() == "":
		page_number = "unknown"

	return f"Diagram from page {page_number}"


def _is_generic_heading(text: str) -> bool:
	compact = " ".join((text or "").split()).strip().lower()
	if not compact:
		return True

	generic = {
		"figure legends",
		"figure legend",
		"legends",
		"figures",
	}
	return compact in generic
