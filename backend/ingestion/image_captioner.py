"""Caption assembly for extracted image metadata."""

from __future__ import annotations


def build_image_caption(image_metadata: dict) -> str:
	metadata = image_metadata or {}

	figure_caption = str(metadata.get("figure_caption", "") or "").strip()
	nearest_heading = str(metadata.get("nearest_heading", "") or "").strip()
	surrounding_text = str(metadata.get("surrounding_text", "") or "").strip()
	page_number = metadata.get("page_number")

	parts = []

	# 1. Strong signal
	if figure_caption:
			parts.append(figure_caption)

	# 2. VERY IMPORTANT (always include if useful)
	if nearest_heading and not _is_generic_heading(nearest_heading):
			parts.append(nearest_heading)
			parts.append(f"diagram of {nearest_heading}")

	# 3. Context
	if surrounding_text:
			parts.append(surrounding_text[:150])

	# 4. Fallback
	if not parts:
			parts.append(f"Diagram from page {page_number or 'unknown'}")

	return " | ".join(parts)


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
