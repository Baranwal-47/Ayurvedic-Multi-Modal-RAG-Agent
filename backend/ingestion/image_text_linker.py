"""Attach extracted images to page text and create caption units."""

from __future__ import annotations

from typing import Any


class ImageTextLinker:
    """Convert extracted image rows into PageModel image records and caption units."""

    def apply(self, page_model: dict[str, Any], image_rows: list[dict[str, Any]]) -> dict[str, Any]:
        images: list[dict[str, Any]] = []
        next_caption_index = len(page_model.get("text_units", []))

        for row in image_rows:
            if int(row.get("page_number") or 0) != int(page_model.get("page_number") or 0):
                continue

            image_id = f"{page_model['doc_id']}:p{page_model['page_number']}:img:{int(row.get('figure_index') or 1)}"
            caption = str(row.get("figure_caption") or "").strip()
            caption_unit_ids: list[str] = []
            if caption:
                caption_unit_id = f"{image_id}:caption"
                caption_unit_ids.append(caption_unit_id)
                page_model.setdefault("text_units", []).append(
                    {
                        "unit_id": caption_unit_id,
                        "kind": "caption",
                        "block_type": "caption",
                        "text": caption,
                        "bbox": list(row.get("figure_bbox") or [0.0, 0.0, 0.0, 0.0]),
                        "column_id": None,
                        "reading_order": next_caption_index,
                        "confidence": None,
                        "languages": ["unknown"],
                        "scripts": ["Zyyy"],
                        "source_engine": "heuristic",
                        "section_path": list(page_model.get("section_path") or []),
                    }
                )
                next_caption_index += 1

            images.append(
                {
                    "image_id": image_id,
                    "bbox": list(row.get("figure_bbox") or [0.0, 0.0, 0.0, 0.0]),
                    "image_type": "table_like" if row.get("content_type") == "table" else "diagram",
                    "caption": caption,
                    "caption_source": "explicit" if caption else "missing",
                    "labels": [],
                    "labels_sparse": True,
                    "caption_unit_ids": caption_unit_ids,
                    "label_unit_ids": [],
                    "surrounding_text": str(row.get("surrounding_text") or ""),
                    "section_path": list(page_model.get("section_path") or []),
                    "association_confidence": 1.0 if caption else 0.5,
                    "figure_index": int(row.get("figure_index") or 1),
                    "image_path": str(row.get("image_path") or ""),
                }
            )

        page_model["images"] = images
        return page_model
