"""Layout classification and reading-order assignment."""

from __future__ import annotations

import re
from typing import Any


class PageLayout:
    """Assign layout type and reading order using simple bbox geometry."""

    def apply(self, page_model: dict[str, Any]) -> dict[str, Any]:
        text_units = [
            unit
            for unit in page_model.get("text_units", [])
            if unit.get("kind") not in {"label", "noise"}
        ]
        text_units = self._dedupe_overlapping_units(text_units)

        keep_ids = {id(unit) for unit in text_units}
        page_model["text_units"] = [
            unit
            for unit in page_model.get("text_units", [])
            if unit.get("kind") in {"label", "noise"} or id(unit) in keep_ids
        ]

        if not text_units:
            page_model["layout_type"] = "single"
            return page_model

        centers = []
        for unit in text_units:
            bbox = unit.get("bbox") or [0, 0, 0, 0]
            x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
            centers.append(x_center)

        layout_type = self._classify_layout(centers)
        page_model["layout_type"] = layout_type

        if layout_type == "two_column":
            midpoint = sorted(centers)[len(centers) // 2]
            left = []
            right = []
            for unit in text_units:
                bbox = unit.get("bbox") or [0, 0, 0, 0]
                x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
                unit["column_id"] = 0 if x_center <= midpoint else 1
                (left if unit["column_id"] == 0 else right).append(unit)
            ordered = sorted(left, key=self._sort_key) + sorted(right, key=self._sort_key)
        else:
            for unit in text_units:
                unit["column_id"] = 0 if layout_type == "single" else None
            ordered = sorted(text_units, key=self._sort_key)

        for index, unit in enumerate(ordered):
            unit["reading_order"] = index
        return page_model

    @staticmethod
    def _sort_key(unit: dict[str, Any]) -> tuple[float, float]:
        bbox = unit.get("bbox") or [0, 0, 0, 0]
        return (float(bbox[1]), float(bbox[0]))

    @staticmethod
    def _classify_layout(centers: list[float]) -> str:
        if len(centers) < 4:
            return "single"
        min_x = min(centers)
        max_x = max(centers)
        spread = max_x - min_x
        if spread <= 0:
            return "single"
        midpoint = (min_x + max_x) / 2.0
        left = [x for x in centers if x <= midpoint]
        right = [x for x in centers if x > midpoint]
        if left and right and abs(len(left) - len(right)) <= max(2, len(centers) // 2):
            return "two_column"
        return "single"

    @staticmethod
    def _dedupe_overlapping_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not units:
            return []

        ordered = sorted(units, key=PageLayout._sort_key)
        kept: list[dict[str, Any]] = []

        for candidate in ordered:
            duplicate_idx: int | None = None
            for idx, existing in enumerate(kept):
                if PageLayout._is_near_duplicate(existing, candidate):
                    duplicate_idx = idx
                    break

            if duplicate_idx is None:
                kept.append(candidate)
                continue

            existing = kept[duplicate_idx]
            if len(PageLayout._normalize_text(candidate.get("text"))) > len(PageLayout._normalize_text(existing.get("text"))):
                kept[duplicate_idx] = candidate

        return kept

    @staticmethod
    def _is_near_duplicate(a: dict[str, Any], b: dict[str, Any]) -> bool:
        text_a = PageLayout._normalize_text(a.get("text"))
        text_b = PageLayout._normalize_text(b.get("text"))
        if len(text_a) < 30 or len(text_b) < 30:
            return False

        similar_text = text_a == text_b or text_a in text_b or text_b in text_a
        if not similar_text:
            return False

        bbox_a = a.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        bbox_b = b.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        y0 = max(float(bbox_a[1]), float(bbox_b[1]))
        y1 = min(float(bbox_a[3]), float(bbox_b[3]))
        overlap = max(0.0, y1 - y0)
        min_h = max(1.0, min(float(bbox_a[3]) - float(bbox_a[1]), float(bbox_b[3]) - float(bbox_b[1])))
        return (overlap / min_h) >= 0.60

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip()).lower()
