"""Layout classification and reading-order assignment."""

from __future__ import annotations

from typing import Any


class PageLayout:
    """Assign layout type and reading order using simple bbox geometry."""

    def apply(self, page_model: dict[str, Any]) -> dict[str, Any]:
        text_units = [unit for unit in page_model.get("text_units", []) if unit.get("kind") != "label"]
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
