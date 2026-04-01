"""Hybrid repair helpers for mixed native/OCR page content."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from normalization.diacritic_normalizer import DiacriticNormalizer


@dataclass(frozen=True)
class HybridRepairResult:
    text_units: list[dict[str, Any]]
    repaired_unit_indexes: list[int]
    legacy_repaired_unit_indexes: list[int]
    used_ocr: bool
    ocr_profile: str


class HybridPageRepair:
    """Repair mixed pages by keeping clean native units and patching suspect units."""

    def __init__(self, legacy_font_map_path: str | Path | None = None, legacy_font_map: dict[str, str] | None = None) -> None:
        self.normalizer = DiacriticNormalizer()
        self.legacy_font_map = dict(legacy_font_map or self._load_legacy_font_map(legacy_font_map_path or os.getenv("LEGACY_FONT_MAP_PATH")))

    def should_use_hybrid_repair(self, native_units: list[dict[str, Any]], parser: Any) -> bool:
        """Return True when a page mixes clean and suspect native blocks."""
        substantial_indexes = self._substantial_unit_indexes(native_units)
        if len(substantial_indexes) < 2:
            return False

        suspect_indexes = self._suspect_unit_indexes(native_units, parser)
        if not suspect_indexes:
            return False

        clean_indexes = [index for index in substantial_indexes if index not in suspect_indexes]
        if not clean_indexes:
            return False

        suspect_ratio = len(suspect_indexes) / max(1, len(substantial_indexes))
        return suspect_ratio < 0.75

    def select_ocr_profile(self, native_units: list[dict[str, Any]], parser: Any) -> str:
        if self._is_garbled_table_candidate(native_units, parser):
            return "garbled_table"
        return "default"

    def repair_units(
        self,
        *,
        native_units: list[dict[str, Any]],
        ocr_result: dict[str, Any] | None,
        parser: Any,
        page_number: int,
        source_file: str,
    ) -> HybridRepairResult:
        """Return a merged unit list that preserves native structure and repairs suspects."""
        repaired_units = [dict(unit) for unit in native_units]
        ocr_units = [dict(unit) for unit in ((ocr_result or {}).get("text_units") or [])]
        ocr_text = str((ocr_result or {}).get("text") or (ocr_result or {}).get("raw_text") or "").strip()

        suspect_indexes = self._suspect_unit_indexes(repaired_units, parser)
        legacy_repaired_indexes: list[int] = []
        ocr_repaired_indexes: list[int] = []

        if not suspect_indexes:
            return HybridRepairResult(
                text_units=repaired_units,
                repaired_unit_indexes=[],
                legacy_repaired_unit_indexes=[],
                used_ocr=bool(ocr_units or ocr_text),
                ocr_profile=self.select_ocr_profile(native_units, parser),
            )

        remaining_indexes = list(suspect_indexes)
        if self.legacy_font_map:
            for index in list(remaining_indexes):
                unit = repaired_units[index]
                decoded_text = self._apply_legacy_font_map(str(unit.get("text") or ""))
                if not decoded_text:
                    continue
                if decoded_text == str(unit.get("text") or ""):
                    continue
                if parser.is_text_garbled(decoded_text):
                    continue

                self._apply_repaired_text(unit, decoded_text, source_engine="legacy_font_map")
                legacy_repaired_indexes.append(index)
                remaining_indexes.remove(index)

        if remaining_indexes:
            ocr_repaired_indexes = self._apply_ocr_repair(
                repaired_units=repaired_units,
                target_indexes=remaining_indexes,
                ocr_units=ocr_units,
                ocr_text=ocr_text,
            )

        return HybridRepairResult(
            text_units=repaired_units,
            repaired_unit_indexes=ocr_repaired_indexes,
            legacy_repaired_unit_indexes=legacy_repaired_indexes,
            used_ocr=bool(ocr_repaired_indexes),
            ocr_profile=self.select_ocr_profile(native_units, parser),
        )

    def _apply_ocr_repair(
        self,
        *,
        repaired_units: list[dict[str, Any]],
        target_indexes: list[int],
        ocr_units: list[dict[str, Any]],
        ocr_text: str,
    ) -> list[int]:
        repaired_indexes: list[int] = []

        for index in target_indexes:
            unit = repaired_units[index]
            matches = self._matching_ocr_units(unit, ocr_units)
            if matches:
                replacement_text = self._join_texts(match.get("text") for match in matches)
                if replacement_text:
                    self._apply_repaired_text(
                        unit,
                        replacement_text,
                        source_engine=str(matches[0].get("source_engine") or "vision"),
                        confidence=self._best_confidence(matches),
                    )
                    repaired_indexes.append(index)

        if repaired_indexes:
            return repaired_indexes

        if not ocr_text:
            return repaired_indexes

        remaining_units = [repaired_units[index] for index in target_indexes]
        segments = self._split_text_to_slots(ocr_text, len(remaining_units))
        for unit, segment, index in zip(remaining_units, segments, target_indexes):
            if not segment:
                continue
            self._apply_repaired_text(unit, segment, source_engine="vision")
            repaired_indexes.append(index)

        return repaired_indexes

    def _apply_repaired_text(
        self,
        unit: dict[str, Any],
        text: str,
        *,
        source_engine: str,
        confidence: float | None = None,
    ) -> None:
        cleaned_text = " ".join(str(text or "").split()).strip()
        unit["text"] = cleaned_text
        unit["source_engine"] = source_engine
        unit["languages"] = self._languages_for_text(cleaned_text)
        unit["scripts"] = self._scripts_for_text(cleaned_text)
        if confidence is not None:
            unit["confidence"] = confidence
        if str(unit.get("kind") or "") == "table_row":
            unit["table_cells"] = self._table_cells_for_text(cleaned_text, unit.get("table_cells"))

    def _apply_legacy_font_map(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned or not self.legacy_font_map:
            return cleaned

        repaired = cleaned
        for old, new in sorted(self.legacy_font_map.items(), key=lambda item: len(str(item[0])), reverse=True):
            if not old:
                continue
            repaired = repaired.replace(str(old), str(new))

        repaired = self.normalizer.normalize(repaired)
        return " ".join(repaired.split()).strip()

    def _matching_ocr_units(self, native_unit: dict[str, Any], ocr_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        native_bbox = self._bbox_to_tuple(native_unit.get("bbox"))
        if native_bbox is None:
            return []

        matches: list[tuple[float, dict[str, Any]]] = []
        for ocr_unit in ocr_units:
            text = str(ocr_unit.get("text") or "").strip()
            if not text:
                continue
            ocr_bbox = self._bbox_to_tuple(ocr_unit.get("bbox"))
            if ocr_bbox is None:
                continue

            overlap = self._bbox_overlap_ratio(native_bbox, ocr_bbox)
            if overlap < 0.15 and not self._bbox_centers_overlap(native_bbox, ocr_bbox):
                continue
            matches.append((overlap, dict(ocr_unit)))

        matches.sort(
            key=lambda item: (
                int(item[1].get("reading_order", 0) or 0),
                -float(item[0]),
            )
        )
        return [item[1] for item in matches]

    def _suspect_unit_indexes(self, native_units: list[dict[str, Any]], parser: Any) -> list[int]:
        indexes: list[int] = []
        for index, unit in enumerate(native_units):
            text = str(unit.get("text") or "").strip()
            if not text or len(text) < 5:
                continue
            if parser.is_text_garbled(text):
                indexes.append(index)
        return indexes

    @staticmethod
    def _substantial_unit_indexes(native_units: list[dict[str, Any]]) -> list[int]:
        indexes: list[int] = []
        for index, unit in enumerate(native_units):
            text = str(unit.get("text") or "").strip()
            if len(text) >= 5:
                indexes.append(index)
        return indexes

    def _is_garbled_table_candidate(self, native_units: list[dict[str, Any]], parser: Any) -> bool:
        if not native_units:
            return False

        has_table_block = any(str((unit or {}).get("block_type") or "") == "table_row" for unit in native_units)
        if has_table_block:
            return True

        garbled_count = 0
        pipe_count = 0
        for unit in native_units:
            text = str((unit or {}).get("text") or "")
            if parser.is_text_garbled(text):
                garbled_count += 1
            if "|" in text:
                pipe_count += 1

        return garbled_count >= 1 and pipe_count >= 1

    @staticmethod
    def _load_legacy_font_map(font_map_path: str | Path | None) -> dict[str, str]:
        if not font_map_path:
            return {}

        path = Path(font_map_path)
        if not path.exists():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        if not isinstance(data, dict):
            return {}
        return {str(key): str(value) for key, value in data.items() if str(key)}

    @staticmethod
    def _join_texts(texts: Any) -> str:
        parts = [" ".join(str(text or "").split()).strip() for text in texts if str(text or "").strip()]
        return " ".join(parts).strip()

    @staticmethod
    def _best_confidence(units: list[dict[str, Any]]) -> float | None:
        confidences = [float(unit.get("confidence")) for unit in units if unit.get("confidence") is not None]
        if not confidences:
            return None
        return max(confidences)

    @staticmethod
    def _split_text_to_slots(text: str, slots: int) -> list[str]:
        if slots <= 0:
            return []

        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        if not lines:
            return [str(text or "").strip()] + [""] * (slots - 1)

        if len(lines) <= slots:
            return lines + [""] * (slots - len(lines))

        base = len(lines) // slots
        extra = len(lines) % slots
        parts: list[str] = []
        index = 0
        for slot_index in range(slots):
            take = base + (1 if slot_index < extra else 0)
            parts.append("\n".join(lines[index : index + take]).strip())
            index += take
        return parts

    @staticmethod
    def _table_cells_for_text(text: str, existing_cells: Any) -> list[str]:
        if isinstance(existing_cells, list) and existing_cells:
            cells = [" ".join(str(cell or "").split()).strip() for cell in existing_cells]
            cells = [cell for cell in cells if cell]
            if cells:
                return cells

        compact = " ".join(str(text or "").split()).strip()
        if not compact:
            return []
        if "|" in compact:
            cells = [part.strip() for part in compact.split("|") if part.strip()]
            if cells:
                return cells
        return [compact]

    def _scripts_for_text(self, text: str) -> list[str]:
        mapping = {
            "devanagari": ["Deva"],
            "telugu": ["Telu"],
            "arabic": ["Arab"],
            "latin": ["Latn"],
            "tamil": ["Taml"],
            "bengali": ["Beng"],
        }
        script = self.normalizer.detect_script(text)
        return mapping.get(script, ["Zyyy"])

    def _languages_for_text(self, text: str) -> list[str]:
        scripts = self._scripts_for_text(text)
        if "Deva" in scripts:
            return ["hi", "sa"]
        if "Telu" in scripts:
            return ["te"]
        if "Arab" in scripts:
            return ["ur"]
        if "Latn" in scripts:
            return ["en"]
        if "Taml" in scripts:
            return ["ta"]
        if "Beng" in scripts:
            return ["bn"]
        return ["unknown"]

    def _bbox_to_tuple(self, bbox: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except Exception:
            return None

    @staticmethod
    def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
        x0, y0, x1, y1 = bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def _bbox_overlap_ratio(
        self,
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        x0 = max(left[0], right[0])
        y0 = max(left[1], right[1])
        x1 = min(left[2], right[2])
        y1 = min(left[3], right[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0

        intersection = (x1 - x0) * (y1 - y0)
        smaller_area = min(self._bbox_area(left), self._bbox_area(right))
        return intersection / max(1.0, smaller_area)

    @staticmethod
    def _bbox_centers_overlap(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> bool:
        left_center_x = (left[0] + left[2]) / 2.0
        left_center_y = (left[1] + left[3]) / 2.0
        right_center_x = (right[0] + right[2]) / 2.0
        right_center_y = (right[1] + right[3]) / 2.0

        left_contains_right = left[0] <= right_center_x <= left[2] and left[1] <= right_center_y <= left[3]
        right_contains_left = right[0] <= left_center_x <= right[2] and right[1] <= left_center_y <= right[3]
        return left_contains_right or right_contains_left
