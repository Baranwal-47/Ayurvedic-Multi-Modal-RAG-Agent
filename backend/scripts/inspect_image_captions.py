"""Run image extraction and show caption mapping per extracted figure."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.image_captioner import build_image_caption
from ingestion.image_extractor import ImageExtractor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect extracted images and generated captions")
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/images/_caption_inspect"),
        help="Directory where extracted figure images are written",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/images/caption_manifest.json"),
        help="Output JSON manifest path",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("data/images/caption_manifest.csv"),
        help="Output CSV manifest path",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=20,
        help="How many rows to print to terminal",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)

    rows = ImageExtractor().extract(args.pdf, args.output_dir)

    enriched: list[dict] = []
    for row in rows:
        entry = dict(row)
        entry["caption"] = build_image_caption(entry)
        enriched.append(entry)

    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "image_path",
        "page_number",
        "figure_index",
        "part_count",
        "figure_bbox",
        "source_file",
        "figure_caption",
        "nearest_heading",
        "surrounding_text",
        "caption",
    ]
    with args.csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in enriched:
            writer.writerow({k: item.get(k, "") for k in fieldnames})

    print(f"rows: {len(enriched)}")
    print(f"json: {args.json_out}")
    print(f"csv: {args.csv_out}")
    print("\nSample rows:")

    for i, item in enumerate(enriched[: max(0, args.print_limit)], start=1):
        print(f"[{i}] page={item.get('page_number')} figure={item.get('figure_index')} parts={item.get('part_count')}")
        print(f"    image: {item.get('image_path')}")
        print(f"    heading: {(item.get('nearest_heading') or '')[:120]}")
        print(f"    caption: {(item.get('caption') or '')[:220]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
