from __future__ import annotations

from pathlib import Path
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.chunker import Chunker
from ingestion.image_extractor import ImageExtractor
from ingestion.hybrid_page_repair import HybridPageRepair
from ingestion.docling_parser import plan_page_windows
from ingestion.native_pdf_parser import NativePDFParser
from ingestion.noise_detector import NoiseDetector
from ingestion.ocr_pipeline import OCRPipeline
from ingestion.page_classifier import PageClassifier
from ingestion.page_layout import PageLayout
from ingestion.page_model_builder import PageModelBuilder
from ingestion.section_detector import SectionDetector
from ingestion.shloka_detector import ShlokaDetector


def _sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Chapter 1")
        page.insert_text((72, 120), "This is a native paragraph that should remain on the parser path.")
        doc.save(pdf_path)
    return pdf_path


def test_native_parser_and_classifier(tmp_path: Path) -> None:
    pdf_path = _sample_pdf(tmp_path)
    parser = NativePDFParser()
    classifier = PageClassifier()

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(0)
        parsed = parser.parse_page(page_number=1, page=page, source_file=pdf_path.name)
        decision = classifier.classify_page(
            pdf_path=pdf_path,
            page_number=1,
            native_units=parsed.text_units,
            parser=parser,
        )

    assert parsed.text_units
    assert any(unit["block_type"] == "heading" for unit in parsed.text_units)
    assert decision.page_type == "digitized"


def test_page_classifier_routes_mojibake_text_to_ocr_fallback(tmp_path: Path) -> None:
    pdf_path = _sample_pdf(tmp_path)
    parser = NativePDFParser()
    classifier = PageClassifier()

    native_units = [
        {
            "text": "à¤¸à¤¾à¤®à¤¾à¤¨ à¤¦à¥‡à¤¹ à¤ªà¥à¤°à¤•à¥ƒà¤¤à¤¿ à¤•à¤¾ à¤µà¤¿à¤à¥ƒà¤¤ à¤°à¥‚à¤ª",
            "block_type": "paragraph",
        }
    ]

    decision = classifier.classify_page(
        pdf_path=pdf_path,
        page_number=1,
        native_units=native_units,
        parser=parser,
    )

    assert decision.page_type == "ocr_fallback"
    assert "mojibake" in decision.reason or "garbled" in decision.reason


def test_page_classifier_keeps_valid_devanagari_as_digitized(tmp_path: Path) -> None:
    pdf_path = _sample_pdf(tmp_path)
    parser = NativePDFParser()
    classifier = PageClassifier()

    native_units = [
        {
            "text": "वातः पित्तं कफश्च देहे तिष्ठन्ति सर्वदा ।",
            "block_type": "paragraph",
        }
    ]

    decision = classifier.classify_page(
        pdf_path=pdf_path,
        page_number=1,
        native_units=native_units,
        parser=parser,
    )

    assert decision.page_type == "digitized"


def test_page_classifier_keeps_index_like_pages_digitized(tmp_path: Path) -> None:
    pdf_path = _sample_pdf(tmp_path)
    parser = NativePDFParser()
    classifier = PageClassifier()

    native_units = [
        {"text": "19 MANDŪRA", "block_type": "table_row"},
        {"text": "653-656", "block_type": "paragraph"},
        {"text": "20 RASAYOGA", "block_type": "table_row"},
        {"text": "657-740", "block_type": "paragraph"},
        {"text": "21 LAUHA", "block_type": "table_row"},
        {"text": "741-756", "block_type": "paragraph"},
    ]

    decision = classifier.classify_page(
        pdf_path=pdf_path,
        page_number=5,
        native_units=native_units,
        parser=parser,
    )

    assert decision.page_type == "digitized"
    assert decision.reason == "index_like_native"


def test_page_model_pipeline_marks_shloka_and_layout() -> None:
    builder = PageModelBuilder()
    layout = PageLayout()
    shloka = ShlokaDetector()
    section = SectionDetector()

    page_model = builder.build(
        doc_id="doc1",
        source_file="sample.pdf",
        page_number=1,
        route="digitized",
        native_units=[
            {
                "unit_id": "u1",
                "text": "Verse 1",
                "block_type": "heading",
                "bbox": [0, 0, 100, 10],
                "reading_order": 0,
                "languages": ["en"],
                "scripts": ["Latn"],
                "source_engine": "pymupdf",
            },
            {
                "unit_id": "u2",
                "text": "1. वातः पित्तं कफश्च ।\nदेहे तिष्ठन्ति सर्वदा ॥",
                "block_type": "paragraph",
                "bbox": [0, 20, 100, 40],
                "reading_order": 1,
                "languages": ["sa"],
                "scripts": ["Deva"],
                "source_engine": "pymupdf",
            },
        ],
    )

    page_model = layout.apply(page_model)
    page_model = shloka.apply(page_model)
    page_model, path = section.apply(page_model)

    assert page_model["layout_type"] == "single"
    assert any(unit["kind"] == "shloka" for unit in page_model["text_units"])
    assert path


def test_shloka_detector_supports_telugu() -> None:
    detector = ShlokaDetector()
    page_model = {
        "text_units": [
            {
                "unit_id": "u_telu_1",
                "kind": "paragraph",
                "block_type": "paragraph",
                "text": "1. శ్రీరామ జయ రామ జయ జయ రామ ॥\nభక్తుల రక్షకుడవు శ్రీరామ ॥",
                "scripts": ["Telu"],
                "reading_order": 0,
            }
        ]
    }

    updated = detector.apply(page_model)
    assert updated["text_units"][0]["kind"] == "shloka"


def test_shloka_detector_rejects_latin_prose() -> None:
    detector = ShlokaDetector()
    page_model = {
        "text_units": [
            {
                "unit_id": "u_latn_1",
                "kind": "paragraph",
                "block_type": "paragraph",
                "text": "1. This is a numbered sentence.\nIt still reads like ordinary English prose.",
                "scripts": ["Latn"],
                "reading_order": 0,
            }
        ]
    }

    updated = detector.apply(page_model)
    assert updated["text_units"][0]["kind"] == "paragraph"


def test_noise_detector_strips_numbered_footer_patterns() -> None:
    detector = NoiseDetector()
    page_models = [
        {
            "text_units": [
                {"text": "1 www.ijaar.in IJAAR VOLUME 1 ISSUE 4 MAR-APR 2014", "bbox": [0, 760, 500, 780]},
                {"text": "Body text", "bbox": [0, 100, 500, 140]},
            ]
        },
        {
            "text_units": [
                {"text": "2 www.ijaar.in IJAAR VOLUME 1 ISSUE 4 MAR-APR 2014", "bbox": [0, 760, 500, 780]},
                {"text": "More body text", "bbox": [0, 100, 500, 140]},
            ]
        },
    ]

    marked = detector.mark_document_noise(page_models)

    assert marked[0]["text_units"][0]["kind"] == "noise"
    assert marked[1]["text_units"][0]["block_type"] == "noise"


def test_chunker_outputs_new_contract() -> None:
    page_models = [
        {
            "doc_id": "doc1",
            "source_file": "sample.pdf",
            "page_number": 1,
            "route": "digitized",
            "layout_type": "single",
            "section_path": ["Chapter 1"],
            "quality": {"ocr_confidence": None},
            "images": [],
            "text_units": [
                {
                    "unit_id": "u1",
                    "kind": "heading",
                    "text": "Chapter 1",
                    "section_path": ["Chapter 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 0,
                },
                {
                    "unit_id": "u2",
                    "kind": "paragraph",
                    "text": " ".join(["body"] * 220),
                    "section_path": ["Chapter 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 1,
                },
            ],
        }
    ]

    chunks = Chunker().chunk_document(page_models)

    assert chunks
    assert "chunk_id" in chunks[0]
    assert "page_numbers" in chunks[0]
    assert chunks[0]["doc_id"] == "doc1"


def test_chunker_marks_table_text_chunks() -> None:
    page_models = [
        {
            "doc_id": "doc1",
            "source_file": "sample.pdf",
            "page_number": 2,
            "route": "digitized",
            "layout_type": "single",
            "section_path": ["Table 1"],
            "quality": {"ocr_confidence": None},
            "images": [],
            "text_units": [
                {
                    "unit_id": "u1",
                    "kind": "paragraph",
                    "text": "Temperature 32 116 120 124",
                    "section_path": ["Table 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 0,
                },
                {
                    "unit_id": "u2",
                    "kind": "paragraph",
                    "text": "% of Free Hg Nil Nil Nil",
                    "section_path": ["Table 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 1,
                },
            ],
        }
    ]

    chunks = Chunker().chunk_document(page_models)

    assert chunks[0]["chunk_type"] == "table_text"
    assert chunks[0]["table_id"] is not None
    assert chunks[0]["table_rows"] is not None
    assert len(chunks[0]["table_rows"]) >= 2
    assert chunks[0]["table_markdown"] is not None


def test_chunker_preserves_structured_table_cells() -> None:
    page_models = [
        {
            "doc_id": "doc1",
            "source_file": "sample.pdf",
            "page_number": 2,
            "route": "digitized",
            "layout_type": "single",
            "section_path": ["Table 1"],
            "quality": {"ocr_confidence": None},
            "images": [],
            "text_units": [
                {
                    "unit_id": "u1",
                    "kind": "table_row",
                    "text": "Table No. 1",
                    "table_cells": ["Table No. 1"],
                    "section_path": ["Table 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 0,
                },
                {
                    "unit_id": "u2",
                    "kind": "table_row",
                    "text": "Parameters | Start",
                    "table_cells": ["Parameters", "Start"],
                    "section_path": ["Table 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 1,
                },
                {
                    "unit_id": "u3",
                    "kind": "table_row",
                    "text": "Consistency | Fine",
                    "table_cells": ["Consistency", "Fine"],
                    "section_path": ["Table 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 2,
                },
            ],
        }
    ]

    chunks = Chunker(target_words=400, hard_cap_words=400).chunk_document(page_models)

    assert chunks
    table_chunk = chunks[0]
    assert table_chunk["chunk_type"] == "table_text"
    assert table_chunk["table_caption"] == "Table No. 1"
    assert table_chunk["table_rows"] == [["Table No. 1"], ["Parameters", "Start"], ["Consistency", "Fine"]]
    assert "| Parameters | Start |" in str(table_chunk["table_markdown"])


def test_page_model_builder_promotes_rows_after_table_anchor() -> None:
    builder = PageModelBuilder()
    page_model = builder.build(
        doc_id="doc1",
        source_file="sample.pdf",
        page_number=2,
        route="digitized",
        native_units=[
            {
                "unit_id": "u1",
                "text": "OBSERVATIONS: Table No. 1 Showing various parameters",
                "block_type": "paragraph",
                "bbox": [0, 0, 100, 10],
                "reading_order": 0,
                "languages": ["en"],
                "scripts": ["Latn"],
                "source_engine": "pymupdf",
            },
            {
                "unit_id": "u2",
                "text": "Reddish powder",
                "block_type": "paragraph",
                "bbox": [0, 12, 100, 22],
                "reading_order": 1,
                "languages": ["en"],
                "scripts": ["Latn"],
                "source_engine": "pymupdf",
            },
            {
                "unit_id": "u3",
                "text": "Consistency Fine, smooth powder",
                "block_type": "paragraph",
                "bbox": [0, 24, 100, 34],
                "reading_order": 2,
                "languages": ["en"],
                "scripts": ["Latn"],
                "source_engine": "pymupdf",
            },
            {
                "unit_id": "u4",
                "text": "Temperature 32 116 120 124",
                "block_type": "paragraph",
                "bbox": [0, 36, 100, 46],
                "reading_order": 3,
                "languages": ["en"],
                "scripts": ["Latn"],
                "source_engine": "pymupdf",
            },
            {
                "unit_id": "u5",
                "text": "Breakable - Not easily Easily Turns into powder",
                "block_type": "paragraph",
                "bbox": [0, 48, 100, 58],
                "reading_order": 4,
                "languages": ["en"],
                "scripts": ["Latn"],
                "source_engine": "pymupdf",
            },
        ],
    )

    promoted = [unit for unit in page_model["text_units"] if unit["kind"] == "table_row"]
    assert len(promoted) >= 3


def test_chunker_splits_at_table_row_boundary() -> None:
    page_models = [
        {
            "doc_id": "doc1",
            "source_file": "sample.pdf",
            "page_number": 2,
            "route": "digitized",
            "layout_type": "single",
            "section_path": ["Table 1"],
            "quality": {"ocr_confidence": None},
            "images": [],
            "text_units": [
                {
                    "unit_id": "u1",
                    "kind": "paragraph",
                    "text": "Observations for the experiment are summarized below",
                    "section_path": ["Table 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 0,
                },
                {
                    "unit_id": "u2",
                    "kind": "table_row",
                    "text": "Temperature 32 116 120 124",
                    "section_path": ["Table 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 1,
                },
                {
                    "unit_id": "u3",
                    "kind": "table_row",
                    "text": "Total loss 4 5 7",
                    "section_path": ["Table 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 2,
                },
                {
                    "unit_id": "u4",
                    "kind": "paragraph",
                    "text": "Discussion continues after the table",
                    "section_path": ["Table 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "reading_order": 3,
                },
            ],
        }
    ]

    chunks = Chunker(target_words=400, hard_cap_words=400).chunk_document(page_models)

    table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "table_text"]
    assert table_chunks
    assert all("Observations for the experiment" not in chunk["text"] for chunk in table_chunks)


def test_image_extractor_drops_table_like_candidate_on_digitized_page() -> None:
    keep = ImageExtractor._should_keep_image_candidate(
        is_full=False,
        area_ratio=0.22,
        has_caption=True,
        page_has_meaningful_text=True,
        is_table_like=True,
        is_scanned_page=False,
        text_overlap_ratio=0.10,
        figure_rect=fitz.Rect(60, 120, 420, 520),
        page_width=595,
        page_height=842,
        resolved_caption="Table No. 1",
        surrounding_text="Tabular values",
    )

    assert keep is False


def test_image_extractor_does_not_mark_figure_caption_as_table() -> None:
    is_table = ImageExtractor.looks_like_table(
        text_blocks=[],
        image_bbox=fitz.Rect(120, 80, 460, 220),
        resolved_caption="Fig. 1. Phases of Namburi Phased Spot test",
        surrounding_text="Fig. 1 description. Table 3 follows below.",
    )

    assert is_table is False


def test_image_extractor_drops_tiny_uncaptioned_asset() -> None:
    keep = ImageExtractor._should_keep_image_candidate(
        is_full=False,
        area_ratio=0.0017,
        has_caption=False,
        page_has_meaningful_text=True,
        is_table_like=False,
        is_scanned_page=False,
        text_overlap_ratio=0.00,
        figure_rect=fitz.Rect(500.9, 148.7, 529.4, 176.8),
        page_width=595,
        page_height=842,
        resolved_caption="",
        surrounding_text="Journal of Ayurveda and Integrative Medicine",
    )

    assert keep is False


def test_image_extractor_document_blocks_preserve_bbox() -> None:
    page_rect = fitz.Rect(0, 0, 1000, 2000)
    blocks = [
        {
            "text": "Right column paragraph",
            "block_type": "paragraph",
            "bbox": [620, 220, 980, 380],
            "column_id": 1,
            "reading_order": 3,
        }
    ]

    converted = ImageExtractor._convert_document_blocks_to_text_blocks(blocks, page_rect)

    assert converted
    rect = converted[0]["rect"]
    assert abs(rect.x0 - 620.0) < 1e-6
    assert abs(rect.x1 - 980.0) < 1e-6
    assert converted[0].get("column_id") == 1


def test_image_extractor_surrounding_text_prefers_prose_context() -> None:
    image_rect = fitz.Rect(1840, 417, 2333, 1031)
    blocks = [
        {"text": "Rag and mud wrap", "rect": fitz.Rect(1856, 393, 2006, 472)},
        {"text": "Iron rod", "rect": fitz.Rect(2135, 446, 2247, 470)},
        {
            "text": "An earthen vessel is filled with the sand at the bottom upto 8\"-9\" thickness. Then the bottle filled with mercurial is placed on it and surrounded by the sand on all sides.",
            "rect": fitz.Rect(1836, 220, 3049, 716),
        },
        {
            "text": "After drying, the whole apparatus is put on the fire and subjected for slow heating.",
            "rect": fitz.Rect(2377, 713, 3053, 1028),
        },
    ]

    text = ImageExtractor._get_surrounding_text(blocks, image_rect)

    assert "8\"-9\" thickness" in text or "whole apparatus" in text


def test_image_extractor_surrounding_text_allows_longer_context() -> None:
    image_rect = fitz.Rect(1840, 417, 2333, 1031)
    blocks = [
        {
            "text": "An earthen vessel is filled with the sand at the bottom upto 8\"-9\" thickness. Then the bottle filled with mercurial is placed on it and surrounded by the sand on all sides.",
            "rect": fitz.Rect(1836, 220, 3049, 716),
        },
        {
            "text": "After drying, the whole apparatus is put on the fire and subjected for slow heating. To moniter the amount of heat, a grass stick is kept on the top of the saucer, which is not supposed to get burnt.",
            "rect": fitz.Rect(2377, 713, 3053, 1028),
        },
        {
            "text": "When insted of sand the apparatus is filled with the salt, it is known as a 'Lavana yantra'. (Rasaratasamuchchaya 9:36-39)",
            "rect": fitz.Rect(1842, 1032, 3053, 1219),
        },
    ]

    text = ImageExtractor._get_surrounding_text(blocks, image_rect)

    assert "Lavana yantra" in text


def test_image_extractor_keeps_table_like_candidate_on_scanned_page() -> None:
    keep = ImageExtractor._should_keep_image_candidate(
        is_full=False,
        area_ratio=0.22,
        has_caption=False,
        page_has_meaningful_text=False,
        is_table_like=True,
        is_scanned_page=True,
        text_overlap_ratio=0.05,
        figure_rect=fitz.Rect(60, 120, 420, 520),
        page_width=595,
        page_height=842,
        resolved_caption="",
        surrounding_text="",
    )

    assert keep is True


def test_hybrid_page_repair_replaces_only_suspect_units() -> None:
    parser = NativePDFParser()
    repair = HybridPageRepair()
    garbled_text = "@@@ abc ###"

    native_units = [
        {
            "unit_id": "u1",
            "text": "Chapter 1",
            "block_type": "heading",
            "bbox": [0, 0, 100, 10],
            "reading_order": 0,
            "source_engine": "pymupdf",
            "languages": ["en"],
            "scripts": ["Latn"],
        },
        {
            "unit_id": "u2",
            "text": garbled_text,
            "block_type": "paragraph",
            "bbox": [0, 20, 100, 40],
            "reading_order": 1,
            "source_engine": "pymupdf",
            "languages": ["en"],
            "scripts": ["Latn"],
        },
    ]
    ocr_result = {
        "text": "Chapter 1\nRecovered shloka text",
        "confidence": 0.91,
        "text_units": [
            {
                "unit_id": "ocr1",
                "text": "Recovered shloka text",
                "bbox": [0, 20, 100, 40],
                "reading_order": 0,
                "source_engine": "vision",
                "confidence": 0.91,
            }
        ],
    }

    assert repair.should_use_hybrid_repair(native_units, parser) is True

    result = repair.repair_units(
        native_units=native_units,
        ocr_result=ocr_result,
        parser=parser,
        page_number=1,
        source_file="sample.pdf",
    )

    assert result.used_ocr is True
    assert result.repaired_unit_indexes == [1]
    assert result.legacy_repaired_unit_indexes == []
    assert result.text_units[0]["text"] == "Chapter 1"
    assert result.text_units[1]["text"] == "Recovered shloka text"
    assert result.text_units[1]["source_engine"] == "vision"
    assert result.text_units[1]["scripts"] == ["Latn"]


def test_hybrid_page_repair_uses_legacy_font_map_before_ocr() -> None:
    parser = NativePDFParser()
    garbled_text = "@@@ abc ###"
    repair = HybridPageRepair(legacy_font_map={garbled_text: "वातः"})

    native_units = [
        {
            "unit_id": "u1",
            "text": garbled_text,
            "block_type": "paragraph",
            "bbox": [0, 0, 100, 10],
            "reading_order": 0,
            "source_engine": "pymupdf",
        }
    ]

    result = repair.repair_units(
        native_units=native_units,
        ocr_result={"text": "", "confidence": 0.0, "text_units": []},
        parser=parser,
        page_number=1,
        source_file="sample.pdf",
    )

    assert result.used_ocr is False
    assert result.legacy_repaired_unit_indexes == [0]
    assert result.repaired_unit_indexes == []
    assert result.text_units[0]["text"] == "वातः"
    assert result.text_units[0]["source_engine"] == "legacy_font_map"
    assert result.text_units[0]["scripts"] == ["Deva"]


def test_chunker_marks_hybrid_ocr_source_only_when_ocr_used() -> None:
    chunker = Chunker(target_words=400, hard_cap_words=400)
    page_models = [
        {
            "doc_id": "doc1",
            "source_file": "sample.pdf",
            "page_number": 1,
            "route": "digitized",
            "layout_type": "single",
            "section_path": [],
            "quality": {"ocr_confidence": 0.91, "hybrid_repair_used_ocr": True},
            "images": [],
            "text_units": [
                {
                    "unit_id": "u1",
                    "kind": "heading",
                    "text": "Chapter 1",
                    "reading_order": 0,
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "source_engine": "pymupdf",
                },
                {
                    "unit_id": "u2",
                    "kind": "paragraph",
                    "text": "Recovered text",
                    "reading_order": 1,
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "source_engine": "vision",
                },
            ],
        }
    ]

    chunks = chunker.chunk_document(page_models)
    assert chunks
    assert chunks[0]["ocr_source"] == "vision"

    page_models[0]["quality"]["hybrid_repair_used_ocr"] = False
    page_models[0]["text_units"][1]["source_engine"] = "legacy_font_map"
    chunks = chunker.chunk_document(page_models)
    assert chunks
    assert chunks[0]["ocr_source"] is None


def test_page_model_builder_preserves_table_cells() -> None:
    builder = PageModelBuilder()
    page_model = builder.build(
        doc_id="doc1",
        source_file="sample.pdf",
        page_number=1,
        route="digitized",
        native_units=[
            {
                "unit_id": "u1",
                "text": "Table No. 1",
                "block_type": "table_row",
                "bbox": [0, 0, 100, 10],
                "reading_order": 0,
                "source_engine": "docling",
                "table_cells": ["Table No. 1"],
                "table_id": "doc1:p1:table:1",
                "table_row_index": 0,
            }
        ],
    )

    unit = page_model["text_units"][0]
    assert unit["table_cells"] == ["Table No. 1"]
    assert unit["table_id"] == "doc1:p1:table:1"
    assert unit["table_row_index"] == 0


def test_plan_page_windows_groups_contiguous_ranges() -> None:
    assert plan_page_windows([1, 2, 3, 6, 7, 10], batch_size=2) == [(1, 2), (3, 3), (6, 7), (10, 10)]


def test_ocr_line_merger_merges_short_lines() -> None:
    line_units = [
        {
            "unit_id": "doc:p1:ocr-line:1",
            "text": "Indo - Romanic Equivalents",
            "bbox": [100.0, 100.0, 420.0, 116.0],
            "reading_order": 0,
            "page_number": 1,
            "source_file": "sample.pdf",
            "source_engine": "vision",
            "languages": ["unknown"],
            "scripts": ["Zyyy"],
            "confidence": 0.90,
        },
        {
            "unit_id": "doc:p1:ocr-line:2",
            "text": "of Devanagari Alphabets",
            "bbox": [102.0, 118.0, 418.0, 134.0],
            "reading_order": 1,
            "page_number": 1,
            "source_file": "sample.pdf",
            "source_engine": "vision",
            "languages": ["unknown"],
            "scripts": ["Zyyy"],
            "confidence": 0.92,
        },
    ]

    merged = OCRPipeline.merge_line_units(line_units)

    assert len(merged) == 1
    assert "Indo - Romanic Equivalents" in merged[0]["text"]
    assert "of Devanagari Alphabets" in merged[0]["text"]
    assert merged[0]["merged_line_count"] == 2


def test_ocr_line_merger_does_not_merge_large_vertical_gaps() -> None:
    line_units = [
        {
            "unit_id": "doc:p1:ocr-line:1",
            "text": "Short heading",
            "bbox": [80.0, 80.0, 260.0, 96.0],
            "reading_order": 0,
            "page_number": 1,
            "source_file": "sample.pdf",
            "source_engine": "vision",
        },
        {
            "unit_id": "doc:p1:ocr-line:2",
            "text": "A long paragraph line that starts after a clear section break.",
            "bbox": [82.0, 180.0, 520.0, 198.0],
            "reading_order": 1,
            "page_number": 1,
            "source_file": "sample.pdf",
            "source_engine": "vision",
        },
    ]

    merged = OCRPipeline.merge_line_units(line_units)

    assert len(merged) == 2
