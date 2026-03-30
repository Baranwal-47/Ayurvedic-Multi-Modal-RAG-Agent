from __future__ import annotations

from pathlib import Path
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.chunker import Chunker
from ingestion.native_pdf_parser import NativePDFParser
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
                "text": "1.57 वातः पित्तं कफश्च",
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
