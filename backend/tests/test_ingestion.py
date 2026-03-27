"""
tests/test_ingestion.py

Tests for all Week 1 ingestion files.
Run from backend/ with:
    python tests/test_ingestion.py

Does NOT require a real PDF for most tests.
For OCR and Docling tests, drop any PDF into backend/data/pdfs/ and
set TEST_PDF_PATH below, or pass as env var:
    TEST_PDF_PATH=data/pdfs/my.pdf python tests/test_ingestion.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Start OCR warmup early so model load overlaps with non-OCR test sections.
from ingestion.ocr_pipeline import warmup_ocr, wait_for_ocr_ready
from ingestion.ocr_routing import apply_ocr_routing_to_document

print("Before warmup")
warmup_ocr()
print("After warmup")
print("[test_ingestion] OCR warming up in background...")

# TEST_PDF_PATH = Path(os.getenv(
#     "TEST_PDF_PATH",
#     ROOT / "data" / "pdfs" / "Official Formularies & Pharmacopoea - PDF Format" / "AFI - PDF Format" /  "AFI PART - I" / "AFI-PART-I_PART_A_FORMULATIONS.pdf"
# ))
TEST_PDF_PATH = Path(os.getenv(
    "TEST_PDF_PATH",
    ROOT / "data" / "pdfs" / "93aa6-145-rasa-shastra.pdf"
))

TEST_OUTPUT_ROOT = ROOT / "data" / "images" / "test_ingestion_test"
# Default to production parity (Docling-first parser path) unless explicitly overridden.
TEST_USE_DOCLING = os.getenv("TEST_USE_DOCLING", "1").strip() in {"1", "true", "True"}


def get_effective_test_pdf_path(pdf_path: Path) -> Path:
    """Use a subset PDF for known heavy targets to keep runtime manageable."""
    if pdf_path.name.lower() == "93aa6-145-rasa-shastra.pdf":
        if not pdf_path.exists():
            return pdf_path

        tmp_dir = TEST_OUTPUT_ROOT / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        subset_pdf = tmp_dir / f"{pdf_path.stem}_p20_p40.pdf"

        try:
            with fitz.open(pdf_path) as src:
                total_pages = src.page_count
                start_page = 20
                end_page = 40

                if total_pages < start_page:
                    print(
                        f"[test_ingestion] WARN: {pdf_path.name} has only {total_pages} pages; "
                        f"cannot create 20-40 subset. Using full PDF"
                    )
                    return pdf_path

                if subset_pdf.exists():
                    try:
                        with fitz.open(subset_pdf) as cached:
                            expected_pages = end_page - start_page + 1
                            if cached.page_count == expected_pages:
                                print(f"[test_ingestion] Using cached pages 20-40 subset: {subset_pdf}")
                                return subset_pdf
                    except Exception:
                        pass

                with fitz.open() as dst:
                    dst.insert_pdf(src, from_page=start_page - 1, to_page=min(end_page - 1, total_pages - 1))
                    dst.save(subset_pdf)

                print(
                    f"[test_ingestion] Limited {pdf_path.name} to pages 20-40 "
                    f"({total_pages} pages total): {subset_pdf}"
                )
                return subset_pdf
        except Exception as exc:
            print(f"[test_ingestion] WARN: could not create 20-40 subset ({exc}); using full PDF")
            return pdf_path

    if pdf_path.name.lower() != "slokas vol 1.pdf":
        return pdf_path

    if not pdf_path.exists():
        return pdf_path

    tmp_dir = TEST_OUTPUT_ROOT / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subset_pdf = tmp_dir / pdf_path.name

    try:
        with fitz.open(pdf_path) as src:
            total_pages = src.page_count
            if total_pages <= 15:
                return pdf_path

            if subset_pdf.exists():
                try:
                    with fitz.open(subset_pdf) as cached:
                        if cached.page_count == 15:
                            print(f"[test_ingestion] Using cached 15-page subset: {subset_pdf}")
                            return subset_pdf
                except Exception:
                    pass

            with fitz.open() as dst:
                dst.insert_pdf(src, from_page=0, to_page=14)
                dst.save(subset_pdf)

            print(
                f"[test_ingestion] Limited Slokas Vol 1.pdf to first 15 pages "
                f"({total_pages} -> 15): {subset_pdf}"
            )
            return subset_pdf
    except Exception as exc:
        print(f"[test_ingestion] WARN: could not create 15-page subset ({exc}); using full PDF")
        return pdf_path


def get_output_paths(pdf_path: Path) -> tuple[Path, Path, Path]:
    """Return (pdf_root_dir, images_dir, json_output_path) for deterministic test artifacts."""
    pdf_name = pdf_path.stem
    pdf_root = TEST_OUTPUT_ROOT / pdf_name
    images_dir = pdf_root / "images"
    json_out = pdf_root / f"ingestion_{pdf_name}_output.json"
    return pdf_root, images_dir, json_out


passed = 0
failed = 0

ACTIVE_TEST_PDF_PATH = get_effective_test_pdf_path(TEST_PDF_PATH)

# Caches — populated in sections 4 and 6, reused in section 7
_cached_parsed_blocks: list | None = None
_cached_image_rows: list | None = None


def ok(label: str):
    global passed
    passed += 1
    print(f"  PASS  {label}")


def fail(label: str, err: Exception):
    global failed
    failed += 1
    print(f"  FAIL  {label} — {err}")


# ─────────────────────────────────────────────
# 1. DiacriticNormalizer
# ─────────────────────────────────────────────
print("\n[1] DiacriticNormalizer")

try:
    from normalization.diacritic_normalizer import DiacriticNormalizer
    n = DiacriticNormalizer()
    ok("import")
except Exception as e:
    fail("import", e)
    n = None

if n:
    try:
        assert n.normalize("") == ""
        ok("normalize empty string")
    except Exception as e:
        fail("normalize empty string", e)

    try:
        result = n.normalize("Āyurveda")
        assert result == "Ayurveda", f"got: {result}"
        ok("normalize ā → a")
    except Exception as e:
        fail("normalize ā → a", e)

    try:
        result = n.normalize("pittaṃ")
        assert result == "pittam", f"got: {result}"
        ok("normalize ṃ → m")
    except Exception as e:
        fail("normalize ṃ → m", e)

    try:
        devanagari = "पित्तं"
        result = n.normalize(devanagari)
        assert result == devanagari, f"Devanagari was modified: {result}"
        ok("Devanagari passes through unchanged")
    except Exception as e:
        fail("Devanagari passes through unchanged", e)

    try:
        assert n.detect_script("पित्तं") == "devanagari"
        ok("detect_script devanagari")
    except Exception as e:
        fail("detect_script devanagari", e)

    try:
        assert n.detect_script("hello world") == "latin"
        ok("detect_script latin")
    except Exception as e:
        fail("detect_script latin", e)

    try:
        assert n.detect_script("") == "latin"
        ok("detect_script empty → latin")
    except Exception as e:
        fail("detect_script empty → latin", e)


# ─────────────────────────────────────────────
# 2. Chunker
# ─────────────────────────────────────────────
print("\n[2] Chunker")

try:
    from ingestion.chunker import Chunker
    chunker = Chunker()
    ok("import")
except Exception as e:
    fail("import", e)
    chunker = None

if chunker:
    try:
        result = chunker.chunk([])
        assert result == []
        ok("chunk empty list → []")
    except Exception as e:
        fail("chunk empty list → []", e)

    try:
        blocks = [
            {
                "text": "Vata is the dosha of movement.",
                "block_type": "paragraph",
                "page_number": 1,
                "source_file": "test.pdf",
                "heading_context": "",
            }
        ]
        chunks = chunker.chunk(blocks)
        assert len(chunks) == 1
        c = chunks[0]
        assert "original_text" in c
        assert "normalized_text" in c
        assert "language" in c
        assert "page_number" in c
        assert "source_file" in c
        assert c["source_file"] == "test.pdf"
        ok("chunk schema keys present")
    except Exception as e:
        fail("chunk schema keys present", e)

    try:
        blocks = [
            {
                "text": "पित्तं दाहकर्मणि प्रमुखम्",
                "block_type": "paragraph",
                "page_number": 2,
                "source_file": "test.pdf",
                "heading_context": "",
            }
        ]
        chunks = chunker.chunk(blocks)
        assert len(chunks) >= 1
        lang = chunks[0]["language"]
        assert lang in ("sanskrit", "hindi", "devanagari"), f"unexpected lang: {lang}"
        assert "पित्तं" in chunks[0]["original_text"]
        ok(f"Devanagari chunk — language={lang}, original preserved")
    except Exception as e:
        fail("Devanagari chunk", e)

    try:
        table_block = {
            "text": "| Step | Action |\n|------|--------|\n| 1 | Do this |",
            "block_type": "table",
            "page_number": 3,
            "source_file": "test.pdf",
            "heading_context": "Procedure",
        }
        chunks = chunker.chunk([table_block])
        assert len(chunks) == 1, f"table split into {len(chunks)} chunks"
        ok("table block kept atomic")
    except Exception as e:
        fail("table block kept atomic", e)

    try:
        long_text = "Ayurveda is the science of life. " * 50
        blocks = [
            {
                "text": long_text,
                "block_type": "paragraph",
                "page_number": 4,
                "source_file": "test.pdf",
                "heading_context": "",
            }
        ]
        chunks = chunker.chunk(blocks)
        assert len(chunks) > 1, "long block should split into multiple chunks"
        ok(f"long block splits into {len(chunks)} chunks with overlap")
    except Exception as e:
        fail("long block splits", e)

    try:
        blocks = [
            {
                "text": "1.5 pittaṃ dahakarmani pramukham",
                "block_type": "paragraph",
                "page_number": 5,
                "source_file": "test.pdf",
                "heading_context": "",
            }
        ]
        chunks = chunker.chunk(blocks)
        assert chunks[0]["shloka_number"] == "1.5", f"got: {chunks[0]['shloka_number']}"
        ok("shloka number extracted from text")
    except Exception as e:
        fail("shloka number extraction", e)


# ─────────────────────────────────────────────
# 3. ImageCaptioner
# ─────────────────────────────────────────────
print("\n[3] ImageCaptioner")

try:
    from ingestion.image_captioner import build_image_caption
    ok("import")
except Exception as e:
    fail("import", e)
    build_image_caption = None

if build_image_caption:
    try:
        result = build_image_caption({})
        assert "Diagram from page" in result or result == "Diagram from page unknown"
        ok("empty metadata → fallback string")
    except Exception as e:
        fail("empty metadata → fallback string", e)

    try:
        result = build_image_caption({"figure_caption": "Fig 1. Marma points diagram"})
        assert result == "Fig 1. Marma points diagram"
        ok("figure_caption takes priority")
    except Exception as e:
        fail("figure_caption priority", e)

    try:
        result = build_image_caption({
            "figure_caption": "",
            "nearest_heading": "Panchakarma Procedures",
            "surrounding_text": "This diagram shows the five cleansing procedures",
        })
        assert result == "Panchakarma Procedures"
        ok("nearest_heading used when no figure_caption")
    except Exception as e:
        fail("nearest_heading fallback", e)

    try:
        result = build_image_caption({
            "figure_caption": "",
            "nearest_heading": "",
            "surrounding_text": "This diagram shows the five cleansing procedures in detail",
            "page_number": 45,
        })
        assert "cleansing" in result
        ok("surrounding_text fallback")
    except Exception as e:
        fail("surrounding_text fallback", e)

    try:
        result = build_image_caption({"page_number": 12})
        assert result == "Diagram from page 12"
        ok("page number fallback")
    except Exception as e:
        fail("page number fallback", e)


# ─────────────────────────────────────────────
# 4. DoclingParser — requires TEST_PDF_PATH
# ─────────────────────────────────────────────
print("\n[4] DoclingParser")

try:
    from ingestion.docling_parser import DoclingParser
    parser = DoclingParser()
    ok("import")
except Exception as e:
    fail("import", e)
    parser = None

if parser:
    try:
        assert parser.is_page_scanned([]) is True
        ok("is_page_scanned: empty blocks → True")
    except Exception as e:
        fail("is_page_scanned: empty blocks", e)

    try:
        blocks = [{"text": "x" * 50}]
        assert parser.is_page_scanned(blocks) is False
        ok("is_page_scanned: normal text → False")
    except Exception as e:
        fail("is_page_scanned: normal text", e)

    try:
        blocks = [{"text": "ab"}]
        assert parser.is_page_scanned(blocks) is True
        ok("is_page_scanned: tiny text → True")
    except Exception as e:
        fail("is_page_scanned: tiny text", e)

    try:
        assert parser.is_text_non_latin("नमस्ते पित्तं वाता") is True
        ok("is_text_non_latin: Devanagari text → True")
    except Exception as e:
        fail("is_text_non_latin: Devanagari text", e)

    try:
        assert parser.is_text_non_latin("This is clean English text") is False
        ok("is_text_non_latin: English text → False")
    except Exception as e:
        fail("is_text_non_latin: English text", e)

    try:
        blocks = [{"text": "अ आ इ ई उ ऊ ऋ"}]
        assert parser.is_page_non_latin(blocks) is True
        ok("is_page_non_latin: Indic page → True")
    except Exception as e:
        fail("is_page_non_latin: Indic page", e)

    try:
        blocks = [{"text": "This page is fully English language content."}]
        assert parser.is_page_non_latin(blocks) is False
        ok("is_page_non_latin: English page → False")
    except Exception as e:
        fail("is_page_non_latin: English page", e)

    try:
        sample = "¶¤¦µ¿¡¢£¬÷×ØÐÞþðøåæçèéêëìíîïñòóôõöùúûüýÿ" * 3
        assert parser.is_text_garbled(sample) is True
        ok("is_text_garbled: mojibake sample → True")
    except Exception as e:
        fail("is_text_garbled: mojibake sample", e)

    try:
        sample = "The quick brown fox jumps over the lazy dog repeatedly."
        assert parser.is_text_garbled(sample) is False
        ok("is_text_garbled: clean English sample → False")
    except Exception as e:
        fail("is_text_garbled: clean English sample", e)

    if ACTIVE_TEST_PDF_PATH.exists():
        try:
            if TEST_USE_DOCLING:
                blocks = parser.parse(ACTIVE_TEST_PDF_PATH)
                backend = "docling"
            else:
                # Default to PyMuPDF in tests to avoid long model initialization/download waits.
                blocks = parser._parse_with_pymupdf(ACTIVE_TEST_PDF_PATH)
                backend = "pymupdf"

            assert isinstance(blocks, list), "parse must return a list"
            assert len(blocks) > 0, "got zero blocks — check _extract_docling_text"
            for b in blocks[:3]:
                assert "text" in b
                assert "block_type" in b
                assert "page_number" in b
                assert "source_file" in b
                assert b["source_file"] == TEST_PDF_PATH.name
            ok(f"parse real PDF ({backend}) → {len(blocks)} blocks, schema OK")
            # Cache for section 7 — no re-parsing needed
            _cached_parsed_blocks = blocks
        except Exception as e:
            fail("parse real PDF", e)
    else:
        print(f"  SKIP  parse real PDF — PDF not found at: {ACTIVE_TEST_PDF_PATH}")


# ─────────────────────────────────────────────
# 5. OCRPipeline — requires TEST_PDF_PATH
# ─────────────────────────────────────────────
print("\n[5] OCRPipeline")

try:
    from ingestion.ocr_pipeline import OCRPipeline
    print("Before wait")
    wait_for_ocr_ready()
    print("After wait")
    ocr = OCRPipeline()
    ok("import + model ready")
except Exception as e:
    fail("import", e)
    ocr = None

if ocr and ACTIVE_TEST_PDF_PATH.exists():
    try:
        result = ocr.process_page(ACTIVE_TEST_PDF_PATH, 1)
        assert "text" in result
        assert "confidence" in result
        assert "engine_used" in result
        assert "page_number" in result
        assert result["page_number"] == 1
        assert result["engine_used"] in ("paddleocr", "tesseract", "indic-ocr")
        assert isinstance(result["confidence"], float)
        ok(f"process_page 1 → engine={result['engine_used']} conf={result['confidence']:.2f} text_len={len(result['text'])}")
    except Exception as e:
        fail("process_page 1", e)
elif ocr:
    print(f"  SKIP  OCR process_page — PDF not found at: {TEST_PDF_PATH}")


# ─────────────────────────────────────────────
# 6. ImageExtractor — requires TEST_PDF_PATH
# ─────────────────────────────────────────────
print("\n[6] ImageExtractor")

try:
    from ingestion.image_extractor import ImageExtractor
    extractor = ImageExtractor()
    ok("import")
except Exception as e:
    fail("import", e)
    extractor = None

if extractor and ACTIVE_TEST_PDF_PATH.exists():
    try:
        pdf_root, images_dir, _ = get_output_paths(TEST_PDF_PATH)
        images_dir.mkdir(parents=True, exist_ok=True)

        images = extractor.extract(ACTIVE_TEST_PDF_PATH, images_dir)
        assert isinstance(images, list)
        for img in images:
            assert "image_path" in img
            assert "page_number" in img
            assert "caption" not in img  # captioner is separate
        ok(f"extract real PDF → {len(images)} images found")

        if images:
            img = images[0]
            assert "source_file" in img
            assert "surrounding_text" in img
            assert "nearest_heading" in img
            ok(f"image metadata schema OK — page={img['page_number']}")

        # Cache for section 7 — no re-extracting needed
        _cached_image_rows = images

    except Exception as e:
        fail("extract real PDF", e)
elif extractor:
    print(f"  SKIP  ImageExtractor extract — PDF not found at: {TEST_PDF_PATH}")


# ─────────────────────────────────────────────
# 7. End-to-end artifact (uses cached results — no re-processing)
# ─────────────────────────────────────────────
print("\n[7] Ingestion Test Artifact")

_can_run_e2e = (
    ACTIVE_TEST_PDF_PATH.exists()
    and parser is not None
    and ocr is not None
    and chunker is not None
    and extractor is not None
    and build_image_caption is not None
    and _cached_parsed_blocks is not None
    and _cached_image_rows is not None
)

if _can_run_e2e:
    try:
        pdf_root, images_dir, json_out = get_output_paths(TEST_PDF_PATH)
        images_dir.mkdir(parents=True, exist_ok=True)

        # Reuse cached parse output, then apply the same OCR routing as production.
        parsed_blocks = _cached_parsed_blocks
        parsed_blocks, ocr_stats = apply_ocr_routing_to_document(
            pdf_path=ACTIVE_TEST_PDF_PATH,
            base_blocks=parsed_blocks,
            parser=parser,
            ocr=ocr,
            force_ocr_all_pages=False,
        )
        ok(
            "OCR routing applied via shared module "
            f"(scanned={len(ocr_stats['scanned_pages'])}, garbled={len(ocr_stats['garbled_pages'])})"
        )

        # Re-extract with scanned-page awareness to mirror production image filtering.
        image_rows = extractor.extract(
            ACTIVE_TEST_PDF_PATH,
            images_dir,
            scanned_pages=set(ocr_stats.get("scanned_pages") or []),
        )

        # Build image caption blocks
        image_caption_blocks = []
        for row in image_rows:
            caption = build_image_caption(row)
            if not str(caption).strip():
                continue
            image_caption_blocks.append(
                {
                    "text": caption,
                    "block_type": "figure_caption",
                    "page_number": int(row.get("page_number") or 1),
                    "source_file": TEST_PDF_PATH.name,
                    "heading_context": str(row.get("nearest_heading") or ""),
                }
            )

        all_blocks = [*parsed_blocks, *image_caption_blocks]
        all_blocks.sort(key=lambda b: int(b.get("page_number") or 1))
        chunks = chunker.chunk(all_blocks)

        payload = {
            "summary": {
                "pdf": str(TEST_PDF_PATH),
                "source_file": TEST_PDF_PATH.name,
                "parser_blocks": len(parsed_blocks),
                "image_caption_blocks": len(image_caption_blocks),
                "chunks_created": len(chunks),
                "images_extracted": len(image_rows),
                "source_file_all_chunks_ok": all(
                    c.get("source_file") == TEST_PDF_PATH.name for c in chunks
                ),
            },
            "chunks": chunks,
            "images": image_rows,
        }

        # Save output JSON for manual inspection
        try:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            with json_out.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            ok(f"output JSON saved → {json_out}")
        except Exception as e:
            print(f"  WARN  could not save JSON artifact (non-blocking) — {e}")

        ok(f"end-to-end complete — {len(chunks)} chunks, {len(image_rows)} images")
        ok(f"images dir → {images_dir}")

        # Quick sanity checks on the final output
        assert len(chunks) > 0, "chunker produced zero chunks"
        ok("chunks > 0")

        assert payload["summary"]["source_file_all_chunks_ok"], \
            "some chunks have wrong source_file"
        ok("all chunks have correct source_file")

        languages_seen = {c.get("language") for c in chunks if c.get("language")}
        ok(f"languages detected in chunks: {sorted(languages_seen)}")

    except Exception as e:
        fail("end-to-end artifact", e)
elif not ACTIVE_TEST_PDF_PATH.exists():
    print(f"  SKIP  end-to-end — PDF not found at: {ACTIVE_TEST_PDF_PATH}")
else:
    missing = []
    if _cached_parsed_blocks is None:
        missing.append("DoclingParser (section 4 failed or skipped)")
    if _cached_image_rows is None:
        missing.append("ImageExtractor (section 6 failed or skipped)")
    print(f"  SKIP  end-to-end — fix failing sections first: {', '.join(missing)}")


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 50)

if failed == 0:
    print("\nAll ingestion tests passed.")
    if ACTIVE_TEST_PDF_PATH.exists():
        _, images_dir, json_out = get_output_paths(TEST_PDF_PATH)
        print("\nOutput files to inspect:")
        print(f"  chunks + metadata  →  {json_out}")
        print(f"  extracted images   →  {images_dir}")
    print("\nNext: drop more Ayurvedic PDFs into data/pdfs/ and run:")
    print("  TEST_PDF_PATH=data/pdfs/yourfile.pdf python tests/test_ingestion.py")
else:
    print("\nFix failing tests before running ingest_documents.py")