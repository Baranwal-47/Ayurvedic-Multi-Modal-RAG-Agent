"""Validation checks for Chunker behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.chunker import Chunker


def _mk_block(
    text: str,
    block_type: str = "paragraph",
    page_number: int = 1,
    source_file: str = "test.pdf",
    heading_context: str = "",
) -> dict:
    return {
        "text": text,
        "block_type": block_type,
        "page_number": page_number,
        "source_file": source_file,
        "heading_context": heading_context,
    }


def _assert(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    if not ok:
        raise AssertionError(name)


def main() -> int:
    chunker = Chunker(max_tokens=400, overlap_tokens=50)
    enc = tiktoken.get_encoding("cl100k_base")

    # 1) Token size
    long_text = " ".join(["token"] * 800)
    long_chunks = chunker.chunk([_mk_block(long_text)])
    token_sizes = [len(enc.encode(c["original_text"])) for c in long_chunks]
    _assert("Token size <= 400", all(t <= 400 for t in token_sizes), f"sizes={token_sizes[:6]}")

    # 2) Overlap
    overlap_ok = True
    if len(long_chunks) >= 2:
        t1 = enc.encode(long_chunks[0]["original_text"])
        t2 = enc.encode(long_chunks[1]["original_text"])
        overlap_ok = t1[-50:] == t2[:50]
    _assert("Overlap 50 tokens", overlap_ok)

    # 3) Table atomic
    table_text = " ".join(["tabledata"] * 600)
    table_chunks = chunker.chunk([_mk_block(table_text, block_type="table")])
    _assert("Table atomic single chunk", len(table_chunks) == 1 and table_chunks[0]["block_type"] == "table")

    # 4) Figure caption atomic
    fig_text = " ".join(["figurecaption"] * 600)
    fig_chunks = chunker.chunk([_mk_block(fig_text, block_type="figure_caption")])
    _assert("Figure caption atomic single chunk", len(fig_chunks) == 1 and fig_chunks[0]["block_type"] == "figure_caption")

    # 5) Shloka extraction positive
    shloka_chunks = chunker.chunk([_mk_block("1.57 वातः पित्तं कफश्च")])
    _assert("Shloka extraction", shloka_chunks[0]["shloka_number"] == "1.57", str(shloka_chunks[0]["shloka_number"]))

    # 6) Shloka absent
    plain_chunks = chunker.chunk([_mk_block("This is a plain paragraph without marker.")])
    _assert("Shloka absent", plain_chunks[0]["shloka_number"] is None)

    # 7) Language detection
    dev_chunks = chunker.chunk([_mk_block("यह एक हिंदी वाक्य है")])
    eng_chunks = chunker.chunk([_mk_block("This is an English paragraph for detection.")])
    _assert("Language Hindi", dev_chunks[0]["language"] in {"hindi", "sanskrit", "devanagari"}, dev_chunks[0]["language"])
    _assert("Language English", eng_chunks[0]["language"] == "english", eng_chunks[0]["language"])

    # 8) Normalized vs original
    norm_chunks = chunker.chunk([_mk_block("ā is transliterated")])
    original = norm_chunks[0]["original_text"]
    normalized = norm_chunks[0]["normalized_text"]
    _assert("Original preserved", "ā" in original)
    _assert("Normalized diacritic", "a" in normalized and "ā" not in normalized, normalized)

    # 9) Empty input
    empty = chunker.chunk([])
    _assert("Empty input", empty == [])

    # 10) Empty text block skipped
    skipped = chunker.chunk([_mk_block(""), _mk_block("valid")])
    _assert("Empty text skipped", len(skipped) == 1 and skipped[0]["original_text"] == "valid")

    # 11) First block metadata wins when merged
    merged = chunker.chunk([
        _mk_block("INTRO", block_type="heading", page_number=3, heading_context="Intro"),
        _mk_block("small paragraph", block_type="paragraph", page_number=4, heading_context="Other"),
    ])
    _assert(
        "Merged first block page",
        len(merged) >= 1 and merged[0]["page_number"] == 3,
        f"page={merged[0]['page_number'] if merged else 'none'}",
    )

    print("All chunker checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
