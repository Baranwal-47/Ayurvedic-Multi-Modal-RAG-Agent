"""Run ingest_documents.py sequentially in separate processes, one PDF at a time."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client.models import FieldCondition, Filter, MatchValue
from vector_db.qdrant_client import QdrantManager


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}")


def _discover_pdfs(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    return sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])


def _load_pdf_list(pdf_list_file: Path) -> list[Path]:
    if not pdf_list_file.exists() or not pdf_list_file.is_file():
        raise FileNotFoundError(f"PDF list file not found: {pdf_list_file}")

    lines = pdf_list_file.read_text(encoding="utf-8").splitlines()
    out: list[Path] = []
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        out.append(Path(raw))

    if not out:
        raise ValueError(f"No PDF paths found in list file: {pdf_list_file}")
    return out


def _count_source_points(qdrant: QdrantManager, collection: str, source_file: str) -> int:
    f = Filter(
        must=[
            FieldCondition(
                key="source_file",
                match=MatchValue(value=source_file),
            )
        ]
    )
    return int(qdrant.client.count(collection_name=collection, count_filter=f).count)


def _already_ingested(qdrant: QdrantManager, source_file: str) -> tuple[bool, int, int]:
    text_count = _count_source_points(qdrant, qdrant.text_collection, source_file)
    image_count = _count_source_points(qdrant, qdrant.image_collection, source_file)
    return (text_count > 0 and image_count > 0), text_count, image_count


def _load_checkpoint(checkpoint_file: Path) -> set[str]:
    if not checkpoint_file.exists():
        return set()

    rows = checkpoint_file.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in rows if line.strip() and not line.strip().startswith("#")}


def _append_checkpoint(checkpoint_file: Path, pdf_path: Path) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_file.open("a", encoding="utf-8") as f:
        f.write(f"{pdf_path}\n")
        f.flush()
        os.fsync(f.fileno())


def _apply_safe_batch_defaults(env: dict[str, str]) -> None:
    """Apply conservative large-PDF defaults unless caller already provided values."""
    env.setdefault("DOCLING_PAGE_BATCH_SIZE", "4")
    env.setdefault("DOCLING_AUTO_BATCH_SIZE", "4")
    env.setdefault("OCR_PAGE_BATCH_SIZE", "4")
    env.setdefault("OCR_AUTO_BATCH_SIZE", "4")
    env.setdefault("OCR_BATCH_COOLDOWN_SEC", "0.15")
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ingestion one PDF at a time using separate child processes"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
        help="Root directory to discover PDFs recursively",
    )
    parser.add_argument(
        "--pdf-list",
        type=Path,
        default=None,
        help="Optional text file with PDF paths in strict execution order (one path per line)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many PDFs to process from the discovered list",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index in the sorted PDF list (useful for resuming)",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable to use for child ingestion runs",
    )
    parser.add_argument(
        "--ingest-script",
        type=Path,
        default=Path("scripts/ingest_documents.py"),
        help="Path to ingest_documents.py",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop batch at first failure",
    )
    parser.add_argument(
        "--resume-skip-existing",
        action="store_true",
        help="Skip PDFs that already have points in both Qdrant collections",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path("data/images/_ingest_success_checkpoint.txt"),
        help="File storing successfully ingested PDFs for crash-safe resume",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Skip PDFs already recorded in checkpoint file",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.perf_counter()

    backend_root = Path(__file__).resolve().parents[1]
    ingest_script = (backend_root / args.ingest_script).resolve()
    pdf_dir = (backend_root / args.pdf_dir).resolve()
    default_pdf_list = (backend_root / Path("scripts/pdf_run_order_example.txt")).resolve()
    pdf_list_file = (backend_root / args.pdf_list).resolve() if args.pdf_list else None
    if pdf_list_file is None and default_pdf_list.exists():
        pdf_list_file = default_pdf_list
    checkpoint_file = (backend_root / args.checkpoint_file).resolve()

    if not ingest_script.exists():
        raise FileNotFoundError(f"ingest script not found: {ingest_script}")

    if pdf_list_file is not None:
        all_pdfs = _load_pdf_list(pdf_list_file)
        all_pdfs = [p if p.is_absolute() else (backend_root / p).resolve() for p in all_pdfs]
    else:
        all_pdfs = _discover_pdfs(pdf_dir)

    all_pdfs = [p.resolve() for p in all_pdfs]
    if not all_pdfs:
        raise FileNotFoundError("No PDFs found for the selected input mode")

    missing = [p for p in all_pdfs if not p.exists()]
    if missing:
        preview = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"Some listed PDFs do not exist:\n{preview}")

    start = max(0, int(args.offset))
    end = start + max(0, int(args.limit))
    selected = all_pdfs[start:end]

    if not selected:
        raise ValueError(
            f"Selection is empty (offset={args.offset}, limit={args.limit}, total={len(all_pdfs)})"
        )

    input_mode = f"pdf-list ({pdf_list_file})" if pdf_list_file else f"pdf-dir ({pdf_dir})"
    _log(f"Discovered PDFs from {input_mode}: total={len(all_pdfs)}")
    _log(f"Selected window: start={start}, end={end - 1}, count={len(selected)}")

    done_from_checkpoint = _load_checkpoint(checkpoint_file) if args.resume_from_checkpoint else set()
    if args.resume_from_checkpoint:
        _log(f"Checkpoint resume enabled: {checkpoint_file}")
        _log(f"Checkpoint entries loaded: {len(done_from_checkpoint)}")

    success: list[Path] = []
    skipped: list[Path] = []
    failed: list[tuple[Path, int]] = []

    qdrant = QdrantManager() if args.resume_skip_existing else None

    for i, pdf in enumerate(selected, start=1):
        pdf_key = str(pdf)

        if args.resume_from_checkpoint and pdf_key in done_from_checkpoint:
            skipped.append(pdf)
            _log(f"[{i}/{len(selected)}] SKIP {pdf.name} (already in checkpoint)")
            continue

        if qdrant is not None:
            done, text_count, image_count = _already_ingested(qdrant, pdf.name)
            if done:
                skipped.append(pdf)
                _log(
                    f"[{i}/{len(selected)}] SKIP {pdf.name} "
                    f"(already in qdrant: text={text_count}, image={image_count})"
                )
                continue

        _log(f"[{i}/{len(selected)}] START {pdf}")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        _apply_safe_batch_defaults(env)

        cmd = [
            args.python_exe,
            "-u",
            str(ingest_script),
            "--pdf",
            str(pdf),
        ]

        _log(f"[{i}/{len(selected)}] CMD {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=backend_root, env=env).returncode

        if rc == 0:
            success.append(pdf)
            if args.resume_from_checkpoint:
                _append_checkpoint(checkpoint_file, pdf)
                done_from_checkpoint.add(pdf_key)
            _log(f"[{i}/{len(selected)}] OK {pdf.name}")
        else:
            failed.append((pdf, rc))
            _log(f"[{i}/{len(selected)}] FAIL {pdf.name} (exit_code={rc})")
            if args.stop_on_failure:
                _log("Stopping due to --stop-on-failure")
                break

    print("\n=== Sequential Batch Summary ===")
    print(
        f"selected={len(selected)} success={len(success)} skipped={len(skipped)} failed={len(failed)}"
    )

    if skipped:
        print("Skipped files:")
        for p in skipped:
            print(f"- {p}")

    if failed:
        print("Failed files:")
        for p, rc in failed:
            print(f"- {p} (exit_code={rc})")

    elapsed = time.perf_counter() - t0
    print(f"Total elapsed: {elapsed:.1f}s")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
