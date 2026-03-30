"""Simple JSON-backed run-state tracking for restart-safe ingestion."""

from __future__ import annotations

import json
from pathlib import Path


class RunState:
    """Track completed and failed pages for a document ingestion run."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def load(self, doc_id: str) -> dict:
        path = self._path(doc_id)
        if not path.exists():
            return {"doc_id": doc_id, "completed_pages": [], "failed_pages": {}, "document_complete": False}
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, state: dict) -> None:
        self._path(state["doc_id"]).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def start_document(self, doc_id: str) -> dict:
        state = self.load(doc_id)
        state["document_complete"] = False
        self.save(state)
        return state

    def mark_page_completed(self, doc_id: str, page_number: int) -> dict:
        state = self.load(doc_id)
        completed = set(state.get("completed_pages", []))
        completed.add(int(page_number))
        state["completed_pages"] = sorted(completed)
        failed = dict(state.get("failed_pages", {}))
        failed.pop(str(page_number), None)
        state["failed_pages"] = failed
        state["document_complete"] = False
        self.save(state)
        return state

    def mark_page_failed(self, doc_id: str, page_number: int, reason: str) -> dict:
        state = self.load(doc_id)
        failed = dict(state.get("failed_pages", {}))
        failed[str(page_number)] = str(reason)
        state["failed_pages"] = failed
        completed = set(state.get("completed_pages", []))
        completed.discard(int(page_number))
        state["completed_pages"] = sorted(completed)
        state["document_complete"] = False
        self.save(state)
        return state

    def mark_document_complete(self, doc_id: str) -> dict:
        state = self.load(doc_id)
        state["document_complete"] = True
        state["failed_pages"] = {}
        self.save(state)
        return state

    def _path(self, doc_id: str) -> Path:
        return self.root_dir / f"{doc_id}.json"
