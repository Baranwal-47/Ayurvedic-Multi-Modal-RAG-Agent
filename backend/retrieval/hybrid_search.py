"""Hybrid retrieval over text and image collections for multimodal RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embeddings.text_embedder import TextEmbedder
from normalization.diacritic_normalizer import DiacriticNormalizer
from vector_db.qdrant_client import QdrantManager


@dataclass
class QueryFilters:
    """Safe user-facing query filters for retrieval."""

    doc_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    languages: list[str] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    chunk_types: list[str] = field(default_factory=list)
    source_file: str | None = None

    def as_qdrant(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.doc_id:
            payload["doc_id"] = self.doc_id
        if self.page_start is not None:
            payload["page_start"] = int(self.page_start)
        if self.page_end is not None:
            payload["page_end"] = int(self.page_end)
        if self.languages:
            payload["languages"] = list(self.languages)
        if self.scripts:
            payload["scripts"] = list(self.scripts)
        if self.chunk_types:
            payload["chunk_types"] = list(self.chunk_types)
        if self.source_file:
            payload["source_file"] = self.source_file
        return payload


@dataclass
class QueryBundle:
    """Normalized request envelope for retrieval and reranking."""

    query: str
    normalized_query: str
    intent: str
    filters: QueryFilters = field(default_factory=QueryFilters)
    include_debug: bool = False
    text_top_k: int = 32
    image_top_k: int = 4
    rerank_top_k: int = 24
    proximity_window: int = 1

    @property
    def is_visual(self) -> bool:
        return self.intent == "visual"

    @property
    def is_table(self) -> bool:
        return self.intent == "table"

    @property
    def is_shloka(self) -> bool:
        return self.intent == "shloka"


@dataclass
class RetrievalCandidate:
    """Unified retrieval candidate spanning text chunks and figures."""

    kind: str
    candidate_id: str
    doc_id: str
    source_file: str
    page_numbers: list[int]
    score: float
    snippet: str
    rerank_text: str
    section_path: list[str] = field(default_factory=list)
    heading_text: str | None = None
    chunk_type: str | None = None
    image_type: str | None = None
    image_url: str | None = None
    cloudinary_public_id: str | None = None
    caption: str | None = None
    labels: list[str] = field(default_factory=list)
    linked_ids: list[str] = field(default_factory=list)
    retrieval_reasons: list[str] = field(default_factory=list)
    table_markdown: str | None = None
    table_rows: list[list[str]] | None = None
    text: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_table(self) -> bool:
        return self.chunk_type == "table_text"

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    def merged_reason(self) -> str:
        return ", ".join(self.retrieval_reasons)


@dataclass
class HybridSearchResult:
    """Hybrid search output with optional debug stats."""

    query_bundle: QueryBundle
    candidates: list[RetrievalCandidate]
    debug: dict[str, Any] = field(default_factory=dict)


class HybridSearcher:
    """Retrieve and merge text, table, and image evidence from Qdrant."""

    VISUAL_KEYWORDS = {
        "diagram",
        "figure",
        "fig",
        "flowchart",
        "chart",
        "graph",
        "image",
        "illustration",
        "yantra",
        "show me",
        "picture",
        "photo",
    }
    TABLE_KEYWORDS = {
        "table",
        "tabular",
        "compare",
        "comparison",
        "rows",
        "columns",
        "values",
        "parameter",
        "specification",
        "dataset",
    }
    SHLOKA_KEYWORDS = {"shloka", "sloka", "verse", "sutra", "śloka", "śloka"}
    DEFINITION_KEYWORDS = {"what is", "define", "meaning of", "explain", "synonyms of"}

    def __init__(
        self,
        qdrant: QdrantManager,
        text_embedder: TextEmbedder,
        normalizer: DiacriticNormalizer | None = None,
    ) -> None:
        self.qdrant = qdrant
        self.text_embedder = text_embedder
        self.normalizer = normalizer or DiacriticNormalizer()

    def build_query_bundle(
        self,
        *,
        query: str,
        doc_id: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
        languages: list[str] | None = None,
        scripts: list[str] | None = None,
        chunk_types: list[str] | None = None,
        source_file: str | None = None,
        include_debug: bool = False,
    ) -> QueryBundle:
        cleaned_query = " ".join(str(query or "").split()).strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty")

        normalized_query = self.normalizer.normalize(cleaned_query, aggressive_latin_fold=False)
        intent = self._infer_intent(cleaned_query)
        image_top_k = 12 if intent == "visual" else 4

        return QueryBundle(
            query=cleaned_query,
            normalized_query=normalized_query,
            intent=intent,
            filters=QueryFilters(
                doc_id=doc_id,
                page_start=page_start,
                page_end=page_end,
                languages=list(languages or []),
                scripts=list(scripts or []),
                chunk_types=list(chunk_types or []),
                source_file=source_file,
            ),
            include_debug=bool(include_debug),
            image_top_k=image_top_k,
        )

    def search(self, query_bundle: QueryBundle) -> HybridSearchResult:
        candidates: dict[str, RetrievalCandidate] = {}
        debug: dict[str, Any] = {
            "intent": query_bundle.intent,
            "query": query_bundle.query,
            "normalized_query": query_bundle.normalized_query,
            "filters": query_bundle.filters.as_qdrant(),
        }

        original_vector = self.text_embedder.embed([query_bundle.query])[0]
        normalized_vector = None
        if query_bundle.normalized_query != query_bundle.query:
            normalized_vector = self.text_embedder.embed([query_bundle.normalized_query])[0]

        text_results_original = self.qdrant.hybrid_search_text(
            dense_vector=original_vector["dense_vector"],
            sparse_indices=original_vector["sparse_indices"],
            sparse_values=original_vector["sparse_values"],
            top_k=24,
            filters=query_bundle.filters.as_qdrant(),
        )
        self._merge_text_rows(candidates, text_results_original, reason="text_original")

        text_results_normalized: list[dict[str, Any]] = []
        if normalized_vector:
            text_results_normalized = self.qdrant.hybrid_search_text(
                dense_vector=normalized_vector["dense_vector"],
                sparse_indices=normalized_vector["sparse_indices"],
                sparse_values=normalized_vector["sparse_values"],
                top_k=24,
                filters=query_bundle.filters.as_qdrant(),
            )
            self._merge_text_rows(candidates, text_results_normalized, reason="text_normalized")

        image_results = self.qdrant.search_images(
            dense_vector=original_vector["dense_vector"],
            top_k=query_bundle.image_top_k,
            filters=query_bundle.filters.as_qdrant(),
            exclude_image_types=["decorative"],
        )
        self._merge_image_rows(candidates, image_results, reason="image_direct")

        sorted_text_candidates = [
            candidate
            for candidate in sorted(candidates.values(), key=lambda item: item.score, reverse=True)
            if candidate.kind == "text"
        ]
        self._rescue_linked_images(candidates, sorted_text_candidates[:8])
        self._rescue_page_proximity_images(query_bundle, candidates, sorted_text_candidates[:6])

        final_candidates = sorted(candidates.values(), key=lambda item: item.score, reverse=True)[: query_bundle.text_top_k]
        if query_bundle.include_debug:
            debug.update(
                {
                    "text_hits_original": len(text_results_original),
                    "text_hits_normalized": len(text_results_normalized),
                    "image_hits_direct": len(image_results),
                    "candidate_count_merged": len(candidates),
                    "candidate_ids": [candidate.candidate_id for candidate in final_candidates],
                }
            )
        return HybridSearchResult(query_bundle=query_bundle, candidates=final_candidates, debug=debug)

    def _merge_text_rows(
        self,
        merged: dict[str, RetrievalCandidate],
        rows: list[dict[str, Any]],
        *,
        reason: str,
    ) -> None:
        for row in rows:
            candidate = self._text_candidate_from_row(row, reason=reason)
            self._store_candidate(merged, candidate)

    def _merge_image_rows(
        self,
        merged: dict[str, RetrievalCandidate],
        rows: list[dict[str, Any]],
        *,
        reason: str,
        score_override: float | None = None,
    ) -> None:
        for row in rows:
            candidate = self._image_candidate_from_row(row, reason=reason, score_override=score_override)
            self._store_candidate(merged, candidate)

    def _rescue_linked_images(
        self,
        merged: dict[str, RetrievalCandidate],
        text_candidates: list[RetrievalCandidate],
    ) -> None:
        seen_image_ids: set[str] = set()
        for text_candidate in text_candidates:
            image_ids = [str(image_id) for image_id in text_candidate.payload.get("image_ids", []) if str(image_id).strip()]
            image_ids = [image_id for image_id in image_ids if image_id not in seen_image_ids]
            if not image_ids:
                continue
            seen_image_ids.update(image_ids)
            rescued_rows = self.qdrant.retrieve_points(collection="image_chunks", point_ids=image_ids)
            self._merge_image_rows(
                merged,
                rescued_rows,
                reason=f"linked_from:{text_candidate.candidate_id}",
                score_override=max(text_candidate.score * 0.92, 0.01),
            )

    def _rescue_page_proximity_images(
        self,
        query_bundle: QueryBundle,
        merged: dict[str, RetrievalCandidate],
        text_candidates: list[RetrievalCandidate],
    ) -> None:
        if not text_candidates:
            return

        seen_pages: set[tuple[str, int]] = set()
        for text_candidate in text_candidates:
            if text_candidate.payload.get("image_ids"):
                continue
            doc_id = text_candidate.doc_id
            for page_number in text_candidate.page_numbers:
                for offset in range(0, query_bundle.proximity_window + 1):
                    for direction in (-1, 1) if offset else (0,):
                        candidate_page = page_number + (offset * direction)
                        if candidate_page < 1:
                            continue
                        key = (doc_id, candidate_page)
                        if key in seen_pages:
                            continue
                        seen_pages.add(key)
                        rows = self.qdrant.scroll_points(
                            collection="image_chunks",
                            filters={"doc_id": doc_id, "page_start": candidate_page, "page_end": candidate_page},
                            limit=6,
                            exclude_image_types=["decorative"],
                        )
                        if not rows:
                            continue
                        reason = f"page_proximity:{text_candidate.candidate_id}:{candidate_page}"
                        penalty = 0.86 if candidate_page == page_number else 0.72
                        self._merge_image_rows(
                            merged,
                            rows,
                            reason=reason,
                            score_override=max(text_candidate.score * penalty, 0.01),
                        )

    def _store_candidate(self, merged: dict[str, RetrievalCandidate], candidate: RetrievalCandidate) -> None:
        existing = merged.get(candidate.candidate_id)
        if not existing:
            merged[candidate.candidate_id] = candidate
            return

        existing.score = max(existing.score, candidate.score)
        existing.retrieval_reasons = self._merge_reasons(existing.retrieval_reasons, candidate.retrieval_reasons)
        if not existing.image_url and candidate.image_url:
            existing.image_url = candidate.image_url
        if not existing.cloudinary_public_id and candidate.cloudinary_public_id:
            existing.cloudinary_public_id = candidate.cloudinary_public_id
        if not existing.table_markdown and candidate.table_markdown:
            existing.table_markdown = candidate.table_markdown
        if not existing.table_rows and candidate.table_rows:
            existing.table_rows = candidate.table_rows
        existing.linked_ids = self._merge_reasons(existing.linked_ids, candidate.linked_ids)

    def _text_candidate_from_row(self, row: dict[str, Any], *, reason: str) -> RetrievalCandidate:
        text = str(row.get("text") or "").strip()
        heading_text = str(row.get("heading_text") or "").strip() or None
        table_markdown = str(row.get("table_markdown") or "").strip() or None
        rerank_parts = [heading_text or "", text]
        if table_markdown:
            rerank_parts.append(table_markdown)
        page_numbers = [int(page) for page in row.get("page_numbers", []) if self._is_int_like(page)]
        if not page_numbers:
            page_start = int(row.get("page_start") or 0)
            page_end = int(row.get("page_end") or page_start)
            page_numbers = [page_start] if page_start == page_end else [page_start, page_end]

        return RetrievalCandidate(
            kind="text",
            candidate_id=str(row.get("chunk_id") or row.get("_id")),
            doc_id=str(row.get("doc_id") or ""),
            source_file=str(row.get("source_file") or ""),
            page_numbers=page_numbers,
            score=float(row.get("_score") or 0.0),
            snippet=self._snippet(text or table_markdown or heading_text or ""),
            rerank_text="\n\n".join(part for part in rerank_parts if part).strip(),
            section_path=[str(item) for item in row.get("section_path", []) if str(item).strip()],
            heading_text=heading_text,
            chunk_type=str(row.get("chunk_type") or ""),
            linked_ids=[str(item) for item in row.get("image_ids", []) if str(item).strip()],
            retrieval_reasons=[reason],
            table_markdown=table_markdown,
            table_rows=row.get("table_rows"),
            text=text,
            payload=dict(row),
        )

    def _image_candidate_from_row(
        self,
        row: dict[str, Any],
        *,
        reason: str,
        score_override: float | None = None,
    ) -> RetrievalCandidate:
        caption = str(row.get("caption") or "").strip()
        labels = [str(label) for label in row.get("labels", []) if str(label).strip()]
        surrounding_text = str(row.get("surrounding_text") or "").strip()
        rerank_parts = [caption, " ".join(labels), surrounding_text, " / ".join(row.get("section_path", []) or [])]
        page_number = int(row.get("page_number") or 0)
        return RetrievalCandidate(
            kind="image",
            candidate_id=str(row.get("image_id") or row.get("_id")),
            doc_id=str(row.get("doc_id") or ""),
            source_file=str(row.get("source_file") or ""),
            page_numbers=[page_number] if page_number else [],
            score=float(score_override if score_override is not None else row.get("_score") or 0.0),
            snippet=self._snippet(caption or surrounding_text or " ".join(labels)),
            rerank_text="\n\n".join(part for part in rerank_parts if part).strip(),
            section_path=[str(item) for item in row.get("section_path", []) if str(item).strip()],
            image_type=str(row.get("image_type") or ""),
            image_url=str(row.get("image_url") or "").strip() or None,
            cloudinary_public_id=str(row.get("cloudinary_public_id") or "").strip() or None,
            caption=caption or None,
            labels=labels,
            linked_ids=[str(item) for item in row.get("linked_chunk_ids", []) if str(item).strip()],
            retrieval_reasons=[reason],
            text=surrounding_text,
            payload=dict(row),
        )

    def _infer_intent(self, query: str) -> str:
        lowered = query.lower()
        if any(keyword in lowered for keyword in self.SHLOKA_KEYWORDS):
            return "shloka"
        if any(keyword in lowered for keyword in self.TABLE_KEYWORDS):
            return "table"
        if any(keyword in lowered for keyword in self.VISUAL_KEYWORDS):
            return "visual"
        if any(keyword in lowered for keyword in self.DEFINITION_KEYWORDS):
            return "definition"
        return "general"

    @staticmethod
    def _merge_reasons(left: list[str], right: list[str]) -> list[str]:
        merged: list[str] = []
        for value in [*left, *right]:
            if value and value not in merged:
                merged.append(value)
        return merged

    @staticmethod
    def _snippet(text: str, limit: int = 320) -> str:
        compact = " ".join(str(text or "").split()).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    @staticmethod
    def _is_int_like(value: object) -> bool:
        try:
            int(value)
            return True
        except Exception:
            return False
