"""Hybrid retrieval over text and image collections for multimodal RAG."""

from __future__ import annotations

import math
import os
import time
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
    route: str = "deep"
    query_token_count: int = 0
    skip_sparse: bool = False
    filters: QueryFilters = field(default_factory=QueryFilters)
    include_debug: bool = False
    text_top_k: int = 12
    image_top_k: int = 4
    rerank_top_k: int = 8
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
    CHITCHAT_KEYWORDS = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "thanks",
        "thank you",
    }
    DEEP_FORCE_KEYWORDS = {"diagram", "table", "image", "yantra"}

    def __init__(
        self,
        qdrant: QdrantManager,
        text_embedder: TextEmbedder,
        normalizer: DiacriticNormalizer | None = None,
    ) -> None:
        self.qdrant = qdrant
        self.text_embedder = text_embedder
        self.normalizer = normalizer or DiacriticNormalizer()
        self._query_vector_cache: dict[str, dict[str, Any]] = {}
        self._query_vector_cache_order: list[str] = []
        self._query_vector_cache_size = self._env_int("QUERY_EMBED_CACHE_SIZE", default=128)

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
        token_count = len(cleaned_query.split())
        deep_force_keywords = set(self._env_csv("ROUTE_DEEP_FORCE_KEYWORDS", default=list(self.DEEP_FORCE_KEYWORDS)))
        route = self._classify_route(
            query=cleaned_query,
            intent=intent,
            token_count=token_count,
            deep_force_keywords=deep_force_keywords,
        )

        image_top_k = 0
        if route == "deep":
            image_top_k = 6 if intent == "visual" else 4

        sparse_skip_threshold = self._env_int("SHORT_QUERY_SPARSE_SKIP_LT_TOKENS", default=4)
        skip_sparse = route == "fast" or token_count < sparse_skip_threshold

        text_top_k = self._env_int("TEXT_CANDIDATE_TOP_K", default=12)
        rerank_top_k = self._env_int("RERANK_TOP_K", default=8) if route == "deep" else 0

        return QueryBundle(
            query=cleaned_query,
            normalized_query=normalized_query,
            intent=intent,
            route=route,
            query_token_count=token_count,
            skip_sparse=skip_sparse,
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
            text_top_k=text_top_k,
            image_top_k=image_top_k,
            rerank_top_k=rerank_top_k,
        )

    def search(self, query_bundle: QueryBundle) -> HybridSearchResult:
        candidates: dict[str, RetrievalCandidate] = {}
        debug: dict[str, Any] = {
            "intent": query_bundle.intent,
            "route": query_bundle.route,
            "query": query_bundle.query,
            "normalized_query": query_bundle.normalized_query,
            "query_token_count": query_bundle.query_token_count,
            "skip_sparse": query_bundle.skip_sparse,
            "filters": query_bundle.filters.as_qdrant(),
        }
        retrieval_timing: dict[str, float] = {
            "embed_query_sec": 0.0,
            "text_original_sec": 0.0,
            "text_normalized_sec": 0.0,
            "image_direct_sec": 0.0,
            "linked_image_rescue_sec": 0.0,
            "page_proximity_rescue_sec": 0.0,
            "hydrate_points_sec": 0.0,
            "merge_sort_sec": 0.0,
        }
        retrieval_counts: dict[str, int] = {
            "text_hits_original": 0,
            "text_hits_normalized": 0,
            "image_hits_direct": 0,
            "rescued_image_hits": 0,
            "rescued_page_hits": 0,
            "candidate_count_merged": 0,
        }
        effective_rerank_target = max(1, int(query_bundle.rerank_top_k or 0))
        text_search_k = min(max(effective_rerank_target, max(1, query_bundle.text_top_k)), 12)
        prefilter_top_k = min(max(text_search_k, max(1, query_bundle.rerank_top_k or 0)), 12)
        include_vectors_for_prefilter = query_bundle.route == "deep"
        text_search_mode = "dense_only" if query_bundle.skip_sparse else "hybrid"

        embed_started = time.perf_counter()
        original_vector = self._embed_query_text(query_bundle.query)
        setattr(query_bundle, "_query_dense_vector", list(original_vector.get("dense_vector", []) or []))
        normalized_vector = None
        normalized_search_ran = self._materially_changes_query(query_bundle.query, query_bundle.normalized_query)
        if normalized_search_ran:
            normalized_vector = self._embed_query_text(query_bundle.normalized_query)
        retrieval_timing["embed_query_sec"] = time.perf_counter() - embed_started

        text_original_started = time.perf_counter()
        if query_bundle.skip_sparse:
            text_results_original = self.qdrant.search_text_dense(
                dense_vector=original_vector["dense_vector"],
                top_k=text_search_k,
                filters=query_bundle.filters.as_qdrant(),
                include_vectors=include_vectors_for_prefilter,
            )
        else:
            text_results_original = self.qdrant.hybrid_search_text(
                dense_vector=original_vector["dense_vector"],
                sparse_indices=original_vector["sparse_indices"],
                sparse_values=original_vector["sparse_values"],
                top_k=text_search_k,
                filters=query_bundle.filters.as_qdrant(),
                include_vectors=include_vectors_for_prefilter,
            )
        retrieval_timing["text_original_sec"] = time.perf_counter() - text_original_started
        retrieval_counts["text_hits_original"] = len(text_results_original)
        self._merge_text_rows(candidates, text_results_original, reason="text_original")

        text_results_normalized: list[dict[str, Any]] = []
        if normalized_vector:
            text_normalized_started = time.perf_counter()
            if query_bundle.skip_sparse:
                text_results_normalized = self.qdrant.search_text_dense(
                    dense_vector=normalized_vector["dense_vector"],
                    top_k=text_search_k,
                    filters=query_bundle.filters.as_qdrant(),
                    include_vectors=include_vectors_for_prefilter,
                )
            else:
                text_results_normalized = self.qdrant.hybrid_search_text(
                    dense_vector=normalized_vector["dense_vector"],
                    sparse_indices=normalized_vector["sparse_indices"],
                    sparse_values=normalized_vector["sparse_values"],
                    top_k=text_search_k,
                    filters=query_bundle.filters.as_qdrant(),
                    include_vectors=include_vectors_for_prefilter,
                )
            retrieval_timing["text_normalized_sec"] = time.perf_counter() - text_normalized_started
            self._merge_text_rows(candidates, text_results_normalized, reason="text_normalized")
        retrieval_counts["text_hits_normalized"] = len(text_results_normalized)

        image_results: list[dict[str, Any]] = []
        image_direct_started = time.perf_counter()
        if query_bundle.image_top_k > 0:
            image_results = self.qdrant.search_images(
                dense_vector=original_vector["dense_vector"],
                top_k=query_bundle.image_top_k,
                filters=query_bundle.filters.as_qdrant(),
                exclude_image_types=["decorative"],
                include_vectors=include_vectors_for_prefilter,
            )
        retrieval_timing["image_direct_sec"] = time.perf_counter() - image_direct_started
        retrieval_counts["image_hits_direct"] = len(image_results)
        self._merge_image_rows(candidates, image_results, reason="image_direct")

        merge_sort_started = time.perf_counter()
        sorted_candidates = sorted(candidates.values(), key=lambda item: (-item.score, item.candidate_id))
        sorted_text_candidates = [candidate for candidate in sorted_candidates if candidate.kind == "text"]
        retrieval_timing["merge_sort_sec"] += time.perf_counter() - merge_sort_started

        linked_rescue_started = time.perf_counter()
        rescued_image_hits, hydrate_points_sec = self._rescue_linked_images(
            candidates,
            sorted_text_candidates[:2],
            max_rescued_images=3,
            include_vectors=include_vectors_for_prefilter,
        )
        retrieval_timing["linked_image_rescue_sec"] = time.perf_counter() - linked_rescue_started
        retrieval_timing["hydrate_points_sec"] = hydrate_points_sec
        retrieval_counts["rescued_image_hits"] = rescued_image_hits

        rescued_page_hits = 0
        if len(image_results) < 2:
            page_rescue_started = time.perf_counter()
            rescued_page_hits = self._rescue_page_proximity_images(
                query_bundle,
                candidates,
                sorted_text_candidates[:2],
                max_neighborhoods=2,
                include_vectors=include_vectors_for_prefilter,
            )
            retrieval_timing["page_proximity_rescue_sec"] = time.perf_counter() - page_rescue_started
        retrieval_counts["rescued_page_hits"] = rescued_page_hits

        merge_sort_started = time.perf_counter()
        merged_sorted = sorted(candidates.values(), key=lambda item: (-item.score, item.candidate_id))
        candidate_count_before_prefilter = len(merged_sorted)
        prefiltered_candidates = self._prefilter_candidates_by_query_similarity(
            query_dense_vector=original_vector.get("dense_vector", []),
            candidates=merged_sorted,
            top_k=prefilter_top_k,
        )
        candidate_count_after_prefilter = len(prefiltered_candidates)
        final_candidates = prefiltered_candidates[: query_bundle.text_top_k]
        retrieval_timing["merge_sort_sec"] += time.perf_counter() - merge_sort_started
        retrieval_counts["candidate_count_merged"] = len(candidates)

        if query_bundle.include_debug:
            debug.update(
                {
                    "retrieval_timing": {key: round(value, 4) for key, value in retrieval_timing.items()},
                    "retrieval_counts": retrieval_counts,
                    "text_hits_original": retrieval_counts["text_hits_original"],
                    "text_hits_normalized": retrieval_counts["text_hits_normalized"],
                    "image_hits_direct": retrieval_counts["image_hits_direct"],
                    "text_search_k": text_search_k,
                    "text_search_mode": text_search_mode,
                    "candidate_count_merged": retrieval_counts["candidate_count_merged"],
                    "candidate_count_before_prefilter": candidate_count_before_prefilter,
                    "candidate_count_after_prefilter": candidate_count_after_prefilter,
                    "normalized_text_search_ran": normalized_search_ran,
                    "candidate_ids": [candidate.candidate_id for candidate in final_candidates],
                }
            )
        return HybridSearchResult(query_bundle=query_bundle, candidates=final_candidates, debug=debug)

    def prewarm(self) -> None:
        self._embed_query_text("warmup")

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
        *,
        max_rescued_images: int = 3,
        include_vectors: bool = False,
    ) -> tuple[int, float]:
        if not text_candidates or max_rescued_images <= 0:
            return 0, 0.0

        existing_candidate_ids = set(merged.keys())
        rescue_sources: dict[str, RetrievalCandidate] = {}
        image_ids_to_hydrate: list[str] = []

        for text_candidate in text_candidates:
            for image_id in [str(value) for value in text_candidate.payload.get("image_ids", []) if str(value).strip()]:
                if image_id in rescue_sources:
                    continue
                if image_id in existing_candidate_ids:
                    continue
                rescue_sources[image_id] = text_candidate
                image_ids_to_hydrate.append(image_id)
                if len(image_ids_to_hydrate) >= max_rescued_images:
                    break
            if len(image_ids_to_hydrate) >= max_rescued_images:
                break

        if not image_ids_to_hydrate:
            return 0, 0.0

        hydrate_started = time.perf_counter()
        rescued_rows = self.qdrant.retrieve_points(
            collection="image_chunks",
            point_ids=image_ids_to_hydrate,
            include_vectors=include_vectors,
        )
        hydrate_points_sec = time.perf_counter() - hydrate_started

        rescued_count = 0
        for row in rescued_rows:
            image_id = self._image_identifier_from_row(row)
            if not image_id or image_id in existing_candidate_ids:
                continue
            source = rescue_sources.get(image_id)
            reason = f"linked_from:{source.candidate_id}" if source else "linked_image_rescue"
            score_override = max(source.score * 0.92, 0.01) if source else None
            self._merge_image_rows(
                merged,
                [row],
                reason=reason,
                score_override=score_override,
            )
            existing_candidate_ids.add(image_id)
            rescued_count += 1
            if rescued_count >= max_rescued_images:
                break

        return rescued_count, hydrate_points_sec

    def _rescue_page_proximity_images(
        self,
        query_bundle: QueryBundle,
        merged: dict[str, RetrievalCandidate],
        text_candidates: list[RetrievalCandidate],
        *,
        max_neighborhoods: int = 2,
        include_vectors: bool = False,
    ) -> int:
        if not text_candidates or max_neighborhoods <= 0:
            return 0

        seen_pages: set[tuple[str, int]] = set()
        existing_candidate_ids = set(merged.keys())
        neighborhoods_scanned = 0
        rescued_count = 0
        for text_candidate in text_candidates:
            if neighborhoods_scanned >= max_neighborhoods:
                break
            if text_candidate.payload.get("image_ids"):
                continue
            doc_id = text_candidate.doc_id
            for page_number in text_candidate.page_numbers:
                if neighborhoods_scanned >= max_neighborhoods:
                    break
                for offset in range(0, query_bundle.proximity_window + 1):
                    if neighborhoods_scanned >= max_neighborhoods:
                        break
                    for direction in (-1, 1) if offset else (0,):
                        if neighborhoods_scanned >= max_neighborhoods:
                            break
                        candidate_page = page_number + (offset * direction)
                        if candidate_page < 1:
                            continue
                        key = (doc_id, candidate_page)
                        if key in seen_pages:
                            continue
                        seen_pages.add(key)
                        neighborhoods_scanned += 1
                        rows = self.qdrant.scroll_points(
                            collection="image_chunks",
                            filters={"doc_id": doc_id, "page_start": candidate_page, "page_end": candidate_page},
                            limit=6,
                            exclude_image_types=["decorative"],
                            include_vectors=include_vectors,
                        )
                        if not rows:
                            continue
                        deduped_rows = [
                            row
                            for row in rows
                            if self._image_identifier_from_row(row)
                            and self._image_identifier_from_row(row) not in existing_candidate_ids
                        ]
                        if not deduped_rows:
                            continue
                        reason = f"page_proximity:{text_candidate.candidate_id}:{candidate_page}"
                        penalty = 0.86 if candidate_page == page_number else 0.72
                        self._merge_image_rows(
                            merged,
                            deduped_rows,
                            reason=reason,
                            score_override=max(text_candidate.score * penalty, 0.01),
                        )
                        for row in deduped_rows:
                            image_id = self._image_identifier_from_row(row)
                            if image_id:
                                existing_candidate_ids.add(image_id)
                        rescued_count += len(deduped_rows)

        return rescued_count

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
        if self._is_chitchat_query(lowered):
            return "chitchat"
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

    @staticmethod
    def _materially_changes_query(original_query: str, normalized_query: str) -> bool:
        original = " ".join(str(original_query or "").split()).strip().casefold()
        normalized = " ".join(str(normalized_query or "").split()).strip().casefold()
        return bool(normalized) and normalized != original

    @staticmethod
    def _image_identifier_from_row(row: dict[str, Any]) -> str:
        return str(row.get("image_id") or row.get("_id") or "").strip()

    def _prefilter_candidates_by_query_similarity(
        self,
        *,
        query_dense_vector: list[float],
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        scored: list[RetrievalCandidate] = []
        for candidate in candidates:
            dense_vector = self._candidate_dense_vector(candidate)
            cosine = self._cosine_similarity(query_dense_vector, dense_vector)
            payload = dict(candidate.payload)
            payload["prefilter_cosine"] = float(cosine)
            candidate.payload = payload
            scored.append(candidate)

        scored.sort(
            key=lambda item: (
                float(item.payload.get("prefilter_cosine", 0.0)),
                float(item.score),
                item.candidate_id,
            ),
            reverse=True,
        )
        return scored[: max(1, int(top_k))]

    @staticmethod
    def _candidate_dense_vector(candidate: RetrievalCandidate) -> list[float] | None:
        dense = dict(candidate.payload or {}).get("_dense_vector")
        if not dense:
            return None
        try:
            return [float(value) for value in dense]
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float] | None) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _classify_route(
        self,
        *,
        query: str,
        intent: str,
        token_count: int,
        deep_force_keywords: set[str],
    ) -> str:
        lowered = query.lower()
        if intent == "chitchat":
            return "simple"
        if any(keyword in lowered for keyword in deep_force_keywords if keyword):
            return "deep"
        fast_token_limit = self._env_int("ROUTE_FAST_MAX_TOKENS", default=6)
        if intent in {"general", "definition"} and token_count <= fast_token_limit:
            return "fast"
        return "deep"

    def _is_chitchat_query(self, lowered_query: str) -> bool:
        compact = " ".join(str(lowered_query or "").split()).strip()
        if compact in self.CHITCHAT_KEYWORDS:
            return True
        return any(compact.startswith(f"{keyword} ") for keyword in {"hi", "hello", "hey"})

    def _embed_query_text(self, text: str) -> dict[str, Any]:
        key = str(text or "").strip()
        cached = self._query_vector_cache.get(key)
        if cached is not None:
            return cached

        vector = self.text_embedder.embed([key])[0]
        self._query_vector_cache[key] = vector
        self._query_vector_cache_order.append(key)

        while len(self._query_vector_cache_order) > self._query_vector_cache_size:
            oldest = self._query_vector_cache_order.pop(0)
            self._query_vector_cache.pop(oldest, None)

        return vector

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = str(os.getenv(name, "") or "").strip()
        if not raw:
            return int(default)
        try:
            return max(1, int(raw))
        except Exception:
            return int(default)

    @staticmethod
    def _env_csv(name: str, default: list[str]) -> list[str]:
        raw = str(os.getenv(name, "") or "").strip()
        if not raw:
            return list(default)
        return [item.strip().lower() for item in raw.split(",") if item.strip()]
