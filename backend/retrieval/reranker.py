"""Rerank and select multimodal retrieval evidence."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from retrieval.hybrid_search import QueryBundle, RetrievalCandidate


def reranker_model_name() -> str:
    return os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3").strip() or "BAAI/bge-reranker-v2-m3"


def reranker_batch_size(default: int = 8) -> int:
    raw = str(os.getenv("RERANKER_BATCH_SIZE", "") or "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except Exception:
        return int(default)


@dataclass
class EvidenceSelection:
    """Selected evidence after reranking and deterministic scoring."""

    reranked_candidates: list[RetrievalCandidate]
    citation_candidates: list[RetrievalCandidate]
    image_candidates: list[RetrievalCandidate]
    table_candidates: list[RetrievalCandidate]
    debug: dict[str, Any] = field(default_factory=dict)


class CandidateReranker:
    """Rerank candidates with a multilingual cross-encoder and light heuristics."""

    def __init__(self, model: Any | None = None, model_name: str | None = None) -> None:
        self._model = model
        self.model_name = str(model_name or reranker_model_name())
        self.batch_size = reranker_batch_size()

    def rerank(self, query_bundle: QueryBundle, candidates: list[RetrievalCandidate]) -> EvidenceSelection:
        if not candidates:
            return EvidenceSelection([], [], [], [], debug={"model": self.model_name, "pool_size": 0})

        pool = sorted(candidates, key=lambda item: item.score, reverse=True)[: query_bundle.rerank_top_k]
        base_scores = self._score_pairs(query_bundle.query, pool)
        present_ids = {candidate.candidate_id for candidate in pool}

        reranked: list[RetrievalCandidate] = []
        for candidate, base_score in zip(pool, base_scores):
            retrieval_score = float(candidate.score)
            final_score = self._apply_post_boosts(
                query_bundle=query_bundle,
                candidate=candidate,
                base_score=float(base_score),
                present_candidate_ids=present_ids,
            )
            payload = dict(candidate.payload)
            payload["retrieval_score"] = retrieval_score
            payload["rerank_score"] = float(base_score)
            payload["final_score"] = float(final_score)
            reranked.append(
                replace(
                    candidate,
                    score=final_score,
                    payload=payload,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)

        citation_candidates = [candidate for candidate in reranked if candidate.kind == "text"][:6]
        citation_ids = {candidate.candidate_id for candidate in citation_candidates}

        image_candidates = [
            candidate
            for candidate in reranked
            if candidate.kind == "image" and self._is_useful_image_candidate(candidate, query_bundle, citation_ids)
        ][:2]

        table_candidates = [
            candidate
            for candidate in reranked
            if candidate.kind == "text" and candidate.is_table and self._is_useful_table_candidate(candidate, query_bundle)
        ][:2]

        return EvidenceSelection(
            reranked_candidates=reranked,
            citation_candidates=citation_candidates,
            image_candidates=image_candidates,
            table_candidates=table_candidates,
            debug={
                "model": self.model_name,
                "pool_size": len(pool),
                "citation_ids": [candidate.candidate_id for candidate in citation_candidates],
                "image_ids": [candidate.candidate_id for candidate in image_candidates],
                "table_ids": [candidate.candidate_id for candidate in table_candidates],
            },
        )

    def _score_pairs(self, query: str, candidates: list[RetrievalCandidate]) -> list[float]:
        model = self._load_model()
        pairs = [(query, candidate.rerank_text or candidate.snippet) for candidate in candidates]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(score) for score in scores]

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, trust_remote_code=True)
        return self._model

    @staticmethod
    def _apply_post_boosts(
        *,
        query_bundle: QueryBundle,
        candidate: RetrievalCandidate,
        base_score: float,
        present_candidate_ids: set[str],
    ) -> float:
        adjusted = float(base_score)

        if candidate.is_table and query_bundle.is_table:
            adjusted += 0.18
        if candidate.chunk_type == "shloka" and query_bundle.is_shloka:
            adjusted += 0.18
        if candidate.kind == "image" and query_bundle.is_visual and candidate.linked_ids:
            adjusted += 0.12
        if candidate.chunk_type == "page_bridge":
            source_ids = {str(item) for item in candidate.payload.get("bridge_source_chunk_ids", []) if str(item).strip()}
            if len(source_ids.intersection(present_candidate_ids)) >= 2:
                adjusted -= 0.12

        return adjusted

    @staticmethod
    def _is_useful_image_candidate(
        candidate: RetrievalCandidate,
        query_bundle: QueryBundle,
        citation_ids: set[str],
    ) -> bool:
        if not candidate.image_url:
            return False
        if candidate.score < -2.5:
            return False
        if query_bundle.is_visual:
            return True
        if citation_ids.intersection(candidate.linked_ids):
            return True
        return candidate.score >= -0.4

    @staticmethod
    def _is_useful_table_candidate(candidate: RetrievalCandidate, query_bundle: QueryBundle) -> bool:
        if not candidate.is_table:
            return False
        if not candidate.table_markdown and not candidate.table_rows:
            return False
        if query_bundle.is_table:
            return True
        row_count = len(candidate.table_rows or [])
        return candidate.score >= 0.0 and row_count >= 2
