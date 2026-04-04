"""Rerank and select multimodal retrieval evidence."""

from __future__ import annotations

import os
import time
from threading import Lock
from dataclasses import dataclass, field, replace
from typing import Any

from retrieval.hybrid_search import QueryBundle, RetrievalCandidate


_RERANKER_MODEL_SINGLETON: Any | None = None
_RERANKER_MODEL_LOAD_COUNT = 0
_RERANKER_MODEL_LOCK = Lock()


def reranker_model_name() -> str:
    return os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3").strip() or "BAAI/bge-reranker-v2-m3"


def reranker_batch_size(default: int = 8) -> int:
    raw = str(os.getenv("RERANKER_BATCH_SIZE", "") or "").strip()
    if not raw:
        return int(default)
    try:
        return max(8, int(raw))
    except Exception:
        return int(default)


def reranker_max_length(default: int = 320) -> int:
    raw = str(os.getenv("RERANKER_MAX_LENGTH", "") or "").strip()
    if not raw:
        return int(default)
    try:
        return min(384, max(256, int(raw)))
    except Exception:
        return int(default)


def rerank_prefilter_top_k(default: int = 12) -> int:
    raw = str(os.getenv("RERANK_PREFILTER_TOP_K", "") or "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except Exception:
        return int(default)


def rerank_pool_top_k(default: int = 8) -> int:
    raw = str(os.getenv("RERANK_POOL_TOP_K", "") or "").strip()
    if not raw:
        return int(default)
    try:
        return min(8, max(1, int(raw)))
    except Exception:
        return int(default)


def reranker_log_load() -> bool:
    return str(os.getenv("RERANKER_LOG_LOAD", "true")).strip().lower() in {"1", "true", "yes"}


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
        self.max_length = reranker_max_length()
        self.prefilter_top_k = rerank_prefilter_top_k()
        self.pool_top_k = rerank_pool_top_k()
        self._prewarmed = model is not None

    def rerank(self, query_bundle: QueryBundle, candidates: list[RetrievalCandidate]) -> EvidenceSelection:
        rerank_timing = {
            "pair_build_sec": 0.0,
            "model_infer_sec": 0.0,
            "postprocess_sec": 0.0,
        }
        candidate_count_before_prefilter = len(candidates)
        prefiltered = self._prefilter_candidates(query_bundle, candidates)
        candidate_count_after_prefilter = len(prefiltered)
        rerank_meta = {
            "device": self._model_device_label(),
            "model": self.model_name,
            "pool_size": 0,
            "warm_model": bool(self._prewarmed),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "model_load_count": _RERANKER_MODEL_LOAD_COUNT,
            "model_loaded_once": _RERANKER_MODEL_LOAD_COUNT <= 1,
        }

        if not prefiltered:
            return EvidenceSelection(
                [],
                [],
                [],
                [],
                debug={
                    "model": self.model_name,
                    "pool_size": 0,
                    "rerank_pool_size": 0,
                    "reranker_infer_time": 0.0,
                    "candidate_count_before_prefilter": candidate_count_before_prefilter,
                    "candidate_count_after_prefilter": candidate_count_after_prefilter,
                    "rerank_timing": rerank_timing,
                    "rerank_meta": rerank_meta,
                },
            )

        pair_build_started = time.perf_counter()
        requested_pool = int(query_bundle.rerank_top_k or self.pool_top_k)
        requested_pool = min(8, max(1, requested_pool, self.pool_top_k))
        pool_limit = min(requested_pool, len(prefiltered))
        pool = prefiltered[:pool_limit]
        pairs = self._build_pairs(query_bundle.query, pool)
        rerank_timing["pair_build_sec"] = time.perf_counter() - pair_build_started

        base_scores, model_infer_sec, effective_batch_size = self._predict_pairs(pairs)
        rerank_timing["model_infer_sec"] = model_infer_sec
        rerank_meta = {
            "device": self._model_device_label(),
            "model": self.model_name,
            "pool_size": len(pool),
            "warm_model": bool(self._prewarmed),
            "batch_size": self.batch_size,
            "effective_batch_size": effective_batch_size,
            "max_length": self.max_length,
            "model_load_count": _RERANKER_MODEL_LOAD_COUNT,
            "model_loaded_once": _RERANKER_MODEL_LOAD_COUNT <= 1,
        }

        postprocess_started = time.perf_counter()
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

        reranked.sort(key=lambda item: (-item.score, item.candidate_id))

        citation_candidates = [candidate for candidate in reranked if candidate.kind == "text"][:6]
        citation_ids = {candidate.candidate_id for candidate in citation_candidates}

        ranked_top_image = reranked[0] if reranked and reranked[0].kind == "image" else None
        image_candidates = [
            candidate
            for candidate in reranked
            if candidate.kind == "image"
            and (
                candidate == ranked_top_image
                or self._is_useful_image_candidate(candidate, query_bundle, citation_ids)
            )
        ][:2]

        table_candidates = [
            candidate
            for candidate in reranked
            if candidate.kind == "text" and candidate.is_table and self._is_useful_table_candidate(candidate, query_bundle)
        ][:2]
        rerank_timing["postprocess_sec"] = time.perf_counter() - postprocess_started

        return EvidenceSelection(
            reranked_candidates=reranked,
            citation_candidates=citation_candidates,
            image_candidates=image_candidates,
            table_candidates=table_candidates,
            debug={
                "model": self.model_name,
                "pool_size": len(pool),
                "rerank_pool_size": len(pool),
                "reranker_infer_time": round(model_infer_sec, 4),
                "candidate_count_before_prefilter": candidate_count_before_prefilter,
                "candidate_count_after_prefilter": candidate_count_after_prefilter,
                "rerank_timing": {key: round(value, 4) for key, value in rerank_timing.items()},
                "rerank_meta": rerank_meta,
                "citation_ids": [candidate.candidate_id for candidate in citation_candidates],
                "image_ids": [candidate.candidate_id for candidate in image_candidates],
                "table_ids": [candidate.candidate_id for candidate in table_candidates],
            },
        )

    def prewarm(self) -> None:
        self._load_model()
        self._prewarmed = True

    def runtime_meta(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "device": self._model_device_label(),
            "pool_size": 0,
            "warm_model": self._model is not None,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "model_load_count": _RERANKER_MODEL_LOAD_COUNT,
            "model_loaded_once": _RERANKER_MODEL_LOAD_COUNT <= 1,
        }

    def _build_pairs(self, query: str, candidates: list[RetrievalCandidate]) -> list[tuple[str, str]]:
        return [(query, self._truncate_text_for_inference(candidate.rerank_text or candidate.snippet)) for candidate in candidates]

    def _truncate_text_for_inference(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return value

        model = self._load_model()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return " ".join(value.split()[: self.max_length])

        token_ids = tokenizer.encode(value, add_special_tokens=False, truncation=True, max_length=self.max_length)
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def _predict_pairs(self, pairs: list[tuple[str, str]]) -> tuple[list[float], float, int]:
        if not pairs:
            return [], 0.0, 0

        model = self._load_model()
        effective_batch_size = min(self.batch_size, len(pairs))
        infer_started = time.perf_counter()
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        infer_sec = time.perf_counter() - infer_started
        return [float(score) for score in scores], infer_sec, effective_batch_size

    def _load_model(self):
        global _RERANKER_MODEL_SINGLETON, _RERANKER_MODEL_LOAD_COUNT
        if self._model is not None:
            return self._model

        if _RERANKER_MODEL_SINGLETON is None:
            with _RERANKER_MODEL_LOCK:
                if _RERANKER_MODEL_SINGLETON is None:
                    if reranker_log_load():
                        print("Loading reranker model...")
                    from sentence_transformers import CrossEncoder

                    _RERANKER_MODEL_SINGLETON = CrossEncoder(self.model_name, trust_remote_code=True)
                    _RERANKER_MODEL_LOAD_COUNT += 1

        self._model = _RERANKER_MODEL_SINGLETON
        return self._model

    def _prefilter_candidates(self, query_bundle: QueryBundle, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        query_dense_vector = list(getattr(query_bundle, "_query_dense_vector", []) or [])
        if not query_dense_vector:
            return sorted(candidates, key=lambda item: (-item.score, item.candidate_id))[: self.prefilter_top_k]

        scored: list[RetrievalCandidate] = []
        for candidate in candidates:
            dense_vector = self._dense_vector_from_candidate(candidate)
            cosine = self._cosine_similarity(query_dense_vector, dense_vector)
            payload = dict(candidate.payload)
            payload["prefilter_cosine"] = float(cosine)
            scored.append(replace(candidate, payload=payload))

        scored.sort(
            key=lambda item: (
                float(item.payload.get("prefilter_cosine", 0.0)),
                float(item.score),
                item.candidate_id,
            ),
            reverse=True,
        )
        return scored[: self.prefilter_top_k]

    @staticmethod
    def _dense_vector_from_candidate(candidate: RetrievalCandidate) -> list[float] | None:
        dense_vector = dict(candidate.payload or {}).get("_dense_vector")
        if not dense_vector:
            return None
        try:
            return [float(value) for value in dense_vector]
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float] | None) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = sum(float(a) * float(a) for a in left) ** 0.5
        right_norm = sum(float(b) * float(b) for b in right) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _model_device_label(self) -> str:
        if self._model is None:
            return "unloaded"
        model_obj = getattr(self._model, "model", None)
        device = getattr(model_obj, "device", None)
        if device is None:
            return "unknown"
        return str(device)

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
        if query_bundle.is_shloka and CandidateReranker._is_shloka_like_candidate(candidate):
            adjusted += 0.1
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
        text_blob = " ".join(
            part
            for part in [
                candidate.caption or "",
                " ".join(candidate.labels or []),
                candidate.text or "",
                " ".join(candidate.section_path or []),
            ]
            if part
        ).lower()
        query_terms = {
            token
            for token in query_bundle.normalized_query.lower().split()
            if len(token) >= 4 and token not in {"show", "image", "images", "figure", "figures", "diagram", "diagrams"}
        }
        keyword_hit = bool(query_terms) and any(term in text_blob for term in query_terms)
        linked_hit = bool(citation_ids.intersection(candidate.linked_ids))
        if query_bundle.is_visual:
            return linked_hit or keyword_hit or candidate.score >= 0.92
        if linked_hit:
            return True
        return keyword_hit and candidate.score >= 0.2

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

    @staticmethod
    def _is_shloka_like_candidate(candidate: RetrievalCandidate) -> bool:
        payload = dict(candidate.payload or {})
        text = str(candidate.text or candidate.snippet or "").strip()
        scripts = {str(script) for script in payload.get("scripts", [])}
        shloka_number = payload.get("shloka_number")

        if candidate.chunk_type == "shloka":
            return True
        if bool(payload.get("is_shloka")):
            return True
        if shloka_number:
            return True
        if "Deva" in scripts and any(mark in text for mark in ("।", "॥", "|")):
            return True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return len(lines) >= 2 and all(len(line.split()) <= 16 for line in lines)
