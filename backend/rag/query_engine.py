"""End-to-end query orchestration for retrieval-backed answers."""

# tests\query_debug.py --interactive --prewarm
# query> what is palika yantra
# query> show me the tlc figure
# query> define rasa shastra

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from groq import Groq

from embeddings.text_embedder import TextEmbedder
from rag.context_builder import ContextBuilder, ContextPack
from retrieval.hybrid_search import HybridSearchResult, HybridSearcher
from retrieval.reranker import CandidateReranker, EvidenceSelection
from vector_db.qdrant_client import QdrantManager


def llm_provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower() or "groq"


def groq_model_name() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"


@dataclass
class QueryPreparation:
    """Prepared retrieval and context state before answer synthesis."""

    query_bundle: Any
    retrieval_result: Any
    evidence: EvidenceSelection
    context_pack: ContextPack
    timings: dict[str, float]


class LLMClient(ABC):
    """Minimal LLM interface so provider changes stay isolated."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream_generate(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        raise NotImplementedError


class GroqLLMClient(LLMClient):
    """Groq-backed chat completion client for sync and streaming answers."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        self.api_key = str(api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self._model_name = str(model_name or groq_model_name())
        self._client: Groq | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.available():
            raise RuntimeError("Groq API key is not configured")
        completion = self._client_instance().chat.completions.create(
            model=self._model_name,
            temperature=0,
            max_completion_tokens=600,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        message = completion.choices[0].message
        return str(getattr(message, "content", "") or "").strip()

    def stream_generate(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        if not self.available():
            raise RuntimeError("Groq API key is not configured")
        stream = self._client_instance().chat.completions.create(
            model=self._model_name,
            temperature=0,
            max_completion_tokens=600,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta else None
            if content:
                yield str(content)

    def _client_instance(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client


class QueryEngine:
    """Shared sync and streaming query pipeline."""

    def __init__(
        self,
        qdrant: QdrantManager | None = None,
        text_embedder: TextEmbedder | None = None,
        searcher: HybridSearcher | None = None,
        reranker: CandidateReranker | None = None,
        context_builder: ContextBuilder | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.qdrant = qdrant or QdrantManager()
        self.text_embedder = text_embedder or TextEmbedder()
        self.searcher = searcher or HybridSearcher(self.qdrant, self.text_embedder)
        self.reranker = reranker or CandidateReranker()
        self.context_builder = context_builder or ContextBuilder()
        self.llm_client = llm_client or self._default_llm_client()

    def prewarm(self, *, load_reranker: bool = True) -> None:
        self.searcher.prewarm()
        if load_reranker:
            self.reranker.prewarm()

    def query(
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
    ) -> dict[str, Any]:
        prepared = self._prepare_query(
            query=query,
            doc_id=doc_id,
            page_start=page_start,
            page_end=page_end,
            languages=languages,
            scripts=scripts,
            chunk_types=chunk_types,
            source_file=source_file,
            include_debug=include_debug,
        )
        answer = self._generate_answer(prepared)
        response = self._build_response(prepared, answer=answer, include_debug=include_debug)
        return response

    def debug_query(
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
    ) -> dict[str, Any]:
        prepared = self._prepare_query(
            query=query,
            doc_id=doc_id,
            page_start=page_start,
            page_end=page_end,
            languages=languages,
            scripts=scripts,
            chunk_types=chunk_types,
            source_file=source_file,
            include_debug=True,
        )
        return {
            "query_bundle": asdict(prepared.query_bundle),
            "retrieved_candidates": [self._serialize_candidate(candidate) for candidate in prepared.retrieval_result.candidates],
            "reranked_candidates": [self._serialize_candidate(candidate) for candidate in prepared.evidence.reranked_candidates],
            "final_context": {
                "system_prompt": prepared.context_pack.system_prompt,
                "user_prompt": prepared.context_pack.user_prompt,
                "citations": prepared.context_pack.citations,
                "images": prepared.context_pack.images,
                "tables": prepared.context_pack.tables,
                "enough_evidence": prepared.context_pack.enough_evidence,
            },
            "timings": dict(prepared.timings),
        }

    def stream_query(
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
    ) -> Iterator[dict[str, Any]]:
        yield {"event": "status", "data": {"stage": "retrieval", "message": "Searching the indexed corpus..."}} 
        prepared = self._prepare_query(
            query=query,
            doc_id=doc_id,
            page_start=page_start,
            page_end=page_end,
            languages=languages,
            scripts=scripts,
            chunk_types=chunk_types,
            source_file=source_file,
            include_debug=include_debug,
        )

        if not prepared.context_pack.enough_evidence or not self.llm_client.available():
            answer = self._generate_answer(prepared)
            yield {"event": "token", "data": answer}
            yield {"event": "final", "data": self._build_response(prepared, answer=answer, include_debug=include_debug)}
            return

        answer_parts: list[str] = []
        llm_started = time.perf_counter()
        yield {"event": "status", "data": {"stage": "generation", "message": "Generating grounded answer..."}} 
        for token in self.llm_client.stream_generate(
            system_prompt=prepared.context_pack.system_prompt,
            user_prompt=prepared.context_pack.user_prompt,
        ):
            answer_parts.append(token)
            yield {"event": "token", "data": token}
        prepared.timings["llm_sec"] = round(time.perf_counter() - llm_started, 4)
        answer = "".join(answer_parts).strip()
        yield {"event": "final", "data": self._build_response(prepared, answer=answer, include_debug=include_debug)}

    def _prepare_query(
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
    ) -> QueryPreparation:
        started = time.perf_counter()
        query_bundle = self.searcher.build_query_bundle(
            query=query,
            doc_id=doc_id,
            page_start=page_start,
            page_end=page_end,
            languages=languages,
            scripts=scripts,
            chunk_types=chunk_types,
            source_file=source_file,
            include_debug=include_debug,
        )

        if query_bundle.route == "simple":
            context_pack = self._simple_context_pack(query_bundle.query)
            rerank_meta = self._reranker_runtime_meta()
            return QueryPreparation(
                query_bundle=query_bundle,
                retrieval_result=HybridSearchResult(
                    query_bundle=query_bundle,
                    candidates=[],
                    debug={
                        "route": "simple",
                        "skipped_retrieval": True,
                        "candidate_count_before_prefilter": 0,
                        "candidate_count_after_prefilter": 0,
                    },
                ),
                evidence=EvidenceSelection(
                    reranked_candidates=[],
                    citation_candidates=[],
                    image_candidates=[],
                    table_candidates=[],
                    debug={
                        "model": self.reranker.model_name,
                        "pool_size": 0,
                        "rerank_pool_size": 0,
                        "reranker_infer_time": 0.0,
                        "candidate_count_before_prefilter": 0,
                        "candidate_count_after_prefilter": 0,
                        "rerank_timing": {
                            "pair_build_sec": 0.0,
                            "model_infer_sec": 0.0,
                            "postprocess_sec": 0.0,
                        },
                        "rerank_meta": rerank_meta,
                        "skipped": True,
                        "reason": "simple_route",
                    },
                ),
                context_pack=context_pack,
                timings={
                    "retrieval_sec": 0.0,
                    "rerank_sec": 0.0,
                    "context_sec": 0.0,
                    "total_pre_llm_sec": round(time.perf_counter() - started, 4),
                },
            )

        retrieval_started = time.perf_counter()
        retrieval_result = self.searcher.search(query_bundle)
        retrieval_sec = time.perf_counter() - retrieval_started

        rerank_sec = 0.0
        if query_bundle.route == "fast":
            evidence = self._fast_route_evidence(query_bundle, retrieval_result.candidates)
        else:
            rerank_started = time.perf_counter()
            evidence = self.reranker.rerank(query_bundle, retrieval_result.candidates)
            rerank_sec = time.perf_counter() - rerank_started

        context_started = time.perf_counter()
        context_pack = self.context_builder.build(query_bundle, evidence)
        context_sec = time.perf_counter() - context_started

        return QueryPreparation(
            query_bundle=query_bundle,
            retrieval_result=retrieval_result,
            evidence=evidence,
            context_pack=context_pack,
            timings={
                "retrieval_sec": round(retrieval_sec, 4),
                "rerank_sec": round(rerank_sec, 4),
                "context_sec": round(context_sec, 4),
                "total_pre_llm_sec": round(time.perf_counter() - started, 4),
            },
        )

    def _generate_answer(self, prepared: QueryPreparation) -> str:
        if getattr(prepared.query_bundle, "route", "deep") == "simple" and not self.llm_client.available():
            return "Hello! How can I help you?"

        if not prepared.context_pack.enough_evidence:
            return "I couldn't find enough reliable evidence in the indexed material to answer confidently. Please refine the question or narrow the source scope."

        if not self.llm_client.available():
            citation_ids = ", ".join(citation["id"] for citation in prepared.context_pack.citations[:3])
            suffix = f" Relevant citations: {citation_ids}." if citation_ids else ""
            return f"Answer generation is not configured, but relevant evidence was found.{suffix}".strip()

        llm_started = time.perf_counter()
        answer = self.llm_client.generate(
            system_prompt=prepared.context_pack.system_prompt,
            user_prompt=prepared.context_pack.user_prompt,
        )
        prepared.timings["llm_sec"] = round(time.perf_counter() - llm_started, 4)
        return answer

    def _build_response(self, prepared: QueryPreparation, *, answer: str, include_debug: bool) -> dict[str, Any]:
        citations = prepared.context_pack.citations if prepared.context_pack.enough_evidence else []
        images = prepared.context_pack.images if prepared.context_pack.enough_evidence else []
        tables = prepared.context_pack.tables if prepared.context_pack.enough_evidence else []
        retrieval_timing = dict(prepared.retrieval_result.debug.get("retrieval_timing", {}) or {})
        retrieval_counts = dict(prepared.retrieval_result.debug.get("retrieval_counts", {}) or {})
        rerank_timing = dict(prepared.evidence.debug.get("rerank_timing", {}) or {})
        rerank_meta = dict(prepared.evidence.debug.get("rerank_meta", {}) or {})
        candidate_count_before_prefilter = prepared.retrieval_result.debug.get(
            "candidate_count_before_prefilter",
            prepared.evidence.debug.get("candidate_count_before_prefilter", 0),
        )
        candidate_count_after_prefilter = prepared.retrieval_result.debug.get(
            "candidate_count_after_prefilter",
            prepared.evidence.debug.get("candidate_count_after_prefilter", 0),
        )
        rerank_pool_size = prepared.evidence.debug.get("rerank_pool_size", prepared.evidence.debug.get("pool_size", 0))
        reranker_infer_time = prepared.evidence.debug.get("reranker_infer_time", rerank_timing.get("model_infer_sec", 0.0))
        public_images = [self._public_image_card(image) for image in images]
        response = {
            "answer": answer,
            "citations": citations,
            "images": public_images,
            "tables": tables,
            "enough_evidence": prepared.context_pack.enough_evidence,
            "query_intent": prepared.query_bundle.intent,
            "model": self.llm_client.model_name if self.llm_client.available() else "none",
            "timings": dict(prepared.timings),
        }
        if include_debug:
            response["debug"] = {
                "query_bundle": asdict(prepared.query_bundle),
                "retrieved_candidates": [self._serialize_candidate(candidate) for candidate in prepared.retrieval_result.candidates],
                "reranked_candidates": [self._serialize_candidate(candidate) for candidate in prepared.evidence.reranked_candidates],
                "retrieval_timing": retrieval_timing,
                "retrieval_counts": retrieval_counts,
                "rerank_timing": rerank_timing,
                "rerank_meta": rerank_meta,
                "candidate_count_before_prefilter": candidate_count_before_prefilter,
                "candidate_count_after_prefilter": candidate_count_after_prefilter,
                "rerank_pool_size": rerank_pool_size,
                "reranker_infer_time": reranker_infer_time,
                "retrieval": prepared.retrieval_result.debug,
                "rerank": prepared.evidence.debug,
                "context": prepared.context_pack.debug,
                "final_context": {
                    "citations": prepared.context_pack.citations,
                    "images": prepared.context_pack.images,
                    "tables": prepared.context_pack.tables,
                    "enough_evidence": prepared.context_pack.enough_evidence,
                    "user_prompt": prepared.context_pack.user_prompt,
                },
            }
        return response

    def _simple_context_pack(self, query: str) -> ContextPack:
        return ContextPack(
            system_prompt=(
                "You are a concise, polite assistant. Respond naturally to greetings and small talk."
            ),
            user_prompt=str(query or "").strip(),
            citations=[],
            images=[],
            tables=[],
            enough_evidence=True,
            debug={"route": "simple", "skipped_retrieval": True, "skipped_rerank": True},
        )

    def _fast_route_evidence(self, query_bundle, candidates: list) -> EvidenceSelection:
        ranked = sorted(candidates, key=lambda item: (-item.score, item.candidate_id))
        citation_candidates = [candidate for candidate in ranked if candidate.kind == "text"][:6]
        image_candidates = [candidate for candidate in ranked if candidate.kind == "image" and candidate.image_url][:2]
        table_candidates = [candidate for candidate in ranked if candidate.kind == "text" and candidate.is_table][:2]
        rerank_meta = self._reranker_runtime_meta()

        return EvidenceSelection(
            reranked_candidates=ranked,
            citation_candidates=citation_candidates,
            image_candidates=image_candidates,
            table_candidates=table_candidates,
            debug={
                "model": "none",
                "pool_size": 0,
                "rerank_pool_size": 0,
                "reranker_infer_time": 0.0,
                "candidate_count_before_prefilter": len(candidates),
                "candidate_count_after_prefilter": len(candidates),
                "rerank_timing": {
                    "pair_build_sec": 0.0,
                    "model_infer_sec": 0.0,
                    "postprocess_sec": 0.0,
                },
                "rerank_meta": rerank_meta,
                "skipped": True,
                "reason": f"{query_bundle.route}_route",
                "citation_ids": [candidate.candidate_id for candidate in citation_candidates],
                "image_ids": [candidate.candidate_id for candidate in image_candidates],
                "table_ids": [candidate.candidate_id for candidate in table_candidates],
            },
        )

    def _reranker_runtime_meta(self) -> dict[str, Any]:
        runtime_meta_fn = getattr(self.reranker, "runtime_meta", None)
        if callable(runtime_meta_fn):
            try:
                meta = dict(runtime_meta_fn() or {})
                if meta:
                    return meta
            except Exception:
                pass

        model_name = str(getattr(self.reranker, "model_name", "unknown") or "unknown")
        model_obj = getattr(self.reranker, "_model", None)
        return {
            "model": model_name,
            "device": "unknown" if model_obj is not None else "unloaded",
            "pool_size": 0,
            "warm_model": bool(model_obj is not None),
        }

    @staticmethod
    def _public_image_card(image: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": image.get("id"),
            "image_id": image.get("image_id"),
            "page_number": image.get("page_number"),
            "caption": image.get("caption") or "",
            "labels": list(image.get("labels") or []),
            "image_url": image.get("image_url"),
            "cloudinary_public_id": image.get("cloudinary_public_id"),
            "source_file": image.get("source_file"),
            "citation_ids": list(image.get("citation_ids") or []),
        }

    @staticmethod
    def _serialize_candidate(candidate) -> dict[str, Any]:
        return {
            "id": candidate.candidate_id,
            "kind": candidate.kind,
            "score": candidate.score,
            "doc_id": candidate.doc_id,
            "source_file": candidate.source_file,
            "page_numbers": list(candidate.page_numbers),
            "chunk_type": candidate.chunk_type,
            "image_type": candidate.image_type,
            "section_path": list(candidate.section_path),
            "snippet": candidate.snippet,
            "retrieval_reasons": list(candidate.retrieval_reasons),
            "linked_ids": list(candidate.linked_ids),
            "table_markdown": candidate.table_markdown,
            "caption": candidate.caption,
            "image_url": candidate.image_url,
        }

    @staticmethod
    def _default_llm_client() -> LLMClient:
        provider = llm_provider_name()
        if provider == "groq":
            return GroqLLMClient()
        raise ValueError(f"Unsupported LLM provider '{provider}'")
