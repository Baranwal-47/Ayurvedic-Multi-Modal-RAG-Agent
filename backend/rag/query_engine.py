"""End-to-end query orchestration for retrieval-backed answers."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from groq import Groq

from embeddings.text_embedder import TextEmbedder
from rag.context_builder import ContextBuilder, ContextPack
from retrieval.hybrid_search import HybridSearcher
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
            temperature=0.2,
            max_completion_tokens=900,
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
            temperature=0.2,
            max_completion_tokens=900,
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

        retrieval_started = time.perf_counter()
        retrieval_result = self.searcher.search(query_bundle)
        retrieval_sec = time.perf_counter() - retrieval_started

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
        response = {
            "answer": answer,
            "citations": prepared.context_pack.citations,
            "images": prepared.context_pack.images,
            "tables": prepared.context_pack.tables,
            "enough_evidence": prepared.context_pack.enough_evidence,
            "query_intent": prepared.query_bundle.intent,
            "model": self.llm_client.model_name if self.llm_client.available() else "none",
            "timings": dict(prepared.timings),
        }
        if include_debug:
            response["debug"] = {
                "query_bundle": asdict(prepared.query_bundle),
                "retrieval": prepared.retrieval_result.debug,
                "rerank": prepared.evidence.debug,
                "context": prepared.context_pack.debug,
            }
        return response

    @staticmethod
    def _default_llm_client() -> LLMClient:
        provider = llm_provider_name()
        if provider == "groq":
            return GroqLLMClient()
        raise ValueError(f"Unsupported LLM provider '{provider}'")
