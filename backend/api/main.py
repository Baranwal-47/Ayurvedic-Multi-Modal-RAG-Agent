"""FastAPI application for retrieval-backed query endpoints."""

from __future__ import annotations

import json
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse

from api.models import HealthResponse, QueryRequest, QueryResponse
from rag.query_engine import QueryEngine, llm_provider_name


app = FastAPI(title="Ayurveda RAG API", version="0.1.0")


def _cors_origins() -> list[str]:
    raw = str(os.getenv("FRONTEND_ORIGINS", "") or "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_query_engine() -> None:
    if str(os.getenv("PREWARM_MODELS_ON_STARTUP", "true")).strip().lower() not in {"1", "true", "yes"}:
        return
    engine = get_query_engine()
    engine.prewarm(load_reranker=True)


@lru_cache(maxsize=1)
def get_query_engine() -> QueryEngine:
    return QueryEngine()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        engine = get_query_engine()
        engine.qdrant.client.get_collections()
        return HealthResponse(
            status="ok",
            llm_provider=llm_provider_name(),
            llm_available=engine.llm_client.available(),
            qdrant_reachable=True,
            text_collection=engine.qdrant.text_collection,
            image_collection=engine.qdrant.image_collection,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        engine = get_query_engine()
        response = engine.query(
            query=request.query,
            doc_id=request.doc_id,
            page_start=request.page_start,
            page_end=request.page_end,
            languages=request.languages,
            scripts=request.scripts,
            chunk_types=request.chunk_types,
            include_debug=request.include_debug,
        )
        return QueryResponse.model_validate(response)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.post("/query/stream")
def stream_query(request: QueryRequest) -> EventSourceResponse:
    engine = get_query_engine()

    def event_generator():
        try:
            for event in engine.stream_query(
                query=request.query,
                doc_id=request.doc_id,
                page_start=request.page_start,
                page_end=request.page_end,
                languages=request.languages,
                scripts=request.scripts,
                chunk_types=request.chunk_types,
                include_debug=request.include_debug,
            ):
                payload = event.get("data")
                if isinstance(payload, dict):
                    payload = json.dumps(payload, ensure_ascii=False)
                yield {"event": event.get("event", "token"), "data": payload}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"detail": f"Query stream failed: {exc}"}, ensure_ascii=False)}

    return EventSourceResponse(event_generator(), ping=15)
