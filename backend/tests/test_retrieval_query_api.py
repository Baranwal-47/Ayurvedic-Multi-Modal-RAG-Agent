from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import main as api_main
from rag.context_builder import ContextBuilder
from rag.query_engine import LLMClient, QueryEngine
from retrieval.hybrid_search import HybridSearcher, QueryBundle, QueryFilters, RetrievalCandidate
from retrieval.reranker import EvidenceSelection


class FakeTextEmbedder:
    def embed(self, texts):
        return [
            {
                "dense_vector": [0.1, 0.2, 0.3],
                "sparse_indices": [1, 2],
                "sparse_values": [0.5, 0.6],
            }
            for _ in texts
        ]


class FakeQdrantManager:
    def __init__(self):
        self.text_collection = "text_chunks"
        self.image_collection = "image_chunks"
        self.last_image_exclusions = None

    def hybrid_search_text(self, **kwargs):
        return [
            {
                "_id": "text-1",
                "_score": 0.91,
                "chunk_id": "doc1:p3-3:paragraph:1",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_numbers": [3],
                "chunk_type": "paragraph",
                "text": "This page discusses the linked figure.",
                "section_path": ["Chapter 1"],
                "heading_text": "Chapter 1",
                "image_ids": ["doc1:p3:img:1"],
            },
            {
                "_id": "text-2",
                "_score": 0.83,
                "chunk_id": "doc1:p7-7:paragraph:2",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_numbers": [7],
                "chunk_type": "paragraph",
                "text": "This page refers to a nearby illustration.",
                "section_path": ["Chapter 2"],
                "heading_text": "Chapter 2",
                "image_ids": [],
            },
        ]

    def search_images(self, **kwargs):
        self.last_image_exclusions = kwargs.get("exclude_image_types")
        return []

    def retrieve_points(self, *, collection, point_ids):
        assert collection == "image_chunks"
        return [
            {
                "_id": "image-1",
                "_score": 0.0,
                "image_id": "doc1:p3:img:1",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_number": 3,
                "image_type": "diagram",
                "image_url": "https://example.com/fig1.png",
                "cloudinary_public_id": "demo/doc1/p3/fig1",
                "linked_chunk_ids": ["doc1:p3-3:paragraph:1"],
                "caption": "Linked figure",
                "labels": ["arm"],
                "surrounding_text": "Linked surrounding text",
                "section_path": ["Chapter 1"],
            }
        ]

    def scroll_points(self, *, collection, filters, limit, exclude_image_types=None):
        assert collection == "image_chunks"
        if filters.get("doc_id") == "doc1" and filters.get("page_start") == 7:
            return [
                {
                    "_id": "image-2",
                    "_score": 0.0,
                    "image_id": "doc1:p7:img:2",
                    "doc_id": "doc1",
                    "source_file": "sample.pdf",
                    "page_number": 7,
                    "image_type": "diagram",
                    "image_url": "https://example.com/fig2.png",
                    "cloudinary_public_id": "demo/doc1/p7/fig2",
                    "linked_chunk_ids": [],
                    "caption": "",
                    "labels": [],
                    "surrounding_text": "Nearby figure context",
                    "section_path": ["Chapter 2"],
                }
            ]
        return []


class FakeRerankModel:
    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        scores = []
        for _, text in pairs:
            lowered = text.lower()
            if "table no. 1" in lowered:
                scores.append(0.7)
            elif "linked figure" in lowered:
                scores.append(0.5)
            else:
                scores.append(0.3)
        return scores


class FakeLLMClient(LLMClient):
    def __init__(self, available=True):
        self._available = available

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def available(self) -> bool:
        return self._available

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "Grounded answer with citations C1."

    def stream_generate(self, *, system_prompt: str, user_prompt: str):
        yield "Grounded "
        yield "stream "
        yield "answer."


def make_query_bundle(intent: str = "general") -> QueryBundle:
    return QueryBundle(
        query="test query",
        normalized_query="test query",
        intent=intent,
        filters=QueryFilters(),
    )


def make_candidate(
    *,
    candidate_id: str,
    score: float,
    chunk_type: str = "paragraph",
    kind: str = "text",
    table_rows=None,
    table_markdown=None,
    image_url: str | None = None,
    linked_ids=None,
):
    return RetrievalCandidate(
        kind=kind,
        candidate_id=candidate_id,
        doc_id="doc1",
        source_file="sample.pdf",
        page_numbers=[3],
        score=score,
        snippet="snippet",
        rerank_text="Table No. 1" if chunk_type == "table_text" else "regular paragraph",
        section_path=["Chapter 1"],
        heading_text="Chapter 1",
        chunk_type=chunk_type,
        image_type="diagram" if kind == "image" else None,
        image_url=image_url,
        cloudinary_public_id="demo/img" if image_url else None,
        caption="figure caption" if kind == "image" else None,
        labels=["label"] if kind == "image" else [],
        linked_ids=list(linked_ids or []),
        retrieval_reasons=["seed"],
        table_markdown=table_markdown,
        table_rows=table_rows,
        text="table body" if chunk_type == "table_text" else "body",
        payload={
            "table_id": "doc1:p3:table:1",
            "table_caption": "Table No. 1",
            "bridge_source_chunk_ids": [],
        },
    )


def test_hybrid_search_rescues_linked_and_proximity_images():
    qdrant = FakeQdrantManager()
    searcher = HybridSearcher(qdrant=qdrant, text_embedder=FakeTextEmbedder())

    result = searcher.search(searcher.build_query_bundle(query="show me the figure"))
    candidate_ids = {candidate.candidate_id for candidate in result.candidates}

    assert "doc1:p3:img:1" in candidate_ids
    assert "doc1:p7:img:2" in candidate_ids
    assert qdrant.last_image_exclusions == ["decorative"]


def test_context_builder_only_displays_table_when_sufficiently_useful():
    builder = ContextBuilder(token_budget=1200)
    weak_table = make_candidate(
        candidate_id="doc1:p3-3:table_text:1",
        score=0.05,
        chunk_type="table_text",
        table_rows=[["A", "B"], ["1", "2"], ["3", "4"]],
        table_markdown="| A | B |\n| --- | --- |\n| 1 | 2 |",
    )
    strong_table = make_candidate(
        candidate_id="doc1:p3-3:table_text:2",
        score=0.45,
        chunk_type="table_text",
        table_rows=[["A", "B"], ["1", "2"], ["3", "4"]],
        table_markdown="| A | B |\n| --- | --- |\n| 1 | 2 |",
    )

    weak_context = builder.build(
        make_query_bundle(intent="general"),
        EvidenceSelection([weak_table], [weak_table], [], [weak_table]),
    )
    strong_context = builder.build(
        make_query_bundle(intent="table"),
        EvidenceSelection([strong_table], [strong_table], [], [strong_table]),
    )

    assert weak_context.tables == []
    assert strong_context.tables


def test_context_builder_skips_image_card_without_url():
    builder = ContextBuilder(token_budget=1200)
    citation = make_candidate(candidate_id="doc1:p3-3:paragraph:1", score=0.6)
    image = make_candidate(candidate_id="doc1:p3:img:1", score=0.7, kind="image", image_url=None, linked_ids=[citation.candidate_id])

    context = builder.build(
        make_query_bundle(intent="visual"),
        EvidenceSelection([citation, image], [citation], [image], []),
    )

    assert context.images == []


def test_query_engine_returns_grounded_fallback_when_llm_unavailable():
    class FakeSearcher:
        def build_query_bundle(self, **kwargs):
            return make_query_bundle(intent="general")

        def search(self, query_bundle):
            return type("Result", (), {"candidates": [], "debug": {"mocked": True}})()

    citation = make_candidate(candidate_id="doc1:p3-3:paragraph:1", score=0.8)
    evidence = EvidenceSelection([citation], [citation], [], [])
    context = ContextBuilder(token_budget=1200).build(make_query_bundle(intent="general"), evidence)

    class FakeReranker:
        def rerank(self, query_bundle, candidates):
            return evidence

    class FakeContextBuilder:
        def build(self, query_bundle, evidence):
            return context

    engine = QueryEngine(
        qdrant=FakeQdrantManager(),
        text_embedder=FakeTextEmbedder(),
        searcher=FakeSearcher(),
        reranker=FakeReranker(),
        context_builder=FakeContextBuilder(),
        llm_client=FakeLLMClient(available=False),
    )

    response = engine.query(query="test")
    assert response["enough_evidence"] is True
    assert "Answer generation is not configured" in response["answer"]


def test_api_query_and_stream_endpoints(monkeypatch):
    class FakeApiEngine:
        def __init__(self):
            self.qdrant = type(
                "Q",
                (),
                {
                    "client": type("Client", (), {"get_collections": lambda self: []})(),
                    "text_collection": "text_chunks",
                    "image_collection": "image_chunks",
                },
            )()
            self.llm_client = FakeLLMClient(available=True)

        def query(self, **kwargs):
            return {
                "answer": "Answer text",
                "citations": [
                    {
                        "id": "C1",
                        "kind": "paragraph",
                        "doc_id": "doc1",
                        "source_file": "sample.pdf",
                        "page_numbers": [1],
                        "section_path": ["Chapter 1"],
                        "snippet": "snippet",
                    }
                ],
                "images": [],
                "tables": [],
                "enough_evidence": True,
                "query_intent": "general",
                "model": "fake-llm",
                "timings": {"retrieval_sec": 0.01},
            }

        def stream_query(self, **kwargs):
            yield {"event": "token", "data": "Hello "}
            yield {"event": "token", "data": "world"}
            yield {"event": "final", "data": self.query(**kwargs)}

    api_main.get_query_engine.cache_clear()
    monkeypatch.setattr(api_main, "get_query_engine", lambda: FakeApiEngine())
    client = TestClient(api_main.app)

    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Answer text"

    with client.stream("POST", "/query/stream", json={"query": "test"}) as stream_response:
        assert stream_response.status_code == 200
        events = list(stream_response.iter_lines())
    joined = "\n".join(line for line in events if line)
    assert "event: token" in joined
    assert "event: final" in joined
    assert "Hello " in joined
    assert "\"answer\": \"Answer text\"" in joined
