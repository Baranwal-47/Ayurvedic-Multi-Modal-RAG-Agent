from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import main as api_main
from rag.context_builder import ContextBuilder
from rag.query_engine import LLMClient, QueryEngine
from retrieval.hybrid_search import HybridSearcher, QueryBundle, QueryFilters, RetrievalCandidate
from retrieval.reranker import CandidateReranker, EvidenceSelection


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


class RecordingTextEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.extend(list(texts))
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
        self.hybrid_search_text_calls = 0
        self.dense_search_text_calls = 0
        self.scroll_calls = 0
        self.retrieve_calls = 0
        self.last_text_top_k = None
        self.last_image_top_k = None
        self.last_retrieve_point_ids = []

    def hybrid_search_text(self, **kwargs):
        self.hybrid_search_text_calls += 1
        self.last_text_top_k = kwargs.get("top_k")
        return self._default_text_rows()

    @staticmethod
    def _default_text_rows():
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

    def search_text_dense(self, **kwargs):
        self.dense_search_text_calls += 1
        self.last_text_top_k = kwargs.get("top_k")
        return self._default_text_rows()

    def search_images(self, **kwargs):
        self.last_image_exclusions = kwargs.get("exclude_image_types")
        self.last_image_top_k = kwargs.get("top_k")
        return []

    def retrieve_points(self, *, collection, point_ids, include_vectors=False):
        assert collection == "image_chunks"
        self.retrieve_calls += 1
        self.last_retrieve_point_ids = list(point_ids)
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

    def scroll_points(self, *, collection, filters, limit, exclude_image_types=None, include_vectors=False):
        assert collection == "image_chunks"
        self.scroll_calls += 1
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


class IdentityNormalizer:
    def normalize(self, text, aggressive_latin_fold=False):
        return str(text)


class RescueStressQdrantManager(FakeQdrantManager):
    def hybrid_search_text(self, **kwargs):
        self.hybrid_search_text_calls += 1
        self.last_text_top_k = kwargs.get("top_k")
        return [
            {
                "_id": "text-1",
                "_score": 0.95,
                "chunk_id": "doc1:p3-3:paragraph:1",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_numbers": [3],
                "chunk_type": "paragraph",
                "text": "Linked images text.",
                "section_path": ["Chapter 1"],
                "heading_text": "Chapter 1",
                "image_ids": [
                    "doc1:p3:img:1",
                    "doc1:p3:img:2",
                    "doc1:p3:img:3",
                    "doc1:p3:img:4",
                ],
            },
            {
                "_id": "text-2",
                "_score": 0.84,
                "chunk_id": "doc1:p7-7:paragraph:2",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_numbers": [7],
                "chunk_type": "paragraph",
                "text": "Nearby page image text.",
                "section_path": ["Chapter 2"],
                "heading_text": "Chapter 2",
                "image_ids": [],
            },
        ]

    def search_text_dense(self, **kwargs):
        self.dense_search_text_calls += 1
        self.last_text_top_k = kwargs.get("top_k")
        return [
            {
                "_id": "text-1",
                "_score": 0.95,
                "chunk_id": "doc1:p3-3:paragraph:1",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_numbers": [3],
                "chunk_type": "paragraph",
                "text": "Linked images text.",
                "section_path": ["Chapter 1"],
                "heading_text": "Chapter 1",
                "image_ids": [
                    "doc1:p3:img:1",
                    "doc1:p3:img:2",
                    "doc1:p3:img:3",
                    "doc1:p3:img:4",
                ],
            },
            {
                "_id": "text-2",
                "_score": 0.84,
                "chunk_id": "doc1:p7-7:paragraph:2",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_numbers": [7],
                "chunk_type": "paragraph",
                "text": "Nearby page image text.",
                "section_path": ["Chapter 2"],
                "heading_text": "Chapter 2",
                "image_ids": [],
            },
        ]

    def search_images(self, **kwargs):
        self.last_image_exclusions = kwargs.get("exclude_image_types")
        self.last_image_top_k = kwargs.get("top_k")
        return []

    def retrieve_points(self, *, collection, point_ids, include_vectors=False):
        assert collection == "image_chunks"
        self.retrieve_calls += 1
        self.last_retrieve_point_ids = list(point_ids)
        rows = []
        for index, point_id in enumerate(point_ids, start=1):
            rows.append(
                {
                    "_id": f"image-linked-{index}",
                    "_score": 0.0,
                    "image_id": str(point_id),
                    "doc_id": "doc1",
                    "source_file": "sample.pdf",
                    "page_number": 3,
                    "image_type": "diagram",
                    "image_url": f"https://example.com/{point_id}.png",
                    "cloudinary_public_id": f"demo/{point_id}",
                    "linked_chunk_ids": ["doc1:p3-3:paragraph:1"],
                    "caption": "Linked figure",
                    "labels": ["label"],
                    "surrounding_text": "linked",
                    "section_path": ["Chapter 1"],
                }
            )
        return rows

    def scroll_points(self, *, collection, filters, limit, exclude_image_types=None, include_vectors=False):
        assert collection == "image_chunks"
        self.scroll_calls += 1
        page = int(filters.get("page_start") or 0)
        return [
            {
                "_id": f"image-prox-{page}",
                "_score": 0.0,
                "image_id": f"doc1:p{page}:img:prox",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_number": page,
                "image_type": "diagram",
                "image_url": f"https://example.com/prox-{page}.png",
                "cloudinary_public_id": f"demo/prox/{page}",
                "linked_chunk_ids": [],
                "caption": "Proximity figure",
                "labels": [],
                "surrounding_text": "proximity",
                "section_path": ["Chapter 2"],
            }
        ]


class VisualHitsQdrantManager(FakeQdrantManager):
    def hybrid_search_text(self, **kwargs):
        self.hybrid_search_text_calls += 1
        self.last_text_top_k = kwargs.get("top_k")
        return [
            {
                "_id": "text-visual-1",
                "_score": 0.9,
                "chunk_id": "doc1:p2-2:paragraph:1",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_numbers": [2],
                "chunk_type": "paragraph",
                "text": "Figure description",
                "section_path": ["Visual"],
                "heading_text": "Visual",
                "image_ids": [],
            }
        ]

    def search_text_dense(self, **kwargs):
        self.dense_search_text_calls += 1
        self.last_text_top_k = kwargs.get("top_k")
        return [
            {
                "_id": "text-visual-1",
                "_score": 0.9,
                "chunk_id": "doc1:p2-2:paragraph:1",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_numbers": [2],
                "chunk_type": "paragraph",
                "text": "Figure description",
                "section_path": ["Visual"],
                "heading_text": "Visual",
                "image_ids": [],
            }
        ]

    def search_images(self, **kwargs):
        self.last_image_exclusions = kwargs.get("exclude_image_types")
        self.last_image_top_k = kwargs.get("top_k")
        return [
            {
                "_id": "img-direct-1",
                "_score": 0.75,
                "image_id": "doc1:p2:img:1",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_number": 2,
                "image_type": "diagram",
                "image_url": "https://example.com/direct-1.png",
                "cloudinary_public_id": "demo/direct-1",
                "linked_chunk_ids": [],
                "caption": "Direct figure one",
                "labels": ["figure"],
                "surrounding_text": "direct",
                "section_path": ["Visual"],
            },
            {
                "_id": "img-direct-2",
                "_score": 0.71,
                "image_id": "doc1:p2:img:2",
                "doc_id": "doc1",
                "source_file": "sample.pdf",
                "page_number": 2,
                "image_type": "diagram",
                "image_url": "https://example.com/direct-2.png",
                "cloudinary_public_id": "demo/direct-2",
                "linked_chunk_ids": [],
                "caption": "Direct figure two",
                "labels": ["figure"],
                "surrounding_text": "direct",
                "section_path": ["Visual"],
            },
        ]

    def retrieve_points(self, *, collection, point_ids, include_vectors=False):
        assert collection == "image_chunks"
        self.retrieve_calls += 1
        self.last_retrieve_point_ids = list(point_ids)
        return []

    def scroll_points(self, *, collection, filters, limit, exclude_image_types=None, include_vectors=False):
        assert collection == "image_chunks"
        self.scroll_calls += 1
        return []


def make_query_bundle(intent: str = "general", query: str = "test query") -> QueryBundle:
    return QueryBundle(
        query=query,
        normalized_query=query,
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
    payload_overrides=None,
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
            **(payload_overrides or {}),
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


def test_visual_query_returns_debug_timing_keys_and_reduced_pools():
    qdrant = VisualHitsQdrantManager()
    searcher = HybridSearcher(qdrant=qdrant, text_embedder=FakeTextEmbedder())

    bundle = searcher.build_query_bundle(query="show me the figure", include_debug=True)
    result = searcher.search(bundle)

    assert bundle.image_top_k == 6
    assert result.debug["text_search_k"] <= 12
    assert result.debug["retrieval_counts"]["image_hits_direct"] == 2

    timing = result.debug["retrieval_timing"]
    counts = result.debug["retrieval_counts"]
    assert set(
        [
            "embed_query_sec",
            "text_original_sec",
            "text_normalized_sec",
            "image_direct_sec",
            "linked_image_rescue_sec",
            "page_proximity_rescue_sec",
            "hydrate_points_sec",
            "merge_sort_sec",
        ]
    ).issubset(set(timing.keys()))
    assert set(
        [
            "text_hits_original",
            "text_hits_normalized",
            "image_hits_direct",
            "rescued_image_hits",
            "rescued_page_hits",
            "candidate_count_merged",
        ]
    ).issubset(set(counts.keys()))
    assert result.debug["candidate_count_before_prefilter"] >= result.debug["candidate_count_after_prefilter"]


def test_unchanged_normalized_query_skips_normalized_search():
    qdrant = FakeQdrantManager()
    searcher = HybridSearcher(
        qdrant=qdrant,
        text_embedder=FakeTextEmbedder(),
        normalizer=IdentityNormalizer(),
    )

    bundle = searcher.build_query_bundle(query="plain query", include_debug=True)
    result = searcher.search(bundle)

    assert (qdrant.hybrid_search_text_calls + qdrant.dense_search_text_calls) == 1
    assert result.debug["text_hits_normalized"] == 0
    assert result.debug["normalized_text_search_ran"] is False


def test_short_query_uses_fast_route_and_dense_only_retrieval():
    qdrant = FakeQdrantManager()
    searcher = HybridSearcher(qdrant=qdrant, text_embedder=FakeTextEmbedder())

    bundle = searcher.build_query_bundle(query="what is rasa", include_debug=True)
    result = searcher.search(bundle)

    assert bundle.route == "fast"
    assert result.debug["text_search_mode"] == "dense_only"
    assert qdrant.dense_search_text_calls >= 1
    assert qdrant.hybrid_search_text_calls == 0


def test_deep_route_override_for_short_visual_keyword_query():
    qdrant = VisualHitsQdrantManager()
    searcher = HybridSearcher(qdrant=qdrant, text_embedder=FakeTextEmbedder())

    bundle = searcher.build_query_bundle(query="palika yantra diagram", include_debug=True)
    result = searcher.search(bundle)

    assert bundle.route == "deep"
    assert bundle.image_top_k == 6
    assert result.debug["route"] == "deep"


def test_sufficient_direct_images_skip_page_proximity_rescue():
    qdrant = VisualHitsQdrantManager()
    searcher = HybridSearcher(qdrant=qdrant, text_embedder=FakeTextEmbedder())

    bundle = searcher.build_query_bundle(query="show me figure", include_debug=True)
    result = searcher.search(bundle)

    assert result.debug["retrieval_counts"]["image_hits_direct"] >= 2
    assert result.debug["retrieval_counts"]["rescued_page_hits"] == 0
    assert qdrant.scroll_calls == 0


def test_poor_direct_image_recall_triggers_capped_rescue_and_batched_hydration():
    qdrant = RescueStressQdrantManager()
    searcher = HybridSearcher(qdrant=qdrant, text_embedder=FakeTextEmbedder())

    bundle = searcher.build_query_bundle(query="show me the figure", include_debug=True)
    result = searcher.search(bundle)
    counts = result.debug["retrieval_counts"]

    assert counts["image_hits_direct"] == 0
    assert counts["rescued_image_hits"] <= 3
    assert counts["rescued_page_hits"] >= 1
    assert qdrant.retrieve_calls == 1
    assert len(qdrant.last_retrieve_point_ids) <= 3
    assert qdrant.scroll_calls <= 2


def test_repeated_identical_query_keeps_stable_candidate_ordering():
    qdrant = RescueStressQdrantManager()
    searcher = HybridSearcher(qdrant=qdrant, text_embedder=FakeTextEmbedder())

    bundle = searcher.build_query_bundle(query="show me the figure", include_debug=True)
    first = searcher.search(bundle)
    second = searcher.search(bundle)

    assert first.debug["candidate_ids"] == second.debug["candidate_ids"]


def test_simple_route_detected_for_greeting_query():
    searcher = HybridSearcher(qdrant=FakeQdrantManager(), text_embedder=FakeTextEmbedder())
    bundle = searcher.build_query_bundle(query="hello")

    assert bundle.intent == "chitchat"
    assert bundle.route == "simple"


def test_candidate_text_is_not_reembedded_during_prefilter():
    embedder = RecordingTextEmbedder()
    qdrant = FakeQdrantManager()
    searcher = HybridSearcher(qdrant=qdrant, text_embedder=embedder)

    bundle = searcher.build_query_bundle(query="what is rasa", include_debug=True)
    searcher.search(bundle)

    assert embedder.calls == ["what is rasa"]


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


def test_context_builder_includes_visual_image_cards_even_without_keyword_overlap():
    builder = ContextBuilder(token_budget=1200)
    citation = make_candidate(candidate_id="doc1:p3-3:paragraph:1", score=0.6)
    image = make_candidate(
        candidate_id="doc1:p3:img:1",
        score=0.55,
        kind="image",
        image_url="https://example.com/unrelated.png",
        linked_ids=[],
    )
    image.caption = "Portrait of Hakim Muhammad Ajmal"
    image.text = "Biography image from another document"
    image.section_path = ["Biography"]

    context = builder.build(
        make_query_bundle(intent="visual", query="show me palika yantra figure"),
        EvidenceSelection([citation, image], [citation], [image], []),
    )

    assert len(context.images) == 1


def test_context_builder_includes_shloka_like_evidence_for_shloka_queries():
    builder = ContextBuilder(token_budget=1200)
    shloka_like = make_candidate(
        candidate_id="doc1:p8-8:paragraph:4",
        score=0.0,
        chunk_type="paragraph",
        payload_overrides={"is_shloka": True, "shloka_number": "12.4", "scripts": ["Deva"]},
    )
    shloka_like.text = "चपके वृत्तुलं लोहैः विनताग्रोर्ध्वदण्डकम् ।\nएतद्वि पालिकायन्त्रं बलिजारणहेतवे ॥"
    shloka_like.snippet = shloka_like.text

    context = builder.build(
        make_query_bundle(intent="shloka", query="give me some shlokas"),
        EvidenceSelection([shloka_like], [shloka_like], [], []),
    )

    assert context.citations
    assert context.enough_evidence is True


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


def test_simple_route_returns_greeting_when_llm_unavailable():
    engine = QueryEngine(
        qdrant=FakeQdrantManager(),
        text_embedder=FakeTextEmbedder(),
        llm_client=FakeLLMClient(available=False),
    )

    response = engine.query(query="hello")
    assert response["answer"] == "Hello! How can I help you?"


def test_fast_route_skips_reranker_call():
    class TrackingReranker:
        def __init__(self):
            self.called = False
            self.model_name = "tracking-reranker"

        def prewarm(self):
            return None

        def rerank(self, query_bundle, candidates):
            self.called = True
            return EvidenceSelection([], [], [], [], debug={})

    reranker = TrackingReranker()
    engine = QueryEngine(
        qdrant=FakeQdrantManager(),
        text_embedder=FakeTextEmbedder(),
        reranker=reranker,
        llm_client=FakeLLMClient(available=False),
    )

    response = engine.query(query="what is rasa")
    assert reranker.called is False
    assert response["timings"]["rerank_sec"] == 0.0


def test_context_builder_prioritizes_top_image_context_when_image_is_ranked_first():
    builder = ContextBuilder(token_budget=1200)

    top_image = make_candidate(
        candidate_id="doc1:p4:img:1",
        score=0.92,
        kind="image",
        image_url="https://example.com/top-image.png",
        linked_ids=[],
    )
    top_image.caption = "Image caption"
    top_image.text = "Primary surrounding text should lead context"

    lower_text = make_candidate(
        candidate_id="doc1:p4-4:paragraph:1",
        score=0.61,
        kind="text",
    )

    evidence = EvidenceSelection(
        reranked_candidates=[top_image, lower_text],
        citation_candidates=[lower_text],
        image_candidates=[top_image],
        table_candidates=[],
    )
    context = builder.build(make_query_bundle(intent="general", query="explain this diagram"), evidence)

    assert context.images
    assert context.enough_evidence is True
    assert "Primary surrounding text should lead context" in context.user_prompt
    assert context.user_prompt.index("Figure Evidence") < context.user_prompt.index("Text Evidence")


def test_reranker_model_load_log_prints_once_per_process(monkeypatch, capsys):
    from retrieval import reranker as reranker_module

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False, truncation=True, max_length=320):
            _ = add_special_tokens, truncation
            tokens = [ord(ch) % 255 for ch in str(text)]
            return tokens[: max_length]

        def decode(self, token_ids, skip_special_tokens=True):
            _ = skip_special_tokens
            return "x" * len(token_ids)

    class FakeCrossEncoder:
        init_calls = 0

        def __init__(self, model_name, trust_remote_code=True):
            _ = model_name, trust_remote_code
            FakeCrossEncoder.init_calls += 1
            self.model = type("M", (), {"device": "cpu"})()
            self.tokenizer = FakeTokenizer()

        def predict(self, pairs, batch_size=8, show_progress_bar=False):
            _ = batch_size, show_progress_bar
            return [0.1 for _ in pairs]

    monkeypatch.setenv("RERANKER_LOG_LOAD", "true")
    monkeypatch.setattr(reranker_module, "_RERANKER_MODEL_SINGLETON", None)
    monkeypatch.setattr(reranker_module, "_RERANKER_MODEL_LOAD_COUNT", 0)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))

    reranker_one = CandidateReranker(model_name="fake-reranker")
    reranker_two = CandidateReranker(model_name="fake-reranker")

    reranker_one._load_model()
    reranker_two._load_model()

    output = capsys.readouterr().out
    assert output.count("Loading reranker model...") == 1
    assert FakeCrossEncoder.init_calls == 1


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
