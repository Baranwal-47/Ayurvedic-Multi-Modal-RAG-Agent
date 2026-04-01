"""Sanity checks for QdrantManager against the rebuilt payload contract."""

from __future__ import annotations

import os
import random
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db.qdrant_client import QdrantManager

DENSE_DIM = 1024


def test_0_point_id_normalization():
    normalized_text = QdrantManager._normalize_point_id("b6a7db2cfb401fff:p1-1:paragraph:1", kind="text")
    normalized_image = QdrantManager._normalize_point_id("b6a7db2cfb401fff:p1:img:1", kind="image")

    uuid.UUID(str(normalized_text))
    uuid.UUID(str(normalized_image))
    assert normalized_text != normalized_image


def random_vector():
    values = [random.uniform(-1, 1) for _ in range(DENSE_DIM)]
    norm = sum(value**2 for value in values) ** 0.5
    return [value / norm for value in values]


def random_sparse(n_terms: int = 20):
    indices = random.sample(range(0, 30000), n_terms)
    values = [random.uniform(0, 1) for _ in range(n_terms)]
    return sorted(indices), values


def test_1_connection():
    manager = QdrantManager()
    assert manager.client is not None
    return manager


def test_2_create_collections(manager):
    manager.create_collections(recreate=True)
    collections = [collection.name for collection in manager.client.get_collections().collections]
    assert manager.text_collection in collections
    assert manager.image_collection in collections


def test_3_upsert_text(manager):
    indices, values = random_sparse()
    points = []
    for index in range(1, 4):
        chunk_id = f"doc1:p{index}-{index}:paragraph:{index}"
        points.append(
            {
                "id": chunk_id,
                "dense_vector": random_vector(),
                "sparse_indices": indices,
                "sparse_values": values,
                "payload": {
                    "chunk_id": chunk_id,
                    "doc_id": "doc1",
                    "source_file": "test_charaka.pdf",
                    "page_start": index,
                    "page_end": index,
                    "page_numbers": [index],
                    "chunk_type": "paragraph",
                    "text": f"sample text chunk {index}",
                    "normalized_text": f"sample text chunk {index}",
                    "section_path": ["Chapter 1"],
                    "heading_text": "Chapter 1",
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "layout_type": "single",
                    "route": "digitized",
                    "ocr_source": None,
                    "ocr_confidence": None,
                    "is_shloka": False,
                    "shloka_id": None,
                    "is_multilingual": False,
                    "has_image_context": False,
                    "image_ids": [],
                    "source_unit_ids": [f"u{index}"],
                    "bridge_source_chunk_ids": [],
                    "shloka_number": None,
                    "table_id": None,
                },
            }
        )
    manager.upsert_text_chunks(points)
    assert manager.client.count(manager.text_collection).count == 3


def test_4_upsert_images(manager):
    points = []
    for index in range(1, 3):
        image_id = f"doc1:p{index}:img:{index}"
        points.append(
            {
                "id": image_id,
                "dense_vector": random_vector(),
                "payload": {
                    "image_id": image_id,
                    "doc_id": "doc1",
                    "source_file": "test_charaka.pdf",
                    "page_number": index,
                    "figure_index": index,
                    "image_type": "diagram",
                    "figure_bbox": [0.0, 0.0, 100.0, 100.0],
                    "caption": f"Fig {index}",
                    "caption_source": "explicit",
                    "labels": [],
                    "labels_sparse": True,
                    "surrounding_text": "",
                    "section_path": ["Chapter 1"],
                    "languages": ["en"],
                    "scripts": ["Latn"],
                    "linked_chunk_ids": [],
                    "caption_unit_ids": [],
                    "label_unit_ids": [],
                    "association_confidence": 1.0,
                    "cloudinary_public_id": f"demo/page_{index}/figure_{index}",
                    "image_url": f"https://example.com/figure_{index}.png",
                },
            }
        )
    manager.upsert_image_chunks(points)
    assert manager.client.count(manager.image_collection).count == 2


def test_5_hybrid_search(manager):
    indices, values = random_sparse()
    results = manager.hybrid_search_text(
        dense_vector=random_vector(),
        sparse_indices=indices,
        sparse_values=values,
        top_k=3,
    )
    assert results
    assert "text" in results[0]


def test_6_image_search(manager):
    results = manager.search_images(dense_vector=random_vector(), top_k=2)
    assert results
    assert "caption" in results[0]


def test_7_delete_by_doc_id(manager):
    deleted = manager.delete_by_doc_id("doc1")
    assert deleted >= 5
    assert manager.client.count(manager.text_collection).count == 0
    assert manager.client.count(manager.image_collection).count == 0


if __name__ == "__main__":
    manager = test_1_connection()
    test_2_create_collections(manager)
    test_3_upsert_text(manager)
    test_4_upsert_images(manager)
    test_5_hybrid_search(manager)
    test_6_image_search(manager)
    test_7_delete_by_doc_id(manager)
    print("Qdrant sanity checks passed.")
