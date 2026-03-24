"""
tests/test_qdrant.py

Run from backend/ with:
    python -m pytest tests/test_qdrant.py -v
or just:
    python tests/test_qdrant.py

Tests:
  1. Connection to Qdrant Cloud
  2. Collection creation
  3. Upsert a dummy text chunk
  4. Upsert a dummy image chunk
  5. Hybrid search returns results
  6. Image search returns results
  7. Collection info prints correctly
  8. Delete by source works
"""

import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db.qdrant_client import QdrantManager

DENSE_DIM = 1024


def random_vector():
    """Generate a random 1024-dim unit vector to simulate bge-m3 output."""
    v = [random.uniform(-1, 1) for _ in range(DENSE_DIM)]
    norm = sum(x**2 for x in v) ** 0.5
    return [x / norm for x in v]


def random_sparse(n_terms=20):
    """Generate random sparse vector indices + values."""
    indices = random.sample(range(0, 30000), n_terms)
    values = [random.uniform(0, 1) for _ in range(n_terms)]
    return sorted(indices), values


def test_1_connection():
    print("\n[Test 1] Connecting to Qdrant Cloud...")
    manager = QdrantManager()
    assert manager.client is not None
    print("  PASS — connected")
    return manager


def test_2_create_collections(manager):
    print("\n[Test 2] Creating collections (recreate=True for clean test)...")
    manager.create_collections(recreate=True)
    collections = [c.name for c in manager.client.get_collections().collections]
    assert "text_chunks" in collections, "text_chunks not found"
    assert "image_chunks" in collections, "image_chunks not found"
    print("  PASS — both collections exist")


def test_3_upsert_text(manager):
    print("\n[Test 3] Upserting 3 dummy text chunks...")
    indices, values = random_sparse()
    points = [
        {
            "id": i,
            "dense_vector": random_vector(),
            "sparse_indices": indices,
            "sparse_values": values,
            "payload": {
                "original_text": f"पित्तं दाहकर्मणि प्रमुखम् — chunk {i}",
                "normalized_text": f"pittam dahakarmani pramukham chunk {i}",
                "page_number": i + 1,
                "source_file": "test_charaka.pdf",
                "block_type": "paragraph",
                "language": "sanskrit",
                "shloka_number": f"1.{i}",
            },
        }
        for i in range(1, 4)
    ]
    manager.upsert_text_chunks(points)
    count = manager.client.count("text_chunks").count
    assert count == 3, f"Expected 3 points, got {count}"
    print(f"  PASS — {count} text chunks upserted")


def test_4_upsert_images(manager):
    print("\n[Test 4] Upserting 2 dummy image chunks...")
    points = [
        {
            "id": 100 + i,
            "dense_vector": random_vector(),
            "payload": {
                "caption": f"Fig {i}. Marma points diagram from Charaka Samhita page {i * 10}",
                "image_path": f"data/images/test_charaka_page{i * 10}.png",
                "page_number": i * 10,
                "source_file": "test_charaka.pdf",
            },
        }
        for i in range(1, 3)
    ]
    manager.upsert_image_chunks(points)
    count = manager.client.count("image_chunks").count
    assert count == 2, f"Expected 2 points, got {count}"
    print(f"  PASS — {count} image chunks upserted")


def test_5_hybrid_search(manager):
    print("\n[Test 5] Hybrid search on text_chunks...")
    indices, values = random_sparse()
    results = manager.hybrid_search_text(
        dense_vector=random_vector(),
        sparse_indices=indices,
        sparse_values=values,
        top_k=3,
    )
    assert len(results) > 0, "No results returned"
    assert "_score" in results[0], "Score missing from result"
    assert "original_text" in results[0], "Payload missing from result"
    print(f"  PASS — {len(results)} results returned")
    print(f"  Top result: {results[0]['original_text'][:60]}...")


def test_6_image_search(manager):
    print("\n[Test 6] Dense search on image_chunks...")
    results = manager.search_images(dense_vector=random_vector(), top_k=2)
    assert len(results) > 0, "No image results returned"
    assert "caption" in results[0], "Caption missing from result"
    print(f"  PASS — {len(results)} image results returned")
    print(f"  Top result: {results[0]['caption'][:60]}...")


def test_7_collection_info(manager):
    print("\n[Test 7] Collection info...")
    manager.get_collection_info()
    print("  PASS — info printed above")


def test_8_delete_by_source(manager):
    print("\n[Test 8] Delete by source file...")
    manager.delete_by_source("test_charaka.pdf")
    text_count = manager.client.count("text_chunks").count
    image_count = manager.client.count("image_chunks").count
    assert text_count == 0, f"Expected 0 text chunks after delete, got {text_count}"
    assert image_count == 0, f"Expected 0 image chunks after delete, got {image_count}"
    print(f"  PASS — all points deleted (text={text_count}, image={image_count})")


if __name__ == "__main__":
    print("=" * 50)
    print("QdrantManager — Sanity Test Suite")
    print("=" * 50)

    passed = 0
    failed = 0

    try:
        manager = test_1_connection()
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")
        print("\nCannot continue without connection. Check QDRANT_URL and QDRANT_API_KEY in .env")
        sys.exit(1)

    tests = [
        test_2_create_collections,
        test_3_upsert_text,
        test_4_upsert_images,
        test_5_hybrid_search,
        test_6_image_search,
        test_7_collection_info,
        test_8_delete_by_source,
    ]

    for test_fn in tests:
        try:
            test_fn(manager)
            passed += 1
        except Exception as e:
            print(f"  FAIL — {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\nAll tests passed. QdrantManager is ready.")
        print("Commit: feat(vector_db): qdrant cloud connection + collection setup")
    else:
        print("\nSome tests failed. Fix before moving to next file.")