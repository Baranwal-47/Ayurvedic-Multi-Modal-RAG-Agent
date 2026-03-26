"""
vector_db/qdrant_client.py

QdrantManager — handles all Qdrant Cloud operations for the Ayurveda RAG system.
Two collections:
  - text_chunks  : stores text with dense + sparse vectors (bge-m3 hybrid search)
  - image_chunks : stores image captions with dense vectors only
"""

import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchValue,
)

load_dotenv()

# Collection names
TEXT_COLLECTION = os.getenv("QDRANT_TEXT_COLLECTION", "text_chunks")
IMAGE_COLLECTION = os.getenv("QDRANT_IMAGE_COLLECTION", "image_chunks")

# bge-m3 dense vector dimension
DENSE_DIM = 1024


class QdrantManager:
    def __init__(self):
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url:
            raise ValueError("QDRANT_URL not set in .env")
        if not api_key:
            raise ValueError("QDRANT_API_KEY not set in .env")

        self.client = QdrantClient(url=url, api_key=api_key)
        self.text_collection = TEXT_COLLECTION
        self.image_collection = IMAGE_COLLECTION
        self._collection_aliases = {
            "text_chunks": self.text_collection,
            "image_chunks": self.image_collection,
            self.text_collection: self.text_collection,
            self.image_collection: self.image_collection,
        }
        print(f"[QdrantManager] Connected to {url}")

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

    def create_collections(self, recreate: bool = False):
        """
        Create text_chunks and image_chunks collections.
        Set recreate=True to wipe and rebuild (useful during development).
        """
        existing = [c.name for c in self.client.get_collections().collections]

        # --- text_chunks ---
        if TEXT_COLLECTION in existing:
            if recreate:
                self.client.delete_collection(TEXT_COLLECTION)
                print(f"[QdrantManager] Deleted existing collection: {TEXT_COLLECTION}")
            else:
                print(f"[QdrantManager] Collection already exists: {TEXT_COLLECTION}")

        if TEXT_COLLECTION not in existing or recreate:
            self.client.create_collection(
                collection_name=TEXT_COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            self.client.create_payload_index(
                collection_name=self.text_collection,
                field_name="source_file",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"[QdrantManager] Created collection: {TEXT_COLLECTION}")

        # --- image_chunks ---
        if IMAGE_COLLECTION in existing:
            if recreate:
                self.client.delete_collection(IMAGE_COLLECTION)
                print(f"[QdrantManager] Deleted existing collection: {IMAGE_COLLECTION}")
            else:
                print(f"[QdrantManager] Collection already exists: {IMAGE_COLLECTION}")

        if IMAGE_COLLECTION not in existing or recreate:
            self.client.create_collection(
                collection_name=IMAGE_COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
                },
            )
            self.client.create_payload_index(
                collection_name=self.image_collection,
                field_name="source_file",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"[QdrantManager] Created collection: {IMAGE_COLLECTION}")

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_text_chunks(self, points: list[dict]):
        """
        Upsert text chunks into text_chunks collection.

        Each point dict must have:
          - id          : str or int (unique)
          - dense_vector: list[float] (1024-dim)
          - sparse_indices: list[int]
          - sparse_values : list[float]
          - payload     : dict with keys like original_text, normalized_text,
                          page_number, source_file, block_type, language,
                          shloka_number (optional)
        """
        qdrant_points = []
        for p in points:
            qdrant_points.append(
                PointStruct(
                    id=p["id"],
                    vector={
                        "dense": p["dense_vector"],
                        "sparse": SparseVector(
                            indices=p["sparse_indices"],
                            values=p["sparse_values"],
                        ),
                    },
                    payload=p["payload"],
                )
            )

        self.client.upsert(collection_name=TEXT_COLLECTION, points=qdrant_points)
        print(f"[QdrantManager] Upserted {len(qdrant_points)} text chunks")

    def upsert_image_chunks(self, points: list[dict]):
        """
        Upsert image caption chunks into image_chunks collection.

        Each point dict must have:
          - id           : str or int (unique)
          - dense_vector : list[float] (1024-dim)
          - payload      : dict with keys like caption, image_path,
                           page_number, source_file
        """
        qdrant_points = [
            PointStruct(
                id=p["id"],
                vector={"dense": p["dense_vector"]},
                payload=p["payload"],
            )
            for p in points
        ]

        self.client.upsert(collection_name=IMAGE_COLLECTION, points=qdrant_points)
        print(f"[QdrantManager] Upserted {len(qdrant_points)} image chunks")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def hybrid_search_text(
        self,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 10,
        language_filter: str | None = None,
    ) -> list[dict]:
        """
        Hybrid search on text_chunks using dense + sparse vectors via RRF fusion.
        Optionally filter results by language metadata.
        Returns list of payload dicts with score attached.
        """
        query_filter = None
        if language_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="language",
                        match=MatchValue(value=language_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.text_collection,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        output = []
        for point in results.points:
            row = {"_score": point.score, "_id": point.id}
            row.update(point.payload)
            output.append(row)
        return output

    def search_images(
        self,
        dense_vector: list[float],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Simple dense search on image_chunks collection.
        Returns list of payload dicts.
        """
        results = self.client.query_points(
            collection_name=self.image_collection,
            query=dense_vector,
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        output = []
        for point in results.points:
            row = {"_score": point.score, "_id": point.id}
            row.update(point.payload)
            output.append(row)
        return output

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_collection_info(self):
        """Print basic stats for both collections."""
        for name in [TEXT_COLLECTION, IMAGE_COLLECTION]:
            try:
                info = self.client.get_collection(name)
                count = self.client.count(name).count
                print(f"[{name}] status={info.status} | vectors={count}")
            except Exception as e:
                print(f"[{name}] not found or error: {e}")

    def delete_by_source(self, source_file: str, collection: str | None = None) -> int:
        """
        Delete all points from a specific collection (or both collections when omitted)
        where payload.source_file == source_file.
        Returns the number of deleted points.
        """
        f = Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_file),
                )
            ]
        )

        if collection is not None:
            collection_names = [self._resolve_collection_name(collection)]
        else:
            collection_names = [self.text_collection, self.image_collection]

        total_deleted = 0
        for col in collection_names:
            try:
                before = self.client.count(col, count_filter=f).count
            except Exception:
                before = 0

            if before > 0:
                self.client.delete(collection_name=col, points_selector=f)

            print(f"[QdrantManager] Deleted points for '{source_file}' from {col}: {before}")
            total_deleted += int(before)

        return total_deleted

    def _resolve_collection_name(self, collection: str) -> str:
        name = str(collection or "").strip()
        resolved = self._collection_aliases.get(name)
        if not resolved:
            raise ValueError(
                f"Unknown collection '{collection}'. Expected one of: "
                f"{self.text_collection}, {self.image_collection}, text_chunks, image_chunks"
            )
        return resolved