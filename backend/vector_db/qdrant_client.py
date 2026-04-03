"""Qdrant collection management and upsert/search helpers for the ingestion rebuild."""

from __future__ import annotations

import os
import uuid
from typing import Any, Iterable
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    Range,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env", encoding="utf-8-sig")

TEXT_COLLECTION = os.getenv("QDRANT_TEXT_COLLECTION", "text_chunks")
IMAGE_COLLECTION = os.getenv("QDRANT_IMAGE_COLLECTION", "image_chunks")
DENSE_DIM = 1024


class QdrantManager:
    """Own Qdrant collection lifecycle and point operations."""

    def __init__(self) -> None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        timeout_sec = float(os.getenv("QDRANT_TIMEOUT_SEC", "120"))

        if not url:
            raise ValueError("QDRANT_URL not set in .env")
        if not api_key:
            raise ValueError("QDRANT_API_KEY not set in .env")

        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout_sec)
        self.text_collection = TEXT_COLLECTION
        self.image_collection = IMAGE_COLLECTION
        self._collection_aliases = {
            "text_chunks": self.text_collection,
            "image_chunks": self.image_collection,
            self.text_collection: self.text_collection,
            self.image_collection: self.image_collection,
        }

    def create_collections(self, recreate: bool = False) -> None:
        existing = {collection.name for collection in self.client.get_collections().collections}

        if recreate and self.text_collection in existing:
            self.client.delete_collection(self.text_collection)
            existing.discard(self.text_collection)
        if recreate and self.image_collection in existing:
            self.client.delete_collection(self.image_collection)
            existing.discard(self.image_collection)

        if self.text_collection not in existing:
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
            )
        self._create_payload_indexes(
            self.text_collection,
            {
                "doc_id": PayloadSchemaType.KEYWORD,
                "source_file": PayloadSchemaType.KEYWORD,
                "page_start": PayloadSchemaType.INTEGER,
                "page_end": PayloadSchemaType.INTEGER,
                "chunk_type": PayloadSchemaType.KEYWORD,
                "route": PayloadSchemaType.KEYWORD,
            },
        )

        if self.image_collection not in existing:
            self.client.create_collection(
                collection_name=self.image_collection,
                vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
            )
        self._create_payload_indexes(
            self.image_collection,
            {
                "doc_id": PayloadSchemaType.KEYWORD,
                "source_file": PayloadSchemaType.KEYWORD,
                "page_number": PayloadSchemaType.INTEGER,
                "image_type": PayloadSchemaType.KEYWORD,
            },
        )

    def upsert_text_chunks(self, points: list[dict]) -> None:
        if not points:
            return
        self.client.upsert(
            collection_name=self.text_collection,
            points=[
                PointStruct(
                    id=self._normalize_point_id(point["id"], kind="text"),
                    vector={
                        "dense": point["dense_vector"],
                        "sparse": SparseVector(indices=point["sparse_indices"], values=point["sparse_values"]),
                    },
                    payload=point["payload"],
                )
                for point in points
            ],
        )

    def upsert_image_chunks(self, points: list[dict]) -> None:
        if not points:
            return
        self.client.upsert(
            collection_name=self.image_collection,
            points=[
                PointStruct(
                    id=self._normalize_point_id(point["id"], kind="image"),
                    vector={"dense": point["dense_vector"]},
                    payload=point["payload"],
                )
                for point in points
            ],
        )

    def hybrid_search_text(
        self,
        *,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        query_filter = self._search_filter(collection_name=self.text_collection, filters=filters)
        results = self.client.query_points(
            collection_name=self.text_collection,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
                Prefetch(query=SparseVector(indices=sparse_indices, values=sparse_values), using="sparse", limit=top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [self._point_to_row(point) for point in results.points]

    def search_images(
        self,
        *,
        dense_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        exclude_image_types: Iterable[str] | None = None,
    ) -> list[dict]:
        query_filter = self._search_filter(
            collection_name=self.image_collection,
            filters=filters,
            exclude_image_types=exclude_image_types,
        )
        results = self.client.query_points(
            collection_name=self.image_collection,
            query=dense_vector,
            using="dense",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [self._point_to_row(point) for point in results.points]

    def scroll_points(
        self,
        *,
        collection: str,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
        exclude_image_types: Iterable[str] | None = None,
    ) -> list[dict]:
        collection_name = self._resolve_collection_name(collection)
        query_filter = self._search_filter(
            collection_name=collection_name,
            filters=filters,
            exclude_image_types=exclude_image_types,
        )
        points, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=max(1, int(limit)),
            with_payload=True,
            with_vectors=False,
        )
        return [self._point_to_row(point) for point in points]

    def retrieve_points(self, *, collection: str, point_ids: Iterable[object]) -> list[dict]:
        collection_name = self._resolve_collection_name(collection)
        point_kind = "text" if collection_name == self.text_collection else "image"
        normalized_ids = [self._normalize_point_id(point_id, kind=point_kind) for point_id in point_ids]
        if not normalized_ids:
            return []
        points = self.client.retrieve(
            collection_name=collection_name,
            ids=normalized_ids,
            with_payload=True,
            with_vectors=False,
        )
        return [self._point_to_row(point) for point in points]

    def delete_by_doc_id(self, doc_id: str, collection: str | None = None) -> int:
        return self._delete_by_filter(Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]), collection)

    def delete_by_source(self, source_file: str, collection: str | None = None) -> int:
        return self._delete_by_filter(Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]), collection)

    def get_collection_info(self) -> None:
        for collection_name in [self.text_collection, self.image_collection]:
            info = self.client.get_collection(collection_name)
            count = self.client.count(collection_name).count
            print(f"[{collection_name}] status={info.status} count={count}")

    def _delete_by_filter(self, selector: Filter, collection: str | None = None) -> int:
        collections = [self._resolve_collection_name(collection)] if collection else [self.text_collection, self.image_collection]
        total_deleted = 0
        for collection_name in collections:
            before = int(self.client.count(collection_name=collection_name, count_filter=selector).count)
            if before > 0:
                self.client.delete(collection_name=collection_name, points_selector=selector)
            total_deleted += before
        return total_deleted

    def _create_payload_indexes(self, collection_name: str, fields: dict[str, PayloadSchemaType]) -> None:
        for field_name, schema in fields.items():
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" in message or "existing index" in message:
                    continue
                raise

    def _search_filter(
        self,
        *,
        collection_name: str,
        filters: dict[str, Any] | None = None,
        exclude_image_types: Iterable[str] | None = None,
    ) -> Filter | None:
        filters = dict(filters or {})
        must: list[FieldCondition] = []
        must_not: list[FieldCondition] = []

        doc_id = self._clean_scalar(filters.get("doc_id"))
        if doc_id:
            must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

        source_file = self._clean_scalar(filters.get("source_file"))
        if source_file:
            must.append(FieldCondition(key="source_file", match=MatchValue(value=source_file)))

        languages = self._clean_list(filters.get("languages"))
        if languages:
            must.append(FieldCondition(key="languages", match=MatchAny(any=languages)))

        scripts = self._clean_list(filters.get("scripts"))
        if scripts:
            must.append(FieldCondition(key="scripts", match=MatchAny(any=scripts)))

        chunk_types = self._clean_list(filters.get("chunk_types"))
        if chunk_types and collection_name == self.text_collection:
            must.append(FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types)))

        image_types = self._clean_list(filters.get("image_types"))
        if image_types and collection_name == self.image_collection:
            must.append(FieldCondition(key="image_type", match=MatchAny(any=image_types)))

        section_path = self._clean_list(filters.get("section_path"))
        if section_path:
            must.append(FieldCondition(key="section_path", match=MatchAny(any=section_path)))

        page_start = self._clean_int(filters.get("page_start"))
        page_end = self._clean_int(filters.get("page_end"))
        if collection_name == self.text_collection:
            if page_start is not None:
                must.append(FieldCondition(key="page_end", range=Range(gte=page_start)))
            if page_end is not None:
                must.append(FieldCondition(key="page_start", range=Range(lte=page_end)))
        else:
            if page_start is not None:
                must.append(FieldCondition(key="page_number", range=Range(gte=page_start)))
            if page_end is not None:
                must.append(FieldCondition(key="page_number", range=Range(lte=page_end)))

        for image_type in self._clean_list(exclude_image_types):
            if collection_name == self.image_collection:
                must_not.append(FieldCondition(key="image_type", match=MatchValue(value=image_type)))

        if not must and not must_not:
            return None
        return Filter(must=must or None, must_not=must_not or None)

    @staticmethod
    def _point_to_row(point) -> dict:
        row = {"_score": float(getattr(point, "score", 0.0) or 0.0), "_id": point.id}
        row.update(point.payload)
        return row

    def _resolve_collection_name(self, collection: str | None) -> str:
        name = str(collection or "").strip()
        resolved = self._collection_aliases.get(name)
        if not resolved:
            raise ValueError(f"Unknown collection '{collection}'")
        return resolved

    @staticmethod
    def _normalize_point_id(point_id: object, *, kind: str) -> str | int:
        if isinstance(point_id, int):
            return point_id

        raw = str(point_id or "").strip()
        if not raw:
            raise ValueError("Point id cannot be empty")

        try:
            return str(uuid.UUID(raw))
        except Exception:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{kind}:{raw}"))

    @staticmethod
    def _clean_scalar(value: object) -> str | None:
        raw = str(value or "").strip()
        return raw or None

    @staticmethod
    def _clean_list(values: object) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            cleaned = [values.strip()]
        else:
            cleaned = [str(value).strip() for value in values if str(value or "").strip()]
        seen: set[str] = set()
        result: list[str] = []
        for value in cleaned:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    @staticmethod
    def _clean_int(value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except Exception:
            return None
