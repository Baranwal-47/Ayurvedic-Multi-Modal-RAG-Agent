"""Convert chunk and image records into Qdrant-ready point dictionaries."""

from __future__ import annotations

from typing import Any


class QdrantMapper:
    """Map internal ingestion records to Qdrant point payloads."""

    def map_text_points(self, chunks: list[dict[str, Any]], vectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            payload = {key: value for key, value in chunk.items() if key != "text_for_embedding"}
            points.append(
                {
                    "id": chunk["chunk_id"],
                    "dense_vector": vector["dense_vector"],
                    "sparse_indices": vector["sparse_indices"],
                    "sparse_values": vector["sparse_values"],
                    "payload": payload,
                }
            )
        return points

    def map_image_points(self, images: list[dict[str, Any]], dense_vectors: list[list[float]]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for image, dense_vector in zip(images, dense_vectors):
            payload = dict(image)
            points.append(
                {
                    "id": image["image_id"],
                    "dense_vector": dense_vector,
                    "payload": payload,
                }
            )
        return points
