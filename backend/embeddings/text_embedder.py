"""
embeddings/text_embedder.py

Requirements:
  pip install FlagEmbedding numpy
"""

from __future__ import annotations

import os
from typing import Any

from FlagEmbedding import BGEM3FlagModel

from embeddings.model_factory import build_bge_m3_model, embedding_batch_size


class TextEmbedder:
	"""Hybrid embedder for text chunks using one shared BGE-M3 configuration."""

	def __init__(self, model: BGEM3FlagModel | None = None) -> None:
		self.model = model or build_bge_m3_model()
		self.default_batch_size = embedding_batch_size(default=12)

	def embed(self, texts: list[str], batch_size: int | None = None) -> list[dict[str, Any]]:
		"""Return dense + sparse vectors for each input text."""
		if not texts:
			return []

		cleaned = [str(t or "").strip() for t in texts]
		cleaned = [text for text in cleaned if text]
		if not cleaned:
			return []

		encoded = self.model.encode(
			cleaned,
			batch_size=int(batch_size or self.default_batch_size),
			return_dense=True,
			return_sparse=True,
			return_colbert_vecs=False,
		)

		dense_vecs = encoded.get("dense_vecs", [])
		lexical = encoded.get("lexical_weights", [])

		rows: list[dict[str, Any]] = []
		for i in range(len(cleaned)):
			sparse_map = lexical[i] if i < len(lexical) else {}
			sparse_indices = [int(token_id) for token_id in sparse_map.keys()]
			sparse_values = [float(weight) for weight in sparse_map.values()]

			rows.append(
				{
					"dense_vector": dense_vecs[i].tolist(),
					"sparse_indices": sparse_indices,
					"sparse_values": sparse_values,
				}
			)

		return rows


if __name__ == "__main__":
	embedder = TextEmbedder()
	sample = ["test Sanskrit text"]
	vectors = embedder.embed(sample)
	first = vectors[0]
	print(f"Dense vector dim: {len(first['dense_vector'])}")
	print(f"Sparse terms: {len(first['sparse_indices'])}")
