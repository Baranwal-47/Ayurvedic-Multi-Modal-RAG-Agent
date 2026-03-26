"""
embeddings/text_embedder.py

Requirements:
  pip install FlagEmbedding numpy
"""

from __future__ import annotations

from typing import Any

from FlagEmbedding import BGEM3FlagModel


class TextEmbedder:
	"""Hybrid embedder for text chunks using BAAI/bge-m3 via FlagEmbedding."""

	def __init__(self, model: BGEM3FlagModel | None = None) -> None:
		self.model = model or BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

	def embed(self, texts: list[str], batch_size: int = 12) -> list[dict[str, Any]]:
		"""Return dense + sparse vectors for each input text."""
		if not texts:
			return []

		cleaned = [str(t or "").strip() for t in texts]
		encoded = self.model.encode(
			cleaned,
			batch_size=batch_size,
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
