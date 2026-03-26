"""
embeddings/image_embedder.py

Requirements:
  pip install FlagEmbedding numpy
"""

from __future__ import annotations

from FlagEmbedding import BGEM3FlagModel


class ImageEmbedder:
	"""Dense embedder for image captions using the same BGE-M3 model."""

	def __init__(self, model: BGEM3FlagModel | None = None) -> None:
		self.model = model or BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

	def embed(self, captions: list[str], batch_size: int = 12) -> list[list[float]]:
		"""Return dense vectors for image captions."""
		if not captions:
			return []

		cleaned = [str(c or "").strip() for c in captions]
		encoded = self.model.encode(
			cleaned,
			batch_size=batch_size,
			return_dense=True,
			return_sparse=False,
			return_colbert_vecs=False,
		)

		dense_vecs = encoded.get("dense_vecs", [])
		return [dense_vecs[i].tolist() for i in range(len(cleaned))]


if __name__ == "__main__":
	embedder = ImageEmbedder()
	sample = ["Fig 3. Nasya therapy administration of medicated oil"]
	vectors = embedder.embed(sample)
	print(f"Dense vector dim: {len(vectors[0])}")
