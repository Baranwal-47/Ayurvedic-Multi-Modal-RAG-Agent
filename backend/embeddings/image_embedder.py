"""
embeddings/image_embedder.py

Requirements:
  pip install FlagEmbedding numpy
"""

from __future__ import annotations

from FlagEmbedding import BGEM3FlagModel

from embeddings.model_factory import build_bge_m3_model, embedding_batch_size


class ImageEmbedder:
	"""Dense embedder for image retrieval text such as captions and surrounding context."""

	def __init__(self, model: BGEM3FlagModel | None = None) -> None:
		self.model = model or build_bge_m3_model()
		self.default_batch_size = embedding_batch_size(default=12)

	def embed(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
		"""Return dense vectors for image-associated text."""
		if not texts:
			return []

		cleaned = [str(text or "").strip() for text in texts]
		cleaned = [text for text in cleaned if text]
		if not cleaned:
			return []

		encoded = self.model.encode(
			cleaned,
			batch_size=int(batch_size or self.default_batch_size),
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
