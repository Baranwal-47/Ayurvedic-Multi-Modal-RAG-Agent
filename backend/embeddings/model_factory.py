"""Shared embedding model factory for ingestion and retrieval."""

from __future__ import annotations

import os

from FlagEmbedding import BGEM3FlagModel


def embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3").strip() or "BAAI/bge-m3"


def embedding_use_fp16() -> bool:
    return os.getenv("EMBEDDING_USE_FP16", "true").strip().lower() in {"1", "true", "yes"}


def embedding_batch_size(default: int = 12) -> int:
    raw = os.getenv("EMBEDDING_BATCH_SIZE", "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except Exception:
        return int(default)


def build_bge_m3_model() -> BGEM3FlagModel:
    return BGEM3FlagModel(embedding_model_name(), use_fp16=embedding_use_fp16())
