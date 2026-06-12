"""Sentence-transformer embedding wrapper."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from django.conf import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns list of float vectors."""
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()
