"""Embedding utilities for GalaxyMem.

Local ONNX embedding via fastembed (default) with a deterministic
hash-based fallback when fastembed is unavailable. The fallback is
NOT semantically meaningful — it only lets the store function for
tests and headless environments.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

# Lazy import so the package can be installed without fastembed
# (the fallback is enough for tests + schema-aware code).
_embed_model = None


def _get_model():
    """Load the fastembed TextEmbedding model once, then cache it."""
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding()
    return _embed_model


def embed_text(text: str) -> list[float]:
    """Embed a single text → list[float]."""
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts → list[list[float]].

    Uses fastembed if importable, else deterministic hash vectors.
    Both code paths produce the same dimension so vectors stay
    interchangeable on the same DB.
    """
    try:
        from . import config as cfg
        model = _get_model()
        # fastembed accepts an iterable of strings and returns a generator
        # of lists; coerce to a fully-materialized list.
        dim = cfg.EMBEDDING_DIM
        out = []
        for vec in model.embed(texts):
            v = list(vec)[:dim]
            if len(v) < dim:
                v = v + [0.0] * (dim - len(v))
            out.append(v)
        return out
    except Exception as e:
        logger.debug("fastembed unavailable (%s); using hash fallback", e)
        return _hash_fallback(texts)


def _hash_fallback(texts: list[str]) -> list[list[float]]:
    """Deterministic, non-semantic vectors for tests + offline mode."""
    from . import config as cfg
    dim = cfg.EMBEDDING_DIM
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = [float(h[i % 32]) / 255.0 for i in range(dim)]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out
