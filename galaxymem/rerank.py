"""Cross-encoder reranker for deep recall.

Re-reads query + each top-K memory pair through a cross-encoder to score
*paired relevance* (not just embedding similarity). Bi-encoders measure
semantic proximity; cross-encoders measure "is this memory actually useful
for THIS query" — the precision gap that makes session search feel random.

Graceful degradation: if the model can't be loaded, rerank is a no-op and
the fused bi-encoder score is used alone. The pipeline never hard-fails
because of the reranker.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from . import config as cfg

logger = logging.getLogger(__name__)

_reranker = None
_reranker_lock = threading.Lock()
_reranker_failed = False  # sticky flag: once it's failed, don't retry each call


class _NullReranker:
    """Fallback when the real cross-encoder can't load — identity passthrough."""
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        # Score = number of query terms present in the memory (cheap lexical fallback)
        scores = []
        for query, memory in pairs:
            qterms = set(query.lower().split())
            mterms = set(memory.lower().split())
            overlap = len(qterms & mterms) / max(1, len(qterms))
            scores.append(overlap)
        return scores


def _load_reranker():
    """Lazy-load the cross-encoder model. Returns None if unavailable."""
    global _reranker, _reranker_failed
    if _reranker is not None or _reranker_failed:
        return _reranker
    if not cfg.RERANKER_MODEL or cfg.RERANKER_MODEL.lower() in ("none", "false", "0", "off"):
        _reranker_failed = True
        return None
    with _reranker_lock:
        if _reranker is not None or _reranker_failed:
            return _reranker
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(cfg.RERANKER_MODEL, max_length=512, device=None)
            logger.info("Loaded reranker: %s", cfg.RERANKER_MODEL)
            return _reranker
        except Exception as e:
            logger.warning(
                "Cross-encoder reranker unavailable (%s), will use bi-encoder scores only. "
                "Check network/model name/GALAXYMEM_RERANKER_MODEL.",
                e,
            )
            _reranker_failed = True
            return None


def rerank(
    query: str,
    memories_with_scores: list[tuple[object, float]],
    *,
    top_k: Optional[int] = None,
    score_weight: Optional[float] = None,
) -> list[tuple[object, float]]:
    """Rerank (memory, fused_score) pairs by query–memory paired relevance.

    Args:
        query: Natural-language query.
        memories_with_scores: [(memory, fused_rrf_score)] as produced by RRF fusion.
        top_k: Maximum results to return.
        score_weight: Blend weight for the cross-encoder score
            (1.0 = use only cross-encoder; 0.0 = use only RRF).

    Returns:
        The same pairs, resorted by blended score, top-k truncated.
    """
    if top_k is None:
        top_k = cfg.RERANKER_TOP_K
    if score_weight is None:
        score_weight = cfg.RERANKER_SCORE_WEIGHT

    if not memories_with_scores:
        return memories_with_scores
    if len(memories_with_scores) <= 1:
        return memories_with_scores

    model = _load_reranker()
    if model is None:
        if not cfg.RERANKER_ALLOW_FALLBACK:
            raise RuntimeError(
                f"Reranker model '{cfg.RERANKER_MODEL}' unavailable and fallback disabled"
            )
        return memories_with_scores

    # Build (query, memory_text) pairs for scoring
    pairs = [(query, m[0].text) for m in memories_with_scores]
    try:
        with _reranker_lock:
            # sentence-transformers v5 renamed .score() → .predict()
            scores = model.predict(pairs)  # blocking call
    except AttributeError:
        # v4 compatibility: .score() existed before v5
        try:
            with _reranker_lock:
                scores = model.score(pairs)
        except Exception as e:
            logger.warning("Reranker scoring failed: %s; using bi-encoder scores", e)
            return memories_with_scores
    except Exception as e:
        logger.warning("Reranker scoring failed: %s; using bi-encoder scores", e)
        return memories_with_scores

    # Blend: final = (1-w) * fused_rrf + w * normalized_rerank
    # Normalize rerank scores to [0,1] via the minibatch sigmoid mapping that
    # CrossEncoder returns for MS MARCO models (negative values → 0, positive → 1).
    # The actual range depends on the model; MS MARCO MiniLM-L-6-v2 is logistic.
    import math
    blended = []
    for (mem, rrf_score), rerank_score in zip(memories_with_scores, scores):
        # rerank_score: positive values → stronger relevance.
        # Map through sigmoid to [0,1].
        normalized_rerank = 1.0 / (1.0 + math.exp(-float(rerank_score)))
        final = (1.0 - score_weight) * rrf_score + score_weight * normalized_rerank
        blended.append((mem, final, rerank_score))

    blended.sort(key=lambda t: t[1], reverse=True)
    return [(mem, final) for mem, final, _ in blended[:top_k]]


def is_available() -> bool:
    """True if the cross-encoder model is loadable in this process."""
    return _load_reranker() is not None
