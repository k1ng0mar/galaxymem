"""Belief confidence scoring — strength from corroboration, not just recency.

Confidence measures how reliable a memory is, derived from:
  - number of source memories (more sources → stronger)
  - edge endorsements (derived_from, shared_entity edges → corroboration)
  - status (active > contested > demoted)
  - temporal stability (reflect_cycles survived)

The score is cached in-memory (no extra DB writes per recall — it's
recomputed cheaply on access) and exposed via gm_recall results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .models import MemoryRecord, MemoryStatus
from .store_sqlite import Store

logger = logging.getLogger(__name__)


def compute_confidence(memory: MemoryRecord, store: Optional[Store] = None) -> float:
    """Compute a 0.0-1.0 confidence score for a memory.

    Components (weighted blend):
      - Base status: active=0.50, contested=0.30, demoted=0.10, archived=0.05
      - Source count bonus: +0.25 * min(1, len(source_memory_ids) / 5)
      - Edge endorsements: +0.15 * (derived_from edges / 10)
      - Reflect cycles: +0.10 * min(1, reflect_cycles / 3)

    Total range: ~0.10 (archived, no sources, no reflections)
    to ~1.00 (active, many sources, reflected, corroborated).
    """
    score = 0.50

    # Status base
    status_scores = {
        MemoryStatus.active: 0.50,
        MemoryStatus.contested: 0.30,
        MemoryStatus.demoted: 0.10,
        MemoryStatus.archived: 0.05,
    }
    score = status_scores.get(memory.status, 0.10)

    # Source corroboration (how many distinct turns/flags support this fact)
    if memory.source_memory_ids:
        if isinstance(memory.source_memory_ids, list):
            src_count = len(memory.source_memory_ids)
        else:
            try:
                import json
                src_count = len(json.loads(memory.source_memory_ids))
            except (json.JSONDecodeError, TypeError):
                src_count = 1
        score += 0.25 * min(1.0, src_count / 5.0)

    # Edge endorsements
    if store is not None:
        try:
            edges = store.get_edges_for_memory(memory.id)
            endorsements = sum(1 for e in edges if e.kind.value == "derived_from")
            score += 0.15 * min(1.0, endorsements / 10.0)
        except Exception:
            pass  # edges are nice-to-have for confidence

    # Reflect cycles survived
    score += 0.10 * min(1.0, memory.reflect_cycles / 3.0)

    return round(min(1.0, max(0.0, score)), 4)


def classify_confidence(score: float) -> str:
    """Human-readable tier for a confidence score."""
    if score >= 0.85:
        return "certain"
    if score >= 0.65:
        return "likely"
    if score >= 0.45:
        return "uncertain"
    return "low"
