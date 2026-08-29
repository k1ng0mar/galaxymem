"""Procedural memory + gap detection — Mnemosyne territory.

Extracts HOW-TO knowledge (not just declarative facts) and uses the
memory graph to surface forgotten associations.

1. Procedural extraction: scans flagged turns for directive/constraint-like
   content and creates procedural memories (distinct from world facts).
2. Gap detection: given recalled memories, uses edges to find neighbor nodes
   that haven't been recalled recently — "you should ALSO remember this..."
3. The synthesis: injected into system prompt as "megatarian" reminders —
   the thing the agent should have recalled but didn't.
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import MemoryRecord, MemoryStatus
from .store_sqlite import Store

logger = logging.getLogger(__name__)


def extract_procedural(store, flagged_turn: str) -> Optional[dict]:
    """Scan for directive/procedural content, return a MemoryRecord dict
    with network='procedural' if the turn describes how to DO something.

    This is distinct from the standard Pass 2 extraction — it's pattern-
    based, running in the same sync_turn pipeline BEFORE the LLM call,
    capturing workflow knowledge that the generic extraction might miss.
    """
    from .retain import _apply_flag_rules

    text = flagged_turn.strip()
    if not text or len(text) < 20:
        return None

    reason = _apply_flag_rules(text)
    if reason in ("directive", "project_constraint"):
        # This is a how-to / constraint — promote to procedural
        return {
            "text": text,
            "network": "observation",
            "procedural": True,
            "entity_ids": [],
            "flag_reason": reason,
        }
    return None


def detect_gaps(
    store: Store,
    results: list[MemoryRecord],
) -> list[str]:
    """Given recalled memories, find their neighbors in the graph that
    have been seen together but weren't recalled this time.

    Uses spread activation: neighbors of the recalled memories get their
    activation score dropped for this recall (they weren't relevant to this
    query), but the EDGE itself being crossed reminds us they exist —
    surfacing them as "things you might have forgotten".

    Returns:
        Formatted reminder strings, e.g.:
        "- You also recall: 'tests gate builds hard in this repo'"
    """
    if not results:
        return []

    gaps = []
    recalled_ids = {r.id for r in results}
    neighbor_of: dict[str, MemoryRecord] = {}
    try:
        neighbor_map = store.neighbors_for_ids([m.id for m in results], min_weight=0.3)
    except Exception:
        neighbor_map = {}
    for mem in results:
        for nid, _edge in neighbor_map.get(mem.id, []):
            if nid in recalled_ids:
                continue
            neighbor_of.setdefault(nid, mem)

    fetched = store.get_memories_by_ids(list(neighbor_of)) if neighbor_of else {}
    for nid, sibling in fetched.items():
        if sibling.status != MemoryStatus.active:
            continue
        mem = neighbor_of.get(nid)
        if mem is None:
            continue
        if sibling.created_at and sibling.created_at > mem.created_at:
            gaps.append(
                f"- You also remember: '{sibling.text[:80]}' "
                f"(linked to '{mem.text[:40]}')"
            )
            if len(gaps) >= 3:
                return gaps
    return gaps
