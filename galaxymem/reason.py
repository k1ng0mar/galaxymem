"""gm_reason — evidence-backed reasoning over GalaxyMem memories.

Mirrors Hindsight's reflect() agentic loop: check consolidated opinions
(observations) first, then raw facts, then synthesize a grounded answer with
source ids and confidence. Runs synchronously on demand.

The point of this tool is that gm_recall returns raw memories; gm_reason
returns a REASONED, evidence-citeded answer the agent can act on directly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .models import MemoryRecord, MemoryStatus, Network
from .store import Store

logger = logging.getLogger(__name__)


def gather_context(
    store: Store,
    query: str,
    entity_ids: Optional[list[str]] = None,
    max_sources: int = 8,
) -> dict[str, Any]:
    """Pull the evidence for a reasoning query.

    Two tiers, in priority order:
      1. Consolidated opinions (network=opinion, active) — these are the
         "observations" that hindsight checks first.
      2. Raw facts (world + experience, active) — the ground truth.

    Both are scoped by entity_ids when provided (hard filter, matching the
    D8 recall scoping). Unscoped world facts ride along, same as recall.

    Returns:
        {"opinions": [...], "facts": [...], "entity_cards": [...]}
        Each memory entry is a dict with id/text/network/status/created_at/
        confidence/recall_count/source_memory_ids.
    """
    from .confidence import compute_confidence, classify_confidence

    def _to_dict(mem: MemoryRecord) -> dict[str, Any]:
        conf = compute_confidence(mem, store)
        return {
            "id": mem.id,
            "text": mem.text,
            "network": mem.network.value,
            "status": mem.status.value,
            "created_at": mem.created_at.isoformat() if mem.created_at else None,
            "confidence": conf,
            "confidence_tier": classify_confidence(conf),
            "recall_count": mem.recall_count,
            "source_memory_ids": mem.source_memory_ids or [],
        }

    opinions: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []

    # Tier 1: opinions (consolidated observations)
    try:
        opinion_mems = store.list_memories(
            network=Network.opinion,
            status=MemoryStatus.active,
            entity_ids=entity_ids,
        )
        opinions = [_to_dict(m) for m in opinion_mems]
    except Exception as e:
        logger.warning("gm_reason opinion gather failed: %s", e)

    # Tier 2: raw facts (world + experience)
    try:
        fact_mems: list[MemoryRecord] = []
        for network in (Network.world, Network.experience):
            fact_mems.extend(store.list_memories(
                network=network,
                status=MemoryStatus.active,
                entity_ids=entity_ids,
            ))
        facts = [_to_dict(m) for m in fact_mems]
    except Exception as e:
        logger.warning("gm_reason fact gather failed: %s", e)

    # Cap sources to keep the LLM prompt bounded.
    opinions = opinions[:max_sources]
    facts = facts[:max_sources]

    # Entity cards for context (status_line + card fields).
    entity_cards: list[dict[str, Any]] = []
    if entity_ids:
        for eid in entity_ids:
            try:
                ent = store.get_entity(eid)
                if ent is not None:
                    entity_cards.append({
                        "id": ent.id,
                        "label": ent.label,
                        "type": ent.type.value if ent.type else None,
                        "status_line": ent.status_line,
                        "card": ent.card,
                    })
            except Exception as e:
                logger.debug("gm_reason entity card failed for %s: %s", eid, e)

    return {
        "opinions": opinions,
        "facts": facts,
        "entity_cards": entity_cards,
    }


# ── Prompt builder ──────────────────────────────────────────────────────────

_REASON_PROMPT = """You are reasoning over a memory store to answer a question with evidence.

QUESTION: {query}

{entity_section}CONSOLIDATED OPINIONS (observations derived from multiple sources — check these first):
{opinion_lines}

RAW FACTS (world + experience memories — the ground truth):
{fact_lines}

Instructions:
1. Answer the question using ONLY the evidence above. Do not invent facts.
2. If an opinion is contradicted by a raw fact, say so explicitly — flag the conflict.
3. Cite each claim with the memory id(s) that support it, in [id] form.
4. If the evidence is insufficient to answer confidently, say so and list what's missing.
5. Note any low-confidence sources (confidence_tier != "certain").

Return ONLY JSON:
{{
  "answer": "the reasoned answer, grounded in evidence, with [id] citations",
  "sources": ["id1", "id2"],
  "confidence": "high|medium|low",
  "conflicts": ["description of any contradiction, or empty list"],
  "gaps": ["what evidence is missing, or empty list"]
}}
"""


def _fmt_mem(mem: dict[str, Any]) -> str:
    """Format a memory dict for the prompt."""
    conf = mem.get("confidence_tier", "unknown")
    srcs = mem.get("source_memory_ids") or []
    src_part = f" (sources: {', '.join(srcs)})" if srcs else ""
    return f"[{mem['id']}] ({conf}{src_part}) {mem['text']}"


def _build_prompt(query: str, ctx: dict[str, Any]) -> str:
    """Build the reasoning prompt from gathered context."""
    opinions = ctx.get("opinions", [])
    facts = ctx.get("facts", [])
    cards = ctx.get("entity_cards", [])

    entity_section = ""
    if cards:
        card_lines = []
        for c in cards:
            status = c.get("status_line") or ""
            card_lines.append(f"- {c.get('label', c.get('id'))}: {status}")
        entity_section = "ENTITY CONTEXT:\n" + "\n".join(card_lines) + "\n\n"

    opinion_lines = "\n".join(_fmt_mem(o) for o in opinions) or "(none)"
    fact_lines = "\n".join(_fmt_mem(f) for f in facts) or "(none)"

    return _REASON_PROMPT.format(
        query=query,
        entity_section=entity_section,
        opinion_lines=opinion_lines,
        fact_lines=fact_lines,
    )


def _parse_json_object(response: str, default: dict) -> dict:
    """Extract the first valid JSON object from an LLM response, else default."""
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(response):
        pos = response.find("{", idx)
        if pos == -1:
            break
        try:
            result, _ = decoder.raw_decode(response, pos)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        idx = pos + 1
    logger.warning("No valid JSON object found in gm_reason response: %.200s...", response)
    return dict(default)


# ── Orchestrator ───────────────────────────────────────────────────────────

def reason(
    store: Store,
    llm_client: Any,
    query: str,
    entity_ids: Optional[list[str]] = None,
    max_sources: int = 8,
) -> dict[str, Any]:
    """Run the reasoning loop: gather → prompt → LLM → parsed answer.

    Args:
        store: The GalaxyMem store.
        llm_client: An object with .chat(messages) -> str (the provider's
            _LLMClientAdapter).
        query: The question to reason about.
        entity_ids: Optional entity scoping (hard filter).
        max_sources: Max opinions + max facts to include.

    Returns:
        {"answer", "sources", "confidence", "conflicts", "gaps", "used": {...}}
        On failure, returns {"error": ...} and logs.
    """
    if not query or not query.strip():
        return {"error": "Missing required parameter: query"}

    ctx = gather_context(store, query, entity_ids, max_sources)

    if not ctx["opinions"] and not ctx["facts"]:
        return {
            "answer": "No relevant memories found for this query.",
            "sources": [],
            "confidence": "low",
            "conflicts": [],
            "gaps": ["No memories matched the query/entity scope."],
            "used": {"opinions": 0, "facts": 0},
        }

    prompt = _build_prompt(query, ctx)

    try:
        response = llm_client.chat([{"role": "user", "content": prompt}])
    except Exception as e:
        logger.error("gm_reason LLM call failed: %s", e)
        return {"error": f"LLM call failed: {e}"}

    parsed = _parse_json_object(response, {
        "answer": "", "sources": [], "confidence": "low",
        "conflicts": [], "gaps": [],
    })

    # Sanitize sources: only return ids that actually exist in the gathered set.
    valid_ids = {m["id"] for m in ctx["opinions"]} | {m["id"] for m in ctx["facts"]}
    sources = [s for s in parsed.get("sources", []) if s in valid_ids]

    return {
        "answer": parsed.get("answer", ""),
        "sources": sources,
        "confidence": parsed.get("confidence", "low"),
        "conflicts": parsed.get("conflicts", []),
        "gaps": parsed.get("gaps", []),
        "used": {"opinions": len(ctx["opinions"]), "facts": len(ctx["facts"])},
    }

