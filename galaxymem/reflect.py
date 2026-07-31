"""Autonomous reflection — Phase 5 of GalaxyMem (D10, D11).

Runs with no approval gate. Each cycle:

5a. Opinion formation — for entities with enough NEW memories since the last
    reflect, derive stable patterns from active world+experience memories.
    Equivalent existing opinions are MERGED (source ids appended — strength
    grows per D6) instead of duplicated; demoted equivalents are revived.
5b. Conflict resolution — same-entity conflicting memories are classified:
    mutable-over-time facts → silent supersession (recency wins, old memory
    archived as `superseded`, entity status_line updated);
    fixed claims → both `contested`, never silently overwritten.
    A newer unambiguous statement (e.g. a correction caught by Pass 1) that
    supersedes one side of a contested pair heals the other side back to
    active.
5c. Opinion invalidation cascade — opinions whose sources were contested or
    superseded are recounted: ≥2 valid sources → keep (drop invalid ids);
    fewer → demoted (revived automatically if re-derived with fresh sources).
5d. Housekeeping — reflect_cycles incremented on active memories (feeds the
    promotion stability threshold), hot caches rebuilt for touched entities,
    lance tables compacted daily to prevent version bloat, recurring
    untracked names nominated for user-approved entity creation (entity
    creation stays explicit per D3).

Trigger: volume — REFLECT_VOLUME_TRIGGER new memories since the last cycle
(state persisted in reflect_state.json inside the DB directory). A cron can
additionally invoke gm_reflect_now on a schedule (OQ1).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from . import config as cfg
from .models import (
    EdgeKind,
    EdgeRecord,
    MemoryRecord,
    MemoryStatus,
    Network,
    ReflectionRecord,
)

logger = logging.getLogger(__name__)

# Cap how many memories go into a single reflection LLM prompt
_MAX_PROMPT_MEMORIES = 30
# Minimum supporting sources for a newly formed opinion
_MIN_OPINION_SOURCES = 2


# ── LLM Client Protocol ────────────────────────────────────────────────────

class LLMClient(Protocol):
    """Protocol for LLM clients used in reflection.

    Must implement a chat() method that accepts a list of messages and returns
    a string response.
    """
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send messages to LLM and return response text."""
        ...


# ── Helpers ────────────────────────────────────────────────────────────────

from .utils import ulid as _ulid  # noqa: E402


def _parse_json_object(response: str, default: dict) -> dict:
    """Extract the first valid JSON object from an LLM response, else default.

    Uses a proper JSON parser that can handle nested structures and
    skips over invalid text that may contain JSON-like substrings.
    Scans each potential start position and lets the parser find the
    valid object — no naive brace matching.
    """
    # Try progressively: first valid JSON object anywhere in the response.
    # Start from the beginning and try each position where "{" occurs.
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
    logger.warning("No valid JSON object found in LLM reflection response: %.200s...", response)
    return dict(default)


# ── Lance compaction (prevents version bloat) ──────────────────────────────

_COMPACT_INTERVAL = 86400  # 24 hours in seconds


def _last_compact_path(store) -> Path:
    return Path(store.db_path) / "compact_state.json"


def _should_compact(store) -> bool:
    """True if the last lance compaction was > 24h ago (or never run)."""
    try:
        raw = _last_compact_path(store).read_text(encoding="utf-8")
        ts = json.loads(raw).get("last_compact_at", 0)
        return (time.time() - ts) >= _COMPACT_INTERVAL
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return True


def _save_compact_state(store) -> None:
    try:
        _last_compact_path(store).write_text(
            json.dumps({"last_compact_at": time.time()}),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Failed to persist compact state: %s", e)


def _compact_lance_tables(store) -> None:
    """Run lance optimize + old-version cleanup on all write-heavy tables.

    LanceDB accumulates a version snapshot per write; without periodic
    compaction the _versions/ directory grows unbounded (observed 1.7 GB
    for 87 MB of actual data). This merges fragments and removes versions
    older than 24 hours.
    """
    if not _should_compact(store):
        return
    table_names = ["memories", "edges", "flags", "promotion_queue"]
    for name in table_names:
        try:
            tbl = store.db.open_table(name)
            tbl.optimize(cleanup_older_than=timedelta(hours=24))
            logger.info("Compacted lance table: %s", name)
        except Exception as e:
            logger.warning("Lance compaction failed for %s: %s", name, e)
    _save_compact_state(store)


# ── Reflect state (drives the volume trigger) ──────────────────────────────

def _state_path(store) -> Path:
    return Path(store.db_path) / "reflect_state.json"


def get_last_reflect_at(store) -> Optional[datetime]:
    """Timestamp of the last completed reflection cycle, or None."""
    try:
        raw = _state_path(store).read_text(encoding="utf-8")
        data = json.loads(raw)
        ts = data.get("last_reflect_at")
        return datetime.fromisoformat(ts) if ts else None
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None


def _save_reflect_state(store) -> None:
    try:
        _state_path(store).write_text(
            json.dumps({"last_reflect_at": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Failed to persist reflect state: %s", e)


def should_reflect(store) -> bool:
    """Volume trigger: >= REFLECT_VOLUME_TRIGGER NEW memories since the last
    reflect cycle (not total corpus size — a full corpus does not mean new
    work to reflect on)."""
    try:
        since = get_last_reflect_at(store)
        new_memories = store.list_memories(since=since)
        if len(new_memories) >= cfg.REFLECT_VOLUME_TRIGGER:
            logger.info(
                "Reflection triggered: %d new memories >= %d threshold",
                len(new_memories), cfg.REFLECT_VOLUME_TRIGGER,
            )
            return True
    except Exception as e:
        logger.warning("Failed to check reflection trigger: %s", e)
    return False


# ── Main entry point ───────────────────────────────────────────────────────

def run_reflection(store, llm_client: LLMClient) -> dict[str, Any]:
    """Run one full reflection cycle. Returns a report dict."""
    logger.info("Starting reflection cycle")

    report: dict[str, Any] = {
        "contradictions_detected": 0,
        "memories_superseded": 0,
        "memories_contested": 0,
        "memories_healed": 0,
        "opinions_formed": 0,
        "opinions_merged": 0,
        "opinions_demoted": 0,
        "entity_suggestions": [],
        "reflection_records": [],
    }

    try:
        since = get_last_reflect_at(store)
        new_memories = store.list_memories(since=since)

        # Group NEW memories by entity — reflection focuses where things changed
        new_by_entity: dict[str, list[MemoryRecord]] = defaultdict(list)
        for mem in new_memories:
            for entity_id in mem.entity_ids:
                new_by_entity[entity_id].append(mem)

        touched_entities: set[str] = set()
        invalidated_ids: set[str] = set()

        # 5b — conflict resolution for every entity with new material
        for entity_id in list(new_by_entity.keys()):
            records = _resolve_conflicts_for_entity(
                store, llm_client, entity_id, report, invalidated_ids,
            )
            if records:
                touched_entities.add(entity_id)
                report["reflection_records"].extend(records)

        # 5a — opinion formation for entities with >= 3 new memories
        for entity_id, mems in new_by_entity.items():
            if len(mems) < 3:
                continue
            records = _form_opinions_for_entity(store, llm_client, entity_id, report)
            if records:
                touched_entities.add(entity_id)
                report["reflection_records"].extend(records)

        # 5c — opinion invalidation cascade
        if invalidated_ids:
            records = _cascade_opinion_invalidation(store, invalidated_ids, report)
            report["reflection_records"].extend(records)

        # 5d — housekeeping
        _increment_reflect_cycles(store)
        report["entity_suggestions"] = _nominate_entities(store)
        from .entities import _cleanup_stale_provisionals
        cleaned = _cleanup_stale_provisionals(store)
        if cleaned:
            logger.info("Cleaned up %d stale provisional entities", cleaned)
            report["provisionals_cleaned"] = cleaned
        _rebuild_hot_caches(store, touched_entities)
        _compact_lance_tables(store)
        _save_reflect_state(store)

        logger.info(
            "Reflection complete: %d contested, %d superseded, %d healed, "
            "%d opinions formed, %d merged, %d demoted",
            report["memories_contested"], report["memories_superseded"],
            report["memories_healed"], report["opinions_formed"],
            report["opinions_merged"], report["opinions_demoted"],
        )

    except Exception as e:
        logger.error("Reflection cycle failed: %s", e, exc_info=True)
        report["error"] = str(e)

    return report


# ── 5b: conflict resolution ────────────────────────────────────────────────

def _conflict_basis(store, entity_id: str) -> list[MemoryRecord]:
    """Memories eligible for conflict analysis: the entity's active world +
    experience memories, plus its contested memories (so a newer correction
    can resolve an open contradiction)."""
    memories: list[MemoryRecord] = []
    for network in (Network.world, Network.experience):
        memories.extend(store.list_memories(network=network, status=MemoryStatus.active,
                                            entity_ids=[entity_id]))
        memories.extend(store.list_memories(network=network, status=MemoryStatus.contested,
                                            entity_ids=[entity_id]))
    memories.sort(key=lambda m: m.created_at)
    return memories[-_MAX_PROMPT_MEMORIES:]


_CONFLICT_PROMPT = """Analyze these memories about entity '{entity_id}' (chronological order, oldest first) for pairs that conflict — they state incompatible things about the same subject.

Memories:
{memory_lines}

For each conflicting pair, classify the underlying fact:
- "mutable": something that legitimately changes over time (location, current project, current preference, status). The newer statement replaces the older one.
- "fixed": a claim that cannot both be true (a trait, a one-time event, a quote). Neither statement can be trusted until clarified.

Entries marked [contested] are earlier unresolved contradictions: if a newer memory clearly settles one, report it as "mutable" with the contested entry as old_memory_id.

Return ONLY JSON:
{{
  "conflicts": [
    {{
      "old_memory_id": "id of the earlier/replaced memory",
      "new_memory_id": "id of the later memory",
      "fact_type": "mutable|fixed",
      "reason": "brief explanation",
      "status_line": "optional: one-line current status for the entity after this update"
    }}
  ]
}}

If nothing conflicts, return {{"conflicts": []}}.
"""


def _resolve_conflicts_for_entity(
    store,
    llm_client: LLMClient,
    entity_id: str,
    report: dict,
    invalidated_ids: set[str],
) -> list[ReflectionRecord]:
    """Detect and resolve conflicts for one entity (spec 5b)."""
    memories = _conflict_basis(store, entity_id)
    if len(memories) < 2:
        return []
    by_id = {m.id: m for m in memories}

    memory_lines = "\n".join(
        f"[{m.id}] ({m.created_at.strftime('%Y-%m-%d')})"
        f"{' [contested]' if m.status == MemoryStatus.contested else ''} {m.text}"
        for m in memories
    )
    prompt = _CONFLICT_PROMPT.format(entity_id=entity_id, memory_lines=memory_lines)

    try:
        response = llm_client.chat([{"role": "user", "content": prompt}])
    except Exception as e:
        logger.error("Conflict analysis failed for entity %s: %s", entity_id, e)
        return []

    analysis = _parse_json_object(response, {"conflicts": []})
    records: list[ReflectionRecord] = []

    for conflict in analysis.get("conflicts", []):
        old_id = conflict.get("old_memory_id")
        new_id = conflict.get("new_memory_id")
        fact_type = conflict.get("fact_type", "fixed")
        reason = conflict.get("reason", "")
        old_mem = by_id.get(old_id)
        new_mem = by_id.get(new_id)
        if old_mem is None or new_mem is None or old_id == new_id:
            continue  # never act on ids the LLM invented

        if fact_type == "mutable":
            records.extend(_apply_supersession(
                store, entity_id, old_mem, new_mem, reason,
                conflict.get("status_line") or "", report, invalidated_ids,
            ))
        else:
            records.extend(_apply_contradiction(
                store, old_mem, new_mem, reason, report, invalidated_ids,
            ))

    return records


def _apply_supersession(store, entity_id: str, old_mem: MemoryRecord,
                        new_mem: MemoryRecord, reason: str, status_line: str,
                        report: dict, invalidated_ids: set[str],
                        ) -> list[ReflectionRecord]:
    """Mutable fact: recency wins silently (D11). Old memory archived as
    superseded; if it was one side of a contested pair, heal the partners."""
    if new_mem.status != MemoryStatus.active:
        return []  # the winner must be a live memory

    records: list[ReflectionRecord] = []

    store.update_memory_status(old_mem.id, MemoryStatus.superseded,
                               superseded_by=new_mem.id)
    store.add_edge(EdgeRecord(from_id=old_mem.id, to_id=new_mem.id,
                              kind=EdgeKind.supersedes, weight=1.0))
    invalidated_ids.add(old_mem.id)
    report["memories_superseded"] += 1
    records.append(ReflectionRecord(
        id=_ulid(), action="supersede", memory_ids=[old_mem.id],
        new_memory_id=new_mem.id, reason=reason,
    ))
    logger.info("Superseded %s -> %s: %s", old_mem.id, new_mem.id, reason)

    # Heal contest partners left with no remaining contest
    for partner_id in old_mem.contested_with or []:
        partner = store.get_memory(partner_id)
        if partner is None or partner.status != MemoryStatus.contested:
            continue
        remaining = [c for c in partner.contested_with if c != old_mem.id]
        if not remaining:
            store.update_memory_status(partner_id, MemoryStatus.active,
                                       contested_with=[])
            report["memories_healed"] += 1
            records.append(ReflectionRecord(
                id=_ulid(), action="heal", memory_ids=[partner_id],
                reason=f"Contradiction resolved by supersession of {old_mem.id}",
            ))
        else:
            store.update_memory_status(partner_id, MemoryStatus.contested,
                                       contested_with=remaining)

    # Keep the entity's status_line current (5b)
    line = status_line.strip() or new_mem.text.strip()
    try:
        store.update_entity(entity_id, status_line=line[:200])
    except Exception as e:
        logger.debug("status_line update failed for %s: %s", entity_id, e)

    return records


def _apply_contradiction(store, mem_a: MemoryRecord, mem_b: MemoryRecord,
                         reason: str, report: dict, invalidated_ids: set[str],
                         ) -> list[ReflectionRecord]:
    """Fixed claim: both kept + contested; never silently overwritten (D11)."""
    if mem_a.status != MemoryStatus.active or mem_b.status != MemoryStatus.active:
        return []  # already-contested pairs stay as they are

    store.update_memory_status(mem_a.id, MemoryStatus.contested,
                               contested_with=list({*(mem_a.contested_with or []), mem_b.id}))
    store.update_memory_status(mem_b.id, MemoryStatus.contested,
                               contested_with=list({*(mem_b.contested_with or []), mem_a.id}))
    store.add_edge(EdgeRecord(from_id=mem_a.id, to_id=mem_b.id,
                              kind=EdgeKind.contests, weight=1.0))
    invalidated_ids.update((mem_a.id, mem_b.id))
    report["memories_contested"] += 2
    report["contradictions_detected"] += 1
    logger.info("Contradiction: %s <-> %s: %s", mem_a.id, mem_b.id, reason)

    return [ReflectionRecord(
        id=_ulid(), action="contest", memory_ids=[mem_a.id, mem_b.id],
        reason=reason,
    )]


# ── 5a: opinion formation ──────────────────────────────────────────────────

_OPINION_PROMPT = """You maintain opinions (revisable beliefs) about entity '{entity_id}' derived from stored facts and events.

Facts and events (each line: [memory_id] text):
{memory_lines}

Existing opinions already held (do NOT repeat these):
{opinion_lines}

Identify up to 3 NEW stable patterns or conclusions supported by at least {min_sources} of the memories above and not already captured by an existing opinion.

Return ONLY JSON:
{{
  "opinions": [
    {{
      "text": "the opinion, stated as a belief in 1-2 sentences",
      "source_memory_ids": ["id1", "id2"]
    }}
  ]
}}

If no new pattern is clearly supported, return {{"opinions": []}}.
"""


def _form_opinions_for_entity(store, llm_client: LLMClient, entity_id: str,
                              report: dict) -> list[ReflectionRecord]:
    """Form opinions from the entity's world+experience memories (spec 5a)."""
    basis: list[MemoryRecord] = []
    for network in (Network.world, Network.experience):
        basis.extend(store.list_memories(network=network, status=MemoryStatus.active,
                                         entity_ids=[entity_id]))
    if len(basis) < _MIN_OPINION_SOURCES:
        return []
    basis.sort(key=lambda m: m.created_at)
    basis = basis[-_MAX_PROMPT_MEMORIES:]
    by_id = {m.id: m for m in basis}

    existing: list[MemoryRecord] = []
    for status in (MemoryStatus.active, MemoryStatus.demoted):
        existing.extend(store.list_memories(network=Network.opinion, status=status,
                                            entity_ids=[entity_id]))

    prompt = _OPINION_PROMPT.format(
        entity_id=entity_id,
        memory_lines="\n".join(f"[{m.id}] {m.text}" for m in basis),
        opinion_lines="\n".join(f"- {o.text}" for o in existing) or "(none)",
        min_sources=_MIN_OPINION_SOURCES,
    )

    try:
        response = llm_client.chat([{"role": "user", "content": prompt}])
    except Exception as e:
        logger.error("Opinion formation failed for entity %s: %s", entity_id, e)
        return []

    analysis = _parse_json_object(response, {"opinions": []})
    records: list[ReflectionRecord] = []

    for item in analysis.get("opinions", []):
        text = (item.get("text") or "").strip()
        source_ids = [sid for sid in item.get("source_memory_ids", []) if sid in by_id]
        if not text or len(source_ids) < _MIN_OPINION_SOURCES:
            continue

        merged = _merge_into_existing_opinion(store, entity_id, text, source_ids, existing)
        if merged is not None:
            report["opinions_merged"] += 1
            records.append(ReflectionRecord(
                id=_ulid(), action="merge_opinion", memory_ids=source_ids,
                new_memory_id=merged, reason="Equivalent opinion strengthened (D6)",
            ))
            continue

        opinion = MemoryRecord(
            id=_ulid(), text=text, network=Network.opinion,
            entity_ids=[entity_id], source_memory_ids=source_ids,
            status=MemoryStatus.active,
        )
        store.add_memory(opinion)
        for sid in source_ids:
            store.add_edge(EdgeRecord(from_id=sid, to_id=opinion.id,
                                      kind=EdgeKind.derived_from, weight=1.0))
        report["opinions_formed"] += 1
        records.append(ReflectionRecord(
            id=_ulid(), action="form_opinion", memory_ids=source_ids,
            new_memory_id=opinion.id, reason="New stable pattern",
        ))
        logger.info("Formed opinion for %s: %.60s (from %d sources)",
                    entity_id, text, len(source_ids))

    return records


def _merge_into_existing_opinion(store, entity_id: str, text: str,
                                 source_ids: list[str],
                                 existing: list[MemoryRecord]) -> Optional[str]:
    """If an equivalent opinion exists, append sources instead of duplicating.

    A demoted equivalent is revived (5c revival) since it now has fresh
    sources. Returns the merged opinion id, or None if no equivalent exists.
    """
    try:
        matches = store.vector_search(
            text, k=3,
            entity_filter=[entity_id],
            network_filter=Network.opinion,
            status_filter=[MemoryStatus.active, MemoryStatus.demoted],
        )
    except Exception:
        matches = []

    score_threshold = (1.0 + cfg.DEDUP_SIMILARITY_THRESHOLD) / 2.0
    for candidate, score in matches:
        if score < score_threshold:
            continue
        combined = list(dict.fromkeys([*candidate.source_memory_ids, *source_ids]))
        store.update_memory_field(candidate.id, source_memory_ids=combined)
        for sid in source_ids:
            if sid not in candidate.source_memory_ids:
                store.add_edge(EdgeRecord(from_id=sid, to_id=candidate.id,
                                          kind=EdgeKind.derived_from, weight=1.0))
        if candidate.status == MemoryStatus.demoted:
            store.update_memory_status(candidate.id, MemoryStatus.active)
            logger.info("Revived demoted opinion %s with fresh sources", candidate.id)
        return candidate.id
    return None


# ── 5c: opinion invalidation cascade ───────────────────────────────────────

def _cascade_opinion_invalidation(store, invalidated_ids: set[str],
                                  report: dict) -> list[ReflectionRecord]:
    """Recount sources of opinions that depended on invalidated memories."""
    records: list[ReflectionRecord] = []
    opinions = store.list_memories(network=Network.opinion, status=MemoryStatus.active)

    for opinion in opinions:
        hit = set(opinion.source_memory_ids) & invalidated_ids
        if not hit:
            continue

        valid: list[str] = []
        for sid in opinion.source_memory_ids:
            src = store.get_memory(sid)
            if src is not None and src.status == MemoryStatus.active:
                valid.append(sid)

        if len(valid) >= 2:
            store.update_memory_field(opinion.id, source_memory_ids=valid)
            records.append(ReflectionRecord(
                id=_ulid(), action="drop_sources", memory_ids=list(hit),
                new_memory_id=opinion.id,
                reason=f"Dropped {len(opinion.source_memory_ids) - len(valid)} invalidated sources",
            ))
        else:
            store.update_memory_status(opinion.id, MemoryStatus.demoted)
            report["opinions_demoted"] += 1
            records.append(ReflectionRecord(
                id=_ulid(), action="demote_opinion", memory_ids=[opinion.id],
                reason="Fewer than 2 valid sources remain",
            ))
            logger.info("Demoted opinion %s (sources invalidated)", opinion.id)

    return records


# ── 5d: housekeeping ───────────────────────────────────────────────────────

def _increment_reflect_cycles(store) -> None:
    """Bump reflect_cycles on active memories — feeds the promotion
    stability threshold (stable across >= N reflect cycles).

    Uses LanceDB batch update with a SQL expression instead of one
    update per memory (O(N) DB writes → O(1)).
    """
    try:
        # Try LanceDB's batch update with a SQL increment expression.
        # This does a single bulk update across the whole table.
        self_expr = "reflect_cycles + 1"
        store._memories.update(
            where='status = "active"',
            values={"reflect_cycles": self_expr},
        )
    except Exception:
        # Fallback: individual updates (small DBs where this fires infrequently)
        try:
            for mem in store.list_memories(status=MemoryStatus.active):
                store.update_memory_field(mem.id, reflect_cycles=mem.reflect_cycles + 1)
        except Exception as e:
            logger.warning("reflect_cycles increment failed: %s", e)


def _rebuild_hot_caches(store, entity_ids: set[str]) -> None:
    try:
        from .recall import update_hot_cache
        for entity_id in entity_ids:
            update_hot_cache(store, entity_id=entity_id)
        update_hot_cache(store, entity_id=None)
    except Exception as e:
        logger.warning("Hot cache rebuild after reflection failed: %s", e)


_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})?)\b")
_NAME_STOPWORDS = {
    "The", "This", "That", "There", "They", "When", "Where", "What", "Which",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Today", "Tomorrow", "Yesterday",
}


def _nominate_entities(store) -> list[dict[str, Any]]:
    """Nominate recurring untracked names for entity creation (5d).

    Surfaced as suggestions only — creating the entity remains an explicit
    user decision (D3's explicit-only spirit).
    """
    try:
        known_labels = {e.label.lower() for e in store.list_entities()}
        counter: Counter[str] = Counter()
        examples: dict[str, list[str]] = defaultdict(list)

        for mem in store.list_memories(status=MemoryStatus.active):
            seen_in_mem = set()
            for match in _NAME_PATTERN.findall(mem.text):
                if match in _NAME_STOPWORDS or match.lower() in known_labels:
                    continue
                if match in seen_in_mem:
                    continue
                seen_in_mem.add(match)
                counter[match] += 1
                if len(examples[match]) < 3:
                    examples[match].append(mem.id)

        return [
            {"name": name, "memory_count": count, "example_memory_ids": examples[name]}
            for name, count in counter.most_common(10)
            if count >= cfg.ENTITY_CREATION_MIN_RECURRING
        ]
    except Exception as e:
        logger.warning("Entity nomination failed: %s", e)
        return []


# ── Compatibility wrappers ─────────────────────────────────────────────────

def detect_contradictions(store, llm_client: LLMClient) -> list[ReflectionRecord]:
    """Run the conflict pass across all entities and return the records."""
    report: dict[str, Any] = {
        "contradictions_detected": 0, "memories_superseded": 0,
        "memories_contested": 0, "memories_healed": 0,
    }
    invalidated: set[str] = set()
    all_records: list[ReflectionRecord] = []
    entity_ids: set[str] = set()
    for network in (Network.world, Network.experience):
        for mem in store.list_memories(network=network, status=MemoryStatus.active):
            entity_ids.update(mem.entity_ids)
    for entity_id in entity_ids:
        all_records.extend(_resolve_conflicts_for_entity(
            store, llm_client, entity_id, report, invalidated,
        ))
    return all_records


def form_opinions(store, llm_client: LLMClient) -> list[ReflectionRecord]:
    """Run opinion formation across all entities and return the records."""
    report: dict[str, Any] = {"opinions_formed": 0, "opinions_merged": 0}
    all_records: list[ReflectionRecord] = []
    entity_ids: set[str] = set()
    for network in (Network.world, Network.experience):
        for mem in store.list_memories(network=network, status=MemoryStatus.active):
            entity_ids.update(mem.entity_ids)
    for entity_id in entity_ids:
        all_records.extend(_form_opinions_for_entity(store, llm_client, entity_id, report))
    return all_records
