"""Integration tests for reflect.py — Phase 5 of GalaxyMem.

Covers the spec checkpoints:
- "Mike is in Berlin" → "Mike moved to Lisbon": silent supersession +
  status_line update (mutable fact, D11).
- Two contradictory fixed claims: both contested + a dependent opinion
  demoted (5c cascade).
- Opinion formation from world+experience with merge-into-existing (D6).
- Volume trigger counts NEW memories since the last reflect cycle.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from galaxymem.models import (
    EdgeKind,
    EntityRecord,
    EntityType,
    MemoryRecord,
    MemoryStatus,
    Network,
)
from galaxymem.reflect import (
    detect_contradictions,
    form_opinions,
    get_last_reflect_at,
    run_reflection,
    should_reflect,
)
from galaxymem.store import Store
from galaxymem import config as cfg


# Mock embedding function for tests
def _mock_embed_text(text: str) -> list[float]:
    return [0.0] * 384


def _mock_embed_texts(texts: list[str]) -> list[list[float]]:
    return [[0.0] * 384 for _ in texts]


class MockLLMClient:
    """Mock LLM routed on the JSON template each reflection prompt embeds."""

    def __init__(self, conflicts: str = '{"conflicts": []}',
                 opinions: str = '{"opinions": []}'):
        self.conflicts_response = conflicts
        self.opinions_response = opinions
        self.call_count = 0
        self.last_prompt = ""

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        self.last_prompt = messages[0]["content"] if messages else ""
        if '"conflicts"' in self.last_prompt:
            return self.conflicts_response
        if '"opinions"' in self.last_prompt:
            return self.opinions_response
        return "{}"


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_db"
        with patch("galaxymem.store.embed_text", side_effect=_mock_embed_text), \
             patch("galaxymem.store.embed_texts", side_effect=_mock_embed_texts):
            store = Store(db_path)
            store.open(create_if_missing=True)
            yield store
            store.close()


def _create_memory(
    text: str,
    network: Network = Network.world,
    entity_ids: list[str] | None = None,
    status: MemoryStatus = MemoryStatus.active,
    recall_count: int = 0,
    created_at: datetime | None = None,
    source_memory_ids: list[str] | None = None,
    mem_id: str | None = None,
) -> MemoryRecord:
    from galaxymem.reflect import _ulid

    return MemoryRecord(
        id=mem_id or _ulid(),
        text=text,
        network=network,
        entity_ids=entity_ids or [],
        source_memory_ids=source_memory_ids or [],
        status=status,
        recall_count=recall_count,
        created_at=created_at or datetime.now(timezone.utc),
    )


def _add_entity(store, entity_id: str, label: str | None = None) -> None:
    store.add_entity(EntityRecord(
        id=entity_id, type=EntityType.person, label=label or entity_id.title(),
    ))


# ── Volume trigger ─────────────────────────────────────────────────────────

def test_should_reflect_volume_trigger(temp_store):
    for i in range(cfg.REFLECT_VOLUME_TRIGGER):
        temp_store.add_memory(_create_memory(f"fact {i}"))
    assert should_reflect(temp_store) is True


def test_should_reflect_below_threshold(temp_store):
    for i in range(5):
        temp_store.add_memory(_create_memory(f"fact {i}"))
    assert should_reflect(temp_store) is False


def test_reflection_resets_volume_trigger(temp_store):
    """After a cycle, only NEW memories count toward the next trigger —
    total corpus size must not re-trigger every session (spec OQ1 shape)."""
    for i in range(cfg.REFLECT_VOLUME_TRIGGER):
        temp_store.add_memory(_create_memory(f"fact {i}"))
    assert should_reflect(temp_store) is True

    run_reflection(temp_store, MockLLMClient())
    assert get_last_reflect_at(temp_store) is not None
    assert should_reflect(temp_store) is False  # nothing new since the cycle


# ── Supersession (mutable facts, D11) ──────────────────────────────────────

def test_supersession_mike_moves_to_lisbon(temp_store):
    """Spec Phase 5 checkpoint: silent supersession + status_line update."""
    _add_entity(temp_store, "mike")
    old = _create_memory("Mike is in Berlin", entity_ids=["mike"], mem_id="m-berlin",
                         created_at=datetime.now(timezone.utc) - timedelta(days=10))
    new = _create_memory("Mike moved to Lisbon", entity_ids=["mike"], mem_id="m-lisbon")
    temp_store.add_memory(old)
    temp_store.add_memory(new)

    llm = MockLLMClient(conflicts=(
        '{"conflicts": [{"old_memory_id": "m-berlin", "new_memory_id": "m-lisbon",'
        ' "fact_type": "mutable", "reason": "moved", "status_line": "Living in Lisbon"}]}'
    ))
    report = run_reflection(temp_store, llm)

    assert report["memories_superseded"] == 1
    updated_old = temp_store.get_memory("m-berlin")
    assert updated_old.status == MemoryStatus.superseded
    assert updated_old.superseded_by == "m-lisbon"
    assert temp_store.get_memory("m-lisbon").status == MemoryStatus.active

    edges = temp_store.get_edges_for_memory("m-berlin")
    assert any(e.kind == EdgeKind.supersedes and e.to_id == "m-lisbon" for e in edges)

    assert temp_store.get_entity("mike").status_line == "Living in Lisbon"


# ── Contradiction (fixed claims, D11) ──────────────────────────────────────

def test_contradiction_both_contested(temp_store):
    _add_entity(temp_store, "mike")
    a = _create_memory("Mike has never been to France", entity_ids=["mike"], mem_id="m-a")
    b = _create_memory("Mike lived in Paris for a year", entity_ids=["mike"], mem_id="m-b")
    temp_store.add_memory(a)
    temp_store.add_memory(b)

    llm = MockLLMClient(conflicts=(
        '{"conflicts": [{"old_memory_id": "m-a", "new_memory_id": "m-b",'
        ' "fact_type": "fixed", "reason": "incompatible claims"}]}'
    ))
    report = run_reflection(temp_store, llm)

    assert report["contradictions_detected"] == 1
    assert report["memories_contested"] == 2
    mem_a = temp_store.get_memory("m-a")
    mem_b = temp_store.get_memory("m-b")
    assert mem_a.status == MemoryStatus.contested
    assert mem_b.status == MemoryStatus.contested
    assert "m-b" in mem_a.contested_with
    assert "m-a" in mem_b.contested_with

    edges = temp_store.get_edges_for_memory("m-a")
    assert any(e.kind == EdgeKind.contests for e in edges)


def test_llm_invented_ids_are_ignored(temp_store):
    _add_entity(temp_store, "mike")
    temp_store.add_memory(_create_memory("Mike likes tea", entity_ids=["mike"]))
    temp_store.add_memory(_create_memory("Mike works remotely", entity_ids=["mike"]))

    llm = MockLLMClient(conflicts=(
        '{"conflicts": [{"old_memory_id": "not-real", "new_memory_id": "also-fake",'
        ' "fact_type": "mutable", "reason": "hallucinated"}]}'
    ))
    report = run_reflection(temp_store, llm)
    assert report["memories_superseded"] == 0
    assert report["memories_contested"] == 0


def test_supersession_heals_contested_partner(temp_store):
    """D11 resolution: a later unambiguous statement that supersedes one side
    of a contested pair returns the other side to active."""
    _add_entity(temp_store, "mike")
    a = _create_memory("Mike is vegetarian", entity_ids=["mike"], mem_id="m-veg",
                       status=MemoryStatus.contested,
                       created_at=datetime.now(timezone.utc) - timedelta(days=5))
    b = _create_memory("Mike eats steak weekly", entity_ids=["mike"], mem_id="m-steak",
                       status=MemoryStatus.contested,
                       created_at=datetime.now(timezone.utc) - timedelta(days=5))
    a.contested_with = ["m-steak"]
    b.contested_with = ["m-veg"]
    temp_store.add_memory(a)
    temp_store.add_memory(b)
    correction = _create_memory("Actually, Mike stopped eating meat last month",
                                entity_ids=["mike"], mem_id="m-correction")
    temp_store.add_memory(correction)

    llm = MockLLMClient(conflicts=(
        '{"conflicts": [{"old_memory_id": "m-steak", "new_memory_id": "m-correction",'
        ' "fact_type": "mutable", "reason": "correction settles it"}]}'
    ))
    report = run_reflection(temp_store, llm)

    assert report["memories_superseded"] == 1
    assert report["memories_healed"] == 1
    assert temp_store.get_memory("m-steak").status == MemoryStatus.superseded
    assert temp_store.get_memory("m-veg").status == MemoryStatus.active


# ── Opinion formation (5a, D6) ─────────────────────────────────────────────

def test_opinion_formation_from_world_and_experience(temp_store):
    _add_entity(temp_store, "mike")
    ids = []
    for i, (text, network) in enumerate([
        ("Mike pushed back on the deadline", Network.experience),
        ("Mike asked for more QA time", Network.experience),
        ("Mike's team runs full regression before every release", Network.world),
    ]):
        mem = _create_memory(text, network=network, entity_ids=["mike"], mem_id=f"m-src-{i}")
        temp_store.add_memory(mem)
        ids.append(mem.id)

    llm = MockLLMClient(opinions=(
        '{"opinions": [{"text": "Mike prioritizes quality over speed",'
        f' "source_memory_ids": ["{ids[0]}", "{ids[1]}"]}}]}}'
    ))
    report = run_reflection(temp_store, llm)

    assert report["opinions_formed"] == 1
    opinions = temp_store.list_memories(network=Network.opinion)
    assert len(opinions) == 1
    opinion = opinions[0]
    assert set(opinion.source_memory_ids) == {ids[0], ids[1]}
    assert opinion.entity_ids == ["mike"]

    edges = temp_store.get_edges_for_memory(opinion.id)
    derived = [e for e in edges if e.kind == EdgeKind.derived_from]
    assert len(derived) == 2


def test_opinion_merges_into_existing_equivalent(temp_store):
    """D6: equivalent opinion → append sources (strength grows), no duplicate.
    (Mock embeddings are identical vectors, so any candidate matches.)"""
    _add_entity(temp_store, "mike")
    existing = _create_memory("Mike values quality", network=Network.opinion,
                              entity_ids=["mike"], mem_id="m-op",
                              source_memory_ids=["m-old-src"])
    temp_store.add_memory(existing)
    for i in range(3):
        temp_store.add_memory(_create_memory(f"Mike did quality thing {i}",
                                             entity_ids=["mike"], mem_id=f"m-q-{i}"))

    llm = MockLLMClient(opinions=(
        '{"opinions": [{"text": "Mike cares about quality",'
        ' "source_memory_ids": ["m-q-0", "m-q-1"]}]}'
    ))
    report = run_reflection(temp_store, llm)

    assert report["opinions_merged"] == 1
    assert report["opinions_formed"] == 0
    opinions = temp_store.list_memories(network=Network.opinion)
    assert len(opinions) == 1  # still just one opinion
    assert set(opinions[0].source_memory_ids) == {"m-old-src", "m-q-0", "m-q-1"}


def test_opinion_requires_min_sources(temp_store):
    _add_entity(temp_store, "mike")
    for i in range(3):
        temp_store.add_memory(_create_memory(f"Mike fact {i}", entity_ids=["mike"],
                                             mem_id=f"m-f-{i}"))
    llm = MockLLMClient(opinions=(
        '{"opinions": [{"text": "Weak hunch", "source_memory_ids": ["m-f-0"]}]}'
    ))
    report = run_reflection(temp_store, llm)
    assert report["opinions_formed"] == 0


# ── Opinion invalidation cascade (5c) ──────────────────────────────────────

def test_cascade_demotes_opinion_when_sources_contested(temp_store):
    """Spec checkpoint: contradiction demotes the dependent opinion."""
    _add_entity(temp_store, "mike")
    a = _create_memory("Mike loves spicy food", entity_ids=["mike"], mem_id="m-spicy")
    b = _create_memory("Mike cannot handle any spice", entity_ids=["mike"], mem_id="m-mild")
    temp_store.add_memory(a)
    temp_store.add_memory(b)
    opinion = _create_memory("Mike is adventurous with food", network=Network.opinion,
                             entity_ids=["mike"], mem_id="m-opinion",
                             source_memory_ids=["m-spicy", "m-mild"])
    temp_store.add_memory(opinion)

    llm = MockLLMClient(conflicts=(
        '{"conflicts": [{"old_memory_id": "m-spicy", "new_memory_id": "m-mild",'
        ' "fact_type": "fixed", "reason": "contradictory preferences"}]}'
    ))
    report = run_reflection(temp_store, llm)

    assert report["opinions_demoted"] == 1
    assert temp_store.get_memory("m-opinion").status == MemoryStatus.demoted


def test_cascade_keeps_opinion_with_two_valid_sources(temp_store):
    _add_entity(temp_store, "mike")
    for i in range(3):
        temp_store.add_memory(_create_memory(f"Mike source {i}", entity_ids=["mike"],
                                             mem_id=f"m-s-{i}"))
    temp_store.add_memory(_create_memory("Mike is contradicted", entity_ids=["mike"],
                                         mem_id="m-bad"))
    opinion = _create_memory("Mike opinion", network=Network.opinion,
                             entity_ids=["mike"], mem_id="m-op3",
                             source_memory_ids=["m-s-0", "m-s-1", "m-bad"])
    temp_store.add_memory(opinion)

    llm = MockLLMClient(conflicts=(
        '{"conflicts": [{"old_memory_id": "m-bad", "new_memory_id": "m-s-2",'
        ' "fact_type": "mutable", "reason": "updated"}]}'
    ))
    report = run_reflection(temp_store, llm)

    assert report["opinions_demoted"] == 0
    kept = temp_store.get_memory("m-op3")
    assert kept.status == MemoryStatus.active
    assert set(kept.source_memory_ids) == {"m-s-0", "m-s-1"}


# ── Housekeeping (5d) ──────────────────────────────────────────────────────

def test_reflect_cycles_incremented(temp_store):
    temp_store.add_memory(_create_memory("stable fact", mem_id="m-cycles"))
    run_reflection(temp_store, MockLLMClient())
    assert temp_store.get_memory("m-cycles").reflect_cycles == 1
    run_reflection(temp_store, MockLLMClient())
    assert temp_store.get_memory("m-cycles").reflect_cycles == 2


def test_entity_nomination_for_recurring_names(temp_store):
    for i in range(cfg.ENTITY_CREATION_MIN_RECURRING):
        temp_store.add_memory(_create_memory(
            f"Zorblatt mentioned the roadmap item {i}", mem_id=f"m-z-{i}"))
    report = run_reflection(temp_store, MockLLMClient())
    names = [s["name"] for s in report["entity_suggestions"]]
    assert "Zorblatt" in names


def test_no_stale_demotion_in_reflection(temp_store):
    """Aging alone must never demote a memory — decay is brightness-based
    at query time, not a status flip (D13 spirit)."""
    old = _create_memory("very old but true fact", mem_id="m-old",
                         created_at=datetime.now(timezone.utc) - timedelta(days=400))
    temp_store.add_memory(old)
    run_reflection(temp_store, MockLLMClient())
    assert temp_store.get_memory("m-old").status == MemoryStatus.active


# ── Report / API surface ───────────────────────────────────────────────────

def test_reflection_report_structure(temp_store):
    report = run_reflection(temp_store, MockLLMClient())
    for key in (
        "contradictions_detected", "memories_superseded", "memories_contested",
        "memories_healed", "opinions_formed", "opinions_merged",
        "opinions_demoted", "entity_suggestions", "reflection_records",
    ):
        assert key in report
    assert "error" not in report


def test_compat_wrappers_run(temp_store):
    _add_entity(temp_store, "mike")
    temp_store.add_memory(_create_memory("Mike fact one", entity_ids=["mike"]))
    temp_store.add_memory(_create_memory("Mike fact two", entity_ids=["mike"]))
    assert detect_contradictions(temp_store, MockLLMClient()) == []
    assert form_opinions(temp_store, MockLLMClient()) == []
