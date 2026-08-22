"""Integration tests for retain.py Pass 1 and Pass 2.

Asserts the locked-decision behavior:
- D3: unresolved entity labels never auto-create entities
- D7: memories are scoped by SUBJECT; the speaker is tracked separately
- D4: an unknown (platform, external_id) speaker gets a provisional entity
- Dedup: re-stating a fact touches the original instead of duplicating
- Failure drill: LLM failure leaves flags unprocessed (retry next trigger)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from galaxymem.entities import create_entity
from galaxymem.models import EntityType, MemoryStatus, Network
from galaxymem.retain import flag_turn, process_pending_flags, should_trigger_pass2
from galaxymem.store import Store


class MockLLMClient:
    """Mock LLM client returning canned responses."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "[]"


class FailingLLMClient:
    """Mock LLM that always raises — for the failure drill."""

    def __init__(self):
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        raise RuntimeError("simulated LLM outage")


TEST_TURNS = [
    ("Remember that I'm allergic to peanuts", "session-1", "telegram", "user123"),
    ("My name is Alice and I live in Seattle", "session-1", "telegram", "user123"),
    ("I'm 28 years old", "session-1", "telegram", "user123"),
    ("We decided to use PostgreSQL for the database", "session-1", "telegram", "user123"),
    ("The deadline is March 15th", "session-1", "telegram", "user123"),
    ("Actually, I meant the deadline is March 20th", "session-1", "telegram", "user123"),
    ("I love working with Python", "session-1", "telegram", "user123"),
    ("I hate JavaScript", "session-1", "telegram", "user123"),
    ("I have a dog named Max", "session-1", "telegram", "user123"),
    ("I'm worried about the project timeline", "session-1", "telegram", "user123"),
    ("The architecture uses microservices", "session-1", "telegram", "user123"),
    ("I'm from Canada", "session-1", "telegram", "user123"),
    ("My favorite color is blue", "session-1", "telegram", "user123"),
    ("I work at Google", "session-1", "telegram", "user123"),
    ("We're building a REST API", "session-1", "telegram", "user123"),
]

MOCK_EXTRACTION = """[
  {"text": "User is allergic to peanuts", "network": "world", "entity_labels": ["self"]},
  {"text": "User's name is Alice and she lives in Seattle", "network": "world", "entity_labels": ["self"]},
  {"text": "The project deadline is March 20th", "network": "world", "entity_labels": ["Hermes Project"]},
  {"text": "The project uses PostgreSQL", "network": "world", "entity_labels": ["Hermes Project"]},
  {"text": "User loves working with Python", "network": "opinion", "entity_labels": ["self"]},
  {"text": "User is worried about the project timeline", "network": "observation", "entity_labels": ["self", "Hermes Project"]},
  {"text": "Sarah from accounting approved the budget", "network": "world", "entity_labels": ["Sarah"]}
]"""


def _fresh_store(tmpdir: str) -> Store:
    return Store(Path(tmpdir) / "test.db").open(create_if_missing=True)


def test_retain_integration():
    """Flag 15 turns, trigger Pass 2, verify locked-decision behavior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)

        # A tracked project entity exists; "Sarah" does NOT.
        create_entity(store, label="Hermes Project", entity_type=EntityType.project,
                      slug="hermes-project")

        # Pass 1: all 15 turns match a flag rule
        flagged = sum(
            flag_turn(store, text, sid, platform, speaker)
            for text, sid, platform, speaker in TEST_TURNS
        )
        assert flagged == 15, f"Expected 15 flagged turns, got {flagged}"
        assert should_trigger_pass2(store), "Pass 2 should trigger with 15 flags"

        # Pass 2
        llm = MockLLMClient([MOCK_EXTRACTION])
        created = process_pending_flags(store, llm, batch_size=20)
        assert created == 7, f"Expected 7 memories, got {created}"
        assert store.unprocessed_flags() == []

        memories = store.list_memories()
        by_text = {m.text: m for m in memories}

        # D7: subject scoping — the project fact is filed under the project,
        # and NOT force-tagged with self
        deadline = by_text["The project deadline is March 20th"]
        assert deadline.entity_ids == ["hermes-project"]

        # D3: "Sarah" is untracked → label dropped, name preserved in text,
        # NO person entity auto-created
        sarah_mem = by_text["Sarah from accounting approved the budget"]
        assert sarah_mem.entity_ids == []
        labels = {e.label for e in store.list_entities()}
        assert "Sarah" not in labels

        # D4/D7: the speaker resolved to a provisional entity, recorded as
        # speaker (who said it), not as subject
        provisionals = [e for e in store.list_entities()
                        if e.type == EntityType.provisional]
        assert len(provisionals) == 1
        assert all(m.speaker_entity_id == provisionals[0].id for m in memories)

        # Self-alias labels resolve to the self entity
        allergy = by_text["User is allergic to peanuts"]
        assert allergy.entity_ids == ["self"]

        # Edges: temporal chain + shared_entity for co-scoped memories
        edge_kinds = {e.kind.value
                      for m in memories
                      for e in store.get_edges_for_memory(m.id)}
        assert "temporal" in edge_kinds
        assert "shared_entity" in edge_kinds

        assert not should_trigger_pass2(store)
        store.close()


def test_dedup_touches_instead_of_duplicating():
    """Phase 3 step 4: an equivalent restatement must not create a second
    memory — the original is touched."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)

        response = '[{"text": "User is allergic to peanuts", "network": "world", "entity_labels": ["self"]}]'

        flag_turn(store, "Remember that I'm allergic to peanuts",
                  "session-1", "cli", "hermes-user")
        assert process_pending_flags(store, MockLLMClient([response])) == 1

        # Same fact re-stated in a later session
        flag_turn(store, "Don't forget I'm allergic to peanuts",
                  "session-2", "cli", "hermes-user")
        assert process_pending_flags(store, MockLLMClient([response])) == 0

        memories = store.list_memories()
        assert len(memories) == 1
        assert memories[0].recall_count == 1  # touched by the dedup hit
        store.close()


def test_failed_llm_leaves_flags_unprocessed():
    """Phase 10 failure drill: flags are never dropped. A failing batch is
    retried once, then left for the next trigger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)

        flag_turn(store, "Remember that the launch date is Friday",
                  "session-1", "cli", "hermes-user")

        failing = FailingLLMClient()
        created = process_pending_flags(store, failing)

        assert created == 0
        assert failing.call_count == 2, "Batch must be retried exactly once"
        remaining = store.unprocessed_flags()
        assert len(remaining) == 1, "Flags must survive an LLM outage"

        # Next trigger with a working LLM picks the same flag up
        response = '[{"text": "The launch date is Friday", "network": "world", "entity_labels": []}]'
        assert process_pending_flags(store, MockLLMClient([response])) == 1
        assert store.unprocessed_flags() == []
        store.close()


def test_tracked_entity_label_flags_turn():
    """Pass-1 rule: plain mention of a tracked entity's label flags the turn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        create_entity(store, label="Qruzpay", entity_type=EntityType.project)

        # No static rule matches this sentence — only the entity label does.
        # (Bust the module-level label cache by expiring it.)
        from galaxymem import retain as retain_mod
        retain_mod._ENTITY_LABEL_CACHE["expires"] = 0.0

        assert flag_turn(store, "How are things over at Qruzpay these days",
                         "session-1", "cli", "hermes-user") is True
        flags = store.unprocessed_flags()
        assert flags[0].flag_reason == "tracked_entity"
        store.close()


def test_flag_turn_redacts_secrets():
    """Pass-1: credential-shaped strings are redacted before persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        secret_turn = "remember that my api key is sk-proj-abc123def456GHI789jklMNO"
        assert flag_turn(store, secret_turn,
                         "session-redact", "cli", "hermes-user") is True
        flags = store.unprocessed_flags()
        assert len(flags) == 1
        assert "sk-proj-abc123def456" not in flags[0].turn_text
        assert "[REDACTED]" in flags[0].turn_text
        store.close()


if __name__ == "__main__":
    test_retain_integration()
    print("All tests passed!")
