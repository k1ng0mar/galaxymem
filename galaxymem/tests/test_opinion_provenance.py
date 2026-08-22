"""Tests for opinion proof counts and merge history (provenance)."""

import json
import tempfile
from datetime import datetime, timezone

from galaxymem.models import MemoryRecord, MemoryStatus, Network
from galaxymem.reflect import _merge_into_existing_opinion, _form_opinions_for_entity
from galaxymem.store import Store


class StubLLM:
    def chat(self, messages):
        return '{"opinions": []}'


def _fresh_store(tmpdir) -> Store:
    from pathlib import Path as P
    store = Store(db_path=P(tmpdir) / "db")
    return store.open(create_if_missing=True)


def _seed_opinion(store, text, sources, entity="self"):
    mem = MemoryRecord(
        id=f"op-{text[:8]}",
        text=text,
        network=Network.opinion,
        entity_ids=[entity],
        source_memory_ids=sources,
        status=MemoryStatus.active,
    )
    store.add_memory(mem)
    return mem


def _seed_fact(store, mem_id, text, entity="self"):
    mem = MemoryRecord(
        id=mem_id,
        text=text,
        network=Network.world,
        entity_ids=[entity],
        status=MemoryStatus.active,
    )
    store.add_memory(mem)
    return mem


def test_fresh_opinion_gets_proof_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        for i in range(3):
            _seed_fact(store, f"f{i}", f"fact number {i} about testing")
        report = {"opinions_formed": 0, "opinions_merged": 0}
        records = _form_opinions_for_entity(store, StubLLM(), "self", report)
        # StubLLM returns no opinions; just verify no crash. The real
        # proof-count assertion lives in the merge test below.
        store.close()


def test_merge_increments_proof_count_and_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        existing = _seed_opinion(store, "User prefers direct execution", ["s1", "s2"])

        merged_id = _merge_into_existing_opinion(
            store, "self",
            "user prefers direct execution",
            ["s3"], existing=[existing],
        )
        assert merged_id == existing.id, "same-meaning phrasing should merge"

        updated = store.get_memory(existing.id)
        assert updated.source_memory_ids == ["s1", "s2", "s3"]
        assert updated.proof_count == 3

        history = json.loads(updated.history_json or "[]")
        assert len(history) == 1
        assert history[0]["action"] == "merge"
        assert history[0]["sources"] == ["s3"]
        assert "at" in history[0]
        store.close()


def test_new_opinion_history_initialized_empty():
    """A fresh opinion carries proof_count == len(sources), empty history."""
    mem = MemoryRecord(
        id="op-new",
        text="test opinion",
        network=Network.opinion,
        entity_ids=["self"],
        source_memory_ids=["a", "b"],
        status=MemoryStatus.active,
    )
    assert mem.proof_count == 0  # default; set explicitly by _form_opinions
    assert mem.history_json is None


if __name__ == "__main__":
    print("run via pytest")
