"""Integration test: temporal retrieval arm in deep_recall."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from galaxymem import config as cfg
from galaxymem.models import MemoryRecord, Network
from galaxymem.recall import deep_recall
from galaxymem.store import Store

NOW = datetime.now(timezone.utc)


def _fresh_store(tmpdir) -> Store:
    store = Store(db_path=Path(tmpdir) / "db")
    return store.open(create_if_missing=True)


def _seed(store, mem_id, text, created_at):
    m = MemoryRecord(
        id=mem_id,
        text=text,
        network=Network.world,
        entity_ids=[],
        created_at=created_at,
    )
    store.add_memory(m)
    # created_at was defaulted into the record; force the row's timestamp
    store.update_memory_field(mem_id, created_at=created_at.isoformat())


def test_july_query_ranks_july_memory_first():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        july = datetime(2026, 7, 10, tzinfo=timezone.utc)
        recent = NOW - timedelta(days=1)

        # July fact: semantically WEAKER match. Recent fact: strong match.
        _seed(store, "m-july", "the backup server hostname was backup-01.internal",
              july)
        _seed(store, "m-recent", "the backup server hostname is backup-02.internal",
              recent)

        results = deep_recall(
            "what was the backup server hostname in 2026-07",
            store, limit=2,
        )
        ids = [m.id for m in results]
        assert "m-july" in ids, "july memory must surface for a july query"
        assert ids[0] == "m-july", (
            f"july memory must rank first, got {ids}"
        )
        store.close()


def test_non_temporal_query_unchanged():
    """No date in query → arm inert, plain ranking applies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        _seed(store, "m-a", "the api runs on port 8010",
              datetime(2026, 3, 1, tzinfo=timezone.utc))
        results = deep_recall("which port does the api run on", store, limit=2)
        assert any(m.id == "m-a" for m in results)
        store.close()


if __name__ == "__main__":
    print("run via pytest")
