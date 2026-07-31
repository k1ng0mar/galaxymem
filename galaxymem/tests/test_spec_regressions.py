"""Regression tests for spec-review fixes.

Each test pins a specific defect found in the 2026-07-05 spec review:
- D8 hard filter must not leak on slug substrings ("sam" vs "samuel")
- Contested memories are excluded from default deep recall (D11)
- Brightness anchors on last_recalled_at — recall arrests decay
- store.as_of() returns real historical state and is read-only
- Recall scope includes unscoped world facts alongside requested entities
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from galaxymem.models import MemoryRecord, MemoryStatus, Network
from galaxymem.recall import _brightness, deep_recall
from galaxymem import config as cfg


def _mem(mem_id: str, text: str, *, entity_ids=None, network=Network.world,
         status=MemoryStatus.active, created_at=None, last_recalled_at=None,
         recall_count=0) -> MemoryRecord:
    return MemoryRecord(
        id=mem_id, text=text, network=network,
        entity_ids=entity_ids or [], status=status,
        created_at=created_at or datetime.now(timezone.utc),
        last_recalled_at=last_recalled_at, recall_count=recall_count,
    )


class TestEntityFilterExactness:
    def test_substring_slug_does_not_leak(self, temp_db):
        """D8: filtering for 'sam' must not return memories scoped to 'samuel'."""
        temp_db.add_memory(_mem("m-sam", "Sam prefers tea", entity_ids=["sam"]))
        temp_db.add_memory(_mem("m-samuel", "Samuel prefers coffee", entity_ids=["samuel"]))

        results = temp_db.vector_search("preferred hot drink", k=10, entity_filter=["sam"])
        ids = {m.id for m, _ in results}
        assert "m-sam" in ids
        assert "m-samuel" not in ids

    def test_list_memories_entity_filter_exact(self, temp_db):
        temp_db.add_memory(_mem("m-nyx", "Nyx heartbeat is active", entity_ids=["nyx"]))
        temp_db.add_memory(_mem("m-nyx-react", "Nyx-react uses vite",
                                entity_ids=["nyx-react"]))
        got = temp_db.list_memories(entity_ids=["nyx"])
        assert [m.id for m in got] == ["m-nyx"]


class TestRecallScoping:
    def test_contested_excluded_from_default_recall(self, temp_db):
        temp_db.add_memory(_mem("m-ok", "The server runs Ubuntu"))
        temp_db.add_memory(_mem("m-contested", "The server runs Debian",
                                status=MemoryStatus.contested))
        results = deep_recall("what OS does the server run", temp_db)
        ids = {m.id for m in results}
        assert "m-ok" in ids
        assert "m-contested" not in ids

    def test_archived_excluded_from_default_recall(self, temp_db):
        temp_db.add_memory(_mem("m-gone", "Forgotten secret fact",
                                status=MemoryStatus.archived))
        results = deep_recall("forgotten secret fact", temp_db)
        assert all(m.id != "m-gone" for m in results)

    def test_unscoped_world_facts_included_with_entity_filter(self, temp_db):
        """D8: scope = requested entities + self + unscoped world facts."""
        temp_db.add_memory(_mem("m-scoped", "Mike deployed the gateway",
                                entity_ids=["mike"]))
        temp_db.add_memory(_mem("m-unscoped", "Gateways restart via hermes gateway restart"))
        temp_db.add_memory(_mem("m-other", "Sarah likes hiking", entity_ids=["sarah"]))

        results = deep_recall("gateway", temp_db, entity_ids=["mike"])
        ids = {m.id for m in results}
        assert "m-scoped" in ids
        assert "m-unscoped" in ids       # unscoped world fact rides along
        assert "m-other" not in ids      # hard filter holds

    def test_recall_touches_returned_memories(self, temp_db):
        temp_db.add_memory(_mem("m-touch", "Hermes uses LanceDB for memory"))
        deep_recall("what does hermes use for memory", temp_db)
        touched = temp_db.get_memory("m-touch")
        assert touched.recall_count == 1
        assert touched.last_recalled_at is not None


class TestBrightness:
    def test_recall_arrests_decay(self):
        """An old memory recalled yesterday outshines an old never-recalled one."""
        old = datetime.now(timezone.utc) - timedelta(days=120)
        never_recalled = _mem("m-a", "old fact", created_at=old)
        recently_recalled = _mem("m-b", "old fact", created_at=old,
                                 last_recalled_at=datetime.now(timezone.utc) - timedelta(days=1),
                                 recall_count=4)
        assert _brightness(recently_recalled) > _brightness(never_recalled)

    def test_floor_holds(self):
        ancient = _mem("m-c", "ancient", created_at=datetime.now(timezone.utc) - timedelta(days=3650))
        assert _brightness(ancient) == pytest.approx(cfg.BRIGHTNESS_FLOOR)


class TestTemporalAsOf:
    def test_as_of_sees_historical_state(self, temp_db):
        temp_db.add_memory(_mem("m-first", "Belief on Monday: project ships in June"))
        time.sleep(0.05)
        checkpoint = datetime.now(timezone.utc)
        time.sleep(0.05)
        temp_db.add_memory(_mem("m-second", "Belief on Friday: project slips to July"))

        historical = temp_db.as_of(checkpoint)
        past_ids = {m.id for m in historical.list_memories()}
        assert "m-first" in past_ids
        assert "m-second" not in past_ids

        # Live store still sees both
        assert len(temp_db.list_memories()) == 2

    def test_as_of_is_read_only(self, temp_db):
        temp_db.add_memory(_mem("m-ro", "immutable past"))
        historical = temp_db.as_of(datetime.now(timezone.utc))
        with pytest.raises(RuntimeError):
            historical.touch_memory("m-ro")

    def test_as_of_before_any_version_raises(self, temp_db):
        temp_db.add_memory(_mem("m-x", "recent fact"))
        with pytest.raises(ValueError):
            temp_db.as_of(datetime(2020, 1, 1, tzinfo=timezone.utc))

    def test_temporal_recall_does_not_touch(self, temp_db):
        temp_db.add_memory(_mem("m-frozen", "Historical belief about the roadmap"))
        time.sleep(0.05)
        checkpoint = datetime.now(timezone.utc)

        results = deep_recall("roadmap belief", temp_db, as_of=checkpoint)
        assert any(m.id == "m-frozen" for m in results)
        assert temp_db.get_memory("m-frozen").recall_count == 0
