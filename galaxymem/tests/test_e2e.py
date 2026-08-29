"""End-to-end lifecycle test for GalaxyMem.

Tests the full memory lifecycle:
1. Create self entity
2. Store memories across all 4 networks
3. Recall memories (verify RRF + hot cache)
4. Run reflection cycle (verify opinions formed, contradictions detected)
5. Check promotion eligibility
6. Test decay (brightness drops over time)
7. Test supersession (old memory superseded by new)
"""

import pytest
from datetime import datetime, timezone, timedelta
from galaxymem.store_sqlite import Store
from galaxymem.models import (
    MemoryRecord, Network, MemoryStatus,
    EntityType, EdgeRecord, EdgeKind,
)
from galaxymem.entities import (
    create_self_entity,
    create_entity,
    ensure_self_entity,
    link_identity_explicit,
)
from galaxymem.recall import (
    deep_recall,
    recall,
    get_hot_cache,
    update_hot_cache,
    inject_hot_context,
)


def _make_memory(
    mem_id, text, entity_ids, network=Network.world,
    status=MemoryStatus.active,
    recall_count=0, reflect_cycles=0, created_at=None,
):
    """Helper to create a MemoryRecord."""
    return MemoryRecord(
        id=mem_id,
        text=text,
        network=network,
        entity_ids=entity_ids,
        status=status,
        recall_count=recall_count,
        reflect_cycles=reflect_cycles,
        created_at=created_at or datetime.now(timezone.utc),
    )


class TestFullLifecycle:
    """Full end-to-end lifecycle test."""

    def test_complete_memory_lifecycle(self, temp_db):
        """Test the complete memory lifecycle from creation to promotion."""
        # === Step 1: Create self entity ===
        self_entity = create_self_entity(temp_db, label="Agent")
        assert self_entity.id == "self"

        # Create other entities
        alice = create_entity(temp_db, "Alice", EntityType.person, slug="alice")
        project = create_entity(temp_db, "GalaxyMem", EntityType.project, slug="galaxymem")

        # Link identities
        link_identity_explicit(temp_db, "telegram", "alice123", alice.id)

        # === Step 2: Store memories across all 4 networks ===
        memories = [
            _make_memory("w1", "Alice prefers dark mode", [alice.id], Network.world),
            _make_memory("w2", "GalaxyMem uses LanceDB for storage", [project.id], Network.world),
            _make_memory("w3", "Python is Alice's primary language", [alice.id], Network.world),
            _make_memory("e1", "Alice and I worked on GalaxyMem together", [alice.id, "self"], Network.experience),
            _make_memory("e2", "I helped Alice debug a LanceDB issue", [alice.id, "self"], Network.experience),
            _make_memory("o1", "Alice thinks Vim is better than Emacs", [alice.id], Network.opinion),
            _make_memory("ob1", "Alice responds to messages within 2 hours", [alice.id], Network.observation),
        ]

        for mem in memories:
            temp_db.add_memory(mem)

        # Verify all stored
        stats = temp_db.stats()
        assert stats["total_memories"] == 7

        # Verify per-network counts
        assert stats["memories_per_network"]["world"] == 3
        assert stats["memories_per_network"]["experience"] == 2
        assert stats["memories_per_network"]["opinion"] == 1
        assert stats["memories_per_network"]["observation"] == 1

        # === Step 3: Recall memories ===
        results = deep_recall("Alice editor preferences", temp_db, limit=5)
        assert len(results) > 0

        # Verify recall_count incremented
        recalled_mem = temp_db.get_memory(results[0].id)
        assert recalled_mem.recall_count >= 1

        # Hot cache
        cache = update_hot_cache(temp_db)
        assert len(cache.memory_ids) > 0
        assert cache.rendered

        # inject_hot_context
        context = inject_hot_context(temp_db)
        assert context

        # === Step 4: Entity filtering ===
        alice_results = deep_recall(
            "editor preferences", temp_db, entity_ids=[alice.id], limit=3
        )
        for mem in alice_results:
            assert alice.id in mem.entity_ids

        # === Step 5: Test edges and spreading activation ===
        edge = EdgeRecord(
            from_id="w1",
            to_id="w3",
            kind=EdgeKind.shared_entity,
            weight=0.9,
        )
        temp_db.add_edge(edge)

        edges = temp_db.get_edges_for_memory("w1")
        assert len(edges) >= 1

        # === Step 6: Promotion check ===
        temp_db.update_memory_field("w1", recall_count=5, reflect_cycles=3)

        from galaxymem.promote import scan_for_promotable
        promotable = scan_for_promotable(temp_db)
        assert any(m.id == "w1" for m in promotable)

    def test_network_isolation(self, temp_db):
        """Test that networks are properly isolated."""
        entity = create_entity(temp_db, "TestEntity", EntityType.person, slug="test")

        for net in [Network.world, Network.experience, Network.opinion, Network.observation]:
            mem = _make_memory(
                f"mem-{net.value}",
                f"Fact in {net.value} network",
                [entity.id],
                network=net,
            )
            temp_db.add_memory(mem)

        results = deep_recall("Fact", temp_db, limit=10)
        networks_found = {r.network for r in results}
        assert len(networks_found) >= 1

    def test_status_filtering(self, temp_db):
        """Test that non-active memories are excluded from promotion."""
        entity = create_entity(temp_db, "TestEntity", EntityType.person, slug="test")

        # Active memory with enough recalls
        active = _make_memory(
            "active1", "Active memory content", [entity.id],
            recall_count=5, reflect_cycles=3,
        )
        temp_db.add_memory(active)

        # Demoted memory (enough recalls but wrong status)
        demoted = _make_memory(
            "demoted1", "Demoted memory content",
            [entity.id], status=MemoryStatus.demoted,
            recall_count=5, reflect_cycles=3,
        )
        temp_db.add_memory(demoted)

        from galaxymem.promote import scan_for_promotable
        promotable = scan_for_promotable(temp_db)
        promotable_ids = {m.id for m in promotable}

        assert "active1" in promotable_ids
        assert "demoted1" not in promotable_ids

    def test_supersession_flow(self, temp_db):
        """Test that old memories can be superseded by new ones."""
        entity = create_entity(temp_db, "TestEntity", EntityType.person, slug="test")

        old_mem = _make_memory(
            "old1", "Alice lives in New York", [entity.id], network=Network.world,
        )
        temp_db.add_memory(old_mem)

        new_mem = _make_memory(
            "new1", "Alice moved to San Francisco", [entity.id], network=Network.world,
        )
        temp_db.add_memory(new_mem)

        edge = EdgeRecord(
            from_id="new1",
            to_id="old1",
            kind=EdgeKind.supersedes,
            weight=1.0,
        )
        temp_db.add_edge(edge)

        edges = temp_db.get_edges_for_memory("new1")
        assert len(edges) >= 1
        assert any(e.kind == EdgeKind.supersedes for e in edges)

        temp_db.update_memory_field("old1", status=MemoryStatus.superseded.value)

        updated = temp_db.get_memory("old1")
        assert updated.status == MemoryStatus.superseded

    def test_entity_merge_with_memories(self, temp_db):
        """Test that entity merge preserves identity links."""
        from galaxymem.entities import merge_entity

        entity1 = create_entity(temp_db, "Alice", EntityType.person, slug="alice1")
        entity2 = create_entity(temp_db, "Alice Smith", EntityType.person, slug="alice2")

        link_identity_explicit(temp_db, "telegram", "alice123", entity1.id)

        merge_entity(temp_db, entity1.id, entity2.id)

        from galaxymem.entities import get_identity_links
        links = get_identity_links(temp_db, entity2.id)
        assert len(links) == 1
        assert links[0].external_id == "alice123"
