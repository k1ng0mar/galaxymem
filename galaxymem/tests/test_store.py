"""Unit tests for Store class — core storage operations."""

import pytest
from datetime import datetime, timezone
from galaxymem.models import (
    MemoryRecord, Network, MemoryStatus,
    EntityRecord, EntityType,
    EdgeRecord, EdgeKind,
    IdentityLink, LinkMethod,
)


class TestStoreMemoryOperations:
    """Test memory CRUD operations."""

    def test_add_and_get_memory(self, temp_db, sample_memory):
        """Test adding and retrieving a memory."""
        temp_db.add_memory(sample_memory)
        retrieved = temp_db.get_memory(sample_memory.id)

        assert retrieved is not None
        assert retrieved.id == sample_memory.id
        assert retrieved.text == sample_memory.text
        assert retrieved.network == sample_memory.network

    def test_get_memory_not_found(self, temp_db):
        """Test retrieving non-existent memory returns None."""
        result = temp_db.get_memory("nonexistent-id")
        assert result is None

    def test_update_memory_field_recall_count(self, temp_db, sample_memory):
        """Test updating memory recall_count via update_memory_field."""
        temp_db.add_memory(sample_memory)
        temp_db.update_memory_field(sample_memory.id, recall_count=3)

        updated = temp_db.get_memory(sample_memory.id)
        assert updated.recall_count == 3

    def test_update_memory_recall_count(self, temp_db, sample_memory):
        """Test that touch_memory increments recall_count."""
        temp_db.add_memory(sample_memory)
        temp_db.touch_memory(sample_memory.id)

        updated = temp_db.get_memory(sample_memory.id)
        assert updated.recall_count == 1

    def test_update_memory_status(self, temp_db, sample_memory):
        """Test updating memory status."""
        temp_db.add_memory(sample_memory)
        temp_db.update_memory_status(sample_memory.id, MemoryStatus.superseded)

        updated = temp_db.get_memory(sample_memory.id)
        assert updated.status == MemoryStatus.superseded

    def test_count_memories(self, temp_db):
        """Test counting memories via stats()."""
        for i in range(3):
            mem = MemoryRecord(
                id=f"mem-{i}",
                text=f"Memory {i}",
                network=Network.world,
                entity_ids=["test"],
                status=MemoryStatus.active,
            )
            temp_db.add_memory(mem)

        s = temp_db.stats()
        assert s["total_memories"] == 3

    def test_count_memories_by_network(self, temp_db):
        """Test counting memories grouped by network via stats()."""
        networks = [Network.world, Network.world, Network.experience, Network.opinion]
        for i, network in enumerate(networks):
            mem = MemoryRecord(
                id=f"mem-{i}",
                text=f"Memory {i}",
                network=network,
                entity_ids=["test"],
                status=MemoryStatus.active,
            )
            temp_db.add_memory(mem)

        s = temp_db.stats()
        assert s["memories_per_network"]["world"] == 2
        assert s["memories_per_network"]["experience"] == 1
        assert s["memories_per_network"]["opinion"] == 1


class TestStoreEntityOperations:
    """Test entity CRUD operations."""

    def test_add_and_get_entity(self, temp_db):
        """Test adding and retrieving an entity."""
        entity = EntityRecord(
            id="alice",
            type=EntityType.person,
            label="Alice",
        )
        temp_db.add_entity(entity)
        retrieved = temp_db.get_entity("alice")

        assert retrieved is not None
        assert retrieved.id == "alice"
        assert retrieved.label == "Alice"

    def test_get_entity_not_found(self, temp_db):
        """Test retrieving non-existent entity returns None."""
        result = temp_db.get_entity("nonexistent-entity")
        assert result is None

    def test_update_entity(self, temp_db):
        """Test updating entity fields."""
        entity = EntityRecord(
            id="alice",
            type=EntityType.person,
            label="Alice",
        )
        temp_db.add_entity(entity)
        temp_db.update_entity("alice", status_line="Updated status")

        updated = temp_db.get_entity("alice")
        assert updated.status_line == "Updated status"

    def test_list_entities(self, temp_db):
        """Test listing all entities."""
        for i in range(2):
            entity = EntityRecord(
                id=f"entity-{i}",
                type=EntityType.person,
                label=f"Entity {i}",
            )
            temp_db.add_entity(entity)

        entities = temp_db.list_entities()
        assert len(entities) == 2


class TestStoreEdgeOperations:
    """Test edge CRUD operations."""

    def test_add_and_get_edges(self, temp_db):
        """Test adding and retrieving edges."""
        # Create two memories first
        mem1 = MemoryRecord(
            id="mem-1", text="Memory 1",
            network=Network.world, entity_ids=["test"],
            status=MemoryStatus.active,
        )
        mem2 = MemoryRecord(
            id="mem-2", text="Memory 2",
            network=Network.world, entity_ids=["test"],
            status=MemoryStatus.active,
        )
        temp_db.add_memory(mem1)
        temp_db.add_memory(mem2)

        # Add edge
        edge = EdgeRecord(
            from_id="mem-1",
            to_id="mem-2",
            kind=EdgeKind.temporal,
            weight=0.8,
        )
        temp_db.add_edge(edge)

        # Retrieve edges
        edges = temp_db.get_edges_for_memory("mem-1")
        assert len(edges) >= 1


class TestStoreIdentityOperations:
    """Test identity link operations."""

    def test_add_and_resolve_identity(self, temp_db):
        """Test adding and resolving identity links."""
        link = IdentityLink(
            platform="telegram",
            external_id="user123",
            entity_id="alice",
            created_by=LinkMethod.explicit,
        )
        temp_db.add_identity_link(link)

        resolved = temp_db.resolve_identity("telegram", "user123")
        assert resolved is not None
        assert resolved.entity_id == "alice"

    def test_resolve_identity_not_found(self, temp_db):
        """Test resolving non-existent identity returns None."""
        result = temp_db.resolve_identity("telegram", "nonexistent")
        assert result is None

    def test_get_identity_links_for_entity(self, temp_db):
        """Test getting all identity links for an entity."""
        for i in range(2):
            link = IdentityLink(
                platform=f"platform-{i}",
                external_id=f"user-{i}",
                entity_id="alice",
                created_by=LinkMethod.explicit,
            )
            temp_db.add_identity_link(link)

        links = temp_db.get_identity_links_for_entity("alice")
        assert len(links) == 2


class TestStorePromotionQueue:
    """Test promotion queue operations."""

    def test_add_and_get_queue(self, temp_db):
        """Test adding and retrieving from promotion queue."""
        temp_db.add_to_promotion_queue("mem-001")
        temp_db.add_to_promotion_queue("mem-002")

        queue = temp_db.list_promotion_queue()
        assert len(queue) == 2

    def test_remove_from_queue(self, temp_db):
        """Test removing from promotion queue."""
        temp_db.add_to_promotion_queue("mem-001")
        temp_db.remove_from_promotion_queue("mem-001")

        queue = temp_db.list_promotion_queue()
        assert len(queue) == 0


class TestStoreSearchOperations:
    """Test search operations."""

    def test_keyword_search(self, temp_db):
        """Test keyword-based search."""
        mem = MemoryRecord(
            id="mem-keyword",
            text="The quick brown fox jumps",
            network=Network.world,
            entity_ids=["test"],
            status=MemoryStatus.active,
        )
        temp_db.add_memory(mem)

        results = temp_db.keyword_search("fox", k=1)
        assert len(results) > 0
