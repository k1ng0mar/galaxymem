"""Unit tests for entities.py — entity management and lifecycle."""

import pytest
from datetime import datetime, timezone
from galaxymem.entities import (
    create_entity,
    create_self_entity,
    get_self_entity,
    ensure_self_entity,
    update_entity,
    create_provisional,
    merge_entity,
    resolve_or_provision,
    link_identity_explicit,
    get_identity_links,
    list_entities,
    list_provisionals,
    get_entity_card,
)
from galaxymem.models import EntityType, MemoryRecord, Network, MemoryStatus


class TestEntityCreation:
    """Test entity creation functions."""

    def test_create_entity_person(self, temp_db):
        """Test creating a person entity."""
        entity = create_entity(
            temp_db,
            label="Alice",
            entity_type=EntityType.person,
            card={"role": "developer"},
        )
        
        assert entity.id is not None
        assert entity.label == "Alice"
        assert entity.type == EntityType.person
        assert entity.card["role"] == "developer"
        
        # Verify it's retrievable
        retrieved = temp_db.get_entity(entity.id)
        assert retrieved is not None
        assert retrieved.label == "Alice"

    def test_create_entity_project(self, temp_db):
        """Test creating a project entity."""
        entity = create_entity(
            temp_db,
            label="GalaxyMem",
            entity_type=EntityType.project,
            status_line="Active development",
        )
        
        assert entity.type == EntityType.project
        assert entity.status_line == "Active development"

    def test_create_entity_auto_slug(self, temp_db):
        """Test that entities get auto-generated slugs."""
        entity = create_entity(temp_db, "Alice Smith", EntityType.person)
        
        # Slug should be derived from label
        assert "alice" in entity.id.lower()

    def test_create_entity_duplicate_slug_raises(self, temp_db):
        """Test that creating entity with duplicate slug raises error."""
        create_entity(temp_db, "Alice", EntityType.person, slug="alice")
        
        with pytest.raises(ValueError, match="already exists"):
            create_entity(temp_db, "Alice 2", EntityType.person, slug="alice")


class TestSelfEntity:
    """Test self entity management."""

    def test_create_self_entity(self, temp_db):
        """Test creating the self entity."""
        entity = create_self_entity(temp_db, label="Self")
        
        assert entity.id == "self"
        assert entity.type == EntityType.self_
        assert entity.label == "Self"

    def test_create_self_entity_only_once(self, temp_db):
        """Test that self entity can only be created once."""
        create_self_entity(temp_db)
        
        with pytest.raises(RuntimeError, match="already exists"):
            create_self_entity(temp_db)

    def test_get_self_entity(self, temp_db):
        """Test retrieving the self entity."""
        create_self_entity(temp_db)
        entity = get_self_entity(temp_db)
        
        assert entity is not None
        assert entity.id == "self"

    def test_get_self_entity_not_created(self, temp_db):
        """Test getting self entity when not created returns None."""
        entity = get_self_entity(temp_db)
        assert entity is None

    def test_ensure_self_entity_creates(self, temp_db):
        """Test ensure_self_entity creates if not exists."""
        entity = ensure_self_entity(temp_db)
        
        assert entity is not None
        assert entity.id == "self"

    def test_ensure_self_entity_idempotent(self, temp_db):
        """Test ensure_self_entity is idempotent."""
        entity1 = ensure_self_entity(temp_db)
        entity2 = ensure_self_entity(temp_db)
        
        assert entity1.id == entity2.id


class TestProvisionalLifecycle:
    """Test provisional entity lifecycle."""

    def test_create_provisional(self, temp_db):
        """Test creating a provisional entity."""
        entity = create_provisional(
            temp_db,
            platform="telegram",
            external_id="user123",
            label="Unknown User",
        )
        
        assert entity.type == EntityType.provisional
        assert entity.card["platform"] == "telegram"
        assert entity.card["external_id"] == "user123"
        
        # Verify identity link was created
        links = get_identity_links(temp_db, entity.id)
        assert len(links) == 1
        assert links[0].platform == "telegram"
        assert links[0].external_id == "user123"

    def test_create_provisional_auto_label(self, temp_db):
        """Test provisional entity with auto-generated label."""
        entity = create_provisional(temp_db, "discord", "user456")
        
        assert "discord" in entity.label
        assert "user456" in entity.label

    def test_list_provisionals(self, temp_db):
        """Test listing provisional entities."""
        create_provisional(temp_db, "telegram", "user1")
        create_provisional(temp_db, "discord", "user2")
        
        provisionals = list_provisionals(temp_db)
        assert len(provisionals) == 2
        assert all(e.type == EntityType.provisional for e in provisionals)


class TestResolveOrProvision:
    """Test resolve_or_provision function."""

    def test_resolve_or_provision_new(self, temp_db):
        """Test resolve_or_provision creates new provisional."""
        entity_id, is_new = resolve_or_provision(
            temp_db,
            platform="telegram",
            external_id="newuser",
        )
        
        assert is_new is True
        assert entity_id is not None
        
        # Verify entity was created
        entity = temp_db.get_entity(entity_id)
        assert entity is not None
        assert entity.type == EntityType.provisional

    def test_resolve_or_provision_existing(self, temp_db):
        """Test resolve_or_provision returns existing entity."""
        # Create first time
        entity_id1, is_new1 = resolve_or_provision(
            temp_db, "telegram", "user123"
        )
        assert is_new1 is True
        
        # Resolve second time
        entity_id2, is_new2 = resolve_or_provision(
            temp_db, "telegram", "user123"
        )
        assert is_new2 is False
        assert entity_id2 == entity_id1


class TestMergeEntity:
    """Test entity merge operations."""

    def test_merge_entity(self, temp_db):
        """Test merging source entity into target."""
        source = create_entity(temp_db, "Source", EntityType.person)
        target = create_entity(temp_db, "Target", EntityType.person)
        
        # Add identity link to source
        link_identity_explicit(temp_db, "telegram", "user1", source.id)
        
        # Merge
        result = merge_entity(temp_db, source.id, target.id)
        
        assert result.id == target.id
        
        # Verify source is marked as merged
        source_updated = temp_db.get_entity(source.id)
        assert source_updated.merged_into == target.id
        
        # Verify identity link was transferred
        links = get_identity_links(temp_db, target.id)
        assert len(links) == 1
        assert links[0].external_id == "user1"

    def test_merge_entity_self_raises(self, temp_db):
        """Test that merging entity into itself raises error."""
        entity = create_entity(temp_db, "Test", EntityType.person)
        
        with pytest.raises(ValueError, match="Cannot merge"):
            merge_entity(temp_db, entity.id, entity.id)

    def test_merge_entity_not_found_raises(self, temp_db):
        """Test that merging non-existent entity raises error."""
        target = create_entity(temp_db, "Target", EntityType.person)
        
        with pytest.raises(ValueError, match="not found"):
            merge_entity(temp_db, "nonexistent", target.id)


class TestIdentityLinking:
    """Test identity linking functions."""

    def test_link_identity_explicit(self, temp_db):
        """Test explicit identity linking."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        
        link_identity_explicit(temp_db, "telegram", "user123", entity.id)
        
        links = get_identity_links(temp_db, entity.id)
        assert len(links) == 1
        assert links[0].platform == "telegram"
        assert links[0].external_id == "user123"

    def test_link_identity_explicit_not_found_raises(self, temp_db):
        """Test linking to non-existent entity raises error."""
        with pytest.raises(ValueError, match="not found"):
            link_identity_explicit(temp_db, "telegram", "user123", "nonexistent")


class TestEntityCard:
    """Test entity card retrieval."""

    def test_get_entity_card(self, temp_db):
        """Test getting full entity card."""
        entity = create_entity(
            temp_db,
            "Alice",
            EntityType.person,
            card={"role": "developer"},
        )
        link_identity_explicit(temp_db, "telegram", "user123", entity.id)
        
        card = get_entity_card(temp_db, entity.id)
        
        assert card is not None
        assert "entity" in card
        assert "identity_links" in card
        assert card["entity"]["label"] == "Alice"
        assert len(card["identity_links"]) == 1

    def test_get_entity_card_not_found(self, temp_db):
        """Test getting card for non-existent entity returns None."""
        card = get_entity_card(temp_db, "nonexistent")
        assert card is None


class TestUpdateEntity:
    """Test entity update operations."""

    def test_update_entity_label(self, temp_db):
        """Test updating entity label."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        
        updated = update_entity(temp_db, entity.id, label="Alice Smith")
        
        assert updated.label == "Alice Smith"

    def test_update_entity_card(self, temp_db):
        """Test updating entity card."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        
        updated = update_entity(
            temp_db,
            entity.id,
            card={"role": "senior developer"},
        )
        
        assert updated.card["role"] == "senior developer"

    def test_update_entity_status_line(self, temp_db):
        """Test updating entity status line."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        
        updated = update_entity(
            temp_db,
            entity.id,
            status_line="On vacation",
        )
        
        assert updated.status_line == "On vacation"
