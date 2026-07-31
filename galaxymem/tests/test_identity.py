"""Unit tests for identity.py — cross-platform identity linking."""

import pytest
from galaxymem.identity import (
    resolve_platform_user,
    link_platforms,
    merge_provisional,
    get_platform_map,
    find_duplicate_provisionals,
    unlink_platform,
    format_identity_card,
)
from galaxymem.entities import create_entity, create_provisional
from galaxymem.models import EntityType


class TestResolvePlatformUser:
    """Test platform user resolution."""

    def test_resolve_new_user_provisional(self, temp_db):
        """Test resolving a new user creates provisional."""
        result = resolve_platform_user(
            temp_db,
            platform="telegram",
            platform_id="user123",
            display_name="Alice",
        )
        
        assert result["is_new"] is True
        assert result["is_provisional"] is True
        assert result["platform"] == "telegram"
        assert result["platform_id"] == "user123"
        assert result["entity_name"] == "Alice"

    def test_resolve_returning_user(self, temp_db):
        """Test resolving a returning user returns existing entity."""
        # First call creates
        result1 = resolve_platform_user(
            temp_db, "telegram", "user123", "Alice"
        )
        assert result1["is_new"] is True
        
        # Second call returns existing
        result2 = resolve_platform_user(
            temp_db, "telegram", "user123", "Alice"
        )
        assert result2["is_new"] is False
        assert result2["entity_id"] == result1["entity_id"]

    def test_resolve_different_platforms(self, temp_db):
        """Test resolving same display name on different platforms."""
        r1 = resolve_platform_user(temp_db, "telegram", "u1", "Alice")
        r2 = resolve_platform_user(temp_db, "discord", "u2", "Alice")
        
        assert r1["entity_id"] != r2["entity_id"]
        assert r1["is_new"] is True
        assert r2["is_new"] is True


class TestLinkPlatforms:
    """Test batch platform linking."""

    def test_link_platforms_batch(self, temp_db):
        """Test linking multiple platforms to one entity."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        
        result = link_platforms(temp_db, entity.id, [
            {"platform": "telegram", "platform_id": "tg123"},
            {"platform": "discord", "platform_id": "dc456"},
            {"platform": "web", "platform_id": "web789"},
        ])
        
        assert len(result["linked"]) == 3
        assert len(result["skipped"]) == 0
        assert len(result["errors"]) == 0

    def test_link_platforms_already_linked(self, temp_db):
        """Test linking an already-linked platform skips."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        
        # First link
        link_platforms(temp_db, entity.id, [
            {"platform": "telegram", "platform_id": "tg123"},
        ])
        
        # Try to link again
        result = link_platforms(temp_db, entity.id, [
            {"platform": "telegram", "platform_id": "tg123"},
        ])
        
        assert len(result["linked"]) == 0
        assert len(result["skipped"]) == 1
        assert "already linked" in result["skipped"][0]["reason"]

    def test_link_platforms_missing_fields(self, temp_db):
        """Test linking with missing platform fields returns error."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        
        result = link_platforms(temp_db, entity.id, [
            {"platform": "", "platform_id": "tg123"},
        ])
        
        assert len(result["errors"]) == 1

    def test_link_platforms_entity_not_found(self, temp_db):
        """Test linking to non-existent entity raises error."""
        with pytest.raises(ValueError, match="not found"):
            link_platforms(temp_db, "nonexistent", [
                {"platform": "telegram", "platform_id": "tg123"},
            ])


class TestMergeProvisional:
    """Test provisional entity merge."""

    def test_merge_provisional_into_real(self, temp_db):
        """Test merging a provisional entity into a real one."""
        # Create provisional
        provisional = create_provisional(temp_db, "telegram", "user123", "Unknown")
        
        # Create real entity
        real_entity = create_entity(temp_db, "Alice", EntityType.person)
        
        # Merge
        result = merge_provisional(temp_db, provisional.id, real_entity.id)
        
        assert result["merged"] is True
        assert result["source_id"] == provisional.id
        assert result["target_id"] == real_entity.id

    def test_merge_provisional_not_provisional_raises(self, temp_db):
        """Test merging non-provisional entity raises error."""
        entity1 = create_entity(temp_db, "Alice", EntityType.person)
        entity2 = create_entity(temp_db, "Bob", EntityType.person)
        
        with pytest.raises(ValueError, match="not provisional"):
            merge_provisional(temp_db, entity1.id, entity2.id)

    def test_merge_provisional_not_found(self, temp_db):
        """Test merging non-existent provisional raises error."""
        target = create_entity(temp_db, "Alice", EntityType.person)
        
        with pytest.raises(ValueError, match="not found"):
            merge_provisional(temp_db, "nonexistent", target.id)


class TestGetPlatformMap:
    """Test platform map generation."""

    def test_get_platform_map_empty(self, temp_db):
        """Test platform map with no identities."""
        result = get_platform_map(temp_db)
        
        assert isinstance(result, dict)
        assert "telegram" in result
        assert "discord" in result
        assert "cli" in result
        assert "web" in result

    def test_get_platform_map_with_links(self, temp_db):
        """Test platform map with identity links."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        from galaxymem.entities import link_identity_explicit
        link_identity_explicit(temp_db, "telegram", "tg123", entity.id)
        link_identity_explicit(temp_db, "discord", "dc456", entity.id)
        
        result = get_platform_map(temp_db)
        
        assert result["telegram"]["tg123"] == entity.id
        assert result["discord"]["dc456"] == entity.id


class TestFindDuplicateProvisionals:
    """Test duplicate provisional detection (no auto-merge)."""

    def test_find_duplicates_none(self, temp_db):
        """Test no duplicates found with unique provisionals."""
        create_provisional(temp_db, "telegram", "u1", "Alice")
        create_provisional(temp_db, "discord", "u2", "Bob")
        
        duplicates = find_duplicate_provisionals(temp_db)
        
        assert len(duplicates) == 0

    def test_find_duplicates_same_name(self, temp_db):
        """Test finding duplicate provisionals with same name."""
        create_provisional(temp_db, "telegram", "u1", "Alice")
        create_provisional(temp_db, "discord", "u2", "Alice")
        
        duplicates = find_duplicate_provisionals(temp_db)
        
        assert len(duplicates) == 1
        assert duplicates[0]["display_name"] == "alice"
        assert len(duplicates[0]["provisionals"]) == 2

    def test_no_auto_merge(self, temp_db):
        """Test that find_duplicates does NOT auto-merge (D3 rule)."""
        p1 = create_provisional(temp_db, "telegram", "u1", "Alice")
        p2 = create_provisional(temp_db, "discord", "u2", "Alice")
        
        find_duplicate_provisionals(temp_db)
        
        # Both entities should still exist independently
        assert temp_db.get_entity(p1.id) is not None
        assert temp_db.get_entity(p2.id) is not None
        assert temp_db.get_entity(p1.id).merged_into is None


class TestUnlinkPlatform:
    """Test platform unlinking."""

    def test_unlink_platform_success(self, temp_db):
        """Test successfully unlinking a platform."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        from galaxymem.entities import link_identity_explicit
        link_identity_explicit(temp_db, "telegram", "tg123", entity.id)
        
        result = unlink_platform(temp_db, entity.id, "telegram", "tg123")
        
        assert result["unlinked"] is True
        assert result["remaining_links"] == 0

    def test_unlink_platform_not_found(self, temp_db):
        """Test unlinking non-existent link returns error."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        
        result = unlink_platform(temp_db, entity.id, "telegram", "nonexistent")
        
        assert result["unlinked"] is False
        assert "error" in result

    def test_unlink_platform_wrong_entity(self, temp_db):
        """Test unlinking a link that belongs to another entity."""
        entity1 = create_entity(temp_db, "Alice", EntityType.person)
        entity2 = create_entity(temp_db, "Bob", EntityType.person)
        from galaxymem.entities import link_identity_explicit
        link_identity_explicit(temp_db, "telegram", "tg123", entity1.id)
        
        result = unlink_platform(temp_db, entity2.id, "telegram", "tg123")
        
        assert result["unlinked"] is False
        assert "belongs to entity" in result["error"]


class TestFormatIdentityCard:
    """Test identity card formatting."""

    def test_format_identity_card(self, temp_db):
        """Test formatting an identity card."""
        entity = create_entity(temp_db, "Alice", EntityType.person)
        from galaxymem.entities import link_identity_explicit
        link_identity_explicit(temp_db, "telegram", "tg123", entity.id)
        
        card = format_identity_card(temp_db, entity.id)
        
        assert "Alice" in card
        assert "telegram" in card
        assert "tg123" in card

    def test_format_identity_card_not_found(self, temp_db):
        """Test formatting card for non-existent entity."""
        card = format_identity_card(temp_db, "nonexistent")
        
        assert "not found" in card
