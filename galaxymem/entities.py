"""Entity management — CRUD, provisional lifecycle, identity resolution.

Handles:
- Entity creation (self, person, project, provisional)
- Provisional entity lifecycle (auto-created for unknown contacts)
- Explicit merge of provisional → real entity
- Identity resolution: resolve_or_provision(platform, external_id) → entity_id
- Slug generation and label normalization
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional

from .models import EntityRecord, EntityType, IdentityLink, LinkMethod, MemoryStatus
from .store import Store

logger = logging.getLogger(__name__)


# ── Slug generation ─────────────────────────────────────────────────────

def _slugify(text: str, max_len: int = 64) -> str:
    """Generate a URL-safe slug from a label.

    Normalizes unicode, lowercases, replaces non-alphanumerics with hyphens,
    collapses runs, strips leading/trailing hyphens, truncates.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:max_len].rstrip("-") or "entity"


def _unique_slug(store: Store, base: str) -> str:
    """Generate a unique slug, appending -2, -3, etc. if needed."""
    slug = _slugify(base)
    if store.get_entity(slug) is None:
        return slug
    for i in range(2, 10000):
        candidate = f"{slug}-{i}"
        if store.get_entity(candidate) is None:
            return candidate
    raise RuntimeError(f"Could not generate unique slug for '{base}' after 10000 attempts")


# ── Entity CRUD ─────────────────────────────────────────────────────────

def create_entity(
    store: Store,
    label: str,
    entity_type: EntityType,
    card: Optional[dict] = None,
    status_line: str = "",
    slug: Optional[str] = None,
) -> EntityRecord:
    """Create a new entity with an auto-generated or explicit slug.

    Args:
        store: The LanceDB store.
        label: Human-readable name.
        entity_type: self, person, project, or provisional.
        card: Optional metadata dict (free-form).
        status_line: Short status summary.
        slug: Explicit slug; auto-generated if None.

    Returns:
        The created EntityRecord.

    Raises:
        ValueError: If slug already exists.
    """
    if slug is None:
        slug = _unique_slug(store, label)
    elif store.get_entity(slug) is not None:
        raise ValueError(f"Entity with slug '{slug}' already exists")

    entity = EntityRecord(
        id=slug,
        type=entity_type,
        label=label,
        card=card or {},
        status_line=status_line,
    )
    store.add_entity(entity)
    logger.info("Created entity: %s (%s)", slug, entity_type.value)
    return entity


def create_self_entity(store: Store, label: str = "Self") -> EntityRecord:
    """Create the self entity. Only one allowed — raises if one exists."""
    if store.has_self_entity():
        raise RuntimeError("Self entity already exists. Use get_self_entity().")
    return create_entity(store, label=label, entity_type=EntityType.self_, slug="self")


def get_self_entity(store: Store) -> Optional[EntityRecord]:
    """Get the self entity, or None if not created yet."""
    return store.get_entity("self")


def ensure_self_entity(store: Store, label: str = "Self") -> EntityRecord:
    """Get or create the self entity."""
    existing = get_self_entity(store)
    if existing is not None:
        return existing
    return create_self_entity(store, label=label)


def update_entity(
    store: Store,
    entity_id: str,
    label: Optional[str] = None,
    card: Optional[dict] = None,
    status_line: Optional[str] = None,
) -> EntityRecord:
    """Update entity fields. Returns the updated entity."""
    kwargs = {}
    if label is not None:
        kwargs["label"] = label
    if card is not None:
        kwargs["card"] = card  # store.update_entity handles JSON serialization
    if status_line is not None:
        kwargs["status_line"] = status_line
    if kwargs:
        store.update_entity(entity_id, **kwargs)
    updated = store.get_entity(entity_id)
    if updated is None:
        raise ValueError(f"Entity '{entity_id}' not found after update")
    return updated


# ── Provisional lifecycle ───────────────────────────────────────────────

def create_provisional(
    store: Store,
    platform: str,
    external_id: str,
    label: Optional[str] = None,
) -> EntityRecord:
    """Create a provisional entity for an unknown contact.

    Provisional entities are auto-created when a new (platform, external_id)
    is encountered that doesn't match any existing identity link.

    Args:
        store: The LanceDB store.
        platform: Platform name (e.g. "telegram", "discord", "cli").
        external_id: Platform-specific user ID.
        label: Optional display label; defaults to "{platform}:{external_id}".

    Returns:
        The created provisional EntityRecord.
    """
    if label is None:
        label = f"{platform}:{external_id}"

    entity = create_entity(
        store,
        label=label,
        entity_type=EntityType.provisional,
        card={"platform": platform, "external_id": external_id},
    )

    # Create the identity link
    link = IdentityLink(
        platform=platform,
        external_id=external_id,
        entity_id=entity.id,
        created_by=LinkMethod.provisional,
    )
    store.add_identity_link(link)
    logger.info("Created provisional entity: %s for %s:%s", entity.id, platform, external_id)
    return entity


def merge_entity(
    store: Store,
    source_id: str,
    target_id: str,
) -> EntityRecord:
    """Merge source entity into target entity (spec Phase 2 step 3).

    Re-points all memories (entity scope AND speaker attribution) and all
    identity links from source to target, marks source as merged (never
    deleted, D13), and rebuilds the target's hot cache.

    This is an explicit user action — never auto-inferred.

    Args:
        store: The LanceDB store.
        source_id: The entity to merge FROM (will be marked merged_into).
        target_id: The entity to merge INTO (receives everything).

    Returns:
        The target EntityRecord after merge.

    Raises:
        ValueError: If source or target not found, or source == target.
    """
    if source_id == target_id:
        raise ValueError("Cannot merge an entity into itself")

    source = store.get_entity(source_id)
    target = store.get_entity(target_id)
    if source is None:
        raise ValueError(f"Source entity '{source_id}' not found")
    if target is None:
        raise ValueError(f"Target entity '{target_id}' not found")

    # Repoint all memories scoped to (or spoken by) the source
    moved = store.re_memory_entity_ids(source_id, target_id)
    store.re_memory_speaker(source_id, target_id)

    # Repoint all identity links from source → target
    store.repoint_identity_links(source_id, target_id)

    # Mark source as merged — audit trail survives
    store.update_entity(source_id, merged_into=target_id)

    # The target's working memory changed — rebuild its hot cache
    try:
        from .recall import update_hot_cache
        update_hot_cache(store, entity_id=target_id)
    except Exception as e:
        logger.warning("Hot cache rebuild after merge failed: %s", e)

    logger.info("Merged entity %s → %s (%d memories moved)", source_id, target_id, moved)
    return target


def resolve_or_provision(
    store: Store,
    platform: str,
    external_id: str,
    label: Optional[str] = None,
) -> tuple[str, bool]:
    """Resolve a (platform, external_id) to an entity, or create a provisional.

    This is the main entry point for identity resolution during conversation.
    If the identity already has a link, returns the linked entity_id.
    Otherwise, creates a new provisional entity and links it.

    Args:
        store: The LanceDB store.
        platform: Platform name (e.g. "telegram", "discord", "cli").
        external_id: Platform-specific user ID.
        label: Optional display label for new provisional entity.

    Returns:
        (entity_id, is_new) tuple. is_new=True if a provisional was just created.
    """
    # Try to resolve existing identity
    existing_link = store.resolve_identity(platform, external_id)
    if existing_link is not None:
        return existing_link.entity_id, False

    # No existing link — create provisional
    entity = create_provisional(store, platform, external_id, label=label)
    return entity.id, True


# ── Identity linking ────────────────────────────────────────────────────

def link_identity_explicit(
    store: Store,
    platform: str,
    external_id: str,
    entity_id: str,
) -> None:
    """Explicitly link a (platform, external_id) to an existing entity.

    Used for user-initiated identity linking (e.g. "this Telegram user is Alice").
    Never auto-inferred.

    Args:
        store: The LanceDB store.
        platform: Platform name.
        external_id: Platform-specific user ID.
        entity_id: The entity to link to.

    Raises:
        ValueError: If entity not found.
    """
    entity = store.get_entity(entity_id)
    if entity is None:
        raise ValueError(f"Entity '{entity_id}' not found")

    link = IdentityLink(
        platform=platform,
        external_id=external_id,
        entity_id=entity_id,
        created_by=LinkMethod.explicit,
    )
    store.add_identity_link(link)
    logger.info("Explicitly linked %s:%s → %s", platform, external_id, entity_id)


def get_identity_links(store: Store, entity_id: str) -> list[IdentityLink]:
    """Get all identity links for an entity."""
    return store.get_identity_links_for_entity(entity_id)


# ── Listing / queries ───────────────────────────────────────────────────

def list_entities(
    store: Store,
    entity_type: Optional[EntityType] = None,
) -> list[EntityRecord]:
    """List entities, optionally filtered by type."""
    all_entities = store.list_entities()
    if entity_type is not None:
        return [e for e in all_entities if e.type == entity_type]
    return all_entities


def list_provisionals(store: Store) -> list[EntityRecord]:
    """List all provisional entities (unknown contacts awaiting merge)."""
    all_entities = store.list_entities()
    return [e for e in all_entities if e.type == EntityType.provisional]


def get_entity_card(store: Store, entity_id: str) -> Optional[dict]:
    """Get the full entity card (entity + identity links + stats).

    Returns None if entity not found.
    """
    entity = store.get_entity(entity_id)
    if entity is None:
        return None

    links = store.get_identity_links_for_entity(entity_id)
    return {
        "entity": entity.model_dump(),
        "identity_links": [link.model_dump() for link in links],
    }


# ── Provisional cleanup ────────────────────────────────────────────────────

def _cleanup_stale_provisionals(store: Store, ttl_days: Optional[int] = None) -> int:
    """Archive provisional entities older than ttl_days with zero memories.

    Called during reflection housekeeping (5d). Prevents the entities table
    from filling with one-time messaging contacts that were never explicitly
    linked.

    Returns the count of provisionals cleaned up.
    """
    from . import config as cfg
    from .models import EntityType

    ttl = ttl_days if ttl_days is not None else cfg.PROVISIONAL_TTL_DAYS
    if ttl <= 0:
        return 0  # expiry disabled

    provisionals = list_provisionals(store)
    if not provisionals:
        return 0

    removed = 0
    for ent in provisionals:
        age_days = (datetime.now(timezone.utc) - ent.created_at).days
        if age_days < ttl:
            continue

        # Count memories associated with this entity
        mems = store.list_memories(entity_ids=[ent.id])
        active_mems = [m for m in mems if m.status == MemoryStatus.active]
        if active_mems:
            continue  # has active memories — keep

        # Check identity links — keep entities that have been explicitly linked
        links = store.get_identity_links_for_entity(ent.id)
        explicit_links = [l for l in links if l.created_by.value == "explicit"]
        if explicit_links:
            continue

        # Safe to archive: no active memories, no explicit links
        # We don't DELETE; we mark as merged into nothing (merged_into = "__archived__")
        # which removes them from list_provisionals but keeps the history.
        store.update_entity(ent.id, merged_into="__archived_provisional__")
        removed += 1
        logger.info("Auto-archived stale provisional (age %dd, 0 active memories): %s",
                    age_days, ent.id)

    return removed
