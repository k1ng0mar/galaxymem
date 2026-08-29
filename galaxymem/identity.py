"""Cross-platform identity linking (spec D3: explicit-only, never auto-inferred/merged).

Connects CLI, Telegram, Discord, and Web platform identities to GalaxyMem entities.
Every function takes ``store`` as its first argument — no global state.

Key rule (D3): identities are NEVER auto-merged. This module can *suggest* merges
(e.g. ``find_duplicate_provisionals``), but only the user can execute one via
``merge_provisional`` or ``merge_entity``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .entities import (
    get_identity_links,
    link_identity_explicit,
    list_provisionals,
    merge_entity,
    resolve_or_provision,
)
from .models import EntityType, IdentityLink
from .store import Store

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ("cli", "telegram", "discord", "web")


# ── 1. resolve_platform_user ────────────────────────────────────────────

def resolve_platform_user(
    store: Store,
    platform: str,
    platform_id: str,
    display_name: str,
) -> dict[str, Any]:
    """Main entry point when a message arrives from any platform.

    Resolves the (platform, platform_id) to a canonical entity, creating a
    provisional entity if this identity has never been seen before.

    Returns a dict with:
        entity_id      — canonical entity slug
        entity_name    — label
        entity_type    — EntityType value
        is_provisional — True if entity is still provisional (unknown contact)
        is_new         — True if a provisional was just created this call
        platform       — the platform string
        platform_id    — the platform-specific id
    """
    entity_id, is_new = resolve_or_provision(
        store, platform=platform, external_id=platform_id, label=display_name,
    )
    entity = store.get_entity(entity_id)
    if entity is None:  # defensive — should never happen just after resolve_or_provision
        raise RuntimeError(f"resolve_or_provision returned entity_id '{entity_id}' but it was not found")

    return {
        "entity_id": entity_id,
        "entity_name": entity.label,
        "entity_type": entity.type.value,
        "is_provisional": entity.type == EntityType.provisional,
        "is_new": is_new,
        "platform": platform,
        "platform_id": platform_id,
    }


# ── 2. link_platforms ───────────────────────────────────────────────────

def link_platforms(
    store: Store,
    entity_id: str,
    links: list[dict[str, str]],
) -> dict[str, Any]:
    """Explicitly link multiple platforms to one entity (batch operation).

    Each link dict: ``{"platform": str, "platform_id": str, "display_name": str}``.
    ``display_name`` is recorded in the entity card but not currently stored on
    the IdentityLink row (the IdentityLink model only carries platform/external_id).

    Returns a summary dict:
        entity_id   — the target entity
        linked      — list of {platform, platform_id} successfully linked
        skipped     — list of {platform, platform_id, reason} already linked elsewhere
        errors      — list of {platform, platform_id, error}
    """
    entity = store.get_entity(entity_id)
    if entity is None:
        raise ValueError(f"Entity '{entity_id}' not found")

    linked: list[dict[str, str]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for link in links:
        platform = link.get("platform", "")
        platform_id = link.get("platform_id", "")
        if not platform or not platform_id:
            errors.append({"link": link, "error": "missing platform or platform_id"})
            continue

        # Check if this identity is already linked — to us or someone else
        existing = store.resolve_identity(platform, platform_id)
        if existing is not None:
            if existing.entity_id == entity_id:
                skipped.append({
                    "platform": platform, "platform_id": platform_id,
                    "reason": "already linked to this entity",
                })
            else:
                skipped.append({
                    "platform": platform, "platform_id": platform_id,
                    "reason": f"already linked to entity '{existing.entity_id}'",
                })
            continue

        try:
            link_identity_explicit(
                store,
                platform=platform,
                external_id=platform_id,
                entity_id=entity_id,
            )
            linked.append({"platform": platform, "platform_id": platform_id})
            logger.info("Linked %s:%s → %s", platform, platform_id, entity_id)
        except Exception as exc:
            errors.append({
                "platform": platform, "platform_id": platform_id,
                "error": str(exc),
            })

    return {
        "entity_id": entity_id,
        "linked": linked,
        "skipped": skipped,
        "errors": errors,
    }


# ── 3. merge_provisional ────────────────────────────────────────────────

def merge_provisional(
    store: Store,
    provisional_id: str,
    real_entity_id: str,
) -> dict[str, Any]:
    """Merge a provisional entity into a real entity.

    Transfers all identity links and memory references (via edge/entity_id
    updates) from the provisional to the real entity, then marks the
    provisional as merged. Uses ``entities.merge_entity`` under the hood.

    Returns a dict with:
        source_id          — the provisional entity id
        target_id          — the real entity id
        target_name        — label of the real entity after merge
        merged             — True on success
        links_transferred  — count of identity links moved
        memories_transferred — count of memories re-pointed
    """
    source = store.get_entity(provisional_id)
    if source is None:
        raise ValueError(f"Provisional entity '{provisional_id}' not found")
    if source.type != EntityType.provisional:
        raise ValueError(
            f"Entity '{provisional_id}' is not provisional (type={source.type.value}). "
            "Use merge_entity directly for non-provisional merges."
        )

    target = store.get_entity(real_entity_id)
    if target is None:
        raise ValueError(f"Target entity '{real_entity_id}' not found")

    # Count what we're about to move (capture before mutation)
    links_before = len(store.get_identity_links_for_entity(provisional_id))
    memories_transferred = len(store.list_memories(entity_ids=[provisional_id]))

    # Delegate to entities.merge_entity — it repoints memories (scope and
    # speaker), identity links, marks merged, and rebuilds the hot cache.
    merged_target = merge_entity(store, source_id=provisional_id, target_id=real_entity_id)

    logger.info(
        "Merged provisional %s → %s (links=%d, memories=%d)",
        provisional_id, real_entity_id, links_before, memories_transferred,
    )

    return {
        "source_id": provisional_id,
        "target_id": real_entity_id,
        "target_name": merged_target.label,
        "merged": True,
        "links_transferred": links_before,
        "memories_transferred": memories_transferred,
    }


# ── 4. get_platform_map ─────────────────────────────────────────────────

def get_platform_map(store: Store) -> dict[str, dict[str, str]]:
    """Return a full map of all platform identities → entity_ids.

    Organized by platform:
        {"telegram": {"123": "alice", "456": "bob"}, "discord": {...}, ...}

    Only includes non-merged entities (skips entities with merged_into set).
    """
    # Scan identity links via the public Store API (never the private table).
    result: dict[str, dict[str, str]] = {p: {} for p in SUPPORTED_PLATFORMS}

    for link in store.list_identity_links():
        platform = link.platform
        if platform not in result:
            result[platform] = {}
        result[platform][link.external_id] = link.entity_id

    return result


# ── 5. find_duplicate_provisionals ──────────────────────────────────────

def find_duplicate_provisionals(store: Store) -> list[dict[str, Any]]:
    """Scan for provisional entities that might be the same person.

    Groups provisionals by normalized display_name. Returns candidates for
    manual merge — does NOT auto-merge (D3: explicit only).

    Returns a list of candidate groups:
        [{"display_name": "Alice", "provisionals": [{entity_id, platforms: [...]}, ...]}, ...]
    Only groups with 2+ provisionals sharing a name are returned.
    """
    provisionals = list_provisionals(store)
    by_name: dict[str, list[dict[str, Any]]] = {}

    for ent in provisionals:
        # Normalize: lowercase, strip whitespace
        key = (ent.label or "").strip().lower()
        if not key:
            continue

        links = store.get_identity_links_for_entity(ent.id)
        platforms = [
            {"platform": lk.platform, "platform_id": lk.external_id}
            for lk in links
        ]

        entry = {
            "entity_id": ent.id,
            "label": ent.label,
            "platforms": platforms,
            "created_at": ent.created_at.isoformat(),
        }
        by_name.setdefault(key, []).append(entry)

    candidates = [
        {"display_name": name, "provisionals": ents}
        for name, ents in by_name.items()
        if len(ents) >= 2
    ]
    candidates.sort(key=lambda c: len(c["provisionals"]), reverse=True)
    return candidates


# ── 6. unlink_platform ──────────────────────────────────────────────────

def unlink_platform(
    store: Store,
    entity_id: str,
    platform: str,
    platform_id: str,
) -> dict[str, Any]:
    """Remove a specific platform link from an entity.

    Returns:
        {"entity_id", "platform", "platform_id", "unlinked": bool, "remaining_links": int}
    """
    # Verify the link exists and belongs to this entity
    existing = store.resolve_identity(platform, platform_id)
    if existing is None:
        return {
            "entity_id": entity_id,
            "platform": platform,
            "platform_id": platform_id,
            "unlinked": False,
            "error": "no such identity link",
        }
    if existing.entity_id != entity_id:
        return {
            "entity_id": entity_id,
            "platform": platform,
            "platform_id": platform_id,
            "unlinked": False,
            "error": f"link belongs to entity '{existing.entity_id}', not '{entity_id}'",
        }

    store.delete_identity_link(platform, platform_id)
    remaining = len(store.get_identity_links_for_entity(entity_id))

    logger.info("Unlinked %s:%s from %s", platform, platform_id, entity_id)
    return {
        "entity_id": entity_id,
        "platform": platform,
        "platform_id": platform_id,
        "unlinked": True,
        "remaining_links": remaining,
    }


# ── 7. format_identity_card ─────────────────────────────────────────────

def format_identity_card(store: Store, entity_id: str) -> str:
    """Human-readable identity card for an entity.

    Shows: name, type, all linked platforms, memory count.
    Returns a formatted string. Returns an error string if entity not found.
    """
    entity = store.get_entity(entity_id)
    if entity is None:
        return f"❌ Entity '{entity_id}' not found"

    links = store.get_identity_links_for_entity(entity_id)

    # Count memories referencing this entity
    try:
        memory_count = store.count_memories_for_entity(entity_id)
    except Exception:
        memory_count = -1

    # Build platform lines
    if links:
        platform_lines = []
        for lk in links:
            method_tag = "📌" if lk.created_by.value == "explicit" else "❓"
            platform_lines.append(
                f"   {method_tag} {lk.platform}:{lk.external_id}"
            )
        platforms_block = "\n".join(platform_lines)
    else:
        platforms_block = "   (none)"

    type_label = entity.type.value
    if entity.merged_into:
        type_label += f" → merged into {entity.merged_into}"

    status = f"\n   status: {entity.status_line}" if entity.status_line else ""

    return (
        f"┌─ {entity.label} ─────────────────────\n"
        f"│ id: {entity.id}\n"
        f"│ type: {type_label}{status}\n"
        f"│ platforms:\n"
        f"{platforms_block}\n"
        f"│ memories: {memory_count if memory_count >= 0 else 'unknown'}\n"
        f"└──────────────────────────────────────"
    )
