"""Pydantic models mirroring §4 of the spec exactly."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class Network(str, Enum):
    world = "world"
    experience = "experience"
    opinion = "opinion"
    observation = "observation"


class MemoryStatus(str, Enum):
    active = "active"
    superseded = "superseded"
    contested = "contested"
    demoted = "demoted"
    promoted = "promoted"  # legacy; promotion now sets promoted_to without a status flip
    archived = "archived"  # gm_forget — explicit user intent only (D13: never hard-deleted)


class EntityType(str, Enum):
    self_ = "self"
    person = "person"
    project = "project"
    provisional = "provisional"


class LinkMethod(str, Enum):
    explicit = "explicit"
    provisional = "provisional"


class EdgeKind(str, Enum):
    shared_entity = "shared_entity"
    temporal = "temporal"
    derived_from = "derived_from"
    supersedes = "supersedes"
    contests = "contests"
    caused_by = "caused_by"


# ── Core tables ──────────────────────────────────────────────────────────

class MemoryRecord(BaseModel):
    """A single memory — fact, event, opinion, or observation."""
    id: str  # ULID
    text: str
    network: Network
    entity_ids: list[str] = Field(default_factory=list)
    source_memory_ids: list[str] = Field(default_factory=list)
    status: MemoryStatus = MemoryStatus.active
    superseded_by: Optional[str] = None
    contested_with: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_recalled_at: Optional[datetime] = None
    recall_count: int = 0
    # How many times this memory was a recall candidate but NOT returned in the
    # final top-k the caller used. Feeds the usefulness policy (promote/demote).
    recall_miss_count: int = 0
    reflect_cycles: int = 0  # Number of reflection cycles this memory has been through
    source_session_id: Optional[str] = None
    source_platform: Optional[str] = None
    speaker_entity_id: Optional[str] = None
    promoted_to: Optional[str] = None  # URL or path of promoted wiki page
    flagged_source: Optional[str] = None  # which Pass-1 rule matched
    # Canonized fact representation: (subject|predicate|object) triple for
    # structured extraction. Format: "subject|predicate|object" in lowercase,
    # e.g. "umar|prefers|concise answers". Used for cross-session consistency:
    # the same fact extracted differently ("Umar likes X" vs "Umar prefers X")
    # maps to one canonical_key and gets merged instead of duplicated.
    canonical_key: Optional[str] = None
    # Provenance strength for derived records (opinions): how many distinct
    # sources currently support this belief. Incremented on reflect merges.
    proof_count: int = 0
    # Verbatim supporting text for consolidated records (opinions). Each entry
    # is an exact quote from a source memory, so a belief is directly checkable
    # against its sources (hindsight observation parity).
    evidence_quotes: list[str] = Field(default_factory=list)
    # Change history as a JSON-encoded list of {at, action, sources} entries.
    history_json: Optional[str] = None
    # When the event described actually happened (if knowable), as opposed to
    # created_at (when we learned it). "Alice got married in June 2024" was
    # learned in 2025 but occurred in 2024. Temporal queries rank by
    # occurred_at when present, falling back to created_at.
    occurred_at: Optional[datetime] = None


class SessionSummary(BaseModel):
    """Rolling compressed summary of a conversation session.

    One per session. Updated as messages accumulate — keeps the last
    N_MEMORY_TOKENS of compressed context. Stored as a regular SQLite table
    so it participates in the same vector/DB fabric as memories (reuses
    creating, searching, and versioning).
    """
    id: str  # session_id
    text: str  # compressed summary
    message_count: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EntityRecord(BaseModel):
    """An entity — self, person, project, or provisional."""
    id: str  # slug
    type: EntityType
    label: str
    card: dict[str, Any] = Field(default_factory=dict)
    status_line: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    merged_into: Optional[str] = None


class IdentityLink(BaseModel):
    """Linking a (platform, external_id) to a canonical entity."""
    platform: str
    external_id: str
    entity_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: LinkMethod


class EdgeRecord(BaseModel):
    """Weighted relationship between two memories."""
    from_id: str
    to_id: str
    kind: EdgeKind
    weight: float = 1.0


class HotCache(BaseModel):
    """Materialized per-entity working memory for injection."""
    entity_id: str
    memory_ids: list[str] = Field(default_factory=list)
    rendered: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FlagRecord(BaseModel):
    """A Pass-1 flagged turn awaiting extraction."""
    id: str  # ULID
    session_id: str
    platform: str
    speaker_external_id: str
    turn_text: str
    flag_reason: str
    processed: bool = False
    attempt_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PromotionQueueRecord(BaseModel):
    """A memory queued for promotion to external knowledge bases."""
    memory_id: str
    enqueued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_systems: list[str] = Field(default_factory=list)  # e.g. ["wiki", "obsidian"]


class ReflectionRecord(BaseModel):
    """A record of a reflection action taken during autonomous reflection."""
    id: str  # ULID
    action: str  # supersede, contest, demote, form_opinion
    memory_ids: list[str] = Field(default_factory=list)
    new_memory_id: Optional[str] = None
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
