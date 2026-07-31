"""LanceDB table schemas for GalaxyMem.

Pure DDL: the seven Pydantic/LanceModel table definitions plus the SQL
literal-escaper `_esc`. No business logic, no dependency on the `Store` class,
so this module is safe to import anywhere (tests, tooling, migrations).
"""

from __future__ import annotations

from typing import Optional

from lancedb.pydantic import LanceModel, Vector

from . import config as cfg


class MemoriesTable(LanceModel):
    """The LanceDB model for the memories table."""
    id: str  # ULID
    text: str
    vector: Vector(cfg.EMBEDDING_DIM)
    network: str
    entity_ids: str  # JSON-encoded list
    source_memory_ids: str  # JSON-encoded list
    status: str
    superseded_by: Optional[str] = None
    contested_with: str = "[]"  # JSON-encoded list
    created_at: str  # ISO datetime string
    last_recalled_at: Optional[str] = None
    recall_count: int = 0
    reflect_cycles: int = 0
    source_session_id: Optional[str] = None
    source_platform: Optional[str] = None
    speaker_entity_id: Optional[str] = None
    promoted_to: Optional[str] = None
    flagged_source: Optional[str] = None
    canonical_key: Optional[str] = None  # subject|predicate|object for structured facts


class EntitiesTable(LanceModel):
    id: str  # slug
    type: str
    label: str
    card: str = "{}"  # JSON-encoded dict
    status_line: str = ""
    created_at: str  # ISO datetime
    merged_into: Optional[str] = None


class IdentityLinksTable(LanceModel):
    platform: str
    external_id: str
    entity_id: str
    created_at: str
    created_by: str


class EdgesTable(LanceModel):
    from_id: str
    to_id: str
    kind: str
    weight: float = 1.0


class HotCacheTable(LanceModel):
    entity_id: str
    memory_ids: str = "[]"
    rendered: str = ""
    updated_at: str


class FlagsTable(LanceModel):
    id: str
    session_id: str
    platform: str
    speaker_external_id: str
    turn_text: str
    flag_reason: str
    processed: bool = False
    created_at: str


class PromotionQueueTable(LanceModel):
    """Queue of memories pending promotion to external knowledge bases."""
    memory_id: str
    enqueued_at: str  # ISO datetime
    target_systems: str = "[]"  # JSON-encoded list


class SessionSummariesTable(LanceModel):
    """Rolling compressed summary per session — metadata-heavy, no vector."""
    id: str  # session_id
    text: str  # compressed summary (max ~1500 chars)
    message_count: int = 0
    last_updated: str  # ISO datetime


# ── SQL litera escaping ────────────────────────────────────────────────────

_LIKE_SPECIALS = {"%": "\\%", "_": "\\_", "[": "\\[", "]": "\\]"}


def _esc(val: str, *, escape_like: bool = False) -> str:
    """Escape a string for embedding in a LanceDB/SQL filter literal.

    Escapes:
    - backslashes
    - double quotes (primary delimiter for string literals)
    - single quotes
    - newlines
    - null bytes

    When escape_like=True, also escapes LIKE wildcards (%) and (_).
    """
    s = str(val)
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("'", "\\'")
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\x00", "")
    if escape_like:
        for old, new in _LIKE_SPECIALS.items():
            s = s.replace(old, new)
    return s
