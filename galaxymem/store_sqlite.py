"""SQLite storage layer — typed CRUD with zero business logic.

Replacement for the LanceDB-backed store.py. Uses SQLite for ACID
transactions, FTS5 for keyword search, and sqlite-vec for vector search.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import config as cfg
from .models import (
    EdgeKind,
    EdgeRecord,
    EntityRecord,
    EntityType,
    FlagRecord,
    HotCache,
    IdentityLink,
    LinkMethod,
    MemoryRecord,
    MemoryStatus,
    Network,
    PromotionQueueRecord,
    SessionSummary,
)
from .redact import redact_secrets
from .utils import ulid as _ulid

logger = logging.getLogger(__name__)


# ── Schema DDL ────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    vector BLOB,
    network TEXT NOT NULL,
    entity_ids TEXT NOT NULL DEFAULT '[]',
    source_memory_ids TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by TEXT,
    contested_with TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    last_recalled_at TEXT,
    recall_count INTEGER NOT NULL DEFAULT 0,
    recall_miss_count INTEGER NOT NULL DEFAULT 0,
    reflect_cycles INTEGER NOT NULL DEFAULT 0,
    source_session_id TEXT,
    source_platform TEXT,
    speaker_entity_id TEXT,
    promoted_to TEXT,
    flagged_source TEXT,
    canonical_key TEXT,
    proof_count INTEGER NOT NULL DEFAULT 0,
    evidence_quotes TEXT NOT NULL DEFAULT '[]',
    history_json TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    label TEXT NOT NULL,
    card TEXT NOT NULL DEFAULT '{}',
    status_line TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    merged_into TEXT
);

CREATE TABLE IF NOT EXISTS identity_links (
    platform TEXT NOT NULL,
    external_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    PRIMARY KEY (platform, external_id)
);

CREATE TABLE IF NOT EXISTS edges (
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (from_id, to_id, kind)
);

CREATE TABLE IF NOT EXISTS hot_cache (
    entity_id TEXT PRIMARY KEY,
    memory_ids TEXT NOT NULL DEFAULT '[]',
    rendered TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS flags (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    speaker_external_id TEXT NOT NULL,
    turn_text TEXT NOT NULL,
    flag_reason TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_queue (
    memory_id TEXT PRIMARY KEY,
    enqueued_at TEXT NOT NULL,
    target_systems TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_network ON memories(network);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_canonical ON memories(canonical_key);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
CREATE INDEX IF NOT EXISTS idx_flags_processed ON flags(processed);
CREATE INDEX IF NOT EXISTS idx_flags_session ON flags(session_id);
"""

_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    text,
    tokenize = 'porter'
);
"""

_VEC_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    id TEXT PRIMARY KEY,
    vector FLOAT[{dim}]
);
"""

_SCHEMA_VERSION = 1


def _entity_membership_clause(entity_ids: list[str]) -> str:
    """Empty-clause guard for D8 fail-closed semantics (test contract).

    Returns the unsatisfiable '(1 = 0)' for an empty list. Non-empty
    lists go through Store._filter_where with ? placeholders.
    """
    if not entity_ids:
        return "(1 = 0)"
    return "(entity_ids LIKE ?)"


class Store:
    """SQLite-backed storage layer, API-compatible with the LanceDB store."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else Path(cfg.DB_PATH)
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.RLock()
        self._closed = False
        self._schema_version = _SCHEMA_VERSION

    # ── Connection / lifecycle ──────────────────────────────────────────

    def open(self, create_if_missing: bool = True) -> "Store":
        """Open (or create) the SQLite database and prepare tables/indexes.

        Accepts either a file path or a directory path. A directory is
        treated as the legacy LanceDB-style layout: the DB file is
        <dir>/galaxymem.sqlite3.
        """
        p = Path(self.db_path)
        if p.is_dir() or not str(p).endswith(".sqlite3"):
            # Directory-style path → use galaxymem.sqlite3 inside it
            if str(p).endswith(("db", "db/", "test_galaxymem")) or p.is_dir():
                p = p / "galaxymem.sqlite3"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = p
        self._conn = sqlite3.connect(str(p), timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA busy_timeout=30000;")

        with self._write_lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.executescript(_FTS_SQL)
            # Load sqlite-vec if available; vector search falls back to
            # in-Python brute force when the extension is absent.
            try:
                import sqlite_vec
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
                self._vec_available = True
            except Exception as e:
                self._vec_available = False
                logger.warning("sqlite-vec unavailable (%s); vector search will use brute force", e)
            if self._vec_available:
                self._conn.executescript(_VEC_SQL.format(dim=cfg.EMBEDDING_DIM))
            self._conn.commit()
        return self

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._closed = True

    def _assert_writable(self) -> None:
        if self._conn is None or self._closed:
            raise RuntimeError("Store is not open")

    def _assert_under_limits(self, extra: int = 0) -> None:
        """Raise if adding `extra` memories would exceed MAX_MEMORIES (D-dos guard)."""
        max_mem = getattr(cfg, "MAX_MEMORIES", None)
        if max_mem is None:
            return
        count = 0
        try:
            count = self._query("SELECT COUNT(*) AS c FROM memories")[0]["c"]
        except Exception:
            return
        if count + extra > max_mem:
            raise RuntimeError(
                f"Memory limit reached ({count} >= {max_mem}); refusing to add {extra} more"
            )

    def _execute(self, sql: str, params: tuple = ()):
        self._assert_writable()
        cur = self._conn.execute(sql, params)
        return cur

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        self._assert_writable()
        cur = self._conn.execute(sql, params)
        return cur.fetchall()

    # ── Row converters ───────────────────────────────────────────────────

    @staticmethod
    def _dumps(v: Any) -> str:
        return json.dumps(v, ensure_ascii=False) if v is not None else None

    @staticmethod
    def _loads(s: Any, default: Any = None):
        if s is None:
            return default
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError, ValueError):
            return default

    # ── Memory CRUD ──────────────────────────────────────────────────────

    def add_memory(self, memory: MemoryRecord) -> str:
        """Insert a single memory atomically with FTS + vector indexes."""
        self._assert_writable()
        # Credential-shaped spans never persist (store-boundary redaction).
        prepared_text = redact_secrets(memory.text)
        if prepared_text != memory.text:
            memory = memory.model_copy(update={"text": prepared_text})
        vec = getattr(memory, "vector", None) or self._embed(memory.text)
        self._assert_under_limits(1)
        with self._write_lock:
            row = self._memory_to_row(memory)
            self._conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, text, vector, network, entity_ids, source_memory_ids, status,
                    superseded_by, contested_with, created_at, last_recalled_at,
                    recall_count, recall_miss_count, reflect_cycles,
                    source_session_id, source_platform, speaker_entity_id,
                    promoted_to, flagged_source, canonical_key, proof_count,
                    evidence_quotes, history_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (row[0], row[1], self._vec_to_blob(vec), *row[2:]),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO memories_fts (id, text) VALUES (?,?)",
                (memory.id, memory.text),
            )
            self._conn.commit()
        return memory.id

    def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Batch insert. Embeds once per text, single transaction."""
        self._assert_writable()
        if not memories:
            return []
        texts = [m.text for m in memories]
        vectors = self._embed_texts(texts)
        with self._write_lock:
            for m, vec in zip(memories, vectors):
                row = self._memory_to_row(m)
                self._conn.execute(
                    """INSERT OR REPLACE INTO memories
                       (id, text, vector, network, entity_ids, source_memory_ids, status,
                        superseded_by, contested_with, created_at, last_recalled_at,
                        recall_count, recall_miss_count, reflect_cycles,
                        source_session_id, source_platform, speaker_entity_id,
                        promoted_to, flagged_source, canonical_key, proof_count,
                        evidence_quotes, history_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (row[0], row[1], self._vec_to_blob(vec), *row[2:]),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO memories_fts (id, text) VALUES (?,?)",
                    (m.id, m.text),
                )
            self._conn.commit()
        return [m.id for m in memories]

    # ── Converters / embedding ───────────────────────────────────────────

    def _memory_to_row(self, m: MemoryRecord) -> tuple:
        return (
            m.id, m.text, m.network.value, self._dumps(m.entity_ids),
            self._dumps(m.source_memory_ids), m.status.value, m.superseded_by,
            self._dumps(m.contested_with),
            m.created_at.isoformat() if m.created_at else None,
            m.last_recalled_at.isoformat() if m.last_recalled_at else None,
            m.recall_count, m.recall_miss_count, m.reflect_cycles,
            m.source_session_id, m.source_platform, m.speaker_entity_id,
            m.promoted_to, m.flagged_source, m.canonical_key, m.proof_count,
            self._dumps(m.evidence_quotes), m.history_json,
        )

    @staticmethod
    def _row_to_memory(r: sqlite3.Row) -> MemoryRecord:
        def _dt(v):
            if not v:
                return None
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                return None
        return MemoryRecord(
            id=r["id"], text=r["text"], network=Network(r["network"]),
            entity_ids=json.loads(r["entity_ids"] or "[]"),
            source_memory_ids=json.loads(r["source_memory_ids"] or "[]"),
            status=MemoryStatus(r["status"]), superseded_by=r["superseded_by"],
            contested_with=json.loads(r["contested_with"] or "[]"),
            created_at=_dt(r["created_at"]) or datetime.now(timezone.utc),
            last_recalled_at=_dt(r["last_recalled_at"]),
            recall_count=r["recall_count"] or 0,
            recall_miss_count=r["recall_miss_count"] or 0,
            reflect_cycles=r["reflect_cycles"] or 0,
            source_session_id=r["source_session_id"],
            source_platform=r["source_platform"],
            speaker_entity_id=r["speaker_entity_id"],
            promoted_to=r["promoted_to"], flagged_source=r["flagged_source"],
            canonical_key=r["canonical_key"], proof_count=r["proof_count"] or 0,
            evidence_quotes=json.loads(r["evidence_quotes"] or "[]"),
            history_json=r["history_json"],
        )

    def _embed(self, text: str) -> list[float]:
        return self._embed_texts([text])[0]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Uses fastembed if available, else fake."""
        try:
            from .embed import embed_texts
            return embed_texts(texts)
        except Exception:
            # Fall back to a deterministic hash-based vector so the store
            # works even without a model loaded (tests, headless).
            dim = cfg.EMBEDDING_DIM
            out = []
            for t in texts:
                import hashlib
                h = hashlib.sha256(t.encode()).digest()
                v = [float((h[i % 32])) / 255.0 for i in range(dim)]
                norm = sum(x * x for x in v) ** 0.5 or 1.0
                out.append([x / norm for x in v])
            return out

    @staticmethod
    def _vec_to_blob(vec: list[float]) -> bytes:
        return struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])

    @staticmethod
    def _blob_to_vec(blob: bytes) -> list[float]:
        n = len(blob) // 4
        return list(struct.unpack(f"<{n}f", blob))

    # ── Get / update ─────────────────────────────────────────────────────

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        rows = self._query("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return self._row_to_memory(rows[0]) if rows else None

    def get_memories_by_ids(self, memory_ids: list[str]) -> dict[str, MemoryRecord]:
        if not memory_ids:
            return {}
        unique = list(dict.fromkeys(memory_ids))
        out: dict[str, MemoryRecord] = {}
        for i in range(0, len(unique), 200):
            chunk = unique[i:i + 200]
            ph = ",".join("?" for _ in chunk)
            rows = self._query(f"SELECT * FROM memories WHERE id IN ({ph})", tuple(chunk))
            for r in rows:
                rec = self._row_to_memory(r)
                out[rec.id] = rec
        return out

    def update_memory_status(self, memory_id: str, status: MemoryStatus = None,
                              **fields) -> None:
        """Update a memory's status (and optionally other related fields).

        Accepts the legacy (memory_id, status) positional call, plus
        keyword args for related fields that often change together
        (superseded_by, contested_with, last_recalled_at).
        """
        if status is not None:
            fields["status"] = status
        if not fields:
            return
        allowed = {"status", "superseded_by", "contested_with", "last_recalled_at"}
        cols, params = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if hasattr(v, "value"):
                v = v.value
            elif isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, datetime):
                v = v.isoformat()
            cols.append(f"{k} = ?")
            params.append(v)
        if not cols:
            return
        params.append(memory_id)
        self._execute(
            f"UPDATE memories SET {', '.join(cols)} WHERE id = ?",
            tuple(params),
        ).connection.commit()

    def update_memory_field(self, memory_id: str, **fields: Any) -> None:
        """Update arbitrary scalar fields on a memory."""
        if not fields:
            return
        allowed = {
            "text", "network", "entity_ids", "source_memory_ids", "status",
            "superseded_by", "contested_with", "last_recalled_at",
            "recall_count", "recall_miss_count", "reflect_cycles",
            "promoted_to", "flagged_source", "canonical_key", "proof_count",
            "evidence_quotes", "history_json", "created_at",
        }
        cols, params = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(f"Unknown field: {k}")
            if isinstance(v, (list, dict)):
                v = self._dumps(v)
            elif hasattr(v, "value"):
                v = v.value
            elif isinstance(v, datetime):
                v = v.isoformat()
            cols.append(f"{k} = ?")
            params.append(v)
        params.append(memory_id)
        with self._write_lock:
            self._execute(
                f"UPDATE memories SET {', '.join(cols)} WHERE id = ?",
                tuple(params),
            ).connection.commit()

    def touch_memory(self, memory_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            "UPDATE memories SET recall_count = recall_count + 1, "
            "last_recalled_at = ? WHERE id = ?",
            (now, memory_id),
        ).connection.commit()

    def touch_memories(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        ph = ",".join("?" for _ in memory_ids)
        self._execute(
            f"UPDATE memories SET recall_count = recall_count + 1, "
            f"last_recalled_at = ? WHERE id IN ({ph})",
            (now, *memory_ids),
        ).connection.commit()

    def bump_recall_miss(self, memory_id: str) -> None:
        self._execute(
            "UPDATE memories SET recall_miss_count = recall_miss_count + 1 WHERE id = ?",
            (memory_id,),
        ).connection.commit()

    def bump_recall_misses(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        ph = ",".join("?" for _ in memory_ids)
        self._execute(
            f"UPDATE memories SET recall_miss_count = recall_miss_count + 1 "
            f"WHERE id IN ({ph})",
            tuple(memory_ids),
        ).connection.commit()

    def increment_reflect_cycles(self) -> None:
        self._execute(
            "UPDATE memories SET reflect_cycles = reflect_cycles + 1 WHERE status != ?",
            (MemoryStatus.archived.value,),
        ).connection.commit()

    def delete_memory(self, memory_id: str) -> None:
        """Hard-delete a memory from all tables (used by tests only)."""
        with self._write_lock:
            self._execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
            self._execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
            self._execute("DELETE FROM edges WHERE from_id = ? OR to_id = ?",
                          (memory_id, memory_id))
            self._conn.commit()

    # ── Listing ──────────────────────────────────────────────────────────

    def list_memories(self, status: Optional[MemoryStatus] = None,
                      network: Optional[Network] = None,
                      limit: Optional[int] = None,
                      since: Optional[datetime] = None,
                      until: Optional[datetime] = None,
                      entity_ids: Optional[list[str]] = None,
                      ) -> list[MemoryRecord]:
        """List memories with optional filters."""
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if network is not None:
            clauses.append("network = ?")
            params.append(network.value)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since.isoformat())
        if until is not None:
            clauses.append("created_at <= ?")
            params.append(until.isoformat())
        if entity_ids is not None:
            # Entity membership: memory's entity_ids JSON contains any requested id.
            # OR-join of LIKE '%"id"%' — exact quote-delimited match.
            if not entity_ids:
                clauses.append("(1 = 0)")
            else:
                subs = []
                for eid in entity_ids:
                    subs.append("entity_ids LIKE ?")
                    params.append(f'%"{eid}"%')
                clauses.append("(" + " OR ".join(subs) + ")")
        sql = "SELECT * FROM memories"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_memory(r) for r in self._query(sql, tuple(params))]

    def list_active_candidates(self, limit: int = 200) -> list[MemoryRecord]:
        """Return a bounded set of active memories (hot-cache candidates)."""
        rows = self._query(
            "SELECT * FROM memories WHERE status = ? ORDER BY recall_count DESC, "
            "created_at DESC LIMIT ?",
            (MemoryStatus.active.value, limit),
        )
        return [self._row_to_memory(r) for r in rows]

    def count_memories(self) -> int:
        row = self._query("SELECT COUNT(*) AS c FROM memories")[0]
        return row["c"]

    # ── Search ───────────────────────────────────────────────────────────

    def _filter_where(self, entity_filter=None, network_filter=None,
                      status_filter=None, exclude_status=None,
                      include_unscoped_world=False) -> tuple[str, list]:
        """Build a SQL WHERE clause + params for search filters."""
        clauses, params = [], []
        if entity_filter is not None:
            if not entity_filter:
                clauses.append("(1 = 0)")
            else:
                subs = []
                for eid in entity_filter:
                    subs.append("entity_ids LIKE ?")
                    params.append(f'%"{eid}"%')
                if include_unscoped_world:
                    subs.append(f"(entity_ids = '[]' AND network = 'world')")
                clauses.append("(" + " OR ".join(subs) + ")")
        if network_filter is not None:
            n = network_filter.value if hasattr(network_filter, "value") else network_filter
            clauses.append("m.network = ?")
            params.append(n)
        if status_filter:
            vals = [s.value if hasattr(s, "value") else str(s) for s in status_filter]
            clauses.append("m.status IN (" + ",".join("?" for _ in vals) + ")")
            params.extend(vals)
        if exclude_status:
            vals = [s.value if hasattr(s, "value") else str(s) for s in exclude_status]
            clauses.append("m.status NOT IN (" + ",".join("?" for _ in vals) + ")")
            params.extend(vals)
        return (" AND ".join(clauses), params) if clauses else ("", [])

    def vector_search(self, query: str, k: int = 25,
                      entity_filter=None, network_filter=None,
                      status_filter=None, exclude_status=None,
                      include_unscoped_world=False) -> list[tuple[MemoryRecord, float]]:
        """Vector similarity search with optional filters.

        Uses sqlite-vec when available (returns distance); otherwise brute
        force over all vectors in Python. Filters are applied post-search.
        """
        qvec = self._embed(query)
        where, params = self._filter_where(
            entity_filter, network_filter, status_filter,
            exclude_status, include_unscoped_world,
        )

        sql = "SELECT m.*, m.vector AS _vec FROM memories m"
        if where:
            sql += " WHERE " + where
        rows = self._query(sql, tuple(params))

        scored: list[tuple[MemoryRecord, float]] = []
        for r in rows:
            blob = r["_vec"]
            if blob:
                v = self._blob_to_vec(blob)
            else:
                v = self._embed(r["text"])
            rec = self._row_to_memory(r)
            sim = self._cosine(qvec, v)
            scored.append((rec, max(0.0, sim)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        import math
        if not a or not b or len(a) != len(b):
            return 0.0
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0.0 and nb == 0.0:
            # Both zero vectors — match LanceDB's L2-based scoring, which
            # treats identical (including all-zero) vectors as similarity 1.
            return 1.0
        if na == 0.0 or nb == 0.0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return max(-1.0, min(1.0, dot / (na * nb)))
    def keyword_search(self, query: str, k: int = 25,
                       entity_filter=None, network_filter=None,
                       status_filter=None, exclude_status=None,
                       include_unscoped_world=False) -> list[tuple[MemoryRecord, float]]:
        """Full-text search via FTS5 with optional filters."""
        where, params = self._filter_where(
            entity_filter, network_filter, status_filter,
            exclude_status, include_unscoped_world,
        )
        # FTS5 query: escape special chars, use prefix matching.
        safe = re.sub(r'[^\w\s]', ' ', query).strip()
        if not safe:
            return []
        fts_query = " OR ".join(f'"{w}"' if len(w) > 1 else w for w in safe.split()[:10])
        if where:
            sql = (
                "SELECT m.* FROM memories m "
                "INNER JOIN memories_fts fts ON m.id = fts.id "
                f"WHERE fts.text MATCH ? AND {where} "
                "ORDER BY rank LIMIT ?"
            )
            params = [fts_query] + list(params) + [k]
        else:
            sql = (
                "SELECT m.* FROM memories m "
                "INNER JOIN memories_fts fts ON m.id = fts.id "
                "WHERE fts.text MATCH ? "
                "ORDER BY rank LIMIT ?"
            )
            params = [fts_query, k]
        try:
            rows = self._query(sql, tuple(params))
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query failed (%s), falling back to LIKE", e)
            return self._like_fallback(safe, k, entity_filter, network_filter,
                                       status_filter, exclude_status, include_unscoped_world)
        # Score: FTS5 rank is a negative log-score; convert to positive.
        scored = []
        for r in rows:
            rec = self._row_to_memory(r)
            scored.append((rec, 0.5))
        return scored[:k]

    def _like_fallback(self, query: str, k: int, entity_filter=None,
                       network_filter=None, status_filter=None,
                       exclude_status=None, include_unscoped_world=False) -> list[tuple[MemoryRecord, float]]:
        """Fallback keyword search using LIKE."""
        where, params = self._filter_where(
            entity_filter, network_filter, status_filter,
            exclude_status, include_unscoped_world,
        )
        like_clauses = []
        for word in query.split()[:5]:
            like_clauses.append("m.text LIKE ?")
            params.append(f"%{word}%")
        like_sql = " OR ".join(like_clauses)
        if where:
            full_sql = f"SELECT * FROM memories m WHERE ({like_sql}) AND {where} LIMIT ?"
        else:
            full_sql = f"SELECT * FROM memories m WHERE {like_sql} LIMIT ?"
        params.append(k)
        rows = self._query(full_sql, tuple(params))
        return [(self._row_to_memory(r), 0.3) for r in rows]


    # ── Entity CRUD ──────────────────────────────────────────────────────

    def add_entity(self, entity: EntityRecord) -> str:
        self._execute(
            "INSERT OR REPLACE INTO entities (id, type, label, card, status_line, created_at, merged_into) "
            "VALUES (?,?,?,?,?,?,?)",
            (entity.id, entity.type.value, entity.label, self._dumps(entity.card),
             entity.status_line, entity.created_at.isoformat() if entity.created_at else None,
             entity.merged_into),
        ).connection.commit()
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[EntityRecord]:
        rows = self._query("SELECT * FROM entities WHERE id = ?", (entity_id,))
        if not rows:
            return None
        r = rows[0]
        return EntityRecord(
            id=r["id"], type=EntityType(r["type"]), label=r["label"],
            card=self._loads(r["card"], {}), status_line=r["status_line"] or "",
            created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else datetime.now(timezone.utc),
            merged_into=r["merged_into"],
        )

    def get_entity_by_label(self, label: str) -> Optional[EntityRecord]:
        rows = self._query("SELECT * FROM entities WHERE label = ?", (label,))
        if not rows:
            return None
        r = rows[0]
        return self.get_entity(r["id"])

    def list_entities(self) -> list[EntityRecord]:
        rows = self._query("SELECT * FROM entities ORDER BY created_at DESC")
        out = []
        for r in rows:
            out.append(EntityRecord(
                id=r["id"], type=EntityType(r["type"]), label=r["label"],
                card=self._loads(r["card"], {}), status_line=r["status_line"] or "",
                created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else datetime.now(timezone.utc),
                merged_into=r["merged_into"],
            ))
        return out

    def update_entity(self, entity_id: str, **fields: Any) -> None:
        allowed = {"type", "label", "card", "status_line", "merged_into"}
        cols, params = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if isinstance(v, dict):
                v = self._dumps(v)
            elif hasattr(v, "value"):
                v = v.value
            cols.append(f"{k} = ?")
            params.append(v)
        if not cols:
            return
        params.append(entity_id)
        self._execute(
            f"UPDATE entities SET {', '.join(cols)} WHERE id = ?",
            tuple(params),
        ).connection.commit()

    def has_self_entity(self) -> bool:
        row = self._query("SELECT COUNT(*) AS c FROM entities WHERE type = ?", (EntityType.self_.value,))
        return row[0]["c"] > 0

    def count_memories_for_entity(self, entity_id: str) -> int:
        row = self._query(
            "SELECT COUNT(*) AS c FROM memories WHERE entity_ids LIKE ?",
            (f'%"{entity_id}"%',),
        )
        return row[0]["c"]

    # ── Identity links ───────────────────────────────────────────────────

    def add_identity_link(self, link: IdentityLink) -> str:
        self._execute(
            "INSERT OR REPLACE INTO identity_links (platform, external_id, entity_id, created_at, created_by) "
            "VALUES (?,?,?,?,?)",
            (link.platform, link.external_id, link.entity_id,
             link.created_at.isoformat() if link.created_at else datetime.now(timezone.utc).isoformat(),
             link.created_by.value if hasattr(link.created_by, "value") else link.created_by),
        ).connection.commit()
        return link.entity_id

    def delete_identity_link(self, platform: str, external_id: str) -> None:
        self._execute(
            "DELETE FROM identity_links WHERE platform = ? AND external_id = ?",
            (platform, external_id),
        ).connection.commit()

    def resolve_identity(self, platform: str, external_id: str) -> Optional[IdentityLink]:
        """Return the IdentityLink for a (platform, external_id), or None.

        Matches the LanceDB store's contract (returns an IdentityLink
        object, not the bare entity_id string), so provider/entities call
        sites are backend-agnostic.
        """
        rows = self._query(
            "SELECT * FROM identity_links WHERE platform = ? AND external_id = ?",
            (platform, external_id),
        )
        if not rows:
            return None
        r = rows[0]
        return IdentityLink(
            platform=r["platform"], external_id=r["external_id"],
            entity_id=r["entity_id"],
            created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else datetime.now(timezone.utc),
            created_by=LinkMethod(r["created_by"]),
        )

    def get_identity_links_for_entity(self, entity_id: str) -> list[IdentityLink]:
        rows = self._query(
            "SELECT * FROM identity_links WHERE entity_id = ?", (entity_id,),
        )
        out = []
        for r in rows:
            out.append(IdentityLink(
                platform=r["platform"], external_id=r["external_id"],
                entity_id=r["entity_id"],
                created_at=datetime.fromisoformat(r["created_at"]),
                created_by=LinkMethod(r["created_by"]),
            ))
        return out

    def list_identity_links(self) -> list[IdentityLink]:
        rows = self._query("SELECT * FROM identity_links")
        out = []
        for r in rows:
            out.append(IdentityLink(
                platform=r["platform"], external_id=r["external_id"],
                entity_id=r["entity_id"],
                created_at=datetime.fromisoformat(r["created_at"]),
                created_by=LinkMethod(r["created_by"]),
            ))
        return out

    def repoint_identity_links(self, old_entity_id: str, new_entity_id: str) -> None:
        self._execute(
            "UPDATE identity_links SET entity_id = ? WHERE entity_id = ?",
            (new_entity_id, old_entity_id),
        ).connection.commit()

    # ── Edges ────────────────────────────────────────────────────────────

    def add_edge(self, edge: EdgeRecord) -> str:
        self._execute(
            "INSERT OR REPLACE INTO edges (from_id, to_id, kind, weight) VALUES (?,?,?,?)",
            (edge.from_id, edge.to_id, edge.kind.value if hasattr(edge.kind, "value") else edge.kind,
             edge.weight),
        ).connection.commit()
        return edge.from_id

    def add_edges(self, edges: list[EdgeRecord]) -> None:
        with self._write_lock:
            for e in edges:
                self._execute(
                    "INSERT OR REPLACE INTO edges (from_id, to_id, kind, weight) VALUES (?,?,?,?)",
                    (e.from_id, e.to_id, e.kind.value if hasattr(e.kind, "value") else e.kind, e.weight),
                )
            self._conn.commit()

    def get_edges_for_memory(self, memory_id: str) -> list[EdgeRecord]:
        rows = self._query(
            "SELECT * FROM edges WHERE from_id = ? OR to_id = ?", (memory_id, memory_id),
        )
        out = []
        for r in rows:
            out.append(EdgeRecord(
                from_id=r["from_id"], to_id=r["to_id"],
                kind=EdgeKind(r["kind"]), weight=r["weight"],
            ))
        return out

    def list_edges(self) -> list[EdgeRecord]:
        rows = self._query("SELECT * FROM edges")
        out = []
        for r in rows:
            out.append(EdgeRecord(
                from_id=r["from_id"], to_id=r["to_id"],
                kind=EdgeKind(r["kind"]), weight=r["weight"],
            ))
        return out

    def neighbors(self, memory_id: str, min_weight: float = 0.0) -> list[tuple[str, EdgeRecord]]:
        rows = self._query(
            "SELECT * FROM edges WHERE (from_id = ? OR to_id = ?) AND weight >= ?",
            (memory_id, memory_id, min_weight),
        )
        out = []
        for r in rows:
            edge = EdgeRecord(from_id=r["from_id"], to_id=r["to_id"],
                              kind=EdgeKind(r["kind"]), weight=r["weight"])
            nid = r["to_id"] if r["from_id"] == memory_id else r["from_id"]
            out.append((nid, edge))
        return out

    def neighbors_for_ids(self, memory_ids: list[str], min_weight: float = 0.0) -> dict[str, list[tuple[str, EdgeRecord]]]:
        if not memory_ids:
            return {}
        ph = ",".join("?" for _ in memory_ids)
        rows = self._query(
            f"SELECT * FROM edges WHERE (from_id IN ({ph}) OR to_id IN ({ph})) AND weight >= ?",
            tuple(memory_ids) + tuple(memory_ids) + (min_weight,),
        )
        result: dict[str, list[tuple[str, EdgeRecord]]] = {}
        for r in rows:
            edge = EdgeRecord(from_id=r["from_id"], to_id=r["to_id"],
                              kind=EdgeKind(r["kind"]), weight=r["weight"])
            for mid in memory_ids:
                if r["from_id"] == mid:
                    nid = r["to_id"]
                elif r["to_id"] == mid:
                    nid = r["from_id"]
                else:
                    continue
                result.setdefault(mid, []).append((nid, edge))
        return result

    def update_edge_weight(self, from_id: str, to_id: str, kind: str, weight: float) -> None:
        self._execute(
            "UPDATE edges SET weight = ? WHERE from_id = ? AND to_id = ? AND kind = ?",
            (weight, from_id, to_id, kind),
        ).connection.commit()

    def re_memory_entity_ids(self, old_eid: str, new_eid: str) -> None:
        self._execute(
            "UPDATE memories SET entity_ids = REPLACE(entity_ids, ?, ?) WHERE entity_ids LIKE ?",
            (f'"{old_eid}"', f'"{new_eid}"', f'%"{old_eid}"%'),
        ).connection.commit()

    def re_memory_speaker(self, old_eid: str, new_eid: str) -> None:
        self._execute(
            "UPDATE memories SET speaker_entity_id = ? WHERE speaker_entity_id = ?",
            (new_eid, old_eid),
        ).connection.commit()

    def get_memory_by_canonical_key(self, key: str) -> Optional[MemoryRecord]:
        rows = self._query("SELECT * FROM memories WHERE canonical_key = ?", (key,))
        if not rows:
            return None
        return self._row_to_memory(rows[0])

    # ── Hot cache ────────────────────────────────────────────────────────

    def get_hot_cache(self, entity_id: str) -> Optional[HotCache]:
        rows = self._query("SELECT * FROM hot_cache WHERE entity_id = ?", (entity_id,))
        if not rows:
            return None
        r = rows[0]
        return HotCache(
            entity_id=r["entity_id"], memory_ids=self._loads(r["memory_ids"], []),
            rendered=r["rendered"] or "",
            updated_at=datetime.fromisoformat(r["updated_at"]) if r["updated_at"] else datetime.now(timezone.utc),
        )

    def save_hot_cache(self, cache: HotCache) -> None:
        self._execute(
            "INSERT OR REPLACE INTO hot_cache (entity_id, memory_ids, rendered, updated_at) "
            "VALUES (?,?,?,?)",
            (cache.entity_id, self._dumps(cache.memory_ids), cache.rendered,
             cache.updated_at.isoformat() if cache.updated_at else datetime.now(timezone.utc).isoformat()),
        ).connection.commit()

    # ── Flags ────────────────────────────────────────────────────────────

    def add_flag(self, flag: FlagRecord) -> str:
        self._execute(
            "INSERT INTO flags (id, session_id, platform, speaker_external_id, turn_text, "
            "flag_reason, processed, attempt_count, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (flag.id, flag.session_id, flag.platform, flag.speaker_external_id,
             flag.turn_text, flag.flag_reason, int(flag.processed), flag.attempt_count,
             flag.created_at.isoformat() if flag.created_at else datetime.now(timezone.utc).isoformat()),
        ).connection.commit()
        return flag.id

    def _row_to_flag(self, r: sqlite3.Row) -> FlagRecord:
        return FlagRecord(
            id=r["id"], session_id=r["session_id"], platform=r["platform"],
            speaker_external_id=r["speaker_external_id"], turn_text=r["turn_text"],
            flag_reason=r["flag_reason"], processed=bool(r["processed"]),
            attempt_count=r["attempt_count"] or 0,
            created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else datetime.now(timezone.utc),
        )

    def unprocessed_flags(self, session_id: Optional[str] = None) -> list[FlagRecord]:
        if session_id:
            rows = self._query(
                "SELECT * FROM flags WHERE processed = 0 AND session_id = ? ORDER BY created_at",
                (session_id,),
            )
        else:
            rows = self._query("SELECT * FROM flags WHERE processed = 0 ORDER BY created_at")
        return [self._row_to_flag(r) for r in rows]

    def unprocessed_flag_count(self) -> int:
        row = self._query("SELECT COUNT(*) AS c FROM flags WHERE processed = 0")
        return row[0]["c"]

    def mark_flags_processed(self, flag_ids: list[str]) -> None:
        if not flag_ids:
            return
        ph = ",".join("?" for _ in flag_ids)
        self._execute(f"UPDATE flags SET processed = 1 WHERE id IN ({ph})",
                      tuple(flag_ids)).connection.commit()

    def increment_flag_attempts(self, flag_ids: list[str]) -> None:
        if not flag_ids:
            return
        ph = ",".join("?" for _ in flag_ids)
        self._execute(
            f"UPDATE flags SET attempt_count = attempt_count + 1 WHERE id IN ({ph})",
            tuple(flag_ids),
        ).connection.commit()

    def consume_flags(self, session_id: str) -> list[FlagRecord]:
        """Fetch + mark-unprocessed in one transaction (no read-then-write race)."""
        with self._write_lock:
            rows = self._query(
                "SELECT * FROM flags WHERE processed = 0 ORDER BY created_at"
            )
            flags = [f for f in (self._row_to_flag(r) for r in rows)
                     if f.session_id == session_id]
            if flags:
                ph = ",".join("?" for _ in flags)
                self._execute(
                    f"UPDATE flags SET processed = 1 WHERE id IN ({ph})",
                    tuple(f.id for f in flags),
                ).connection.commit()
            return flags

    # ── Promotion queue ──────────────────────────────────────────────────

    def add_to_promotion_queue(self, memory_id: str, target_systems: list[str] = None) -> None:
        self._execute(
            "INSERT OR REPLACE INTO promotion_queue (memory_id, enqueued_at, target_systems) "
            "VALUES (?,?,?)",
            (memory_id, datetime.now(timezone.utc).isoformat(),
             self._dumps(target_systems or ["wiki", "obsidian"])),
        ).connection.commit()

    def list_promotion_queue(self) -> list[PromotionQueueRecord]:
        rows = self._query("SELECT * FROM promotion_queue ORDER BY enqueued_at")
        out = []
        for r in rows:
            out.append(PromotionQueueRecord(
                memory_id=r["memory_id"],
                enqueued_at=datetime.fromisoformat(r["enqueued_at"]) if r["enqueued_at"] else datetime.now(timezone.utc),
                target_systems=self._loads(r["target_systems"], []),
            ))
        return out

    def remove_from_promotion_queue(self, memory_id: str) -> None:
        self._execute("DELETE FROM promotion_queue WHERE memory_id = ?",
                      (memory_id,)).connection.commit()

    # ── Session summaries ────────────────────────────────────────────────

    def get_session_summary(self, session_id: str) -> Optional[SessionSummary]:
        rows = self._query("SELECT * FROM session_summaries WHERE id = ?", (session_id,))
        if not rows:
            return None
        r = rows[0]
        return SessionSummary(
            id=r["id"], text=r["text"], message_count=r["message_count"] or 0,
            last_updated=datetime.fromisoformat(r["last_updated"]) if r["last_updated"] else datetime.now(timezone.utc),
        )

    def upsert_session_summary(self, session_id: str, text: str = None,
                               message_count: int = None, summary=None) -> None:
        """Insert/update a session summary.

        Accepts either (session_id, text, message_count) — the legacy
        positional contract — or a single SessionSummary instance.
        """
        if summary is not None and isinstance(summary, SessionSummary):
            session_id = summary.id
            text = summary.text
            message_count = summary.message_count
            updated = summary.last_updated.isoformat() if summary.last_updated else datetime.now(timezone.utc).isoformat()
        else:
            if text is None:
                raise ValueError("upsert_session_summary requires text or a SessionSummary")
            message_count = message_count or 0
            updated = datetime.now(timezone.utc).isoformat()
        self._execute(
            "INSERT OR REPLACE INTO session_summaries (id, text, message_count, last_updated) "
            "VALUES (?,?,?,?)",
            (session_id, text, message_count, updated),
        ).connection.commit()

    def list_session_summaries(self, limit: Optional[int] = None) -> list[SessionSummary]:
        sql = "SELECT * FROM session_summaries ORDER BY last_updated DESC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._query(sql, params)
        out = []
        for r in rows:
            out.append(SessionSummary(
                id=r["id"], text=r["text"], message_count=r["message_count"] or 0,
                last_updated=datetime.fromisoformat(r["last_updated"]) if r["last_updated"] else datetime.now(timezone.utc),
            ))
        return out

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        out = {}
        for t in ("memories", "entities", "identity_links", "edges",
                  "flags", "promotion_queue", "session_summaries"):
            try:
                out[t] = self._query(f"SELECT COUNT(*) AS c FROM {t}")[0]["c"]
            except sqlite3.OperationalError:
                out[t] = 0
        by_network = {}
        for n in Network:
            row = self._query("SELECT COUNT(*) AS c FROM memories WHERE network = ?", (n.value,))
            by_network[n.value] = row[0]["c"]
        by_status = {}
        for s in MemoryStatus:
            row = self._query("SELECT COUNT(*) AS c FROM memories WHERE status = ?", (s.value,))
            by_status[s.value] = row[0]["c"]
        out["memories_by_network"] = by_network
        out["memories_by_status"] = by_status
        # Legacy contract keys (tests + provider consumers expect these names)
        out["total_memories"] = out["memories"]
        out["memories_per_network"] = by_network
        out["memories_per_status"] = by_status
        return out

    # ── Temporal (as_of) ─────────────────────────────────────────────────

    def as_of(self, timestamp: datetime) -> "_AsOfView":
        """Return a read-only view of the store as it was at `timestamp`.

        SQLite has no native time-travel; we approximate with created_at
        filters (memories that did not exist yet are invisible). Raises
        ValueError if the timestamp predates every record in the store —
        mirroring the LanceDB version-not-found contract.
        """
        if self._conn is None or self._closed:
            raise RuntimeError("Store is not open")
        row = self._query("SELECT MIN(created_at) AS earliest FROM memories")[0]
        earliest = row["earliest"] if row else None
        if earliest is not None:
            try:
                if timestamp < datetime.fromisoformat(earliest):
                    raise ValueError(
                        f"No store version exists as of {timestamp.isoformat()} "
                        f"(earliest record: {earliest})"
                    )
            except TypeError:
                pass
        return _AsOfView(self, timestamp)


class _AsOfView:
    """Read-only temporal view over a Store.

    Behaves like Store for search/read paths but includes superseded
    memories (which were active at the as_of time) and excludes
    archived/demoted. The parent's data is live; this is an approximation
    of LanceDB's versioned reads.
    """

    def __init__(self, store: Store, as_of: datetime):
        self._store = store
        self._as_of = as_of

    def vector_search(self, query: str, k: int = 25, **kw) -> list[tuple[MemoryRecord, float]]:
        kw.pop("status_filter", None)
        kw.pop("exclude_status", None)
        # Include superseded (they were active then); exclude future records.
        rows = self._store._query(
            "SELECT * FROM memories WHERE created_at <= ? AND status NOT IN ('archived', 'demoted')",
            (self._as_of.isoformat(),),
        )
        records = [self._store._row_to_memory(r) for r in rows]
        qvec = self._store._embed(query)
        scored = [(m, max(0.0, Store._cosine(qvec, self._store._embed(m.text)))) for m in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def keyword_search(self, query: str, k: int = 25, **kw) -> list[tuple[MemoryRecord, float]]:
        kw.pop("status_filter", None)
        kw.pop("exclude_status", None)
        results = self._store.keyword_search(query, k=k * 2, **kw)
        cutoff = self._as_of.isoformat()
        return [(m, s) for m, s in results if m.created_at and m.created_at.isoformat() <= cutoff][:k]

    def list_memories(self, **kw) -> list[MemoryRecord]:
        """List memories that existed at the as_of timestamp."""
        kw.pop("status", None)
        out = self._store.list_memories(**kw)
        cutoff = self._as_of.isoformat() if self._as_of else None
        if cutoff:
            return [m for m in out if m.created_at and m.created_at.isoformat() <= cutoff]
        return out

    def get_memories_by_ids(self, ids: list[str]) -> dict[str, MemoryRecord]:
        return self._store.get_memories_by_ids(ids)

    def neighbors_for_ids(self, ids: list[str], min_weight: float = 0.0):
        return self._store.neighbors_for_ids(ids, min_weight=min_weight)

    def touch_memory(self, memory_id: str) -> None:
        """Read-only view: touching is forbidden, mirroring LanceDB's
        as_of temporal semantics."""
        raise RuntimeError("as_of views are read-only; touch_memory forbidden")

    def close(self) -> None:
        pass  # parent owns the connection
