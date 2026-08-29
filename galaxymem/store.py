"""LanceDB storage layer — typed CRUD over the six tables with zero business logic.

All business logic lives in retain.py, recall.py, reflect.py, etc.
This module handles: table creation, embedding, CRUD, vector/keyword search,
edge operations, hot cache, flag queue, and temporal versioning.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import lancedb
import numpy as np
from lancedb.table import Table as LanceTable

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
)
from .redact import redact_secrets
from .schema import _esc, _in_list
from .utils import ulid as _ulid

logger = logging.getLogger(__name__)


def _table_row_count(table: LanceTable) -> int:
    """Return the number of rows in a LanceDB table across supported versions.

    LanceDB 0.34 exposes count_rows(); older versions have no working count()
    on the sync Table, so fall back to a full scan.
    """
    if table is None:
        return 0
    try:
        return table.count_rows()
    except Exception:
        pass
    try:
        count = table.count()
        if isinstance(count, int):
            return count
    except Exception:
        pass
    try:
        return len(table.search().to_pandas())
    except Exception:
        return 0


# ── Embedding function ──────────────────────────────────────────────────

_embed_fn = None
_embed_lock = threading.Lock()


def _get_embedding_function():
    """Lazy-init embedding function. Returns a callable that embeds strings.

    Thread-safe: the lock prevents two threads from both seeing None and
    double-initializing the model (which wastes ~200MB of memory).
    """
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn

    with _embed_lock:
        if _embed_fn is not None:
            return _embed_fn

        if cfg.EMBEDDING_BACKEND == "fastembed":
            from fastembed import TextEmbedding

            _model = TextEmbedding(
                model_name=cfg.EMBEDDING_MODEL,
                max_length=512,
                cache_dir=str(Path.home() / ".cache" / "galaxymem"),
            )

            def _make_embed_fn(model_ref):
                def embed(texts):
                    return list(model_ref.embed(texts))
                return embed

            _embed_fn = _make_embed_fn(_model)
        elif cfg.EMBEDDING_BACKEND == "api" and cfg.EMBEDDING_API_URL:
            import requests

            parsed = urlparse(cfg.EMBEDDING_API_URL)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise RuntimeError(
                    f"Invalid GALAXYMEM_EMBEDDING_API_URL: {cfg.EMBEDDING_API_URL!r} "
                    "(must be http(s)://host/...)"
                )
            url = cfg.EMBEDDING_API_URL.rstrip("/")
            headers = {"Content-Type": "application/json"}
            if cfg.EMBEDDING_API_KEY:
                headers["Authorization"] = f"Bearer {cfg.EMBEDDING_API_KEY}"

            def _api_embed(texts):
                if not texts:
                    return []
                items = list(texts)
                out: list[list[float]] = []
                # Chunk so a runaway batch cannot DoS the API, but never
                # silently drop the tail of a legitimate batch.
                for i in range(0, len(items), 64):
                    chunk = items[i:i + 64]
                    resp = requests.post(
                        f"{url}/embeddings",
                        json={"input": chunk, "model": cfg.EMBEDDING_MODEL},
                        headers=headers,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list):
                        out.extend(d["embedding"] for d in data)
                    else:
                        out.extend(d["embedding"] for d in data["data"])
                if len(out) != len(items):
                    raise RuntimeError(
                        f"Embedding API returned {len(out)} vectors for {len(items)} inputs"
                    )
                return out

            _embed_fn = _api_embed
        else:
            raise RuntimeError(
                f"Unknown embedding backend: {cfg.EMBEDDING_BACKEND}. "
                f"Set GALAXYMEM_EMBEDDING_BACKEND to 'fastembed' or 'api' and configure URL."
            )

    return _embed_fn


def embed_text(text: str) -> list[float]:
    return list(_get_embedding_function()([text])[0])


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [list(v) for v in _get_embedding_function()(texts)]


# ── LanceDB Pydantic models (table schemas) ────────────────────────────
from .schema import (  # noqa: F401
    MemoriesTable,
    EntitiesTable,
    IdentityLinksTable,
    EdgesTable,
    HotCacheTable,
    FlagsTable,
    PromotionQueueTable,
    SessionSummariesTable,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _close_tbl(tbl) -> None:
    """Best-effort close of a LanceDB table handle to release file descriptors."""
    if tbl is None:
        return
    try:
        inner = getattr(tbl, "_table", tbl)
        if hasattr(inner, "close") and getattr(inner, "is_open", True):
            inner.close()
    except Exception as e:
        logger.debug("close table handle failed (non-fatal): %s", e)


def _entity_membership_clause(entity_ids: list[str]) -> str:
    """Exact-membership filter over the JSON-encoded entity_ids column.

    Matches the quote-delimited element ("sam" never matches "samuel") —
    the D8 hard filter must not leak on slug substrings.
    """
    if not entity_ids:
        return "(1 = 0)"
    parts = [f'entity_ids LIKE \'%"{_esc(e, escape_like=True)}"%\'' for e in entity_ids]
    return f"({' OR '.join(parts)})"


def _parse_json_field(val, default):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return json.loads(default) if isinstance(default, str) else default
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError, ValueError):
        return json.loads(default) if isinstance(default, str) else default


def _safe_str(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val)


def _parse_dt(val) -> Optional[datetime]:
    s = _safe_str(val)
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _from_memory(t) -> MemoryRecord:
    """Convert a MemoriesTable row (or dict-like) to MemoryRecord."""

    def g(key, default=None):
        if isinstance(t, dict):
            return t.get(key, default)
        return getattr(t, key, default)

    created = _parse_dt(g("created_at")) or datetime.now(timezone.utc)
    raw_network = g("network")
    try:
        network = Network(raw_network)
    except (ValueError, TypeError):
        logger.warning("Invalid network %r on memory %s; treating as observation",
                       raw_network, g("id"))
        network = Network.observation
    raw_status = g("status")
    try:
        status = MemoryStatus(raw_status)
    except (ValueError, TypeError):
        logger.warning("Invalid status %r on memory %s; treating as archived",
                       raw_status, g("id"))
        status = MemoryStatus.archived
    return MemoryRecord(
        id=str(g("id")),
        text=str(g("text") or ""),
        network=network,
        entity_ids=_parse_json_field(g("entity_ids"), "[]"),
        source_memory_ids=_parse_json_field(g("source_memory_ids"), "[]"),
        status=status,
        superseded_by=_safe_str(g("superseded_by")),
        contested_with=_parse_json_field(g("contested_with"), "[]"),
        created_at=created,
        last_recalled_at=_parse_dt(g("last_recalled_at")),
        recall_count=int(g("recall_count") or 0),
        recall_miss_count=int(g("recall_miss_count") or 0),
        reflect_cycles=int(g("reflect_cycles") or 0),
        source_session_id=_safe_str(g("source_session_id")),
        source_platform=_safe_str(g("source_platform")),
        speaker_entity_id=_safe_str(g("speaker_entity_id")),
        promoted_to=_safe_str(g("promoted_to")),
        flagged_source=_safe_str(g("flagged_source")),
        canonical_key=_safe_str(g("canonical_key")),
        proof_count=int(g("proof_count") or 0),
        history_json=_safe_str(g("history_json")),
        evidence_quotes=_parse_json_field(g("evidence_quotes"), "[]"),
    )


def _prepare_memory_text(text: str) -> str:
    """Redact secrets and cap length before a row is written."""
    from . import config as cfg
    if not text:
        return text
    redacted = redact_secrets(text)
    cap = cfg.MAX_MEMORY_TEXT_CHARS
    if len(redacted) > cap:
        redacted = redacted[:cap]
    return redacted


def _to_memory_row(m: MemoryRecord, vector: Optional[list[float]] = None) -> dict:
    text = _prepare_memory_text(m.text)
    return {
        "id": m.id,
        "text": text,
        "vector": vector if vector is not None else embed_text(text),
        "network": m.network.value,
        "entity_ids": json.dumps(m.entity_ids),
        "source_memory_ids": json.dumps(m.source_memory_ids),
        "status": m.status.value,
        "superseded_by": m.superseded_by,
        "contested_with": json.dumps(m.contested_with),
        "created_at": m.created_at.isoformat() if isinstance(m.created_at, datetime) else m.created_at,
        "last_recalled_at": m.last_recalled_at.isoformat() if m.last_recalled_at else None,
        "recall_count": m.recall_count,
        "recall_miss_count": m.recall_miss_count,
        "reflect_cycles": m.reflect_cycles,
        "source_session_id": m.source_session_id,
        "source_platform": m.source_platform,
        "speaker_entity_id": m.speaker_entity_id,
        "promoted_to": m.promoted_to,
        "flagged_source": m.flagged_source,
        "canonical_key": m.canonical_key,
        "proof_count": m.proof_count,
        "history_json": m.history_json,
        "evidence_quotes": json.dumps(m.evidence_quotes),
    }


def _to_entity_row(e: EntityRecord) -> dict:
    return {
        "id": e.id,
        "type": e.type.value if isinstance(e.type, EntityType) else e.type,
        "label": e.label,
        "card": json.dumps(e.card),
        "status_line": e.status_line,
        "created_at": e.created_at.isoformat() if isinstance(e.created_at, datetime) else e.created_at,
        "merged_into": e.merged_into,
    }


def _to_identity_row(link: IdentityLink) -> dict:
    return {
        "platform": link.platform,
        "external_id": link.external_id,
        "entity_id": link.entity_id,
        "created_at": link.created_at.isoformat() if isinstance(link.created_at, datetime) else link.created_at,
        "created_by": link.created_by.value if isinstance(link.created_by, LinkMethod) else link.created_by,
    }


def _to_edge_row(edge: EdgeRecord) -> dict:
    return {
        "from_id": edge.from_id,
        "to_id": edge.to_id,
        "kind": edge.kind.value if isinstance(edge.kind, EdgeKind) else edge.kind,
        "weight": edge.weight,
    }


def _to_flag_row(f: FlagRecord) -> dict:
    return {
        "id": f.id,
        "session_id": f.session_id,
        "platform": f.platform,
        "speaker_external_id": f.speaker_external_id,
        "turn_text": redact_secrets(f.turn_text) if f.turn_text else f.turn_text,
        "flag_reason": f.flag_reason,
        "processed": f.processed,
        "attempt_count": getattr(f, "attempt_count", 0),
        "created_at": f.created_at.isoformat() if isinstance(f.created_at, datetime) else f.created_at,
    }


def _to_hot_cache_row(hc: HotCache) -> dict:
    return {
        "entity_id": hc.entity_id,
        "memory_ids": json.dumps(hc.memory_ids),
        "rendered": hc.rendered,
        "updated_at": hc.updated_at.isoformat() if isinstance(hc.updated_at, datetime) else hc.updated_at,
    }


# ── The Store ────────────────────────────────────────────────────────────

class Store:
    """Typed CRUD over the six LanceDB tables.

    Uses LanceDB native versioning for temporal queries (store.as_of(timestamp)).
    No business logic — just reads, writes, and searches.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._read_only = False  # True on historical as_of() handles
        self._write_lock = threading.RLock()
        self.db: Optional[lancedb.DBConnection] = None
        self._memories: Optional[LanceTable] = None
        self._entities: Optional[LanceTable] = None
        self._identities: Optional[LanceTable] = None
        self._edges: Optional[LanceTable] = None
        self._hot_cache: Optional[LanceTable] = None
        self._flags: Optional[LanceTable] = None
        self._promotion_queue: Optional[LanceTable] = None
        self._session_summaries: Optional[LanceTable] = None

    # ── Initialization ──────────────────────────────────────────────────

    def open(self, create_if_missing: bool = True) -> Store:
        """Connect to LanceDB and open/create all tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

        opened: list[tuple[str, Any]] = []
        try:
            for name, schema in [
                ("memories", MemoriesTable),
                ("entities", EntitiesTable),
                ("identity_links", IdentityLinksTable),
                ("edges", EdgesTable),
                ("hot_cache", HotCacheTable),
                ("flags", FlagsTable),
                ("promotion_queue", PromotionQueueTable),
                ("session_summaries", SessionSummariesTable),
            ]:
                try:
                    tbl = self.db.open_table(name)
                except Exception:
                    if create_if_missing:
                        tbl = self.db.create_table(name, schema=schema, exist_ok=True)
                        logger.info("Created table: %s", name)
                    else:
                        raise

                opened.append((name, tbl))

                if name == "memories":
                    self._memories = tbl
                    self._migrate_memories_schema(tbl)
                    self._ensure_indexes()
                elif name == "entities":
                    self._entities = tbl
                elif name == "identity_links":
                    self._identities = tbl
                elif name == "edges":
                    self._edges = tbl
                elif name == "hot_cache":
                    self._hot_cache = tbl
                elif name == "flags":
                    self._flags = tbl
                    self._migrate_flags_schema(tbl)
                elif name == "promotion_queue":
                    self._promotion_queue = tbl
                elif name == "session_summaries":
                    self._session_summaries = tbl

            return self
        except Exception:
            for _, tbl in opened:
                _close_tbl(tbl)
            self._memories = None
            self._entities = None
            self._identities = None
            self._edges = None
            self._hot_cache = None
            self._flags = None
            self._promotion_queue = None
            self._session_summaries = None
            raise

    def _migrate_flags_schema(self, tbl) -> None:
        """Add attempt_count to pre-existing flags tables (idempotent)."""
        try:
            existing = {f.name for f in tbl.schema}
            if "attempt_count" not in existing:
                import pyarrow as pa

                tbl.add_columns({"attempt_count": pa.int64()})
                logger.info("Migrated flags table: added column attempt_count")
        except Exception as e:
            logger.warning("flags schema migration skipped: %s", e)

    def _migrate_memories_schema(self, tbl) -> None:
        """Add columns introduced after the initial schema to existing tables."""
        try:
            existing = {f.name for f in tbl.schema}
            import pyarrow as pa

            if "proof_count" not in existing:
                tbl.add_columns({"proof_count": pa.int64()})
                logger.info("Migrated memories table: added column proof_count")
            if "recall_miss_count" not in existing:
                tbl.add_columns({"recall_miss_count": pa.int64()})
                logger.info("Migrated memories table: added column recall_miss_count")
            if "history_json" not in existing:
                tbl.add_columns({"history_json": pa.string()})
                logger.info("Migrated memories table: added column history_json")
            if "evidence_quotes" not in existing:
                tbl.add_columns({"evidence_quotes": pa.string()})
                logger.info("Migrated memories table: added column evidence_quotes")
        except Exception as e:
            logger.warning("memories schema migration skipped: %s", e)

    def _ensure_indexes(self) -> None:
        """Create vector + FTS indexes when the corpus is large enough."""
        n = _table_row_count(self._memories)
        if n >= cfg.VECTOR_INDEX_MIN_ROWS:
            self._ensure_vector_index()
        self._ensure_fts_index()

    def _ensure_vector_index(self):
        """Create IVF-PQ vector index on memories if not present."""
        try:
            existing_indices = self._memories.list_indices()
            if not any(
                (idx.get("name") if isinstance(idx, dict) else getattr(idx, "name", None))
                == "vector_idx"
                for idx in existing_indices
            ):
                n = max(_table_row_count(self._memories), 1)
                partitions = max(2, min(256, n // 40 or 2))
                try:
                    from lancedb.index import IvfPq
                    self._memories.create_index(
                        "vector",
                        config=IvfPq(distance_type="l2", num_partitions=partitions),
                    )
                except (ImportError, TypeError):
                    self._memories.create_index(
                        vector_column_name="vector",
                        index_type="IVF_PQ",
                        metric="L2",
                        num_partitions=partitions,
                        num_sub_vectors=8,
                    )
                logger.info("Created vector index on memories.vector")
        except Exception as e:
            logger.warning("Vector index creation skipped: %s", e)

    def _ensure_fts_index(self) -> None:
        """Create a full-text index on memories.text (idempotent)."""
        try:
            try:
                from lancedb.index import FTS
                self._memories.create_index("text", config=FTS())
            except (ImportError, TypeError, ValueError):
                self._memories.create_fts_index("text", replace=False)
        except Exception as e:
            logger.debug("FTS index creation skipped: %s", e)

    def as_of(self, timestamp: datetime) -> Store:
        """Return a read-only Store handle to a historical LanceDB version."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        tbl = self.db.open_table("memories")
        try:
            versions = tbl.list_versions()
        except Exception as e:
            _close_tbl(tbl)
            raise RuntimeError(
                f"Temporal queries unsupported: LanceDB versioning unavailable ({e})"
            ) from e

        target = None
        for v in versions:
            vts = v.get("timestamp")
            if vts is None:
                continue
            if vts.tzinfo is None:
                vts = vts.replace(tzinfo=timezone.utc)
            if vts <= timestamp and (target is None or v["version"] > target):
                target = v["version"]
        if target is None:
            _close_tbl(tbl)
            raise ValueError(f"No memory-table version exists at or before {timestamp.isoformat()}")

        tbl.checkout(target)

        historical = Store.__new__(Store)
        historical.db_path = self.db_path
        historical._read_only = True
        historical._write_lock = threading.RLock()
        historical.db = self.db
        historical._memories = tbl
        historical._entities = None
        historical._identities = None
        historical._edges = None
        historical._hot_cache = None
        historical._flags = None
        historical._promotion_queue = None
        historical._session_summaries = None
        return historical

    def _assert_writable(self) -> None:
        if getattr(self, "_read_only", False):
            raise RuntimeError("This Store is a read-only historical handle (as_of); writes are forbidden")

    # ── Memories ────────────────────────────────────────────────────────

    def _assert_under_limits(self, extra_memories: int = 1) -> None:
        """Raise RuntimeError if the DB size limits (config) are exceeded."""
        from . import config as cfg
        if cfg.MAX_MEMORIES > 0:
            count = _table_row_count(self._memories)
            if count + extra_memories > cfg.MAX_MEMORIES:
                raise RuntimeError(
                    f"Memory limit reached ({count} + {extra_memories} > {cfg.MAX_MEMORIES}). "
                    f"Archive or export older memories before adding more."
                )
        if cfg.MAX_EDGES > 0:
            count = _table_row_count(self._edges)
            if count >= cfg.MAX_EDGES:
                raise RuntimeError(
                    f"Edge limit reached ({count} >= {cfg.MAX_EDGES})."
                )

    def add_memory(self, memory: MemoryRecord) -> str:
        """Insert a memory. Returns its id."""
        self._assert_writable()
        with self._write_lock:
            self._assert_under_limits(1)
            prepared = _prepare_memory_text(memory.text)
            if prepared != memory.text:
                memory = memory.model_copy(update={"text": prepared})
            row = _to_memory_row(memory)
            self._memories.add([row])
            return memory.id

    def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Batch insert memories. Embeds once per text (not twice)."""
        self._assert_writable()
        if not memories:
            return []
        with self._write_lock:
            self._assert_under_limits(len(memories))
            prepared: list[MemoryRecord] = []
            texts: list[str] = []
            for m in memories:
                text = _prepare_memory_text(m.text)
                if text != m.text:
                    m = m.model_copy(update={"text": text})
                prepared.append(m)
                texts.append(m.text)
            vectors = embed_texts(texts)
            rows = [_to_memory_row(m, vector=vec) for m, vec in zip(prepared, vectors)]
            self._memories.add(rows)
            return [m.id for m in prepared]

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        tbl = self._memories
        df = tbl.search().where(f'id = "{_esc(memory_id)}"').limit(1).to_pandas()
        if df.empty:
            return None
        return _from_memory(df.iloc[0].to_dict())

    def get_memories_by_ids(self, memory_ids: list[str]) -> dict[str, MemoryRecord]:
        """Batch-fetch memories by id. Missing ids are omitted."""
        if not memory_ids:
            return {}
        unique = list(dict.fromkeys(memory_ids))
        out: dict[str, MemoryRecord] = {}
        # Chunk to keep IN clauses bounded.
        for i in range(0, len(unique), 200):
            chunk = unique[i:i + 200]
            df = self._memories.search().where(f"id IN {_in_list(chunk)}").to_pandas()
            for _, row in df.iterrows():
                rec = _from_memory(row.to_dict())
                out[rec.id] = rec
        return out

    def update_memory_status(self, memory_id: str, status: MemoryStatus,
                             superseded_by: Optional[str] = None,
                             contested_with: Optional[list[str]] = None) -> int:
        self._assert_writable()
        updates = {"status": status.value}
        if superseded_by is not None:
            updates["superseded_by"] = superseded_by
        if contested_with is not None:
            updates["contested_with"] = json.dumps(contested_with)
        with self._write_lock:
            self._memories.update(where=f'id = "{_esc(memory_id)}"', values=updates)
        return 1

    def touch_memory(self, memory_id: str) -> None:
        """Update last_recalled_at and increment recall_count."""
        self.touch_memories([memory_id])

    def touch_memories(self, memory_ids: list[str]) -> None:
        """Batch-touch memories (one read+write per id, serialized)."""
        self._assert_writable()
        if not memory_ids:
            return
        now = _now_iso()
        with self._write_lock:
            found = self.get_memories_by_ids(list(dict.fromkeys(memory_ids)))
            for mid, mem in found.items():
                self._memories.update(
                    where=f'id = "{_esc(mid)}"',
                    values={
                        "last_recalled_at": now,
                        "recall_count": mem.recall_count + 1,
                    },
                )

    def bump_recall_miss(self, memory_id: str) -> None:
        self.bump_recall_misses([memory_id])

    def bump_recall_misses(self, memory_ids: list[str]) -> None:
        """Increment recall_miss_count for memories retrieved but unused."""
        self._assert_writable()
        if not memory_ids:
            return
        with self._write_lock:
            found = self.get_memories_by_ids(list(dict.fromkeys(memory_ids)))
            for mid, mem in found.items():
                self._memories.update(
                    where=f'id = "{_esc(mid)}"',
                    values={"recall_miss_count": mem.recall_miss_count + 1},
                )

    def increment_reflect_cycles(self) -> None:
        """Bump reflect_cycles on every active memory (one SQL update)."""
        self._assert_writable()
        with self._write_lock:
            try:
                self._memories.update(
                    where='status = "active"',
                    values_sql={"reflect_cycles": "reflect_cycles + 1"},
                )
                return
            except TypeError:
                pass
            for mem in self.list_memories(status=MemoryStatus.active):
                self._memories.update(
                    where=f'id = "{_esc(mem.id)}"',
                    values={"reflect_cycles": mem.reflect_cycles + 1},
                )

    def update_memory_field(self, memory_id: str, **kwargs) -> None:
        """Generic field update on a memory by id."""
        self._assert_writable()
        updates = {}
        for k, v in kwargs.items():
            if k == "text" and isinstance(v, str):
                v = _prepare_memory_text(v)
            if isinstance(v, list):
                v = json.dumps(v)
            elif isinstance(v, datetime):
                v = v.isoformat()
            elif isinstance(v, Enum):
                v = v.value
            updates[k] = v
        with self._write_lock:
            self._memories.update(where=f'id = "{_esc(memory_id)}"', values=updates)

    def delete_memory(self, memory_id: str) -> None:
        """Soft-delete by setting status to archived. Never hard-deletes (D13)."""
        self.update_memory_status(memory_id, MemoryStatus.archived)

    def list_memories(
        self,
        network: Optional[Network | str] = None,
        status: Optional[MemoryStatus | str] = None,
        entity_ids: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryRecord]:
        """List memories with optional filters."""
        clauses = []
        if network is not None:
            n = network.value if isinstance(network, Network) else network
            clauses.append(f'network = "{_esc(n)}"')
        if status is not None:
            s = status.value if isinstance(status, MemoryStatus) else status
            clauses.append(f'status = "{_esc(s)}"')
        if entity_ids is not None:
            clauses.append(_entity_membership_clause(entity_ids))
        if since is not None:
            since_iso = since.isoformat() if isinstance(since, datetime) else since
            clauses.append(f'created_at > "{_esc(str(since_iso))}"')
        if until is not None:
            until_iso = until.isoformat() if isinstance(until, datetime) else until
            clauses.append(f'created_at <= "{_esc(str(until_iso))}"')

        where_clause = " AND ".join(clauses) if clauses else None
        q = self._memories.search()
        if where_clause:
            q = q.where(where_clause)
        if limit:
            q = q.limit(limit)

        df = q.to_pandas()
        records = []
        for _, row in df.iterrows():
            rec = _from_memory(row.to_dict())
            records.append(rec)
        return records

    def list_active_candidates(self, limit: int = 200) -> list[MemoryRecord]:
        """Active memories for hot-cache ranking, newest/most-recalled first.

        Tries an ordered query; falls back to a capped scan + Python sort so
        an unordered LIMIT cannot drop the hottest rows.
        """
        limit = max(1, int(limit))

        def _sort_head(frame):
            if frame is None or getattr(frame, "empty", True):
                return frame
            sort_cols = [c for c in ("recall_count", "created_at") if c in frame.columns]
            if sort_cols:
                frame = frame.sort_values(sort_cols, ascending=False)
            return frame.head(limit)

        try:
            q = self._memories.search().where('status = "active"')
            ordered = False
            try:
                from lancedb.query import ColumnOrdering
                q = q.order_by([
                    ColumnOrdering(column="recall_count", ascending=False),
                    ColumnOrdering(column="created_at", ascending=False),
                ])
                ordered = True
            except Exception:
                try:
                    q = q.order_by([
                        {"column": "recall_count", "ascending": False},
                        {"column": "created_at", "ascending": False},
                    ])
                    ordered = True
                except Exception:
                    pass
            if ordered:
                df = q.limit(limit).to_pandas()
            else:
                # Unordered scan: cap then sort so we still prefer hot rows.
                df = _sort_head(q.limit(max(limit * 20, 1000)).to_pandas())
        except Exception:
            df = self._memories.search().where('status = "active"').to_pandas()
            df = _sort_head(df)

        if df is None or df.empty:
            return []
        return [_from_memory(row.to_dict()) for _, row in df.iterrows()]

    def _build_search_filter(self,
                             entity_ids: Optional[list[str]] = None,
                             network: Optional[Network | str] = None,
                             status_filter: Optional[list[MemoryStatus | str]] = None,
                             exclude_status: Optional[list[MemoryStatus | str]] = None,
                             include_unscoped_world: bool = False,
                             ) -> str:
        clauses = []
        if entity_ids is not None:
            entity_clause = _entity_membership_clause(entity_ids)
            if include_unscoped_world:
                entity_clause = f'({entity_clause} OR (entity_ids = \'[]\' AND network = "{Network.world.value}"))'
            clauses.append(entity_clause)
        if network is not None:
            n = network.value if isinstance(network, Network) else network
            clauses.append(f'network = "{_esc(n)}"')
        if status_filter is not None:
            vals = [s.value if isinstance(s, MemoryStatus) else str(s) for s in status_filter]
            clauses.append(f"status IN {_in_list(vals)}")
        if exclude_status is not None:
            vals = [s.value if isinstance(s, MemoryStatus) else str(s) for s in exclude_status]
            clauses.append(f"status NOT IN {_in_list(vals)}")
        return " AND ".join(clauses) if clauses else None

    def vector_search(self, query: str, k: int = 25,
                      entity_filter: Optional[list[str]] = None,
                      network_filter: Optional[Network | str] = None,
                      status_filter: Optional[list[MemoryStatus | str]] = None,
                      exclude_status: Optional[list[MemoryStatus | str]] = None,
                      include_unscoped_world: bool = False,
                      ) -> list[tuple[MemoryRecord, float]]:
        """Vector similarity search with LanceDB pre-filters."""
        query_vec = embed_text(query)
        filter_str = self._build_search_filter(
            entity_ids=entity_filter,
            network=network_filter,
            status_filter=status_filter,
            exclude_status=exclude_status,
            include_unscoped_world=include_unscoped_world,
        )

        q = self._memories.search(query_vec).limit(max(1, k)).metric("L2")
        if filter_str:
            try:
                q = q.where(filter_str, prefilter=True)
            except TypeError:
                q = q.where(filter_str)

        results = q.to_pandas()
        records = []
        for _, row in results.iterrows():
            rec = _from_memory(row.to_dict())
            score = float(row.get("_distance", 0.0))
            sim_score = max(0.0, 1.0 - score / 4.0)
            records.append((rec, sim_score))
        return records

    def keyword_search(self, query: str, k: int = 25,
                       entity_filter: Optional[list[str]] = None,
                       network_filter: Optional[Network | str] = None,
                       status_filter: Optional[list[MemoryStatus | str]] = None,
                       exclude_status: Optional[list[MemoryStatus | str]] = None,
                       include_unscoped_world: bool = False,
                       ) -> list[tuple[MemoryRecord, float]]:
        """Full-text search over memories.text using LanceDB FTS."""
        filter_str = self._build_search_filter(
            entity_ids=entity_filter,
            network=network_filter,
            status_filter=status_filter,
            exclude_status=exclude_status,
            include_unscoped_world=include_unscoped_world,
        )
        k = max(1, k)

        try:
            q = self._memories.search(query, query_type="fts").limit(k)
            if filter_str:
                q = q.where(filter_str)
            results = q.to_pandas()
        except Exception as e:
            logger.debug("FTS search failed (%s), falling back to text search", e)
            fts_filter = f'text LIKE "%{_esc(query, escape_like=True)}%"'
            if filter_str:
                fts_filter = f'({filter_str}) AND {fts_filter}'
            q = self._memories.search().where(fts_filter).limit(k)
            results = q.to_pandas()

        records = []
        for _, row in results.iterrows():
            rec = _from_memory(row.to_dict())
            score = float(row.get("_relevance_score", row.get("_score", 1.0)))
            records.append((rec, score))
        return records

    # ── Entities ────────────────────────────────────────────────────────

    def add_entity(self, entity: EntityRecord) -> str:
        self._assert_writable()
        from . import config as cfg
        with self._write_lock:
            if cfg.MAX_ENTITIES > 0:
                count = _table_row_count(self._entities)
                if count >= cfg.MAX_ENTITIES:
                    raise RuntimeError(
                        f"Entity limit reached ({count} >= {cfg.MAX_ENTITIES})."
                    )
            self._entities.add([_to_entity_row(entity)])
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[EntityRecord]:
        df = self._entities.search().where(f'id = "{_esc(entity_id)}"').limit(1).to_pandas()
        if df.empty:
            return None
        return _from_entity(df.iloc[0])

    def get_entity_by_label(self, label: str) -> Optional[EntityRecord]:
        df = self._entities.search().where(f'label = "{_esc(label)}"').limit(1).to_pandas()
        if df.empty:
            return None
        return _from_entity(df.iloc[0])

    def list_entities(self) -> list[EntityRecord]:
        df = self._entities.search().to_pandas()
        return [_from_entity(row) for _, row in df.iterrows()]

    def update_entity(self, entity_id: str, **kwargs) -> None:
        self._assert_writable()
        updates = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = json.dumps(v)
            elif isinstance(v, Enum):
                v = v.value
            elif isinstance(v, datetime):
                v = v.isoformat()
            updates[k] = v
        with self._write_lock:
            self._entities.update(where=f'id = "{_esc(entity_id)}"', values=updates)

    def has_self_entity(self) -> bool:
        df = self._entities.search().where(f'type = "{EntityType.self_.value}"').limit(1).to_pandas()
        return not df.empty

    def count_memories_for_entity(self, entity_id: str) -> int:
        """Count memories whose entity_ids JSON contains this exact id."""
        try:
            df = self._memories.search().where(
                _entity_membership_clause([entity_id])
            ).to_pandas()
            return len(df)
        except Exception:
            return -1

    # ── Identity Links ──────────────────────────────────────────────────

    def add_identity_link(self, link: IdentityLink) -> IdentityLink:
        self._assert_writable()
        with self._write_lock:
            self._identities.add([_to_identity_row(link)])
        return link

    def delete_identity_link(self, platform: str, external_id: str) -> None:
        self._assert_writable()
        with self._write_lock:
            self._identities.delete(
                where=f'platform = "{_esc(platform)}" AND external_id = "{_esc(external_id)}"'
            )

    def resolve_identity(self, platform: str, external_id: str) -> Optional[IdentityLink]:
        df = self._identities.search().where(
            f'platform = "{_esc(platform)}" AND external_id = "{_esc(external_id)}"'
        ).limit(1).to_pandas()
        if df.empty:
            return None
        row = df.iloc[0]
        return IdentityLink(
            platform=row["platform"],
            external_id=row["external_id"],
            entity_id=row["entity_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            created_by=LinkMethod(row["created_by"]),
        )

    def get_identity_links_for_entity(self, entity_id: str) -> list[IdentityLink]:
        df = self._identities.search().where(f'entity_id = "{_esc(entity_id)}"').to_pandas()
        return [
            IdentityLink(
                platform=row["platform"],
                external_id=row["external_id"],
                entity_id=row["entity_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                created_by=LinkMethod(row["created_by"]),
            )
            for _, row in df.iterrows()
        ]

    def list_identity_links(self) -> list[IdentityLink]:
        df = self._identities.search().to_pandas()
        return [
            IdentityLink(
                platform=row["platform"],
                external_id=row["external_id"],
                entity_id=row["entity_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                created_by=LinkMethod(row["created_by"]),
            )
            for _, row in df.iterrows()
        ]

    def repoint_identity_links(self, old_entity_id: str, new_entity_id: str) -> int:
        self._assert_writable()
        with self._write_lock:
            self._identities.update(
                where=f'entity_id = "{_esc(old_entity_id)}"',
                values={"entity_id": new_entity_id},
            )
        return 1

    # ── Edges ───────────────────────────────────────────────────────────

    def add_edge(self, edge: EdgeRecord) -> None:
        self._assert_writable()
        with self._write_lock:
            if cfg.MAX_EDGES > 0 and _table_row_count(self._edges) >= cfg.MAX_EDGES:
                raise RuntimeError(f"Edge limit reached (>= {cfg.MAX_EDGES}).")
            self._edges.add([_to_edge_row(edge)])

    def add_edges(self, edges: list[EdgeRecord]) -> None:
        self._assert_writable()
        if not edges:
            return
        with self._write_lock:
            if cfg.MAX_EDGES > 0:
                count = _table_row_count(self._edges)
                if count + len(edges) > cfg.MAX_EDGES:
                    raise RuntimeError(
                        f"Edge limit reached ({count} + {len(edges)} > {cfg.MAX_EDGES})."
                    )
            self._edges.add([_to_edge_row(e) for e in edges])

    def get_edges_for_memory(self, memory_id: str) -> list[EdgeRecord]:
        df = self._edges.search().where(
            f'from_id = "{_esc(memory_id)}" OR to_id = "{_esc(memory_id)}"'
        ).to_pandas()
        return [_from_edge(row) for _, row in df.iterrows()]

    def list_edges(self) -> list[EdgeRecord]:
        """All edges — used by export so we don't N+1 per memory."""
        df = self._edges.search().to_pandas()
        return [_from_edge(row) for _, row in df.iterrows()]

    def neighbors(self, memory_id: str, min_weight: float = 0.4) -> list[tuple[str, EdgeRecord]]:
        return self.neighbors_for_ids([memory_id], min_weight=min_weight).get(memory_id, [])

    def neighbors_for_ids(
        self, memory_ids: list[str], min_weight: float = 0.4,
    ) -> dict[str, list[tuple[str, EdgeRecord]]]:
        """Batch-fetch neighbors for many memories in one (chunked) query."""
        if not memory_ids:
            return {}
        unique = list(dict.fromkeys(memory_ids))
        out: dict[str, list[tuple[str, EdgeRecord]]] = {mid: [] for mid in unique}
        id_set = set(unique)
        weight = float(min_weight)
        for i in range(0, len(unique), 200):
            chunk = unique[i:i + 200]
            clause = _in_list(chunk)
            df = self._edges.search().where(
                f'(from_id IN {clause} OR to_id IN {clause}) AND weight >= {weight}'
            ).to_pandas()
            for _, row in df.iterrows():
                edge = _from_edge(row)
                if edge.from_id in id_set:
                    out[edge.from_id].append((edge.to_id, edge))
                if edge.to_id in id_set and edge.to_id != edge.from_id:
                    out[edge.to_id].append((edge.from_id, edge))
        return out

    def update_edge_weight(self, from_id: str, to_id: str, kind: str, weight: float) -> None:
        self._assert_writable()
        with self._write_lock:
            self._edges.update(
                where=f'from_id = "{_esc(from_id)}" AND to_id = "{_esc(to_id)}" AND kind = "{_esc(kind)}"',
                values={"weight": float(weight)},
            )

    def re_memory_entity_ids(self, old_entity_id: str, new_entity_id: str) -> int:
        self._assert_writable()
        count = 0
        df = self._memories.search().where(_entity_membership_clause([old_entity_id])).to_pandas()
        with self._write_lock:
            for _, row in df.iterrows():
                old_ids = _parse_json_field(row["entity_ids"], "[]")
                if old_entity_id in old_ids:
                    new_ids = [new_entity_id if e == old_entity_id else e for e in old_ids]
                    new_ids = list(dict.fromkeys(new_ids))
                    self._memories.update(
                        where=f'id = "{_esc(row["id"])}"',
                        values={"entity_ids": json.dumps(new_ids)},
                    )
                    count += 1
        return count

    def re_memory_speaker(self, old_entity_id: str, new_entity_id: str) -> int:
        self._assert_writable()
        with self._write_lock:
            self._memories.update(
                where=f'speaker_entity_id = "{_esc(old_entity_id)}"',
                values={"speaker_entity_id": new_entity_id},
            )
        return 1

    def get_memory_by_canonical_key(self, canonical_key: str) -> Optional[MemoryRecord]:
        df = self._memories.search().where(
            f'canonical_key = "{_esc(canonical_key)}"'
        ).limit(1).to_pandas()
        if df.empty:
            return None
        return _from_memory(df.iloc[0].to_dict())

    # ── Hot Cache ───────────────────────────────────────────────────────

    def get_hot_cache(self, entity_id: str) -> Optional[HotCache]:
        df = self._hot_cache.search().where(f'entity_id = "{_esc(entity_id)}"').limit(1).to_pandas()
        if df.empty:
            return None
        row = df.iloc[0]
        return HotCache(
            entity_id=row["entity_id"],
            memory_ids=_parse_json_field(row["memory_ids"], "[]"),
            rendered=row["rendered"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def save_hot_cache(self, cache: HotCache) -> None:
        self._assert_writable()
        with self._write_lock:
            existing = self.get_hot_cache(cache.entity_id)
            if existing:
                self._hot_cache.update(
                    where=f'entity_id = "{_esc(cache.entity_id)}"',
                    values=_to_hot_cache_row(cache),
                )
            else:
                self._hot_cache.add([_to_hot_cache_row(cache)])

    # ── Flags ───────────────────────────────────────────────────────────

    def add_flag(self, flag: FlagRecord) -> str:
        self._assert_writable()
        from . import config as cfg
        with self._write_lock:
            if cfg.MAX_FLAGS_PER_SESSION > 0:
                count = self.unprocessed_flag_count(flag.session_id)
                if count >= cfg.MAX_FLAGS_PER_SESSION:
                    raise RuntimeError(
                        f"Session flag limit reached ({count} >= {cfg.MAX_FLAGS_PER_SESSION}). "
                        f"Run gm_flush or gm_reflect_now to process pending flags first."
                    )
            self._flags.add([_to_flag_row(flag)])
        return flag.id

    def unprocessed_flags(self, session_id: Optional[str] = None) -> list[FlagRecord]:
        where = 'processed = false'
        if session_id:
            where += f' AND session_id = "{_esc(session_id)}"'
        df = self._flags.search().where(where).to_pandas()
        return [_from_flag(row) for _, row in df.iterrows()]

    def unprocessed_flag_count(self, session_id: Optional[str] = None) -> int:
        where = 'processed = false'
        if session_id:
            where += f' AND session_id = "{_esc(session_id)}"'
        try:
            return self._flags.count_rows(where)
        except Exception:
            return len(self.unprocessed_flags(session_id))

    def mark_flags_processed(self, flag_ids: list[str]) -> None:
        self._assert_writable()
        with self._write_lock:
            for fid in flag_ids:
                self._flags.update(where=f'id = "{_esc(fid)}"', values={"processed": True})

    def increment_flag_attempts(self, flag_ids: list[str]) -> None:
        """Bump attempt_count on flags whose extraction batch just failed."""
        self._assert_writable()
        with self._write_lock:
            for fid in flag_ids:
                try:
                    self._flags.update(
                        where=f'id = "{_esc(fid)}"',
                        values_sql={"attempt_count": "attempt_count + 1"},
                    )
                    continue
                except TypeError:
                    pass
                # Fallback: read-modify-write (LanceDB builds without values_sql).
                df = self._flags.search().where(f'id = "{_esc(fid)}"').limit(1).to_pandas()
                if df.empty:
                    continue
                current = int(df.iloc[0].get("attempt_count", 0) or 0)
                self._flags.update(
                    where=f'id = "{_esc(fid)}"',
                    values={"attempt_count": current + 1},
                )

    def consume_flags(self, session_id: str) -> list[FlagRecord]:
        self._assert_writable()
        flags = [f for f in self.unprocessed_flags(session_id) if f.session_id == session_id]
        if flags:
            self.mark_flags_processed([f.id for f in flags])
        return flags

    # ── Promotion Queue ─────────────────────────────────────────────────

    def add_to_promotion_queue(self, memory_id: str, target_systems: list[str] = None) -> None:
        self._assert_writable()
        record = PromotionQueueRecord(
            memory_id=memory_id,
            enqueued_at=datetime.now(timezone.utc),
            target_systems=target_systems or ["wiki", "obsidian"]
        )
        row = {
            "memory_id": record.memory_id,
            "enqueued_at": record.enqueued_at.isoformat(),
            "target_systems": json.dumps(record.target_systems)
        }
        with self._write_lock:
            self._promotion_queue.add([row])

    def list_promotion_queue(self) -> list[PromotionQueueRecord]:
        df = self._promotion_queue.search().to_pandas()
        records = []
        for _, row in df.iterrows():
            record = PromotionQueueRecord(
                memory_id=row["memory_id"],
                enqueued_at=datetime.fromisoformat(row["enqueued_at"]),
                target_systems=_parse_json_field(row["target_systems"], "[]"),
            )
            records.append(record)
        return records

    def remove_from_promotion_queue(self, memory_id: str) -> None:
        self._assert_writable()
        with self._write_lock:
            self._promotion_queue.delete(where=f'memory_id = "{_esc(memory_id)}"')

    # ── Session summaries ───────────────────────────────────────────────

    def get_session_summary(self, session_id: str) -> Optional[dict]:
        if self._session_summaries is None:
            return None
        try:
            df = self._session_summaries.search().where(
                f'id = "{_esc(session_id)}"'
            ).limit(1).to_pandas()
            if df.empty:
                return None
            row = df.iloc[0]
            return {
                "id": row["id"],
                "text": row["text"],
                "message_count": int(row["message_count"]),
                "last_updated": row["last_updated"],
            }
        except Exception as e:
            logger.debug("get_session_summary failed for %s: %s", session_id, e)
            return None

    def upsert_session_summary(self, session_id: str, text: str,
                               message_count: int) -> None:
        self._assert_writable()
        if self._session_summaries is None:
            return
        now = _now_iso()
        text = redact_secrets(text) if text else text
        with self._write_lock:
            existing = self.get_session_summary(session_id)
            if existing:
                self._session_summaries.update(
                    where=f'id = "{_esc(session_id)}"',
                    values={
                        "text": text,
                        "message_count": message_count,
                        "last_updated": now,
                    },
                )
            else:
                self._session_summaries.add([{
                    "id": session_id,
                    "text": text,
                    "message_count": message_count,
                    "last_updated": now,
                }])

    def list_session_summaries(self, limit: int = 50) -> list[dict]:
        if self._session_summaries is None:
            return []
        try:
            df = self._session_summaries.search().to_pandas()
            if df.empty:
                return []
            df = df.sort_values("last_updated", ascending=False).head(max(1, limit))
            return [
                {
                    "id": row["id"],
                    "text": row["text"],
                    "message_count": int(row["message_count"]),
                    "last_updated": row["last_updated"],
                }
                for _, row in df.iterrows()
            ]
        except Exception as e:
            logger.debug("list_session_summaries failed: %s", e)
            return []

    # ── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return DB statistics without materializing embedding vectors."""
        mem_count = _table_row_count(self._memories)
        ent_count = _table_row_count(self._entities)
        flag_count = self.unprocessed_flag_count()
        edge_count = _table_row_count(self._edges)

        network_counts = {n.value: 0 for n in Network}
        status_counts = {s.value: 0 for s in MemoryStatus}
        type_counts = {t.value: 0 for t in EntityType}

        try:
            inner = getattr(self._memories, "to_lance", None)
            if callable(inner):
                table = inner().to_table(columns=["network", "status"])
                nets = table.column("network").to_pylist()
                stats_col = table.column("status").to_pylist()
                for n, s in zip(nets, stats_col):
                    if n in network_counts:
                        network_counts[n] += 1
                    if s in status_counts:
                        status_counts[s] += 1
            else:
                raise AttributeError("no to_lance")
        except Exception:
            mem_df = self._memories.search().to_pandas()
            for n in Network:
                network_counts[n.value] = int((mem_df.network == n.value).sum()) if not mem_df.empty else 0
            for s in MemoryStatus:
                status_counts[s.value] = int((mem_df.status == s.value).sum()) if not mem_df.empty else 0

        try:
            ent_df = self._entities.search().to_pandas()
            for t in EntityType:
                type_counts[t.value] = int((ent_df.type == t.value).sum()) if not ent_df.empty else 0
            ent_count = len(ent_df)
        except Exception:
            pass

        return {
            "total_memories": mem_count,
            "memories_per_network": network_counts,
            "memories_per_status": status_counts,
            "total_entities": ent_count,
            "entities_per_type": type_counts,
            "total_edges": edge_count,
            "unprocessed_flags": flag_count,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        """Close all LanceDB table handles to release OS-level file descriptors."""
        for tbl in (self._memories, self._entities, self._identities,
                    self._edges, self._hot_cache, self._flags,
                    self._promotion_queue, self._session_summaries):
            if tbl is None:
                continue
            try:
                inner = getattr(tbl, "_table", tbl)
                if hasattr(inner, "close") and getattr(inner, "is_open", True):
                    inner.close()
            except Exception as e:
                logger.debug("close() on table failed (non-fatal): %s", e)
        self._memories = None
        self._entities = None
        self._identities = None
        self._edges = None
        self._hot_cache = None
        self._flags = None
        self._promotion_queue = None
        self._session_summaries = None
        self.db = None


# ── Helper constructors ──────────────────────────────────────────────────

def _from_entity(row) -> EntityRecord:
    merged = row.get("merged_into")
    if merged is None or (isinstance(merged, float) and np.isnan(merged)):
        merged = None
    else:
        merged = str(merged)
    card_raw = row.get("card", "{}")
    return EntityRecord(
        id=row["id"],
        type=EntityType(row["type"]),
        label=row["label"],
        card=_parse_json_field(card_raw, "{}"),
        status_line=row.get("status_line", "") or "",
        created_at=datetime.fromisoformat(row["created_at"]),
        merged_into=merged,
    )


def _from_edge(row) -> EdgeRecord:
    return EdgeRecord(
        from_id=row["from_id"],
        to_id=row["to_id"],
        kind=EdgeKind(row["kind"]),
        weight=float(row["weight"]),
    )


def _from_flag(row) -> FlagRecord:
    return FlagRecord(
        id=row["id"],
        session_id=row["session_id"],
        platform=row["platform"],
        speaker_external_id=row["speaker_external_id"],
        turn_text=row["turn_text"],
        flag_reason=row["flag_reason"],
        processed=bool(row["processed"]),
        attempt_count=int(row.get("attempt_count", 0) or 0),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
