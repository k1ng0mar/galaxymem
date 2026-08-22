"""LanceDB storage layer — typed CRUD over the six tables with zero business logic.

All business logic lives in retain.py, recall.py, reflect.py, etc.
This module handles: table creation, embedding, CRUD, vector/keyword search,
edge operations, hot cache, flag queue, and temporal versioning.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import lancedb
import numpy as np
from lancedb.embeddings import EmbeddingFunctionConfig, EmbeddingFunctionRegistry
from lancedb.table import Table as LanceTable
from lancedb.query import LanceQueryBuilder

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

logger = logging.getLogger(__name__)


def _table_row_count(table: LanceTable) -> int:
    """Return the number of rows in a LanceDB table across supported versions.

    LanceDB 0.34 exposes count_rows(); older versions have no working count()
    on the sync Table, so fall back to a full scan.
    """
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
    return len(table.search().to_pandas())


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
        # Double-check after acquiring lock — another thread may have init'd
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
                """Closure that holds a strong reference to the model."""
                def embed(texts):
                    return list(model_ref.embed(texts))
                return embed

            _embed_fn = _make_embed_fn(_model)
        elif cfg.EMBEDDING_BACKEND == "api" and cfg.EMBEDDING_API_URL:
            # External embedding API (e.g. OpenAI-compatible)
            import requests

            url = cfg.EMBEDDING_API_URL.rstrip("/")

            def _api_embed(texts):
                resp = requests.post(
                    f"{url}/embeddings",
                    json={"input": texts, "model": cfg.EMBEDDING_MODEL},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    return [d["embedding"] for d in data]
                return [d["embedding"] for d in data["data"]]

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
    return [list(v) for v in _get_embedding_function()(texts)]


# ── LanceDB Pydantic models (table schemas) ────────────────────────────
# Defined in schema.py to keep this storage module focused on CRUD/search.
from .schema import (  # noqa: F401
    MemoriesTable,
    EntitiesTable,
    IdentityLinksTable,
    EdgesTable,
    HotCacheTable,
    FlagsTable,
    PromotionQueueTable,
    SessionSummariesTable,
    _esc,
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
    parts = [f'entity_ids LIKE \'%"{_esc(e)}"%\'' for e in entity_ids]
    return f"({' OR '.join(parts)})"


from .utils import ulid as _ulid


def _from_memory(t) -> MemoryRecord:
    """Convert a MemoriesTable row (or dict-like) to MemoryRecord."""
    def _safe_json(val, default="[]"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return json.loads(default)
        return json.loads(val)

    def _safe_str(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return str(val)

    return MemoryRecord(
        id=str(t["id"]) if isinstance(t, dict) else t.id,
        text=str(t["text"]) if isinstance(t, dict) else t.text,
        network=Network(t["network"] if isinstance(t, dict) else t.network),
        entity_ids=_safe_json(t["entity_ids"] if isinstance(t, dict) else t.entity_ids),
        source_memory_ids=_safe_json(t["source_memory_ids"] if isinstance(t, dict) else t.source_memory_ids),
        status=MemoryStatus(t["status"] if isinstance(t, dict) else t.status),
        superseded_by=_safe_str(t["superseded_by"] if isinstance(t, dict) else t.superseded_by),
        contested_with=_safe_json(t["contested_with"] if isinstance(t, dict) else t.contested_with, "[]"),
        created_at=datetime.fromisoformat(str(t["created_at"] if isinstance(t, dict) else t.created_at)),
        last_recalled_at=datetime.fromisoformat(str(t["last_recalled_at"])) if _safe_str(t["last_recalled_at"] if isinstance(t, dict) else t.last_recalled_at) else None,
        recall_count=int(t["recall_count"] if isinstance(t, dict) else t.recall_count),
        reflect_cycles=int(t["reflect_cycles"] if isinstance(t, dict) else getattr(t, 'reflect_cycles', 0)),
        source_session_id=_safe_str(t["source_session_id"] if isinstance(t, dict) else t.source_session_id),
        source_platform=_safe_str(t["source_platform"] if isinstance(t, dict) else t.source_platform),
        speaker_entity_id=_safe_str(t["speaker_entity_id"] if isinstance(t, dict) else t.speaker_entity_id),
        promoted_to=_safe_str(t["promoted_to"] if isinstance(t, dict) else t.promoted_to),
        flagged_source=_safe_str(t["flagged_source"] if isinstance(t, dict) else getattr(t, 'flagged_source', None)),
        canonical_key=_safe_str(t.get("canonical_key") if isinstance(t, dict) else getattr(t, 'canonical_key', None)),
        proof_count=int(t.get("proof_count", 0) or 0) if isinstance(t, dict) else int(getattr(t, 'proof_count', 0) or 0),
        history_json=_safe_str(t.get("history_json") if isinstance(t, dict) else getattr(t, 'history_json', None)),
    )


def _to_memory_row(m: MemoryRecord) -> dict:
    return {
        "id": m.id,
        "text": m.text,
        "vector": embed_text(m.text),
        "network": m.network.value,
        "entity_ids": json.dumps(m.entity_ids),
        "source_memory_ids": json.dumps(m.source_memory_ids),
        "status": m.status.value,
        "superseded_by": m.superseded_by,
        "contested_with": json.dumps(m.contested_with),
        "created_at": m.created_at.isoformat() if isinstance(m.created_at, datetime) else m.created_at,
        "last_recalled_at": m.last_recalled_at.isoformat() if m.last_recalled_at else None,
        "recall_count": m.recall_count,
        "reflect_cycles": m.reflect_cycles,
        "source_session_id": m.source_session_id,
        "source_platform": m.source_platform,
        "speaker_entity_id": m.speaker_entity_id,
        "promoted_to": m.promoted_to,
        "flagged_source": m.flagged_source,
        "canonical_key": m.canonical_key,
        "proof_count": m.proof_count,
        "history_json": m.history_json,
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
        "turn_text": f.turn_text,
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
                    # Create vector index if not already present
                    self._ensure_vector_index()
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
            # Close any tables we managed to open before the failure,
            # then reset attrs so the Store isn't left in a half-open state.
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
        """Add columns introduced after the initial schema to existing tables.

        LanceDB create_table(exist_ok=True) does NOT alter an existing
        table's schema, so new nullable/defaulted fields need an explicit
        add_columns pass. Safe to run on every open (idempotent).
        """
        try:
            existing = {f.name for f in tbl.schema}
            import pyarrow as pa

            if "proof_count" not in existing:
                tbl.add_columns({"proof_count": pa.int64()})
                logger.info("Migrated memories table: added column proof_count")
            if "history_json" not in existing:
                tbl.add_columns({"history_json": pa.string()})
                logger.info("Migrated memories table: added column history_json")
        except Exception as e:
            logger.warning("memories schema migration skipped: %s", e)

    def _ensure_vector_index(self):
        """Create IVF-PQ vector index on memories if not present."""
        try:
            existing_indices = self._memories.list_indices()
            if not any(
                (idx.get("name") if isinstance(idx, dict) else getattr(idx, "name", None))
                == "vector_idx"
                for idx in existing_indices
            ):
                try:
                    from lancedb.index import IvfPq
                    self._memories.create_index(
                        "vector",
                        config=IvfPq(distance_type="l2"),
                    )
                except (ImportError, TypeError):
                    # Fallback for older LanceDB versions
                    self._memories.create_index(
                        vector_column_name="vector",
                        index_type="IVF_PQ",
                        metric="L2",
                        num_partitions=32,
                        num_sub_vectors=8,
                    )
                logger.info("Created vector index on memories.vector")
        except Exception as e:
            logger.warning("Vector index creation skipped: %s", e)

    def as_of(self, timestamp: datetime) -> Store:
        """Return a read-only Store handle to a historical LanceDB version.

        Uses LanceDB native version pinning: finds the latest memories-table
        version at or before `timestamp` and checks out a fresh table handle
        pinned to it. Only memory queries work in temporal mode; mutations
        raise RuntimeError.

        Raises:
            ValueError: if no table version exists at or before `timestamp`.
            RuntimeError: if this LanceDB build does not expose versioning.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Fresh handle so the pinned checkout never mutates the live table.
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
                # LanceDB reports naive LOCAL timestamps; make them aware
                vts = vts.astimezone()
            if vts <= timestamp and (target is None or v["version"] > target):
                target = v["version"]
        if target is None:
            _close_tbl(tbl)
            raise ValueError(f"No memory-table version exists at or before {timestamp.isoformat()}")

        tbl.checkout(target)

        historical = Store.__new__(Store)
        historical.db_path = self.db_path
        historical._read_only = True
        historical.db = self.db  # share the DB connection (not the table handle)
        historical._memories = tbl
        historical._entities = None
        historical._identities = None
        historical._edges = None
        historical._hot_cache = None
        historical._flags = None
        historical._promotion_queue = None
        return historical

    def _assert_writable(self) -> None:
        if getattr(self, "_read_only", False):
            raise RuntimeError("This Store is a read-only historical handle (as_of); writes are forbidden")

    # ── Memories ────────────────────────────────────────────────────────

    def _assert_under_limits(self) -> None:
        """Raise RuntimeError if the DB size limits (config) are exceeded."""
        from . import config as cfg
        if cfg.MAX_MEMORIES > 0:
            count = _table_row_count(self._memories)
            if count >= cfg.MAX_MEMORIES:
                raise RuntimeError(
                    f"Memory limit reached ({count} >= {cfg.MAX_MEMORIES}). "
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
        self._assert_under_limits()
        row = _to_memory_row(memory)
        self._memories.add([row])
        return memory.id

    def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Batch insert memories. Returns their ids."""
        self._assert_writable()
        self._assert_under_limits()
        if not memories:
            return []
        data = embed_texts([m.text for m in memories])
        rows = []
        for m, vec in zip(memories, data):
            row = _to_memory_row(m)
            row["vector"] = vec
            rows.append(row)
        self._memories.add(rows)
        return [m.id for m in memories]

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        tbl = self._memories
        df = tbl.search().where(f'id = "{_esc(memory_id)}"').limit(1).to_pandas()
        if df.empty:
            return None
        return _from_memory(df.iloc[0].to_dict())

    def update_memory_status(self, memory_id: str, status: MemoryStatus,
                             superseded_by: Optional[str] = None,
                             contested_with: Optional[list[str]] = None) -> int:
        self._assert_writable()
        updates = {"status": status.value}
        if superseded_by is not None:
            updates["superseded_by"] = superseded_by
        if contested_with is not None:
            updates["contested_with"] = json.dumps(contested_with)
        self._memories.update(where=f'id = "{_esc(memory_id)}"', values=updates)
        return 1

    def touch_memory(self, memory_id: str) -> None:
        """Update last_recalled_at and increment recall_count."""
        self._assert_writable()
        mem = self.get_memory(memory_id)
        if not mem:
            return
        self._memories.update(
            where=f'id = "{_esc(memory_id)}"',
                values={
                    "last_recalled_at": _now_iso(),
                "recall_count": mem.recall_count + 1,
            },
        )

    def update_memory_field(self, memory_id: str, **kwargs) -> None:
        """Generic field update on a memory by id."""
        self._assert_writable()
        updates = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                v = json.dumps(v)
            elif isinstance(v, datetime):
                v = v.isoformat()
            elif isinstance(v, Enum):
                v = v.value
            updates[k] = v
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
        limit: Optional[int] = None,
    ) -> list[MemoryRecord]:
        """List memories with optional filters.

        Args:
            network: Filter by network (world/experience/opinion/observation)
            status: Filter by status (active/superseded/contested/demoted)
            entity_ids: Filter to memories containing any of these entity IDs
            since: Filter to memories created after this datetime
            limit: Maximum number of memories to return

        Returns:
            List of MemoryRecord objects matching the filters.
        """
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
            clauses.append(f'created_at > "{since_iso}"')

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
                # D8: recall scope = requested entities + unscoped world facts
                entity_clause = f'({entity_clause} OR (entity_ids = \'[]\' AND network = "{Network.world.value}"))'
            clauses.append(entity_clause)
        if network is not None:
            n = network.value if isinstance(network, Network) else network
            clauses.append(f'network = "{_esc(n)}"')
        if status_filter is not None:
            vals = [s.value if isinstance(s, MemoryStatus) else s for s in status_filter]
            quoted = ",".join(f'"{v}"' for v in vals)
            clauses.append(f"status IN ({quoted})")
        if exclude_status is not None:
            vals = [s.value if isinstance(s, MemoryStatus) else s for s in exclude_status]
            quoted = ",".join(f'"{v}"' for v in vals)
            clauses.append(f"status NOT IN ({quoted})")
        return " AND ".join(clauses) if clauses else None

    def vector_search(self, query: str, k: int = 25,
                      entity_filter: Optional[list[str]] = None,
                      network_filter: Optional[Network | str] = None,
                      status_filter: Optional[list[MemoryStatus | str]] = None,
                      exclude_status: Optional[list[MemoryStatus | str]] = None,
                      include_unscoped_world: bool = False,
                      ) -> list[tuple[MemoryRecord, float]]:
        """Vector similarity search with LanceDB pre-filters.

        Returns list of (MemoryRecord, score) tuples.
        """
        query_vec = embed_text(query)
        filter_str = self._build_search_filter(
            entity_ids=entity_filter,
            network=network_filter,
            status_filter=status_filter,
            exclude_status=exclude_status,
            include_unscoped_world=include_unscoped_world,
        )

        q = self._memories.search(query_vec).limit(k).metric("L2")
        if filter_str:
            # prefilter=True: filter BEFORE the vector index scan (D1/D8 —
            # a hard filter, not post-filtering of the top-k).
            try:
                q = q.where(filter_str, prefilter=True)
            except TypeError:
                q = q.where(filter_str)

        results = q.to_pandas()
        records = []
        for _, row in results.iterrows():
            rec = _from_memory(row.to_dict())
            # LanceDB returns distance; convert to similarity score
            score = float(row.get("_distance", 0.0))
            # Normalize: lower distance = higher score
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
        """Full-text search over memories.text using LanceDB FTS.

        Returns list of (MemoryRecord, score) tuples.
        """
        filter_str = self._build_search_filter(
            entity_ids=entity_filter,
            network=network_filter,
            status_filter=status_filter,
            exclude_status=exclude_status,
            include_unscoped_world=include_unscoped_world,
        )

        # LanceDB FTS via full_text_search property
        try:
            q = self._memories.search(query, query_type="fts").limit(k)
            if filter_str:
                q = q.where(filter_str)
            results = q.to_pandas()
        except Exception as e:
            logger.debug("FTS search failed (%s), falling back to text search", e)
            # Fallback: text filter
            # Use escape_like=True so user queries can't inject % or _ wildcards
            fts_filter = f'text LIKE "%{_esc(query, escape_like=True)}%"'
            if filter_str:
                fts_filter = f'({filter_str}) AND text LIKE "%{_esc(query, escape_like=True)}%"'

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
        from . import config as cfg
        if cfg.MAX_ENTITIES > 0:
            try:
                count = self._entities.count()
            except Exception:
                count = len(self._entities.search().limit(cfg.MAX_ENTITIES + 1).to_pandas())
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
        updates = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = json.dumps(v)
            elif isinstance(v, Enum):
                v = v.value
            elif isinstance(v, datetime):
                v = v.isoformat()
            updates[k] = v
        self._entities.update(where=f'id = "{_esc(entity_id)}"', values=updates)

    def has_self_entity(self) -> bool:
        df = self._entities.search().where(f'type = "{EntityType.self_.value}"').limit(1).to_pandas()
        return not df.empty

    # ── Identity Links ──────────────────────────────────────────────────

    def add_identity_link(self, link: IdentityLink) -> IdentityLink:
        self._identities.add([_to_identity_row(link)])
        return link

    def delete_identity_link(self, platform: str, external_id: str) -> None:
        """Delete an identity link. Returns nothing; raises on failure."""
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
        """List all identity links."""
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
        """Move all identity links from old entity to new entity."""
        self._identities.update(
            where=f'entity_id = "{_esc(old_entity_id)}"',
            values={"entity_id": new_entity_id},
        )
        return 1

    # ── Edges ───────────────────────────────────────────────────────────

    def add_edge(self, edge: EdgeRecord) -> None:
        self._edges.add([_to_edge_row(edge)])

    def add_edges(self, edges: list[EdgeRecord]) -> None:
        if not edges:
            return
        self._edges.add([_to_edge_row(e) for e in edges])

    def get_edges_for_memory(self, memory_id: str) -> list[EdgeRecord]:
        df = self._edges.search().where(
            f'from_id = "{_esc(memory_id)}" OR to_id = "{_esc(memory_id)}"'
        ).to_pandas()
        return [_from_edge(row) for _, row in df.iterrows()]

    def neighbors(self, memory_id: str, min_weight: float = 0.4) -> list[tuple[str, EdgeRecord]]:
        """Get neighbor memory IDs and edges for a memory, filtered by min weight.

        Returns list of (neighbor_id, edge) — both directions.
        """
        df = self._edges.search().where(
            f'(from_id = "{_esc(memory_id)}" OR to_id = "{_esc(memory_id)}") AND weight >= {min_weight}'
        ).to_pandas()
        results = []
        for _, row in df.iterrows():
            edge = _from_edge(row)
            if edge.from_id == memory_id:
                results.append((edge.to_id, edge))
            else:
                results.append((edge.from_id, edge))
        return results

    def update_edge_weight(self, from_id: str, to_id: str, kind: str, weight: float) -> None:
        self._edges.update(
            where=f'from_id = "{_esc(from_id)}" AND to_id = "{_esc(to_id)}" AND kind = "{_esc(kind)}"',
            values={"weight": weight},
        )

    def re_memory_entity_ids(self, old_entity_id: str, new_entity_id: str) -> int:
        """Repoint entity_ids in memories from old to new entity."""
        self._assert_writable()
        count = 0
        df = self._memories.search().where(_entity_membership_clause([old_entity_id])).to_pandas()
        for _, row in df.iterrows():
            old_ids = json.loads(row["entity_ids"])
            if old_entity_id in old_ids:
                new_ids = [new_entity_id if e == old_entity_id else e for e in old_ids]
                # De-dupe in case the memory was already scoped to both
                new_ids = list(dict.fromkeys(new_ids))
                self._memories.update(
                    where=f'id = "{_esc(row["id"])}"',
                    values={"entity_ids": json.dumps(new_ids)},
                )
                count += 1
        return count

    def re_memory_speaker(self, old_entity_id: str, new_entity_id: str) -> int:
        """Repoint speaker_entity_id in memories from old to new entity."""
        self._assert_writable()
        self._memories.update(
            where=f'speaker_entity_id = "{_esc(old_entity_id)}"',
            values={"speaker_entity_id": new_entity_id},
        )
        return 1

    def get_memory_by_canonical_key(self, canonical_key: str) -> Optional[MemoryRecord]:
        """Fetch the memory with this exact canonical_key, if any. Used for
        fact canonization: same triple extracted from different phrasings maps
        to one memory record (source_ids get merged, text gets the latest
        phrasing).

        Args:
            canonical_key: The unique fact key "subject|predicate|object".

        Returns:
            The matching MemoryRecord, or None if not found.
        """
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
            memory_ids=json.loads(row["memory_ids"]),
            rendered=row["rendered"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def save_hot_cache(self, cache: HotCache) -> None:
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
        from . import config as cfg
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
        return len(self.unprocessed_flags(session_id))

    def mark_flags_processed(self, flag_ids: list[str]) -> None:
        for fid in flag_ids:
            self._flags.update(where=f'id = "{_esc(fid)}"', values={"processed": True})

    def increment_flag_attempts(self, flag_ids: list[str]) -> None:
        """Bump attempt_count on flags whose extraction batch just failed."""
        for fid in flag_ids:
            self._flags.update(
                where=f'id = "{_esc(fid)}"',
                values_sql={"attempt_count": "attempt_count + 1"},
            )

    def consume_flags(self, session_id: str) -> list[FlagRecord]:
        """Get and mark all unprocessed flags for a session."""
        flags = [f for f in self.unprocessed_flags(session_id) if f.session_id == session_id]
        if flags:
            self.mark_flags_processed([f.id for f in flags])
        return flags

    # ── Promotion Queue ─────────────────────────────────────────────────

    def add_to_promotion_queue(self, memory_id: str, target_systems: list[str] = None) -> None:
        """Add a memory to the promotion queue."""
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
        self._promotion_queue.add([row])

    def list_promotion_queue(self) -> list[PromotionQueueRecord]:
        """List all items in the promotion queue."""
        df = self._promotion_queue.search().to_pandas()
        records = []
        for _, row in df.iterrows():
            record = PromotionQueueRecord(
                memory_id=row["memory_id"],
                enqueued_at=datetime.fromisoformat(row["enqueued_at"]),
                target_systems=json.loads(row["target_systems"])
            )
            records.append(record)
        return records

    def remove_from_promotion_queue(self, memory_id: str) -> None:
        """Remove a memory from the promotion queue."""
        self._promotion_queue.delete(where=f'memory_id = "{_esc(memory_id)}"')

    # ── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return DB statistics without materializing full tables into pandas.

        Uses LanceDB count queries for totals, and returns the computed
        breakdowns from the (small) metadata slices we actually need.
        """
        # Use DB-level counting; only load what's needed for the breakdowns
        mem_count = _table_row_count(self._memories)

        ent_df = self._entities.search().to_pandas()  # small table
        flag_count = self.unprocessed_flag_count()
        edge_count = _table_row_count(self._edges)

        # For network/status breakdowns on mem_df we need the actual rows.
        # If the DB is huge this still loads everything but at least total
        # counts are O(1). For moderation-sized usage (<10k memories) this
        # is acceptable; for huge deployments a dedicated stats service is
        # recommended.
        mem_df = self._memories.search().to_pandas()

        return {
            "total_memories": mem_count if mem_count >= 0 else len(mem_df),
            "memories_per_network": {
                n.value: len(mem_df[mem_df.network == n.value]) for n in Network
            },
            "memories_per_status": {
                s.value: len(mem_df[mem_df.status == s.value])
                for s in MemoryStatus
            },
            "total_entities": len(ent_df),
            "entities_per_type": {
                t.value: len(ent_df[ent_df.type == t.value]) for t in EntityType
            },
            "total_edges": edge_count,
            "unprocessed_flags": flag_count,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        """Close all LanceDB table handles to release OS-level file descriptors.

        LanceDB 0.34 AsyncTable exposes close(); calling it releases the
        underlying lance fragment readers. Without this, every search/store
        operation leaks OS file descriptors (one per .lance data fragment),
        eventually hitting the process ulimit.
        """
        for tbl in (self._memories, self._entities, self._identities,
                    self._edges, self._hot_cache, self._flags,
                    self._promotion_queue, self._session_summaries):
            if tbl is None:
                continue
            try:
                # AsyncTable (lancedb ≥0.30) has a proper close()
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
    return EntityRecord(
        id=row["id"],
        type=EntityType(row["type"]),
        label=row["label"],
        card=json.loads(row.get("card", "{}") or "{}"),
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
