"""One-shot migration: LanceDB JSONL export → SQLite.

Reads the JSONL backups produced during the audit and inserts them into
the new SQLite store. Idempotent (INSERT OR REPLACE).

Usage:
    python -m galaxymem.migrate_sqlite --backup-dir /home/ubuntu/backups/gm-migration \
        --db ~/.galaxymem/galaxymem.sqlite3
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    EdgeRecord, EdgeKind, EntityRecord, EntityType, FlagRecord, HotCache,
    IdentityLink, LinkMethod, MemoryRecord, MemoryStatus, Network,
    PromotionQueueRecord, SessionSummary,
)
from .store_sqlite import Store

logger = logging.getLogger(__name__)


def _parse_dt(v):
    if not v:
        return None
    try:
        d = datetime.fromisoformat(v)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _vec_to_blob(vec) -> bytes:
    if not vec:
        return b""
    return struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])


def migrate(backup_dir: Path, db_path: Path) -> dict:
    store = Store(db_path=db_path).open(create_if_missing=True)
    counts = {}

    # entities
    p = backup_dir / "entities.jsonl"
    if p.exists():
        n = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            store.add_entity(EntityRecord(
                id=r["id"], type=EntityType(r["type"]), label=r["label"],
                card=json.loads(r.get("card") or "{}"),
                status_line=r.get("status_line") or "",
                created_at=_parse_dt(r.get("created_at")) or datetime.now(timezone.utc),
                merged_into=r.get("merged_into"),
            ))
            n += 1
        counts["entities"] = n

    # identity_links
    p = backup_dir / "identity_links.jsonl"
    if p.exists():
        n = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            store.add_identity_link(IdentityLink(
                platform=r["platform"], external_id=r["external_id"],
                entity_id=r["entity_id"],
                created_at=_parse_dt(r.get("created_at")) or datetime.now(timezone.utc),
                created_by=LinkMethod(r.get("created_by") or "explicit"),
            ))
            n += 1
        counts["identity_links"] = n

    # edges
    p = backup_dir / "edges.jsonl"
    if p.exists():
        n = 0
        edges = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            edges.append(EdgeRecord(
                from_id=r["from_id"], to_id=r["to_id"],
                kind=EdgeKind(r["kind"]), weight=float(r.get("weight") or 1.0),
            ))
            n += 1
        store.add_edges(edges)
        counts["edges"] = n

    # memories (with vectors from the backup — no re-embedding needed)
    p = backup_dir / "memories.jsonl"
    if p.exists():
        n = 0
        import struct as _struct
        conn = store._conn
        with store._write_lock:
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                try:
                    vec = r.get("vector") or []
                    blob = _struct.pack(f"<{len(vec)}f", *[float(x) for x in vec]) if vec else None
                    conn.execute(
                        """INSERT OR REPLACE INTO memories
                           (id, text, vector, network, entity_ids, source_memory_ids, status,
                            superseded_by, contested_with, created_at, last_recalled_at,
                            recall_count, recall_miss_count, reflect_cycles,
                            source_session_id, source_platform, speaker_entity_id,
                            promoted_to, flagged_source, canonical_key, proof_count,
                            evidence_quotes, history_json)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (r["id"], r["text"], blob, r["network"],
                         r.get("entity_ids") or "[]",
                         r.get("source_memory_ids") or "[]",
                         r.get("status") or "active",
                         r.get("superseded_by"),
                         r.get("contested_with") or "[]",
                         r.get("created_at"),
                         r.get("last_recalled_at"),
                         int(r.get("recall_count") or 0),
                         int(r.get("recall_miss_count") or 0),
                         int(r.get("reflect_cycles") or 0),
                         r.get("source_session_id"), r.get("source_platform"),
                         r.get("speaker_entity_id"), r.get("promoted_to"),
                         r.get("flagged_source"), r.get("canonical_key"),
                         int(r.get("proof_count") or 0),
                         r.get("evidence_quotes") or "[]",
                         r.get("history_json")),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO memories_fts (id, text) VALUES (?,?)",
                        (r["id"], r["text"]),
                    )
                    n += 1
                except Exception as e:
                    logger.warning("skipping bad memory row %s: %s", r.get("id"), e)
            conn.commit()
        counts["memories"] = n

    # flags
    p = backup_dir / "flags.jsonl"
    if p.exists():
        n = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            store.add_flag(FlagRecord(
                id=r["id"], session_id=r["session_id"], platform=r["platform"],
                speaker_external_id=r["speaker_external_id"],
                turn_text=r["turn_text"], flag_reason=r["flag_reason"],
                processed=bool(r.get("processed")),
                attempt_count=int(r.get("attempt_count") or 0),
                created_at=_parse_dt(r.get("created_at")) or datetime.now(timezone.utc),
            ))
            n += 1
        counts["flags"] = n

    # promotion_queue
    p = backup_dir / "promotion_queue.jsonl"
    if p.exists():
        n = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            store.add_to_promotion_queue(
                r["memory_id"], json.loads(r.get("target_systems") or "[]"),
            )
            n += 1
        counts["promotion_queue"] = n

    # session_summaries
    p = backup_dir / "session_summaries.jsonl"
    if p.exists():
        n = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            store.upsert_session_summary(SessionSummary(
                id=r["id"], text=r["text"],
                message_count=int(r.get("message_count") or 0),
                last_updated=_parse_dt(r.get("last_updated")) or datetime.now(timezone.utc),
            ))
            n += 1
        counts["session_summaries"] = n

    store.close()
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--db", required=True)
    args = ap.parse_args()
    result = migrate(Path(args.backup_dir), Path(args.db))
    print(json.dumps(result, indent=2))


def verify(db_path: Path) -> dict:
    """Sanity check the migrated DB: count rows + run a few queries."""
    from .store_sqlite import Store
    store = Store(db_path=db_path).open()
    try:
        st = store.stats()
        sample = store.vector_search("test", k=3)
        kw = store.keyword_search("test", k=3)
        return {"stats": st, "vec_sample_size": len(sample), "kw_sample_size": len(kw)}
    finally:
        store.close()
