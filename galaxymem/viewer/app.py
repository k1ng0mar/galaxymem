"""FastAPI backend for Memory Galaxy viewer."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from ..store import Store
    from .. import config as cfg
    from ..models import MemoryStatus
except ImportError:
    # Standalone mode (e.g., cloned from GitHub without parent package)
    import sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).resolve().parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from galaxymem.store import Store
    import galaxymem.config as cfg
    from galaxymem.models import MemoryStatus

logger = logging.getLogger(__name__)

# ── Rate limiting (in-memory, simple) ───────────────────────────────────────

_rate_limits: dict[str, list[float]] = {}
_RATE_LIMIT_MAX = int(os.environ.get("GALAXYMEM_VIEWER_RATE_LIMIT", "60"))  # req/min
_RATE_LIMIT_WINDOW = 60.0  # seconds


def _check_rate_limit(ip: str) -> None:
    """Raise HTTP 429 if the IP has exceeded the rate limit."""
    now = __import__("time").time()
    window_start = now - _RATE_LIMIT_WINDOW
    timestamps = _rate_limits.get(ip, [])
    timestamps = [t for t in timestamps if t > window_start]
    if len(timestamps) >= _RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Slow down.")
    timestamps.append(now)
    _rate_limits[ip] = timestamps


# ── Lifespan ─────────────────────────────────────────────────────────────────

_store: Optional[Store] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage store lifecycle — open on startup, close on shutdown."""
    global _store
    yield
    if _store is not None:
        try:
            _store.close()
        except Exception:
            pass
        _store = None


# Initialize FastAPI app
app = FastAPI(
    title="Memory Galaxy Viewer",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — restrict to localhost; viewer is a local debug tool, not production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000", "http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# Global store instance (populated by lifespan or lazily)
def get_store() -> Store:
    """Get or initialize the Store instance."""
    global _store
    if _store is None:
        db_path = cfg.resolve_db_path()
        _store = Store(db_path=db_path).open(create_if_missing=True)
    return _store


def _get_client_ip(request: Request) -> str:
    """Extract client IP for rate limiting."""
    if request.client:
        return request.client.host or "127.0.0.1"
    return "127.0.0.1"


# ── API Endpoints ───────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Health check endpoint (no auth required)."""
    try:
        store = get_store()
        stats = store.stats()
        return {
            "status": "ok",
            "db_path": str(stats.get("db_path", "unknown")),
            "total_memories": stats.get("total_memories", 0),
            "total_entities": stats.get("total_entities", 0),
            "embedding_backend": cfg.EMBEDDING_BACKEND,
            "embedding_model": cfg.EMBEDDING_MODEL,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )


@app.get("/api/stats")
def get_stats(request: Request):
    """Get memory store statistics."""
    _check_rate_limit(_get_client_ip(request))
    try:
        store = get_store()
        return store.stats()
    except Exception as e:
        logger.error("Stats error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memories")
def list_memories(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    network: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
):
    """List memories with pagination and filters."""
    try:
        store = get_store()

        # Search mode
        if q:
            results = store.vector_search(q, k=limit)
            memories = [rec for rec, _ in results]
            return {
                "memories": [m.model_dump(mode="json") for m in memories],
                "total": len(results),
                "offset": 0,
                "limit": limit,
            }

        # List mode with filters
        memories = store.list_memories(
            network=network,
            status=status,
            limit=limit + offset,  # Fetch more to handle offset
        )

        # Apply offset
        memories = memories[offset : offset + limit]

        return {
            "memories": [m.model_dump(mode="json") for m in memories],
            "total": len(memories),
            "offset": offset,
            "limit": limit,
        }
    except Exception as e:
        logger.error("List memories error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memories/{memory_id}")
def get_memory(request: Request, memory_id: str):
    """Get a single memory with its edges."""
    try:
        store = get_store()
        memory = store.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        edges = store.get_edges_for_memory(memory_id)

        return {
            "memory": memory.model_dump(mode="json"),
            "edges": [e.model_dump(mode="json") for e in edges],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get memory error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entities")
def list_entities(request: Request):
    """List all entities."""
    _check_rate_limit(_get_client_ip(request))
    try:
        store = get_store()
        entities = store.list_entities()
        return {"entities": [e.model_dump(mode="json") for e in entities]}
    except Exception as e:
        logger.error("List entities error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entities/{entity_id}")
def get_entity(request: Request, entity_id: str):
    """Get entity with its memories."""
    _check_rate_limit(_get_client_ip(request))
    try:
        store = get_store()
        entity = store.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")

        # Get memories for this entity
        memories = store.list_memories(entity_ids=[entity_id], limit=100)

        return {
            "entity": entity.model_dump(mode="json"),
            "memories": [m.model_dump(mode="json") for m in memories],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get entity error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph")
def get_graph(
    request: Request,
    limit: int = Query(200, ge=1, le=1000),
    network: Optional[str] = None,
):
    """Get network graph data (nodes + edges)."""
    try:
        store = get_store()

        # Get memories
        memories = store.list_memories(network=network, limit=limit)
        memory_ids = {m.id for m in memories}

        # Get entities
        entities = store.list_entities()

        # Build nodes
        nodes = []
        for m in memories:
            nodes.append(
                {
                    "id": m.id,
                    "type": "memory",
                    "label": m.text[:80] + "..." if len(m.text) > 80 else m.text,
                    "network": m.network.value,
                    "status": m.status.value,
                    "recall_count": m.recall_count,
                }
            )

        for e in entities:
            nodes.append(
                {
                    "id": e.id,
                    "type": "entity",
                    "label": e.label,
                    "entity_type": e.type.value,
                }
            )

        # Build a set of all visible node IDs (memories + entities)
        entity_ids = {e.id for e in entities}
        all_node_ids = memory_ids | entity_ids

        # Get edges for visible memories
        edges = []
        for m in memories:
            # Explicit graph edges (supersedes, contests, etc.)
            mem_edges = store.get_edges_for_memory(m.id)
            for edge in mem_edges:
                if edge.from_id in all_node_ids and edge.to_id in all_node_ids:
                    edges.append(
                        {
                            "source": edge.from_id,
                            "target": edge.to_id,
                            "kind": edge.kind.value,
                            "weight": edge.weight,
                        }
                    )

            # Synthesize entity-memory edges from entity_ids association
            for eid in getattr(m, 'entity_ids', None) or []:
                if eid in entity_ids:
                    edges.append(
                        {
                            "source": eid,
                            "target": m.id,
                            "kind": "shared_entity",
                            "weight": 1.0,
                        }
                    )

        # Deduplicate edges
        seen = set()
        unique_edges = []
        for e in edges:
            key = (e["source"], e["target"], e["kind"])
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        return {"nodes": nodes, "edges": unique_edges}
    except Exception as e:
        logger.error("Graph error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/similarity")
def get_similarity_graph(
    request: Request,
    limit: int = Query(50, ge=1, le=50),  # hard cap to prevent N+1 DoS
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    network: Optional[str] = None,
):
    _check_rate_limit(_get_client_ip(request))
    """Get graph with semantic similarity edges based on embedding distance

    For each memory, finds the top-K most similar memories and creates edges
    where cosine similarity exceeds the threshold.

    WARNING: N+1 vector searches. Hard-capped at 50 nodes to prevent DoS.
    """
    try:
        store = get_store()
        memories = store.list_memories(network=network, limit=limit)

        if not memories:
            return {"nodes": [], "edges": []}

        # Build nodes
        nodes = []
        for m in memories:
            nodes.append({
                "id": m.id,
                "type": "memory",
                "label": m.text[:80] + "..." if len(m.text) > 80 else m.text,
                "network": m.network.value,
                "status": m.status.value,
                "recall_count": m.recall_count,
            })

        # Build similarity edges via vector_search for each memory
        edges = []
        seen = set()
        memory_ids = {m.id for m in memories}

        for m in memories:
            try:
                results = store.vector_search(m.text, k=6)
                for rec, score in results:
                    if rec.id == m.id or rec.id not in memory_ids:
                        continue
                    if score < threshold:
                        continue
                    # Dedupe bidirectional
                    key = tuple(sorted([m.id, rec.id]))
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append({
                        "source": m.id,
                        "target": rec.id,
                        "kind": "semantic",
                        "weight": round(score, 3),
                    })
            except Exception:
                continue

        # Also include entity nodes and their edges
        entities = store.list_entities()
        entity_ids = {e.id for e in entities}
        for e in entities:
            nodes.append({
                "id": e.id,
                "type": "entity",
                "label": e.label,
                "entity_type": e.type.value,
            })

        for m in memories:
            for eid in getattr(m, 'entity_ids', None) or []:
                if eid in entity_ids:
                    edges.append({
                        "source": eid,
                        "target": m.id,
                        "kind": "shared_entity",
                        "weight": 1.0,
                    })

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        logger.error("Similarity graph error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/entity/{entity_id}")
def get_entity_graph(request: Request, entity_id: str):
    """Get subgraph centered on an entity."""
    _check_rate_limit(_get_client_ip(request))
    try:
        store = get_store()

        # Verify entity exists
        entity = store.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")

        # Get memories for this entity
        memories = store.list_memories(entity_ids=[entity_id], limit=100)
        memory_ids = {m.id for m in memories}

        # Build nodes
        nodes = [
            {
                "id": entity.id,
                "type": "entity",
                "label": entity.label,
                "entity_type": entity.type.value,
            }
        ]

        for m in memories:
            nodes.append(
                {
                    "id": m.id,
                    "type": "memory",
                    "label": m.text[:80] + "..." if len(m.text) > 80 else m.text,
                    "network": m.network.value,
                    "status": m.status.value,
                    "recall_count": m.recall_count,
                }
            )

        # Get edges
        edges = []
        for m in memories:
            mem_edges = store.get_edges_for_memory(m.id)
            for edge in mem_edges:
                if edge.from_id in memory_ids and edge.to_id in memory_ids:
                    edges.append(
                        {
                            "source": edge.from_id,
                            "target": edge.to_id,
                            "kind": edge.kind.value,
                            "weight": edge.weight,
                        }
                    )

        # Deduplicate
        seen = set()
        unique_edges = []
        for e in edges:
            key = (e["source"], e["target"], e["kind"])
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        return {"nodes": nodes, "edges": unique_edges}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Entity graph error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Spec-contract endpoints (Phase 9 viewer) ────────────────────────────


def _compute_brightness(memory) -> float:
    """Brightness = recall decay. Computed once, not per-frame.

    brightness = max(brightness_floor, exp(-days_since_last_recall / half_life) * 0.7
                     + importance_proxy * 0.3)

    Demoted opinions are dimmed below the normal decay floor.
    Superseded memories get brightness 0 (hidden except in time-travel).
    """
    import math
    from datetime import datetime, timezone

    if memory.status.value == MemoryStatus.superseded.value:
        return 0.0
    if memory.status.value == MemoryStatus.demoted.value:
        return cfg.BRIGHTNESS_FLOOR * 0.4  # below normal decay floor

    now = datetime.now(timezone.utc)
    last_recalled = memory.last_recalled_at or memory.created_at
    if last_recalled.tzinfo is None:
        last_recalled = last_recalled.replace(tzinfo=timezone.utc)
    days_since = (now - last_recalled).total_seconds() / 86400.0

    # importance_proxy = normalized recall_count (cap at 10)
    importance = min(memory.recall_count / 10.0, 1.0)

    brightness = math.exp(-days_since / cfg.DECAY_HALF_LIFE_DAYS) * 0.7 + importance * 0.3
    return max(cfg.BRIGHTNESS_FLOOR, brightness)


@app.get("/graph")
def get_spec_graph(
    request: Request,
    limit: Optional[int] = Query(None, ge=1, le=10000),
    network: Optional[str] = None,
):
    """Spec-contract /graph endpoint: entities + memories + edges with brightness precomputed.

    Returns:
        { entities: [...], memories: [...], edges: [...], brightness_floor: float }
    """
    try:
        store = get_store()

        # Memories (exclude superseded by default — they're only for time-travel)
        memories = store.list_memories(network=network, limit=limit)
        memory_ids = {m.id for m in memories}

        # Entities
        entities = store.list_entities()
        entity_ids = {e.id for e in entities}

        # Build memory nodes with precomputed brightness
        mem_nodes = []
        for m in memories:
            mem_nodes.append({
                "id": m.id,
                "type": "memory",
                "label": m.text[:100] + "..." if len(m.text) > 100 else m.text,
                "text": m.text,
                "network": m.network.value,
                "status": m.status.value,
                "recall_count": m.recall_count,
                "entity_ids": m.entity_ids,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "last_recalled_at": m.last_recalled_at.isoformat() if m.last_recalled_at else None,
                "brightness": round(_compute_brightness(m), 4),
                "superseded_by": m.superseded_by,
                "contested_with": m.contested_with,
            })

        # Build entity nodes
        ent_nodes = []
        for e in entities:
            ent_nodes.append({
                "id": e.id,
                "type": "entity",
                "label": e.label,
                "entity_type": e.type.value,
                "status_line": e.status_line,
            })

        # Build edges
        edges = []
        seen_edges = set()
        all_node_ids = memory_ids | entity_ids

        for m in memories:
            # Explicit graph edges
            mem_edges = store.get_edges_for_memory(m.id)
            for edge in mem_edges:
                if edge.from_id in all_node_ids and edge.to_id in all_node_ids:
                    key = (edge.from_id, edge.to_id, edge.kind.value)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append({
                            "source": edge.from_id,
                            "target": edge.to_id,
                            "kind": edge.kind.value,
                            "weight": edge.weight,
                        })

            # Synthesize entity-memory edges from entity_ids association
            for eid in m.entity_ids or []:
                if eid in entity_ids:
                    key = (eid, m.id, "shared_entity")
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append({
                            "source": eid,
                            "target": m.id,
                            "kind": "shared_entity",
                            "weight": 1.0,
                        })

        return {
            "entities": ent_nodes,
            "memories": mem_nodes,
            "edges": edges,
            "brightness_floor": cfg.BRIGHTNESS_FLOOR,
            "total_nodes": len(ent_nodes) + len(mem_nodes),
        }
    except Exception as e:
        logger.error("Spec graph error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entity/{entity_id}")
def get_spec_entity(request: Request, entity_id: str):
    """Spec-contract entity detail endpoint."""
    _check_rate_limit(_get_client_ip(request))
    try:
        store = get_store()
        entity = store.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")

        memories = store.list_memories(entity_ids=[entity_id], limit=200)

        return {
            "entity": entity.model_dump(mode="json"),
            "memories": [
                {
                    **m.model_dump(mode="json"),
                    "brightness": round(_compute_brightness(m), 4),
                }
                for m in memories
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Spec entity error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{memory_id}")
def get_spec_memory(request: Request, memory_id: str):
    """Spec-contract memory detail endpoint."""
    _check_rate_limit(_get_client_ip(request))
    try:
        store = get_store()
        memory = store.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        edges = store.get_edges_for_memory(memory_id)

        return {
            "memory": {
                **memory.model_dump(mode="json"),
                "brightness": round(_compute_brightness(memory), 4),
            },
            "edges": [e.model_dump(mode="json") for e in edges],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Spec memory error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/as-of/{timestamp}")
def get_as_of(timestamp: str):
    """Time-travel: return the graph as it was at a given timestamp.

    Uses LanceDB versioning via store.as_of() to pin to the actual historical
    state of the memories table — shows the true state (status, recall_count,
    superseded_by, etc.) as it existed at that moment.

    The timestamp should be ISO format.
    """
    try:
        store = get_store()

        # Parse timestamp
        from datetime import datetime, timezone
        try:
            as_of_dt = datetime.fromisoformat(timestamp)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO 8601.")

        # Get the historical store handle — a read-only view at that timestamp
        try:
            historical_store = store.as_of(as_of_dt)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"No memory-table version exists at or before {timestamp}: {e}"
            )

        # Historical store is read-only but list_memories works on it.
        # Get ALL memories (including superseded) as they existed then.
        memories = historical_store.list_memories(limit=1000)
        entities = store.list_entities()
        entity_ids = {e.id for e in entities}

        mem_nodes = []
        for m in memories:
            mem_nodes.append({
                "id": m.id,
                "type": "memory",
                "label": m.text[:100] + "..." if len(m.text) > 100 else m.text,
                "text": m.text,
                "network": m.network.value,
                "status": m.status.value,
                "recall_count": m.recall_count,
                "entity_ids": m.entity_ids,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "brightness": round(_compute_brightness(m), 4),
            })

        ent_nodes = [
            {
                "id": e.id,
                "type": "entity",
                "label": e.label,
                "entity_type": e.type.value,
                "status_line": e.status_line,
            }
            for e in entities
        ]

        # Build edges (same as /graph but include superseded)
        edges = []
        seen_edges = set()
        all_node_ids = {m.id for m in memories} | entity_ids

        for m in memories:
            mem_edges = store.get_edges_for_memory(m.id)
            for edge in mem_edges:
                if edge.from_id in all_node_ids and edge.to_id in all_node_ids:
                    key = (edge.from_id, edge.to_id, edge.kind.value)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append({
                            "source": edge.from_id,
                            "target": edge.to_id,
                            "kind": edge.kind.value,
                            "weight": edge.weight,
                        })

            for eid in m.entity_ids or []:
                if eid in entity_ids:
                    key = (eid, m.id, "shared_entity")
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append({
                            "source": eid,
                            "target": m.id,
                            "kind": "shared_entity",
                            "weight": 1.0,
                        })

        return {
            "entities": ent_nodes,
            "memories": mem_nodes,
            "edges": edges,
            "brightness_floor": cfg.BRIGHTNESS_FLOOR,
            "as_of": timestamp,
            "total_nodes": len(ent_nodes) + len(mem_nodes),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("As-of error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Static Files ────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def serve_index():
    """Serve the SPA."""
    return FileResponse(STATIC_DIR / "index.html")


# Mount static files (must be after API routes)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Runner ──────────────────────────────────────────────────────────────


def run_viewer(port: int = 7331, host: str = "127.0.0.1"):
    """Launch the Memory Galaxy viewer."""
    import uvicorn

    logger.info("Starting Memory Galaxy viewer on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_viewer()
