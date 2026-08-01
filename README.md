# GalaxyMem

**Entity-scoped memory with four-network epistemic split, decay, and autonomous reflection.**

LanceDB-backed memory for AI agents — built for [Hermes Agent](https://hermes-agent.nousresearch.com) but usable standalone. Provides hybrid vector+keyword recall, spreading activation, decay-based relevance, autonomous reflection cycles, and a built-in interactive galaxy graph viewer.

---

## Why?

Most agent memory systems are flat key-value stores or naive vector DBs. GalaxyMem models memory the way cognition works:

| Feature | What it does |
|---|---|
| **Four epistemic networks** | Memories are classified as `world` (facts), `experience` (events), `opinion` (preferences), or `observation` (patterns) — each with different decay and reflection rules |
| **Entity scoping** | Every memory is scoped to *who it's about* (the subject), not who said it. Recall filters by active entities in conversation |
| **Decay + recall arrest** | Memories fade over time (30-day half-life by default). Every recall "touches" a memory, arresting its decay — things you use stay fresh |
| **Spreading activation** | Recall doesn't just find direct matches — it activates connected memories through entity and temporal edges (one-hop) |
| **Autonomous reflection** | Periodically analyzes memories for contradictions, supersedes outdated facts, forms opinions from observations, and nominates items for promotion |
| **Promotion bridge** | High-confidence memories can be promoted into an external knowledge base (wiki, Obsidian vault) through a human-gated proposal pipeline |
| **Identity resolution** | Cross-platform identity links — map Telegram/Discord/CLI users to entities explicitly, never inferred |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GalaxyMem Pipeline                       │
│                                                              │
│  Turn ──► Pass 1 (flag) ──► Pass 2 (extract) ──► Store      │
│                                  │                           │
│  Prompt ◄── Hot Cache ◄── Deep Recall ◄── RRF Fusion       │
│               │                  │                           │
│               │          Spreading Activation                │
│               │                  │                           │
│          Reflect ◄──────── 100+ new memories                 │
│               │                                              │
│          Promote ──► Vault Proposals (human-gated)           │
│                                                              │
│  Viewer (FastAPI + React) ──► Interactive Galaxy Graph       │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline stages

| Stage | When | What |
|---|---|---|
| **Pass 1** (flag) | Every turn, no LLM | Rule-based flagging of memorable turns into a queue |
| **Pass 2** (extract) | ≥12 flags, 20 min idle, `gm_flush`, or session end | One batched LLM call → self-contained memories, entity-scoped by subject |
| **Hot path** | Every prompt | Top-K brightest memories for active entities (≤800 tokens), injected via system prompt |
| **Deep recall** | `gm_recall` tool | Hybrid vector+keyword search → RRF fusion → decay boost → single-hop spreading activation |
| **Reflect** | ≥100 new memories or `gm_reflect_now` | Supersession (mutable facts), contradiction detection (fixed claims), opinion formation, invalidation cascade |
| **Promote** | After reflect / session end | Writes `workflow:draft` proposal notes into the vault inbox |

## Quick Start

### As a Hermes Agent plugin

```bash
# Clone into your Hermes plugins directory
git clone https://github.com/k1ng0mar/galaxymem.git ~/.hermes/plugins/galaxymem

# Install dependencies
cd ~/.hermes/plugins/galaxymem
pip install -e ".[viewer]"

# The plugin auto-registers on next session start
```

Set in `config.yaml`:
```yaml
memory_enabled: false  # Disable built-in memory
# GalaxyMem registers itself as the memory provider
```

### Standalone (Python library)

```bash
git clone https://github.com/k1ng0mar/galaxymem.git
cd galaxymem
pip install -e ".[viewer]"
```

Using the memory store directly:

```python
from galaxymem.store import Store
from pathlib import Path

db = Store(db_path=Path("./my_memory"))
db.open(create_if_missing=True)

# Store a memory
mem_id = db.add_memory(
    text="The API runs on port 8010",
    network="world",
    entity_ids=["self"],
)

# Search
results = db.vector_search("what port is the API on", k=5)
for mem, score in results:
    print(f"  {score:.3f}  {mem.text}")
```

### Standalone (with the galaxy viewer)

```bash
python -c "from galaxymem.viewer.app import run_viewer; run_viewer()"
# Open http://127.0.0.1:7331
```

## Configuration

All tunables are in `config.py` and overridable via `GALAXYMEM_*` environment variables:

| Setting | Default | Description |
|---|---|---|
| `GALAXYMEM_DB_PATH` | `~/.galaxymem/db` | LanceDB database path |
| `GALAXYMEM_DECAY_HALF_LIFE_DAYS` | `30` | How fast memories fade |
| `GALAXYMEM_HOT_CACHE_K` | `8` | Max memories in hot cache |
| `GALAXYMEM_HOT_CACHE_TOKEN_BUDGET` | `800` | Token budget for injected context |
| `GALAXYMEM_PASS2_FLAG_THRESHOLD` | `12` | Flags needed to trigger extraction |
| `GALAXYMEM_REFLECT_VOLUME_TRIGGER` | `100` | New memories needed to trigger reflection |
| `GALAXYMEM_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (via fastembed) |

## API Tools (when used with Hermes Agent)

| Tool | Description |
|---|---|
| `gm_recall` | Deep recall with hybrid search + spreading activation |
| `gm_store` | Manually store a memory |
| `gm_stats` | Memory store statistics |
| `gm_reflect` | Trigger reflection cycle (background) |
| `gm_reflect_now` | Immediate reflection + extraction |
| `gm_flush` | Force Pass 2 extraction of pending flags |
| `gm_forget` | Archive a memory (soft delete) |
| `gm_create_entity` | Create a new entity (person, project, etc.) |
| `gm_entity_card` | Get entity details + memories |
| `gm_update_entity` | Update entity card fields |
| `gm_merge_entity` | Merge two entities |
| `gm_link_identity` | Link platform identity to entity |

## Galaxy Viewer

The built-in viewer provides an interactive force-directed graph visualization of your memory store:

- **Galaxy view** — Default landing page. All memories as nodes, entities as gravitational centers, connections as curved edges. Zoom, pan, drag, hover for details
- **Stats view** — Memory counts by network, status breakdown, entity types
- **Memories browser** — Paginated card view with network/status filters and full-text search

```bash
python -c "from galaxymem.viewer.app import run_viewer; run_viewer(port=7331)"
```

## Project Structure

```
galaxymem/                 # the Python package (pip-installable)
├── __init__.py            # version + public re-exports
├── config.py              # All tunables (env-overridable)
├── models.py              # Pydantic models (MemoryRecord, EntityRecord, EdgeRecord)
├── schema.py              # LanceDB table schemas (the 7 LanceModel classes)
├── store.py               # Storage layer — CRUD, vector/keyword search, edges, stats
├── entities.py            # Entity CRUD, provisioning, slug resolution
├── identity.py            # Cross-platform identity management
├── retain.py              # Pass 1 flagging + Pass 2 LLM extraction
├── recall.py              # Hybrid search, RRF fusion, hot cache, spreading activation
├── reflect.py             # Supersession, contradiction, opinion formation
├── promote.py             # Wiki/vault proposal bridge
├── provider.py            # Hermes Agent memory provider integration (tool schemas, init)
├── viewer/                # Optional FastAPI + JS galaxy-graph viewer ([viewer] extra)
│   ├── app.py
│   └── static/
└── tests/                 # full coverage (~229 tests; count grows as tests are added)

# Plugin glue (lives at the plugin root, not inside the package):
__init__.py                # Hermes plugin entry: register(GalaxyMemProvider)
plugin.yaml               # Hermes plugin manifest
pyproject.toml            # Python package config
requirements.txt          # Dependency list
```

> **Layout note:** the importable package is `galaxymem/` (so `from galaxymem.store import Store` works after `pip install -e .`). The plugin root only holds Hermes-specific glue (`__init__.py` + `plugin.yaml`). This keeps the engine usable standalone AND as a Hermes plugin.

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/k1ng0mar/galaxymem.git
cd galaxymem
pip install -e ".[dev,viewer]"

# Run tests
pytest galaxymem/tests/ -v

# Run viewer in dev mode
uvicorn galaxymem.viewer.app:app --reload --port 7331
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [AGENTS.md](AGENTS.md) for AI-agent setup instructions.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — module boundaries, data model, pipeline internals, design decisions (read this before contributing)
- **[README.md](README.md)** — this file (setup, config, tools)

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [LanceDB](https://lancedb.com/) — vector database
- [fastembed](https://github.com/qdrant/fastembed) — local embeddings
- [Hermes Agent](https://hermes-agent.nousresearch.com) — the agent framework this was built for
