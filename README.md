# GalaxyMem

![GalaxyMem — connected memory nodes in a spiral galaxy](assets/galaxymem-logo.png)

**Entity-scoped memory with four-network epistemic split, decay, evidence-backed reasoning, and autonomous reflection.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LanceDB](https://img.shields.io/badge/storage-LanceDB-purple)](https://lancedb.com/)
[![Local-first](https://img.shields.io/badge/local--first-zero--server-green)](#)

LanceDB-backed memory for AI agents — built for [Hermes Agent](https://hermes-agent.nousresearch.com) but usable standalone. Provides hybrid vector+keyword recall fused via RRF, cross-encoder reranking, temporal queries, spreading activation, decay-based relevance, evidence-quoted opinion consolidation, staleness re-verification, usefulness feedback, and an on-demand reasoning loop (`gm_reason`).

**Local-first.** No server, no cloud, no per-memory LLM cost. Embeddings run on-device via fastembed; everything is a directory of Lance tables.

---

## Why?

Most agent memory systems are flat key-value stores or naive vector DBs. GalaxyMem models memory the way cognition works:

| Feature | What it does |
|---|---|
| **Four epistemic networks** | Memories are classified as `world` (facts), `experience` (events), `opinion` (preferences), or `observation` (patterns) — each with different decay and reflection rules |
| **Entity scoping** | Every memory is scoped to *who it's about* (the subject), not who said it. Recall filters by active entities in conversation |
| **Hybrid retrieval (TEMPR-style)** | Vector + BM25 keyword search fused via Reciprocal Rank Fusion, then cross-encoder reranked — names, technical terms, and paraphrases all match |
| **Temporal queries** | `as_of` recall answers "what did I believe then"; date-window parsing surfaces "last spring" style queries |
| **Decay + recall arrest** | Memories fade over time (30-day half-life by default). Every recall "touches" a memory, arresting its decay — things you use stay fresh |
| **Spreading activation** | Recall activates connected memories through entity and temporal edges — associative recall |
| **Evidence-quoted consolidation** | Reflection forms opinions from corroborated facts, storing verbatim source quotes + proof counts — every belief is checkable |
| **Staleness re-verification** | Existing opinions are re-checked against newer evidence; contradicted beliefs are demoted automatically |
| **Usefulness feedback** | Memories that are retrieved but never used get demoted; consistently useful ones stay — the store learns what matters |
| **On-demand reasoning (`gm_reason`)** | Ask a question, get a *reasoned, cited answer* — not a memory dump. Mirrors Hindsight's reflect() agentic loop |
| **Autonomous reflection** | Analyzes memories for contradictions, supersedes outdated facts, forms opinions, runs the usefulness policy |
| **Secret redaction** | Credential-shaped spans (API keys, tokens, passwords) are redacted at ingest — never stored |
| **Identity resolution** | Cross-platform identity links — map Telegram/Discord/CLI users to entities explicitly, never inferred |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    GalaxyMem Pipeline                        │
│                                                              │
│  Turn ──► Pass 1 (flag + redact) ──► Pass 2 (extract) ──► Store
│                                   │                          │
│  Prompt ◄── Hot Cache ◄── Deep Recall ◄── RRF Fusion         │
│               │                  │                           │
│               │          Spreading Activation                │
│               │                  │                           │
│          Reflect ◄──────── 100+ new memories                 │
│               │  ├─ conflicts (supersede / contest)          │
│               │  ├─ opinions w/ evidence quotes              │
│               │  ├─ staleness re-verify                      │
│               │  └─ usefulness policy (demote/revive)        │
│               │                                              │
│          Promote ──► Vault Proposals (human-gated)           │
│                                                              │
│  gm_reason ◄── opinions + facts + confidence ──► cited answer│
└──────────────────────────────────────────────────────────────┘
```

### Pipeline stages

| Stage | When | What |
|---|---|---|
| **Pass 1** (flag + redact) | Every turn, no LLM | Rule-based flagging of memorable turns; credential-shaped spans redacted |
| **Pass 2** (extract) | ≥12 flags, 20 min idle, `gm_flush`, or session end | One batched LLM call → self-contained memories, entity-scoped by subject |
| **Hot path** | Every prompt | Top-K brightest memories for active entities (≤800 tokens), injected via system prompt |
| **Deep recall** | `gm_recall` tool | Hybrid vector+keyword → RRF fusion → decay boost → spreading activation → cross-encoder rerank |
| **Reflect** | ≥100 new memories or `gm_reflect_now` | Supersession (mutable), contradiction (fixed), opinions w/ evidence quotes, staleness re-verify, usefulness policy |
| **Reason** | `gm_reason` tool | On-demand agentic loop: opinions → facts → cited answer with confidence, conflicts, gaps |
| **Promote** | After reflect / session end | Writes `workflow:draft` proposal notes into the vault inbox |

## Quick Start

### As a Hermes Agent plugin

```bash
# Clone into your Hermes plugins directory
git clone https://github.com/k1ng0mar/galaxymem.git ~/.hermes/plugins/galaxymem

# Install dependencies
cd ~/.hermes/plugins/galaxymem
pip install -e .
```

Select it as the active memory provider:

```bash
hermes memory setup   # pick "galaxymem"
# or
hermes config set memory.provider galaxymem
```

The plugin auto-registers on next session start. Built-in MEMORY.md / USER.md
continue to work underneath — GalaxyMem is additive.

### Standalone (Python library)

```bash
git clone https://github.com/k1ng0mar/galaxymem.git
cd galaxymem
pip install -e .
```

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
| `GALAXYMEM_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for recall reranking |

## API Tools (when used with Hermes Agent)

| Tool | Description |
|---|---|
| `gm_recall` | Deep recall — hybrid search + RRF fusion + spreading activation + rerank |
| `gm_reason` | **On-demand reasoning** — opinions → facts → cited answer with confidence/conflicts/gaps |
| `gm_store` | Manually store a memory |
| `gm_stats` | Memory store statistics |
| `gm_reflect` | Trigger reflection cycle (background) |
| `gm_reflect_now` | Immediate reflection + extraction |
| `gm_flush` | Force Pass 2 extraction of pending flags |
| `gm_forget` | Archive a memory (soft delete) |
| `gm_create_entity` | Create a new entity (person, project, etc.) |
| `gm_entity_card` | Get entity details + memories (incl. evidence quotes) |
| `gm_update_entity` | Update entity card fields |
| `gm_merge_entity` | Merge two entities |
| `gm_link_identity` | Link platform identity to entity |
| `gm_session_search` | Search past sessions by summary |

## Project Structure

```
galaxymem/                 # the Python package (pip-installable)
├── __init__.py            # version + public re-exports
├── config.py              # All tunables (env-overridable)
├── models.py              # Pydantic models (MemoryRecord, EntityRecord, EdgeRecord)
├── schema.py              # LanceDB table schemas (the LanceModel classes)
├── store.py               # Storage layer — CRUD, vector/keyword search, edges, stats
├── entities.py            # Entity CRUD, provisioning, slug resolution
├── identity.py            # Cross-platform identity management
├── retain.py              # Pass 1 flagging + redaction + Pass 2 LLM extraction
├── redact.py              # Credential-shaped secret detection + redaction
├── recall.py              # Hybrid search, RRF fusion, hot cache, spreading activation, usefulness tracking
├── rerank.py              # Cross-encoder reranking
├── temporal_parse.py      # Date-window parsing for temporal queries
├── queryexpansion.py     # LLM query expansion
├── reflect.py             # Supersession, contradiction, opinions w/ evidence quotes, staleness re-verify, usefulness policy
├── reason.py              # gm_reason — on-demand evidence-backed reasoning loop
├── promote.py             # Wiki/vault proposal bridge
├── provider.py            # Hermes Agent memory provider integration (tool schemas, init)
└── tests/                 # full coverage (270+ tests)
```

> **Layout note:** the importable package is `galaxymem/` (so `from galaxymem.store import Store` works after `pip install -e .`). The plugin root only holds Hermes-specific glue (`__init__.py` + `plugin.yaml`). This keeps the engine usable standalone AND as a Hermes plugin.

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/k1ng0mar/galaxymem.git
cd galaxymem
pip install -e ".[dev]"

# Run tests
pytest galaxymem/tests/ -v
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
