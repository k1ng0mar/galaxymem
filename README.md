# GalaxyMem

![GalaxyMem — connected memory nodes in a spiral galaxy](assets/galaxymem-logo.png)

**Entity-scoped memory for AI agents — four epistemic networks, decay, evidence-backed reasoning, and autonomous reflection.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/storage-SQLite-purple)](https://sqlite.org/)
[![Local-first](https://img.shields.io/badge/local--first-zero--server-green)](#)

Local-first agent memory built for [Hermes Agent](https://hermes-agent.nousresearch.com), usable anywhere Python runs. Hybrid vector + keyword recall fused with RRF, cross-encoder reranking, temporal queries, spreading activation, decay-based relevance, evidence-quoted opinions, staleness re-verification, usefulness feedback, and a reasoning loop you can call on demand.

Everything lives in one SQLite file. No server, no cloud account, no per-memory LLM cost. Embeddings run on-device via fastembed.

---

## Why

Most agent memory is a flat list of facts in a vector database. Find similar text, return the top matches, done. That works until the agent has to remember who a preference belongs to, notice that a fact went stale, or explain *why* it believes something.

GalaxyMem models memory closer to how it actually works:

| | |
|---|---|
| **Four epistemic networks** | Memories are `world` facts, `experience` events, `opinion` preferences, or `observation` patterns. Each decays and reflects differently |
| **Entity scoping** | Every memory is scoped to *who or what it's about*, not who said it. Recall filters to the entities active in the conversation |
| **Hybrid retrieval** | Vector + BM25 keyword, fused via Reciprocal Rank Fusion, then cross-encoder reranked. Names, error codes, and paraphrases all land |
| **Temporal queries** | Ask "what did I believe in July" and get what was true then, not what's true now |
| **Decay with recall arrest** | Memories fade over a 30-day half-life by default. Every recall touches a memory and arrests decay — things you use stay fresh |
| **Spreading activation** | Recall walks out through entity and temporal edges, surfacing connected memories you didn't ask for but probably wanted |
| **Evidence-quoted consolidation** | Reflection forms opinions from corroborating facts and stores the verbatim source quotes. Every belief can be checked against its evidence |
| **Staleness re-verification** | Existing opinions get re-checked against newer evidence. Contradicted ones are demoted, not quietly left wrong |
| **Usefulness feedback** | Memories that surface but never help get demoted. Consistently useful ones stay |
| **On-demand reasoning** | `gm_reason` returns a cited answer with confidence, conflicts, and gaps — not a raw memory dump |
| **Secret redaction** | Credential-shaped spans get redacted at ingest, so an API key ends up in neither storage nor the prompt |
| **Explicit identity** | Cross-platform identity links are explicit only. Telegram user X maps to an entity because you said so, never because a model guessed |

## How it works

```
┌──────────────────────────────────────────────────────────┐
│ Turn ──► Pass 1 (flag + redact) ──► Pass 2 (extract) ──► store
│                                                            │
│ Prompt ◄── Hot cache ◄── Deep recall ◄── RRF fusion        │
│                │                  │                        │
│                │         Spreading activation              │
│                │                  │                        │
│         Reflect ◄── 100+ new memories (or gm_reflect_now)  │
│                │   ├─ conflicts → supersede / contest      │
│                │   ├─ opinions with evidence quotes        │
│                │   ├─ staleness re-verify                  │
│                │   └─ usefulness policy                    │
│                │                                           │
│ gm_reason ◄── opinions + facts + confidence ──► answer    │
└──────────────────────────────────────────────────────────┘
```

| Stage | When | What happens |
|---|---|---|
| **Pass 1** | Every turn, no LLM | Rules decide if a turn is worth remembering. Credential-shaped spans get redacted before anything is stored |
| **Pass 2** | 12+ flags, 20 min idle, `gm_flush`, or session end | One batched LLM call extracts entity-scoped memories |
| **Hot cache** | Every prompt | Top-K brightest memories for active entities (token-budgeted) injected as context |
| **Deep recall** | `gm_recall` | Vector + keyword → RRF → decay boost → spreading activation → rerank |
| **Reflect** | 100+ new memories or on demand | Resolves contradictions, merges opinions, demotes stale beliefs |
| **Reason** | `gm_reason` | Agentic loop over opinions and facts, returns a cited answer |
| **Promote** | After reflection | Drafts vault proposals for memories worth promoting — human-gated |

## Install

As a Hermes Agent plugin:

```bash
git clone https://github.com/k1ng0mar/galaxymem.git ~/.hermes/plugins/galaxymem
cd ~/.hermes/plugins/galaxymem
pip install -e ".[dev]"
```

then point Hermes at it:

```bash
hermes config set memory.provider galaxymem
```

The plugin registers on the next session start. Built-in MEMORY.md and USER.md keep working underneath; GalaxyMem is additive.

Standalone, as a library:

```bash
git clone https://github.com/k1ng0mar/galaxymem.git
cd galaxymem
pip install -e .
```

```python
from pathlib import Path
from galaxymem.store_sqlite import Store
from galaxymem.models import MemoryRecord, Network

store = Store(db_path=Path("./my_memory")).open()

store.add_memory(MemoryRecord(
    id="example-1",
    text="The API runs on port 8010",
    network=Network.world,
    entity_ids=["self"],
))

results = store.vector_search("what port is the API on", k=5)
for mem, score in results:
    print(f"{score:.3f}  {mem.text}")
```

## Configuration

Everything is an environment variable with a `GALAXYMEM_` prefix:

| Variable | Default | What it controls |
|---|---|---|
| `GALAXYMEM_DB_PATH` | `~/.galaxymem/db` | Where the database lives |
| `GALAXYMEM_DECAY_HALF_LIFE_DAYS` | `30` | Memory fade rate |
| `GALAXYMEM_HOT_CACHE_K` | `8` | Memories in the hot cache |
| `GALAXYMEM_HOT_CACHE_TOKEN_BUDGET` | `800` | Token cap for injected context |
| `GALAXYMEM_PASS2_FLAG_THRESHOLD` | `12` | Flags before extraction runs |
| `GALAXYMEM_REFLECT_VOLUME_TRIGGER` | `100` | New memories before reflection |
| `GALAXYMEM_MAX_MEMORIES` | `50000` | Hard cap on stored memories |
| `GALAXYMEM_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `GALAXYMEM_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Recall reranker |

## Tools (inside Hermes)

| Tool | What it does |
|---|---|
| `gm_recall` | Deep recall: hybrid search, fusion, activation, rerank |
| `gm_explain_recall` | Same results with provenance — which retrieval arms selected each memory and why |
| `gm_reason` | Cited, evidence-backed answer over memories |
| `gm_store` | Manually store a memory |
| `gm_stats` | Store statistics |
| `gm_reflect` | Kick off a reflection cycle in the background |
| `gm_reflect_now` | Extraction + reflection, blocking |
| `gm_flush` | Force Pass 2 extraction of pending flags |
| `gm_forget` | Archive a memory (soft delete — it's never destroyed) |
| `gm_create_entity` / `gm_update_entity` / `gm_merge_entity` | Entity lifecycle |
| `gm_entity_card` | Full entity profile with links and memories |
| `gm_link_identity` | Map a platform identity to an entity |
| `gm_session_search` | Search past sessions by summary |

## Project layout

```
galaxymem/                 # repo root (also the Hermes plugin dir)
├── __init__.py            # Hermes entry point — lazy register()
├── plugin.yaml            # plugin manifest
├── conftest.py            # pytest boundary (do not delete)
├── pyproject.toml
└── galaxymem/             # the installable package
    ├── config.py          # tunables, env-overridable
    ├── models.py          # MemoryRecord, EntityRecord, EdgeRecord, ...
    ├── store_sqlite.py    # storage layer: CRUD, vector + keyword search, edges
    ├── embed.py           # embeddings (fastembed, with a deterministic fallback)
    ├── entities.py        # entity lifecycle + provisioning
    ├── identity.py        # cross-platform identity links
    ├── retain.py          # Pass 1 flagging + Pass 2 extraction
    ├── redact.py          # credential detection and redaction
    ├── recall.py          # hybrid search, RRF, hot cache, spreading activation
    ├── rerank.py          # cross-encoder reranking
    ├── temporal_parse.py  # "last July", "in March" → date windows
    ├── queryexpansion.py  # LLM query expansion
    ├── reflect.py         # supersession, opinions, staleness, usefulness
    ├── reason.py          # the gm_reason loop
    ├── promote.py         # vault proposal bridge
    ├── summaries.py       # rolling session summaries
    └── tests/             # 280+ tests
```

## Development

```bash
pip install -e ".[dev]"
pytest galaxymem/tests/ -v
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for module boundaries and design decisions, [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute, and [AGENTS.md](AGENTS.md) if you're an AI agent setting this up for someone.

## License

MIT — see [LICENSE](LICENSE).

Built on [SQLite](https://sqlite.org/) with [sqlite-vec](https://github.com/asg017/sqlite-vec) and [fastembed](https://github.com/qdrant/fastembed). Made for [Hermes Agent](https://hermes-agent.nousresearch.com).
