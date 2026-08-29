# GalaxyMem Architecture

This document explains how GalaxyMem is structured internally — the module
boundaries, the data flow, and *why* things are split the way they are. If
you're contributing, start here; if you're just using it, the README is enough.

---

## The one-paragraph version

A memory *turn* flows through a two-pass pipeline (cheap rule-based flagging →
batched LLM extraction) into a LanceDB store. At recall time, a hybrid
vector+keyword search is fused (RRF), boosted by decay, and expanded through
one-hop graph activation. Periodically, a reflection pass repairs the store
(contradictions, supersession, opinion formation). Everything is scoped to
*entities* (who the memory is about) and tagged with an *epistemic network*
(fact / event / opinion / observation).

---

## Module boundaries

GalaxyMem is deliberately split into **storage**, **records**, **logic**, and
**integration** layers. No module imports across layers except downward.

```
                    provider.py  ← Hermes integration (tool schemas, init, prefetch)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   retain.py         recall.py        reflect.py      ← pipeline logic
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                      store.py  ← storage: CRUD, search, edges, stats
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         schema.py   models.py   config.py   ← data + tunables (leaf modules)
```

| Module | Responsibility | Depends on |
|---|---|---|
| `config.py` | All tunables, env-overridable | `sanitize` (env parsers) |
| `sanitize.py` | Prompt/YAML/JSON/path sandbox helpers | nothing |
| `models.py` | Pydantic record types (Memory, Entity, Edge, Identity) | `config` |
| `schema.py` | LanceDB `LanceModel` table definitions + `_esc` | `config` |
| `store.py` | The `Store` class: CRUD, vector/keyword search, edges, hot cache, flags, stats, versioning | `schema`, `models`, `config`, `redact` |
| `entities.py` | Entity CRUD, provisional provisioning, slug resolution | `store`, `models` |
| `identity.py` | Platform↔entity link resolution | `store`, `models` |
| `retain.py` | Pass 1 flag rules + Pass 2 LLM extraction | `store`, `entities`, `models`, `config` |
| `recall.py` | `deep_recall`, RRF fusion, spreading activation, hot cache | `store`, `models`, `config` |
| `reflect.py` | Supersession, contradiction, opinion formation, cascades | `store`, `models`, `config` |
| `promote.py` | Export high-value memories to vault/wiki (human-gated) | `store`, `models`, `config` |
| `provider.py` | Hermes `MemoryProvider` impl: tool schemas, `initialize()`, prefetch | everything above |

**Key rule:** `store.py` contains *zero* business logic — it moves rows in and
out of LanceDB and runs searches. All "what should we remember / what does this
mean" logic lives in `retain.py` / `recall.py` / `reflect.py`. This is why
`store.py` is the largest file but the easiest to reason about: it has no
branching on meaning, only on data shape.

---

## Data model

### Four epistemic networks

Every memory carries a `network` tag. This is the core modeling decision — it
lets different *kinds* of knowledge age and reflect differently.

| Network | Meaning | Example | Reflection behavior |
|---|---|---|---|
| `world` | Objective facts | "API runs on port 8010" | Superseded when a newer fact arrives |
| `experience` | Events/actions | "User deployed v2 at 2am" | Rarely contested; timestamped |
| `opinion` | Preferences/views | "User prefers direct execution" | Formed from repeated world+experience signals |
| `observation` | Patterns/insights | "User ignores planning loops" | Demoted if its sources get contested |

### Entity scoping

Memories are scoped to **who they're about** (`entity_ids`), not who said them.
Recall filters by the active entities in the current conversation (plus
unscoped `world` facts — see D8 below). This prevents "Bob told me X" from
leaking into a conversation with Alice.

### Identity resolution

Platform identities (telegram:123, discord:456) are linked to entities **only
via explicit `gm_link_identity` calls** — never inferred. A provisional entity
is auto-created on first contact and surfaced to the user once ("want to link
this?"), but the link is never made automatically.

---

## The pipeline (detailed)

### Pass 1 — flag (every turn, no LLM)

`retain.flag_turn()` applies rule-based heuristics to each incoming turn:
- self-references ("my", "i prefer")
- decisions / commitments ("let's", "will do")
- contradictions of prior state
- explicit user-requested memory

Flagged turns go into a `flags` queue. No LLM call — this is pure string
matching, so it's free to run every turn.

### Pass 2 — extract (batched, LLM)

Triggered when ≥`PASS2_FLAG_THRESHOLD` (12) flags accumulate, after
`PASS2_IDLE_MINUTES` (20) idle, on `gm_flush`, or at session end. One batched
LLM call turns the flag queue into self-contained `MemoryRecord`s, each scoped
to the subject entity. Flags are marked processed so they're never double-counted.

### Hot path (every prompt)

Before each agent turn, `provider.prefetch()` runs `deep_recall` for the active
entities and writes the top-K brightest memories into a `hot_cache` row
(≤`HOT_CACHE_TOKEN_BUDGET` tokens). The agent reads the hot cache via its
system prompt — cheap, always-fresh context.

### Deep recall (`gm_recall`)

```
query ──┬──► vector_search()  ──┐
       └──► keyword_search() ──┼──► RRF fusion ──► decay boost ──► spreading activation
                                                                       │
                                                          neighbors via edges (1-hop)
                                                                       │
                                                          └──► ranked MemoryRecords
```

- **RRF (Reciprocal Rank Fusion):** `score = Σ 1/(k+rank)`. Blends vector and
  keyword ranks without needing comparable score scales.
- **Decay boost:** `brightness = 0.5^age/half_life`. Recalled memories get
  "touched" (recall_count++, last_recalled_at=now), arresting their decay.
- **Spreading activation:** each top result's neighbors (through `edges`) get a
  damped relevance contribution, surfacing connected context the query didn't
  name directly.

### Reflection (`gm_reflect` / `gm_reflect_now`)

Runs when ≥`REFLECT_VOLUME_TRIGGER` (100) new memories accumulate, or on demand:
1. **Supersession** — a newer `world` fact about the same subject marks the old
   one `superseded` (kept for history, excluded from recall).
2. **Contradiction** — two fixed claims conflict → both `contested`, surfaced
   for human resolution.
3. **Opinion formation** — ≥`OPINION_REQUIRES_MIN_SOURCES` (3) world/experience
   signals about a preference → an `opinion` memory is synthesized.
4. **Invalidation cascade** — if an opinion's source memories get contested,
   the opinion is `demoted` (kept but deprioritized).
5. **Entity nomination** — names that recur but aren't linked get nominated for
   explicit `gm_create_entity`.

### Promotion (`gm_promote` / session end)

High-confidence memories (≥`PROMOTION_MIN_RECALLS` recalls, ≥`PROMOTION_MIN_CYCLES`
reflection cycles) are written as `workflow:draft` proposal notes into the vault
inbox. A human approves before they enter the wiki/Obsidian — GalaxyMem never
writes to your knowledge base unilaterally.

---

## Design decisions (the "why")

These are the non-obvious calls. Each is tagged with the design rule that
motivated it.

- **D1 — Two-pass, not one.** Flagging every turn is free; extracting every
  turn is not. Decoupling them lets us batch LLM calls.
- **D3 — Links never inferred.** Identity is sensitive; a wrong auto-link
  corrupts memory ownership permanently. Explicit-only is safer.
- **D8 — Recall scope = requested entities + unscoped world.** A memory with no
  entity ("the sky is blue") is universally relevant; a memory scoped to Bob is
  not. The `OR (entity_ids = '[]' AND network = 'world')` clause encodes this.
- **D12 — Promotion is human-gated.** Memory the agent writes to its own store
  is low-stakes; memory it writes to *your* wiki is high-stakes. Gate it.
- **D13 — Never hard-delete.** `delete_memory` is a soft `archived` status.
  Memory bugs are silent; being able to audit "what did it used to believe" is
  the only recourse.
- **Store has no business logic (separation of concerns).** Makes `store.py`
  testable without an LLM and lets `recall`/`reflect` evolve independently.

---

## Testing layout

`galaxymem/tests/` mirrors the module structure:

| Test file | Covers |
|---|---|
| `test_store.py` | CRUD, search, edges, versioning, compaction |
| `test_entities.py` | Entity CRUD, provisioning, merge |
| `test_identity.py` | Platform↔entity link resolution |
| `test_retrieval.py` | `deep_recall`, RRF, spreading activation |
| `test_reflect_root.py` | Supersession, contradiction, opinion cascades |
| `test_retain_root.py` | Flag→extract flow, dedup, LLM-failure handling |
| `test_promote.py` | Promotion queue + vault draft writing |
| `test_e2e.py` | Full pipeline: flag → extract → recall → reflect |
| `test_spec_regressions.py` | Schema/contract stability guards |

Run with: `pytest galaxymem/tests/ -v` (after `pip install -e ".[dev]"`).

---

## Extending GalaxyMem

- **New memory source?** Add a flag rule in `retain.flag_turn()` — no other
  changes needed; Pass 2 + store handle the rest.
- **New recall signal?** Add a search method to `store.py`, fuse it in
  `recall.deep_recall()` via RRF.
- **New reflection rule?** Add a function in `reflect.py` and call it from
  `run_reflection()`.
- **New Hermes tool?** Add a schema dict in `provider.py` + a handler method.
  The tool auto-registers (12 tools currently).
- **Standalone (no Hermes)?** Use `store.Store` + `retain`/`recall`/`reflect`
  directly. `provider.py` is the only Hermes-coupled module.
