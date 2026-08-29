# AGENTS.md — GalaxyMem setup for AI agents

This file is for AI coding agents (Claude, GPT, Hermes, etc.) that are asked to
install, configure, or integrate GalaxyMem on a user's behalf. It is designed to
be read in full before any installation steps.

---

## What this project is

GalaxyMem is an entity-scoped memory engine for AI agents. It stores memories
in a single SQLite database file (with sqlite-vec for vector search and FTS5
for keyword search), classifies them into four epistemic
networks (world / experience / opinion / observation), decays them over time,
and runs autonomous reflection cycles to resolve contradictions and form
opinions.

It works in two modes:
1. **Hermes Agent plugin** — drop into `~/.hermes/plugins/`, auto-registers
2. **Standalone Python library** — `pip install` + use `Store` directly

---

## Pre-flight checks

Before installing, verify the environment:

```bash
python3 --version   # needs >= 3.10
pip --version       # needs pip
```

GalaxyMem's heavy dependencies are `sqlite-vec` (SQLite vector extension) and
`fastembed` (local ONNX embeddings). These pull in `onnxruntime` and `numpy`.
The first install downloads a ~100MB embedding model
(`BAAI/bge-small-en-v1.5`) on first use.

**Disk:** ~500MB for dependencies + model.  
**RAM:** ~2GB with embedding model loaded.  
**GPU:** Not required (CPU inference via ONNX Runtime).

---

## Installation (pick one mode)

### Mode 1: Hermes Agent plugin

```bash
git clone https://github.com/k1ng0mar/galaxymem.git ~/.hermes/plugins/galaxymem
cd ~/.hermes/plugins/galaxymem
pip install -e ".[dev]"
```

Then in Hermes `config.yaml`:
```yaml
memory_enabled: false   # disable built-in memory
# GalaxyMem auto-registers as the memory provider
```

Restart the Hermes session. Verify with:
```bash
hermes tools list | grep gm_
```

If the `gm_*` tools don't appear, check:
- `pip show galaxymem` returns the package
- `python -c "from galaxymem.store_sqlite import Store"` works
- Hermes logs show "GalaxyMem store opened (backend=sqlite)"

### Mode 2: Standalone (no Hermes)

```bash
git clone https://github.com/k1ng0mar/galaxymem.git
cd galaxymem
pip install -e .
```

Verify:
```bash
python -c "from galaxymem.store_sqlite import Store; print('OK')"
python -c "import galaxymem; print(galaxymem.__version__)"
```

---

## Critical: the package layout

The repo has a nested structure:

```
galaxymem/                 # repo root (also the Hermes plugin dir)
├── __init__.py            # Hermes plugin entry point (lazy imports only)
├── plugin.yaml            # Hermes plugin manifest
├── conftest.py            # pytest boundary (do not delete)
├── galaxymem/             # the actual Python package
│   ├── __init__.py        # public API re-exports
│   ├── store_sqlite.py    # Store class (SQLite + FTS5 + sqlite-vec)
│   ├── embed.py           # embeddings (fastembed + fallback)
│   ├── ...
│   └── tests/
└── pyproject.toml
```

**Do NOT delete the root `conftest.py`.** Without it, pytest's prepend import
mode walks up from `galaxymem/tests/`, finds the plugin-root `__init__.py`,
treats the entire repo as a package named `galaxymem`, and shadows the real
inner package. The root `conftest.py` stops this walk.

**Do NOT move the plugin-root `__init__.py` import of `GalaxyMemProvider` to
module level.** It must stay inside `register()` as a lazy import. The inner
package's `galaxymem/__init__.py` wraps the provider import in a try/except so
the engine imports standalone (without Hermes's `agent.*` modules). The plugin
root follows the same pattern — import inside `register()` only.

---

## Common failures and fixes

### `ModuleNotFoundError: No module named 'galaxymem.provider'`

The editable install didn't map the package correctly. Reinstall:
```bash
pip install -e . --no-build-isolation
```

### `ModuleNotFoundError: No module named 'agent'`

Expected when running standalone (outside Hermes). The `provider.py` module
imports Hermes internals (`agent.memory_provider`, `tools.registry`). The
package guards this with try/except — `GalaxyMemProvider` will be `None` when
Hermes isn't present. This is fine for standalone use; you don't need the
provider class.

### `ModuleNotFoundError: No module named 'fastapi'`

The optional viewer isn't part of the core package. Install fastapi +
uvicorn separately only if you're running the viewer.

### SQLite lock errors

SQLite recovers from crashed writes automatically via its WAL journal on
next open. If `.db-wal`/`.db-shm` files linger after every process has
exited, delete them — the main `.sqlite3` file is always consistent.

---

## Configuration

All settings are environment variables prefixed `GALAXYMEM_`:

| Variable | Default | Purpose |
|---|---|---|
| `GALAXYMEM_DB_PATH` | `~/.galaxymem/db` | Where SQLite stores data |
| `GALAXYMEM_DECAY_HALF_LIFE_DAYS` | `30` | Memory fade rate |
| `GALAXYMEM_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `GALAXYMEM_HOT_CACHE_K` | `8` | Max memories in hot cache |
| `GALAXYMEM_PASS2_FLAG_THRESHOLD` | `12` | Flags before LLM extraction |
| `GALAXYMEM_REFLECT_VOLUME_TRIGGER` | `100` | New memories before reflection |

For Hermes Agent, config can also go in `$HERMES_HOME/galaxymem.json`.

---

## When modifying the codebase

1. **Read ARCHITECTURE.md first.** It explains the module boundaries and why
   the store has zero business logic.
2. **The store layer is Hermes-agnostic.** Only `provider.py` imports Hermes
   internals. Keep it that way.
3. **No new dependencies without justification.** The current deps are
   sqlite-vec, fastembed, pydantic, numpy, pandas, requests. The viewer adds
   fastapi + uvicorn.
4. **Python 3.10+ required.** The codebase uses `from __future__ import
   annotations` and union type syntax.

---

## Integration checklist for agents

When an AI agent is setting up GalaxyMem for a user, verify each item:

- [ ] `python3 --version` shows 3.10+
- [ ] `pip install -e .` completed without errors
- [ ] `python -c "from galaxymem.store_sqlite import Store"` succeeds
- [ ] `python -c "import galaxymem; print(galaxymem.__version__)"` prints `0.2.0`
- [ ] If Hermes: `GalaxyMemProvider` is not `None` when imported inside Hermes
- [ ] If standalone: `GalaxyMemProvider` is `None` (expected, not an error)
- [ ] The DB path is writable and has disk space (~500MB headroom)
