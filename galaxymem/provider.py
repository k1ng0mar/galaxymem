"""GalaxyMem MemoryProvider — entity-scoped memory for Hermes Agent.

LanceDB-backed memory with four-network epistemic split (world / experience /
opinion / observation), decay-based relevance, spreading activation recall,
autonomous reflection, and promotion to external knowledge bases.

Pipeline:
  Pass 1 (real-time): flag_turn() applies rule-based heuristics to detect
      memorable content in conversation turns.
  Pass 2 (batched): process_pending_flags() sends flagged turns to an LLM for
      structured memory extraction.
  Recall: deep_recall() fuses vector + keyword search via RRF, then applies
      spreading activation through the memory graph.
  Reflect: run_reflection() detects contradictions, supersedes outdated facts,
      forms opinions from observations, demotes stale memories.
  Promote: run_promotion_cycle() exports high-value memories to wikis/Obsidian.

Config via $HERMES_HOME/galaxymem.json and GALAXYMEM_* env vars.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause calls
# for cooldown to avoid hammering a broken LanceDB or embedding backend.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(hermes_home: Optional[str] = None) -> dict:
    """Load config from env vars, with $HERMES_HOME/galaxymem.json overrides.

    Environment variables provide defaults; galaxymem.json (if present)
    overrides individual keys.
    """
    if hermes_home is None:
        try:
            from hermes_constants import get_hermes_home

            hermes_home = str(get_hermes_home())
        except ImportError:
            # Standalone mode — fall back to default DB path
            hermes_home = None

    from . import config as cfg

    if hermes_home is not None:
        default_db_path = str(Path(hermes_home) / "galaxymem" / "db")
        config_path = Path(hermes_home) / "galaxymem.json"
    else:
        # Standalone mode — use config defaults
        default_db_path = str(cfg.DB_PATH)
        config_path = None

    config: dict[str, Any] = {
        "db_path": os.environ.get("GALAXYMEM_DB_PATH", default_db_path),
        "embedding_backend": cfg.EMBEDDING_BACKEND,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "embedding_dim": cfg.EMBEDDING_DIM,
    }

    if config_path is not None and config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except (json.JSONDecodeError, OSError) as e:
            logger.error("GalaxyMem config load failed at %s: %s", config_path, e)
            raise RuntimeError(f"Invalid galaxymem.json at {config_path}: {e}") from e

    Path(config["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    return config


# ---------------------------------------------------------------------------
# LLM client adapter
# ---------------------------------------------------------------------------

# Auxiliary task key: users pin a provider/model via `auxiliary.galaxymem`
# in config.yaml (hermes model → Configure auxiliary models).
AUX_TASK_KEY = "galaxymem"

# Fallback defaults for the auxiliary task. Overridable via:
#   1. galaxymem.json: {"aux": {"provider": "...", "model": "..."}}
#   2. Environment variables: GALAXYMEM_AUX_PROVIDER / GALAXYMEM_AUX_MODEL
_AUX_ENV_PROVIDER = os.environ.get("GALAXYMEM_AUX_PROVIDER", "custom:kilo")
_AUX_ENV_MODEL = os.environ.get("GALAXYMEM_AUX_MODEL", "kilo-auto/free")


def _load_aux_defaults(hermes_home: Optional[str] = None) -> dict:
    """Load auxiliary LLM defaults from galaxymem.json; fall back to env/hardcoded."""
    if hermes_home:
        config_path = Path(hermes_home) / "galaxymem.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                aux = data.get("aux", {})
                if isinstance(aux, dict):
                    return {
                        "provider": aux.get("provider", _AUX_ENV_PROVIDER),
                        "model": aux.get("model", _AUX_ENV_MODEL),
                    }
            except Exception as e:
                logger.warning(
                    "Failed to load aux defaults from galaxymem.json: %s", e)
    return {"provider": _AUX_ENV_PROVIDER, "model": _AUX_ENV_MODEL}


class _LLMClientAdapter:
    """Adapts Hermes's auxiliary LLM plumbing to the .complete()/.chat()
    protocols expected by retain.py and reflect.py.

    Calls route through agent.auxiliary_client.call_llm(task="galaxymem"),
    which resolves provider/model from the registered auxiliary task config
    (free-tier default, user-overridable) with Hermes's own fallback chain.
    A ``model_complete`` kwarg to initialize(), when provided, overrides this
    (used by tests).
    """

    def __init__(self, complete_fn: Optional[Any] = None):
        self._complete_fn = complete_fn

    def _call(self, messages: list[dict[str, str]]) -> str:
        if self._complete_fn is not None:
            parts = [f"[{m.get('role', 'user')}] {m.get('content', '')}" for m in messages]
            return self._complete_fn("\n".join(parts))
        from agent.auxiliary_client import call_llm, extract_content_or_reasoning
        response = call_llm(task=AUX_TASK_KEY, messages=messages, max_tokens=2000)
        return extract_content_or_reasoning(response) or ""

    def complete(self, prompt: str) -> str:
        return self._call([{"role": "user", "content": prompt}])

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._call(messages)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

GM_STORE_SCHEMA = {
    "name": "gm_store",
    "description": (
        "Manually store a memory in GalaxyMem. Use for explicit facts, "
        "decisions, or durable information the agent should remember. "
        "Memories are entity-scoped and decay over time unless recalled."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The memory text to store.",
            },
            "network": {
                "type": "string",
                "description": (
                    "Epistemic network: 'world' (objective facts), "
                    "'experience' (events/actions), 'opinion' (preferences/views), "
                    "'observation' (patterns/insights). Default: world."
                ),
                "default": "world",
                "enum": ["world", "experience", "opinion", "observation"],
            },
            "entity_labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Entity labels to associate with this memory.",
                "default": [],
            },
        },
        "required": ["text"],
    },
}

GM_RECALL_SCHEMA = {
    "name": "gm_recall",
    "description": (
        "Deep recall from GalaxyMem. Uses hybrid vector + keyword search "
        "fused via Reciprocal Rank Fusion, with spreading activation through "
        "the memory graph. Returns ranked memories."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 8, max 50).",
                "default": 8,
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Entity ids/labels to scope the recall to (hard filter; "
                    "self and unscoped world facts are always included)."
                ),
                "default": [],
            },
            "as_of": {
                "type": "string",
                "description": (
                    "ISO timestamp for temporal recall — 'what did I believe "
                    "then'. Returns the memory state as of that moment."
                ),
            },
        },
        "required": ["query"],
    },
}

GM_REFLECT_SCHEMA = {
    "name": "gm_reflect",
    "description": (
        "Trigger a GalaxyMem reflection cycle. Analyzes memories for "
        "contradictions, supersedes outdated facts, forms opinions from "
        "observations, and demotes stale memories. Runs in background."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GM_CREATE_ENTITY_SCHEMA = {
    "name": "gm_create_entity",
    "description": (
        "Create a new entity in GalaxyMem (person, project, etc.). "
        "Entities scope memories and enable identity resolution."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "Human-readable name."},
            "entity_type": {
                "type": "string",
                "description": "Entity type: person, project, or self.",
                "default": "person",
                "enum": ["person", "project", "self"],
            },
            "status_line": {
                "type": "string",
                "description": "Short status summary.",
                "default": "",
            },
        },
        "required": ["label"],
    },
}

GM_MERGE_ENTITY_SCHEMA = {
    "name": "gm_merge_entity",
    "description": (
        "Merge two entities in GalaxyMem. Moves all identity links from "
        "source to target and marks source as merged. Explicit user action."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source_id": {
                "type": "string",
                "description": "Entity ID/slug to merge FROM (will be marked merged).",
            },
            "target_id": {
                "type": "string",
                "description": "Entity ID/slug to merge INTO (receives links).",
            },
        },
        "required": ["source_id", "target_id"],
    },
}

GM_ENTITY_CARD_SCHEMA = {
    "name": "gm_entity_card",
    "description": (
        "Get an entity card from GalaxyMem — full entity record, identity "
        "links, and associated memories. Use to profile a person or project."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "Entity ID/slug to look up.",
            },
        },
        "required": ["entity_id"],
    },
}

GM_STATS_SCHEMA = {
    "name": "gm_stats",
    "description": (
        "Show GalaxyMem memory statistics — total memories, entities, edges, "
        "memories per network/status, unprocessed flags, DB path."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GM_REFLECT_NOW_SCHEMA = {
    "name": "gm_reflect_now",
    "description": (
        "Trigger immediate GalaxyMem reflection + Pass 2 extraction. "
        "Processes pending flags via LLM, then runs conflict resolution "
        "(supersession/contradiction), opinion formation, and promotion "
        "nomination."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GM_FLUSH_SCHEMA = {
    "name": "gm_flush",
    "description": (
        "Force GalaxyMem Pass 2 extraction NOW: process all pending flagged "
        "turns into structured memories. Does not run reflection."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GM_FORGET_SCHEMA = {
    "name": "gm_forget",
    "description": (
        "Archive a memory by id (soft delete — excluded from recall but "
        "never destroyed). Use ONLY on explicit user request to forget "
        "something."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The memory id to archive."},
        },
        "required": ["memory_id"],
    },
}

GM_UPDATE_ENTITY_SCHEMA = {
    "name": "gm_update_entity",
    "description": (
        "Update an entity's card (merge JSON fields), status line, or label. "
        "Use for curated facts about a person/project — e.g. relationship "
        "tracking (warmth/depth/notes), role, timezone. Card updates merge: "
        "existing keys are kept unless overwritten; set a key to null to "
        "remove it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "Entity id/slug to update (e.g. 'self', 'ven').",
            },
            "card_patch": {
                "type": "object",
                "description": "JSON fields to merge into the entity card.",
            },
            "status_line": {
                "type": "string",
                "description": "Optional new one-line current status.",
            },
            "label": {
                "type": "string",
                "description": "Optional new display name.",
            },
        },
        "required": ["entity_id"],
    },
}

GM_LINK_IDENTITY_SCHEMA = {
    "name": "gm_link_identity",
    "description": (
        "Explicitly link a platform identity to an entity, e.g. 'this "
        "telegram user is Sarah' or 'that discord account is me'. If the "
        "identity currently points at an auto-created provisional entity, "
        "the provisional is merged into the target (memories move with it). "
        "Identity links are ONLY ever created this way — never inferred."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "description": "Platform name: cli, telegram, discord, ...",
            },
            "external_id": {
                "type": "string",
                "description": "The platform-native user id to link.",
            },
            "entity_id": {
                "type": "string",
                "description": "Target entity id/slug (e.g. 'self', 'sarah').",
            },
        },
        "required": ["platform", "external_id", "entity_id"],
    },
}

GM_EXPORT_SCHEMA = {
    "name": "gm_export",
    "description": (
        "Export all memories, entities, edges, flags and promotion queue "
        "to a JSON file. Used for backup, migration, or debugging."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "output_path": {
                "type": "string",
                "description": "File path to write the JSON export to.",
            },
        },
        "required": ["output_path"],
    },
}

GM_SESSION_SEARCH_SCHEMA = {
    "name": "gm_session_search",
    "description": (
        "Search past conversation sessions by keyword or approximate date. "
        "More relevant than gm_recall for 'when did we discuss X' or "
        "'find the session from last Tuesday' — uses session summaries, "
        "not individual memory vectors."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to match against session summaries.",
            },
            "limit": {
                "type": "integer",
                "description": "Max sessions to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# GalaxyMemProvider
# ---------------------------------------------------------------------------

class GalaxyMemProvider(MemoryProvider):
    """GalaxyMem — entity-scoped LanceDB memory with decay and reflection."""

    def __init__(self):
        self._config: Optional[dict] = None
        self._db_path: str = ""
        self._store: Optional[Any] = None  # Store instance
        self._session_id: str = "default"
        self._platform: str = "cli"
        self._speaker_external_id: str = "hermes-user"
        self._speaker_entity_id: Optional[str] = None
        self._provisional_notice: str = ""
        self._llm_client: Optional[_LLMClientAdapter] = None
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._reflect_thread: Optional[threading.Thread] = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        # FD leak prevention: release stale fragment readers periodically.
        # _store_lock serializes fd release against concurrent operations.
        self._store_lock = threading.RLock()  # guards _store + fd release
        self._gc_counter = 0
        self._gc_interval = 30  # gc.collect() every N fd-release calls

    @property
    def name(self) -> str:
        return "galaxymem"

    # -- availability ---------------------------------------------------

    def is_available(self) -> bool:
        """Check if LanceDB and fastembed are importable."""
        try:
            import lancedb  # noqa: F401
            import fastembed  # noqa: F401
            return True
        except ImportError:
            return False

    # -- config ---------------------------------------------------------

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to $HERMES_HOME/galaxymem.json."""
        from utils import atomic_json_write
        config_path = Path(hermes_home) / "galaxymem.json"
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not read existing galaxymem.json; overwriting.")
        existing.update(values)
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "db_path",
                "description": "Path to the GalaxyMem LanceDB database directory",
                "default": "$HERMES_HOME/galaxymem/db",
                "env_var": "GALAXYMEM_DB_PATH",
            },
            {
                "key": "embedding_backend",
                "description": "Embedding backend: 'fastembed' (local) or 'api' (external)",
                "default": "fastembed",
                "env_var": "GALAXYMEM_EMBEDDING_BACKEND",
            },
            {
                "key": "embedding_model",
                "description": "Embedding model name (fastembed or API)",
                "default": "BAAI/bge-small-en-v1.5",
                "env_var": "GALAXYMEM_EMBEDDING_MODEL",
            },
        ]

    # -- initialization -------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        self._config = _load_config(hermes_home)
        self._db_path = self._config["db_path"]
        self._session_id = session_id or "default"
        self._platform = kwargs.get("platform", "cli")
        self._speaker_external_id = kwargs.get("user_id", "hermes-user")
        # Per the MemoryProvider contract: only primary agent sessions may
        # write (cron/subagent turns would corrupt user representations).
        self._agent_context = kwargs.get("agent_context", "primary")

        # Wire up LLM client if gateway provided a complete function
        model_complete = kwargs.get("model_complete")
        self._llm_client = _LLMClientAdapter(complete_fn=model_complete)

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Close any previously-opened store before opening a new one,
        # so re-initialization doesn't leak file descriptors.
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None

        # Initialize the Store
        try:
            from .store import Store
            self._store = Store(db_path=Path(self._db_path))
            self._store.open(create_if_missing=True)
            logger.info("GalaxyMem store opened at %s", self._db_path)
        except Exception as e:
            logger.error("GalaxyMem store init failed: %s", e)
            self._store = None

        # Ensure self entity exists, then resolve the session's identity
        # (Phase 7): known (platform, external_id) → canonical entity;
        # unknown → provisional entity (D4), mentioned once so the user can
        # link it explicitly (D3 — links are never inferred).
        if self._store is not None:
            try:
                from .entities import ensure_self_entity, resolve_or_provision
                ensure_self_entity(self._store)
                if self._agent_context != "primary":
                    # Read-only session: resolve if linked, never provision
                    link = self._store.resolve_identity(
                        self._platform, self._speaker_external_id)
                    self._speaker_entity_id = link.entity_id if link else "self"
                    return
                entity_id, is_new = resolve_or_provision(
                    self._store, self._platform, self._speaker_external_id,
                )
                self._speaker_entity_id = entity_id
                entity = self._store.get_entity(entity_id)
                if entity is not None and entity.type.value == "provisional":
                    self._provisional_notice = (
                        f"Messages on {self._platform} from id "
                        f"'{self._speaker_external_id}' are filed under the "
                        f"provisional entity '{entity_id}'. If you know who this "
                        f"is, call gm_link_identity(platform='{self._platform}', "
                        f"external_id='{self._speaker_external_id}', "
                        f"entity_id=<who>) to merge it (use 'self' for the owner)."
                    )
            except Exception as e:
                logger.error("GalaxyMem identity resolution failed: %s", e)

    # -- circuit breaker --------------------------------------------------

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "GalaxyMem circuit breaker tripped after %d consecutive failures. "
                "Pausing for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    # -- error logging helper --------------------------------------------

    @staticmethod
    def _warn_on_exc(msg: str, exc: Exception, *args) -> None:
        """Log a warning for swallowed exceptions so they're not invisible."""
        logger.warning("%s — %s: %s", msg, type(exc).__name__, exc, *args)

    # -- fragment FD management -------------------------------------------

    def _maybe_release_fragment_fds(self) -> None:
        """Release stale fragment readers without destroying the store.

        LanceDB 0.34 opens lance data fragment readers on every search/query
        and doesn't reliably close them. Over time these accumulate to the
        process ulimit. Calling close_lsm_writers() on each table releases
        the fragment handles WITHOUT destroying the store or invalidating
        existing references — background threads keep working.

        Thread-safety: acquires _store_lock so concurrent tool calls /
        background threads don't see a half-refreshed store.
        """
        if self._store is None:
            return
        with self._store_lock:
            if self._store is None:
                return
            try:
                # close_lsm_writers is on the LanceTable (sync wrapper) itself
                for tbl_name in ("memories", "entities", "edges", "hot_cache",
                                 "flags", "promotion_queue", "identity_links"):
                    try:
                        tbl = getattr(self._store, f"_{tbl_name}")
                        if tbl is not None and hasattr(tbl, "close_lsm_writers"):
                            tbl.close_lsm_writers()
                    except Exception as e:
                        self._warn_on_exc(
                            f"GalaxyMem fd release failed for {tbl_name}", e)
                # Periodically force GC to release Python-side references to
                # lance fragment readers that close_lsm_writers doesn't catch.
                self._gc_counter += 1
                if self._gc_counter >= self._gc_interval:
                    self._gc_counter = 0
                    import gc
                    gc.collect()
                logger.debug("GalaxyMem fragment readers released")
            except Exception as e:
                self._warn_on_exc("GalaxyMem fd release failed", e)

    # -- system prompt -----------------------------------------------------

    def system_prompt_block(self) -> str:
        parts = [
            "# GalaxyMem Memory",
            f"Active. DB: {self._db_path}.",
            "Use gm_recall to search memories (entities=[...] to scope, "
            "as_of<<ISO date> for 'what did I believe then'), gm_store to "
            "manually store facts, gm_forget to archive one, gm_entity_card "
            "to profile a person/project, gm_link_identity to map platform "
            "ids to entities, gm_stats for statistics, gm_flush to force "
            "extraction, gm_reflect_now for extraction+reflection.",
        ]

        # Hot path (Phase 4): inject working memory for the active entities —
        # self + the resolved conversation partner — within the token budget.
        if self._store is not None and not self._is_breaker_open():
            try:
                from .recall import get_hot_cache, format_memories_for_prompt
                from .summaries import get_summary

                active_entities = ["self"]
                if self._speaker_entity_id and self._speaker_entity_id != "self":
                    active_entities.append(self._speaker_entity_id)

                # Session summary injection: inject a compressed summary of the
                # current session before the hot cache. This gives turn 1 of a
                # new conversation the same context as turn 20 — the "what we
                # were talking about just now" that vector recall misses.
                session_summary = get_summary(self._store, self._session_id)
                if session_summary and session_summary.get("text"):
                    parts.append(
                        f"**Current session context** (rolling summary):\n"
                        f"{session_summary['text']}"
                    )

                with self._store_lock:
                    memories = get_hot_cache(self._store, entity_ids=active_entities)
                rendered = format_memories_for_prompt(memories)
                if rendered:
                    parts.append(rendered)

                # Gap detection: neighbors of recalled memories that weren't
                # recalled this turn. Mnemosyne-style associative recall —
                # "you should ALSO remember Y because X came up". Uses edges
                # that were crossed during spreading activation but their
                # target memory wasn't in the final top-k.
                try:
                    from .procedural import detect_gaps
                    gaps = detect_gaps(self._store, memories)
                    if gaps:
                        parts.append("**Also remember (linked context):**\n" + "\n".join(gaps))
                except Exception as e:
                    logger.debug("Gap detection skipped: %s", e)

                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("GalaxyMem hot cache injection failed: %s", e)

        if self._provisional_notice:
            parts.append(f"NOTE: {self._provisional_notice}")

        return "\n".join(parts)

    # -- prefetch ----------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return pre-computed recall results from queue_prefetch()."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## GalaxyMem Memory (recalled for: {query!r})\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background deep_recall for the next turn."""
        if self._is_breaker_open():
            return
        if not query or not query.strip():
            return
        if self._store is None:
            return

        def _run() -> None:
            try:
                from .recall import deep_recall
                with self._store_lock:
                    results = deep_recall(query, self._store)
                if results:
                    lines = []
                    for mem in results:
                        tag = mem.network.value[0].upper()
                        lines.append(f"- [{tag}] {mem.text}")
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("GalaxyMem prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="galaxymem-prefetch",
        )
        self._prefetch_thread.start()

    # -- sync_turn -----------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Feed the turn to the retain pipeline (Pass 1 flagging).

        Non-blocking — flags the turn if it matches memorability heuristics.
        Pass 2 extraction runs later via on_session_end or gm_reflect_now.
        """
        if self._is_breaker_open():
            return
        if not user_content or not self._store:
            return
        if self._agent_context != "primary":
            return  # cron/subagent turns never write memories

        def _sync() -> None:
            try:
                from .retain import flag_turn, should_trigger_pass2, process_pending_flags
                from .summaries import update_summary
                # Rolling session summary: cheap concatenation, no LLM cost per turn.
                # LLM compression only fires when the summary overflows the cap
                # and the aux model is available.
                from .summaries import _llm_compress
                _compress_fn = None
                try:
                    if self._llm_client:
                        _compress_fn = lambda p: self._llm_client.complete(p)
                except (AttributeError, TypeError):
                    pass  # lambda closure over a None is fine to ignore
                update_summary(
                    self._store,
                    session_id=session_id or self._session_id,
                    user_message=user_content,
                    assistant_message=assistant_content,
                    llm_summarize_fn=_compress_fn,
                )

                # Pass 1: flag the turn for later rich extraction
                from .retain import flag_turn, should_trigger_pass2, process_pending_flags
                turn_text = f"{user_content}\n{assistant_content}".strip()
                sid = session_id or self._session_id
                with self._store_lock:
                    flag_turn(
                        store=self._store,
                        turn_text=turn_text,
                        session_id=sid,
                        platform=self._platform,
                        speaker_external_id=self._speaker_external_id,
                    )
                    # Mid-session Pass 2: fire on flag-count/idle thresholds
                    # instead of waiting for session end (spec D5 triggers).
                    if self._llm_client is not None and should_trigger_pass2(self._store, sid):
                        count = process_pending_flags(
                            self._store, self._llm_client, session_id=sid,
                        )
                        if count:
                            logger.info("GalaxyMem mid-session Pass 2: %d memories", count)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("GalaxyMem sync_turn failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="galaxymem-sync",
        )
        self._sync_thread.start()

    # -- session end ---------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Run Pass 2 extraction, reflection, and promotion at session end.

        Blocking — caller expects this to complete before teardown.
        """
        if self._store is None:
            return
        if self._agent_context != "primary":
            return  # cron/subagent sessions never run extraction/reflection

        with self._store_lock:
            # Pass 2: extract memories from flagged turns
            try:
                from .retain import should_trigger_pass2, process_pending_flags
                if should_trigger_pass2(self._store) and self._llm_client is not None:
                    count = process_pending_flags(
                        self._store, self._llm_client,
                        session_id=self._session_id,
                    )
                    if count > 0:
                        logger.info("GalaxyMem Pass 2: extracted %d memories", count)
            except Exception as e:
                self._warn_on_exc("GalaxyMem Pass 2 at session end failed", e)

            # Reflection: contradictions, supersessions, opinions, demotion
            try:
                from .reflect import should_reflect, run_reflection
                if should_reflect(self._store) and self._llm_client is not None:
                    report = run_reflection(self._store, self._llm_client)
                    logger.info("GalaxyMem reflection: %s", report)
            except Exception as e:
                self._warn_on_exc("GalaxyMem reflection at session end failed", e)

            # Promotion: export high-value memories to external KBs
            try:
                from .promote import run_promotion_cycle
                run_promotion_cycle(self._store)
            except Exception as e:
                self._warn_on_exc("GalaxyMem promotion at session end failed", e)

    # -- tool schemas -----------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            GM_STORE_SCHEMA,
            GM_RECALL_SCHEMA,
            GM_REFLECT_SCHEMA,
            GM_CREATE_ENTITY_SCHEMA,
            GM_MERGE_ENTITY_SCHEMA,
            GM_ENTITY_CARD_SCHEMA,
            GM_STATS_SCHEMA,
            GM_REFLECT_NOW_SCHEMA,
            GM_FLUSH_SCHEMA,
            GM_FORGET_SCHEMA,
            GM_LINK_IDENTITY_SCHEMA,
            GM_UPDATE_ENTITY_SCHEMA,
            GM_EXPORT_SCHEMA,
            GM_SESSION_SEARCH_SCHEMA,
        ]

    # -- tool dispatch ------------------------------------------------------------

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": (
                    "GalaxyMem temporarily unavailable (multiple consecutive "
                    "failures). Will retry automatically."
                ),
            })

        if self._store is None:
            return tool_error(
                "GalaxyMem store not initialized. Check $HERMES_HOME/galaxymem.json "
                "and DB path permissions."
            )

        try:
            # -- gm_reflect (background) — spawns its own thread with its
            # own _store_lock acquisition, so dispatch OUTSIDE the lock.
            if tool_name == "gm_reflect":
                return self._handle_reflect(args)

            # All other handlers access self._store synchronously. Hold the
            # lock so bg threads (prefetch, sync, reflect) don't race.
            with self._store_lock:
                if tool_name == "gm_store":
                    return self._handle_store(args)

                elif tool_name == "gm_recall":
                    return self._handle_recall(args)

                elif tool_name == "gm_reflect_now":
                    return self._handle_reflect_now(args)

                elif tool_name == "gm_create_entity":
                    return self._handle_create_entity(args)

                elif tool_name == "gm_merge_entity":
                    return self._handle_merge_entity(args)

                elif tool_name == "gm_entity_card":
                    return self._handle_entity_card(args)

                elif tool_name == "gm_stats":
                    return self._handle_stats(args)

                elif tool_name == "gm_flush":
                    return self._handle_flush(args)

                elif tool_name == "gm_forget":
                    return self._handle_forget(args)

                elif tool_name == "gm_link_identity":
                    return self._handle_link_identity(args)

                elif tool_name == "gm_update_entity":
                    return self._handle_update_entity(args)

                elif tool_name == "gm_export":
                    return self._handle_export(args)

                elif tool_name == "gm_session_search":
                    return self._handle_session_search(args)

                return tool_error(f"Unknown tool: {tool_name}")

        except Exception as e:
            self._record_failure()
            self._warn_on_exc(f"GalaxyMem tool '{tool_name}' failed", e)
            return tool_error(f"{type(e).__name__}: {e}")

    # -- tool handlers -------------------------------------------------------------

    def _handle_store(self, args: dict) -> str:
        """gm_store: manually store a memory."""
        from .models import MemoryRecord, MemoryStatus, Network
        from .entities import ensure_self_entity
        from .store import _ulid

        text = (args.get("text") or "").strip()
        if not text:
            return tool_error("Missing required parameter: text")

        network_str = args.get("network", "world")
        try:
            network = Network(network_str)
        except ValueError:
            network = Network.world

        # Resolve labels to EXISTING entities only — memories are scoped by
        # who/what they are ABOUT (D7), and entities are never auto-created
        # from labels (D3). Unresolved labels stay in the text.
        ensure_self_entity(self._store)
        entity_ids = self._resolve_entity_args(args.get("entity_labels", []))

        mem = MemoryRecord(
            id=_ulid(),
            text=text,
            network=network,
            entity_ids=entity_ids,
            source_session_id=self._session_id,
            source_platform=self._platform,
            speaker_entity_id=self._speaker_entity_id,
            status=MemoryStatus.active,
        )
        self._store.add_memory(mem)
        self._record_success()
        self._maybe_release_fragment_fds()
        return json.dumps({
            "result": "Memory stored.",
            "id": mem.id,
            "network": network.value,
            "entity_ids": entity_ids,
        })

    def _handle_recall(self, args: dict) -> str:
        """gm_recall: deep recall with RRF, decay boost, spreading activation.

        entities=[...] applies the D8 hard filter; as_of=<ISO> runs the same
        pipeline against the historical store (temporal mode, no touching).
        """
        from datetime import datetime as _dt, timezone
        from .recall import recall as do_recall
        from .queryexpansion import expand_query, should_expand

        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required parameter: query")

        limit = min(int(args.get("limit", 8)), 50)

        entity_ids = self._resolve_entity_args(args.get("entities") or [])

        as_of = None
        as_of_raw = (args.get("as_of") or "").strip()
        if as_of_raw:
            try:
                as_of = _dt.fromisoformat(as_of_raw)
            except ValueError:
                return tool_error(f"Invalid as_of timestamp: {as_of_raw!r} (use ISO format)")

        try:
            # Query expansion: use the aux LLM to broaden the query before
            # vector/keyword search. One cheap (~150 tok) call that closes
            # the "what the user means" → "what the memory actually says" gap.
            if self._llm_client is not None and should_expand(query, entity_ids):
                try:
                    query = expand_query(query, self._llm_client)
                except Exception as e:
                    logger.debug("Query expansion skipped: %s", e)

            results = do_recall(query, self._store, entity_ids=entity_ids or None,
                                limit=limit, as_of=as_of)
        except (ValueError, RuntimeError) as e:
            # Temporal-mode errors are user-meaningful (no version that old, etc.)
            return tool_error(f"Recall failed: {e}")
        except Exception as e:
            self._record_failure()
            self._warn_on_exc("GalaxyMem recall failed", e)
            return tool_error(f"Recall failed: {e}")

        self._record_success()
        self._maybe_release_fragment_fds()
        if not results:
            return json.dumps({"results": [], "count": 0, "query": query})

        now = _dt.now(timezone.utc)
        # Confidence scoring: augment each memory with its confidence metadata.
        # The blend (sources × edges × reflections × status) replaces the
        # simplistic "recall_count × brightness" ranking for trust-aware
        # retrieval — a memory with 5 sources and 3 reflections outranks one
        # with 1 source from yesterday even if it's buried deeper in time.
        items = []
        for mem in results:
            from .confidence import compute_confidence
            conf = compute_confidence(mem, self._store)
            items.append({
                "id": mem.id,
                "text": mem.text,
                "network": mem.network.value,
                "status": mem.status.value,
                "entity_ids": mem.entity_ids,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "age_days": (now - mem.created_at).days if mem.created_at else None,
                "recall_count": mem.recall_count,
                "canonical_key": mem.canonical_key,
                "reflect_cycles": mem.reflect_cycles,
                "confidence": conf,
                "confidence_tier": classify_confidence(conf),
            })
        payload = {"results": items, "count": len(items), "query": query}
        if as_of is not None:
            payload["as_of"] = as_of.isoformat()
        return json.dumps(payload)

    def _resolve_entity_args(self, raw: list[str]) -> list[str]:
        """Resolve tool-supplied entity ids/labels to entity ids."""
        from .entities import _slugify

        resolved: list[str] = []
        for item in raw:
            item = (item or "").strip()
            if not item:
                continue
            entity = self._store.get_entity(item) or \
                self._store.get_entity_by_label(item) or \
                self._store.get_entity(_slugify(item))
            if entity is not None:
                if entity.merged_into:
                    entity = self._store.get_entity(entity.merged_into) or entity
                if entity.id not in resolved:
                    resolved.append(entity.id)
        return resolved

    def _handle_flush(self, args: dict) -> str:
        """gm_flush: force Pass 2 extraction now."""
        from .retain import process_pending_flags

        if self._llm_client is None:
            return tool_error("No LLM available for extraction")
        count = process_pending_flags(self._store, self._llm_client,
                                      session_id=self._session_id)
        self._record_success()
        return json.dumps({"result": "Pass 2 complete.", "memories_created": count})

    def _handle_forget(self, args: dict) -> str:
        """gm_forget: archive a memory (soft, D13 — explicit user intent only)."""
        memory_id = (args.get("memory_id") or "").strip()
        if not memory_id:
            return tool_error("Missing required parameter: memory_id")

        mem = self._store.get_memory(memory_id)
        if mem is None:
            return tool_error(f"Memory '{memory_id}' not found")

        self._store.delete_memory(memory_id)  # sets status=archived
        self._record_success()
        return json.dumps({
            "result": "Memory archived (soft — recoverable, never hard-deleted).",
            "id": memory_id,
        })

    def _handle_link_identity(self, args: dict) -> str:
        """gm_link_identity: explicit identity link (D3). Linking away from a
        provisional merges the provisional into the target."""
        from .entities import link_identity_explicit, merge_entity
        from .models import EntityType

        platform = (args.get("platform") or "").strip()
        external_id = str(args.get("external_id") or "").strip()
        entity_id = (args.get("entity_id") or "").strip()
        if not platform or not external_id or not entity_id:
            return tool_error("Missing required parameters: platform, external_id, entity_id")

        target = self._store.get_entity(entity_id)
        if target is None:
            return tool_error(f"Entity '{entity_id}' not found — create it first with gm_create_entity")

        merged_from = None
        existing_link = self._store.resolve_identity(platform, external_id)
        if existing_link is not None:
            if existing_link.entity_id == entity_id:
                return json.dumps({"result": "Already linked.", "entity_id": entity_id})
            current = self._store.get_entity(existing_link.entity_id)
            if current is not None and current.type == EntityType.provisional:
                merge_entity(self._store, existing_link.entity_id, entity_id)
                merged_from = existing_link.entity_id
            else:
                return tool_error(
                    f"{platform}:{external_id} is already explicitly linked to "
                    f"'{existing_link.entity_id}'. Merge entities explicitly if intended."
                )
        else:
            link_identity_explicit(self._store, platform, external_id, entity_id)

        # If this session's speaker was that identity, update the resolution
        if platform == self._platform and external_id == self._speaker_external_id:
            self._speaker_entity_id = entity_id
            self._provisional_notice = ""

        self._record_success()
        return json.dumps({
            "result": "Identity linked.",
            "platform": platform,
            "external_id": external_id,
            "entity_id": entity_id,
            "merged_provisional": merged_from,
        })

    def _handle_reflect(self, args: dict) -> str:
        """gm_reflect: queue reflection in background thread."""
        def _run() -> None:
            try:
                from .retain import should_trigger_pass2, process_pending_flags
                from .reflect import should_reflect, run_reflection

                with self._store_lock:
                    if should_trigger_pass2(self._store) and self._llm_client:
                        process_pending_flags(
                            self._store, self._llm_client,
                            session_id=self._session_id,
                        )
                    if should_reflect(self._store) and self._llm_client:
                        run_reflection(self._store, self._llm_client)

                self._record_success()
            except Exception as e:
                self._record_failure()
                self._warn_on_exc("GalaxyMem background reflect failed", e)

        t = threading.Thread(target=_run, daemon=True, name="galaxymem-reflect")
        self._reflect_thread = t  # tracked for shutdown()
        t.start()
        return json.dumps({
            "result": "Reflection queued. Returns immediately; runs in background.",
        })

    def _handle_reflect_now(self, args: dict) -> str:
        """gm_reflect_now: immediate Pass 2 + reflection (blocking)."""
        report: dict[str, Any] = {
            "pass2_memories": 0,
            "reflection": {},
        }

        # Pass 2 extraction
        try:
            from .retain import process_pending_flags
            if self._llm_client is not None:
                report["pass2_memories"] = process_pending_flags(
                    self._store, self._llm_client,
                    session_id=self._session_id,
                )
        except Exception as e:
            report["pass2_error"] = str(e)
            self._record_failure()
            self._warn_on_exc("Pass 2 failed", e)

        # Reflection
        try:
            from .reflect import run_reflection
            if self._llm_client is not None:
                report["reflection"] = run_reflection(self._store, self._llm_client)
        except Exception as e:
            report["reflection_error"] = str(e)
            self._record_failure()
            self._warn_on_exc("Reflection failed", e)

        self._record_success()
        return json.dumps(report, default=str)

    def _handle_create_entity(self, args: dict) -> str:
        """gm_create_entity: create a new entity."""
        from .entities import create_entity
        from .models import EntityType

        label = (args.get("label") or "").strip()
        if not label:
            return tool_error("Missing required parameter: label")

        type_str = args.get("entity_type", "person")
        # Map "self" to EntityType.self_
        if type_str == "self":
            entity_type = EntityType.self_
        else:
            try:
                entity_type = EntityType(type_str)
            except ValueError:
                entity_type = EntityType.person

        status_line = args.get("status_line", "")

        entity = create_entity(
            store=self._store,
            label=label,
            entity_type=entity_type,
            status_line=status_line,
        )
        self._record_success()
        return json.dumps({
            "result": "Entity created.",
            "id": entity.id,
            "label": entity.label,
            "type": entity.type.value,
        })

    def _handle_merge_entity(self, args: dict) -> str:
        """gm_merge_entity: merge source entity into target."""
        from .entities import merge_entity

        source_id = (args.get("source_id") or "").strip()
        target_id = (args.get("target_id") or "").strip()
        if not source_id or not target_id:
            return tool_error("Missing required parameters: source_id and target_id")

        target = merge_entity(self._store, source_id, target_id)
        self._record_success()
        return json.dumps({
            "result": "Entities merged.",
            "source_id": source_id,
            "target_id": target.id,
            "target_label": target.label,
        })

    def _handle_update_entity(self, args: dict) -> str:
        """gm_update_entity: merge card fields / set status_line / rename."""
        entity_id = (args.get("entity_id") or "").strip()
        if not entity_id:
            return tool_error("Missing required parameter: entity_id")

        entity = self._store.get_entity(entity_id)
        if entity is None:
            return tool_error(f"Entity '{entity_id}' not found")

        updates: dict = {}
        card_patch = args.get("card_patch")
        if isinstance(card_patch, dict) and card_patch:
            card = dict(entity.card)
            for k, v in card_patch.items():
                if v is None:
                    card.pop(k, None)
                else:
                    card[k] = v
            updates["card"] = card
        status_line = args.get("status_line")
        if isinstance(status_line, str) and status_line.strip():
            updates["status_line"] = status_line.strip()[:200]
        label = args.get("label")
        if isinstance(label, str) and label.strip():
            updates["label"] = label.strip()

        if not updates:
            return tool_error("Nothing to update: provide card_patch, status_line, or label")

        self._store.update_entity(entity_id, **updates)
        updated = self._store.get_entity(entity_id)
        self._record_success()
        return json.dumps({
            "result": "Entity updated.",
            "id": updated.id,
            "label": updated.label,
            "status_line": updated.status_line,
            "card": updated.card,
        })

    def _handle_entity_card(self, args: dict) -> str:
        """gm_entity_card: get entity record + identity links + memories."""
        from .entities import get_entity_card

        entity_id = (args.get("entity_id") or "").strip()
        if not entity_id:
            return tool_error("Missing required parameter: entity_id")

        card = get_entity_card(self._store, entity_id)
        if card is None:
            return tool_error(f"Entity '{entity_id}' not found")

        # Also fetch memories for this entity
        memories = self._store.list_memories(entity_ids=[entity_id])
        card["memories"] = [
            {
                "id": m.id,
                "text": m.text,
                "network": m.network.value,
                "status": m.status.value,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "recall_count": m.recall_count,
            }
            for m in memories
        ]
        card["memory_count"] = len(memories)

        self._record_success()
        return json.dumps(card, default=str)

    def _handle_stats(self, args: dict) -> str:
        """gm_stats: memory statistics."""
        try:
            stats = self._store.stats()
        except Exception as e:
            self._record_failure()
            self._warn_on_exc("GalaxyMem stats failed", e)
            return tool_error(f"Stats failed: {e}")

        self._record_success()
        return json.dumps({"stats": stats, "db_path": self._db_path})

    def _handle_session_search(self, args: dict) -> str:
        """gm_session_search: search past sessions by keyword."""
        from .summaries import search_sessions_by_text

        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required parameter: query")

        limit = min(int(args.get("limit", 5)), 20)
        matches = search_sessions_by_text(self._store, query, limit=limit)

        results = []
        for s in matches:
            results.append({
                "session_id": s.get("id", "unknown"),
                "summary": s.get("text", "")[:200] + ("..." if len(s.get("text", "")) > 200 else ""),
                "message_count": s.get("message_count", 0),
                "last_updated": s.get("last_updated", ""),
            })

        self._record_success()
        return json.dumps({
            "results": results,
            "count": len(results),
            "query": query,
        })

    def _handle_export(self, args: dict) -> str:
        """gm_export: export all tables to JSON."""
        output_path_raw = (args.get("output_path") or "").strip()
        if not output_path_raw:
            return tool_error("Missing required parameter: output_path")

        output_path = Path(output_path_raw)
        try:
            export_data = {}
            # Memories
            export_data["memories"] = [
                m.model_dump(mode="json")
                for m in self._store.list_memories()
            ]
            # Entities
            export_data["entities"] = [
                e.model_dump(mode="json")
                for e in self._store.list_entities()
            ]
            # Edges
            edges = []
            for mem in self._store.list_memories():
                edges.extend(self._store.get_edges_for_memory(mem.id))
            # Deduplicate edges (from_id, to_id, kind)
            seen_edges = set()
            unique_edges = []
            for e in edges:
                key = (e.from_id, e.to_id, e.kind.value)
                if key not in seen_edges:
                    seen_edges.add(key)
                    unique_edges.append(e.model_dump(mode="json"))
            export_data["edges"] = unique_edges
            # Identity links
            export_data["identity_links"] = [
                link.model_dump(mode="json")
                for link in self._store._identities.search().to_pandas().itertuples()
            ] if False else []  # safer to iterate entities
            # Promotion queue
            export_data["promotion_queue"] = [
                q.model_dump(mode="json")
                for q in self._store.list_promotion_queue()
            ]
            # Flags
            export_data["flags"] = [
                f.model_dump(mode="json")
                for f in self._store.unprocessed_flags()
            ]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(export_data, indent=2, default=str),
                encoding="utf-8",
            )
            self._record_success()
            return json.dumps({
                "result": "Export complete.",
                "path": str(output_path),
                "memories": len(export_data["memories"]),
                "entities": len(export_data["entities"]),
                "edges": len(export_data["edges"]),
            })
        except Exception as e:
            self._record_failure()
            self._warn_on_exc("Export failed", e)
            return tool_error(f"Export failed: {e}")

    # -- shutdown ---------------------------------------------------------------

    def shutdown(self) -> None:
        """Wait for background threads, close the store.

        All bg threads acquire _store_lock before touching the store, so
        holding it here guarantees no thread is mid-operation when we close.
        """
        # Wait for bg threads to finish their current work (they'll be
        # holding or waiting for _store_lock). timeout=10 gives the reflect
        # thread extra room since it does LLM calls.
        for t in (self._prefetch_thread, self._sync_thread, self._reflect_thread):
            if t and t.is_alive():
                t.join(timeout=10.0)

        # Now acquire the lock so no bg thread can start new work while we close.
        with self._store_lock:
            if self._store is not None:
                try:
                    self._store.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register GalaxyMem as a memory provider plugin."""
    ctx.register_memory_provider(GalaxyMemProvider())

    # LLM-backed extraction/reflection routes through auxiliary.galaxymem.
    # Defaults are loaded from galaxymem.json via _load_aux_defaults; the user
    # can repoint via `hermes model → Configure auxiliary models` or env vars.
    aux_defaults = _load_aux_defaults(getattr(ctx, "hermes_home", None))
    try:
        ctx.register_auxiliary_task(
            AUX_TASK_KEY,
            display_name="GalaxyMem extraction",
            description="Memory extraction + reflection LLM for the galaxymem provider",
            defaults=aux_defaults,
        )
    except AttributeError:
        logger.debug("PluginContext.register_auxiliary_task unavailable; "
                     "auxiliary.galaxymem must be configured manually")
