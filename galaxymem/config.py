"""All tunables. Overridable via GALAXYMEM_* env vars."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .sanitize import env_bool, env_float, env_int

# Storage
# Default to ~/.galaxymem/db for standalone use.
# When loaded as a Hermes plugin, provider.py overrides via resolve_db_path(hermes_home).
DB_PATH = Path(os.environ.get("GALAXYMEM_DB_PATH", str(Path.home() / ".galaxymem" / "db")))
EMBEDDING_BACKEND = os.environ.get("GALAXYMEM_EMBEDDING_BACKEND", "fastembed")
EMBEDDING_API_URL = os.environ.get("GALAXYMEM_EMBEDDING_API_URL", "")
EMBEDDING_API_KEY = os.environ.get("GALAXYMEM_EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("GALAXYMEM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = env_int("GALAXYMEM_EMBEDDING_DIM", 384, minimum=8, maximum=4096)

# Hot cache
HOT_CACHE_K = env_int("GALAXYMEM_HOT_CACHE_K", 8, minimum=1, maximum=64)
HOT_CACHE_TOKEN_BUDGET = env_int("GALAXYMEM_HOT_CACHE_TOKEN_BUDGET", 800, minimum=64, maximum=8000)

# Pass 2 triggers
PASS2_FLAG_THRESHOLD = env_int("GALAXYMEM_PASS2_FLAG_THRESHOLD", 12, minimum=1, maximum=1000)
PASS2_IDLE_MINUTES = env_int("GALAXYMEM_PASS2_IDLE_MINUTES", 20, minimum=1, maximum=24 * 60)
# A batch whose extraction failed this many times is parked (left
# unprocessed but never retried) instead of poisoning every trigger.
PASS2_MAX_ATTEMPTS = env_int("GALAXYMEM_PASS2_MAX_ATTEMPTS", 3, minimum=1, maximum=20)

# Decay — floor at 0.1 days so brightness never divides by zero.
DECAY_HALF_LIFE_DAYS = env_float("GALAXYMEM_DECAY_HALF_LIFE_DAYS", 30.0, minimum=0.1, maximum=36500.0)
BRIGHTNESS_FLOOR = env_float("GALAXYMEM_BRIGHTNESS_FLOOR", 0.15, minimum=0.0, maximum=1.0)

# Recall
RRF_K = env_int("GALAXYMEM_RRF_K", 60, minimum=1, maximum=1000)
ACTIVATION_MIN_WEIGHT = env_float("GALAXYMEM_ACTIVATION_MIN_WEIGHT", 0.4, minimum=0.0, maximum=1.0)
ACTIVATION_DAMPING = env_float("GALAXYMEM_ACTIVATION_DAMPING", 0.5, minimum=0.0, maximum=1.0)
RECALL_DEFAULT_K = env_int("GALAXYMEM_RECALL_DEFAULT_K", 8, minimum=1, maximum=100)
RECALL_SEARCH_K = env_int("GALAXYMEM_RECALL_SEARCH_K", 25, minimum=1, maximum=200)
# Temporal arm: parse date expressions from the query and fuse a
# window-ranked retrieval signal into RRF (hindsight-inspired).
TEMPORAL_ARM_ENABLED = env_bool("GALAXYMEM_TEMPORAL_ARM", True)
RECALL_TEMPORAL_K = env_int("GALAXYMEM_RECALL_TEMPORAL_K", 25, minimum=1, maximum=200)

# Reflect
REFLECT_CRON = os.environ.get("GALAXYMEM_REFLECT_CRON", "30 3 * * *")
REFLECT_VOLUME_TRIGGER = env_int("GALAXYMEM_REFLECT_VOLUME_TRIGGER", 100, minimum=1, maximum=100000)

# Promotion (Phase 8 — proposals into the EXISTING human-gated wiki pipeline, D12)
PROMOTION_MIN_RECALLS = env_int("GALAXYMEM_PROMOTION_MIN_RECALLS", 3, minimum=1, maximum=1000)
PROMOTION_MIN_CYCLES = env_int("GALAXYMEM_PROMOTION_MIN_CYCLES", 2, minimum=1, maximum=1000)
# Vault paths are Hermes-specific; make them optional for standalone use.
# Only used if the paths exist — promotion gracefully skips if vault is absent.
_VAULT_NOTES = os.environ.get("GALAXYMEM_VAULT_NOTES_PATH")
_VAULT_WIKI = os.environ.get("GALAXYMEM_WIKI_INDEX_PATH")
VAULT_NOTES_PATH = Path(_VAULT_NOTES) if _VAULT_NOTES else None
WIKI_INDEX_PATH = Path(_VAULT_WIKI) if _VAULT_WIKI else None

# Dedup
DEDUP_SIMILARITY_THRESHOLD = env_float(
    "GALAXYMEM_DEDUP_SIMILARITY_THRESHOLD", 0.92, minimum=0.0, maximum=1.0
)

# Query expansion
QUERY_EXPANSION_ENABLED = env_bool("GALAXYMEM_QUERY_EXPANSION", True)
QUERY_EXPANSION_MAX_TOKENS = env_int("GALAXYMEM_QUERY_EXPANSION_MAX_TOKENS", 150, minimum=16, maximum=2000)

# ── Reranker ────────────────────────────────────────────────────────────
# Set to empty string or "none"/"false" to disable reranking entirely.
RERANKER_MODEL = os.environ.get(
    "GALAXYMEM_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANKER_TOP_K = env_int("GALAXYMEM_RERANKER_TOP_K", 25, minimum=1, maximum=200)
RERANKER_SCORE_WEIGHT = env_float("GALAXYMEM_RERANKER_SCORE_WEIGHT", 0.7, minimum=0.0, maximum=1.0)
# If the reranker can't load (network offline, disk full, etc.) fall back to
# bi-encoder scores only with this toggle — no hard failure.
RERANKER_ALLOW_FALLBACK = env_bool("GALAXYMEM_RERANKER_ALLOW_FALLBACK", True)

# Provisional entity TTL — provisional entities older than this many days
# with zero associated memories are automatically cleaned up during reflection.
# Set to 0 to disable expiry (old behaviour).
PROVISIONAL_TTL_DAYS = env_int("GALAXYMEM_PROVISIONAL_TTL_DAYS", 90, minimum=0, maximum=36500)

# Max DB size limits — when hit, new memories are rejected.
MAX_MEMORIES = env_int("GALAXYMEM_MAX_MEMORIES", 50000, minimum=0, maximum=10_000_000)
MAX_ENTITIES = env_int("GALAXYMEM_MAX_ENTITIES", 500, minimum=0, maximum=1_000_000)
MAX_FLAGS_PER_SESSION = env_int("GALAXYMEM_MAX_FLAGS_PER_SESSION", 100, minimum=0, maximum=100000)
MAX_EDGES = env_int("GALAXYMEM_MAX_EDGES", 100000, minimum=0, maximum=10_000_000)
MAX_MEMORY_TEXT_CHARS = env_int("GALAXYMEM_MAX_MEMORY_TEXT_CHARS", 16000, minimum=256, maximum=200000)

# Entity creation suggestion
ENTITY_CREATION_MIN_RECURRING = env_int("GALAXYMEM_ENTITY_CREATION_MIN_RECURRING", 3, minimum=1, maximum=100)

# IVF-PQ needs a minimum corpus; below this we skip vector-index creation.
VECTOR_INDEX_MIN_ROWS = env_int("GALAXYMEM_VECTOR_INDEX_MIN_ROWS", 256, minimum=32, maximum=100000)


def resolve_db_path(hermes_home: Optional[str] = None) -> Path:
    """Resolve DB path, preferring hermes_home-scoped path if hermes_home is given."""
    if hermes_home:
        env_override = os.environ.get("GALAXYMEM_DB_PATH")
        if env_override:
            return Path(env_override)
        return Path(hermes_home) / "galaxymem" / "db"
    return DB_PATH
