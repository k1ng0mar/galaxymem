"""All tunables. Overridable via GALAXYMEM_* env vars."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Storage
# Default to ~/.galaxymem/db for standalone use.
# When loaded as a Hermes plugin, provider.py overrides via resolve_db_path(hermes_home).
DB_PATH = Path(os.environ.get("GALAXYMEM_DB_PATH", str(Path.home() / ".galaxymem" / "db")))
EMBEDDING_BACKEND = os.environ.get("GALAXYMEM_EMBEDDING_BACKEND", "fastembed")
EMBEDDING_API_URL = os.environ.get("GALAXYMEM_EMBEDDING_API_URL", "")
EMBEDDING_MODEL = os.environ.get("GALAXYMEM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("GALAXYMEM_EMBEDDING_DIM", "384"))

# Hot cache
HOT_CACHE_K = int(os.environ.get("GALAXYMEM_HOT_CACHE_K", "8"))
HOT_CACHE_TOKEN_BUDGET = int(os.environ.get("GALAXYMEM_HOT_CACHE_TOKEN_BUDGET", "800"))

# Pass 2 triggers
PASS2_FLAG_THRESHOLD = int(os.environ.get("GALAXYMEM_PASS2_FLAG_THRESHOLD", "12"))
PASS2_IDLE_MINUTES = int(os.environ.get("GALAXYMEM_PASS2_IDLE_MINUTES", "20"))

# Decay
DECAY_HALF_LIFE_DAYS = float(os.environ.get("GALAXYMEM_DECAY_HALF_LIFE_DAYS", "30"))
BRIGHTNESS_FLOOR = float(os.environ.get("GALAXYMEM_BRIGHTNESS_FLOOR", "0.15"))

# Recall
RRF_K = int(os.environ.get("GALAXYMEM_RRF_K", "60"))
ACTIVATION_MIN_WEIGHT = float(os.environ.get("GALAXYMEM_ACTIVATION_MIN_WEIGHT", "0.4"))
ACTIVATION_DAMPING = float(os.environ.get("GALAXYMEM_ACTIVATION_DAMPING", "0.5"))
RECALL_DEFAULT_K = int(os.environ.get("GALAXYMEM_RECALL_DEFAULT_K", "8"))
RECALL_SEARCH_K = int(os.environ.get("GALAXYMEM_RECALL_SEARCH_K", "25"))

# Reflect
REFLECT_CRON = os.environ.get("GALAXYMEM_REFLECT_CRON", "30 3 * * *")
REFLECT_VOLUME_TRIGGER = int(os.environ.get("GALAXYMEM_REFLECT_VOLUME_TRIGGER", "100"))

# Promotion (Phase 8 — proposals into the EXISTING human-gated wiki pipeline, D12)
PROMOTION_MIN_RECALLS = int(os.environ.get("GALAXYMEM_PROMOTION_MIN_RECALLS", "3"))
PROMOTION_MIN_CYCLES = int(os.environ.get("GALAXYMEM_PROMOTION_MIN_CYCLES", "2"))
# Vault paths are Hermes-specific; make them optional for standalone use.
# Only used if the paths exist — promotion gracefully skips if vault is absent.
_VAULT_NOTES = os.environ.get("GALAXYMEM_VAULT_NOTES_PATH")
_VAULT_WIKI = os.environ.get("GALAXYMEM_WIKI_INDEX_PATH")
VAULT_NOTES_PATH = Path(_VAULT_NOTES) if _VAULT_NOTES else None
WIKI_INDEX_PATH = Path(_VAULT_WIKI) if _VAULT_WIKI else None

# Dedup
DEDUP_SIMILARITY_THRESHOLD = float(os.environ.get("GALAXYMEM_DEDUP_SIMILARITY_THRESHOLD", "0.92"))

# Query expansion
QUERY_EXPANSION_ENABLED = os.environ.get(
    "GALAXYMEM_QUERY_EXPANSION", "true"
).lower() not in ("false", "0", "no", "off")
QUERY_EXPANSION_MAX_TOKENS = int(os.environ.get("GALAXYMEM_QUERY_EXPANSION_MAX_TOKENS", "150"))

# ── Reranker ────────────────────────────────────────────────────────────
# Set to empty string or "none"/"false" to disable reranking entirely.
RERANKER_MODEL = os.environ.get(
    "GALAXYMEM_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANKER_TOP_K = int(os.environ.get("GALAXYMEM_RERANKER_TOP_K", "25"))
RERANKER_SCORE_WEIGHT = float(os.environ.get("GALAXYMEM_RERANKER_SCORE_WEIGHT", "0.7"))
# If the reranker can't load (network offline, disk full, etc.) fall back to
# bi-encoder scores only with this toggle — no hard failure.
RERANKER_ALLOW_FALLBACK = os.environ.get(
    "GALAXYMEM_RERANKER_ALLOW_FALLBACK", "true"
).lower() not in ("false", "0", "no", "off")

# Provisional entity TTL — provisional entities older than this many days
# with zero associated memories are automatically cleaned up during reflection.
# Set to 0 to disable expiry (old behaviour).
PROVISIONAL_TTL_DAYS = int(os.environ.get("GALAXYMEM_PROVISIONAL_TTL_DAYS", "90"))

# Max DB size limits — when hit, new memories are rejected.
MAX_MEMORIES = int(os.environ.get("GALAXYMEM_MAX_MEMORIES", "50000"))
MAX_ENTITIES = int(os.environ.get("GALAXYMEM_MAX_ENTITIES", "500"))
MAX_FLAGS_PER_SESSION = int(os.environ.get("GALAXYMEM_MAX_FLAGS_PER_SESSION", "100"))
MAX_EDGES = int(os.environ.get("GALAXYMEM_MAX_EDGES", "100000"))

# Entity creation suggestion
ENTITY_CREATION_MIN_RECURRING = int(os.environ.get("GALAXYMEM_ENTITY_CREATION_MIN_RECURRING", "3"))


def resolve_db_path(hermes_home: Optional[str] = None) -> Path:
    """Resolve DB path, preferring hermes_home-scoped path if hermes_home is given."""
    if hermes_home:
        env_override = os.environ.get("GALAXYMEM_DB_PATH")
        if env_override:
            return Path(env_override)
        return Path(hermes_home) / "galaxymem" / "db"
    return DB_PATH
