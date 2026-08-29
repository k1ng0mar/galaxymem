"""GalaxyMem — entity-scoped memory provider for AI agents.

LanceDB-backed, four-network epistemic split, decay-based relevance,
autonomous reflection. Built for Hermes Agent but usable standalone.
"""

PACKAGE_NAME = "galaxymem"
__version__ = "0.1.1"

# `provider` is the ONLY Hermes-coupled module — it imports agent.* / tools.*
# at module load. Guard it so the engine (store / retain / recall / reflect /
# promote) imports and runs standalone, without a Hermes Agent runtime present.
try:  # pragma: no cover - depends on host environment
    from .provider import GalaxyMemProvider
except ImportError:
    GalaxyMemProvider = None  # type: ignore[assignment]

from .store import Store
from .models import (
    MemoryRecord,
    EntityRecord,
    EdgeRecord,
    IdentityLink,
    Network,
    MemoryStatus,
    EntityType,
    EdgeKind,
    LinkMethod,
)
from .config import (
    DB_PATH,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

__all__ = [
    "GalaxyMemProvider",
    "Store",
    "MemoryRecord",
    "EntityRecord",
    "EdgeRecord",
    "IdentityLink",
    "Network",
    "MemoryStatus",
    "EntityType",
    "EdgeKind",
    "LinkMethod",
    "DB_PATH",
    "EMBEDDING_BACKEND",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
]
