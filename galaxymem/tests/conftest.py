"""Shared pytest fixtures for GalaxyMem test suite.

Each test gets its own fresh temporary Store (LanceDB-backed). 
We never share DB state between tests.
"""

import pytest
from pathlib import Path
from datetime import datetime, timezone

from galaxymem.store import Store
from galaxymem.models import (
    MemoryRecord,
    EntityRecord,
    EntityType,
    Network,
    MemoryStatus,
    IdentityLink,
    LinkMethod,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a fresh Store in a unique tmp directory per test."""
    db_path = tmp_path / "test_galaxymem"
    store = Store(db_path=db_path)
    store.open(create_if_missing=True)
    yield store
    try:
        store.close()
    except Exception:
        pass


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide just the path, for tests that want to construct Store themselves."""
    return tmp_path / "test_galaxymem"


@pytest.fixture
def sample_memory():
    """A reusable MemoryRecord for tests."""
    return MemoryRecord(
        id="mem-test-001",
        text="The user prefers dark mode in all applications",
        network=Network.world,
        entity_ids=["user-1"],
        status=MemoryStatus.active,
        recall_count=0,
        reflect_cycles=0,
    )


@pytest.fixture
def sample_entity(temp_db):
    """A reusable EntityRecord created and stored."""
    entity = EntityRecord(
        id="alice",
        type=EntityType.person,
        label="Alice",
        card={"role": "engineer"},
        status_line="Software engineer at Acme",
        created_at=datetime.now(timezone.utc),
    )
    temp_db.add_entity(entity)
    return entity


@pytest.fixture
def sample_identity_link():
    """Reusable IdentityLink object (not auto-added)."""
    return IdentityLink(
        platform="telegram",
        external_id="123456",
        entity_id="alice",
        created_at=datetime.now(timezone.utc),
        created_by=LinkMethod.explicit,
    )
