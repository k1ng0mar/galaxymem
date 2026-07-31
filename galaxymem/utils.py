"""Shared utilities for GalaxyMem."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def ulid() -> str:
    """Generate a sortable ULID-like ID (timestamp + random suffix)."""
    timestamp = datetime.now(timezone.utc)
    suffix = uuid.uuid4().hex[:16]
    return f"{timestamp.strftime('%Y%m%d%H%M%S%f')}-{suffix}"


def log_and_retain(level: int, msg: str, *args, exc=None, **kwargs) -> None:
    """Log at the given level, including exception info when provided."""
    if exc is not None:
        msg = f"{msg} — {type(exc).__name__}: {exc}"
    logger.log(level, msg, *args, **kwargs)
