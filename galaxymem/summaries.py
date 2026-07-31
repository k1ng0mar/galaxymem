"""Rolling session summaries — Honcho's "summary" pipeline, GalaxyMem-style.

Maintains a compressed, per-session summary of the conversation. Stored in
the dedicated `session_summaries` table. Updated as messages accumulate —
keeps a rolling ~1500 char window so turn 1 of a new conversation has the
same context as turn 20.

If an LLM summarization fn is provided, compression fires when the buffer
overflows the cap (no cost per-turn; only when the summary gets too big).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .store import Store

logger = logging.getLogger(__name__)

_SUMMARY_MAX_CHARS = 1500
_UPDATE_CHARS_PER_TURN = 400


def update_summary(
    store: Store,
    session_id: str,
    user_message: str,
    assistant_message: str,
    *,
    llm_summarize_fn=None,
) -> None:
    """Append a turn to the rolling session summary, compressing if needed.

    Uses a cheap truncation fallback when no LLM fn is provided — keeps the
    most recent portion of the merged text. When an LLM fn is provided,
    compression fires on overflow.
    """
    existing = get_summary(store, session_id)
    old_text = existing["text"] if existing else ""

    turn_text = f"[user] {user_message}\n[assistant] {assistant_message}\n"
    turn_text = turn_text[:_UPDATE_CHARS_PER_TURN]

    if not old_text:
        new_text = turn_text
    else:
        merged = old_text + turn_text
        if len(merged) <= _SUMMARY_MAX_CHARS:
            new_text = merged
        else:
            if llm_summarize_fn:
                new_text = _llm_compress(old_text, turn_text, llm_summarize_fn)
            else:
                new_text = merged[-_SUMMARY_MAX_CHARS:]

    _upsert_summary(store, session_id, new_text, existing)


def get_summary(store: Store, session_id: str) -> Optional[dict]:
    """Get the current summary for a session, or None."""
    from .schema import _esc
    where_clause = f'id = "{_esc(session_id)}"'
    try:
        df = store._session_summaries.search().where(where_clause).to_pandas()
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "id": row["id"],
            "text": row["text"],
            "message_count": int(row["message_count"]),
            "last_updated": row["last_updated"],
        }
    except Exception as e:
        logger.debug("get_summary failed for %s: %s", session_id, e)
        return None


def list_summaries(store: Store, limit: int = 50) -> list[dict]:
    """List all session summaries, most recent first."""
    try:
        df = store._session_summaries.search().limit(limit).to_pandas()
        df = df.sort_values("last_updated", ascending=False).head(limit)
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "message_count": int(row["message_count"]),
                "last_updated": row["last_updated"],
            }
            for _, row in df.iterrows()
        ]
    except Exception as e:
        logger.debug("list_summaries failed: %s", e)
        return []


def search_sessions_by_text(store: Store, query: str, limit: int = 10) -> list[dict]:
    """Keyword search over session summaries text content."""
    all_summaries = list_summaries(store, limit=500)
    qwords = set(query.lower().split())
    if not qwords:
        return []

    results = []
    for s in all_summaries:
        if not s["text"]:
            continue
        stext = s["text"].lower()
        overlap = sum(1 for w in qwords if w in stext)
        if overlap >= max(1, len(qwords) // 2):
            results.append((overlap, s))
    results.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in results[:limit]]


def _llm_compress(old_text: str, new_text: str, llm_fn) -> str:
    """Use an LLM to compress old + new into a ~1500 char summary."""
    merged = old_text + new_text
    prompt = (
        "Merge and compress this rolling session summary. "
        "Keep only the most important facts, decisions, and actions. "
        "Drop redundancy and chit-chat. Maximum 1500 characters.\n\n"
        f"OLD summary: {old_text}\n\n"
        f"NEW messages to merge: {new_text}\n\n"
        "Return ONLY the compressed summary text, no commentary."
    )
    try:
        compressed = llm_fn(prompt)
        return compressed[:_SUMMARY_MAX_CHARS]
    except Exception as e:
        logger.warning("LLM compress failed, falling back to truncation: %s", e)
        return merged[-_SUMMARY_MAX_CHARS:]


def _upsert_summary(store: Store, session_id: str, text: str, existing: Optional[dict]) -> None:
    """Insert or update a session summary row."""
    from datetime import datetime as _dt
    now = _dt.now(timezone.utc).isoformat()
    from .schema import _esc
    where_clause = f'id = "{_esc(session_id)}"'

    if existing:
        store._session_summaries.update(
            where=where_clause,
            values={
                "text": text,
                "message_count": existing["message_count"] + 1,
                "last_updated": now,
            },
        )
    else:
        store._session_summaries.add([{
            "id": session_id,
            "text": text,
            "message_count": 1,
            "last_updated": now,
        }])
