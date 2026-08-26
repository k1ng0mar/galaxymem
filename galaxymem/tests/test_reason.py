"""Tests for gm_reason — evidence-backed reasoning over GalaxyMem memories."""

from __future__ import annotations

import json

import pytest
from datetime import datetime, timezone

from galaxymem.models import (
    MemoryRecord, Network, MemoryStatus,
    EntityRecord, EntityType,
)
from galaxymem.reason import (
    gather_context,
    _build_prompt,
    reason,
)


class FakeLLM:
    """Stand-in for _LLMClientAdapter — returns canned JSON."""

    def __init__(self, response: str):
        self.response = response
        self.last_messages = None

    def chat(self, messages: list[dict]) -> str:
        self.last_messages = messages
        return self.response


def _mem(text: str, network=Network.world, entity_ids=None, source_ids=None):
    from galaxymem.store import _ulid
    return MemoryRecord(
        id=_ulid(),
        text=text,
        network=network,
        entity_ids=entity_ids or [],
        source_memory_ids=source_ids or [],
        status=MemoryStatus.active,
    )


def _add(store, text, network=Network.world, entity_ids=None, source_ids=None):
    m = _mem(text, network, entity_ids, source_ids)
    store.add_memory(m)
    return m.id


class TestGatherContext:
    """gather_context: pulling opinions + facts from the store."""

    def test_returns_empty_when_nothing_stored(self, temp_db):
        ctx = gather_context(temp_db, "something")
        assert ctx["opinions"] == []
        assert ctx["facts"] == []
        assert ctx["entity_cards"] == []

    def test_returns_opinions_first(self, temp_db, sample_memory):
        # sample_memory is world; add an opinion too
        temp_db.add_memory(sample_memory)
        op_id = _add(temp_db, "User likes concise answers", network=Network.opinion)

        ctx = gather_context(temp_db, "preferences")

        assert len(ctx["opinions"]) == 1
        assert ctx["opinions"][0]["id"] == op_id
        assert len(ctx["facts"]) >= 1


class TestReason:
    """reason(): the orchestration + LLM parsing."""

    def test_reason_synthesizes_cited_answer(self, temp_db):
        fact_id = _add(temp_db, "User prefers Python over JavaScript")
        llm = FakeLLM(json.dumps({
            "answer": "User prefers Python. [%s]" % fact_id,
            "sources": [fact_id],
            "confidence": "high",
            "conflicts": [],
            "gaps": [],
        }))

        result = reason(temp_db, llm, "what language does user prefer?")

        assert result["answer"].startswith("User prefers Python")
        assert fact_id in result["sources"]
        assert result["confidence"] == "high"
        assert result["used"]["facts"] >= 1

    def test_reason_returns_error_on_missing_query(self, temp_db):
        llm = FakeLLM("{}")
        result = reason(temp_db, llm, "   ")
        assert "error" in result

    def test_reason_handles_empty_store(self, temp_db):
        llm = FakeLLM("{}")
        result = reason(temp_db, llm, "anything")
        assert result["confidence"] == "low"
        assert "No relevant memories" in result["answer"]

    def test_reason_sanitizes_invalid_sources(self, temp_db):
        fact_id = _add(temp_db, "Real fact here")
        llm = FakeLLM(json.dumps({
            "answer": "Something",
            "sources": [fact_id, "made-up-id"],
            "confidence": "medium",
            "conflicts": [],
            "gaps": [],
        }))

        result = reason(temp_db, llm, "test")

        assert fact_id in result["sources"]
        assert "made-up-id" not in result["sources"]

    def test_reason_flags_conflict_when_opinion_contradicts_fact(self, temp_db):
        _add(temp_db, "User likes dark mode", network=Network.opinion)
        _add(temp_db, "User switched to light mode")
        llm = FakeLLM(json.dumps({
            "answer": "Conflicting evidence.",
            "sources": [],
            "confidence": "low",
            "conflicts": ["opinion says dark, fact says light"],
            "gaps": [],
        }))

        result = reason(temp_db, llm, "user theme preference")

        assert result["conflicts"], "should surface the contradiction"
        assert result["confidence"] == "low"
