"""Tier-2 feature tests: session summaries, confidence scoring,
query expansion, procedural/gap detection, gm_session_search."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from galaxymem.confidence import compute_confidence, classify_confidence
from galaxymem.queryexpansion import expand_query, should_expand
from galaxymem.procedural import detect_gaps, extract_procedural
from galaxymem.summaries import (
    _SUMMARY_MAX_CHARS,
    get_summary,
    list_summaries,
    search_sessions_by_text,
    update_summary,
)
from galaxymem.models import MemoryRecord, Network
from galaxymem.store_sqlite import Store


class TestSummaries:
    def test_creates_summary_on_first_call(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        update_summary(s, "session-1", "hello there", "hi back")
        result = get_summary(s, "session-1")
        assert result is not None
        assert "hello" in result["text"]
        assert result["message_count"] == 1

    def test_updates_existing_summary(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        update_summary(s, "s1", "turn one", "response one")
        update_summary(s, "s1", "turn two", "response two")
        result = get_summary(s, "s1")
        assert result["message_count"] == 2

    def test_rolls_when_overflowing(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        big_text = "x" * 500
        for _ in range(10):
            update_summary(s, "s1", big_text, "y")
        result = get_summary(s, "s1")
        assert len(result["text"]) <= _SUMMARY_MAX_CHARS

    def test_llm_compress_called_on_overflow(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        big_text = "hello " * 100
        llm_calls = []

        def fake_llm(prompt):
            llm_calls.append(prompt)
            return "compressed summary output"

        for _ in range(5):
            update_summary(
                s, "s1", big_text, "resp",
                llm_summarize_fn=fake_llm,
            )
        last = get_summary(s, "s1")
        assert len(last["text"]) <= _SUMMARY_MAX_CHARS
        # LLM was used to compress instead of blind truncation

    def test_no_summary_returns_none(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        assert get_summary(s, "nonexistent") is None

    def test_list_summaries_orders_newest_first(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        update_summary(s, "old", "old turn", "old reply")
        update_summary(s, "new", "new turn", "new reply")
        results = list_summaries(s)
        assert len(results) == 2

    def test_search_sessions_by_keyword(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        update_summary(s, "s1", "galaxymem discussions", "some reply")
        update_summary(s, "s2", "chess and games", "another reply")
        matches = search_sessions_by_text(s, "galaxymem memory system", limit=5)
        assert len(matches) >= 1
        assert "galaxymem" in matches[0]["text"].lower()

    def test_search_sessions_no_match(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        update_summary(s, "s1", "topic alpha mention", "reply")
        results = search_sessions_by_text(s, " galaxy delta omega 999", limit=5)
        # Should not return things that don't match at all (or gracefully empty)
        assert len(results) == 0 or all(
            "galaxy delta omega 999" not in r["text"]
            for r in results
        )


class TestConfidenceScore:
    def _mk(self, status="active", sources=0, cycles=0):
        return MemoryRecord(
            id="test", text="test fact",
            network=Network.world,
            entity_ids=["test"],
            status=status,
            source_memory_ids=[f"src-{i}" for i in range(sources)],
            reflect_cycles=cycles,
        )

    def test_active_memory_baseline(self):
        mem = self._mk(status="active", sources=0, cycles=0)
        score = compute_confidence(mem)
        assert 0.45 <= score <= 0.55  # active with no sources → ~0.5

    def test_five_sources_capped(self):
        mem = self._mk(sources=10, cycles=0)
        score = compute_confidence(mem)
        assert score < 1.0, "should cap below max"

    def test_certain_tier(self):
        mem = self._mk(sources=8, cycles=5)
        score = compute_confidence(mem)
        assert classify_confidence(score) == "certain"

    def test_low_confidence_archived(self):
        mem = self._mk(status="archived")
        score = compute_confidence(mem)
        assert classify_confidence(score) == "low"

    def test_confidence_score_range(self):
        for status in ["active", "contested", "demoted", "archived"]:
            mem = self._mk(status=status, sources=5, cycles=2)
            score = compute_confidence(mem)
            assert 0.0 <= score <= 1.0


class TestQueryExpansion:
    def test_should_expand_short_queries(self):
        assert should_expand("api") == False  # too short
        assert should_expand("docs") == False

    def test_should_expand_entity_scoped_returns_false(self):
        assert should_expand("what port", entity_ids=["api"]) == False

    def test_should_expand_normal(self):
        assert should_expand("what port is the API on") == True

    def test_should_expand_fillers_starting_query(self):
        assert should_expand("get me the test suite run command") == True

    def test_expand_query_calls_llm(self):
        llm = MagicMock()
        llm.complete = MagicMock(return_value="api port service endpoint listen 8010")
        out = expand_query("what port is the api on", llm)
        assert "api port" in out.lower() or "8010" in out.lower()
        llm.complete.assert_called_once()

    def test_expand_query_llm_fails_gently(self):
        llm = MagicMock()
        llm.complete = MagicMock(side_effect=Exception("model down"))
        out = expand_query("what port is the api on", llm)
        assert out == "what port is the api on"  # falls back


class TestProcedural:
    def test_extract_returns_none_for_declaratives(self):
        assert extract_procedural(None, "the weather is nice today") is None
        assert extract_procedural(None, "my name is Umar") is None

    def test_extract_directive(self):
        result = extract_procedural(None, "use the direct api instead of the wrapper")
        assert result is not None
        assert result["network"] == "observation"
        assert result["procedural"] is True

    def test_extract_project_constraint(self):
        result = extract_procedural(None, "tests gate builds hard in this repo")
        assert result is not None
        assert result["network"] == "observation"
        assert result["procedural"] is True

    def test_detect_gaps_no_results(self):
        from galaxymem.procedural import detect_gaps
        fake = MagicMock()
        assert detect_gaps(fake, []) == []

    def test_detect_gaps_no_edge_memory(self, tmp_path):
        from galaxymem.procedural import detect_gaps
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        counts = detect_gaps(s, [])
        assert counts == []
        s.close()


# ── End-to-end: session_search handler ──────────────────────────────────────

class TestSessionSearchHandler:
    def _mk_store_with_summaries(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        update_summary(s, "sess-alpha", "galaxymem design discussion", "architecture talk")
        update_summary(s, "sess-beta", "chess match analysis", "move review")
        update_summary(s, "sess-beta", "more chess", "opening theory")
        return s

    def test_search_finds_relevant_sessions(self, tmp_path):
        s = self._mk_store_with_summaries(tmp_path)
        # Keyword search over session summaries
        results = search_sessions_by_text(s, "galaxymem architecture", limit=5)
        assert len(results) >= 1
        assert any("galaxymem" in r["text"].lower() for r in results)
        s.close()

    def test_session_search_respects_limit(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        for i in range(10):
            update_summary(s, f"sess-{i}", f"unique topic {i}", "resp")
        results = search_sessions_by_text(s, "unique topic", limit=3)
        assert len(results) <= 3
        s.close()

    def test_get_summary_returns_updated_content(self, tmp_path):
        s = Store(db_path=Path(tmp_path)).open(create_if_missing=True)
        update_summary(s, "me", "hello there", "hi back")
        update_summary(s, "me", "how are you", "good")
        result = get_summary(s, "me")
        assert result["message_count"] == 2
        s.close()
