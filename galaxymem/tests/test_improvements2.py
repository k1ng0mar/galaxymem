"""Tests for gm round-2 improvements: evidence_quotes, staleness re-verify,
usefulness feedback."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from galaxymem.models import (
    MemoryRecord, Network, MemoryStatus,
)
from galaxymem.reflect import (
    _validate_evidence_quotes,
    _usefulness_ratio,
    _apply_usefulness_policy,
)
from galaxymem.reason import gather_context


class TestEvidenceQuotes:
    def test_validate_keeps_only_verbatim_quotes(self, temp_db, sample_memory):
        # sample_memory has some text; create by_id with it
        from galaxymem.store import _from_memory
        # just use the sample_memory object directly
        by_id = {sample_memory.id: sample_memory}
        quotes = _validate_evidence_quotes(
            [sample_memory.text[:30], "not in source at all", ""],
            [sample_memory.id],
            by_id,
        )
        # first is verbatim substring (stripped), second isn't, third empty
        assert sample_memory.text[:30].strip() in quotes
        assert "not in source at all" not in quotes

    def test_opinion_stores_quotes(self, temp_db):
        # Add a source memory, form an opinion with quotes via the model
        mem = MemoryRecord(
            id="src-1", text="User prefers TypeScript over JavaScript",
            network=Network.world, status=MemoryStatus.active,
        )
        temp_db.add_memory(mem)
        opinion = MemoryRecord(
            id="op-1", text="User is a TypeScript fan",
            network=Network.opinion, status=MemoryStatus.active,
            source_memory_ids=["src-1"],
            evidence_quotes=["User prefers TypeScript over JavaScript"],
        )
        temp_db.add_memory(opinion)
        got = temp_db.get_memory("op-1")
        assert got is not None
        assert "TypeScript" in got.evidence_quotes[0]


class TestUsefulness:
    def test_usefulness_ratio(self, temp_db, sample_memory):
        mem = sample_memory
        mem.recall_count = 5
        mem.recall_miss_count = 15
        ratio = _usefulness_ratio(mem)
        assert ratio is not None
        assert abs(ratio - 0.25) < 1e-6

    def test_usefulness_ratio_needs_min_recalls(self, temp_db, sample_memory):
        mem = sample_memory
        mem.recall_count = 1
        mem.recall_miss_count = 0
        assert _usefulness_ratio(mem) is None

    def test_policy_demotes_low_usefulness(self, temp_db):
        low = MemoryRecord(
            id="low-1", text="stale fact that never gets used",
            network=Network.world, status=MemoryStatus.active,
        )
        low.recall_count = 3
        low.recall_miss_count = 50  # very low usefulness
        temp_db.add_memory(low)

        result = _apply_usefulness_policy(temp_db, {})

        assert result["demoted"] >= 1
        got = temp_db.get_memory("low-1")
        assert got.status == MemoryStatus.demoted

    def test_policy_revives_high_usefulness(self, temp_db):
        high = MemoryRecord(
            id="high-1", text="frequently used fact",
            network=Network.world, status=MemoryStatus.demoted,
        )
        high.recall_count = 40
        high.recall_miss_count = 2  # very high usefulness
        temp_db.add_memory(high)

        result = _apply_usefulness_policy(temp_db, {})

        assert result["revived"] >= 1
        got = temp_db.get_memory("high-1")
        assert got.status == MemoryStatus.active
