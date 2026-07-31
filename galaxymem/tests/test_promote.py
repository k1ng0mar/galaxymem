"""Tests for promote.py — the proposal-only wiki bridge (Phase 8 / D12).

Promotion writes proposal notes into the vault inbox for the EXISTING
human-gated pipeline. It never writes wiki pages and never flips a memory's
status — approval only sets the promoted_to pointer.
"""

from __future__ import annotations

from datetime import datetime, timezone

from galaxymem import config as cfg
from galaxymem.models import MemoryRecord, MemoryStatus, Network
from galaxymem.promote import (
    check_approved_promotions,
    check_promotion_eligibility,
    format_proposal_note,
    is_in_wiki,
    run_promotion_cycle,
    scan_for_promotable,
    write_proposal,
)


def _make_memory(mem_id: str, *, recall_count: int = 5, reflect_cycles: int = 3,
                 status: MemoryStatus = MemoryStatus.active,
                 promoted_to: str | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=mem_id,
        text=f"Stable fact {mem_id}: the project uses LanceDB for storage",
        network=Network.world,
        entity_ids=["hermes-mem"],
        status=status,
        recall_count=recall_count,
        reflect_cycles=reflect_cycles,
        promoted_to=promoted_to,
        created_at=datetime.now(timezone.utc),
    )


class TestEligibility:
    def test_scan_returns_empty_when_no_memories(self, temp_db):
        assert scan_for_promotable(temp_db) == []

    def test_eligible_memory_found(self, temp_db):
        temp_db.add_memory(_make_memory("m-eligible"))
        eligible = check_promotion_eligibility(temp_db)
        assert [m.id for m in eligible] == ["m-eligible"]

    def test_low_recall_count_not_eligible(self, temp_db):
        temp_db.add_memory(_make_memory("m-low-recall",
                                        recall_count=cfg.PROMOTION_MIN_RECALLS - 1))
        assert check_promotion_eligibility(temp_db) == []

    def test_low_reflect_cycles_not_eligible(self, temp_db):
        temp_db.add_memory(_make_memory("m-young",
                                        reflect_cycles=cfg.PROMOTION_MIN_CYCLES - 1))
        assert check_promotion_eligibility(temp_db) == []

    def test_non_active_not_eligible(self, temp_db):
        temp_db.add_memory(_make_memory("m-contested", status=MemoryStatus.contested))
        assert check_promotion_eligibility(temp_db) == []

    def test_already_promoted_not_eligible(self, temp_db):
        temp_db.add_memory(_make_memory("m-done", promoted_to="/vault/somewhere.md"))
        assert check_promotion_eligibility(temp_db) == []

    def test_already_proposed_not_eligible(self, temp_db, tmp_path):
        temp_db.add_memory(_make_memory("m-proposed"))
        mem = temp_db.get_memory("m-proposed")
        write_proposal(temp_db, mem, notes_path=tmp_path / "notes")
        assert check_promotion_eligibility(temp_db) == []


class TestProposalNote:
    def test_note_has_draft_workflow_and_provenance(self, temp_db):
        mem = _make_memory("m-note")
        temp_db.add_memory(mem)
        note = format_proposal_note(temp_db.get_memory("m-note"), temp_db)
        assert "workflow:draft" in note
        assert "m-note" in note              # provenance memory id
        assert "topic:hermes-mem" in note    # entity topic tag
        assert "source:galaxymem" in note
        assert note.startswith("---")

    def test_write_proposal_creates_exactly_one_note(self, temp_db, tmp_path):
        inbox = tmp_path / "notes"
        temp_db.add_memory(_make_memory("m-once"))
        mem = temp_db.get_memory("m-once")

        path = write_proposal(temp_db, mem, notes_path=inbox)
        assert path is not None and path.exists()

        # Re-running the cycle never duplicates the proposal (Phase 8 checkpoint)
        report = run_promotion_cycle(temp_db, notes_path=inbox,
                                     wiki_index_path=tmp_path / "missing-index.md")
        assert report["nominated_count"] == 0
        assert len(list(inbox.glob("*.md"))) == 1

    def test_memory_status_unchanged_by_proposal(self, temp_db, tmp_path):
        temp_db.add_memory(_make_memory("m-status"))
        write_proposal(temp_db, temp_db.get_memory("m-status"),
                       notes_path=tmp_path / "notes")
        assert temp_db.get_memory("m-status").status == MemoryStatus.active


class TestWikiCheck:
    def test_memory_id_in_index_counts_as_covered(self, tmp_path):
        mem = _make_memory("m-wiki")
        index = tmp_path / "index.md"
        index.write_text("# Wiki index\n- some page (from m-wiki)\n", encoding="utf-8")
        assert is_in_wiki(mem, wiki_index_path=index) is True

    def test_missing_index_means_not_covered(self, tmp_path):
        mem = _make_memory("m-noindex")
        assert is_in_wiki(mem, wiki_index_path=tmp_path / "nope.md") is False

    def test_cycle_skips_wiki_covered(self, temp_db, tmp_path):
        temp_db.add_memory(_make_memory("m-covered"))
        index = tmp_path / "index.md"
        index.write_text("m-covered already documented", encoding="utf-8")
        report = run_promotion_cycle(temp_db, notes_path=tmp_path / "notes",
                                     wiki_index_path=index)
        assert report["skipped_in_wiki"] == 1
        assert report["nominated_count"] == 0


class TestApproval:
    def test_approved_note_sets_promoted_to(self, temp_db, tmp_path):
        inbox = tmp_path / "notes"
        temp_db.add_memory(_make_memory("m-approve"))
        mem = temp_db.get_memory("m-approve")
        note_path = write_proposal(temp_db, mem, notes_path=inbox)

        # Simulate the user's workflow approving: draft → promoted
        content = note_path.read_text(encoding="utf-8")
        note_path.write_text(content.replace("workflow:draft", "workflow:promoted"),
                             encoding="utf-8")

        approved = check_approved_promotions(temp_db, vault_root=tmp_path)
        assert approved == 1
        updated = temp_db.get_memory("m-approve")
        assert updated.promoted_to == str(note_path)
        # Approval is a pointer, not a status flip (D13)
        assert updated.status == MemoryStatus.active
        # Ledger cleared
        assert temp_db.list_promotion_queue() == []

    def test_unapproved_note_stays_pending(self, temp_db, tmp_path):
        inbox = tmp_path / "notes"
        temp_db.add_memory(_make_memory("m-pending"))
        write_proposal(temp_db, temp_db.get_memory("m-pending"), notes_path=inbox)

        approved = check_approved_promotions(temp_db, vault_root=tmp_path)
        assert approved == 0
        assert temp_db.get_memory("m-pending").promoted_to is None


class TestFullCycle:
    def test_cycle_nominates_and_reports(self, temp_db, tmp_path):
        temp_db.add_memory(_make_memory("m-a"))
        temp_db.add_memory(_make_memory("m-b"))
        temp_db.add_memory(_make_memory("m-under", recall_count=0))

        report = run_promotion_cycle(temp_db, notes_path=tmp_path / "notes",
                                     wiki_index_path=tmp_path / "missing.md")
        assert report["eligible_count"] == 2
        assert report["nominated_count"] == 2
        assert len(list((tmp_path / "notes").glob("*.md"))) == 2
