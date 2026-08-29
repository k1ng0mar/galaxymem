"""Wiki promotion bridge — Phase 8 (D12).

This module writes PROPOSALS, never wiki pages. Stable Layer-1 memories are
nominated into the user's EXISTING approval-gated pipeline: a proposal note
lands in the vault inbox (`notes/`) tagged `workflow:draft`, and from there
the user's curation workflow (scan → propose → human approval → wiki page)
takes over untouched.

Lifecycle:
1. Eligibility (mirrors the existing thresholds): recalled >= 3 times AND
   stable (active, not superseded/contested) across >= 2 reflect cycles AND
   not already represented in the wiki index AND not already proposed.
2. A proposal note is written once per memory; the promotion_queue table is
   the ledger of written proposals (prevents re-nomination).
3. When the user's workflow approves a proposal (the note gains a
   `workflow:promoted` tag, wherever it now lives in the vault), the source
   memory gets a `promoted_to` pointer so future reflects skip it. The memory
   itself stays active — promotion is a pointer, not a status change (D13).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config as cfg
from .models import MemoryRecord, MemoryStatus
from .store_sqlite import Store

logger = logging.getLogger(__name__)


# ── Eligibility ─────────────────────────────────────────────────────────

def check_promotion_eligibility(store: Store) -> list[MemoryRecord]:
    """Find memories meeting the promotion thresholds (Phase 8 step 1)."""
    active_memories = store.list_memories(status=MemoryStatus.active)

    proposed_ids = {q.memory_id for q in _safe_queue(store)}

    eligible = []
    for mem in active_memories:
        if mem.id in proposed_ids:
            continue
        if mem.promoted_to:
            continue
        if (mem.recall_count >= cfg.PROMOTION_MIN_RECALLS
                and mem.reflect_cycles >= cfg.PROMOTION_MIN_CYCLES):
            eligible.append(mem)

    return eligible


def scan_for_promotable(store: Store) -> list[MemoryRecord]:
    """Alias for check_promotion_eligibility."""
    return check_promotion_eligibility(store)


def _safe_queue(store: Store) -> list:
    try:
        return store.list_promotion_queue()
    except Exception:
        return []


# ── Wiki-coverage check ─────────────────────────────────────────────────

def is_in_wiki(memory: MemoryRecord, wiki_index_path: Optional[Path] = None) -> bool:
    """Best-effort check whether the memory is already represented in the
    wiki: its id (provenance) or its title line appears in the wiki index."""
    index_path = wiki_index_path or cfg.WIKI_INDEX_PATH
    if index_path is None:
        return False  # No wiki configured, assume not in wiki
    try:
        content = index_path.read_text(encoding="utf-8").lower()
    except (FileNotFoundError, OSError):
        return False

    if memory.id.lower() in content:
        return True
    title = _proposal_title(memory).lower()
    return len(title) >= 20 and title in content


# ── Proposal note ───────────────────────────────────────────────────────

def _proposal_title(memory: MemoryRecord) -> str:
    title = memory.text[:60].replace("\n", " ").strip()
    if len(memory.text) > 60:
        title += "..."
    return title


def _proposal_filename(memory: MemoryRecord) -> str:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    short_id = re.sub(r"[^a-zA-Z0-9]", "", memory.id)[-8:] or "memory"
    return f"{date}-galaxymem-{short_id}.md"


def format_proposal_note(memory: MemoryRecord, store: Store) -> str:
    """Render a proposal note matching the vault's frontmatter conventions."""
    from .sanitize import yaml_quote, prompt_escape

    created = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "---",
        f"title: {yaml_quote(_proposal_title(memory))}",
        f"created: {created}",
        "type: memory-promotion",
        "tags:",
        "  - workflow:draft",
        "  - source:galaxymem",
    ]
    for eid in memory.entity_ids:
        safe_eid = re.sub(r"[^a-zA-Z0-9._-]", "-", str(eid))[:64]
        if safe_eid:
            lines.append(f"  - topic:{safe_eid}")
    lines += [
        "provenance:",
        "  memory_ids:",
        f"    - {yaml_quote(memory.id)}",
    ]
    for sid in memory.source_memory_ids:
        lines.append(f"    - {yaml_quote(sid)}")
    lines += [
        f"  network: {memory.network.value}",
        f"  recall_count: {memory.recall_count}",
        f"  reflect_cycles: {memory.reflect_cycles}",
        f"  first_seen: {yaml_quote(memory.created_at.isoformat())}",
        "---",
        "",
        memory.text.replace("\x00", ""),
        "",
    ]

    if memory.source_memory_ids:
        lines.append("## Supporting memories")
        sources = store.get_memories_by_ids(list(memory.source_memory_ids))
        for sid in memory.source_memory_ids:
            src = sources.get(sid)
            snippet = prompt_escape(src.text, max_len=200) if src else "(not found)"
            lines.append(f"- `{sid}` — {snippet}")
        lines.append("")

    return "\n".join(lines)


def write_proposal(store: Store, memory: MemoryRecord,
                   notes_path: Optional[Path] = None) -> Optional[Path]:
    """Write exactly one proposal note for a memory into the vault inbox."""
    from .sanitize import resolve_under

    inbox = notes_path or cfg.VAULT_NOTES_PATH
    if inbox is None:
        return None
    try:
        inbox = Path(inbox).expanduser().resolve()
        inbox.mkdir(parents=True, exist_ok=True)
        note_path = resolve_under(_proposal_filename(memory), inbox)
        note_path.write_text(format_proposal_note(memory, store), encoding="utf-8")
    except (OSError, ValueError) as e:
        logger.error("Failed to write proposal note for %s: %s", memory.id, e)
        return None

    store.add_to_promotion_queue(memory.id, target_systems=[str(note_path)])
    logger.info("Promotion proposal written: %s -> %s", memory.id, note_path)
    return note_path


# ── Approval detection ──────────────────────────────────────────────────

def check_approved_promotions(store: Store,
                              vault_root: Optional[Path] = None) -> int:
    """Detect proposals the user's workflow approved.

    A proposal counts as approved when a vault note referencing the memory id
    carries a `workflow:promoted` tag (the note may have been moved). The
    source memory gets its `promoted_to` pointer and leaves the ledger.
    """
    queue = _safe_queue(store)
    if not queue:
        return 0

    root = vault_root or (cfg.VAULT_NOTES_PATH.parent if cfg.VAULT_NOTES_PATH else None)
    if root is None:
        return 0  # No vault configured
    approved = 0

    for item in queue:
        note_hint = item.target_systems[0] if item.target_systems else None
        promoted_path = _find_promoted_note(root, item.memory_id, note_hint)
        if promoted_path is None:
            continue
        try:
            store.update_memory_field(item.memory_id, promoted_to=str(promoted_path))
            store.remove_from_promotion_queue(item.memory_id)
            approved += 1
            logger.info("Promotion approved: %s -> %s", item.memory_id, promoted_path)
        except Exception as e:
            logger.warning("Failed to record approved promotion %s: %s",
                           item.memory_id, e)

    return approved


def _find_promoted_note(root: Path, memory_id: str,
                        note_hint: Optional[str]) -> Optional[Path]:
    """Find a note referencing memory_id whose frontmatter says workflow:promoted."""
    from .sanitize import is_under

    root = Path(root).expanduser().resolve()
    candidates: list[Path] = []
    if note_hint:
        hint = Path(note_hint).expanduser()
        try:
            hint = hint.resolve()
        except OSError:
            hint = None
        if hint is not None and hint.exists() and is_under(hint, root):
            candidates.append(hint)
    if not candidates:
        try:
            scanned = 0
            for p in root.rglob("*.md"):
                scanned += 1
                if scanned > 500:
                    break
                if not is_under(p, root):
                    continue
                try:
                    if memory_id in p.read_text(encoding="utf-8", errors="ignore"):
                        candidates.append(p)
                except OSError:
                    continue
        except OSError:
            return None

    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if memory_id in text and re.search(r"workflow:\s*promoted", text):
            return path
    return None


# ── Full cycle ──────────────────────────────────────────────────────────

def run_promotion_cycle(store: Store,
                        notes_path: Optional[Path] = None,
                        wiki_index_path: Optional[Path] = None) -> dict[str, Any]:
    """One promotion pass: record approvals, then write new proposals.

    Never touches the wiki — output is proposal notes for the human gate.
    """
    report: dict[str, Any] = {
        "approved_count": 0,
        "eligible_count": 0,
        "nominated_count": 0,
        "skipped_in_wiki": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        report["approved_count"] = check_approved_promotions(store)
    except Exception as e:
        logger.warning("Approval check failed: %s", e)

    eligible = check_promotion_eligibility(store)
    report["eligible_count"] = len(eligible)

    for mem in eligible:
        if is_in_wiki(mem, wiki_index_path):
            report["skipped_in_wiki"] += 1
            continue
        if write_proposal(store, mem, notes_path) is not None:
            report["nominated_count"] += 1

    if report["nominated_count"] or report["approved_count"]:
        logger.info("Promotion cycle: %s", report)
    return report
