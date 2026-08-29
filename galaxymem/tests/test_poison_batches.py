"""Tests for Pass 2 poison-batch parking (attempt_count cap)."""

import tempfile

from galaxymem import config as cfg
from galaxymem.retain import flag_turn, process_pending_flags
from galaxymem.store_sqlite import Store


class BoomClient:
    """LLM client whose every call fails."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError("llm down")


def _fresh_store(tmpdir) -> Store:
    from pathlib import Path
    store = Store(db_path=Path(tmpdir) / "db")
    return store.open(create_if_missing=True)


def test_poison_batch_parked_after_max_attempts():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        flag_turn(store, "remember that the deploy deadline is friday",
                  "s-poison", "cli", "u1")

        # Run Pass 2 to exhaustion
        for _ in range(cfg.PASS2_MAX_ATTEMPTS):
            process_pending_flags(store, BoomClient())

        flags = store.unprocessed_flags()
        assert len(flags) == 1
        assert flags[0].attempt_count == cfg.PASS2_MAX_ATTEMPTS

        # A further trigger does NOT retry parked flags
        process_pending_flags(store, BoomClient())
        assert store.unprocessed_flags()[0].attempt_count == cfg.PASS2_MAX_ATTEMPTS
        store.close()


def test_failed_batch_below_cap_still_retryable():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        flag_turn(store, "remember that the launch date is monday",
                  "s-retry", "cli", "u1")

        process_pending_flags(store, BoomClient())  # one failure
        flags = store.unprocessed_flags()
        assert len(flags) == 1
        assert flags[0].attempt_count == 1  # still retryable
        store.close()


def test_healthy_path_unaffected():
    """A working LLM processes flags regardless of attempt_count plumbing."""

    class MockLLM:
        def complete(self, prompt):
            return ('[{"text": "The deploy deadline is friday", '
                    '"network": "world", "entity_labels": [], '
                    '"memory_ids": []}]')

    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        flag_turn(store, "remember that the deploy deadline is friday",
                  "s-ok", "cli", "u1")
        assert process_pending_flags(store, MockLLM()) >= 0
        store.close()


if __name__ == "__main__":
    test_poison_batch_parked_after_max_attempts()
    print("ok")
