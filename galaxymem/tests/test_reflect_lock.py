"""Tests for concurrency-safe reflection (lockfile claims)."""

import tempfile
import time
from pathlib import Path

from galaxymem import reflect
from galaxymem.store import Store


class StubLLM:
    """LLM client that answers reflect prompts with a no-op JSON object."""

    def chat(self, messages):
        return '{"conflicts": [], "opinions": []}'


def _fresh_store(tmpdir) -> Store:
    from pathlib import Path as P
    store = Store(db_path=P(tmpdir) / "db")
    return store.open(create_if_missing=True)


def test_reflect_skipped_when_lock_held():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        lock_path = Path(store.db_path) / "reflect.lock"
        lock_path.write_text(str(time.time()))

        report = reflect.run_reflection(store, StubLLM())
        assert report.get("status") == "skipped"
        store.close()


def test_reflect_steals_stale_lock():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        lock_path = Path(store.db_path) / "reflect.lock"
        # Lock from 31 minutes ago — holder must have crashed.
        stale = time.time() - (reflect._LOCK_STALE_SECS + 60)
        lock_path.write_text(str(stale))

        report = reflect.run_reflection(store, StubLLM())
        assert report.get("status") != "skipped"
        assert "error" not in report or report.get("status") != "skipped"
        # Lock released after the run
        assert not lock_path.exists()
        store.close()


def test_reflect_releases_lock_on_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _fresh_store(tmpdir)
        report = reflect.run_reflection(store, StubLLM())
        assert report.get("status") != "skipped"
        lock_path = Path(store.db_path) / "reflect.lock"
        assert not lock_path.exists()
        store.close()


if __name__ == "__main__":
    print("run via pytest")
