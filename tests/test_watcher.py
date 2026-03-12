"""Tests for ckg.watcher.GraphWatcher and helpers."""

from __future__ import annotations

import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from ckg.store import GraphStore
from ckg.watcher import GraphWatcher, _PythonFileHandler, _is_tracked, _rel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "service.py").write_text(textwrap.dedent("""\
        def run():
            pass
        def helper():
            pass
    """))
    (tmp_path / "db.py").write_text(textwrap.dedent("""\
        class Database:
            def add(self, key, value):
                self._data = {}
    """))
    return tmp_path


# ---------------------------------------------------------------------------
# _is_tracked
# ---------------------------------------------------------------------------

class TestIsTracked:
    def test_python_file_tracked(self, tmp_path: Path) -> None:
        assert _is_tracked(str(tmp_path / "foo.py")) is True

    def test_non_python_not_tracked(self, tmp_path: Path) -> None:
        assert _is_tracked(str(tmp_path / "README.md")) is False
        assert _is_tracked(str(tmp_path / "data.json")) is False

    def test_venv_not_tracked(self, tmp_path: Path) -> None:
        assert _is_tracked(str(tmp_path / ".venv" / "lib" / "foo.py")) is False

    def test_pycache_not_tracked(self, tmp_path: Path) -> None:
        assert _is_tracked(str(tmp_path / "__pycache__" / "foo.cpython-312.pyc")) is False

    def test_git_not_tracked(self, tmp_path: Path) -> None:
        assert _is_tracked(str(tmp_path / ".git" / "ORIG_HEAD")) is False

    def test_build_dir_not_tracked(self, tmp_path: Path) -> None:
        assert _is_tracked(str(tmp_path / "build" / "lib" / "foo.py")) is False

    def test_egg_info_not_tracked(self, tmp_path: Path) -> None:
        assert _is_tracked(str(tmp_path / "mypackage.egg-info" / "top_level.py")) is False

    def test_nested_python_file_tracked(self, tmp_path: Path) -> None:
        assert _is_tracked(str(tmp_path / "subdir" / "module.py")) is True


# ---------------------------------------------------------------------------
# _rel
# ---------------------------------------------------------------------------

class TestRel:
    def test_strips_root_prefix(self, tmp_path: Path) -> None:
        result = _rel(str(tmp_path / "sub" / "foo.py"), tmp_path)
        assert result == "sub/foo.py"

    def test_returns_original_if_not_under_root(self, tmp_path: Path) -> None:
        result = _rel("/some/other/path/foo.py", tmp_path)
        assert result == "/some/other/path/foo.py"


# ---------------------------------------------------------------------------
# _PythonFileHandler — debounce logic
# ---------------------------------------------------------------------------

class TestPythonFileHandler:
    def test_on_modified_queues_path(self, tmp_path: Path) -> None:
        fired: list[tuple[set, set]] = []
        handler = _PythonFileHandler(
            root=tmp_path,
            on_change=lambda m, d: fired.append((set(m), set(d))),
            debounce_seconds=0.05,
        )
        from watchdog.events import FileModifiedEvent
        handler.on_modified(FileModifiedEvent(str(tmp_path / "foo.py")))
        time.sleep(0.15)
        assert len(fired) == 1
        assert "foo.py" in fired[0][0]

    def test_on_created_queues_path(self, tmp_path: Path) -> None:
        fired: list[tuple[set, set]] = []
        handler = _PythonFileHandler(
            root=tmp_path,
            on_change=lambda m, d: fired.append((set(m), set(d))),
            debounce_seconds=0.05,
        )
        from watchdog.events import FileCreatedEvent
        handler.on_created(FileCreatedEvent(str(tmp_path / "new.py")))
        time.sleep(0.15)
        assert len(fired) == 1
        assert "new.py" in fired[0][0]

    def test_on_deleted_queues_path(self, tmp_path: Path) -> None:
        fired: list[tuple[set, set]] = []
        handler = _PythonFileHandler(
            root=tmp_path,
            on_change=lambda m, d: fired.append((set(m), set(d))),
            debounce_seconds=0.05,
        )
        from watchdog.events import FileDeletedEvent
        handler.on_deleted(FileDeletedEvent(str(tmp_path / "old.py")))
        time.sleep(0.15)
        assert len(fired) == 1
        assert "old.py" in fired[0][1]

    def test_on_moved_invalidates_src_queues_dst(self, tmp_path: Path) -> None:
        fired: list[tuple[set, set]] = []
        handler = _PythonFileHandler(
            root=tmp_path,
            on_change=lambda m, d: fired.append((set(m), set(d))),
            debounce_seconds=0.05,
        )
        from watchdog.events import FileMovedEvent
        handler.on_moved(
            FileMovedEvent(str(tmp_path / "old.py"), str(tmp_path / "new.py"))
        )
        time.sleep(0.15)
        assert len(fired) == 1
        modified, deleted = fired[0]
        assert "new.py" in modified
        assert "old.py" in deleted

    def test_non_python_file_ignored(self, tmp_path: Path) -> None:
        fired: list[tuple[set, set]] = []
        handler = _PythonFileHandler(
            root=tmp_path,
            on_change=lambda m, d: fired.append((set(m), set(d))),
            debounce_seconds=0.05,
        )
        from watchdog.events import FileModifiedEvent
        handler.on_modified(FileModifiedEvent(str(tmp_path / "README.md")))
        time.sleep(0.15)
        assert fired == []

    def test_venv_file_ignored(self, tmp_path: Path) -> None:
        fired: list[tuple[set, set]] = []
        handler = _PythonFileHandler(
            root=tmp_path,
            on_change=lambda m, d: fired.append((set(m), set(d))),
            debounce_seconds=0.05,
        )
        from watchdog.events import FileModifiedEvent
        handler.on_modified(
            FileModifiedEvent(str(tmp_path / ".venv" / "site-packages" / "x.py"))
        )
        time.sleep(0.15)
        assert fired == []

    def test_debounce_merges_rapid_events(self, tmp_path: Path) -> None:
        """Multiple rapid events within debounce window → single callback."""
        fired: list[tuple[set, set]] = []
        handler = _PythonFileHandler(
            root=tmp_path,
            on_change=lambda m, d: fired.append((set(m), set(d))),
            debounce_seconds=0.1,
        )
        from watchdog.events import FileModifiedEvent
        for _ in range(5):
            handler.on_modified(FileModifiedEvent(str(tmp_path / "foo.py")))
            time.sleep(0.01)
        time.sleep(0.25)
        # Should have fired exactly once (debounced)
        assert len(fired) == 1

    def test_debounce_accumulates_multiple_files(self, tmp_path: Path) -> None:
        fired: list[tuple[set, set]] = []
        handler = _PythonFileHandler(
            root=tmp_path,
            on_change=lambda m, d: fired.append((set(m), set(d))),
            debounce_seconds=0.1,
        )
        from watchdog.events import FileModifiedEvent
        handler.on_modified(FileModifiedEvent(str(tmp_path / "a.py")))
        handler.on_modified(FileModifiedEvent(str(tmp_path / "b.py")))
        time.sleep(0.25)
        assert len(fired) == 1
        assert {"a.py", "b.py"} == fired[0][0]

    def test_delete_cancels_pending_modify(self, tmp_path: Path) -> None:
        """If a file is queued for modify then deleted, it should appear only in deleted."""
        fired: list[tuple[set, set]] = []
        handler = _PythonFileHandler(
            root=tmp_path,
            on_change=lambda m, d: fired.append((set(m), set(d))),
            debounce_seconds=0.1,
        )
        from watchdog.events import FileModifiedEvent, FileDeletedEvent
        handler.on_modified(FileModifiedEvent(str(tmp_path / "foo.py")))
        handler.on_deleted(FileDeletedEvent(str(tmp_path / "foo.py")))
        time.sleep(0.25)
        assert len(fired) == 1
        modified, deleted = fired[0]
        assert "foo.py" not in modified
        assert "foo.py" in deleted


# ---------------------------------------------------------------------------
# GraphWatcher — integration with real filesystem
# ---------------------------------------------------------------------------

class TestGraphWatcher:
    def test_watcher_starts_and_stops(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)

        watcher = GraphWatcher(repo, store=store, debounce_seconds=0.05)
        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()
        time.sleep(0.1)
        assert watcher.is_running
        watcher.stop()
        time.sleep(0.2)
        assert not watcher.is_running

    def test_modified_file_triggers_rebuild(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)
        initial_count = store.db_stats()["nodes"]

        rebuilt: list[list[str]] = []

        def cb(reparsed, deleted):
            rebuilt.append(reparsed)

        watcher = GraphWatcher(repo, store=store, debounce_seconds=0.05, on_rebuild=cb)
        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()
        time.sleep(0.2)

        # Modify a file and add a new function
        time.sleep(0.05)
        service = repo / "service.py"
        service.write_text(service.read_text() + "\ndef brand_new(): pass\n")

        # Wait for rebuild to complete
        deadline = time.time() + 3.0
        while not rebuilt and time.time() < deadline:
            time.sleep(0.1)

        watcher.stop()

        assert rebuilt, "Rebuild callback was never called"
        assert "service.py" in rebuilt[0]

    def test_new_node_in_graph_after_rebuild(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)

        rebuilt = threading.Event()

        def cb(reparsed, deleted):
            rebuilt.set()

        watcher = GraphWatcher(repo, store=store, debounce_seconds=0.05, on_rebuild=cb)
        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()
        time.sleep(0.2)

        time.sleep(0.05)
        (repo / "service.py").write_text(
            (repo / "service.py").read_text() + "\ndef watched_fn(): pass\n"
        )

        assert rebuilt.wait(timeout=3.0), "Rebuild never triggered"
        watcher.stop()

        g = store.load()
        assert g.get_node("service.py::watched_fn") is not None

    def test_created_file_triggers_rebuild(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)

        rebuilt = threading.Event()

        def cb(reparsed, deleted):
            rebuilt.set()

        watcher = GraphWatcher(repo, store=store, debounce_seconds=0.05, on_rebuild=cb)
        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()
        time.sleep(0.2)

        time.sleep(0.05)
        (repo / "new_module.py").write_text("def new_fn(): pass\n")

        assert rebuilt.wait(timeout=3.0), "Rebuild never triggered"
        watcher.stop()

        g = store.load()
        assert g.get_node("new_module.py::new_fn") is not None

    def test_deleted_file_invalidated(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)

        # Confirm service.py nodes are there initially
        g = store.load()
        assert g.get_node("service.py::run") is not None

        rebuilt = threading.Event()

        def cb(reparsed, deleted):
            if deleted:
                rebuilt.set()

        watcher = GraphWatcher(repo, store=store, debounce_seconds=0.05, on_rebuild=cb)
        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()
        time.sleep(0.2)

        time.sleep(0.05)
        (repo / "service.py").unlink()

        assert rebuilt.wait(timeout=3.0), "Rebuild never triggered"
        watcher.stop()

        g = store.load()
        assert g.get_node("service.py::run") is None

    def test_on_rebuild_callback_called(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)

        calls: list[tuple] = []
        done = threading.Event()

        def cb(reparsed, deleted):
            calls.append((list(reparsed), list(deleted)))
            done.set()

        watcher = GraphWatcher(repo, store=store, debounce_seconds=0.05, on_rebuild=cb)
        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()
        time.sleep(0.2)

        time.sleep(0.05)
        (repo / "db.py").write_text((repo / "db.py").read_text() + "\n# change\n")

        assert done.wait(timeout=3.0), "Callback never fired"
        watcher.stop()

        assert len(calls) >= 1


# ---------------------------------------------------------------------------
# GraphWatcher + embedder integration
# ---------------------------------------------------------------------------

class TestGraphWatcherWithEmbedder:
    def test_embedder_invalidated_on_delete(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)

        mock_embedder = MagicMock()

        rebuilt = threading.Event()

        def cb(reparsed, deleted):
            if deleted:
                rebuilt.set()

        watcher = GraphWatcher(
            repo,
            store=store,
            embedder=mock_embedder,
            debounce_seconds=0.05,
            on_rebuild=cb,
        )
        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()
        time.sleep(0.2)

        time.sleep(0.05)
        (repo / "service.py").unlink()

        assert rebuilt.wait(timeout=3.0)
        watcher.stop()

        mock_embedder.invalidate_file.assert_called_with("service.py")

    def test_embed_all_called_after_rebuild(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)

        mock_embedder = MagicMock()

        rebuilt = threading.Event()

        def cb(reparsed, deleted):
            if reparsed:
                rebuilt.set()

        watcher = GraphWatcher(
            repo,
            store=store,
            embedder=mock_embedder,
            debounce_seconds=0.05,
            on_rebuild=cb,
        )
        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()
        time.sleep(0.2)

        time.sleep(0.05)
        (repo / "db.py").write_text(
            (repo / "db.py").read_text() + "\ndef extra(): pass\n"
        )

        assert rebuilt.wait(timeout=3.0)
        watcher.stop()

        mock_embedder.embed_all.assert_called()


# ---------------------------------------------------------------------------
# CLI — watch command (smoke test with mocked watcher)
# ---------------------------------------------------------------------------

class TestWatchCLI:
    def test_watch_command_exists(self) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch for file changes" in result.output

    def test_watch_starts_then_stops(self, tmp_path: Path) -> None:
        """Watch command should start, and stop when watcher is stopped."""
        from click.testing import CliRunner
        from ckg.cli import cli

        repo = _make_repo(tmp_path / "repo")
        db = str(tmp_path / "g.db")

        # Pre-build so watch doesn't trigger a full parse
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)

        runner = CliRunner()

        # Patch GraphWatcher.start to stop immediately after a tick
        original_init = GraphWatcher.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Stop after 0.3 s via a thread
            def _auto_stop():
                time.sleep(0.3)
                self.stop()
            threading.Thread(target=_auto_stop, daemon=True).start()

        with patch.object(GraphWatcher, "__init__", patched_init):
            result = runner.invoke(
                cli,
                ["--db", db, "watch", "--repo", str(repo)],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        assert "stopped" in result.output.lower() or "watching" in result.output.lower()
