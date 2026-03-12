"""Tests for ckg.store.GraphStore."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import pytest

from ckg.graph import PropertyGraph
from ckg.models import FunctionNode, ClassNode, FileNode, ModuleNode
from ckg.store import GraphStore, _invalidate_in_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    """Three-file repo used across most tests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "db.py").write_text(textwrap.dedent("""\
        import os

        class Database:
            def __init__(self):
                self._data = {}

            def add(self, key, value):
                if key in self._data:
                    raise KeyError("dup")
                self._data[key] = value

            def get(self, key):
                return self._data.get(key)
    """))
    (tmp_path / "service.py").write_text(textwrap.dedent("""\
        from db import Database

        def create(db, key, value):
            db.add(key, value)

        def fetch(db, key):
            return db.get(key)

        def helper():
            pass
    """))
    (tmp_path / "cli.py").write_text(textwrap.dedent("""\
        from service import create, fetch

        def run(db, key, value):
            create(db, key, value)
            return fetch(db, key)
    """))
    return tmp_path


# ---------------------------------------------------------------------------
# GraphStore initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        db = tmp_path / "subdir" / "graph.db"
        store = GraphStore(db)
        assert db.parent.exists()

    def test_default_path_is_home_ckg(self) -> None:
        store = GraphStore()
        assert str(store._db_path).endswith("graph.db")
        assert ".ckg" in str(store._db_path)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        store = GraphStore(str(tmp_path / "g.db"))
        assert store._db_path.suffix == ".db"


# ---------------------------------------------------------------------------
# Save and load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_node_count_survives_round_trip(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)

        g1 = store.build_and_save(repo)
        g2 = store.load()

        assert g2.node_count() == g1.node_count()

    def test_edge_count_survives_round_trip(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)

        g1 = store.build_and_save(repo)
        g2 = store.load()

        assert g2.edge_count() == g1.edge_count()

    def test_function_node_fields_survive(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        g = store.load()
        node = g.get_node("service.py::create")
        assert isinstance(node, FunctionNode)
        assert node.name == "create"
        assert node.file_path == "service.py"
        assert node.line_start > 0

    def test_file_node_fields_survive(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        g = store.load()
        node = g.get_node("db.py")
        assert isinstance(node, FileNode)
        assert node.path == "db.py"
        assert node.line_count > 0

    def test_class_node_fields_survive(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        g = store.load()
        node = g.get_node("db.py::Database")
        assert isinstance(node, ClassNode)
        assert node.name == "Database"
        assert node.method_count >= 3

    def test_module_node_fields_survive(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        g = store.load()
        node = g.get_node("os")
        assert isinstance(node, ModuleNode)
        assert node.is_stdlib is True

    def test_edge_types_survive(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        g1 = store.build_and_save(repo)
        g2 = store.load()

        assert g2.edge_count_by_type().get("CALLS", 0) > 0
        assert g2.edge_count_by_type().get("IMPORTS", 0) > 0
        assert g2.edge_count_by_type().get("DEFINES", 0) > 0

    def test_traversal_works_after_load(self, tmp_path: Path) -> None:
        from ckg.queries import GraphQueries
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        g = store.load()
        fn = g.get_node("service.py::create")
        assert fn is not None
        # Use GraphQueries.callers() which handles bare-name CALLS edges
        # (cli.py::run calls create(...) → dst_id="create", not full ID)
        q = GraphQueries(g)
        callers = q.callers("service.py::create")
        assert any(n.id == "cli.py::run" for n in callers)

    def test_empty_db_load_returns_empty_graph(self, tmp_path: Path) -> None:
        store = GraphStore(tmp_path / "empty.db")
        g = store.load()
        assert g.node_count() == 0
        assert g.edge_count() == 0

    def test_save_overwrites_previous(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)
        count1 = store.db_stats()["nodes"]

        # Second save with the same data should give same count
        store.build_and_save(repo)
        count2 = store.db_stats()["nodes"]
        assert count1 == count2


# ---------------------------------------------------------------------------
# Mtime tracking
# ---------------------------------------------------------------------------

class TestMtime:
    def test_needs_reparse_true_for_new_file(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        new_file = repo / "new.py"
        new_file.write_text("def fresh(): pass\n")
        assert store.needs_reparse("new.py", repo) is True

    def test_needs_reparse_false_for_unchanged_file(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)
        assert store.needs_reparse("db.py", repo) is False

    def test_needs_reparse_true_after_modification(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        # Modify the file slightly in the future so mtime definitely changes
        target = repo / "service.py"
        time.sleep(0.05)
        target.write_text(target.read_text() + "\n# modified\n")
        assert store.needs_reparse("service.py", repo) is True

    def test_stored_files_lists_tracked_paths(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        tracked = store.stored_files()
        assert "db.py" in tracked
        assert "service.py" in tracked
        assert "cli.py" in tracked


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

class TestInvalidation:
    def test_invalidate_removes_file_node(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        store.invalidate_file("service.py")
        g = store.load()
        assert g.get_node("service.py") is None

    def test_invalidate_removes_function_nodes(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        store.invalidate_file("service.py")
        g = store.load()
        assert g.get_node("service.py::create") is None
        assert g.get_node("service.py::fetch") is None

    def test_invalidate_removes_mtime_record(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        store.invalidate_file("service.py")
        assert "service.py" not in store.stored_files()

    def test_invalidate_does_not_remove_other_files(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        store.invalidate_file("service.py")
        g = store.load()
        assert g.get_node("db.py") is not None
        assert g.get_node("db.py::Database") is not None


# ---------------------------------------------------------------------------
# Incremental rebuild
# ---------------------------------------------------------------------------

class TestIncrementalRebuild:
    def test_unchanged_repo_reparses_nothing(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        _, reparsed = store.rebuild_incremental(repo)
        assert reparsed == []

    def test_modified_file_is_reparsed(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        time.sleep(0.05)
        (repo / "service.py").write_text(
            (repo / "service.py").read_text() + "\ndef new_fn(): pass\n"
        )

        _, reparsed = store.rebuild_incremental(repo)
        assert "service.py" in reparsed
        assert "db.py" not in reparsed

    def test_incremental_preserves_unmodified_nodes(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        time.sleep(0.05)
        (repo / "service.py").write_text(
            (repo / "service.py").read_text() + "\ndef added(): pass\n"
        )

        g, _ = store.rebuild_incremental(repo)
        # db.py untouched — its nodes must still be present
        assert g.get_node("db.py::Database") is not None
        # New function added to service.py
        assert g.get_node("service.py::added") is not None

    def test_new_file_is_parsed(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        time.sleep(0.05)
        (repo / "extra.py").write_text("def extra_fn(): pass\n")

        g, reparsed = store.rebuild_incremental(repo)
        assert "extra.py" in reparsed
        assert g.get_node("extra.py::extra_fn") is not None

    def test_node_edge_counts_consistent_after_incremental(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        g_full = store.build_and_save(repo)

        time.sleep(0.05)
        (repo / "cli.py").write_text(
            (repo / "cli.py").read_text() + "\ndef extra(): pass\n"
        )

        g_inc, _ = store.rebuild_incremental(repo)
        # Incremental graph should have at least as many nodes as full build
        # (one extra function added)
        assert g_inc.node_count() >= g_full.node_count()


# ---------------------------------------------------------------------------
# db_stats
# ---------------------------------------------------------------------------

class TestDbStats:
    def test_stats_after_build(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)

        stats = store.db_stats()
        assert stats["nodes"] > 0
        assert stats["edges"] > 0
        assert stats["tracked_files"] == 3  # db.py, service.py, cli.py


# ---------------------------------------------------------------------------
# _invalidate_in_graph helper
# ---------------------------------------------------------------------------

class TestInvalidateInGraph:
    def test_removes_nodes_with_prefix(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        g = PropertyGraph()
        g.build_from_directory(repo)

        assert g.get_node("service.py") is not None
        _invalidate_in_graph(g, "service.py")
        assert g.get_node("service.py") is None
        assert g.get_node("service.py::create") is None

    def test_leaves_other_nodes_intact(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        g = PropertyGraph()
        g.build_from_directory(repo)

        _invalidate_in_graph(g, "service.py")
        assert g.get_node("db.py") is not None
        assert g.get_node("db.py::Database") is not None
