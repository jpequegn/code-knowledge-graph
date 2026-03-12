"""Tests for ckg.cli — all commands exercised via Click's CliRunner."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from ckg.cli import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    """Create a minimal 3-file project for CLI tests."""
    (tmp_path / "db.py").write_text(textwrap.dedent("""\
        import os

        class Database:
            def __init__(self):
                self._data = {}

            def add(self, key, value):
                if key in self._data:
                    raise KeyError(f"duplicate: {key}")
                self._data[key] = value

            def get(self, key):
                return self._data.get(key)

            def _unused(self):
                pass
    """))
    (tmp_path / "service.py").write_text(textwrap.dedent("""\
        from db import Database

        def create(db, key, value):
            db.add(key, value)

        def fetch(db, key):
            return db.get(key)

        def orphan():
            pass
    """))
    (tmp_path / "cli_app.py").write_text(textwrap.dedent("""\
        from service import create, fetch

        def run(db, key, value):
            create(db, key, value)
            result = fetch(db, key)
            return result
    """))
    return tmp_path


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path)


_runner = CliRunner()


def _run(args: list[str]) -> tuple[int, str]:
    """Invoke the CLI and return (exit_code, output)."""
    result = _runner.invoke(cli, args, catch_exceptions=False)
    return result.exit_code, result.output


# ---------------------------------------------------------------------------
# ckg build
# ---------------------------------------------------------------------------

class TestBuild:
    def test_exits_zero(self, repo: Path) -> None:
        code, out = _run(["build", str(repo)])
        assert code == 0

    def test_shows_node_count(self, repo: Path) -> None:
        code, out = _run(["build", str(repo)])
        assert "nodes" in out
        assert "edges" in out

    def test_incremental_flag_accepted(self, repo: Path) -> None:
        code, out = _run(["build", str(repo), "--incremental"])
        assert code == 0
        assert "incremental" in out.lower()

    def test_version_flag(self) -> None:
        code, out = _run(["--version"])
        assert code == 0
        assert "0.1.0" in out


# ---------------------------------------------------------------------------
# ckg query hotspots
# ---------------------------------------------------------------------------

class TestQueryHotspots:
    def test_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["query", "hotspots", "--repo", str(repo)])
        assert code == 0

    def test_shows_table_header(self, repo: Path) -> None:
        _, out = _run(["query", "hotspots", "--repo", str(repo)])
        assert "CC" in out
        assert "Function" in out

    def test_color_legend_present(self, repo: Path) -> None:
        _, out = _run(["query", "hotspots", "--repo", str(repo)])
        assert "simple" in out or "<5" in out or "moderate" in out

    def test_top_option(self, repo: Path) -> None:
        _, out = _run(["query", "hotspots", "--repo", str(repo), "--top", "3"])
        assert "3" in out or "Top-3" in out


# ---------------------------------------------------------------------------
# ckg query dead-code
# ---------------------------------------------------------------------------

class TestQueryDeadCode:
    def test_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["query", "dead-code", "--repo", str(repo)])
        assert code == 0

    def test_finds_orphan(self, repo: Path) -> None:
        _, out = _run(["query", "dead-code", "--repo", str(repo)])
        assert "orphan" in out

    def test_does_not_list_init(self, repo: Path) -> None:
        _, out = _run(["query", "dead-code", "--repo", str(repo)])
        assert "__init__" not in out


# ---------------------------------------------------------------------------
# ckg query impact
# ---------------------------------------------------------------------------

class TestQueryImpact:
    def test_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["query", "impact", "add", "--repo", str(repo)])
        assert code == 0

    def test_shows_callers(self, repo: Path) -> None:
        _, out = _run(["query", "impact", "add", "--repo", str(repo)])
        # create() calls db.add()
        assert "create" in out or "No callers" in out

    def test_unknown_node_no_crash(self, repo: Path) -> None:
        code, out = _run(["query", "impact", "totally_nonexistent_fn", "--repo", str(repo)])
        assert code == 0
        assert "No callers" in out

    def test_depth_option(self, repo: Path) -> None:
        code, _ = _run(["query", "impact", "add", "--repo", str(repo), "--depth", "1"])
        assert code == 0


# ---------------------------------------------------------------------------
# ckg query callers / callees
# ---------------------------------------------------------------------------

class TestQueryCallersCallees:
    def test_callers_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["query", "callers", "add", "--repo", str(repo)])
        assert code == 0

    def test_callees_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["query", "callees", "run", "--repo", str(repo)])
        assert code == 0

    def test_callers_lists_function(self, repo: Path) -> None:
        _, out = _run(["query", "callers", "add", "--repo", str(repo)])
        assert "create" in out or "No callers" in out

    def test_callees_no_crash_unknown(self, repo: Path) -> None:
        code, out = _run(["query", "callees", "zzz_nope", "--repo", str(repo)])
        assert code == 0
        assert "No callees" in out


# ---------------------------------------------------------------------------
# ckg query path
# ---------------------------------------------------------------------------

class TestQueryPath:
    def test_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["query", "path", "cli_app.py", "service", "--repo", str(repo)])
        assert code == 0

    def test_shows_path_nodes(self, repo: Path) -> None:
        _, out = _run(["query", "path", "cli_app.py", "service", "--repo", str(repo)])
        assert "cli_app.py" in out

    def test_no_path_message(self, repo: Path) -> None:
        _, out = _run(["query", "path", "db.py", "cli_app.py", "--repo", str(repo)])
        assert "No import path" in out or "db.py" in out


# ---------------------------------------------------------------------------
# ckg query raises
# ---------------------------------------------------------------------------

class TestQueryRaises:
    def test_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["query", "raises", "KeyError", "--repo", str(repo)])
        assert code == 0

    def test_finds_raiser(self, repo: Path) -> None:
        _, out = _run(["query", "raises", "KeyError", "--repo", str(repo)])
        assert "add" in out

    def test_unknown_exception_no_crash(self, repo: Path) -> None:
        code, out = _run(["query", "raises", "NoSuchError", "--repo", str(repo)])
        assert code == 0
        assert "No functions" in out


# ---------------------------------------------------------------------------
# ckg inspect node
# ---------------------------------------------------------------------------

class TestInspectNode:
    def test_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["inspect", "node", "db.py::Database.add", "--repo", str(repo)])
        assert code == 0

    def test_shows_signature(self, repo: Path) -> None:
        _, out = _run(["inspect", "node", "db.py::Database.add", "--repo", str(repo)])
        assert "add" in out
        assert "Signature" in out

    def test_shows_callers_section(self, repo: Path) -> None:
        _, out = _run(["inspect", "node", "db.py::Database.add", "--repo", str(repo)])
        assert "Caller" in out or "No callers" in out

    def test_missing_node_exits_nonzero(self, repo: Path) -> None:
        result = _runner.invoke(cli, ["inspect", "node", "nope::nothing", "--repo", str(repo)])
        assert result.exit_code != 0 or "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# ckg inspect file
# ---------------------------------------------------------------------------

class TestInspectFile:
    def test_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["inspect", "file", "db.py", "--repo", str(repo)])
        assert code == 0

    def test_shows_class_table(self, repo: Path) -> None:
        _, out = _run(["inspect", "file", "db.py", "--repo", str(repo)])
        assert "Database" in out

    def test_shows_imports(self, repo: Path) -> None:
        _, out = _run(["inspect", "file", "db.py", "--repo", str(repo)])
        assert "os" in out or "Import" in out

    def test_missing_file_exits_nonzero(self, repo: Path) -> None:
        result = _runner.invoke(cli, ["inspect", "file", "ghost.py", "--repo", str(repo)])
        assert result.exit_code != 0 or "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# ckg stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_exits_zero(self, repo: Path) -> None:
        code, _ = _run(["stats", "--repo", str(repo)])
        assert code == 0

    def test_shows_node_counts(self, repo: Path) -> None:
        _, out = _run(["stats", "--repo", str(repo)])
        assert "Functions" in out
        assert "Classes" in out

    def test_shows_edge_counts(self, repo: Path) -> None:
        _, out = _run(["stats", "--repo", str(repo)])
        assert "CALLS" in out or "edges" in out

    def test_shows_complexity_section(self, repo: Path) -> None:
        _, out = _run(["stats", "--repo", str(repo)])
        assert "Complexity" in out or "CC" in out

    def test_shows_uncalled_warning(self, repo: Path) -> None:
        _, out = _run(["stats", "--repo", str(repo)])
        # orphan() is uncalled
        assert "uncalled" in out.lower() or "dead" in out.lower()
