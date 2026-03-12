"""Tests for decorator capture, transitive queries, and new CLI subcommands."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ckg.graph import PropertyGraph
from ckg.models import FunctionNode, ClassNode
from ckg.parsers.python import parse_file
from ckg.queries import GraphQueries
from ckg.store import GraphStore


# ---------------------------------------------------------------------------
# Test repo
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)

    (tmp_path / "api.py").write_text(textwrap.dedent("""\
        from fastapi import FastAPI
        from typing import List

        app = FastAPI()

        @app.get("/items")
        async def list_items() -> List[dict]:
            \"\"\"List all items.\"\"\"
            return []

        @app.post("/items")
        @app.get("/items/{item_id}")
        async def get_item(item_id: int) -> dict:
            return {}

        def internal_helper():
            pass
    """))

    (tmp_path / "cli.py").write_text(textwrap.dedent("""\
        import click
        from service import process, fetch

        @click.group()
        def main():
            pass

        @main.command()
        @click.argument("name")
        def run(name: str) -> None:
            \"\"\"Run the command.\"\"\"
            result = process(name)
            return fetch(result)

        @main.command()
        def status() -> None:
            pass
    """))

    (tmp_path / "service.py").write_text(textwrap.dedent("""\
        from db import Database

        def process(name: str) -> str:
            db = Database()
            db.save(name)
            return name.upper()

        def fetch(key: str) -> str:
            return key
    """))

    (tmp_path / "db.py").write_text(textwrap.dedent("""\
        class Database:
            \"\"\"Simple DB.\"\"\"
            def save(self, value: str) -> None:
                pass

            def load(self, key: str) -> str:
                return key
    """))

    (tmp_path / "models.py").write_text(textwrap.dedent("""\
        from pydantic import BaseModel

        @staticmethod
        def standalone():
            pass

        class Item(BaseModel):
            name: str

        @dataclass_decorator
        class Config:
            debug: bool = False
    """))

    return tmp_path


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path / "repo")


@pytest.fixture()
def graph(repo: Path) -> PropertyGraph:
    g = PropertyGraph()
    g.build_from_directory(repo)
    return g


@pytest.fixture()
def queries(graph: PropertyGraph) -> GraphQueries:
    return GraphQueries(graph)


# ---------------------------------------------------------------------------
# Parser — decorator capture on FunctionNode
# ---------------------------------------------------------------------------

class TestParserDecorators:
    def test_single_decorator_captured(self, repo: Path) -> None:
        result = parse_file(repo / "api.py", repo)
        fn = next(f for f in result.functions if f.name == "list_items")
        assert len(fn.decorators) == 1
        assert "app.get" in fn.decorators[0]

    def test_multiple_decorators_captured(self, repo: Path) -> None:
        result = parse_file(repo / "api.py", repo)
        fn = next(f for f in result.functions if f.name == "get_item")
        assert len(fn.decorators) == 2

    def test_click_decorator_captured(self, repo: Path) -> None:
        result = parse_file(repo / "cli.py", repo)
        fn = next(f for f in result.functions if f.name == "run")
        dec_strs = " ".join(fn.decorators)
        assert "main.command" in dec_strs
        assert "click.argument" in dec_strs

    def test_no_decorator_gives_empty_list(self, repo: Path) -> None:
        result = parse_file(repo / "api.py", repo)
        fn = next(f for f in result.functions if f.name == "internal_helper")
        assert fn.decorators == []

    def test_decorator_is_string(self, repo: Path) -> None:
        result = parse_file(repo / "api.py", repo)
        fn = next(f for f in result.functions if f.name == "list_items")
        assert all(isinstance(d, str) for d in fn.decorators)

    def test_decorator_contains_route_path(self, repo: Path) -> None:
        result = parse_file(repo / "api.py", repo)
        fn = next(f for f in result.functions if f.name == "list_items")
        assert "/items" in fn.decorators[0]


# ---------------------------------------------------------------------------
# Parser — decorator capture on ClassNode
# ---------------------------------------------------------------------------

class TestParserClassDecorators:
    def test_class_decorator_captured(self, repo: Path) -> None:
        result = parse_file(repo / "models.py", repo)
        cls = next(c for c in result.classes if c.name == "Config")
        assert "dataclass_decorator" in cls.decorators[0]

    def test_class_without_decorator_empty(self, repo: Path) -> None:
        result = parse_file(repo / "models.py", repo)
        cls = next(c for c in result.classes if c.name == "Item")
        assert cls.decorators == []


# ---------------------------------------------------------------------------
# Store round-trip — decorators
# ---------------------------------------------------------------------------

class TestStoreDecoratorsRoundTrip:
    def test_function_decorators_survive(self, repo: Path, tmp_path: Path) -> None:
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)
        g = store.load()

        fn = g.get_node("api.py::list_items")
        assert isinstance(fn, FunctionNode)
        assert len(fn.decorators) == 1
        assert "app.get" in fn.decorators[0]

    def test_multiple_decorators_survive(self, repo: Path, tmp_path: Path) -> None:
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)
        g = store.load()

        fn = g.get_node("api.py::get_item")
        assert isinstance(fn, FunctionNode)
        assert len(fn.decorators) == 2

    def test_empty_decorators_survive(self, repo: Path, tmp_path: Path) -> None:
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)
        g = store.load()

        fn = g.get_node("api.py::internal_helper")
        assert isinstance(fn, FunctionNode)
        assert fn.decorators == []

    def test_class_decorators_survive(self, repo: Path, tmp_path: Path) -> None:
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)
        g = store.load()

        cls = g.get_node("models.py::Config")
        assert isinstance(cls, ClassNode)
        assert any("dataclass_decorator" in d for d in cls.decorators)


# ---------------------------------------------------------------------------
# GraphQueries — functions_with_decorator
# ---------------------------------------------------------------------------

class TestFunctionsWithDecorator:
    def test_finds_app_get_routes(self, queries: GraphQueries) -> None:
        fns = queries.functions_with_decorator("app.get")
        ids = [f.id for f in fns]
        assert "api.py::list_items" in ids
        assert "api.py::get_item" in ids

    def test_finds_click_commands(self, queries: GraphQueries) -> None:
        fns = queries.functions_with_decorator("command")
        ids = [f.id for f in fns]
        assert "cli.py::run" in ids
        assert "cli.py::status" in ids

    def test_excludes_undecorated(self, queries: GraphQueries) -> None:
        fns = queries.functions_with_decorator("app.get")
        ids = [f.id for f in fns]
        assert "api.py::internal_helper" not in ids
        assert "service.py::process" not in ids

    def test_substring_match(self, queries: GraphQueries) -> None:
        # "app" matches both "app.get" and "app.post"
        fns = queries.functions_with_decorator("app.")
        assert len(fns) >= 2

    def test_exact_match(self, queries: GraphQueries) -> None:
        # Only "app.get('/items')" should match exactly
        fns = queries.functions_with_decorator(
            "app.get('/items')", substring=False
        )
        assert len(fns) == 1
        assert fns[0].id == "api.py::list_items"

    def test_no_match_returns_empty(self, queries: GraphQueries) -> None:
        assert queries.functions_with_decorator("nonexistent_decorator") == []

    def test_each_function_appears_once(self, queries: GraphQueries) -> None:
        # get_item has two decorators matching "app." — should appear only once
        fns = queries.functions_with_decorator("app.")
        ids = [f.id for f in fns]
        assert len(ids) == len(set(ids))

    def test_sorted_by_id(self, queries: GraphQueries) -> None:
        fns = queries.functions_with_decorator("app.")
        ids = [f.id for f in fns]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# GraphQueries — transitive_deps
# ---------------------------------------------------------------------------

class TestTransitiveDeps:
    def test_cli_sees_service_and_db(self, queries: GraphQueries) -> None:
        deps = queries.transitive_deps("cli.py")
        # cli imports service, service imports db
        assert "service" in deps or "service.py" in deps
        assert "db" in deps or "db.py" in deps

    def test_direct_import_included(self, queries: GraphQueries) -> None:
        deps = queries.transitive_deps("cli.py")
        # cli directly imports click
        assert "click" in deps

    def test_excludes_self(self, queries: GraphQueries) -> None:
        deps = queries.transitive_deps("cli.py")
        assert "cli.py" not in deps
        assert "cli" not in deps

    def test_nonexistent_file_returns_empty(self, queries: GraphQueries) -> None:
        deps = queries.transitive_deps("nonexistent.py")
        assert deps == set()

    def test_leaf_file_has_only_direct_deps(self, queries: GraphQueries) -> None:
        # db.py imports nothing
        deps = queries.transitive_deps("db.py")
        # may be empty or just stdlib
        assert "cli.py" not in deps

    def test_returns_set(self, queries: GraphQueries) -> None:
        result = queries.transitive_deps("cli.py")
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# GraphQueries — transitive_callers
# ---------------------------------------------------------------------------

class TestTransitiveCallers:
    def test_finds_direct_caller(self, queries: GraphQueries) -> None:
        # process() is called by run()
        callers = queries.transitive_callers("service.py::process")
        ids = [f.id for f in callers]
        assert "cli.py::run" in ids

    def test_finds_indirect_callers(self, queries: GraphQueries) -> None:
        # fetch() is called by run(); run itself has no callers
        # so transitive callers of fetch should include run
        callers = queries.transitive_callers("service.py::fetch")
        ids = [f.id for f in callers]
        # run calls fetch directly
        assert "cli.py::run" in ids

    def test_no_callers_returns_empty(self, queries: GraphQueries) -> None:
        # list_items is an endpoint — nothing calls it
        callers = queries.transitive_callers("api.py::list_items")
        assert callers == []

    def test_returns_function_nodes_only(self, queries: GraphQueries) -> None:
        callers = queries.transitive_callers("service.py::process")
        assert all(isinstance(f, FunctionNode) for f in callers)

    def test_no_duplicates(self, queries: GraphQueries) -> None:
        callers = queries.transitive_callers("service.py::process")
        ids = [f.id for f in callers]
        assert len(ids) == len(set(ids))

    def test_sorted_by_id(self, queries: GraphQueries) -> None:
        callers = queries.transitive_callers("service.py::process")
        ids = [f.id for f in callers]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# CLI — new subcommands
# ---------------------------------------------------------------------------

class TestNewCLISubcommands:
    def _build_db(self, tmp_path: Path) -> tuple[Path, Path]:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        GraphStore(db).build_and_save(repo)
        return repo, db

    def test_decorator_command_finds_routes(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "query", "decorator", "app.get",
                  "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "list_items" in result.output
        assert "get_item" in result.output

    def test_decorator_command_no_match(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "query", "decorator", "nonexistent",
                  "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "No functions" in result.output

    def test_transitive_deps_command(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "query", "transitive-deps", "cli.py",
                  "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # Should show total count and some dependency names
        assert "Transitive dependencies" in result.output
        assert "service" in result.output or "click" in result.output

    def test_transitive_callers_command(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "query", "transitive-callers",
                  "service.py::process", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "run" in result.output

    def test_inspect_node_shows_decorators(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "inspect", "node", "api.py::list_items",
                  "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Decorators" in result.output
        assert "app.get" in result.output

    def test_inspect_node_no_decorators_row_absent(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "inspect", "node", "api.py::internal_helper",
                  "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # No decorators → row should not appear
        assert "Decorators" not in result.output
