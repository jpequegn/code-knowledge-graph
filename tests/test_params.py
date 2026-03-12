"""Tests for structured parameter info (ParamInfo), new queries, and CLI commands."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ckg.models import FunctionNode, ClassNode, ParamInfo
from ckg.graph import PropertyGraph
from ckg.parsers.python import parse_file, _build_params
from ckg.queries import GraphQueries
from ckg.store import GraphStore

import ast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models.py").write_text(textwrap.dedent("""\
        from pydantic import BaseModel
        from typing import Optional, List
        from datetime import datetime

        class Episode(BaseModel):
            \"\"\"A podcast episode.\"\"\"
            title: str
            url: str

        class Feed(BaseModel):
            \"\"\"A podcast feed.\"\"\"
            name: str
            episodes: List[Episode]

        class Timestamped(Episode):
            created: datetime
    """))
    (tmp_path / "service.py").write_text(textwrap.dedent("""\
        from datetime import datetime
        from typing import Optional

        async def fetch_episode(url: str, timeout: int = 30) -> dict:
            \"\"\"Fetch an episode by URL.\"\"\"
            return {}

        async def list_episodes(
            page: int = 1,
            since: Optional[datetime] = None,
        ) -> list:
            return []

        def process(name: str, count: int, tags: list = None) -> bool:
            return True

        def no_annotations(x, y, z=0):
            return x + y + z

        def varargs_fn(*args: str, **kwargs: int) -> None:
            pass

        def kwonly_fn(a: str, *, b: int = 5, c: float) -> str:
            return a
    """))
    (tmp_path / "db.py").write_text(textwrap.dedent("""\
        class Database:
            def __init__(self):
                self._data = {}

            def add(self, key: str, value: int) -> None:
                self._data[key] = value

            def get(self, key: str) -> int:
                return self._data[key]
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
# ParamInfo — model
# ---------------------------------------------------------------------------

class TestParamInfoModel:
    def test_default_values(self) -> None:
        p = ParamInfo(name="x")
        assert p.annotation is None
        assert p.default is None

    def test_full_construction(self) -> None:
        p = ParamInfo(name="count", annotation="int", default="0")
        assert p.name == "count"
        assert p.annotation == "int"
        assert p.default == "0"


# ---------------------------------------------------------------------------
# _build_params — parser helper
# ---------------------------------------------------------------------------

class TestBuildParams:
    def _parse_fn(self, src: str) -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(src))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        raise ValueError("no function found")

    def test_simple_annotated_params(self) -> None:
        fn = self._parse_fn("def f(name: str, count: int): pass")
        params = _build_params(fn)
        assert len(params) == 2
        assert params[0].name == "name"
        assert params[0].annotation == "str"
        assert params[0].default is None
        assert params[1].name == "count"
        assert params[1].annotation == "int"

    def test_default_values(self) -> None:
        fn = self._parse_fn("def f(x: int = 0, y: str = 'hi'): pass")
        params = _build_params(fn)
        assert params[0].default == "0"
        assert params[1].default == "'hi'"

    def test_no_annotations(self) -> None:
        fn = self._parse_fn("def f(a, b, c=None): pass")
        params = _build_params(fn)
        assert all(p.annotation is None for p in params)
        assert params[2].default == "None"

    def test_varargs(self) -> None:
        fn = self._parse_fn("def f(*args: str, **kwargs: int): pass")
        params = _build_params(fn)
        names = [p.name for p in params]
        assert "*args" in names
        assert "**kwargs" in names

    def test_kwonly_args(self) -> None:
        fn = self._parse_fn("def f(a, *, b: int = 5, c: float): pass")
        params = _build_params(fn)
        names = [p.name for p in params]
        assert "a" in names
        assert "b" in names
        assert "c" in names
        b_param = next(p for p in params if p.name == "b")
        assert b_param.annotation == "int"
        assert b_param.default == "5"

    def test_optional_type(self) -> None:
        fn = self._parse_fn("def f(x: Optional[datetime] = None): pass")
        params = _build_params(fn)
        assert params[0].annotation == "Optional[datetime]"
        assert params[0].default == "None"

    def test_empty_params(self) -> None:
        fn = self._parse_fn("def f(): pass")
        params = _build_params(fn)
        assert params == []


# ---------------------------------------------------------------------------
# Parser — FunctionNode.params populated
# ---------------------------------------------------------------------------

class TestParserPopulatesParams:
    def test_params_list_populated(self, repo: Path) -> None:
        result = parse_file(repo / "service.py", repo)
        fn = next(f for f in result.functions if f.name == "fetch_episode")
        assert len(fn.params) == 2

    def test_param_names_correct(self, repo: Path) -> None:
        result = parse_file(repo / "service.py", repo)
        fn = next(f for f in result.functions if f.name == "fetch_episode")
        assert fn.params[0].name == "url"
        assert fn.params[1].name == "timeout"

    def test_param_annotations_correct(self, repo: Path) -> None:
        result = parse_file(repo / "service.py", repo)
        fn = next(f for f in result.functions if f.name == "fetch_episode")
        assert fn.params[0].annotation == "str"
        assert fn.params[1].annotation == "int"

    def test_param_default_captured(self, repo: Path) -> None:
        result = parse_file(repo / "service.py", repo)
        fn = next(f for f in result.functions if f.name == "fetch_episode")
        assert fn.params[0].default is None   # required
        assert fn.params[1].default == "30"

    def test_optional_datetime_param(self, repo: Path) -> None:
        result = parse_file(repo / "service.py", repo)
        fn = next(f for f in result.functions if f.name == "list_episodes")
        since_param = next(p for p in fn.params if p.name == "since")
        assert "datetime" in (since_param.annotation or "")

    def test_no_annotation_params_captured(self, repo: Path) -> None:
        result = parse_file(repo / "service.py", repo)
        fn = next(f for f in result.functions if f.name == "no_annotations")
        assert len(fn.params) == 3
        assert all(p.annotation is None for p in fn.params)

    def test_varargs_params(self, repo: Path) -> None:
        result = parse_file(repo / "service.py", repo)
        fn = next(f for f in result.functions if f.name == "varargs_fn")
        names = [p.name for p in fn.params]
        assert "*args" in names
        assert "**kwargs" in names

    def test_kwonly_params(self, repo: Path) -> None:
        result = parse_file(repo / "service.py", repo)
        fn = next(f for f in result.functions if f.name == "kwonly_fn")
        names = [p.name for p in fn.params]
        assert "b" in names
        assert "c" in names

    def test_method_params_include_self(self, repo: Path) -> None:
        result = parse_file(repo / "db.py", repo)
        fn = next(f for f in result.functions if f.name == "add")
        assert fn.params[0].name == "self"

    def test_param_count_consistent(self, repo: Path) -> None:
        """param_count should equal number of non-vararg params."""
        result = parse_file(repo / "service.py", repo)
        fn = next(f for f in result.functions if f.name == "process")
        # process(name, count, tags) — 3 positional params
        assert fn.param_count == 3


# ---------------------------------------------------------------------------
# Store round-trip for params
# ---------------------------------------------------------------------------

class TestStoreParamsRoundTrip:
    def test_params_survive_save_load(self, repo: Path, tmp_path: Path) -> None:
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)
        g = store.load()

        fn = g.get_node("service.py::fetch_episode")
        assert isinstance(fn, FunctionNode)
        assert len(fn.params) == 2
        assert fn.params[0].name == "url"
        assert fn.params[0].annotation == "str"
        assert fn.params[1].default == "30"

    def test_empty_params_survive_save_load(self, repo: Path, tmp_path: Path) -> None:
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)
        g = store.load()

        # db.py::Database.__init__ has only self
        fn = g.get_node("db.py::Database.__init__")
        assert isinstance(fn, FunctionNode)
        assert fn.params[0].name == "self"

    def test_optional_param_annotation_survives(self, repo: Path, tmp_path: Path) -> None:
        store = GraphStore(tmp_path / "g.db")
        store.build_and_save(repo)
        g = store.load()

        fn = g.get_node("service.py::list_episodes")
        assert isinstance(fn, FunctionNode)
        since_param = next(p for p in fn.params if p.name == "since")
        assert "datetime" in (since_param.annotation or "")


# ---------------------------------------------------------------------------
# GraphQueries — async_functions
# ---------------------------------------------------------------------------

class TestAsyncFunctions:
    def test_returns_async_only(self, queries: GraphQueries) -> None:
        async_fns = queries.async_functions()
        assert all(f.is_async for f in async_fns)

    def test_finds_fetch_episode(self, queries: GraphQueries) -> None:
        async_fns = queries.async_functions()
        ids = [f.id for f in async_fns]
        assert "service.py::fetch_episode" in ids

    def test_finds_list_episodes(self, queries: GraphQueries) -> None:
        async_fns = queries.async_functions()
        ids = [f.id for f in async_fns]
        assert "service.py::list_episodes" in ids

    def test_excludes_sync_functions(self, queries: GraphQueries) -> None:
        async_fns = queries.async_functions()
        ids = [f.id for f in async_fns]
        assert "service.py::process" not in ids

    def test_sorted_by_id(self, queries: GraphQueries) -> None:
        async_fns = queries.async_functions()
        ids = [f.id for f in async_fns]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# GraphQueries — subclasses
# ---------------------------------------------------------------------------

class TestSubclasses:
    def test_finds_pydantic_subclasses(self, queries: GraphQueries) -> None:
        subs = queries.subclasses("BaseModel")
        names = [c.name for c in subs]
        assert "Episode" in names
        assert "Feed" in names

    def test_direct_only(self, queries: GraphQueries) -> None:
        # Timestamped inherits Episode (not BaseModel directly)
        subs = queries.subclasses("BaseModel")
        names = [c.name for c in subs]
        assert "Timestamped" not in names

    def test_direct_subclass_of_episode(self, queries: GraphQueries) -> None:
        subs = queries.subclasses("Episode")
        names = [c.name for c in subs]
        assert "Timestamped" in names

    def test_no_match_returns_empty(self, queries: GraphQueries) -> None:
        assert queries.subclasses("NonExistentBase") == []

    def test_sorted_by_id(self, queries: GraphQueries) -> None:
        subs = queries.subclasses("BaseModel")
        ids = [c.id for c in subs]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# GraphQueries — functions_with_param_type
# ---------------------------------------------------------------------------

class TestFunctionsWithParamType:
    def test_finds_datetime_params(self, queries: GraphQueries) -> None:
        fns = queries.functions_with_param_type("datetime")
        ids = [f.id for f in fns]
        assert "service.py::list_episodes" in ids

    def test_finds_str_params(self, queries: GraphQueries) -> None:
        fns = queries.functions_with_param_type("str")
        ids = [f.id for f in fns]
        assert "service.py::fetch_episode" in ids
        assert "service.py::process" in ids

    def test_substring_match(self, queries: GraphQueries) -> None:
        # "datetime" should match "Optional[datetime]"
        fns = queries.functions_with_param_type("datetime", substring=True)
        ids = [f.id for f in fns]
        assert "service.py::list_episodes" in ids

    def test_exact_match_excludes_optional(self, queries: GraphQueries) -> None:
        # Exact "datetime" should NOT match "Optional[datetime]"
        fns = queries.functions_with_param_type("datetime", substring=False)
        ids = [f.id for f in fns]
        assert "service.py::list_episodes" not in ids

    def test_no_match_returns_empty(self, queries: GraphQueries) -> None:
        assert queries.functions_with_param_type("NonExistentType") == []

    def test_each_function_appears_once(self, queries: GraphQueries) -> None:
        """Even if a function has multiple str params, it appears only once."""
        fns = queries.functions_with_param_type("str")
        ids = [f.id for f in fns]
        assert len(ids) == len(set(ids))

    def test_sorted_by_id(self, queries: GraphQueries) -> None:
        fns = queries.functions_with_param_type("str")
        ids = [f.id for f in fns]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# CLI — new subcommands
# ---------------------------------------------------------------------------

class TestNewCLICommands:
    def _build_db(self, tmp_path: Path) -> tuple[Path, Path]:
        repo = _make_repo(tmp_path / "repo")
        db = tmp_path / "g.db"
        GraphStore(db).build_and_save(repo)
        return repo, db

    def test_async_command(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "query", "async", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "fetch_episode" in result.output
        assert "list_episodes" in result.output

    def test_inherits_command(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "query", "inherits", "BaseModel", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Episode" in result.output
        assert "Feed" in result.output

    def test_inherits_no_match(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "query", "inherits", "Ghost", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "No classes" in result.output

    def test_param_type_command(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "query", "param-type", "datetime", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "list_episodes" in result.output

    def test_file_fan_in_command(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "query", "file-fan-in", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Most-Imported" in result.output

    def test_inspect_node_shows_params(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db", str(db), "inspect", "node", "service.py::fetch_episode",
             "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # Should show param names and types
        assert "url" in result.output
        assert "str" in result.output
