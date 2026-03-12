"""Tests for ckg.queries.GraphQueries."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ckg.graph import PropertyGraph
from ckg.models import FunctionNode, FileNode
from ckg.queries import GraphQueries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(tmp_path: Path, name: str, source: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(source))
    return p


def _fn(g: PropertyGraph, id: str, name: str, file_path: str = "a.py",
        complexity: int = 1, is_method: bool = False,
        class_name: str | None = None) -> FunctionNode:
    node = FunctionNode(
        id=id, name=name, file_path=file_path,
        line_start=1, line_end=5, signature=f"def {name}()",
        docstring=None, return_type=None,
        cyclomatic_complexity=complexity, is_async=False,
        is_method=is_method, class_name=class_name,
    )
    g.add_node(node)
    return node


def _file(g: PropertyGraph, path: str) -> FileNode:
    node = FileNode(id=path, path=path, line_count=10)
    g.add_node(node)
    return node


# ---------------------------------------------------------------------------
# impact_radius
# ---------------------------------------------------------------------------

class TestImpactRadius:
    def _make_call_chain(self) -> PropertyGraph:
        """
        d calls c, c calls b, b calls a  (a is the deepest dependency)
        query: impact_radius('a') should return b at d=1, c at d=2, d at d=3
        """
        g = PropertyGraph()
        a = _fn(g, "f.py::a", "a")
        b = _fn(g, "f.py::b", "b")
        c = _fn(g, "f.py::c", "c")
        d = _fn(g, "f.py::d", "d")
        g.add_edge("f.py::b", "f.py::a", "CALLS")
        g.add_edge("f.py::c", "f.py::b", "CALLS")
        g.add_edge("f.py::d", "f.py::c", "CALLS")
        return g

    def test_direct_callers_at_depth_1(self) -> None:
        g = self._make_call_chain()
        q = GraphQueries(g)
        result = q.impact_radius("f.py::a", depth=3)
        assert 1 in result
        assert any(fn.name == "b" for fn in result[1])

    def test_transitive_callers_at_depth_2(self) -> None:
        g = self._make_call_chain()
        q = GraphQueries(g)
        result = q.impact_radius("f.py::a", depth=3)
        assert 2 in result
        assert any(fn.name == "c" for fn in result[2])

    def test_depth_limit_respected(self) -> None:
        g = self._make_call_chain()
        q = GraphQueries(g)
        result = q.impact_radius("f.py::a", depth=1)
        assert 1 in result
        assert 2 not in result

    def test_no_callers_returns_empty(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::lonely", "lonely")
        q = GraphQueries(g)
        assert q.impact_radius("f.py::lonely") == {}

    def test_resolves_bare_name(self) -> None:
        g = self._make_call_chain()
        q = GraphQueries(g)
        result = q.impact_radius("a", depth=1)
        assert 1 in result

    def test_diamond_deduplicates(self) -> None:
        """Two paths to a — each caller counted only once."""
        g = PropertyGraph()
        _fn(g, "f.py::a", "a")
        _fn(g, "f.py::b", "b")
        _fn(g, "f.py::c", "c")
        _fn(g, "f.py::d", "d")
        g.add_edge("f.py::b", "f.py::a", "CALLS")
        g.add_edge("f.py::c", "f.py::a", "CALLS")
        g.add_edge("f.py::d", "f.py::b", "CALLS")
        g.add_edge("f.py::d", "f.py::c", "CALLS")
        q = GraphQueries(g)
        result = q.impact_radius("f.py::a", depth=3)
        # d appears at depth 2 only once
        d_nodes = [fn for layer in result.values() for fn in layer if fn.name == "d"]
        assert len(d_nodes) == 1


# ---------------------------------------------------------------------------
# fan_in
# ---------------------------------------------------------------------------

class TestFanIn:
    def test_most_called_first(self) -> None:
        g = PropertyGraph()
        popular  = _fn(g, "f.py::popular",  "popular")
        caller1  = _fn(g, "f.py::caller1",  "caller1")
        caller2  = _fn(g, "f.py::caller2",  "caller2")
        caller3  = _fn(g, "f.py::caller3",  "caller3")
        loner    = _fn(g, "f.py::loner",    "loner")
        for src in ("f.py::caller1", "f.py::caller2", "f.py::caller3"):
            g.add_edge(src, "f.py::popular", "CALLS")

        q = GraphQueries(g)
        result = q.fan_in(top_k=3)
        assert result[0][0].name == "popular"
        assert result[0][1] == 3

    def test_top_k_limits_results(self) -> None:
        g = PropertyGraph()
        for i in range(10):
            _fn(g, f"f.py::fn{i}", f"fn{i}")
        q = GraphQueries(g)
        assert len(q.fan_in(top_k=3)) == 3

    def test_zero_callers_included(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::unused", "unused")
        q = GraphQueries(g)
        result = q.fan_in(top_k=5)
        assert any(fn.name == "unused" and cnt == 0 for fn, cnt in result)


# ---------------------------------------------------------------------------
# uncalled_functions
# ---------------------------------------------------------------------------

class TestUncalledFunctions:
    def test_finds_unused_function(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::used",   "used")
        _fn(g, "f.py::unused", "unused")
        _fn(g, "f.py::caller", "caller")
        g.add_edge("f.py::caller", "f.py::used", "CALLS")
        q = GraphQueries(g)
        result = q.uncalled_functions()
        names = {fn.name for fn in result}
        assert "unused" in names
        assert "used" not in names

    def test_excludes_dunder_methods(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::__init__", "__init__", is_method=True, class_name="Foo")
        _fn(g, "f.py::__str__",  "__str__",  is_method=True, class_name="Foo")
        q = GraphQueries(g)
        result = q.uncalled_functions()
        names = {fn.name for fn in result}
        assert "__init__" not in names
        assert "__str__" not in names

    def test_excludes_main(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::main", "main")
        q = GraphQueries(g)
        result = q.uncalled_functions()
        assert not any(fn.name == "main" for fn in result)

    def test_excludes_test_functions(self) -> None:
        g = PropertyGraph()
        _fn(g, "t.py::test_foo", "test_foo")
        q = GraphQueries(g)
        result = q.uncalled_functions()
        assert not any(fn.name == "test_foo" for fn in result)

    def test_result_sorted_by_file_then_name(self) -> None:
        g = PropertyGraph()
        _fn(g, "b.py::z", "z", file_path="b.py")
        _fn(g, "a.py::m", "m", file_path="a.py")
        _fn(g, "a.py::a", "a", file_path="a.py")
        q = GraphQueries(g)
        result = q.uncalled_functions()
        keys = [(fn.file_path, fn.name) for fn in result]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# complexity_hotspots
# ---------------------------------------------------------------------------

class TestComplexityHotspots:
    def test_highest_complexity_first(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::simple",  "simple",  complexity=1)
        _fn(g, "f.py::medium",  "medium",  complexity=5)
        _fn(g, "f.py::complex", "complex", complexity=12)
        q = GraphQueries(g)
        result = q.complexity_hotspots(top_k=3)
        assert result[0][0].name == "complex"
        assert result[0][1] == 12

    def test_top_k_respected(self) -> None:
        g = PropertyGraph()
        for i in range(10):
            _fn(g, f"f.py::fn{i}", f"fn{i}", complexity=i)
        q = GraphQueries(g)
        assert len(q.complexity_hotspots(top_k=3)) == 3

    def test_returns_function_and_complexity(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::foo", "foo", complexity=7)
        q = GraphQueries(g)
        fn, complexity = q.complexity_hotspots(top_k=1)[0]
        assert isinstance(fn, FunctionNode)
        assert complexity == 7


# ---------------------------------------------------------------------------
# dependency_path
# ---------------------------------------------------------------------------

class TestDependencyPath:
    def _graph_with_imports(self) -> PropertyGraph:
        g = PropertyGraph()
        for p in ("a.py", "b.py", "c.py"):
            _file(g, p)
        g.add_edge("a.py", "b.py", "IMPORTS")
        g.add_edge("b.py", "c.py", "IMPORTS")
        return g

    def test_direct_path(self) -> None:
        g = self._graph_with_imports()
        q = GraphQueries(g)
        path = q.dependency_path("a.py", "b.py")
        assert path == ["a.py", "b.py"]

    def test_transitive_path(self) -> None:
        g = self._graph_with_imports()
        q = GraphQueries(g)
        path = q.dependency_path("a.py", "c.py")
        assert path == ["a.py", "b.py", "c.py"]

    def test_no_path_returns_none(self) -> None:
        g = self._graph_with_imports()
        q = GraphQueries(g)
        assert q.dependency_path("c.py", "a.py") is None

    def test_missing_node_returns_none(self) -> None:
        g = self._graph_with_imports()
        q = GraphQueries(g)
        assert q.dependency_path("a.py", "nonexistent.py") is None

    def test_shortest_path_chosen(self) -> None:
        """Longer alternative path exists — shortest_path should pick the direct one."""
        g = PropertyGraph()
        for p in ("a.py", "b.py", "c.py", "d.py"):
            _file(g, p)
        g.add_edge("a.py", "b.py", "IMPORTS")
        g.add_edge("b.py", "d.py", "IMPORTS")
        g.add_edge("a.py", "c.py", "IMPORTS")
        g.add_edge("c.py", "d.py", "IMPORTS")
        q = GraphQueries(g)
        path = q.dependency_path("a.py", "d.py")
        assert path is not None
        assert len(path) == 3  # a→b→d  or  a→c→d (both length 3)


# ---------------------------------------------------------------------------
# raises_exception
# ---------------------------------------------------------------------------

class TestRaisesException:
    def test_finds_raiser(self) -> None:
        g = PropertyGraph()
        fn = _fn(g, "f.py::boom", "boom")
        g.add_edge("f.py::boom", "ValueError", "RAISES")
        q = GraphQueries(g)
        result = q.raises_exception("ValueError")
        assert len(result) == 1
        assert result[0].name == "boom"

    def test_no_match_returns_empty(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::safe", "safe")
        q = GraphQueries(g)
        assert q.raises_exception("KeyError") == []

    def test_does_not_match_other_exceptions(self) -> None:
        g = PropertyGraph()
        fn = _fn(g, "f.py::foo", "foo")
        g.add_edge("f.py::foo", "TypeError", "RAISES")
        q = GraphQueries(g)
        assert q.raises_exception("ValueError") == []

    def test_multiple_functions_same_exception(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::a", "a")
        _fn(g, "f.py::b", "b")
        g.add_edge("f.py::a", "ValueError", "RAISES")
        g.add_edge("f.py::b", "ValueError", "RAISES")
        q = GraphQueries(g)
        result = q.raises_exception("ValueError")
        assert len(result) == 2

    def test_sorted_by_id(self) -> None:
        g = PropertyGraph()
        _fn(g, "f.py::z", "z")
        _fn(g, "f.py::a", "a")
        g.add_edge("f.py::z", "ValueError", "RAISES")
        g.add_edge("f.py::a", "ValueError", "RAISES")
        q = GraphQueries(g)
        result = q.raises_exception("ValueError")
        ids = [fn.id for fn in result]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# callers / callees
# ---------------------------------------------------------------------------

class TestCallersCallees:
    def _call_graph(self) -> PropertyGraph:
        g = PropertyGraph()
        _fn(g, "f.py::a", "a")
        _fn(g, "f.py::b", "b")
        _fn(g, "f.py::c", "c")
        g.add_edge("f.py::b", "f.py::a", "CALLS")
        g.add_edge("f.py::c", "f.py::a", "CALLS")
        g.add_edge("f.py::a", "f.py::c", "CALLS")
        return g

    def test_callers(self) -> None:
        q = GraphQueries(self._call_graph())
        result = q.callers("f.py::a")
        names = {fn.name for fn in result}
        assert names == {"b", "c"}

    def test_callees(self) -> None:
        q = GraphQueries(self._call_graph())
        result = q.callees("f.py::a")
        names = {fn.name for fn in result}
        assert names == {"c"}

    def test_callers_bare_name(self) -> None:
        q = GraphQueries(self._call_graph())
        result = q.callers("a")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# file_fan_in
# ---------------------------------------------------------------------------

class TestFileFanIn:
    def test_most_imported_first(self) -> None:
        g = PropertyGraph()
        core = _file(g, "core.py")
        for i in range(3):
            f = _file(g, f"user{i}.py")
            g.add_edge(f"user{i}.py", "core.py", "IMPORTS")
        q = GraphQueries(g)
        result = q.file_fan_in(top_k=5)
        assert result[0][0].path == "core.py"
        assert result[0][1] == 3


# ---------------------------------------------------------------------------
# Integration — build from real source and run all queries
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline(self, tmp_path: Path) -> None:
        _make_source(tmp_path, "db.py", """\
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

                def _internal(self):
                    pass
        """)
        _make_source(tmp_path, "service.py", """\
            from db import Database

            def create(db, key, value):
                db.add(key, value)

            def fetch(db, key):
                return db.get(key)

            def helper():
                pass
        """)
        _make_source(tmp_path, "cli.py", """\
            from service import create, fetch

            def run(db, key, value):
                create(db, key, value)
                result = fetch(db, key)
                return result
        """)

        g = PropertyGraph()
        g.build_from_directory(tmp_path)
        q = GraphQueries(g)

        # node/edge sanity
        assert g.node_count(node_type="function") >= 5
        assert g.edge_count(edge_type="CALLS") >= 2
        assert g.edge_count(edge_type="IMPORTS") >= 2

        # complexity hotspots returns something
        hotspots = q.complexity_hotspots(top_k=5)
        assert len(hotspots) > 0
        assert all(isinstance(fn, FunctionNode) for fn, _ in hotspots)

        # fan_in returns something
        fi = q.fan_in(top_k=5)
        assert len(fi) > 0

        # uncalled_functions: helper() is never called
        uncalled = q.uncalled_functions()
        assert any(fn.name == "helper" for fn in uncalled)

        # raises_exception
        raisers = q.raises_exception("KeyError")
        assert len(raisers) >= 1

        # dependency_path: cli → service
        # The parser emits IMPORTS edges to the module name ("service"),
        # so dependency_path resolves "service.py" → "service" internally.
        path = q.dependency_path("cli.py", "service.py")
        assert path is not None
        assert path[0] == "cli.py"
        assert path[-1] in ("service.py", "service")
