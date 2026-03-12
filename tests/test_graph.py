"""Tests for ckg.graph.PropertyGraph."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ckg.graph import PropertyGraph
from ckg.models import FunctionNode, ClassNode, FileNode, ModuleNode, Edge, ParseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fn(id: str, name: str = "fn", file_path: str = "a.py") -> FunctionNode:
    return FunctionNode(
        id=id, name=name, file_path=file_path,
        line_start=1, line_end=5, signature=f"def {name}()",
        docstring=None, return_type=None,
        cyclomatic_complexity=1, is_async=False,
        is_method=False, class_name=None,
    )


def _file(path: str = "a.py") -> FileNode:
    return FileNode(id=path, path=path, line_count=10)


def _mod(name: str = "os") -> ModuleNode:
    return ModuleNode(id=name, name=name, is_stdlib=True, is_local=False)


def _edge(src: str, dst: str, etype: str = "CALLS", line: int = 1) -> Edge:
    return Edge(src_id=src, dst_id=dst, edge_type=etype, line=line)  # type: ignore[arg-type]


def _make_source(tmp_path: Path, name: str, source: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(source))
    return p


# ---------------------------------------------------------------------------
# add_node / get_node
# ---------------------------------------------------------------------------

class TestAddGetNode:
    def test_get_node_returns_dataclass(self) -> None:
        g = PropertyGraph()
        fn = _fn("a.py::foo")
        g.add_node(fn)
        assert g.get_node("a.py::foo") is fn

    def test_get_node_missing_returns_none(self) -> None:
        g = PropertyGraph()
        assert g.get_node("nope") is None

    def test_add_node_idempotent(self) -> None:
        g = PropertyGraph()
        fn1 = _fn("a.py::foo")
        fn2 = _fn("a.py::foo")
        g.add_node(fn1)
        g.add_node(fn2)
        assert g.node_count() == 1

    def test_different_node_types(self) -> None:
        g = PropertyGraph()
        g.add_node(_file("a.py"))
        g.add_node(_fn("a.py::foo"))
        g.add_node(_mod("os"))
        assert g.node_count() == 3


# ---------------------------------------------------------------------------
# add_edge
# ---------------------------------------------------------------------------

class TestAddEdge:
    def test_edge_added(self) -> None:
        g = PropertyGraph()
        g.add_node(_fn("a.py::foo"))
        g.add_node(_fn("a.py::bar"))
        g.add_edge("a.py::foo", "a.py::bar", "CALLS")
        assert g.edge_count() == 1

    def test_edge_creates_placeholder_nodes(self) -> None:
        """Endpoints that are not typed nodes should still allow traversal."""
        g = PropertyGraph()
        g.add_edge("unknown_src", "unknown_dst", "IMPORTS")
        assert g.edge_count() == 1

    def test_multiple_edge_types_between_same_pair(self) -> None:
        g = PropertyGraph()
        g.add_node(_fn("a.py::foo"))
        g.add_node(_fn("a.py::bar"))
        g.add_edge("a.py::foo", "a.py::bar", "CALLS")
        g.add_edge("a.py::foo", "a.py::bar", "DEFINES")
        assert g.edge_count() == 2


# ---------------------------------------------------------------------------
# successors / predecessors
# ---------------------------------------------------------------------------

class TestTraversal:
    def _graph_with_calls(self) -> PropertyGraph:
        g = PropertyGraph()
        foo = _fn("a.py::foo", name="foo")
        bar = _fn("a.py::bar", name="bar")
        baz = _fn("a.py::baz", name="baz")
        for n in (foo, bar, baz):
            g.add_node(n)
        g.add_edge("a.py::foo", "a.py::bar", "CALLS")
        g.add_edge("a.py::foo", "a.py::baz", "CALLS")
        g.add_edge("a.py::bar", "a.py::baz", "CALLS")
        return g

    def test_successors_unfiltered(self) -> None:
        g = self._graph_with_calls()
        result = g.successors("a.py::foo")
        ids = {n.id for n in result}
        assert ids == {"a.py::bar", "a.py::baz"}

    def test_successors_filtered_by_type(self) -> None:
        g = self._graph_with_calls()
        g.add_node(_file("a.py"))
        g.add_edge("a.py", "a.py::foo", "DEFINES")
        g.add_edge("a.py", "a.py::bar", "DEFINES")
        calls = g.successors("a.py::foo", edge_type="CALLS")
        defines = g.successors("a.py", edge_type="DEFINES")
        assert {n.id for n in calls} == {"a.py::bar", "a.py::baz"}
        assert {n.id for n in defines} == {"a.py::foo", "a.py::bar"}

    def test_predecessors_unfiltered(self) -> None:
        g = self._graph_with_calls()
        result = g.predecessors("a.py::baz")
        ids = {n.id for n in result}
        assert ids == {"a.py::foo", "a.py::bar"}

    def test_predecessors_filtered(self) -> None:
        g = self._graph_with_calls()
        result = g.predecessors("a.py::bar", edge_type="CALLS")
        assert len(result) == 1
        assert result[0].id == "a.py::foo"

    def test_successors_unknown_node_returns_empty(self) -> None:
        g = PropertyGraph()
        assert g.successors("nope") == []

    def test_predecessors_unknown_node_returns_empty(self) -> None:
        g = PropertyGraph()
        assert g.predecessors("nope") == []

    def test_successors_deduplicates(self) -> None:
        """Two edges of different types to same dst → node appears once."""
        g = PropertyGraph()
        g.add_node(_fn("a.py::foo"))
        g.add_node(_fn("a.py::bar"))
        g.add_edge("a.py::foo", "a.py::bar", "CALLS")
        g.add_edge("a.py::foo", "a.py::bar", "DEFINES")
        result = g.successors("a.py::foo")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# edges_between
# ---------------------------------------------------------------------------

class TestEdgesBetween:
    def test_returns_edge_data(self) -> None:
        g = PropertyGraph()
        g.add_edge("a", "b", "CALLS", line=10, weight=2)
        edges = g.edges_between("a", "b")
        assert len(edges) == 1
        assert edges[0]["edge_type"] == "CALLS"
        assert edges[0]["line"] == 10
        assert edges[0]["weight"] == 2

    def test_filtered_by_type(self) -> None:
        g = PropertyGraph()
        g.add_edge("a", "b", "CALLS")
        g.add_edge("a", "b", "IMPORTS")
        calls = g.edges_between("a", "b", edge_type="CALLS")
        assert len(calls) == 1
        assert calls[0]["edge_type"] == "CALLS"

    def test_missing_pair_returns_empty(self) -> None:
        g = PropertyGraph()
        assert g.edges_between("x", "y") == []


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_node_count_total(self) -> None:
        g = PropertyGraph()
        g.add_node(_file("a.py"))
        g.add_node(_fn("a.py::foo"))
        g.add_node(_fn("a.py::bar"))
        assert g.node_count() == 3

    def test_node_count_by_type_filter(self) -> None:
        g = PropertyGraph()
        g.add_node(_file("a.py"))
        g.add_node(_fn("a.py::foo"))
        g.add_node(_fn("a.py::bar"))
        assert g.node_count(node_type="function") == 2
        assert g.node_count(node_type="file") == 1

    def test_edge_count_total(self) -> None:
        g = PropertyGraph()
        g.add_edge("a", "b", "CALLS")
        g.add_edge("a", "c", "IMPORTS")
        assert g.edge_count() == 2

    def test_edge_count_filtered(self) -> None:
        g = PropertyGraph()
        g.add_edge("a", "b", "CALLS")
        g.add_edge("a", "c", "CALLS")
        g.add_edge("a", "d", "IMPORTS")
        assert g.edge_count(edge_type="CALLS") == 2
        assert g.edge_count(edge_type="IMPORTS") == 1

    def test_edge_count_by_type(self) -> None:
        g = PropertyGraph()
        g.add_edge("a", "b", "CALLS")
        g.add_edge("c", "d", "IMPORTS")
        g.add_edge("e", "f", "DEFINES")
        counts = g.edge_count_by_type()
        assert counts["CALLS"] == 1
        assert counts["IMPORTS"] == 1
        assert counts["DEFINES"] == 1

    def test_node_count_by_type(self) -> None:
        g = PropertyGraph()
        g.add_node(_file("a.py"))
        g.add_node(_fn("a.py::foo"))
        g.add_node(_mod("os"))
        by_type = g.node_count_by_type()
        assert by_type["file"] == 1
        assert by_type["function"] == 1
        assert by_type["module"] == 1


# ---------------------------------------------------------------------------
# iter_nodes / iter_edges
# ---------------------------------------------------------------------------

class TestIteration:
    def test_iter_nodes_all(self) -> None:
        g = PropertyGraph()
        g.add_node(_file("a.py"))
        g.add_node(_fn("a.py::foo"))
        assert len(list(g.iter_nodes())) == 2

    def test_iter_nodes_filtered(self) -> None:
        g = PropertyGraph()
        g.add_node(_file("a.py"))
        g.add_node(_fn("a.py::foo"))
        g.add_node(_fn("a.py::bar"))
        fns = list(g.iter_nodes(node_type="function"))
        assert len(fns) == 2

    def test_iter_edges_all(self) -> None:
        g = PropertyGraph()
        g.add_edge("a", "b", "CALLS")
        g.add_edge("c", "d", "IMPORTS")
        assert len(list(g.iter_edges())) == 2

    def test_iter_edges_filtered(self) -> None:
        g = PropertyGraph()
        g.add_edge("a", "b", "CALLS")
        g.add_edge("c", "d", "IMPORTS")
        calls = list(g.iter_edges(edge_type="CALLS"))
        assert len(calls) == 1
        assert calls[0][2]["edge_type"] == "CALLS"


# ---------------------------------------------------------------------------
# build_from_directory (integration)
# ---------------------------------------------------------------------------

class TestBuildFromDirectory:
    def test_parses_py_files(self, tmp_path: Path) -> None:
        _make_source(tmp_path, "a.py", """\
            import os

            def foo(x: int) -> int:
                return x + 1
        """)
        _make_source(tmp_path, "b.py", """\
            from a import foo

            class Bar:
                def method(self) -> None:
                    foo(1)
        """)
        g = PropertyGraph()
        g.build_from_directory(tmp_path)

        assert g.node_count() > 0
        assert g.node_count(node_type="function") >= 2
        assert g.node_count(node_type="class") >= 1
        assert g.edge_count(edge_type="IMPORTS") >= 2
        assert g.edge_count(edge_type="DEFINES") >= 2
        assert g.edge_count(edge_type="CALLS") >= 1

    def test_edge_count_by_type_has_expected_keys(self, tmp_path: Path) -> None:
        _make_source(tmp_path, "a.py", """\
            import os

            def foo(): pass
            def bar(): foo()
        """)
        g = PropertyGraph()
        g.build_from_directory(tmp_path)
        by_type = g.edge_count_by_type()
        assert "IMPORTS" in by_type
        assert "DEFINES" in by_type
        assert "CALLS" in by_type

    def test_get_node_by_id_after_build(self, tmp_path: Path) -> None:
        _make_source(tmp_path, "mod.py", "def greet(name): pass\n")
        g = PropertyGraph()
        g.build_from_directory(tmp_path)
        node = g.get_node("mod.py::greet")
        assert node is not None
        assert isinstance(node, FunctionNode)
        assert node.name == "greet"

    def test_nx_graph_accessible(self, tmp_path: Path) -> None:
        _make_source(tmp_path, "a.py", "def foo(): pass\n")
        g = PropertyGraph()
        g.build_from_directory(tmp_path)
        import networkx as nx
        assert isinstance(g.nx_graph, nx.MultiDiGraph)
        assert g.nx_graph.number_of_nodes() > 0

    def test_build_from_parse_results(self, tmp_path: Path) -> None:
        from ckg.parsers.python import parse_directory
        _make_source(tmp_path, "a.py", "def foo(): pass\n")
        results = parse_directory(tmp_path)
        g = PropertyGraph()
        g.build_from_parse_results(results)
        assert g.node_count() > 0
