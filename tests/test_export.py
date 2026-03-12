"""Tests for ckg.export and the ckg export / ckg query fan-in CLI commands."""

from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path

import pytest

from ckg.export import export_csv, export_dot, export_json, _node_to_dict
from ckg.graph import PropertyGraph
from ckg.models import FunctionNode, ClassNode, FileNode, ModuleNode


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_graph(tmp_path: Path) -> PropertyGraph:
    """Small three-file graph for export tests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "db.py").write_text(textwrap.dedent("""\
        class Database:
            \"\"\"In-memory store.\"\"\"
            def add(self, key: str, value: int) -> None:
                \"\"\"Add a key/value pair.\"\"\"
                self._data[key] = value

            def get(self, key: str) -> int:
                return self._data[key]
    """))
    (tmp_path / "service.py").write_text(textwrap.dedent("""\
        from db import Database
        import os

        def create(db, key, value):
            db.add(key, value)

        def fetch(db, key):
            return db.get(key)

        def unused():
            pass
    """))
    (tmp_path / "cli.py").write_text(textwrap.dedent("""\
        from service import create, fetch

        def run(db, key, value):
            create(db, key, value)
            return fetch(db, key)
    """))
    g = PropertyGraph()
    g.build_from_directory(tmp_path)
    return g


@pytest.fixture()
def graph(tmp_path: Path) -> PropertyGraph:
    return _make_graph(tmp_path)


# ---------------------------------------------------------------------------
# _node_to_dict
# ---------------------------------------------------------------------------

class TestNodeToDict:
    def test_function_node_fields(self) -> None:
        fn = FunctionNode(
            id="f.py::foo", name="foo", file_path="f.py",
            line_start=1, line_end=5, signature="def foo(x: int) -> str",
            docstring="Does foo.", return_type="str", param_count=1,
            cyclomatic_complexity=2, is_async=True, is_method=False,
            class_name=None,
        )
        d = _node_to_dict(fn)
        assert d["id"] == "f.py::foo"
        assert d["type"] == "function"
        assert d["name"] == "foo"
        assert d["signature"] == "def foo(x: int) -> str"
        assert d["return_type"] == "str"
        assert d["is_async"] is True
        assert d["cyclomatic_complexity"] == 2

    def test_class_node_fields(self) -> None:
        cls = ClassNode(
            id="f.py::Bar", name="Bar", file_path="f.py",
            line_start=1, line_end=20, bases=["Base"],
            docstring="A class.", method_count=3,
        )
        d = _node_to_dict(cls)
        assert d["type"] == "class"
        assert d["bases"] == ["Base"]
        assert d["method_count"] == 3

    def test_file_node_fields(self) -> None:
        fnode = FileNode(id="f.py", path="f.py", line_count=42, avg_complexity=3.5)
        d = _node_to_dict(fnode)
        assert d["type"] == "file"
        assert d["line_count"] == 42
        assert d["avg_complexity"] == 3.5

    def test_module_node_fields(self) -> None:
        m = ModuleNode(id="os", name="os", is_stdlib=True, is_local=False)
        d = _node_to_dict(m)
        assert d["type"] == "module"
        assert d["is_stdlib"] is True
        assert d["is_local"] is False


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_returns_valid_json(self, graph: PropertyGraph) -> None:
        text = export_json(graph)
        data = json.loads(text)
        assert "nodes" in data
        assert "edges" in data

    def test_nodes_count(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph))
        assert len(data["nodes"]) == graph.node_count()

    def test_edges_count(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph))
        assert len(data["edges"]) == graph.edge_count()

    def test_node_has_required_fields(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph))
        for node in data["nodes"]:
            assert "id" in node
            assert "type" in node

    def test_edge_has_required_fields(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph))
        for edge in data["edges"]:
            assert "src" in edge
            assert "dst" in edge
            assert "type" in edge

    def test_only_nodes(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph, only="nodes"))
        assert isinstance(data, list)
        assert all("id" in n for n in data)

    def test_only_edges(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph, only="edges"))
        assert isinstance(data, list)
        assert all("src" in e for e in data)

    def test_function_node_signature_present(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph))
        fn_nodes = [n for n in data["nodes"] if n["type"] == "function"]
        assert fn_nodes, "expected at least one function node"
        # signature field should be present (may be empty string but not missing)
        assert all("signature" in n for n in fn_nodes)

    def test_class_bases_is_list(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph))
        cls_nodes = [n for n in data["nodes"] if n["type"] == "class"]
        for cls in cls_nodes:
            assert isinstance(cls.get("bases", []), list)

    def test_nodes_sorted_by_id(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph))
        ids = [n["id"] for n in data["nodes"]]
        assert ids == sorted(ids)

    def test_calls_edges_present(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph))
        types = {e["type"] for e in data["edges"]}
        assert "CALLS" in types

    def test_imports_edges_present(self, graph: PropertyGraph) -> None:
        data = json.loads(export_json(graph))
        types = {e["type"] for e in data["edges"]}
        assert "IMPORTS" in types


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------

class TestExportCsv:
    def test_creates_two_files(self, graph: PropertyGraph, tmp_path: Path) -> None:
        out = tmp_path / "out"
        nodes_p, edges_p = export_csv(graph, output_dir=out)
        assert nodes_p.exists()
        assert edges_p.exists()

    def test_nodes_csv_row_count(self, graph: PropertyGraph, tmp_path: Path) -> None:
        out = tmp_path / "out"
        nodes_p, _ = export_csv(graph, output_dir=out)
        with nodes_p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == graph.node_count()

    def test_edges_csv_row_count(self, graph: PropertyGraph, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _, edges_p = export_csv(graph, output_dir=out)
        with edges_p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == graph.edge_count()

    def test_nodes_csv_has_id_column(self, graph: PropertyGraph, tmp_path: Path) -> None:
        out = tmp_path / "out"
        nodes_p, _ = export_csv(graph, output_dir=out)
        with nodes_p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "id" in (reader.fieldnames or [])

    def test_edges_csv_has_src_dst_type(self, graph: PropertyGraph, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _, edges_p = export_csv(graph, output_dir=out)
        with edges_p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for col in ("src", "dst", "type"):
                assert col in (reader.fieldnames or [])

    def test_list_fields_semicolon_joined(self, graph: PropertyGraph, tmp_path: Path) -> None:
        out = tmp_path / "out"
        nodes_p, _ = export_csv(graph, output_dir=out)
        with nodes_p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cls_rows = [r for r in rows if r["type"] == "class"]
        # bases should not contain JSON brackets — should be plain text or empty
        for row in cls_rows:
            assert "[" not in row.get("bases", "")
            assert "]" not in row.get("bases", "")

    def test_creates_output_dir_if_missing(self, graph: PropertyGraph, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "out"
        assert not out.exists()
        export_csv(graph, output_dir=out)
        assert out.exists()

    def test_returns_correct_paths(self, graph: PropertyGraph, tmp_path: Path) -> None:
        out = tmp_path / "out"
        nodes_p, edges_p = export_csv(graph, output_dir=out)
        assert nodes_p.name == "nodes.csv"
        assert edges_p.name == "edges.csv"


# ---------------------------------------------------------------------------
# export_dot
# ---------------------------------------------------------------------------

class TestExportDot:
    def test_starts_with_digraph(self, graph: PropertyGraph) -> None:
        dot = export_dot(graph)
        assert dot.strip().startswith("digraph")

    def test_ends_with_closing_brace(self, graph: PropertyGraph) -> None:
        dot = export_dot(graph)
        assert dot.strip().endswith("}")

    def test_contains_node_ids(self, graph: PropertyGraph) -> None:
        dot = export_dot(graph)
        # At least one function node ID should appear
        assert "service.py::create" in dot

    def test_contains_edge_arrows(self, graph: PropertyGraph) -> None:
        dot = export_dot(graph)
        assert " -> " in dot

    def test_calls_edges_blue(self, graph: PropertyGraph) -> None:
        dot = export_dot(graph)
        # There should be at least one CALLS edge coloured blue
        assert 'color="blue"' in dot

    def test_imports_edges_dashed(self, graph: PropertyGraph) -> None:
        dot = export_dot(graph)
        assert 'style="dashed"' in dot

    def test_all_node_ids_present(self, graph: PropertyGraph) -> None:
        dot = export_dot(graph)
        for node in graph.iter_nodes():
            assert node.id in dot

    def test_valid_dot_structure(self, graph: PropertyGraph) -> None:
        """Basic structural validity: balanced braces, -> present."""
        dot = export_dot(graph)
        assert dot.count("{") == dot.count("}")
        assert "->" in dot


# ---------------------------------------------------------------------------
# CLI — ckg export
# ---------------------------------------------------------------------------

class TestExportCLI:
    def _build_db(self, tmp_path: Path) -> tuple[Path, Path]:
        from ckg.store import GraphStore
        repo = tmp_path / "repo"
        _make_graph(repo)  # writes files into repo
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)
        return repo, db

    def test_export_json_stdout(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "export", "--format", "json", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # Strip Rich console preamble before the JSON payload
        json_start = result.output.index("{")
        data = json.loads(result.output[json_start:])
        assert "nodes" in data
        assert "edges" in data

    def test_export_json_to_file(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        out_file = str(tmp_path / "graph.json")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db", str(db), "export", "--format", "json",
             "--output", out_file, "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(Path(out_file).read_text())
        assert "nodes" in data

    def test_export_csv_creates_files(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        out_dir = str(tmp_path / "csv_out")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db", str(db), "export", "--format", "csv",
             "--output", out_dir, "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert (tmp_path / "csv_out" / "nodes.csv").exists()
        assert (tmp_path / "csv_out" / "edges.csv").exists()

    def test_export_dot_stdout(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "export", "--format", "dot", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "digraph" in result.output
        assert "->" in result.output

    def test_export_json_only_nodes(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db", str(db), "export", "--format", "json",
             "--only", "nodes", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        json_start = result.output.index("[")
        data = json.loads(result.output[json_start:])
        assert isinstance(data, list)
        assert all("id" in n for n in data)

    def test_export_json_only_edges(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db", str(db), "export", "--format", "json",
             "--only", "edges", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        json_start = result.output.index("[")
        data = json.loads(result.output[json_start:])
        assert isinstance(data, list)
        assert all("src" in e for e in data)


# ---------------------------------------------------------------------------
# CLI — ckg query fan-in
# ---------------------------------------------------------------------------

class TestFanInCLI:
    def _build_db(self, tmp_path: Path) -> tuple[Path, Path]:
        from ckg.store import GraphStore
        repo = tmp_path / "repo"
        _make_graph(repo)
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.build_and_save(repo)
        return repo, db

    def test_fan_in_returns_results(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db", str(db), "query", "fan-in", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # create and fetch are called by run → both should appear
        assert "create" in result.output or "fetch" in result.output

    def test_fan_in_top_limits_output(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db", str(db), "query", "fan-in", "--top", "2", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Top-2" in result.output

    def test_fan_in_shows_caller_count(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli
        repo, db = self._build_db(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db", str(db), "query", "fan-in", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # Should display a numeric caller count in the table
        assert re.search(r"\b[0-9]+\b", result.output)
