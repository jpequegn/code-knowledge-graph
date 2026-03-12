"""Tests for ckg.parsers.python."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ckg.parsers.python import parse_file, parse_directory
from ckg.models import FunctionNode, ClassNode, FileNode, ModuleNode, Edge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(tmp_path: Path, name: str, source: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(source))
    return p


# ---------------------------------------------------------------------------
# FileNode
# ---------------------------------------------------------------------------

class TestFileNode:
    def test_id_is_relative_path(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", "x = 1\n")
        result = parse_file(src, tmp_path)
        assert result.file_node.id == "foo.py"
        assert result.file_node.path == "foo.py"

    def test_line_count(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", "a = 1\nb = 2\nc = 3\n")
        result = parse_file(src, tmp_path)
        assert result.file_node.line_count == 3

    def test_avg_complexity_no_functions(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", "x = 1\n")
        result = parse_file(src, tmp_path)
        assert result.file_node.avg_complexity == 0.0


# ---------------------------------------------------------------------------
# FunctionNode
# ---------------------------------------------------------------------------

class TestFunctionNode:
    def test_basic_function(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            def greet(name: str) -> str:
                return f"Hello, {name}"
        """)
        result = parse_file(src, tmp_path)
        assert len(result.functions) == 1
        fn = result.functions[0]
        assert fn.name == "greet"
        assert fn.id == "foo.py::greet"
        assert fn.is_async is False
        assert fn.is_method is False
        assert fn.class_name is None
        assert fn.return_type == "str"
        assert fn.cyclomatic_complexity == 1
        assert fn.param_count == 1

    def test_async_function(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            async def fetch(url: str) -> bytes:
                pass
        """)
        result = parse_file(src, tmp_path)
        fn = result.functions[0]
        assert fn.is_async is True

    def test_docstring_extracted(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", '''\
            def foo():
                """My docstring."""
                pass
        ''')
        result = parse_file(src, tmp_path)
        assert result.functions[0].docstring == "My docstring."

    def test_cyclomatic_complexity(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            def process(x):
                if x > 0:
                    for i in range(x):
                        if i % 2 == 0:
                            pass
                elif x < 0:
                    pass
                return x
        """)
        result = parse_file(src, tmp_path)
        fn = result.functions[0]
        # base 1 + if + for + nested_if + elif = 5
        assert fn.cyclomatic_complexity == 5

    def test_method_in_class(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            class MyClass:
                def method(self) -> None:
                    pass
        """)
        result = parse_file(src, tmp_path)
        methods = [f for f in result.functions if f.is_method]
        assert len(methods) == 1
        m = methods[0]
        assert m.class_name == "MyClass"
        assert m.id == "foo.py::MyClass.method"

    def test_line_numbers(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            def foo():
                pass

            def bar():
                pass
        """)
        result = parse_file(src, tmp_path)
        names = {f.name: f for f in result.functions}
        assert names["foo"].line_start == 1
        assert names["bar"].line_start == 4


# ---------------------------------------------------------------------------
# ClassNode
# ---------------------------------------------------------------------------

class TestClassNode:
    def test_basic_class(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            class Animal:
                pass
        """)
        result = parse_file(src, tmp_path)
        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Animal"
        assert cls.id == "foo.py::Animal"
        assert cls.bases == []

    def test_class_with_base(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            class Dog(Animal):
                pass
        """)
        result = parse_file(src, tmp_path)
        cls = result.classes[0]
        assert cls.bases == ["Animal"]

    def test_method_count(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            class Foo:
                def a(self): pass
                def b(self): pass
                def c(self): pass
        """)
        result = parse_file(src, tmp_path)
        assert result.classes[0].method_count == 3

    def test_class_docstring(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", '''\
            class Foo:
                """A foo."""
                pass
        ''')
        result = parse_file(src, tmp_path)
        assert result.classes[0].docstring == "A foo."


# ---------------------------------------------------------------------------
# IMPORTS edges
# ---------------------------------------------------------------------------

class TestImports:
    def test_stdlib_import(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", "import os\n")
        result = parse_file(src, tmp_path)
        import_edges = [e for e in result.edges if e.edge_type == "IMPORTS"]
        assert any(e.dst_id == "os" for e in import_edges)

    def test_from_import(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", "from pathlib import Path\n")
        result = parse_file(src, tmp_path)
        import_edges = [e for e in result.edges if e.edge_type == "IMPORTS"]
        assert any(e.dst_id == "pathlib" for e in import_edges)

    def test_import_src_is_file(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", "import os\n")
        result = parse_file(src, tmp_path)
        import_edges = [e for e in result.edges if e.edge_type == "IMPORTS"]
        assert all(e.src_id == "foo.py" for e in import_edges)

    def test_stdlib_classified(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", "import os\nimport pathlib\n")
        result = parse_file(src, tmp_path)
        for m in result.modules:
            assert m.is_stdlib is True


# ---------------------------------------------------------------------------
# CALLS edges
# ---------------------------------------------------------------------------

class TestCalls:
    def test_simple_call(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            def helper():
                pass

            def main():
                helper()
        """)
        result = parse_file(src, tmp_path)
        call_edges = [e for e in result.edges if e.edge_type == "CALLS"]
        assert any(e.src_id == "foo.py::main" and e.dst_id == "helper" for e in call_edges)

    def test_self_method_call(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            class Foo:
                def bar(self):
                    self.baz()
                def baz(self):
                    pass
        """)
        result = parse_file(src, tmp_path)
        call_edges = [e for e in result.edges if e.edge_type == "CALLS"]
        assert any(
            e.src_id == "foo.py::Foo.bar" and e.dst_id == "foo.py::Foo.baz"
            for e in call_edges
        )

    def test_repeated_calls_weight(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            def helper(): pass
            def main():
                helper()
                helper()
                helper()
        """)
        result = parse_file(src, tmp_path)
        call_edges = [e for e in result.edges if e.edge_type == "CALLS"]
        edge = next(
            (e for e in call_edges if e.src_id == "foo.py::main" and e.dst_id == "helper"),
            None,
        )
        assert edge is not None
        assert edge.weight == 3


# ---------------------------------------------------------------------------
# RAISES edges
# ---------------------------------------------------------------------------

class TestRaises:
    def test_raise_name(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            def boom():
                raise ValueError("oops")
        """)
        result = parse_file(src, tmp_path)
        raise_edges = [e for e in result.edges if e.edge_type == "RAISES"]
        assert any(e.dst_id == "ValueError" for e in raise_edges)

    def test_bare_reraise_ignored(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            def safe():
                try:
                    pass
                except Exception:
                    raise
        """)
        result = parse_file(src, tmp_path)
        raise_edges = [e for e in result.edges if e.edge_type == "RAISES"]
        assert len(raise_edges) == 0


# ---------------------------------------------------------------------------
# DEFINES / CONTAINS edges
# ---------------------------------------------------------------------------

class TestDefinesContains:
    def test_defines_function(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", "def foo(): pass\n")
        result = parse_file(src, tmp_path)
        defines = [e for e in result.edges if e.edge_type == "DEFINES"]
        assert any(e.src_id == "foo.py" and e.dst_id == "foo.py::foo" for e in defines)

    def test_defines_class(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", "class Foo: pass\n")
        result = parse_file(src, tmp_path)
        defines = [e for e in result.edges if e.edge_type == "DEFINES"]
        assert any(e.src_id == "foo.py" and e.dst_id == "foo.py::Foo" for e in defines)

    def test_contains_method(self, tmp_path: Path) -> None:
        src = _make_source(tmp_path, "foo.py", """\
            class Foo:
                def bar(self): pass
        """)
        result = parse_file(src, tmp_path)
        contains = [e for e in result.edges if e.edge_type == "CONTAINS"]
        assert any(
            e.src_id == "foo.py::Foo" and e.dst_id == "foo.py::Foo.bar"
            for e in contains
        )


# ---------------------------------------------------------------------------
# parse_directory
# ---------------------------------------------------------------------------

class TestParseDirectory:
    def test_finds_all_py_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.py").write_text("def b(): pass\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("def c(): pass\n")

        results = parse_directory(tmp_path)
        paths = {r.file_node.path for r in results}
        assert "a.py" in paths
        assert "b.py" in paths
        assert "sub/c.py" in paths

    def test_skips_venv(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x = 1\n")
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "fake.py").write_text("y = 2\n")

        results = parse_directory(tmp_path)
        paths = {r.file_node.path for r in results}
        assert "real.py" in paths
        assert not any(".venv" in p for p in paths)

    def test_skips_syntax_errors(self, tmp_path: Path) -> None:
        (tmp_path / "good.py").write_text("def ok(): pass\n")
        (tmp_path / "bad.py").write_text("def (: pass\n")

        results = parse_directory(tmp_path)
        assert len(results) == 1
        assert results[0].file_node.path == "good.py"
