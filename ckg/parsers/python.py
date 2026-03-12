"""AST-based parser for Python source files.

Walks the Python AST of a single file and produces typed node and edge
objects (see ckg.models).  No third-party dependencies — stdlib only.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

from ckg.models import (
    ClassNode,
    Edge,
    FileNode,
    FunctionNode,
    ModuleNode,
    ParamInfo,
    ParseResult,
)

# ---------------------------------------------------------------------------
# Stdlib module names (Python 3.12 top-level packages)
# Used to classify imports as stdlib vs. third-party.
# ---------------------------------------------------------------------------
_STDLIB_TOP_LEVEL: frozenset[str] = frozenset(sys.stdlib_module_names)  # type: ignore[attr-defined]


def _is_stdlib(name: str) -> bool:
    top = name.split(".")[0]
    return top in _STDLIB_TOP_LEVEL


# ---------------------------------------------------------------------------
# Cyclomatic complexity visitor
# ---------------------------------------------------------------------------

class _ComplexityVisitor(ast.NodeVisitor):
    """Count decision points within a single function body."""

    _BOOL_OPS = (ast.And, ast.Or)
    _BRANCH_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ExceptHandler,
        ast.With,
        ast.AsyncWith,
        ast.Assert,
        ast.comprehension,
    )

    def __init__(self) -> None:
        self.complexity = 1  # base complexity

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # each extra operand in an and/or chain adds one branch
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_MatchCase(self, node: ast.MatchCase) -> None:  # Python 3.10+
        self.complexity += 1
        self.generic_visit(node)


def _cyclomatic_complexity(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    v = _ComplexityVisitor()
    v.visit(func_node)
    return v.complexity


# ---------------------------------------------------------------------------
# Annotation → string helper
# ---------------------------------------------------------------------------

def _annotation_to_str(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parameter info extractor
# ---------------------------------------------------------------------------

def _build_params(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ParamInfo]:
    """Return a structured list of :class:`~ckg.models.ParamInfo` for *func*.

    Handles positional-only, regular, keyword-only, *args, and **kwargs.
    Defaults are aligned right (the last N regular args have defaults).
    """
    args_obj = func.args
    params: list[ParamInfo] = []

    # Positional-only args (Python 3.8+) — no defaults support in pos-only
    posonlyargs = args_obj.posonlyargs
    # Regular args
    regargs = args_obj.args
    # Keyword-only args (after *)
    kwonlyargs = args_obj.kwonlyargs

    # Defaults apply right-to-left across posonlyargs + regargs combined
    all_positional = posonlyargs + regargs
    n_defaults = len(args_obj.defaults)
    default_offset = len(all_positional) - n_defaults

    for i, arg in enumerate(all_positional):
        default_idx = i - default_offset
        default_str: str | None = None
        if default_idx >= 0:
            try:
                default_str = ast.unparse(args_obj.defaults[default_idx])
            except Exception:
                default_str = "..."
        params.append(ParamInfo(
            name=arg.arg,
            annotation=_annotation_to_str(arg.annotation),
            default=default_str,
        ))

    # *args
    if args_obj.vararg:
        params.append(ParamInfo(
            name=f"*{args_obj.vararg.arg}",
            annotation=_annotation_to_str(args_obj.vararg.annotation),
            default=None,
        ))

    # Keyword-only args (each may have an individual default from kw_defaults)
    for i, arg in enumerate(kwonlyargs):
        kw_default = args_obj.kw_defaults[i] if i < len(args_obj.kw_defaults) else None
        default_str = None
        if kw_default is not None:
            try:
                default_str = ast.unparse(kw_default)
            except Exception:
                default_str = "..."
        params.append(ParamInfo(
            name=arg.arg,
            annotation=_annotation_to_str(arg.annotation),
            default=default_str,
        ))

    # **kwargs
    if args_obj.kwarg:
        params.append(ParamInfo(
            name=f"**{args_obj.kwarg.arg}",
            annotation=_annotation_to_str(args_obj.kwarg.annotation),
            default=None,
        ))

    return params


# ---------------------------------------------------------------------------
# Signature builder
# ---------------------------------------------------------------------------

def _build_signature(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    class_name: str | None,
) -> str:
    prefix = "async def " if isinstance(func, ast.AsyncFunctionDef) else "def "
    try:
        args_str = ast.unparse(func.args)
    except Exception:
        args_str = "..."
    ret = _annotation_to_str(func.returns)
    ret_part = f" -> {ret}" if ret else ""
    qual = f"{class_name}.{func.name}" if class_name else func.name
    return f"{prefix}{qual}({args_str}){ret_part}"


# ---------------------------------------------------------------------------
# Call target resolution
# ---------------------------------------------------------------------------

def _resolve_call_target(
    call: ast.Call,
    file_path: str,
    class_name: str | None,
    func_name: str,
) -> str | None:
    """Return a best-effort string ID for the callee, or None if unresolvable."""
    node = call.func
    if isinstance(node, ast.Name):
        # bare name: could be a local function or builtin
        return node.id
    if isinstance(node, ast.Attribute):
        attr = node.attr
        if isinstance(node.value, ast.Name):
            obj = node.value.id
            if obj == "self" and class_name:
                # self.method() → resolve to class method in same file
                return f"{file_path}::{class_name}.{attr}"
            if obj == "cls" and class_name:
                return f"{file_path}::{class_name}.{attr}"
            return f"{obj}.{attr}"
        return attr
    return None


# ---------------------------------------------------------------------------
# Main file parser
# ---------------------------------------------------------------------------

class _FileParser(ast.NodeVisitor):
    """Single-pass AST visitor that populates a ParseResult."""

    def __init__(self, source: str, rel_path: str) -> None:
        self._source = source
        self._rel_path = rel_path  # e.g. 'p3/database.py'

        # current scope tracking
        self._current_class: str | None = None
        self._current_func: str | None = None

        # output collectors
        self.functions: list[FunctionNode] = []
        self.classes: list[ClassNode] = []
        self.modules: list[ModuleNode] = []
        self.edges: list[Edge] = []

        # dedup helpers
        self._seen_modules: set[str] = set()
        # call edges: (src_id, dst_raw) → edge  (for weight accumulation)
        self._call_edge_map: dict[tuple[str, str], Edge] = {}

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def _add_module(self, name: str, is_local: bool) -> ModuleNode:
        if name not in self._seen_modules:
            m = ModuleNode(
                id=name,
                name=name,
                is_stdlib=_is_stdlib(name),
                is_local=is_local,
            )
            self.modules.append(m)
            self._seen_modules.add(name)
        return next(m for m in self.modules if m.id == name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod_name = alias.name
            self._add_module(mod_name, is_local=False)
            self.edges.append(Edge(
                src_id=self._rel_path,
                dst_id=mod_name,
                edge_type="IMPORTS",
                line=node.lineno,
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        mod_name = node.module
        # Relative imports (level > 0) are local
        is_local = (node.level or 0) > 0
        self._add_module(mod_name, is_local=is_local)
        self.edges.append(Edge(
            src_id=self._rel_path,
            dst_id=mod_name,
            edge_type="IMPORTS",
            line=node.lineno,
        ))
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = f"{self._rel_path}::{node.name}"
        bases = [ast.unparse(b) for b in node.bases]
        docstring = ast.get_docstring(node)

        # Count methods (visit children first, then count)
        method_names = [
            n.name
            for n in ast.walk(node)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n is not node
        ]

        cls = ClassNode(
            id=class_id,
            name=node.name,
            file_path=self._rel_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            bases=bases,
            docstring=docstring,
            method_count=len(method_names),
            decorators=[ast.unparse(d) for d in node.decorator_list],
        )
        self.classes.append(cls)

        # DEFINES edge: file → class
        self.edges.append(Edge(
            src_id=self._rel_path,
            dst_id=class_id,
            edge_type="DEFINES",
            line=node.lineno,
        ))

        # INHERITS edges
        for base in bases:
            self.edges.append(Edge(
                src_id=class_id,
                dst_id=base,
                edge_type="INHERITS",
                line=node.lineno,
            ))

        # Recurse into class body with class context
        prev_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = prev_class

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        class_name = self._current_class
        func_name = node.name

        # Build qualified name (ClassName.method or plain function)
        qualified = f"{class_name}.{func_name}" if class_name else func_name
        func_id = f"{self._rel_path}::{qualified}"

        signature = _build_signature(node, class_name)
        docstring = ast.get_docstring(node)
        return_type = _annotation_to_str(node.returns)
        complexity = _cyclomatic_complexity(node)
        param_count = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
        params = _build_params(node)
        decorators = [ast.unparse(d) for d in node.decorator_list]

        fn = FunctionNode(
            id=func_id,
            name=func_name,
            file_path=self._rel_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=signature,
            docstring=docstring,
            return_type=return_type,
            cyclomatic_complexity=complexity,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=class_name is not None,
            class_name=class_name,
            param_count=param_count,
            params=params,
            decorators=decorators,
        )
        self.functions.append(fn)

        # DEFINES edge: file → function
        self.edges.append(Edge(
            src_id=self._rel_path,
            dst_id=func_id,
            edge_type="DEFINES",
            line=node.lineno,
        ))

        # CONTAINS edge: class → method
        if class_name:
            class_id = f"{self._rel_path}::{class_name}"
            self.edges.append(Edge(
                src_id=class_id,
                dst_id=func_id,
                edge_type="CONTAINS",
                line=node.lineno,
            ))

        # Walk body for CALLS and RAISES
        prev_func = self._current_func
        self._current_func = func_id
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self._process_call(child, func_id, class_name)
            elif isinstance(child, ast.Raise):
                self._process_raise(child, func_id)
        self._current_func = prev_func

        # Do NOT call generic_visit here — we handled the body manually above.
        # But we still need to recurse for nested classes/functions.
        # Use generic_visit with the class context unchanged.
        prev_class = self._current_class
        self._current_class = class_name  # keep same class for nested defs
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(child)
        self._current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def _process_call(self, call: ast.Call, src_func_id: str, class_name: str | None) -> None:
        target = _resolve_call_target(call, self._rel_path, class_name, src_func_id)
        if target is None:
            return
        key = (src_func_id, target)
        if key in self._call_edge_map:
            self._call_edge_map[key].weight += 1
        else:
            edge = Edge(
                src_id=src_func_id,
                dst_id=target,
                edge_type="CALLS",
                line=call.lineno if hasattr(call, "lineno") else None,
                weight=1,
            )
            self._call_edge_map[key] = edge
            self.edges.append(edge)

    # ------------------------------------------------------------------
    # Raises
    # ------------------------------------------------------------------

    def _process_raise(self, raise_node: ast.Raise, src_func_id: str) -> None:
        exc = raise_node.exc
        if exc is None:
            return  # bare re-raise
        exc_name: str | None = None
        if isinstance(exc, ast.Name):
            exc_name = exc.id
        elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            exc_name = exc.func.id
        elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Attribute):
            exc_name = exc.func.attr
        if exc_name:
            self.edges.append(Edge(
                src_id=src_func_id,
                dst_id=exc_name,
                edge_type="RAISES",
                line=raise_node.lineno,
            ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(path: Path | str, project_root: Path | str) -> ParseResult:
    """Parse a single Python file and return a :class:`ParseResult`.

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.py`` file.
    project_root:
        Root of the project.  Used to compute the relative path that
        becomes the node ID (e.g. ``p3/database.py``).
    """
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()

    source = path.read_text(encoding="utf-8", errors="replace")
    rel_path = str(path.relative_to(project_root))

    lines = source.splitlines()
    line_count = len(lines)

    tree = ast.parse(source, filename=str(path))

    visitor = _FileParser(source, rel_path)
    visitor.visit(tree)

    avg_complexity = (
        sum(f.cyclomatic_complexity for f in visitor.functions) / len(visitor.functions)
        if visitor.functions
        else 0.0
    )

    file_node = FileNode(
        id=rel_path,
        path=rel_path,
        line_count=line_count,
        avg_complexity=round(avg_complexity, 2),
    )

    return ParseResult(
        file_node=file_node,
        functions=visitor.functions,
        classes=visitor.classes,
        modules=visitor.modules,
        edges=visitor.edges,
    )


def parse_directory(root: Path | str) -> list[ParseResult]:
    """Recursively parse all ``.py`` files under *root*.

    Skips files inside ``.venv``, ``__pycache__``, ``.git``, ``dist``,
    ``build``, and ``*.egg-info`` directories.
    """
    root = Path(root).resolve()
    _SKIP_DIRS = {".venv", "__pycache__", ".git", "dist", "build", ".mypy_cache", ".ruff_cache"}

    results: list[ParseResult] = []
    for py_file in sorted(root.rglob("*.py")):
        # Skip unwanted directories
        if any(part in _SKIP_DIRS or part.endswith(".egg-info") for part in py_file.parts):
            continue
        try:
            results.append(parse_file(py_file, root))
        except SyntaxError:
            # Skip files with syntax errors
            continue
    return results
