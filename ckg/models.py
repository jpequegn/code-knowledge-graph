"""Typed node and edge dataclasses for the code knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Parameter info (structured parameter metadata for FunctionNode)
# ---------------------------------------------------------------------------

@dataclass
class ParamInfo:
    """Metadata for a single function parameter."""
    name: str
    annotation: str | None = None   # e.g. "str", "Optional[datetime]", None
    default: str | None = None      # e.g. "None", "0", None (= required param)


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

NodeType = Literal["file", "function", "class", "module"]
EdgeType = Literal["IMPORTS", "CALLS", "DEFINES", "CONTAINS", "RAISES", "INHERITS"]


@dataclass
class FileNode:
    id: str           # relative path, e.g. 'p3/database.py'
    path: str         # same as id (relative to project root)
    line_count: int
    avg_complexity: float = 0.0

    node_type: NodeType = "file"


@dataclass
class FunctionNode:
    id: str           # 'p3/database.py::add_episode'
    name: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: str | None
    return_type: str | None
    cyclomatic_complexity: int   # branches + 1
    is_async: bool
    is_method: bool
    class_name: str | None       # set when this is a method
    param_count: int = 0
    params: list[ParamInfo] = field(default_factory=list)

    node_type: NodeType = "function"


@dataclass
class ClassNode:
    id: str           # 'p3/database.py::P3Database'
    name: str
    file_path: str
    line_start: int
    line_end: int
    bases: list[str]
    docstring: str | None
    method_count: int = 0

    node_type: NodeType = "class"


@dataclass
class ModuleNode:
    id: str           # module name, e.g. 'duckdb' or 'pathlib'
    name: str
    is_stdlib: bool
    is_local: bool

    node_type: NodeType = "module"


Node = FileNode | FunctionNode | ClassNode | ModuleNode


# ---------------------------------------------------------------------------
# Edge type
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    src_id: str
    dst_id: str
    edge_type: EdgeType
    line: int | None = None
    weight: int = 1
    properties: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parse result (output of parse_file)
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    file_node: FileNode
    functions: list[FunctionNode] = field(default_factory=list)
    classes: list[ClassNode] = field(default_factory=list)
    modules: list[ModuleNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
