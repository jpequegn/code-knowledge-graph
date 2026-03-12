"""Export a PropertyGraph to standard interchange formats.

Supported formats
-----------------
json
    Single JSON object with ``nodes`` and ``edges`` arrays.  Every scalar
    field from the dataclass is included; list fields (e.g. ``bases``) are
    kept as JSON arrays.

csv
    Two files: ``nodes.csv`` and ``edges.csv``.  Written to *output_dir*
    (default: current directory).  List fields are serialised as
    semicolon-separated strings.

dot
    Graphviz DOT language.  Directed graph; node label = name; edge label =
    edge type.  CALLS edges are solid blue, IMPORTS dashed grey, DEFINES /
    CONTAINS light green, INHERITS orange, RAISES red.

Usage::

    from ckg.graph import PropertyGraph
    from ckg.export import export_json, export_csv, export_dot

    g = PropertyGraph()
    g.build_from_directory("my_project/")

    json_str = export_json(g)
    export_csv(g, output_dir=".")
    dot_str  = export_dot(g)
"""

from __future__ import annotations

import csv
import json
import io
from pathlib import Path
from typing import Literal

from ckg.graph import PropertyGraph
from ckg.models import FunctionNode, ClassNode, FileNode, ModuleNode, Node


# ---------------------------------------------------------------------------
# Node → dict
# ---------------------------------------------------------------------------

def _node_to_dict(node: Node) -> dict:
    """Serialise a node dataclass to a plain dict."""
    d: dict = {
        "id": node.id,
        "type": node.node_type,
    }
    if isinstance(node, FunctionNode):
        d.update({
            "name": node.name,
            "file": node.file_path,
            "line_start": node.line_start,
            "line_end": node.line_end,
            "signature": node.signature,
            "docstring": node.docstring,
            "return_type": node.return_type,
            "param_count": node.param_count,
            "cyclomatic_complexity": node.cyclomatic_complexity,
            "is_async": node.is_async,
            "is_method": node.is_method,
            "class_name": node.class_name,
        })
    elif isinstance(node, ClassNode):
        d.update({
            "name": node.name,
            "file": node.file_path,
            "line_start": node.line_start,
            "line_end": node.line_end,
            "bases": node.bases,
            "docstring": node.docstring,
            "method_count": node.method_count,
        })
    elif isinstance(node, FileNode):
        d.update({
            "path": node.path,
            "line_count": node.line_count,
            "avg_complexity": node.avg_complexity,
        })
    elif isinstance(node, ModuleNode):
        d.update({
            "name": node.name,
            "is_stdlib": node.is_stdlib,
            "is_local": node.is_local,
        })
    return d


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def export_json(
    graph: PropertyGraph,
    *,
    only: Literal["nodes", "edges", "both"] = "both",
    indent: int = 2,
) -> str:
    """Return a JSON string representing the graph.

    Parameters
    ----------
    graph:
        The graph to export.
    only:
        ``"nodes"`` — only the nodes array; ``"edges"`` — only edges;
        ``"both"`` (default) — full ``{nodes, edges}`` object.
    indent:
        JSON indentation level.
    """
    nodes_list = sorted(
        (_node_to_dict(n) for n in graph.iter_nodes()),
        key=lambda d: d["id"],
    )
    edges_list = [
        {
            "src": src,
            "dst": dst,
            "type": data.get("edge_type", ""),
            "weight": data.get("weight", 1),
            "line": data.get("line"),
        }
        for src, dst, data in sorted(
            graph.nx_graph.edges(data=True),
            key=lambda t: (t[0], t[1], t[2].get("edge_type", "")),
        )
    ]

    if only == "nodes":
        payload = nodes_list
    elif only == "edges":
        payload = edges_list
    else:
        payload = {"nodes": nodes_list, "edges": edges_list}

    return json.dumps(payload, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

# Canonical column order for each file
_NODE_COLUMNS = [
    "id", "type", "name", "file", "line_start", "line_end",
    "signature", "docstring", "return_type", "param_count",
    "cyclomatic_complexity", "is_async", "is_method", "class_name",
    # class extras
    "bases", "method_count",
    # file extras
    "path", "line_count", "avg_complexity",
    # module extras
    "is_stdlib", "is_local",
]
_EDGE_COLUMNS = ["src", "dst", "type", "weight", "line"]


def _flatten_for_csv(d: dict) -> dict:
    """Convert list fields to semicolon-joined strings for CSV export."""
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = ";".join(str(x) for x in v)
        elif v is None:
            out[k] = ""
        else:
            out[k] = v
    return out


def export_csv(
    graph: PropertyGraph,
    *,
    output_dir: str | Path = ".",
) -> tuple[Path, Path]:
    """Write ``nodes.csv`` and ``edges.csv`` to *output_dir*.

    Returns
    -------
    tuple of (nodes_path, edges_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes_path = out / "nodes.csv"
    edges_path = out / "edges.csv"

    # Nodes
    nodes_rows = sorted(
        (_flatten_for_csv(_node_to_dict(n)) for n in graph.iter_nodes()),
        key=lambda d: d["id"],
    )
    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=_NODE_COLUMNS,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        writer.writerows(nodes_rows)

    # Edges
    edges_rows = [
        {
            "src": src,
            "dst": dst,
            "type": data.get("edge_type", ""),
            "weight": data.get("weight", 1),
            "line": data.get("line", ""),
        }
        for src, dst, data in sorted(
            graph.nx_graph.edges(data=True),
            key=lambda t: (t[0], t[1], t[2].get("edge_type", "")),
        )
    ]
    with edges_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EDGE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(edges_rows)

    return nodes_path, edges_path


# ---------------------------------------------------------------------------
# DOT (Graphviz)
# ---------------------------------------------------------------------------

# Edge style by type
_EDGE_STYLES: dict[str, str] = {
    "CALLS":    'color="blue" style="solid"',
    "IMPORTS":  'color="gray60" style="dashed"',
    "DEFINES":  'color="darkgreen" style="solid"',
    "CONTAINS": 'color="green3" style="solid"',
    "INHERITS": 'color="darkorange" style="solid"',
    "RAISES":   'color="red" style="solid"',
}
_DEFAULT_EDGE_STYLE = 'color="black"'

# Node shape by type
_NODE_SHAPES: dict[str, str] = {
    "function": "ellipse",
    "class":    "box",
    "file":     "folder",
    "module":   "cylinder",
}
_DEFAULT_SHAPE = "ellipse"


def _dot_id(node_id: str) -> str:
    """Escape a node ID for use as a DOT identifier."""
    return '"' + node_id.replace('"', '\\"') + '"'


def export_dot(graph: PropertyGraph) -> str:
    """Return a Graphviz DOT string for the graph.

    Produces a directed graph (``digraph``) with:
    - Node labels = bare name; shape encodes node type
    - Edge labels = edge type; colour/style encodes edge type
    - CALLS edges in blue, IMPORTS dashed grey, DEFINES/CONTAINS green,
      INHERITS orange, RAISES red
    """
    buf = io.StringIO()
    buf.write("digraph ckg {\n")
    buf.write('  graph [rankdir="LR" fontname="Helvetica"];\n')
    buf.write('  node  [fontname="Helvetica" fontsize=10];\n')
    buf.write('  edge  [fontname="Helvetica" fontsize=8];\n\n')

    # Nodes
    for node in sorted(graph.iter_nodes(), key=lambda n: n.id):
        shape = _NODE_SHAPES.get(node.node_type, _DEFAULT_SHAPE)
        label = getattr(node, "name", node.id)
        # Escape special chars in label
        label = label.replace('"', '\\"').replace("\n", "\\n")
        buf.write(
            f"  {_dot_id(node.id)} "
            f'[label="{label}" shape="{shape}"];\n'
        )

    buf.write("\n")

    # Edges
    for src, dst, data in sorted(
        graph.nx_graph.edges(data=True),
        key=lambda t: (t[0], t[1], t[2].get("edge_type", "")),
    ):
        etype = data.get("edge_type", "")
        style = _EDGE_STYLES.get(etype, _DEFAULT_EDGE_STYLE)
        label = etype.replace('"', '\\"')
        buf.write(
            f"  {_dot_id(src)} -> {_dot_id(dst)} "
            f'[label="{label}" {style}];\n'
        )

    buf.write("}\n")
    return buf.getvalue()
