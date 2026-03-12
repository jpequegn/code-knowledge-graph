"""In-memory property graph backed by networkx MultiDiGraph.

Usage
-----
    from ckg.graph import PropertyGraph

    graph = PropertyGraph()
    graph.build_from_directory("p3/")

    fn = graph.get_node("p3/database.py::add_episode")
    callers = graph.predecessors("p3/database.py::add_episode", edge_type="CALLS")
    callees = graph.successors("p3/database.py::add_episode", edge_type="CALLS")

    print(graph.node_count())
    print(graph.edge_count_by_type())
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import networkx as nx

from ckg.models import (
    ClassNode,
    Edge,
    EdgeType,
    FileNode,
    FunctionNode,
    ModuleNode,
    Node,
    ParseResult,
)
from ckg.parsers.python import parse_directory, parse_file


class PropertyGraph:
    """A directed property graph over a Python codebase.

    Backed by :class:`networkx.MultiDiGraph` — supports multiple typed
    edges between the same pair of nodes (e.g. a file that imports a
    module *and* is defined by it).

    Node data is stored in a flat dict keyed by node ID for O(1) lookup;
    the networkx graph is used for traversal only.
    """

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        # id → dataclass instance
        self._nodes: dict[str, Node] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add *node* to the graph (idempotent on repeated calls with same id)."""
        self._nodes[node.id] = node
        self._graph.add_node(node.id, node_type=node.node_type)

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: EdgeType,
        *,
        line: int | None = None,
        weight: int = 1,
        **props,
    ) -> None:
        """Add a directed edge *src_id → dst_id* of *edge_type*.

        If either endpoint is not yet in the graph a placeholder node is
        created so that traversal never raises KeyError.
        """
        for nid in (src_id, dst_id):
            if nid not in self._graph:
                self._graph.add_node(nid)

        self._graph.add_edge(
            src_id,
            dst_id,
            edge_type=edge_type,
            line=line,
            weight=weight,
            **props,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Node | None:
        """Return the typed node for *node_id*, or ``None`` if not found."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    # ------------------------------------------------------------------
    # Typed traversal
    # ------------------------------------------------------------------

    def successors(
        self,
        node_id: str,
        *,
        edge_type: EdgeType | None = None,
    ) -> list[Node]:
        """Return nodes reachable from *node_id* via one outgoing edge.

        Parameters
        ----------
        node_id:
            Source node ID.
        edge_type:
            If given, only edges of this type are followed.
        """
        results: list[Node] = []
        for dst, edge_data_dict in self._graph[node_id].items() if node_id in self._graph else []:
            for _, data in edge_data_dict.items():
                if edge_type is None or data.get("edge_type") == edge_type:
                    if (n := self._nodes.get(dst)) is not None:
                        results.append(n)
                    break  # one entry per (src, dst) pair is enough for node list
        # deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[Node] = []
        for n in results:
            if n.id not in seen:
                seen.add(n.id)
                deduped.append(n)
        return deduped

    def predecessors(
        self,
        node_id: str,
        *,
        edge_type: EdgeType | None = None,
    ) -> list[Node]:
        """Return nodes that have an edge pointing *to* *node_id*.

        Parameters
        ----------
        node_id:
            Target node ID.
        edge_type:
            If given, only edges of this type are considered.
        """
        results: list[Node] = []
        if node_id not in self._graph:
            return results
        for src in self._graph.predecessors(node_id):
            edge_data_dict = self._graph[src][node_id]
            for _, data in edge_data_dict.items():
                if edge_type is None or data.get("edge_type") == edge_type:
                    if (n := self._nodes.get(src)) is not None:
                        results.append(n)
                    break
        seen: set[str] = set()
        deduped: list[Node] = []
        for n in results:
            if n.id not in seen:
                seen.add(n.id)
                deduped.append(n)
        return deduped

    def edges_between(
        self,
        src_id: str,
        dst_id: str,
        *,
        edge_type: EdgeType | None = None,
    ) -> list[dict]:
        """Return edge attribute dicts for all edges from *src_id* to *dst_id*."""
        if src_id not in self._graph or dst_id not in self._graph[src_id]:
            return []
        return [
            data
            for data in self._graph[src_id][dst_id].values()
            if edge_type is None or data.get("edge_type") == edge_type
        ]

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_nodes(
        self, *, node_type: str | None = None
    ) -> Iterator[Node]:
        """Iterate over all typed nodes, optionally filtered by *node_type*."""
        for node in self._nodes.values():
            if node_type is None or node.node_type == node_type:
                yield node

    def iter_edges(
        self, *, edge_type: EdgeType | None = None
    ) -> Iterator[tuple[str, str, dict]]:
        """Yield ``(src_id, dst_id, data)`` triples, optionally filtered."""
        for src, dst, data in self._graph.edges(data=True):
            if edge_type is None or data.get("edge_type") == edge_type:
                yield src, dst, data

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def node_count(self, *, node_type: str | None = None) -> int:
        """Total number of typed nodes (optionally filtered by *node_type*)."""
        if node_type is None:
            return len(self._nodes)
        return sum(1 for n in self._nodes.values() if n.node_type == node_type)

    def edge_count(self, *, edge_type: EdgeType | None = None) -> int:
        """Total number of edges (optionally filtered by *edge_type*)."""
        if edge_type is None:
            return self._graph.number_of_edges()
        return sum(1 for _, _, d in self._graph.edges(data=True) if d.get("edge_type") == edge_type)

    def edge_count_by_type(self) -> dict[str, int]:
        """Return a mapping ``{edge_type: count}`` for all edge types present."""
        counts: dict[str, int] = defaultdict(int)
        for _, _, data in self._graph.edges(data=True):
            et = data.get("edge_type", "UNKNOWN")
            counts[et] += 1
        return dict(counts)

    def node_count_by_type(self) -> dict[str, int]:
        """Return a mapping ``{node_type: count}`` for all typed nodes."""
        counts: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            counts[node.node_type] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # Build from source
    # ------------------------------------------------------------------

    def _ingest_parse_result(self, result: ParseResult) -> None:
        """Add all nodes and edges from one :class:`ParseResult`."""
        # Nodes
        self.add_node(result.file_node)
        for fn in result.functions:
            self.add_node(fn)
        for cls in result.classes:
            self.add_node(cls)
        for mod in result.modules:
            self.add_node(mod)

        # Edges
        for edge in result.edges:
            self.add_edge(
                edge.src_id,
                edge.dst_id,
                edge.edge_type,
                line=edge.line,
                weight=edge.weight,
            )

    def build_from_parse_results(self, results: list[ParseResult]) -> None:
        """Populate the graph from a pre-computed list of parse results."""
        for result in results:
            self._ingest_parse_result(result)

    def build_from_file(
        self, path: Path | str, project_root: Path | str
    ) -> None:
        """Parse a single file and add its nodes/edges to the graph."""
        result = parse_file(path, project_root)
        self._ingest_parse_result(result)

    def build_from_directory(self, root: Path | str) -> None:
        """Recursively parse all ``.py`` files under *root* and build the graph."""
        results = parse_directory(root)
        self.build_from_parse_results(results)

    # ------------------------------------------------------------------
    # NetworkX passthrough (for queries that need raw graph access)
    # ------------------------------------------------------------------

    @property
    def nx_graph(self) -> nx.MultiDiGraph:
        """Direct access to the underlying :class:`networkx.MultiDiGraph`."""
        return self._graph

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        by_type = self.node_count_by_type()
        by_edge = self.edge_count_by_type()
        return (
            f"PropertyGraph("
            f"nodes={self.node_count()} {dict(by_type)}, "
            f"edges={self.edge_count()} {dict(by_edge)})"
        )
