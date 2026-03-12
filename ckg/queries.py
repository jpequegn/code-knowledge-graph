"""Structural queries over a PropertyGraph.

These answer real engineering questions that flat text search cannot:

    from ckg.graph import PropertyGraph
    from ckg.queries import GraphQueries

    g = PropertyGraph()
    g.build_from_directory("my_project/")
    q = GraphQueries(g)

    q.impact_radius("database.py::add_episode", depth=3)
    q.fan_in(top_k=10)
    q.uncalled_functions()
    q.complexity_hotspots(top_k=10)
    q.dependency_path("cli.py", "database.py")
    q.raises_exception("ValueError")
    q.callers("add_episode")
    q.callees("add_episode")
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import networkx as nx

from ckg.models import FunctionNode, FileNode, ModuleNode, Node

if TYPE_CHECKING:
    from ckg.graph import PropertyGraph

# Functions excluded from "uncalled" analysis — entry points / magic methods
# that are legitimately never called from within the codebase itself.
_UNCALLED_EXCLUDE: frozenset[str] = frozenset(
    {
        "__init__",
        "__new__",
        "__del__",
        "__repr__",
        "__str__",
        "__len__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__contains__",
        "__iter__",
        "__next__",
        "__enter__",
        "__exit__",
        "__eq__",
        "__hash__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__call__",
        "__class_getitem__",
        "main",  # CLI entry points
        "setup",
        "teardown",
        # pytest conventions
        "test_",  # prefix — checked separately
    }
)


def _is_excluded_from_uncalled(fn: FunctionNode) -> bool:
    """Return True if *fn* should be excluded from the uncalled-functions list."""
    name = fn.name
    # dunder methods
    if name.startswith("__") and name.endswith("__"):
        return True
    # pytest test functions / methods
    if name.startswith("test_") or name.startswith("Test"):
        return True
    # explicit exclude list
    if name in _UNCALLED_EXCLUDE:
        return True
    return False


class GraphQueries:
    """Named structural queries over a :class:`~ckg.graph.PropertyGraph`."""

    def __init__(self, graph: "PropertyGraph") -> None:
        self._g = graph

    # ------------------------------------------------------------------
    # 1. Impact radius
    # ------------------------------------------------------------------

    def impact_radius(
        self,
        node_id: str,
        *,
        depth: int = 3,
    ) -> dict[int, list[FunctionNode]]:
        """BFS over CALLS *predecessors* up to *depth* hops.

        Returns a dict mapping distance → list of FunctionNodes at that
        distance.  Distance 1 = direct callers, 2 = their callers, etc.

        Parameters
        ----------
        node_id:
            Full node ID of the function whose impact you want to measure.
            Accepts a bare function name and will resolve it if unambiguous.
        depth:
            Maximum BFS depth (default 3).
        """
        node_id = self._resolve_id(node_id)
        result: dict[int, list[FunctionNode]] = {}

        visited: set[str] = {node_id}
        frontier: set[str] = {node_id}

        for d in range(1, depth + 1):
            next_frontier: set[str] = set()
            for nid in frontier:
                # Expand via exact node-ID in-edges
                for src, _, data in self._g.nx_graph.in_edges(nid, data=True):
                    if data.get("edge_type") == "CALLS" and src not in visited:
                        next_frontier.add(src)
                        visited.add(src)
                # Also expand via bare-name in-edges (unresolved self.obj.method() calls)
                node = self._g.get_node(nid)
                if isinstance(node, FunctionNode):
                    bare = node.name
                    if bare != nid and bare in self._g.nx_graph:
                        for src, _, data in self._g.nx_graph.in_edges(bare, data=True):
                            if data.get("edge_type") == "CALLS" and src not in visited:
                                next_frontier.add(src)
                                visited.add(src)
            if not next_frontier:
                break
            frontier = next_frontier  # advance BFS wave
            layer: list[FunctionNode] = []
            for nid in sorted(next_frontier):
                node = self._g.get_node(nid)
                if isinstance(node, FunctionNode):
                    layer.append(node)
            if layer:
                result[d] = layer

        return result

    # ------------------------------------------------------------------
    # 2. Fan-in  (most-called functions)
    # ------------------------------------------------------------------

    def fan_in(self, *, top_k: int = 10) -> list[tuple[FunctionNode, int]]:
        """Return the *top_k* functions with the most distinct callers.

        Returns a list of ``(FunctionNode, caller_count)`` tuples sorted
        descending by caller count.
        """
        counts: dict[str, int] = {}
        for nid, node in self._g._nodes.items():
            if not isinstance(node, FunctionNode):
                continue
            caller_count = sum(
                1
                for _, _, data in self._g.nx_graph.in_edges(nid, data=True)
                if data.get("edge_type") == "CALLS"
            )
            counts[nid] = caller_count

        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            (self._g.get_node(nid), cnt)  # type: ignore[misc]
            for nid, cnt in ranked
            if self._g.get_node(nid) is not None
        ]

    # ------------------------------------------------------------------
    # 3. Uncalled functions  (dead code candidates)
    # ------------------------------------------------------------------

    def uncalled_functions(self) -> list[FunctionNode]:
        """Return functions with zero CALLS-predecessors inside the graph.

        Excludes dunder methods, ``main``, pytest entry points, and other
        functions that are legitimately never called from within the
        codebase itself.
        """
        result: list[FunctionNode] = []
        for nid, node in self._g._nodes.items():
            if not isinstance(node, FunctionNode):
                continue
            if _is_excluded_from_uncalled(node):
                continue
            caller_count = sum(
                1
                for _, _, data in self._g.nx_graph.in_edges(nid, data=True)
                if data.get("edge_type") == "CALLS"
            )
            if caller_count == 0:
                result.append(node)

        return sorted(result, key=lambda fn: (fn.file_path, fn.name))

    # ------------------------------------------------------------------
    # 4. Complexity hotspots
    # ------------------------------------------------------------------

    def complexity_hotspots(
        self, *, top_k: int = 10
    ) -> list[tuple[FunctionNode, int]]:
        """Return the *top_k* functions ranked by cyclomatic complexity (desc).

        Returns a list of ``(FunctionNode, complexity)`` tuples.
        """
        fns = [
            (node, node.cyclomatic_complexity)
            for node in self._g._nodes.values()
            if isinstance(node, FunctionNode)
        ]
        fns.sort(key=lambda t: t[1], reverse=True)
        return fns[:top_k]

    # ------------------------------------------------------------------
    # 5. Dependency path  (file-level shortest path via IMPORTS)
    # ------------------------------------------------------------------

    def dependency_path(
        self, src_file: str, dst_file: str
    ) -> list[str] | None:
        """Return the shortest IMPORTS-edge path from *src_file* to *dst_file*.

        Returns a list of file/module IDs representing the path, or
        ``None`` if no path exists.

        Accepts relative file paths (e.g. ``cli.py``) or bare module names.
        The parser emits IMPORTS edges whose destination is the module name
        (e.g. ``service``), so this method tries to resolve a ``.py`` filename
        to its bare module name when looking up the destination node.
        """
        # Build a simple DiGraph of IMPORTS edges only (faster for path search)
        imports_view = nx.DiGraph(
            (src, dst)
            for src, dst, data in self._g.nx_graph.edges(data=True)
            if data.get("edge_type") == "IMPORTS"
        )

        # Resolve file paths → best reachable node ID.
        # The parser emits IMPORTS edges whose destination is the module name
        # (e.g. "service"), not the file path (e.g. "service.py").
        # So for a given "service.py" argument we try both forms and pick
        # whichever has the most in-edges (i.e. is actually a dependency target).
        def _resolve(name: str) -> str:
            stem = name[:-3] if name.endswith(".py") else name
            candidates = [name, stem]
            # prefer the candidate that appears as a destination of IMPORTS edges
            for c in candidates:
                if imports_view.in_degree(c) > 0 if c in imports_view else False:
                    return c
            # fallback: first candidate present in graph
            for c in candidates:
                if c in imports_view:
                    return c
            return name

        src = _resolve(src_file)
        dst = _resolve(dst_file)

        if src not in imports_view or dst not in imports_view:
            return None

        try:
            return nx.shortest_path(imports_view, src, dst)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    # ------------------------------------------------------------------
    # 6. Raises exception
    # ------------------------------------------------------------------

    def raises_exception(self, exception_name: str) -> list[FunctionNode]:
        """Return all functions that raise *exception_name*.

        Matches the destination of RAISES edges by exact name.
        """
        result: list[FunctionNode] = []
        for src, dst, data in self._g.nx_graph.edges(data=True):
            if data.get("edge_type") != "RAISES":
                continue
            if dst != exception_name:
                continue
            node = self._g.get_node(src)
            if isinstance(node, FunctionNode):
                result.append(node)
        return sorted(result, key=lambda fn: fn.id)

    # ------------------------------------------------------------------
    # 7. Callers / callees  (by name or full ID)
    # ------------------------------------------------------------------

    def callers(self, name_or_id: str) -> list[FunctionNode]:
        """Return all functions that call *name_or_id*.

        Accepts a bare name (resolved if unambiguous) or a full node ID.

        Handles two cases:
        1. Direct graph predecessors via CALLS edges (fully qualified dst_id).
        2. Unresolved call edges whose dst_id is the bare function name
           (e.g. ``self.db.add_episode(...)`` → dst_id ``"add_episode"``).
        """
        node_id = self._resolve_id(name_or_id)
        node = self._g.get_node(node_id)
        bare_name = node.name if isinstance(node, FunctionNode) else name_or_id

        # Case 1: edges pointing to the full qualified node ID
        direct = {
            n.id: n
            for n in self._g.predecessors(node_id, edge_type="CALLS")
            if isinstance(n, FunctionNode)
        }

        # Case 2: edges pointing to the bare function name as dst_id
        # (unresolved method calls like self.obj.method())
        via_bare: dict[str, FunctionNode] = {}
        if bare_name != node_id and bare_name in self._g.nx_graph:
            for n in self._g.predecessors(bare_name, edge_type="CALLS"):
                if isinstance(n, FunctionNode) and n.id not in direct:
                    via_bare[n.id] = n

        # Also scan all CALLS edges for bare-name matches not yet in graph as a node
        for src, dst, data in self._g.nx_graph.in_edges(bare_name, data=True):
            if data.get("edge_type") == "CALLS":
                caller = self._g.get_node(src)
                if isinstance(caller, FunctionNode) and caller.id not in direct:
                    via_bare[caller.id] = caller

        merged = {**direct, **via_bare}
        return sorted(merged.values(), key=lambda f: f.id)

    def callees(self, name_or_id: str) -> list[FunctionNode]:
        """Return all functions called by *name_or_id*.

        Accepts a bare name (resolved if unambiguous) or a full node ID.
        """
        node_id = self._resolve_id(name_or_id)
        return [
            n for n in self._g.successors(node_id, edge_type="CALLS")
            if isinstance(n, FunctionNode)
        ]

    # ------------------------------------------------------------------
    # 8. File fan-in  (most-imported files)
    # ------------------------------------------------------------------

    def file_fan_in(self, *, top_k: int = 10) -> list[tuple[FileNode, int]]:
        """Return the *top_k* files imported by the most other files (desc).

        Counts both direct FileNode in-edges (absolute imports) and
        local ModuleNode in-edges (relative imports like ``from .database import …``
        which produce a ``ModuleNode("database")`` rather than a ``FileNode``).
        The two counts are merged by matching each FileNode's stem name against
        the module node ID.
        """
        from pathlib import Path as _Path

        # Build stem → FileNode mapping for local module resolution
        stem_to_file: dict[str, FileNode] = {}
        for nid, node in self._g._nodes.items():
            if isinstance(node, FileNode):
                stem = _Path(node.path).stem  # e.g. "database" from "database.py"
                stem_to_file[stem] = node

        counts: dict[str, int] = {}

        # Direct FileNode in-edges
        for nid, node in self._g._nodes.items():
            if not isinstance(node, FileNode):
                continue
            count = sum(
                1
                for _, _, data in self._g.nx_graph.in_edges(nid, data=True)
                if data.get("edge_type") == "IMPORTS"
            )
            counts[nid] = counts.get(nid, 0) + count

        # Local ModuleNode in-edges — map back to FileNode by stem
        for nid, node in self._g._nodes.items():
            if not isinstance(node, ModuleNode) or not node.is_local:
                continue
            count = sum(
                1
                for _, _, data in self._g.nx_graph.in_edges(nid, data=True)
                if data.get("edge_type") == "IMPORTS"
            )
            if count and node.name in stem_to_file:
                file_nid = stem_to_file[node.name].id
                counts[file_nid] = counts.get(file_nid, 0) + count

        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            (self._g.get_node(nid), cnt)  # type: ignore[misc]
            for nid, cnt in ranked
            if self._g.get_node(nid) is not None
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_id(self, name_or_id: str) -> str:
        """Resolve a bare function name to a full node ID if unambiguous.

        If *name_or_id* already exists as a node ID it is returned as-is.
        Otherwise all FunctionNodes whose ``name`` matches are collected;
        if exactly one match exists that ID is returned, otherwise the
        original string is returned unchanged (traversal will just find
        nothing, which is safe).
        """
        if self._g.has_node(name_or_id):
            return name_or_id

        matches = [
            nid
            for nid, node in self._g._nodes.items()
            if isinstance(node, FunctionNode) and node.name == name_or_id
        ]
        if len(matches) == 1:
            return matches[0]
        return name_or_id
