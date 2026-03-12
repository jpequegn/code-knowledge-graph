"""DuckDB persistence layer for the Code Knowledge Graph.

Serialises a :class:`~ckg.graph.PropertyGraph` into two DuckDB tables
(``nodes`` and ``edges``) plus a ``file_mtimes`` table that drives
incremental rebuilds.

Usage
-----
    from ckg.store import GraphStore

    store = GraphStore()               # default: ~/.ckg/graph.db
    store.build_and_save("my_repo/")   # parse + persist

    # Later session — instant load, no re-parsing
    graph = store.load()

    # After editing a file — only re-parse what changed
    store.rebuild_incremental("my_repo/")
    graph = store.load()
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ckg.models import (
    ClassNode,
    FileNode,
    FunctionNode,
    ModuleNode,
    Node,
    ParamInfo,
)

if TYPE_CHECKING:
    from ckg.graph import PropertyGraph


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    node_type   TEXT    NOT NULL,
    name        TEXT,
    file_path   TEXT,
    line_start  INTEGER,
    properties  JSON    NOT NULL DEFAULT '{}',
    parsed_at   TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY,
    src_id      TEXT    NOT NULL,
    dst_id      TEXT    NOT NULL,
    edge_type   TEXT    NOT NULL,
    weight      INTEGER NOT NULL DEFAULT 1,
    line        INTEGER,
    properties  JSON    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS file_mtimes (
    path        TEXT PRIMARY KEY,
    mtime       DOUBLE  NOT NULL,
    parsed_at   TIMESTAMP NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _node_to_row(node: Node, now: datetime) -> tuple:
    """Convert a typed node dataclass to a ``(id, node_type, name, file_path,
    line_start, properties_json, parsed_at)`` tuple."""
    d = asdict(node)
    nid       = d.pop("id")
    node_type = d.pop("node_type")
    name      = d.pop("name", None)
    file_path = d.pop("file_path", None)
    line_start = d.pop("line_start", None)
    # everything else goes into the JSON blob
    props = json.dumps(d)
    return (nid, node_type, name, file_path, line_start, props, now)


def _row_to_node(row: tuple) -> Node:
    """Reconstruct a typed node from a DB row."""
    nid, node_type, name, file_path, line_start, props_json, _ = row
    props: dict = json.loads(props_json) if props_json else {}

    if node_type == "file":
        return FileNode(
            id=nid,
            path=props.get("path", nid),
            line_count=props.get("line_count", 0),
            avg_complexity=props.get("avg_complexity", 0.0),
        )
    if node_type == "function":
        raw_params = props.get("params", [])
        params = [
            ParamInfo(
                name=p["name"],
                annotation=p.get("annotation"),
                default=p.get("default"),
            )
            for p in raw_params
        ]
        return FunctionNode(
            id=nid,
            name=name or "",
            file_path=file_path or "",
            line_start=line_start or 0,
            line_end=props.get("line_end", line_start or 0),
            signature=props.get("signature", ""),
            docstring=props.get("docstring"),
            return_type=props.get("return_type"),
            cyclomatic_complexity=props.get("cyclomatic_complexity", 1),
            is_async=props.get("is_async", False),
            is_method=props.get("is_method", False),
            class_name=props.get("class_name"),
            param_count=props.get("param_count", 0),
            params=params,
        )
    if node_type == "class":
        return ClassNode(
            id=nid,
            name=name or "",
            file_path=file_path or "",
            line_start=line_start or 0,
            line_end=props.get("line_end", line_start or 0),
            bases=props.get("bases", []),
            docstring=props.get("docstring"),
            method_count=props.get("method_count", 0),
        )
    if node_type == "module":
        return ModuleNode(
            id=nid,
            name=name or nid,
            is_stdlib=props.get("is_stdlib", False),
            is_local=props.get("is_local", False),
        )
    raise ValueError(f"Unknown node_type {node_type!r} for id={nid!r}")


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """Persist and load a :class:`~ckg.graph.PropertyGraph` via DuckDB.

    Parameters
    ----------
    db_path:
        Path to the DuckDB file.  Defaults to ``~/.ckg/graph.db``.
        The parent directory is created automatically.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".ckg" / "graph.db"
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal connection management
    # ------------------------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(str(self._db_path))
        # DuckDB doesn't have executescript — run each statement individually
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        return conn

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, graph: "PropertyGraph", project_root: str | Path | None = None) -> None:
        """Serialise *graph* to DuckDB, replacing any previous data.

        Parameters
        ----------
        graph:
            The populated :class:`~ckg.graph.PropertyGraph` to persist.
        project_root:
            When provided, mtime records are written for every
            ``FileNode`` whose path can be resolved under *project_root*.
        """
        now = datetime.now(timezone.utc)
        conn = self._connect()

        try:
            conn.execute("BEGIN")

            # Wipe existing data
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")
            conn.execute("DELETE FROM file_mtimes")

            # Insert nodes
            node_rows = [_node_to_row(n, now) for n in graph._nodes.values()]
            if node_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
                    node_rows,
                )

            # Insert edges
            edge_rows = []
            for i, (src, dst, data) in enumerate(graph.nx_graph.edges(data=True)):
                edge_rows.append((
                    i,
                    src,
                    dst,
                    data.get("edge_type", "UNKNOWN"),
                    data.get("weight", 1),
                    data.get("line"),
                    json.dumps({
                        k: v for k, v in data.items()
                        if k not in ("edge_type", "weight", "line")
                    }),
                ))
            if edge_rows:
                conn.executemany(
                    "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?)",
                    edge_rows,
                )

            # Write file mtimes
            if project_root is not None:
                root = Path(project_root).resolve()
                mtime_rows = []
                for node in graph._nodes.values():
                    if not isinstance(node, FileNode):
                        continue
                    abs_path = root / node.path
                    if abs_path.exists():
                        mtime_rows.append((node.path, abs_path.stat().st_mtime, now))
                if mtime_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO file_mtimes VALUES (?, ?, ?)",
                        mtime_rows,
                    )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> "PropertyGraph":
        """Reconstruct a :class:`~ckg.graph.PropertyGraph` from DuckDB.

        Returns an empty graph if the database contains no data.
        """
        from ckg.graph import PropertyGraph

        conn = self._connect()
        try:
            graph = PropertyGraph()

            # Restore nodes
            rows = conn.execute(
                "SELECT id, node_type, name, file_path, line_start, properties, parsed_at "
                "FROM nodes"
            ).fetchall()
            for row in rows:
                node = _row_to_node(row)
                graph.add_node(node)

            # Restore edges
            edge_rows = conn.execute(
                "SELECT src_id, dst_id, edge_type, weight, line FROM edges"
            ).fetchall()
            for src, dst, edge_type, weight, line in edge_rows:
                graph.add_edge(src, dst, edge_type, weight=weight, line=line)  # type: ignore[arg-type]

            return graph
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Mtime helpers
    # ------------------------------------------------------------------

    def needs_reparse(self, file_path: str, project_root: str | Path) -> bool:
        """Return ``True`` if *file_path* is new or has been modified since
        the last ``save()`` call.

        Parameters
        ----------
        file_path:
            Relative path as stored in the graph (e.g. ``database.py``).
        project_root:
            Absolute path to the repository root.
        """
        abs_path = Path(project_root).resolve() / file_path
        if not abs_path.exists():
            return False  # file deleted — handled by invalidate_file

        current_mtime = abs_path.stat().st_mtime

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT mtime FROM file_mtimes WHERE path = ?", (file_path,)
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return True  # never seen before
        return current_mtime > row[0]

    def stored_files(self) -> list[str]:
        """Return all file paths currently tracked in the mtime table."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT path FROM file_mtimes").fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_file(self, file_path: str) -> None:
        """Remove all nodes and edges that originated from *file_path*,
        and delete its mtime record so it is re-parsed next time.

        Node IDs that start with ``file_path + "::"`` are removed, along
        with the ``FileNode`` whose id equals *file_path*.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            prefix = file_path + "::"

            # Nodes: file node itself + all function/class nodes in the file
            conn.execute(
                "DELETE FROM nodes WHERE id = ? OR id LIKE ?",
                (file_path, prefix + "%"),
            )
            # Edges: any edge whose src or dst originated in this file
            conn.execute(
                "DELETE FROM edges WHERE src_id = ? OR src_id LIKE ? "
                "OR dst_id = ? OR dst_id LIKE ?",
                (file_path, prefix + "%", file_path, prefix + "%"),
            )
            # Mtime record
            conn.execute("DELETE FROM file_mtimes WHERE path = ?", (file_path,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Incremental rebuild
    # ------------------------------------------------------------------

    def rebuild_incremental(
        self,
        root: str | Path,
        *,
        verbose: bool = False,
    ) -> tuple["PropertyGraph", list[str]]:
        """Re-parse only files that have changed since the last save.

        Algorithm
        ---------
        1. Load the existing graph from DuckDB.
        2. Walk all ``.py`` files under *root*.
        3. For each file that ``needs_reparse()`` returns ``True``:
           a. Invalidate its nodes/edges in the DB.
           b. Parse the file and ingest the result into *graph*.
        4. Persist the updated graph and return it together with the
           list of re-parsed relative paths.

        Parameters
        ----------
        root:
            Repository root to scan.
        verbose:
            If ``True`` print each re-parsed file to stdout.

        Returns
        -------
        (graph, reparsed_paths)
            The updated :class:`~ckg.graph.PropertyGraph` and the list
            of relative file paths that were re-parsed.
        """
        from ckg.parsers.python import parse_directory, parse_file

        root = Path(root).resolve()
        _SKIP_DIRS = {".venv", "__pycache__", ".git", "dist", "build",
                      ".mypy_cache", ".ruff_cache"}

        # 1. Load current state
        graph = self.load()

        reparsed: list[str] = []

        # 2. Walk .py files
        for py_file in sorted(root.rglob("*.py")):
            if any(part in _SKIP_DIRS or part.endswith(".egg-info")
                   for part in py_file.parts):
                continue
            rel = str(py_file.relative_to(root))

            # 3. Check mtime
            if not self.needs_reparse(rel, root):
                continue

            if verbose:
                print(f"  re-parsing {rel}")

            # 3a. Remove stale data from in-memory graph too
            _invalidate_in_graph(graph, rel)

            # 3b. Parse and ingest
            try:
                from ckg.parsers.python import parse_file as _pf
                result = _pf(py_file, root)
                graph._ingest_parse_result(result)
                reparsed.append(rel)
            except SyntaxError:
                if verbose:
                    print(f"    syntax error — skipped")

        # 4. Persist updated graph
        self.save(graph, project_root=root)

        return graph, reparsed

    # ------------------------------------------------------------------
    # Convenience: build + save in one call
    # ------------------------------------------------------------------

    def build_and_save(self, root: str | Path) -> "PropertyGraph":
        """Full build: parse *root*, save to DB, return the graph."""
        from ckg.graph import PropertyGraph

        graph = PropertyGraph()
        graph.build_from_directory(root)
        self.save(graph, project_root=root)
        return graph

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def db_stats(self) -> dict[str, int]:
        """Return ``{nodes, edges, tracked_files}`` counts from the DB."""
        conn = self._connect()
        try:
            n = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            e = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            f = conn.execute("SELECT COUNT(*) FROM file_mtimes").fetchone()[0]
            return {"nodes": n, "edges": e, "tracked_files": f}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"GraphStore({self._db_path})"


# ---------------------------------------------------------------------------
# Internal helper — remove a file's contribution from an in-memory graph
# ---------------------------------------------------------------------------

def _invalidate_in_graph(graph: "PropertyGraph", file_path: str) -> None:
    """Remove all nodes and edges originating from *file_path* from the
    in-memory *graph* (mirrors :meth:`GraphStore.invalidate_file` for DB)."""
    prefix = file_path + "::"

    to_remove = [
        nid for nid in graph._nodes
        if nid == file_path or nid.startswith(prefix)
    ]
    for nid in to_remove:
        del graph._nodes[nid]
        if nid in graph._graph:
            graph._graph.remove_node(nid)

    # Also remove edges where src or dst matches
    edges_to_remove = [
        (src, dst, key)
        for src, dst, key, data in graph._graph.edges(data=True, keys=True)
        if (src == file_path or src.startswith(prefix)
            or dst == file_path or dst.startswith(prefix))
    ]
    for src, dst, key in edges_to_remove:
        if graph._graph.has_edge(src, dst, key):
            graph._graph.remove_edge(src, dst, key)
