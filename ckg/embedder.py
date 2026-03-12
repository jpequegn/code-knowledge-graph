"""Semantic embedding layer for the Code Knowledge Graph.

Embeds FunctionNode and ClassNode descriptions (name + signature + docstring)
using a local sentence-transformers model and stores the vectors in DuckDB
alongside the graph.  Enables both pure semantic search and hybrid
structural + semantic queries.

Usage::

    from ckg.store import GraphStore
    from ckg.embedder import NodeEmbedder

    store = GraphStore()
    graph = store.load()

    embedder = NodeEmbedder(store)
    n_embedded = embedder.embed_all(graph)

    # Pure semantic
    results = embedder.search("audio transcription", top_k=5)
    for node, score in results:
        print(f"{score:.3f}  {node.id}")

    # Hybrid: semantic rank + structural filter
    from ckg.queries import GraphQueries
    q = GraphQueries(graph)
    callers = {n.id for n in q.callers("database.py::add_episode")}
    results = embedder.search(
        "episode storage", top_k=10,
        node_filter=lambda n: n.id in callers,
    )
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Sequence

import numpy as np

from ckg.graph import PropertyGraph
from ckg.models import ClassNode, FunctionNode, Node
from ckg.store import GraphStore

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_EMBED_DDL = """
CREATE TABLE IF NOT EXISTS embeddings (
    node_id   TEXT PRIMARY KEY,
    embedding TEXT NOT NULL,    -- JSON-serialised float list
    text_used TEXT NOT NULL
)
"""


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def _node_text(node: Node) -> str | None:
    """Return the text to embed for *node*, or None if the node should be skipped."""
    if isinstance(node, FunctionNode):
        parts = [node.name]
        if node.signature:
            parts.append(node.signature)
        if node.docstring:
            parts.append(node.docstring)
        return "\n".join(parts)
    if isinstance(node, ClassNode):
        parts = [node.name]
        if node.docstring:
            parts.append(node.docstring)
        if node.bases:
            parts.append("inherits: " + ", ".join(node.bases))
        return "\n".join(parts)
    return None  # FileNode / ModuleNode — not embedded


# ---------------------------------------------------------------------------
# NodeEmbedder
# ---------------------------------------------------------------------------

class NodeEmbedder:
    """Embed graph nodes and search by cosine similarity.

    Parameters
    ----------
    store:
        A :class:`~ckg.store.GraphStore` instance.  The embeddings table
        is created inside the same DuckDB file as the graph.
    model_name:
        Sentence-transformers model identifier.  Defaults to
        ``all-MiniLM-L6-v2`` (384-dim, ~22 MB, CPU-friendly).
    """

    def __init__(
        self,
        store: GraphStore,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._store = store
        self._model_name = model_name
        self._model = None  # lazy-load
        self._ensure_table()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        conn = self._store._connect()
        conn.execute(_EMBED_DDL)
        conn.close()

    # ------------------------------------------------------------------
    # Model (lazy)
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed_all(
        self,
        graph: PropertyGraph,
        force: bool = False,
        batch_size: int = 64,
    ) -> int:
        """Embed all eligible nodes and persist vectors to DuckDB.

        Parameters
        ----------
        graph:
            The graph whose nodes to embed.
        force:
            If True, re-embed every node even if already stored.
        batch_size:
            Number of texts to send to the model at once.

        Returns
        -------
        int
            Number of nodes newly embedded.
        """
        already_stored: set[str] = set()
        if not force:
            already_stored = self._stored_ids()

        # Collect (node, text) pairs to embed
        to_embed: list[tuple[Node, str]] = []
        for node in graph.iter_nodes():
            if node.id in already_stored:
                continue
            text = _node_text(node)
            if text:
                to_embed.append((node, text))

        if not to_embed:
            return 0

        model = self._get_model()
        texts = [t for _, t in to_embed]
        nodes_list = [n for n, _ in to_embed]

        # Encode in batches
        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            vectors.extend(vecs)

        # Persist
        conn = self._store._connect()
        rows = [
            (node.id, json.dumps(vec.tolist()), text)
            for node, text, vec in zip(nodes_list, texts, vectors)
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (node_id, embedding, text_used) VALUES (?, ?, ?)",
            rows,
        )
        conn.close()
        return len(rows)

    def embed_node(self, node: Node) -> bool:
        """Embed a single node and persist it.  Returns True if embedded."""
        text = _node_text(node)
        if text is None:
            return False
        model = self._get_model()
        vec: np.ndarray = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        conn = self._store._connect()
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, embedding, text_used) VALUES (?, ?, ?)",
            (node.id, json.dumps(vec.tolist()), text),
        )
        conn.close()
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        graph: PropertyGraph,
        top_k: int = 10,
        node_filter: Callable[[Node], bool] | None = None,
    ) -> list[tuple[Node, float]]:
        """Return nodes most similar to *query*, sorted by cosine similarity.

        Parameters
        ----------
        query:
            Free-text query string.
        graph:
            Graph used for node lookups.
        top_k:
            Maximum number of results.
        node_filter:
            Optional callable; only nodes for which it returns True are
            considered.  Use this to combine semantic ranking with
            structural constraints.

        Returns
        -------
        list of ``(node, score)`` sorted descending by score.
        """
        model = self._get_model()
        q_vec: np.ndarray = model.encode(query, normalize_embeddings=True, show_progress_bar=False)

        conn = self._store._connect()
        rows = conn.execute(
            "SELECT node_id, embedding FROM embeddings"
        ).fetchall()
        conn.close()

        if not rows:
            return []

        # Build matrix
        ids = [r[0] for r in rows]
        matrix = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)  # (N, D)

        # Cosine similarity (vectors are already normalised)
        scores: np.ndarray = matrix @ q_vec  # (N,)

        # Rank
        ranked = sorted(zip(ids, scores.tolist()), key=lambda x: -x[1])

        results: list[tuple[Node, float]] = []
        for node_id, score in ranked:
            if len(results) >= top_k:
                break
            node = graph.get_node(node_id)
            if node is None:
                continue
            if node_filter is not None and not node_filter(node):
                continue
            results.append((node, float(score)))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stored_ids(self) -> set[str]:
        conn = self._store._connect()
        rows = conn.execute("SELECT node_id FROM embeddings").fetchall()
        conn.close()
        return {r[0] for r in rows}

    def embed_count(self) -> int:
        """Return the number of nodes currently embedded in DuckDB."""
        conn = self._store._connect()
        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        conn.close()
        return count

    def invalidate_file(self, file_path: str) -> int:
        """Remove all embeddings for nodes belonging to *file_path*.

        Returns the number of rows deleted.
        """
        conn = self._store._connect()
        # Match node_id prefix (e.g. "service.py" or "service.py::")
        result = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE node_id = ? OR node_id LIKE ?",
            (file_path, f"{file_path}::%"),
        ).fetchone()[0]
        conn.execute(
            "DELETE FROM embeddings WHERE node_id = ? OR node_id LIKE ?",
            (file_path, f"{file_path}::%"),
        )
        conn.close()
        return result
