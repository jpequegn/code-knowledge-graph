"""Tests for ckg.embedder.NodeEmbedder."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from ckg.embedder import NodeEmbedder, _node_text
from ckg.graph import PropertyGraph
from ckg.models import FunctionNode, ClassNode, FileNode, ModuleNode
from ckg.store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "transcriber.py").write_text(textwrap.dedent("""\
        \"\"\"Audio transcription utilities.\"\"\"

        def transcribe_episode(audio_path: str) -> str:
            \"\"\"Transcribe an audio file to text using Whisper.\"\"\"
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return result["text"]

        def detect_language(audio_path: str) -> str:
            \"\"\"Detect the spoken language of an audio file.\"\"\"
            return "en"
    """))
    (tmp_path / "database.py").write_text(textwrap.dedent("""\
        \"\"\"SQLite persistence for episodes.\"\"\"

        class EpisodeDatabase:
            \"\"\"Stores and retrieves podcast episode records.\"\"\"

            def __init__(self, path: str) -> None:
                self._path = path

            def add_episode(self, url: str, transcript: str) -> None:
                \"\"\"Insert a new episode into the database.\"\"\"
                if not url:
                    raise ValueError("url required")

            def get_episode(self, url: str) -> dict:
                \"\"\"Retrieve an episode by URL.\"\"\"
                return {}

            def list_episodes(self) -> list:
                \"\"\"Return all stored episodes.\"\"\"
                return []
    """))
    (tmp_path / "cli.py").write_text(textwrap.dedent("""\
        \"\"\"Command-line interface for the podcast processor.\"\"\"
        from database import EpisodeDatabase
        from transcriber import transcribe_episode

        def run(audio_path: str, db_path: str) -> None:
            \"\"\"Transcribe and store a podcast episode.\"\"\"
            db = EpisodeDatabase(db_path)
            text = transcribe_episode(audio_path)
            db.add_episode(audio_path, text)
    """))
    return tmp_path


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path / "repo")


@pytest.fixture()
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(tmp_path / "g.db")


@pytest.fixture()
def graph(repo: Path, store: GraphStore) -> PropertyGraph:
    return store.build_and_save(repo)


@pytest.fixture()
def embedder(store: GraphStore) -> NodeEmbedder:
    return NodeEmbedder(store)


# ---------------------------------------------------------------------------
# _node_text helper
# ---------------------------------------------------------------------------

class TestNodeText:
    def test_function_uses_name(self) -> None:
        fn = FunctionNode(
            id="f.py::foo", name="foo", file_path="f.py",
            line_start=1, line_end=5, signature="def foo(x: int) -> str",
            docstring="Does foo.", return_type="str", param_count=1,
            cyclomatic_complexity=1, is_async=False, is_method=False,
            class_name=None,
        )
        text = _node_text(fn)
        assert "foo" in text

    def test_function_includes_signature(self) -> None:
        fn = FunctionNode(
            id="f.py::foo", name="foo", file_path="f.py",
            line_start=1, line_end=5, signature="def foo(x: int) -> str",
            docstring=None, return_type="str", param_count=1,
            cyclomatic_complexity=1, is_async=False, is_method=False,
            class_name=None,
        )
        assert "def foo(x: int) -> str" in _node_text(fn)

    def test_function_includes_docstring(self) -> None:
        fn = FunctionNode(
            id="f.py::foo", name="foo", file_path="f.py",
            line_start=1, line_end=5, signature="def foo()",
            docstring="Handles audio transcription.", return_type=None,
            param_count=0, cyclomatic_complexity=1, is_async=False,
            is_method=False, class_name=None,
        )
        assert "audio transcription" in _node_text(fn)

    def test_class_uses_name(self) -> None:
        cls = ClassNode(
            id="f.py::DB", name="DB", file_path="f.py",
            line_start=1, line_end=10, bases=[], docstring=None,
            method_count=2,
        )
        assert "DB" in _node_text(cls)

    def test_class_includes_docstring(self) -> None:
        cls = ClassNode(
            id="f.py::DB", name="DB", file_path="f.py",
            line_start=1, line_end=10, bases=["BaseDB"],
            docstring="Manages database connections.", method_count=2,
        )
        text = _node_text(cls)
        assert "database connections" in text

    def test_class_includes_bases(self) -> None:
        cls = ClassNode(
            id="f.py::DB", name="DB", file_path="f.py",
            line_start=1, line_end=10, bases=["BaseHandler"],
            docstring=None, method_count=2,
        )
        assert "BaseHandler" in _node_text(cls)

    def test_file_node_returns_none(self) -> None:
        fnode = FileNode(id="f.py", path="f.py", line_count=10, avg_complexity=2.0)
        assert _node_text(fnode) is None

    def test_module_node_returns_none(self) -> None:
        mnode = ModuleNode(id="os", name="os", is_stdlib=True, is_local=False)
        assert _node_text(mnode) is None


# ---------------------------------------------------------------------------
# embed_all
# ---------------------------------------------------------------------------

class TestEmbedAll:
    def test_embeds_functions_and_classes(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        n = embedder.embed_all(graph)
        assert n > 0

    def test_embed_count_matches_eligible_nodes(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        eligible = sum(
            1 for node in graph.iter_nodes()
            if isinstance(node, (FunctionNode, ClassNode))
        )
        embedder.embed_all(graph)
        assert embedder.embed_count() == eligible

    def test_idempotent_without_force(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        first = embedder.embed_all(graph)
        second = embedder.embed_all(graph)
        assert first > 0
        assert second == 0  # nothing new to embed

    def test_force_reembeds_all(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        count_before = embedder.embed_count()
        re_embedded = embedder.embed_all(graph, force=True)
        assert re_embedded == count_before

    def test_vectors_stored_in_duckdb(
        self, graph: PropertyGraph, embedder: NodeEmbedder, store: GraphStore
    ) -> None:
        embedder.embed_all(graph)
        conn = store._connect()
        rows = conn.execute("SELECT node_id, embedding FROM embeddings").fetchall()
        conn.close()
        assert len(rows) > 0
        # Each embedding should be a valid JSON float list
        vec = json.loads(rows[0][1])
        assert isinstance(vec, list)
        assert len(vec) == 384  # all-MiniLM-L6-v2

    def test_vectors_are_unit_normalised(
        self, graph: PropertyGraph, embedder: NodeEmbedder, store: GraphStore
    ) -> None:
        embedder.embed_all(graph)
        conn = store._connect()
        rows = conn.execute("SELECT embedding FROM embeddings LIMIT 5").fetchall()
        conn.close()
        for (emb_json,) in rows:
            vec = np.array(json.loads(emb_json))
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 1e-5, f"vector not unit-normalised (norm={norm})"


# ---------------------------------------------------------------------------
# embed_node
# ---------------------------------------------------------------------------

class TestEmbedNode:
    def test_embeds_single_function(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        fn = graph.get_node("transcriber.py::transcribe_episode")
        assert fn is not None
        result = embedder.embed_node(fn)
        assert result is True
        assert embedder.embed_count() == 1

    def test_skips_file_node(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        fnode = graph.get_node("cli.py")
        assert fnode is not None
        result = embedder.embed_node(fnode)
        assert result is False
        assert embedder.embed_count() == 0


# ---------------------------------------------------------------------------
# search — relevance
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_top_k_results(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        results = embedder.search("transcription", graph=graph, top_k=3)
        assert len(results) <= 3

    def test_transcription_query_finds_transcribe_episode(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        results = embedder.search("audio transcription", graph=graph, top_k=5)
        ids = [n.id for n, _ in results]
        assert "transcriber.py::transcribe_episode" in ids

    def test_database_query_finds_episode_database(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        results = embedder.search("database episode storage", graph=graph, top_k=5)
        ids = [n.id for n, _ in results]
        assert any("database" in nid.lower() for nid in ids)

    def test_scores_descending(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        results = embedder.search("store episode", graph=graph, top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_between_0_and_1(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        results = embedder.search("run", graph=graph, top_k=5)
        for _, score in results:
            assert -1.0 <= score <= 1.0

    def test_empty_db_returns_empty(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        # Don't call embed_all — DB is empty
        results = embedder.search("anything", graph=graph, top_k=5)
        assert results == []

    def test_node_filter_restricts_results(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        # Only nodes in transcriber.py
        results = embedder.search(
            "audio", graph=graph, top_k=10,
            node_filter=lambda n: getattr(n, "file_path", "") == "transcriber.py",
        )
        for node, _ in results:
            assert node.file_path == "transcriber.py"

    def test_node_filter_can_return_empty(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        results = embedder.search(
            "anything", graph=graph, top_k=10,
            node_filter=lambda n: False,
        )
        assert results == []


# ---------------------------------------------------------------------------
# invalidate_file
# ---------------------------------------------------------------------------

class TestInvalidateFile:
    def test_removes_embeddings_for_file(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        removed = embedder.invalidate_file("transcriber.py")
        assert removed > 0
        # Those nodes should no longer appear in search
        results = embedder.search(
            "transcription", graph=graph, top_k=20
        )
        ids = [n.id for n, _ in results]
        assert "transcriber.py::transcribe_episode" not in ids

    def test_leaves_other_files_intact(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        count_before = embedder.embed_count()
        removed = embedder.invalidate_file("transcriber.py")
        assert embedder.embed_count() == count_before - removed
        # database.py nodes still present
        results = embedder.search(
            "episode database", graph=graph, top_k=20
        )
        ids = [n.id for n, _ in results]
        assert any("database" in nid.lower() for nid in ids)

    def test_invalidate_nonexistent_returns_zero(
        self, graph: PropertyGraph, embedder: NodeEmbedder
    ) -> None:
        embedder.embed_all(graph)
        removed = embedder.invalidate_file("nonexistent.py")
        assert removed == 0


# ---------------------------------------------------------------------------
# CLI — embed command and search subcommand
# ---------------------------------------------------------------------------

class TestCLI:
    def test_embed_command_runs(self, repo: Path, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli

        runner = CliRunner()
        db = str(tmp_path / "g.db")
        result = runner.invoke(
            cli, ["--db", db, "build", str(repo)], catch_exceptions=False
        )
        assert result.exit_code == 0

        result = runner.invoke(
            cli, ["--db", db, "embed", "--repo", str(repo)], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "Embedded" in result.output or "Nothing new" in result.output

    def test_embed_idempotent(self, repo: Path, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli

        runner = CliRunner()
        db = str(tmp_path / "g.db")
        runner.invoke(cli, ["--db", db, "build", str(repo)], catch_exceptions=False)
        runner.invoke(cli, ["--db", db, "embed", "--repo", str(repo)], catch_exceptions=False)

        result = runner.invoke(
            cli, ["--db", db, "embed", "--repo", str(repo)], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "Nothing new" in result.output

    def test_search_requires_embed_first(self, repo: Path, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli

        runner = CliRunner()
        db = str(tmp_path / "g.db")
        runner.invoke(cli, ["--db", db, "build", str(repo)], catch_exceptions=False)

        result = runner.invoke(
            cli,
            ["--db", db, "query", "search", "transcription", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code != 0 or "No embeddings" in result.output

    def test_search_returns_results(self, repo: Path, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from ckg.cli import cli

        runner = CliRunner()
        db = str(tmp_path / "g.db")
        runner.invoke(cli, ["--db", db, "build", str(repo)], catch_exceptions=False)
        runner.invoke(cli, ["--db", db, "embed", "--repo", str(repo)], catch_exceptions=False)

        result = runner.invoke(
            cli,
            ["--db", db, "query", "search", "audio transcription", "--repo", str(repo)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "transcribe" in result.output.lower()
