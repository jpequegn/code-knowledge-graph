"""Filesystem watcher for automatic incremental graph updates.

Monitors a repository for Python file changes and keeps the DuckDB graph
cache up-to-date automatically.

Usage::

    from ckg.watcher import GraphWatcher
    from ckg.store import GraphStore

    store = GraphStore()
    store.build_and_save("my_project/")

    watcher = GraphWatcher("my_project/", store=store)
    watcher.start()   # blocks until Ctrl-C

With semantic re-embed::

    from ckg.embedder import NodeEmbedder
    embedder = NodeEmbedder(store)
    watcher = GraphWatcher("my_project/", store=store, embedder=embedder)
    watcher.start()
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# Only react to Python source files
_PYTHON_SUFFIX = ".py"
# Directories to ignore (mirrors the parser skip list)
_SKIP_DIRS: frozenset[str] = frozenset(
    {".venv", "__pycache__", ".git", "dist", "build", ".mypy_cache", ".ruff_cache"}
)


def _is_tracked(path: str) -> bool:
    """Return True if *path* is a Python file outside skip directories."""
    p = Path(path)
    if p.suffix != _PYTHON_SUFFIX:
        return False
    for part in p.parts:
        if part in _SKIP_DIRS or part.endswith(".egg-info"):
            return False
    return True


def _rel(path: str, root: Path) -> str:
    """Return path relative to *root*, or the original string if not under root."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


# ---------------------------------------------------------------------------
# Debounced event handler
# ---------------------------------------------------------------------------

class _PythonFileHandler(FileSystemEventHandler):
    """Watchdog event handler that debounces filesystem events.

    Collects changed / deleted / moved paths and fires *on_change* after
    *debounce_seconds* of quiet time.

    Parameters
    ----------
    root:
        Absolute repository root path.
    on_change:
        Callback invoked with ``(modified: set[str], deleted: set[str])``
        where each set contains *relative* paths.
    debounce_seconds:
        How long to wait (in seconds) after the last event before firing.
    """

    def __init__(
        self,
        root: Path,
        on_change: Callable[[set[str], set[str]], None],
        debounce_seconds: float = 0.5,
    ) -> None:
        super().__init__()
        self._root = root
        self._on_change = on_change
        self._debounce = debounce_seconds
        self._modified: set[str] = set()
        self._deleted: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    # -- watchdog event callbacks -------------------------------------------

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_tracked(event.src_path):
            self._queue_modified(_rel(event.src_path, self._root))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_tracked(event.src_path):
            self._queue_modified(_rel(event.src_path, self._root))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_tracked(event.src_path):
            self._queue_deleted(_rel(event.src_path, self._root))

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            if _is_tracked(event.src_path):
                self._queue_deleted(_rel(event.src_path, self._root))
            if _is_tracked(event.dest_path):
                self._queue_modified(_rel(event.dest_path, self._root))

    # -- internal -----------------------------------------------------------

    def _queue_modified(self, rel_path: str) -> None:
        with self._lock:
            self._modified.add(rel_path)
            self._deleted.discard(rel_path)
            self._reschedule()

    def _queue_deleted(self, rel_path: str) -> None:
        with self._lock:
            self._deleted.add(rel_path)
            self._modified.discard(rel_path)
            self._reschedule()

    def _reschedule(self) -> None:
        """Reset the debounce timer (must be called with self._lock held)."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            modified = set(self._modified)
            deleted = set(self._deleted)
            self._modified.clear()
            self._deleted.clear()
            self._timer = None
        if modified or deleted:
            self._on_change(modified, deleted)


# ---------------------------------------------------------------------------
# GraphWatcher
# ---------------------------------------------------------------------------

class GraphWatcher:
    """Watch a repository and keep the graph cache up-to-date.

    Parameters
    ----------
    repo:
        Path to the repository root to monitor.
    store:
        A :class:`~ckg.store.GraphStore` instance (already initialised).
    embedder:
        Optional :class:`~ckg.embedder.NodeEmbedder`.  When provided,
        changed nodes are re-embedded after each rebuild.
    debounce_seconds:
        Quiet-time window before triggering a rebuild (default 0.5 s).
    on_rebuild:
        Optional callback invoked after each successful rebuild with
        ``(reparsed: list[str], deleted: list[str])``.  Useful for tests
        and custom notification hooks.
    """

    def __init__(
        self,
        repo: str | Path,
        *,
        store,  # GraphStore  (avoid circular import at module level)
        embedder=None,  # NodeEmbedder | None
        debounce_seconds: float = 0.5,
        on_rebuild: Callable[[list[str], list[str]], None] | None = None,
    ) -> None:
        self._root = Path(repo).resolve()
        self._store = store
        self._embedder = embedder
        self._debounce = debounce_seconds
        self._on_rebuild = on_rebuild
        self._observer: Observer | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start watching the repository.  Blocks until :meth:`stop` is called
        or a ``KeyboardInterrupt`` is raised (Ctrl-C)."""
        handler = _PythonFileHandler(
            root=self._root,
            on_change=self._handle_change,
            debounce_seconds=self._debounce,
        )
        self._observer = Observer()
        self._observer.schedule(handler, str(self._root), recursive=True)
        self._observer.start()
        logger.info("Watching %s", self._root)
        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the watcher gracefully."""
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    @property
    def is_running(self) -> bool:
        """True while the watcher is active."""
        return self._observer is not None and self._observer.is_alive()

    # ------------------------------------------------------------------
    # Internal: handle debounced change batch
    # ------------------------------------------------------------------

    def _handle_change(self, modified: set[str], deleted: set[str]) -> None:
        """Process a batch of filesystem changes."""
        reparsed: list[str] = []
        invalidated: list[str] = []

        # 1. Invalidate deleted files
        for rel_path in sorted(deleted):
            logger.info("Deleted %s — invalidating", rel_path)
            self._store.invalidate_file(rel_path)
            if self._embedder is not None:
                self._embedder.invalidate_file(rel_path)
            invalidated.append(rel_path)

        # 2. Invalidate + re-parse modified / created files
        if modified:
            try:
                graph, reparsed = self._store.rebuild_incremental(self._root)
                logger.info(
                    "Rebuilt: %d reparsed, %d deleted",
                    len(reparsed), len(invalidated),
                )
                # Re-embed only newly parsed nodes
                if self._embedder is not None and reparsed:
                    self._embedder.embed_all(graph, force=False)
            except Exception:
                logger.exception("Incremental rebuild failed")

        if self._on_rebuild is not None:
            self._on_rebuild(reparsed, invalidated)
