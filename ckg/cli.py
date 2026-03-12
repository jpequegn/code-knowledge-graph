"""CLI entrypoint for the Code Knowledge Graph tool."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from ckg.embedder import NodeEmbedder
from ckg.graph import PropertyGraph
from ckg.models import FunctionNode, ClassNode, FileNode, ModuleNode
from ckg.queries import GraphQueries
from ckg.store import GraphStore

console = Console()

# Default DB location (can be overridden with --db)
_DEFAULT_DB = Path.home() / ".ckg" / "graph.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _complexity_style(cc: int) -> str:
    if cc < 5:
        return "green"
    if cc < 10:
        return "yellow"
    return "red"


def _complexity_text(cc: int) -> Text:
    style = _complexity_style(cc)
    return Text(str(cc), style=style)


def _print_graph_summary(g: PropertyGraph, *, prefix: str = "✓") -> None:
    by_type = g.node_count_by_type()
    by_edge = g.edge_count_by_type()
    console.print(
        f"[green]{prefix}[/green] Graph: "
        f"[bold]{g.node_count()}[/bold] nodes "
        f"([cyan]{by_type.get('function', 0)}[/cyan] functions, "
        f"[cyan]{by_type.get('class', 0)}[/cyan] classes, "
        f"[cyan]{by_type.get('file', 0)}[/cyan] files)  "
        f"[bold]{g.edge_count()}[/bold] edges "
        f"([cyan]{by_edge.get('CALLS', 0)}[/cyan] calls, "
        f"[cyan]{by_edge.get('IMPORTS', 0)}[/cyan] imports)"
    )


def _load_or_build_graph(repo: str, db_path: Path, *, force: bool = False) -> PropertyGraph:
    """Return a graph, loading from the DB cache when available and fresh.

    If *force* is True, always re-parse from source.
    Falls back to a full parse when no DB exists yet.
    """
    store = GraphStore(db_path)
    root = Path(repo).resolve()

    if not force and db_path.exists():
        # Check whether the cache is stale (any file needs re-parse)
        stale = any(
            store.needs_reparse(f, root)
            for f in store.stored_files()
        )
        # Also check for new files not yet in cache
        _SKIP = {".venv", "__pycache__", ".git", "dist", "build",
                 ".mypy_cache", ".ruff_cache"}
        known = set(store.stored_files())
        new_files = [
            str(p.relative_to(root))
            for p in sorted(root.rglob("*.py"))
            if not any(part in _SKIP or part.endswith(".egg-info") for part in p.parts)
            and str(p.relative_to(root)) not in known
        ]

        if not stale and not new_files:
            console.print(f"[dim]Loading graph from cache[/dim] [cyan]{db_path}[/cyan]…")
            g = store.load()
            _print_graph_summary(g, prefix="✓ Loaded")
            return g

        console.print(f"[dim]Cache stale — rebuilding[/dim] [cyan]{root}[/cyan]…")
        g, reparsed = store.rebuild_incremental(root)
        console.print(f"[green]✓[/green] Re-parsed [bold]{len(reparsed)}[/bold] file(s)")
        _print_graph_summary(g)
        return g

    # No cache — full parse
    console.print(f"[dim]Scanning[/dim] [cyan]{root}[/cyan]…")
    g = store.build_and_save(root)
    _print_graph_summary(g, prefix="✓ Built")
    return g


def _require_arg(args: tuple[str, ...], n: int, usage: str) -> list[str]:
    if len(args) < n:
        console.print(f"[red]Error:[/red] {usage}")
        sys.exit(1)
    return list(args[:n])


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="code-knowledge-graph")
@click.option(
    "--db",
    default=str(_DEFAULT_DB),
    show_default=True,
    envvar="CKG_DB",
    help="Path to the DuckDB graph cache.",
)
@click.pass_context
def cli(ctx: click.Context, db: str) -> None:
    """Code Knowledge Graph — structural analysis for Python codebases.

    Build a property graph from your Python source code and answer
    structural questions: impact radius, dead code, complexity hotspots,
    dependency paths, and more.

    The graph is cached in a DuckDB file (default: ~/.ckg/graph.db).
    Subsequent queries load from cache and only re-parse changed files.
    """
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = Path(db)


# ---------------------------------------------------------------------------
# ckg build
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("repo", default=".", type=click.Path(exists=True))
@click.option("--incremental", is_flag=True,
              help="Only re-parse files changed since last build.")
@click.option("--force", is_flag=True,
              help="Force full rebuild, ignoring the cache.")
@click.pass_context
def build(ctx: click.Context, repo: str, incremental: bool, force: bool) -> None:
    """Build (or update) the knowledge graph from a Python repository.

    REPO is the path to the root of the repository (default: current directory).
    The graph is persisted to the DuckDB cache for fast subsequent queries.
    """
    db_path: Path = ctx.obj["db_path"]
    root = Path(repo).resolve()
    store = GraphStore(db_path)

    if force:
        console.print(f"[dim]Force rebuild from[/dim] [cyan]{root}[/cyan]…")
        g = store.build_and_save(root)
        _print_graph_summary(g, prefix="✓ Built")
    elif incremental and db_path.exists():
        console.print(f"[dim]Incremental rebuild of[/dim] [cyan]{root}[/cyan]…")
        g, reparsed = store.rebuild_incremental(root)
        if reparsed:
            console.print(
                f"[green]✓[/green] Re-parsed [bold]{len(reparsed)}[/bold] file(s): "
                + ", ".join(reparsed[:5])
                + (" …" if len(reparsed) > 5 else "")
            )
        else:
            console.print("[green]✓[/green] Nothing changed — cache is up to date.")
        _print_graph_summary(g)
    else:
        if incremental:
            console.print("[yellow]No cache found — running full build.[/yellow]")
        console.print(f"[dim]Scanning[/dim] [cyan]{root}[/cyan]…")
        g = store.build_and_save(root)
        _print_graph_summary(g, prefix="✓ Built")

    stats = store.db_stats()
    console.print(
        f"[dim]Cache:[/dim] [cyan]{db_path}[/cyan] "
        f"({stats['nodes']} nodes, {stats['edges']} edges, "
        f"{stats['tracked_files']} tracked files)"
    )


# ---------------------------------------------------------------------------
# ckg query  (sub-dispatched by subcommand name)
# ---------------------------------------------------------------------------

@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("subcommand", type=click.Choice(
    ["impact", "callers", "callees", "hotspots", "dead-code", "path", "raises", "search"],
    case_sensitive=False,
))
@click.argument("args", nargs=-1)
@click.option("--repo", default=".", type=click.Path(exists=True),
              help="Repository root (default: current directory).")
@click.option("--depth", default=3, show_default=True,
              help="BFS depth for impact query.")
@click.option("--top", default=10, show_default=True,
              help="Number of results for hotspots / fan-in queries.")
@click.pass_context
def query(ctx: click.Context, subcommand: str, args: tuple[str, ...], repo: str, depth: int, top: int) -> None:
    """Run a structural query against the knowledge graph.

    \b
    Subcommands:
      impact  <node_id>           Impact radius of a function change
      callers <name_or_id>        All callers of a function
      callees <name_or_id>        All callees of a function
      hotspots                    Top-N complexity hotspots (--top N)
      dead-code                   Functions never called anywhere
      path    <file_a> <file_b>   Dependency path between two files
      raises  <ExceptionName>     Functions that raise an exception
      search  <query text>        Semantic similarity search over nodes
    """
    db_path: Path = ctx.obj["db_path"]
    g = _load_or_build_graph(repo, db_path)
    q = GraphQueries(g)

    sub = subcommand.lower()

    # ---- impact -----------------------------------------------------------
    if sub == "impact":
        [node_id] = _require_arg(args, 1, "Usage: ckg query impact <node_id>")
        result = q.impact_radius(node_id, depth=depth)
        if not result:
            console.print(f"[yellow]No callers found for[/yellow] [cyan]{node_id}[/cyan]")
            return
        console.print(Panel(
            f"Impact radius of [bold cyan]{node_id}[/bold cyan] (depth={depth})",
            expand=False,
        ))
        t = Table(box=box.SIMPLE_HEAD, show_header=True)
        t.add_column("Distance", style="bold", justify="center", width=10)
        t.add_column("Caller", style="cyan")
        t.add_column("File", style="dim")
        t.add_column("CC", justify="right", width=4)
        for d in sorted(result):
            for fn in result[d]:
                t.add_row(str(d), fn.id, fn.file_path, _complexity_text(fn.cyclomatic_complexity))
        console.print(t)

    # ---- callers ----------------------------------------------------------
    elif sub == "callers":
        [node_id] = _require_arg(args, 1, "Usage: ckg query callers <name_or_id>")
        callers = q.callers(node_id)
        if not callers:
            console.print(f"[yellow]No callers found for[/yellow] [cyan]{node_id}[/cyan]")
            return
        t = Table(title=f"Callers of {node_id}", box=box.SIMPLE_HEAD)
        t.add_column("Caller", style="cyan")
        t.add_column("File", style="dim")
        t.add_column("Line", justify="right", width=6)
        t.add_column("CC", justify="right", width=4)
        for fn in sorted(callers, key=lambda f: f.id):
            t.add_row(fn.id, fn.file_path, str(fn.line_start), _complexity_text(fn.cyclomatic_complexity))
        console.print(t)

    # ---- callees ----------------------------------------------------------
    elif sub == "callees":
        [node_id] = _require_arg(args, 1, "Usage: ckg query callees <name_or_id>")
        callees = q.callees(node_id)
        if not callees:
            console.print(f"[yellow]No callees found for[/yellow] [cyan]{node_id}[/cyan]")
            return
        t = Table(title=f"Callees of {node_id}", box=box.SIMPLE_HEAD)
        t.add_column("Callee", style="cyan")
        t.add_column("File", style="dim")
        t.add_column("Line", justify="right", width=6)
        t.add_column("CC", justify="right", width=4)
        for fn in sorted(callees, key=lambda f: f.id):
            t.add_row(fn.id, fn.file_path, str(fn.line_start), _complexity_text(fn.cyclomatic_complexity))
        console.print(t)

    # ---- hotspots ---------------------------------------------------------
    elif sub == "hotspots":
        hotspots = q.complexity_hotspots(top_k=top)
        if not hotspots:
            console.print("[yellow]No functions found.[/yellow]")
            return
        t = Table(title=f"Top-{top} Complexity Hotspots", box=box.SIMPLE_HEAD)
        t.add_column("#", justify="right", width=4, style="dim")
        t.add_column("Function", style="cyan")
        t.add_column("File", style="dim")
        t.add_column("Lines", justify="right", width=8)
        t.add_column("CC", justify="right", width=4)
        for i, (fn, cc) in enumerate(hotspots, 1):
            lines = f"{fn.line_start}–{fn.line_end}"
            t.add_row(str(i), fn.id, fn.file_path, lines, _complexity_text(cc))
        console.print(t)
        console.print(
            "[dim]CC: [green]<5[/green] simple  "
            "[yellow]5–9[/yellow] moderate  "
            "[red]≥10[/red] complex[/dim]"
        )

    # ---- dead-code --------------------------------------------------------
    elif sub == "dead-code":
        uncalled = q.uncalled_functions()
        if not uncalled:
            console.print("[green]No uncalled functions found.[/green]")
            return
        t = Table(title="Uncalled Functions (dead code candidates)", box=box.SIMPLE_HEAD)
        t.add_column("Function", style="cyan")
        t.add_column("File", style="dim")
        t.add_column("Line", justify="right", width=6)
        t.add_column("CC", justify="right", width=4)
        for fn in uncalled:
            t.add_row(fn.id, fn.file_path, str(fn.line_start), _complexity_text(fn.cyclomatic_complexity))
        console.print(t)

    # ---- path -------------------------------------------------------------
    elif sub == "path":
        [src, dst] = _require_arg(args, 2, "Usage: ckg query path <file_a> <file_b>")
        path = q.dependency_path(src, dst)
        if path is None:
            console.print(
                f"[yellow]No import path found from[/yellow] [cyan]{src}[/cyan] "
                f"[yellow]to[/yellow] [cyan]{dst}[/cyan]"
            )
            return
        arrow = Text(" → ", style="dim")
        result = Text()
        for i, node in enumerate(path):
            if i:
                result.append(" → ", style="dim")
            result.append(node, style="cyan")
        console.print(Panel(result, title=f"Dependency path: {src} → {dst}", expand=False))

    # ---- raises -----------------------------------------------------------
    elif sub == "raises":
        [exc] = _require_arg(args, 1, "Usage: ckg query raises <ExceptionName>")
        raisers = q.raises_exception(exc)
        if not raisers:
            console.print(f"[yellow]No functions raise[/yellow] [cyan]{exc}[/cyan]")
            return
        t = Table(title=f"Functions that raise {exc}", box=box.SIMPLE_HEAD)
        t.add_column("Function", style="cyan")
        t.add_column("File", style="dim")
        t.add_column("Line", justify="right", width=6)
        for fn in raisers:
            t.add_row(fn.id, fn.file_path, str(fn.line_start))
        console.print(t)

    # ---- search -----------------------------------------------------------
    elif sub == "search":
        if not args:
            console.print("[red]Error:[/red] Usage: ckg query search <query text>")
            sys.exit(1)
        query_text = " ".join(args)
        embedder = NodeEmbedder(GraphStore(db_path))
        if embedder.embed_count() == 0:
            console.print(
                "[yellow]No embeddings found.[/yellow] "
                "Run [cyan]ckg embed --repo .[/cyan] first."
            )
            sys.exit(1)
        results = embedder.search(query_text, graph=g, top_k=top)
        if not results:
            console.print(f"[yellow]No results for[/yellow] [cyan]{query_text}[/cyan]")
            return
        t = Table(title=f'Semantic search: "{query_text}"', box=box.SIMPLE_HEAD)
        t.add_column("Score", justify="right", width=7, style="bold")
        t.add_column("Node", style="cyan")
        t.add_column("File", style="dim")
        t.add_column("Type", style="dim", width=10)
        for node, score in results:
            t.add_row(
                f"{score:.3f}",
                node.id,
                getattr(node, "file_path", ""),
                node.node_type,
            )
        console.print(t)


# ---------------------------------------------------------------------------
# ckg inspect
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("kind", type=click.Choice(["node", "file"], case_sensitive=False))
@click.argument("target")
@click.option("--repo", default=".", type=click.Path(exists=True),
              help="Repository root (default: current directory).")
@click.pass_context
def inspect(ctx: click.Context, kind: str, target: str, repo: str) -> None:
    """Inspect a node or file in the knowledge graph.

    \b
    Kinds:
      node <node_id>   Full details for a single function or class node
      file <path>      All nodes defined in a file
    """
    db_path: Path = ctx.obj["db_path"]
    g = _load_or_build_graph(repo, db_path)
    q = GraphQueries(g)

    if kind == "node":
        node = g.get_node(target)
        if node is None:
            # try bare-name resolution
            resolved = q._resolve_id(target)
            node = g.get_node(resolved)
        if node is None:
            console.print(f"[red]Node not found:[/red] [cyan]{target}[/cyan]")
            sys.exit(1)

        if isinstance(node, FunctionNode):
            _inspect_function(node, g, q)
        elif isinstance(node, ClassNode):
            _inspect_class(node, g)
        elif isinstance(node, FileNode):
            _inspect_file_node(node, g)
        else:
            console.print(node)

    elif kind == "file":
        # find all nodes in this file
        fnode = g.get_node(target)
        if fnode is None:
            console.print(f"[red]File not found in graph:[/red] [cyan]{target}[/cyan]")
            sys.exit(1)
        _inspect_file_node(fnode, g)


def _inspect_function(fn: FunctionNode, g: PropertyGraph, q: GraphQueries) -> None:
    console.print(Panel(
        f"[bold cyan]{fn.id}[/bold cyan]",
        title="Function",
        expand=False,
    ))

    # Properties table
    props = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    props.add_column("Key", style="dim", width=20)
    props.add_column("Value")
    props.add_row("Signature", Text(fn.signature, style="cyan"))
    props.add_row("File", fn.file_path)
    props.add_row("Lines", f"{fn.line_start}–{fn.line_end}")
    props.add_row("Async", "yes" if fn.is_async else "no")
    props.add_row("Method", f"yes ({fn.class_name})" if fn.is_method else "no")
    props.add_row("Params", str(fn.param_count))
    props.add_row("Return type", fn.return_type or "—")
    props.add_row("Complexity", _complexity_text(fn.cyclomatic_complexity))
    props.add_row("Docstring", fn.docstring or "—")
    console.print(props)

    # Callers
    callers = q.callers(fn.id)
    if callers:
        t = Table(title="Callers", box=box.SIMPLE_HEAD)
        t.add_column("Caller", style="cyan")
        t.add_column("File", style="dim")
        for c in sorted(callers, key=lambda f: f.id):
            t.add_row(c.id, c.file_path)
        console.print(t)
    else:
        console.print("[dim]No callers.[/dim]")

    # Callees
    callees = q.callees(fn.id)
    if callees:
        t = Table(title="Callees", box=box.SIMPLE_HEAD)
        t.add_column("Callee", style="cyan")
        t.add_column("File", style="dim")
        for c in sorted(callees, key=lambda f: f.id):
            t.add_row(c.id, c.file_path)
        console.print(t)
    else:
        console.print("[dim]No callees.[/dim]")


def _inspect_class(cls: ClassNode, g: PropertyGraph) -> None:
    console.print(Panel(
        f"[bold cyan]{cls.id}[/bold cyan]",
        title="Class",
        expand=False,
    ))
    props = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    props.add_column("Key", style="dim", width=20)
    props.add_column("Value")
    props.add_row("File", cls.file_path)
    props.add_row("Lines", f"{cls.line_start}–{cls.line_end}")
    props.add_row("Bases", ", ".join(cls.bases) or "—")
    props.add_row("Methods", str(cls.method_count))
    props.add_row("Docstring", cls.docstring or "—")
    console.print(props)

    # Methods via CONTAINS edges
    methods = [
        n for n in g.successors(cls.id, edge_type="CONTAINS")
        if isinstance(n, FunctionNode)
    ]
    if methods:
        t = Table(title="Methods", box=box.SIMPLE_HEAD)
        t.add_column("Method", style="cyan")
        t.add_column("Line", justify="right", width=6)
        t.add_column("CC", justify="right", width=4)
        for m in sorted(methods, key=lambda f: f.line_start):
            t.add_row(m.name, str(m.line_start), _complexity_text(m.cyclomatic_complexity))
        console.print(t)


def _inspect_file_node(fnode: FileNode, g: PropertyGraph) -> None:
    console.print(Panel(
        f"[bold cyan]{fnode.path}[/bold cyan]",
        title="File",
        expand=False,
    ))
    props = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    props.add_column("Key", style="dim", width=20)
    props.add_column("Value")
    props.add_row("Lines", str(fnode.line_count))
    props.add_row("Avg complexity", f"{fnode.avg_complexity:.1f}")
    console.print(props)

    # Functions defined in this file
    fns = [
        n for n in g.successors(fnode.id, edge_type="DEFINES")
        if isinstance(n, FunctionNode)
    ]
    classes = [
        n for n in g.successors(fnode.id, edge_type="DEFINES")
        if isinstance(n, ClassNode)
    ]

    if classes:
        t = Table(title="Classes", box=box.SIMPLE_HEAD)
        t.add_column("Class", style="cyan")
        t.add_column("Line", justify="right", width=6)
        t.add_column("Bases", style="dim")
        t.add_column("Methods", justify="right", width=8)
        for cls in sorted(classes, key=lambda c: c.line_start):
            t.add_row(cls.name, str(cls.line_start), ", ".join(cls.bases) or "—", str(cls.method_count))
        console.print(t)

    if fns:
        t = Table(title="Functions", box=box.SIMPLE_HEAD)
        t.add_column("Function", style="cyan")
        t.add_column("Line", justify="right", width=6)
        t.add_column("CC", justify="right", width=4)
        t.add_column("Async", width=6)
        for fn in sorted(fns, key=lambda f: f.line_start):
            t.add_row(
                fn.name,
                str(fn.line_start),
                _complexity_text(fn.cyclomatic_complexity),
                "async" if fn.is_async else "",
            )
        console.print(t)

    # Imports
    imports = [
        n for n in g.successors(fnode.id, edge_type="IMPORTS")
    ]
    if imports:
        t = Table(title="Imports", box=box.SIMPLE_HEAD)
        t.add_column("Module / File", style="cyan")
        t.add_column("Type", style="dim")
        for imp in sorted(imports, key=lambda n: n.id):
            if isinstance(imp, ModuleNode):
                kind = "stdlib" if imp.is_stdlib else ("local" if imp.is_local else "third-party")
            else:
                kind = imp.node_type
            t.add_row(imp.id, kind)
        console.print(t)


# ---------------------------------------------------------------------------
# ckg embed
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--repo", default=".", type=click.Path(exists=True),
              help="Repository root (default: current directory).")
@click.option("--force", is_flag=True,
              help="Re-embed all nodes even if already stored.")
@click.option("--model", default="all-MiniLM-L6-v2", show_default=True,
              help="Sentence-transformers model to use.")
@click.pass_context
def embed(ctx: click.Context, repo: str, force: bool, model: str) -> None:
    """Embed all function and class nodes for semantic search.

    Vectors are stored in the DuckDB cache alongside the graph and used
    by [cyan]ckg query search[/cyan].  Running this command again is safe
    (already-embedded nodes are skipped unless --force is given).
    """
    db_path: Path = ctx.obj["db_path"]
    g = _load_or_build_graph(repo, db_path)
    store = GraphStore(db_path)
    embedder = NodeEmbedder(store, model_name=model)

    total_eligible = sum(
        1 for n in g.iter_nodes()
        if isinstance(n, (FunctionNode, ClassNode))
    )
    console.print(
        f"[dim]Embedding[/dim] [bold]{total_eligible}[/bold] nodes "
        f"using [cyan]{model}[/cyan]…"
    )
    n_embedded = embedder.embed_all(g, force=force)
    total_stored = embedder.embed_count()
    if n_embedded == 0:
        console.print(
            f"[green]✓[/green] Nothing new to embed "
            f"([bold]{total_stored}[/bold] nodes already embedded). "
            "Use [cyan]--force[/cyan] to re-embed."
        )
    else:
        console.print(
            f"[green]✓[/green] Embedded [bold]{n_embedded}[/bold] node(s) "
            f"([bold]{total_stored}[/bold] total in cache)"
        )


# ---------------------------------------------------------------------------
# ckg stats
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--repo", default=".", type=click.Path(exists=True),
              help="Repository root (default: current directory).")
@click.pass_context
def stats(ctx: click.Context, repo: str) -> None:
    """Show summary statistics for the knowledge graph."""
    db_path: Path = ctx.obj["db_path"]
    g = _load_or_build_graph(repo, db_path)
    q = GraphQueries(g)

    by_type = g.node_count_by_type()
    by_edge = g.edge_count_by_type()

    # Summary table
    t = Table(title="Graph Summary", box=box.SIMPLE_HEAD)
    t.add_column("Metric", style="dim")
    t.add_column("Value", justify="right", style="bold")
    t.add_row("Total nodes", str(g.node_count()))
    t.add_row("  Functions", str(by_type.get("function", 0)))
    t.add_row("  Classes", str(by_type.get("class", 0)))
    t.add_row("  Files", str(by_type.get("file", 0)))
    t.add_row("  Modules (imported)", str(by_type.get("module", 0)))
    t.add_row("Total edges", str(g.edge_count()))
    for etype in sorted(by_edge):
        t.add_row(f"  {etype}", str(by_edge[etype]))
    console.print(t)

    # Complexity stats
    fns = list(g.iter_nodes(node_type="function"))
    if fns:
        complexities = [n.cyclomatic_complexity for n in fns if isinstance(n, FunctionNode)]  # type: ignore[union-attr]
        avg_cc = sum(complexities) / len(complexities)
        max_cc = max(complexities)
        over10 = sum(1 for c in complexities if c >= 10)
        ct = Table(title="Complexity", box=box.SIMPLE_HEAD)
        ct.add_column("Metric", style="dim")
        ct.add_column("Value", justify="right", style="bold")
        ct.add_row("Average CC", f"{avg_cc:.2f}")
        ct.add_row("Max CC", str(max_cc))
        ct.add_row("Functions CC ≥ 10", str(over10))
        ct.add_row("Functions CC < 5", str(sum(1 for c in complexities if c < 5)))
        console.print(ct)

    # Top hotspots preview
    hotspots = q.complexity_hotspots(top_k=5)
    if hotspots:
        ht = Table(title="Top-5 Complexity Hotspots", box=box.SIMPLE_HEAD)
        ht.add_column("Function", style="cyan")
        ht.add_column("File", style="dim")
        ht.add_column("CC", justify="right", width=4)
        for fn, cc in hotspots:
            ht.add_row(fn.name, fn.file_path, _complexity_text(cc))
        console.print(ht)

    # Uncalled functions count
    uncalled = q.uncalled_functions()
    if uncalled:
        console.print(
            f"[yellow]⚠[/yellow]  [bold]{len(uncalled)}[/bold] uncalled function(s) "
            f"— run [cyan]ckg query dead-code[/cyan] for details"
        )

    # Most-imported files
    fi = q.file_fan_in(top_k=5)
    if fi:
        fit = Table(title="Most-Imported Files", box=box.SIMPLE_HEAD)
        fit.add_column("File", style="cyan")
        fit.add_column("Importers", justify="right", width=10)
        for file_node, cnt in fi:
            if cnt > 0:
                fit.add_row(file_node.path, str(cnt))
        console.print(fit)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    cli()
