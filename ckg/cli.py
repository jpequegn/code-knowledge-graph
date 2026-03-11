"""CLI entrypoint for the Code Knowledge Graph tool."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """Code Knowledge Graph — structural analysis for Python codebases.

    Build a property graph from your Python source code and answer
    structural questions: impact radius, dead code, complexity hotspots,
    dependency paths, and more.
    """


@cli.command()
@click.argument("repo", default=".", type=click.Path(exists=True))
@click.option("--incremental", is_flag=True, help="Only re-parse changed files.")
def build(repo: str, incremental: bool) -> None:
    """Build the knowledge graph from a Python repository.

    REPO is the path to the root of the repository (default: current directory).
    """
    mode = "incremental" if incremental else "full"
    console.print(f"[bold green]Building[/bold green] graph from [cyan]{repo}[/cyan] ({mode})…")
    console.print("[yellow]Not yet implemented — coming in issue #6 (parser) and #4 (graph).[/yellow]")


@cli.command()
@click.argument("subcommand", type=click.Choice(["impact", "callers", "hotspots", "dead-code", "path"]))
@click.argument("args", nargs=-1)
def query(subcommand: str, args: tuple[str, ...]) -> None:
    """Run a structural query against the knowledge graph.

    \b
    Subcommands:
      impact <node_id>          Impact radius of a function change
      callers <name>            All callers of a function
      hotspots                  Top-10 complexity hotspots
      dead-code                 Functions never called
      path <file_a> <file_b>    Dependency path between two files
    """
    console.print(f"[bold green]Query:[/bold green] {subcommand} {' '.join(args)}")
    console.print("[yellow]Not yet implemented — coming in issue #7 (queries).[/yellow]")


@cli.command()
@click.argument("kind", type=click.Choice(["node", "file"]))
@click.argument("target")
def inspect(kind: str, target: str) -> None:
    """Inspect a node or file in the knowledge graph.

    \b
    Kinds:
      node <node_id>   Full details for a single function or class node
      file <path>      All nodes defined in a file
    """
    console.print(f"[bold green]Inspect {kind}:[/bold green] [cyan]{target}[/cyan]")
    console.print("[yellow]Not yet implemented — coming in issue #4 (graph).[/yellow]")


@cli.command()
def stats() -> None:
    """Show summary statistics for the knowledge graph."""
    console.print("[bold green]Stats[/bold green]")
    console.print("[yellow]Not yet implemented — coming in issue #4 (graph).[/yellow]")


def main() -> None:
    cli()
