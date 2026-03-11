# Code Knowledge Graph

A property graph over Python codebases that answers structural questions flat vector search cannot.

## What it does

Build a typed graph of your Python source code — functions, classes, files, and their relationships (calls, imports, defines, raises) — then query it:

```bash
ckg build --repo ~/Code/my-project
ckg query hotspots                              # top-10 complexity
ckg query impact 'database.py::add_episode'    # what breaks if this changes?
ckg query dead-code                            # functions never called
ckg query path 'cli.py' 'database.py'          # dependency chain
ckg inspect node 'database.py::add_episode'    # full node details
ckg stats                                      # graph summary
```

## Installation

```bash
uv sync
uv run ckg --help
```

## Development status

| Issue | Feature | Status |
|-------|---------|--------|
| #5 | Project setup | ✅ Done |
| #6 | AST parser | 🔜 Next |
| #4 | Property graph | 🔜 Planned |
| #2 | DuckDB persistence | 🔜 Planned |
| #7 | Structural queries | 🔜 Planned |
| #1 | CLI (full) | 🔜 Planned |
| #3 | Eval on P³ | 🔜 Planned |
