# Code Knowledge Graph — Implementation Plan

## What We're Building

A property graph of a Python codebase where nodes carry rich attributes (functions, classes, files, variables) and edges express typed structural relationships (calls, imports, inherits, reads, writes, raises). The graph answers structural questions that flat vector search fundamentally cannot.

## Why This Matters

The TWiML episode on Blesses (autonomous code generation platform) identified their knowledge graph as the key differentiator — not the LLM. Vector search finds *similar* text. A property graph answers *structural* questions:

- "What would break if I change the signature of `add_episode()`?"
- "Which functions write to the database?"
- "What is the full transitive dependency tree of `p3/cli.py`?"
- "Which files are most central to the codebase? (highest fan-in)"

This is a direct upgrade to `codebase-context-engine`: instead of call graphs built from import edges alone, we get a full property graph with typed relationships and rich node metadata.

## Graph Schema

### Node Types

```
File
  path: str
  line_count: int
  complexity: float          # average cyclomatic complexity of functions

Function
  name: str
  file_path: str
  line_start: int
  line_end: int
  signature: str             # full signature string
  docstring: str | None
  return_type: str | None
  param_count: int
  cyclomatic_complexity: int # number of branches + 1
  is_async: bool
  is_method: bool
  class_name: str | None     # if it's a method

Class
  name: str
  file_path: str
  line_start: int
  bases: list[str]           # parent class names
  docstring: str | None
  method_count: int

Module
  name: str                  # e.g. 'anthropic', 'duckdb'
  is_stdlib: bool
  is_local: bool
```

### Edge Types

```
CALLS         Function → Function    (function f calls function g)
IMPORTS       File → File | Module   (file imports another file or module)
INHERITS      Class → Class          (class inherits from class)
DEFINES       File → Function|Class  (file defines this symbol)
CONTAINS      Class → Function       (class contains this method)
READS         Function → variable    (function reads a module-level var)
RAISES        Function → Exception   (function raises this exception type)
RETURNS       Function → type        (return type annotation, if present)
```

### Edge Properties
```
weight: int      # call count (how many times f calls g in its body)
line: int        # line number where the edge occurs
```

## Architecture

```
ckg/
├── __init__.py
├── parser.py          # AST walker: produces nodes and edges from .py files
├── graph.py           # PropertyGraph: build, query, serialize, load
├── queries.py         # Named structural queries (impact_radius, fan_in, etc.)
├── store.py           # DuckDB persistence layer (nodes + edges as tables)
├── embedder.py        # Optional: embed node docstrings for hybrid search
└── cli.py             # `ckg build`, `ckg query`, `ckg inspect`, `ckg serve`

tests/
├── test_parser.py
├── test_graph.py
└── test_queries.py

pyproject.toml
README.md
```

## Implementation Phases

### Phase 1: AST parser (parser.py)

Walk the Python AST of a file and extract all nodes and edges.

```python
@dataclass
class ParseResult:
    file_node: FileNode
    functions: list[FunctionNode]
    classes: list[ClassNode]
    edges: list[Edge]          # all edges found in this file

result = parse_file("p3/database.py", project_root="p3/")
```

Key extractions:
- `ast.FunctionDef` / `ast.AsyncFunctionDef` → `FunctionNode`
- `ast.ClassDef` → `ClassNode` (with bases resolved to local names where possible)
- `ast.Import` / `ast.ImportFrom` → `IMPORTS` edges
- `ast.Call` → `CALLS` edges (resolve `self.method()` to class method)
- `ast.Raise` → `RAISES` edges
- Return type annotations → `RETURNS` edges

Cyclomatic complexity: count `if`, `for`, `while`, `except`, `with`, `assert`, boolean operators (`and`/`or`) + 1.

### Phase 2: Property graph (graph.py)

In-memory graph backed by networkx `MultiDiGraph` (directed, multiple edge types between same nodes).

```python
graph = PropertyGraph()
graph.build_from_directory("p3/")

# Node access
node = graph.get_node("p3/database.py::add_episode")
# → FunctionNode(name='add_episode', complexity=4, ...)

# Neighbor traversal
graph.successors("p3/database.py::add_episode", edge_type="CALLS")
# → [FunctionNode('p3/database.py::get_episode_by_url'), ...]

graph.predecessors("p3/database.py::add_episode", edge_type="CALLS")
# → all callers of add_episode
```

### Phase 3: DuckDB persistence (store.py)

Serialize the graph to DuckDB so it survives across sessions and can be queried with SQL.

```sql
CREATE TABLE nodes (id TEXT PRIMARY KEY, type TEXT, name TEXT, file_path TEXT,
                    line_start INT, properties JSON);
CREATE TABLE edges (id INTEGER PRIMARY KEY, src TEXT, dst TEXT, edge_type TEXT,
                    weight INT, line INT, properties JSON);
```

```python
store = GraphStore("~/.ckg/graph.db")
store.save(graph)
store.load() -> PropertyGraph   # reconstruct from DB
store.invalidate_file("p3/database.py")  # remove and re-parse one file
```

Incremental rebuild: compare file mtime against last-parsed timestamp. Only re-parse changed files.

### Phase 4: Structural queries (queries.py)

The payoff — named queries that answer real engineering questions:

```python
queries = GraphQueries(graph)

# Impact analysis: what would break if I change this function?
queries.impact_radius("p3/database.py::add_episode", depth=3)
# → {distance_1: [callers], distance_2: [callers of callers], ...}

# Fan-in: which functions are called by the most other functions?
queries.fan_in(top_k=10)
# → [(function, caller_count), ...]  → "database.py::add_episode" is called by 8 functions

# Dependency path: how does cli.py depend on database.py?
queries.dependency_path("p3/cli.py", "p3/database.py")
# → cli.py → transcriber.py → database.py

# Complexity hotspots
queries.complexity_hotspots(top_k=10)
# → functions with highest cyclomatic complexity

# Dead code candidates: functions defined but never called
queries.uncalled_functions()

# Exception surfaces: all functions that raise a given exception type
queries.raises_exception("DuplicateKeyError")
```

### Phase 5: Hybrid search (embedder.py)

Combine structural queries with semantic search: embed each function's docstring + signature, store in DuckDB alongside the graph. Now you can query "functions related to audio processing" AND intersect with structural results.

```python
# Semantic + structural: functions similar to 'transcription' that are also callers of database functions
results = hybrid_search(
    semantic_query="audio transcription",
    structural_filter=lambda n: graph.has_path(n, "p3/database.py")
)
```

### Phase 6: CLI

```bash
ckg build --repo ~/Code/parakeet-podcast-processor   # build and persist graph
ckg build --incremental                               # only re-parse changed files

ckg query impact "p3/database.py::add_episode"        # impact radius
ckg query callers "add_episode"                       # all callers
ckg query hotspots                                    # complexity top 10
ckg query dead-code                                   # uncalled functions
ckg query path "cli.py" "database.py"                 # dependency path

ckg inspect node "p3/database.py::add_episode"        # full node details
ckg inspect file "p3/database.py"                     # all nodes in file

ckg stats                                             # node/edge counts, complexity avg
```

### Phase 7: Eval — answer 10 structural questions about P³

Write 10 structural questions about the P³ codebase. Answer them manually (ground truth). Then answer them with the graph. Measure:
- Accuracy: does the graph give the right answer?
- Speed: graph query vs. manual grep
- Insight: did the graph reveal something you didn't know?

## Key Design Decisions

**Why MultiDiGraph (multiple edges between same nodes)?**
A function can call another function in multiple places (different lines, different weights). A file can import the same module for different symbols. The multi-edge model captures this.

**Why DuckDB for persistence, not a graph DB (Neo4j)?**
Zero extra infrastructure. DuckDB can express graph traversal via recursive CTEs. Follow-on: export to Neo4j for richer visualization.

**Why cyclomatic complexity?**
It's the simplest metric that predicts test difficulty and change risk. Functions with complexity > 10 are candidates for refactoring. The graph makes these visible at the codebase level.

**What we're NOT building**
- Dynamic analysis (runtime call graphs) — static only
- Multi-language support — Python only
- IDE plugin
- Real-time file watching

## Acceptance Criteria

1. `ckg build p3/` completes without errors, builds graph with ≥50 nodes and ≥100 edges
2. `ckg query impact "database.py::add_episode"` returns correct callers (verified manually)
3. `ckg query dead-code` returns at least 1 genuinely uncalled function
4. Incremental rebuild: modify one file, only that file re-parsed (verified by timing)
5. 10-question eval: ≥8/10 correct answers, all faster than manual grep

## Learning Outcomes

After building this you will understand:
- Why property graphs beat flat vector search for structural questions
- What cyclomatic complexity actually measures and why it matters
- How Blesses and similar autonomous code systems build understanding of a codebase
- Why impact analysis is the killer feature of structural code tools
- The difference between static and dynamic call graphs (and when each matters)
