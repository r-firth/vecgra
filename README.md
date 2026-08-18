# VectorGraph

VectorGraph is an experimental vector-native embedded graph database written in
Rust. It stores a labelled property multigraph, node embeddings, relationship
embeddings, and multivectors in one portable file. Vector retrieval and graph
traversal share one execution engine; vectors are not a plugin bolted onto a
separate graph store.

The current build is an end-to-end engine, not yet a production release. It has:

- stable `u64` node and edge IDs, parallel edges, self-edges, labels, and typed
  properties;
- native vectors on both nodes and edges, with one or more vectors per element;
- weighted late interaction across multiple query and element vectors, with
  inspectable facet matches and whole-element approximate candidate ranking;
- exact top-k search plus a persisted randomized-binary coarse tier with exact
  F16/F32 reranking, native FP16 conversion, and adaptive prefiltering;
- compressed typed candidate sets with union/intersection/difference, one-hop
  and bounded multi-hop expansion, plus pre-budget graph filtering for both
  vector and MaxSim search;
- exact bounded shortest paths with direction and relationship-label filters,
  an inspectable work-costed, expansion-budget-aware bidirectional plan for
  multi-hop work, per-endpoint expansion counters, deterministic evidence
  chains, and explicit hop/work-limit diagnostics;
- bounded semantic path retrieval in which node and relationship embeddings
  both affect path score;
- graph-range vector search that evaluates bounded reachability and ordinary
  node predicates before spending an adaptive exact/ANN candidate budget;
- one-call filtered vector execution that orders/intersects typed equality and
  numeric-range access paths before adaptive scoring, returning every scalar
  and vector plan as structured diagnostics;
- a single-file, append-only transactional log with torn-tail recovery;
- eager whole-file integrity verification on demand while selective reads keep
  vector checksum blocks lazy;
- v8 columnar checkpoints, directly mapped fixed records, CSR adjacency,
  vector sketches, compressed label postings, automatic typed-property
  equality and ordered numeric postings, lazy properties, F16/F32 persistence,
  block checksums, and an F32 write delta;
- an atomic direct bulk loader that builds the indexed checkpoint without an
  intermediate in-memory graph or transaction-log copy;
- a small Cypher-compatible one-hop surface with an inspectable costed choice
  between relationship scans and selective endpoint adjacency, a hybrid
  semantic-pattern operator that ranks nodes and relationships together, plus
  a native Rust API;
- generic atomic JSONL graph/multivector ingestion, streaming fbin plus typed
  node-metadata ingestion, a resumable GitHub engineering-history crawler,
  Tree-sitter Rust repository ingestion, and batched OpenRouter/Qwen embeddings;
- reproducible graph, vector, recovery, LadybugDB, and Neo4j comparison harnesses.

On the official million-vector VIBE/Yandex held-out corpus (200 dimensions), a
query-confidence-weighted 512-bit scan plus 12k reranks reaches 0.9967 official
recall@10 and 0.9975 recall against the exact engine at 5.42 ms p50, versus
about 21.6 ms compressed exact (16.27 ms
with an explicit 800 MB F32 cache). On the official
Graphalytics wiki-Talk graph (2.39M vertices, 5.02M directed edges), BFS agrees
with every supplied reference distance and runs at 69.2 ms p50. The ripgrep
Qwen graph still reaches 1.000 min/mean recall@10 at 5.12 ms p50. WCC and
PageRank also match every official wiki-Talk output row; PageRank runs in 546.8
ms p50 for the specified ten iterations. On the 99,560-vector MoReVec movies
workload (768 dimensions), an ordered rating predicate selects 20,588 nodes in
~0.46 ms p50; adaptive prefilter plus 5,147-row exact reranking reaches 0.9992
official recall@10 in ~2.06 ms total p50. A follow-up audit against a local
Neo4j Enterprise 2026.07 server uses its dedicated `VECTOR<FLOAT32>` property,
native filtered vector index, optimized `neo4j-admin` importer, and GDS BFS.
On these exact corpora, VectorGraph is 3.5x faster at higher-recall unfiltered
vector search and 4.8--26x faster on native filtered search. Its single-thread
BFS is 1.26x faster than GDS using four threads (and about 3x faster at equal
one-thread concurrency). VectorGraph's vector files are 3.3--3.4x smaller than
Neo4j's live database directories excluding transaction logs; Neo4j's pure
graph store is 28% smaller. These are development-machine measurements, not
cross-machine claims; see
[`docs/benchmarks.md`](docs/benchmarks.md) for methodology and caveats.

## Build and try it

```sh
cargo build --release

# Generic crawler/interchange path; external IDs may be strings or integers.
target/release/vg import-jsonl nodes.jsonl edges.jsonl graph.vg 256 f16

# Stream one fbin vector and one typed metadata JSON record per node.
target/release/vg import-node-fbin train.fbin metadata.jsonl vectors.vg f16

# Offline structural embeddings, useful for a smoke test.
target/release/vg import-rust /path/to/rust/repo repo.vg 256 hash 256

# Crawl engineering history rather than source syntax. Authentication comes
# from GITHUB_TOKEN/GH_TOKEN or an existing `gh auth login` session.
target/release/vg import-github zed-industries/zed zed.vg \
  500 500 150 50 256 hash 512

# Or use Qwen3-Embedding-8B through OpenRouter.
OPENROUTER_API_KEY=... target/release/vg \
  import-github zed-industries/zed zed-qwen.vg \
  500 500 150 50 256 qwen 128

target/release/vg stats repo.vg
target/release/vg check repo.vg
target/release/vg plan-search repo.vg both
target/release/vg numeric-range repo.vg nodes start_line int 100 200 10
target/release/vg search-text repo.vg "where is retry policy configured" 10 hash
target/release/vg range-text repo.vg 42 "nearby retry behavior" 2 10 hash both
target/release/vg shortest-path repo.vg 42 84 6 both HAS_SYNTAX 100000
target/release/vg search-facets repo.vg \
  'retry policy || exponential backoff' 10 hash both
target/release/vg semantic-text repo.vg "where is retry policy configured" 20 2 hash
target/release/vg query repo.vg \
  'MATCH (a:File)-[e:HAS_SYNTAX]->(b:Syntax) RETURN a,e,b LIMIT 10'
target/release/vg query-text repo.vg \
  'MATCH (a:Syntax)-[e:AST_CHILD]->(b:Syntax) RETURN a,e,b LIMIT 10' \
  'error handling around retries' hash
target/release/vg compact repo.vg repo.compact.vg f16
```

The GitHub path produces a general engineering graph of repositories, work
items, discussions, people, reviews, comments, commits, files, releases, and
taxonomy. Relationships such as `CLOSES`, `REVIEWS`, `CHANGES`, `ANSWERS`, and
`REPLIES_TO` are typed graph elements with their own embeddings, not metadata
on their endpoints. The crawl cache is request-fingerprinted and resumable; see
[`docs/github-import.md`](docs/github-import.md) for the schema, completeness
markers, fan-out policy, and example hybrid queries.

## Native Studio

VectorGraph Studio lives in this workspace and opens the same `.vg` file
directly in read-only mode. The database and CLI remain the default workspace
members, so desktop dependencies are only built when Studio is requested.

```sh
cargo run --release -p vectorgraph-studio -- repo.vg
```

The current native macOS vertical slice includes background file loading,
topology-aware Auto layout (radial for small forests, clustered constellations
for large forests, bounded force for general graphs), semantic level-of-detail
aggregation, GPU-painted nodes and type-aware directed relationships,
pointer-anchored deep zoom through 102,400% with individual-element recovery,
pan/pinch, direct node dragging with persistent pins
and bounded neighbor physics, interruptible spring transitions between Force,
resistance-distance Structure, and Orbit arrangements, node/edge selection,
clickable node/relationship taxonomy lenses, relationship inspection, and
ranked Text, Semantic, and Hybrid search over both nodes and relationships.
An exact evidence-path workbench turns a selected node into a path origin and
promotes a second selected node into a copper destination card and focusable
`Trace exact path` action in the evidence rail at every supported width,
without requiring either ID to be memorized. The wide Inspector mirrors that
action as a convenience rather than owning the workflow; Enter runs it from a
ready destination. Endpoint markers remain visible while the complete database
search runs on a background worker, hydrates the full node/relationship
records, and injects any sampled-out evidence into an ordered one-for-one
directed canvas chain. Stored direction, the selected physical plan, split
start/end expansion work, relationship filters, visited/read work, hop bounds,
and incomplete work-limit outcomes stay visible.
Facet lenses promote a chosen label directly into the canvas relevance field,
with a structural active marker and an exact Escape/Overview return. Hybrid
results expose separate Text and Vector contribution
rails instead of hiding retrieval behind one score, and title-bar controls
reflow into a two-tier instrument rail at narrow window widths. Press `Cmd-K`,
type a query, and press Enter; a second Enter or a result click animates into
its bounded one-hop context. Search itself frames and highlights all visible
matches through a relevance lens; unrelated structure recedes without leaving
the graph. Escape or Overview springs the saved arrangement and camera back
into place. With no exact-path draft active, double-clicking a node (or pressing
Enter on a selected node) opens a branch-balanced, relationship-diverse
two-hop context and keeps the overview available as an exact animated return.
The same box retains the
`node <id>`, `edge <id>`, `zoom <level>`, `layout auto`, `layout force`,
`layout structure`, `layout orbit`, `focus <node-id>`, `release`, `fit`, and
`clear` command surface, plus `facet node <label>` and
`facet relationship <label>`, plus
`path-start <node> [both|out|in] [max-hops]` and
`path <start> <end>` with optional direction, relationship-label, and max-hop
arguments. See
[`docs/studio.md`](docs/studio.md) for the architecture, interaction contract,
measured scope, and remaining release gates.

Studio defaults to the deterministic `hash` embedding used by local fixtures.
For a database imported with Qwen, select the same single embedding model at
launch (the database intentionally stores no per-vector model metadata):

```sh
OPENROUTER_API_KEY=... VG_EMBEDDER=qwen \
  cargo run --release -p vectorgraph-studio -- repo.vg
```

The generic JSONL records are deliberately small. Nodes use
`{"id":"doc:1","label":"Document","properties":{"title":"..."},"vectors":[[...]]}`;
edges use
`{"source":"doc:1","target":"claim:2","label":"SUPPORTS","properties":{},"vectors":[[...]]}`.
All node records precede edges, scalar JSON values map to typed properties, and
the external IDs are only join keys unless also included as properties.
For existing vector corpora, `import-node-fbin` accepts standard little-endian
fbin and one ordered JSONL record of
`{"label":"Document","properties":{"year":2026}}` per vector.

The hash embedder is deterministic test machinery, not a semantic model. A
database deliberately assumes one embedding space: dimension and similarity
are database-level invariants, and per-vector model metadata is not stored.
Different modalities can share that space through multivectors. Equal- or
custom-weight MaxSim lets each query modality/facet independently match its
best element facet before the graph element receives one aggregate score.

Mapped F16 remains the default compute source as well as the disk format.
Repeated search never silently doubles vector memory: applications that value
hot full-scan latency over footprint can explicitly call `warm_vector_cache`,
inspect its returned allocation size, and query `vector_cache_bytes`.
F32 checkpoints score directly from aligned mapped floats on little-endian
machines; they do not allocate a redundant decoded column.

## Design stance

VectorGraph is general-purpose graph infrastructure. Agent context is a strong
target workload because it needs evidence relationships, semantic entry points,
and fast bounded traversal, but the storage model does not contain agent-only
TTL, ACL, or framework concepts.

The next major engine work is richer costed multi-hop query plans and a
partitioned coarse tier for larger-than-memory files. Exact search stays the
small/selective path and the recall oracle.

Read [`docs/architecture.md`](docs/architecture.md) for the storage model and
[`docs/research.md`](docs/research.md) for the frontier work informing the
design.
