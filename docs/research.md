# Research notes

These papers and experiments shaped Vecgra's design. A cited idea is not
necessarily implemented.

## Native vector/graph execution

[TigerVector (2025)](https://arxiv.org/abs/2501.11216) stores embeddings as a
graph attribute type and composes distributed vector search with graph query
blocks. Vecgra makes a different storage tradeoff. Nodes and relationships own
vectors in one local checkpoint, and semantic traversal can score relationships
without retrieving vertices and joining them later.

[NaviX (2025)](https://arxiv.org/abs/2506.23397) is the closest ANN comparison.
It stores HNSW through graph database operations, pre-evaluates an arbitrary
predicate into a candidate subset, and adapts filtered search using local
selectivity. Vecgra uses the same plan shape. Compressed typed
candidate sets can come from labels, traversal, or application subplans; exact,
binary-sketch, and MaxSim execution all consume them before scoring/candidate
budgeting. v8 adds automatic typed equality and ordered numeric-range postings
whose candidate ranges are reconciled with the WAL. Correlation-aware partition
choice and compound predicates remain open work.

[Filtered Vector Search: State-of-the-art and Research Opportunities
(2025)](https://research.google/pubs/filtered-vector-search-state-of-the-art-and-research-opportunities/)
explains why prefilter, inline-filter, and postfilter strategies change with
selectivity and vector/filter correlation. External VIBE measurements already
show distinct local cost shapes: a contiguous unfiltered exact scan remains
competitive through about 64M float comparisons, while scattered candidates
cross at roughly 12M floats for 512+-D vectors and 32M for narrower vectors.
Correlation and partition locality are the missing inputs to that policy.

[Filtered Approximate Nearest Neighbor Search in Vector Databases
(2026)](https://arxiv.org/abs/2602.11443) introduces the MoReVec dataset, which
couples 768-dimensional text embeddings to real scalar
filters, and its Global-Local Selectivity metric distinguishes global predicate
cardinality from the predicate density near a query. Its experiments also find
that hybrid exact/approximate execution avoids optimizer mistakes and that IVF
can beat HNSW for low-selectivity filtered search. Vecgra now runs the
official 99,560-row Movies/medium workload through fbin and typed JSONL.
It exactly matches all tested filter cardinalities. At 10.8% selectivity the
planner retains exact search; at 20.7% it switches to a 5,147-row sketch rerank,
reaching 0.9992 official recall@10 at ~2.03 ms total p50 including ordered range
evaluation. Any partitioned index should earn its place on MoReVec, not on
unfiltered recall alone.

[Approximate Nearest Neighbor Search with Graph Range Filters
(2026)](https://arxiv.org/abs/2607.00727) defines the ANNGR workload directly:
nearest-neighbor search constrained to the `r`-hop neighborhood of a query
node. Its DLH design converts distance-aware graph labels into a small number
of set intersections, compresses large sets with Bloom filters, and memoizes
intermediate state for repeated query nodes. Vecgra prefilters the exact
version. It intersects a compressed node range with an optional label or
property predicate before choosing exact gather or persisted sketch/rerank. The result reports
both reachable cardinality and the chosen vector plan. Avoiding traversed-edge
materialization cuts wiki-Talk radius-3 and radius-6 expansion time by roughly
one quarter. Broad or repeated neighborhoods may be cheaper with memoized or
approximate membership than with a rebuilt exact frontier.

[BoomHQ (2026)](https://arxiv.org/abs/2604.24552) learns vector-attribute
correlation and query-neighborhood patterns to choose execution hints. Requiring
a learned optimizer would be a poor default for a small embedded engine. Vecgra
can still record the useful inputs: predicate and neighborhood cardinality, ANN
survivors, rerank work, and achieved recall. Those measurements can refine the
deterministic property, label, and full-scan plan.

[AkasicDB (2026)](https://arxiv.org/abs/2608.09214) composes ANN as an iterator
inside a traversal and join plan. It coordinates separate vector, graph, and
PostgreSQL stores. Vecgra instead keeps node and relationship multivectors,
candidate sets, and graph records in one local file. In both designs, operators
need to compose incrementally rather than hide behind an application RAG
pipeline.

## Graph algorithm execution

[Algorithm Support for Graph Databases, Done Right
(2026)](https://arxiv.org/abs/2601.06705) argues that graph algorithms belong
inside the graph system and share its storage and execution machinery. For
Vecgra, that means a fast mapped scan, frontier, and pull API. Algorithms and
hybrid retrieval plans should compose from those operations rather than require
a separate analytics export.

The official Graphalytics wiki-Talk dataset exercises three physical access
patterns against complete supplied outputs: directed frontier
BFS, undirected WCC, and iterative incoming-pull PageRank with dangling-mass
redistribution. Semantic candidate sets use the same machinery as graph
frontiers: compressed sets, endpoint projection, CSR locality, and bounded
scoring.

## Quantized coarse search

[The RaBitQ Library (2025)](https://openreview.net/forum?id=OeZHhOsFir) and its
[official implementation](https://github.com/VectorDB-NTU/RaBitQ-Library)
show that randomized quantization can provide very compact similarity estimates
with useful error bounds and fast bitwise kernels. [VSAG
(2025)](https://arxiv.org/abs/2503.17911) measures the same systems concerns:
cache layout, low-precision distance computation, and parameter selection can
matter as much as the ANN graph.

Vecgra v6 uses a simpler measured first tier: two
structured randomized Hadamard rotations followed by up to a 512-bit sign
sketch (earlier 256-bit v5 files remain readable).
The sketch is scanned sequentially. On datasets with one vector per element, a
query-only confidence mask gives above-mean rotated coordinates twice the
mismatch weight; source F16/F32 vectors are then reranked exactly. This
asymmetric score raised
official VIBE recall from 0.9938 at 20k reranks to 0.9967 at only 12k while
reducing p50 from 5.963 to 5.419 ms, with no format or file-size change. It is
not presented as RaBitQ and has no equivalent theoretical error bound; exact
recall curves decide whether it is used. It adds little to the file, builds in
portable Rust, reads contiguously, handles node and edge multivectors, and does
not keep a pointer-heavy ANN graph in RAM.

## Update-friendly vector storage

[SPFresh](https://arxiv.org/abs/2410.14452) uses local incremental rebalancing
of centroid partitions instead of periodically rebuilding a global ANN index.
[Turbopuffer's architecture](https://turbopuffer.com/docs/architecture) applies
that family of ideas to object storage: a compact centroid directory chooses a
small number of partitions, partition data is fetched in large ranges, and
recent WAL data remains immediately searchable through exhaustive scan. Its
[native filtering design](https://turbopuffer.com/blog/native-filtering) aligns
attribute postings with vector clusters so filters and ANN maintenance share
partition boundaries.

Object-store latency differs from local mmap and SSD access. The useful pieces
are:

- immutable indexed base plus an exact recent delta;
- large contiguous partitions instead of pointer-heavy random I/O;
- local split/merge maintenance;
- filter postings aware of vector partitions;
- exact fallback based on candidate cardinality;
- cold, warm, and hot representations selected by measured work and explicit
  memory policy.

The 2026 storage tutorial [Vector Search for the Future: From Memory-Resident,
Static Heterogeneous Storage, to Cloud-Native
Architectures](https://arxiv.org/abs/2601.01937) compares block layouts,
memory and storage tiers, query strategies, and update maintenance. Vecgra's
scale boundary is mmap page and SSD-range locality, not object-store RPC. A
future partition directory should therefore point to contiguous searchable
blocks, not an in-memory centroid graph over scattered data.

Vecgra already implements the immutable F16 base, persisted binary coarse
index, exact F32 delta, explicit inspectable hot promotion, and block-ranged
verification. The next scale tier should compare an SPFresh-like contiguous partition layer
with NaviX-like filtered HNSW, using exact recall@k as the oracle.

Two no-format-change shortcuts failed on the
million-row VIBE workload. Eight 16-bit multi-probe tables over disjoint pieces
of the 512-bit signature inspected 188k rows at radius three but reached only
0.790 official recall@10 at 6.94 ms; radius four inspected 480k, reached 0.960,
and took 12.51 ms. A half-signature cascade selecting 100k rows before the full
signature reached 0.972 at 7.96 ms. The existing sequential full-signature
path was both faster (roughly 5.5--5.9 ms in those paired runs) and more
accurate (0.990 on the 200-query sample). Neither experiment remains in the
engine API.

A reproducible Faiss IVF design probe (`scripts/ivf_frontier_probe.py`) did
better. With 4,096 centroids, 512 probes inspect about 118k
VIBE rows and reach 0.998 official recall@10 at 1.76 ms in Faiss; 256 probes
inspect 59k but fall to 0.989. This is not a Vecgra performance claim.
The result points to a concrete design: train a compact centroid directory, physically
co-locate vectors/sketches and filter postings by partition, probe enough
contiguous partitions to meet measured recall, then scan the exact WAL delta.
Random short-code postings and scattered gathers are not an adequate stand-in
for that physical layout.

## Relationship and path semantics

[PathRAG (2025)](https://arxiv.org/abs/2502.14902) argues for relational path
retrieval with redundancy and flow-aware pruning rather than flat entity
retrieval. Vecgra ranks paths as evidence objects by combining seed, edge, and
destination-node similarity with path decay, hop penalty, degree penalty, and
hard expansion bounds. The weights still need calibration. That work belongs
in retrieval, not in an agent framework baked into the database.

Vecgra embeds every relationship. This avoids a schema-time decision about
which predicates are "semantic" and lets a query distinguish relations between
similar endpoints. Path embeddings should be derived or cached per query, not
stored for every possible path.

The engine exposes an exact bounded shortest-path operator as the
non-semantic evidence baseline. It normalizes traversal order by stable node
and relationship IDs, supports outgoing/incoming/undirected and relationship
label constraints, and reports expansion-budget truncation separately from
"not found within the hop bound." Start- and end-side expansion counters expose
how its costed bidirectional plan spent that budget, and always sum to the total
expanded count. This gives Studio and retrieval planners a reproducible path
object to compare against learned or embedding-weighted path ranking without
confusing an incomplete search with graph absence.

## Multimodal graphs

[MG²-RAG (2026)](https://arxiv.org/abs/2604.04969) constructs multi-granularity
multimodal graphs with unified textual and visual nodes, then propagates dense
relevance over the graph. Vecgra stores multiple vectors per element in one
embedding space instead of adding separate image tables. Typed properties and
edges hold provenance. A query can select the "best matching view" or use
weighted late interaction. Each text, image, or query facet selects its best stored facet, their
scores are averaged, and the result exposes the matched facet indices. The
binary coarse tier mirrors that MaxSim reduction at the whole-element level
before exact reranking, so elements with more views do not consume more
candidate slots.

One embedding model per database keeps this simple. A multimodal embedding
model can place text, image regions, audio, or other views in one space without
storing redundant model metadata beside every vector. Changing embedding space
is a database migration, not a row-level concern.

## Product boundary

Agent context is a demanding benchmark, not the schema. Vecgra needs fast
semantic entry, relationship evidence, bounded traversal, incremental writes,
and compact local deployment. It does not need agent memories, TTL policy,
ACLs, prompt formats, or a RAG framework in the storage model.

The product is a small embedded property graph whose storage and planner treat
vectors, filters, relationships, and paths as one workload. "Faster Cypher" is
useful compatibility, not the reason to build it.
