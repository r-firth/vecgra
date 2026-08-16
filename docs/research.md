# Frontier research and design implications

This is the working research map, not a claim that every cited idea is already
implemented.

## Native vector/graph execution

[TigerVector (2025)](https://arxiv.org/abs/2501.11216) makes embeddings a graph
attribute type and composes distributed vector search with graph query blocks.
It validates the unified-system direction, but VectorGraph goes further at the
local storage level: nodes and relationships both own vectors, the checkpoint
physically co-designs graph and vector access, and semantic traversal can score
relationships rather than retrieving vertices then joining.

[NaviX (2025)](https://arxiv.org/abs/2506.23397) is the most directly relevant
ANN work. It stores HNSW through graph-DB primitives, pre-evaluates an arbitrary
predicate into a candidate subset, and adapts filtered search using local
selectivity. VectorGraph now has that execution contract: compressed typed
candidate sets can come from labels, traversal, or application subplans; exact,
binary-sketch, and MaxSim execution all consume them before scoring/candidate
budgeting. v8 adds automatic typed equality and ordered numeric-range postings
whose candidate ranges are reconciled with the WAL. The remaining work is
correlation-aware partition choice and richer compound predicate classes,
rather than separate “vector DB” behavior.

[Filtered Vector Search: State-of-the-art and Research Opportunities
(2025)](https://research.google/pubs/filtered-vector-search-state-of-the-art-and-research-opportunities/)
formalizes why prefilter, inline-filter, and postfilter strategies change with
selectivity and vector/filter correlation. External VIBE measurements already
show distinct local cost shapes: a contiguous unfiltered exact scan remains
competitive through about 64M float comparisons, while scattered candidates
cross at roughly 12M floats for 512+-D vectors and 32M for narrower vectors.
Observed correlation and partition locality should refine those measured
priors.

[Filtered Approximate Nearest Neighbor Search in Vector Databases
(2026)](https://arxiv.org/abs/2602.11443) adds two especially useful pieces:
the MoReVec corpus couples 768-dimensional text embeddings to real scalar
filters, and its Global-Local Selectivity metric distinguishes global predicate
cardinality from the predicate density near a query. Its experiments also find
that hybrid exact/approximate execution avoids optimizer mistakes and that IVF
can beat HNSW for low-selectivity filtered search. VectorGraph now runs the
official 99,560-row Movies/medium workload through ordinary fbin + typed JSONL.
It exactly matches all tested filter cardinalities. At 10.8% selectivity the
planner retains exact search; at 20.7% it switches to a 5,147-row sketch rerank,
reaching 0.9992 official recall@10 at ~2.03 ms total p50 including ordered range
evaluation. This is the paper's adaptive exact/approximate lesson implemented
as a measured local policy, not merely cited. A partition tier should still be
judged on MoReVec rather than justified by unfiltered recall alone.

[Approximate Nearest Neighbor Search with Graph Range Filters
(2026)](https://arxiv.org/abs/2607.00727) defines the ANNGR workload directly:
nearest-neighbor search constrained to the `r`-hop neighborhood of a query
node. Its DLH design converts distance-aware graph labels into a small number
of set intersections, compresses large sets with Bloom filters, and memoizes
intermediate state for repeated query nodes. VectorGraph already exposes the
exact prefilter version of this contract: a node-only compressed bounded range
is intersected with an optional label/property predicate, and only then does
the engine choose exact gather or persisted sketch/rerank. The result reports
both reachable cardinality and the chosen vector plan. Avoiding traversed-edge
materialization cuts wiki-Talk radius-3 and radius-6 expansion time by roughly
one quarter. The paper identifies the next adaptive branch: broad or repeatedly
queried neighborhoods may favor memoized/approximate membership and
postfiltering over rebuilding a complete exact frontier.

[BoomHQ (2026)](https://arxiv.org/abs/2604.24552) goes beyond scalar
selectivity by learning vector–attribute correlation and query-neighborhood
patterns to choose execution hints. A small embedded engine should not require
a learned model to function, but it should collect the same evidence:
predicate cardinality, neighborhood cardinality, ANN survivors, rerank work,
and achieved recall. The inspectable property/label/full-scan plan is the
deterministic base on which observed correlation can later refine choices.

[AkasicDB (2026)](https://arxiv.org/abs/2608.09214) independently validates
unified vector, graph, and relational execution, composing ANN as an iterator
inside a traversal/join plan. It uses dedicated vector, graph, and PostgreSQL
stores coordinated by one execution layer. VectorGraph's differentiator is
physical rather than merely logical integration: one local file, native node
and relationship multivectors, shared candidate sets, and no separate service
or relational runtime. Its lesson is still important: native operators must be
composable and incremental instead of hidden behind an application RAG
pipeline.

## Graph algorithm execution

[Algorithm Support for Graph Databases, Done Right
(2026)](https://arxiv.org/abs/2601.06705) argues that graph algorithms belong
inside the graph system and should share its storage and execution machinery,
rather than requiring a separate export-oriented analytics stack. That is a
useful boundary for this engine: the primitive should be a fast mapped graph
scan/frontier/pull API on which algorithms and hybrid retrieval plans compose,
not a catalogue of opaque agent-specific routines.

The official Graphalytics wiki-Talk corpus now exercises three different
physical access patterns against complete supplied outputs: directed frontier
BFS, undirected WCC, and iterative incoming-pull PageRank with dangling-mass
redistribution. This matters to the vector-native design because a semantic
candidate set is also a graph frontier: compressed sets, endpoint projection,
CSR locality, and bounded scoring should remain shared execution primitives.

## Quantized coarse search

[The RaBitQ Library (2025)](https://openreview.net/forum?id=OeZHhOsFir) and its
[official implementation](https://github.com/VectorDB-NTU/RaBitQ-Library)
show that randomized quantization can provide very compact similarity estimates
with useful error bounds and fast bitwise kernels. [VSAG
(2025)](https://arxiv.org/abs/2503.17911) reinforces the systems lessons:
cache-friendly organization, low-precision distance computation, and automatic
parameter selection matter as much as the high-level ANN graph.

VectorGraph v6 persists a deliberately simpler measured first tier: two
structured randomized Hadamard rotations followed by up to a 512-bit sign
sketch (earlier 256-bit v5 files remain readable).
The sketch is scanned sequentially. On one-vector corpora, a query-only
confidence mask gives above-mean rotated coordinates twice the mismatch weight;
source F16/F32 vectors are then reranked exactly. This asymmetric score raised
official VIBE recall from 0.9938 at 20k reranks to 0.9967 at only 12k while
reducing p50 from 5.963 to 5.419 ms, with no format or file-size change. It is
not presented as RaBitQ and has no equivalent theoretical error bound; exact
recall curves decide whether it is used. Its useful properties are tiny in-file
overhead, portable construction, contiguous reads, natural node/edge
multivectors, and no pointer-heavy ANN graph in RAM.

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

The local embedded setting has different latency economics, but the transferable
ideas are strong:

- immutable indexed base plus an exact recent delta;
- large contiguous partitions instead of pointer-heavy random I/O;
- local split/merge maintenance;
- filter postings aware of vector partitions;
- exact fallback based on candidate cardinality;
- cold, warm, and hot representations selected by measured work and explicit
  memory policy.

The 2026 storage tutorial [Vector Search for the Future: From Memory-Resident,
Static Heterogeneous Storage, to Cloud-Native
Architectures](https://arxiv.org/abs/2601.01937) independently organizes the
frontier around block layout, memory/SSD/object tiers, query strategy, and
update maintenance. For this local engine the relevant scale boundary is mmap
page and SSD-range locality, not object-store RPC, but it reinforces that a
future partition directory must point to contiguous independently searchable
blocks rather than merely adding an in-memory centroid graph.

VectorGraph already implements the immutable F16 base, persisted binary coarse
index, exact F32 delta, explicit inspectable hot promotion, and block-ranged
verification.
The next scale tier should compare an SPFresh-like contiguous partition layer
with NaviX-like filtered HNSW, using exact recall@k as the oracle.

Two tempting no-format-change shortcuts were also tested and rejected on the
million-row VIBE workload. Eight 16-bit multi-probe tables over disjoint pieces
of the 512-bit signature inspected 188k rows at radius three but reached only
0.790 official recall@10 at 6.94 ms; radius four inspected 480k, reached 0.960,
and took 12.51 ms. A half-signature cascade selecting 100k rows before the full
signature reached 0.972 at 7.96 ms. The existing sequential full-signature
path was both faster (roughly 5.5--5.9 ms in those paired runs) and more
accurate (0.990 on the 200-query sample). Neither experiment remains in the
engine API.

A reproducible Faiss IVF design probe (`scripts/ivf_frontier_probe.py`) gives a
more constructive result. With 4,096 centroids, 512 probes inspect about 118k
VIBE rows and reach 0.998 official recall@10 at 1.76 ms in Faiss; 256 probes
inspect 59k but fall to 0.989. This is not a VectorGraph performance claim.
It narrows the durable design: train a compact centroid directory, physically
co-locate vectors/sketches and filter postings by partition, probe enough
contiguous partitions to meet measured recall, then scan the exact WAL delta.
Random short-code postings and scattered gathers are not an adequate stand-in
for that physical layout.

## Relationship and path semantics

[PathRAG (2025)](https://arxiv.org/abs/2502.14902) argues for relational path
retrieval with redundancy and flow-aware pruning rather than flat entity
retrieval. This supports treating a path as a ranked evidence object. The engine
currently combines seed, edge, and destination-node similarity with path decay,
hop penalty, degree penalty, and hard expansion bounds. Future work should learn
or calibrate these components without turning the database into an agent
framework.

Embedding every relationship is intentional. It removes a schema-time decision
about which predicates are “semantic,” allows a query to distinguish relations
between equally similar endpoints, and provides the primitive needed for path
ranking. Path embeddings themselves should be derived/cacheable query artifacts,
not a permanently materialized vector for every combinatorial path.

## Multimodal graphs

[MG²-RAG (2026)](https://arxiv.org/abs/2604.04969) constructs multi-granularity
multimodal graphs with unified textual and visual nodes, then propagates dense
relevance over the graph. The storage implication is not separate image tables;
it is multiple vectors per element in one shared embedding space, plus typed
provenance in ordinary properties and edges. VectorGraph's multivector model
supports both a single query's “best matching view” and weighted late
interaction: each text/image/query facet selects its best stored facet, their
scores are averaged, and the result exposes the matched facet indices. The
binary coarse tier mirrors that MaxSim reduction at the whole-element level
before exact reranking, so elements with more views do not consume more
candidate slots.

The one-model-per-database assumption remains useful. A multimodal embedding
model can place text, image regions, audio, or other views in one space without
storing redundant model metadata beside every vector. Changing embedding space
is a database migration, not a row-level concern.

## Product boundary

Agent context is a demanding benchmark, not the schema. The database should be
excellent at semantic entry, evidence relationships, bounded traversal,
incremental writes, provenance-like properties, and compact local deployment.
It should not hard-code agents, memories, TTL policy, ACLs, prompt formats, or a
particular RAG framework.

The defensible position is: **a small embedded property graph whose physical
storage and planner treat vectors, filters, relationships, and paths as one
workload**. “Faster Cypher” is useful compatibility; it is not the core design.
