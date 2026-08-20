# Development benchmarks

These numbers are guardrails for engineering decisions, not published product
claims. They were collected on the same Apple Silicon development machine with
the Rust release profile (`thin` LTO, one codegen unit). Files were generally in
the OS cache unless a result explicitly says otherwise.

## Real graph datasets

The dataset is a shallow checkout of `BurntSushi/ripgrep`. The original Qwen
build uses one vector on every node and edge:

| Item | Count |
|---|---:|
| Rust files | 110 |
| Source bytes | 1,902,104 |
| Syntax nodes | 253,864 |
| Total graph nodes | 253,975 |
| Total graph edges | 253,974 |
| Native node + edge vectors | 507,949 |
| Vector dimension | 256 |
| Unique Qwen embedding payloads | 16,201 |

Every graph node and relationship owns an embedding. Exact-text deduplication
reduced remote Qwen3-Embedding-8B calls by roughly 31× while preserving a
vector for every element. The original Qwen import took about 169 seconds and
was dominated by the remote provider. The same topology with the offline hash
embedder imported in about 2.0 seconds, which is useful as a storage/parser
smoke measurement but not a semantic-quality comparison.

The indexed v5 form is 342,300,360 bytes (about 326 MiB): approximately 260 MB
is vector data; the remainder is dictionary, fixed records, properties, both
CSR directions, checksums, and 16.3 MB of persisted binary sketches.

The current importer uses two vector facets on syntax nodes and AST edges: a
deduplicated structural meaning plus a shared enclosing-symbol/file context.
On the same source it produces:

| Item | Count |
|---|---:|
| Graph nodes / edges | 253,975 / 253,974 |
| Native node + edge vectors | 1,015,677 |
| Unique embedding payloads | 20,599 |
| Indexed v7 F16 file | 702,270,012 bytes |
| Direct hash bulk build | 6.70 s |
| Direct hash private peak | 462 MB |

This doubles semantic facets for only about 4,400 additional unique model
inputs over the older importer. The hash embedder measures parsing/storage, not
semantic quality. The v7 result also contains widened 512-bit sketches,
mapped owner columns, and 3.47M automatic property postings. The earlier direct
v5 builder took 4.36 s / 453 MB; v7 spends more build work and 72.8 MB of file
space on those query structures while remaining below 500 MB private.

## Current engine

Representative results:

| Operation | Result |
|---|---:|
| Open indexed v5 checkpoint | about 20–30 ms |
| Stats-only max RSS / private footprint, Qwen v5 | about 92 MB / 8.8 MB |
| Point adjacency, degree 1, Rust API | 0.000083 ms p50 |
| `File-HAS_SYNTAX-Syntax`, limit 100 | about 0.003 ms p50 |
| Full exact search, 507,949 × 256, hot F32 | 9.8 ms p50 |
| v7 sketch + 20k exact rerank, 100 derived queries | 5.12 ms p50 |
| Recall@10 for that 20k run | min 1.000, mean 1.000 |
| First full compressed search | about 85 ms with cached file pages |
| Label-filtered `File` search, 110 candidates | about 0.015 ms p50 |
| First `File` search with block verification | about 12.6 ms |
| First persisted-index search (including search) | 7.9 ms |

The v7 compaction of the original Qwen graph adds automatic postings for all
3.47 million scalar properties. Adaptive 12-byte entries occupy 41.6 MB. The
complete file is 389,635,280 bytes versus 342,300,360 bytes for v5; that 13.8%
total difference also includes v6's mapped vector-owner columns, not just the
property table. Repeated exact equality measurements were:

| Predicate | Matches | v5 mapped scan p50 | v7 posting + exact verify p50 |
|---|---:|---:|---:|
| `kind = "identifier"` | 54,690 | 4.880 ms | 1.624 ms |
| `path = "build.rs"` | 1 | 1.200 ms | 0.000083 ms |

Warm open moved from 20.4 ms to about 29–32 ms because every fixed posting is
structurally validated. Posting fingerprints are only candidate keys; encoded
typed values are compared exactly, and WAL property changes are reconciled by
an exact overlay scan. The unique-value number is a hot repeated lookup, not a
cold storage latency claim.

The 100-query approximate result uses deterministic 85/15 mixtures of stored
Qwen vectors, with the exact engine as ground truth. It is a regression dataset,
not a substitute for an external query distribution or ANN-Benchmarks. A 10k
budget measured 3.7 ms p50 but only 0.90 mean recall@10, which is why latency is
never reported without recall and why the adaptive planner uses a larger budget.

The filtered first search touches multiple blocks because file-node vectors are
spread through ID-ordered physical vector data. A future vector posting layout
can cluster filter/partition peers while preserving stable graph IDs.

The checkpoint work began at roughly 157–163 ms internal open time and
285–298 MB RSS. Hardware CRC32C, lazy properties, mapped CSR/vectors/sketches,
direct-mapped fixed records, and compressed label postings reduced a cached
Qwen open to about 21 ms. A stats-only run now reports roughly 92 MB maximum
resident pages but only 8.8 MB private footprint; mapped file pages can be
reclaimed by the OS. A post-checkpoint vector insertion appends to the F32
delta instead of hydrating the base.

Checkpoint construction also has a large-graph memory guardrail. Directly
importing wiki-Talk text (2.39M nodes / 5.02M edges) takes about 1.02 s with an
839 MB private peak; compacting an existing mapped file into v7 takes about
1.40 s with a 644 MB
private peak. The first in-memory metadata writer took about 1.03 s but peaked
at 1.78 GB private; direct buffered fixed-record/CSR serialization removed a
full metadata image and its node/edge byte copies. On the vector/property-rich
Qwen graph, v7 compaction takes about 3.6 s with roughly 218 MB private peak.
The destination remains failure-clean and is synced only after metadata,
vectors, checksums, and final header agree.

## LadybugDB directional baseline

The identical AST topology was exported to typed CSV and loaded into
LadybugDB 0.19.1 (formerly Kuzu) through its Python package. Ladybug stored the
non-vector graph in about 18 MB and measured:

| Operation | LadybugDB | Vecgra |
|---|---:|---:|
| Open | 11.77 ms | ~20 ms |
| `File-HAS_SYNTAX-Syntax`, limit 100 | 1.160 ms | ~0.003 ms |
| `Syntax-AST_CHILD-Syntax`, limit 100 | 12.215 ms | ~0.003 ms |
| Point degree/property query | 0.428 ms | ~0.000083 ms adjacency API |

This is not a clean head-to-head. Ladybug was called through Python and a
declarative query engine; Vecgra used a release Rust CLI and native operators.
Ladybug's file contains no vectors, and the schemas are not byte-equivalent.
The result says only that Ladybug opened faster while Vecgra's warm adjacency
operators were faster. A Rust binding and LSQB or LDBC benchmark would make a
fairer comparison.

## Reproduce

Reproduction commands:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release

target/release/vecgra bench-neighbors graph.vg NODE_ID 100000
target/release/vecgra bench-pattern graph.vg \
  'MATCH (a:File)-[e:HAS_SYNTAX]->(b:Syntax) RETURN a,e,b LIMIT 100' 1000
target/release/vecgra bench-search graph.vg 'regex configuration builder' \
  25 hash both
target/release/vecgra bench-search graph.vg 'regex configuration builder' \
  25 hash nodes File
target/release/vecgra bench-ann graph.vg 100 20000 both
target/release/vecgra plan-search graph.vg both
```

Vector recall is always reported against the exact engine; a latency number
without recall is not accepted as an optimization.

## External held-out vector datasets

### Partition-tier design probes (not shipped paths)

Before adding another durable index section, two storage-free coarse shortcuts
were compared directly with the then-current equal-weight 512-bit selector on
200 held-out VIBE queries (`k=10`, 20k exact reranks):

| Coarse method | Median full signatures inspected | Official recall | p50 |
|---|---:|---:|---:|
| Previous equal-weight sequential 512-bit scan | 1,000,000 | 0.990 | 5.49--5.91 ms |
| 8-table 16-bit multi-probe, radius 3 | 188,134 | 0.790 | 6.94 ms |
| 8-table 16-bit multi-probe, radius 4 | 479,609 | 0.960 | 12.51 ms |
| 256-bit pass, 100k survivors, then 512-bit | 100,000 | 0.972 | 7.96 ms |

The reduced comparison count did not translate to latency because bucket or
survivor gathers destroyed sequential locality, and both shortcuts discarded
too many good candidates. Their implementations were removed after the
measurement.

The retained query-confidence scorer described below supersedes the first row:
it reaches 0.9945 recall on the same hard 200-query prefix with only 10k exact
reranks and 4.99 ms p50, without changing the persisted index.

An external single-thread Faiss IVFFlat probe is retained as a reproducible
design tool rather than a product comparison. On the same million vectors,
4,096 centroids and 512 probes inspected about 117,918 rows per query, reached
0.998 official recall@10, and measured 1.76 ms/query inside Faiss. At 256 probes
it inspected 58,573 rows and reached only 0.989. A native partition tier would
need similarly contiguous vector, sketch, and filter rows. Adding an in-memory
Faiss dependency would not fix the current file layout.

### VIBE/Yandex: one million modern multimodal vectors

The official [VIBE](https://github.com/vector-index-bench/vibe) Yandex-200
cosine dataset contains 1,000,000 image vectors, 1,000 held-out text queries,
dimension 200, and official top-100 neighbors. It is a better frontier guardrail
than queries synthesized from stored vectors. The current v7 build takes 5.45
s, uses about 203 MB peak RSS, and produces a 545,000,992-byte file. The v6
writer took 5.15 s and about 537 MB: direct buffered record serialization plus
streaming the already-built sketch signatures cut construction memory by about
62% without changing a byte of the resulting vector representation.
Both are about 13 MB larger than v5 because owner/kind/label columns are durable
and directly mapped; v7 adds only 272 bytes on this property-free dataset.

On all 1,000 held-out queries at `k=10` and 12,000 exact reranks:

| Metric | Result |
|---|---:|
| Approx recall vs official | 0.9967 |
| Approx recall vs exact F16/F32 engine | 0.9975 |
| Exact recall vs official F32 truth | 0.9992 |
| Approx p50 / p95, compressed mode | 5.419 / 5.659 ms |
| Exact p50 / p95, compressed F16 in the same loaded session | 21.459 / 23.522 ms |
| Exact p50 / p95, explicit hot F32 | 16.269 / 16.406 ms |
| Full F32 cache / compressed-run private peak | 800.0 / 9.8 MB |
| Stats-only private footprint, v7 / v5 | 2.7 / 15.8 MB |
| Eager `vecgra check`, 400 MB vector column | 85.3 ms / 2.9 MB private peak |

The earlier 256-bit sketch reached only 0.9585 recall at a 20k budget; widening
the projection was a measured quality change, not cosmetic metadata. Equal
weight Hamming on the current 512 bits subsequently reached 0.9938 official
recall with 20k reranks at 5.963 ms p50. The new scorer retains the real rotated
query projection long enough to mark above-mean-confidence bits. A mismatch on
those bits receives one additional unit of distance. This query-asymmetric
signal cuts the necessary exact budget by 40% while improving both quality
and latency; stored signatures and the 545 MB file are byte-for-byte unchanged.

The common one-vector path still keeps only a two-byte distance per row, finds
the exact cutoff through a small histogram, and emits selected rows in physical
order for local exact reranking. AArch64 computes base and confidence-weighted
popcounts together from each loaded 128-bit pair; other architectures use the
same deterministic portable score. The adaptive planner selects the 12k budget
only when all eligible elements own exactly one vector and this fast path
applies. Multivector, labelled, and filtered plans retain their separately
measured budgets.

The two exact rows make a deliberate memory policy visible. The previous
automatic policy synchronously decoded the whole base on the second full scan:
that query took about 217 ms, even though ordinary compressed scans took about
20 ms. Full F32 promotion is now explicit. It buys roughly 4.5 ms on a repeated
million-vector exact scan but owns 800 MB; compressed mode remains stable and
never surprises a query with that allocation. Approximate reranking keeps using
F16 even when the cache exists because its sparse physical-order gather is
faster from the smaller working set.

An F32-on-disk build provides a third point in the trade space. Its
945,001,376-byte file (versus 545,000,992 bytes for F16) scores directly from
the aligned mmap and uses no decoded heap cache. Across the same 1,000 queries,
exact p50/p95 was 17.909/18.229 ms and ANN was 8.299/8.606 ms, with about 10.9
MB private peak. Mapped resident pages reached roughly the file size but are
clean and OS-reclaimable. F16 remains the better default: it is smaller and its
sparse rerank is faster; F32 mapping is useful when exact latency matters and
an 800 MB owned cache is undesirable.

The same dataset now exercises graph-derived candidate sets. IDs divisible by
the filter stride form a deterministic compressed prefilter; recall is measured
against exact search within that set, never against unfiltered truth.

| Eligibility | Allowed | Rerank | Recall vs filtered exact | Approx p50 | Exact gather p50 | Adaptive choice |
|---:|---:|---:|---:|---:|---:|---|
| 50% | 500,000 | 20,000 | 0.9965 | 11.264 ms | 20.346 ms | sketch/rerank |
| 20% | 200,000 | 20,000 | 1.0000 | 13.057 ms | 15.744 ms | sketch/rerank |
| 10% | 100,000 | 20,000 | 1.0000 | 10.980 ms | 10.231 ms | exact |

The non-monotonic gather cost is real: unrestricted exact search streams a
contiguous vector column, whereas a sparse set gathers record/vector rows. The
planner therefore uses separately measured 64M-float contiguous, 32M-float
narrow-vector gather, and 12M-float 512+-D gather priors. ANN merge-scans the
ordered Roaring set, applying eligibility before candidate budgeting. Compact
F16 materially helps the gather shape: at 10%, exact is now both lossless and
faster than explicit ANN, so the adaptive choice is supported by latency as
well as recall risk.

Reproduction:

```sh
uv run --with h5py --with numpy scripts/hdf5_to_fbin.py \
  yandex-200-cosine.hdf5 /tmp/yandex
target/release/vecgra import-fbin /tmp/yandex.train.fbin /tmp/yandex.vg
target/release/vecgra bench-fbin /tmp/yandex.vg /tmp/yandex.test.fbin \
  /tmp/yandex.neighbors.ibin 1000 12000 10 compressed
target/release/vecgra bench-filtered-fbin /tmp/yandex.vg \
  /tmp/yandex.test.fbin 200 5 20000 10
```

### MoReVec typed numeric filters

The 2026 MoReVec Movies/medium workload contributes 99,560 real movie text
embeddings (768 dimensions), scalar metadata filters, 1,000 queries per
filter, and official filtered top-100 neighbors. The generic conversion keeps
vectors in fbin and emits typed JSONL metadata; the database loader contains no
MoReVec-specific schema path. The resulting v8 F16 database is 186,446,572
bytes. Direct construction took 1.60 s with about 94 MB maximum RSS / 60 MB
private peak on the development machine.

The following single-threaded results use 200 queries for the two smallest
subsets and 500 for the larger subsets. "Total" includes rebuilding the exact
ordered range set for every query as well as vector search. Official recall
below 1.0 is largely the expected F16/normalization difference: exact F16
itself is 0.9986–0.9995 on the affected rows.

| Inclusive filter | Selectivity | Allowed | Range p50 | Adaptive plan | Official recall@10 | Total p50 |
|---|---:|---:|---:|---|---:|---:|
| `avg_rating >= 9.6` | 0.52% | 519 | 0.012 ms | exact 519 | 0.9995 | 0.051 ms |
| `avg_rating >= 9.2` | 2.20% | 2,192 | 0.074 ms | exact 2,192 | 0.9995 | 0.239 ms |
| `avg_rating >= 8.5` | 10.76% | 10,712 | 0.624 ms | exact 10,712 | 0.9986 | 1.747 ms |
| `avg_rating >= 8.1` | 20.68% | 20,588 | 0.466 ms | sketch + 5,147 rerank | 0.9992 | 2.026 ms |

At the widest filter, isolated exact gather is about 2.86 ms p50 while the
5,147-candidate sketch/rerank is about 1.38 ms. The previous 32M-float-only
heuristic incorrectly retained exact search and measured roughly 4.1 ms total;
the dimension-aware crossover halves that path without sacrificing meaningful
recall. This directly exercises MoReVec's central systems claim: filtered ANN
needs an explicit cardinality-sensitive exact/approximate branch.

Reproduction:

```sh
uv run --with h5py --with numpy scripts/morevec_to_interchange.py \
  movies.hdf5 movies_filters.hdf5 /tmp/morevec
target/release/vecgra import-node-fbin /tmp/morevec/train.fbin \
  /tmp/morevec/metadata.jsonl /tmp/morevec.vg f16
target/release/vecgra bench-range-fbin /tmp/morevec.vg \
  /tmp/morevec/query-6.fbin /tmp/morevec/truth-6.ibin \
  avg_rating 8.1 500 5000 10
```

#### Directional DuckDB VSS comparison

As a same-machine embedded-system check, DuckDB 1.5.5 plus its official VSS
extension loaded the identical 99,560 F32 vectors and built a default cosine
HNSW index. One thread and 500 `avg_rating >= 8.1`, k=10 queries were used.
DuckDB's physical plan was an HNSW index scan followed by the rating filter.
It did not prefilter the vector search.

| Measurement | DuckDB VSS | Vecgra |
|---|---:|---:|
| Indexed file | 798,765,056 bytes | 186,446,572 bytes (F16) |
| Table + index build | 1.01 s + 57.45 s | 1.60 s total |
| Maximum RSS during build/benchmark process | 1.96 GB | 94 MB during build |
| Filtered adaptive/index p50 | 44.45 ms | 2.03 ms total |
| Official recall@10 | 0.2094 | 0.9992 |
| Minimum results returned | 0 | 10 |
| Forced exact p50 / recall | 104.43 ms / 0.9996 | 2.86 ms isolated / 0.9994 |

This does not compare the databases as a whole. DuckDB was called through
Python with a 768-value bound parameter, retained F32, and used extension
defaults. Vecgra ran inside Rust, stored F16, and used a purpose-built prefilter.
`EXPLAIN` shows the relevant difference: DuckDB filtered an unfiltered HNSW
result, while Vecgra fed an ordered metadata range into its exact or ANN choice.

### ANN-Benchmarks COCO text-to-image

The COCO-I2I/T2I angular dataset distributed by ANN-Benchmarks supplies 113,287
image vectors, 10,000 held-out text-query vectors, dimension 512, and official
top-100 neighbors. This distribution exposed a weakness that the derived Qwen
queries did not: the original 256-bit sketch reached only 0.683 / 0.812 / 0.897
official recall@10 at 5k / 10k / 20k exact-rerank candidates (100 queries).

Widening the deterministic rotated sketch to 512 bits produced the following
200-query sweep. Exact F16 search reached 0.993 official recall@10; the small
difference from the supplied F32 ground truth is quantization/normalization.

| Rerank candidates | Official recall@10 | Recall vs exact | Approx p50 | Exact p50 |
|---:|---:|---:|---:|---:|
| 5,000 | 0.9765 | 0.9830 | 1.474 ms | 5.156 ms |
| 10,000 | 0.9870 | 0.9940 | 2.387 ms | 5.126 ms |
| 20,000 | 0.9895 | 0.9965 | 3.716 ms | 5.097 ms |
| 30,000 | 0.9905 | 0.9975 | 4.594 ms | 5.188 ms |

The query-confidence score improves every measured budget and makes approximate
search faster than exact even at 30k candidates. Adaptive search nevertheless
remains exact through roughly 64 million scalar comparisons (about 131k vectors
at 512 dimensions or 262k at 256 dimensions), because this 113k-row dataset can
still return the exact engine's higher 0.993 official recall cheaply. Callers
who knowingly accept a lower recall target can request an explicit approximate
budget. The unchanged 512-bit direct build took 1.23 s and about 62 MiB peak RSS
on the development machine.

Reproduction:

```sh
uv run --with h5py --with numpy scripts/hdf5_to_fbin.py \
  coco-t2i-512-angular.hdf5 /tmp/coco
target/release/vecgra import-fbin /tmp/coco.train.fbin /tmp/coco.vg
target/release/vecgra bench-fbin /tmp/coco.vg /tmp/coco.test.fbin \
  /tmp/coco.neighbors.ibin 200 5000 10
```

## Official Graphalytics traversal dataset

The [LDBC Graphalytics wiki-Talk
dataset](https://ldbcouncil.org/benchmarks/graphalytics/datasets/) contains
2,394,385 vertices and 5,021,410 directed edges. The v5 file is 558 MB. Direct
bulk import took 1.26 s on the development machine. BFS from official source 2
reached 2,354,316 vertices with maximum distance 6 and matched every one of the
2,394,385 supplied reference values.

| Operation | Result |
|---|---:|
| Cached stats/open wall | 0.23 s |
| Stats-only private footprint with compressed postings | 2.7 MB |
| BFS p50 / p95, 10 iterations | 60.328 / 61.436 ms |
| BFS private peak footprint | about 199 MB |
| WCC p50 / p95, 3 iterations | 205.415 / 206.865 ms |
| WCC components / reference agreement | 2,555 / all 2.39M rows |
| PageRank p50 / p95, 5 iterations | 546.817 / 553.023 ms |
| PageRank rank sum / maximum absolute error | 1.000000000014 / 2.212e-17 |

The same source node exercises exact graph-range candidate materialization,
including both reached nodes and traversed edges in one compressed typed set:

| Outgoing radius | Reached nodes | Traversed edges | Expansion p50 |
|---:|---:|---:|---:|
| 1 | 110 | 110 | 0.004 ms |
| 2 | 35,230 | 46,993 | 4.401 ms |
| 3 | 1,458,427 | 2,761,554 | 108.276 ms |
| 6 | 2,354,315 | 4,949,279 | 194.651 ms |

Graph-range vector retrieval needs reachable nodes but not a materialized copy
of every relationship encountered along the way. The node-only range operator
returns the same node set and retains the same roughly 5.7 MB private footprint
on the widest case, while avoiding the edge posting result:

| Radius | Full node+edge expansion p50 | Node-only range p50 |
|---:|---:|---:|
| 2 | 4.427 ms | 3.493 ms |
| 3 | 109.613 ms | 82.841 ms |
| 6 | 197.280 ms | 151.587 ms |

On the ripgrep hash-embedding graph, `range-text` from `build.rs` over six
outgoing hops produced 131 node candidates (262 vector facets) and correctly
selected the exact gathered path. The run covers graph-range construction,
node filtering, adaptive planning, and vector ranking. Hash-embedding scores do
not measure semantic quality.

The six-hop run used about 5.7 MB private memory (mapped resident pages are
excluded). Radius two is a practical exact vector prefilter; the power-law
explosion at radius three is evidence for an adaptive broad-range/postfilter or
memoized labeling path, not for applying the same plan at every radius.

Mapped-record column projection reduced BFS from 133.0 ms to 60.3 ms: the
inner loop reads only edge endpoint fields and entirely bypasses empty WAL
maps. Compressed label postings cut roughly another 58 MB from the BFS private
footprint without changing traversal latency. Undirected CSR suppresses the
incoming copy of self-loops by endpoint comparison, so WCC does not allocate a
dedup hash set per visited vertex; parallel self-edges still appear exactly
once each. PageRank uses the Graphalytics parameters (damping 0.85, ten
iterations, dangling-mass redistribution) and validates all 2.39 million
supplied floating-point ranks. Its measured private peak footprint was about
81 MB; the operating system's mapped-file RSS is intentionally not reported as
owned heap.

```sh
target/release/vecgra import-graphalytics wiki-Talk.v wiki-Talk.e wiki-talk.vg
target/release/vecgra bench-bfs wiki-talk.vg 2 wiki-Talk-BFS 10
target/release/vecgra bench-wcc wiki-talk.vg wiki-Talk-WCC 3
target/release/vecgra bench-pagerank wiki-talk.vg wiki-Talk-PR 5
target/release/vecgra bench-expand wiki-talk.vg 2 2 10 out
```

## Neo4j comparison

### Pass 1: Neo4j 2026.06 Community embedded

The first competitor baseline uses Neo4j Community 2026.06.0 embedded in the
benchmark JVM on Java 25, with no Bolt, HTTP, Docker,
or client serialization. Lucene's Java Vector API was active. The JVM used a
2--4 GiB ZGC heap and Neo4j used a 1 GiB page cache. Both engines ran warm,
single-threaded queries on the same 14-core Apple M4 Pro / 24 GB machine on
2026-08-18. The checked-in benchmark and complete reproduction commands are in
[`benchmarks/neo4j`](../benchmarks/neo4j/README.md).

The Community embedded artifact stores embeddings as supported
`LIST<FLOAT>` properties. Neo4j documents its contiguous dedicated `VECTOR`
property type as an Enterprise storage feature. The 2026.06 vector index
provider accepts both, so this does not change the Lucene search path, but it
is relevant to the whole-store size comparison.

This comparison does not answer "which database is better".
Neo4j supports far more mutable, transactional, server, language, and tooling
workloads. Vecgra exploits an immutable indexed base and direct native API.
Conversely, embedding Neo4j removes network and driver overhead. Vector search
uses the supported Cypher 25 `SEARCH` path with one transaction per query; BFS
uses Neo4j's embedded relationship API because Community does not ship the GDS
BFS implementation. Neo4j's offline `neo4j-admin` bulk importer and GDS were
not measured.

#### Unfiltered VIBE/Yandex vectors

Neo4j's default scalar-quantized HNSW import stored all 1,000,000 200-D vectors
in 6.461 s and populated the index in 41.708 s. The process peaked at about
5.36 GB and the resulting store was 3,107,186,279 bytes. Vecgra's current
v7 build takes 5.45 s, peaks at about 203 MB, and is 545,000,992 bytes. The
Neo4j build result is its normal embedded transactional loader, not its offline
admin importer.

The critical comparison is recall-matched query latency, not either engine's
default knob:

| Engine / index configuration | Recall@10 | p50 | p95 | Store |
|---|---:|---:|---:|---:|
| Vecgra, confidence-weighted 512-bit + 12k F16 rerank | 0.9967 | 5.419 ms | 5.659 ms | 545.0 MB |
| Neo4j scalar, default expansion 1.5 | 0.8301 | 7.959 ms | 12.156 ms | 3.107 GB |
| Neo4j scalar, expansion 100 | 0.9892 | 15.722 ms | 18.717 ms | 3.108 GB |
| Neo4j scalar, expansion 200 | 0.9958 | 20.502 ms | 23.492 ms | 3.108 GB |
| Neo4j unquantized, M=16 / efConstruction=100 / expansion 200 | 0.9961 | 14.076 ms | 16.376 ms | 2.893 GB |
| Neo4j unquantized, M=32 / efConstruction=400 / expansion 100 | 0.9967 | 15.472 ms | 17.656 ms | 2.911 GB |

Neo4j's best measured recall-matched point is therefore about 2.86x slower at
p50 and 5.3x larger on disk, despite returning slightly higher recall. Raising
HNSW construction quality took 121.959 s for the index rebuild and did not
improve latency: its extra connectivity crossed the recall target at a lower
search expansion, but traversal cost rose. At one million vectors, Vecgra's
contiguous coarse scan and rerank was faster. That does not predict its result
against HNSW or partitioned indexes at other scales.

#### Native filtered MoReVec vectors

Neo4j used its 2026.06 vector provider's indexed additional-property filter,
not a postfilter. Its best measured configuration here was unquantized HNSW
with M=16, efConstruction=100, and search expansion 6. Vecgra used the
same typed `avg_rating` thresholds and official truth files. The Vecgra
latency is the complete adaptive call, including numeric-posting lookup;
Neo4j's includes its Cypher transaction and `SEARCH` execution.

| Threshold | Eligible | Vecgra recall / p50 | Neo4j recall / p50 | Vecgra speedup |
|---|---:|---:|---:|---:|
| `avg_rating >= 9.6` | 519 | 0.9994 / 0.056 ms | 0.9998 / 0.542 ms | 9.68x |
| `avg_rating >= 9.2` | 2,192 | 0.9994 / 0.245 ms | 1.0000 / 0.753 ms | 3.07x |
| `avg_rating >= 8.5` | 10,712 | 0.9986 / 1.351 ms | 1.0000 / 4.277 ms | 3.17x |
| `avg_rating >= 8.1` | 20,588 | 0.9992 / 2.062 ms | 0.9998 / 6.693 ms | 3.25x |

The 99,560 x 768-D Vecgra file is 186,446,572 bytes. The final
unquantized Neo4j store was about 1.035 GB, 5.55x larger. A fresh Vecgra
build took 1.15 s and 99 MB maximum RSS. Neo4j's initial default-scalar load
took 2.241 s for data plus 8.494 s for index population and peaked at about
4.25 GB maximum RSS; the scalar store was 1.111 GB. Query and build settings
are separated here because testing Neo4j fairly found that unquantized search
was both faster and smaller for this workload.

#### Graphalytics wiki-Talk BFS

Both engines imported the same dense Graphalytics text files, then BFS from
official source 2 validated every one of the 2,394,385 supplied distances.
Vecgra's fresh rerun includes its complete direct checkpoint builder;
Neo4j's uses batched embedded transactions and its normal relationship store.

| Metric | Vecgra | Neo4j | Ratio |
|---|---:|---:|---:|
| Import wall time | 2.03 s | 34.674 s | Vecgra 17.1x faster |
| Store bytes | 584,617,440 | 1,289,647,007 | Vecgra 2.21x smaller |
| BFS p50 | 69.221 ms | 946.753 ms | Vecgra 13.7x faster |
| BFS p95 | 77.829 ms | 1,004.580 ms | Vecgra 12.9x faster |
| Reached / maximum distance | 2,354,316 / 6 | 2,354,316 / 6 | identical |

The traversal result is the clearest payoff from persisted bidirectional CSR:
the hot loop projects compact endpoint columns, while Neo4j follows general
relationship records and entity proxies. It is also the least transferable
result to an update-heavy workload: Vecgra pays for this layout at
checkpoint construction and has not yet demonstrated Neo4j-like sustained
mutation behavior.

### Pass 2: Neo4j Enterprise 2026.07 Desktop audit

The second pass targets the strongest paths available in the user's local
Neo4j Desktop installation rather than extrapolating from Community Edition.
The server is Neo4j Enterprise 2026.07.0 on Java 21 with a 2--4 GiB heap, a
1 GiB page cache, the incubating Java Vector API enabled, and the
`vector-2026.07` provider. Vector properties are the dedicated contiguous
`VECTOR<FLOAT32 NOT NULL>(d) NOT NULL` type. The Java 25 benchmark client uses
the official driver over loopback Bolt; connection setup is reported but
excluded from query samples. GDS 2026.07 and `neo4j-admin` are the versions
installed by Desktop.

The second pass uses the normal product path and produces smaller margins than
the embedded Community run.
The generated databases were isolated from the default `neo4j` database.

#### VIBE/Yandex with native vectors

All one million 200-D vectors use Neo4j's native vector property and an
unquantized HNSW index with M=16 and efConstruction=100. Expansion 160 cleared
Vecgra's previous 0.9938 result. After query-confidence weighting, even
Neo4j's highest measured expansion-200 point remains below Vecgra's 0.9967
official recall. An auto-commit query is the normal product path; the
one-transaction rows are lower bounds that reuse one transaction for all 1,000
requests.

| Engine / path | Recall@10 | p50 | p95 | Live database bytes |
|---|---:|---:|---:|---:|
| Vecgra, confidence-weighted 512-bit + 12k F16 rerank | 0.9967 | 5.419 ms | 5.659 ms | 545,000,992 |
| Neo4j Enterprise, expansion 160, one transaction | 0.9944 | 16.137 ms | 18.657 ms | 1,788,661,939 |
| Neo4j Enterprise, expansion 160, auto-commit | 0.9944 | 17.273 ms | 21.017 ms | 1,788,661,939 |
| Neo4j Enterprise, expansion 200, one transaction | 0.9961 | 17.546 ms | 20.315 ms | 1,788,661,939 |
| Neo4j Enterprise, expansion 200, auto-commit | 0.9961 | 18.790 ms | 21.693 ms | 1,788,661,939 |

At lower recall, the expansion-160 product path is 3.19x slower. At Neo4j's
highest-recall expansion-200 point, the product path is 3.47x slower and even
the reused-transaction lower bound is 3.24x slower. The
database-directory ratio is 3.28x, substantially
better for Neo4j than the Community result but still not close to the
single-file F16 layout. Neo4j's directory breaks down into about 809.1 MB of
native vector-property storage, 850.7 MB of vector schema index, and 128 MB of
primary block records plus small metadata. Its retained transaction logs add
859.5 MB and are disclosed separately rather than folded into the headline.

Loading over Bolt took 76.489 s and expansion-200 index population took
40.019 s. That is a real ingestion path, but it is not compared with
Vecgra's offline build as an optimized-import claim because Neo4j's admin
importer also accepts vector columns and was not exercised on this 200-D source.

#### MoReVec native prefiltered search

Neo4j's vector index includes `avg_rating` as an additional indexed property,
so these are native prefilters rather than postfilters. To avoid favoring
Vecgra through transient server load, the table uses Neo4j's best measured
auto-commit pass; a later repetition was slower at every threshold.

| Threshold | Eligible | Vecgra recall / p50 | Neo4j Enterprise recall / p50 | Vecgra speedup |
|---|---:|---:|---:|---:|
| `avg_rating >= 9.6` | 519 | 0.9994 / 0.056 ms | 0.9998 / 1.463 ms | 26.1x |
| `avg_rating >= 9.2` | 2,192 | 0.9994 / 0.245 ms | 1.0000 / 1.788 ms | 7.30x |
| `avg_rating >= 8.5` | 10,712 | 0.9986 / 1.351 ms | 1.0000 / 6.963 ms | 5.15x |
| `avg_rating >= 8.1` | 20,588 | 0.9992 / 2.062 ms | 0.9994 / 9.805 ms | 4.75x |

Neo4j's live database directory is 630,625,939 bytes versus Vecgra's
186,446,572-byte file, a 3.38x ratio. Retained Neo4j transaction logs add
312,344,840 bytes. The narrow-filter lead is not evidence that HNSW is bad; it
is evidence that an ordered scalar posting feeding a cardinality-sensitive
exact/ANN branch is unusually effective for vector-native graph workloads.

#### `neo4j-admin` import and GDS BFS

The official wiki-Talk files were imported with `neo4j-admin database import
full`, Enterprise block format, integer IDs, high-parallel IO, and all 14
available importer threads while the server was stopped. The GDS projection
contains the same 2,394,385 nodes and 5,021,410 directed relationships. It took
587 ms to project and occupies 79,406,959 bytes (reported as 75 MiB) of
additional in-memory graph state.

| Metric | Vecgra | Neo4j Enterprise best path | Result |
|---|---:|---:|---:|
| Import engine / complete process wall | 2.21 / 2.21 s | 4.778 / 6.27 s | Vecgra 2.16x / 2.84x faster |
| Import maximum RSS | 840 MB | 2.65 GB | Vecgra 3.16x lower |
| Durable graph bytes | 584,617,440 | 423,290,781 | Neo4j 27.6% smaller |
| BFS p50, concurrency 1 | 80.286 ms | 236 ms server compute | Vecgra 2.94x faster |
| BFS p50, GDS concurrency 4 | 80.286 ms (one thread) | 101 ms server / 111 ms client | Vecgra 1.26x / 1.38x faster |
| Reached nodes | 2,354,316 | 2,354,316 | identical |

The four-thread BFS and Vecgra latency rows use alternating warm batches
from the same audit session; the one-thread GDS row is its best 20-run warm
batch. Vecgra's earlier unloaded 69.221 ms result is retained in the
first-pass table but is not used for these ratios. Vecgra validates every one of the
2,394,385 official distance values and maximum distance 6. GDS BFS exposes the
visited node list rather than the distance vector, so this pass validates its
reachable-set cardinality; the first-pass Neo4j implementation separately
validated every distance. The installed unlicensed GDS build caps concurrency
at four, so the four-thread row is the strongest executable result from this
Desktop instance, not a claim about a separately licensed higher-concurrency
GDS deployment.

#### Revised conclusions

With native Enterprise vector storage and matched recall, Vecgra's vector path
measured about 3.3x faster and used about 3.3x less disk, excluding Neo4j's
transaction logs. The first Community run overstated both margins because it
stored vectors as lists. `neo4j-admin` and GDS also reduce the import and BFS
margins to roughly 2--3x for import and 1.26x against four-thread GDS. Neo4j
uses less disk for the graph-only dataset.

The measured design case is narrow. Sequential binary coarse search plus exact
rerank works at one million vectors, typed postings belong inside the vector
planner, and persisted CSR can compete with a projected analytics graph. Larger
or colder datasets still need a contiguous partition tier and an exact
recent-write delta. Both Neo4j passes remain in the repository so later changes
can be compared against the same product overhead and recall.
