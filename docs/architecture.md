# Architecture

## Mental model

Vecgra has two durable layers in one file:

1. A checkpoint is the compact, read-optimized state of the graph. Most of it
   is used directly through a read-only memory map; opening a database does not
   deserialize every property or vector.
2. A transaction log follows the checkpoint. Commits append complete,
   checksummed frames. On open, valid frames are replayed over the checkpoint.

This is similar to a table snapshot followed by a WAL, except the checkpoint is
laid out for graph traversal and vector search. Compaction writes a new
checkpoint without changing stable element IDs.

The important storage terms are:

- **Column-projected records:** source ID, target ID, label, and vector offsets
  live in fixed-width mapped records. Hot operators read only the fields they
  require without allocating a Rust object for every graph element.
- **Memory mapping:** the OS makes file pages addressable as memory and loads
  only pages that are touched. This enables fast open and lets cold data leave
  RAM naturally.
- **CSR adjacency:** outgoing or incoming edge IDs for each node occupy one
  contiguous array. A second offsets array says where each node's slice begins
  and ends. A degree-one traversal is two offset reads plus one edge read, with
  no pointer chasing.
- **Delta:** immutable checkpoint data stays mapped. New vectors and adjacency
  changes go to small in-memory structures and the durable log. Compaction
  folds them into a new base.

## Logical model and invariants

- The database is a labelled property multigraph. Parallel edges and self-edges
  are valid and every edge is independently addressable.
- Node and edge IDs are stable unsigned 64-bit values.
- Properties are sorted `(u32 key, typed value)` records. The symbol dictionary
  interns label names and property keys.
- One database has one vector dimension, similarity metric, and embedding
  space. Nodes and edges may own zero, one, or multiple vectors.
- Cosine vectors are normalized once on entry. Search is therefore a dot
  product for both cosine and dot-product databases.
- Forward and reverse adjacency agree with the current edge record. Checkpoint
  CSR is immutable; updates are overlaid and stale base positions are rejected
  by endpoint validation.
- A commit is visible in memory only after its complete frame has been appended
  and, when configured, synced.

## Container format v8

```text
+---------------------------+ offset 0
| 64-byte header            | magic, version, dimension, metric,
|                           | encoding, checkpoint ranges, CRC32C
+---------------------------+
| columnar checkpoint       | dictionary + vector-block checksums
|                           | fixed node records
|                           | fixed edge records
|                           | encoded property blob
|                           | outgoing CSR offsets + edge IDs
|                           | incoming CSR offsets + edge IDs
|                           | randomized binary vector sketches
|                           | typed equality-fingerprint postings
|                           | ordered typed numeric postings
|                           | metadata CRC32C
+---------------------------+
| contiguous vector section | F16 or F32 values + global CRC32C
+---------------------------+
| transaction frame 1       | length, tx id, operations, CRC32C, tail magic
+---------------------------+
| transaction frame 2 ...   |
+---------------------------+
```

The reader remains compatible with the earlier transaction-only v1/v2 format,
operation-stream v3 checkpoints, non-indexed columnar v4 checkpoints, and both
256/512-bit v5/v6 indexed checkpoints, and v7 equality-indexed checkpoints.
Compaction and the direct bulk loader emit v8.

Node records are 48 bytes and edge records are 64 bytes. They are consumed
directly from the mapping on little-endian machines and decoded on big-endian
machines; point/traversal/vector paths project only existence, endpoints, or
vector-span fields as appropriate. They point into the property and vector
sections. Properties remain encoded until an API call or predicate needs them.
The compact mapped descriptor intentionally has no per-record decode cache:
avoiding millions of `OnceLock` values matters more than caching a cold
property. CSR arrays are mapped under the same endian policy. Every v8 section
starts on a 64-byte cache-line boundary, so adding a format descriptor cannot
shift all later hot columns within cache lines.

Exact per-label ID sets use compressed 64-bit Roaring postings. They are kept
current across updates/deletes and double as query candidate sets, replacing
stale posting histories plus per-query hash deduplication. Dense label runs cost
very little memory; wiki-Talk's stats-only private footprint is about 2.7 MB.

Metadata has one checksum because opening necessarily touches its compact
structural fields. Vectors use 4 MiB block checksums in addition to a global
checksum. A selective query verifies only blocks containing its candidates;
full scans, hydration, and compaction eventually verify every block. Older
checkpoints without the block table fall back to the global checksum.
`verify_integrity`/`vecgra check` eagerly touches every remaining vector block
without decoding it; all metadata, records, indexes, CSR, and WAL frames have
already been structurally and checksum-validated by `Database::open`.

The current writer stores up to 512 randomized sign bits per vector (bounded by
the transform dimension). The reader also accepts the earlier 256-bit v5
variant, so widening the coarse representation did not strand existing files.
Sketches and compact parallel owner-ID, owner-kind, and label columns all map
directly in v6; v5 reconstructs the owner columns for compatibility. Vectors
belonging to one element remain contiguous. When owner IDs are monotonic within
their node/edge kind, filters merge-scan compressed candidate sets; unusual
physical orders automatically use exact bitmap membership instead.

v8 indexes every scalar property, but gives numbers a physically different
access path. Nulls, booleans, strings, byte strings, and graph references use
the v7 typed 32-bit fingerprint posting (12 or 16 bytes); the encoded property
is compared exactly, so collisions affect work rather than correctness.
Integers and non-NaN floats instead use a collision-free, order-preserving
64-bit key in a 20- or 24-byte posting. Signed integer order and IEEE float
order are preserved independently, including infinities and canonical signed
zero. Numeric equality is a one-value ordered range, while inclusive,
exclusive, and unbounded range predicates use two binary searches. NaN remains
eligible only for ordinary typed equality semantics because it has no useful
range order. Storing each number in the ordered table only—not both tables—cuts
3.58 MB from the 99,560-row MoReVec checkpoint.

The smallest property range competes with label cardinality at query time. WAL
overlays are scanned exactly, making inserts and changed properties visible
without synchronous base-index maintenance. `element_filter_plan` and
`numeric_range_plan` expose the chosen path and conservative candidate bound
before materialization.

For one-vector-per-element, unfiltered search ranks the signature stream
without touching owner columns until the bounded winners are known. On the
million-vector VIBE corpus this simultaneously lowered stats-only private
footprint from 15.8 MB to 2.7 MB and improved ANN latency.

## Vector tiers

The default disk representation is F16, while newly written vectors and the hot
compute representation are F32:

```text
mapped F16 checkpoint  ->  exact compressed scan for cold/sparse work
                       ->  sequential binary-sketch scan + bounded exact rerank
                       ->  optional explicit full F32 cache for hot exact scans
small F32 delta        ->  immediately searchable, no base rewrite
```

Promotion is explicit rather than a hidden query side effect. `warm_vector_cache`
returns the owned allocation size, `vector_cache_bytes` exposes current use,
and all ordinary searches stay compressed until the caller opts in. This avoids
a large synchronous latency spike and makes the memory/throughput tradeoff
controllable. Sparse ANN reranking deliberately continues to use F16 even when
the cache exists because its smaller random-gather working set measures faster.
A full search scores the base and delta in one logical offset space. Updating
one vector never hydrates the base.

The exact engine uses runtime AVX2 on x86/x86-64, baseline NEON on AArch64, and
a scalar fallback. F16 reranking uses ARM FP16 or x86 F16C when available, with
a portable table-conversion kernel otherwise. Exact search is intentional: it
is competitive for local graphs in the hundreds of thousands of elements and
is the truth set for approximate recall measurement.

F32 checkpoints are genuinely zero-copy on little-endian targets. Full scans
verify the mapped vector blocks once and give the SIMD loop one aligned native
slice; sparse access verifies only touched blocks. Big-endian readers retain
portable decoding. This avoids the former redundant full-size heap copy.

The approximate base is an immutable, cache-friendly coarse index. Two
deterministic randomized Hadamard rotations produce a compact angular sketch.
For the common one-vector-per-element path, query-time scoring retains the real
rotated query long enough to mark above-mean-confidence coordinates. Every bit
uses ordinary Hamming distance and mismatches on that confidence mask receive
one additional unit. This asymmetric signal improves the ranking without
adding a byte to stored signatures. The bounded vector set is then reranked
exactly by the ordinary SIMD scorer. The recent F32 WAL delta is always scanned
exhaustively, so a commit never waits for index maintenance.

The AArch64 kernel computes base and confidence-weighted byte popcounts
together from each loaded 128-bit pair; ordinary x86 builds use scalar POPCNT
and other targets retain a portable fallback. Format and ranking are identical
across machines. Bounded two-byte distances feed a 1025-bin histogram: an exact
cutoff replaces million-row tuple selection, and qualifying ordinals remain in
physical order so reranking gathers monotonically through the mapped vector
column. Filtered, labelled, and multivector paths retain their whole-element
equal-weight sketch score until separately benchmarked confidence variants
justify a change.
The planner estimates exact work as eligible vectors × dimension, keeps
small/selective scans exact, and assigns broad scans a bounded rerank budget.
When every eligible element has one vector and the confidence-weighted fast
path applies, the broad budget starts near 1% with a 12k floor; other shapes keep
the previous 20k floor.
Candidate-set policy has separately measured high-dimensional and ordinary
vector-width crossovers; this avoids treating 20k × 768 and 100k × 200 as the
same gather shape.
Callers can request exact, explicit-budget, or adaptive execution and inspect
the adaptive plan.

## Query execution

Current native operators include:

- exact vector scan over nodes, edges, or both;
- persisted sketch candidate generation with exact reranking;
- weighted late interaction across query and element multivectors, with both
  exact and whole-element sketch/rerank execution;
- adaptive exact/approximate selection and exact label prefiltering;
- typed compressed `ElementSet` algebra, exact bounded multi-hop set expansion,
  and graph-first prefiltering for exact, sketch/rerank, and multivector MaxSim
  search;
- first-class graph-range nearest-neighbor search: node-only bounded expansion,
  optional label/property intersection, then an inspectable adaptive vector
  plan whose budget is spent only inside the reachable range;
- exact label plus arbitrary typed-property predicates materialized directly
  into `ElementSet`, comparing mapped encodings without record hydration;
- exact integer/float range predicates backed by ordered mapped postings and
  directly consumable by vector and graph set algebra;
- fused filtered vector execution that costs one conjunctive equality filter
  and any number of numeric ranges, evaluates the smallest conservative access
  path first, intersects compressed sets, and exposes every scalar plan plus
  the final exact/sketch vector plan in `FilteredVectorSearchResult`;
- outgoing, incoming, or undirected adjacency expansion;
- exact unweighted shortest-path search with direction and relationship-label
  filters, stable-ID tie ordering, a hop bound, a frontier-expansion budget,
  and an inspectable adaptive choice between one-hop BFS and frontier-balanced
  bidirectional BFS;
- one-hop labelled/property pattern matching with an inspectable costed choice
  between an edge/edge-label scan, start-node adjacency, and reverse adjacency
  from the end-node predicate;
- semantic one-hop matching that jointly ranks the seed node, relationship,
  and destination node;
- semantic seed search followed by bounded best-first traversal.

Semantic traversal scores both the relationship and destination node. It has
separate node/edge weights, path decay, hop and degree penalties, direction and
label filters, a hop bound, and an expansion budget. This is a database
operator, not an agent framework feature.

`shortest_path` is the exact evidence-chain primitive beneath higher-level
path ranking. One-hop work stays on a simple BFS. Deeper work searches from
both endpoints, reverses traversal direction on the destination side, and
costs each complete frontier by node expansions plus a conservative adjacency
read upper bound. The estimate is an O(frontier) view over mapped CSR and WAL
overlay lengths; correctness does not depend on its accuracy, and relationship
filters remain exact during expansion. Because the shortest-path proof advances
only after a complete layer, the planner first prefers the only frontier that
fits the remaining expansion budget when exactly one does; otherwise the
lower-cost frontier advances. This preserves the same global hop and expansion
limits without spending a tight budget on a layer that cannot complete. A path
is returned only after the sum of completed search depths proves it shortest.
The result exposes `ShortestPathStrategy`, ordered node and relationship IDs,
visited-node and examined-relationship counts, plus start-side, end-side, and
total expanded-node counts. The two endpoint counts always sum to the total;
one-sided BFS reports zero end-side work. A typed termination reason preserves
completeness. A zero-hop self path succeeds even with zero work budget; missing
endpoints remain typed `NotFound` errors. Stable frontier and adjacency ordering
plus deterministic meeting-path selection make equal-length and parallel-edge
choices reproducible across WAL state and compacted mapped checkpoints.

The cost regression fixture gives one origin 256 dead ends beside a five-hop
spine. Cardinality-only bidirectional search examined 261 relationships; the
work-costed plan starts at the cheap destination frontier and examines five,
while returning the same path in five expansions.

A separate tight-budget fixture leaves two nodes in the forward layer and one
in the reverse layer with one expansion remaining. Completing the reverse layer
finds and proves the two-hop path at exactly the two-expansion limit; partially
advancing the nominally cheaper forward layer would have returned an
inconclusive `ExpansionLimit` instead.

On the 110,303-node repository fixture, the same seven-hop `0 → 100000`
either-direction query changed from 15,293 expanded nodes and 38,219 examined
relationships to 62 and 188 respectively, with the identical evidence chain.
A current run exposes that 62-node plan as 27 expansions from the requested
origin and 35 from the destination, rather than hiding the adaptive split.
A single local debug smoke changed from 23.94 ms to 2.02 ms; the work counters
are deterministic, while those timings are not a distribution or
cross-machine claim.

`vecgra-studio-core::evidence_path_database` is the owned presentation
boundary above that primitive. It opens a read-only view, resolves an optional
relationship label, executes the bounded exact path, and hydrates node and
relationship properties, vector counts, titles, labels,
stored-versus-traversed direction, physical strategy, termination, and work
diagnostics before releasing the read guard. Native clients can therefore run
the complete unit on a background executor without leaking mapped records or
database locks into UI state. It deliberately preserves `ExpansionLimit` as an
incomplete result rather than collapsing it into absence.

`one_hop_plan` uses the same conservative label/property posting cardinalities
as scalar filtering, estimates directional adjacency work, and exposes the
selected physical strategy before execution. This is the first general pattern
cost boundary; the Cypher-compatible parser remains a convenience surface for
one-hop patterns. The internal plan model will remain vector-aware rather than
forcing all features through Cypher syntax.

## Recovery and compaction

Each transaction frame contains its byte length, transaction ID, operation
count, payload, CRC32C, and a tail marker. Recovery stops at a partial final
frame and truncates that tail before the next commit. A checksum or marker
mismatch inside a supposedly complete frame is corruption, not a torn write.

Compaction is non-destructive: it requires a new destination path, writes a new
header/checkpoint, flushes and syncs it, and removes the partial destination on
failure. Vector output is spooled, checksummed incrementally, and copied raw
when source and destination encodings match. This avoids decoding F16 merely to
encode the same bytes again.

For initial construction, `BulkLoader` writes the indexed checkpoint directly.
It retains compact record/property/sketch state but spools vector bytes,
avoiding the full mutable graph, F32 vector copy, and temporary transaction log.
At finalization, fixed records, CSR columns, property postings, sketch owner
columns, and the already-built signature rows stream through a 1 MiB writer
directly into the new destination; CRC32C is accumulated during that one pass,
then vectors are copied and the header is patched and synced. A failed partial
destination is removed. On the million-vector VIBE corpus this keeps peak build
RSS to about 203 MB on the million-vector VIBE build. The v8 writer adds the
ordered numeric section without changing the streamed vector/sketch path.

## Known gaps

- The current coarse tier is an in-file binary sketch, not yet a partitioned
  disk index. Larger-than-memory datasets still need contiguous partitions and
  local split/merge maintenance.
- Equality and same-typed numeric range predicates have automatic postings;
  prefix, full-text, and compound covering property indexes are not yet present.
- The planner does not yet compose arbitrary multi-hop patterns into one costed
  plan; graph-range, semantic one-hop, and bounded path operators are currently
  specialized plans.
- Compressed label postings are rebuilt from mapped record headers at open;
  persisting them would reduce CPU open work on very large non-vector graphs.
- Concurrency is a single writer with shared readers; there is no MVCC snapshot
  API yet.
