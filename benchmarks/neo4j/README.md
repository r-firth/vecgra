# Neo4j comparison harness

This directory contains two reproducible Neo4j baselines for VectorGraph's
public benchmark corpora:

- a same-process Neo4j Community 2026.06 baseline that deliberately avoids
  Bolt and client serialization; and
- a loopback Neo4j Enterprise 2026.07/Desktop baseline using dedicated native
  vector properties, Cypher 25 `SEARCH`, `neo4j-admin`, and GDS.

The harness requires Java 25. Its Gradle wrapper downloads all other Java
dependencies:

```sh
JAVA_HOME=/path/to/jdk-25 ./gradlew installDist
export NEO4J_BENCH=build/install/vectorgraph-neo4j-benchmark/bin/vectorgraph-neo4j-benchmark
```

The embedded application has a 1 GiB Neo4j page cache, a 2--4 GiB ZGC heap, and enables
Lucene's incubating Java Vector API. Commands fail rather than append to a
non-empty import directory. This is Community Edition, so embeddings are
stored as supported `LIST<FLOAT>` properties; Neo4j's dedicated `VECTOR`
property storage is an Enterprise feature. Both property forms use the same
2026.06 vector-index provider.

## Embedded Community baseline

Inputs use the little-endian `fbin`/`ibin` interchange files produced by the
scripts in `../../scripts`.

```sh
# VIBE/Yandex: one million unfiltered vectors.
$NEO4J_BENCH import-fbin /tmp/neo4j-vibe yandex.train.fbin - 5000 scalar 1.5
$NEO4J_BENCH bench-fbin /tmp/neo4j-vibe yandex.test.fbin \
  yandex.neighbors.ibin 1000 10 - 100

# Change quantization, search expansion, and optionally HNSW construction.
$NEO4J_BENCH reindex /tmp/neo4j-vibe 200 unfiltered none 200
$NEO4J_BENCH reindex /tmp/neo4j-vibe 200 unfiltered none 100 32 400

# MoReVec: native numeric filtering inside Neo4j's vector index.
$NEO4J_BENCH import-fbin /tmp/neo4j-morevec morevec/train.fbin \
  morevec/metadata.jsonl 5000 none 6
$NEO4J_BENCH bench-fbin /tmp/neo4j-morevec morevec/query-6.fbin \
  morevec/truth-6.ibin 500 10 8.1 50
```

Every result includes recall against the corpus truth, the filter cardinality,
warm open time, p50/p95/max latency, and recursive store size. One transaction
is opened per measured vector query, matching ordinary embedded Neo4j usage.

## Graphalytics BFS

```sh
$NEO4J_BENCH import-graphalytics /tmp/neo4j-wiki-talk \
  wiki-Talk.v wiki-Talk.e 100000
$NEO4J_BENCH bench-bfs /tmp/neo4j-wiki-talk 2 wiki-Talk-BFS 10 1
```

The importer verifies that Graphalytics' IDs map densely to Neo4j internal
IDs. BFS validates all supplied output rows before timing and performs warm,
single-threaded outgoing traversal. This is an embedded storage/traversal
comparison, not a claim about Neo4j GDS or clustered server throughput.

## Enterprise/Desktop baseline

Start an isolated local Enterprise instance, create empty benchmark databases,
and provide credentials only through the environment:

```sh
export NEO4J_URI=neo4j://127.0.0.1:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=...

$NEO4J_BENCH remote probe vgbench
$NEO4J_BENCH remote smoke vgbench
```

The remote loader binds `float[]` through the official driver as native vector
values. The final sentinel argument prevents accidentally running this path
with list-backed properties.

```sh
# VIBE/Yandex native-vector load, index, and recall-matched product query.
$NEO4J_BENCH remote import-fbin vgbench yandex.train.fbin - \
  5000 none 200 16 100 native-vector
$NEO4J_BENCH remote reindex vgbench 200 unfiltered \
  none 200 16 100 native-vector
$NEO4J_BENCH remote bench-fbin vgbench yandex.test.fbin \
  yandex.neighbors.ibin 1000 10 - 20 autocommit 200

# MoReVec native additional-property prefilter.
$NEO4J_BENCH remote import-fbin vgbenchmore morevec/train.fbin \
  morevec/metadata.jsonl 5000 none 6 16 100 native-vector
$NEO4J_BENCH remote bench-fbin vgbenchmore morevec/query-6.fbin \
  morevec/truth-6.ibin 500 10 8.1 20 autocommit 768
```

For the optimized graph import, stop the server and target a database that does
not exist. The checked-in headers map the official space-delimited Graphalytics
files without rewriting them:

```sh
JAVA_HOME=/path/to/desktop-jdk-21 "$DBMS_HOME/bin/neo4j-admin" \
  database import full \
  --nodes=GraphVertex="graphalytics-vertices.header,wiki-Talk.v" \
  --relationships=LINK="graphalytics-edges.header,wiki-Talk.e" \
  --delimiter=" " --id-type=integer --format=block --high-parallel-io=on \
  vgbenchgraph
```

After creating/starting `vgbenchgraph`, project its directed topology once:

```cypher
CALL gds.graph.project(
  'wikiTalk',
  'GraphVertex',
  {LINK: {orientation: 'NATURAL'}}
);
```

The official input is dense and ordered, so external source 2 maps to internal
node ID 2. The harness first checks the reached cardinality through
`gds.bfs.stream`, then times server-reported computation and full driver
latency through `gds.bfs.stats`:

```sh
$NEO4J_BENCH remote bench-gds-bfs vgbenchgraph wikiTalk 2 3 20 1
$NEO4J_BENCH remote bench-gds-bfs vgbenchgraph wikiTalk 2 3 20 4
```

Keep database-directory bytes and retained transaction-log bytes separate.
The former includes both native vector-property storage and the complete HNSW
schema index; measuring only the vector block before index population is not a
valid whole-store comparison.

Full measured results and caveats live in
[`../../docs/benchmarks.md`](../../docs/benchmarks.md).
