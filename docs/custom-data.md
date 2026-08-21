# Import your own data

Vecgra is not tied to GitHub. The GitHub command is one adapter built on top of
the same graph database APIs available to your code.

There are three general ingestion paths:

| Need | Use |
| --- | --- |
| Turn node and edge files into a new database | `vecgra import-jsonl` |
| Create, update, or delete data from a Rust application | `Database::transaction()` |
| Build a large new database from Rust | `BulkLoader` |

Labels and property keys do not need advance registration. Vecgra accepts
directed edges, parallel edges, and self-edges. Nodes and edges can each hold
zero, one, or many vectors.

## Import JSONL

The repository includes a small customer and product graph. Import it with:

```sh
cargo build --release

target/release/vecgra import-jsonl \
  examples/custom-data/nodes.jsonl \
  examples/custom-data/edges.jsonl \
  customer-orders.vg \
  4
```

The last argument is the vector dimension. The command prints:

```text
nodes	3
edges	2
vectors	5
```

Inspect and query the result:

```sh
target/release/vecgra stats customer-orders.vg
target/release/vecgra check customer-orders.vg

target/release/vecgra query customer-orders.vg \
  'MATCH (c:Customer)-[r:PURCHASED]->(p:Product) RETURN c,r,p LIMIT 10'
```

You can also open the file in Studio:

```sh
cargo run --release -p vecgra-studio -- customer-orders.vg
```

The checked-in files are
[`examples/custom-data/nodes.jsonl`](../examples/custom-data/nodes.jsonl) and
[`examples/custom-data/edges.jsonl`](../examples/custom-data/edges.jsonl).
Their four-value vectors are test data, not useful semantic embeddings.

### Node records

Write one JSON object per line:

```json
{"id":"product:keyboard","label":"Product","properties":{"name":"Mechanical keyboard","price":129.0,"sku":"KB-01"},"vectors":[[0.8,0.4,0.0,0.0]]}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | String or integer used by edge records during this import |
| `label` | yes | Node label |
| `properties` | no | Object containing scalar JSON values |
| `vectors` | no | Array containing zero or more vectors |

### Edge records

Write one JSON object per line:

```json
{"source":"customer:ada","target":"product:keyboard","label":"PURCHASED","properties":{"order_id":"order-1001","quantity":1},"vectors":[[0.9,0.3,0.0,0.0]]}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `source` | yes | `id` of a node in the node file |
| `target` | yes | `id` of a node in the node file |
| `label` | yes | Edge label |
| `properties` | no | Object containing scalar JSON values |
| `vectors` | no | Array containing zero or more vectors |

The importer reads all nodes before it reads edges. External IDs only resolve
edge endpoints and are not stored automatically. Add an ID to `properties` if
you need to query it later.

Properties may be `null`, booleans, numbers, or strings. JSON arrays and
objects are not property values in this format. Use the Rust API if you need
`Value::Bytes`, `Value::Node`, or `Value::Edge`.

Every supplied vector must match the database dimension and contain finite
numbers. The JSONL importer uses cosine similarity and normalizes vectors while
loading them. You may omit `vectors` for a graph-only element. A vectorless
database still needs a dimension, so choose the size you would use if you add
vectors later.

`import-jsonl` creates a new database and refuses to overwrite an existing
path. It ignores blank lines, reports input filenames and line numbers for bad
records, rejects duplicate node IDs, and rejects edges with missing endpoints.
Pass an empty file as the edge input when importing nodes without edges.
Use `f32` as the optional final argument if you do not want the default F16
checkpoint encoding:

```sh
target/release/vecgra import-jsonl nodes.jsonl edges.jsonl graph.vg 768 f32
```

Vecgra stores vectors that you supply. This command does not turn property text
into embeddings. Generate embeddings with your chosen model before writing the
JSONL, and use the same model and dimension for query vectors.

## Ingest from Rust

Use transactions for an application that adds or changes data over time. The
normal pattern is to keep a map from IDs in your source system to the `NodeId`
values returned by Vecgra, then create edges with those IDs.

The checked-in [`custom_ingest.rs`](../crates/vecgra/examples/custom_ingest.rs)
example does this with customer, product, and purchase records:

```sh
cargo run -p vecgra --example custom_ingest -- customer-orders-rust.vg
```

The essential write path is:

```rust
use std::collections::HashMap;
use vecgra::{Database, DatabaseOptions, Value};

let database = Database::create("graph.vg", DatabaseOptions::new(4))?;
let mut transaction = database.transaction();
let mut ids = HashMap::new();

for record in source_nodes {
    let id = transaction.create_node(
        record.label,
        record.properties,
        &record.vectors,
    );
    ids.insert(record.external_id, id);
}

for record in source_edges {
    transaction.create_edge(
        ids[&record.source],
        ids[&record.target],
        record.label,
        record.properties,
        &record.vectors,
    );
}

transaction.commit()?;
```

Call `Database::open` instead of `Database::create` for later writes. A
transaction also has `update_node`, `update_edge`, `delete_node`, and
`delete_edge`. Updates replace the complete label, property set, and vector set
for that element.

Commits are durable and atomic. If validation fails, none of that
transaction's mutations become visible. Vecgra assigns internal node and edge
IDs. Persist the source-to-Vecgra ID mapping in your application if later
imports need it.

## Bulk load from Rust

`BulkLoader` creates a new indexed checkpoint without retaining the whole
mutable graph or a transaction log in memory. It is the Rust equivalent of the
JSONL import path.

```rust
use vecgra::{BulkLoader, DatabaseOptions};

let mut loader = BulkLoader::new("graph.vg", DatabaseOptions::new(768))?;
let customer = loader.create_node("Customer", customer_properties, &customer_vectors)?;
let product = loader.create_node("Product", product_properties, &product_vectors)?;
loader.create_edge(customer, product, "PURCHASED", purchase_properties, &purchase_vectors)?;
let stats = loader.finish()?;
```

Append every node before any edge that refers to it. `BulkLoader::finish`
writes the final database file. The destination must not exist.
