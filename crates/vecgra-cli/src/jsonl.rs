use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use vecgra::{
    BulkLoader, Database, DatabaseOptions, GraphStats, ReadGuard, Similarity, Value, VectorEncoding,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AppendStats {
    pub(crate) nodes: usize,
    pub(crate) edges: usize,
    pub(crate) indexed_vectors: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(untagged)]
enum ExternalId {
    String(String),
    Signed(i64),
    Unsigned(u64),
}

impl fmt::Display for ExternalId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(value) => write!(formatter, "{value:?}"),
            Self::Signed(value) => value.fmt(formatter),
            Self::Unsigned(value) => value.fmt(formatter),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonNode {
    id: ExternalId,
    label: String,
    #[serde(default)]
    properties: BTreeMap<String, JsonValue>,
    #[serde(default)]
    vectors: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonEdge {
    source: Endpoint,
    target: Endpoint,
    label: String,
    #[serde(default)]
    properties: BTreeMap<String, JsonValue>,
    #[serde(default)]
    vectors: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum Endpoint {
    Batch(ExternalId),
    Existing { node: u64 },
}

impl Endpoint {
    fn resolve(
        &self,
        node_ids: &HashMap<ExternalId, u64>,
        existing: Option<&ReadGuard<'_>>,
    ) -> Result<u64, Box<dyn Error>> {
        match self {
            Self::Batch(id) => node_ids.get(id).copied().ok_or_else(|| {
                format!("edge endpoint {id} does not name a node in this batch").into()
            }),
            Self::Existing { node } => {
                let read = existing.ok_or("existing node references require append-jsonl")?;
                read.node(*node)
                    .map(|record| record.id)
                    .ok_or_else(|| format!("existing node {node} was not found").into())
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonNodeMetadata {
    label: String,
    #[serde(default)]
    properties: BTreeMap<String, JsonValue>,
}

pub(crate) fn import_jsonl(
    nodes_path: &Path,
    edges_path: &Path,
    database_path: &Path,
    dimension: usize,
    vector_encoding: VectorEncoding,
) -> Result<GraphStats, Box<dyn Error>> {
    let mut loader = BulkLoader::new(
        database_path,
        DatabaseOptions {
            vector_dimension: dimension,
            similarity: Similarity::Cosine,
            vector_encoding,
            sync_on_commit: true,
        },
    )?;
    let mut node_ids = HashMap::new();

    for_json_lines::<JsonNode>(nodes_path, |node| {
        let external_id = node.id.clone();
        if node_ids.contains_key(&external_id) {
            return Err(format!("duplicate node id {external_id}").into());
        }
        let properties = convert_properties(node.properties)?;
        let id = loader.create_node(node.label, properties, &node.vectors)?;
        node_ids.insert(external_id, id);
        Ok(())
    })?;

    for_json_lines::<JsonEdge>(edges_path, |edge| {
        let source = edge.source.resolve(&node_ids, None)?;
        let target = edge.target.resolve(&node_ids, None)?;
        let properties = convert_properties(edge.properties)?;
        loader.create_edge(source, target, edge.label, properties, &edge.vectors)?;
        Ok(())
    })?;

    Ok(loader.finish()?)
}

/// Durably appends one JSONL batch to an existing database.
///
/// Scalar endpoints name batch-local IDs; `{ "node": id }` endpoints refer to
/// existing database nodes. Either input file may be empty.
pub(crate) fn append_jsonl(
    database_path: &Path,
    nodes_path: &Path,
    edges_path: &Path,
) -> Result<AppendStats, Box<dyn Error>> {
    let database = Database::open(database_path)?;
    let mut transaction = database.transaction();
    let mut node_ids = HashMap::new();
    let mut stats = AppendStats::default();

    for_json_lines::<JsonNode>(nodes_path, |node| {
        let external_id = node.id.clone();
        if node_ids.contains_key(&external_id) {
            return Err(format!("duplicate node id {external_id}").into());
        }
        let properties = convert_properties(node.properties)?;
        stats.indexed_vectors += node.vectors.len();
        let id = transaction.create_node(node.label, properties, &node.vectors);
        node_ids.insert(external_id, id);
        stats.nodes += 1;
        Ok(())
    })?;

    let read = database.read();
    for_json_lines::<JsonEdge>(edges_path, |edge| {
        let source = edge.source.resolve(&node_ids, Some(&read))?;
        let target = edge.target.resolve(&node_ids, Some(&read))?;
        let properties = convert_properties(edge.properties)?;
        stats.indexed_vectors += edge.vectors.len();
        transaction.create_edge(source, target, edge.label, properties, &edge.vectors);
        stats.edges += 1;
        Ok(())
    })?;

    drop(read);
    transaction.commit()?;
    Ok(stats)
}

/// Streams one fbin vector and one JSON metadata record into each node. This
/// keeps vector datasets in their standard ANN interchange format while still
/// preserving a typed property graph schema. Blank metadata lines are ignored.
pub(crate) fn import_node_fbin(
    vectors_path: &Path,
    metadata_path: &Path,
    database_path: &Path,
    vector_encoding: VectorEncoding,
) -> Result<GraphStats, Box<dyn Error>> {
    let (mut vectors, vector_count, dimension) =
        crate::ann_benchmark::open_matrix(vectors_path, 4)?;
    let mut loader = BulkLoader::new(
        database_path,
        DatabaseOptions {
            vector_dimension: dimension,
            similarity: Similarity::Cosine,
            vector_encoding,
            sync_on_commit: true,
        },
    )?;
    let mut encoded = vec![0u8; dimension * size_of::<f32>()];
    let mut batch = vec![vec![0.0f32; dimension]];
    let mut imported = 0usize;

    for_json_lines::<JsonNodeMetadata>(metadata_path, |metadata| {
        if imported == vector_count {
            return Err(format!(
                "metadata has more records than the {vector_count} vectors in {}",
                vectors_path.display()
            )
            .into());
        }
        vectors.read_exact(&mut encoded)?;
        for (value, bytes) in batch[0].iter_mut().zip(encoded.chunks_exact(4)) {
            *value = f32::from_le_bytes(bytes.try_into().unwrap());
        }
        let properties = convert_properties(metadata.properties)?;
        loader.create_node(metadata.label, properties, &batch)?;
        imported += 1;
        if imported.is_multiple_of(100_000) {
            eprintln!("stored {imported}/{vector_count} vectors with metadata");
        }
        Ok(())
    })?;

    if imported != vector_count {
        return Err(format!(
            "{} has {imported} metadata records but {} contains {vector_count} vectors",
            metadata_path.display(),
            vectors_path.display()
        )
        .into());
    }
    Ok(loader.finish()?)
}

fn for_json_lines<T>(
    path: &Path,
    mut visitor: impl FnMut(T) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>>
where
    T: for<'de> Deserialize<'de>,
{
    let input = BufReader::new(File::open(path)?);
    for (index, line) in input.lines().enumerate() {
        let line_number = index + 1;
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{line_number}: invalid JSON: {error}", path.display()))?;
        visitor(value).map_err(|error| format!("{}:{line_number}: {error}", path.display()))?;
    }
    Ok(())
}

pub(crate) fn convert_properties(
    properties: BTreeMap<String, JsonValue>,
) -> Result<Vec<(String, Value)>, String> {
    properties
        .into_iter()
        .map(|(key, value)| {
            let value = match value {
                JsonValue::Null => Value::Null,
                JsonValue::Bool(value) => Value::Bool(value),
                JsonValue::Number(value) => {
                    if let Some(value) = value.as_i64() {
                        Value::Int(value)
                    } else if let Some(value) = value.as_f64() {
                        Value::Float(value)
                    } else {
                        return Err(format!("property {key:?} is outside the numeric range"));
                    }
                }
                JsonValue::String(value) => Value::String(Arc::from(value)),
                JsonValue::Array(_) | JsonValue::Object(_) => {
                    return Err(format!(
                        "property {key:?} must be a null, boolean, number, or string"
                    ));
                }
            };
            Ok((key, value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};
    use vecgra::{Database, Direction, EdgeFilter};

    fn write_fbin(path: &Path, rows: &[&[f32]]) {
        let dimension = rows.first().map_or(0, |row| row.len());
        let mut bytes = Vec::with_capacity(8 + rows.len() * dimension * 4);
        bytes.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(dimension as u32).to_le_bytes());
        for row in rows {
            assert_eq!(row.len(), dimension);
            for value in *row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        fs::write(path, bytes).unwrap();
    }

    fn path(suffix: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "vecgra-jsonl-{}-{nonce}-{suffix}",
            std::process::id()
        ))
    }

    #[test]
    fn generic_jsonl_import_preserves_graph_properties_and_vectors() {
        let nodes = path("nodes.jsonl");
        let edges = path("edges.jsonl");
        let database_path = path("graph.vg");
        fs::write(
            &nodes,
            concat!(
                "{\"id\":\"doc:a\",\"label\":\"Document\",\"properties\":{\"title\":\"Alpha\",\"year\":2026},\"vectors\":[[2,0,0,0]]}\n",
                "{\"id\":2,\"label\":\"Claim\",\"properties\":{\"grounded\":true},\"vectors\":[[0,1,0,0],[0,0,1,0]]}\n"
            ),
        )
        .unwrap();
        fs::write(
            &edges,
            "{\"source\":\"doc:a\",\"target\":2,\"label\":\"SUPPORTS\",\"properties\":{\"weight\":0.75},\"vectors\":[[1,1,0,0]]}\n",
        )
        .unwrap();

        let stats = import_jsonl(&nodes, &edges, &database_path, 4, VectorEncoding::F16).unwrap();
        assert_eq!((stats.nodes, stats.edges, stats.indexed_vectors), (2, 1, 4));

        let database = Database::open(&database_path).unwrap();
        let read = database.read();
        let first = read.node(0).unwrap();
        assert_eq!(
            read.property(&first.properties, "title"),
            Some(&Value::String(Arc::from("Alpha")))
        );
        assert_eq!(read.node(1).unwrap().vector_count, 2);
        let edges_from_first = read
            .neighbors(0, Direction::Outgoing, EdgeFilter::default())
            .unwrap();
        assert_eq!(edges_from_first.len(), 1);
        assert_eq!(edges_from_first[0].target, 1);
        assert_eq!(read.symbol(edges_from_first[0].label), Some("SUPPORTS"));
        drop(read);
        drop(database);

        fs::remove_file(nodes).unwrap();
        fs::remove_file(edges).unwrap();
        fs::remove_file(database_path).unwrap();
    }

    #[test]
    fn generic_jsonl_append_adds_a_durable_batch() {
        let initial_nodes = path("initial-nodes.jsonl");
        let appended_nodes = path("appended-nodes.jsonl");
        let edges = path("empty-edges.jsonl");
        let database_path = path("append.vg");
        fs::write(
            &initial_nodes,
            "{\"id\":\"old\",\"label\":\"Memory\",\"properties\":{\"text\":\"old\"},\"vectors\":[[1,0]]}\n",
        )
        .unwrap();
        fs::write(
            &appended_nodes,
            "{\"id\":\"new\",\"label\":\"Memory\",\"properties\":{\"text\":\"new\"},\"vectors\":[[0,1]]}\n",
        )
        .unwrap();
        fs::write(&edges, "").unwrap();

        import_jsonl(
            &initial_nodes,
            &edges,
            &database_path,
            2,
            VectorEncoding::F16,
        )
        .unwrap();
        let appended = append_jsonl(&database_path, &appended_nodes, &edges).unwrap();
        assert_eq!(
            appended,
            AppendStats {
                nodes: 1,
                edges: 0,
                indexed_vectors: 1
            }
        );

        let database = Database::open(&database_path).unwrap();
        let stats = database.read().stats();
        assert_eq!((stats.nodes, stats.edges, stats.indexed_vectors), (2, 0, 2));
        drop(database);

        fs::remove_file(initial_nodes).unwrap();
        fs::remove_file(appended_nodes).unwrap();
        fs::remove_file(edges).unwrap();
        fs::remove_file(database_path).unwrap();
    }

    #[test]
    fn node_fbin_import_streams_vectors_and_typed_metadata_in_lockstep() {
        let vectors = path("vectors.fbin");
        let metadata = path("metadata.jsonl");
        let database_path = path("metadata.vg");
        write_fbin(&vectors, &[&[2.0, 0.0], &[0.0, 3.0]]);
        fs::write(
            &metadata,
            concat!(
                "{\"label\":\"Movie\",\"properties\":{\"mid\":\"m1\",\"rating\":9.3}}\n",
                "{\"label\":\"Movie\",\"properties\":{\"mid\":\"m2\",\"year\":2026}}\n"
            ),
        )
        .unwrap();

        let stats =
            import_node_fbin(&vectors, &metadata, &database_path, VectorEncoding::F16).unwrap();
        assert_eq!((stats.nodes, stats.indexed_vectors), (2, 2));
        let database = Database::open(&database_path).unwrap();
        let read = database.read();
        let movie = read.label_id("Movie").unwrap();
        assert_eq!(
            read.elements_with_label(movie, vecgra::VectorTarget::Nodes)
                .len(),
            2
        );
        let first = read.node(0).unwrap();
        assert_eq!(
            read.property(&first.properties, "rating"),
            Some(&Value::Float(9.3))
        );
        let normalized = read.node_vector_owned(1, 0).unwrap().unwrap();
        assert!((normalized[1] - 1.0).abs() < 0.001);

        drop(read);
        drop(database);
        fs::remove_file(vectors).unwrap();
        fs::remove_file(metadata).unwrap();
        fs::remove_file(database_path).unwrap();
    }
}
