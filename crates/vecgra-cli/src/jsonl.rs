use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use vecgra::{BulkLoader, DatabaseOptions, GraphStats, Similarity, Value, VectorEncoding};

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
    source: ExternalId,
    target: ExternalId,
    label: String,
    #[serde(default)]
    properties: BTreeMap<String, JsonValue>,
    #[serde(default)]
    vectors: Vec<Vec<f32>>,
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

    for_json_lines::<JsonNode>(nodes_path, |line, node| {
        let external_id = node.id.clone();
        if node_ids.contains_key(&external_id) {
            return Err(format!(
                "{}:{line}: duplicate node id {external_id}",
                nodes_path.display()
            )
            .into());
        }
        let properties = convert_properties(node.properties)
            .map_err(|message| format!("{}:{line}: {message}", nodes_path.display()))?;
        let id = loader.create_node(node.label, properties, &node.vectors)?;
        node_ids.insert(external_id, id);
        Ok(())
    })?;

    for_json_lines::<JsonEdge>(edges_path, |line, edge| {
        let source = node_ids.get(&edge.source).copied().ok_or_else(|| {
            format!(
                "{}:{line}: edge source {} does not name an imported node",
                edges_path.display(),
                edge.source
            )
        })?;
        let target = node_ids.get(&edge.target).copied().ok_or_else(|| {
            format!(
                "{}:{line}: edge target {} does not name an imported node",
                edges_path.display(),
                edge.target
            )
        })?;
        let properties = convert_properties(edge.properties)
            .map_err(|message| format!("{}:{line}: {message}", edges_path.display()))?;
        loader.create_edge(source, target, edge.label, properties, &edge.vectors)?;
        Ok(())
    })?;

    Ok(loader.finish()?)
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

    for_json_lines::<JsonNodeMetadata>(metadata_path, |line, metadata| {
        if imported == vector_count {
            return Err(format!(
                "{}:{line}: metadata has more records than the {vector_count} vectors in {}",
                metadata_path.display(),
                vectors_path.display()
            )
            .into());
        }
        vectors.read_exact(&mut encoded)?;
        for (value, bytes) in batch[0].iter_mut().zip(encoded.chunks_exact(4)) {
            *value = f32::from_le_bytes(bytes.try_into().unwrap());
        }
        let properties = convert_properties(metadata.properties)
            .map_err(|message| format!("{}:{line}: {message}", metadata_path.display()))?;
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
    mut visitor: impl FnMut(usize, T) -> Result<(), Box<dyn Error>>,
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
        visitor(line_number, value)?;
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
