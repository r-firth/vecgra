use crate::codec::{self, Header, SnapshotBuilder};
use crate::database::DatabaseOptions;
use crate::error::{Error, Result};
use crate::graph::{GraphStats, Operation};
use crate::model::{EdgeId, NodeId, Property, Value};
use crate::vector::{self, Similarity};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Builds an immutable indexed checkpoint directly, without first retaining a
/// mutable in-memory graph or transaction log. IDs are assigned densely and
/// edges may refer to nodes already appended to this builder.
pub struct BulkLoader {
    path: PathBuf,
    options: DatabaseOptions,
    snapshot: SnapshotBuilder,
    symbol_ids: HashMap<Arc<str>, u32>,
    symbols: usize,
    nodes: u64,
    edges: u64,
    indexed_vectors: usize,
}

impl BulkLoader {
    /// Prepares a direct checkpoint writer at a new destination path.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero vector dimension, an existing destination,
    /// or an unavailable temporary vector spool.
    pub fn new(path: impl AsRef<Path>, options: DatabaseOptions) -> Result<Self> {
        if options.vector_dimension == 0 {
            return Err(Error::InvalidArgument(
                "vector dimension must be greater than zero".into(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        if path.try_exists()? {
            return Err(Error::Conflict(format!(
                "bulk-load destination already exists: {}",
                path.display()
            )));
        }
        Ok(Self {
            snapshot: SnapshotBuilder::new(options.vector_dimension, options.vector_encoding)?,
            path,
            options,
            symbol_ids: HashMap::new(),
            symbols: 0,
            nodes: 0,
            edges: 0,
            indexed_vectors: 0,
        })
    }

    /// Appends a densely numbered node and returns its identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when properties cannot be encoded or a vector has the
    /// wrong dimension, contains a non-finite value, or is zero for cosine
    /// similarity.
    pub fn create_node<I, K>(
        &mut self,
        label: impl Into<Arc<str>>,
        properties: I,
        vectors: &[Vec<f32>],
    ) -> Result<NodeId>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<Arc<str>>,
    {
        let id = self.nodes;
        let label = self.intern(label.into())?;
        let properties = self.resolve_properties(properties)?;
        let encoded_properties = codec::encode_properties_blob(&properties)?;
        let vector_count = u32::try_from(vectors.len())
            .map_err(|_| Error::InvalidArgument("too many vectors on one node".into()))?;
        let vectors = prepare_vectors(
            vectors,
            self.options.vector_dimension,
            self.options.similarity,
        )?;
        self.snapshot
            .append_node_f32(id, 1, label, &encoded_properties, vector_count, &vectors)?;
        self.nodes += 1;
        self.indexed_vectors += vectors.len() / self.options.vector_dimension;
        Ok(id)
    }

    /// Appends a densely numbered relationship between existing nodes.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing endpoint, an invalid vector, or data
    /// that cannot be represented in the checkpoint format.
    pub fn create_edge<I, K>(
        &mut self,
        source: NodeId,
        target: NodeId,
        label: impl Into<Arc<str>>,
        properties: I,
        vectors: &[Vec<f32>],
    ) -> Result<EdgeId>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<Arc<str>>,
    {
        if source >= self.nodes || target >= self.nodes {
            return Err(Error::NotFound("bulk edge endpoint", source.max(target)));
        }
        let id = self.edges;
        let label = self.intern(label.into())?;
        let properties = self.resolve_properties(properties)?;
        let encoded_properties = codec::encode_properties_blob(&properties)?;
        let vector_count = u32::try_from(vectors.len())
            .map_err(|_| Error::InvalidArgument("too many vectors on one edge".into()))?;
        let vectors = prepare_vectors(
            vectors,
            self.options.vector_dimension,
            self.options.similarity,
        )?;
        self.snapshot.append_edge_f32(
            id,
            1,
            source,
            target,
            label,
            &encoded_properties,
            vector_count,
            &vectors,
        )?;
        self.edges += 1;
        self.indexed_vectors += vectors.len() / self.options.vector_dimension;
        Ok(id)
    }

    /// Atomically completes the checkpoint and returns its graph statistics.
    ///
    /// # Errors
    ///
    /// Returns an error when the destination cannot be created or written. A
    /// partially written destination is removed before the error is returned.
    pub fn finish(self) -> Result<GraphStats> {
        let Self {
            path,
            options,
            snapshot,
            symbols,
            nodes,
            edges,
            indexed_vectors,
            ..
        } = self;
        let mut output = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        let result = (|| -> Result<()> {
            let lengths = snapshot.finish_to(&mut output)?;
            codec::write_header(
                &mut output,
                Header {
                    dimension: options.vector_dimension,
                    similarity: options.similarity,
                    vector_encoding: options.vector_encoding,
                    snapshot_metadata_len: lengths.metadata_len,
                    snapshot_vector_offset: codec::HEADER_LEN + lengths.metadata_len,
                    snapshot_vector_len: lengths.vector_len,
                },
            )?;
            output.flush()?;
            output.sync_all()?;
            Ok(())
        })();
        drop(output);
        if let Err(error) = result {
            let _ = std::fs::remove_file(path);
            return Err(error);
        }
        Ok(GraphStats {
            nodes: nodes as usize,
            edges: edges as usize,
            labels: symbols,
            indexed_vectors,
            transactions: 1,
        })
    }

    fn intern(&mut self, value: Arc<str>) -> Result<u32> {
        if let Some(id) = self.symbol_ids.get(&value) {
            return Ok(*id);
        }
        let id = u32::try_from(self.symbols)
            .map_err(|_| Error::InvalidArgument("too many bulk-load symbols".into()))?;
        self.snapshot.append(&[Operation::InternSymbol {
            id,
            value: value.clone(),
        }])?;
        self.symbol_ids.insert(value, id);
        self.symbols += 1;
        Ok(id)
    }

    fn resolve_properties<I, K>(&mut self, properties: I) -> Result<Vec<Property>>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<Arc<str>>,
    {
        let mut result = Vec::new();
        for (key, value) in properties {
            result.push(Property {
                key: self.intern(key.into())?,
                value,
            });
        }
        result.sort_unstable_by_key(|property| property.key);
        result.dedup_by_key(|property| property.key);
        Ok(result)
    }
}

fn prepare_vectors(
    vectors: &[Vec<f32>],
    dimension: usize,
    similarity: Similarity,
) -> Result<Vec<f32>> {
    let mut flattened = Vec::with_capacity(vectors.len().saturating_mul(dimension));
    for vector in vectors {
        if vector.len() != dimension {
            return Err(Error::InvalidArgument(format!(
                "vector dimension {} does not match database dimension {dimension}",
                vector.len()
            )));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(
                "vectors may only contain finite values".into(),
            ));
        }
        let start = flattened.len();
        flattened.extend_from_slice(vector);
        if similarity == Similarity::Cosine {
            vector::normalize(&mut flattened[start..])?;
        }
    }
    Ok(flattened)
}
