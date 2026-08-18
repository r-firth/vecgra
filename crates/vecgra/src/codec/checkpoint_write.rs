use super::*;

struct SnapshotNodeRecord {
    id: u64,
    generation: u64,
    label: u32,
    property_offset: u64,
    property_len: u32,
    vector_count: u32,
    vector_offset: u64,
}

#[derive(Debug)]
struct SnapshotEdgeRecord {
    id: u64,
    generation: u64,
    source: u64,
    target: u64,
    label: u32,
    property_offset: u64,
    property_len: u32,
    vector_count: u32,
    vector_offset: u64,
}

pub(crate) struct SnapshotBuilder {
    dimension: usize,
    vector_encoding: VectorEncoding,
    symbols: Vec<Arc<str>>,
    nodes: Vec<SnapshotNodeRecord>,
    edges: Vec<SnapshotEdgeRecord>,
    properties: Vec<u8>,
    property_count: usize,
    numeric_property_count: usize,
    vectors: VectorSpool,
    vector_float_offset: u64,
    sketches: Vec<u64>,
    sketch_workspace: Vec<f32>,
    sketch_vector: Vec<f32>,
}

impl SnapshotBuilder {
    pub(crate) fn new(dimension: usize, vector_encoding: VectorEncoding) -> Result<Self> {
        Ok(Self {
            dimension,
            vector_encoding,
            symbols: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            properties: Vec::new(),
            property_count: 0,
            numeric_property_count: 0,
            vectors: VectorSpool::create()?,
            vector_float_offset: 0,
            sketches: Vec::new(),
            sketch_workspace: Vec::new(),
            sketch_vector: vec![0.0; dimension],
        })
    }

    pub(crate) fn append(&mut self, operations: &[Operation]) -> Result<()> {
        for operation in operations {
            match operation {
                Operation::InternSymbol { id, value } => {
                    if *id as usize != self.symbols.len() {
                        return Err(Error::InvalidArgument(format!(
                            "checkpoint symbol id {id} is not contiguous"
                        )));
                    }
                    self.symbols.push(value.clone());
                }
                Operation::PutNode(node) => {
                    let mut properties = Vec::new();
                    encode_properties(&mut properties, &node.properties)?;
                    self.append_node_f32(
                        node.id,
                        node.generation,
                        node.label,
                        &properties,
                        node.vector_count,
                        &node.pending_vectors,
                    )?;
                }
                Operation::PutEdge(edge) => {
                    let mut properties = Vec::new();
                    encode_properties(&mut properties, &edge.properties)?;
                    self.append_edge_f32(
                        edge.id,
                        edge.generation,
                        edge.source,
                        edge.target,
                        edge.label,
                        &properties,
                        edge.vector_count,
                        &edge.pending_vectors,
                    )?;
                }
                Operation::DeleteNode(_) | Operation::DeleteEdge(_) => {
                    return Err(Error::InvalidArgument(
                        "checkpoint may only contain live graph records".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn vector_encoding(&self) -> VectorEncoding {
        self.vector_encoding
    }

    pub(crate) fn append_node_f32(
        &mut self,
        id: u64,
        generation: u64,
        label: u32,
        encoded_properties: &[u8],
        vector_count: u32,
        vectors: &[f32],
    ) -> Result<()> {
        self.validate_float_count(vector_count, vectors.len())?;
        let mut encoded = Vec::with_capacity(vectors.len() * self.bytes_per_float());
        encode_floats(&mut encoded, vectors, self.vector_encoding);
        self.append_node_encoded(
            id,
            generation,
            label,
            encoded_properties,
            vector_count,
            &encoded,
        )
    }

    pub(crate) fn append_node_encoded(
        &mut self,
        id: u64,
        generation: u64,
        label: u32,
        encoded_properties: &[u8],
        vector_count: u32,
        encoded_vectors: &[u8],
    ) -> Result<()> {
        self.validate_encoded_vector_len(vector_count, encoded_vectors.len())?;
        let (property_offset, property_len) = self.append_properties(encoded_properties)?;
        let vector_offset = self.append_vectors(vector_count, encoded_vectors)?;
        self.nodes.push(SnapshotNodeRecord {
            id,
            generation,
            label,
            property_offset,
            property_len,
            vector_count,
            vector_offset,
        });
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "arguments mirror the stable on-disk edge record"
    )]
    pub(crate) fn append_edge_f32(
        &mut self,
        id: u64,
        generation: u64,
        source: u64,
        target: u64,
        label: u32,
        encoded_properties: &[u8],
        vector_count: u32,
        vectors: &[f32],
    ) -> Result<()> {
        self.validate_float_count(vector_count, vectors.len())?;
        let mut encoded = Vec::with_capacity(vectors.len() * self.bytes_per_float());
        encode_floats(&mut encoded, vectors, self.vector_encoding);
        self.append_edge_encoded(
            id,
            generation,
            source,
            target,
            label,
            encoded_properties,
            vector_count,
            &encoded,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "arguments mirror the stable encoded edge record"
    )]
    pub(crate) fn append_edge_encoded(
        &mut self,
        id: u64,
        generation: u64,
        source: u64,
        target: u64,
        label: u32,
        encoded_properties: &[u8],
        vector_count: u32,
        encoded_vectors: &[u8],
    ) -> Result<()> {
        self.validate_encoded_vector_len(vector_count, encoded_vectors.len())?;
        let (property_offset, property_len) = self.append_properties(encoded_properties)?;
        let vector_offset = self.append_vectors(vector_count, encoded_vectors)?;
        self.edges.push(SnapshotEdgeRecord {
            id,
            generation,
            source,
            target,
            label,
            property_offset,
            property_len,
            vector_count,
            vector_offset,
        });
        Ok(())
    }

    fn bytes_per_float(&self) -> usize {
        match self.vector_encoding {
            VectorEncoding::F32 => 4,
            VectorEncoding::F16 => 2,
        }
    }

    fn validate_float_count(&self, vector_count: u32, actual: usize) -> Result<()> {
        let expected = self
            .dimension
            .checked_mul(vector_count as usize)
            .ok_or_else(|| Error::InvalidArgument("checkpoint vector length overflow".into()))?;
        if actual != expected {
            return Err(Error::InvalidArgument(
                "checkpoint vector float count does not match dimension".into(),
            ));
        }
        Ok(())
    }

    fn validate_encoded_vector_len(&self, vector_count: u32, actual: usize) -> Result<()> {
        let expected = self
            .dimension
            .checked_mul(vector_count as usize)
            .and_then(|floats| floats.checked_mul(self.bytes_per_float()))
            .ok_or_else(|| {
                Error::InvalidArgument("checkpoint vector byte length overflow".into())
            })?;
        if actual != expected {
            return Err(Error::InvalidArgument(
                "checkpoint encoded vector length does not match dimension".into(),
            ));
        }
        Ok(())
    }

    fn append_properties(&mut self, encoded: &[u8]) -> Result<(u64, u32)> {
        if encoded.len() < 4 {
            return Err(Error::InvalidArgument(
                "encoded property record is truncated".into(),
            ));
        }
        let offset = self.properties.len() as u64;
        let count = u32::from_le_bytes(encoded[..4].try_into().unwrap()) as usize;
        self.property_count = self
            .property_count
            .checked_add(count)
            .ok_or_else(|| Error::InvalidArgument("checkpoint property count overflows".into()))?;
        self.numeric_property_count = self
            .numeric_property_count
            .checked_add(encoded_numeric_property_count(encoded)?)
            .ok_or_else(|| {
                Error::InvalidArgument("checkpoint numeric property count overflows".into())
            })?;
        let len = u32::try_from(encoded.len())
            .map_err(|_| Error::InvalidArgument("property record exceeds u32".into()))?;
        self.properties.extend_from_slice(encoded);
        Ok((offset, len))
    }

    fn append_vectors(&mut self, vector_count: u32, encoded: &[u8]) -> Result<u64> {
        let offset = self.vector_float_offset;
        let bytes_per_float = self.bytes_per_float();
        let bytes_per_vector = self.dimension * bytes_per_float;
        for encoded_vector in encoded.chunks_exact(bytes_per_vector) {
            match self.vector_encoding {
                VectorEncoding::F32 => {
                    for (value, chunk) in self
                        .sketch_vector
                        .iter_mut()
                        .zip(encoded_vector.chunks_exact(4))
                    {
                        *value = f32::from_le_bytes(chunk.try_into().unwrap());
                    }
                }
                VectorEncoding::F16 => {
                    for (value, chunk) in self
                        .sketch_vector
                        .iter_mut()
                        .zip(encoded_vector.chunks_exact(2))
                    {
                        *value = f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap()));
                    }
                }
            }
            crate::ann::append_vector_signature(
                &self.sketch_vector,
                &mut self.sketch_workspace,
                &mut self.sketches,
            );
        }
        self.vectors.write_all(encoded)?;
        self.vector_float_offset = self
            .vector_float_offset
            .checked_add(self.dimension as u64 * vector_count as u64)
            .ok_or_else(|| Error::InvalidArgument("checkpoint vector offset overflow".into()))?;
        Ok(offset)
    }

    pub(crate) fn finish_to(mut self, output: &mut File) -> Result<SnapshotLengths> {
        self.vectors.finish_blocks();
        self.nodes.sort_unstable_by_key(|node| node.id);
        self.edges.sort_unstable_by_key(|edge| edge.id);
        if self.nodes.windows(2).any(|pair| pair[0].id == pair[1].id)
            || self.edges.windows(2).any(|pair| pair[0].id == pair[1].id)
        {
            return Err(Error::InvalidArgument(
                "checkpoint contains duplicate element ids".into(),
            ));
        }
        let node_slots = self.nodes.last().map_or(0, |node| node.id + 1);
        let edge_slots = self.edges.last().map_or(0, |edge| edge.id + 1);
        let (out_offsets, out_ids) = build_snapshot_csr(&self.edges, node_slots, true)?;
        let (in_offsets, in_ids) = build_snapshot_csr(&self.edges, node_slots, false)?;

        let indexed_vectors = self
            .nodes
            .iter()
            .map(|node| node.vector_count as u64)
            .chain(self.edges.iter().map(|edge| edge.vector_count as u64))
            .sum::<u64>();
        let entry_count = usize::try_from(indexed_vectors)
            .map_err(|_| Error::InvalidArgument("indexed vector count exceeds usize".into()))?;
        let words_per_signature = crate::ann::signature_word_count(self.dimension);
        let expected_sketch_words = entry_count
            .checked_mul(words_per_signature)
            .ok_or_else(|| Error::InvalidArgument("checkpoint sketch count overflows".into()))?;
        if self.sketches.len() != expected_sketch_words {
            return Err(Error::InvalidArgument(
                "checkpoint sketch count does not match indexed vectors".into(),
            ));
        }
        let owner_offset = 24usize;
        let owner_kind_offset =
            owner_offset
                .checked_add(entry_count.checked_mul(8).ok_or_else(|| {
                    Error::InvalidArgument("sketch owner column overflows".into())
                })?)
                .ok_or_else(|| Error::InvalidArgument("sketch owner column overflows".into()))?;
        let label_offset = align_up(
            owner_kind_offset
                .checked_add(entry_count)
                .ok_or_else(|| Error::InvalidArgument("sketch kind column overflows".into()))?,
            4,
        )?;
        let signature_offset = align_up(
            label_offset
                .checked_add(entry_count.checked_mul(4).ok_or_else(|| {
                    Error::InvalidArgument("sketch label column overflows".into())
                })?)
                .ok_or_else(|| Error::InvalidArgument("sketch label column overflows".into()))?,
            8,
        )?;
        let sketch_len = signature_offset
            .checked_add(self.sketches.len().checked_mul(8).ok_or_else(|| {
                Error::InvalidArgument("sketch signature column overflows".into())
            })?)
            .ok_or_else(|| Error::InvalidArgument("sketch section overflows".into()))?;

        let mut symbols = Vec::new();
        for (id, symbol) in self.symbols.iter().enumerate() {
            put_u32(&mut symbols, id as u32);
            put_bytes(&mut symbols, symbol.as_bytes())?;
        }
        for checksum in &self.vectors.block_checksums {
            put_u32(&mut symbols, *checksum);
        }
        let maximum_packed_element = self
            .nodes
            .last()
            .map(|node| node.id.checked_mul(2))
            .into_iter()
            .chain(
                self.edges
                    .last()
                    .map(|edge| edge.id.checked_mul(2).and_then(|id| id.checked_add(1))),
            )
            .flatten()
            .max()
            .unwrap_or(0);
        let packed_width = if maximum_packed_element <= u32::MAX as u64 {
            4u8
        } else {
            8u8
        };
        let equality_property_count = self
            .property_count
            .checked_sub(self.numeric_property_count)
            .ok_or_else(|| {
                Error::InvalidArgument("numeric property count exceeds all properties".into())
            })?;
        let property_index_len = 24usize
            .checked_add(
                equality_property_count
                    .checked_mul(8 + packed_width as usize)
                    .ok_or_else(|| {
                        Error::InvalidArgument("property index length overflows".into())
                    })?,
            )
            .ok_or_else(|| Error::InvalidArgument("property index length overflows".into()))?;
        let numeric_property_index_len = 24usize
            .checked_add(
                self.numeric_property_count
                    .checked_mul(16 + packed_width as usize)
                    .ok_or_else(|| {
                        Error::InvalidArgument("numeric property index length overflows".into())
                    })?,
            )
            .ok_or_else(|| {
                Error::InvalidArgument("numeric property index length overflows".into())
            })?;
        let section_lengths = [
            symbols.len(),
            checked_column_len(self.nodes.len(), 48)?,
            checked_column_len(self.edges.len(), 64)?,
            self.properties.len(),
            checked_column_len(out_offsets.len(), 8)?,
            checked_column_len(out_ids.len(), 8)?,
            checked_column_len(in_offsets.len(), 8)?,
            checked_column_len(in_ids.len(), 8)?,
            sketch_len,
            property_index_len,
            numeric_property_index_len,
        ];
        let sections = plan_columnar_sections(&section_lengths)?;
        let mut metadata_header = vec![0u8; RANGE_INDEXED_COLUMNAR_HEADER_LEN];
        metadata_header[..8].copy_from_slice(RANGE_INDEXED_SNAPSHOT_MAGIC);
        metadata_header[8..16].copy_from_slice(&(self.nodes.len() as u64).to_le_bytes());
        metadata_header[16..24].copy_from_slice(&(self.edges.len() as u64).to_le_bytes());
        metadata_header[24..32].copy_from_slice(&(self.symbols.len() as u64).to_le_bytes());
        metadata_header[32..40].copy_from_slice(&node_slots.to_le_bytes());
        metadata_header[40..48].copy_from_slice(&edge_slots.to_le_bytes());
        metadata_header[48..56].copy_from_slice(&indexed_vectors.to_le_bytes());
        metadata_header[56..64]
            .copy_from_slice(&(self.vectors.block_checksums.len() as u64).to_le_bytes());
        for (index, &(offset, len)) in sections.iter().enumerate() {
            let start = 64 + index * 16;
            metadata_header[start..start + 8].copy_from_slice(&offset.to_le_bytes());
            metadata_header[start + 8..start + 16].copy_from_slice(&len.to_le_bytes());
        }
        let mut metadata = MetadataWriter::new(output)?;
        metadata.write_all(&metadata_header)?;
        append_columnar_section(&mut metadata, &sections[0], &symbols)?;

        append_node_record_section(&mut metadata, &sections[1], &self.nodes)?;
        append_edge_record_section(&mut metadata, &sections[2], &self.edges)?;
        append_columnar_section(&mut metadata, &sections[3], &self.properties)?;
        append_u64_columnar_section(&mut metadata, &sections[4], &out_offsets)?;
        append_u64_columnar_section(&mut metadata, &sections[5], &out_ids)?;
        append_u64_columnar_section(&mut metadata, &sections[6], &in_offsets)?;
        append_u64_columnar_section(&mut metadata, &sections[7], &in_ids)?;

        let mut sketch_owners = vec![0u64; entry_count];
        let mut sketch_owner_kinds = vec![0u8; entry_count];
        let mut sketch_labels = vec![0u32; entry_count];
        let mut populated = vec![false; entry_count];
        for node in &self.nodes {
            populate_sketch_owner_columns(
                &mut sketch_owners,
                &mut sketch_owner_kinds,
                &mut sketch_labels,
                &mut populated,
                self.dimension,
                node.id,
                0,
                node.label,
                node.vector_offset,
                node.vector_count,
            )?;
        }
        for edge in &self.edges {
            populate_sketch_owner_columns(
                &mut sketch_owners,
                &mut sketch_owner_kinds,
                &mut sketch_labels,
                &mut populated,
                self.dimension,
                edge.id,
                1,
                edge.label,
                edge.vector_offset,
                edge.vector_count,
            )?;
        }
        if populated.iter().any(|populated| !populated) {
            return Err(Error::InvalidArgument(
                "sketch owner columns do not cover every vector".into(),
            ));
        }
        append_sketch_section(
            &mut metadata,
            &sections[8],
            indexed_vectors,
            words_per_signature,
            owner_offset,
            owner_kind_offset,
            label_offset,
            signature_offset,
            &sketch_owners,
            &sketch_owner_kinds,
            &sketch_labels,
            self.sketches,
        )?;

        let mut property_entries = Vec::new();
        let mut numeric_property_entries = Vec::new();
        for node in &self.nodes {
            append_property_index_entries(
                &mut property_entries,
                &mut numeric_property_entries,
                &self.properties,
                node.property_offset,
                node.property_len,
                0,
                node.id,
            )?;
        }
        for edge in &self.edges {
            append_property_index_entries(
                &mut property_entries,
                &mut numeric_property_entries,
                &self.properties,
                edge.property_offset,
                edge.property_len,
                1,
                edge.id,
            )?;
        }
        property_entries.sort_unstable();
        if property_entries.len() != equality_property_count
            || (packed_width == 4
                && property_entries
                    .iter()
                    .any(|entry| entry.2 > u32::MAX as u64))
        {
            return Err(Error::InvalidArgument(
                "property index plan does not match encoded properties".into(),
            ));
        }
        numeric_property_entries.sort_unstable();
        if numeric_property_entries.len() != self.numeric_property_count
            || (packed_width == 4
                && numeric_property_entries
                    .iter()
                    .any(|entry| entry.3 > u32::MAX as u64))
        {
            return Err(Error::InvalidArgument(
                "numeric property index plan does not match encoded properties".into(),
            ));
        }
        append_property_index_section(&mut metadata, &sections[9], packed_width, property_entries)?;
        append_numeric_property_index_section(
            &mut metadata,
            &sections[10],
            packed_width,
            numeric_property_entries,
        )?;
        let metadata_len = metadata.finish()?;
        let vector_len =
            self.vectors.byte_len.checked_add(4).ok_or_else(|| {
                Error::InvalidArgument("checkpoint vector length overflows".into())
            })?;
        self.vectors.copy_to(output)?;
        Ok(SnapshotLengths {
            metadata_len,
            vector_len,
        })
    }
}

fn append_property_index_entries(
    output: &mut Vec<(u32, u32, u64)>,
    numeric_output: &mut Vec<(u32, u8, u64, u64)>,
    properties: &[u8],
    property_offset: u64,
    property_len: u32,
    kind: u8,
    id: u64,
) -> Result<()> {
    let packed_element = id
        .checked_mul(2)
        .and_then(|value| value.checked_add(kind as u64))
        .ok_or_else(|| {
            Error::InvalidArgument("property-indexed element id exceeds 63 bits".into())
        })?;
    let start = usize::try_from(property_offset)
        .map_err(|_| Error::InvalidArgument("property offset exceeds usize".into()))?;
    let end = start
        .checked_add(property_len as usize)
        .ok_or_else(|| Error::InvalidArgument("property range overflows".into()))?;
    let encoded = properties
        .get(start..end)
        .ok_or_else(|| Error::InvalidArgument("property range exceeds checkpoint".into()))?;
    visit_encoded_property_index_keys(encoded, |key, fingerprint, numeric| {
        if let Some((tag, sortable)) = numeric {
            numeric_output.push((key, tag, sortable, packed_element));
        } else {
            output.push((key, fingerprint, packed_element));
        }
    })?;
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "parallel slices are the compact persisted sketch columns"
)]
pub(super) fn populate_sketch_owner_columns(
    owners: &mut [u64],
    kinds: &mut [u8],
    labels: &mut [u32],
    populated: &mut [bool],
    dimension: usize,
    owner: u64,
    kind: u8,
    label: u32,
    vector_float_offset: u64,
    vector_count: u32,
) -> Result<()> {
    let dimension = u64::try_from(dimension)
        .map_err(|_| Error::InvalidArgument("vector dimension exceeds u64".into()))?;
    if dimension == 0 {
        return Err(Error::InvalidArgument(
            "vector dimension must be greater than zero".into(),
        ));
    }
    if !vector_float_offset.is_multiple_of(dimension) {
        return Err(Error::InvalidArgument(
            "element vector offset is not dimension-aligned".into(),
        ));
    }
    let first = usize::try_from(vector_float_offset / dimension)
        .map_err(|_| Error::InvalidArgument("sketch ordinal exceeds usize".into()))?;
    for vector_index in 0..vector_count as usize {
        let ordinal = first
            .checked_add(vector_index)
            .ok_or_else(|| Error::InvalidArgument("sketch ordinal overflows".into()))?;
        let slot = populated
            .get_mut(ordinal)
            .ok_or_else(|| Error::InvalidArgument("sketch ordinal exceeds entries".into()))?;
        if *slot {
            return Err(Error::InvalidArgument(
                "multiple elements own one sketch ordinal".into(),
            ));
        }
        owners[ordinal] = owner;
        kinds[ordinal] = kind;
        labels[ordinal] = label;
        *slot = true;
    }
    Ok(())
}

struct VectorSpool {
    writer: Option<BufWriter<File>>,
    path: PathBuf,
    byte_len: u64,
    checksum: u32,
    block_len: usize,
    block_checksum: u32,
    block_checksums: Vec<u32>,
}

impl VectorSpool {
    fn create() -> Result<Self> {
        static NEXT_SPOOL: AtomicU64 = AtomicU64::new(0);
        for _ in 0..128 {
            let nonce = NEXT_SPOOL.fetch_add(1, AtomicOrdering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!(".vecgra-spool-{}-{nonce}.tmp", std::process::id()));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    return Ok(Self {
                        writer: Some(BufWriter::with_capacity(1024 * 1024, file)),
                        path,
                        byte_len: 0,
                        checksum: 0,
                        block_len: 0,
                        block_checksum: 0,
                        block_checksums: Vec::new(),
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(Error::Conflict(
            "could not allocate a unique vector spool file".into(),
        ))
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.as_mut().unwrap().write_all(bytes)?;
        self.byte_len = self
            .byte_len
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::InvalidArgument("vector spool length overflow".into()))?;
        self.checksum = crc32c_append(self.checksum, bytes);
        let mut remaining = bytes;
        while !remaining.is_empty() {
            let available = VECTOR_CHECKSUM_BLOCK_SIZE - self.block_len;
            let count = available.min(remaining.len());
            self.block_checksum = crc32c_append(self.block_checksum, &remaining[..count]);
            self.block_len += count;
            remaining = &remaining[count..];
            if self.block_len == VECTOR_CHECKSUM_BLOCK_SIZE {
                self.block_checksums.push(self.block_checksum);
                self.block_len = 0;
                self.block_checksum = 0;
            }
        }
        Ok(())
    }

    fn finish_blocks(&mut self) {
        if self.block_len != 0 {
            self.block_checksums.push(self.block_checksum);
            self.block_len = 0;
            self.block_checksum = 0;
        }
    }

    fn copy_to(&mut self, output: &mut impl Write) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
            writer.into_inner().map_err(|error| error.into_error())?;
        }
        let mut input = File::open(&self.path)?;
        let copied = io::copy(&mut input, output)?;
        if copied != self.byte_len {
            return Err(Error::Corrupt(
                "vector spool length changed before checkpoint write".into(),
            ));
        }
        output.write_all(&self.checksum.to_le_bytes())?;
        Ok(())
    }
}

impl Drop for VectorSpool {
    fn drop(&mut self) {
        let _ = self.writer.take();
        let _ = std::fs::remove_file(&self.path);
    }
}

struct MetadataWriter<'a> {
    writer: BufWriter<&'a mut File>,
    byte_len: u64,
    checksum: u32,
}

impl<'a> MetadataWriter<'a> {
    fn new(file: &'a mut File) -> Result<Self> {
        file.seek(SeekFrom::Start(0))?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);
        writer.write_all(&[0; HEADER_LEN as usize])?;
        Ok(Self {
            writer,
            byte_len: 0,
            checksum: 0,
        })
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes)?;
        self.checksum = crc32c_append(self.checksum, bytes);
        self.byte_len = self
            .byte_len
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::InvalidArgument("metadata spool length overflow".into()))?;
        Ok(())
    }

    fn align(&mut self, alignment: u64) -> Result<()> {
        let padding = (alignment - self.byte_len % alignment) % alignment;
        if padding != 0 {
            const ZEROES: [u8; 64] = [0; 64];
            self.write_all(&ZEROES[..padding as usize])?;
        }
        Ok(())
    }

    fn finish(self) -> Result<u64> {
        let byte_len = self.byte_len;
        let checksum = self.checksum;
        let file = self
            .writer
            .into_inner()
            .map_err(|error| error.into_error())?;
        file.write_all(&checksum.to_le_bytes())?;
        let metadata_len = byte_len
            .checked_add(4)
            .ok_or_else(|| Error::InvalidArgument("metadata checksum length overflows".into()))?;
        Ok(metadata_len)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotLengths {
    pub metadata_len: u64,
    pub vector_len: u64,
}

fn append_columnar_section(
    metadata: &mut MetadataWriter<'_>,
    descriptor: &(u64, u64),
    bytes: &[u8],
) -> Result<()> {
    // Stable cache-line alignment matters for mapped fixed records and CSR;
    // adding a descriptor must not shift every hot column within its lines.
    metadata.align(64)?;
    validate_planned_section(*descriptor, metadata.byte_len, bytes.len())?;
    metadata.write_all(bytes)
}

fn append_node_record_section(
    metadata: &mut MetadataWriter<'_>,
    descriptor: &(u64, u64),
    records: &[SnapshotNodeRecord],
) -> Result<()> {
    begin_columnar_section(metadata, descriptor, records.len(), 48)?;
    let mut encoded = [0u8; 48];
    for record in records {
        encoded[0..8].copy_from_slice(&record.id.to_le_bytes());
        encoded[8..16].copy_from_slice(&record.generation.to_le_bytes());
        encoded[16..24].copy_from_slice(&record.vector_offset.to_le_bytes());
        encoded[24..32].copy_from_slice(&record.property_offset.to_le_bytes());
        encoded[32..36].copy_from_slice(&record.label.to_le_bytes());
        encoded[36..40].copy_from_slice(&record.property_len.to_le_bytes());
        encoded[40..44].copy_from_slice(&record.vector_count.to_le_bytes());
        encoded[44..48].fill(0);
        metadata.write_all(&encoded)?;
    }
    Ok(())
}

fn append_edge_record_section(
    metadata: &mut MetadataWriter<'_>,
    descriptor: &(u64, u64),
    records: &[SnapshotEdgeRecord],
) -> Result<()> {
    begin_columnar_section(metadata, descriptor, records.len(), 64)?;
    let mut encoded = [0u8; 64];
    for record in records {
        encoded[0..8].copy_from_slice(&record.id.to_le_bytes());
        encoded[8..16].copy_from_slice(&record.generation.to_le_bytes());
        encoded[16..24].copy_from_slice(&record.source.to_le_bytes());
        encoded[24..32].copy_from_slice(&record.target.to_le_bytes());
        encoded[32..40].copy_from_slice(&record.vector_offset.to_le_bytes());
        encoded[40..48].copy_from_slice(&record.property_offset.to_le_bytes());
        encoded[48..52].copy_from_slice(&record.label.to_le_bytes());
        encoded[52..56].copy_from_slice(&record.property_len.to_le_bytes());
        encoded[56..60].copy_from_slice(&record.vector_count.to_le_bytes());
        encoded[60..64].fill(0);
        metadata.write_all(&encoded)?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "arguments describe independent offsets in the persisted sketch section"
)]
fn append_sketch_section(
    metadata: &mut MetadataWriter<'_>,
    descriptor: &(u64, u64),
    entry_count: u64,
    words_per_signature: usize,
    owner_offset: usize,
    owner_kind_offset: usize,
    label_offset: usize,
    signature_offset: usize,
    owners: &[u64],
    kinds: &[u8],
    labels: &[u32],
    signatures: Vec<u64>,
) -> Result<()> {
    let entry_count_usize = usize::try_from(entry_count)
        .map_err(|_| Error::InvalidArgument("sketch entry count exceeds usize".into()))?;
    if owners.len() != entry_count_usize
        || kinds.len() != entry_count_usize
        || labels.len() != entry_count_usize
        || signatures.len()
            != entry_count_usize
                .checked_mul(words_per_signature)
                .ok_or_else(|| Error::InvalidArgument("sketch word count overflows".into()))?
    {
        return Err(Error::InvalidArgument(
            "streamed sketch columns have inconsistent lengths".into(),
        ));
    }
    let section_len = signature_offset
        .checked_add(checked_column_len(signatures.len(), 8)?)
        .ok_or_else(|| Error::InvalidArgument("sketch section length overflows".into()))?;
    metadata.align(64)?;
    validate_planned_section(*descriptor, metadata.byte_len, section_len)?;
    let section_start = metadata.byte_len;
    let mut header = [0u8; 24];
    header[..8].copy_from_slice(SKETCH_COLUMNS_MAGIC);
    header[8..16].copy_from_slice(&entry_count.to_le_bytes());
    header[16..20].copy_from_slice(
        &u32::try_from(words_per_signature)
            .map_err(|_| Error::InvalidArgument("sketch width exceeds u32".into()))?
            .to_le_bytes(),
    );
    metadata.write_all(&header)?;
    write_local_padding(metadata, section_start, owner_offset)?;
    for owner in owners {
        metadata.write_all(&owner.to_le_bytes())?;
    }
    write_local_padding(metadata, section_start, owner_kind_offset)?;
    metadata.write_all(kinds)?;
    write_local_padding(metadata, section_start, label_offset)?;
    for label in labels {
        metadata.write_all(&label.to_le_bytes())?;
    }
    write_local_padding(metadata, section_start, signature_offset)?;
    for signature in signatures {
        metadata.write_all(&signature.to_le_bytes())?;
    }
    Ok(())
}

fn write_local_padding(
    metadata: &mut MetadataWriter<'_>,
    section_start: u64,
    target_local_offset: usize,
) -> Result<()> {
    let current = metadata
        .byte_len
        .checked_sub(section_start)
        .ok_or_else(|| Error::InvalidArgument("section cursor precedes its start".into()))?;
    let target = u64::try_from(target_local_offset)
        .map_err(|_| Error::InvalidArgument("section offset exceeds u64".into()))?;
    let padding = target
        .checked_sub(current)
        .ok_or_else(|| Error::InvalidArgument("streamed section columns overlap".into()))?;
    const ZEROES: [u8; 8] = [0; 8];
    if padding > ZEROES.len() as u64 {
        return Err(Error::InvalidArgument(
            "unexpectedly large streamed section padding".into(),
        ));
    }
    metadata.write_all(&ZEROES[..padding as usize])
}

fn append_property_index_section(
    metadata: &mut MetadataWriter<'_>,
    descriptor: &(u64, u64),
    packed_width: u8,
    entries: Vec<(u32, u32, u64)>,
) -> Result<()> {
    metadata.align(64)?;
    let entry_width = 8 + packed_width as usize;
    let length = 24usize
        .checked_add(checked_column_len(entries.len(), entry_width)?)
        .ok_or_else(|| Error::InvalidArgument("property index section overflows".into()))?;
    validate_planned_section(*descriptor, metadata.byte_len, length)?;
    let mut header = [0u8; 24];
    header[..8].copy_from_slice(PROPERTY_INDEX_MAGIC);
    header[8..16].copy_from_slice(
        &u64::try_from(entries.len())
            .map_err(|_| Error::InvalidArgument("property index count exceeds u64".into()))?
            .to_le_bytes(),
    );
    header[16] = packed_width;
    metadata.write_all(&header)?;

    let mut encoded = [0u8; 16];
    for (key, fingerprint, packed_element) in entries {
        encoded[0..4].copy_from_slice(&key.to_le_bytes());
        encoded[4..8].copy_from_slice(&fingerprint.to_le_bytes());
        if packed_width == 4 {
            encoded[8..12].copy_from_slice(&(packed_element as u32).to_le_bytes());
            metadata.write_all(&encoded[..12])?;
        } else {
            encoded[8..16].copy_from_slice(&packed_element.to_le_bytes());
            metadata.write_all(&encoded)?;
        }
    }
    Ok(())
}

fn append_numeric_property_index_section(
    metadata: &mut MetadataWriter<'_>,
    descriptor: &(u64, u64),
    packed_width: u8,
    entries: Vec<(u32, u8, u64, u64)>,
) -> Result<()> {
    metadata.align(64)?;
    let entry_width = 16 + packed_width as usize;
    let length = 24usize
        .checked_add(checked_column_len(entries.len(), entry_width)?)
        .ok_or_else(|| Error::InvalidArgument("numeric property index section overflows".into()))?;
    validate_planned_section(*descriptor, metadata.byte_len, length)?;
    let mut header = [0u8; 24];
    header[..8].copy_from_slice(NUMERIC_PROPERTY_INDEX_MAGIC);
    header[8..16].copy_from_slice(
        &u64::try_from(entries.len())
            .map_err(|_| Error::InvalidArgument("numeric property index count exceeds u64".into()))?
            .to_le_bytes(),
    );
    header[16] = packed_width;
    metadata.write_all(&header)?;

    let mut encoded = [0u8; 24];
    for (key, tag, sortable, packed_element) in entries {
        encoded.fill(0);
        encoded[0..4].copy_from_slice(&key.to_le_bytes());
        encoded[4] = tag;
        encoded[8..16].copy_from_slice(&sortable.to_le_bytes());
        if packed_width == 4 {
            encoded[16..20].copy_from_slice(&(packed_element as u32).to_le_bytes());
            metadata.write_all(&encoded[..20])?;
        } else {
            encoded[16..24].copy_from_slice(&packed_element.to_le_bytes());
            metadata.write_all(&encoded)?;
        }
    }
    Ok(())
}

fn begin_columnar_section(
    metadata: &mut MetadataWriter<'_>,
    descriptor: &(u64, u64),
    count: usize,
    width: usize,
) -> Result<()> {
    metadata.align(64)?;
    let length = count
        .checked_mul(width)
        .ok_or_else(|| Error::InvalidArgument("columnar section length overflows".into()))?;
    validate_planned_section(*descriptor, metadata.byte_len, length)
}

fn validate_planned_section(
    planned: (u64, u64),
    actual_offset: u64,
    actual_length: usize,
) -> Result<()> {
    let actual_length = u64::try_from(actual_length)
        .map_err(|_| Error::InvalidArgument("columnar section length exceeds u64".into()))?;
    if planned != (actual_offset, actual_length) {
        return Err(Error::InvalidArgument(
            "columnar section does not match its precomputed descriptor".into(),
        ));
    }
    Ok(())
}

fn checked_column_len(count: usize, width: usize) -> Result<usize> {
    count
        .checked_mul(width)
        .ok_or_else(|| Error::InvalidArgument("columnar section length overflows".into()))
}

fn plan_columnar_sections(
    lengths: &[usize; RANGE_INDEXED_COLUMNAR_SECTION_COUNT],
) -> Result<[(u64, u64); RANGE_INDEXED_COLUMNAR_SECTION_COUNT]> {
    let mut sections = [(0u64, 0u64); RANGE_INDEXED_COLUMNAR_SECTION_COUNT];
    let mut cursor = RANGE_INDEXED_COLUMNAR_HEADER_LEN;
    for (section, &length) in sections.iter_mut().zip(lengths) {
        cursor = align_up(cursor, 64)?;
        section.0 = u64::try_from(cursor)
            .map_err(|_| Error::InvalidArgument("columnar offset exceeds u64".into()))?;
        section.1 = u64::try_from(length)
            .map_err(|_| Error::InvalidArgument("columnar length exceeds u64".into()))?;
        cursor = cursor
            .checked_add(length)
            .ok_or_else(|| Error::InvalidArgument("columnar metadata length overflows".into()))?;
    }
    Ok(sections)
}

pub(super) fn align_up(value: usize, alignment: usize) -> Result<usize> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| Error::InvalidArgument("aligned section offset overflows".into()))
}

fn append_u64_columnar_section(
    metadata: &mut MetadataWriter<'_>,
    descriptor: &(u64, u64),
    values: &[u64],
) -> Result<()> {
    metadata.align(64)?;
    validate_planned_section(
        *descriptor,
        metadata.byte_len,
        checked_column_len(values.len(), 8)?,
    )?;
    for value in values {
        metadata.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn build_snapshot_csr(
    edges: &[SnapshotEdgeRecord],
    node_slots: u64,
    outgoing: bool,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let slots = usize::try_from(node_slots)
        .map_err(|_| Error::InvalidArgument("node slots exceed usize".into()))?;
    let mut offsets = vec![0u64; slots + 1];
    for edge in edges {
        let endpoint = if outgoing { edge.source } else { edge.target };
        let endpoint = usize::try_from(endpoint)
            .map_err(|_| Error::InvalidArgument("edge endpoint exceeds usize".into()))?;
        if endpoint >= slots {
            return Err(Error::InvalidArgument(format!(
                "edge {} endpoint exceeds checkpoint node slots",
                edge.id
            )));
        }
        offsets[endpoint + 1] += 1;
    }
    for index in 1..offsets.len() {
        offsets[index] = offsets[index]
            .checked_add(offsets[index - 1])
            .ok_or_else(|| Error::InvalidArgument("CSR offset overflow".into()))?;
    }
    let mut ids = vec![0u64; edges.len()];
    let mut cursors = offsets[..slots].to_vec();
    for edge in edges {
        let endpoint = (if outgoing { edge.source } else { edge.target }) as usize;
        let position = cursors[endpoint] as usize;
        ids[position] = edge.id;
        cursors[endpoint] += 1;
    }
    Ok((offsets, ids))
}
