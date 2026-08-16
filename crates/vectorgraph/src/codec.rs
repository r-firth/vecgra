use crate::error::{Error, Result};
use crate::graph::{Operation, SnapshotOperation};
use crate::model::{Edge, Node, Property, Value};
use crate::vector::{Similarity, VectorEncoding};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};

pub(crate) const HEADER_LEN: u64 = 64;
const HEADER_MAGIC: &[u8; 8] = b"VGRPHDB\0";
const SNAPSHOT_MAGIC: &[u8; 8] = b"VGSNAP01";
const COLUMNAR_SNAPSHOT_MAGIC: &[u8; 8] = b"VGSNAP02";
const LEGACY_INDEXED_SNAPSHOT_MAGIC: &[u8; 8] = b"VGSNAP03";
const INDEXED_SNAPSHOT_MAGIC: &[u8; 8] = b"VGSNAP04";
const RANGE_INDEXED_SNAPSHOT_MAGIC: &[u8; 8] = b"VGSNAP05";
const SKETCH_MAGIC: &[u8; 8] = b"VGSIG001";
const SKETCH_COLUMNS_MAGIC: &[u8; 8] = b"VGSIG002";
const PROPERTY_INDEX_MAGIC: &[u8; 8] = b"VGPROP01";
const NUMERIC_PROPERTY_INDEX_MAGIC: &[u8; 8] = b"VGNUM001";
const FRAME_MAGIC: &[u8; 8] = b"VGRTXN01";
const TAIL_MAGIC: u32 = 0x5647_454e;
const FORMAT_VERSION: u32 = 8;
const MAX_FRAME_SIZE: usize = 2 * 1024 * 1024 * 1024;
const COLUMNAR_HEADER_LEN: usize = 192;
const COLUMNAR_SECTION_COUNT: usize = 8;
const LEGACY_INDEXED_COLUMNAR_HEADER_LEN: usize = 208;
const LEGACY_INDEXED_COLUMNAR_SECTION_COUNT: usize = 9;
const INDEXED_COLUMNAR_HEADER_LEN: usize = 224;
const INDEXED_COLUMNAR_SECTION_COUNT: usize = 10;
const RANGE_INDEXED_COLUMNAR_HEADER_LEN: usize = 240;
const RANGE_INDEXED_COLUMNAR_SECTION_COUNT: usize = 11;
pub(crate) const VECTOR_CHECKSUM_BLOCK_SIZE: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Header {
    pub dimension: usize,
    pub similarity: Similarity,
    pub vector_encoding: VectorEncoding,
    pub snapshot_metadata_len: u64,
    pub snapshot_vector_offset: u64,
    pub snapshot_vector_len: u64,
}

impl Header {
    pub(crate) fn log_offset(self) -> u64 {
        if self.snapshot_metadata_len == 0 {
            HEADER_LEN
        } else {
            self.snapshot_vector_offset + self.snapshot_vector_len
        }
    }

    pub(crate) fn has_snapshot(self) -> bool {
        self.snapshot_metadata_len != 0
    }
}

pub(crate) fn write_header(file: &mut (impl Write + Seek), header: Header) -> Result<()> {
    let dimension = u32::try_from(header.dimension)
        .map_err(|_| Error::InvalidArgument("vector dimension exceeds u32".into()))?;
    let mut bytes = [0u8; HEADER_LEN as usize];
    bytes[..8].copy_from_slice(HEADER_MAGIC);
    bytes[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes[12..16].copy_from_slice(&dimension.to_le_bytes());
    bytes[16] = match header.similarity {
        Similarity::Cosine => 1,
        Similarity::Dot => 2,
    };
    bytes[17] = match header.vector_encoding {
        VectorEncoding::F32 => 1,
        VectorEncoding::F16 => 2,
    };
    bytes[24..32].copy_from_slice(&header.snapshot_metadata_len.to_le_bytes());
    bytes[32..40].copy_from_slice(&header.snapshot_vector_offset.to_le_bytes());
    bytes[40..48].copy_from_slice(&header.snapshot_vector_len.to_le_bytes());
    let checksum = crc32c(&bytes[..60]);
    bytes[60..64].copy_from_slice(&checksum.to_le_bytes());
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&bytes)?;
    Ok(())
}

pub(crate) fn read_header(file: &mut (impl Read + Seek)) -> Result<Header> {
    let mut bytes = [0u8; HEADER_LEN as usize];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut bytes)?;
    if &bytes[..8] != HEADER_MAGIC {
        return Err(Error::Corrupt("invalid file magic".into()));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if !(1..=FORMAT_VERSION).contains(&version) {
        return Err(Error::Corrupt(format!(
            "unsupported format version {version}"
        )));
    }
    let expected = u32::from_le_bytes(bytes[60..64].try_into().unwrap());
    if crc32c(&bytes[..60]) != expected {
        return Err(Error::Corrupt("header checksum mismatch".into()));
    }
    let dimension = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if dimension == 0 {
        return Err(Error::Corrupt("vector dimension is zero".into()));
    }
    let similarity = match bytes[16] {
        1 => Similarity::Cosine,
        2 => Similarity::Dot,
        other => return Err(Error::Corrupt(format!("unknown similarity metric {other}"))),
    };
    let vector_encoding = if version == 1 {
        VectorEncoding::F32
    } else {
        match bytes[17] {
            1 => VectorEncoding::F32,
            2 => VectorEncoding::F16,
            other => return Err(Error::Corrupt(format!("unknown vector encoding {other}"))),
        }
    };
    let (snapshot_metadata_len, snapshot_vector_offset, snapshot_vector_len) = if version >= 3 {
        (
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
        )
    } else {
        (0, 0, 0)
    };
    if snapshot_metadata_len != 0
        && (snapshot_metadata_len < 20
            || snapshot_vector_len < 4
            || snapshot_vector_offset != HEADER_LEN + snapshot_metadata_len)
    {
        return Err(Error::Corrupt(
            "invalid checkpoint section offsets in header".into(),
        ));
    }
    Ok(Header {
        dimension,
        similarity,
        vector_encoding,
        snapshot_metadata_len,
        snapshot_vector_offset,
        snapshot_vector_len,
    })
}

pub(crate) fn encode_frame(
    transaction_id: u64,
    operations: &[Operation],
    vector_encoding: VectorEncoding,
) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    for operation in operations {
        encode_operation(&mut payload, operation, vector_encoding)?;
    }
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| Error::InvalidArgument("transaction is too large".into()))?;
    let operation_count = u32::try_from(operations.len())
        .map_err(|_| Error::InvalidArgument("too many transaction operations".into()))?;
    let mut frame = Vec::with_capacity(32 + payload.len() + 8);
    frame.extend_from_slice(FRAME_MAGIC);
    frame.extend_from_slice(&payload_len.to_le_bytes());
    frame.extend_from_slice(&transaction_id.to_le_bytes());
    frame.extend_from_slice(&operation_count.to_le_bytes());
    frame.extend_from_slice(&0u32.to_le_bytes());
    frame.extend_from_slice(&payload);
    let checksum = crc32c(&frame[8..]);
    frame.extend_from_slice(&checksum.to_le_bytes());
    frame.extend_from_slice(&TAIL_MAGIC.to_le_bytes());
    Ok(frame)
}

pub(crate) fn append_frame(file: &mut (impl Write + Seek), frame: &[u8]) -> Result<()> {
    file.seek(SeekFrom::End(0))?;
    file.write_all(frame)?;
    Ok(())
}

#[derive(Debug)]
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

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
fn populate_sketch_owner_columns(
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
            let path = std::env::temp_dir().join(format!(
                ".vectorgraph-spool-{}-{nonce}.tmp",
                std::process::id()
            ));
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

#[allow(clippy::too_many_arguments)]
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

fn align_up(value: usize, alignment: usize) -> Result<usize> {
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

#[derive(Clone, Debug)]
pub(crate) struct SnapshotVectorSection {
    pub byte_offset: usize,
    pub byte_len: usize,
    pub checksum: u32,
    pub block_checksums: Option<Arc<[u32]>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotCsrSection {
    pub byte_offset: usize,
    pub value_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotCsrSections {
    pub out_offsets: SnapshotCsrSection,
    pub out_ids: SnapshotCsrSection,
    pub in_offsets: SnapshotCsrSection,
    pub in_ids: SnapshotCsrSection,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotSketchSection {
    pub byte_offset: usize,
    pub entry_count: usize,
    pub words_per_signature: usize,
    pub owner_columns: Option<SnapshotSketchOwnerColumns>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotSketchOwnerColumns {
    pub owner_byte_offset: usize,
    pub owner_kind_byte_offset: usize,
    pub label_byte_offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotPropertyIndexSection {
    pub byte_offset: usize,
    pub entry_count: usize,
    pub entry_width: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotNumericPropertyIndexSection {
    pub byte_offset: usize,
    pub entry_count: usize,
    pub entry_width: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotRecordSections {
    pub node_byte_offset: usize,
    pub node_count: usize,
    pub node_slots: usize,
    pub edge_byte_offset: usize,
    pub edge_count: usize,
    pub edge_slots: usize,
    pub property_byte_offset: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct SnapshotSections {
    pub vectors: SnapshotVectorSection,
    pub csr: Option<SnapshotCsrSections>,
    pub sketches: Option<SnapshotSketchSection>,
    pub property_index: Option<SnapshotPropertyIndexSection>,
    pub numeric_property_index: Option<SnapshotNumericPropertyIndexSection>,
    pub records: Option<SnapshotRecordSections>,
}

pub(crate) fn read_snapshot(
    file_bytes: &[u8],
    header: Header,
    mut apply: impl FnMut(SnapshotOperation) -> Result<()>,
) -> Result<SnapshotSections> {
    if !header.has_snapshot() {
        return Err(Error::Corrupt("database has no checkpoint snapshot".into()));
    }
    let metadata_len = usize::try_from(header.snapshot_metadata_len)
        .map_err(|_| Error::Corrupt("checkpoint metadata is too large".into()))?;
    let metadata_start = HEADER_LEN as usize;
    let metadata_end = metadata_start
        .checked_add(metadata_len)
        .ok_or_else(|| Error::Corrupt("checkpoint metadata range overflow".into()))?;
    let metadata = file_bytes
        .get(metadata_start..metadata_end)
        .ok_or_else(|| Error::Corrupt("checkpoint metadata exceeds the file".into()))?;
    let content_len = metadata_len - 4;
    let expected = u32::from_le_bytes(metadata[content_len..].try_into().unwrap());
    if crc32c(&metadata[..content_len]) != expected {
        return Err(Error::Corrupt(
            "checkpoint metadata checksum mismatch".into(),
        ));
    }
    let (csr, block_checksums, sketches, property_index, numeric_property_index, records) =
        match &metadata[..8] {
            magic if magic == SNAPSHOT_MAGIC => {
                let operation_count =
                    usize::try_from(u64::from_le_bytes(metadata[8..16].try_into().unwrap()))
                        .map_err(|_| {
                            Error::Corrupt("checkpoint operation count exceeds usize".into())
                        })?;
                decode_snapshot_operations(
                    &metadata[16..content_len],
                    operation_count,
                    metadata_start + 16,
                    &mut apply,
                )?;
                (None, None, None, None, None, None)
            }
            magic if magic == COLUMNAR_SNAPSHOT_MAGIC => {
                let decoded = decode_columnar_snapshot(
                    &metadata[..content_len],
                    metadata_start,
                    false,
                    false,
                    false,
                    header.dimension,
                    &mut apply,
                )?;
                (
                    Some(decoded.csr),
                    decoded.block_checksums,
                    None,
                    None,
                    None,
                    Some(decoded.records),
                )
            }
            magic if magic == LEGACY_INDEXED_SNAPSHOT_MAGIC => {
                let decoded = decode_columnar_snapshot(
                    &metadata[..content_len],
                    metadata_start,
                    true,
                    false,
                    false,
                    header.dimension,
                    &mut apply,
                )?;
                (
                    Some(decoded.csr),
                    decoded.block_checksums,
                    decoded.sketches,
                    None,
                    None,
                    Some(decoded.records),
                )
            }
            magic if magic == INDEXED_SNAPSHOT_MAGIC => {
                let decoded = decode_columnar_snapshot(
                    &metadata[..content_len],
                    metadata_start,
                    true,
                    true,
                    false,
                    header.dimension,
                    &mut apply,
                )?;
                (
                    Some(decoded.csr),
                    decoded.block_checksums,
                    decoded.sketches,
                    decoded.property_index,
                    None,
                    Some(decoded.records),
                )
            }
            magic if magic == RANGE_INDEXED_SNAPSHOT_MAGIC => {
                let decoded = decode_columnar_snapshot(
                    &metadata[..content_len],
                    metadata_start,
                    true,
                    true,
                    true,
                    header.dimension,
                    &mut apply,
                )?;
                (
                    Some(decoded.csr),
                    decoded.block_checksums,
                    decoded.sketches,
                    decoded.property_index,
                    decoded.numeric_property_index,
                    Some(decoded.records),
                )
            }
            _ => return Err(Error::Corrupt("invalid checkpoint metadata magic".into())),
        };

    let vector_len = usize::try_from(header.snapshot_vector_len)
        .map_err(|_| Error::Corrupt("checkpoint vector section is too large".into()))?;
    let vector_offset = usize::try_from(header.snapshot_vector_offset)
        .map_err(|_| Error::Corrupt("checkpoint vector offset exceeds usize".into()))?;
    if let Some(checksums) = &block_checksums {
        let data_len = vector_len - 4;
        let expected_blocks = data_len.div_ceil(VECTOR_CHECKSUM_BLOCK_SIZE);
        if checksums.len() != expected_blocks {
            return Err(Error::Corrupt(
                "vector block checksum count does not match vector section".into(),
            ));
        }
    }
    let checksum_offset = vector_offset
        .checked_add(vector_len - 4)
        .ok_or_else(|| Error::Corrupt("checkpoint vector range overflow".into()))?;
    let checksum = file_bytes
        .get(checksum_offset..checksum_offset + 4)
        .ok_or_else(|| Error::Corrupt("checkpoint vector section exceeds the file".into()))?;
    Ok(SnapshotSections {
        vectors: SnapshotVectorSection {
            byte_offset: vector_offset,
            byte_len: vector_len - 4,
            checksum: u32::from_le_bytes(checksum.try_into().unwrap()),
            block_checksums,
        },
        csr,
        sketches,
        property_index,
        numeric_property_index,
        records,
    })
}

struct DecodedColumnarSnapshot {
    csr: SnapshotCsrSections,
    block_checksums: Option<Arc<[u32]>>,
    sketches: Option<SnapshotSketchSection>,
    property_index: Option<SnapshotPropertyIndexSection>,
    numeric_property_index: Option<SnapshotNumericPropertyIndexSection>,
    records: SnapshotRecordSections,
}

fn decode_columnar_snapshot(
    metadata: &[u8],
    metadata_file_offset: usize,
    indexed: bool,
    property_indexed: bool,
    numeric_property_indexed: bool,
    dimension: usize,
    apply: &mut impl FnMut(SnapshotOperation) -> Result<()>,
) -> Result<DecodedColumnarSnapshot> {
    let header_len = if numeric_property_indexed {
        RANGE_INDEXED_COLUMNAR_HEADER_LEN
    } else if property_indexed {
        INDEXED_COLUMNAR_HEADER_LEN
    } else if indexed {
        LEGACY_INDEXED_COLUMNAR_HEADER_LEN
    } else {
        COLUMNAR_HEADER_LEN
    };
    let section_count = if numeric_property_indexed {
        RANGE_INDEXED_COLUMNAR_SECTION_COUNT
    } else if property_indexed {
        INDEXED_COLUMNAR_SECTION_COUNT
    } else if indexed {
        LEGACY_INDEXED_COLUMNAR_SECTION_COUNT
    } else {
        COLUMNAR_SECTION_COUNT
    };
    if metadata.len() < header_len {
        return Err(Error::Corrupt(
            "columnar checkpoint header is truncated".into(),
        ));
    }
    let node_count = columnar_u64(metadata, 8)?;
    let edge_count = columnar_u64(metadata, 16)?;
    let symbol_count = columnar_u64(metadata, 24)?;
    let node_slots = columnar_u64(metadata, 32)?;
    let edge_slots = columnar_u64(metadata, 40)?;
    let indexed_vectors = columnar_u64(metadata, 48)?;
    let vector_checksum_count = usize::try_from(columnar_u64(metadata, 56)?)
        .map_err(|_| Error::Corrupt("vector block checksum count exceeds usize".into()))?;
    let mut descriptors = vec![(0usize, 0usize); section_count];
    let mut previous_end = header_len;
    for (index, descriptor) in descriptors.iter_mut().enumerate() {
        let start = 64 + index * 16;
        let offset = usize::try_from(columnar_u64(metadata, start)?)
            .map_err(|_| Error::Corrupt("columnar section offset exceeds usize".into()))?;
        let len = usize::try_from(columnar_u64(metadata, start + 8)?)
            .map_err(|_| Error::Corrupt("columnar section length exceeds usize".into()))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::Corrupt("columnar section range overflow".into()))?;
        if offset < header_len
            || offset < previous_end
            || !offset.is_multiple_of(8)
            || end > metadata.len()
        {
            return Err(Error::Corrupt(format!(
                "invalid columnar section {index} range"
            )));
        }
        *descriptor = (offset, len);
        previous_end = end;
    }
    let section = |index: usize| {
        let (offset, len) = descriptors[index];
        &metadata[offset..offset + len]
    };

    let mut symbols = Decoder::new(section(0));
    for expected in 0..symbol_count {
        let id = symbols.u32()?;
        if id as u64 != expected {
            return Err(Error::Corrupt(format!(
                "columnar symbol id {id} is not contiguous"
            )));
        }
        apply(SnapshotOperation::InternSymbol {
            id,
            value: symbols.string()?.into(),
        })?;
    }
    let mut block_checksums = Vec::with_capacity(vector_checksum_count);
    for _ in 0..vector_checksum_count {
        block_checksums.push(symbols.u32()?);
    }
    if !symbols.is_empty() {
        return Err(Error::Corrupt(
            "columnar symbol/checksum section contains trailing bytes".into(),
        ));
    }

    let node_bytes = section(1);
    let expected_node_bytes = usize::try_from(node_count)
        .ok()
        .and_then(|count| count.checked_mul(48))
        .ok_or_else(|| Error::Corrupt("columnar node section size overflow".into()))?;
    if node_bytes.len() != expected_node_bytes {
        return Err(Error::Corrupt(
            "columnar node record count does not match its section".into(),
        ));
    }
    let (_, property_section_len) = descriptors[3];
    let mut counted_vectors = 0u64;
    let mut previous_node = None;
    for record in node_bytes.chunks_exact(48) {
        let id = columnar_u64(record, 0)?;
        if previous_node.is_some_and(|previous| id <= previous) || id >= node_slots {
            return Err(Error::Corrupt(
                "columnar node ids are not strictly ordered within their slots".into(),
            ));
        }
        previous_node = Some(id);
        let property_offset = columnar_u64(record, 24)?;
        let property_len = columnar_u32(record, 36)?;
        validate_property_range(property_offset, property_len, property_section_len)?;
        if columnar_u64(record, 8)? == 0 || columnar_u32(record, 32)? as u64 >= symbol_count {
            return Err(Error::Corrupt(
                "columnar node generation or label is invalid".into(),
            ));
        }
        let vector_count = columnar_u32(record, 40)?;
        counted_vectors = counted_vectors
            .checked_add(vector_count as u64)
            .ok_or_else(|| Error::Corrupt("indexed vector count overflow".into()))?;
        usize::try_from(columnar_u64(record, 16)?)
            .map_err(|_| Error::Corrupt("node vector offset exceeds usize".into()))?;
    }

    let edge_bytes = section(2);
    let expected_edge_bytes = usize::try_from(edge_count)
        .ok()
        .and_then(|count| count.checked_mul(64))
        .ok_or_else(|| Error::Corrupt("columnar edge section size overflow".into()))?;
    if edge_bytes.len() != expected_edge_bytes {
        return Err(Error::Corrupt(
            "columnar edge record count does not match its section".into(),
        ));
    }
    let mut previous_edge = None;
    for record in edge_bytes.chunks_exact(64) {
        let id = columnar_u64(record, 0)?;
        if previous_edge.is_some_and(|previous| id <= previous) || id >= edge_slots {
            return Err(Error::Corrupt(
                "columnar edge ids are not strictly ordered within their slots".into(),
            ));
        }
        previous_edge = Some(id);
        let source = columnar_u64(record, 16)?;
        let target = columnar_u64(record, 24)?;
        if source >= node_slots
            || target >= node_slots
            || !columnar_node_exists(node_bytes, node_count, node_slots, source)
            || !columnar_node_exists(node_bytes, node_count, node_slots, target)
        {
            return Err(Error::Corrupt(
                "columnar edge endpoint exceeds node slots".into(),
            ));
        }
        let property_offset = columnar_u64(record, 40)?;
        let property_len = columnar_u32(record, 52)?;
        validate_property_range(property_offset, property_len, property_section_len)?;
        if columnar_u64(record, 8)? == 0 || columnar_u32(record, 48)? as u64 >= symbol_count {
            return Err(Error::Corrupt(
                "columnar edge generation or label is invalid".into(),
            ));
        }
        let vector_count = columnar_u32(record, 56)?;
        counted_vectors = counted_vectors
            .checked_add(vector_count as u64)
            .ok_or_else(|| Error::Corrupt("indexed vector count overflow".into()))?;
        usize::try_from(columnar_u64(record, 32)?)
            .map_err(|_| Error::Corrupt("edge vector offset exceeds usize".into()))?;
    }
    if counted_vectors != indexed_vectors {
        return Err(Error::Corrupt(
            "columnar indexed vector count does not match records".into(),
        ));
    }

    let expected_offsets = usize::try_from(node_slots)
        .ok()
        .and_then(|slots| slots.checked_add(1))
        .and_then(|count| count.checked_mul(8))
        .ok_or_else(|| Error::Corrupt("columnar CSR offset size overflow".into()))?;
    let expected_ids = usize::try_from(edge_count)
        .ok()
        .and_then(|count| count.checked_mul(8))
        .ok_or_else(|| Error::Corrupt("columnar CSR id size overflow".into()))?;
    if section(4).len() != expected_offsets
        || section(6).len() != expected_offsets
        || section(5).len() != expected_ids
        || section(7).len() != expected_ids
    {
        return Err(Error::Corrupt(
            "columnar CSR section sizes do not match graph counts".into(),
        ));
    }
    validate_csr_sections(section(4), section(5), edge_count, edge_slots)?;
    validate_csr_sections(section(6), section(7), edge_count, edge_slots)?;
    let mapped_section = |index: usize| SnapshotCsrSection {
        byte_offset: metadata_file_offset + descriptors[index].0,
        value_count: descriptors[index].1 / 8,
    };
    let sketches = if indexed {
        let (section_offset, section_len) = descriptors[8];
        let section = section(8);
        if section.len() < 24
            || (&section[..8] != SKETCH_MAGIC && &section[..8] != SKETCH_COLUMNS_MAGIC)
        {
            return Err(Error::Corrupt(
                "indexed checkpoint sketch section is truncated or invalid".into(),
            ));
        }
        let entry_count = usize::try_from(columnar_u64(section, 8)?)
            .map_err(|_| Error::Corrupt("sketch entry count exceeds usize".into()))?;
        let words_per_signature = columnar_u32(section, 16)? as usize;
        let maximum_words = crate::ann::signature_word_count(dimension);
        if entry_count != indexed_vectors as usize
            || words_per_signature == 0
            || words_per_signature > maximum_words
        {
            return Err(Error::Corrupt(
                "sketch header does not match checkpoint vector count".into(),
            ));
        }
        let word_count = entry_count
            .checked_mul(words_per_signature)
            .ok_or_else(|| Error::Corrupt("sketch word count overflow".into()))?;
        let (signature_offset, owner_columns) = if &section[..8] == SKETCH_MAGIC {
            (24usize, None)
        } else {
            let owner_offset = 24usize;
            let owner_kind_offset =
                owner_offset
                    .checked_add(entry_count.checked_mul(8).ok_or_else(|| {
                        Error::Corrupt("sketch owner column length overflow".into())
                    })?)
                    .ok_or_else(|| Error::Corrupt("sketch owner offset overflow".into()))?;
            let label_offset = align_up(
                owner_kind_offset
                    .checked_add(entry_count)
                    .ok_or_else(|| Error::Corrupt("sketch owner-kind offset overflow".into()))?,
                4,
            )
            .map_err(|_| Error::Corrupt("sketch label offset overflow".into()))?;
            let signature_offset = align_up(
                label_offset
                    .checked_add(entry_count.checked_mul(4).ok_or_else(|| {
                        Error::Corrupt("sketch label column length overflow".into())
                    })?)
                    .ok_or_else(|| Error::Corrupt("sketch label offset overflow".into()))?,
                8,
            )
            .map_err(|_| Error::Corrupt("sketch signature offset overflow".into()))?;
            validate_sketch_owner_columns(
                section,
                owner_offset,
                owner_kind_offset,
                label_offset,
                node_bytes,
                edge_bytes,
                dimension,
                entry_count,
                symbol_count,
            )?;
            (
                signature_offset,
                Some(SnapshotSketchOwnerColumns {
                    owner_byte_offset: metadata_file_offset + section_offset + owner_offset,
                    owner_kind_byte_offset: metadata_file_offset
                        + section_offset
                        + owner_kind_offset,
                    label_byte_offset: metadata_file_offset + section_offset + label_offset,
                }),
            )
        };
        let expected_len = signature_offset
            .checked_add(
                word_count
                    .checked_mul(8)
                    .ok_or_else(|| Error::Corrupt("sketch word bytes overflow".into()))?,
            )
            .ok_or_else(|| Error::Corrupt("sketch section length overflow".into()))?;
        if section_len != expected_len {
            return Err(Error::Corrupt(
                "sketch section length does not match its header".into(),
            ));
        }
        let byte_offset = metadata_file_offset + section_offset + signature_offset;
        if !byte_offset.is_multiple_of(8) {
            return Err(Error::Corrupt("sketch word section is not aligned".into()));
        }
        Some(SnapshotSketchSection {
            byte_offset,
            entry_count,
            words_per_signature,
            owner_columns,
        })
    } else {
        None
    };
    let property_index = if property_indexed {
        let (section_offset, section_len) = descriptors[9];
        let property_section = section(9);
        if property_section.len() < 24
            || &property_section[..8] != PROPERTY_INDEX_MAGIC
            || property_section[17..24] != [0; 7]
        {
            return Err(Error::Corrupt(
                "property index header is truncated or invalid".into(),
            ));
        }
        let entry_count = usize::try_from(columnar_u64(property_section, 8)?)
            .map_err(|_| Error::Corrupt("property index count exceeds usize".into()))?;
        let packed_width = match property_section[16] {
            4 => 4usize,
            8 => 8usize,
            _ => return Err(Error::Corrupt("property index ID width is invalid".into())),
        };
        let entry_width = 8 + packed_width;
        let expected_len = 24usize
            .checked_add(
                entry_count
                    .checked_mul(entry_width)
                    .ok_or_else(|| Error::Corrupt("property index length overflows".into()))?,
            )
            .ok_or_else(|| Error::Corrupt("property index section overflows".into()))?;
        if section_len != expected_len {
            return Err(Error::Corrupt(
                "property index length does not match its header".into(),
            ));
        }
        let entries = &property_section[24..];
        validate_property_index(
            entries,
            packed_width,
            node_bytes,
            edge_bytes,
            node_count,
            node_slots,
            edge_count,
            edge_slots,
            symbol_count,
        )?;
        Some(SnapshotPropertyIndexSection {
            byte_offset: metadata_file_offset
                .checked_add(section_offset)
                .and_then(|offset| offset.checked_add(24))
                .ok_or_else(|| Error::Corrupt("property index offset overflow".into()))?,
            entry_count,
            entry_width,
        })
    } else {
        None
    };
    let numeric_property_index = if numeric_property_indexed {
        let (section_offset, section_len) = descriptors[10];
        let numeric_section = section(10);
        if numeric_section.len() < 24
            || &numeric_section[..8] != NUMERIC_PROPERTY_INDEX_MAGIC
            || numeric_section[17..24] != [0; 7]
        {
            return Err(Error::Corrupt(
                "numeric property index header is truncated or invalid".into(),
            ));
        }
        let entry_count = usize::try_from(columnar_u64(numeric_section, 8)?)
            .map_err(|_| Error::Corrupt("numeric property index count exceeds usize".into()))?;
        let packed_width = match numeric_section[16] {
            4 => 4usize,
            8 => 8usize,
            _ => {
                return Err(Error::Corrupt(
                    "numeric property index ID width is invalid".into(),
                ));
            }
        };
        let entry_width = 16 + packed_width;
        let expected_len =
            24usize
                .checked_add(entry_count.checked_mul(entry_width).ok_or_else(|| {
                    Error::Corrupt("numeric property index length overflows".into())
                })?)
                .ok_or_else(|| Error::Corrupt("numeric property index section overflows".into()))?;
        if section_len != expected_len {
            return Err(Error::Corrupt(
                "numeric property index length does not match its header".into(),
            ));
        }
        validate_numeric_property_index(
            &numeric_section[24..],
            packed_width,
            node_bytes,
            edge_bytes,
            node_count,
            node_slots,
            edge_count,
            edge_slots,
            symbol_count,
        )?;
        Some(SnapshotNumericPropertyIndexSection {
            byte_offset: metadata_file_offset
                .checked_add(section_offset)
                .and_then(|offset| offset.checked_add(24))
                .ok_or_else(|| Error::Corrupt("numeric property index offset overflow".into()))?,
            entry_count,
            entry_width,
        })
    } else {
        None
    };
    Ok(DecodedColumnarSnapshot {
        csr: SnapshotCsrSections {
            out_offsets: mapped_section(4),
            out_ids: mapped_section(5),
            in_offsets: mapped_section(6),
            in_ids: mapped_section(7),
        },
        block_checksums: (!block_checksums.is_empty()).then(|| block_checksums.into()),
        sketches,
        property_index,
        numeric_property_index,
        records: SnapshotRecordSections {
            node_byte_offset: metadata_file_offset + descriptors[1].0,
            node_count: usize::try_from(node_count)
                .map_err(|_| Error::Corrupt("node count exceeds usize".into()))?,
            node_slots: usize::try_from(node_slots)
                .map_err(|_| Error::Corrupt("node slots exceed usize".into()))?,
            edge_byte_offset: metadata_file_offset + descriptors[2].0,
            edge_count: usize::try_from(edge_count)
                .map_err(|_| Error::Corrupt("edge count exceeds usize".into()))?,
            edge_slots: usize::try_from(edge_slots)
                .map_err(|_| Error::Corrupt("edge slots exceed usize".into()))?,
            property_byte_offset: metadata_file_offset + descriptors[3].0,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_property_index(
    bytes: &[u8],
    packed_width: usize,
    node_records: &[u8],
    edge_records: &[u8],
    node_count: u64,
    node_slots: u64,
    edge_count: u64,
    edge_slots: u64,
    symbol_count: u64,
) -> Result<()> {
    let entry_width = 8 + packed_width;
    if !bytes.len().is_multiple_of(entry_width) {
        return Err(Error::Corrupt(
            "property index length is not a whole number of entries".into(),
        ));
    }
    let mut previous = None;
    for entry in bytes.chunks_exact(entry_width) {
        let key = columnar_u32(entry, 0)?;
        if key as u64 >= symbol_count {
            return Err(Error::Corrupt(
                "property index entry has invalid key".into(),
            ));
        }
        let fingerprint = columnar_u32(entry, 4)?;
        let packed_element = if packed_width == 4 {
            columnar_u32(entry, 8)? as u64
        } else {
            columnar_u64(entry, 8)?
        };
        let kind = (packed_element & 1) as u8;
        let id = packed_element >> 1;
        let ordering_key = (key, fingerprint, packed_element);
        if previous.is_some_and(|previous| previous >= ordering_key) {
            return Err(Error::Corrupt(
                "property index entries are not strictly ordered".into(),
            ));
        }
        let exists = if kind == 0 {
            columnar_record_exists(node_records, 48, node_count, node_slots, id)
        } else {
            columnar_record_exists(edge_records, 64, edge_count, edge_slots, id)
        };
        if !exists {
            return Err(Error::Corrupt(
                "property index references a missing element".into(),
            ));
        }
        previous = Some(ordering_key);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_numeric_property_index(
    bytes: &[u8],
    packed_width: usize,
    node_records: &[u8],
    edge_records: &[u8],
    node_count: u64,
    node_slots: u64,
    edge_count: u64,
    edge_slots: u64,
    symbol_count: u64,
) -> Result<()> {
    let entry_width = 16 + packed_width;
    if !bytes.len().is_multiple_of(entry_width) {
        return Err(Error::Corrupt(
            "numeric property index length is not a whole number of entries".into(),
        ));
    }
    let mut previous = None;
    for entry in bytes.chunks_exact(entry_width) {
        let key = columnar_u32(entry, 0)?;
        if key as u64 >= symbol_count {
            return Err(Error::Corrupt(
                "numeric property index entry has invalid key".into(),
            ));
        }
        let tag = entry[4];
        if !matches!(tag, 3 | 4) || entry[5..8] != [0; 3] {
            return Err(Error::Corrupt(
                "numeric property index entry has an invalid type".into(),
            ));
        }
        let sortable = columnar_u64(entry, 8)?;
        let packed_element = if packed_width == 4 {
            columnar_u32(entry, 16)? as u64
        } else {
            columnar_u64(entry, 16)?
        };
        let kind = (packed_element & 1) as u8;
        let id = packed_element >> 1;
        let ordering_key = (key, tag, sortable, packed_element);
        if previous.is_some_and(|previous| previous >= ordering_key) {
            return Err(Error::Corrupt(
                "numeric property index entries are not strictly ordered".into(),
            ));
        }
        let exists = if kind == 0 {
            columnar_record_exists(node_records, 48, node_count, node_slots, id)
        } else {
            columnar_record_exists(edge_records, 64, edge_count, edge_slots, id)
        };
        if !exists {
            return Err(Error::Corrupt(
                "numeric property index references a missing element".into(),
            ));
        }
        previous = Some(ordering_key);
    }
    Ok(())
}

fn columnar_record_exists(
    bytes: &[u8],
    record_size: usize,
    count: u64,
    slots: u64,
    id: u64,
) -> bool {
    if id >= slots {
        return false;
    }
    if count == slots {
        return true;
    }
    let mut left = 0usize;
    let mut right = bytes.len() / record_size;
    while left < right {
        let middle = left + (right - left) / 2;
        let start = middle * record_size;
        let candidate = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        match candidate.cmp(&id) {
            std::cmp::Ordering::Less => left = middle + 1,
            std::cmp::Ordering::Greater => right = middle,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn validate_csr_sections(
    offsets: &[u8],
    ids: &[u8],
    edge_count: u64,
    edge_slots: u64,
) -> Result<()> {
    let mut previous = 0u64;
    for (index, chunk) in offsets.chunks_exact(8).enumerate() {
        let offset = u64::from_le_bytes(chunk.try_into().unwrap());
        if (index == 0 && offset != 0) || offset < previous || offset > edge_count {
            return Err(Error::Corrupt(
                "columnar CSR offsets are not monotonic".into(),
            ));
        }
        previous = offset;
    }
    if previous != edge_count {
        return Err(Error::Corrupt(
            "columnar CSR final offset does not equal edge count".into(),
        ));
    }
    if ids
        .chunks_exact(8)
        .any(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) >= edge_slots)
    {
        return Err(Error::Corrupt(
            "columnar CSR contains an edge id outside edge slots".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_sketch_owner_columns(
    section: &[u8],
    owner_offset: usize,
    owner_kind_offset: usize,
    label_offset: usize,
    node_records: &[u8],
    edge_records: &[u8],
    dimension: usize,
    entry_count: usize,
    symbol_count: u64,
) -> Result<()> {
    let owners_len = entry_count
        .checked_mul(8)
        .ok_or_else(|| Error::Corrupt("sketch owner byte length overflow".into()))?;
    let labels_len = entry_count
        .checked_mul(4)
        .ok_or_else(|| Error::Corrupt("sketch label byte length overflow".into()))?;
    let owners_end = owner_offset
        .checked_add(owners_len)
        .ok_or_else(|| Error::Corrupt("sketch owner range overflow".into()))?;
    let owner_kinds_end = owner_kind_offset
        .checked_add(entry_count)
        .ok_or_else(|| Error::Corrupt("sketch owner-kind range overflow".into()))?;
    let labels_end = label_offset
        .checked_add(labels_len)
        .ok_or_else(|| Error::Corrupt("sketch label range overflow".into()))?;
    let owners = section
        .get(owner_offset..owners_end)
        .ok_or_else(|| Error::Corrupt("sketch owner column is truncated".into()))?;
    let owner_kinds = section
        .get(owner_kind_offset..owner_kinds_end)
        .ok_or_else(|| Error::Corrupt("sketch owner-kind column is truncated".into()))?;
    let labels = section
        .get(label_offset..labels_end)
        .ok_or_else(|| Error::Corrupt("sketch label column is truncated".into()))?;
    let mut populated = vec![false; entry_count];
    let mut validate_record =
        |id: u64, label: u32, vector_offset: u64, vector_count: u32, kind: u8| -> Result<()> {
            if label as u64 >= symbol_count || !vector_offset.is_multiple_of(dimension as u64) {
                return Err(Error::Corrupt(
                    "sketch owner record has invalid label or vector offset".into(),
                ));
            }
            let first = usize::try_from(vector_offset / dimension as u64)
                .map_err(|_| Error::Corrupt("sketch owner ordinal exceeds usize".into()))?;
            for vector_index in 0..vector_count as usize {
                let ordinal = first
                    .checked_add(vector_index)
                    .ok_or_else(|| Error::Corrupt("sketch owner ordinal overflow".into()))?;
                let slot = populated
                    .get_mut(ordinal)
                    .ok_or_else(|| Error::Corrupt("sketch owner ordinal exceeds entries".into()))?;
                let owner_start = ordinal * 8;
                let label_start = ordinal * 4;
                if *slot
                    || u64::from_le_bytes(owners[owner_start..owner_start + 8].try_into().unwrap())
                        != id
                    || owner_kinds[ordinal] != kind
                    || u32::from_le_bytes(labels[label_start..label_start + 4].try_into().unwrap())
                        != label
                {
                    return Err(Error::Corrupt(
                        "sketch owner columns disagree with element records".into(),
                    ));
                }
                *slot = true;
            }
            Ok(())
        };
    for record in node_records.chunks_exact(48) {
        validate_record(
            columnar_u64(record, 0)?,
            columnar_u32(record, 32)?,
            columnar_u64(record, 16)?,
            columnar_u32(record, 40)?,
            0,
        )?;
    }
    for record in edge_records.chunks_exact(64) {
        validate_record(
            columnar_u64(record, 0)?,
            columnar_u32(record, 48)?,
            columnar_u64(record, 32)?,
            columnar_u32(record, 56)?,
            1,
        )?;
    }
    if populated.iter().any(|populated| !populated) {
        return Err(Error::Corrupt(
            "sketch owner columns do not densely cover entries".into(),
        ));
    }
    Ok(())
}

fn columnar_node_exists(bytes: &[u8], count: u64, slots: u64, id: u64) -> bool {
    if id >= slots {
        return false;
    }
    if count == slots {
        return true;
    }
    let mut left = 0usize;
    let mut right = bytes.len() / 48;
    while left < right {
        let middle = left + (right - left) / 2;
        let start = middle * 48;
        let candidate = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        match candidate.cmp(&id) {
            std::cmp::Ordering::Less => left = middle + 1,
            std::cmp::Ordering::Greater => right = middle,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn validate_property_range(offset: u64, len: u32, section_len: usize) -> Result<()> {
    let start = usize::try_from(offset)
        .map_err(|_| Error::Corrupt("property offset exceeds usize".into()))?;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| Error::Corrupt("property range overflow".into()))?;
    if len < 4 || end > section_len {
        return Err(Error::Corrupt(
            "property record exceeds columnar property section".into(),
        ));
    }
    Ok(())
}

fn columnar_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::Corrupt("truncated columnar u32".into()))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn columnar_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Error::Corrupt("truncated columnar u64".into()))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

pub(crate) fn replay_frames(
    file: &mut (impl Read + Seek),
    dimension: usize,
    vector_encoding: VectorEncoding,
    start_offset: u64,
    mut apply: impl FnMut(u64, &[Operation]) -> Result<()>,
) -> Result<u64> {
    file.seek(SeekFrom::Start(start_offset))?;
    let mut valid_end = start_offset;
    loop {
        let mut fixed = [0u8; 32];
        match read_exact_or_eof(file, &mut fixed)? {
            ReadStatus::Eof | ReadStatus::Partial => break,
            ReadStatus::Complete => {}
        }
        if &fixed[..8] != FRAME_MAGIC {
            return Err(Error::Corrupt("invalid transaction frame magic".into()));
        }
        let payload_len = u64::from_le_bytes(fixed[8..16].try_into().unwrap()) as usize;
        if payload_len > MAX_FRAME_SIZE {
            return Err(Error::Corrupt(format!(
                "transaction frame is implausibly large: {payload_len} bytes"
            )));
        }
        let transaction_id = u64::from_le_bytes(fixed[16..24].try_into().unwrap());
        let operation_count = u32::from_le_bytes(fixed[24..28].try_into().unwrap()) as usize;
        let mut tail = vec![0u8; payload_len + 8];
        if read_exact_or_eof(file, &mut tail)? != ReadStatus::Complete {
            break;
        }
        let checksum = u32::from_le_bytes(tail[payload_len..payload_len + 4].try_into().unwrap());
        let tail_magic = u32::from_le_bytes(tail[payload_len + 4..].try_into().unwrap());
        if tail_magic != TAIL_MAGIC {
            return Err(Error::Corrupt("transaction tail marker mismatch".into()));
        }
        let mut checksum_input = Vec::with_capacity(24 + payload_len);
        checksum_input.extend_from_slice(&fixed[8..]);
        checksum_input.extend_from_slice(&tail[..payload_len]);
        if crc32c(&checksum_input) != checksum {
            return Err(Error::Corrupt("transaction checksum mismatch".into()));
        }
        let operations = decode_operations(
            &tail[..payload_len],
            operation_count,
            dimension,
            vector_encoding,
        )?;
        apply(transaction_id, &operations)?;
        valid_end = file.stream_position()?;
    }
    Ok(valid_end)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadStatus {
    Complete,
    Eof,
    Partial,
}

fn read_exact_or_eof(reader: &mut impl Read, buffer: &mut [u8]) -> io::Result<ReadStatus> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..])? {
            0 if filled == 0 => return Ok(ReadStatus::Eof),
            0 => return Ok(ReadStatus::Partial),
            count => filled += count,
        }
    }
    Ok(ReadStatus::Complete)
}

fn encode_operation(
    buffer: &mut Vec<u8>,
    operation: &Operation,
    vector_encoding: VectorEncoding,
) -> Result<()> {
    match operation {
        Operation::InternSymbol { id, value } => {
            buffer.push(1);
            put_u32(buffer, *id);
            put_bytes(buffer, value.as_bytes())?;
        }
        Operation::PutNode(node) => {
            buffer.push(2);
            put_u64(buffer, node.id);
            put_u64(buffer, node.generation);
            put_u32(buffer, node.label);
            encode_properties(buffer, &node.properties)?;
            put_u32(buffer, node.vector_count);
            encode_floats(buffer, &node.pending_vectors, vector_encoding);
        }
        Operation::PutEdge(edge) => {
            buffer.push(3);
            put_u64(buffer, edge.id);
            put_u64(buffer, edge.generation);
            put_u64(buffer, edge.source);
            put_u64(buffer, edge.target);
            put_u32(buffer, edge.label);
            encode_properties(buffer, &edge.properties)?;
            put_u32(buffer, edge.vector_count);
            encode_floats(buffer, &edge.pending_vectors, vector_encoding);
        }
        Operation::DeleteNode(id) => {
            buffer.push(4);
            put_u64(buffer, *id);
        }
        Operation::DeleteEdge(id) => {
            buffer.push(5);
            put_u64(buffer, *id);
        }
    }
    Ok(())
}

fn decode_operations(
    bytes: &[u8],
    operation_count: usize,
    dimension: usize,
    vector_encoding: VectorEncoding,
) -> Result<Vec<Operation>> {
    let mut decoder = Decoder::new(bytes);
    let mut operations = Vec::with_capacity(operation_count);
    for _ in 0..operation_count {
        let tag = decoder.u8()?;
        let operation = match tag {
            1 => Operation::InternSymbol {
                id: decoder.u32()?,
                value: decoder.string()?.into(),
            },
            2 => {
                let id = decoder.u64()?;
                let generation = decoder.u64()?;
                let label = decoder.u32()?;
                let properties = decode_properties(&mut decoder)?;
                let vector_count = decoder.u32()?;
                let vectors = decoder
                    .floats(dimension, vector_count, vector_encoding)?
                    .into();
                Operation::PutNode(Node {
                    id,
                    label,
                    properties: properties.into(),
                    vector_count,
                    generation,
                    pending_vectors: vectors,
                })
            }
            3 => {
                let id = decoder.u64()?;
                let generation = decoder.u64()?;
                let source = decoder.u64()?;
                let target = decoder.u64()?;
                let label = decoder.u32()?;
                let properties = decode_properties(&mut decoder)?;
                let vector_count = decoder.u32()?;
                let vectors = decoder
                    .floats(dimension, vector_count, vector_encoding)?
                    .into();
                Operation::PutEdge(Edge {
                    id,
                    generation,
                    source,
                    target,
                    label,
                    properties: properties.into(),
                    vector_count,
                    vector_offset: 0,
                    pending_vectors: vectors,
                })
            }
            4 => Operation::DeleteNode(decoder.u64()?),
            5 => Operation::DeleteEdge(decoder.u64()?),
            other => return Err(Error::Corrupt(format!("unknown operation tag {other}"))),
        };
        operations.push(operation);
    }
    if !decoder.is_empty() {
        return Err(Error::Corrupt(
            "transaction contains trailing payload bytes".into(),
        ));
    }
    Ok(operations)
}

fn decode_snapshot_operations(
    bytes: &[u8],
    operation_count: usize,
    absolute_byte_offset: usize,
    apply: &mut impl FnMut(SnapshotOperation) -> Result<()>,
) -> Result<()> {
    let mut decoder = Decoder::new(bytes);
    for _ in 0..operation_count {
        let operation = match decoder.u8()? {
            1 => SnapshotOperation::InternSymbol {
                id: decoder.u32()?,
                value: decoder.string()?.into(),
            },
            2 => {
                let id = decoder.u64()?;
                let generation = decoder.u64()?;
                let label = decoder.u32()?;
                let (property_byte_offset, property_byte_len) =
                    decoder.validated_properties(absolute_byte_offset)?;
                let vector_count = decoder.u32()?;
                let vector_offset = usize::try_from(decoder.u64()?)
                    .map_err(|_| Error::Corrupt("node vector offset exceeds usize".into()))?;
                SnapshotOperation::PutNode {
                    id,
                    label,
                    vector_count,
                    generation,
                    vector_offset,
                    property_byte_offset,
                    property_byte_len,
                }
            }
            3 => {
                let id = decoder.u64()?;
                let generation = decoder.u64()?;
                let source = decoder.u64()?;
                let target = decoder.u64()?;
                let label = decoder.u32()?;
                let (property_byte_offset, property_byte_len) =
                    decoder.validated_properties(absolute_byte_offset)?;
                let vector_count = decoder.u32()?;
                let vector_offset = usize::try_from(decoder.u64()?)
                    .map_err(|_| Error::Corrupt("edge vector offset exceeds usize".into()))?;
                SnapshotOperation::PutEdge {
                    id,
                    generation,
                    source,
                    target,
                    label,
                    vector_count,
                    vector_offset,
                    property_byte_offset,
                    property_byte_len,
                }
            }
            other => {
                return Err(Error::Corrupt(format!(
                    "unknown checkpoint operation tag {other}"
                )));
            }
        };
        apply(operation)?;
    }
    if !decoder.is_empty() {
        return Err(Error::Corrupt(
            "checkpoint metadata contains trailing bytes".into(),
        ));
    }
    Ok(())
}

pub(crate) fn decode_properties_blob(bytes: &[u8]) -> Result<Vec<Property>> {
    let mut decoder = Decoder::new(bytes);
    let properties = decode_properties(&mut decoder)?;
    if !decoder.is_empty() {
        return Err(Error::Corrupt(
            "mapped property record contains trailing bytes".into(),
        ));
    }
    Ok(properties)
}

/// Tests sorted encoded properties without allocating decoded strings, byte
/// arrays, or a property vector. Checkpoint validation has already established
/// structural integrity, but this remains fallible for reuse by recovery code.
pub(crate) fn properties_blob_matches_all(
    bytes: &[u8],
    predicates: &[(u32, Value)],
) -> Result<bool> {
    for (expected_key, expected_value) in predicates {
        let mut decoder = Decoder::new(bytes);
        let count = decoder.u32()? as usize;
        let mut found = false;
        for _ in 0..count {
            let key = decoder.u32()?;
            if key < *expected_key {
                decoder.validate_value()?;
            } else if key == *expected_key {
                found = decoder.value_matches(expected_value)?;
                break;
            } else {
                return Ok(false);
            }
        }
        if !found {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Reads one sorted numeric property without allocating or hydrating the
/// record. The returned key uses the same order-preserving representation as
/// the mapped numeric posting table.
pub(crate) fn property_blob_numeric_key(
    bytes: &[u8],
    expected_key: u32,
) -> Result<Option<(u8, u64)>> {
    let mut decoder = Decoder::new(bytes);
    let count = decoder.u32()? as usize;
    for _ in 0..count {
        let key = decoder.u32()?;
        if key < expected_key {
            decoder.validate_value()?;
        } else if key > expected_key {
            return Ok(None);
        } else {
            let tag = decoder.u8()?;
            return Ok(match tag {
                3 => Some((tag, (decoder.i64()? as u64) ^ (1u64 << 63))),
                4 => {
                    let value = decoder.f64()?;
                    (!value.is_nan()).then(|| (tag, sortable_f64(value)))
                }
                _ => None,
            });
        }
    }
    Ok(None)
}

/// Stable typed fingerprint used by the exact-verified property posting table.
/// Hash collisions can add candidates but can never add results.
pub(crate) fn property_value_fingerprint(value: &Value) -> u32 {
    match value {
        Value::Null => fingerprint_parts(0, &[]),
        Value::Bool(false) => fingerprint_parts(1, &[]),
        Value::Bool(true) => fingerprint_parts(2, &[]),
        Value::Int(value) => fingerprint_parts(3, &value.to_le_bytes()),
        Value::Float(value) => {
            let canonical = if *value == 0.0 { 0.0 } else { *value };
            fingerprint_parts(4, &canonical.to_le_bytes())
        }
        Value::String(value) => fingerprint_parts(5, value.as_bytes()),
        Value::Bytes(value) => fingerprint_parts(6, value),
        Value::Node(value) => fingerprint_parts(7, &value.to_le_bytes()),
        Value::Edge(value) => fingerprint_parts(8, &value.to_le_bytes()),
    }
}

/// Stable, order-preserving key for same-typed numeric property values.
/// NaNs have no range ordering and are consequently absent from this index.
pub(crate) fn numeric_value_index_key(value: &Value) -> Option<(u8, u64)> {
    match value {
        Value::Int(value) => Some((3, (*value as u64) ^ (1u64 << 63))),
        Value::Float(value) if !value.is_nan() => Some((4, sortable_f64(*value))),
        _ => None,
    }
}

fn sortable_f64(value: f64) -> u64 {
    let value = if value == 0.0 { 0.0 } else { value };
    let bits = value.to_bits();
    if bits & (1u64 << 63) != 0 {
        !bits
    } else {
        bits ^ (1u64 << 63)
    }
}

fn encoded_numeric_property_count(bytes: &[u8]) -> Result<usize> {
    let mut count = 0usize;
    visit_encoded_property_index_keys(bytes, |_, _, numeric| {
        count += usize::from(numeric.is_some());
    })?;
    Ok(count)
}

fn visit_encoded_property_index_keys(
    bytes: &[u8],
    mut visitor: impl FnMut(u32, u32, Option<(u8, u64)>),
) -> Result<()> {
    let mut decoder = Decoder::new(bytes);
    let count = decoder.u32()? as usize;
    for _ in 0..count {
        let key = decoder.u32()?;
        let tag = decoder.u8()?;
        let (fingerprint, numeric) = match tag {
            0..=2 => (fingerprint_parts(tag, &[]), None),
            3 => {
                let bytes = decoder.take(8)?;
                let value = i64::from_le_bytes(bytes.try_into().unwrap());
                (
                    fingerprint_parts(tag, bytes),
                    Some((tag, (value as u64) ^ (1u64 << 63))),
                )
            }
            4 => {
                let encoded = decoder.take(8)?;
                let value = f64::from_le_bytes(encoded.try_into().unwrap());
                let canonical = if value == 0.0 { 0.0 } else { value };
                (
                    fingerprint_parts(tag, &canonical.to_le_bytes()),
                    (!value.is_nan()).then(|| (tag, sortable_f64(value))),
                )
            }
            5 | 6 => (fingerprint_parts(tag, decoder.bytes()?), None),
            7 | 8 => (fingerprint_parts(tag, decoder.take(8)?), None),
            other => return Err(Error::Corrupt(format!("unknown value tag {other}"))),
        };
        visitor(key, fingerprint, numeric);
    }
    if !decoder.is_empty() {
        return Err(Error::Corrupt(
            "encoded property record contains trailing bytes".into(),
        ));
    }
    Ok(())
}

fn fingerprint_parts(tag: u8, bytes: &[u8]) -> u32 {
    // FNV-1a is intentionally simple and portable here. Exact property
    // comparison follows every hit, so this is a locality key, not identity.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash ^= tag as u64;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (hash as u32) ^ (hash >> 32) as u32
}

fn encode_properties(buffer: &mut Vec<u8>, properties: &[Property]) -> Result<()> {
    put_u32(
        buffer,
        u32::try_from(properties.len())
            .map_err(|_| Error::InvalidArgument("too many properties".into()))?,
    );
    for property in properties {
        put_u32(buffer, property.key);
        encode_value(buffer, &property.value)?;
    }
    Ok(())
}

pub(crate) fn encode_properties_blob(properties: &[Property]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    encode_properties(&mut bytes, properties)?;
    Ok(bytes)
}

fn decode_properties(decoder: &mut Decoder<'_>) -> Result<Vec<Property>> {
    let count = decoder.u32()? as usize;
    let mut properties = Vec::with_capacity(count);
    for _ in 0..count {
        properties.push(Property {
            key: decoder.u32()?,
            value: decode_value(decoder)?,
        });
    }
    Ok(properties)
}

fn encode_value(buffer: &mut Vec<u8>, value: &Value) -> Result<()> {
    match value {
        Value::Null => buffer.push(0),
        Value::Bool(false) => buffer.push(1),
        Value::Bool(true) => buffer.push(2),
        Value::Int(value) => {
            buffer.push(3);
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        Value::Float(value) => {
            buffer.push(4);
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        Value::String(value) => {
            buffer.push(5);
            put_bytes(buffer, value.as_bytes())?;
        }
        Value::Bytes(value) => {
            buffer.push(6);
            put_bytes(buffer, value)?;
        }
        Value::Node(value) => {
            buffer.push(7);
            put_u64(buffer, *value);
        }
        Value::Edge(value) => {
            buffer.push(8);
            put_u64(buffer, *value);
        }
    }
    Ok(())
}

fn decode_value(decoder: &mut Decoder<'_>) -> Result<Value> {
    Ok(match decoder.u8()? {
        0 => Value::Null,
        1 => Value::Bool(false),
        2 => Value::Bool(true),
        3 => Value::Int(decoder.i64()?),
        4 => Value::Float(decoder.f64()?),
        5 => Value::String(decoder.string()?.into()),
        6 => Value::Bytes(decoder.bytes()?.to_vec().into()),
        7 => Value::Node(decoder.u64()?),
        8 => Value::Edge(decoder.u64()?),
        other => return Err(Error::Corrupt(format!("unknown value tag {other}"))),
    })
}

fn encode_floats(buffer: &mut Vec<u8>, values: &[f32], encoding: VectorEncoding) {
    match encoding {
        VectorEncoding::F32 => {
            for value in values {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
        }
        VectorEncoding::F16 => {
            for value in values {
                buffer.extend_from_slice(&f32_to_f16(*value).to_le_bytes());
            }
        }
    }
}

fn put_u32(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(buffer: &mut Vec<u8>, value: u64) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(buffer: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    put_u32(
        buffer,
        u32::try_from(value.len())
            .map_err(|_| Error::InvalidArgument("byte value exceeds u32 length".into()))?,
    );
    buffer.extend_from_slice(value);
    Ok(())
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn validated_properties(&mut self, absolute_byte_offset: usize) -> Result<(u64, u32)> {
        let start = self.offset;
        let count = self.u32()? as usize;
        for _ in 0..count {
            self.u32()?;
            self.validate_value()?;
        }
        let absolute = absolute_byte_offset
            .checked_add(start)
            .ok_or_else(|| Error::Corrupt("property byte offset overflow".into()))?;
        let length = self.offset - start;
        Ok((
            u64::try_from(absolute)
                .map_err(|_| Error::Corrupt("property byte offset exceeds u64".into()))?,
            u32::try_from(length)
                .map_err(|_| Error::Corrupt("property byte length exceeds u32".into()))?,
        ))
    }

    fn validate_value(&mut self) -> Result<()> {
        match self.u8()? {
            0..=2 => {}
            3 | 4 | 7 | 8 => {
                self.take(8)?;
            }
            5 => {
                std::str::from_utf8(self.bytes()?)
                    .map_err(|_| Error::Corrupt("invalid UTF-8 string".into()))?;
            }
            6 => {
                self.bytes()?;
            }
            other => return Err(Error::Corrupt(format!("unknown value tag {other}"))),
        }
        Ok(())
    }

    fn value_matches(&mut self, expected: &Value) -> Result<bool> {
        Ok(match self.u8()? {
            0 => matches!(expected, Value::Null),
            1 => matches!(expected, Value::Bool(false)),
            2 => matches!(expected, Value::Bool(true)),
            3 => matches!(expected, Value::Int(value) if *value == self.i64()?),
            4 => matches!(expected, Value::Float(value) if *value == self.f64()?),
            5 => {
                let encoded = self.bytes()?;
                std::str::from_utf8(encoded)
                    .map_err(|_| Error::Corrupt("invalid UTF-8 string".into()))?;
                matches!(expected, Value::String(value) if value.as_bytes() == encoded)
            }
            6 => {
                let encoded = self.bytes()?;
                matches!(expected, Value::Bytes(value) if value.as_ref() == encoded)
            }
            7 => matches!(expected, Value::Node(value) if *value == self.u64()?),
            8 => matches!(expected, Value::Edge(value) if *value == self.u64()?),
            other => return Err(Error::Corrupt(format!("unknown value tag {other}"))),
        })
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| Error::Corrupt("payload offset overflow".into()))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| Error::Corrupt("truncated transaction payload".into()))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let count = self.u32()? as usize;
        self.take(count)
    }

    fn string(&mut self) -> Result<String> {
        String::from_utf8(self.bytes()?.to_vec())
            .map_err(|_| Error::Corrupt("invalid UTF-8 string".into()))
    }

    fn floats(
        &mut self,
        dimension: usize,
        count: u32,
        encoding: VectorEncoding,
    ) -> Result<Vec<f32>> {
        let float_count = dimension
            .checked_mul(count as usize)
            .ok_or_else(|| Error::Corrupt("vector length overflow".into()))?;
        let bytes_per_float = match encoding {
            VectorEncoding::F32 => 4,
            VectorEncoding::F16 => 2,
        };
        let bytes = self.take(
            float_count
                .checked_mul(bytes_per_float)
                .ok_or_else(|| Error::Corrupt("vector byte length overflow".into()))?,
        )?;
        Ok(match encoding {
            VectorEncoding::F32 => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            VectorEncoding::F16 => bytes
                .chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap())))
                .collect(),
        })
    }
}

pub(crate) fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let raw_exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if raw_exponent == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }

    let exponent = raw_exponent - 127 + 15;
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        let mut rounded = mantissa >> shift;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }

    let mut rounded = ((exponent as u32) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
        rounded += 1;
    }
    sign | rounded as u16
}

pub(crate) fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value as u32) & 0x8000) << 16;
    let exponent = ((value >> 10) & 0x1f) as i32;
    let mut mantissa = (value & 0x03ff) as u32;
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut unbiased = -14;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                unbiased -= 1;
            }
            mantissa &= 0x03ff;
            sign | (((unbiased + 127) as u32) << 23) | (mantissa << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | (((exponent - 15 + 127) as u32) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

pub(crate) fn decode_vector_blob(bytes: &[u8], encoding: VectorEncoding) -> Result<Vec<f32>> {
    let values: Vec<f32> = match encoding {
        VectorEncoding::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(Error::Corrupt(
                    "F32 vector section has a partial value".into(),
                ));
            }
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        }
        VectorEncoding::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(Error::Corrupt(
                    "F16 vector section has a partial value".into(),
                ));
            }
            bytes
                .chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap())))
                .collect()
        }
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Corrupt(
            "checkpoint vector section contains a non-finite value".into(),
        ));
    }
    Ok(values)
}

pub(crate) fn crc32c(bytes: &[u8]) -> u32 {
    crc32c_append(0, bytes)
}

fn crc32c_append(previous: u32, bytes: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        // SAFETY: the feature is checked at runtime and the implementation
        // only performs unaligned-safe integer loads from `bytes`.
        return unsafe { crc32c_aarch64(previous, bytes) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.2") {
        // SAFETY: SSE4.2 is checked at runtime. The CRC instructions operate
        // on integer registers; no aligned loads are required.
        return unsafe { crc32c_x86_64(previous, bytes) };
    }
    crc32c_software(previous, bytes)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_aarch64(previous: u32, mut bytes: &[u8]) -> u32 {
    use core::arch::aarch64::{__crc32cb, __crc32cd};

    let mut crc = !previous;
    while bytes.len() >= 8 {
        let word = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        crc = __crc32cd(crc, word);
        bytes = &bytes[8..];
    }
    for &byte in bytes {
        crc = __crc32cb(crc, byte);
    }
    !crc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_x86_64(previous: u32, mut bytes: &[u8]) -> u32 {
    use core::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};

    let mut crc = (!previous) as u64;
    while bytes.len() >= 8 {
        let word = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        crc = _mm_crc32_u64(crc, word);
        bytes = &bytes[8..];
    }
    let mut tail = crc as u32;
    for &byte in bytes {
        tail = _mm_crc32_u8(tail, byte);
    }
    !tail
}

fn crc32c_software(previous: u32, bytes: &[u8]) -> u32 {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (index, slot) in table.iter_mut().enumerate() {
            let mut value = index as u32;
            for _ in 0..8 {
                value = if value & 1 != 0 {
                    0x82f6_3b78 ^ (value >> 1)
                } else {
                    value >> 1
                };
            }
            *slot = value;
        }
        table
    });
    let mut crc = !previous;
    for byte in bytes {
        crc = table[((crc as u8) ^ byte) as usize] ^ (crc >> 8);
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_covers_normal_and_subnormal_values() {
        for value in [
            0.0,
            -0.0,
            1.0,
            -2.0,
            0.333_251_95,
            65_504.0,
            0.000_061_035_156,
            0.000_000_059_604_645,
        ] {
            let decoded = f16_to_f32(f32_to_f16(value));
            let tolerance = value.abs().max(1.0) * 0.001;
            assert!((decoded - value).abs() <= tolerance, "{value} -> {decoded}");
        }
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f16_to_f32(f32_to_f16(f32::INFINITY)), f32::INFINITY);
    }

    #[test]
    fn crc32c_matches_the_castagnoli_check_value_and_fallback() {
        let input = b"123456789";
        assert_eq!(crc32c(input), 0xe306_9283);
        assert_eq!(crc32c(input), crc32c_software(0, input));

        let unaligned = &b"xvectorgraph-checksum"[..][1..];
        assert_eq!(crc32c(unaligned), crc32c_software(0, unaligned));
        let split = crc32c_append(crc32c(&input[..4]), &input[4..]);
        assert_eq!(split, crc32c(input));
    }

    #[test]
    fn encoded_property_predicates_match_without_hydration() {
        let properties = [
            Property {
                key: 1,
                value: Value::Int(42),
            },
            Property {
                key: 3,
                value: Value::String(Arc::from("evidence")),
            },
            Property {
                key: 7,
                value: Value::Bytes(Arc::from(&b"raw"[..])),
            },
        ];
        let encoded = encode_properties_blob(&properties).unwrap();
        assert!(
            properties_blob_matches_all(
                &encoded,
                &[
                    (1, Value::Int(42)),
                    (3, Value::String(Arc::from("evidence"))),
                ],
            )
            .unwrap()
        );
        assert!(!properties_blob_matches_all(&encoded, &[(2, Value::Bool(true))]).unwrap());
        assert!(
            !properties_blob_matches_all(&encoded, &[(3, Value::String(Arc::from("claim")))])
                .unwrap()
        );
        assert!(!properties_blob_matches_all(&encoded, &[(7, Value::Int(0))]).unwrap());
    }

    #[test]
    fn property_fingerprints_match_encoded_values_and_normalize_signed_zero() {
        let properties = [
            Property {
                key: 1,
                value: Value::Float(-0.0),
            },
            Property {
                key: 2,
                value: Value::String(Arc::from("frontier")),
            },
            Property {
                key: 3,
                value: Value::Node(42),
            },
        ];
        let encoded = encode_properties_blob(&properties).unwrap();
        let mut fingerprints = Vec::new();
        visit_encoded_property_index_keys(&encoded, |key, fingerprint, numeric| {
            fingerprints.push((key, fingerprint, numeric));
        })
        .unwrap();
        assert_eq!(fingerprints.len(), properties.len());
        for ((key, fingerprint, _), property) in fingerprints.iter().zip(&properties) {
            assert_eq!(*key, property.key);
            assert_eq!(*fingerprint, property_value_fingerprint(&property.value));
        }
        assert_eq!(
            property_value_fingerprint(&Value::Float(-0.0)),
            property_value_fingerprint(&Value::Float(0.0))
        );
    }

    #[test]
    fn sketch_owner_columns_are_cross_checked_against_records() {
        let entry_count = 2usize;
        let owner_offset = 24usize;
        let owner_kind_offset = owner_offset + entry_count * 8;
        let label_offset = align_up(owner_kind_offset + entry_count, 4).unwrap();
        let mut section = vec![0; label_offset + entry_count * 4];
        let mut owners = vec![0; entry_count];
        let mut kinds = vec![0; entry_count];
        let mut labels = vec![0; entry_count];
        let mut populated = vec![false; entry_count];
        populate_sketch_owner_columns(
            &mut owners,
            &mut kinds,
            &mut labels,
            &mut populated,
            4,
            7,
            0,
            3,
            0,
            2,
        )
        .unwrap();
        for (ordinal, owner) in owners.into_iter().enumerate() {
            let start = owner_offset + ordinal * 8;
            section[start..start + 8].copy_from_slice(&owner.to_le_bytes());
        }
        section[owner_kind_offset..owner_kind_offset + entry_count].copy_from_slice(&kinds);
        for (ordinal, label) in labels.into_iter().enumerate() {
            let start = label_offset + ordinal * 4;
            section[start..start + 4].copy_from_slice(&label.to_le_bytes());
        }

        let mut nodes = Vec::new();
        put_u64(&mut nodes, 7); // id
        put_u64(&mut nodes, 1); // generation
        put_u64(&mut nodes, 0); // vector float offset
        put_u64(&mut nodes, 0); // property offset
        put_u32(&mut nodes, 3); // label
        put_u32(&mut nodes, 0); // property bytes
        put_u32(&mut nodes, 2); // vectors
        put_u32(&mut nodes, 0); // reserved
        assert_eq!(nodes.len(), 48);

        validate_sketch_owner_columns(
            &section,
            owner_offset,
            owner_kind_offset,
            label_offset,
            &nodes,
            &[],
            4,
            entry_count,
            4,
        )
        .unwrap();

        section[owner_offset] ^= 1;
        let error = validate_sketch_owner_columns(
            &section,
            owner_offset,
            owner_kind_offset,
            label_offset,
            &nodes,
            &[],
            4,
            entry_count,
            4,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Corrupt(message) if message.contains("disagree")));
    }
}
