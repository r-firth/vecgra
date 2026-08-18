use crate::error::{Error, Result};
use crate::graph::{Operation, SnapshotOperation};
use crate::model::{Edge, Node, Property, Value};
use crate::vector::{Similarity, VectorEncoding};
pub(crate) use checksum::crc32c;
use checksum::crc32c_append;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

mod checkpoint_read;
mod checkpoint_write;
mod checksum;

#[cfg(test)]
use checkpoint_read::validate_sketch_owner_columns;
pub(crate) use checkpoint_read::{
    SnapshotCsrSection, SnapshotCsrSections, SnapshotNumericPropertyIndexSection,
    SnapshotPropertyIndexSection, SnapshotRecordSections, SnapshotSketchSection, read_snapshot,
};
pub(crate) use checkpoint_write::SnapshotBuilder;
use checkpoint_write::align_up;
#[cfg(test)]
use checkpoint_write::populate_sketch_owner_columns;

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
    pub(crate) fn log_offset(self) -> Result<u64> {
        if self.snapshot_metadata_len == 0 {
            Ok(HEADER_LEN)
        } else {
            self.snapshot_vector_offset
                .checked_add(self.snapshot_vector_len)
                .ok_or_else(|| Error::Corrupt("checkpoint log offset overflow".into()))
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
    if snapshot_metadata_len != 0 {
        let expected_vector_offset = HEADER_LEN.checked_add(snapshot_metadata_len);
        let log_offset = snapshot_vector_offset.checked_add(snapshot_vector_len);
        if snapshot_metadata_len < 20
            || snapshot_vector_len < 4
            || expected_vector_offset != Some(snapshot_vector_offset)
            || log_offset.is_none()
        {
            return Err(Error::Corrupt(
                "invalid checkpoint section offsets in header".into(),
            ));
        }
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
    if payload.len() > MAX_FRAME_SIZE {
        return Err(Error::InvalidArgument(format!(
            "transaction frame exceeds {MAX_FRAME_SIZE} bytes"
        )));
    }
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| Error::InvalidArgument("transaction is too large".into()))?;
    let operation_count = u32::try_from(operations.len())
        .map_err(|_| Error::InvalidArgument("too many transaction operations".into()))?;
    let frame_capacity = payload
        .len()
        .checked_add(40)
        .ok_or_else(|| Error::InvalidArgument("transaction frame length overflow".into()))?;
    let mut frame = Vec::with_capacity(frame_capacity);
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

pub(crate) fn replay_frames(
    file: &mut (impl Read + Seek),
    dimension: usize,
    vector_encoding: VectorEncoding,
    start_offset: u64,
    mut apply: impl FnMut(u64, &[Operation]) -> Result<()>,
) -> Result<u64> {
    let file_end = file.seek(SeekFrom::End(0))?;
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
        let encoded_payload_len = u64::from_le_bytes(fixed[8..16].try_into().unwrap());
        if encoded_payload_len > MAX_FRAME_SIZE as u64 {
            return Err(Error::Corrupt(format!(
                "transaction frame is implausibly large: {encoded_payload_len} bytes"
            )));
        }
        let payload_len = usize::try_from(encoded_payload_len)
            .map_err(|_| Error::Corrupt("transaction frame exceeds usize".into()))?;
        let transaction_id = u64::from_le_bytes(fixed[16..24].try_into().unwrap());
        let operation_count = u32::from_le_bytes(fixed[24..28].try_into().unwrap()) as usize;
        let tail_len = payload_len
            .checked_add(8)
            .ok_or_else(|| Error::Corrupt("transaction tail length overflow".into()))?;
        let tail_start = file.stream_position()?;
        let remaining = file_end.saturating_sub(tail_start);
        if u64::try_from(tail_len).unwrap_or(u64::MAX) > remaining {
            break;
        }
        let mut tail = vec![0u8; tail_len];
        if read_exact_or_eof(file, &mut tail)? != ReadStatus::Complete {
            break;
        }
        let checksum = u32::from_le_bytes(tail[payload_len..payload_len + 4].try_into().unwrap());
        let tail_magic = u32::from_le_bytes(tail[payload_len + 4..].try_into().unwrap());
        if tail_magic != TAIL_MAGIC {
            return Err(Error::Corrupt("transaction tail marker mismatch".into()));
        }
        let calculated_checksum = crc32c_append(crc32c(&fixed[8..]), &tail[..payload_len]);
        if calculated_checksum != checksum {
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
    // Every operation requires at least a tag and eight bytes of payload.
    // Basing the reservation on bytes actually present prevents a corrupt
    // count field from requesting an attacker-sized allocation.
    let mut operations = Vec::with_capacity(operation_count.min(bytes.len() / 9));
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
    // A property needs at least a four-byte key and one-byte value tag.
    let mut properties = Vec::with_capacity(count.min(decoder.remaining() / 5));
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

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
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
