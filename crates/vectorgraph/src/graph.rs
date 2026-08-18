use crate::VectorEncoding;
use crate::ann::{BinarySketchIndex, SketchEntry};
use crate::error::{Error, Result};
use crate::model::{
    Edge, EdgeId, ElementRef, ElementSet, LabelId, Node, NodeId, Property, PropertyKeyId, Value,
};
use crate::vector::{
    self, LateInteractionHit, Similarity, TopK, VectorHit, VectorSearchPlan, VectorSearchStrategy,
    VectorTarget,
};
use memmap2::Mmap;
use roaring::RoaringTreemap;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::num::NonZeroU64;
use std::ops::Bound;
use std::sync::{Arc, OnceLock};

#[derive(Debug)]
enum VectorData {
    Owned(Vec<f32>),
    Mapped {
        map: Arc<Mmap>,
        byte_offset: usize,
        byte_len: usize,
        encoding: VectorEncoding,
        checksum: u32,
        block_checksums: Option<Arc<[u32]>>,
        verified: OnceLock<std::result::Result<(), String>>,
        verified_blocks: Box<[OnceLock<std::result::Result<(), String>>]>,
        decoded: OnceLock<std::result::Result<Vec<f32>, String>>,
        delta: Vec<f32>,
    },
}

#[derive(Debug)]
enum CsrValues {
    Owned(Vec<u64>),
    Mapped {
        map: Arc<Mmap>,
        byte_offset: usize,
        value_count: usize,
    },
}

impl CsrValues {
    fn as_slice(&self) -> &[u64] {
        match self {
            Self::Owned(values) => values,
            Self::Mapped {
                map,
                byte_offset,
                value_count,
            } => {
                let pointer = map[*byte_offset..].as_ptr();
                debug_assert_eq!(pointer.align_offset(std::mem::align_of::<u64>()), 0);
                // SAFETY: v4 checkpoint sections are aligned to eight bytes,
                // their complete ranges were bounds-checked at open, the map
                // outlives this view, and the on-disk integers are native on
                // supported little-endian targets.
                unsafe { std::slice::from_raw_parts(pointer.cast::<u64>(), *value_count) }
            }
        }
    }
}

#[derive(Debug)]
struct CsrAdjacency {
    offsets: CsrValues,
    edge_ids: CsrValues,
}

impl CsrAdjacency {
    fn from_edges(edges: &[Option<StoredEdge>], node_slots: usize, outgoing: bool) -> Result<Self> {
        let mut offsets = vec![0u64; node_slots + 1];
        for edge in edges.iter().flatten() {
            let node = if outgoing { edge.source } else { edge.target };
            let slot = usize::try_from(node)
                .map_err(|_| Error::Corrupt("adjacency node id exceeds usize".into()))?;
            let next = slot
                .checked_add(1)
                .filter(|next| *next < offsets.len())
                .ok_or_else(|| Error::Corrupt("edge endpoint exceeds node slots".into()))?;
            offsets[next] = offsets[next]
                .checked_add(1)
                .ok_or_else(|| Error::Corrupt("adjacency degree overflow".into()))?;
        }
        for slot in 1..offsets.len() {
            offsets[slot] = offsets[slot]
                .checked_add(offsets[slot - 1])
                .ok_or_else(|| Error::Corrupt("adjacency offset overflow".into()))?;
        }
        let edge_count = usize::try_from(*offsets.last().unwrap_or(&0))
            .map_err(|_| Error::Corrupt("adjacency exceeds usize".into()))?;
        let mut edge_ids = vec![0; edge_count];
        let mut cursors = offsets[..node_slots].to_vec();
        for edge in edges.iter().flatten() {
            let node = (if outgoing { edge.source } else { edge.target }) as usize;
            let position = usize::try_from(cursors[node])
                .map_err(|_| Error::Corrupt("adjacency cursor exceeds usize".into()))?;
            edge_ids[position] = edge.id;
            cursors[node] += 1;
        }
        Ok(Self {
            offsets: CsrValues::Owned(offsets),
            edge_ids: CsrValues::Owned(edge_ids),
        })
    }

    fn mapped(
        map: Arc<Mmap>,
        offsets: crate::codec::SnapshotCsrSection,
        edge_ids: crate::codec::SnapshotCsrSection,
    ) -> Result<Self> {
        #[cfg(target_endian = "little")]
        {
            for section in [offsets, edge_ids] {
                let byte_len = section
                    .value_count
                    .checked_mul(8)
                    .ok_or_else(|| Error::Corrupt("mapped CSR byte length overflow".into()))?;
                let end = section
                    .byte_offset
                    .checked_add(byte_len)
                    .ok_or_else(|| Error::Corrupt("mapped CSR range overflow".into()))?;
                let bytes = map
                    .get(section.byte_offset..end)
                    .ok_or_else(|| Error::Corrupt("mapped CSR range exceeds file".into()))?;
                if bytes.as_ptr().align_offset(std::mem::align_of::<u64>()) != 0 {
                    return Err(Error::Corrupt("mapped CSR section is not aligned".into()));
                }
            }
            Ok(Self {
                offsets: CsrValues::Mapped {
                    map: map.clone(),
                    byte_offset: offsets.byte_offset,
                    value_count: offsets.value_count,
                },
                edge_ids: CsrValues::Mapped {
                    map,
                    byte_offset: edge_ids.byte_offset,
                    value_count: edge_ids.value_count,
                },
            })
        }
        #[cfg(target_endian = "big")]
        {
            let decode = |section: crate::codec::SnapshotCsrSection| -> Result<Vec<u64>> {
                let byte_len = section
                    .value_count
                    .checked_mul(8)
                    .ok_or_else(|| Error::Corrupt("mapped CSR byte length overflow".into()))?;
                let bytes = map
                    .get(section.byte_offset..section.byte_offset + byte_len)
                    .ok_or_else(|| Error::Corrupt("mapped CSR range exceeds file".into()))?;
                Ok(bytes
                    .chunks_exact(8)
                    .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect())
            };
            Ok(Self {
                offsets: CsrValues::Owned(decode(offsets)?),
                edge_ids: CsrValues::Owned(decode(edge_ids)?),
            })
        }
    }

    fn get(&self, node: NodeId) -> &[EdgeId] {
        let Ok(node) = usize::try_from(node) else {
            return &[];
        };
        let offsets = self.offsets.as_slice();
        let Some((&start, &end)) = offsets.get(node).zip(offsets.get(node + 1)) else {
            return &[];
        };
        &self.edge_ids.as_slice()[start as usize..end as usize]
    }
}

#[derive(Debug)]
enum MutableAdjacency {
    Dense {
        slots: Vec<Vec<EdgeId>>,
        entries: usize,
    },
    Sparse {
        slots: HashMap<NodeId, Vec<EdgeId>>,
        entries: usize,
    },
}

#[derive(Clone, Copy)]
enum AdjacencySide {
    Outgoing,
    Incoming { skip_self: bool },
}

impl MutableAdjacency {
    fn dense() -> Self {
        Self::Dense {
            slots: Vec::new(),
            entries: 0,
        }
    }

    fn sparse() -> Self {
        Self::Sparse {
            slots: HashMap::new(),
            entries: 0,
        }
    }

    fn grow(&mut self, node: usize) {
        if let Self::Dense { slots, .. } = self {
            grow_adjacency(slots, node);
        }
    }

    fn push(&mut self, node: NodeId, edge: EdgeId) {
        match self {
            Self::Dense { slots, entries } => {
                let node = node as usize;
                grow_adjacency(slots, node);
                slots[node].push(edge);
                *entries += 1;
            }
            Self::Sparse { slots, entries } => {
                slots.entry(node).or_default().push(edge);
                *entries += 1;
            }
        }
    }

    fn get(&self, node: NodeId) -> &[EdgeId] {
        match self {
            Self::Dense { slots, .. } => slots
                .get(node as usize)
                .map(Vec::as_slice)
                .unwrap_or_default(),
            Self::Sparse { slots, .. } => slots.get(&node).map(Vec::as_slice).unwrap_or_default(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        match self {
            Self::Dense { entries, .. } | Self::Sparse { entries, .. } => *entries == 0,
        }
    }

    fn remove(&mut self, node: NodeId, edge: EdgeId) {
        match self {
            Self::Dense { slots, entries } => {
                if let Some(edges) = slots.get_mut(node as usize) {
                    let before = edges.len();
                    edges.retain(|id| *id != edge);
                    *entries -= before - edges.len();
                }
            }
            Self::Sparse { slots, entries } => {
                let remove_slot = if let Some(edges) = slots.get_mut(&node) {
                    let before = edges.len();
                    edges.retain(|id| *id != edge);
                    *entries -= before - edges.len();
                    edges.is_empty()
                } else {
                    false
                };
                if remove_slot {
                    slots.remove(&node);
                }
            }
        }
    }
}

impl VectorData {
    fn base_float_count(&self) -> usize {
        match self {
            Self::Owned(_) => 0,
            Self::Mapped {
                byte_len, encoding, ..
            } => {
                byte_len
                    / match encoding {
                        VectorEncoding::F32 => 4,
                        VectorEncoding::F16 => 2,
                    }
            }
        }
    }

    fn copy_vector(&self, float_offset: usize, output: &mut [f32]) -> Result<()> {
        let end = float_offset
            .checked_add(output.len())
            .ok_or_else(|| Error::Corrupt("vector copy range overflow".into()))?;
        match self {
            Self::Owned(values) => output.copy_from_slice(
                values
                    .get(float_offset..end)
                    .ok_or_else(|| Error::Corrupt("vector copy exceeds owned values".into()))?,
            ),
            Self::Mapped {
                byte_len,
                encoding,
                delta,
                ..
            } => {
                let base_float_count = self.base_float_count();
                if float_offset >= base_float_count {
                    let start = float_offset - base_float_count;
                    output.copy_from_slice(delta.get(start..start + output.len()).ok_or_else(
                        || Error::Corrupt("vector copy exceeds delta values".into()),
                    )?);
                } else {
                    if end > base_float_count {
                        return Err(Error::Corrupt(
                            "vector copy crosses base and delta storage".into(),
                        ));
                    }
                    let bytes_per_value = match encoding {
                        VectorEncoding::F32 => 4,
                        VectorEncoding::F16 => 2,
                    };
                    let byte_offset = float_offset * bytes_per_value;
                    let bytes = self.mapped_range_bytes(
                        byte_offset,
                        output.len().saturating_mul(bytes_per_value),
                    )?;
                    match encoding {
                        VectorEncoding::F32 => {
                            for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(4)) {
                                *value = f32::from_le_bytes(chunk.try_into().unwrap());
                            }
                        }
                        VectorEncoding::F16 => {
                            for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                                *value = crate::codec::f16_to_f32(u16::from_le_bytes(
                                    chunk.try_into().unwrap(),
                                ));
                            }
                        }
                    }
                    debug_assert_eq!(base_float_count * bytes_per_value, *byte_len);
                }
            }
        }
        Ok(())
    }

    fn raw_mapped_bytes(&self) -> Result<Option<(&[u8], VectorEncoding)>> {
        let Self::Mapped {
            map,
            byte_offset,
            byte_len,
            encoding,
            ..
        } = self
        else {
            return Ok(None);
        };
        let end = byte_offset
            .checked_add(*byte_len)
            .ok_or_else(|| Error::Corrupt("mapped vector range overflow".into()))?;
        let bytes = map
            .get(*byte_offset..end)
            .ok_or_else(|| Error::Corrupt("mapped vector range exceeds the file".into()))?;
        Ok(Some((bytes, *encoding)))
    }

    fn mapped_range_bytes(&self, start: usize, len: usize) -> Result<&[u8]> {
        let Self::Mapped {
            checksum,
            block_checksums,
            verified,
            verified_blocks,
            ..
        } = self
        else {
            return Err(Error::Corrupt(
                "mapped vector range requested from owned storage".into(),
            ));
        };
        let (bytes, _) = self.raw_mapped_bytes()?.unwrap();
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::Corrupt("mapped vector subrange overflow".into()))?;
        let range = bytes
            .get(start..end)
            .ok_or_else(|| Error::Corrupt("mapped vector subrange exceeds section".into()))?;
        if let Some(checksums) = block_checksums {
            if len != 0 {
                let first = start / crate::codec::VECTOR_CHECKSUM_BLOCK_SIZE;
                let last = (end - 1) / crate::codec::VECTOR_CHECKSUM_BLOCK_SIZE;
                for block in first..=last {
                    verified_blocks[block]
                        .get_or_init(|| {
                            let block_start = block * crate::codec::VECTOR_CHECKSUM_BLOCK_SIZE;
                            let block_end = (block_start
                                + crate::codec::VECTOR_CHECKSUM_BLOCK_SIZE)
                                .min(bytes.len());
                            (crate::codec::crc32c(&bytes[block_start..block_end])
                                == checksums[block])
                                .then_some(())
                                .ok_or_else(|| {
                                    format!("mapped vector block {block} checksum mismatch")
                                })
                        })
                        .as_ref()
                        .map_err(|message| Error::Corrupt(message.clone()))?;
                }
            }
        } else {
            verified
                .get_or_init(|| {
                    (crate::codec::crc32c(bytes) == *checksum)
                        .then_some(())
                        .ok_or_else(|| "mapped vector section checksum mismatch".to_owned())
                })
                .as_ref()
                .map_err(|message| Error::Corrupt(message.clone()))?;
        }
        Ok(range)
    }

    fn mapped_bytes(&self) -> Result<Option<(&[u8], VectorEncoding)>> {
        let Some((bytes, encoding)) = self.raw_mapped_bytes()? else {
            return Ok(None);
        };
        self.mapped_range_bytes(0, bytes.len())?;
        Ok(Some((bytes, encoding)))
    }

    fn verify_all_bytes(&self) -> Result<(usize, usize)> {
        match self {
            Self::Owned(values) => Ok((values.len().saturating_mul(size_of::<f32>()), 0)),
            Self::Mapped {
                block_checksums, ..
            } => {
                let bytes = self
                    .mapped_bytes()?
                    .expect("mapped vector data returns mapped bytes")
                    .0
                    .len();
                Ok((
                    bytes,
                    block_checksums.as_ref().map_or(1, |blocks| blocks.len()),
                ))
            }
        }
    }

    #[cfg(target_endian = "little")]
    fn mapped_f32_range(&self, float_offset: usize, float_count: usize) -> Result<&[f32]> {
        let byte_offset = float_offset
            .checked_mul(4)
            .ok_or_else(|| Error::Corrupt("mapped F32 offset overflows".into()))?;
        let byte_len = float_count
            .checked_mul(4)
            .ok_or_else(|| Error::Corrupt("mapped F32 length overflows".into()))?;
        let bytes = self.mapped_range_bytes(byte_offset, byte_len)?;
        if bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) != 0 {
            return Err(Error::Corrupt(
                "mapped F32 vector section is not naturally aligned".into(),
            ));
        }
        // SAFETY: the checkpoint reader validates the complete vector range,
        // this method verifies the requested checksum blocks, F32 storage is
        // little-endian on this target, and alignment was checked above.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), float_count) })
    }

    #[cfg(target_endian = "little")]
    fn mapped_f32_slice(&self) -> Result<&[f32]> {
        let (bytes, encoding) = self
            .mapped_bytes()?
            .ok_or_else(|| Error::Corrupt("mapped F32 data is not mapped".into()))?;
        if encoding != VectorEncoding::F32
            || bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) != 0
        {
            return Err(Error::Corrupt(
                "mapped F32 vector section has invalid encoding or alignment".into(),
            ));
        }
        // SAFETY: `mapped_bytes` verifies the whole section, F32 is native
        // endian on this target, the byte length was validated at open, and
        // alignment was checked above.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) })
    }

    fn scorer(&self, prefer_f32_cache: bool) -> Result<VectorScorer<'_>> {
        match self {
            Self::Owned(values) => Ok(VectorScorer::F32(values)),
            Self::Mapped {
                encoding: VectorEncoding::F16,
                decoded,
                delta,
                byte_len,
                ..
            } => {
                let base_float_count = *byte_len / 2;
                if prefer_f32_cache && let Some(decoded) = decoded.get() {
                    return Ok(VectorScorer::LayeredF32 {
                        base: decoded
                            .as_ref()
                            .map(Vec::as_slice)
                            .map_err(|message| Error::Corrupt(message.clone()))?,
                        delta,
                        base_float_count,
                    });
                }
                Ok(VectorScorer::LayeredF16 {
                    data: self,
                    delta,
                    base_float_count,
                })
            }
            Self::Mapped {
                encoding: VectorEncoding::F32,
                delta,
                byte_len,
                ..
            } => {
                #[cfg(target_endian = "little")]
                {
                    if prefer_f32_cache {
                        Ok(VectorScorer::LayeredF32 {
                            base: self.mapped_f32_slice()?,
                            delta,
                            base_float_count: *byte_len / 4,
                        })
                    } else {
                        Ok(VectorScorer::LayeredMappedF32 {
                            data: self,
                            delta,
                            base_float_count: *byte_len / 4,
                        })
                    }
                }
                #[cfg(target_endian = "big")]
                {
                    Ok(VectorScorer::LayeredF32 {
                        base: self.as_slice()?,
                        delta,
                        base_float_count: *byte_len / 4,
                    })
                }
            }
        }
    }

    /// Materializes an F32 copy of a mapped F16 base on explicit request.
    /// Returning the allocation size makes the memory tradeoff visible to the
    /// caller. Ordinary searches never trigger this potentially large and
    /// latency-spiky conversion implicitly.
    fn warm_f32_cache(&self) -> Result<usize> {
        match self {
            Self::Owned(_) => Ok(0),
            Self::Mapped {
                encoding: VectorEncoding::F16,
                byte_len,
                ..
            } => {
                self.as_slice()?;
                Ok((*byte_len / 2).saturating_mul(4))
            }
            Self::Mapped {
                encoding: VectorEncoding::F32,
                ..
            } => Ok(0),
        }
    }

    fn f32_cache_bytes(&self) -> usize {
        match self {
            Self::Owned(_) => 0,
            Self::Mapped { decoded, .. } => decoded
                .get()
                .and_then(|result| result.as_ref().ok())
                .map_or(0, |values| values.len().saturating_mul(4)),
        }
    }

    fn snapshot_range(
        &self,
        float_offset: usize,
        float_count: usize,
        target: VectorEncoding,
    ) -> Result<SnapshotVectorRange<'_>> {
        match self {
            Self::Owned(values) => {
                let end = float_offset
                    .checked_add(float_count)
                    .ok_or_else(|| Error::Corrupt("snapshot vector range overflow".into()))?;
                Ok(SnapshotVectorRange::F32(
                    values.get(float_offset..end).ok_or_else(|| {
                        Error::Corrupt("snapshot vector range exceeds owned values".into())
                    })?,
                ))
            }
            Self::Mapped {
                encoding,
                byte_len,
                delta,
                ..
            } => {
                let base_float_count = *byte_len
                    / match encoding {
                        VectorEncoding::F32 => 4,
                        VectorEncoding::F16 => 2,
                    };
                let end_float = float_offset
                    .checked_add(float_count)
                    .ok_or_else(|| Error::Corrupt("snapshot vector range overflow".into()))?;
                if float_offset >= base_float_count {
                    let start = float_offset - base_float_count;
                    let end = end_float - base_float_count;
                    return Ok(SnapshotVectorRange::F32(delta.get(start..end).ok_or_else(
                        || Error::Corrupt("snapshot delta vector range exceeds values".into()),
                    )?));
                }
                if end_float > base_float_count {
                    return Err(Error::Corrupt(
                        "an element vector range crosses base and delta storage".into(),
                    ));
                }
                let source = *encoding;
                let bytes_per_float = match source {
                    VectorEncoding::F32 => 4,
                    VectorEncoding::F16 => 2,
                };
                let start = float_offset
                    .checked_mul(bytes_per_float)
                    .ok_or_else(|| Error::Corrupt("snapshot vector byte offset overflow".into()))?;
                let end = float_count
                    .checked_mul(bytes_per_float)
                    .and_then(|len| start.checked_add(len))
                    .ok_or_else(|| Error::Corrupt("snapshot vector byte range overflow".into()))?;
                let encoded = self.mapped_range_bytes(start, end - start)?;
                if *encoding == target {
                    Ok(SnapshotVectorRange::Encoded(encoded))
                } else {
                    Ok(SnapshotVectorRange::OwnedF32(
                        crate::codec::decode_vector_blob(encoded, source)?,
                    ))
                }
            }
        }
    }

    fn as_slice(&self) -> Result<&[f32]> {
        match self {
            Self::Owned(values) => Ok(values),
            Self::Mapped {
                encoding, decoded, ..
            } => decoded
                .get_or_init(|| {
                    let (bytes, _) = self
                        .mapped_bytes()
                        .map_err(|error| error.to_string())?
                        .unwrap();
                    crate::codec::decode_vector_blob(bytes, *encoding)
                        .map_err(|error| error.to_string())
                })
                .as_ref()
                .map(Vec::as_slice)
                .map_err(|message| Error::Corrupt(message.clone())),
        }
    }

    fn f32_range(&self, float_offset: usize, float_count: usize) -> Result<Option<&[f32]>> {
        let end = float_offset
            .checked_add(float_count)
            .ok_or_else(|| Error::Corrupt("vector range overflow".into()))?;
        match self {
            Self::Owned(values) => Ok(values.get(float_offset..end)),
            Self::Mapped {
                byte_len,
                encoding,
                delta,
                ..
            } => {
                let base_float_count = *byte_len
                    / match encoding {
                        VectorEncoding::F32 => 4,
                        VectorEncoding::F16 => 2,
                    };
                if float_offset >= base_float_count {
                    return Ok(delta.get(float_offset - base_float_count..end - base_float_count));
                }
                if end > base_float_count {
                    return Err(Error::Corrupt(
                        "an element vector range crosses base and delta storage".into(),
                    ));
                }
                #[cfg(target_endian = "little")]
                if *encoding == VectorEncoding::F32 {
                    return self.mapped_f32_range(float_offset, float_count).map(Some);
                }
                Ok(self.as_slice()?.get(float_offset..end))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum VectorScorer<'a> {
    F32(&'a [f32]),
    LayeredF32 {
        base: &'a [f32],
        delta: &'a [f32],
        base_float_count: usize,
    },
    LayeredF16 {
        data: &'a VectorData,
        delta: &'a [f32],
        base_float_count: usize,
    },
    #[cfg(target_endian = "little")]
    LayeredMappedF32 {
        data: &'a VectorData,
        delta: &'a [f32],
        base_float_count: usize,
    },
}

enum SnapshotVectorRange<'a> {
    Encoded(&'a [u8]),
    F32(&'a [f32]),
    OwnedF32(Vec<f32>),
}

impl VectorScorer<'_> {
    #[inline]
    fn score(&self, query: &[f32], float_offset: usize) -> Result<f32> {
        Ok(match self {
            Self::F32(values) => {
                vector::score(query, &values[float_offset..float_offset + query.len()])
            }
            Self::LayeredF32 {
                base,
                delta,
                base_float_count,
            } => {
                if float_offset >= *base_float_count {
                    let start = float_offset - *base_float_count;
                    vector::score(query, &delta[start..start + query.len()])
                } else {
                    vector::score(query, &base[float_offset..float_offset + query.len()])
                }
            }
            Self::LayeredF16 {
                data,
                delta,
                base_float_count,
            } => {
                if float_offset >= *base_float_count {
                    let start = float_offset - *base_float_count;
                    vector::score(query, &delta[start..start + query.len()])
                } else {
                    let start = float_offset * 2;
                    crate::simd::dot_f16(query, data.mapped_range_bytes(start, query.len() * 2)?)
                }
            }
            #[cfg(target_endian = "little")]
            Self::LayeredMappedF32 {
                data,
                delta,
                base_float_count,
            } => {
                if float_offset >= *base_float_count {
                    let start = float_offset - *base_float_count;
                    vector::score(query, &delta[start..start + query.len()])
                } else {
                    vector::score(query, data.mapped_f32_range(float_offset, query.len())?)
                }
            }
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct StoredProperties {
    /// Mapped byte offset, or `OWNED_PROPERTY_BIT | arena_index`.
    location: u64,
    byte_len: u32,
}

const OWNED_PROPERTY_BIT: u64 = 1 << 63;

impl StoredProperties {
    fn owned(arena_index: usize) -> Result<Self> {
        let arena_index = u64::try_from(arena_index)
            .map_err(|_| Error::InvalidArgument("owned property arena exceeds u64".into()))?;
        if arena_index & OWNED_PROPERTY_BIT != 0 {
            return Err(Error::InvalidArgument(
                "owned property arena exceeds compact reference range".into(),
            ));
        }
        Ok(Self {
            location: arena_index | OWNED_PROPERTY_BIT,
            byte_len: 0,
        })
    }

    fn mapped(byte_offset: u64, byte_len: u32) -> Self {
        debug_assert_eq!(byte_offset & OWNED_PROPERTY_BIT, 0);
        Self {
            location: byte_offset,
            byte_len,
        }
    }

    fn get(&self, map: Option<&Mmap>, owned: &[Arc<[Property]>]) -> Arc<[Property]> {
        if self.location & OWNED_PROPERTY_BIT != 0 {
            let index = (self.location & !OWNED_PROPERTY_BIT) as usize;
            owned[index].clone()
        } else {
            let map = map.expect("mapped properties require their checkpoint mapping");
            let start = usize::try_from(self.location)
                .expect("validated property offset fits this platform");
            let end = start + self.byte_len as usize;
            crate::codec::decode_properties_blob(&map[start..end])
                .expect("mapped properties were validated while opening the checkpoint")
                .into()
        }
    }

    fn matches(
        &self,
        predicates: &[(PropertyKeyId, Value)],
        map: Option<&Mmap>,
        owned: &[Arc<[Property]>],
    ) -> bool {
        if predicates.is_empty() {
            return true;
        }
        if self.location & OWNED_PROPERTY_BIT != 0 {
            let index = (self.location & !OWNED_PROPERTY_BIT) as usize;
            let properties = &owned[index];
            predicates.iter().all(|(key, expected)| {
                properties
                    .binary_search_by_key(key, |property| property.key)
                    .ok()
                    .is_some_and(|index| properties[index].value == *expected)
            })
        } else {
            let map = map.expect("mapped properties require their checkpoint mapping");
            let start = usize::try_from(self.location)
                .expect("validated property offset fits this platform");
            let end = start + self.byte_len as usize;
            crate::codec::properties_blob_matches_all(&map[start..end], predicates)
                .expect("mapped properties were validated while opening the checkpoint")
        }
    }

    fn numeric_key(
        &self,
        key: PropertyKeyId,
        map: Option<&Mmap>,
        owned: &[Arc<[Property]>],
    ) -> Option<(u8, u64)> {
        if self.location & OWNED_PROPERTY_BIT != 0 {
            let index = (self.location & !OWNED_PROPERTY_BIT) as usize;
            let properties = &owned[index];
            properties
                .binary_search_by_key(&key, |property| property.key)
                .ok()
                .and_then(|index| crate::codec::numeric_value_index_key(&properties[index].value))
        } else {
            let map = map.expect("mapped properties require their checkpoint mapping");
            let start = usize::try_from(self.location)
                .expect("validated property offset fits this platform");
            let end = start + self.byte_len as usize;
            crate::codec::property_blob_numeric_key(&map[start..end], key)
                .expect("mapped properties were validated while opening the checkpoint")
        }
    }

    fn encoded<'a>(
        &'a self,
        map: Option<&'a Mmap>,
        owned: &[Arc<[Property]>],
    ) -> Result<Cow<'a, [u8]>> {
        if self.location & OWNED_PROPERTY_BIT != 0 {
            let index = (self.location & !OWNED_PROPERTY_BIT) as usize;
            Ok(Cow::Owned(crate::codec::encode_properties_blob(
                &owned[index],
            )?))
        } else {
            let map = map.ok_or_else(|| {
                Error::Corrupt("mapped properties are missing their checkpoint map".into())
            })?;
            let start = usize::try_from(self.location)
                .map_err(|_| Error::Corrupt("property offset exceeds usize".into()))?;
            let end = start
                .checked_add(self.byte_len as usize)
                .ok_or_else(|| Error::Corrupt("property byte range overflow".into()))?;
            Ok(Cow::Borrowed(map.get(start..end).ok_or_else(|| {
                Error::Corrupt("property byte range exceeds checkpoint".into())
            })?))
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct StoredNode {
    id: NodeId,
    label: LabelId,
    properties: StoredProperties,
    vector_count: u32,
    generation: NonZeroU64,
    vector_offset: usize,
}

#[derive(Clone, Copy, Debug)]
struct StoredEdge {
    id: EdgeId,
    source: NodeId,
    target: NodeId,
    label: LabelId,
    properties: StoredProperties,
    vector_count: u32,
    generation: NonZeroU64,
    vector_offset: usize,
}

#[derive(Debug)]
struct MappedNodeRecords {
    map: Arc<Mmap>,
    byte_offset: usize,
    count: usize,
    slots: usize,
    property_byte_offset: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DiskNodeRecord {
    id: u64,
    generation: u64,
    vector_offset: u64,
    property_offset: u64,
    label: u32,
    property_len: u32,
    vector_count: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DiskEdgeRecord {
    id: u64,
    generation: u64,
    source: u64,
    target: u64,
    vector_offset: u64,
    property_offset: u64,
    label: u32,
    property_len: u32,
    vector_count: u32,
    reserved: u32,
}

// The on-disk structs are read directly on little-endian systems. Keep these
// assertions beside the definitions so a future field edit cannot silently
// invalidate the portable file layout or its eight-byte section alignment.
const _: () = assert!(std::mem::size_of::<DiskNodeRecord>() == 48);
const _: () = assert!(std::mem::align_of::<DiskNodeRecord>() <= 8);
const _: () = assert!(std::mem::size_of::<DiskEdgeRecord>() == 64);
const _: () = assert!(std::mem::align_of::<DiskEdgeRecord>() <= 8);

impl MappedNodeRecords {
    #[inline(always)]
    fn disk_record_at(&self, index: usize) -> DiskNodeRecord {
        #[cfg(target_endian = "little")]
        {
            let pointer = self.map[self.byte_offset..].as_ptr();
            debug_assert_eq!(
                pointer.align_offset(std::mem::align_of::<DiskNodeRecord>()),
                0
            );
            // SAFETY: the columnar decoder validates the aligned complete
            // fixed-width section, and the read-only map outlives this table.
            unsafe { *pointer.cast::<DiskNodeRecord>().add(index) }
        }
        #[cfg(target_endian = "big")]
        {
            let start = self.byte_offset + index * 48;
            let bytes = &self.map[start..start + 48];
            DiskNodeRecord {
                id: record_u64(bytes, 0),
                generation: record_u64(bytes, 8),
                vector_offset: record_u64(bytes, 16),
                property_offset: record_u64(bytes, 24),
                label: record_u32(bytes, 32),
                property_len: record_u32(bytes, 36),
                vector_count: record_u32(bytes, 40),
                reserved: record_u32(bytes, 44),
            }
        }
    }

    #[inline]
    fn contains(&self, id: NodeId) -> bool {
        let Ok(index) = usize::try_from(id) else {
            return false;
        };
        if self.count == self.slots {
            // Strictly increasing IDs, count == slots, and id < slots are
            // validated by the codec, so a dense checkpoint contains 0..N.
            return index < self.count;
        }
        self.get(id).is_some()
    }

    fn record_at(&self, index: usize) -> StoredNode {
        let record = self.disk_record_at(index);
        StoredNode {
            id: record.id,
            generation: NonZeroU64::new(record.generation).unwrap(),
            vector_offset: record.vector_offset as usize,
            properties: StoredProperties::mapped(
                (self.property_byte_offset as u64) + record.property_offset,
                record.property_len,
            ),
            label: record.label,
            vector_count: record.vector_count,
        }
    }

    fn get(&self, id: NodeId) -> Option<StoredNode> {
        let id = usize::try_from(id).ok()?;
        if self.count == self.slots && id < self.count {
            let record = self.record_at(id);
            return (record.id as usize == id).then_some(record);
        }
        let mut left = 0usize;
        let mut right = self.count;
        while left < right {
            let middle = left + (right - left) / 2;
            match self.record_at(middle).id.cmp(&(id as u64)) {
                Ordering::Less => left = middle + 1,
                Ordering::Greater => right = middle,
                Ordering::Equal => return Some(self.record_at(middle)),
            }
        }
        None
    }

    fn iter(&self) -> impl Iterator<Item = StoredNode> + '_ {
        (0..self.count).map(|index| self.record_at(index))
    }

    #[inline(always)]
    fn vector_fields_at(&self, index: usize) -> (NodeId, LabelId, usize, u32) {
        let record = self.disk_record_at(index);
        (
            record.id,
            record.label,
            record.vector_offset as usize,
            record.vector_count,
        )
    }

    #[inline(always)]
    fn vector_fields(&self, id: NodeId) -> Option<(LabelId, usize, u32)> {
        let index = usize::try_from(id).ok()?;
        if self.count == self.slots && index < self.count {
            let record = self.disk_record_at(index);
            return (record.id == id).then_some((
                record.label,
                record.vector_offset as usize,
                record.vector_count,
            ));
        }
        self.get(id)
            .map(|record| (record.label, record.vector_offset, record.vector_count))
    }
}

#[derive(Debug)]
struct MappedEdgeRecords {
    map: Arc<Mmap>,
    byte_offset: usize,
    count: usize,
    slots: usize,
    property_byte_offset: usize,
}

#[derive(Clone, Copy, Debug)]
struct PropertyIndexEntry {
    key: PropertyKeyId,
    fingerprint: u32,
    kind: u8,
    id: u64,
}

#[derive(Debug)]
struct MappedPropertyIndex {
    map: Arc<Mmap>,
    byte_offset: usize,
    count: usize,
    entry_width: usize,
}

impl MappedPropertyIndex {
    fn new(map: Arc<Mmap>, section: crate::codec::SnapshotPropertyIndexSection) -> Result<Self> {
        let byte_len = section
            .entry_count
            .checked_mul(section.entry_width)
            .ok_or_else(|| Error::Corrupt("property index byte length overflow".into()))?;
        let end = section
            .byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| Error::Corrupt("property index range overflow".into()))?;
        map.get(section.byte_offset..end)
            .ok_or_else(|| Error::Corrupt("property index exceeds mapped file".into()))?;
        Ok(Self {
            map,
            byte_offset: section.byte_offset,
            count: section.entry_count,
            entry_width: section.entry_width,
        })
    }

    #[inline]
    fn entry_at(&self, index: usize) -> PropertyIndexEntry {
        let start = self.byte_offset + index * self.entry_width;
        let bytes = &self.map[start..start + self.entry_width];
        let packed_element = if self.entry_width == 12 {
            record_u32(bytes, 8) as u64
        } else {
            record_u64(bytes, 8)
        };
        PropertyIndexEntry {
            key: record_u32(bytes, 0),
            fingerprint: record_u32(bytes, 4),
            kind: (packed_element & 1) as u8,
            id: packed_element >> 1,
        }
    }

    fn range(&self, key: PropertyKeyId, fingerprint: u32) -> std::ops::Range<usize> {
        let needle = (key, fingerprint);
        let lower = self.partition_point(|entry| (entry.key, entry.fingerprint) < needle);
        let upper = self.partition_point(|entry| (entry.key, entry.fingerprint) <= needle);
        lower..upper
    }

    fn partition_point(&self, mut predicate: impl FnMut(PropertyIndexEntry) -> bool) -> usize {
        let mut left = 0usize;
        let mut right = self.count;
        while left < right {
            let middle = left + (right - left) / 2;
            if predicate(self.entry_at(middle)) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        left
    }
}

#[derive(Clone, Copy, Debug)]
struct NumericPropertyIndexEntry {
    key: PropertyKeyId,
    tag: u8,
    sortable: u64,
    kind: u8,
    id: u64,
}

#[derive(Debug)]
struct MappedNumericPropertyIndex {
    map: Arc<Mmap>,
    byte_offset: usize,
    count: usize,
    entry_width: usize,
}

impl MappedNumericPropertyIndex {
    fn new(
        map: Arc<Mmap>,
        section: crate::codec::SnapshotNumericPropertyIndexSection,
    ) -> Result<Self> {
        let byte_len = section
            .entry_count
            .checked_mul(section.entry_width)
            .ok_or_else(|| Error::Corrupt("numeric property index byte length overflow".into()))?;
        let end = section
            .byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| Error::Corrupt("numeric property index range overflow".into()))?;
        map.get(section.byte_offset..end)
            .ok_or_else(|| Error::Corrupt("numeric property index exceeds mapped file".into()))?;
        Ok(Self {
            map,
            byte_offset: section.byte_offset,
            count: section.entry_count,
            entry_width: section.entry_width,
        })
    }

    #[inline]
    fn entry_at(&self, index: usize) -> NumericPropertyIndexEntry {
        let start = self.byte_offset + index * self.entry_width;
        let bytes = &self.map[start..start + self.entry_width];
        let packed_element = if self.entry_width == 20 {
            record_u32(bytes, 16) as u64
        } else {
            record_u64(bytes, 16)
        };
        NumericPropertyIndexEntry {
            key: record_u32(bytes, 0),
            tag: bytes[4],
            sortable: record_u64(bytes, 8),
            kind: (packed_element & 1) as u8,
            id: packed_element >> 1,
        }
    }

    fn range(
        &self,
        key: PropertyKeyId,
        tag: u8,
        lower: Bound<u64>,
        upper: Bound<u64>,
    ) -> std::ops::Range<usize> {
        let lower = match lower {
            Bound::Included(value) => self.partition_point(|entry| {
                (entry.key, entry.tag, entry.sortable) < (key, tag, value)
            }),
            Bound::Excluded(value) => self.partition_point(|entry| {
                (entry.key, entry.tag, entry.sortable) <= (key, tag, value)
            }),
            Bound::Unbounded => self.partition_point(|entry| (entry.key, entry.tag) < (key, tag)),
        };
        let upper = match upper {
            Bound::Included(value) => self.partition_point(|entry| {
                (entry.key, entry.tag, entry.sortable) <= (key, tag, value)
            }),
            Bound::Excluded(value) => self.partition_point(|entry| {
                (entry.key, entry.tag, entry.sortable) < (key, tag, value)
            }),
            Bound::Unbounded => self.partition_point(|entry| (entry.key, entry.tag) <= (key, tag)),
        };
        lower.min(upper)..upper
    }

    fn partition_point(
        &self,
        mut predicate: impl FnMut(NumericPropertyIndexEntry) -> bool,
    ) -> usize {
        let mut left = 0usize;
        let mut right = self.count;
        while left < right {
            let middle = left + (right - left) / 2;
            if predicate(self.entry_at(middle)) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        left
    }
}

impl MappedEdgeRecords {
    #[inline(always)]
    fn disk_record_at(&self, index: usize) -> DiskEdgeRecord {
        #[cfg(target_endian = "little")]
        {
            let pointer = self.map[self.byte_offset..].as_ptr();
            debug_assert_eq!(
                pointer.align_offset(std::mem::align_of::<DiskEdgeRecord>()),
                0
            );
            // SAFETY: the decoder validates the aligned, complete fixed-width
            // section and the read-only map outlives this table.
            unsafe { *pointer.cast::<DiskEdgeRecord>().add(index) }
        }
        #[cfg(target_endian = "big")]
        {
            let start = self.byte_offset + index * 64;
            let bytes = &self.map[start..start + 64];
            DiskEdgeRecord {
                id: record_u64(bytes, 0),
                generation: record_u64(bytes, 8),
                source: record_u64(bytes, 16),
                target: record_u64(bytes, 24),
                vector_offset: record_u64(bytes, 32),
                property_offset: record_u64(bytes, 40),
                label: record_u32(bytes, 48),
                property_len: record_u32(bytes, 52),
                vector_count: record_u32(bytes, 56),
                reserved: record_u32(bytes, 60),
            }
        }
    }

    fn record_at(&self, index: usize) -> StoredEdge {
        let record = self.disk_record_at(index);
        StoredEdge {
            id: record.id,
            generation: NonZeroU64::new(record.generation).unwrap(),
            source: record.source,
            target: record.target,
            vector_offset: record.vector_offset as usize,
            properties: StoredProperties::mapped(
                (self.property_byte_offset as u64) + record.property_offset,
                record.property_len,
            ),
            label: record.label,
            vector_count: record.vector_count,
        }
    }

    fn get(&self, id: EdgeId) -> Option<StoredEdge> {
        let id = usize::try_from(id).ok()?;
        if self.count == self.slots && id < self.count {
            let record = self.record_at(id);
            return (record.id as usize == id).then_some(record);
        }
        let mut left = 0usize;
        let mut right = self.count;
        while left < right {
            let middle = left + (right - left) / 2;
            match self.record_at(middle).id.cmp(&(id as u64)) {
                Ordering::Less => left = middle + 1,
                Ordering::Greater => right = middle,
                Ordering::Equal => return Some(self.record_at(middle)),
            }
        }
        None
    }

    /// Resolves only the fields required by traversal, avoiding construction
    /// of properties, generations, and vector metadata in the inner CSR loop.
    #[inline(always)]
    fn neighbor(
        &self,
        id: EdgeId,
        node: NodeId,
        outgoing: bool,
        label: Option<LabelId>,
    ) -> Option<NodeId> {
        let id_index = usize::try_from(id).ok()?;
        let record = if self.count == self.slots && id_index < self.count {
            self.disk_record_at(id_index)
        } else {
            let mut left = 0usize;
            let mut right = self.count;
            loop {
                if left >= right {
                    return None;
                }
                let middle = left + (right - left) / 2;
                let record = self.disk_record_at(middle);
                match record.id.cmp(&id) {
                    Ordering::Less => left = middle + 1,
                    Ordering::Greater => right = middle,
                    Ordering::Equal => break record,
                }
            }
        };
        if record.id != id
            || (outgoing && record.source != node)
            || (!outgoing && record.target != node)
            || label.is_some_and(|label| record.label != label)
        {
            return None;
        }
        Some(if outgoing {
            record.target
        } else {
            record.source
        })
    }

    fn iter(&self) -> impl Iterator<Item = StoredEdge> + '_ {
        (0..self.count).map(|index| self.record_at(index))
    }

    #[inline(always)]
    fn vector_fields_at(&self, index: usize) -> (EdgeId, LabelId, usize, u32) {
        let record = self.disk_record_at(index);
        (
            record.id,
            record.label,
            record.vector_offset as usize,
            record.vector_count,
        )
    }

    #[inline(always)]
    fn vector_fields(&self, id: EdgeId) -> Option<(LabelId, usize, u32)> {
        let index = usize::try_from(id).ok()?;
        if self.count == self.slots && index < self.count {
            let record = self.disk_record_at(index);
            return (record.id == id).then_some((
                record.label,
                record.vector_offset as usize,
                record.vector_count,
            ));
        }
        self.get(id)
            .map(|record| (record.label, record.vector_offset, record.vector_count))
    }
}

fn record_u32(record: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(record[offset..offset + 4].try_into().unwrap())
}

fn record_u64(record: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(record[offset..offset + 8].try_into().unwrap())
}

#[derive(Clone, Debug)]
pub(crate) enum SnapshotOperation {
    InternSymbol {
        id: u32,
        value: Arc<str>,
    },
    PutNode {
        id: NodeId,
        generation: u64,
        label: LabelId,
        property_byte_offset: u64,
        property_byte_len: u32,
        vector_count: u32,
        vector_offset: usize,
    },
    PutEdge {
        id: EdgeId,
        generation: u64,
        source: NodeId,
        target: NodeId,
        label: LabelId,
        property_byte_offset: u64,
        property_byte_len: u32,
        vector_count: u32,
        vector_offset: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeFilter {
    pub label: Option<LabelId>,
}

/// Bounds and traversal semantics for an exact unweighted shortest path.
///
/// `max_expansions` counts frontier nodes whose adjacency is expanded. This is
/// deliberately separate from `max_hops`: a shallow search over a broad graph
/// can still have a finite application-controlled work budget.
#[derive(Clone, Copy, Debug)]
pub struct ShortestPathOptions {
    pub max_hops: usize,
    pub max_expansions: usize,
    pub direction: Direction,
    pub edge_filter: EdgeFilter,
}

impl Default for ShortestPathOptions {
    fn default() -> Self {
        Self {
            max_hops: 6,
            max_expansions: 100_000,
            direction: Direction::Both,
            edge_filter: EdgeFilter::default(),
        }
    }
}

/// One exact graph path. `edges[index]` connects `nodes[index]` to
/// `nodes[index + 1]` under the requested traversal direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShortestPath {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
}

/// Physical traversal selected for an exact unweighted path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortestPathStrategy {
    /// A single start-side frontier, used for one-hop work.
    BreadthFirst,
    /// Two complete frontiers with the next side chosen by estimated node and
    /// adjacency work.
    BidirectionalBreadthFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortestPathTermination {
    Found,
    NotFoundWithinHops,
    ExpansionLimit,
}

/// Inspectable result of bounded exact shortest-path search.
///
/// A missing `path` is conclusive only when `termination` is
/// `NotFoundWithinHops`; `ExpansionLimit` reports a deliberately incomplete
/// search rather than silently looking like graph absence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShortestPathResult {
    pub path: Option<ShortestPath>,
    pub strategy: ShortestPathStrategy,
    pub termination: ShortestPathTermination,
    pub visited_nodes: usize,
    /// Nodes expanded from the requested start endpoint.
    pub start_expanded_nodes: usize,
    /// Nodes expanded from the requested end endpoint. This is zero for the
    /// one-sided breadth-first strategy.
    pub end_expanded_nodes: usize,
    /// Total nodes expanded across both endpoints. This always equals
    /// `start_expanded_nodes + end_expanded_nodes`.
    pub expanded_nodes: usize,
    pub examined_relationships: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    pub nodes: usize,
    pub edges: usize,
    pub labels: usize,
    pub indexed_vectors: usize,
    pub transactions: u64,
}

#[derive(Clone, Debug)]
pub struct SemanticPathOptions {
    pub seed_count: usize,
    pub limit: usize,
    pub max_hops: usize,
    pub max_expansions: usize,
    pub direction: Direction,
    pub seed_label: Option<LabelId>,
    pub edge_label: Option<LabelId>,
    pub node_weight: f32,
    pub edge_weight: f32,
    pub path_decay: f32,
    pub hop_penalty: f32,
    pub degree_penalty: f32,
    pub include_seeds: bool,
}

impl Default for SemanticPathOptions {
    fn default() -> Self {
        Self {
            seed_count: 32,
            limit: 20,
            max_hops: 2,
            max_expansions: 10_000,
            direction: Direction::Both,
            seed_label: None,
            edge_label: None,
            node_weight: 0.6,
            edge_weight: 0.4,
            path_decay: 0.55,
            hop_penalty: 0.92,
            degree_penalty: 0.04,
            include_seeds: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticPathHit {
    pub seed: NodeId,
    pub node: NodeId,
    pub score: f32,
    pub seed_score: f32,
    pub path: Vec<EdgeId>,
}

/// Result of a nearest-neighbor query constrained to a bounded graph range.
///
/// The range is evaluated before the vector candidate budget, so `hits` never
/// relies on post-filtering a global ANN result. `plan` describes the exact or
/// sketch/rerank decision made after the reachable-node cardinality is known.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphRangeSearchResult {
    pub hits: Vec<VectorHit>,
    pub candidate_nodes: u64,
    pub plan: VectorSearchPlan,
}

/// Result of a vector query whose ordinary property and numeric-range
/// predicates were evaluated before vector candidate selection. The access
/// plans remain visible so applications can diagnose selectivity/cost without
/// parsing an engine-specific explain string.
#[derive(Clone, Debug, PartialEq)]
pub struct FilteredVectorSearchResult {
    pub hits: Vec<VectorHit>,
    pub candidate_elements: u64,
    pub equality_plan: Option<ElementFilterPlan>,
    pub numeric_range_plans: Vec<NumericRangePlan>,
    pub vector_plan: VectorSearchPlan,
}

#[derive(Clone, Debug)]
pub struct GraphRangeSearchOptions {
    pub max_hops: usize,
    pub limit: usize,
    pub direction: Direction,
    pub edge_filter: EdgeFilter,
    pub include_seeds: bool,
    pub node_filter: Option<ElementFilter>,
}

impl Default for GraphRangeSearchOptions {
    fn default() -> Self {
        Self {
            max_hops: 2,
            limit: 10,
            direction: Direction::Both,
            edge_filter: EdgeFilter::default(),
            include_seeds: true,
            node_filter: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ElementFilter {
    pub label: Option<LabelId>,
    pub properties: Vec<(PropertyKeyId, Value)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementFilterStrategy {
    FullScan,
    LabelPosting,
    PropertyPosting,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementFilterPlan {
    pub strategy: ElementFilterStrategy,
    /// Conservative number of records the selected access path may inspect.
    pub candidate_upper_bound: usize,
    /// Predicate ordinal in `ElementFilter::properties` when a property
    /// posting is selected.
    pub property_predicate: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumericValue {
    Int(i64),
    Float(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericRangeFilter {
    pub label: Option<LabelId>,
    pub key: PropertyKeyId,
    pub lower: Bound<NumericValue>,
    pub upper: Bound<NumericValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericRangeStrategy {
    FullScan,
    LabelPosting,
    NumericPosting,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericRangePlan {
    pub strategy: NumericRangeStrategy,
    pub candidate_upper_bound: usize,
}

/// Backward-compatible name used by the node positions of `OneHopQuery`.
pub type NodeFilter = ElementFilter;

#[derive(Clone, Debug)]
pub struct OneHopQuery {
    pub start: NodeFilter,
    pub edge_label: Option<LabelId>,
    pub end: NodeFilter,
    pub direction: Direction,
    pub limit: usize,
}

impl Default for OneHopQuery {
    fn default() -> Self {
        Self {
            start: NodeFilter::default(),
            edge_label: None,
            end: NodeFilter::default(),
            direction: Direction::Outgoing,
            limit: 100,
        }
    }
}

/// Physical access path selected for an exact one-hop pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneHopStrategy {
    /// Scan all relationships, or an exact relationship-label posting.
    EdgeScan,
    /// Materialize the start-node predicate, then expand matching adjacency.
    StartAdjacency,
    /// Materialize the end-node predicate, then expand reverse adjacency.
    EndAdjacency,
}

/// Inspectable cost decision for an exact one-hop pattern.
///
/// Candidate bounds come from the same label/property postings used during
/// execution. `estimated_edge_visits` is a cardinality-weighted average-degree
/// cost for ordering access paths, not a runtime cardinality promise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHopPlan {
    pub strategy: OneHopStrategy,
    pub estimated_edge_visits: usize,
    pub edge_candidate_upper_bound: usize,
    pub start_candidate_upper_bound: usize,
    pub end_candidate_upper_bound: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatternMatch {
    pub start: NodeId,
    pub edge: EdgeId,
    pub end: NodeId,
}

#[derive(Clone, Debug)]
pub struct SemanticOneHopQuery {
    pub pattern: OneHopQuery,
    pub seed_count: usize,
    pub start_weight: f32,
    pub edge_weight: f32,
    pub end_weight: f32,
}

impl Default for SemanticOneHopQuery {
    fn default() -> Self {
        Self {
            pattern: OneHopQuery::default(),
            seed_count: 64,
            start_weight: 0.5,
            edge_weight: 0.25,
            end_weight: 0.25,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticPatternMatch {
    pub pattern: PatternMatch,
    pub score: f32,
    pub start_score: f32,
    pub edge_score: Option<f32>,
    pub end_score: Option<f32>,
}

#[derive(Clone, Debug)]
struct PathState {
    seed: NodeId,
    node: NodeId,
    score: f32,
    seed_score: f32,
    path: Vec<EdgeId>,
}

impl PartialEq for PathState {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.node == other.node
    }
}

impl Eq for PathState {}

impl PartialOrd for PathState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.node.cmp(&self.node))
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Mutation {
    PutNode {
        id: NodeId,
        label: Arc<str>,
        properties: Vec<(Arc<str>, Value)>,
        vectors: Vec<f32>,
        vector_count: u32,
    },
    PutEdge {
        id: EdgeId,
        source: NodeId,
        target: NodeId,
        label: Arc<str>,
        properties: Vec<(Arc<str>, Value)>,
        vectors: Vec<f32>,
        vector_count: u32,
    },
    DeleteNode {
        id: NodeId,
        detach: bool,
    },
    DeleteEdge {
        id: EdgeId,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum Operation {
    InternSymbol { id: u32, value: Arc<str> },
    PutNode(Node),
    PutEdge(Edge),
    DeleteNode(NodeId),
    DeleteEdge(EdgeId),
}

#[derive(Debug)]
pub(crate) struct Graph {
    dimension: usize,
    similarity: Similarity,
    symbols: Vec<Arc<str>>,
    symbol_ids: HashMap<Arc<str>, u32>,
    nodes: Vec<Option<StoredNode>>,
    edges: Vec<Option<StoredEdge>>,
    mapped_nodes: Option<MappedNodeRecords>,
    mapped_edges: Option<MappedEdgeRecords>,
    mapped_property_index: Option<MappedPropertyIndex>,
    mapped_numeric_property_index: Option<MappedNumericPropertyIndex>,
    node_overlays: HashMap<NodeId, Option<StoredNode>>,
    edge_overlays: HashMap<EdgeId, Option<StoredEdge>>,
    outgoing: MutableAdjacency,
    incoming: MutableAdjacency,
    base_outgoing: Option<CsrAdjacency>,
    base_incoming: Option<CsrAdjacency>,
    loading_snapshot: bool,
    snapshot_map: Option<Arc<Mmap>>,
    owned_properties: Vec<Arc<[Property]>>,
    nodes_by_label: HashMap<LabelId, RoaringTreemap>,
    edges_by_label: HashMap<LabelId, RoaringTreemap>,
    vector_data: VectorData,
    sketch_index: OnceLock<std::result::Result<BinarySketchIndex, String>>,
    indexed_vectors: usize,
    indexed_node_vectors: usize,
    indexed_edge_vectors: usize,
    node_count: usize,
    edge_count: usize,
    transactions: u64,
}

impl Graph {
    pub(crate) fn new(dimension: usize, similarity: Similarity) -> Self {
        Self {
            dimension,
            similarity,
            symbols: Vec::new(),
            symbol_ids: HashMap::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            mapped_nodes: None,
            mapped_edges: None,
            mapped_property_index: None,
            mapped_numeric_property_index: None,
            node_overlays: HashMap::new(),
            edge_overlays: HashMap::new(),
            outgoing: MutableAdjacency::dense(),
            incoming: MutableAdjacency::dense(),
            base_outgoing: None,
            base_incoming: None,
            loading_snapshot: false,
            snapshot_map: None,
            owned_properties: Vec::new(),
            nodes_by_label: HashMap::new(),
            edges_by_label: HashMap::new(),
            vector_data: VectorData::Owned(Vec::new()),
            sketch_index: OnceLock::new(),
            indexed_vectors: 0,
            indexed_node_vectors: 0,
            indexed_edge_vectors: 0,
            node_count: 0,
            edge_count: 0,
            transactions: 0,
        }
    }

    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn similarity(&self) -> Similarity {
        self.similarity
    }

    pub(crate) fn mark_transaction_applied(&mut self) {
        self.transactions += 1;
    }

    pub(crate) fn next_node_id(&self) -> NodeId {
        self.nodes
            .len()
            .max(
                self.mapped_nodes
                    .as_ref()
                    .map_or(0, |records| records.slots),
            )
            .max(
                self.node_overlays
                    .keys()
                    .max()
                    .and_then(|id| usize::try_from(*id).ok())
                    .map_or(0, |id| id.saturating_add(1)),
            ) as NodeId
    }

    pub(crate) fn next_edge_id(&self) -> EdgeId {
        self.edges
            .len()
            .max(
                self.mapped_edges
                    .as_ref()
                    .map_or(0, |records| records.slots),
            )
            .max(
                self.edge_overlays
                    .keys()
                    .max()
                    .and_then(|id| usize::try_from(*id).ok())
                    .map_or(0, |id| id.saturating_add(1)),
            ) as EdgeId
    }

    fn node_record(&self, id: NodeId) -> Option<StoredNode> {
        if !self.node_overlays.is_empty()
            && let Some(record) = self.node_overlays.get(&id)
        {
            return *record;
        }
        if !self.nodes.is_empty()
            && let Some(record) = self.nodes.get(id as usize).and_then(|record| *record)
        {
            return Some(record);
        }
        self.mapped_nodes.as_ref()?.get(id)
    }

    fn edge_record(&self, id: EdgeId) -> Option<StoredEdge> {
        if !self.edge_overlays.is_empty()
            && let Some(record) = self.edge_overlays.get(&id)
        {
            return *record;
        }
        if !self.edges.is_empty()
            && let Some(record) = self.edges.get(id as usize).and_then(|record| *record)
        {
            return Some(record);
        }
        self.mapped_edges.as_ref()?.get(id)
    }

    fn node_records(&self) -> Box<dyn Iterator<Item = StoredNode> + '_> {
        if let Some(mapped) = &self.mapped_nodes {
            if self.node_overlays.is_empty() {
                Box::new(mapped.iter())
            } else {
                Box::new(
                    mapped
                        .iter()
                        .filter(|record| !self.node_overlays.contains_key(&record.id))
                        .chain(self.node_overlays.values().filter_map(|record| *record)),
                )
            }
        } else {
            Box::new(self.nodes.iter().filter_map(|record| *record))
        }
    }

    fn edge_records(&self) -> Box<dyn Iterator<Item = StoredEdge> + '_> {
        if let Some(mapped) = &self.mapped_edges {
            if self.edge_overlays.is_empty() {
                Box::new(mapped.iter())
            } else {
                Box::new(
                    mapped
                        .iter()
                        .filter(|record| !self.edge_overlays.contains_key(&record.id))
                        .chain(self.edge_overlays.values().filter_map(|record| *record)),
                )
            }
        } else {
            Box::new(self.edges.iter().filter_map(|record| *record))
        }
    }

    /// Streams the vector-relevant node columns without decoding properties or
    /// generations. Checkpoint-only databases take a monomorphic mapped path;
    /// mutable/legacy graphs retain the unified record semantics.
    #[inline]
    fn visit_node_vector_fields(
        &self,
        mut visitor: impl FnMut(NodeId, LabelId, usize, u32) -> Result<()>,
    ) -> Result<()> {
        if self.nodes.is_empty()
            && self.node_overlays.is_empty()
            && let Some(records) = &self.mapped_nodes
        {
            for index in 0..records.count {
                let (id, label, offset, count) = records.vector_fields_at(index);
                visitor(id, label, offset, count)?;
            }
        } else {
            for node in self.node_records() {
                visitor(node.id, node.label, node.vector_offset, node.vector_count)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn visit_edge_vector_fields(
        &self,
        mut visitor: impl FnMut(EdgeId, LabelId, usize, u32) -> Result<()>,
    ) -> Result<()> {
        if self.edges.is_empty()
            && self.edge_overlays.is_empty()
            && let Some(records) = &self.mapped_edges
        {
            for index in 0..records.count {
                let (id, label, offset, count) = records.vector_fields_at(index);
                visitor(id, label, offset, count)?;
            }
        } else {
            for edge in self.edge_records() {
                visitor(edge.id, edge.label, edge.vector_offset, edge.vector_count)?;
            }
        }
        Ok(())
    }

    pub(crate) fn install_mapped_records(
        &mut self,
        map: Arc<Mmap>,
        sections: crate::codec::SnapshotRecordSections,
    ) -> Result<()> {
        self.mapped_nodes = Some(MappedNodeRecords {
            map: map.clone(),
            byte_offset: sections.node_byte_offset,
            count: sections.node_count,
            slots: sections.node_slots,
            property_byte_offset: sections.property_byte_offset,
        });
        self.mapped_edges = Some(MappedEdgeRecords {
            map,
            byte_offset: sections.edge_byte_offset,
            count: sections.edge_count,
            slots: sections.edge_slots,
            property_byte_offset: sections.property_byte_offset,
        });
        self.nodes.clear();
        self.nodes.shrink_to_fit();
        self.edges.clear();
        self.edges.shrink_to_fit();
        self.node_count = sections.node_count;
        self.edge_count = sections.edge_count;
        self.indexed_vectors = 0;
        self.indexed_node_vectors = 0;
        self.indexed_edge_vectors = 0;
        self.nodes_by_label.clear();
        self.edges_by_label.clear();
        for node in self.mapped_nodes.as_ref().unwrap().iter() {
            if node.label as usize >= self.symbols.len() {
                return Err(Error::Corrupt(
                    "mapped node label exceeds dictionary".into(),
                ));
            }
            self.indexed_vectors = self
                .indexed_vectors
                .checked_add(node.vector_count as usize)
                .ok_or_else(|| Error::Corrupt("mapped vector count overflow".into()))?;
            self.indexed_node_vectors = self
                .indexed_node_vectors
                .checked_add(node.vector_count as usize)
                .ok_or_else(|| Error::Corrupt("mapped node vector count overflow".into()))?;
            self.nodes_by_label
                .entry(node.label)
                .or_default()
                .insert(node.id);
        }
        for edge in self.mapped_edges.as_ref().unwrap().iter() {
            if edge.label as usize >= self.symbols.len() {
                return Err(Error::Corrupt(
                    "mapped edge label exceeds dictionary".into(),
                ));
            }
            self.indexed_vectors = self
                .indexed_vectors
                .checked_add(edge.vector_count as usize)
                .ok_or_else(|| Error::Corrupt("mapped vector count overflow".into()))?;
            self.indexed_edge_vectors = self
                .indexed_edge_vectors
                .checked_add(edge.vector_count as usize)
                .ok_or_else(|| Error::Corrupt("mapped edge vector count overflow".into()))?;
            self.edges_by_label
                .entry(edge.label)
                .or_default()
                .insert(edge.id);
        }
        Ok(())
    }

    pub(crate) fn install_mapped_property_index(
        &mut self,
        map: Arc<Mmap>,
        section: crate::codec::SnapshotPropertyIndexSection,
    ) -> Result<()> {
        self.mapped_property_index = Some(MappedPropertyIndex::new(map, section)?);
        Ok(())
    }

    pub(crate) fn install_mapped_numeric_property_index(
        &mut self,
        map: Arc<Mmap>,
        section: crate::codec::SnapshotNumericPropertyIndexSection,
    ) -> Result<()> {
        self.mapped_numeric_property_index = Some(MappedNumericPropertyIndex::new(map, section)?);
        Ok(())
    }

    pub(crate) fn node(&self, id: NodeId) -> Option<Node> {
        let node = self.node_record(id)?;
        Some(Node {
            id: node.id,
            label: node.label,
            properties: node
                .properties
                .get(self.snapshot_map.as_deref(), &self.owned_properties),
            vector_count: node.vector_count,
            generation: node.generation.get(),
            pending_vectors: Arc::from([]),
        })
    }

    pub(crate) fn edge(&self, id: EdgeId) -> Option<Edge> {
        let edge = self.edge_record(id)?;
        Some(Edge {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.label,
            properties: edge
                .properties
                .get(self.snapshot_map.as_deref(), &self.owned_properties),
            vector_count: edge.vector_count,
            generation: edge.generation.get(),
            vector_offset: edge.vector_offset,
            pending_vectors: Arc::from([]),
        })
    }

    #[inline]
    fn has_node(&self, id: NodeId) -> bool {
        if let Some(record) = self.node_overlays.get(&id) {
            return record.is_some();
        }
        if self.nodes.get(id as usize).is_some_and(Option::is_some) {
            return true;
        }
        self.mapped_nodes
            .as_ref()
            .is_some_and(|records| records.contains(id))
    }

    #[inline]
    fn stored_edge(&self, id: EdgeId) -> Option<StoredEdge> {
        self.edge_record(id)
    }

    pub(crate) fn node_ids(&self) -> Vec<NodeId> {
        self.node_records().map(|node| node.id).collect()
    }

    pub(crate) fn edge_ids(&self) -> Vec<EdgeId> {
        self.edge_records().map(|edge| edge.id).collect()
    }

    pub(crate) fn symbol_id(&self, value: &str) -> Option<u32> {
        self.symbol_ids.get(value).copied()
    }

    pub(crate) fn symbol(&self, id: u32) -> Option<&str> {
        self.symbols.get(id as usize).map(AsRef::as_ref)
    }

    pub(crate) fn stats(&self) -> GraphStats {
        GraphStats {
            nodes: self.node_count,
            edges: self.edge_count,
            labels: self.symbols.len(),
            indexed_vectors: self.indexed_vectors,
            transactions: self.transactions,
        }
    }

    pub(crate) fn warm_vector_cache(&self) -> Result<usize> {
        self.vector_data.warm_f32_cache()
    }

    pub(crate) fn vector_cache_bytes(&self) -> usize {
        self.vector_data.f32_cache_bytes()
    }

    pub(crate) fn verify_integrity(&self) -> Result<(usize, usize)> {
        self.vector_data.verify_all_bytes()
    }

    pub(crate) fn snapshot_symbols(&self) -> Vec<Operation> {
        self.symbols
            .iter()
            .enumerate()
            .map(|(id, value)| Operation::InternSymbol {
                id: id as u32,
                value: value.clone(),
            })
            .collect()
    }

    #[cfg(unix)]
    pub(crate) fn advise_checkpoint(&self, advice: memmap2::Advice) {
        if let Some(map) = &self.snapshot_map {
            let _ = map.advise(advice);
        }
    }

    pub(crate) fn apply_snapshot_operation(&mut self, operation: SnapshotOperation) -> Result<()> {
        match operation {
            SnapshotOperation::InternSymbol { id, value } => {
                if id as usize != self.symbols.len() {
                    return Err(Error::Corrupt(format!(
                        "snapshot symbol id {id} is not the next dictionary id {}",
                        self.symbols.len()
                    )));
                }
                self.symbols.push(value.clone());
                self.symbol_ids.insert(value, id);
            }
            SnapshotOperation::PutNode {
                id,
                generation,
                label,
                property_byte_offset,
                property_byte_len,
                vector_count,
                vector_offset,
            } => self.install_snapshot_node(StoredNode {
                id,
                label,
                properties: StoredProperties::mapped(property_byte_offset, property_byte_len),
                vector_count,
                generation: nonzero_generation(generation)?,
                vector_offset,
            })?,
            SnapshotOperation::PutEdge {
                id,
                generation,
                source,
                target,
                label,
                property_byte_offset,
                property_byte_len,
                vector_count,
                vector_offset,
            } => self.install_snapshot_edge(StoredEdge {
                id,
                source,
                target,
                label,
                properties: StoredProperties::mapped(property_byte_offset, property_byte_len),
                vector_count,
                generation: nonzero_generation(generation)?,
                vector_offset,
            })?,
        }
        Ok(())
    }

    pub(crate) fn begin_snapshot_load(&mut self, map: Arc<Mmap>) {
        self.loading_snapshot = true;
        self.snapshot_map = Some(map);
        self.outgoing = MutableAdjacency::sparse();
        self.incoming = MutableAdjacency::sparse();
    }

    pub(crate) fn finish_snapshot_load(
        &mut self,
        csr: Option<crate::codec::SnapshotCsrSections>,
    ) -> Result<()> {
        if let Some(csr) = csr {
            let map = self.snapshot_map.as_ref().unwrap().clone();
            self.base_outgoing = Some(CsrAdjacency::mapped(
                map.clone(),
                csr.out_offsets,
                csr.out_ids,
            )?);
            self.base_incoming = Some(CsrAdjacency::mapped(map, csr.in_offsets, csr.in_ids)?);
        } else {
            self.base_outgoing = Some(CsrAdjacency::from_edges(
                &self.edges,
                self.nodes.len(),
                true,
            )?);
            self.base_incoming = Some(CsrAdjacency::from_edges(
                &self.edges,
                self.nodes.len(),
                false,
            )?);
        }
        self.loading_snapshot = false;
        Ok(())
    }

    pub(crate) fn install_mapped_vectors(
        &mut self,
        map: Arc<Mmap>,
        byte_offset: usize,
        byte_len: usize,
        encoding: VectorEncoding,
        checksum: u32,
        block_checksums: Option<Arc<[u32]>>,
    ) -> Result<()> {
        let bytes_per_vector_value = match encoding {
            VectorEncoding::F32 => 4,
            VectorEncoding::F16 => 2,
        };
        if !byte_len.is_multiple_of(bytes_per_vector_value) {
            return Err(Error::Corrupt(
                "checkpoint vector section has a partial value".into(),
            ));
        }
        let float_count = byte_len / bytes_per_vector_value;
        let required = self
            .node_records()
            .map(|node| node.vector_offset + node.vector_count as usize * self.dimension)
            .chain(
                self.edge_records()
                    .map(|edge| edge.vector_offset + edge.vector_count as usize * self.dimension),
            )
            .max()
            .unwrap_or(0);
        if required > float_count {
            return Err(Error::Corrupt(format!(
                "checkpoint metadata references {required} floats but vector section has {float_count}"
            )));
        }
        let verified_blocks = (0..block_checksums
            .as_ref()
            .map_or(0, |checksums| checksums.len()))
            .map(|_| OnceLock::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.vector_data = VectorData::Mapped {
            map,
            byte_offset,
            byte_len,
            encoding,
            checksum,
            block_checksums,
            verified: OnceLock::new(),
            verified_blocks,
            decoded: OnceLock::new(),
            delta: Vec::new(),
        };
        Ok(())
    }

    pub(crate) fn install_mapped_sketches(
        &mut self,
        map: Arc<Mmap>,
        section: crate::codec::SnapshotSketchSection,
    ) -> Result<()> {
        let word_count = section
            .entry_count
            .checked_mul(section.words_per_signature)
            .ok_or_else(|| Error::Corrupt("mapped sketch word count overflow".into()))?;
        #[cfg(target_endian = "little")]
        if let Some(columns) = section.owner_columns {
            let index = BinarySketchIndex::mapped_columns(
                self.dimension,
                map,
                columns.owner_byte_offset,
                columns.owner_kind_byte_offset,
                columns.label_byte_offset,
                section.byte_offset,
                section.entry_count,
                section.words_per_signature,
                word_count,
            );
            self.sketch_index.set(Ok(index)).map_err(|_| {
                Error::Corrupt("checkpoint installs sketches more than once".into())
            })?;
            return Ok(());
        }
        let mut owners = vec![0u64; section.entry_count];
        let mut owner_kinds = vec![0u8; section.entry_count];
        let mut labels = vec![0u32; section.entry_count];
        let mut populated = vec![false; section.entry_count];
        for node in self.node_records() {
            for vector_index in 0..node.vector_count {
                let float_offset = node.vector_offset + vector_index as usize * self.dimension;
                let ordinal = float_offset / self.dimension;
                if !float_offset.is_multiple_of(self.dimension)
                    || ordinal >= owners.len()
                    || populated[ordinal]
                {
                    return Err(Error::Corrupt(
                        "indexed checkpoint node sketch ordinal is invalid".into(),
                    ));
                }
                owners[ordinal] = node.id;
                owner_kinds[ordinal] = 0;
                labels[ordinal] = node.label;
                populated[ordinal] = true;
            }
        }
        for edge in self.edge_records() {
            for vector_index in 0..edge.vector_count {
                let float_offset = edge.vector_offset + vector_index as usize * self.dimension;
                let ordinal = float_offset / self.dimension;
                if !float_offset.is_multiple_of(self.dimension)
                    || ordinal >= owners.len()
                    || populated[ordinal]
                {
                    return Err(Error::Corrupt(
                        "indexed checkpoint edge sketch ordinal is invalid".into(),
                    ));
                }
                owners[ordinal] = edge.id;
                owner_kinds[ordinal] = 1;
                labels[ordinal] = edge.label;
                populated[ordinal] = true;
            }
        }
        if populated.iter().any(|populated| !populated) {
            return Err(Error::Corrupt(
                "sketch entries do not densely cover checkpoint vectors".into(),
            ));
        }
        let index = BinarySketchIndex::mapped(
            self.dimension,
            owners,
            owner_kinds,
            labels,
            map,
            section.byte_offset,
            section.words_per_signature,
            word_count,
        );
        self.sketch_index
            .set(Ok(index))
            .map_err(|_| Error::Corrupt("checkpoint installs sketches more than once".into()))?;
        Ok(())
    }

    fn install_snapshot_node(&mut self, node: StoredNode) -> Result<()> {
        let index = node.id as usize;
        grow_slots(&mut self.nodes, index);
        self.outgoing.grow(index);
        self.incoming.grow(index);
        if self.nodes[index].is_some() {
            return Err(Error::Corrupt(format!(
                "checkpoint contains duplicate node {}",
                node.id
            )));
        }
        self.node_count += 1;
        self.indexed_vectors += node.vector_count as usize;
        self.indexed_node_vectors += node.vector_count as usize;
        self.nodes_by_label
            .entry(node.label)
            .or_default()
            .insert(node.id);
        self.nodes[index] = Some(node);
        Ok(())
    }

    fn install_snapshot_edge(&mut self, edge: StoredEdge) -> Result<()> {
        if !self.has_node(edge.source) || !self.has_node(edge.target) {
            return Err(Error::Corrupt(format!(
                "checkpoint edge {} refers to a missing endpoint",
                edge.id
            )));
        }
        let index = edge.id as usize;
        grow_slots(&mut self.edges, index);
        if self.edges[index].is_some() {
            return Err(Error::Corrupt(format!(
                "checkpoint contains duplicate edge {}",
                edge.id
            )));
        }
        self.edge_count += 1;
        self.indexed_vectors += edge.vector_count as usize;
        self.indexed_edge_vectors += edge.vector_count as usize;
        self.edges_by_label
            .entry(edge.label)
            .or_default()
            .insert(edge.id);
        if !self.loading_snapshot {
            self.outgoing.push(edge.source, edge.id);
            self.incoming.push(edge.target, edge.id);
        }
        self.edges[index] = Some(edge);
        Ok(())
    }

    pub(crate) fn append_snapshot_node_batch(
        &self,
        cursor: &mut usize,
        batch_size: usize,
        snapshot: &mut crate::codec::SnapshotBuilder,
    ) -> Result<usize> {
        let mut appended = 0;
        let slots = self.next_node_id() as usize;
        while *cursor < slots && appended < batch_size {
            if let Some(node) = self.node_record(*cursor as NodeId) {
                let start = node.vector_offset;
                let float_count = node.vector_count as usize * self.dimension;
                let properties = node
                    .properties
                    .encoded(self.snapshot_map.as_deref(), &self.owned_properties)?;
                match self.vector_data.snapshot_range(
                    start,
                    float_count,
                    snapshot.vector_encoding(),
                )? {
                    SnapshotVectorRange::Encoded(vectors) => snapshot.append_node_encoded(
                        node.id,
                        node.generation.get(),
                        node.label,
                        &properties,
                        node.vector_count,
                        vectors,
                    )?,
                    SnapshotVectorRange::F32(vectors) => snapshot.append_node_f32(
                        node.id,
                        node.generation.get(),
                        node.label,
                        &properties,
                        node.vector_count,
                        vectors,
                    )?,
                    SnapshotVectorRange::OwnedF32(vectors) => snapshot.append_node_f32(
                        node.id,
                        node.generation.get(),
                        node.label,
                        &properties,
                        node.vector_count,
                        &vectors,
                    )?,
                }
                appended += 1;
            }
            *cursor += 1;
        }
        Ok(appended)
    }

    pub(crate) fn append_snapshot_edge_batch(
        &self,
        cursor: &mut usize,
        batch_size: usize,
        snapshot: &mut crate::codec::SnapshotBuilder,
    ) -> Result<usize> {
        let mut appended = 0;
        let slots = self.next_edge_id() as usize;
        while *cursor < slots && appended < batch_size {
            if let Some(edge) = self.edge_record(*cursor as EdgeId) {
                let start = edge.vector_offset;
                let float_count = edge.vector_count as usize * self.dimension;
                let properties = edge
                    .properties
                    .encoded(self.snapshot_map.as_deref(), &self.owned_properties)?;
                match self.vector_data.snapshot_range(
                    start,
                    float_count,
                    snapshot.vector_encoding(),
                )? {
                    SnapshotVectorRange::Encoded(vectors) => snapshot.append_edge_encoded(
                        edge.id,
                        edge.generation.get(),
                        edge.source,
                        edge.target,
                        edge.label,
                        &properties,
                        edge.vector_count,
                        vectors,
                    )?,
                    SnapshotVectorRange::F32(vectors) => snapshot.append_edge_f32(
                        edge.id,
                        edge.generation.get(),
                        edge.source,
                        edge.target,
                        edge.label,
                        &properties,
                        edge.vector_count,
                        vectors,
                    )?,
                    SnapshotVectorRange::OwnedF32(vectors) => snapshot.append_edge_f32(
                        edge.id,
                        edge.generation.get(),
                        edge.source,
                        edge.target,
                        edge.label,
                        &properties,
                        edge.vector_count,
                        &vectors,
                    )?,
                }
                appended += 1;
            }
            *cursor += 1;
        }
        Ok(appended)
    }

    pub(crate) fn nodes_with_label(&self, label: LabelId) -> Vec<Node> {
        let Some(ids) = self.nodes_by_label.get(&label) else {
            return Vec::new();
        };
        ids.iter()
            .rev()
            .filter_map(|id| {
                let node = self.node(id)?;
                (node.label == label).then_some(node)
            })
            .collect()
    }

    pub(crate) fn elements_with_label(&self, label: LabelId, target: VectorTarget) -> ElementSet {
        let mut result = ElementSet::new();
        if target.accepts(ElementRef::Node(0))
            && let Some(ids) = self.nodes_by_label.get(&label)
        {
            result.clone_nodes_from(ids);
        }
        if target.accepts(ElementRef::Edge(0))
            && let Some(ids) = self.edges_by_label.get(&label)
        {
            result.clone_edges_from(ids);
        }
        result
    }

    pub(crate) fn element_filter_plan(
        &self,
        target: VectorTarget,
        filter: &ElementFilter,
    ) -> ElementFilterPlan {
        let full_count = if target.accepts(ElementRef::Node(0)) {
            self.node_count
        } else {
            0
        }
        .saturating_add(if target.accepts(ElementRef::Edge(0)) {
            self.edge_count
        } else {
            0
        });
        let label_count = filter.label.map(|label| {
            let mut count = 0u64;
            if target.accepts(ElementRef::Node(0)) {
                count = count.saturating_add(
                    self.nodes_by_label
                        .get(&label)
                        .map_or(0, RoaringTreemap::len),
                );
            }
            if target.accepts(ElementRef::Edge(0)) {
                count = count.saturating_add(
                    self.edges_by_label
                        .get(&label)
                        .map_or(0, RoaringTreemap::len),
                );
            }
            usize::try_from(count).unwrap_or(usize::MAX)
        });
        let overlay_count = if target.accepts(ElementRef::Node(0)) {
            self.node_overlays.len()
        } else {
            0
        }
        .saturating_add(if target.accepts(ElementRef::Edge(0)) {
            self.edge_overlays.len()
        } else {
            0
        });
        let property = self.mapped_property_index.as_ref().and_then(|index| {
            filter
                .properties
                .iter()
                .enumerate()
                .map(|(predicate, (key, value))| {
                    let range_len = crate::codec::numeric_value_index_key(value)
                        .and_then(|(tag, sortable)| {
                            self.mapped_numeric_property_index.as_ref().map(|index| {
                                index
                                    .range(
                                        *key,
                                        tag,
                                        Bound::Included(sortable),
                                        Bound::Included(sortable),
                                    )
                                    .len()
                            })
                        })
                        .unwrap_or_else(|| {
                            index
                                .range(*key, crate::codec::property_value_fingerprint(value))
                                .len()
                        });
                    (predicate, range_len.saturating_add(overlay_count))
                })
                .min_by_key(|(_, count)| *count)
        });

        match (label_count, property) {
            (Some(label), Some((predicate, property))) if property < label => ElementFilterPlan {
                strategy: ElementFilterStrategy::PropertyPosting,
                candidate_upper_bound: property,
                property_predicate: Some(predicate),
            },
            (Some(label), _) => ElementFilterPlan {
                strategy: ElementFilterStrategy::LabelPosting,
                candidate_upper_bound: label,
                property_predicate: None,
            },
            (None, Some((predicate, property))) => ElementFilterPlan {
                strategy: ElementFilterStrategy::PropertyPosting,
                candidate_upper_bound: property,
                property_predicate: Some(predicate),
            },
            (None, None) => ElementFilterPlan {
                strategy: ElementFilterStrategy::FullScan,
                candidate_upper_bound: full_count,
                property_predicate: None,
            },
        }
    }

    /// Evaluates an exact label/property predicate into the same compressed
    /// candidate representation consumed by traversal and vector operators.
    pub(crate) fn elements_matching(
        &self,
        target: VectorTarget,
        filter: &ElementFilter,
    ) -> ElementSet {
        let plan = self.element_filter_plan(target, filter);
        if plan.strategy == ElementFilterStrategy::PropertyPosting {
            let predicate = plan
                .property_predicate
                .expect("property posting plans identify their predicate");
            let (key, value) = &filter.properties[predicate];
            if let Some((tag, sortable)) = crate::codec::numeric_value_index_key(value)
                && let Some(index) = self.mapped_numeric_property_index.as_ref()
            {
                let range = index.range(
                    *key,
                    tag,
                    Bound::Included(sortable),
                    Bound::Included(sortable),
                );
                return self.elements_matching_numeric_property_range(target, filter, index, range);
            }
            let index = self
                .mapped_property_index
                .as_ref()
                .expect("property posting plans require a mapped index");
            let range = index.range(*key, crate::codec::property_value_fingerprint(value));
            return self.elements_matching_property_range(target, filter, index, range);
        }

        let mut result = ElementSet::new();
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = filter.label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten() {
                    if let Some(node) = self.node_record(id)
                        && stored_element_matches(
                            node.label,
                            node.properties,
                            filter,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_node(id);
                    }
                }
            } else {
                for node in self.node_records() {
                    if stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    ) {
                        result.insert_node(node.id);
                    }
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = filter.label {
                for id in self.edges_by_label.get(&label).into_iter().flatten() {
                    if let Some(edge) = self.edge_record(id)
                        && stored_element_matches(
                            edge.label,
                            edge.properties,
                            filter,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_edge(id);
                    }
                }
            } else {
                for edge in self.edge_records() {
                    if stored_element_matches(
                        edge.label,
                        edge.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    ) {
                        result.insert_edge(edge.id);
                    }
                }
            }
        }
        result
    }

    fn elements_matching_property_range(
        &self,
        target: VectorTarget,
        filter: &ElementFilter,
        index: &MappedPropertyIndex,
        range: std::ops::Range<usize>,
    ) -> ElementSet {
        let mut result = ElementSet::new();
        for ordinal in range {
            let entry = index.entry_at(ordinal);
            if entry.kind == 0 && target.accepts(ElementRef::Node(entry.id)) {
                if let Some(node) = self.node_record(entry.id)
                    && stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_node(entry.id);
                }
            } else if entry.kind == 1
                && target.accepts(ElementRef::Edge(entry.id))
                && let Some(edge) = self.edge_record(entry.id)
                && stored_element_matches(
                    edge.label,
                    edge.properties,
                    filter,
                    self.snapshot_map.as_deref(),
                    &self.owned_properties,
                )
            {
                result.insert_edge(entry.id);
            }
        }

        // The mapped posting table describes the immutable checkpoint. WAL
        // overlays are small and scanned exactly so property changes become
        // visible immediately without synchronous index maintenance.
        if target.accepts(ElementRef::Node(0)) {
            for (&id, record) in &self.node_overlays {
                if let Some(node) = record
                    && stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_node(id);
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            for (&id, record) in &self.edge_overlays {
                if let Some(edge) = record
                    && stored_element_matches(
                        edge.label,
                        edge.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_edge(id);
                }
            }
        }
        result
    }

    fn elements_matching_numeric_property_range(
        &self,
        target: VectorTarget,
        filter: &ElementFilter,
        index: &MappedNumericPropertyIndex,
        range: std::ops::Range<usize>,
    ) -> ElementSet {
        let mut result = ElementSet::new();
        for ordinal in range {
            let entry = index.entry_at(ordinal);
            if entry.kind == 0 && target.accepts(ElementRef::Node(entry.id)) {
                if let Some(node) = self.node_record(entry.id)
                    && stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_node(entry.id);
                }
            } else if entry.kind == 1
                && target.accepts(ElementRef::Edge(entry.id))
                && let Some(edge) = self.edge_record(entry.id)
                && stored_element_matches(
                    edge.label,
                    edge.properties,
                    filter,
                    self.snapshot_map.as_deref(),
                    &self.owned_properties,
                )
            {
                result.insert_edge(entry.id);
            }
        }
        if target.accepts(ElementRef::Node(0)) {
            for (&id, record) in &self.node_overlays {
                if let Some(node) = record
                    && stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_node(id);
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            for (&id, record) in &self.edge_overlays {
                if let Some(edge) = record
                    && stored_element_matches(
                        edge.label,
                        edge.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_edge(id);
                }
            }
        }
        result
    }

    pub(crate) fn numeric_range_plan(
        &self,
        target: VectorTarget,
        filter: &NumericRangeFilter,
    ) -> Result<NumericRangePlan> {
        let prepared = prepare_numeric_range(filter)?;
        let full_count = if target.accepts(ElementRef::Node(0)) {
            self.node_count
        } else {
            0
        }
        .saturating_add(if target.accepts(ElementRef::Edge(0)) {
            self.edge_count
        } else {
            0
        });
        let label_count = filter.label.map(|label| {
            let mut count = 0u64;
            if target.accepts(ElementRef::Node(0)) {
                count = count.saturating_add(
                    self.nodes_by_label
                        .get(&label)
                        .map_or(0, RoaringTreemap::len),
                );
            }
            if target.accepts(ElementRef::Edge(0)) {
                count = count.saturating_add(
                    self.edges_by_label
                        .get(&label)
                        .map_or(0, RoaringTreemap::len),
                );
            }
            usize::try_from(count).unwrap_or(usize::MAX)
        });
        let overlay_count = if target.accepts(ElementRef::Node(0)) {
            self.node_overlays.len()
        } else {
            0
        }
        .saturating_add(if target.accepts(ElementRef::Edge(0)) {
            self.edge_overlays.len()
        } else {
            0
        });
        let numeric_count = self.mapped_numeric_property_index.as_ref().map(|index| {
            index
                .range(filter.key, prepared.tag, prepared.lower, prepared.upper)
                .len()
                .saturating_add(overlay_count)
        });
        Ok(match (label_count, numeric_count) {
            (Some(label), Some(numeric)) if numeric < label => NumericRangePlan {
                strategy: NumericRangeStrategy::NumericPosting,
                candidate_upper_bound: numeric,
            },
            (Some(label), _) => NumericRangePlan {
                strategy: NumericRangeStrategy::LabelPosting,
                candidate_upper_bound: label,
            },
            (None, Some(numeric)) => NumericRangePlan {
                strategy: NumericRangeStrategy::NumericPosting,
                candidate_upper_bound: numeric,
            },
            (None, None) => NumericRangePlan {
                strategy: NumericRangeStrategy::FullScan,
                candidate_upper_bound: full_count,
            },
        })
    }

    /// Evaluates a same-typed integer or floating-point range without
    /// hydrating mapped records. Checkpoint postings are reconciled against WAL
    /// overlays so inserts and property changes are immediately visible.
    pub(crate) fn elements_matching_numeric_range(
        &self,
        target: VectorTarget,
        filter: &NumericRangeFilter,
    ) -> Result<ElementSet> {
        let prepared = prepare_numeric_range(filter)?;
        let plan = self.numeric_range_plan(target, filter)?;
        if plan.strategy == NumericRangeStrategy::NumericPosting {
            let index = self
                .mapped_numeric_property_index
                .as_ref()
                .expect("numeric posting plans require a mapped index");
            let range = index.range(filter.key, prepared.tag, prepared.lower, prepared.upper);
            let mut result = ElementSet::new();
            for ordinal in range {
                let entry = index.entry_at(ordinal);
                if entry.kind == 0 && target.accepts(ElementRef::Node(entry.id)) {
                    if let Some(overlay) = self.node_overlays.get(&entry.id) {
                        if let Some(node) = overlay
                            && stored_element_matches_numeric_range(
                                node.label,
                                node.properties,
                                filter,
                                prepared,
                                self.snapshot_map.as_deref(),
                                &self.owned_properties,
                            )
                        {
                            result.insert_node(entry.id);
                        }
                    } else if self
                        .node_record(entry.id)
                        .is_some_and(|node| filter.label.is_none_or(|label| node.label == label))
                    {
                        // Unlike equality fingerprints, numeric sort keys are
                        // exact and collision-free. Metadata CRC + open-time
                        // index validation lets immutable rows skip reparsing
                        // their property blobs here.
                        result.insert_node(entry.id);
                    }
                } else if entry.kind == 1 && target.accepts(ElementRef::Edge(entry.id)) {
                    if let Some(overlay) = self.edge_overlays.get(&entry.id) {
                        if let Some(edge) = overlay
                            && stored_element_matches_numeric_range(
                                edge.label,
                                edge.properties,
                                filter,
                                prepared,
                                self.snapshot_map.as_deref(),
                                &self.owned_properties,
                            )
                        {
                            result.insert_edge(entry.id);
                        }
                    } else if self
                        .edge_record(entry.id)
                        .is_some_and(|edge| filter.label.is_none_or(|label| edge.label == label))
                    {
                        result.insert_edge(entry.id);
                    }
                }
            }
            if target.accepts(ElementRef::Node(0)) {
                for (&id, record) in &self.node_overlays {
                    if let Some(node) = record
                        && stored_element_matches_numeric_range(
                            node.label,
                            node.properties,
                            filter,
                            prepared,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_node(id);
                    }
                }
            }
            if target.accepts(ElementRef::Edge(0)) {
                for (&id, record) in &self.edge_overlays {
                    if let Some(edge) = record
                        && stored_element_matches_numeric_range(
                            edge.label,
                            edge.properties,
                            filter,
                            prepared,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_edge(id);
                    }
                }
            }
            return Ok(result);
        }

        let mut result = ElementSet::new();
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = filter.label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten() {
                    if let Some(node) = self.node_record(id)
                        && stored_element_matches_numeric_range(
                            node.label,
                            node.properties,
                            filter,
                            prepared,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_node(id);
                    }
                }
            } else {
                for node in self.node_records() {
                    if stored_element_matches_numeric_range(
                        node.label,
                        node.properties,
                        filter,
                        prepared,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    ) {
                        result.insert_node(node.id);
                    }
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = filter.label {
                for id in self.edges_by_label.get(&label).into_iter().flatten() {
                    if let Some(edge) = self.edge_record(id)
                        && stored_element_matches_numeric_range(
                            edge.label,
                            edge.properties,
                            filter,
                            prepared,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_edge(id);
                    }
                }
            } else {
                for edge in self.edge_records() {
                    if stored_element_matches_numeric_range(
                        edge.label,
                        edge.properties,
                        filter,
                        prepared,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    ) {
                        result.insert_edge(edge.id);
                    }
                }
            }
        }
        Ok(result)
    }

    /// Executes scalar predicates into compressed sets, intersects them, and
    /// only then chooses exact or sketch/rerank vector execution. Predicate
    /// plans are computed first and evaluated from the lowest conservative
    /// candidate bound to minimize the live intermediate set.
    pub(crate) fn vector_search_filtered_adaptive(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        equality: Option<&ElementFilter>,
        numeric_ranges: &[NumericRangeFilter],
    ) -> Result<FilteredVectorSearchResult> {
        #[derive(Clone, Copy)]
        enum Constraint {
            Equality,
            Numeric(usize),
        }

        let equality_plan = equality.map(|filter| self.element_filter_plan(target, filter));
        let numeric_range_plans = numeric_ranges
            .iter()
            .map(|filter| self.numeric_range_plan(target, filter))
            .collect::<Result<Vec<_>>>()?;
        let mut execution =
            Vec::with_capacity(numeric_ranges.len() + usize::from(equality.is_some()));
        if let Some(plan) = equality_plan {
            execution.push((plan.candidate_upper_bound, Constraint::Equality));
        }
        execution.extend(
            numeric_range_plans
                .iter()
                .enumerate()
                .map(|(index, plan)| (plan.candidate_upper_bound, Constraint::Numeric(index))),
        );
        execution.sort_unstable_by_key(|(candidate_upper_bound, _)| *candidate_upper_bound);

        let mut candidates: Option<ElementSet> = None;
        for (_, constraint) in execution {
            let current = match constraint {
                Constraint::Equality => {
                    self.elements_matching(target, equality.expect("planned equality filter"))
                }
                Constraint::Numeric(index) => {
                    self.elements_matching_numeric_range(target, &numeric_ranges[index])?
                }
            };
            candidates = Some(match candidates {
                Some(previous) => previous.intersection(&current),
                None => current,
            });
            if candidates.as_ref().is_some_and(ElementSet::is_empty) {
                break;
            }
        }

        if let Some(candidates) = candidates {
            let vector_plan = self.vector_search_within_plan(&candidates);
            let hits = match vector_plan.strategy {
                VectorSearchStrategy::Exact => {
                    self.vector_search_within(query, &candidates, limit)?
                }
                VectorSearchStrategy::BinarySketchRerank => self.vector_search_within_approximate(
                    query,
                    &candidates,
                    limit,
                    vector_plan.candidate_vectors,
                )?,
            };
            Ok(FilteredVectorSearchResult {
                hits,
                candidate_elements: candidates.len(),
                equality_plan,
                numeric_range_plans,
                vector_plan,
            })
        } else {
            let vector_plan = self.vector_search_plan(target, None);
            let hits = match vector_plan.strategy {
                VectorSearchStrategy::Exact => self.vector_search(query, target, limit, None)?,
                VectorSearchStrategy::BinarySketchRerank => self.vector_search_approximate(
                    query,
                    target,
                    limit,
                    None,
                    vector_plan.candidate_vectors,
                )?,
            };
            Ok(FilteredVectorSearchResult {
                hits,
                candidate_elements: self.eligible_element_upper_bound(target, None) as u64,
                equality_plan,
                numeric_range_plans,
                vector_plan,
            })
        }
    }

    pub(crate) fn expand_element_set(
        &self,
        seeds: &ElementSet,
        direction: Direction,
        filter: EdgeFilter,
    ) -> Result<ElementSet> {
        let mut result = ElementSet::new();
        for node in seeds.node_ids() {
            self.visit_neighbors(node, direction, filter, |neighbor, edge| {
                result.insert_node(neighbor);
                result.insert_edge(edge);
            })?;
        }
        Ok(result)
    }

    /// Computes the exact bounded neighborhood of a typed candidate set.
    ///
    /// Seed nodes are not included in the result. Every newly reached node is
    /// expanded at most once (at its shortest hop depth), while every matching
    /// edge encountered from a frontier is retained. This gives cyclic and
    /// parallel-edge graphs deterministic set semantics without application-side
    /// frontier materialization.
    pub(crate) fn expand_element_set_hops(
        &self,
        seeds: &ElementSet,
        direction: Direction,
        filter: EdgeFilter,
        max_hops: usize,
    ) -> Result<ElementSet> {
        let mut result = ElementSet::new();
        if max_hops == 0 || seeds.node_len() == 0 {
            return Ok(result);
        }

        let mut visited = ElementSet::new();
        let mut frontier = ElementSet::new();
        for node in seeds.node_ids() {
            visited.insert_node(node);
            frontier.insert_node(node);
        }

        for _ in 0..max_hops {
            let mut next = ElementSet::new();
            for node in frontier.node_ids() {
                self.visit_neighbors(node, direction, filter, |neighbor, edge| {
                    result.insert_edge(edge);
                    if !visited.contains(ElementRef::Node(neighbor)) {
                        visited.insert_node(neighbor);
                        next.insert_node(neighbor);
                        result.insert_node(neighbor);
                    }
                })?;
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        Ok(result)
    }

    /// Computes a node-only bounded neighborhood for graph-range retrieval.
    ///
    /// Unlike `expand_element_set_hops`, this does not retain every traversed
    /// edge. That distinction is material for broad ranges where edge IDs can
    /// outnumber reachable nodes several times over.
    pub(crate) fn nodes_within_hops(
        &self,
        seeds: &ElementSet,
        direction: Direction,
        edge_filter: EdgeFilter,
        max_hops: usize,
        include_seeds: bool,
        node_filter: Option<&ElementFilter>,
    ) -> Result<ElementSet> {
        let mut visited = ElementSet::new();
        for node in seeds.node_ids() {
            if !self.has_node(node) {
                return Err(Error::NotFound("node", node));
            }
            visited.insert_node(node);
        }
        let mut frontier = visited.clone();
        let mut result = if include_seeds {
            visited.clone()
        } else {
            ElementSet::new()
        };

        for _ in 0..max_hops {
            let mut next = ElementSet::new();
            for node in frontier.node_ids() {
                self.visit_neighbors(node, direction, edge_filter, |neighbor, _edge| {
                    if !visited.contains(ElementRef::Node(neighbor)) {
                        visited.insert_node(neighbor);
                        next.insert_node(neighbor);
                        result.insert_node(neighbor);
                    }
                })?;
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }

        if let Some(filter) = node_filter {
            let matching = self.elements_matching(VectorTarget::Nodes, filter);
            result = result.intersection(&matching);
        }
        Ok(result)
    }

    /// Finds an exact unweighted shortest path within explicit hop and work
    /// bounds. Frontier and adjacency order are normalized by stable IDs so a
    /// graph with multiple shortest paths returns the same evidence chain
    /// before and after checkpoint compaction.
    pub(crate) fn shortest_path(
        &self,
        start: NodeId,
        end: NodeId,
        options: &ShortestPathOptions,
    ) -> Result<ShortestPathResult> {
        if !self.has_node(start) {
            return Err(Error::NotFound("node", start));
        }
        if !self.has_node(end) {
            return Err(Error::NotFound("node", end));
        }
        if start == end {
            return Ok(ShortestPathResult {
                path: Some(ShortestPath {
                    nodes: vec![start],
                    edges: Vec::new(),
                }),
                strategy: ShortestPathStrategy::BreadthFirst,
                termination: ShortestPathTermination::Found,
                visited_nodes: 1,
                start_expanded_nodes: 0,
                end_expanded_nodes: 0,
                expanded_nodes: 0,
                examined_relationships: 0,
            });
        }
        if options.max_hops == 0 {
            return Ok(ShortestPathResult {
                path: None,
                strategy: ShortestPathStrategy::BreadthFirst,
                termination: ShortestPathTermination::NotFoundWithinHops,
                visited_nodes: 1,
                start_expanded_nodes: 0,
                end_expanded_nodes: 0,
                expanded_nodes: 0,
                examined_relationships: 0,
            });
        }

        if options.max_hops == 1 {
            self.shortest_path_breadth_first(start, end, options)
        } else {
            self.shortest_path_bidirectional(start, end, options)
        }
    }

    fn shortest_path_breadth_first(
        &self,
        start: NodeId,
        end: NodeId,
        options: &ShortestPathOptions,
    ) -> Result<ShortestPathResult> {
        let mut visited = HashSet::from([start]);
        let mut parents: HashMap<NodeId, (NodeId, EdgeId)> = HashMap::new();
        let mut frontier = vec![start];
        let mut expanded_nodes = 0;
        let mut examined_relationships = 0;

        for _depth in 0..options.max_hops {
            frontier.sort_unstable();
            let mut next = Vec::new();
            for node in frontier {
                if expanded_nodes == options.max_expansions {
                    return Ok(ShortestPathResult {
                        path: None,
                        strategy: ShortestPathStrategy::BreadthFirst,
                        termination: ShortestPathTermination::ExpansionLimit,
                        visited_nodes: visited.len(),
                        start_expanded_nodes: expanded_nodes,
                        end_expanded_nodes: 0,
                        expanded_nodes,
                        examined_relationships,
                    });
                }
                expanded_nodes += 1;
                let mut adjacent = Vec::new();
                self.visit_neighbors(
                    node,
                    options.direction,
                    options.edge_filter,
                    |neighbor, edge| adjacent.push((neighbor, edge)),
                )?;
                adjacent.sort_unstable();
                examined_relationships = examined_relationships.saturating_add(adjacent.len());
                for (neighbor, edge) in adjacent {
                    if !visited.insert(neighbor) {
                        continue;
                    }
                    parents.insert(neighbor, (node, edge));
                    if neighbor == end {
                        let path = reconstruct_shortest_path(start, end, &parents);
                        return Ok(ShortestPathResult {
                            path: Some(path),
                            strategy: ShortestPathStrategy::BreadthFirst,
                            termination: ShortestPathTermination::Found,
                            visited_nodes: visited.len(),
                            start_expanded_nodes: expanded_nodes,
                            end_expanded_nodes: 0,
                            expanded_nodes,
                            examined_relationships,
                        });
                    }
                    next.push(neighbor);
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }

        Ok(ShortestPathResult {
            path: None,
            strategy: ShortestPathStrategy::BreadthFirst,
            termination: ShortestPathTermination::NotFoundWithinHops,
            visited_nodes: visited.len(),
            start_expanded_nodes: expanded_nodes,
            end_expanded_nodes: 0,
            expanded_nodes,
            examined_relationships,
        })
    }

    fn shortest_path_bidirectional(
        &self,
        start: NodeId,
        end: NodeId,
        options: &ShortestPathOptions,
    ) -> Result<ShortestPathResult> {
        let mut forward_depths = HashMap::from([(start, 0usize)]);
        let mut reverse_depths = HashMap::from([(end, 0usize)]);
        let mut forward_parents: HashMap<NodeId, (NodeId, EdgeId)> = HashMap::new();
        let mut reverse_next: HashMap<NodeId, (NodeId, EdgeId)> = HashMap::new();
        let mut forward_frontier = vec![start];
        let mut reverse_frontier = vec![end];
        let mut forward_depth = 0usize;
        let mut reverse_depth = 0usize;
        let mut best_length = None;
        let mut expanded_nodes = 0usize;
        let mut start_expanded_nodes = 0usize;
        let mut end_expanded_nodes = 0usize;
        let mut examined_relationships = 0usize;

        loop {
            let proven_length =
                best_length.filter(|&length| forward_depth.saturating_add(reverse_depth) >= length);
            if let Some(length) = proven_length {
                return Ok(ShortestPathResult {
                    path: best_bidirectional_path(
                        start,
                        end,
                        length,
                        &forward_depths,
                        &reverse_depths,
                        &forward_parents,
                        &reverse_next,
                    ),
                    strategy: ShortestPathStrategy::BidirectionalBreadthFirst,
                    termination: ShortestPathTermination::Found,
                    visited_nodes: visited_union_len(&forward_depths, &reverse_depths),
                    start_expanded_nodes,
                    end_expanded_nodes,
                    expanded_nodes,
                    examined_relationships,
                });
            }
            if forward_depth.saturating_add(reverse_depth) >= options.max_hops
                || forward_frontier.is_empty()
                || reverse_frontier.is_empty()
            {
                let path = best_length.and_then(|length| {
                    best_bidirectional_path(
                        start,
                        end,
                        length,
                        &forward_depths,
                        &reverse_depths,
                        &forward_parents,
                        &reverse_next,
                    )
                });
                return Ok(ShortestPathResult {
                    termination: if path.is_some() {
                        ShortestPathTermination::Found
                    } else {
                        ShortestPathTermination::NotFoundWithinHops
                    },
                    path,
                    strategy: ShortestPathStrategy::BidirectionalBreadthFirst,
                    visited_nodes: visited_union_len(&forward_depths, &reverse_depths),
                    start_expanded_nodes,
                    end_expanded_nodes,
                    expanded_nodes,
                    examined_relationships,
                });
            }

            // A bidirectional proof advances only after a complete layer. If
            // exactly one frontier fits in the remaining expansion budget,
            // prefer it even when its adjacency estimate is more expensive;
            // partially expanding the cheaper layer could otherwise exhaust
            // the budget without proving a path already visible from the
            // other side. When both (or neither) fit, score the complete next
            // layer by node expansions plus a cheap upper bound on adjacency
            // reads, then retain forward order as the deterministic tie-break.
            // The estimate is conservative across mapped CSR and WAL overlays;
            // filters are applied during the actual expansion.
            let reverse_search_direction = reverse_direction(options.direction);
            let remaining_expansions = options.max_expansions.saturating_sub(expanded_nodes);
            let forward_fits = forward_frontier.len() <= remaining_expansions;
            let reverse_fits = reverse_frontier.len() <= remaining_expansions;
            let expand_forward = match (forward_fits, reverse_fits) {
                (true, false) => true,
                (false, true) => false,
                (true, true) | (false, false) => {
                    let forward_work =
                        self.frontier_work_upper_bound(&forward_frontier, options.direction);
                    let reverse_work =
                        self.frontier_work_upper_bound(&reverse_frontier, reverse_search_direction);
                    forward_work < reverse_work
                        || (forward_work == reverse_work
                            && forward_frontier.len() <= reverse_frontier.len())
                }
            };
            let (frontier, own_depths, other_depths, parents, direction) = if expand_forward {
                (
                    &mut forward_frontier,
                    &mut forward_depths,
                    &reverse_depths,
                    &mut forward_parents,
                    options.direction,
                )
            } else {
                (
                    &mut reverse_frontier,
                    &mut reverse_depths,
                    &forward_depths,
                    &mut reverse_next,
                    reverse_search_direction,
                )
            };
            frontier.sort_unstable();
            let mut next = Vec::new();
            for node in std::mem::take(frontier) {
                if expanded_nodes == options.max_expansions {
                    return Ok(ShortestPathResult {
                        path: None,
                        strategy: ShortestPathStrategy::BidirectionalBreadthFirst,
                        termination: ShortestPathTermination::ExpansionLimit,
                        visited_nodes: visited_union_len(&forward_depths, &reverse_depths),
                        start_expanded_nodes,
                        end_expanded_nodes,
                        expanded_nodes,
                        examined_relationships,
                    });
                }
                expanded_nodes += 1;
                if expand_forward {
                    start_expanded_nodes += 1;
                } else {
                    end_expanded_nodes += 1;
                }
                let mut adjacent = Vec::new();
                self.visit_neighbors(node, direction, options.edge_filter, |neighbor, edge| {
                    adjacent.push((neighbor, edge));
                })?;
                adjacent.sort_unstable();
                examined_relationships = examined_relationships.saturating_add(adjacent.len());
                let next_depth = own_depths[&node] + 1;
                for (neighbor, edge) in adjacent {
                    if own_depths.contains_key(&neighbor) {
                        continue;
                    }
                    own_depths.insert(neighbor, next_depth);
                    parents.insert(neighbor, (node, edge));
                    if let Some(&other_depth) = other_depths.get(&neighbor) {
                        let length = next_depth.saturating_add(other_depth);
                        if length <= options.max_hops {
                            best_length = Some(best_length.map_or(length, |best| best.min(length)));
                        }
                    }
                    next.push(neighbor);
                }
            }
            *frontier = next;
            if expand_forward {
                forward_depth += 1;
            } else {
                reverse_depth += 1;
            }
        }
    }

    fn frontier_work_upper_bound(&self, frontier: &[NodeId], direction: Direction) -> usize {
        frontier.iter().fold(0usize, |work, &node| {
            work.saturating_add(1)
                .saturating_add(self.adjacency_len_upper_bound(node, direction))
        })
    }

    fn adjacency_len_upper_bound(&self, node: NodeId, direction: Direction) -> usize {
        let mut count = 0usize;
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            count = count.saturating_add(
                self.base_outgoing
                    .as_ref()
                    .map_or(0, |adjacency| adjacency.get(node).len()),
            );
            count = count.saturating_add(self.outgoing.get(node).len());
        }
        if matches!(direction, Direction::Incoming | Direction::Both) {
            count = count.saturating_add(
                self.base_incoming
                    .as_ref()
                    .map_or(0, |adjacency| adjacency.get(node).len()),
            );
            count = count.saturating_add(self.incoming.get(node).len());
        }
        count
    }

    pub(crate) fn vector_search_graph_range_adaptive(
        &self,
        query: &[f32],
        seeds: &ElementSet,
        options: &GraphRangeSearchOptions,
    ) -> Result<GraphRangeSearchResult> {
        let candidates = self.nodes_within_hops(
            seeds,
            options.direction,
            options.edge_filter,
            options.max_hops,
            options.include_seeds,
            options.node_filter.as_ref(),
        )?;
        let plan = self.vector_search_within_plan(&candidates);
        let hits = match plan.strategy {
            VectorSearchStrategy::Exact => {
                self.vector_search_within(query, &candidates, options.limit)?
            }
            VectorSearchStrategy::BinarySketchRerank => self.vector_search_within_approximate(
                query,
                &candidates,
                options.limit,
                plan.candidate_vectors,
            )?,
        };
        Ok(GraphRangeSearchResult {
            hits,
            candidate_nodes: candidates.node_len(),
            plan,
        })
    }

    pub(crate) fn neighbors(
        &self,
        node: NodeId,
        direction: Direction,
        filter: EdgeFilter,
    ) -> Result<Vec<Edge>> {
        if self.node_record(node).is_none() {
            return Err(Error::NotFound("node", node));
        }
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            if let Some(base) = &self.base_outgoing {
                self.collect_edges(base.get(node), node, true, filter, &mut seen, &mut result);
            }
            self.collect_edges(
                self.outgoing.get(node),
                node,
                true,
                filter,
                &mut seen,
                &mut result,
            );
        }
        if matches!(direction, Direction::Incoming | Direction::Both) {
            if let Some(base) = &self.base_incoming {
                self.collect_edges(base.get(node), node, false, filter, &mut seen, &mut result);
            }
            self.collect_edges(
                self.incoming.get(node),
                node,
                false,
                filter,
                &mut seen,
                &mut result,
            );
        }
        Ok(result)
    }

    /// Visits adjacent node/edge IDs without materializing edge records or a
    /// result vector. Immutable CSR-only nodes take a zero-allocation path;
    /// nodes touched by the WAL use a small dedup set to reconcile base and
    /// delta adjacency while preserving parallel edges.
    pub(crate) fn visit_neighbors(
        &self,
        node: NodeId,
        direction: Direction,
        filter: EdgeFilter,
        mut visitor: impl FnMut(NodeId, EdgeId),
    ) -> Result<()> {
        if !self.has_node(node) {
            return Err(Error::NotFound("node", node));
        }
        let out_delta_empty = self.outgoing.is_empty();
        let in_delta_empty = self.incoming.is_empty();
        let has_out_delta = !out_delta_empty && !self.outgoing.get(node).is_empty();
        let has_in_delta = !in_delta_empty && !self.incoming.get(node).is_empty();
        let needs_dedup = (matches!(direction, Direction::Outgoing | Direction::Both)
            && has_out_delta)
            || (matches!(direction, Direction::Incoming | Direction::Both) && has_in_delta);
        let mut seen = needs_dedup.then(HashSet::new);

        if matches!(direction, Direction::Outgoing | Direction::Both) {
            if let Some(base) = &self.base_outgoing {
                self.visit_neighbor_slice(
                    base.get(node),
                    node,
                    AdjacencySide::Outgoing,
                    filter,
                    seen.as_mut(),
                    &mut visitor,
                );
            }
            if !out_delta_empty {
                self.visit_neighbor_slice(
                    self.outgoing.get(node),
                    node,
                    AdjacencySide::Outgoing,
                    filter,
                    seen.as_mut(),
                    &mut visitor,
                );
            }
        }
        // In an immutable bidirectional CSR, the only edge present in both
        // slices for one node is a self-loop. Outgoing already emitted it.
        let skip_incoming_self = direction == Direction::Both && seen.is_none();
        if matches!(direction, Direction::Incoming | Direction::Both) {
            if let Some(base) = &self.base_incoming {
                self.visit_neighbor_slice(
                    base.get(node),
                    node,
                    AdjacencySide::Incoming {
                        skip_self: skip_incoming_self,
                    },
                    filter,
                    seen.as_mut(),
                    &mut visitor,
                );
            }
            if !in_delta_empty {
                self.visit_neighbor_slice(
                    self.incoming.get(node),
                    node,
                    AdjacencySide::Incoming {
                        skip_self: skip_incoming_self,
                    },
                    filter,
                    seen.as_mut(),
                    &mut visitor,
                );
            }
        }
        Ok(())
    }

    pub(crate) fn one_hop_plan(&self, query: &OneHopQuery) -> OneHopPlan {
        let edge_candidate_upper_bound = query.edge_label.map_or(self.edge_count, |label| {
            self.edges_by_label
                .get(&label)
                .map_or(0, |ids| usize::try_from(ids.len()).unwrap_or(usize::MAX))
        });
        let start_candidate_upper_bound = self
            .element_filter_plan(VectorTarget::Nodes, &query.start)
            .candidate_upper_bound;
        let end_candidate_upper_bound = self
            .element_filter_plan(VectorTarget::Nodes, &query.end)
            .candidate_upper_bound;
        let average_directional_degree = if self.node_count == 0 {
            0
        } else {
            self.edge_count.div_ceil(self.node_count)
        };
        let direction_factor = if query.direction == Direction::Both {
            2
        } else {
            1
        };
        let start_edge_visits = start_candidate_upper_bound
            .saturating_mul(average_directional_degree)
            .saturating_mul(direction_factor);
        let end_edge_visits = end_candidate_upper_bound
            .saturating_mul(average_directional_degree)
            .saturating_mul(direction_factor);
        let (estimated_edge_visits, _, strategy) = [
            (edge_candidate_upper_bound, 0_u8, OneHopStrategy::EdgeScan),
            (start_edge_visits, 1, OneHopStrategy::StartAdjacency),
            (end_edge_visits, 2, OneHopStrategy::EndAdjacency),
        ]
        .into_iter()
        .min_by_key(|&(cost, priority, _)| (cost, priority))
        .expect("one-hop planner always has physical alternatives");
        OneHopPlan {
            strategy,
            estimated_edge_visits,
            edge_candidate_upper_bound,
            start_candidate_upper_bound,
            end_candidate_upper_bound,
        }
    }

    pub(crate) fn match_one_hop(&self, query: &OneHopQuery) -> Vec<PatternMatch> {
        if query.limit == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(query.limit.min(1024));
        match self.one_hop_plan(query).strategy {
            OneHopStrategy::EdgeScan => {
                let candidates: Box<dyn Iterator<Item = EdgeId> + '_> = match query.edge_label {
                    Some(label) => Box::new(self.edges_by_label.get(&label).into_iter().flatten()),
                    None => Box::new(self.edge_records().map(|edge| edge.id)),
                };
                for edge_id in candidates {
                    let Some(edge) = self.edge_record(edge_id) else {
                        continue;
                    };
                    if query.edge_label.is_some_and(|label| edge.label != label) {
                        continue;
                    }
                    let (orientations, orientation_count) = match query.direction {
                        Direction::Outgoing => ([(edge.source, edge.target); 2], 1),
                        Direction::Incoming => ([(edge.target, edge.source); 2], 1),
                        Direction::Both if edge.source != edge.target => {
                            ([(edge.source, edge.target), (edge.target, edge.source)], 2)
                        }
                        Direction::Both => ([(edge.source, edge.target); 2], 1),
                    };
                    for &(start, end) in &orientations[..orientation_count] {
                        let Some(start_node) = self.node_record(start) else {
                            continue;
                        };
                        let Some(end_node) = self.node_record(end) else {
                            continue;
                        };
                        if stored_node_matches(
                            &start_node,
                            &query.start,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        ) && stored_node_matches(
                            &end_node,
                            &query.end,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        ) {
                            result.push(PatternMatch {
                                start,
                                edge: edge.id,
                                end,
                            });
                            if result.len() == query.limit {
                                return result;
                            }
                        }
                    }
                }
            }
            OneHopStrategy::StartAdjacency => {
                let starts = self.elements_matching(VectorTarget::Nodes, &query.start);
                for start in starts.node_ids() {
                    self.visit_neighbors(
                        start,
                        query.direction,
                        EdgeFilter {
                            label: query.edge_label,
                        },
                        |end, edge| {
                            if result.len() == query.limit {
                                return;
                            }
                            let Some(end_node) = self.node_record(end) else {
                                return;
                            };
                            if stored_node_matches(
                                &end_node,
                                &query.end,
                                self.snapshot_map.as_deref(),
                                &self.owned_properties,
                            ) {
                                result.push(PatternMatch { start, edge, end });
                            }
                        },
                    )
                    .expect("planned start candidates are existing nodes");
                    if result.len() == query.limit {
                        break;
                    }
                }
            }
            OneHopStrategy::EndAdjacency => {
                let ends = self.elements_matching(VectorTarget::Nodes, &query.end);
                let reverse_direction = match query.direction {
                    Direction::Outgoing => Direction::Incoming,
                    Direction::Incoming => Direction::Outgoing,
                    Direction::Both => Direction::Both,
                };
                for end in ends.node_ids() {
                    self.visit_neighbors(
                        end,
                        reverse_direction,
                        EdgeFilter {
                            label: query.edge_label,
                        },
                        |start, edge| {
                            if result.len() == query.limit {
                                return;
                            }
                            let Some(start_node) = self.node_record(start) else {
                                return;
                            };
                            if stored_node_matches(
                                &start_node,
                                &query.start,
                                self.snapshot_map.as_deref(),
                                &self.owned_properties,
                            ) {
                                result.push(PatternMatch { start, edge, end });
                            }
                        },
                    )
                    .expect("planned end candidates are existing nodes");
                    if result.len() == query.limit {
                        break;
                    }
                }
            }
        }
        result
    }

    pub(crate) fn match_semantic_one_hop(
        &self,
        vector_query: &[f32],
        query: &SemanticOneHopQuery,
    ) -> Result<Vec<SemanticPatternMatch>> {
        validate_semantic_one_hop(query)?;
        if query.pattern.limit == 0 {
            return Ok(Vec::new());
        }
        let seeds = self.vector_search_adaptive(
            vector_query,
            VectorTarget::Nodes,
            query.seed_count,
            query.pattern.start.label,
        )?;
        let prepared = vector::prepare_query(vector_query, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(false)?;
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for seed in seeds {
            let ElementRef::Node(start) = seed.element else {
                continue;
            };
            let Some(start_node) = self.node_record(start) else {
                continue;
            };
            if !stored_node_matches(
                &start_node,
                &query.pattern.start,
                self.snapshot_map.as_deref(),
                &self.owned_properties,
            ) {
                continue;
            }
            for edge in self.neighbors(
                start,
                query.pattern.direction,
                EdgeFilter {
                    label: query.pattern.edge_label,
                },
            )? {
                let end = match query.pattern.direction {
                    Direction::Outgoing if edge.source == start => edge.target,
                    Direction::Incoming if edge.target == start => edge.source,
                    Direction::Both if edge.source == start => edge.target,
                    Direction::Both if edge.target == start => edge.source,
                    _ => continue,
                };
                let Some(end_node) = self.node_record(end) else {
                    continue;
                };
                if !stored_node_matches(
                    &end_node,
                    &query.pattern.end,
                    self.snapshot_map.as_deref(),
                    &self.owned_properties,
                ) {
                    continue;
                }
                let pattern = PatternMatch {
                    start,
                    edge: edge.id,
                    end,
                };
                if !seen.insert((pattern.start, pattern.edge, pattern.end)) {
                    continue;
                }
                let stored_edge = self.edge_record(edge.id).unwrap();
                let edge_score = self
                    .element_score(
                        &prepared,
                        stored_edge.vector_offset,
                        stored_edge.vector_count,
                        &scorer,
                    )?
                    .map(|score| score.0);
                let end_score = self
                    .element_score(
                        &prepared,
                        end_node.vector_offset,
                        end_node.vector_count,
                        &scorer,
                    )?
                    .map(|score| score.0);
                let mut weighted = seed.score * query.start_weight;
                let mut total_weight = query.start_weight;
                if let Some(score) = edge_score {
                    weighted += score * query.edge_weight;
                    total_weight += query.edge_weight;
                }
                if let Some(score) = end_score {
                    weighted += score * query.end_weight;
                    total_weight += query.end_weight;
                }
                result.push(SemanticPatternMatch {
                    pattern,
                    score: weighted / total_weight,
                    start_score: seed.score,
                    edge_score,
                    end_score,
                });
            }
        }
        result.sort_unstable_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.pattern.start.cmp(&right.pattern.start))
                .then_with(|| left.pattern.edge.cmp(&right.pattern.edge))
                .then_with(|| left.pattern.end.cmp(&right.pattern.end))
        });
        result.truncate(query.pattern.limit);
        Ok(result)
    }

    fn collect_edges(
        &self,
        ids: &[EdgeId],
        node: NodeId,
        outgoing: bool,
        filter: EdgeFilter,
        seen: &mut HashSet<EdgeId>,
        result: &mut Vec<Edge>,
    ) {
        for id in ids {
            if !seen.insert(*id) {
                continue;
            }
            let Some(edge) = self.edge(*id) else {
                continue;
            };
            if (outgoing && edge.source != node) || (!outgoing && edge.target != node) {
                continue;
            }
            if filter.label.is_none_or(|label| edge.label == label) {
                result.push(edge);
            }
        }
    }

    fn visit_neighbor_slice(
        &self,
        ids: &[EdgeId],
        node: NodeId,
        side: AdjacencySide,
        filter: EdgeFilter,
        mut seen: Option<&mut HashSet<EdgeId>>,
        visitor: &mut impl FnMut(NodeId, EdgeId),
    ) {
        let outgoing = matches!(side, AdjacencySide::Outgoing);
        let skip_self = matches!(side, AdjacencySide::Incoming { skip_self: true });
        if self.edges.is_empty()
            && self.edge_overlays.is_empty()
            && let Some(records) = &self.mapped_edges
        {
            for id in ids {
                if seen.as_mut().is_some_and(|seen| !seen.insert(*id)) {
                    continue;
                }
                if let Some(neighbor) = records.neighbor(*id, node, outgoing, filter.label) {
                    if skip_self && neighbor == node {
                        continue;
                    }
                    visitor(neighbor, *id);
                }
            }
            return;
        }
        for id in ids {
            if seen.as_mut().is_some_and(|seen| !seen.insert(*id)) {
                continue;
            }
            let Some(edge) = self.edge_record(*id) else {
                continue;
            };
            if (outgoing && edge.source != node)
                || (!outgoing && edge.target != node)
                || filter.label.is_some_and(|label| edge.label != label)
            {
                continue;
            }
            let neighbor = if outgoing { edge.target } else { edge.source };
            if !skip_self || neighbor != node {
                visitor(neighbor, edge.id);
            }
        }
    }

    pub(crate) fn vector_search(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<VectorHit>> {
        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(true)?;
        let mut top = TopK::new(limit);
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten().rev() {
                    let Some(node) = self.node_record(id) else {
                        continue;
                    };
                    if node.label != label {
                        continue;
                    }
                    self.score_element(
                        &query,
                        ElementRef::Node(node.id),
                        node.vector_offset,
                        node.vector_count,
                        &scorer,
                        &mut top,
                    )?;
                }
            } else {
                self.visit_node_vector_fields(|id, _label, vector_offset, vector_count| {
                    self.score_element(
                        &query,
                        ElementRef::Node(id),
                        vector_offset,
                        vector_count,
                        &scorer,
                        &mut top,
                    )
                })?;
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = label {
                for id in self.edges_by_label.get(&label).into_iter().flatten().rev() {
                    let Some(edge) = self.edge_record(id) else {
                        continue;
                    };
                    if edge.label != label {
                        continue;
                    }
                    self.score_element(
                        &query,
                        ElementRef::Edge(edge.id),
                        edge.vector_offset,
                        edge.vector_count,
                        &scorer,
                        &mut top,
                    )?;
                }
            } else {
                self.visit_edge_vector_fields(|id, _label, vector_offset, vector_count| {
                    self.score_element(
                        &query,
                        ElementRef::Edge(id),
                        vector_offset,
                        vector_count,
                        &scorer,
                        &mut top,
                    )
                })?;
            }
        }
        Ok(top.finish())
    }

    /// Exact vector search over a compressed graph-derived candidate set. The
    /// set is applied before scoring, so graph constraints do not suffer the
    /// recall loss of post-filtering a global top-k result.
    pub(crate) fn vector_search_within(
        &self,
        query: &[f32],
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<VectorHit>> {
        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        for id in allowed.node_ids() {
            let element = ElementRef::Node(id);
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            self.score_element(
                &query,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        for id in allowed.edge_ids() {
            let element = ElementRef::Edge(id);
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            self.score_element(
                &query,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        Ok(top.finish())
    }

    pub(crate) fn vector_search_within_approximate(
        &self,
        query: &[f32],
        allowed: &ElementSet,
        limit: usize,
        candidate_elements: usize,
    ) -> Result<Vec<VectorHit>> {
        if candidate_elements == 0 {
            return Err(Error::InvalidArgument(
                "approximate candidate budget must be greater than zero".into(),
            ));
        }
        let allowed_elements = usize::try_from(allowed.len()).unwrap_or(usize::MAX);
        let base_float_count = self.vector_data.base_float_count();
        if self.similarity != Similarity::Cosine
            || base_float_count == 0
            || candidate_elements >= allowed_elements
        {
            return self.vector_search_within(query, allowed, limit);
        }

        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let index = self
            .sketch_index
            .get_or_init(|| self.build_sketch_index().map_err(|error| error.to_string()))
            .as_ref()
            .map_err(|message| Error::Corrupt(message.clone()))?;
        let candidates = index.candidate_entries(
            &query,
            VectorTarget::Both,
            None,
            Some(allowed),
            candidate_elements.max(limit),
        );
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        let mut scored = HashSet::with_capacity(candidates.len());
        for candidate in candidates {
            let Some((label, vector_offset, vector_count)) =
                self.element_vector_fields(candidate.element)
            else {
                continue;
            };
            if label != candidate.label
                || candidate.float_offset < vector_offset
                || candidate.float_offset >= vector_offset + vector_count as usize * self.dimension
                || !scored.insert(candidate.element)
            {
                continue;
            }
            self.score_element(
                &query,
                candidate.element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }

        // The persisted sketch covers the immutable base. Search only allowed
        // WAL elements exhaustively, preserving read-your-writes semantics.
        for element in allowed
            .node_ids()
            .map(ElementRef::Node)
            .chain(allowed.edge_ids().map(ElementRef::Edge))
        {
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            if vector_offset < base_float_count || !scored.insert(element) {
                continue;
            }
            self.score_element(
                &query,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        Ok(top.finish())
    }

    pub(crate) fn vector_search_within_plan(&self, allowed: &ElementSet) -> VectorSearchPlan {
        let estimated_vectors = self.estimated_set_vector_count(allowed);
        let estimated_floats = estimated_vectors.saturating_mul(self.dimension);
        // Candidate-set scans gather non-contiguous records. Measured
        // crossovers differ substantially by vector width: reranking 5k of
        // 20k 768-D MoReVec rows wins, while exact still beats reranking 20k
        // of 100k 200-D VIBE rows. Keep this policy separate from contiguous
        // whole-column search and make the candidate fraction explicit.
        let candidate_vectors =
            adaptive_candidate_budget_for_set(estimated_vectors, self.dimension);
        let strategy = if self.similarity == Similarity::Cosine
            && self.vector_data.base_float_count() != 0
            && candidate_vectors < estimated_vectors
        {
            VectorSearchStrategy::BinarySketchRerank
        } else {
            VectorSearchStrategy::Exact
        };
        VectorSearchPlan {
            strategy,
            estimated_vectors,
            estimated_floats,
            candidate_vectors: if strategy == VectorSearchStrategy::Exact {
                estimated_vectors
            } else {
                candidate_vectors
            },
        }
    }

    pub(crate) fn vector_search_within_adaptive(
        &self,
        query: &[f32],
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<VectorHit>> {
        let plan = self.vector_search_within_plan(allowed);
        match plan.strategy {
            VectorSearchStrategy::Exact => self.vector_search_within(query, allowed, limit),
            VectorSearchStrategy::BinarySketchRerank => {
                self.vector_search_within_approximate(query, allowed, limit, plan.candidate_vectors)
            }
        }
    }

    /// Scores whole graph elements with weighted late interaction: each query
    /// vector takes its best matching vector facet on an element, then those
    /// per-query maxima are averaged by weight. This naturally supports token,
    /// chunk, structural/context, and multimodal facets in one embedding space.
    pub(crate) fn late_interaction_search(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<LateInteractionHit>> {
        let (queries, weights) =
            prepare_late_interaction_queries(queries, weights, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(true)?;
        let mut top = TopK::new(limit);
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten().rev() {
                    let Some(node) = self.node_record(id) else {
                        continue;
                    };
                    if node.label == label {
                        self.score_late_interaction_element(
                            &queries,
                            &weights,
                            ElementRef::Node(node.id),
                            node.vector_offset,
                            node.vector_count,
                            &scorer,
                            &mut top,
                        )?;
                    }
                }
            } else {
                for node in self.node_records() {
                    self.score_late_interaction_element(
                        &queries,
                        &weights,
                        ElementRef::Node(node.id),
                        node.vector_offset,
                        node.vector_count,
                        &scorer,
                        &mut top,
                    )?;
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = label {
                for id in self.edges_by_label.get(&label).into_iter().flatten().rev() {
                    let Some(edge) = self.edge_record(id) else {
                        continue;
                    };
                    if edge.label == label {
                        self.score_late_interaction_element(
                            &queries,
                            &weights,
                            ElementRef::Edge(edge.id),
                            edge.vector_offset,
                            edge.vector_count,
                            &scorer,
                            &mut top,
                        )?;
                    }
                }
            } else {
                for edge in self.edge_records() {
                    self.score_late_interaction_element(
                        &queries,
                        &weights,
                        ElementRef::Edge(edge.id),
                        edge.vector_offset,
                        edge.vector_count,
                        &scorer,
                        &mut top,
                    )?;
                }
            }
        }
        self.finish_late_interaction_hits(&queries, &weights, &scorer, top.finish())
    }

    pub(crate) fn late_interaction_search_within(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        let (queries, weights) =
            prepare_late_interaction_queries(queries, weights, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        for element in allowed
            .node_ids()
            .map(ElementRef::Node)
            .chain(allowed.edge_ids().map(ElementRef::Edge))
        {
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            self.score_late_interaction_element(
                &queries,
                &weights,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        self.finish_late_interaction_hits(&queries, &weights, &scorer, top.finish())
    }

    pub(crate) fn late_interaction_search_within_approximate(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        allowed: &ElementSet,
        limit: usize,
        candidate_elements: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        if candidate_elements == 0 {
            return Err(Error::InvalidArgument(
                "late-interaction candidate budget must be greater than zero".into(),
            ));
        }
        let allowed_elements = usize::try_from(allowed.len()).unwrap_or(usize::MAX);
        let base_float_count = self.vector_data.base_float_count();
        if self.similarity != Similarity::Cosine
            || base_float_count == 0
            || candidate_elements >= allowed_elements
        {
            return self.late_interaction_search_within(queries, weights, allowed, limit);
        }
        let (queries, weights) =
            prepare_late_interaction_queries(queries, weights, self.dimension, self.similarity)?;
        let index = self
            .sketch_index
            .get_or_init(|| self.build_sketch_index().map_err(|error| error.to_string()))
            .as_ref()
            .map_err(|message| Error::Corrupt(message.clone()))?;
        let candidates = index.candidate_elements_multivector(
            &queries,
            &weights,
            VectorTarget::Both,
            None,
            Some(allowed),
            candidate_elements.max(limit),
        );
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        let mut scored = HashSet::with_capacity(candidates.len());
        for candidate in candidates {
            if !self.sketch_entry_is_current(candidate) || !scored.insert(candidate.element) {
                continue;
            }
            let Some((vector_offset, vector_count)) = self.element_vector_span(candidate.element)
            else {
                continue;
            };
            self.score_late_interaction_element(
                &queries,
                &weights,
                candidate.element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        for element in allowed
            .node_ids()
            .map(ElementRef::Node)
            .chain(allowed.edge_ids().map(ElementRef::Edge))
        {
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            if vector_offset < base_float_count || !scored.insert(element) {
                continue;
            }
            self.score_late_interaction_element(
                &queries,
                &weights,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        self.finish_late_interaction_hits(&queries, &weights, &scorer, top.finish())
    }

    pub(crate) fn late_interaction_search_within_adaptive(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        let plan = self.vector_search_within_plan(allowed);
        match plan.strategy {
            VectorSearchStrategy::Exact => {
                self.late_interaction_search_within(queries, weights, allowed, limit)
            }
            VectorSearchStrategy::BinarySketchRerank => self
                .late_interaction_search_within_approximate(
                    queries,
                    weights,
                    allowed,
                    limit,
                    plan.candidate_vectors,
                ),
        }
    }

    pub(crate) fn late_interaction_search_approximate(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
        candidate_elements: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        if candidate_elements == 0 {
            return Err(Error::InvalidArgument(
                "late-interaction candidate budget must be greater than zero".into(),
            ));
        }
        let eligible_elements = self.eligible_element_upper_bound(target, label);
        if self.similarity != Similarity::Cosine
            || self.vector_data.base_float_count() == 0
            || candidate_elements >= eligible_elements
        {
            return self.late_interaction_search(queries, weights, target, limit, label);
        }
        let (queries, weights) =
            prepare_late_interaction_queries(queries, weights, self.dimension, self.similarity)?;
        let index = self
            .sketch_index
            .get_or_init(|| self.build_sketch_index().map_err(|error| error.to_string()))
            .as_ref()
            .map_err(|message| Error::Corrupt(message.clone()))?;
        let candidates = index.candidate_elements_multivector(
            &queries,
            &weights,
            target,
            label,
            None,
            candidate_elements.max(limit),
        );
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        let mut scored = HashSet::with_capacity(candidates.len());
        for candidate in candidates {
            if !self.sketch_entry_is_current(candidate) || !scored.insert(candidate.element) {
                continue;
            }
            let Some((vector_offset, vector_count)) = self.element_vector_span(candidate.element)
            else {
                continue;
            };
            self.score_late_interaction_element(
                &queries,
                &weights,
                candidate.element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }

        // As with single-vector ANN, the immutable checkpoint index is paired
        // with an exhaustive mutable delta so new and replaced elements are
        // immediately visible.
        let base_float_count = self.vector_data.base_float_count();
        if target.accepts(ElementRef::Node(0))
            && (!self.node_overlays.is_empty() || !self.nodes.is_empty())
        {
            for node in self.node_records() {
                let element = ElementRef::Node(node.id);
                if node.vector_offset < base_float_count
                    || label.is_some_and(|label| node.label != label)
                    || !scored.insert(element)
                {
                    continue;
                }
                self.score_late_interaction_element(
                    &queries,
                    &weights,
                    element,
                    node.vector_offset,
                    node.vector_count,
                    &scorer,
                    &mut top,
                )?;
            }
        }
        if target.accepts(ElementRef::Edge(0))
            && (!self.edge_overlays.is_empty() || !self.edges.is_empty())
        {
            for edge in self.edge_records() {
                let element = ElementRef::Edge(edge.id);
                if edge.vector_offset < base_float_count
                    || label.is_some_and(|label| edge.label != label)
                    || !scored.insert(element)
                {
                    continue;
                }
                self.score_late_interaction_element(
                    &queries,
                    &weights,
                    element,
                    edge.vector_offset,
                    edge.vector_count,
                    &scorer,
                    &mut top,
                )?;
            }
        }
        self.finish_late_interaction_hits(&queries, &weights, &scorer, top.finish())
    }

    pub(crate) fn late_interaction_search_adaptive(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<LateInteractionHit>> {
        let plan = self.vector_search_plan(target, label);
        match plan.strategy {
            VectorSearchStrategy::Exact => {
                self.late_interaction_search(queries, weights, target, limit, label)
            }
            VectorSearchStrategy::BinarySketchRerank => self.late_interaction_search_approximate(
                queries,
                weights,
                target,
                limit,
                label,
                plan.candidate_vectors,
            ),
        }
    }

    pub(crate) fn vector_search_approximate(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
        candidate_vectors: usize,
    ) -> Result<Vec<VectorHit>> {
        if candidate_vectors == 0 {
            return Err(Error::InvalidArgument(
                "approximate search candidate budget must be greater than zero".into(),
            ));
        }
        let base_float_count = self.vector_data.base_float_count();
        if self.similarity != Similarity::Cosine
            || base_float_count == 0
            || candidate_vectors >= self.eligible_element_upper_bound(target, label)
        {
            return self.vector_search(query, target, limit, label);
        }

        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let index = self
            .sketch_index
            .get_or_init(|| self.build_sketch_index().map_err(|error| error.to_string()))
            .as_ref()
            .map_err(|message| Error::Corrupt(message.clone()))?;
        if candidate_vectors >= index.element_count() {
            return self.vector_search(&query, target, limit, label);
        }
        let candidates =
            index.candidate_entries(&query, target, label, None, candidate_vectors.max(limit));
        self.rerank_approximate_candidates(
            &query,
            target,
            limit,
            label,
            base_float_count,
            candidates,
        )
    }

    fn rerank_approximate_candidates(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
        base_float_count: usize,
        candidates: Vec<SketchEntry>,
    ) -> Result<Vec<VectorHit>> {
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        let mut scored = HashSet::with_capacity(candidates.len());
        for candidate in candidates {
            let Some((current_label, vector_offset, vector_count)) =
                self.element_vector_fields(candidate.element)
            else {
                continue;
            };
            if current_label != candidate.label
                || candidate.float_offset < vector_offset
                || candidate.float_offset >= vector_offset + vector_count as usize * self.dimension
                || !scored.insert(candidate.element)
            {
                continue;
            }
            self.score_element(
                query,
                candidate.element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }

        // The checkpoint sketch is immutable. WAL vectors are deliberately
        // searched exhaustively, mirroring an LSM delta and guaranteeing that
        // fresh writes are immediately visible without rebuilding the base.
        if target.accepts(ElementRef::Node(0))
            && (!self.node_overlays.is_empty() || !self.nodes.is_empty())
        {
            for node in self.node_records() {
                if node.vector_offset < base_float_count
                    || label.is_some_and(|label| node.label != label)
                    || !scored.insert(ElementRef::Node(node.id))
                {
                    continue;
                }
                self.score_element(
                    query,
                    ElementRef::Node(node.id),
                    node.vector_offset,
                    node.vector_count,
                    &scorer,
                    &mut top,
                )?;
            }
        }
        if target.accepts(ElementRef::Edge(0))
            && (!self.edge_overlays.is_empty() || !self.edges.is_empty())
        {
            for edge in self.edge_records() {
                if edge.vector_offset < base_float_count
                    || label.is_some_and(|label| edge.label != label)
                    || !scored.insert(ElementRef::Edge(edge.id))
                {
                    continue;
                }
                self.score_element(
                    query,
                    ElementRef::Edge(edge.id),
                    edge.vector_offset,
                    edge.vector_count,
                    &scorer,
                    &mut top,
                )?;
            }
        }
        Ok(top.finish())
    }

    pub(crate) fn vector_search_plan(
        &self,
        target: VectorTarget,
        label: Option<LabelId>,
    ) -> VectorSearchPlan {
        let estimated_vectors = self.eligible_vector_count(target, label);
        let estimated_floats = estimated_vectors.saturating_mul(self.dimension);
        let target_covers_all_indexed = (self.indexed_node_vectors == 0
            || target.accepts(ElementRef::Node(0)))
            && (self.indexed_edge_vectors == 0 || target.accepts(ElementRef::Edge(0)));
        let high_fidelity_sketch = label.is_none()
            && target_covers_all_indexed
            && estimated_vectors == self.eligible_element_upper_bound(target, label);
        let candidate_vectors =
            adaptive_candidate_budget(estimated_vectors, self.dimension, high_fidelity_sketch);
        let strategy = if self.similarity == Similarity::Cosine
            && self.vector_data.base_float_count() != 0
            && candidate_vectors < estimated_vectors
        {
            VectorSearchStrategy::BinarySketchRerank
        } else {
            VectorSearchStrategy::Exact
        };
        VectorSearchPlan {
            strategy,
            estimated_vectors,
            estimated_floats,
            candidate_vectors: if strategy == VectorSearchStrategy::Exact {
                estimated_vectors
            } else {
                candidate_vectors
            },
        }
    }

    pub(crate) fn vector_search_adaptive(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<VectorHit>> {
        let plan = self.vector_search_plan(target, label);
        match plan.strategy {
            VectorSearchStrategy::Exact => self.vector_search(query, target, limit, label),
            VectorSearchStrategy::BinarySketchRerank => {
                self.vector_search_approximate(query, target, limit, label, plan.candidate_vectors)
            }
        }
    }

    fn eligible_vector_count(&self, target: VectorTarget, label: Option<LabelId>) -> usize {
        if label.is_none() {
            let mut count = 0usize;
            if target.accepts(ElementRef::Node(0)) {
                count = count.saturating_add(self.indexed_node_vectors);
            }
            if target.accepts(ElementRef::Edge(0)) {
                count = count.saturating_add(self.indexed_edge_vectors);
            }
            return count;
        }
        let mut count = 0usize;
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten().rev() {
                    if let Some(node) = self.node_record(id)
                        && node.label == label
                    {
                        count = count.saturating_add(node.vector_count as usize);
                    }
                }
            } else {
                count = count.saturating_add(
                    self.node_records()
                        .map(|node| node.vector_count as usize)
                        .sum(),
                );
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = label {
                for id in self.edges_by_label.get(&label).into_iter().flatten().rev() {
                    if let Some(edge) = self.edge_record(id)
                        && edge.label == label
                    {
                        count = count.saturating_add(edge.vector_count as usize);
                    }
                }
            } else {
                count = count.saturating_add(
                    self.edge_records()
                        .map(|edge| edge.vector_count as usize)
                        .sum(),
                );
            }
        }
        count
    }

    fn estimated_set_vector_count(&self, allowed: &ElementSet) -> usize {
        let node_average = if self.node_count == 0 {
            0
        } else {
            self.indexed_node_vectors.div_ceil(self.node_count)
        };
        let edge_average = if self.edge_count == 0 {
            0
        } else {
            self.indexed_edge_vectors.div_ceil(self.edge_count)
        };
        usize::try_from(allowed.node_len())
            .unwrap_or(usize::MAX)
            .saturating_mul(node_average)
            .saturating_add(
                usize::try_from(allowed.edge_len())
                    .unwrap_or(usize::MAX)
                    .saturating_mul(edge_average),
            )
    }

    fn eligible_element_upper_bound(&self, target: VectorTarget, label: Option<LabelId>) -> usize {
        let mut count = 0usize;
        if target.accepts(ElementRef::Node(0)) {
            count = count.saturating_add(label.map_or(self.node_count, |label| {
                self.nodes_by_label
                    .get(&label)
                    .map_or(0, |ids| usize::try_from(ids.len()).unwrap_or(usize::MAX))
            }));
        }
        if target.accepts(ElementRef::Edge(0)) {
            count = count.saturating_add(label.map_or(self.edge_count, |label| {
                self.edges_by_label
                    .get(&label)
                    .map_or(0, |ids| usize::try_from(ids.len()).unwrap_or(usize::MAX))
            }));
        }
        count
    }

    fn build_sketch_index(&self) -> Result<BinarySketchIndex> {
        let base_float_count = self.vector_data.base_float_count();
        let base_vectors = base_float_count / self.dimension;
        let mut index = BinarySketchIndex::new(self.dimension, base_vectors);
        let mut vector = vec![0.0; self.dimension];
        let mut workspace = Vec::new();
        for node in self.node_records() {
            for vector_index in 0..node.vector_count {
                let float_offset = node.vector_offset + vector_index as usize * self.dimension;
                if float_offset + self.dimension > base_float_count {
                    continue;
                }
                self.vector_data.copy_vector(float_offset, &mut vector)?;
                index.push(
                    SketchEntry {
                        element: ElementRef::Node(node.id),
                        label: node.label,
                        float_offset,
                    },
                    &vector,
                    &mut workspace,
                );
            }
        }
        for edge in self.edge_records() {
            for vector_index in 0..edge.vector_count {
                let float_offset = edge.vector_offset + vector_index as usize * self.dimension;
                if float_offset + self.dimension > base_float_count {
                    continue;
                }
                self.vector_data.copy_vector(float_offset, &mut vector)?;
                index.push(
                    SketchEntry {
                        element: ElementRef::Edge(edge.id),
                        label: edge.label,
                        float_offset,
                    },
                    &vector,
                    &mut workspace,
                );
            }
        }
        Ok(index)
    }

    fn sketch_entry_is_current(&self, entry: SketchEntry) -> bool {
        self.element_vector_fields(entry.element).is_some_and(
            |(label, vector_offset, vector_count)| {
                label == entry.label
                    && entry.float_offset >= vector_offset
                    && entry.float_offset < vector_offset + vector_count as usize * self.dimension
            },
        )
    }

    pub(crate) fn semantic_paths(
        &self,
        query: &[f32],
        options: &SemanticPathOptions,
    ) -> Result<Vec<SemanticPathHit>> {
        let seeds = self.vector_search_adaptive(
            query,
            VectorTarget::Nodes,
            options.seed_count,
            options.seed_label,
        )?;
        let starts: Vec<_> = seeds
            .into_iter()
            .filter_map(|hit| match hit.element {
                ElementRef::Node(id) => Some(id),
                ElementRef::Edge(_) => None,
            })
            .collect();
        self.semantic_expand(query, &starts, options)
    }

    pub(crate) fn semantic_expand(
        &self,
        query: &[f32],
        starts: &[NodeId],
        options: &SemanticPathOptions,
    ) -> Result<Vec<SemanticPathHit>> {
        validate_semantic_options(options)?;
        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(false)?;
        let mut frontier = BinaryHeap::new();
        let mut best_scores = HashMap::new();
        let mut results: HashMap<NodeId, SemanticPathHit> = HashMap::new();

        for &node in starts {
            let Some(node_record) = self.node_record(node) else {
                return Err(Error::NotFound("start node", node));
            };
            let Some((seed_score, _)) = self.element_score(
                &query,
                node_record.vector_offset,
                node_record.vector_count,
                &scorer,
            )?
            else {
                continue;
            };
            if best_scores
                .get(&node)
                .is_none_or(|score| seed_score > *score)
            {
                best_scores.insert(node, seed_score);
                frontier.push(PathState {
                    seed: node,
                    node,
                    score: seed_score,
                    seed_score,
                    path: Vec::new(),
                });
            }
        }

        let mut expansions = 0;
        while let Some(state) = frontier.pop() {
            if best_scores
                .get(&state.node)
                .is_some_and(|score| state.score < *score)
            {
                continue;
            }
            if options.include_seeds || !state.path.is_empty() {
                results.insert(
                    state.node,
                    SemanticPathHit {
                        seed: state.seed,
                        node: state.node,
                        score: state.score,
                        seed_score: state.seed_score,
                        path: state.path.clone(),
                    },
                );
            }
            if state.path.len() >= options.max_hops || expansions >= options.max_expansions {
                continue;
            }
            expansions += 1;
            for edge in self.neighbors(
                state.node,
                options.direction,
                EdgeFilter {
                    label: options.edge_label,
                },
            )? {
                let next = if edge.source == state.node {
                    edge.target
                } else {
                    edge.source
                };
                if state.path.contains(&edge.id) {
                    continue;
                }
                let Some(next_node) = self.node_record(next) else {
                    continue;
                };
                let Some((edge_score, _)) =
                    self.element_score(&query, edge.vector_offset, edge.vector_count, &scorer)?
                else {
                    continue;
                };
                let Some((node_score, _)) = self.element_score(
                    &query,
                    next_node.vector_offset,
                    next_node.vector_count,
                    &scorer,
                )?
                else {
                    continue;
                };
                let semantic_score = (options.node_weight * node_score
                    + options.edge_weight * edge_score)
                    / (options.node_weight + options.edge_weight);
                let degree_penalty = options.degree_penalty * (self.degree(next) as f32).ln_1p();
                let score = (options.path_decay * state.score
                    + (1.0 - options.path_decay) * semantic_score)
                    * options.hop_penalty
                    - degree_penalty;
                if best_scores.get(&next).is_some_and(|best| score <= *best) {
                    continue;
                }
                best_scores.insert(next, score);
                let mut path = state.path.clone();
                path.push(edge.id);
                frontier.push(PathState {
                    seed: state.seed,
                    node: next,
                    score,
                    seed_score: state.seed_score,
                    path,
                });
            }
        }

        let mut results: Vec<_> = results.into_values().collect();
        results.sort_unstable_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.node.cmp(&right.node))
        });
        results.truncate(options.limit);
        Ok(results)
    }

    fn score_element(
        &self,
        query: &[f32],
        element: ElementRef,
        vector_offset: usize,
        vector_count: u32,
        scorer: &VectorScorer<'_>,
        top: &mut TopK,
    ) -> Result<()> {
        let best = self.element_score(query, vector_offset, vector_count, scorer)?;
        if let Some((score, vector_index)) = best {
            top.push(VectorHit {
                element,
                score,
                vector_index,
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn score_late_interaction_element(
        &self,
        queries: &[Vec<f32>],
        weights: &[f32],
        element: ElementRef,
        vector_offset: usize,
        vector_count: u32,
        scorer: &VectorScorer<'_>,
        top: &mut TopK,
    ) -> Result<()> {
        if let Some(score) =
            self.late_interaction_score(queries, weights, vector_offset, vector_count, scorer)?
        {
            top.push(VectorHit {
                element,
                score,
                vector_index: 0,
            });
        }
        Ok(())
    }

    fn late_interaction_score(
        &self,
        queries: &[Vec<f32>],
        weights: &[f32],
        vector_offset: usize,
        vector_count: u32,
        scorer: &VectorScorer<'_>,
    ) -> Result<Option<f32>> {
        if vector_count == 0 {
            return Ok(None);
        }
        let mut score = 0.0;
        for (query, weight) in queries.iter().zip(weights) {
            let Some((best, _)) = self.element_score(query, vector_offset, vector_count, scorer)?
            else {
                return Ok(None);
            };
            score += best * weight;
        }
        Ok(Some(score))
    }

    fn finish_late_interaction_hits(
        &self,
        queries: &[Vec<f32>],
        weights: &[f32],
        scorer: &VectorScorer<'_>,
        hits: Vec<VectorHit>,
    ) -> Result<Vec<LateInteractionHit>> {
        let mut result = Vec::with_capacity(hits.len());
        for hit in hits {
            let Some((vector_offset, vector_count)) = self.element_vector_span(hit.element) else {
                continue;
            };
            let mut matched_vector_indices = Vec::with_capacity(queries.len());
            let mut score = 0.0;
            for (query, weight) in queries.iter().zip(weights) {
                let Some((best, vector_index)) =
                    self.element_score(query, vector_offset, vector_count, scorer)?
                else {
                    continue;
                };
                score += best * weight;
                matched_vector_indices.push(vector_index);
            }
            result.push(LateInteractionHit {
                element: hit.element,
                score,
                matched_vector_indices,
            });
        }
        Ok(result)
    }

    fn element_vector_span(&self, element: ElementRef) -> Option<(usize, u32)> {
        self.element_vector_fields(element)
            .map(|(_label, offset, count)| (offset, count))
    }

    #[inline]
    fn element_vector_fields(&self, element: ElementRef) -> Option<(LabelId, usize, u32)> {
        match element {
            ElementRef::Node(id) => {
                if let Some(record) = self.node_overlays.get(&id) {
                    return record
                        .map(|record| (record.label, record.vector_offset, record.vector_count));
                }
                if let Some(record) = self.nodes.get(id as usize).and_then(|record| *record) {
                    return Some((record.label, record.vector_offset, record.vector_count));
                }
                self.mapped_nodes.as_ref()?.vector_fields(id)
            }
            ElementRef::Edge(id) => {
                if let Some(record) = self.edge_overlays.get(&id) {
                    return record
                        .map(|record| (record.label, record.vector_offset, record.vector_count));
                }
                if let Some(record) = self.edges.get(id as usize).and_then(|record| *record) {
                    return Some((record.label, record.vector_offset, record.vector_count));
                }
                self.mapped_edges.as_ref()?.vector_fields(id)
            }
        }
    }

    fn element_score(
        &self,
        query: &[f32],
        vector_offset: usize,
        vector_count: u32,
        scorer: &VectorScorer<'_>,
    ) -> Result<Option<(f32, u32)>> {
        let mut best = None;
        for vector_index in 0..vector_count {
            let start = vector_offset + vector_index as usize * self.dimension;
            let score = scorer.score(query, start)?;
            if best.is_none_or(|(current, _)| score > current) {
                best = Some((score, vector_index));
            }
        }
        Ok(best)
    }

    fn degree(&self, node: NodeId) -> usize {
        let mut edges = HashSet::new();
        self.collect_incident_ids(node, &mut edges);
        edges.len()
    }

    pub(crate) fn element_vector(
        &self,
        offset: usize,
        vector_count: u32,
        index: usize,
    ) -> Result<Option<&[f32]>> {
        if index >= vector_count as usize {
            return Ok(None);
        }
        let Some(start) = index
            .checked_mul(self.dimension)
            .and_then(|index| offset.checked_add(index))
        else {
            return Ok(None);
        };
        self.vector_data.f32_range(start, self.dimension)
    }

    pub(crate) fn node_vector(&self, id: NodeId, index: usize) -> Result<Option<&[f32]>> {
        let Some(node) = self.node_record(id) else {
            return Ok(None);
        };
        self.element_vector(node.vector_offset, node.vector_count, index)
    }

    pub(crate) fn edge_vector(&self, id: EdgeId, index: usize) -> Result<Option<&[f32]>> {
        let Some(edge) = self.edge_record(id) else {
            return Ok(None);
        };
        self.element_vector(edge.vector_offset, edge.vector_count, index)
    }

    fn element_vector_owned(
        &self,
        offset: usize,
        vector_count: u32,
        index: usize,
    ) -> Result<Option<Vec<f32>>> {
        if index >= vector_count as usize {
            return Ok(None);
        }
        let Some(start) = index
            .checked_mul(self.dimension)
            .and_then(|index| offset.checked_add(index))
        else {
            return Ok(None);
        };
        let mut vector = vec![0.0; self.dimension];
        self.vector_data.copy_vector(start, &mut vector)?;
        Ok(Some(vector))
    }

    pub(crate) fn node_vector_owned(&self, id: NodeId, index: usize) -> Result<Option<Vec<f32>>> {
        let Some(node) = self.node_record(id) else {
            return Ok(None);
        };
        self.element_vector_owned(node.vector_offset, node.vector_count, index)
    }

    pub(crate) fn edge_vector_owned(&self, id: EdgeId, index: usize) -> Result<Option<Vec<f32>>> {
        let Some(edge) = self.edge_record(id) else {
            return Ok(None);
        };
        self.element_vector_owned(edge.vector_offset, edge.vector_count, index)
    }

    pub(crate) fn prepare(&self, mutations: &[Mutation]) -> Result<Vec<Operation>> {
        let mut operations = Vec::with_capacity(mutations.len() * 2);
        let mut pending_symbols: HashMap<Arc<str>, u32> = HashMap::new();
        let mut next_symbol = self.symbols.len() as u32;
        let mut touched_nodes = HashSet::new();
        let mut touched_edges = HashSet::new();
        let mut staged_nodes = HashSet::new();
        let mut deleted_edges = HashSet::new();

        for mutation in mutations {
            match mutation {
                Mutation::PutNode { id, .. } => {
                    if !touched_nodes.insert(*id) {
                        return Err(Error::Conflict(format!(
                            "node {id} is mutated more than once"
                        )));
                    }
                    staged_nodes.insert(*id);
                }
                Mutation::PutEdge { id, .. } | Mutation::DeleteEdge { id } => {
                    if !touched_edges.insert(*id) {
                        return Err(Error::Conflict(format!(
                            "edge {id} is mutated more than once"
                        )));
                    }
                    if matches!(mutation, Mutation::DeleteEdge { .. }) {
                        deleted_edges.insert(*id);
                    }
                }
                Mutation::DeleteNode { id, .. } => {
                    if !touched_nodes.insert(*id) {
                        return Err(Error::Conflict(format!(
                            "node {id} is mutated more than once"
                        )));
                    }
                }
            }
        }

        for mutation in mutations {
            match mutation {
                Mutation::PutNode {
                    id,
                    label,
                    properties,
                    vectors,
                    vector_count,
                } => {
                    validate_vectors(self.dimension, self.similarity, vectors, *vector_count)?;
                    let label = self.resolve_symbol(
                        label,
                        &mut pending_symbols,
                        &mut next_symbol,
                        &mut operations,
                    );
                    let properties = self.resolve_properties(
                        properties,
                        &mut pending_symbols,
                        &mut next_symbol,
                        &mut operations,
                    );
                    let generation = match self.node_record(*id) {
                        Some(existing) => {
                            existing.generation.get().checked_add(1).ok_or_else(|| {
                                Error::Conflict(format!("node {id} generation is exhausted"))
                            })?
                        }
                        None => 1,
                    };
                    operations.push(Operation::PutNode(Node {
                        id: *id,
                        label,
                        properties: properties.into(),
                        vector_count: *vector_count,
                        generation,
                        pending_vectors: vectors.clone().into(),
                    }));
                }
                Mutation::PutEdge {
                    id,
                    source,
                    target,
                    label,
                    properties,
                    vectors,
                    vector_count,
                } => {
                    for endpoint in [source, target] {
                        if !self.has_node(*endpoint) && !staged_nodes.contains(endpoint) {
                            return Err(Error::NotFound("endpoint node", *endpoint));
                        }
                    }
                    validate_vectors(self.dimension, self.similarity, vectors, *vector_count)?;
                    let label = self.resolve_symbol(
                        label,
                        &mut pending_symbols,
                        &mut next_symbol,
                        &mut operations,
                    );
                    let properties = self.resolve_properties(
                        properties,
                        &mut pending_symbols,
                        &mut next_symbol,
                        &mut operations,
                    );
                    let generation = match self.stored_edge(*id) {
                        Some(existing) => {
                            existing.generation.get().checked_add(1).ok_or_else(|| {
                                Error::Conflict(format!("edge {id} generation is exhausted"))
                            })?
                        }
                        None => 1,
                    };
                    operations.push(Operation::PutEdge(Edge {
                        id: *id,
                        source: *source,
                        target: *target,
                        label,
                        properties: properties.into(),
                        vector_count: *vector_count,
                        generation,
                        vector_offset: 0,
                        pending_vectors: vectors.clone().into(),
                    }));
                }
                Mutation::DeleteEdge { id } => {
                    if self.stored_edge(*id).is_none() {
                        return Err(Error::NotFound("edge", *id));
                    }
                    operations.push(Operation::DeleteEdge(*id));
                }
                Mutation::DeleteNode { id, detach } => {
                    if !self.has_node(*id) {
                        return Err(Error::NotFound("node", *id));
                    }
                    let incident = self.incident_edge_ids(*id);
                    if !*detach && incident.iter().any(|edge| !deleted_edges.contains(edge)) {
                        return Err(Error::Conflict(format!(
                            "node {id} still has incident edges; use detach deletion"
                        )));
                    }
                    if *detach {
                        for edge in incident {
                            if touched_edges.insert(edge) {
                                operations.push(Operation::DeleteEdge(edge));
                            }
                        }
                    }
                    operations.push(Operation::DeleteNode(*id));
                }
            }
        }
        // Persistence and replay must never depend on the order in which the
        // caller happened to stage mutations. Dictionary entries precede all
        // users, nodes precede their edges, and incident edges disappear
        // before their nodes.
        operations.sort_by_key(operation_order);
        Ok(operations)
    }

    fn incident_edge_ids(&self, node: NodeId) -> HashSet<EdgeId> {
        let mut edges = HashSet::new();
        self.collect_incident_ids(node, &mut edges);
        edges
    }

    fn collect_incident_ids(&self, node: NodeId, result: &mut HashSet<EdgeId>) {
        let outgoing_base = self
            .base_outgoing
            .as_ref()
            .map(|base| base.get(node))
            .unwrap_or_default();
        let incoming_base = self
            .base_incoming
            .as_ref()
            .map(|base| base.get(node))
            .unwrap_or_default();
        for &id in outgoing_base
            .iter()
            .chain(self.outgoing.get(node))
            .chain(incoming_base)
            .chain(self.incoming.get(node))
        {
            if self
                .stored_edge(id)
                .is_some_and(|edge| edge.source == node || edge.target == node)
            {
                result.insert(id);
            }
        }
    }

    fn resolve_symbol(
        &self,
        value: &Arc<str>,
        pending: &mut HashMap<Arc<str>, u32>,
        next: &mut u32,
        operations: &mut Vec<Operation>,
    ) -> u32 {
        if let Some(id) = self.symbol_ids.get(value).copied() {
            return id;
        }
        if let Some(id) = pending.get(value).copied() {
            return id;
        }
        let id = *next;
        *next += 1;
        pending.insert(value.clone(), id);
        operations.push(Operation::InternSymbol {
            id,
            value: value.clone(),
        });
        id
    }

    fn resolve_properties(
        &self,
        properties: &[(Arc<str>, Value)],
        pending: &mut HashMap<Arc<str>, u32>,
        next: &mut u32,
        operations: &mut Vec<Operation>,
    ) -> Vec<Property> {
        let mut result: Vec<_> = properties
            .iter()
            .map(|(key, value)| Property {
                key: self.resolve_symbol(key, pending, next, operations),
                value: value.clone(),
            })
            .collect();
        result.sort_unstable_by_key(|property| property.key);
        result.dedup_by_key(|property| property.key);
        result
    }

    pub(crate) fn apply(&mut self, operations: &[Operation]) -> Result<()> {
        for operation in operations {
            match operation {
                Operation::InternSymbol { id, value } => {
                    if *id as usize != self.symbols.len() {
                        return Err(Error::Corrupt(format!(
                            "symbol id {id} is not the next dictionary id {}",
                            self.symbols.len()
                        )));
                    }
                    self.symbols.push(value.clone());
                    self.symbol_ids.insert(value.clone(), *id);
                }
                Operation::PutNode(node) => self.apply_node(node.clone())?,
                Operation::PutEdge(edge) => self.apply_edge(edge.clone())?,
                Operation::DeleteNode(id) => {
                    if let Some(node) = self.node_record(*id) {
                        self.node_count -= 1;
                        self.indexed_vectors -= node.vector_count as usize;
                        self.indexed_node_vectors -= node.vector_count as usize;
                        if let Some(ids) = self.nodes_by_label.get_mut(&node.label) {
                            ids.remove(node.id);
                        }
                        if self.mapped_nodes.is_some() {
                            self.node_overlays.insert(*id, None);
                        } else if let Some(slot) = self.nodes.get_mut(*id as usize) {
                            *slot = None;
                        }
                    }
                }
                Operation::DeleteEdge(id) => {
                    if let Some(edge) = self.edge_record(*id) {
                        self.edge_count -= 1;
                        self.indexed_vectors -= edge.vector_count as usize;
                        self.indexed_edge_vectors -= edge.vector_count as usize;
                        if let Some(ids) = self.edges_by_label.get_mut(&edge.label) {
                            ids.remove(edge.id);
                        }
                        self.remove_from_adjacency(edge.source, edge.target, edge.id);
                        if self.mapped_edges.is_some() {
                            self.edge_overlays.insert(*id, None);
                        } else if let Some(slot) = self.edges.get_mut(*id as usize) {
                            *slot = None;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_node(&mut self, node: Node) -> Result<()> {
        validate_vectors(
            self.dimension,
            self.similarity,
            &node.pending_vectors,
            node.vector_count,
        )?;
        let index = node.id as usize;
        if self.mapped_nodes.is_none() {
            grow_slots(&mut self.nodes, index);
            self.outgoing.grow(index);
            self.incoming.grow(index);
        }
        if let Some(existing) = self.node_record(node.id) {
            self.indexed_vectors -= existing.vector_count as usize;
            self.indexed_node_vectors -= existing.vector_count as usize;
            if let Some(ids) = self.nodes_by_label.get_mut(&existing.label) {
                ids.remove(existing.id);
            }
        } else {
            self.node_count += 1;
        }
        let vector_offset = self.append_vectors(&node.pending_vectors)?;
        self.indexed_vectors += node.vector_count as usize;
        self.indexed_node_vectors += node.vector_count as usize;
        self.nodes_by_label
            .entry(node.label)
            .or_default()
            .insert(node.id);
        let properties = self.store_owned_properties(node.properties)?;
        let stored = StoredNode {
            id: node.id,
            label: node.label,
            properties,
            vector_count: node.vector_count,
            generation: nonzero_generation(node.generation)?,
            vector_offset,
        };
        if self.mapped_nodes.is_some() {
            self.node_overlays.insert(node.id, Some(stored));
        } else {
            self.nodes[index] = Some(stored);
        }
        Ok(())
    }

    fn apply_edge(&mut self, edge: Edge) -> Result<()> {
        validate_vectors(
            self.dimension,
            self.similarity,
            &edge.pending_vectors,
            edge.vector_count,
        )?;
        if !self.has_node(edge.source) || !self.has_node(edge.target) {
            return Err(Error::Corrupt(format!(
                "edge {} refers to a missing endpoint",
                edge.id
            )));
        }
        let index = edge.id as usize;
        if self.mapped_edges.is_none() {
            grow_slots(&mut self.edges, index);
        }
        if let Some(existing) = self.edge_record(edge.id) {
            self.indexed_vectors -= existing.vector_count as usize;
            self.indexed_edge_vectors -= existing.vector_count as usize;
            if let Some(ids) = self.edges_by_label.get_mut(&existing.label) {
                ids.remove(existing.id);
            }
            let (source, target, id) = (existing.source, existing.target, existing.id);
            self.remove_from_adjacency(source, target, id);
        } else {
            self.edge_count += 1;
        }
        let vector_offset = self.append_vectors(&edge.pending_vectors)?;
        self.indexed_vectors += edge.vector_count as usize;
        self.indexed_edge_vectors += edge.vector_count as usize;
        self.edges_by_label
            .entry(edge.label)
            .or_default()
            .insert(edge.id);
        self.outgoing.push(edge.source, edge.id);
        self.incoming.push(edge.target, edge.id);
        let properties = self.store_owned_properties(edge.properties)?;
        let stored = StoredEdge {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.label,
            properties,
            vector_count: edge.vector_count,
            generation: nonzero_generation(edge.generation)?,
            vector_offset,
        };
        if self.mapped_edges.is_some() {
            self.edge_overlays.insert(edge.id, Some(stored));
        } else {
            self.edges[index] = Some(stored);
        }
        Ok(())
    }

    fn remove_from_adjacency(&mut self, source: NodeId, target: NodeId, edge: EdgeId) {
        self.outgoing.remove(source, edge);
        self.incoming.remove(target, edge);
    }

    fn store_owned_properties(&mut self, properties: Arc<[Property]>) -> Result<StoredProperties> {
        let reference = StoredProperties::owned(self.owned_properties.len())?;
        self.owned_properties.push(properties);
        Ok(reference)
    }

    fn append_vectors(&mut self, vectors: &[f32]) -> Result<usize> {
        let dimension = self.dimension;
        let similarity = self.similarity;
        let (vector_data, offset) = match &mut self.vector_data {
            VectorData::Owned(vector_data) => {
                let offset = vector_data.len();
                (vector_data, offset)
            }
            VectorData::Mapped {
                byte_len,
                encoding,
                delta,
                ..
            } => {
                let base_float_count = *byte_len
                    / match encoding {
                        VectorEncoding::F32 => 4,
                        VectorEncoding::F16 => 2,
                    };
                let offset = base_float_count + delta.len();
                (delta, offset)
            }
        };
        if similarity == Similarity::Dot {
            vector_data.extend_from_slice(vectors);
            return Ok(offset);
        }
        vector_data.reserve(vectors.len());
        for vector in vectors.chunks_exact(dimension) {
            let start = vector_data.len();
            vector_data.extend_from_slice(vector);
            vector::normalize(&mut vector_data[start..])?;
        }
        Ok(offset)
    }
}

fn adaptive_candidate_budget(
    estimated_vectors: usize,
    dimension: usize,
    high_fidelity_sketch: bool,
) -> usize {
    adaptive_candidate_budget_with_exact_floats(
        estimated_vectors,
        dimension,
        64 * 1024 * 1024,
        high_fidelity_sketch,
    )
}

fn adaptive_candidate_budget_with_exact_floats(
    estimated_vectors: usize,
    dimension: usize,
    exact_float_budget: usize,
    high_fidelity_sketch: bool,
) -> usize {
    if estimated_vectors.saturating_mul(dimension) <= exact_float_budget {
        estimated_vectors
    } else if high_fidelity_sketch {
        // Query-confidence weighting reached higher VIBE recall with 12k
        // candidates than equal-weight Hamming did with 20k. Keep the smaller
        // floor exclusive to the exact one-vector execution shape measured by
        // that benchmark; multivector and filtered gathers use other policies.
        (estimated_vectors / 100).clamp(12_000, 50_000)
    } else {
        (estimated_vectors / 50).clamp(20_000, 50_000)
    }
}

fn adaptive_candidate_budget_for_set(estimated_vectors: usize, dimension: usize) -> usize {
    let exact_float_budget = if dimension >= 512 {
        12 * 1024 * 1024
    } else {
        32 * 1024 * 1024
    };
    if estimated_vectors.saturating_mul(dimension) <= exact_float_budget {
        estimated_vectors
    } else {
        (estimated_vectors / 4)
            .clamp(5_000, 20_000)
            .min(estimated_vectors)
    }
}

fn prepare_late_interaction_queries(
    queries: &[Vec<f32>],
    weights: Option<&[f32]>,
    dimension: usize,
    similarity: Similarity,
) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
    if queries.is_empty() {
        return Err(Error::InvalidArgument(
            "late interaction requires at least one query vector".into(),
        ));
    }
    let mut prepared = Vec::with_capacity(queries.len());
    for query in queries {
        prepared.push(vector::prepare_query(query, dimension, similarity)?);
    }
    let mut normalized_weights = match weights {
        Some(weights) if weights.len() != queries.len() => {
            return Err(Error::InvalidArgument(format!(
                "{} late-interaction weights do not match {} query vectors",
                weights.len(),
                queries.len()
            )));
        }
        Some(weights) => weights.to_vec(),
        None => vec![1.0; queries.len()],
    };
    if normalized_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(Error::InvalidArgument(
            "late-interaction weights must be finite and non-negative".into(),
        ));
    }
    let total: f32 = normalized_weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::InvalidArgument(
            "at least one late-interaction weight must be positive".into(),
        ));
    }
    for weight in &mut normalized_weights {
        *weight /= total;
    }
    Ok((prepared, normalized_weights))
}

fn nonzero_generation(generation: u64) -> Result<NonZeroU64> {
    NonZeroU64::new(generation)
        .ok_or_else(|| Error::Corrupt("element generation must be non-zero".into()))
}

fn stored_node_matches(
    node: &StoredNode,
    filter: &NodeFilter,
    map: Option<&Mmap>,
    owned_properties: &[Arc<[Property]>],
) -> bool {
    stored_element_matches(node.label, node.properties, filter, map, owned_properties)
}

fn stored_element_matches(
    label: LabelId,
    stored_properties: StoredProperties,
    filter: &ElementFilter,
    map: Option<&Mmap>,
    owned_properties: &[Arc<[Property]>],
) -> bool {
    if filter.label.is_some_and(|expected| label != expected) {
        return false;
    }
    if filter.properties.is_empty() {
        return true;
    }
    stored_properties.matches(&filter.properties, map, owned_properties)
}

#[derive(Clone, Copy)]
struct PreparedNumericRange {
    tag: u8,
    lower: Bound<u64>,
    upper: Bound<u64>,
}

fn prepare_numeric_range(filter: &NumericRangeFilter) -> Result<PreparedNumericRange> {
    fn prepare_bound(bound: Bound<NumericValue>) -> Result<(Bound<u64>, Option<u8>)> {
        let wrap = match bound {
            Bound::Included(value) => Bound::Included(value),
            Bound::Excluded(value) => Bound::Excluded(value),
            Bound::Unbounded => return Ok((Bound::Unbounded, None)),
        };
        let (value, included) = match wrap {
            Bound::Included(value) => (value, true),
            Bound::Excluded(value) => (value, false),
            Bound::Unbounded => unreachable!(),
        };
        let value = match value {
            NumericValue::Int(value) => Value::Int(value),
            NumericValue::Float(value) => Value::Float(value),
        };
        let (tag, sortable) = crate::codec::numeric_value_index_key(&value).ok_or_else(|| {
            Error::InvalidArgument("numeric range bounds may not contain NaN".into())
        })?;
        Ok((
            if included {
                Bound::Included(sortable)
            } else {
                Bound::Excluded(sortable)
            },
            Some(tag),
        ))
    }

    let (lower, lower_tag) = prepare_bound(filter.lower)?;
    let (upper, upper_tag) = prepare_bound(filter.upper)?;
    let tag = match (lower_tag, upper_tag) {
        (Some(left), Some(right)) if left != right => {
            return Err(Error::InvalidArgument(
                "numeric range bounds must have the same numeric type".into(),
            ));
        }
        (Some(tag), _) | (_, Some(tag)) => tag,
        (None, None) => {
            return Err(Error::InvalidArgument(
                "a numeric range needs at least one typed bound".into(),
            ));
        }
    };
    Ok(PreparedNumericRange { tag, lower, upper })
}

fn numeric_key_in_range(key: (u8, u64), range: PreparedNumericRange) -> bool {
    if key.0 != range.tag {
        return false;
    }
    let lower = match range.lower {
        Bound::Included(value) => key.1 >= value,
        Bound::Excluded(value) => key.1 > value,
        Bound::Unbounded => true,
    };
    let upper = match range.upper {
        Bound::Included(value) => key.1 <= value,
        Bound::Excluded(value) => key.1 < value,
        Bound::Unbounded => true,
    };
    lower && upper
}

fn stored_element_matches_numeric_range(
    label: LabelId,
    stored_properties: StoredProperties,
    filter: &NumericRangeFilter,
    range: PreparedNumericRange,
    map: Option<&Mmap>,
    owned_properties: &[Arc<[Property]>],
) -> bool {
    if filter.label.is_some_and(|expected| label != expected) {
        return false;
    }
    stored_properties
        .numeric_key(filter.key, map, owned_properties)
        .is_some_and(|key| numeric_key_in_range(key, range))
}

fn validate_vectors(
    dimension: usize,
    similarity: Similarity,
    vectors: &[f32],
    vector_count: u32,
) -> Result<()> {
    if vectors.len() != dimension * vector_count as usize {
        return Err(Error::InvalidArgument(format!(
            "{} floats do not encode {vector_count} vectors of dimension {dimension}",
            vectors.len()
        )));
    }
    for vector in vectors.chunks_exact(dimension) {
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(
                "vectors may only contain finite values".into(),
            ));
        }
        if similarity == Similarity::Cosine
            && vector.iter().map(|value| value * value).sum::<f32>() <= f32::EPSILON
        {
            return Err(Error::InvalidArgument(
                "cosine vectors must be non-zero".into(),
            ));
        }
    }
    Ok(())
}

fn reconstruct_shortest_path(
    start: NodeId,
    end: NodeId,
    parents: &HashMap<NodeId, (NodeId, EdgeId)>,
) -> ShortestPath {
    let mut nodes = vec![end];
    let mut edges = Vec::new();
    let mut cursor = end;
    while cursor != start {
        let &(parent, edge) = parents
            .get(&cursor)
            .expect("every discovered non-root node has one parent");
        edges.push(edge);
        nodes.push(parent);
        cursor = parent;
    }
    nodes.reverse();
    edges.reverse();
    ShortestPath { nodes, edges }
}

const fn reverse_direction(direction: Direction) -> Direction {
    match direction {
        Direction::Outgoing => Direction::Incoming,
        Direction::Incoming => Direction::Outgoing,
        Direction::Both => Direction::Both,
    }
}

fn visited_union_len(
    forward_depths: &HashMap<NodeId, usize>,
    reverse_depths: &HashMap<NodeId, usize>,
) -> usize {
    forward_depths.len() + reverse_depths.len()
        - forward_depths
            .keys()
            .filter(|node| reverse_depths.contains_key(node))
            .count()
}

fn best_bidirectional_path(
    start: NodeId,
    end: NodeId,
    length: usize,
    forward_depths: &HashMap<NodeId, usize>,
    reverse_depths: &HashMap<NodeId, usize>,
    forward_parents: &HashMap<NodeId, (NodeId, EdgeId)>,
    reverse_next: &HashMap<NodeId, (NodeId, EdgeId)>,
) -> Option<ShortestPath> {
    forward_depths
        .iter()
        .filter_map(|(&meeting, &forward_depth)| {
            let reverse_depth = *reverse_depths.get(&meeting)?;
            (forward_depth + reverse_depth == length).then(|| {
                reconstruct_bidirectional_path(start, end, meeting, forward_parents, reverse_next)
            })
        })
        .min_by(|left, right| {
            left.nodes
                .cmp(&right.nodes)
                .then_with(|| left.edges.cmp(&right.edges))
        })
}

fn reconstruct_bidirectional_path(
    start: NodeId,
    end: NodeId,
    meeting: NodeId,
    forward_parents: &HashMap<NodeId, (NodeId, EdgeId)>,
    reverse_next: &HashMap<NodeId, (NodeId, EdgeId)>,
) -> ShortestPath {
    let mut path = reconstruct_shortest_path(start, meeting, forward_parents);
    let mut cursor = meeting;
    while cursor != end {
        let &(next, edge) = reverse_next
            .get(&cursor)
            .expect("every reverse-discovered non-root node has one next step");
        path.edges.push(edge);
        path.nodes.push(next);
        cursor = next;
    }
    path
}

fn validate_semantic_options(options: &SemanticPathOptions) -> Result<()> {
    if options.seed_count == 0 {
        return Err(Error::InvalidArgument(
            "semantic path seed_count must be greater than zero".into(),
        ));
    }
    if !options.node_weight.is_finite()
        || !options.edge_weight.is_finite()
        || options.node_weight < 0.0
        || options.edge_weight < 0.0
        || options.node_weight + options.edge_weight <= f32::EPSILON
    {
        return Err(Error::InvalidArgument(
            "semantic path node and edge weights must be finite, non-negative, and not both zero"
                .into(),
        ));
    }
    if !options.path_decay.is_finite() || !(0.0..=1.0).contains(&options.path_decay) {
        return Err(Error::InvalidArgument(
            "semantic path decay must be between zero and one".into(),
        ));
    }
    if !options.hop_penalty.is_finite() || !(0.0..=1.0).contains(&options.hop_penalty) {
        return Err(Error::InvalidArgument(
            "semantic path hop penalty must be between zero and one".into(),
        ));
    }
    if !options.degree_penalty.is_finite() || options.degree_penalty < 0.0 {
        return Err(Error::InvalidArgument(
            "semantic path degree penalty must be finite and non-negative".into(),
        ));
    }
    Ok(())
}

fn validate_semantic_one_hop(query: &SemanticOneHopQuery) -> Result<()> {
    if query.seed_count == 0 {
        return Err(Error::InvalidArgument(
            "semantic pattern seed_count must be greater than zero".into(),
        ));
    }
    if !query.start_weight.is_finite()
        || !query.edge_weight.is_finite()
        || !query.end_weight.is_finite()
        || query.start_weight <= 0.0
        || query.edge_weight < 0.0
        || query.end_weight < 0.0
    {
        return Err(Error::InvalidArgument(
            "semantic pattern weights must be finite and non-negative, with a positive start weight"
                .into(),
        ));
    }
    Ok(())
}

fn grow_slots<T>(slots: &mut Vec<Option<T>>, index: usize) {
    if slots.len() <= index {
        slots.resize_with(index + 1, || None);
    }
}

fn grow_adjacency(adjacency: &mut Vec<Vec<EdgeId>>, index: usize) {
    if adjacency.len() <= index {
        adjacency.resize_with(index + 1, Vec::new);
    }
}

fn operation_order(operation: &Operation) -> u8 {
    match operation {
        Operation::InternSymbol { .. } => 0,
        Operation::PutNode(_) => 1,
        Operation::PutEdge(_) => 2,
        Operation::DeleteEdge(_) => 3,
        Operation::DeleteNode(_) => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        StoredEdge, StoredNode, StoredProperties, adaptive_candidate_budget,
        adaptive_candidate_budget_for_set,
    };

    #[test]
    fn stored_record_layout_stays_compact() {
        let properties = std::mem::size_of::<StoredProperties>();
        let node = std::mem::size_of::<StoredNode>();
        let edge = std::mem::size_of::<StoredEdge>();
        let node_slot = std::mem::size_of::<Option<StoredNode>>();
        let edge_slot = std::mem::size_of::<Option<StoredEdge>>();
        eprintln!(
            "stored layout: properties={properties}, node={node}/{node_slot}, edge={edge}/{edge_slot}"
        );
        assert!(properties <= 24);
        assert!(node <= 56);
        assert!(edge <= 72);
        assert!(node_slot <= 56);
        assert!(edge_slot <= 72);
    }

    #[test]
    fn adaptive_vector_budget_preserves_small_exact_scans_and_caps_broad_ones() {
        assert_eq!(adaptive_candidate_budget(131_072, 512, false), 131_072);
        assert_eq!(adaptive_candidate_budget(131_073, 512, false), 20_000);
        assert_eq!(adaptive_candidate_budget(262_144, 256, false), 262_144);
        assert_eq!(adaptive_candidate_budget(507_949, 256, false), 20_000);
        assert_eq!(adaptive_candidate_budget(2_000_000, 256, false), 40_000);
        assert_eq!(adaptive_candidate_budget(1_000_000, 4, false), 1_000_000);
        assert_eq!(adaptive_candidate_budget(1_000_000, 200, true), 12_000);
        assert_eq!(adaptive_candidate_budget(2_000_000, 200, true), 20_000);
        assert_eq!(adaptive_candidate_budget_for_set(10_712, 768), 10_712);
        assert_eq!(adaptive_candidate_budget_for_set(20_588, 768), 5_147);
        assert_eq!(adaptive_candidate_budget_for_set(100_000, 200), 100_000);
        assert_eq!(adaptive_candidate_budget_for_set(200_000, 200), 20_000);
    }
}
