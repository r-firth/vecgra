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

mod filters;
mod search;
mod state;
mod traversal;
mod types;

use types::PathState;
pub use types::{
    Direction, EdgeFilter, ElementFilter, ElementFilterPlan, ElementFilterStrategy,
    FilteredVectorSearchResult, GraphRangeSearchOptions, GraphRangeSearchResult, GraphStats,
    NodeFilter, NumericRangeFilter, NumericRangePlan, NumericRangeStrategy, NumericValue,
    OneHopPlan, OneHopQuery, OneHopStrategy, PatternMatch, SemanticOneHopQuery, SemanticPathHit,
    SemanticPathOptions, SemanticPatternMatch, ShortestPath, ShortestPathOptions,
    ShortestPathResult, ShortestPathStrategy, ShortestPathTermination,
};

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
