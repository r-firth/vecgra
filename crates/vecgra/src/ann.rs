use crate::model::{ElementRef, ElementSet, LabelId};
use crate::vector::VectorTarget;
use memmap2::Mmap;
use std::marker::PhantomData;
use std::sync::Arc;

const MAX_SIGNATURE_BITS: usize = 512;
const ROTATION_SEED_1: u64 = 0x243f_6a88_85a3_08d3;
const ROTATION_SEED_2: u64 = 0x1319_8a2e_0370_7344;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SketchEntry {
    pub(crate) element: ElementRef,
    pub(crate) label: LabelId,
    pub(crate) float_offset: usize,
}

/// A cache-friendly angular coarse index. Every vector receives a compact
/// sign sketch after two deterministic randomized Hadamard rotations. Search
/// scans only the sketches, then the graph kernel exactly reranks a bounded
/// candidate set from its mmap vector column.
#[derive(Debug)]
pub(crate) struct BinarySketchIndex {
    dimension: usize,
    transform_dimension: usize,
    words_per_signature: usize,
    owners: IndexColumn<u64>,
    owner_kinds: IndexColumn<u8>,
    labels: IndexColumn<LabelId>,
    element_count: usize,
    owners_ordered: bool,
    last_owner_by_kind: [Option<u64>; 2],
    owner_kind_mask: u8,
    signatures: SketchSignatures,
}

#[derive(Debug)]
enum IndexColumn<T> {
    Owned(Vec<T>),
    Mapped {
        map: Arc<Mmap>,
        byte_offset: usize,
        count: usize,
        marker: PhantomData<T>,
    },
}

impl<T: Copy> IndexColumn<T> {
    fn mapped(map: Arc<Mmap>, byte_offset: usize, count: usize) -> Self {
        Self::Mapped {
            map,
            byte_offset,
            count,
            marker: PhantomData,
        }
    }

    #[inline]
    fn as_slice(&self) -> &[T] {
        match self {
            Self::Owned(values) => values,
            Self::Mapped {
                map,
                byte_offset,
                count,
                ..
            } => {
                let pointer = map[*byte_offset..].as_ptr();
                debug_assert_eq!(pointer.align_offset(std::mem::align_of::<T>()), 0);
                // SAFETY: the checkpoint decoder validates alignment and the
                // complete range for these fixed-width columns.
                unsafe { std::slice::from_raw_parts(pointer.cast::<T>(), *count) }
            }
        }
    }
}

#[derive(Debug)]
enum SketchSignatures {
    Owned(Vec<u64>),
    Mapped {
        map: Arc<Mmap>,
        byte_offset: usize,
        word_count: usize,
    },
}

impl SketchSignatures {
    fn as_slice(&self) -> &[u64] {
        match self {
            Self::Owned(words) => words,
            Self::Mapped {
                map,
                byte_offset,
                word_count,
            } => {
                let pointer = map[*byte_offset..].as_ptr();
                debug_assert_eq!(pointer.align_offset(std::mem::align_of::<u64>()), 0);
                // SAFETY: the indexed checkpoint parser validates the aligned
                // complete range, and the read-only map outlives this view.
                unsafe { std::slice::from_raw_parts(pointer.cast::<u64>(), *word_count) }
            }
        }
    }
}

impl BinarySketchIndex {
    pub(crate) fn new(dimension: usize, capacity: usize) -> Self {
        let transform_dimension = writer_transform_dimension(dimension);
        let bit_count = transform_dimension.min(MAX_SIGNATURE_BITS);
        let words_per_signature = bit_count.div_ceil(64);
        Self {
            dimension,
            transform_dimension,
            words_per_signature,
            owners: IndexColumn::Owned(Vec::with_capacity(capacity)),
            owner_kinds: IndexColumn::Owned(Vec::with_capacity(capacity)),
            labels: IndexColumn::Owned(Vec::with_capacity(capacity)),
            element_count: 0,
            owners_ordered: true,
            last_owner_by_kind: [None, None],
            owner_kind_mask: 0,
            signatures: SketchSignatures::Owned(Vec::with_capacity(
                capacity.saturating_mul(words_per_signature),
            )),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "constructor arguments mirror independent persisted sketch columns"
    )]
    pub(crate) fn mapped(
        dimension: usize,
        owners: Vec<u64>,
        owner_kinds: Vec<u8>,
        labels: Vec<LabelId>,
        map: Arc<Mmap>,
        byte_offset: usize,
        words_per_signature: usize,
        word_count: usize,
    ) -> Self {
        // Persisted word count disambiguates the widened transform for
        // sub-512-dimensional files. Older 256-bit files retain their original
        // transform, while dimensions whose natural transform was already 512
        // continue to decode correctly even when only 256 bits were stored.
        let natural_transform = dimension.next_power_of_two();
        let transform_dimension = if dimension >= 128 {
            natural_transform.max(words_per_signature.saturating_mul(64))
        } else {
            // A signature word is the minimum storage unit, so dimensions
            // below 64 have padding bits rather than a 64-wide transform.
            natural_transform
        };
        debug_assert_eq!(owners.len(), owner_kinds.len());
        debug_assert_eq!(owners.len(), labels.len());
        debug_assert_eq!(word_count, owners.len() * words_per_signature);
        let element_count = owners
            .iter()
            .enumerate()
            .filter(|(index, owner)| {
                *index == 0
                    || owners[*index - 1] != **owner
                    || owner_kinds[*index - 1] != owner_kinds[*index]
            })
            .count();
        let (owners_ordered, last_owner_by_kind) = owner_order_state(&owners, &owner_kinds);
        let owner_kind_mask = owner_kinds
            .iter()
            .fold(0u8, |mask, kind| mask | (1u8 << *kind));
        #[cfg(target_endian = "little")]
        let signatures = SketchSignatures::Mapped {
            map,
            byte_offset,
            word_count,
        };
        #[cfg(target_endian = "big")]
        let signatures = {
            let byte_len = word_count * 8;
            SketchSignatures::Owned(
                map[byte_offset..byte_offset + byte_len]
                    .chunks_exact(8)
                    .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                    .collect(),
            )
        };
        Self {
            dimension,
            transform_dimension,
            words_per_signature,
            owners: IndexColumn::Owned(owners),
            owner_kinds: IndexColumn::Owned(owner_kinds),
            labels: IndexColumn::Owned(labels),
            element_count,
            owners_ordered,
            last_owner_by_kind,
            owner_kind_mask,
            signatures,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "constructor arguments mirror independent memory-mapped sketch columns"
    )]
    #[cfg(target_endian = "little")]
    pub(crate) fn mapped_columns(
        dimension: usize,
        map: Arc<Mmap>,
        owner_byte_offset: usize,
        owner_kind_byte_offset: usize,
        label_byte_offset: usize,
        signature_byte_offset: usize,
        entry_count: usize,
        words_per_signature: usize,
        word_count: usize,
    ) -> Self {
        let natural_transform = dimension.next_power_of_two();
        let transform_dimension = if dimension >= 128 {
            natural_transform.max(words_per_signature.saturating_mul(64))
        } else {
            natural_transform
        };
        let owners = IndexColumn::mapped(map.clone(), owner_byte_offset, entry_count);
        let owner_kinds = IndexColumn::mapped(map.clone(), owner_kind_byte_offset, entry_count);
        let labels = IndexColumn::mapped(map.clone(), label_byte_offset, entry_count);
        let owner_slice = owners.as_slice();
        let kind_slice = owner_kinds.as_slice();
        let element_count = owner_slice
            .iter()
            .enumerate()
            .filter(|(index, owner)| {
                *index == 0
                    || owner_slice[*index - 1] != **owner
                    || kind_slice[*index - 1] != kind_slice[*index]
            })
            .count();
        let (owners_ordered, last_owner_by_kind) = owner_order_state(owner_slice, kind_slice);
        let owner_kind_mask = kind_slice
            .iter()
            .fold(0u8, |mask, kind| mask | (1u8 << *kind));
        Self {
            dimension,
            transform_dimension,
            words_per_signature,
            owners,
            owner_kinds,
            labels,
            element_count,
            owners_ordered,
            last_owner_by_kind,
            owner_kind_mask,
            signatures: SketchSignatures::Mapped {
                map,
                byte_offset: signature_byte_offset,
                word_count,
            },
        }
    }

    pub(crate) fn push(&mut self, entry: SketchEntry, vector: &[f32], workspace: &mut Vec<f32>) {
        debug_assert_eq!(vector.len(), self.dimension);
        let (owner, owner_kind) = element_parts(entry.element);
        self.owner_kind_mask |= 1u8 << owner_kind;
        let kind_index = owner_kind as usize;
        self.owners_ordered &= self.last_owner_by_kind[kind_index]
            .is_none_or(|previous_owner| previous_owner <= owner);
        self.last_owner_by_kind[kind_index] = Some(owner);
        let SketchSignatures::Owned(signatures) = &mut self.signatures else {
            unreachable!("cannot append to a mapped sketch index")
        };
        append_signature(
            vector,
            self.transform_dimension,
            self.words_per_signature,
            workspace,
            signatures,
        );
        if self
            .owners
            .as_slice()
            .last()
            .is_none_or(|previous| *previous != owner)
            || self
                .owner_kinds
                .as_slice()
                .last()
                .is_none_or(|kind| *kind != owner_kind)
        {
            self.element_count += 1;
        }
        let IndexColumn::Owned(owners) = &mut self.owners else {
            unreachable!("cannot append to mapped sketch owners")
        };
        let IndexColumn::Owned(owner_kinds) = &mut self.owner_kinds else {
            unreachable!("cannot append to mapped sketch owner kinds")
        };
        let IndexColumn::Owned(labels) = &mut self.labels else {
            unreachable!("cannot append to mapped sketch labels")
        };
        owners.push(owner);
        owner_kinds.push(owner_kind);
        labels.push(entry.label);
    }

    pub(crate) fn element_count(&self) -> usize {
        self.element_count
    }

    pub(crate) fn candidate_entries(
        &self,
        query: &[f32],
        target: VectorTarget,
        label: Option<LabelId>,
        allowed: Option<&ElementSet>,
        budget: usize,
    ) -> Vec<SketchEntry> {
        let owners = self.owners.as_slice();
        let owner_kinds = self.owner_kinds.as_slice();
        let labels = self.labels.as_slice();
        if budget == 0 || owners.is_empty() {
            return Vec::new();
        }
        let mut workspace = Vec::new();
        let mut query_signature = Vec::with_capacity(self.words_per_signature);
        append_signature(
            query,
            self.transform_dimension,
            self.words_per_signature,
            &mut workspace,
            &mut query_signature,
        );

        // Rank whole graph elements, retaining the best sketch distance among
        // all their vector facets. A richly represented node therefore spends
        // one candidate slot rather than crowding the budget with its facets.
        let mut nearest: Vec<(u16, u32)> = Vec::new();
        let signatures = self.signatures.as_slice();
        let target_covers_all_owners = (self.owner_kind_mask & 1 == 0
            || target.accepts(ElementRef::Node(0)))
            && (self.owner_kind_mask & 2 == 0 || target.accepts(ElementRef::Edge(0)));
        if allowed.is_none()
            && label.is_none()
            && target_covers_all_owners
            && self.element_count == owners.len()
        {
            // Common one-vector-per-element corpora need only the sequential
            // signature stream during coarse ranking. Distances are bounded
            // by 1024, so a tiny histogram finds the exact budget cutoff. This
            // stores two bytes per row and avoids partially sorting an
            // eight-byte `(distance, ordinal)` tuple for every vector.
            let confidence_masks = query_confidence_masks(&workspace, self.words_per_signature);
            let maximum_distance = self.words_per_signature * 64 * 2;
            let mut histogram = vec![0usize; maximum_distance + 1];
            let mut distances = Vec::with_capacity(owners.len());
            for index in 0..owners.len() {
                let signature_start = index * self.words_per_signature;
                let distance = confidence_hamming_distance(
                    &signatures[signature_start..signature_start + self.words_per_signature],
                    &query_signature,
                    &confidence_masks,
                );
                histogram[distance as usize] += 1;
                distances.push(distance);
            }
            return histogram_candidate_ordinals(&distances, &histogram, budget)
                .into_iter()
                .map(|index| self.entry_at(index as usize))
                .collect();
        }
        // Both checkpoint owners and Roaring iterators are ordered. Merge them
        // once instead of performing a tree/container membership lookup for
        // every indexed element during a filtered sketch scan.
        let mut allowed_nodes = allowed.map(ElementSet::node_ids);
        let mut allowed_edges = allowed.map(ElementSet::edge_ids);
        let mut next_allowed_node = allowed_nodes.as_mut().and_then(Iterator::next);
        let mut next_allowed_edge = allowed_edges.as_mut().and_then(Iterator::next);
        let mut start = 0usize;
        while start < owners.len() {
            let owner = owners[start];
            let owner_kind = owner_kinds[start];
            let mut end = start + 1;
            while end < owners.len() && owners[end] == owner && owner_kinds[end] == owner_kind {
                end += 1;
            }
            let element = element_from_parts(owner, owner_kind);
            let element_allowed = if allowed.is_none() {
                true
            } else if !self.owners_ordered {
                allowed.is_some_and(|allowed| allowed.contains(element))
            } else if owner_kind == 0 {
                while next_allowed_node.is_some_and(|allowed| allowed < owner) {
                    next_allowed_node = allowed_nodes.as_mut().and_then(Iterator::next);
                }
                next_allowed_node == Some(owner)
            } else {
                while next_allowed_edge.is_some_and(|allowed| allowed < owner) {
                    next_allowed_edge = allowed_edges.as_mut().and_then(Iterator::next);
                }
                next_allowed_edge == Some(owner)
            };
            if target.accepts(element)
                && label.is_none_or(|label| labels[start] == label)
                && element_allowed
            {
                let mut minimum = u16::MAX;
                for index in start..end {
                    let signature_start = index * self.words_per_signature;
                    let distance = hamming_distance(
                        &signatures[signature_start..signature_start + self.words_per_signature],
                        &query_signature,
                    );
                    minimum = minimum.min(distance);
                }
                nearest.push((minimum, start as u32));
            }
            start = end;
        }
        histogram_ranked_ordinals(&nearest, self.words_per_signature * 64, budget)
            .into_iter()
            .map(|index| self.entry_at(index as usize))
            .collect()
    }

    /// Coarse late-interaction search over whole graph elements. For each
    /// query facet, the best (smallest) Hamming distance among an element's
    /// stored vector facets is retained, then those minima are summed. This is
    /// the binary-sketch analogue of weighted MaxSim and avoids allowing a
    /// multivector element to consume the candidate budget once per facet.
    pub(crate) fn candidate_elements_multivector(
        &self,
        queries: &[Vec<f32>],
        weights: &[f32],
        target: VectorTarget,
        label: Option<LabelId>,
        allowed: Option<&ElementSet>,
        budget: usize,
    ) -> Vec<SketchEntry> {
        let owners = self.owners.as_slice();
        let owner_kinds = self.owner_kinds.as_slice();
        let labels = self.labels.as_slice();
        if budget == 0 || owners.is_empty() || queries.is_empty() {
            return Vec::new();
        }
        debug_assert_eq!(queries.len(), weights.len());

        let mut workspace = Vec::new();
        let mut query_signatures =
            Vec::with_capacity(queries.len().saturating_mul(self.words_per_signature));
        for query in queries {
            append_signature(
                query,
                self.transform_dimension,
                self.words_per_signature,
                &mut workspace,
                &mut query_signatures,
            );
        }

        // Checkpoint vectors belonging to one element are contiguous. Reduce
        // every such run to one coarse score and one representative entry.
        // The weighted integer score preserves deterministic ordering without
        // allocating a query-sized accumulator for every database element.
        let signatures = self.signatures.as_slice();
        let mut nearest: Vec<(u64, u32)> = Vec::new();
        let mut allowed_nodes = allowed.map(ElementSet::node_ids);
        let mut allowed_edges = allowed.map(ElementSet::edge_ids);
        let mut next_allowed_node = allowed_nodes.as_mut().and_then(Iterator::next);
        let mut next_allowed_edge = allowed_edges.as_mut().and_then(Iterator::next);
        let mut start = 0usize;
        let mut minima = vec![u16::MAX; queries.len()];
        while start < owners.len() {
            let owner = owners[start];
            let owner_kind = owner_kinds[start];
            let mut end = start + 1;
            while end < owners.len() && owners[end] == owner && owner_kinds[end] == owner_kind {
                end += 1;
            }
            let element = element_from_parts(owner, owner_kind);
            let element_allowed = if allowed.is_none() {
                true
            } else if !self.owners_ordered {
                allowed.is_some_and(|allowed| allowed.contains(element))
            } else if owner_kind == 0 {
                while next_allowed_node.is_some_and(|allowed| allowed < owner) {
                    next_allowed_node = allowed_nodes.as_mut().and_then(Iterator::next);
                }
                next_allowed_node == Some(owner)
            } else {
                while next_allowed_edge.is_some_and(|allowed| allowed < owner) {
                    next_allowed_edge = allowed_edges.as_mut().and_then(Iterator::next);
                }
                next_allowed_edge == Some(owner)
            };
            if target.accepts(element)
                && label.is_none_or(|label| labels[start] == label)
                && element_allowed
            {
                minima.fill(u16::MAX);
                for vector_index in start..end {
                    let signature_start = vector_index * self.words_per_signature;
                    let signature =
                        &signatures[signature_start..signature_start + self.words_per_signature];
                    for (query_index, minimum) in minima.iter_mut().enumerate() {
                        let query_start = query_index * self.words_per_signature;
                        let distance = hamming_distance(
                            signature,
                            &query_signatures[query_start..query_start + self.words_per_signature],
                        );
                        *minimum = (*minimum).min(distance);
                    }
                }
                // Fixed-point weights keep the sortable key compact. The
                // exact scorer applies the original f32 weights during rerank.
                let distance = minima
                    .iter()
                    .zip(weights)
                    .map(|(distance, weight)| {
                        (*distance as f64 * *weight as f64 * 65_536.0).round() as u64
                    })
                    .sum();
                nearest.push((distance, start as u32));
            }
            start = end;
        }
        if nearest.len() > budget {
            nearest.select_nth_unstable(budget);
            nearest.truncate(budget);
        }
        nearest.sort_unstable();
        nearest
            .into_iter()
            .map(|(_, index)| self.entry_at(index as usize))
            .collect()
    }

    fn entry_at(&self, ordinal: usize) -> SketchEntry {
        let owners = self.owners.as_slice();
        let owner_kinds = self.owner_kinds.as_slice();
        let labels = self.labels.as_slice();
        SketchEntry {
            element: element_from_parts(owners[ordinal], owner_kinds[ordinal]),
            label: labels[ordinal],
            float_offset: ordinal * self.dimension,
        }
    }
}

#[inline(always)]
fn hamming_distance(left: &[u64], right: &[u64]) -> u16 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "aarch64")]
    {
        // AArch64 guarantees NEON. Four unaligned 128-bit loads cover the
        // common 512-bit signature and byte-wise CNT computes 16 popcounts per
        // instruction. The portable tail also handles older 256-bit and small
        // dimension formats.
        // SAFETY: both pointers address complete equally sized slices; the
        // helper uses unaligned loads and stops before any partial pair.
        unsafe { hamming_distance_neon(left, right) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left ^ right).count_ones())
            .sum::<u32>() as u16
    }
}

fn query_confidence_masks(query_projection: &[f32], words_per_signature: usize) -> Vec<u64> {
    let bit_count = (words_per_signature * 64).min(query_projection.len());
    let mean = query_projection
        .iter()
        .take(bit_count)
        .map(|value| value.abs())
        .sum::<f32>()
        / bit_count.max(1) as f32;
    let mut masks = vec![0u64; words_per_signature];
    for (index, value) in query_projection.iter().take(bit_count).enumerate() {
        if value.abs() >= mean {
            masks[index / 64] |= 1u64 << (index % 64);
        }
    }
    masks
}

#[inline(always)]
fn confidence_hamming_distance(left: &[u64], right: &[u64], confidence: &[u64]) -> u16 {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), confidence.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: all slices have equal complete lengths; the helper uses
        // unaligned loads and handles any trailing word portably.
        unsafe { confidence_hamming_distance_neon(left, right, confidence) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        left.iter()
            .zip(right)
            .zip(confidence)
            .map(|((&left, &right), &confidence)| {
                let mismatch = left ^ right;
                mismatch.count_ones() + (mismatch & confidence).count_ones()
            })
            .sum::<u32>() as u16
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn confidence_hamming_distance_neon(left: &[u64], right: &[u64], confidence: &[u64]) -> u16 {
    use std::arch::aarch64::{vaddlvq_u8, vandq_u8, vcntq_u8, veorq_u8, vld1q_u8};

    let pair_count = left.len() / 2;
    let mut total = 0u16;
    for pair in 0..pair_count {
        let byte_offset = pair * 16;
        // SAFETY: `pair < len / 2` guarantees 16 readable bytes from every
        // slice, and vld1q_u8 permits unaligned pointers.
        let left_words = unsafe { vld1q_u8(left.as_ptr().cast::<u8>().add(byte_offset)) };
        // SAFETY: same bound and alignment argument as `left_words`.
        let right_words = unsafe { vld1q_u8(right.as_ptr().cast::<u8>().add(byte_offset)) };
        // SAFETY: same bound and alignment argument as `left_words`.
        let confidence_words =
            unsafe { vld1q_u8(confidence.as_ptr().cast::<u8>().add(byte_offset)) };
        let mismatch = veorq_u8(left_words, right_words);
        total += vaddlvq_u8(vcntq_u8(mismatch));
        total += vaddlvq_u8(vcntq_u8(vandq_u8(mismatch, confidence_words)));
    }
    for index in pair_count * 2..left.len() {
        let mismatch = left[index] ^ right[index];
        total += mismatch.count_ones() as u16;
        total += (mismatch & confidence[index]).count_ones() as u16;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hamming_distance_neon(left: &[u64], right: &[u64]) -> u16 {
    use std::arch::aarch64::{vaddlvq_u8, vcntq_u8, veorq_u8, vld1q_u8};

    let pair_count = left.len() / 2;
    let mut total = 0u16;
    for pair in 0..pair_count {
        let byte_offset = pair * 16;
        // SAFETY: `pair < len / 2` guarantees 16 readable bytes from each
        // slice, and vld1q_u8 permits unaligned pointers.
        let left_words = unsafe { vld1q_u8(left.as_ptr().cast::<u8>().add(byte_offset)) };
        // SAFETY: same bound and alignment argument as `left_words`.
        let right_words = unsafe { vld1q_u8(right.as_ptr().cast::<u8>().add(byte_offset)) };
        total += vaddlvq_u8(vcntq_u8(veorq_u8(left_words, right_words)));
    }
    for index in pair_count * 2..left.len() {
        total += (left[index] ^ right[index]).count_ones() as u16;
    }
    total
}

fn histogram_candidate_ordinals(distances: &[u16], histogram: &[usize], budget: usize) -> Vec<u32> {
    let wanted = budget.min(distances.len());
    if wanted == 0 {
        return Vec::new();
    }
    let mut below_cutoff = 0usize;
    let mut cutoff = 0u16;
    for (distance, &count) in histogram.iter().enumerate() {
        if below_cutoff.saturating_add(count) >= wanted {
            cutoff = distance as u16;
            break;
        }
        below_cutoff += count;
    }

    let mut selected = Vec::with_capacity(wanted);
    for (ordinal, &distance) in distances.iter().enumerate() {
        if distance < cutoff {
            selected.push(ordinal as u32);
        }
    }
    for (ordinal, &distance) in distances.iter().enumerate() {
        if selected.len() == wanted {
            break;
        }
        if distance == cutoff {
            selected.push(ordinal as u32);
        }
    }
    debug_assert_eq!(selected.len(), wanted);
    selected
}

fn histogram_ranked_ordinals(
    ranked: &[(u16, u32)],
    maximum_distance: usize,
    budget: usize,
) -> Vec<u32> {
    let wanted = budget.min(ranked.len());
    if wanted == 0 {
        return Vec::new();
    }
    let mut histogram = vec![0usize; maximum_distance + 1];
    for &(distance, _) in ranked {
        histogram[distance as usize] += 1;
    }
    let mut below_cutoff = 0usize;
    let mut cutoff = 0u16;
    for (distance, &count) in histogram.iter().enumerate() {
        if below_cutoff.saturating_add(count) >= wanted {
            cutoff = distance as u16;
            break;
        }
        below_cutoff += count;
    }

    let mut selected = Vec::with_capacity(wanted);
    selected.extend(
        ranked
            .iter()
            .filter(|(distance, _)| *distance < cutoff)
            .map(|(_, ordinal)| *ordinal),
    );
    selected.extend(
        ranked
            .iter()
            .filter(|(distance, _)| *distance == cutoff)
            .take(wanted - selected.len())
            .map(|(_, ordinal)| *ordinal),
    );
    debug_assert_eq!(selected.len(), wanted);
    selected
}

fn element_parts(element: ElementRef) -> (u64, u8) {
    match element {
        ElementRef::Node(id) => (id, 0),
        ElementRef::Edge(id) => (id, 1),
    }
}

fn element_from_parts(owner: u64, kind: u8) -> ElementRef {
    if kind == 0 {
        ElementRef::Node(owner)
    } else {
        ElementRef::Edge(owner)
    }
}

fn owner_order_state(owners: &[u64], kinds: &[u8]) -> (bool, [Option<u64>; 2]) {
    let mut ordered = true;
    let mut last = [None, None];
    for (&owner, &kind) in owners.iter().zip(kinds) {
        let Some(slot) = last.get_mut(kind as usize) else {
            return (false, last);
        };
        ordered &= slot.is_none_or(|previous| previous <= owner);
        *slot = Some(owner);
    }
    (ordered, last)
}

pub(crate) fn signature_word_count(dimension: usize) -> usize {
    writer_transform_dimension(dimension)
        .min(MAX_SIGNATURE_BITS)
        .div_ceil(64)
}

pub(crate) fn append_vector_signature(
    vector: &[f32],
    workspace: &mut Vec<f32>,
    output: &mut Vec<u64>,
) {
    append_signature(
        vector,
        writer_transform_dimension(vector.len()),
        signature_word_count(vector.len()),
        workspace,
        output,
    );
}

fn writer_transform_dimension(dimension: usize) -> usize {
    let natural = dimension.next_power_of_two();
    if dimension >= 128 {
        natural.max(MAX_SIGNATURE_BITS)
    } else {
        natural
    }
}

fn append_signature(
    vector: &[f32],
    transform_dimension: usize,
    words_per_signature: usize,
    workspace: &mut Vec<f32>,
    output: &mut Vec<u64>,
) {
    workspace.clear();
    workspace.resize(transform_dimension, 0.0);
    for (index, value) in vector.iter().enumerate() {
        workspace[index] = value * random_sign(index, ROTATION_SEED_1);
    }
    hadamard(workspace);
    for (index, value) in workspace.iter_mut().enumerate() {
        *value *= random_sign(index, ROTATION_SEED_2);
    }
    hadamard(workspace);

    for word in 0..words_per_signature {
        let mut bits = 0u64;
        for bit in 0..64 {
            let index = word * 64 + bit;
            if index >= workspace.len() {
                break;
            }
            if workspace[index] >= 0.0 {
                bits |= 1u64 << bit;
            }
        }
        output.push(bits);
    }
}

fn hadamard(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut width = 1;
    while width < values.len() {
        for block in (0..values.len()).step_by(width * 2) {
            for lane in 0..width {
                let left = values[block + lane];
                let right = values[block + width + lane];
                values[block + lane] = left + right;
                values[block + width + lane] = left - right;
            }
        }
        width *= 2;
    }
}

#[inline]
fn random_sign(index: usize, seed: u64) -> f32 {
    let mut value = (index as u64).wrapping_add(seed);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    if value & 1 == 0 { 1.0 } else { -1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_kernel_matches_scalar_for_full_pairs_and_tail() {
        let left = [0, u64::MAX, 0x0123_4567_89ab_cdef, 17, 42];
        let right = [u64::MAX, 0, 0xfedc_ba98_7654_3210, 19, 7];
        let expected = left
            .iter()
            .zip(right)
            .map(|(left, right)| (*left ^ right).count_ones())
            .sum::<u32>() as u16;
        assert_eq!(hamming_distance(&left, &right), expected);

        let confidence = [u64::MAX, 0, 0x0f0f_0f0f_0f0f_0f0f, 1, u64::MAX];
        let confidence_expected = left
            .iter()
            .zip(right)
            .zip(confidence)
            .map(|((left, right), confidence)| {
                let mismatch = *left ^ right;
                mismatch.count_ones() + (mismatch & confidence).count_ones()
            })
            .sum::<u32>() as u16;
        assert_eq!(
            confidence_hamming_distance(&left, &right, &confidence),
            confidence_expected
        );
    }

    fn entry(id: u64) -> SketchEntry {
        SketchEntry {
            element: ElementRef::Node(id),
            label: 1,
            float_offset: id as usize * 4,
        }
    }

    #[test]
    fn identical_vectors_have_identical_signatures() {
        let vector = [0.1, -0.7, 0.2, 0.6];
        let mut index = BinarySketchIndex::new(4, 2);
        let mut workspace = Vec::new();
        index.push(entry(0), &vector, &mut workspace);
        index.push(entry(1), &[-0.1, 0.7, -0.2, -0.6], &mut workspace);
        let hits = index.candidate_entries(&vector, VectorTarget::Nodes, None, None, 1);
        assert_eq!(hits[0].element, ElementRef::Node(0));
    }

    #[test]
    fn query_confidence_marks_above_mean_projection_bits() {
        assert_eq!(query_confidence_masks(&[1.0, -3.0, 0.0, 0.0], 1), vec![3]);
    }

    #[test]
    fn label_and_target_are_prefiltered_before_budgeting() {
        let mut index = BinarySketchIndex::new(4, 2);
        let mut workspace = Vec::new();
        index.push(entry(0), &[1.0, 0.0, 0.0, 0.0], &mut workspace);
        let mut other = entry(1);
        other.label = 2;
        index.push(other, &[1.0, 0.0, 0.0, 0.0], &mut workspace);
        let hits =
            index.candidate_entries(&[1.0, 0.0, 0.0, 0.0], VectorTarget::Nodes, Some(2), None, 1);
        assert_eq!(hits[0].element, ElementRef::Node(1));
    }

    #[test]
    fn graph_candidate_set_is_applied_before_budgeting() {
        let mut index = BinarySketchIndex::new(4, 2);
        let mut workspace = Vec::new();
        index.push(entry(0), &[1.0, 0.0, 0.0, 0.0], &mut workspace);
        index.push(entry(1), &[0.0, 1.0, 0.0, 0.0], &mut workspace);
        let mut allowed = ElementSet::new();
        allowed.insert(ElementRef::Node(1));
        let hits = index.candidate_entries(
            &[1.0, 0.0, 0.0, 0.0],
            VectorTarget::Nodes,
            None,
            Some(&allowed),
            1,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].element, ElementRef::Node(1));
    }

    #[test]
    fn multivector_candidates_are_ranked_as_elements() {
        let mut index = BinarySketchIndex::new(4, 3);
        let mut workspace = Vec::new();
        index.push(entry(0), &[1.0, 0.0, 0.0, 0.0], &mut workspace);
        index.push(entry(0), &[0.0, 1.0, 0.0, 0.0], &mut workspace);
        index.push(entry(1), &[1.0, 0.0, 0.0, 0.0], &mut workspace);
        let hits = index.candidate_elements_multivector(
            &[vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            &[1.0, 1.0],
            VectorTarget::Nodes,
            None,
            None,
            1,
        );
        assert_eq!(hits[0].element, ElementRef::Node(0));
    }

    #[test]
    fn medium_dimensions_receive_a_wider_projection() {
        assert_eq!(signature_word_count(127), 2);
        assert_eq!(signature_word_count(128), 8);
        assert_eq!(signature_word_count(200), 8);
        assert_eq!(signature_word_count(512), 8);
        assert_eq!(signature_word_count(1024), 8);
    }

    #[test]
    fn histogram_selection_honors_budget_and_stable_cutoff_ties() {
        let distances = [4, 1, 3, 1, 2, 1];
        let mut histogram = vec![0; 5];
        for distance in distances {
            histogram[distance as usize] += 1;
        }
        assert_eq!(
            histogram_candidate_ordinals(&distances, &histogram, 4),
            vec![1, 3, 5, 4]
        );
        assert_eq!(
            histogram_candidate_ordinals(&distances, &histogram, 2),
            vec![1, 3]
        );
        assert!(histogram_candidate_ordinals(&distances, &histogram, 0).is_empty());

        let ranked = [(4, 9), (1, 7), (3, 6), (1, 5), (2, 4), (1, 3)];
        assert_eq!(histogram_ranked_ordinals(&ranked, 4, 4), vec![7, 5, 3, 4]);
    }
}
