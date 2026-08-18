use crate::error::{Error, Result};
use crate::model::ElementRef;
use crate::simd;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Similarity function used by every vector in a database.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Similarity {
    /// Cosine similarity over normalized stored and query vectors.
    Cosine,
    /// Raw inner-product similarity.
    Dot,
}

/// Physical encoding used for checkpoint vector storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorEncoding {
    /// Native 32-bit floating-point components.
    F32,
    /// IEEE 754 binary16 components decoded while scoring.
    F16,
}

/// Kinds of graph elements eligible for vector search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorTarget {
    /// Search node vectors only.
    Nodes,
    /// Search relationship vectors only.
    Edges,
    /// Search both node and relationship vectors.
    Both,
}

/// Physical strategy selected for a vector query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorSearchStrategy {
    /// Score every eligible vector exactly.
    Exact,
    /// Select candidates with persisted binary sketches, then rerank exactly.
    BinarySketchRerank,
}

/// Inspectable cost and candidate decision for a vector query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorSearchPlan {
    /// Selected physical search strategy.
    pub strategy: VectorSearchStrategy,
    /// Number of vectors eligible before applying an approximate budget.
    pub estimated_vectors: usize,
    /// Approximate number of stored scalar components an exact scan scores.
    pub estimated_floats: usize,
    /// Maximum number of vectors sent to exact reranking.
    pub candidate_vectors: usize,
}

/// One vector facet returned by nearest-neighbor search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorHit {
    /// Node or relationship that owns the matching vector.
    pub element: ElementRef,
    /// Similarity score; larger values rank first.
    pub score: f32,
    /// Zero-based vector-facet index on the element.
    pub vector_index: u32,
}

/// A whole-element result from weighted late interaction. Each query vector
/// independently selects its best matching vector facet on the element; the
/// final score is their weighted mean.
#[derive(Clone, Debug, PartialEq)]
pub struct LateInteractionHit {
    /// Node or relationship ranked as a whole element.
    pub element: ElementRef,
    /// Weighted mean of the best score for each query facet.
    pub score: f32,
    /// One best matching element-vector index for every query vector, in query
    /// order. This makes multivector and multimodal matches inspectable.
    pub matched_vector_indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RankedHit(VectorHit);

impl Eq for RankedHit {}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .score
            .total_cmp(&other.0.score)
            .then_with(|| self.0.element.cmp(&other.0.element))
            .then_with(|| self.0.vector_index.cmp(&other.0.vector_index))
    }
}

pub(crate) struct TopK {
    limit: usize,
    heap: BinaryHeap<Reverse<RankedHit>>,
}

impl TopK {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit.saturating_add(1)),
        }
    }

    pub(crate) fn push(&mut self, hit: VectorHit) {
        if self.limit == 0 {
            return;
        }
        let hit = RankedHit(hit);
        if self.heap.len() < self.limit {
            self.heap.push(Reverse(hit));
        } else if self.heap.peek().is_some_and(|minimum| hit > minimum.0) {
            self.heap.pop();
            self.heap.push(Reverse(hit));
        }
    }

    pub(crate) fn finish(self) -> Vec<VectorHit> {
        let mut hits: Vec<_> = self.heap.into_iter().map(|item| item.0.0).collect();
        hits.sort_unstable_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.element.cmp(&right.element))
        });
        hits
    }
}

impl VectorTarget {
    #[inline]
    pub(crate) fn accepts(self, element: ElementRef) -> bool {
        matches!(
            (self, element),
            (Self::Nodes, ElementRef::Node(_))
                | (Self::Edges, ElementRef::Edge(_))
                | (Self::Both, _)
        )
    }
}

pub(crate) fn prepare_query(
    query: &[f32],
    dimension: usize,
    similarity: Similarity,
) -> Result<Vec<f32>> {
    if query.len() != dimension {
        return Err(Error::InvalidArgument(format!(
            "query dimension {} does not match database dimension {dimension}",
            query.len()
        )));
    }
    let mut query = query.to_vec();
    if similarity == Similarity::Cosine {
        normalize(&mut query)?;
    } else if query.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidArgument(
            "query vectors may only contain finite values".into(),
        ));
    }
    Ok(query)
}

#[inline]
pub(crate) fn score(left: &[f32], right: &[f32]) -> f32 {
    simd::dot(left, right)
}

pub(crate) fn normalize(vector: &mut [f32]) -> Result<()> {
    let magnitude = simd::dot(vector, vector).sqrt();
    if !magnitude.is_finite() || magnitude <= f32::EPSILON {
        return Err(Error::InvalidArgument(
            "cosine vectors must be finite and non-zero".into(),
        ));
    }
    let inverse = magnitude.recip();
    for value in vector {
        *value *= inverse;
    }
    Ok(())
}
