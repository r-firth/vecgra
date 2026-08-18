use super::*;

/// Direction in which an operation follows directed relationships.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Follow source-to-target relationships.
    Outgoing,
    /// Follow target-to-source relationships.
    Incoming,
    /// Follow relationships in either direction.
    Both,
}

/// Optional relationship predicate used during traversal.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeFilter {
    /// Required relationship label, or `None` to accept every label.
    pub label: Option<LabelId>,
}

/// Bounds and traversal semantics for an exact unweighted shortest path.
///
/// `max_expansions` counts frontier nodes whose adjacency is expanded. This is
/// deliberately separate from `max_hops`: a shallow search over a broad graph
/// can still have a finite application-controlled work budget.
#[derive(Clone, Copy, Debug)]
pub struct ShortestPathOptions {
    /// Maximum number of relationships in the returned path.
    pub max_hops: usize,
    /// Maximum number of frontier nodes whose adjacency may be expanded.
    pub max_expansions: usize,
    /// Directions in which relationships may be traversed.
    pub direction: Direction,
    /// Relationship-label constraint applied during traversal.
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
    /// Ordered nodes from start to destination, inclusive.
    pub nodes: Vec<NodeId>,
    /// Ordered relationships connecting adjacent entries in `nodes`.
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

/// Reason an exact shortest-path traversal stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortestPathTermination {
    /// A path was found within both bounds.
    Found,
    /// Exhaustive traversal found no path within `max_hops`.
    NotFoundWithinHops,
    /// Traversal stopped before a conclusive result at `max_expansions`.
    ExpansionLimit,
}

/// Inspectable result of bounded exact shortest-path search.
///
/// A missing `path` is conclusive only when `termination` is
/// `NotFoundWithinHops`; `ExpansionLimit` reports a deliberately incomplete
/// search rather than silently looking like graph absence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShortestPathResult {
    /// Found path, or `None` when termination was unsuccessful or incomplete.
    pub path: Option<ShortestPath>,
    /// Physical traversal used for the query.
    pub strategy: ShortestPathStrategy,
    /// Why traversal stopped.
    pub termination: ShortestPathTermination,
    /// Distinct nodes discovered across both frontiers.
    pub visited_nodes: usize,
    /// Nodes expanded from the requested start endpoint.
    pub start_expanded_nodes: usize,
    /// Nodes expanded from the requested end endpoint. This is zero for the
    /// one-sided breadth-first strategy.
    pub end_expanded_nodes: usize,
    /// Total nodes expanded across both endpoints. This always equals
    /// `start_expanded_nodes + end_expanded_nodes`.
    pub expanded_nodes: usize,
    /// Relationships examined while expanding adjacency.
    pub examined_relationships: usize,
}

/// Current logical size and transaction count of a graph snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    /// Number of live nodes.
    pub nodes: usize,
    /// Number of live relationships.
    pub edges: usize,
    /// Number of interned labels and property keys.
    pub labels: usize,
    /// Number of vector facets across nodes and relationships.
    pub indexed_vectors: usize,
    /// Number of applied transactions, including a checkpoint base.
    pub transactions: u64,
}

/// Bounds and scoring weights for semantic path expansion.
#[derive(Clone, Debug)]
pub struct SemanticPathOptions {
    /// Number of vector-search seeds considered before traversal.
    pub seed_count: usize,
    /// Maximum number of ranked path hits returned.
    pub limit: usize,
    /// Maximum relationship count in a returned path.
    pub max_hops: usize,
    /// Maximum number of path states expanded.
    pub max_expansions: usize,
    /// Directions in which relationships may be traversed.
    pub direction: Direction,
    /// Optional label required for semantic seed nodes.
    pub seed_label: Option<LabelId>,
    /// Optional label required for traversed relationships.
    pub edge_label: Option<LabelId>,
    /// Relative contribution of destination-node similarity.
    pub node_weight: f32,
    /// Relative contribution of relationship similarity.
    pub edge_weight: f32,
    /// Multiplicative score decay applied at every hop.
    pub path_decay: f32,
    /// Multiplicative penalty applied for each additional hop.
    pub hop_penalty: f32,
    /// Penalty applied to high-degree expansion points.
    pub degree_penalty: f32,
    /// Whether zero-hop seed results may be returned.
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

/// One node reached through a semantically ranked relationship path.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticPathHit {
    /// Vector-search seed from which traversal began.
    pub seed: NodeId,
    /// Reached destination node.
    pub node: NodeId,
    /// Combined semantic and structural path score.
    pub score: f32,
    /// Original seed-node similarity score.
    pub seed_score: f32,
    /// Ordered relationships from `seed` to `node`.
    pub path: Vec<EdgeId>,
}

/// Result of a nearest-neighbor query constrained to a bounded graph range.
///
/// The range is evaluated before the vector candidate budget, so `hits` never
/// relies on post-filtering a global ANN result. `plan` describes the exact or
/// sketch/rerank decision made after the reachable-node cardinality is known.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphRangeSearchResult {
    /// Nearest eligible node-vector facets.
    pub hits: Vec<VectorHit>,
    /// Number of nodes surviving graph and scalar predicates.
    pub candidate_nodes: u64,
    /// Vector access plan selected after range evaluation.
    pub plan: VectorSearchPlan,
}

/// Result of a vector query whose ordinary property and numeric-range
/// predicates were evaluated before vector candidate selection. The access
/// plans remain visible so applications can diagnose selectivity/cost without
/// parsing an engine-specific explain string.
#[derive(Clone, Debug, PartialEq)]
pub struct FilteredVectorSearchResult {
    /// Nearest eligible vector facets.
    pub hits: Vec<VectorHit>,
    /// Number of graph elements surviving all scalar predicates.
    pub candidate_elements: u64,
    /// Equality access plan, when an equality predicate was supplied.
    pub equality_plan: Option<ElementFilterPlan>,
    /// Access plan for each numeric range in request order.
    pub numeric_range_plans: Vec<NumericRangePlan>,
    /// Vector access plan selected after scalar evaluation.
    pub vector_plan: VectorSearchPlan,
}

/// Reachability and filtering options for graph-range vector search.
#[derive(Clone, Debug)]
pub struct GraphRangeSearchOptions {
    /// Maximum traversal depth from the seed set.
    pub max_hops: usize,
    /// Maximum number of nearest-neighbor hits returned.
    pub limit: usize,
    /// Directions in which relationships may be traversed.
    pub direction: Direction,
    /// Relationship predicate applied during range expansion.
    pub edge_filter: EdgeFilter,
    /// Whether seed nodes remain eligible search candidates.
    pub include_seeds: bool,
    /// Optional predicate applied to reachable nodes.
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

/// Conjunction of an optional label and exact property predicates.
#[derive(Clone, Debug, Default)]
pub struct ElementFilter {
    /// Required node or relationship label.
    pub label: Option<LabelId>,
    /// Exact property predicates, all of which must match.
    pub properties: Vec<(PropertyKeyId, Value)>,
}

/// Physical access path selected for an equality filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementFilterStrategy {
    /// Inspect every target record.
    FullScan,
    /// Start from a persisted label posting.
    LabelPosting,
    /// Start from a persisted exact-property posting.
    PropertyPosting,
}

/// Inspectable access decision for an equality filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementFilterPlan {
    /// Selected physical access strategy.
    pub strategy: ElementFilterStrategy,
    /// Conservative number of records the selected access path may inspect.
    pub candidate_upper_bound: usize,
    /// Predicate ordinal in `ElementFilter::properties` when a property
    /// posting is selected.
    pub property_predicate: Option<usize>,
}

/// Numeric scalar accepted by an ordered range predicate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumericValue {
    /// Signed 64-bit integer bound.
    Int(i64),
    /// 64-bit floating-point bound.
    Float(f64),
}

/// Same-typed inclusive or exclusive numeric property range.
#[derive(Clone, Debug, PartialEq)]
pub struct NumericRangeFilter {
    /// Optional required element label.
    pub label: Option<LabelId>,
    /// Interned property key whose value is compared.
    pub key: PropertyKeyId,
    /// Lower bound, including whether it is inclusive.
    pub lower: Bound<NumericValue>,
    /// Upper bound, including whether it is inclusive.
    pub upper: Bound<NumericValue>,
}

/// Physical access path selected for a numeric range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericRangeStrategy {
    /// Inspect every target record.
    FullScan,
    /// Start from a persisted label posting, then test the property.
    LabelPosting,
    /// Use the ordered numeric property posting directly.
    NumericPosting,
}

/// Inspectable access decision for a numeric range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericRangePlan {
    /// Selected physical access strategy.
    pub strategy: NumericRangeStrategy,
    /// Conservative number of records the strategy may inspect.
    pub candidate_upper_bound: usize,
}

/// Backward-compatible name used by the node positions of `OneHopQuery`.
pub type NodeFilter = ElementFilter;

/// Exact labelled-property pattern over one relationship.
#[derive(Clone, Debug)]
pub struct OneHopQuery {
    /// Predicate applied to the pattern's start node.
    pub start: NodeFilter,
    /// Optional required relationship label.
    pub edge_label: Option<LabelId>,
    /// Predicate applied to the pattern's end node.
    pub end: NodeFilter,
    /// Relationship direction relative to start and end.
    pub direction: Direction,
    /// Maximum number of matches returned.
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
    /// Selected physical access strategy.
    pub strategy: OneHopStrategy,
    /// Estimated relationships inspected by the selected strategy.
    pub estimated_edge_visits: usize,
    /// Conservative relationship candidates for an edge scan.
    pub edge_candidate_upper_bound: usize,
    /// Conservative nodes matching the start predicate.
    pub start_candidate_upper_bound: usize,
    /// Conservative nodes matching the end predicate.
    pub end_candidate_upper_bound: usize,
}

/// One exact `(start)-[relationship]->(end)` pattern match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatternMatch {
    /// Matched start node.
    pub start: NodeId,
    /// Matched relationship.
    pub edge: EdgeId,
    /// Matched end node.
    pub end: NodeId,
}

/// One-hop graph pattern augmented with vector-scoring weights.
#[derive(Clone, Debug)]
pub struct SemanticOneHopQuery {
    /// Structural and property pattern that candidates must satisfy.
    pub pattern: OneHopQuery,
    /// Number of semantic candidates gathered for selective seeding.
    pub seed_count: usize,
    /// Contribution of start-node similarity.
    pub start_weight: f32,
    /// Contribution of relationship similarity.
    pub edge_weight: f32,
    /// Contribution of end-node similarity.
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

/// One structurally valid one-hop pattern ranked by semantic similarity.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticPatternMatch {
    /// Exact graph pattern that matched.
    pub pattern: PatternMatch,
    /// Weighted combined similarity score.
    pub score: f32,
    /// Start-node similarity contribution before weighting.
    pub start_score: f32,
    /// Relationship similarity when the relationship owns vectors.
    pub edge_score: Option<f32>,
    /// End-node similarity when the node owns vectors.
    pub end_score: Option<f32>,
}

#[derive(Clone, Debug)]
pub(super) struct PathState {
    pub(super) seed: NodeId,
    pub(super) node: NodeId,
    pub(super) score: f32,
    pub(super) seed_score: f32,
    pub(super) path: Vec<EdgeId>,
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
