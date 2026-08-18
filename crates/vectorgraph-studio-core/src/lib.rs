//! UI-independent graph snapshots, layout, camera math, LOD, and hit testing
//! for VectorGraph Studio.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use petgraph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::prelude::Undirected;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_omega::Omega;
use petgraph_layout_sgd::{Scheduler as _, SchedulerExponential};
use petgraph_linalg_rdmds::RdMds;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vectorgraph::{
    Database, Direction, EdgeFilter, EdgeId, ElementRef, GraphStats, NodeId, Property, ReadGuard,
    ShortestPathOptions, ShortestPathStrategy, ShortestPathTermination, Value, VectorTarget,
};

pub use vectorgraph::{
    Direction as PathDirection, ShortestPathStrategy as EvidencePathStrategy,
    ShortestPathTermination as EvidencePathTermination, Value as PropertyValue,
};

/// Which retrieval signals Studio uses for a search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchMode {
    Text,
    Semantic,
    #[default]
    Hybrid,
}

impl SearchMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::Semantic => "Semantic",
            Self::Hybrid => "Hybrid",
        }
    }
}

/// One whole-element search result. Nodes and relationships share the same
/// ranked result surface because both can carry native vectors.
#[derive(Clone, Debug)]
pub struct SearchHit {
    pub element: ElementRef,
    pub label: Arc<str>,
    pub title: Arc<str>,
    pub detail: Arc<str>,
    pub score: f32,
    pub lexical_score: Option<f32>,
    pub semantic_score: Option<f32>,
    pub vector_index: Option<u32>,
}

impl SearchHit {
    pub const fn kind_label(&self) -> &'static str {
        match self.element {
            ElementRef::Node(_) => "NODE",
            ElementRef::Edge(_) => "EDGE",
        }
    }

    pub const fn id(&self) -> u64 {
        match self.element {
            ElementRef::Node(id) | ElementRef::Edge(id) => id,
        }
    }

    pub fn scene_selection(&self, snapshot: &SceneSnapshot) -> Option<SceneSelection> {
        match self.element {
            ElementRef::Node(id) => snapshot.node_index(id).map(SceneSelection::Node),
            ElementRef::Edge(id) => snapshot.edge_index(id).map(SceneSelection::Edge),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchReport {
    pub query: Arc<str>,
    pub mode: SearchMode,
    pub hits: Arc<[SearchHit]>,
    pub elapsed: Duration,
    pub embedding_model: Option<Arc<str>>,
    /// Hybrid search can still return text results if its semantic provider is
    /// unavailable; the reason remains visible instead of silently degrading.
    pub warning: Option<Arc<str>>,
}

/// One owned node in an exact evidence path. Keeping presentation data out of
/// the read guard lets Studio render the result after the database lock and
/// background task have both been released.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceNode {
    pub id: NodeId,
    pub label: Arc<str>,
    pub title: Arc<str>,
    pub vector_count: u32,
    pub properties: Arc<[SceneProperty]>,
}

/// One directed traversal step. `forward` describes whether the path follows
/// the relationship's stored source-to-target orientation.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceStep {
    pub edge_id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub label: Arc<str>,
    pub title: Arc<str>,
    pub forward: bool,
    pub vector_count: u32,
    pub properties: Arc<[SceneProperty]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidencePath {
    pub nodes: Arc<[EvidenceNode]>,
    pub steps: Arc<[EvidenceStep]>,
}

/// A presentation-ready exact path result with the engine's completeness and
/// work diagnostics preserved. `ExpansionLimit` must never be presented as
/// proof that no relationship exists.
#[derive(Clone, Debug)]
pub struct EvidencePathReport {
    pub start: EvidenceNode,
    pub end: EvidenceNode,
    pub path: Option<EvidencePath>,
    pub strategy: ShortestPathStrategy,
    pub termination: ShortestPathTermination,
    pub direction: Direction,
    pub relationship_label: Option<Arc<str>>,
    pub max_hops: usize,
    pub visited_nodes: usize,
    pub start_expanded_nodes: usize,
    pub end_expanded_nodes: usize,
    pub expanded_nodes: usize,
    pub examined_relationships: usize,
    pub elapsed: Duration,
}

/// Finds and hydrates an exact, bounded evidence path in a read-only database.
/// The whole operation is synchronous so UI clients can move it onto their
/// background executor as one cancellable unit.
pub fn evidence_path_database(
    path: &Path,
    start: NodeId,
    end: NodeId,
    direction: Direction,
    relationship_label: Option<&str>,
    max_hops: usize,
    max_expansions: usize,
) -> Result<EvidencePathReport, String> {
    let started = Instant::now();
    let database = Database::open_read_only(path)
        .map_err(|error| format!("could not open database for path search: {error}"))?;
    let read = database.read();
    let start_node = hydrate_evidence_node(&read, start)
        .ok_or_else(|| format!("start node {start} does not exist"))?;
    let end_node = hydrate_evidence_node(&read, end)
        .ok_or_else(|| format!("end node {end} does not exist"))?;
    let relationship_label = relationship_label
        .map(str::trim)
        .filter(|label| !label.is_empty());
    let edge_label = relationship_label
        .map(|label| {
            read.label_id(label)
                .ok_or_else(|| format!("relationship label “{label}” does not exist"))
        })
        .transpose()?;
    let result = read
        .shortest_path(
            start,
            end,
            &ShortestPathOptions {
                max_hops,
                max_expansions,
                direction,
                edge_filter: EdgeFilter { label: edge_label },
            },
        )
        .map_err(|error| format!("path search failed: {error}"))?;
    let hydrated_path = result
        .path
        .as_ref()
        .map(|path| hydrate_evidence_path(&read, path))
        .transpose()?;

    Ok(EvidencePathReport {
        start: start_node,
        end: end_node,
        path: hydrated_path,
        strategy: result.strategy,
        termination: result.termination,
        direction,
        relationship_label: relationship_label.map(Arc::from),
        max_hops,
        visited_nodes: result.visited_nodes,
        start_expanded_nodes: result.start_expanded_nodes,
        end_expanded_nodes: result.end_expanded_nodes,
        expanded_nodes: result.expanded_nodes,
        examined_relationships: result.examined_relationships,
        elapsed: started.elapsed(),
    })
}

/// Searches the complete database without materializing all matching rows.
///
/// Text search retains a bounded heap while scanning properties. Semantic
/// search delegates to VectorGraph's adaptive native index. Hybrid retrieval
/// fuses their normalized scores and deduplicates whole elements, so multiple
/// vector facets do not crowd the result list.
pub fn search_database(
    path: &Path,
    query: &str,
    mode: SearchMode,
    embedding_model: &str,
    limit: usize,
) -> Result<SearchReport, String> {
    let query = query.trim();
    if query.is_empty() {
        return Err("search query is empty".into());
    }
    let started = Instant::now();
    let database = Database::open_read_only(path)
        .map_err(|error| format!("could not open database for search: {error}"))?;
    let candidate_limit = limit.max(1).saturating_mul(6).max(32);

    let semantic_query = if mode == SearchMode::Text {
        None
    } else {
        Some(vectorgraph_embedding::embed_query(
            embedding_model,
            database.vector_dimension(),
            query,
        ))
    };
    let read = database.read();
    let lexical = if mode == SearchMode::Semantic {
        Vec::new()
    } else {
        lexical_candidates(&read, query, candidate_limit)
    };

    let mut warning = None;
    let semantic = match semantic_query {
        None => Vec::new(),
        Some(Ok(query_vector)) => semantic_candidates(&read, &query_vector, candidate_limit)?,
        Some(Err(error)) if mode == SearchMode::Hybrid => {
            warning = Some(Arc::from(format!(
                "Semantic retrieval unavailable ({error}); showing text matches"
            )));
            Vec::new()
        }
        Some(Err(error)) => return Err(error),
    };

    let lexical_max = lexical
        .first()
        .map_or(1.0, |candidate| candidate.score.max(f32::EPSILON));
    let mut signals = HashMap::<ElementRef, SignalScores>::with_capacity(
        lexical.len().saturating_add(semantic.len()),
    );
    for candidate in lexical {
        signals.entry(candidate.element).or_default().lexical =
            Some((candidate.score / lexical_max).clamp(0.0, 1.0));
    }
    for candidate in semantic {
        let entry = signals.entry(candidate.element).or_default();
        let normalized = ((candidate.score + 1.0) * 0.5).clamp(0.0, 1.0);
        if entry.semantic.is_none_or(|score| normalized > score) {
            entry.semantic = Some(normalized);
            entry.vector_index = Some(candidate.vector_index);
        }
    }

    let mut ranked: Vec<_> = signals
        .into_iter()
        .map(|(element, signals)| {
            let lexical = signals.lexical.unwrap_or(0.0);
            let semantic = signals.semantic.unwrap_or(0.0);
            let score = match mode {
                SearchMode::Text => lexical,
                SearchMode::Semantic => semantic,
                SearchMode::Hybrid => 0.46_f32.mul_add(lexical, 0.54 * semantic),
            };
            (element, signals, score)
        })
        .collect();
    ranked.sort_unstable_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(limit.max(1));

    let hits = ranked
        .into_iter()
        .filter_map(|(element, signals, score)| hydrate_search_hit(&read, element, score, &signals))
        .collect::<Vec<_>>();
    Ok(SearchReport {
        query: Arc::from(query),
        mode,
        hits: hits.into(),
        elapsed: started.elapsed(),
        embedding_model: (mode != SearchMode::Text).then(|| Arc::from(embedding_model)),
        warning,
    })
}

#[derive(Clone, Copy, Debug)]
struct RankedElement {
    element: ElementRef,
    score: f32,
    vector_index: u32,
}

impl PartialEq for RankedElement {
    fn eq(&self, other: &Self) -> bool {
        self.element == other.element
            && self.score.to_bits() == other.score.to_bits()
            && self.vector_index == other.vector_index
    }
}

impl Eq for RankedElement {}

impl PartialOrd for RankedElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedElement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.element.cmp(&other.element))
            .then_with(|| self.vector_index.cmp(&other.vector_index))
    }
}

fn lexical_candidates(read: &ReadGuard<'_>, query: &str, limit: usize) -> Vec<RankedElement> {
    let phrase = query.to_ascii_lowercase();
    let terms: Vec<_> = phrase
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .filter(|term| !term.is_empty())
        .collect();
    let mut heap = BinaryHeap::<Reverse<RankedElement>>::with_capacity(limit.saturating_add(1));

    for id in read.node_ids() {
        if let Some(node) = read.node(id) {
            let label = read.symbol(node.label).unwrap_or("Unknown");
            let score = lexical_score(read, label, &node.properties, &phrase, &terms);
            retain_candidate(
                &mut heap,
                RankedElement {
                    element: ElementRef::Node(id),
                    score,
                    vector_index: 0,
                },
                limit,
            );
        }
    }
    for id in read.edge_ids() {
        if let Some(edge) = read.edge(id) {
            let label = read.symbol(edge.label).unwrap_or("Unknown");
            let score = lexical_score(read, label, &edge.properties, &phrase, &terms);
            retain_candidate(
                &mut heap,
                RankedElement {
                    element: ElementRef::Edge(id),
                    score,
                    vector_index: 0,
                },
                limit,
            );
        }
    }
    let mut candidates: Vec<_> = heap.into_iter().map(|candidate| candidate.0).collect();
    candidates.sort_unstable_by(|left, right| right.cmp(left));
    candidates
}

fn retain_candidate(
    heap: &mut BinaryHeap<Reverse<RankedElement>>,
    candidate: RankedElement,
    limit: usize,
) {
    if candidate.score <= 0.0 || limit == 0 {
        return;
    }
    if heap.len() < limit {
        heap.push(Reverse(candidate));
    } else if heap.peek().is_some_and(|worst| candidate > worst.0) {
        heap.pop();
        heap.push(Reverse(candidate));
    }
}

fn lexical_score(
    read: &ReadGuard<'_>,
    label: &str,
    properties: &[Property],
    phrase: &str,
    terms: &[&str],
) -> f32 {
    let mut score = score_text(&label.to_ascii_lowercase(), phrase, terms, 1.2);
    for property in properties {
        let Some(value) = searchable_value(&property.value) else {
            continue;
        };
        let key = read.symbol(property.key).unwrap_or("");
        let weight = match key {
            "title" | "name" | "headline" | "path" | "tag_name" | "login" => 1.5,
            "body" | "description" | "message" | "url" => 0.85,
            _ => 0.55,
        };
        score += score_text(&value.to_ascii_lowercase(), phrase, terms, weight);
    }
    score
}

fn score_text(text: &str, phrase: &str, terms: &[&str], weight: f32) -> f32 {
    let phrase_score = if text == phrase {
        1.4
    } else if !phrase.is_empty() && text.contains(phrase) {
        0.9
    } else {
        0.0
    };
    let matched = terms.iter().filter(|term| text.contains(**term)).count();
    if phrase_score == 0.0 && matched == 0 {
        0.0
    } else {
        weight * (phrase_score + matched as f32 / terms.len().max(1) as f32)
    }
}

fn searchable_value(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Int(value) => Some(value.to_string()),
        Value::Float(value) => Some(value.to_string()),
        Value::Node(value) | Value::Edge(value) => Some(value.to_string()),
        Value::Null | Value::Bytes(_) => None,
    }
}

fn semantic_candidates(
    read: &ReadGuard<'_>,
    query: &[f32],
    limit: usize,
) -> Result<Vec<RankedElement>, String> {
    read.vector_search_adaptive(query, VectorTarget::Both, limit, None)
        .map(|hits| {
            hits.into_iter()
                .map(|hit| RankedElement {
                    element: hit.element,
                    score: hit.score,
                    vector_index: hit.vector_index,
                })
                .collect()
        })
        .map_err(|error| format!("semantic search failed: {error}"))
}

fn hydrate_search_hit(
    read: &ReadGuard<'_>,
    element: ElementRef,
    score: f32,
    signals: &impl SearchSignals,
) -> Option<SearchHit> {
    let (label, title, detail) = match element {
        ElementRef::Node(id) => {
            let node = read.node(id)?;
            let label = read.symbol(node.label).unwrap_or("Unknown");
            let title = preferred_property(read, &node.properties)
                .unwrap_or_else(|| format!("{label} {id}"));
            let detail = secondary_property(read, &node.properties, &title)
                .unwrap_or_else(|| format!("node:{id}"));
            (label.to_string(), title, detail)
        }
        ElementRef::Edge(id) => {
            let edge = read.edge(id)?;
            let label = read.symbol(edge.label).unwrap_or("Unknown");
            let title = preferred_property(read, &edge.properties)
                .unwrap_or_else(|| label.replace('_', " "));
            let detail = secondary_property(read, &edge.properties, &title)
                .unwrap_or_else(|| format!("node:{} → node:{}", edge.source, edge.target));
            (label.to_string(), title, detail)
        }
    };
    Some(SearchHit {
        element,
        label: label.into(),
        title: truncate_text(&title, 92).into(),
        detail: truncate_text(&detail, 132).into(),
        score,
        lexical_score: signals.lexical(),
        semantic_score: signals.semantic(),
        vector_index: signals.vector_index(),
    })
}

fn hydrate_evidence_node(read: &ReadGuard<'_>, id: NodeId) -> Option<EvidenceNode> {
    let node = read.node(id)?;
    let label = read.symbol(node.label).unwrap_or("Unknown");
    let title =
        preferred_property(read, &node.properties).unwrap_or_else(|| format!("{label} {id}"));
    Some(EvidenceNode {
        id,
        label: Arc::from(label),
        title: Arc::from(truncate_text(&title, 92)),
        vector_count: node.vector_count,
        properties: owned_properties(read, &node.properties),
    })
}

fn hydrate_evidence_path(
    read: &ReadGuard<'_>,
    path: &vectorgraph::ShortestPath,
) -> Result<EvidencePath, String> {
    let nodes = path
        .nodes
        .iter()
        .copied()
        .map(|id| {
            hydrate_evidence_node(read, id)
                .ok_or_else(|| format!("path node {id} disappeared during hydration"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut steps = Vec::with_capacity(path.edges.len());
    for (index, &edge_id) in path.edges.iter().enumerate() {
        let from = path.nodes[index];
        let to = path.nodes[index + 1];
        let edge = read
            .edge(edge_id)
            .ok_or_else(|| format!("path relationship {edge_id} disappeared during hydration"))?;
        let label = read.symbol(edge.label).unwrap_or("Unknown");
        let title =
            preferred_property(read, &edge.properties).unwrap_or_else(|| label.replace('_', " "));
        steps.push(EvidenceStep {
            edge_id,
            from,
            to,
            label: Arc::from(label),
            title: Arc::from(truncate_text(&title, 92)),
            forward: edge.source == from && edge.target == to,
            vector_count: edge.vector_count,
            properties: owned_properties(read, &edge.properties),
        });
    }
    Ok(EvidencePath {
        nodes: nodes.into(),
        steps: steps.into(),
    })
}

trait SearchSignals {
    fn lexical(&self) -> Option<f32>;
    fn semantic(&self) -> Option<f32>;
    fn vector_index(&self) -> Option<u32>;
}

#[derive(Default)]
struct SignalScores {
    lexical: Option<f32>,
    semantic: Option<f32>,
    vector_index: Option<u32>,
}

impl SearchSignals for SignalScores {
    fn lexical(&self) -> Option<f32> {
        self.lexical
    }

    fn semantic(&self) -> Option<f32> {
        self.semantic
    }

    fn vector_index(&self) -> Option<u32> {
        self.vector_index
    }
}

fn preferred_property(read: &ReadGuard<'_>, properties: &[Property]) -> Option<String> {
    ["title", "name", "headline", "path", "tag_name", "login"]
        .into_iter()
        .find_map(|key| read.property(properties, key).and_then(searchable_value))
}

fn secondary_property(
    read: &ReadGuard<'_>,
    properties: &[Property],
    title: &str,
) -> Option<String> {
    ["body", "description", "message", "url", "state", "kind"]
        .into_iter()
        .filter_map(|key| read.property(properties, key).and_then(searchable_value))
        .find(|value| value != title && !value.is_empty())
}

fn truncate_text(text: &str, limit: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= limit {
        normalized
    } else {
        let mut output: String = normalized.chars().take(limit.saturating_sub(1)).collect();
        output.push('…');
        output
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length_squared(self) -> f32 {
        self.x.mul_add(self.x, self.y * self.y)
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Default for Rect {
    fn default() -> Self {
        Self {
            min: Vec2::new(-1.0, -1.0),
            max: Vec2::new(1.0, 1.0),
        }
    }
}

impl Rect {
    pub fn from_points(points: &[Vec2]) -> Self {
        let Some(first) = points.first().copied() else {
            return Self::default();
        };
        let mut bounds = Self {
            min: first,
            max: first,
        };
        for point in points.iter().copied().skip(1) {
            bounds.min.x = bounds.min.x.min(point.x);
            bounds.min.y = bounds.min.y.min(point.y);
            bounds.max.x = bounds.max.x.max(point.x);
            bounds.max.y = bounds.max.y.max(point.y);
        }
        if bounds.width() < 1.0 {
            bounds.min.x -= 0.5;
            bounds.max.x += 0.5;
        }
        if bounds.height() < 1.0 {
            bounds.min.y -= 0.5;
            bounds.max.y += 0.5;
        }
        bounds
    }

    pub fn width(self) -> f32 {
        self.max.x - self.min.x
    }

    pub fn height(self) -> f32 {
        self.max.y - self.min.y
    }

    pub fn center(self) -> Vec2 {
        (self.min + self.max) * 0.5
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SceneProperty {
    pub key: Arc<str>,
    pub value: Value,
}

#[derive(Clone, Debug, Default)]
pub struct SceneNodes {
    pub ids: Vec<NodeId>,
    pub labels: Vec<Arc<str>>,
    pub positions: Vec<Vec2>,
    pub degrees: Vec<u32>,
    pub vector_counts: Vec<u32>,
    pub properties: Vec<Arc<[SceneProperty]>>,
}

#[derive(Clone, Debug, Default)]
pub struct SceneEdges {
    pub ids: Vec<EdgeId>,
    pub sources: Vec<u32>,
    pub targets: Vec<u32>,
    pub labels: Vec<Arc<str>>,
    pub vector_counts: Vec<u32>,
    pub properties: Vec<Arc<[SceneProperty]>>,
}

#[derive(Clone, Debug)]
pub struct SceneSnapshot {
    pub revision: u64,
    pub database_name: Arc<str>,
    pub source_stats: GraphStats,
    pub nodes: SceneNodes,
    pub edges: SceneEdges,
    pub bounds: Rect,
    pub sampled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneSelection {
    Node(usize),
    Edge(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneFocus {
    pub roots: Vec<usize>,
    pub nodes: Vec<usize>,
    pub edges: Vec<usize>,
    /// Breadth-first layers, including roots at layer zero.
    pub layers: Vec<Vec<usize>>,
    /// First-discovery parent for each non-root node.
    pub parents: Vec<(usize, usize)>,
}

impl SceneSelection {
    pub const fn node(self) -> Option<usize> {
        match self {
            Self::Node(index) => Some(index),
            Self::Edge(_) => None,
        }
    }

    pub const fn edge(self) -> Option<usize> {
        match self {
            Self::Node(_) => None,
            Self::Edge(index) => Some(index),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SnapshotOptions {
    pub max_nodes: usize,
    pub max_edges: usize,
}

impl Default for SnapshotOptions {
    fn default() -> Self {
        Self {
            max_nodes: 50_000,
            max_edges: 150_000,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LayoutOptions {
    pub iterations: usize,
    pub edge_length: f32,
    pub attraction: f32,
    pub repulsion: f32,
    pub gravity: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayoutKind {
    #[default]
    Auto,
    Force,
    Structure,
    Orbit,
}

impl LayoutKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Force => "Force",
            Self::Structure => "Structure",
            Self::Orbit => "Orbit",
        }
    }
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            iterations: 72,
            edge_length: 42.0,
            attraction: 0.018,
            repulsion: 1_600.0,
            gravity: 0.0025,
        }
    }
}

impl SceneSnapshot {
    pub fn open(
        path: impl AsRef<Path>,
        snapshot_options: SnapshotOptions,
        layout_options: LayoutOptions,
    ) -> vectorgraph::Result<Self> {
        let database = Database::open_read_only(path)?;
        let mut snapshot = Self::from_database(&database, snapshot_options)?;
        snapshot.layout(layout_options);
        Ok(snapshot)
    }

    pub fn from_database(
        database: &Database,
        options: SnapshotOptions,
    ) -> vectorgraph::Result<Self> {
        let read = database.read();
        let source_stats = read.stats();
        let all_node_ids = read.node_ids();
        let all_edge_ids = read.edge_ids();
        let node_limit = options.max_nodes.max(1);
        let edge_limit = options.max_edges.max(1);

        let mut selected = HashSet::with_capacity(node_limit.min(all_node_ids.len()));
        if all_node_ids.len() <= node_limit {
            selected.extend(all_node_ids.iter().copied());
        } else {
            // Prefer an endpoint-connected overview over an ID-prefix sample.
            // A stable stride keeps this bounded and reproducible on large files.
            let edge_stride = all_edge_ids.len().div_ceil(edge_limit).max(1);
            for edge_id in all_edge_ids.iter().step_by(edge_stride) {
                let Some(edge) = read.edge(*edge_id) else {
                    continue;
                };
                let needed = usize::from(!selected.contains(&edge.source))
                    + usize::from(!selected.contains(&edge.target));
                if selected.len() + needed <= node_limit {
                    selected.insert(edge.source);
                    selected.insert(edge.target);
                }
                if selected.len() == node_limit {
                    break;
                }
            }

            if selected.len() < node_limit {
                let remaining = node_limit - selected.len();
                let stride = all_node_ids.len().div_ceil(remaining).max(1);
                for node_id in all_node_ids.iter().step_by(stride) {
                    selected.insert(*node_id);
                    if selected.len() == node_limit {
                        break;
                    }
                }
            }
        }

        let mut selected_ids: Vec<_> = selected.into_iter().collect();
        selected_ids.sort_unstable();
        let index_by_id: HashMap<_, _> = selected_ids
            .iter()
            .enumerate()
            .map(|(index, id)| (*id, index as u32))
            .collect();

        let mut nodes = SceneNodes::default();
        nodes.ids.reserve(selected_ids.len());
        nodes.labels.reserve(selected_ids.len());
        nodes.positions.resize(selected_ids.len(), Vec2::ZERO);
        nodes.degrees.resize(selected_ids.len(), 0);
        nodes.vector_counts.reserve(selected_ids.len());
        nodes.properties.reserve(selected_ids.len());

        for node_id in selected_ids {
            let Some(node) = read.node(node_id) else {
                continue;
            };
            nodes.ids.push(node.id);
            nodes.labels.push(symbol_or_unknown(&read, node.label));
            nodes.vector_counts.push(node.vector_count);
            nodes
                .properties
                .push(owned_properties(&read, &node.properties));
        }

        let mut edges = SceneEdges::default();
        edges.ids.reserve(edge_limit.min(all_edge_ids.len()));
        edges.sources.reserve(edge_limit.min(all_edge_ids.len()));
        edges.targets.reserve(edge_limit.min(all_edge_ids.len()));
        edges.labels.reserve(edge_limit.min(all_edge_ids.len()));
        edges
            .vector_counts
            .reserve(edge_limit.min(all_edge_ids.len()));
        edges.properties.reserve(edge_limit.min(all_edge_ids.len()));

        for edge_id in all_edge_ids {
            if edges.ids.len() == edge_limit {
                break;
            }
            let Some(edge) = read.edge(edge_id) else {
                continue;
            };
            let (Some(&source), Some(&target)) =
                (index_by_id.get(&edge.source), index_by_id.get(&edge.target))
            else {
                continue;
            };
            edges.ids.push(edge.id);
            edges.sources.push(source);
            edges.targets.push(target);
            edges.labels.push(symbol_or_unknown(&read, edge.label));
            edges.vector_counts.push(edge.vector_count);
            edges
                .properties
                .push(owned_properties(&read, &edge.properties));
            nodes.degrees[source as usize] = nodes.degrees[source as usize].saturating_add(1);
            nodes.degrees[target as usize] = nodes.degrees[target as usize].saturating_add(1);
        }

        let database_name = database
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Untitled.vg")
            .into();
        let sampled = nodes.ids.len() < source_stats.nodes || edges.ids.len() < source_stats.edges;

        Ok(Self {
            revision: source_stats.transactions,
            database_name,
            source_stats,
            nodes,
            edges,
            bounds: Rect::default(),
            sampled,
        })
    }

    pub fn demo() -> Self {
        const LABELS: [&str; 4] = ["Document", "Concept", "Claim", "Source"];
        let mut nodes = SceneNodes::default();
        for index in 0..96_u64 {
            nodes.ids.push(index);
            nodes
                .labels
                .push(LABELS[index as usize % LABELS.len()].into());
            nodes.positions.push(Vec2::ZERO);
            nodes.degrees.push(0);
            nodes.vector_counts.push(u32::from(index % 3 != 0));
            nodes.properties.push(
                vec![SceneProperty {
                    key: "title".into(),
                    value: Value::String(format!("Context element {index}").into()),
                }]
                .into(),
            );
        }

        let mut edges = SceneEdges::default();
        for source in 0..96_u32 {
            for offset in [1_u32, 7, 19] {
                let target = (source + offset) % 96;
                if source >= target && offset != 1 {
                    continue;
                }
                edges.ids.push(edges.ids.len() as u64);
                edges.sources.push(source);
                edges.targets.push(target);
                edges
                    .labels
                    .push(if offset == 1 { "NEXT" } else { "SUPPORTS" }.into());
                edges.vector_counts.push(u32::from(offset != 1));
                edges.properties.push(Arc::from([]));
                nodes.degrees[source as usize] += 1;
                nodes.degrees[target as usize] += 1;
            }
        }

        let mut snapshot = Self {
            revision: 0,
            database_name: "VectorGraph demo".into(),
            source_stats: GraphStats {
                nodes: nodes.ids.len(),
                edges: edges.ids.len(),
                labels: LABELS.len() + 2,
                indexed_vectors: 0,
                transactions: 0,
            },
            nodes,
            edges,
            bounds: Rect::default(),
            sampled: false,
        };
        snapshot.layout(LayoutOptions::default());
        snapshot
    }

    pub fn layout(&mut self, options: LayoutOptions) {
        self.nodes.positions = self.arranged_positions(LayoutKind::Auto, options, None, None);
        self.bounds = Rect::from_points(&self.nodes.positions);
    }

    pub fn arranged_positions(
        &self,
        kind: LayoutKind,
        options: LayoutOptions,
        starting_positions: Option<&[Vec2]>,
        pinned: Option<&[bool]>,
    ) -> Vec<Vec2> {
        let count = self.nodes.ids.len();
        if count == 0 {
            return Vec::new();
        }

        let has_start = starting_positions.is_some_and(|positions| positions.len() == count);
        let mut positions = starting_positions
            .filter(|positions| positions.len() == count)
            .map_or_else(|| vec![Vec2::ZERO; count], <[Vec2]>::to_vec);

        // Trees and forests are common in ASTs, taxonomies, dependency
        // hierarchies, and provenance graphs. Auto keeps that structure in a
        // linear-time radial layout. Force is always available as an explicit
        // physical rearrangement, including from the user's current positions.
        const AUTO_RADIAL_NODE_LIMIT: usize = 2_000;
        if kind == LayoutKind::Auto
            && count <= AUTO_RADIAL_NODE_LIMIT
            && try_layout_radial_forest(&mut positions, &self.edges, options.edge_length)
        {
            recenter(&mut positions);
            return positions;
        }

        if kind == LayoutKind::Auto
            && count > AUTO_RADIAL_NODE_LIMIT
            && try_layout_clustered_forest(
                &mut positions,
                &self.nodes.ids,
                &self.edges,
                options.edge_length,
            )
        {
            run_force_layout(&mut positions, &self.edges, options, None);
            return positions;
        }

        if kind == LayoutKind::Orbit {
            layout_orbits(
                &mut positions,
                &self.nodes.ids,
                &self.nodes.labels,
                &self.nodes.degrees,
                options.edge_length,
            );
            recenter(&mut positions);
            return positions;
        }

        if kind == LayoutKind::Structure {
            if !has_start {
                initialize_by_label(&mut positions, &self.nodes.ids, &self.nodes.labels);
            }
            run_resistance_structure_layout(&mut positions, &self.edges, options);
            return positions;
        }

        if !has_start {
            initialize_by_label(&mut positions, &self.nodes.ids, &self.nodes.labels);
        }
        run_force_layout(
            &mut positions,
            &self.edges,
            options,
            pinned.filter(|pins| pins.len() == count),
        );
        if !pinned.is_some_and(|pins| pins.iter().any(|&pin| pin)) {
            recenter(&mut positions);
        }
        positions
    }

    pub fn label_counts(&self) -> Vec<(Arc<str>, usize)> {
        count_labels(&self.nodes.labels)
    }

    pub fn relationship_counts(&self) -> Vec<(Arc<str>, usize)> {
        count_labels(&self.edges.labels)
    }

    pub fn node_index(&self, id: NodeId) -> Option<usize> {
        self.nodes.ids.binary_search(&id).ok()
    }

    pub fn edge_index(&self, id: EdgeId) -> Option<usize> {
        self.edges.ids.binary_search(&id).ok()
    }

    /// Returns a scene that retains the current overview and injects every
    /// fully hydrated element in an exact evidence path that sampling omitted.
    /// Existing presentation positions are preserved; missing path nodes are
    /// deterministically interpolated between the nearest visible anchors.
    ///
    /// Cloning and reindexing are linear in the bounded scene, so callers
    /// should run this on a background executor. `None` means the current scene
    /// already contains the complete path.
    pub fn including_evidence_path(
        &self,
        report: &EvidencePathReport,
        overview_positions: &[Vec2],
    ) -> Result<Option<Self>, String> {
        let Some(path) = &report.path else {
            return Ok(None);
        };
        if overview_positions.len() != self.nodes.ids.len()
            || overview_positions
                .iter()
                .any(|position| !position.x.is_finite() || !position.y.is_finite())
        {
            return Err("evidence scene positions do not match the current snapshot".into());
        }
        let missing_node = path
            .nodes
            .iter()
            .any(|node| self.node_index(node.id).is_none());
        let missing_edge = path
            .steps
            .iter()
            .any(|step| self.edge_index(step.edge_id).is_none());
        if !missing_node && !missing_edge {
            return Ok(None);
        }

        let path_positions = evidence_path_positions(self, path, overview_positions);
        let mut snapshot = self.clone();
        snapshot.nodes.positions = overview_positions.to_vec();
        let existing_edge_endpoints: Vec<_> = snapshot
            .edges
            .sources
            .iter()
            .zip(&snapshot.edges.targets)
            .map(|(&source, &target)| {
                (
                    snapshot.nodes.ids[source as usize],
                    snapshot.nodes.ids[target as usize],
                )
            })
            .collect();

        let mut included_node_ids: HashSet<_> = snapshot.nodes.ids.iter().copied().collect();
        for node in path.nodes.iter() {
            if !included_node_ids.insert(node.id) {
                continue;
            }
            snapshot.nodes.ids.push(node.id);
            snapshot.nodes.labels.push(node.label.clone());
            snapshot.nodes.positions.push(path_positions[&node.id]);
            snapshot.nodes.degrees.push(0);
            snapshot.nodes.vector_counts.push(node.vector_count);
            snapshot.nodes.properties.push(node.properties.clone());
        }
        sort_scene_nodes(&mut snapshot.nodes);
        let index_by_id: HashMap<_, _> = snapshot
            .nodes
            .ids
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index as u32))
            .collect();
        for (index, &(source, target)) in existing_edge_endpoints.iter().enumerate() {
            snapshot.edges.sources[index] = index_by_id[&source];
            snapshot.edges.targets[index] = index_by_id[&target];
        }

        let mut included_edge_ids: HashSet<_> = snapshot.edges.ids.iter().copied().collect();
        for step in path.steps.iter() {
            if !included_edge_ids.insert(step.edge_id) {
                continue;
            }
            let (source, target) = if step.forward {
                (step.from, step.to)
            } else {
                (step.to, step.from)
            };
            let (Some(&source), Some(&target)) =
                (index_by_id.get(&source), index_by_id.get(&target))
            else {
                return Err(format!(
                    "evidence relationship {} references a missing endpoint",
                    step.edge_id
                ));
            };
            snapshot.edges.ids.push(step.edge_id);
            snapshot.edges.sources.push(source);
            snapshot.edges.targets.push(target);
            snapshot.edges.labels.push(step.label.clone());
            snapshot.edges.vector_counts.push(step.vector_count);
            snapshot.edges.properties.push(step.properties.clone());
        }
        sort_scene_edges(&mut snapshot.edges);
        snapshot.nodes.degrees.fill(0);
        for (&source, &target) in snapshot.edges.sources.iter().zip(&snapshot.edges.targets) {
            let source = source as usize;
            let target = target as usize;
            snapshot.nodes.degrees[source] = snapshot.nodes.degrees[source].saturating_add(1);
            snapshot.nodes.degrees[target] = snapshot.nodes.degrees[target].saturating_add(1);
        }
        snapshot.bounds = Rect::from_points(&snapshot.nodes.positions);
        snapshot.sampled = snapshot.nodes.ids.len() < snapshot.source_stats.nodes
            || snapshot.edges.ids.len() < snapshot.source_stats.edges;
        Ok(Some(snapshot))
    }

    /// Returns a deterministic, bounded one-hop context for a selected graph
    /// element. The selected edge is always retained even at the edge budget.
    pub fn focus_neighborhood(
        &self,
        selection: SceneSelection,
        edge_budget: usize,
    ) -> Option<SceneFocus> {
        self.focus_neighborhood_layers(selection, 1, edge_budget.saturating_add(2), edge_budget)
    }

    /// Builds a deterministic, type-diverse, branch-balanced context without
    /// allowing hubs to consume an unbounded scene. Earlier layers reserve
    /// capacity for later context, and expansion rotates across both frontier
    /// nodes and relationship labels before taking another edge of one type.
    pub fn focus_neighborhood_layers(
        &self,
        selection: SceneSelection,
        max_depth: usize,
        node_budget: usize,
        edge_budget: usize,
    ) -> Option<SceneFocus> {
        let mut node_seen = vec![false; self.nodes.ids.len()];
        let mut edge_seen = vec![false; self.edges.ids.len()];
        let roots = match selection {
            SceneSelection::Node(index) if index < node_seen.len() => {
                node_seen[index] = true;
                vec![index]
            }
            SceneSelection::Edge(index) if index < edge_seen.len() => {
                let source = self.edges.sources[index] as usize;
                let target = self.edges.targets[index] as usize;
                if source >= node_seen.len() || target >= node_seen.len() {
                    return None;
                }
                node_seen[source] = true;
                node_seen[target] = true;
                edge_seen[index] = true;
                vec![source, target]
            }
            SceneSelection::Node(_) | SceneSelection::Edge(_) => return None,
        };

        let node_budget = node_budget.max(roots.len());
        let edge_budget = edge_budget.max(edge_seen.iter().filter(|&&seen| seen).count());
        let mut layers = vec![roots.clone()];
        let mut parents = Vec::new();
        let mut node_count = roots.len();
        let mut edge_count = edge_seen.iter().filter(|&&seen| seen).count();
        let mut frontier_slot = vec![usize::MAX; self.nodes.ids.len()];

        for depth in 1..=max_depth {
            if node_count >= node_budget || edge_count >= edge_budget {
                break;
            }
            let frontier = layers.last().unwrap();
            for (slot, &node) in frontier.iter().enumerate() {
                frontier_slot[node] = slot;
            }
            let mut candidates =
                vec![BTreeMap::<Arc<str>, Vec<(usize, usize)>>::new(); frontier.len()];
            for edge_index in 0..self.edges.ids.len() {
                let source = self.edges.sources[edge_index] as usize;
                let target = self.edges.targets[edge_index] as usize;
                if source >= frontier_slot.len() || target >= frontier_slot.len() {
                    continue;
                }
                let label = &self.edges.labels[edge_index];
                if frontier_slot[source] != usize::MAX && source != target {
                    candidates[frontier_slot[source]]
                        .entry(label.clone())
                        .or_default()
                        .push((edge_index, target));
                }
                if frontier_slot[target] != usize::MAX && source != target {
                    candidates[frontier_slot[target]]
                        .entry(label.clone())
                        .or_default()
                        .push((edge_index, source));
                }
            }
            for &node in frontier {
                frontier_slot[node] = usize::MAX;
            }

            let remaining_nodes = node_budget - node_count;
            let remaining_layers = max_depth - depth + 1;
            let layer_budget = if remaining_layers == 1 {
                remaining_nodes
            } else {
                (remaining_nodes / (remaining_layers + 1)).max(1)
            };
            type FocusEdgeBucket = (Vec<(usize, usize)>, usize);
            let mut buckets: Vec<Vec<FocusEdgeBucket>> = candidates
                .into_iter()
                .map(|types| {
                    types
                        .into_values()
                        .map(|mut edges| {
                            edges.sort_unstable_by(|left, right| {
                                self.nodes.degrees[left.1]
                                    .cmp(&self.nodes.degrees[right.1])
                                    .then_with(|| {
                                        self.nodes.ids[left.1].cmp(&self.nodes.ids[right.1])
                                    })
                                    .then_with(|| left.0.cmp(&right.0))
                            });
                            (edges, 0)
                        })
                        .collect()
                })
                .collect();
            let mut next_layer = Vec::with_capacity(layer_budget);
            while next_layer.len() < layer_budget && edge_count < edge_budget {
                let mut progressed = false;
                for (frontier_index, type_buckets) in buckets.iter_mut().enumerate() {
                    for (bucket, cursor) in type_buckets {
                        while *cursor < bucket.len() {
                            let (edge_index, neighbor) = bucket[*cursor];
                            *cursor += 1;
                            if node_seen[neighbor] {
                                continue;
                            }
                            node_seen[neighbor] = true;
                            edge_seen[edge_index] = true;
                            edge_count += 1;
                            node_count += 1;
                            next_layer.push(neighbor);
                            parents.push((neighbor, frontier[frontier_index]));
                            progressed = true;
                            break;
                        }
                        if next_layer.len() == layer_budget || edge_count == edge_budget {
                            break;
                        }
                    }
                    if next_layer.len() == layer_budget || edge_count == edge_budget {
                        break;
                    }
                }
                if !progressed {
                    break;
                }
            }
            if next_layer.is_empty() {
                break;
            }
            layers.push(next_layer);
        }

        // Preserve cross-links and parallel relationship evidence between the
        // selected nodes after expansion has established the node budget.
        for (edge_index, seen) in edge_seen.iter_mut().enumerate() {
            if edge_count >= edge_budget {
                break;
            }
            if *seen {
                continue;
            }
            let source = self.edges.sources[edge_index] as usize;
            let target = self.edges.targets[edge_index] as usize;
            if source < node_seen.len()
                && target < node_seen.len()
                && node_seen[source]
                && node_seen[target]
            {
                *seen = true;
                edge_count += 1;
            }
        }

        let nodes = layers.iter().flatten().copied().collect();

        Some(SceneFocus {
            roots,
            nodes,
            edges: edge_seen
                .into_iter()
                .enumerate()
                .filter_map(|(index, seen)| seen.then_some(index))
                .collect(),
            layers,
            parents,
        })
    }
}

fn count_labels(labels: &[Arc<str>]) -> Vec<(Arc<str>, usize)> {
    let mut counts: BTreeMap<Arc<str>, usize> = BTreeMap::new();
    for label in labels {
        *counts.entry(label.clone()).or_default() += 1;
    }
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    counts
}

fn evidence_path_positions(
    snapshot: &SceneSnapshot,
    path: &EvidencePath,
    overview_positions: &[Vec2],
) -> HashMap<NodeId, Vec2> {
    let anchors: Vec<_> = path
        .nodes
        .iter()
        .map(|node| {
            snapshot
                .node_index(node.id)
                .map(|index| overview_positions[index])
        })
        .collect();
    let start = path.nodes.first().map_or(0, |node| node.id);
    let end = path.nodes.last().map_or(start, |node| node.id);
    let angle = stable_unit(start ^ end.rotate_left(29)) * std::f32::consts::TAU;
    let fallback_direction = Vec2::new(angle.cos(), angle.sin());
    const STEP: f32 = 42.0;
    let center = snapshot.bounds.center();
    let middle = (path.nodes.len().saturating_sub(1)) as f32 * 0.5;
    let mut positions = HashMap::with_capacity(path.nodes.len());

    for (index, node) in path.nodes.iter().enumerate() {
        let position = if let Some(position) = anchors[index] {
            position
        } else {
            let previous = anchors[..index]
                .iter()
                .enumerate()
                .rev()
                .find_map(|(index, position)| position.map(|position| (index, position)));
            let next = anchors[index + 1..]
                .iter()
                .enumerate()
                .find_map(|(offset, position)| {
                    position.map(|position| (index + 1 + offset, position))
                });
            match (previous, next) {
                (Some((left_index, left)), Some((right_index, right))) => {
                    let mix = (index - left_index) as f32 / (right_index - left_index) as f32;
                    left * (1.0 - mix) + right * mix
                }
                (Some((left_index, left)), None) => {
                    let earlier = anchors[..left_index]
                        .iter()
                        .rev()
                        .find_map(|position| *position);
                    let direction = earlier.map_or(fallback_direction, |earlier| {
                        unit_or(left - earlier, fallback_direction)
                    });
                    left + direction * (STEP * (index - left_index) as f32)
                }
                (None, Some((right_index, right))) => {
                    let later = anchors[right_index + 1..]
                        .iter()
                        .find_map(|position| *position);
                    let direction = later.map_or(fallback_direction * -1.0, |later| {
                        unit_or(right - later, fallback_direction * -1.0)
                    });
                    right + direction * (STEP * (right_index - index) as f32)
                }
                (None, None) => center + Vec2::new((index as f32 - middle) * STEP, 0.0),
            }
        };
        positions.insert(node.id, position);
    }
    positions
}

fn unit_or(vector: Vec2, fallback: Vec2) -> Vec2 {
    let length = vector.length();
    if length > 0.001 {
        vector / length
    } else {
        fallback
    }
}

fn sort_scene_nodes(nodes: &mut SceneNodes) {
    let mut order: Vec<_> = (0..nodes.ids.len()).collect();
    order.sort_unstable_by_key(|&index| nodes.ids[index]);
    nodes.ids = reordered(&nodes.ids, &order);
    nodes.labels = reordered(&nodes.labels, &order);
    nodes.positions = reordered(&nodes.positions, &order);
    nodes.degrees = reordered(&nodes.degrees, &order);
    nodes.vector_counts = reordered(&nodes.vector_counts, &order);
    nodes.properties = reordered(&nodes.properties, &order);
}

fn sort_scene_edges(edges: &mut SceneEdges) {
    let mut order: Vec<_> = (0..edges.ids.len()).collect();
    order.sort_unstable_by_key(|&index| edges.ids[index]);
    edges.ids = reordered(&edges.ids, &order);
    edges.sources = reordered(&edges.sources, &order);
    edges.targets = reordered(&edges.targets, &order);
    edges.labels = reordered(&edges.labels, &order);
    edges.vector_counts = reordered(&edges.vector_counts, &order);
    edges.properties = reordered(&edges.properties, &order);
}

fn reordered<T: Clone>(values: &[T], order: &[usize]) -> Vec<T> {
    order.iter().map(|&index| values[index].clone()).collect()
}

fn run_force_layout(
    positions: &mut [Vec2],
    edges: &SceneEdges,
    options: LayoutOptions,
    pinned: Option<&[bool]>,
) {
    // The cell-centroid approximation below is excellent for small, direct
    // manipulation layouts, but its square neighborhoods become visible as a
    // lattice once thousands of nodes are present. Large general graphs use
    // sampled t-distribution forces instead: work and auxiliary memory remain
    // linear in edges, while repulsion has no screen-aligned spatial grid.
    //
    // This follows the objective introduced by SNAP-tFDP (Wang et al., IEEE
    // VIS 2026), with a deterministic RNG and portable scalar Rust. The hot
    // loop is deliberately contiguous and branch-light so it can later gain
    // platform SIMD or a GPU backend without changing layout semantics.
    const SAMPLED_T_FORCE_NODE_THRESHOLD: usize = 256;
    if positions.len() >= SAMPLED_T_FORCE_NODE_THRESHOLD && !edges.ids.is_empty() {
        run_sampled_t_force_layout(positions, edges, options, pinned);
    } else {
        run_spatial_force_layout(positions, edges, options, pinned);
    }
}

fn run_spatial_force_layout(
    positions: &mut [Vec2],
    edges: &SceneEdges,
    options: LayoutOptions,
    pinned: Option<&[bool]>,
) {
    let count = positions.len();
    let mut forces = vec![Vec2::ZERO; count];
    let cell_size = (options.edge_length * 1.8).max(1.0);
    let iterations = options.iterations.min(match count {
        0..=2_000 => options.iterations,
        2_001..=12_000 => 32,
        _ => 12,
    });

    for iteration in 0..iterations {
        forces.fill(Vec2::ZERO);
        let mut cells: HashMap<(i32, i32), (u32, Vec2)> = HashMap::new();
        for position in positions.iter().copied() {
            let key = cell_key(position, cell_size);
            let entry = cells.entry(key).or_insert((0, Vec2::ZERO));
            entry.0 += 1;
            entry.1 += position;
        }

        for (index, position) in positions.iter().copied().enumerate() {
            let own_cell = cell_key(position, cell_size);
            let mut force = position * -options.gravity;
            for cell_y in (own_cell.1 - 2)..=(own_cell.1 + 2) {
                for cell_x in (own_cell.0 - 2)..=(own_cell.0 + 2) {
                    let Some(&(cell_count, sum)) = cells.get(&(cell_x, cell_y)) else {
                        continue;
                    };
                    let center = sum / cell_count as f32;
                    let mut delta = position - center;
                    let mut distance_squared = delta.length_squared();
                    if distance_squared < 0.01 {
                        let angle =
                            stable_unit(index as u64 ^ iteration as u64) * std::f32::consts::TAU;
                        delta = Vec2::new(angle.cos(), angle.sin());
                        distance_squared = 1.0;
                    }
                    let magnitude = options.repulsion * cell_count as f32
                        / (distance_squared + options.edge_length);
                    force += delta * (magnitude / distance_squared.sqrt());
                }
            }
            forces[index] = force;
        }

        for (&source, &target) in edges.sources.iter().zip(&edges.targets) {
            let source = source as usize;
            let target = target as usize;
            let delta = positions[target] - positions[source];
            let distance = delta.length().max(0.01);
            let magnitude = (distance - options.edge_length) * options.attraction;
            let spring = delta * (magnitude / distance);
            forces[source] += spring;
            forces[target] += spring * -1.0;
        }

        let progress = iteration as f32 / iterations.max(1) as f32;
        let temperature = options.edge_length * (1.0 - progress).mul_add(0.22, 0.025);
        for (index, (position, force)) in positions.iter_mut().zip(&forces).enumerate() {
            if pinned.is_some_and(|pins| pins[index]) {
                continue;
            }
            let length = force.length();
            if length > 0.0 && length.is_finite() {
                *position += *force * (temperature.min(length) / length);
            }
        }
    }
}

fn run_sampled_t_force_layout(
    positions: &mut [Vec2],
    edges: &SceneEdges,
    options: LayoutOptions,
    pinned: Option<&[bool]>,
) {
    let node_count = positions.len();
    if node_count < 2 {
        return;
    }

    let original_frame = Rect::from_points(positions);
    let original_center = original_frame.center();
    let original_extent = original_frame.width().max(original_frame.height());
    if original_extent > 0.000_1 && original_extent.is_finite() {
        let scale = 20.0 / original_extent;
        for position in positions.iter_mut() {
            *position = (*position - original_center) * scale;
        }
    } else {
        // A deterministic phyllotaxis seed avoids coincident gradients when a
        // caller supplies a collapsed layout.
        const GOLDEN_ANGLE: f32 = 2.399_963_1;
        for (index, position) in positions.iter_mut().enumerate() {
            let radius = 0.12 * (index as f32 + 1.0).sqrt();
            let angle = index as f32 * GOLDEN_ANGLE + stable_unit(index as u64);
            *position = Vec2::new(angle.cos(), angle.sin()) * radius;
        }
    }

    // SNAP-tFDP treats the graph as undirected and stores both orientations.
    // Keeping that representation here also gives each endpoint equal chances
    // to anchor negative samples, regardless of the relationship direction.
    let mut directed_edges = Vec::with_capacity(edges.ids.len().saturating_mul(2));
    for (&source, &target) in edges.sources.iter().zip(&edges.targets) {
        let source = source as usize;
        let target = target as usize;
        if source < node_count && target < node_count && source != target {
            directed_edges.push((source, target));
            directed_edges.push((target, source));
        }
    }
    if directed_edges.is_empty() {
        if !pinned.is_some_and(|pins| pins.iter().any(|&pin| pin)) {
            scale_sampled_layout(positions, options.edge_length);
        }
        return;
    }

    let mut rng = LayoutRng::new(
        0x5647_534e_4150_5446_u64
            ^ node_count as u64
            ^ (directed_edges.len() as u64).rotate_left(29),
    );
    for index in (1..directed_edges.len()).rev() {
        let other = rng.index(index + 1);
        directed_edges.swap(index, other);
    }

    const NEGATIVE_SAMPLES: usize = 3;
    const MIN_STEP: f32 = 0.01;
    let epochs = options.iterations.clamp(1, 50);
    let cooling = 1.0 - 0.02_f32.powf(1.0 / epochs as f32);
    let mut step = 1.0_f32;
    for _ in 0..epochs {
        for &(source, target) in &directed_edges {
            let delta = positions[target] - positions[source];
            let distance_squared = delta.length_squared();
            let attraction = 0.1 + 0.8 / (1.0 + distance_squared);
            move_sampled_pair(
                positions,
                pinned,
                source,
                target,
                delta * (step * attraction),
            );

            for _ in 0..NEGATIVE_SAMPLES {
                // Sampling N - 1 and skipping over `source` removes a retry
                // branch and guarantees exactly K useful negatives.
                let sampled = rng.index(node_count - 1);
                let other = sampled + usize::from(sampled >= source);
                let delta = positions[other] - positions[source];
                let denominator = 1.0 + delta.length_squared();
                let repulsion = -1.0 / (denominator * denominator);
                move_sampled_pair(positions, pinned, source, other, delta * (step * repulsion));
            }
        }
        step += (MIN_STEP - step) * cooling;
    }

    if pinned.is_some_and(|pins| pins.iter().any(|&pin| pin)) {
        // Return to the caller's coordinate system so fixed nodes remain true
        // anchors. GraphWorkspace preserves them again during animated fit.
        if original_extent > 0.000_1 && original_extent.is_finite() {
            let inverse_scale = original_extent / 20.0;
            for position in positions.iter_mut() {
                *position = *position * inverse_scale + original_center;
            }
        }
    } else {
        scale_sampled_layout(positions, options.edge_length);
    }
}

fn move_sampled_pair(
    positions: &mut [Vec2],
    pinned: Option<&[bool]>,
    left: usize,
    right: usize,
    movement: Vec2,
) {
    let left_pinned = pinned.is_some_and(|pins| pins[left]);
    let right_pinned = pinned.is_some_and(|pins| pins[right]);
    match (left_pinned, right_pinned) {
        (false, false) => {
            positions[left] += movement;
            positions[right] += movement * -1.0;
        }
        (false, true) => positions[left] += movement * 2.0,
        (true, false) => positions[right] += movement * -2.0,
        (true, true) => {}
    }
}

fn scale_sampled_layout(positions: &mut [Vec2], edge_length: f32) {
    recenter(positions);
    let bounds = Rect::from_points(positions);
    let extent = bounds.width().max(bounds.height());
    if extent <= 0.000_1 || !extent.is_finite() {
        return;
    }
    // Area grows with node count, keeping density and picking tolerances
    // stable across graph sizes. Final Studio fitting still controls how much
    // of that world-space canvas is visible at once.
    let target_extent = edge_length.max(8.0) * (positions.len() as f32).sqrt();
    let scale = target_extent / extent;
    for position in positions {
        *position = *position * scale;
    }
}

struct LayoutRng {
    state: u64,
}

impl LayoutRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64 is small, reproducible on every target, and has ample
        // quality for edge shuffling and unbiased negative sampling.
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper_bound: usize) -> usize {
        debug_assert!(upper_bound > 0);
        ((u128::from(self.next_u64()) * upper_bound as u128) >> 64) as usize
    }
}

/// Lays out each connected component with Omega's low-rank resistance-distance
/// stress model, then packs the independent components without inventing edges
/// between them. This mode deliberately favors global topology and community
/// faithfulness over the latency of the default sampled-force arrangement.
fn run_resistance_structure_layout(
    positions: &mut [Vec2],
    edges: &SceneEdges,
    options: LayoutOptions,
) {
    let node_count = positions.len();
    if node_count < 3 || edges.sources.is_empty() {
        run_spatial_force_layout(positions, edges, options, None);
        recenter(positions);
        return;
    }

    let mut union = LayoutUnionFind::new(node_count);
    let mut unique_pairs = Vec::with_capacity(edges.sources.len());
    let mut pair_seen = HashSet::with_capacity(edges.sources.len());
    for (&source, &target) in edges.sources.iter().zip(&edges.targets) {
        let source = source as usize;
        let target = target as usize;
        if source >= node_count || target >= node_count || source == target {
            continue;
        }
        let pair = if source < target {
            (source, target)
        } else {
            (target, source)
        };
        if pair_seen.insert(pair) {
            union.join(pair.0, pair.1);
            unique_pairs.push(pair);
        }
    }
    if unique_pairs.is_empty() {
        layout_orbits(
            positions,
            &(0..node_count as NodeId).collect::<Vec<_>>(),
            &vec![Arc::from("Node"); node_count],
            &vec![0; node_count],
            options.edge_length,
        );
        return;
    }

    let mut component_nodes = BTreeMap::<usize, Vec<usize>>::new();
    for node in 0..node_count {
        component_nodes
            .entry(union.root(node))
            .or_default()
            .push(node);
    }
    let mut components: Vec<_> = component_nodes.into_values().collect();
    components.sort_unstable_by(|left, right| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| left[0].cmp(&right[0]))
    });

    let mut component_of = vec![usize::MAX; node_count];
    let mut local_index = vec![usize::MAX; node_count];
    for (component_index, nodes) in components.iter().enumerate() {
        for (local, &global) in nodes.iter().enumerate() {
            component_of[global] = component_index;
            local_index[global] = local;
        }
    }
    let mut component_edges = vec![Vec::new(); components.len()];
    for &(source, target) in &unique_pairs {
        let component = component_of[source];
        debug_assert_eq!(component, component_of[target]);
        component_edges[component].push((local_index[source], local_index[target]));
    }

    let mut drawings = Vec::with_capacity(components.len());
    for (component_index, nodes) in components.into_iter().enumerate() {
        let coordinates = resistance_component(
            nodes.len(),
            &component_edges[component_index],
            nodes[0] as u64,
            options,
        );
        let radius = coordinates
            .iter()
            .map(|position| position.length())
            .fold(0.0_f32, f32::max)
            .max(0.5);
        drawings.push(PackedComponent {
            nodes,
            coordinates,
            center: Vec2::ZERO,
            radius,
        });
    }
    pack_structure_components(&mut drawings, options.edge_length.max(8.0) * 0.12);

    for component in drawings {
        for (&node, coordinate) in component.nodes.iter().zip(component.coordinates) {
            positions[node] = coordinate + component.center;
        }
    }
    scale_sampled_layout(positions, options.edge_length);
}

fn resistance_component(
    node_count: usize,
    edges: &[(usize, usize)],
    seed: u64,
    options: LayoutOptions,
) -> Vec<Vec2> {
    match node_count {
        0 => return Vec::new(),
        1 => return vec![Vec2::ZERO],
        2 => return vec![Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.0)],
        _ => {}
    }

    let mut graph = Graph::<(), (), Undirected>::with_capacity(node_count, edges.len());
    for _ in 0..node_count {
        graph.add_node(());
    }
    for &(source, target) in edges {
        graph.add_edge(NodeIndex::new(source), NodeIndex::new(target), ());
    }

    // The paper's quality/performance operating point is rank 10, fifty
    // sampled pairs per vertex, epsilon_d=0.01, and fifteen SGD passes. Rank is
    // naturally capped for tiny connected components.
    let rank = 10.min(node_count - 1);
    let mut rng = StdRng::seed_from_u64(
        0x4f4d_4547_4152_4453_u64 ^ seed ^ (node_count as u64).rotate_left(17),
    );
    let embedding = RdMds::new()
        .d(rank)
        .shift(1e-6)
        .eigenvalue_max_iterations(96)
        .cg_max_iterations(64)
        .eigenvalue_tolerance(1e-5)
        .cg_tolerance(1e-5)
        .embedding(&graph, |_| 1.0_f32, &mut rng);

    let mut drawing = DrawingEuclidean2d::<NodeIndex, f32>::new(&graph);
    for index in 0..node_count {
        let node = NodeIndex::new(index);
        _ = drawing.set_x(node, embedding[[index, 0]]);
        _ = drawing.set_y(node, if rank > 1 { embedding[[index, 1]] } else { 0.0 });
    }

    let mut omega = Omega::new();
    omega.k(50).min_dist(0.01);
    let mut sgd = omega.build(&graph, &embedding, &mut rng);
    let iterations = (options.iterations / 4).clamp(8, 15);
    let mut scheduler = sgd.scheduler::<SchedulerExponential<f32>>(iterations, 0.1);
    scheduler.run(&mut |eta| {
        sgd.shuffle(&mut rng);
        sgd.apply(&mut drawing, eta);
    });

    let mut coordinates: Vec<_> = (0..node_count)
        .map(|index| {
            let node = NodeIndex::new(index);
            Vec2::new(
                drawing.x(node).unwrap_or(0.0),
                drawing.y(node).unwrap_or(0.0),
            )
        })
        .collect();
    if coordinates
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        // Numerical failure should never poison the Studio scene. A stable
        // phyllotaxis component remains inspectable and makes the fallback
        // visually explicit instead of returning a collapsed origin.
        const GOLDEN_ANGLE: f32 = 2.399_963_1;
        for (index, position) in coordinates.iter_mut().enumerate() {
            let radius = (index as f32 + 1.0).sqrt();
            let angle = index as f32 * GOLDEN_ANGLE + stable_unit(seed ^ index as u64);
            *position = Vec2::new(angle.cos(), angle.sin()) * radius;
        }
    }
    recenter(&mut coordinates);
    let extent = Rect::from_points(&coordinates)
        .width()
        .max(Rect::from_points(&coordinates).height())
        .max(0.000_1);
    let target_extent = (node_count as f32).sqrt().max(1.0);
    for position in &mut coordinates {
        *position = *position * (target_extent / extent);
    }

    coordinates
}

struct PackedComponent {
    nodes: Vec<usize>,
    coordinates: Vec<Vec2>,
    center: Vec2,
    radius: f32,
}

fn pack_structure_components(components: &mut [PackedComponent], gap: f32) {
    let Some((largest, rest)) = components.split_first_mut() else {
        return;
    };
    largest.center = Vec2::ZERO;
    if rest.is_empty() {
        return;
    }

    let mut ring_radius = largest.radius + rest[0].radius + gap;
    let mut ring_height = rest[0].radius;
    let mut cursor = 0.0_f32;
    let mut previous_radius = rest[0].radius;
    for component in rest {
        let chord = previous_radius + component.radius + gap;
        let angular_gap = 2.0 * (chord / (2.0 * ring_radius).max(chord)).asin();
        if cursor > 0.0 && cursor + angular_gap > std::f32::consts::TAU {
            ring_radius += ring_height + component.radius + gap;
            ring_height = component.radius;
            cursor = 0.0;
        }
        let angle = cursor + angular_gap * 0.5;
        component.center = Vec2::new(angle.cos(), angle.sin()) * ring_radius;
        cursor += angular_gap;
        ring_height = ring_height.max(component.radius);
        previous_radius = component.radius;
    }
}

struct LayoutUnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl LayoutUnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn root(&mut self, node: usize) -> usize {
        let parent = self.parent[node];
        if parent != node {
            self.parent[node] = self.root(parent);
        }
        self.parent[node]
    }

    fn join(&mut self, left: usize, right: usize) {
        let mut left_root = self.root(left);
        let mut right_root = self.root(right);
        if left_root == right_root {
            return;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] = self.rank[left_root].saturating_add(1);
        }
    }
}

#[derive(Debug)]
pub struct GraphWorkspace {
    positions: Vec<Vec2>,
    base_targets: Vec<Vec2>,
    targets: Vec<Vec2>,
    velocities: Vec<Vec2>,
    pinned: Vec<bool>,
    adjacency: Vec<Vec<u32>>,
    moving: bool,
}

impl GraphWorkspace {
    pub fn new(snapshot: &SceneSnapshot) -> Self {
        let mut adjacency = vec![Vec::new(); snapshot.nodes.ids.len()];
        for (&source, &target) in snapshot.edges.sources.iter().zip(&snapshot.edges.targets) {
            let source = source as usize;
            let target = target as usize;
            if source < adjacency.len() && target < adjacency.len() && source != target {
                adjacency[source].push(target as u32);
                adjacency[target].push(source as u32);
            }
        }
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let positions = snapshot.nodes.positions.clone();
        Self {
            base_targets: positions.clone(),
            targets: positions.clone(),
            velocities: vec![Vec2::ZERO; positions.len()],
            pinned: vec![false; positions.len()],
            adjacency,
            positions,
            moving: false,
        }
    }

    pub fn positions(&self) -> &[Vec2] {
        &self.positions
    }

    /// Stable overview targets, excluding temporary search/context layouts.
    pub fn overview_positions(&self) -> &[Vec2] {
        &self.base_targets
    }

    pub fn position(&self, index: usize) -> Option<Vec2> {
        self.positions.get(index).copied()
    }

    pub fn presentation_bounds(&self) -> Rect {
        Rect::from_points(&self.positions)
    }

    pub fn is_pinned(&self, index: usize) -> bool {
        self.pinned.get(index).copied().unwrap_or(false)
    }

    pub fn pinned_count(&self) -> usize {
        self.pinned.iter().filter(|&&pinned| pinned).count()
    }

    pub fn pins(&self) -> &[bool] {
        &self.pinned
    }

    pub fn is_moving(&self) -> bool {
        self.moving
    }

    pub fn begin_drag(&mut self, index: usize) -> bool {
        let was_pinned = self.is_pinned(index);
        if index < self.positions.len() {
            self.pinned[index] = true;
            self.targets[index] = self.positions[index];
            self.velocities[index] = Vec2::ZERO;
        }
        was_pinned
    }

    pub fn drag_to(&mut self, index: usize, position: Vec2, edge_length: f32) {
        if index >= self.positions.len() || !position.x.is_finite() || !position.y.is_finite() {
            return;
        }
        self.positions[index] = position;
        self.targets[index] = position;
        self.velocities[index] = Vec2::ZERO;

        // Direct manipulation remains O(degree), not O(graph size). A bounded
        // one-hop physical response keeps connected structure attached to the
        // pointer without making a hub node capable of blowing the frame.
        const NEIGHBOR_BUDGET: usize = 512;
        let desired_length = edge_length.max(8.0);
        for &neighbor in self.adjacency[index].iter().take(NEIGHBOR_BUDGET) {
            let neighbor = neighbor as usize;
            if self.pinned[neighbor] {
                continue;
            }
            let mut delta = self.positions[neighbor] - position;
            let distance = delta.length();
            if distance < 0.001 {
                let angle = stable_unit(neighbor as u64 ^ index as u64) * std::f32::consts::TAU;
                delta = Vec2::new(angle.cos(), angle.sin());
            }
            let direction = delta / delta.length().max(0.001);
            let desired = position + direction * desired_length;
            let influence = if distance > desired_length * 2.5 {
                0.34
            } else {
                0.18
            };
            let target = self.targets[neighbor] * (1.0 - influence) + desired * influence;
            self.targets[neighbor] = target;
            self.base_targets[neighbor] = target;
        }
        self.moving = true;
    }

    pub fn restore_pin(&mut self, index: usize, was_pinned: bool) {
        if index >= self.positions.len() {
            return;
        }
        self.pinned[index] = was_pinned;
        self.targets[index] = if was_pinned {
            self.positions[index]
        } else {
            self.base_targets[index]
        };
        self.moving |= !was_pinned && self.targets[index] != self.positions[index];
    }

    pub fn set_pinned(&mut self, index: usize, pinned: bool) {
        if index >= self.positions.len() {
            return;
        }
        self.pinned[index] = pinned;
        self.velocities[index] = Vec2::ZERO;
        self.targets[index] = if pinned {
            self.positions[index]
        } else {
            self.base_targets[index]
        };
        self.moving |= !pinned && self.targets[index] != self.positions[index];
    }

    pub fn retarget_layout(&mut self, mut targets: Vec<Vec2>, frame: Rect) -> bool {
        if targets.len() != self.positions.len()
            || targets
                .iter()
                .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            return false;
        }
        fit_positions_to_frame(&mut targets, frame);
        self.base_targets = targets;
        for index in 0..self.positions.len() {
            if self.pinned[index] {
                self.targets[index] = self.positions[index];
            } else {
                self.targets[index] = self.base_targets[index];
            }
        }
        self.moving = self.positions.iter().zip(&self.targets).enumerate().any(
            |(index, (&position, &target))| {
                !self.pinned[index] && (target - position).length_squared() > 0.000_4
            },
        );
        true
    }

    /// Retargets a bounded neighborhood without changing the active base
    /// arrangement. Switching focus preserves spring velocity; restoring the
    /// overview returns every unpinned node to its base target.
    pub fn retarget_focus(&mut self, focus: &SceneFocus, spacing: f32) -> Option<Rect> {
        if focus.roots.is_empty()
            || focus.nodes.is_empty()
            || focus
                .nodes
                .iter()
                .chain(&focus.roots)
                .any(|&index| index >= self.positions.len())
        {
            return None;
        }
        self.restore_targets();

        let anchor = focus
            .roots
            .iter()
            .map(|&index| self.positions[index])
            .fold(Vec2::ZERO, |sum, position| sum + position)
            / focus.roots.len() as f32;
        let spacing = spacing.max(18.0);
        for (slot, &root) in focus.roots.iter().enumerate() {
            if self.pinned[root] {
                continue;
            }
            self.targets[root] = if focus.roots.len() == 1 {
                anchor
            } else {
                let offset = (slot as f32 - (focus.roots.len() - 1) as f32 * 0.5) * spacing * 1.8;
                anchor + Vec2::new(offset, 0.0)
            };
        }

        let mut parent_of = vec![usize::MAX; self.positions.len()];
        for &(child, parent) in &focus.parents {
            if child < parent_of.len() && parent < parent_of.len() {
                parent_of[child] = parent;
            }
        }
        let mut node_angles = vec![f32::NAN; self.positions.len()];
        for (slot, &root) in focus.roots.iter().enumerate() {
            node_angles[root] = std::f32::consts::TAU * slot as f32 / focus.roots.len() as f32;
        }
        let mut previous_outer_radius = if focus.roots.len() > 1 { spacing } else { 0.0 };
        for (depth, layer) in focus.layers.iter().enumerate().skip(1) {
            let mut ordered = layer.clone();
            ordered.sort_unstable_by(|&left, &right| {
                let left_parent = parent_of[left];
                let right_parent = parent_of[right];
                let left_angle = node_angles.get(left_parent).copied().unwrap_or(f32::NAN);
                let right_angle = node_angles.get(right_parent).copied().unwrap_or(f32::NAN);
                left_angle
                    .total_cmp(&right_angle)
                    .then_with(|| left_parent.cmp(&right_parent))
                    .then_with(|| left.cmp(&right))
            });

            // Each semantic hop owns a separate radial band. Dense layers may
            // use several physical rings inside their band; this keeps nodes
            // pickable without making a high-degree first hop push the second
            // hop off-screen. Angle is global across the band, avoiding the
            // repeated spokes produced by restarting at zero for every ring.
            let mut radius = previous_outer_radius + spacing * 1.45;
            let mut rings = Vec::new();
            let mut total_capacity = 0_usize;
            while total_capacity < ordered.len() {
                let capacity =
                    ((std::f32::consts::TAU * radius / (spacing * 0.92)).floor() as usize).max(8);
                rings.push((radius, capacity));
                total_capacity += capacity;
                radius += spacing * 1.05;
            }
            let phase = if depth == 1 {
                stable_unit(
                    focus.roots[0] as u64
                        ^ (depth as u64).rotate_left(17)
                        ^ (ordered.len() as u64).rotate_left(31),
                ) * 0.28
            } else {
                // Rotate the child sequence to best align its contiguous
                // parent groups with their parents. This circular mean removes
                // avoidable parent-child crossings without an iterative solve.
                let mut direction = Vec2::ZERO;
                for (slot, &node) in ordered.iter().enumerate() {
                    let base = std::f32::consts::TAU * slot as f32 / ordered.len() as f32;
                    let parent_angle = node_angles[parent_of[node]];
                    let delta = parent_angle - base;
                    direction += Vec2::new(delta.cos(), delta.sin());
                }
                direction.y.atan2(direction.x)
            };
            let mut occupancy = vec![0_usize; rings.len()];
            for (slot, &node) in ordered.iter().enumerate() {
                let ring = (0..rings.len())
                    .filter(|&ring| occupancy[ring] < rings[ring].1)
                    .min_by(|&left, &right| {
                        let left_fill = occupancy[left] as f32 / rings[left].1 as f32;
                        let right_fill = occupancy[right] as f32 / rings[right].1 as f32;
                        left_fill.total_cmp(&right_fill).then(left.cmp(&right))
                    })
                    .unwrap_or(0);
                occupancy[ring] += 1;
                let angle = std::f32::consts::TAU * slot as f32 / ordered.len() as f32 + phase;
                node_angles[node] = angle.rem_euclid(std::f32::consts::TAU);
                if !self.pinned[node] {
                    self.targets[node] =
                        anchor + Vec2::new(angle.cos(), angle.sin()) * rings[ring].0;
                }
            }
            previous_outer_radius = rings.last().unwrap().0 + spacing * 0.45;
        }

        self.moving = self.positions.iter().zip(&self.targets).enumerate().any(
            |(index, (&position, &target))| {
                !self.pinned[index] && (target - position).length_squared() > 0.000_4
            },
        );
        let focus_targets: Vec<_> = focus
            .nodes
            .iter()
            .map(|&index| {
                if self.pinned[index] {
                    self.positions[index]
                } else {
                    self.targets[index]
                }
            })
            .collect();
        Some(Rect::from_points(&focus_targets))
    }

    pub fn restore_layout(&mut self) {
        self.restore_targets();
        self.moving = self.positions.iter().zip(&self.targets).enumerate().any(
            |(index, (&position, &target))| {
                !self.pinned[index] && (target - position).length_squared() > 0.000_4
            },
        );
    }

    fn restore_targets(&mut self) {
        for index in 0..self.positions.len() {
            self.targets[index] = if self.pinned[index] {
                self.positions[index]
            } else {
                self.base_targets[index]
            };
        }
    }

    pub fn step(&mut self, elapsed_seconds: f32, reduce_motion: bool) -> bool {
        if !self.moving {
            return false;
        }
        if reduce_motion {
            for index in 0..self.positions.len() {
                if !self.pinned[index] {
                    self.positions[index] = self.targets[index];
                }
                self.velocities[index] = Vec2::ZERO;
            }
            self.moving = false;
            return false;
        }

        let elapsed = elapsed_seconds.clamp(0.0, 0.05);
        if elapsed <= f32::EPSILON {
            return true;
        }
        const MAX_STEP: f32 = 1.0 / 120.0;
        const STIFFNESS: f32 = 175.0;
        const DAMPING: f32 = 27.0;
        let steps = (elapsed / MAX_STEP).ceil().max(1.0) as usize;
        let dt = elapsed / steps as f32;
        for _ in 0..steps {
            for index in 0..self.positions.len() {
                if self.pinned[index] {
                    continue;
                }
                let displacement = self.targets[index] - self.positions[index];
                let acceleration = displacement * STIFFNESS - self.velocities[index] * DAMPING;
                self.velocities[index] += acceleration * dt;
                self.positions[index] += self.velocities[index] * dt;
            }
        }

        let mut moving = false;
        for index in 0..self.positions.len() {
            if self.pinned[index] {
                continue;
            }
            let distance_squared = (self.targets[index] - self.positions[index]).length_squared();
            let speed_squared = self.velocities[index].length_squared();
            if distance_squared <= 0.000_4 && speed_squared <= 0.002_5 {
                self.positions[index] = self.targets[index];
                self.velocities[index] = Vec2::ZERO;
            } else if self.positions[index].x.is_finite()
                && self.positions[index].y.is_finite()
                && self.velocities[index].x.is_finite()
                && self.velocities[index].y.is_finite()
            {
                moving = true;
            } else {
                self.positions[index] = self.targets[index];
                self.velocities[index] = Vec2::ZERO;
            }
        }
        self.moving = moving;
        moving
    }
}

struct ForestTopology {
    children: Vec<Vec<u32>>,
    roots: Vec<u32>,
    weights: Vec<u64>,
    subtree_sizes: Vec<usize>,
}

fn directed_forest_topology(node_count: usize, edges: &SceneEdges) -> Option<ForestTopology> {
    if node_count == 0 || edges.ids.len() >= node_count {
        return None;
    }
    let mut children = vec![Vec::<u32>::new(); node_count];
    let mut indegree = vec![0_u8; node_count];
    for (&source, &target) in edges.sources.iter().zip(&edges.targets) {
        let source = source as usize;
        let target = target as usize;
        if source >= node_count || target >= node_count || source == target {
            return None;
        }
        indegree[target] = indegree[target].saturating_add(1);
        if indegree[target] > 1 {
            return None;
        }
        children[source].push(target as u32);
    }

    let roots: Vec<u32> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &degree)| (degree == 0).then_some(index as u32))
        .collect();
    if roots.is_empty() {
        return None;
    }

    // Produce a parent-before-child order while proving that every node is
    // reachable exactly once. That rejects cycles and disconnected cycles.
    let mut order = Vec::with_capacity(node_count);
    let mut visited = vec![false; node_count];
    let mut stack = roots.clone();
    while let Some(node) = stack.pop() {
        let index = node as usize;
        if std::mem::replace(&mut visited[index], true) {
            return None;
        }
        order.push(node);
        stack.extend(children[index].iter().copied());
    }
    if order.len() != node_count {
        return None;
    }

    // Leaves own one unit of angular space; internal nodes own the sum of
    // their descendants. The resulting dendrogram keeps branch identity at
    // overview scale without a quadratic force simulation.
    let mut weights = vec![1_u64; node_count];
    let mut subtree_sizes = vec![1_usize; node_count];
    for &node in order.iter().rev() {
        let index = node as usize;
        if !children[index].is_empty() {
            weights[index] = children[index]
                .iter()
                .fold(0_u64, |sum, &child| {
                    sum.saturating_add(weights[child as usize])
                })
                .max(1);
            subtree_sizes[index] += children[index]
                .iter()
                .map(|&child| subtree_sizes[child as usize])
                .sum::<usize>();
        }
    }
    Some(ForestTopology {
        children,
        roots,
        weights,
        subtree_sizes,
    })
}

fn try_layout_radial_forest(positions: &mut [Vec2], edges: &SceneEdges, edge_length: f32) -> bool {
    let Some(topology) = directed_forest_topology(positions.len(), edges) else {
        return false;
    };
    let total_weight = topology
        .roots
        .iter()
        .fold(0_u64, |sum, &root| {
            sum.saturating_add(topology.weights[root as usize])
        })
        .max(1) as f32;

    let radial_step = edge_length.max(12.0);
    let mut root_cursor = 0.0_f32;
    let mut assignments = Vec::with_capacity(positions.len());
    for &root in &topology.roots {
        let span = std::f32::consts::TAU * topology.weights[root as usize] as f32 / total_weight;
        assignments.push((root, root_cursor, root_cursor + span, 0_u32));
        root_cursor += span;
    }

    while let Some((node, start, end, depth)) = assignments.pop() {
        let index = node as usize;
        let angle = (start + end) * 0.5;
        let radius = if topology.roots.len() == 1 && depth == 0 {
            0.0
        } else {
            (depth + 1) as f32 * radial_step
        };
        positions[index] = Vec2::new(angle.cos(), angle.sin()) * radius;

        let child_total = topology.weights[index].max(1) as f32;
        let mut cursor = start;
        for &child in &topology.children[index] {
            let child_span = (end - start) * topology.weights[child as usize] as f32 / child_total;
            assignments.push((child, cursor, cursor + child_span, depth + 1));
            cursor += child_span;
        }
    }

    true
}

fn try_layout_clustered_forest(
    positions: &mut [Vec2],
    ids: &[NodeId],
    edges: &SceneEdges,
    edge_length: f32,
) -> bool {
    let Some(topology) = directed_forest_topology(positions.len(), edges) else {
        return false;
    };
    let (branches, central_root) = if topology.roots.len() == 1 {
        let root = topology.roots[0];
        let branches = topology.children[root as usize].clone();
        (branches, Some(root))
    } else {
        (topology.roots.clone(), None)
    };
    if !(2..=64).contains(&branches.len()) {
        return false;
    }

    // A large forest is more legible as a constellation of top-level
    // subtrees than as global depth rings. Each subtree gets a deterministic
    // topology-preserving phyllotaxis seed; the ordinary bounded force pass
    // then relaxes edges without erasing the top-level partition.
    const GOLDEN_ANGLE: f32 = 2.399_963_1;
    let local_spacing = (edge_length * 0.24).max(7.0);
    // Leave enough negative space for the force relaxation to expand each
    // disk without visually welding neighboring top-level branches together.
    let gap = edge_length.max(24.0) * 10.0;
    let mut ranked: Vec<_> = branches
        .iter()
        .copied()
        .map(|branch| (branch, topology.subtree_sizes[branch as usize]))
        .collect();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| ids[left.0 as usize].cmp(&ids[right.0 as usize]))
    });

    let count = ranked.len();
    let mut slots = vec![0_usize; count];
    for (rank, slot) in slots.iter_mut().enumerate() {
        *slot = if rank.is_multiple_of(2) {
            rank / 2
        } else {
            count.div_ceil(2) + rank / 2
        };
    }
    let mut radii_by_slot = vec![0.0_f32; count];
    for ((_, size), &slot) in ranked.iter().zip(&slots) {
        radii_by_slot[slot] = local_spacing * (*size as f32).sqrt();
    }
    let chord = 2.0 * (std::f32::consts::PI / count as f32).sin();
    let mut ring_radius = 0.0_f32;
    for slot in 0..count {
        let next = (slot + 1) % count;
        ring_radius =
            ring_radius.max((radii_by_slot[slot] + radii_by_slot[next] + gap) / chord.max(0.1));
    }
    ring_radius = ring_radius.max(radii_by_slot.iter().copied().fold(0.0_f32, f32::max) + gap);

    if let Some(root) = central_root {
        positions[root as usize] = Vec2::ZERO;
    }
    for ((branch, _), &slot) in ranked.iter().zip(&slots) {
        let angle = std::f32::consts::TAU * slot as f32 / count as f32;
        let center = Vec2::new(angle.cos(), angle.sin()) * ring_radius;
        let phase = stable_unit(ids[*branch as usize]) * std::f32::consts::TAU;
        let mut local_index = 0_usize;
        let mut stack = vec![*branch];
        while let Some(node) = stack.pop() {
            let radius = local_spacing * (local_index as f32).sqrt();
            let local_angle = local_index as f32 * GOLDEN_ANGLE + phase;
            positions[node as usize] =
                center + Vec2::new(local_angle.cos(), local_angle.sin()) * radius;
            local_index += 1;
            stack.extend(topology.children[node as usize].iter().rev().copied());
        }
    }
    true
}

fn symbol_or_unknown(read: &vectorgraph::ReadGuard<'_>, symbol: u32) -> Arc<str> {
    read.symbol(symbol).unwrap_or("Unknown").into()
}

fn owned_properties(
    read: &vectorgraph::ReadGuard<'_>,
    properties: &[vectorgraph::Property],
) -> Arc<[SceneProperty]> {
    properties
        .iter()
        .map(|property| SceneProperty {
            key: symbol_or_unknown(read, property.key),
            value: property.value.clone(),
        })
        .collect()
}

fn initialize_by_label(positions: &mut [Vec2], ids: &[NodeId], node_labels: &[Arc<str>]) {
    let mut labels = BTreeMap::<Arc<str>, usize>::new();
    for label in node_labels {
        let next = labels.len();
        labels.entry(label.clone()).or_insert(next);
    }
    let label_count = labels.len().max(1);
    let mut label_offsets = vec![0_usize; label_count];
    let cluster_radius = 180.0 + (ids.len() as f32).sqrt() * 4.0;
    const GOLDEN_ANGLE: f32 = 2.399_963_1;

    for index in 0..ids.len() {
        let group = labels[&node_labels[index]];
        let group_angle = std::f32::consts::TAU * group as f32 / label_count as f32;
        let center = Vec2::new(group_angle.cos(), group_angle.sin()) * cluster_radius;
        let local_index = label_offsets[group];
        label_offsets[group] += 1;
        let local_radius = 8.0 * (local_index as f32 + 1.0).sqrt();
        let local_angle = local_index as f32 * GOLDEN_ANGLE + stable_unit(ids[index]);
        positions[index] = center + Vec2::new(local_angle.cos(), local_angle.sin()) * local_radius;
    }
}

fn layout_orbits(
    positions: &mut [Vec2],
    ids: &[NodeId],
    labels: &[Arc<str>],
    degrees: &[u32],
    edge_length: f32,
) {
    let mut order: Vec<_> = (0..positions.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        degrees[right]
            .cmp(&degrees[left])
            .then_with(|| labels[left].cmp(&labels[right]))
            .then_with(|| ids[left].cmp(&ids[right]))
    });

    if let Some(&center) = order.first() {
        positions[center] = Vec2::ZERO;
    }
    let spacing = edge_length.max(16.0);
    let mut cursor = 1_usize;
    let mut ring = 1_usize;
    while cursor < order.len() {
        let radius = ring as f32 * spacing;
        let capacity = ((std::f32::consts::TAU * radius / spacing).floor() as usize).max(6);
        let count = capacity.min(order.len() - cursor);
        let phase = if ring.is_multiple_of(2) {
            std::f32::consts::PI / count as f32
        } else {
            0.0
        };
        for offset in 0..count {
            let angle = std::f32::consts::TAU * offset as f32 / count as f32 + phase;
            positions[order[cursor + offset]] = Vec2::new(angle.cos(), angle.sin()) * radius;
        }
        cursor += count;
        ring += 1;
    }
}

fn fit_positions_to_frame(positions: &mut [Vec2], frame: Rect) {
    if positions.is_empty() {
        return;
    }
    let source = Rect::from_points(positions);
    let scale = (frame.width() / source.width())
        .min(frame.height() / source.height())
        .max(0.000_1)
        * 0.94;
    let source_center = source.center();
    let frame_center = frame.center();
    for position in positions {
        *position = (*position - source_center) * scale + frame_center;
    }
}

fn stable_unit(mut value: u64) -> f32 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value as u32) as f32 / u32::MAX as f32
}

fn cell_key(position: Vec2, cell_size: f32) -> (i32, i32) {
    (
        (position.x / cell_size).floor() as i32,
        (position.y / cell_size).floor() as i32,
    )
}

fn recenter(positions: &mut [Vec2]) {
    if positions.is_empty() {
        return;
    }
    let center = positions
        .iter()
        .copied()
        .fold(Vec2::ZERO, |sum, position| sum + position)
        / positions.len() as f32;
    for position in positions {
        *position = *position - center;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub center: Vec2,
    pub zoom: f32,
}

pub const MIN_CAMERA_ZOOM: f32 = 0.08;
pub const MAX_CAMERA_ZOOM: f32 = 1_024.0;

impl Default for Camera {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

impl Camera {
    pub fn fit(bounds: Rect) -> Self {
        Self {
            center: bounds.center(),
            zoom: 1.0,
        }
    }

    /// Frames a subregion while retaining the full scene as the projection's
    /// scale reference. This lets search/focus navigation animate the camera
    /// without rebuilding world coordinates.
    pub fn framed(focus: Rect, world: Rect, viewport: Vec2, padding: f32) -> Self {
        let padding = padding.max(0.0);
        let available_width = (viewport.x - padding * 2.0).max(1.0);
        let available_height = (viewport.y - padding * 2.0).max(1.0);
        let desired_scale = (available_width / focus.width())
            .min(available_height / focus.height())
            .max(0.000_1);
        let base_scale = Self::fit(world).scale(world, viewport);
        Self {
            center: focus.center(),
            zoom: (desired_scale / base_scale).clamp(0.08, 24.0),
        }
    }

    pub fn scale(self, bounds: Rect, viewport: Vec2) -> f32 {
        let available_width = (viewport.x - 64.0).max(1.0);
        let available_height = (viewport.y - 64.0).max(1.0);
        let fitted = (available_width / bounds.width())
            .min(available_height / bounds.height())
            .max(0.000_1);
        fitted * self.zoom
    }

    pub fn project(self, point: Vec2, bounds: Rect, viewport: Vec2) -> Vec2 {
        let scale = self.scale(bounds, viewport);
        (point - self.center) * scale + viewport * 0.5
    }

    pub fn unproject(self, point: Vec2, bounds: Rect, viewport: Vec2) -> Vec2 {
        let scale = self.scale(bounds, viewport);
        (point - viewport * 0.5) / scale + self.center
    }

    pub fn pan_screen(&mut self, delta: Vec2, bounds: Rect, viewport: Vec2) {
        let scale = self.scale(bounds, viewport);
        self.center = self.center - delta / scale;
    }

    pub fn zoom_about(&mut self, screen_point: Vec2, factor: f32, bounds: Rect, viewport: Vec2) {
        let before = self.unproject(screen_point, bounds, viewport);
        self.zoom = (self.zoom * factor).clamp(MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM);
        let after = self.unproject(screen_point, bounds, viewport);
        self.center += before - after;
    }
}

/// Interruptible camera spring used by search and graph-focus navigation.
/// Zoom integrates in log space so magnification stays positive and feels
/// symmetric when entering or leaving a neighborhood.
#[derive(Clone, Copy, Debug)]
pub struct CameraMotion {
    target: Camera,
    center_velocity: Vec2,
    log_zoom_velocity: f32,
    moving: bool,
}

impl CameraMotion {
    pub fn new(camera: Camera) -> Self {
        Self {
            target: camera,
            center_velocity: Vec2::ZERO,
            log_zoom_velocity: 0.0,
            moving: false,
        }
    }

    pub fn retarget(&mut self, camera: Camera) {
        if !camera.center.x.is_finite()
            || !camera.center.y.is_finite()
            || !camera.zoom.is_finite()
            || camera.zoom <= 0.0
        {
            return;
        }
        self.target = camera;
        self.moving = true;
    }

    pub fn cancel_at(&mut self, camera: Camera) {
        self.target = camera;
        self.center_velocity = Vec2::ZERO;
        self.log_zoom_velocity = 0.0;
        self.moving = false;
    }

    pub const fn is_moving(&self) -> bool {
        self.moving
    }

    pub fn step(&mut self, camera: &mut Camera, elapsed_seconds: f32, reduce_motion: bool) -> bool {
        if !self.moving {
            return false;
        }
        if reduce_motion {
            *camera = self.target;
            self.cancel_at(*camera);
            return false;
        }
        let elapsed = elapsed_seconds.clamp(0.0, 0.05);
        if elapsed <= f32::EPSILON {
            return true;
        }
        const MAX_STEP: f32 = 1.0 / 120.0;
        const STIFFNESS: f32 = 145.0;
        const DAMPING: f32 = 24.0;
        let steps = (elapsed / MAX_STEP).ceil().max(1.0) as usize;
        let dt = elapsed / steps as f32;
        let target_log_zoom = self.target.zoom.max(0.001).ln();
        let mut log_zoom = camera.zoom.max(0.001).ln();
        for _ in 0..steps {
            let center_displacement = self.target.center - camera.center;
            let center_acceleration =
                center_displacement * STIFFNESS - self.center_velocity * DAMPING;
            self.center_velocity += center_acceleration * dt;
            camera.center += self.center_velocity * dt;

            let zoom_displacement = target_log_zoom - log_zoom;
            let zoom_acceleration =
                zoom_displacement * STIFFNESS - self.log_zoom_velocity * DAMPING;
            self.log_zoom_velocity += zoom_acceleration * dt;
            log_zoom += self.log_zoom_velocity * dt;
        }
        camera.zoom = log_zoom.exp().clamp(MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM);

        let center_distance = (self.target.center - camera.center).length_squared();
        let zoom_distance = (target_log_zoom - log_zoom).abs();
        if center_distance <= 0.000_4
            && self.center_velocity.length_squared() <= 0.002_5
            && zoom_distance <= 0.000_5
            && self.log_zoom_velocity.abs() <= 0.002
        {
            *camera = self.target;
            self.cancel_at(*camera);
            false
        } else if camera.center.x.is_finite()
            && camera.center.y.is_finite()
            && camera.zoom.is_finite()
        {
            true
        } else {
            *camera = self.target;
            self.cancel_at(*camera);
            false
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetailLevel {
    Overview,
    Communities,
    Elements,
}

pub fn detail_level(camera: Camera, node_count: usize) -> DetailLevel {
    if camera.zoom < 0.45 || (node_count >= 12_000 && camera.zoom < 1.6) {
        DetailLevel::Overview
    } else if camera.zoom < 2.8
        || (node_count > 25_000 && camera.zoom < 24.0)
        || (node_count > 12_000 && camera.zoom < 12.0)
    {
        DetailLevel::Communities
    } else {
        DetailLevel::Elements
    }
}

pub fn hit_test_node(
    snapshot: &SceneSnapshot,
    camera: Camera,
    viewport: Vec2,
    screen_point: Vec2,
    radius: f32,
) -> Option<usize> {
    hit_test_positions(
        &snapshot.nodes.positions,
        camera,
        snapshot.bounds,
        viewport,
        screen_point,
        radius,
    )
}

pub fn hit_test_positions(
    positions: &[Vec2],
    camera: Camera,
    world_bounds: Rect,
    viewport: Vec2,
    screen_point: Vec2,
    radius: f32,
) -> Option<usize> {
    let radius_squared = radius * radius;
    let scale = camera.scale(world_bounds, viewport);
    let project = |position: Vec2| (position - camera.center) * scale + viewport * 0.5;
    positions
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, position)| {
            let delta = project(position) - screen_point;
            let distance = delta.length_squared();
            (distance <= radius_squared).then_some((index, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
}

pub fn hit_test_edges(
    positions: &[Vec2],
    edges: &SceneEdges,
    camera: Camera,
    world_bounds: Rect,
    viewport: Vec2,
    screen_point: Vec2,
    radius: f32,
) -> Option<usize> {
    let radius_squared = radius * radius;
    let scale = camera.scale(world_bounds, viewport);
    let project = |position: Vec2| (position - camera.center) * scale + viewport * 0.5;
    edges
        .sources
        .iter()
        .zip(&edges.targets)
        .enumerate()
        .filter_map(|(index, (&source, &target))| {
            let from = project(*positions.get(source as usize)?);
            let to = project(*positions.get(target as usize)?);
            if screen_point.x < from.x.min(to.x) - radius
                || screen_point.x > from.x.max(to.x) + radius
                || screen_point.y < from.y.min(to.y) - radius
                || screen_point.y > from.y.max(to.y) + radius
            {
                return None;
            }
            let distance = point_segment_distance_squared(screen_point, from, to);
            (distance <= radius_squared).then_some((index, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
}

fn point_segment_distance_squared(point: Vec2, from: Vec2, to: Vec2) -> f32 {
    let segment = to - from;
    let length_squared = segment.length_squared();
    if length_squared <= f32::EPSILON {
        return (point - from).length_squared();
    }
    let from_to_point = point - from;
    let projection = from_to_point
        .x
        .mul_add(segment.x, from_to_point.y * segment.y)
        / length_squared;
    let nearest = from + segment * projection.clamp(0.0, 1.0);
    (point - nearest).length_squared()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vectorgraph::{DatabaseOptions, Similarity, VectorEncoding};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    struct TestFile(std::path::PathBuf);

    impl TestFile {
        fn new() -> Self {
            let serial = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "vectorgraph-studio-{}-{serial}.vg",
                std::process::id()
            )))
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn snapshot_reads_real_database_and_preserves_symbols() {
        let file = TestFile::new();
        let database = Database::create(
            &file.0,
            DatabaseOptions {
                vector_dimension: 4,
                similarity: Similarity::Cosine,
                vector_encoding: VectorEncoding::F32,
                sync_on_commit: false,
            },
        )
        .unwrap();
        let mut transaction = database.transaction();
        let first = transaction.create_node(
            "Document",
            [("title", Value::String("The first".into()))],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        );
        let second = transaction.create_node(
            "Claim",
            [("confidence", Value::Float(0.92))],
            &[vec![0.0, 1.0, 0.0, 0.0]],
        );
        transaction.create_edge(
            first,
            second,
            "SUPPORTS",
            std::iter::empty::<(&str, Value)>(),
            &[vec![0.0, 0.0, 1.0, 0.0]],
        );
        transaction.commit().unwrap();

        let snapshot = SceneSnapshot::from_database(&database, SnapshotOptions::default()).unwrap();
        assert_eq!(snapshot.nodes.ids, vec![first, second]);
        assert_eq!(&*snapshot.nodes.labels[0], "Document");
        assert_eq!(&*snapshot.edges.labels[0], "SUPPORTS");
        assert_eq!(snapshot.nodes.vector_counts, vec![1, 1]);
        assert_eq!(snapshot.edges.sources, vec![0]);
        assert_eq!(snapshot.edges.targets, vec![1]);
        assert_eq!(
            snapshot.relationship_counts(),
            vec![(Arc::from("SUPPORTS"), 1)]
        );
    }

    #[test]
    fn evidence_paths_preserve_direction_hydration_and_incomplete_outcomes() {
        let file = TestFile::new();
        let database = Database::create(
            &file.0,
            DatabaseOptions {
                vector_dimension: 4,
                similarity: Similarity::Cosine,
                vector_encoding: VectorEncoding::F32,
                sync_on_commit: false,
            },
        )
        .unwrap();
        let mut transaction = database.transaction();
        let alpha = transaction.create_node(
            "Document",
            [("title", Value::String("Alpha brief".into()))],
            &[],
        );
        let beta =
            transaction.create_node("Claim", [("name", Value::String("Beta claim".into()))], &[]);
        let gamma = transaction.create_node(
            "Source",
            [("title", Value::String("Gamma source".into()))],
            &[],
        );
        let first = transaction.create_edge(
            alpha,
            beta,
            "SUPPORTS",
            [("title", Value::String("Primary support".into()))],
            &[],
        );
        let second = transaction.create_edge(
            beta,
            gamma,
            "SUPPORTS",
            std::iter::empty::<(&str, Value)>(),
            &[],
        );
        transaction.commit().unwrap();

        let outgoing = evidence_path_database(
            &file.0,
            alpha,
            gamma,
            Direction::Outgoing,
            Some("SUPPORTS"),
            4,
            32,
        )
        .unwrap();
        assert_eq!(outgoing.termination, ShortestPathTermination::Found);
        assert_eq!(
            outgoing.strategy,
            ShortestPathStrategy::BidirectionalBreadthFirst
        );
        assert_eq!(
            outgoing.start_expanded_nodes + outgoing.end_expanded_nodes,
            outgoing.expanded_nodes
        );
        assert_eq!(outgoing.start.title.as_ref(), "Alpha brief");
        let path = outgoing.path.as_ref().unwrap();
        assert_eq!(
            path.nodes.iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![alpha, beta, gamma]
        );
        assert_eq!(
            path.steps
                .iter()
                .map(|step| step.edge_id)
                .collect::<Vec<_>>(),
            vec![first, second]
        );
        assert_eq!(path.steps[0].title.as_ref(), "Primary support");
        assert!(path.steps.iter().all(|step| step.forward));
        assert_eq!(path.nodes[0].vector_count, 0);
        assert_eq!(path.steps[0].properties.len(), 1);

        let mut sampled = SceneSnapshot::from_database(
            &database,
            SnapshotOptions {
                max_nodes: 2,
                max_edges: 1,
            },
        )
        .unwrap();
        sampled.layout(LayoutOptions::default());
        assert!(sampled.node_index(gamma).is_none());
        assert!(sampled.edge_index(second).is_none());
        let overview_positions = sampled.nodes.positions.clone();
        let enriched = sampled
            .including_evidence_path(&outgoing, &overview_positions)
            .unwrap()
            .unwrap();
        assert_eq!(enriched.nodes.ids, vec![alpha, beta, gamma]);
        assert_eq!(enriched.edges.ids, vec![first, second]);
        assert_eq!(
            enriched.nodes.positions[..overview_positions.len()],
            overview_positions
        );
        let gamma_index = enriched.node_index(gamma).unwrap();
        assert_eq!(enriched.nodes.labels[gamma_index].as_ref(), "Source");
        assert_eq!(
            enriched.nodes.properties[gamma_index][0].value,
            Value::String("Gamma source".into())
        );
        let second_index = enriched.edge_index(second).unwrap();
        assert_eq!(
            enriched.nodes.ids[enriched.edges.sources[second_index] as usize],
            beta
        );
        assert_eq!(
            enriched.nodes.ids[enriched.edges.targets[second_index] as usize],
            gamma
        );
        assert!(!enriched.sampled);
        assert!(
            enriched
                .including_evidence_path(&outgoing, &enriched.nodes.positions)
                .unwrap()
                .is_none()
        );

        let incoming =
            evidence_path_database(&file.0, gamma, alpha, Direction::Incoming, None, 4, 32)
                .unwrap();
        assert_eq!(incoming.termination, ShortestPathTermination::Found);
        assert!(
            incoming
                .path
                .unwrap()
                .steps
                .iter()
                .all(|step| !step.forward)
        );

        let limited =
            evidence_path_database(&file.0, alpha, gamma, Direction::Outgoing, None, 4, 0).unwrap();
        assert_eq!(limited.termination, ShortestPathTermination::ExpansionLimit);
        assert!(limited.path.is_none());
        assert_eq!(limited.start_expanded_nodes, 0);
        assert_eq!(limited.end_expanded_nodes, 0);
        assert_eq!(limited.expanded_nodes, 0);

        let error = evidence_path_database(
            &file.0,
            alpha,
            gamma,
            Direction::Outgoing,
            Some("MISSING"),
            4,
            32,
        )
        .unwrap_err();
        assert!(error.contains("relationship label “MISSING” does not exist"));
    }

    #[test]
    fn hybrid_search_ranks_native_node_and_edge_vectors() {
        let file = TestFile::new();
        let dimension = 64;
        let database = Database::create(
            &file.0,
            DatabaseOptions {
                vector_dimension: dimension,
                similarity: Similarity::Cosine,
                vector_encoding: VectorEncoding::F32,
                sync_on_commit: false,
            },
        )
        .unwrap();
        let mut transaction = database.transaction();
        let rust = transaction.create_node(
            "Document",
            [("title", Value::String("Rust memory safety".into()))],
            &[vectorgraph_embedding::feature_vector(
                "ownership makes systems programming memory safe",
                dimension,
            )],
        );
        let fruit = transaction.create_node(
            "Document",
            [("title", Value::String("Banana bread recipe".into()))],
            &[vectorgraph_embedding::feature_vector(
                "fruit flour baking recipe",
                dimension,
            )],
        );
        let relationship = transaction.create_edge(
            rust,
            fruit,
            "CONTRASTS_WITH",
            [(
                "body",
                Value::String("ownership prevents use after free".into()),
            )],
            &[vectorgraph_embedding::feature_vector(
                "ownership prevents memory bugs",
                dimension,
            )],
        );
        transaction.commit().unwrap();

        let report = search_database(
            &file.0,
            "ownership memory safety",
            SearchMode::Hybrid,
            "hash",
            8,
        )
        .unwrap();
        assert!(!report.hits.is_empty());
        assert!(
            report
                .hits
                .iter()
                .any(|hit| hit.element == ElementRef::Node(rust))
        );
        assert!(
            report
                .hits
                .iter()
                .any(|hit| hit.element == ElementRef::Edge(relationship))
        );
        assert!(report.hits[0].semantic_score.is_some());
        assert!(report.hits.iter().any(|hit| hit.lexical_score.is_some()));
    }

    #[test]
    fn layout_is_deterministic_and_finite() {
        let mut first = SceneSnapshot::demo();
        let mut second = SceneSnapshot::demo();
        first.layout(LayoutOptions::default());
        second.layout(LayoutOptions::default());
        assert_eq!(first.nodes.positions, second.nodes.positions);
        assert!(
            first
                .nodes
                .positions
                .iter()
                .all(|point| point.x.is_finite() && point.y.is_finite())
        );
    }

    #[test]
    fn explicit_layouts_are_deterministic_and_finite() {
        let snapshot = SceneSnapshot::demo();
        for kind in [LayoutKind::Force, LayoutKind::Structure, LayoutKind::Orbit] {
            let first = snapshot.arranged_positions(kind, LayoutOptions::default(), None, None);
            let second = snapshot.arranged_positions(kind, LayoutOptions::default(), None, None);
            assert_eq!(first, second, "{kind:?} must be deterministic");
            assert!(
                first
                    .iter()
                    .all(|point| point.x.is_finite() && point.y.is_finite())
            );
        }
    }

    #[test]
    fn sampled_t_force_is_deterministic_finite_and_not_axis_biased() {
        const NODE_COUNT: usize = 512;
        let mut edges = SceneEdges::default();
        for source in 0..NODE_COUNT {
            for target in [(source + 1) % NODE_COUNT, (source * 37 + 101) % NODE_COUNT] {
                if source == target {
                    continue;
                }
                edges.ids.push(edges.ids.len() as EdgeId);
                edges.sources.push(source as u32);
                edges.targets.push(target as u32);
            }
        }
        let seed: Vec<_> = (0..NODE_COUNT)
            .map(|index| {
                let radius = 8.0 * (index as f32 + 1.0).sqrt();
                let angle = index as f32 * 2.399_963_1 + stable_unit(index as u64);
                Vec2::new(angle.cos(), angle.sin()) * radius
            })
            .collect();
        let mut first = seed.clone();
        let mut second = seed;
        let options = LayoutOptions {
            iterations: 32,
            ..LayoutOptions::default()
        };

        run_force_layout(&mut first, &edges, options, None);
        run_force_layout(&mut second, &edges, options, None);

        assert_eq!(first, second);
        assert!(
            first
                .iter()
                .all(|point| point.x.is_finite() && point.y.is_finite())
        );
        assert!(Rect::from_points(&first).width() > options.edge_length * 10.0);

        // The previous large-graph approximation accumulated nodes into a
        // Cartesian cell lattice. Nearest-neighbor bearings catch that visual
        // failure even when graph edges themselves point in arbitrary ways.
        let mut axis_aligned = 0_usize;
        for (index, &position) in first.iter().enumerate() {
            let nearest = first
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, &other)| other - position)
                .min_by(|left, right| left.length_squared().total_cmp(&right.length_squared()))
                .unwrap();
            let axis_ratio = nearest.x.abs().min(nearest.y.abs()) / nearest.length().max(0.000_1);
            axis_aligned += usize::from(axis_ratio < 3_f32.to_radians().sin());
        }
        assert!(
            axis_aligned * 5 < NODE_COUNT,
            "nearest-neighbor axes reveal a lattice: {axis_aligned}/{NODE_COUNT}"
        );
    }

    #[test]
    fn workspace_drag_is_direct_pinned_and_degree_bounded() {
        let snapshot = SceneSnapshot::demo();
        let mut workspace = GraphWorkspace::new(&snapshot);
        let untouched = workspace.position(50).unwrap();
        let destination = Vec2::new(420.0, -180.0);

        assert!(!workspace.begin_drag(0));
        workspace.drag_to(0, destination, LayoutOptions::default().edge_length);

        assert_eq!(workspace.position(0), Some(destination));
        assert!(workspace.is_pinned(0));
        assert_eq!(workspace.position(50), Some(untouched));
        assert!(workspace.is_moving());
    }

    #[test]
    fn focus_neighborhood_retargets_and_restores_the_base_layout() {
        let snapshot = SceneSnapshot::demo();
        let selection = SceneSelection::Node(7);
        let focus = snapshot.focus_neighborhood(selection, 18).unwrap();
        assert_eq!(focus.roots, vec![7]);
        assert!(focus.edges.len() <= 18);
        assert!(focus.nodes.contains(&7));

        let mut workspace = GraphWorkspace::new(&snapshot);
        let original = workspace.positions().to_vec();
        let focus_bounds = workspace.retarget_focus(&focus, 46.0).unwrap();
        assert!(focus_bounds.width().is_finite());
        workspace.step(1.0 / 60.0, true);
        assert_ne!(workspace.positions(), original);

        workspace.restore_layout();
        workspace.step(1.0 / 60.0, true);
        assert_eq!(workspace.positions(), original);
    }

    #[test]
    fn layered_focus_reserves_two_hop_context_and_relationship_diversity() {
        let node_count = 9;
        let edges = SceneEdges {
            ids: (0..8).collect(),
            sources: vec![0, 0, 0, 0, 1, 2, 3, 4],
            targets: vec![1, 2, 3, 4, 5, 6, 7, 8],
            labels: vec![
                "COMMON".into(),
                "COMMON".into(),
                "COMMON".into(),
                "RARE".into(),
                "CHILD".into(),
                "CHILD".into(),
                "CHILD".into(),
                "CHILD".into(),
            ],
            vector_counts: vec![0; 8],
            properties: vec![Arc::from([]); 8],
        };
        let snapshot = SceneSnapshot {
            revision: 1,
            database_name: "focus-test".into(),
            source_stats: GraphStats::default(),
            nodes: SceneNodes {
                ids: (0..node_count as u64).collect(),
                labels: vec!["Node".into(); node_count],
                positions: vec![Vec2::ZERO; node_count],
                degrees: vec![4, 2, 2, 2, 2, 1, 1, 1, 1],
                vector_counts: vec![0; node_count],
                properties: vec![Arc::from([]); node_count],
            },
            edges,
            bounds: Rect::default(),
            sampled: false,
        };

        let focus = snapshot
            .focus_neighborhood_layers(SceneSelection::Node(0), 2, 7, 10)
            .unwrap();
        assert_eq!(focus.layers.len(), 3);
        assert!(focus.nodes.len() <= 7);
        assert!(focus.edges.len() <= 10);
        assert!(focus.layers[1].contains(&4), "rare edge type must survive");
        assert!(
            focus.layers[1].iter().any(|node| (1..=3).contains(node)),
            "common edge type must survive"
        );
        assert!(!focus.layers[2].is_empty(), "capacity must reach hop two");
        assert!(focus.parents.iter().all(|(child, parent)| {
            focus.nodes.contains(child) && focus.nodes.contains(parent)
        }));
    }

    #[test]
    fn workspace_motion_converges_and_reduced_motion_snaps() {
        let snapshot = SceneSnapshot::demo();
        let mut workspace = GraphWorkspace::new(&snapshot);
        let mut targets = snapshot.nodes.positions.clone();
        targets[5] += Vec2::new(100.0, -80.0);
        assert!(workspace.retarget_layout(targets, snapshot.bounds));

        for _ in 0..600 {
            if !workspace.step(1.0 / 120.0, false) {
                break;
            }
        }
        assert!(!workspace.is_moving());
        assert!(
            workspace
                .positions()
                .iter()
                .all(|point| point.x.is_finite())
        );

        let mut targets = snapshot.nodes.positions.clone();
        targets[7] += Vec2::new(-140.0, 95.0);
        assert!(workspace.retarget_layout(targets, snapshot.bounds));
        assert!(!workspace.step(1.0 / 120.0, true));
        assert!(!workspace.is_moving());
    }

    #[test]
    fn workspace_retarget_preserves_velocity_and_survives_long_frames() {
        let snapshot = SceneSnapshot::demo();
        let mut workspace = GraphWorkspace::new(&snapshot);
        let mut first_targets = snapshot.nodes.positions.clone();
        first_targets[9] += Vec2::new(160.0, 20.0);
        assert!(workspace.retarget_layout(first_targets, snapshot.bounds));
        workspace.step(1.0 / 60.0, false);
        let velocity = workspace.velocities[9];
        assert!(velocity.length_squared() > 0.0);

        let mut second_targets = snapshot.nodes.positions.clone();
        second_targets[9] += Vec2::new(-180.0, -30.0);
        assert!(workspace.retarget_layout(second_targets, snapshot.bounds));
        assert_eq!(workspace.velocities[9], velocity);
        workspace.step(8.0, false);
        assert!(
            workspace
                .positions()
                .iter()
                .all(|point| point.x.is_finite())
        );
        assert!(workspace.velocities.iter().all(|point| point.y.is_finite()));
    }

    #[test]
    fn radial_forest_layout_preserves_depth_in_linear_pass() {
        let mut positions = vec![Vec2::ZERO; 4];
        let edges = SceneEdges {
            ids: vec![0, 1, 2],
            sources: vec![0, 0, 1],
            targets: vec![1, 2, 3],
            ..SceneEdges::default()
        };

        assert!(try_layout_radial_forest(&mut positions, &edges, 40.0));
        assert_eq!(positions[0], Vec2::ZERO);
        assert!((positions[1].length() - 80.0).abs() < 0.01);
        assert!((positions[2].length() - 80.0).abs() < 0.01);
        assert!((positions[3].length() - 120.0).abs() < 0.01);
        assert_ne!(positions[1], positions[2]);
    }

    #[test]
    fn clustered_forest_layout_separates_top_level_subtrees() {
        let mut positions = vec![Vec2::ZERO; 13];
        let ids: Vec<NodeId> = (0..13).collect();
        let edges = SceneEdges {
            ids: (0..12).collect(),
            sources: vec![0, 0, 0, 1, 1, 1, 5, 5, 5, 9, 9, 9],
            targets: vec![1, 5, 9, 2, 3, 4, 6, 7, 8, 10, 11, 12],
            ..SceneEdges::default()
        };

        assert!(try_layout_clustered_forest(
            &mut positions,
            &ids,
            &edges,
            40.0
        ));
        assert_eq!(positions[0], Vec2::ZERO);
        for (left, right) in [(1, 5), (5, 9), (9, 1)] {
            assert!((positions[left] - positions[right]).length() > 100.0);
        }
        assert!(
            positions
                .iter()
                .all(|point| point.x.is_finite() && point.y.is_finite())
        );
    }

    #[test]
    fn zoom_anchor_stays_under_pointer() {
        let bounds = Rect {
            min: Vec2::new(-100.0, -50.0),
            max: Vec2::new(100.0, 50.0),
        };
        let viewport = Vec2::new(1000.0, 700.0);
        let pointer = Vec2::new(700.0, 220.0);
        let mut camera = Camera::fit(bounds);
        let before = camera.unproject(pointer, bounds, viewport);
        camera.zoom_about(pointer, 1.8, bounds, viewport);
        let after = camera.unproject(pointer, bounds, viewport);
        assert!((before.x - after.x).abs() < 0.001);
        assert!((before.y - after.y).abs() < 0.001);
    }

    #[test]
    fn deep_zoom_reaches_individual_elements_in_large_scenes() {
        let mut camera = Camera::default();
        camera.zoom_about(
            Vec2::new(400.0, 300.0),
            1_000_000.0,
            Rect {
                min: Vec2::new(-500.0, -500.0),
                max: Vec2::new(500.0, 500.0),
            },
            Vec2::new(800.0, 600.0),
        );
        assert_eq!(camera.zoom, MAX_CAMERA_ZOOM);
        assert_eq!(detail_level(camera, 50_000), DetailLevel::Elements);
        assert_eq!(
            detail_level(
                Camera {
                    zoom: 4.0,
                    ..camera
                },
                50_000
            ),
            DetailLevel::Communities
        );
    }

    #[test]
    fn framed_camera_and_interruptible_motion_converge() {
        let world = Rect {
            min: Vec2::new(-500.0, -400.0),
            max: Vec2::new(500.0, 400.0),
        };
        let focus = Rect {
            min: Vec2::new(100.0, 80.0),
            max: Vec2::new(220.0, 180.0),
        };
        let viewport = Vec2::new(1_000.0, 700.0);
        let target = Camera::framed(focus, world, viewport, 100.0);
        assert_eq!(target.center, focus.center());
        assert!(target.zoom > 1.0);

        let mut camera = Camera::fit(world);
        let mut motion = CameraMotion::new(camera);
        motion.retarget(target);
        for _ in 0..240 {
            if !motion.step(&mut camera, 1.0 / 120.0, false) {
                break;
            }
        }
        assert!(!motion.is_moving());
        assert_eq!(camera, target);

        let next = Camera {
            center: Vec2::new(-140.0, 40.0),
            zoom: 2.4,
        };
        motion.retarget(next);
        motion.step(&mut camera, 1.0 / 120.0, false);
        let interrupted = camera;
        motion.retarget(target);
        motion.step(&mut camera, 1.0 / 120.0, false);
        assert_ne!(camera, target, "retargeting must not snap");
        assert_ne!(camera, interrupted, "retargeting must keep moving");
        motion.step(&mut camera, 1.0 / 120.0, true);
        assert_eq!(camera, target);
    }

    #[test]
    fn edge_hit_testing_uses_screen_space_and_nearest_segment() {
        let positions = [
            Vec2::new(-100.0, -20.0),
            Vec2::new(100.0, -20.0),
            Vec2::new(-100.0, 20.0),
            Vec2::new(100.0, 20.0),
        ];
        let edges = SceneEdges {
            ids: vec![10, 11],
            sources: vec![0, 2],
            targets: vec![1, 3],
            labels: vec!["ABOVE".into(), "BELOW".into()],
            vector_counts: vec![0, 0],
            properties: vec![Arc::from([]), Arc::from([])],
        };
        let bounds = Rect::from_points(&positions);
        let viewport = Vec2::new(800.0, 400.0);
        let camera = Camera::fit(bounds);
        let on_second = camera.project(Vec2::new(0.0, 20.0), bounds, viewport);

        assert_eq!(
            hit_test_edges(&positions, &edges, camera, bounds, viewport, on_second, 6.0),
            Some(1)
        );
        assert_eq!(
            hit_test_edges(
                &positions,
                &edges,
                camera,
                bounds,
                viewport,
                on_second + Vec2::new(0.0, 9.0),
                6.0,
            ),
            None
        );
    }
}
