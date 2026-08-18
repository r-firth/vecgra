use super::*;

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
/// search delegates to Vecgra's adaptive native index. Hybrid retrieval
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
        Some(vecgra_embedding::embed_query(
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
    path: &vecgra::ShortestPath,
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
