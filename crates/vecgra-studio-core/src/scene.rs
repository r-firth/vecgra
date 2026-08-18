use super::*;

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
    ) -> vecgra::Result<Self> {
        let database = Database::open_read_only(path)?;
        let mut snapshot = Self::from_database(&database, snapshot_options)?;
        snapshot.layout(layout_options);
        Ok(snapshot)
    }

    pub fn from_database(database: &Database, options: SnapshotOptions) -> vecgra::Result<Self> {
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
            database_name: "Demo graph".into(),
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
