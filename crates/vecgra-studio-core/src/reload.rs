use super::*;

use std::collections::VecDeque;
use std::time::SystemTime;

/// Cheap identity of a database file. Another process appending a
/// transaction, or compacting into a replacement file, changes it, so Studio
/// can notice new commits without reopening the database on every poll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DatabaseFingerprint {
    len: u64,
    modified: Option<SystemTime>,
    inode: u64,
}

impl DatabaseFingerprint {
    /// Reads the fingerprint from file metadata only.
    ///
    /// # Errors
    ///
    /// Returns an error when the file metadata is unavailable.
    pub fn read(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let metadata = std::fs::metadata(path)?;
        #[cfg(unix)]
        let inode = std::os::unix::fs::MetadataExt::ino(&metadata);
        #[cfg(not(unix))]
        let inode = 0;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            inode,
        })
    }
}

/// Element-level difference between two scene snapshots, keyed by stable IDs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneDelta {
    pub added_nodes: usize,
    pub removed_nodes: usize,
    /// Surviving nodes whose label, properties, or vector count changed.
    pub updated_nodes: usize,
    pub added_edges: usize,
    pub removed_edges: usize,
    /// Surviving relationships whose label, properties, or vector count
    /// changed.
    pub updated_edges: usize,
}

impl SceneDelta {
    pub const fn is_empty(self) -> bool {
        self.added_nodes == 0
            && self.removed_nodes == 0
            && self.updated_nodes == 0
            && self.added_edges == 0
            && self.removed_edges == 0
            && self.updated_edges == 0
    }
}

/// A freshly read scene anchored on the presentation it replaces.
#[derive(Clone, Debug)]
pub struct ReloadedScene {
    pub snapshot: SceneSnapshot,
    /// For each node that was not in the previous scene but grew from one
    /// that was, the presented position of that anchor. Renderers start the
    /// node there so it visibly enters from the structure it attached to.
    pub entrances: Vec<Option<Vec2>>,
    pub delta: SceneDelta,
    /// True when too little of the previous scene survived to anchor the new
    /// one, so it was laid out from scratch instead.
    pub relaid: bool,
}

impl SceneSnapshot {
    /// Rereads a database and anchors it on the current presentation.
    ///
    /// Nodes that survive keep their presented positions, so a live view does
    /// not jump when another process commits. New nodes are placed around the
    /// surviving neighbors they attach to, breadth-first, and new components
    /// with no surviving anchor are packed just outside the previous scene.
    ///
    /// # Errors
    ///
    /// Returns an error when the database cannot be opened or read.
    pub fn reload(
        path: impl AsRef<Path>,
        previous: &SceneSnapshot,
        previous_positions: &[Vec2],
        snapshot_options: SnapshotOptions,
        layout_options: LayoutOptions,
    ) -> vecgra::Result<ReloadedScene> {
        let database = Database::open_read_only(path)?;
        let snapshot = Self::from_database(&database, snapshot_options)?;
        Ok(snapshot.anchored_on(previous, previous_positions, layout_options))
    }

    /// Anchors this unlaid snapshot on a previous scene's presented positions.
    /// See [`SceneSnapshot::reload`].
    pub fn anchored_on(
        mut self,
        previous: &SceneSnapshot,
        previous_positions: &[Vec2],
        layout_options: LayoutOptions,
    ) -> ReloadedScene {
        let count = self.nodes.ids.len();
        let usable_previous = previous_positions.len() == previous.nodes.ids.len()
            && previous_positions
                .iter()
                .all(|position| position.x.is_finite() && position.y.is_finite());
        let mut positions: Vec<Option<Vec2>> = vec![None; count];
        let mut delta = SceneDelta::default();
        let mut retained = 0;
        for (index, &id) in self.nodes.ids.iter().enumerate() {
            let Some(previous_index) = previous.node_index(id) else {
                continue;
            };
            retained += 1;
            if usable_previous {
                positions[index] = Some(previous_positions[previous_index]);
            }
            if self.nodes.labels[index] != previous.nodes.labels[previous_index]
                || self.nodes.vector_counts[index] != previous.nodes.vector_counts[previous_index]
                || self.nodes.properties[index] != previous.nodes.properties[previous_index]
            {
                delta.updated_nodes += 1;
            }
        }
        let mut retained_edges = 0;
        for (index, &id) in self.edges.ids.iter().enumerate() {
            let Some(previous_index) = previous.edge_index(id) else {
                continue;
            };
            retained_edges += 1;
            if self.edges.labels[index] != previous.edges.labels[previous_index]
                || self.edges.vector_counts[index] != previous.edges.vector_counts[previous_index]
                || self.edges.properties[index] != previous.edges.properties[previous_index]
            {
                delta.updated_edges += 1;
            }
        }
        delta.added_nodes = count - retained;
        delta.removed_nodes = previous.nodes.ids.len() - retained;
        delta.added_edges = self.edges.ids.len() - retained_edges;
        delta.removed_edges = previous.edges.ids.len() - retained_edges;

        // Anchoring a scene that is mostly new would leave the old fragment
        // in charge of a much larger layout, so start over instead.
        if !usable_previous || retained == 0 || retained * 2 < count {
            self.layout(layout_options);
            return ReloadedScene {
                entrances: vec![None; count],
                snapshot: self,
                delta,
                relaid: true,
            };
        }

        let mut adjacency = vec![Vec::new(); count];
        for (&source, &target) in self.edges.sources.iter().zip(&self.edges.targets) {
            let (source, target) = (source as usize, target as usize);
            if source != target {
                adjacency[source].push(target);
                adjacency[target].push(source);
            }
        }
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let spacing = layout_options.edge_length.max(8.0);
        let previous_bounds =
            Rect::from_points(&positions.iter().flatten().copied().collect::<Vec<_>>());
        let mut entrances = vec![None; count];
        let mut queue: VecDeque<usize> = (0..count).filter(|&i| positions[i].is_some()).collect();
        grow_from_anchors(
            &self.nodes.ids,
            &adjacency,
            &mut positions,
            &mut entrances,
            &mut queue,
            spacing,
        );

        // Components with no surviving anchor enter on a spiral just outside
        // the previous scene, each one grown from its lowest-ID node.
        let center = previous_bounds.center();
        let clearance = previous_bounds.width().max(previous_bounds.height()) * 0.5 + spacing * 3.0;
        let mut slot = 0_usize;
        for seed in 0..count {
            if positions[seed].is_some() {
                continue;
            }
            let angle = slot as f32 * GOLDEN_ANGLE;
            let radius = clearance + spacing * 2.5 * (slot as f32).sqrt();
            positions[seed] = Some(center + Vec2::new(angle.cos(), angle.sin()) * radius);
            slot += 1;
            queue.push_back(seed);
            grow_from_anchors(
                &self.nodes.ids,
                &adjacency,
                &mut positions,
                &mut entrances,
                &mut queue,
                spacing,
            );
        }

        self.nodes.positions = positions
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect();
        self.bounds = Rect::from_points(&self.nodes.positions);
        ReloadedScene {
            snapshot: self,
            entrances,
            delta,
            relaid: false,
        }
    }
}

const GOLDEN_ANGLE: f32 = 2.399_963_2;

/// Breadth-first placement of unplaced nodes around placed neighbors. Each
/// parent fans its new children along a golden-angle spiral, so a hub that
/// gains many neighbors at once spreads them instead of stacking them.
fn grow_from_anchors(
    ids: &[NodeId],
    adjacency: &[Vec<usize>],
    positions: &mut [Option<Vec2>],
    entrances: &mut [Option<Vec2>],
    queue: &mut VecDeque<usize>,
    spacing: f32,
) {
    while let Some(parent) = queue.pop_front() {
        let Some(origin) = positions[parent] else {
            continue;
        };
        let base = stable_unit(ids[parent]) * std::f32::consts::TAU;
        let mut fanned = 0_usize;
        for &child in &adjacency[parent] {
            if positions[child].is_some() {
                continue;
            }
            let angle = base + fanned as f32 * GOLDEN_ANGLE;
            let radius = spacing * (1.0 + 0.45 * (fanned as f32).sqrt());
            positions[child] = Some(origin + Vec2::new(angle.cos(), angle.sin()) * radius);
            entrances[child] = Some(origin);
            fanned += 1;
            queue.push_back(child);
        }
    }
}
