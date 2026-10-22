//! Hiding nodes from view: a hub that touches everything, or a node beside
//! the point, can be taken off the canvas with its relationships, and shown
//! again later beside its neighbors.

use std::collections::{BTreeSet, HashMap};

use super::*;

/// The nodes hidden from view, and the scene they were hidden from, so that
/// showing them again does not need the database (or works without one).
pub(super) struct HiddenNodes {
    /// The scene as last read, with every node.
    pub(super) full: Arc<SceneSnapshot>,
    pub(super) ids: BTreeSet<u64>,
    /// Where each hidden node was laid out, so it comes back to the same
    /// place.
    positions: HashMap<u64, Vec2>,
}

impl StudioView {
    pub(super) fn hidden_count(&self) -> usize {
        self.hidden.as_ref().map_or(0, |hidden| hidden.ids.len())
    }

    pub(super) fn hide_selected(&mut self, cx: &mut Context<Self>) {
        match self.selection.and_then(SceneSelection::node) {
            Some(index) => self.hide_node(index, cx),
            None => {
                self.status = "Select a node to hide it".into();
                cx.notify();
            }
        }
    }

    /// Takes a node and its relationships off the canvas; everything else
    /// stays where it is.
    pub(super) fn hide_node(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(current) = self.snapshot().cloned() else {
            return;
        };
        let Some(&id) = current.nodes.ids.get(index) else {
            return;
        };
        let name = status_node_name(&current, index);
        let position = self
            .workspace
            .as_ref()
            .and_then(|workspace| workspace.borrow().overview_positions().get(index).copied());
        let hidden = self.hidden.get_or_insert_with(|| HiddenNodes {
            full: current.clone(),
            ids: BTreeSet::new(),
            positions: HashMap::new(),
        });
        hidden.ids.insert(id);
        if let Some(position) = position {
            hidden.positions.insert(id, position);
        }
        let scene = hidden.full.without_nodes(&hidden.ids);
        self.show_scene(scene, &current, &HashMap::new());
        self.status = format!(
            "Hid {name} · {} hidden · ⇧⌘H shows them",
            self.hidden_count()
        )
        .into();
        cx.notify();
    }

    /// Puts every hidden node back where it was, each entering from a
    /// neighbor and ringed like nodes a reload adds.
    pub(super) fn show_hidden_nodes(&mut self, cx: &mut Context<Self>) {
        let (Some(current), Some(hidden)) = (self.snapshot().cloned(), self.hidden.take()) else {
            self.status = "No nodes are hidden".into();
            cx.notify();
            return;
        };
        let count = hidden.ids.len();
        self.show_scene(
            Arc::unwrap_or_clone(hidden.full),
            &current,
            &hidden.positions,
        );
        self.status = format!(
            "Showing {count} hidden node{}",
            if count == 1 { "" } else { "s" }
        )
        .into();
        cx.notify();
    }

    /// Replaces the scene in view with one holding fewer or more of the same
    /// elements: those in both stay put, and the rest enter beside their
    /// neighbors, or settle where `places` says they were.
    fn show_scene(
        &mut self,
        scene: SceneSnapshot,
        current: &SceneSnapshot,
        places: &HashMap<u64, Vec2>,
    ) {
        let Some(workspace) = self.workspace.clone() else {
            return;
        };
        if self.has_transient_presentation() {
            self.retire_transient_presentation();
        }
        // A reload in flight was read for the scene being replaced; drop it,
        // and let live refresh look again on its next tick.
        if self.reload_in_progress {
            self.generation = self.generation.wrapping_add(1);
            self.reload_in_progress = false;
            self.load_task = None;
            self.fingerprint = None;
        }
        let (positions, pinned_ids) = anchor_positions(&workspace.borrow(), current);
        let selected = self.selected_element(self.selection);
        let mut reloaded = scene.anchored_on(current, &positions, LayoutOptions::default());
        if !reloaded.relaid {
            let nodes = &mut reloaded.snapshot.nodes;
            for (index, id) in nodes.ids.iter().enumerate() {
                if current.node_index(*id).is_none()
                    && let Some(&place) = places.get(id)
                {
                    nodes.positions[index] = place;
                }
            }
            reloaded.snapshot.bounds = Rect::from_points(&reloaded.snapshot.nodes.positions);
        }
        self.install_reloaded_scene(reloaded, &pinned_ids, selected);
    }
}
