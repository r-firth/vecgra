//! Hiding nodes from view: a hub that touches everything, a node beside the
//! point, or every node of a label can be taken off the canvas with their
//! relationships, and shown again later where they were.

use std::collections::HashMap;

use vecgra_studio_core::HiddenNodes;

use super::*;

/// What is hidden from view, and the scene it was hidden from, so that
/// showing it again does not need the database (or works without one).
pub(super) struct HiddenScene {
    /// The scene as last read, with every node.
    pub(super) full: Arc<SceneSnapshot>,
    pub(super) hidden: HiddenNodes,
    /// Where each hidden node was laid out, so it comes back to the same
    /// place.
    positions: HashMap<u64, Vec2>,
}

impl StudioView {
    /// How many of the scene's nodes are hidden.
    pub(super) fn hidden_count(&self) -> usize {
        match (&self.hidden, self.snapshot()) {
            (Some(hidden), Some(snapshot)) => hidden
                .full
                .nodes
                .ids
                .len()
                .saturating_sub(snapshot.nodes.ids.len()),
            _ => 0,
        }
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
        self.hide_where(&current, |hidden| {
            hidden.ids.insert(id);
        });
        self.status = format!(
            "Hid {name} · {} hidden · ⇧⌘H shows them",
            self.hidden_count()
        )
        .into();
        cx.notify();
    }

    /// Takes every node with `label` off the canvas, including any that a
    /// later reload brings in.
    pub(super) fn hide_label(&mut self, label: &str, cx: &mut Context<Self>) {
        let Some(current) = self.snapshot().cloned() else {
            return;
        };
        let count = current
            .nodes
            .labels
            .iter()
            .filter(|other| &***other == label)
            .count();
        if count == 0 {
            self.status = format!("No {label} nodes are in view").into();
            cx.notify();
            return;
        }
        let label: Arc<str> = label.into();
        self.hide_where(&current, |hidden| {
            hidden.labels.insert(label.clone());
        });
        self.status = format!(
            "Hid {count} {label} node{} · {} hidden · ⇧⌘H shows them",
            if count == 1 { "" } else { "s" },
            self.hidden_count()
        )
        .into();
        cx.notify();
    }

    /// Adds to what is hidden and shows the scene without it.
    fn hide_where(&mut self, current: &Arc<SceneSnapshot>, add: impl FnOnce(&mut HiddenNodes)) {
        let overview = self
            .workspace
            .as_ref()
            .map(|workspace| workspace.borrow().overview_positions().to_vec())
            .unwrap_or_default();
        let scene = self.hidden.get_or_insert_with(|| HiddenScene {
            full: current.clone(),
            hidden: HiddenNodes::default(),
            positions: HashMap::new(),
        });
        add(&mut scene.hidden);
        for (index, &id) in current.nodes.ids.iter().enumerate() {
            if scene.hidden.hides(id, &current.nodes.labels[index])
                && let Some(&position) = overview.get(index)
            {
                scene.positions.insert(id, position);
            }
        }
        let shown = scene.full.without_nodes(&scene.hidden);
        self.show_scene(shown, current, &HashMap::new());
    }

    /// Puts every hidden node back where it was, each entering from a
    /// neighbor and ringed like nodes a reload adds.
    pub(super) fn show_hidden_nodes(&mut self, cx: &mut Context<Self>) {
        let count = self.hidden_count();
        let (Some(current), Some(hidden)) = (self.snapshot().cloned(), self.hidden.take()) else {
            self.status = "No nodes are hidden".into();
            cx.notify();
            return;
        };
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

    /// Brings back the nodes hidden by `label`, where they were; nodes also
    /// hidden one by one stay hidden.
    pub(super) fn show_hidden_label(&mut self, label: &str, cx: &mut Context<Self>) {
        let Some(current) = self.snapshot().cloned() else {
            return;
        };
        let before = self.hidden_count();
        let Some(scene) = self.hidden.as_mut() else {
            return;
        };
        if !scene.hidden.labels.remove(label) {
            return;
        }
        let (shown, places) = if scene.hidden.is_empty() {
            let Some(scene) = self.hidden.take() else {
                return;
            };
            (Arc::unwrap_or_clone(scene.full), scene.positions)
        } else {
            (
                scene.full.without_nodes(&scene.hidden),
                scene.positions.clone(),
            )
        };
        self.show_scene(shown, &current, &places);
        let count = before.saturating_sub(self.hidden_count());
        self.status = format!(
            "Showing {count} {label} node{}",
            if count == 1 { "" } else { "s" }
        )
        .into();
        cx.notify();
    }

    /// Overview rows for the hidden labels, under the labels in view; each
    /// shows its label again when chosen.
    pub(super) fn hidden_label_rows(&self, cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        let Some(scene) = &self.hidden else {
            return Vec::new();
        };
        scene
            .hidden
            .labels
            .iter()
            .map(|label| {
                let count = scene
                    .full
                    .nodes
                    .labels
                    .iter()
                    .filter(|other| *other == label)
                    .count();
                let shown = label.clone();
                let selector = format!("hidden-label:{label}");
                let button = Button::new(selector.clone())
                    .ghost()
                    .small()
                    .compact()
                    .w_full()
                    .label(label.to_string())
                    .accessibility_id(format!("hidden-label:{label}"))
                    .tooltip(format!("Show the hidden {label} nodes"))
                    .cursor_pointer()
                    .opacity(0.6)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.show_hidden_label(&shown, cx);
                    }))
                    .child(
                        h_flex()
                            .flex_1()
                            .min_w_0()
                            .items_center()
                            .child(div().flex_1())
                            .child(
                                div()
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .text_xs()
                                    .text_color(palette().copper)
                                    .child(format!("{} hidden", panels::format_count(count))),
                            ),
                    );
                div()
                    .w_full()
                    .debug_selector(move || selector)
                    .child(button)
                    .into_any_element()
            })
            .collect()
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
