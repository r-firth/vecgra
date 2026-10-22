//! Hiding elements from view: a hub that touches everything, a node beside
//! the point, every node of a label, or every relationship of a type can be
//! taken off the canvas, and shown again later where it was.

use std::collections::HashMap;

use vecgra_studio_core::HiddenElements;

use super::*;

/// What is hidden from view, and the scene it was hidden from, so that
/// showing it again does not need the database (or works without one).
pub(super) struct HiddenScene {
    /// The scene as last read, with every element.
    pub(super) full: Arc<SceneSnapshot>,
    pub(super) hidden: HiddenElements,
    /// Where each hidden node was laid out, so it comes back to the same
    /// place.
    positions: HashMap<u64, Vec2>,
    /// How many of the full scene's nodes are hidden.
    nodes: usize,
    /// How many of the full scene's relationships are hidden by type.
    relationships: usize,
    /// What was hidden, in order, so ⌘Z can take back the last.
    steps: Vec<HideStep>,
}

impl HiddenScene {
    /// Counts what is hidden again, after what is hidden or the full scene
    /// changes.
    pub(super) fn recount(&mut self) {
        let nodes = &self.full.nodes;
        self.nodes = nodes
            .ids
            .iter()
            .zip(&nodes.labels)
            .filter(|(id, label)| self.hidden.hides(**id, label))
            .count();
        self.relationships = self
            .full
            .edges
            .labels
            .iter()
            .filter(|label| self.hidden.hides_relationship(label))
            .count();
    }

    /// "45 nodes", "117 relationships" or both, for the status bar.
    pub(super) fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.nodes > 0 {
            parts.push(panels::counted(self.nodes, "node", "nodes"));
        }
        if self.relationships > 0 {
            parts.push(panels::counted(
                self.relationships,
                "relationship",
                "relationships",
            ));
        }
        parts.join(" · ")
    }
}

/// What an overview row for a hidden label or type brings back.
#[derive(Clone, PartialEq)]
enum HiddenGroup {
    Label(Arc<str>),
    Relationship(Arc<str>),
}

/// One thing hidden: a node by ID, a label or type, or every label or type
/// but one.
#[derive(Clone, PartialEq)]
enum HideStep {
    Node(u64),
    Group(HiddenGroup),
    Groups(Vec<HiddenGroup>),
}

impl HiddenGroup {
    fn hide_in(&self, hidden: &mut HiddenElements) {
        match self {
            Self::Label(label) => hidden.labels.insert(label.clone()),
            Self::Relationship(label) => hidden.relationships.insert(label.clone()),
        };
    }

    /// Shows the group again; says whether it was hidden.
    fn show_in(&self, hidden: &mut HiddenElements) -> bool {
        match self {
            Self::Label(label) => hidden.labels.remove(label),
            Self::Relationship(label) => hidden.relationships.remove(label),
        }
    }
}

impl StudioView {
    /// How many of the scene's nodes are hidden.
    #[cfg(test)]
    pub(super) fn hidden_count(&self) -> usize {
        self.hidden.as_ref().map_or(0, |scene| scene.nodes)
    }

    /// Whether a search hit is in the scene but hidden from view.
    pub(super) fn hit_is_hidden(&self, hit: &vecgra_studio_core::SearchHit) -> bool {
        let (Some(scene), Some(snapshot)) = (self.hidden.as_ref(), self.snapshot()) else {
            return false;
        };
        hit.scene_selection(snapshot).is_none() && hit.scene_selection(&scene.full).is_some()
    }

    /// The status bar's account of what is hidden, if anything is.
    pub(super) fn hidden_summary(&self) -> Option<String> {
        self.hidden
            .as_ref()
            .map(HiddenScene::summary)
            .filter(|summary| !summary.is_empty())
    }

    /// Hides the selected node, or with nothing selected, the label or type
    /// an overview lens is on.
    pub(super) fn hide_selected(&mut self, cx: &mut Context<Self>) {
        if let Some(index) = self.selection.and_then(SceneSelection::node) {
            self.hide_node(index, cx);
            return;
        }
        match self.active_facet.clone() {
            Some(FacetLens::NodeLabel(label)) => self.hide_label(&label, cx),
            Some(FacetLens::Relationship(label)) => self.hide_relationship(&label, cx),
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
        self.hide_where(&current, HideStep::Node(id));
        self.report_hidden(format!("Hid {name}"), cx);
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
        self.hide_where(&current, HideStep::Group(HiddenGroup::Label(label.clone())));
        self.report_hidden(
            format!(
                "Hid {count} {label} node{}",
                if count == 1 { "" } else { "s" }
            ),
            cx,
        );
    }

    /// Takes every relationship of type `label` off the canvas, including any
    /// that a later reload brings in; every node stays.
    pub(super) fn hide_relationship(&mut self, label: &str, cx: &mut Context<Self>) {
        let Some(current) = self.snapshot().cloned() else {
            return;
        };
        let count = current
            .edges
            .labels
            .iter()
            .filter(|other| &***other == label)
            .count();
        if count == 0 {
            self.status = format!("No {label} relationships are in view").into();
            cx.notify();
            return;
        }
        let label: Arc<str> = label.into();
        self.hide_where(
            &current,
            HideStep::Group(HiddenGroup::Relationship(label.clone())),
        );
        self.report_hidden(
            format!(
                "Hid {count} {label} relationship{}",
                if count == 1 { "" } else { "s" }
            ),
            cx,
        );
    }

    fn report_hidden(&mut self, what: String, cx: &mut Context<Self>) {
        // The status bar's hidden count already says how much is hidden.
        self.status = format!("{what} · ⌘Z undoes").into();
        cx.notify();
    }

    /// Adds to what is hidden and shows the scene without it.
    fn hide_where(&mut self, current: &Arc<SceneSnapshot>, step: HideStep) {
        let overview = self
            .workspace
            .as_ref()
            .map(|workspace| workspace.borrow().overview_positions().to_vec())
            .unwrap_or_default();
        let scene = self.hidden.get_or_insert_with(|| HiddenScene {
            full: current.clone(),
            hidden: HiddenElements::default(),
            positions: HashMap::new(),
            nodes: 0,
            relationships: 0,
            steps: Vec::new(),
        });
        match &step {
            HideStep::Node(id) => {
                scene.hidden.ids.insert(*id);
            }
            HideStep::Group(group) => group.hide_in(&mut scene.hidden),
            HideStep::Groups(groups) => {
                for group in groups {
                    group.hide_in(&mut scene.hidden);
                }
            }
        }
        // Hiding a label again hides any of its nodes shown as exceptions.
        let full = scene.full.clone();
        let hidden = &mut scene.hidden;
        hidden.shown.retain(|id| {
            full.node_index(*id)
                .is_none_or(|index| !hidden.labels.contains(&full.nodes.labels[index]))
        });
        scene.steps.push(step);
        scene.recount();
        for (index, &id) in current.nodes.ids.iter().enumerate() {
            if scene.hidden.hides(id, &current.nodes.labels[index])
                && let Some(&position) = overview.get(index)
            {
                scene.positions.insert(id, position);
            }
        }
        let shown = scene.full.without_hidden(&scene.hidden);
        self.show_scene(shown, current, &HashMap::new());
    }

    /// Puts everything hidden back: each node where it was, entering from a
    /// neighbor and ringed like nodes a reload adds.
    pub(super) fn show_hidden_nodes(&mut self, cx: &mut Context<Self>) {
        let summary = self.hidden_summary();
        let (Some(current), Some(hidden), Some(summary)) =
            (self.snapshot().cloned(), self.hidden.take(), summary)
        else {
            self.hidden = None;
            self.status = "Nothing is hidden".into();
            cx.notify();
            return;
        };
        self.show_scene(
            Arc::unwrap_or_clone(hidden.full),
            &current,
            &hidden.positions,
        );
        self.status = format!("Showing {summary} again").into();
        cx.notify();
    }

    /// Hides every node label in view but `label`, or with `relationship`
    /// every relationship type but `label`, as one step.
    pub(super) fn show_only(&mut self, label: &str, relationship: bool, cx: &mut Context<Self>) {
        let Some(current) = self.snapshot().cloned() else {
            return;
        };
        let counts = if relationship {
            self.relationship_counts.clone()
        } else {
            self.node_label_counts.clone()
        };
        let (kind, kinds, elements) = if relationship {
            ("relationship type", "relationship types", "relationships")
        } else {
            ("label", "labels", "nodes")
        };
        let others: Vec<HiddenGroup> = counts
            .iter()
            .filter(|(other, _)| &**other != label)
            .map(|(other, _)| {
                if relationship {
                    HiddenGroup::Relationship(other.clone())
                } else {
                    HiddenGroup::Label(other.clone())
                }
            })
            .collect();
        if others.is_empty() {
            self.status = format!("Only {label} {elements} are in view").into();
            cx.notify();
            return;
        }
        let count = others.len();
        self.hide_where(&current, HideStep::Groups(others));
        // What is left is what the reader asked to see, so frame it.
        self.fit(cx);
        self.report_hidden(
            format!(
                "Showing only {label} {elements} · hid {}",
                panels::counted(count, &format!("other {kind}"), &format!("other {kinds}"))
            ),
            cx,
        );
    }

    /// Takes back the last hiding still in effect.
    pub(super) fn undo_hide(&mut self, cx: &mut Context<Self>) {
        while let Some(step) = self.hidden.as_mut().and_then(|scene| scene.steps.pop()) {
            if self.show_again(&step, cx) {
                return;
            }
        }
        self.status = "Nothing to undo".into();
        cx.notify();
    }

    /// Brings back what a hidden label or type hid.
    fn show_hidden_group(&mut self, group: HiddenGroup, cx: &mut Context<Self>) {
        self.show_again(&HideStep::Group(group), cx);
    }

    /// Brings back what one step hid, nodes where they were; anything also
    /// hidden another way stays hidden. Says whether the step was still in
    /// effect.
    fn show_again(&mut self, step: &HideStep, cx: &mut Context<Self>) -> bool {
        let Some(current) = self.snapshot().cloned() else {
            return false;
        };
        let Some(scene) = self.hidden.as_mut() else {
            return false;
        };
        let (removed, what) = match step {
            HideStep::Node(id) => (
                scene.hidden.ids.remove(id),
                scene.full.node_index(*id).map_or_else(
                    || format!("node {id}"),
                    |index| status_node_name(&scene.full, index),
                ),
            ),
            HideStep::Group(group) => (
                group.show_in(&mut scene.hidden),
                match group {
                    HiddenGroup::Label(label) => format!("{label} nodes"),
                    HiddenGroup::Relationship(label) => format!("{label} relationships"),
                },
            ),
            HideStep::Groups(groups) => {
                // Every group is shown, even after one already was.
                let shown = groups
                    .iter()
                    .filter(|group| group.show_in(&mut scene.hidden))
                    .count();
                let kind = match groups.first() {
                    Some(HiddenGroup::Relationship(_)) => {
                        ("relationship type", "relationship types")
                    }
                    _ => ("label", "labels"),
                };
                (shown > 0, panels::counted(shown, kind.0, kind.1))
            }
        };
        if !removed {
            return false;
        }
        scene.steps.retain(|other| other != step);
        self.show_less_hidden(&current);
        if matches!(step, HideStep::Groups(_)) {
            // Showing only one label framed it; frame everything again.
            self.fit(cx);
        }
        self.status = format!("Showing {what} again").into();
        cx.notify();
        true
    }

    /// Shows one hidden node again: by ID if it was hidden on its own, or as
    /// an exception to its hidden label. Says whether it was hidden.
    pub(super) fn reveal_node(&mut self, id: u64) -> bool {
        let Some(current) = self.snapshot().cloned() else {
            return false;
        };
        let Some(scene) = self.hidden.as_mut() else {
            return false;
        };
        let Some(index) = scene.full.node_index(id) else {
            return false;
        };
        let label = scene.full.nodes.labels[index].clone();
        if !scene.hidden.hides(id, &label) {
            return false;
        }
        scene.hidden.ids.remove(&id);
        scene.steps.retain(|step| step != &HideStep::Node(id));
        if scene.hidden.hides(id, &label) {
            scene.hidden.shown.insert(id);
        }
        self.show_less_hidden(&current);
        true
    }

    /// Shows the scene after something hidden was shown again, the returning
    /// nodes where they were.
    fn show_less_hidden(&mut self, current: &SceneSnapshot) {
        let Some(scene) = self.hidden.as_mut() else {
            return;
        };
        let (shown, places) = if scene.hidden.is_empty() {
            let Some(scene) = self.hidden.take() else {
                return;
            };
            (Arc::unwrap_or_clone(scene.full), scene.positions)
        } else {
            scene.recount();
            (
                scene.full.without_hidden(&scene.hidden),
                scene.positions.clone(),
            )
        };
        self.show_scene(shown, current, &places);
    }

    /// Overview rows for the hidden node labels, under the labels in view;
    /// each shows its label again when chosen.
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
                hidden_group_row(HiddenGroup::Label(label.clone()), count, cx)
            })
            .collect()
    }

    /// Overview rows for the hidden relationship types, under the types in
    /// view; each shows its type again when chosen.
    pub(super) fn hidden_relationship_rows(&self, cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        let Some(scene) = &self.hidden else {
            return Vec::new();
        };
        scene
            .hidden
            .relationships
            .iter()
            .map(|label| {
                let count = scene
                    .full
                    .edges
                    .labels
                    .iter()
                    .filter(|other| *other == label)
                    .count();
                hidden_group_row(HiddenGroup::Relationship(label.clone()), count, cx)
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

/// A dimmed overview row for a hidden label or type, with its hidden count;
/// choosing it shows the group again.
fn hidden_group_row(
    group: HiddenGroup,
    count: usize,
    cx: &Context<StudioView>,
) -> gpui::AnyElement {
    let (kind, label, what) = match &group {
        HiddenGroup::Label(label) => ("label", label.clone(), "nodes"),
        HiddenGroup::Relationship(label) => ("relationship", label.clone(), "relationships"),
    };
    let selector = format!("hidden-{kind}:{label}");
    let button = Button::new(selector.clone())
        .ghost()
        .small()
        .compact()
        .w_full()
        .label(label.to_string())
        .accessibility_id(selector.clone())
        .tooltip(format!("Show the hidden {label} {what}"))
        .cursor_pointer()
        .opacity(0.6)
        .on_click(cx.listener(move |this, event, window, cx| {
            this.focus_graph_after(event, window, cx);
            this.show_hidden_group(group.clone(), cx);
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
}
