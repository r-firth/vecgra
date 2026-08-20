use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use gpui::prelude::FluentBuilder as _;
use gpui::{
    AppContext as _, Bounds, Context, Entity, FocusHandle, Focusable as _, InteractiveElement as _,
    IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _,
    PinchEvent, Pixels, Render, Role, ScrollHandle, ScrollWheelEvent, SharedString,
    StatefulInteractiveElement as _, Styled, Subscription, Task, Window, actions, div, px,
    relative, rgb,
};
use gpui_component::{
    ActiveTheme as _, Disableable as _, ElementExt as _, InteractiveElementExt as _,
    Selectable as _, Sizable as _, TitleBar,
    button::{Button, ButtonVariants as _},
    h_flex,
    input::Input,
    input::InputEvent,
    input::InputState,
    scroll::ScrollableElement as _,
    v_flex,
};
use vecgra_studio_core::{
    Camera, CameraMotion, DetailLevel, EvidenceNode, EvidencePathReport, EvidencePathStrategy,
    EvidencePathTermination, GraphWorkspace, LayoutKind, LayoutOptions, MAX_CAMERA_ZOOM,
    MIN_CAMERA_ZOOM, PathDirection, PropertyValue, Rect, SceneSelection, SceneSnapshot, SearchMode,
    SearchReport, SnapshotOptions, Vec2, detail_level, evidence_path_database, hit_test_edges,
    hit_test_positions, search_database,
};

mod bezel_controls;
mod panels;

use panels::path_direction_label;
#[cfg(test)]
use panels::visible_facet_counts;

use crate::graph_canvas::{
    GraphCanvasPresentation, GraphEmphasis, GraphPathEndpoints, graph_canvas,
};
use crate::graph_navigator::{graph_navigator, navigator_world_position};
use crate::theme::{palette, relationship_color};

actions!(
    vecgra_studio,
    [
        FitView,
        ZoomIn,
        ZoomOut,
        ClearSelection,
        ArrangeAuto,
        ArrangeForce,
        ArrangeStructure,
        ArrangeOrbit,
        ReleaseSelected,
        ActivateSelection,
        FocusSearch,
        NextSearchResult,
        PreviousSearchResult
    ]
);

enum LoadState {
    Loading { name: SharedString },
    Ready(Arc<SceneSnapshot>),
    Failed(SharedString),
}

enum DragState {
    Canvas {
        last: gpui::Point<Pixels>,
        distance: f32,
        pressed_edge: Option<usize>,
    },
    Node {
        index: usize,
        grab_offset: Vec2,
        last: gpui::Point<Pixels>,
        distance: f32,
        was_pinned: bool,
    },
}

enum SearchState {
    Idle,
    Searching {
        query: SharedString,
        mode: SearchMode,
    },
    Ready(Arc<SearchReport>),
    Failed {
        query: SharedString,
        error: SharedString,
    },
}

enum PathState {
    Idle,
    ChoosingEnd(PathDraft),
    Searching {
        start: u64,
        end: u64,
    },
    Ready(Arc<EvidencePathReport>),
    Failed {
        start: u64,
        end: u64,
        error: SharedString,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathDraft {
    start: u64,
    direction: PathDirection,
    max_hops: usize,
}

struct EvidencePathQuery {
    start: u64,
    end: u64,
    direction: PathDirection,
    relationship_label: Option<String>,
    max_hops: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum FacetLens {
    NodeLabel(Arc<str>),
    Relationship(Arc<str>),
}

impl FacetLens {
    fn label(&self) -> &Arc<str> {
        match self {
            Self::NodeLabel(label) | Self::Relationship(label) => label,
        }
    }

    const fn kind_label(&self) -> &'static str {
        match self {
            Self::NodeLabel(_) => "node label",
            Self::Relationship(_) => "relationship",
        }
    }
}

struct LensTransition {
    from_nodes: Arc<[f32]>,
    target_nodes: Arc<[f32]>,
    from_edges: Arc<[f32]>,
    target_edges: Arc<[f32]>,
    from_dim: f32,
    target_dim: f32,
    progress: f32,
    velocity: f32,
    moving: bool,
}

impl LensTransition {
    fn new(node_count: usize, edge_count: usize) -> Self {
        Self {
            from_nodes: vec![0.0; node_count].into(),
            target_nodes: vec![0.0; node_count].into(),
            from_edges: vec![0.0; edge_count].into(),
            target_edges: vec![0.0; edge_count].into(),
            from_dim: 0.0,
            target_dim: 0.0,
            progress: 1.0,
            velocity: 0.0,
            moving: false,
        }
    }

    fn emphasis(&self) -> GraphEmphasis {
        GraphEmphasis {
            from_nodes: self.from_nodes.clone(),
            target_nodes: self.target_nodes.clone(),
            from_edges: self.from_edges.clone(),
            target_edges: self.target_edges.clone(),
            mix: self.progress,
            dim: self.from_dim + (self.target_dim - self.from_dim) * self.progress,
        }
    }

    fn retarget(&mut self, nodes: Arc<[f32]>, edges: Arc<[f32]>, dim: f32) {
        self.from_nodes = interpolate_scores(&self.from_nodes, &self.target_nodes, self.progress);
        self.from_edges = interpolate_scores(&self.from_edges, &self.target_edges, self.progress);
        self.from_dim = self.from_dim + (self.target_dim - self.from_dim) * self.progress;
        self.target_nodes = nodes;
        self.target_edges = edges;
        self.target_dim = dim.clamp(0.0, 1.0);
        self.progress = 0.0;
        self.velocity = self.velocity.max(0.0);
        self.moving = true;
    }

    fn step(&mut self, elapsed_seconds: f32, reduce_motion: bool) -> bool {
        if !self.moving {
            return false;
        }
        if reduce_motion {
            self.progress = 1.0;
            self.velocity = 0.0;
            self.moving = false;
            return false;
        }
        let elapsed = elapsed_seconds.clamp(0.0, 0.05);
        if elapsed <= f32::EPSILON {
            return true;
        }
        const MAX_STEP: f32 = 1.0 / 120.0;
        const STIFFNESS: f32 = 150.0;
        const DAMPING: f32 = 25.0;
        let steps = (elapsed / MAX_STEP).ceil().max(1.0) as usize;
        let dt = elapsed / steps as f32;
        for _ in 0..steps {
            let acceleration = (1.0 - self.progress) * STIFFNESS - self.velocity * DAMPING;
            self.velocity += acceleration * dt;
            self.progress += self.velocity * dt;
        }
        if (1.0 - self.progress).abs() <= 0.000_5 && self.velocity.abs() <= 0.002 {
            self.progress = 1.0;
            self.velocity = 0.0;
            self.moving = false;
            false
        } else if self.progress.is_finite() && self.velocity.is_finite() {
            true
        } else {
            self.progress = 1.0;
            self.velocity = 0.0;
            self.moving = false;
            false
        }
    }

    fn is_moving(&self) -> bool {
        self.moving
    }

    fn is_cleared(&self) -> bool {
        !self.moving && self.target_dim <= f32::EPSILON
    }
}

fn interpolate_scores(from: &[f32], target: &[f32], mix: f32) -> Arc<[f32]> {
    let mix = mix.clamp(0.0, 1.0);
    from.iter()
        .zip(target)
        .map(|(&from, &target)| from + (target - from) * mix)
        .collect()
}

type ReadyCallback = Box<dyn FnOnce(&mut Window, &mut Context<StudioView>)>;

pub struct StudioView {
    state: LoadState,
    database_path: Option<PathBuf>,
    workspace: Option<Rc<RefCell<GraphWorkspace>>>,
    node_label_counts: Arc<[(Arc<str>, usize)]>,
    relationship_counts: Arc<[(Arc<str>, usize)]>,
    camera: Camera,
    camera_motion: CameraMotion,
    saved_overview_camera: Option<Camera>,
    context_focus_active: bool,
    active_facet: Option<FacetLens>,
    lens: Option<LensTransition>,
    world_bounds: Rect,
    selection: Option<SceneSelection>,
    canvas_bounds: Option<Bounds<Pixels>>,
    navigator_bounds: Option<Bounds<Pixels>>,
    drag: Option<DragState>,
    query_input: Entity<InputState>,
    focus_handle: FocusHandle,
    bezel_search_focus: [FocusHandle; 3],
    bezel_zoom_focus: [FocusHandle; 3],
    bezel_layout_focus: [FocusHandle; 4],
    bezel_release_focus: FocusHandle,
    bezel_overview_focus: FocusHandle,
    inspector_scroll_handle: ScrollHandle,
    generation: u64,
    load_task: Option<Task<()>>,
    search_state: SearchState,
    search_mode: SearchMode,
    search_generation: u64,
    search_task: Option<Task<()>>,
    selected_search_result: usize,
    path_state: PathState,
    path_generation: u64,
    path_task: Option<Task<()>>,
    path_endpoints: Option<GraphPathEndpoints>,
    embedding_model: Arc<str>,
    layout_generation: u64,
    layout_task: Option<Task<()>>,
    layout_in_progress: bool,
    layout_kind: LayoutKind,
    last_motion_frame: Option<Instant>,
    load_ms: Option<f64>,
    status: SharedString,
    on_ready: Option<ReadyCallback>,
    on_search_ready: Option<ReadyCallback>,
    on_path_ready: Option<ReadyCallback>,
    on_layout_ready: Option<ReadyCallback>,
    _subscriptions: Vec<Subscription>,
}

impl StudioView {
    pub fn new(path: Option<PathBuf>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        crate::ensure_bezel_theme(cx);
        let query_input =
            cx.new(|cx| InputState::new(window, cx).placeholder("Search nodes + relationships…"));
        let subscription = cx.subscribe_in(&query_input, window, Self::on_query_event);
        let focus_handle = cx.focus_handle().tab_stop(true);
        let bezel_search_focus = std::array::from_fn(|_| cx.focus_handle());
        let bezel_zoom_focus = std::array::from_fn(|_| cx.focus_handle());
        let bezel_layout_focus = std::array::from_fn(|_| cx.focus_handle());
        let bezel_release_focus = cx.focus_handle();
        let bezel_overview_focus = cx.focus_handle();
        let (state, workspace, world_bounds) = if let Some(path) = &path {
            (
                LoadState::Loading {
                    name: path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("database")
                        .to_string()
                        .into(),
                },
                None,
                Rect::default(),
            )
        } else {
            let snapshot = Arc::new(SceneSnapshot::demo());
            let workspace = Rc::new(RefCell::new(GraphWorkspace::new(&snapshot)));
            let bounds = snapshot.bounds;
            (LoadState::Ready(snapshot), Some(workspace), bounds)
        };
        let (node_label_counts, relationship_counts) = match &state {
            LoadState::Ready(snapshot) => (
                Arc::from(snapshot.label_counts()),
                Arc::from(snapshot.relationship_counts()),
            ),
            LoadState::Loading { .. } | LoadState::Failed(_) => (Arc::from([]), Arc::from([])),
        };
        let mut view = Self {
            state,
            database_path: path.clone(),
            workspace,
            node_label_counts,
            relationship_counts,
            camera: Camera::default(),
            camera_motion: CameraMotion::new(Camera::default()),
            saved_overview_camera: None,
            context_focus_active: false,
            active_facet: None,
            lens: None,
            world_bounds,
            selection: None,
            canvas_bounds: None,
            navigator_bounds: None,
            drag: None,
            query_input,
            focus_handle,
            bezel_search_focus,
            bezel_zoom_focus,
            bezel_layout_focus,
            bezel_release_focus,
            bezel_overview_focus,
            inspector_scroll_handle: ScrollHandle::new(),
            generation: 0,
            load_task: None,
            search_state: SearchState::Idle,
            search_mode: SearchMode::Hybrid,
            search_generation: 0,
            search_task: None,
            selected_search_result: 0,
            path_state: PathState::Idle,
            path_generation: 0,
            path_task: None,
            path_endpoints: None,
            embedding_model: std::env::var("VECGRA_EMBEDDER")
                .unwrap_or_else(|_| "hash".into())
                .into(),
            layout_generation: 0,
            layout_task: None,
            layout_in_progress: false,
            layout_kind: LayoutKind::Auto,
            last_motion_frame: None,
            load_ms: None,
            status: if path.is_some() {
                "Opening database…".into()
            } else {
                "Demo scene · pass a .vg path to inspect a database".into()
            },
            on_ready: None,
            on_search_ready: None,
            on_path_ready: None,
            on_layout_ready: None,
            _subscriptions: vec![subscription],
        };
        if let Some(path) = path {
            view.open(path, window, cx);
        } else if matches!(view.state, LoadState::Ready(_)) {
            view.camera = Camera::fit(view.world_bounds);
            view.camera_motion.cancel_at(view.camera);
        }
        view
    }

    pub fn set_on_ready(
        &mut self,
        callback: impl FnOnce(&mut Window, &mut Context<Self>) + 'static,
    ) {
        self.on_ready = Some(Box::new(callback));
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.state, LoadState::Ready(_))
    }

    pub fn is_searching(&self) -> bool {
        matches!(self.search_state, SearchState::Searching { .. })
    }

    pub fn set_on_search_ready(
        &mut self,
        callback: impl FnOnce(&mut Window, &mut Context<Self>) + 'static,
    ) {
        self.on_search_ready = Some(Box::new(callback));
    }

    pub fn is_finding_path(&self) -> bool {
        matches!(self.path_state, PathState::Searching { .. })
    }

    pub fn set_on_path_ready(
        &mut self,
        callback: impl FnOnce(&mut Window, &mut Context<Self>) + 'static,
    ) {
        self.on_path_ready = Some(Box::new(callback));
    }

    #[doc(hidden)]
    pub fn is_arranging(&self) -> bool {
        self.layout_in_progress
    }

    #[doc(hidden)]
    pub fn set_on_layout_ready(
        &mut self,
        callback: impl FnOnce(&mut Window, &mut Context<Self>) + 'static,
    ) {
        self.on_layout_ready = Some(Box::new(callback));
    }

    pub fn is_presentation_moving(&self) -> bool {
        self.workspace
            .as_ref()
            .is_some_and(|workspace| workspace.borrow().is_moving())
            || self.camera_motion.is_moving()
            || self.lens.as_ref().is_some_and(LensTransition::is_moving)
    }

    /// Deterministically advances presentation springs for the offscreen
    /// visual harness, whose compositor does not deliver display-link frames.
    #[doc(hidden)]
    pub fn settle_presentation_for_capture(&mut self) {
        for _ in 0..600 {
            let workspace_moving = self
                .workspace
                .as_ref()
                .is_some_and(|workspace| workspace.borrow_mut().step(1.0 / 120.0, false));
            let camera_moving = self
                .camera_motion
                .step(&mut self.camera, 1.0 / 120.0, false);
            let lens_moving = self
                .lens
                .as_mut()
                .is_some_and(|lens| lens.step(1.0 / 120.0, false));
            if !workspace_moving && !camera_moving && !lens_moving {
                break;
            }
        }
        if self.is_presentation_moving() {
            if let Some(workspace) = self.workspace.as_ref() {
                workspace.borrow_mut().step(1.0 / 60.0, true);
            }
            self.camera_motion.step(&mut self.camera, 1.0 / 60.0, true);
            if let Some(lens) = self.lens.as_mut() {
                lens.step(1.0 / 60.0, true);
            }
        }
        if self.lens.as_ref().is_some_and(LensTransition::is_cleared) {
            self.lens = None;
            self.saved_overview_camera = None;
        }
        self.last_motion_frame = None;
    }

    fn open(&mut self, path: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        self.load_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let started = Instant::now();
                    let snapshot = SceneSnapshot::open(
                        path,
                        SnapshotOptions::default(),
                        LayoutOptions::default(),
                    );
                    (snapshot, started.elapsed())
                })
                .await;
            this.update_in(cx, |this, window, cx| {
                if this.generation != generation {
                    return;
                }
                let (snapshot, elapsed) = result;
                this.load_ms = Some(elapsed.as_secs_f64() * 1_000.0);
                match snapshot {
                    Ok(snapshot) => {
                        let snapshot = Arc::new(snapshot);
                        this.camera = Camera::fit(snapshot.bounds);
                        this.camera_motion.cancel_at(this.camera);
                        this.saved_overview_camera = None;
                        this.context_focus_active = false;
                        this.active_facet = None;
                        this.lens = None;
                        this.world_bounds = snapshot.bounds;
                        this.workspace =
                            Some(Rc::new(RefCell::new(GraphWorkspace::new(&snapshot))));
                        this.node_label_counts = Arc::from(snapshot.label_counts());
                        this.relationship_counts = Arc::from(snapshot.relationship_counts());
                        this.selection = None;
                        this.layout_in_progress = false;
                        this.layout_kind = LayoutKind::Auto;
                        this.last_motion_frame = None;
                        this.status = if snapshot.sampled {
                            "Ready · deterministic structural sample".into()
                        } else {
                            "Ready · complete graph view".into()
                        };
                        this.state = LoadState::Ready(snapshot);
                    }
                    Err(error) => {
                        this.status = "Could not open database".into();
                        this.state = LoadState::Failed(error.to_string().into());
                    }
                }
                cx.notify();
                if matches!(this.state, LoadState::Ready(_))
                    && let Some(callback) = this.on_ready.take()
                {
                    callback(window, cx);
                }
            })
            .ok();
        }));
    }

    fn snapshot(&self) -> Option<&Arc<SceneSnapshot>> {
        match &self.state {
            LoadState::Ready(snapshot) => Some(snapshot),
            LoadState::Loading { .. } | LoadState::Failed(_) => None,
        }
    }

    fn fit(&mut self, cx: &mut Context<Self>) {
        if let Some(workspace) = self.workspace.clone() {
            self.world_bounds = workspace.borrow().presentation_bounds();
            self.camera = Camera::fit(self.world_bounds);
            self.camera_motion.cancel_at(self.camera);
            cx.notify();
        }
    }

    fn zoom(&mut self, factor: f32, cx: &mut Context<Self>) {
        if self.snapshot().is_none() {
            return;
        }
        let Some(bounds) = self.canvas_bounds else {
            return;
        };
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        self.camera
            .zoom_about(viewport * 0.5, factor, self.world_bounds, viewport);
        self.camera_motion.cancel_at(self.camera);
        cx.notify();
    }

    fn arrange(&mut self, kind: LayoutKind, window: &mut Window, cx: &mut Context<Self>) {
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        let Some(workspace) = self.workspace.clone() else {
            return;
        };
        let (positions, pins) = {
            let workspace = workspace.borrow();
            (workspace.positions().to_vec(), workspace.pins().to_vec())
        };
        self.layout_generation = self.layout_generation.wrapping_add(1);
        let generation = self.layout_generation;
        self.layout_in_progress = true;
        self.status = format!("Computing {} arrangement…", kind.label()).into();
        self.layout_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let started = Instant::now();
                    let targets = snapshot.arranged_positions(
                        kind,
                        LayoutOptions::default(),
                        Some(&positions),
                        Some(&pins),
                    );
                    (targets, started.elapsed())
                })
                .await;
            this.update_in(cx, |this, window, cx| {
                if this.layout_generation != generation {
                    return;
                }
                let (targets, elapsed) = result;
                this.layout_in_progress = false;
                let accepted = this.workspace.as_ref().is_some_and(|workspace| {
                    workspace
                        .borrow_mut()
                        .retarget_layout(targets, this.world_bounds)
                });
                if accepted {
                    this.layout_kind = kind;
                    this.last_motion_frame = Some(Instant::now());
                    this.status = format!(
                        "{} arrangement · {:.1} ms · drag any node to pin it",
                        kind.label(),
                        elapsed.as_secs_f64() * 1_000.0
                    )
                    .into();
                } else {
                    this.status = "Arrangement result no longer matches this scene".into();
                }
                cx.notify();
                if let Some(callback) = this.on_layout_ready.take() {
                    callback(window, cx);
                }
            })
            .ok();
        }));
        cx.notify();
    }

    fn release_selected(&mut self, cx: &mut Context<Self>) {
        let Some(index) = self.selection.and_then(SceneSelection::node) else {
            return;
        };
        let Some(workspace) = self.workspace.as_ref() else {
            return;
        };
        if workspace.borrow().is_pinned(index) {
            workspace.borrow_mut().set_pinned(index, false);
            self.last_motion_frame = Some(Instant::now());
            self.status = "Released node back into the active layout".into();
            cx.notify();
        }
    }

    fn clear_selection(&mut self, cx: &mut Context<Self>) {
        if self.selection.is_some() {
            self.set_selection(None);
            self.status = "Selection cleared".into();
            cx.notify();
        }
    }

    fn set_selection(&mut self, selection: Option<SceneSelection>) {
        if self.selection != selection {
            self.inspector_scroll_handle.set_offset(Default::default());
        }
        self.selection = selection;
    }

    fn begin_presentation_motion(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.last_motion_frame = Some(Instant::now());
        cx.notify();
    }

    fn retarget_lens(&mut self, nodes: Vec<f32>, edges: Vec<f32>, dim: f32) {
        let node_count = nodes.len();
        let edge_count = edges.len();
        let lens = self
            .lens
            .get_or_insert_with(|| LensTransition::new(node_count, edge_count));
        lens.retarget(nodes.into(), edges.into(), dim);
    }

    fn activate_facet(&mut self, facet: FacetLens, window: &mut Window, cx: &mut Context<Self>) {
        if self.active_facet.as_ref() == Some(&facet) {
            self.show_overview(window, cx);
            return;
        }
        let had_path = !matches!(self.path_state, PathState::Idle);
        self.path_generation = self.path_generation.wrapping_add(1);
        drop(self.path_task.take());
        self.path_state = PathState::Idle;
        self.path_endpoints = None;
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        let mut node_scores = vec![0.0_f32; snapshot.nodes.ids.len()];
        let mut edge_scores = vec![0.0_f32; snapshot.edges.ids.len()];
        let match_count = match &facet {
            FacetLens::NodeLabel(label) => {
                for (index, candidate) in snapshot.nodes.labels.iter().enumerate() {
                    if candidate == label {
                        node_scores[index] = 1.0;
                    }
                }
                for (index, (&source, &target)) in snapshot
                    .edges
                    .sources
                    .iter()
                    .zip(&snapshot.edges.targets)
                    .enumerate()
                {
                    let source_matches = node_scores[source as usize] > 0.0;
                    let target_matches = node_scores[target as usize] > 0.0;
                    edge_scores[index] = match (source_matches, target_matches) {
                        (true, true) => 0.52,
                        (true, false) | (false, true) => 0.18,
                        (false, false) => 0.0,
                    };
                }
                node_scores.iter().filter(|&&score| score > 0.0).count()
            }
            FacetLens::Relationship(label) => {
                let mut matches = 0;
                for (index, candidate) in snapshot.edges.labels.iter().enumerate() {
                    if candidate != label {
                        continue;
                    }
                    matches += 1;
                    edge_scores[index] = 1.0;
                    let source = snapshot.edges.sources[index] as usize;
                    let target = snapshot.edges.targets[index] as usize;
                    node_scores[source] = node_scores[source].max(0.72);
                    node_scores[target] = node_scores[target].max(0.72);
                }
                matches
            }
        };

        if self.context_focus_active || had_path {
            if let Some(workspace) = self.workspace.as_ref() {
                workspace.borrow_mut().restore_layout();
            }
            if let Some(camera) = self.saved_overview_camera {
                self.camera_motion.retarget(camera);
            }
        }
        self.context_focus_active = false;
        self.active_facet = Some(facet.clone());
        self.retarget_lens(node_scores, edge_scores, 0.9);
        self.status = format!(
            "{} lens · {} · {match_count} matches · Esc clears",
            facet.kind_label(),
            facet.label()
        )
        .into();
        self.begin_presentation_motion(window, cx);
    }

    fn present_search_report(
        &mut self,
        report: &SearchReport,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        let Some(workspace) = self.workspace.clone() else {
            return;
        };
        self.context_focus_active = false;
        self.active_facet = None;
        let mut node_scores = vec![0.0_f32; snapshot.nodes.ids.len()];
        let mut edge_scores = vec![0.0_f32; snapshot.edges.ids.len()];
        let mut focus_points = Vec::with_capacity(report.hits.len() * 2);
        let maximum = report
            .hits
            .first()
            .map_or(1.0, |hit| hit.score.max(f32::EPSILON));
        let workspace = workspace.borrow();
        for hit in report.hits.iter() {
            let score = (hit.score / maximum).clamp(0.18, 1.0);
            let Some(selection) = hit.scene_selection(&snapshot) else {
                continue;
            };
            match selection {
                SceneSelection::Node(index) => {
                    node_scores[index] = node_scores[index].max(score);
                    if let Some(position) = workspace.position(index) {
                        focus_points.push(position);
                    }
                }
                SceneSelection::Edge(index) => {
                    edge_scores[index] = edge_scores[index].max(score);
                    for endpoint in [
                        snapshot.edges.sources[index] as usize,
                        snapshot.edges.targets[index] as usize,
                    ] {
                        node_scores[endpoint] = node_scores[endpoint].max(score * 0.68);
                        if let Some(position) = workspace.position(endpoint) {
                            focus_points.push(position);
                        }
                    }
                }
            }
        }
        drop(workspace);
        self.retarget_lens(node_scores, edge_scores, 0.84);
        self.saved_overview_camera.get_or_insert(self.camera);
        if let Some(bounds) = self.canvas_bounds
            && !focus_points.is_empty()
        {
            let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
            let mut target = Camera::framed(
                Rect::from_points(&focus_points),
                self.world_bounds,
                viewport,
                96.0,
            );
            target.zoom = target.zoom.min(4.8);
            self.camera_motion.retarget(target);
        }
        self.begin_presentation_motion(window, cx);
    }

    fn start_evidence_path(
        &mut self,
        query: EvidencePathQuery,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let EvidencePathQuery {
            start,
            end,
            direction,
            relationship_label,
            max_hops,
        } = query;
        let had_presentation = self.context_focus_active
            || self.active_facet.is_some()
            || !matches!(self.search_state, SearchState::Idle)
            || !matches!(self.path_state, PathState::Idle);
        let overview_camera = self.saved_overview_camera.unwrap_or(self.camera);
        self.search_generation = self.search_generation.wrapping_add(1);
        drop(self.search_task.take());
        self.search_state = SearchState::Idle;
        self.path_generation = self.path_generation.wrapping_add(1);
        drop(self.path_task.take());
        self.path_endpoints = self.snapshot().and_then(|snapshot| {
            snapshot.node_index(start).map(|start| GraphPathEndpoints {
                start,
                end: snapshot.node_index(end),
            })
        });
        if had_presentation {
            if let Some(workspace) = self.workspace.as_ref() {
                workspace.borrow_mut().restore_layout();
            }
            if let Some(snapshot) = self.snapshot() {
                self.retarget_lens(
                    vec![0.0; snapshot.nodes.ids.len()],
                    vec![0.0; snapshot.edges.ids.len()],
                    0.0,
                );
            }
            if let Some(camera) = self.saved_overview_camera {
                self.camera_motion.retarget(camera);
            }
        }
        self.context_focus_active = false;
        self.active_facet = None;
        self.saved_overview_camera = Some(overview_camera);

        let enrichment_input =
            self.snapshot()
                .cloned()
                .zip(self.workspace.clone())
                .map(|(snapshot, workspace)| {
                    let workspace = workspace.borrow();
                    let mut overview_positions = workspace.overview_positions().to_vec();
                    let pinned_ids: Arc<[u64]> = workspace
                        .pins()
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &pinned)| {
                            if pinned {
                                overview_positions[index] = workspace.positions()[index];
                                Some(snapshot.nodes.ids[index])
                            } else {
                                None
                            }
                        })
                        .collect();
                    (snapshot, overview_positions, pinned_ids)
                });

        let Some(database_path) = self.database_path.clone() else {
            self.path_state = PathState::Failed {
                start,
                end,
                error: "Open a .vg database to trace exact evidence paths".into(),
            };
            self.status = "Path search needs an open database".into();
            cx.notify();
            return;
        };
        let generation = self.path_generation;
        self.path_state = PathState::Searching { start, end };
        self.status = format!("Tracing exact path · node:{start} → node:{end}…").into();
        self.path_task = Some(cx.spawn_in(window, async move |this, cx| {
            let worker_label = relationship_label.clone();
            let result = cx
                .background_spawn(async move {
                    let report = evidence_path_database(
                        &database_path,
                        start,
                        end,
                        direction,
                        worker_label.as_deref(),
                        max_hops,
                        100_000,
                    )?;
                    let enriched = match enrichment_input {
                        Some((snapshot, overview_positions, pinned_ids)) => snapshot
                            .including_evidence_path(&report, &overview_positions)?
                            .map(|snapshot| (snapshot, pinned_ids)),
                        None => None,
                    };
                    Ok::<_, String>((report, enriched))
                })
                .await;
            this.update_in(cx, |this, window, cx| {
                if this.path_generation != generation {
                    return;
                }
                match result {
                    Ok((report, enriched)) => {
                        if let Some((snapshot, pinned_ids)) = enriched {
                            this.install_evidence_snapshot(snapshot, &pinned_ids);
                        }
                        let report = Arc::new(report);
                        this.present_evidence_path(&report, window, cx);
                        let elapsed_ms = report.elapsed.as_secs_f64() * 1_000.0;
                        this.status = match report.termination {
                            EvidencePathTermination::Found => {
                                let hops = report.path.as_ref().map_or(0, |path| path.steps.len());
                                format!(
                                    "Exact evidence path · {hops} hop{} · {elapsed_ms:.1} ms · Esc restores overview",
                                    if hops == 1 { "" } else { "s" }
                                )
                                .into()
                            }
                            EvidencePathTermination::NotFoundWithinHops => format!(
                                "No path within {} hops · conclusive for this bound · {elapsed_ms:.1} ms",
                                report.max_hops
                            )
                            .into(),
                            EvidencePathTermination::ExpansionLimit => format!(
                                "Path work limit reached · result incomplete · {elapsed_ms:.1} ms"
                            )
                            .into(),
                        };
                        this.path_state = PathState::Ready(report);
                    }
                    Err(error) => {
                        this.status = "Path search failed".into();
                        this.path_state = PathState::Failed {
                            start,
                            end,
                            error: error.into(),
                        };
                    }
                }
                cx.notify();
                if let Some(callback) = this.on_path_ready.take() {
                    callback(window, cx);
                }
            })
            .ok();
        }));
        cx.notify();
    }

    fn install_evidence_snapshot(&mut self, snapshot: SceneSnapshot, pinned_ids: &[u64]) {
        let snapshot = Arc::new(snapshot);
        let mut workspace = GraphWorkspace::new(&snapshot);
        for &id in pinned_ids {
            if let Some(index) = snapshot.node_index(id) {
                workspace.set_pinned(index, true);
            }
        }
        self.world_bounds = snapshot.bounds;
        self.node_label_counts = Arc::from(snapshot.label_counts());
        self.relationship_counts = Arc::from(snapshot.relationship_counts());
        self.workspace = Some(Rc::new(RefCell::new(workspace)));
        self.state = LoadState::Ready(snapshot);
        self.last_motion_frame = None;
    }

    fn choose_path_start(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(start) = self
            .snapshot()
            .and_then(|snapshot| snapshot.nodes.ids.get(index))
            .copied()
        else {
            return;
        };
        if self.database_path.is_none() {
            self.status = "Open a .vg database to trace exact evidence paths".into();
            cx.notify();
            return;
        }
        let (direction, max_hops) = match self.path_state {
            PathState::ChoosingEnd(draft) => (draft.direction, draft.max_hops),
            PathState::Idle
            | PathState::Searching { .. }
            | PathState::Ready(_)
            | PathState::Failed { .. } => (PathDirection::Both, 6),
        };

        let overview_camera = self.saved_overview_camera.unwrap_or(self.camera);
        self.search_generation = self.search_generation.wrapping_add(1);
        drop(self.search_task.take());
        self.search_state = SearchState::Idle;
        self.path_generation = self.path_generation.wrapping_add(1);
        drop(self.path_task.take());
        if let Some(workspace) = self.workspace.as_ref() {
            workspace.borrow_mut().restore_layout();
        }
        if let Some(snapshot) = self.snapshot() {
            self.retarget_lens(
                vec![0.0; snapshot.nodes.ids.len()],
                vec![0.0; snapshot.edges.ids.len()],
                0.0,
            );
        }
        if let Some(camera) = self.saved_overview_camera {
            self.camera_motion.retarget(camera);
        }
        self.context_focus_active = false;
        self.active_facet = None;
        self.saved_overview_camera = Some(overview_camera);
        let draft = PathDraft {
            start,
            direction,
            max_hops,
        };
        self.path_state = PathState::ChoosingEnd(draft);
        self.path_endpoints = Some(GraphPathEndpoints {
            start: index,
            end: None,
        });
        self.set_selection(Some(SceneSelection::Node(index)));
        self.status = self.path_draft_status(draft);
        self.begin_presentation_motion(window, cx);
    }

    fn set_path_direction(&mut self, direction: PathDirection, cx: &mut Context<Self>) {
        let PathState::ChoosingEnd(draft) = &mut self.path_state else {
            return;
        };
        if draft.direction == direction {
            return;
        }
        draft.direction = direction;
        let draft = *draft;
        self.status = self.path_draft_status(draft);
        cx.notify();
    }

    fn set_path_max_hops(&mut self, max_hops: usize, cx: &mut Context<Self>) {
        let PathState::ChoosingEnd(draft) = &mut self.path_state else {
            return;
        };
        if draft.max_hops == max_hops || !(1..=64).contains(&max_hops) {
            return;
        }
        draft.max_hops = max_hops;
        let draft = *draft;
        self.status = self.path_draft_status(draft);
        cx.notify();
    }

    fn trace_path_to_node(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(query) = self.path_query_to_node(index) else {
            return;
        };
        self.start_evidence_path(query, window, cx);
    }

    fn path_destination_candidate(&self) -> Option<usize> {
        let PathState::ChoosingEnd(draft) = self.path_state else {
            return None;
        };
        let index = self.selection.and_then(SceneSelection::node)?;
        self.snapshot()?
            .nodes
            .ids
            .get(index)
            .is_some_and(|&id| id != draft.start)
            .then_some(index)
    }

    fn path_draft_status(&self, draft: PathDraft) -> SharedString {
        if let Some(end) = self
            .path_destination_candidate()
            .and_then(|index| self.snapshot()?.nodes.ids.get(index).copied())
        {
            format!(
                "Destination ready · node:{end} · {} · up to {} hops · Enter traces exact path",
                path_direction_label(draft.direction),
                draft.max_hops
            )
            .into()
        } else {
            format!(
                "Path origin · node:{} · {} · up to {} hops · select a destination",
                draft.start,
                path_direction_label(draft.direction),
                draft.max_hops
            )
            .into()
        }
    }

    fn path_query_to_node(&self, index: usize) -> Option<EvidencePathQuery> {
        let PathState::ChoosingEnd(draft) = self.path_state else {
            return None;
        };
        let end = *self.snapshot()?.nodes.ids.get(index)?;
        Some(EvidencePathQuery {
            start: draft.start,
            end,
            direction: draft.direction,
            relationship_label: None,
            max_hops: draft.max_hops,
        })
    }

    fn present_evidence_path(
        &mut self,
        report: &EvidencePathReport,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        let Some(workspace) = self.workspace.clone() else {
            return;
        };
        let mut node_scores = vec![0.0_f32; snapshot.nodes.ids.len()];
        let mut edge_scores = vec![0.0_f32; snapshot.edges.ids.len()];
        let mut focus_points = Vec::new();
        let workspace = workspace.borrow();
        let path_nodes = report.path.as_ref().map_or_else(
            || vec![report.start.id, report.end.id],
            |path| path.nodes.iter().map(|node| node.id).collect(),
        );
        for (position, id) in path_nodes.iter().copied().enumerate() {
            if let Some(index) = snapshot.node_index(id) {
                let endpoint = position == 0 || position + 1 == path_nodes.len();
                node_scores[index] = if endpoint { 1.0 } else { 0.76 };
                if let Some(point) = workspace.position(index) {
                    focus_points.push(point);
                }
            }
        }
        if let Some(path) = &report.path {
            for step in path.steps.iter() {
                if let Some(index) = snapshot.edge_index(step.edge_id) {
                    edge_scores[index] = 1.0;
                }
            }
        }
        drop(workspace);
        self.path_endpoints =
            snapshot
                .node_index(report.start.id)
                .map(|start| GraphPathEndpoints {
                    start,
                    end: snapshot.node_index(report.end.id),
                });
        self.retarget_lens(
            node_scores,
            edge_scores,
            if report.termination == EvidencePathTermination::Found {
                0.96
            } else {
                0.78
            },
        );
        self.saved_overview_camera.get_or_insert(self.camera);
        if let Some(bounds) = self.canvas_bounds
            && !focus_points.is_empty()
        {
            let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
            let mut target = Camera::framed(
                Rect::from_points(&focus_points),
                self.world_bounds,
                viewport,
                116.0,
            );
            target.zoom = target.zoom.clamp(1.8, 24.0);
            self.camera_motion.retarget(target);
        }
        self.set_selection(
            snapshot
                .node_index(report.start.id)
                .map(SceneSelection::Node),
        );
        self.begin_presentation_motion(window, cx);
    }

    fn activate_evidence_step(
        &mut self,
        step_index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some((edge_id, from, to)) = (match &self.path_state {
            PathState::Ready(report) => report.path.as_ref().and_then(|path| {
                path.steps
                    .get(step_index)
                    .map(|step| (step.edge_id, step.from, step.to))
            }),
            PathState::Idle
            | PathState::ChoosingEnd(_)
            | PathState::Searching { .. }
            | PathState::Failed { .. } => None,
        }) else {
            return;
        };
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        let Some(edge_index) = snapshot.edge_index(edge_id) else {
            self.status = format!(
                "Relationship {edge_id} is outside this sampled scene · path evidence remains exact"
            )
            .into();
            cx.notify();
            return;
        };
        self.set_selection(Some(SceneSelection::Edge(edge_index)));
        if let (Some(bounds), Some(workspace)) = (self.canvas_bounds, self.workspace.as_ref()) {
            let workspace = workspace.borrow();
            let endpoints = snapshot
                .node_index(from)
                .zip(snapshot.node_index(to))
                .and_then(|(from, to)| workspace.position(from).zip(workspace.position(to)));
            if let Some((from, to)) = endpoints {
                let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
                let mut target = Camera::framed(
                    Rect::from_points(&[from, to]),
                    self.world_bounds,
                    viewport,
                    164.0,
                );
                target.zoom = target.zoom.clamp(3.0, 24.0);
                self.camera_motion.retarget(target);
            }
        }
        self.status = format!("Evidence step {} · relationship:{edge_id}", step_index + 1).into();
        self.begin_presentation_motion(window, cx);
    }

    fn focus_search_result(
        &mut self,
        selection: SceneSelection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        let Some(focus) = snapshot.focus_neighborhood(selection, 96) else {
            return;
        };
        let focus_bounds = self
            .workspace
            .as_ref()
            .and_then(|workspace| workspace.borrow_mut().retarget_focus(&focus, 46.0));
        let mut node_scores = vec![0.0_f32; snapshot.nodes.ids.len()];
        let mut edge_scores = vec![0.0_f32; snapshot.edges.ids.len()];
        for &index in &focus.nodes {
            node_scores[index] = 0.58;
        }
        for &index in &focus.roots {
            node_scores[index] = 1.0;
        }
        for &index in &focus.edges {
            edge_scores[index] = 0.68;
        }
        if let SceneSelection::Edge(index) = selection {
            edge_scores[index] = 1.0;
        }
        self.retarget_lens(node_scores, edge_scores, 0.96);
        if let (Some(bounds), Some(focus_bounds)) = (self.canvas_bounds, focus_bounds) {
            let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
            let mut target = Camera::framed(focus_bounds, self.world_bounds, viewport, 112.0);
            target.zoom = target.zoom.clamp(2.2, 9.0);
            self.camera_motion.retarget(target);
        }
        self.begin_presentation_motion(window, cx);
    }

    /// Presents a bounded two-hop context around a node. The expansion is
    /// branch-balanced and relationship-type-diverse in the scene layer, so a
    /// hub or one prolific relationship cannot monopolise the useful context.
    fn focus_node_context(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.path_generation = self.path_generation.wrapping_add(1);
        drop(self.path_task.take());
        self.path_state = PathState::Idle;
        self.path_endpoints = None;
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        let Some(focus) =
            snapshot.focus_neighborhood_layers(SceneSelection::Node(index), 2, 112, 168)
        else {
            return;
        };
        let focus_bounds = self
            .workspace
            .as_ref()
            .and_then(|workspace| workspace.borrow_mut().retarget_focus(&focus, 42.0));
        let mut node_scores = vec![0.0_f32; snapshot.nodes.ids.len()];
        let mut edge_scores = vec![0.0_f32; snapshot.edges.ids.len()];
        for (depth, layer) in focus.layers.iter().enumerate() {
            let score = match depth {
                0 => 1.0,
                1 => 0.72,
                _ => 0.46,
            };
            for &node in layer {
                node_scores[node] = score;
            }
        }
        for &edge in &focus.edges {
            let source = snapshot.edges.sources[edge] as usize;
            let target = snapshot.edges.targets[edge] as usize;
            let is_discovery_edge = focus.parents.iter().any(|&(child, parent)| {
                (child == source && parent == target) || (child == target && parent == source)
            });
            edge_scores[edge] = if is_discovery_edge { 0.72 } else { 0.22 };
        }
        self.retarget_lens(node_scores, edge_scores, 0.97);
        self.saved_overview_camera.get_or_insert(self.camera);
        self.context_focus_active = true;
        self.active_facet = None;
        self.set_selection(Some(SceneSelection::Node(index)));
        if let (Some(bounds), Some(focus_bounds)) = (self.canvas_bounds, focus_bounds) {
            let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
            let mut target = Camera::framed(focus_bounds, self.world_bounds, viewport, 112.0);
            target.zoom = target.zoom.clamp(1.4, 9.0);
            self.camera_motion.retarget(target);
        }
        let direct = focus.layers.get(1).map_or(0, Vec::len);
        let indirect = focus.layers.get(2).map_or(0, Vec::len);
        self.status = format!(
            "Context · node {} · {direct} direct · {indirect} two-hop · {} relationships · Esc restores overview",
            snapshot.nodes.ids[index],
            focus.edges.len()
        )
        .into();
        self.begin_presentation_motion(window, cx);
    }

    fn activate_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self
            .query_input
            .read(cx)
            .focus_handle(cx)
            .is_focused(window)
        {
            return;
        }
        if matches!(self.path_state, PathState::ChoosingEnd(_)) {
            if let Some(index) = self.path_destination_candidate() {
                self.trace_path_to_node(index, window, cx);
            }
            return;
        }
        if let Some(index) = self.selection.and_then(SceneSelection::node) {
            self.focus_node_context(index, window, cx);
        }
    }

    fn show_overview(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.search_generation = self.search_generation.wrapping_add(1);
        drop(self.search_task.take());
        self.search_state = SearchState::Idle;
        self.path_generation = self.path_generation.wrapping_add(1);
        drop(self.path_task.take());
        self.path_state = PathState::Idle;
        self.path_endpoints = None;
        self.context_focus_active = false;
        self.active_facet = None;
        self.set_selection(None);
        if let Some(workspace) = self.workspace.as_ref() {
            workspace.borrow_mut().restore_layout();
        }
        if let Some(snapshot) = self.snapshot() {
            self.retarget_lens(
                vec![0.0; snapshot.nodes.ids.len()],
                vec![0.0; snapshot.edges.ids.len()],
                0.0,
            );
        }
        if let Some(camera) = self.saved_overview_camera {
            self.camera_motion.retarget(camera);
        }
        self.status = "Returning to overview…".into();
        window.focus(&self.focus_handle, cx);
        self.begin_presentation_motion(window, cx);
    }

    fn escape(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.context_focus_active
            || self.active_facet.is_some()
            || !matches!(self.search_state, SearchState::Idle)
            || !matches!(self.path_state, PathState::Idle)
        {
            self.show_overview(window, cx);
        } else {
            self.clear_selection(cx);
            window.focus(&self.focus_handle, cx);
        }
    }

    fn focus_search(&self, window: &mut Window, cx: &mut Context<Self>) {
        self.query_input.read(cx).focus_handle(cx).focus(window, cx);
    }

    fn set_search_mode(&mut self, mode: SearchMode, window: &mut Window, cx: &mut Context<Self>) {
        if self.search_mode == mode {
            return;
        }
        self.search_mode = mode;
        let query = self.query_input.read(cx).value().trim().to_string();
        if query.is_empty() {
            cx.notify();
        } else {
            self.start_search(query, mode, window, cx);
        }
    }

    fn move_search_selection(&mut self, delta: isize, cx: &mut Context<Self>) {
        let SearchState::Ready(report) = &self.search_state else {
            return;
        };
        if report.hits.is_empty() {
            return;
        }
        self.selected_search_result = self
            .selected_search_result
            .saturating_add_signed(delta)
            .min(report.hits.len() - 1);
        cx.notify();
    }

    pub fn activate_search_result(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let SearchState::Ready(report) = &self.search_state else {
            return;
        };
        let Some(hit) = report.hits.get(index).cloned() else {
            return;
        };
        let Some(snapshot) = self.snapshot() else {
            return;
        };
        let Some(selection) = hit.scene_selection(snapshot) else {
            self.status = format!(
                "{} {} matches, but is outside this bounded graph view",
                hit.kind_label().to_lowercase(),
                hit.id()
            )
            .into();
            cx.notify();
            return;
        };
        self.set_selection(Some(selection));
        self.selected_search_result = index;
        self.focus_search_result(selection, window, cx);
        self.status = format!(
            "Focused {} {} · {} · {:.0}% match · {}-element context",
            hit.kind_label().to_lowercase(),
            hit.id(),
            hit.label,
            hit.score * 100.0,
            self.snapshot()
                .and_then(|snapshot| snapshot.focus_neighborhood(selection, 96))
                .map_or(0, |focus| focus.nodes.len() + focus.edges.len())
        )
        .into();
    }

    fn start_search(
        &mut self,
        query: String,
        mode: SearchMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.context_focus_active
            || self.active_facet.is_some()
            || !matches!(self.path_state, PathState::Idle)
        {
            let overview_camera = self.saved_overview_camera.unwrap_or(self.camera);
            if let Some(workspace) = self.workspace.as_ref() {
                workspace.borrow_mut().restore_layout();
            }
            if let Some(snapshot) = self.snapshot() {
                self.retarget_lens(
                    vec![0.0; snapshot.nodes.ids.len()],
                    vec![0.0; snapshot.edges.ids.len()],
                    0.0,
                );
            }
            if let Some(camera) = self.saved_overview_camera {
                self.camera_motion.retarget(camera);
            }
            self.context_focus_active = false;
            self.active_facet = None;
            self.path_endpoints = None;
            self.saved_overview_camera = Some(overview_camera);
            self.begin_presentation_motion(window, cx);
        }
        self.path_generation = self.path_generation.wrapping_add(1);
        drop(self.path_task.take());
        self.path_state = PathState::Idle;
        let Some(path) = self.database_path.clone() else {
            self.search_state = SearchState::Failed {
                query: query.into(),
                error: "Open a .vg database to search its complete contents".into(),
            };
            self.status = "Search needs an open database".into();
            cx.notify();
            return;
        };
        self.search_generation = self.search_generation.wrapping_add(1);
        let generation = self.search_generation;
        let embedding_model = self.embedding_model.clone();
        self.selected_search_result = 0;
        self.search_state = SearchState::Searching {
            query: query.clone().into(),
            mode,
        };
        self.status = format!("{} search…", mode.label()).into();
        self.search_task = Some(cx.spawn_in(window, async move |this, cx| {
            let worker_query = query.clone();
            let result = cx
                .background_spawn(async move {
                    search_database(&path, &worker_query, mode, &embedding_model, 24)
                })
                .await;
            this.update_in(cx, |this, window, cx| {
                if this.search_generation != generation {
                    return;
                }
                match result {
                    Ok(report) => {
                        let count = report.hits.len();
                        let elapsed_ms = report.elapsed.as_secs_f64() * 1_000.0;
                        this.status = if count == 0 {
                            format!(
                                "No {} matches · {elapsed_ms:.1} ms",
                                mode.label().to_lowercase()
                            )
                            .into()
                        } else {
                            format!(
                                "{count} {} matches · {elapsed_ms:.1} ms",
                                mode.label().to_lowercase()
                            )
                            .into()
                        };
                        let report = Arc::new(report);
                        this.present_search_report(&report, window, cx);
                        this.search_state = SearchState::Ready(report);
                    }
                    Err(error) => {
                        this.status = "Search failed".into();
                        this.search_state = SearchState::Failed {
                            query: query.into(),
                            error: error.into(),
                        };
                    }
                }
                cx.notify();
                if let Some(callback) = this.on_search_ready.take() {
                    callback(window, cx);
                }
            })
            .ok();
        }));
        cx.notify();
    }

    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&self.focus_handle, cx);
        self.camera_motion.cancel_at(self.camera);
        let Some(bounds) = self.canvas_bounds else {
            return;
        };
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        let Some(workspace) = self.workspace.clone() else {
            return;
        };
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let local = Vec2::new(
            (event.position.x - bounds.origin.x).into(),
            (event.position.y - bounds.origin.y).into(),
        );
        let hit = {
            let workspace = workspace.borrow();
            hit_test_positions(
                workspace.positions(),
                self.camera,
                self.world_bounds,
                viewport,
                local,
                15.0,
            )
        };
        if let Some(index) = hit {
            if event.click_count >= 2 {
                self.drag = None;
                self.focus_node_context(index, window, cx);
                cx.stop_propagation();
                return;
            }
            let pointer_world = self.camera.unproject(local, self.world_bounds, viewport);
            let mut workspace = workspace.borrow_mut();
            let position = workspace.position(index).unwrap_or(pointer_world);
            let was_pinned = workspace.begin_drag(index);
            self.set_selection(Some(SceneSelection::Node(index)));
            let node_id = snapshot.nodes.ids[index];
            self.status = match self.path_state {
                PathState::ChoosingEnd(draft) => self.path_draft_status(draft),
                PathState::Idle
                | PathState::Searching { .. }
                | PathState::Ready(_)
                | PathState::Failed { .. } => {
                    format!("Selected node {node_id} · drag to arrange").into()
                }
            };
            self.drag = Some(DragState::Node {
                index,
                grab_offset: position - pointer_world,
                last: event.position,
                distance: 0.0,
                was_pinned,
            });
        } else {
            let pressed_edge = (detail_level(self.camera, snapshot.nodes.ids.len())
                != DetailLevel::Overview)
                .then(|| {
                    let workspace = workspace.borrow();
                    hit_test_edges(
                        workspace.positions(),
                        &snapshot.edges,
                        self.camera,
                        self.world_bounds,
                        viewport,
                        local,
                        7.0,
                    )
                })
                .flatten();
            self.drag = Some(DragState::Canvas {
                last: event.position,
                distance: 0.0,
                pressed_edge,
            });
        }
        cx.notify();
    }

    fn on_navigator_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.stop_propagation();
        let Some(bounds) = self.navigator_bounds else {
            return;
        };
        let center = navigator_world_position(bounds, self.world_bounds, event.position);
        self.camera_motion.retarget(Camera {
            center,
            zoom: self.camera.zoom,
        });
        self.status = "Navigator · recentering graph view".into();
        window.focus(&self.focus_handle, cx);
        self.begin_presentation_motion(window, cx);
    }

    fn on_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !event.dragging() {
            return;
        }
        let Some(drag) = self.drag.take() else {
            return;
        };
        let Some(bounds) = self.canvas_bounds else {
            self.drag = Some(drag);
            return;
        };
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        match drag {
            DragState::Canvas {
                last,
                distance,
                pressed_edge,
            } => {
                let delta = Vec2::new(
                    (event.position.x - last.x).into(),
                    (event.position.y - last.y).into(),
                );
                self.camera.pan_screen(delta, self.world_bounds, viewport);
                self.drag = Some(DragState::Canvas {
                    last: event.position,
                    distance: distance + delta.length(),
                    pressed_edge,
                });
            }
            DragState::Node {
                index,
                grab_offset,
                last,
                distance,
                was_pinned,
            } => {
                let delta = Vec2::new(
                    (event.position.x - last.x).into(),
                    (event.position.y - last.y).into(),
                );
                let local = Vec2::new(
                    (event.position.x - bounds.origin.x).into(),
                    (event.position.y - bounds.origin.y).into(),
                );
                let position =
                    self.camera.unproject(local, self.world_bounds, viewport) + grab_offset;
                if let Some(workspace) = self.workspace.as_ref() {
                    workspace.borrow_mut().drag_to(
                        index,
                        position,
                        LayoutOptions::default().edge_length,
                    );
                    self.last_motion_frame = Some(Instant::now());
                }
                self.drag = Some(DragState::Node {
                    index,
                    grab_offset,
                    last: event.position,
                    distance: distance + delta.length(),
                    was_pinned,
                });
            }
        }
        cx.notify();
    }

    fn on_mouse_up(&mut self, event: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(drag) = self.drag.take() else {
            return;
        };
        match drag {
            DragState::Node {
                index,
                distance,
                was_pinned,
                ..
            } => {
                if distance < 4.0 {
                    if let Some(workspace) = self.workspace.as_ref() {
                        workspace.borrow_mut().restore_pin(index, was_pinned);
                    }
                } else {
                    self.status = "Node pinned · Release returns it to the active layout".into();
                }
            }
            DragState::Canvas {
                distance,
                pressed_edge,
                ..
            } if distance < 4.0 => {
                if let Some(edge_index) = pressed_edge {
                    self.set_selection(Some(SceneSelection::Edge(edge_index)));
                    if let Some(snapshot) = self.snapshot() {
                        self.status = format!(
                            "Selected relationship {} · {}",
                            snapshot.edges.ids[edge_index], snapshot.edges.labels[edge_index]
                        )
                        .into();
                    }
                } else {
                    self.set_selection(None);
                    self.status = format!(
                        "No graph element at ({:.0}, {:.0})",
                        f32::from(event.position.x),
                        f32::from(event.position.y)
                    )
                    .into();
                }
            }
            DragState::Canvas { .. } => {}
        }
        cx.notify();
    }

    fn on_scroll(
        &mut self,
        event: &ScrollWheelEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.snapshot().is_none() {
            return;
        }
        self.camera_motion.cancel_at(self.camera);
        let Some(bounds) = self.canvas_bounds else {
            return;
        };
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let delta = event.delta.pixel_delta(px(20.0));
        let zoom_gesture =
            !event.delta.precise() || event.modifiers.platform || event.modifiers.control;
        if zoom_gesture {
            let local = Vec2::new(
                (event.position.x - bounds.origin.x).into(),
                (event.position.y - bounds.origin.y).into(),
            );
            let delta_y: f32 = delta.y.into();
            let factor = (-delta_y * if event.delta.precise() { 0.006 } else { 0.035 }).exp();
            self.camera
                .zoom_about(local, factor, self.world_bounds, viewport);
        } else {
            let delta_x: f32 = delta.x.into();
            let delta_y: f32 = delta.y.into();
            self.camera
                .pan_screen(Vec2::new(-delta_x, -delta_y), self.world_bounds, viewport);
        }
        cx.notify();
    }

    fn on_pinch(&mut self, event: &PinchEvent, _window: &mut Window, cx: &mut Context<Self>) {
        if self.snapshot().is_none() {
            return;
        }
        self.camera_motion.cancel_at(self.camera);
        let Some(bounds) = self.canvas_bounds else {
            return;
        };
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let local = Vec2::new(
            (event.position.x - bounds.origin.x).into(),
            (event.position.y - bounds.origin.y).into(),
        );
        self.camera.zoom_about(
            local,
            (1.0 + event.delta).max(0.2),
            self.world_bounds,
            viewport,
        );
        cx.notify();
    }

    fn on_query_event(
        &mut self,
        input: &Entity<InputState>,
        event: &InputEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !matches!(event, InputEvent::PressEnter { .. }) {
            return;
        }
        let query = input.read(cx).value().trim().to_string();
        if matches!(
            &self.search_state,
            SearchState::Ready(report)
                if report.query.as_ref() == query && report.mode == self.search_mode
        ) {
            self.activate_search_result(self.selected_search_result, window, cx);
            return;
        }
        self.execute_command(&query, window, cx);
    }

    pub fn execute_command(&mut self, query: &str, window: &mut Window, cx: &mut Context<Self>) {
        let mut parts = query.split_whitespace();
        match parts.next() {
            Some("fit") => self.fit(cx),
            Some("zoom") => {
                let Some(zoom) = parts.next().and_then(|zoom| zoom.parse::<f32>().ok()) else {
                    self.status =
                        format!("Expected: zoom <{MIN_CAMERA_ZOOM}..{MAX_CAMERA_ZOOM}>").into();
                    cx.notify();
                    return;
                };
                self.camera.zoom = zoom.clamp(MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM);
                self.status = format!("Zoom set to {:.0}%", self.camera.zoom * 100.0).into();
                cx.notify();
            }
            Some("clear") => self.clear_selection(cx),
            Some("release") => self.release_selected(cx),
            Some("layout") => match parts.next() {
                Some("auto") => self.arrange(LayoutKind::Auto, window, cx),
                Some("force") => self.arrange(LayoutKind::Force, window, cx),
                Some("structure") => self.arrange(LayoutKind::Structure, window, cx),
                Some("orbit") => self.arrange(LayoutKind::Orbit, window, cx),
                _ => {
                    self.status =
                        "Expected: layout auto | layout force | layout structure | layout orbit"
                            .into();
                    cx.notify();
                }
            },
            Some("node") => {
                let Some(id) = parts.next().and_then(|id| id.parse().ok()) else {
                    self.status = "Expected: node <numeric-id>".into();
                    cx.notify();
                    return;
                };
                let Some(snapshot) = self.snapshot() else {
                    return;
                };
                let selection = snapshot.node_index(id).map(SceneSelection::Node);
                self.set_selection(selection);
                self.status = if self.selection.is_some() {
                    format!("Selected node {id}").into()
                } else {
                    format!("Node {id} is not in this view").into()
                };
                cx.notify();
            }
            Some("center") => {
                let Some(id) = parts.next().and_then(|id| id.parse().ok()) else {
                    self.status = "Expected: center <numeric-node-id>".into();
                    cx.notify();
                    return;
                };
                let Some(index) = self.snapshot().and_then(|snapshot| snapshot.node_index(id))
                else {
                    self.status = format!("Node {id} is not in this view").into();
                    cx.notify();
                    return;
                };
                let Some(position) = self
                    .workspace
                    .as_ref()
                    .and_then(|workspace| workspace.borrow().position(index))
                else {
                    return;
                };
                self.camera.center = position;
                self.camera_motion.cancel_at(self.camera);
                self.set_selection(Some(SceneSelection::Node(index)));
                self.status = format!("Centered node {id}").into();
                cx.notify();
            }
            Some("focus") | Some("context") => {
                let Some(id) = parts.next().and_then(|id| id.parse().ok()) else {
                    self.status = "Expected: focus <numeric-node-id>".into();
                    cx.notify();
                    return;
                };
                let Some(index) = self.snapshot().and_then(|snapshot| snapshot.node_index(id))
                else {
                    self.status = format!("Node {id} is not in this view").into();
                    cx.notify();
                    return;
                };
                self.focus_node_context(index, window, cx);
            }
            Some("edge") | Some("relationship") => {
                let Some(id) = parts.next().and_then(|id| id.parse().ok()) else {
                    self.status = "Expected: edge <numeric-id>".into();
                    cx.notify();
                    return;
                };
                let Some(snapshot) = self.snapshot() else {
                    return;
                };
                let selection = snapshot.edge_index(id).map(SceneSelection::Edge);
                self.set_selection(selection);
                self.status = if self.selection.is_some() {
                    format!("Selected relationship {id}").into()
                } else {
                    format!("Relationship {id} is not in this view").into()
                };
                cx.notify();
            }
            Some("path-start") => {
                let expected = "Expected: path-start <node-id> [both|out|in] [max-hops]";
                let Some(id) = parts.next().and_then(|id| id.parse::<u64>().ok()) else {
                    self.status = expected.into();
                    cx.notify();
                    return;
                };
                let direction = match parts.next() {
                    None | Some("both") => PathDirection::Both,
                    Some("out") | Some("outgoing") => PathDirection::Outgoing,
                    Some("in") | Some("incoming") => PathDirection::Incoming,
                    Some(_) => {
                        self.status = expected.into();
                        cx.notify();
                        return;
                    }
                };
                let max_hops = match parts.next() {
                    None => 6,
                    Some(value) => match value.parse::<usize>() {
                        Ok(value) if (1..=64).contains(&value) => value,
                        _ => {
                            self.status = "Path max-hops must be an integer from 1 to 64".into();
                            cx.notify();
                            return;
                        }
                    },
                };
                if parts.next().is_some() {
                    self.status = expected.into();
                    cx.notify();
                    return;
                }
                let Some(index) = self.snapshot().and_then(|snapshot| snapshot.node_index(id))
                else {
                    self.status = format!("Node {id} is not in this view").into();
                    cx.notify();
                    return;
                };
                self.choose_path_start(index, window, cx);
                self.set_path_direction(direction, cx);
                self.set_path_max_hops(max_hops, cx);
            }
            Some("path") => {
                let expected = "Expected: path <start-node-id> <end-node-id> [both|out|in] [relationship-label|-] [max-hops]";
                let Some(start) = parts.next().and_then(|id| id.parse::<u64>().ok()) else {
                    self.status = expected.into();
                    cx.notify();
                    return;
                };
                let Some(end) = parts.next().and_then(|id| id.parse::<u64>().ok()) else {
                    self.status = expected.into();
                    cx.notify();
                    return;
                };
                let direction = match parts.next() {
                    None | Some("both") => PathDirection::Both,
                    Some("out") | Some("outgoing") => PathDirection::Outgoing,
                    Some("in") | Some("incoming") => PathDirection::Incoming,
                    Some(_) => {
                        self.status = expected.into();
                        cx.notify();
                        return;
                    }
                };
                let relationship_label = parts
                    .next()
                    .filter(|label| *label != "-")
                    .map(str::to_string);
                let max_hops = match parts.next() {
                    None => 6,
                    Some(value) => match value.parse::<usize>() {
                        Ok(value) if value <= 64 => value,
                        _ => {
                            self.status = "Path max-hops must be an integer from 0 to 64".into();
                            cx.notify();
                            return;
                        }
                    },
                };
                if parts.next().is_some() {
                    self.status = expected.into();
                    cx.notify();
                    return;
                }
                self.start_evidence_path(
                    EvidencePathQuery {
                        start,
                        end,
                        direction,
                        relationship_label,
                        max_hops,
                    },
                    window,
                    cx,
                );
            }
            Some("facet") | Some("lens") => {
                let kind = parts.next();
                let label = parts.collect::<Vec<_>>().join(" ");
                if label.is_empty() {
                    self.status =
                        "Expected: facet node <label> | facet relationship <label>".into();
                    cx.notify();
                    return;
                }
                let facet = match kind {
                    Some("node") | Some("label") => FacetLens::NodeLabel(label.clone().into()),
                    Some("edge") | Some("relationship") => {
                        FacetLens::Relationship(label.clone().into())
                    }
                    _ => {
                        self.status =
                            "Expected: facet node <label> | facet relationship <label>".into();
                        cx.notify();
                        return;
                    }
                };
                let exists = match &facet {
                    FacetLens::NodeLabel(label) => self
                        .node_label_counts
                        .iter()
                        .any(|(candidate, _)| candidate == label),
                    FacetLens::Relationship(label) => self
                        .relationship_counts
                        .iter()
                        .any(|(candidate, _)| candidate == label),
                };
                if exists {
                    self.activate_facet(facet, window, cx);
                } else {
                    self.status =
                        format!("No visible {} named {label:?}", facet.kind_label()).into();
                    cx.notify();
                }
            }
            Some("text") => {
                let query = parts.collect::<Vec<_>>().join(" ");
                self.search_mode = SearchMode::Text;
                self.start_search(query, SearchMode::Text, window, cx);
            }
            Some("semantic") => {
                let query = parts.collect::<Vec<_>>().join(" ");
                self.search_mode = SearchMode::Semantic;
                self.start_search(query, SearchMode::Semantic, window, cx);
            }
            Some("hybrid") | Some("search") => {
                let query = parts.collect::<Vec<_>>().join(" ");
                self.search_mode = SearchMode::Hybrid;
                self.start_search(query, SearchMode::Hybrid, window, cx);
            }
            Some(_) => self.start_search(query.to_string(), self.search_mode, window, cx),
            None => {}
        }
    }
}

impl Render for StudioView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let presentation_moving = self.is_presentation_moving();
        if presentation_moving {
            let now = Instant::now();
            let elapsed = self
                .last_motion_frame
                .replace(now)
                .map_or(0.0, |last| (now - last).as_secs_f32());
            let reduce_motion = cx.reduce_motion();
            let workspace_moving = self
                .workspace
                .as_ref()
                .is_some_and(|workspace| workspace.borrow_mut().step(elapsed, reduce_motion));
            let camera_moving = self
                .camera_motion
                .step(&mut self.camera, elapsed, reduce_motion);
            let lens_moving = self
                .lens
                .as_mut()
                .is_some_and(|lens| lens.step(elapsed, reduce_motion));
            let lens_cleared = self.lens.as_ref().is_some_and(LensTransition::is_cleared);
            if lens_cleared {
                self.lens = None;
                self.saved_overview_camera = None;
                if matches!(self.search_state, SearchState::Idle) {
                    self.status = "Overview · full graph restored".into();
                }
            }
            if workspace_moving || camera_moving || lens_moving {
                window.request_animation_frame();
            } else {
                self.last_motion_frame = None;
            }
        } else {
            self.last_motion_frame = None;
        }
        let viewport_width = window.viewport_size().width;
        let show_inspector = viewport_width >= px(980.0);
        let compact_toolbar = viewport_width < px(1_280.0);
        let show_bezel_controls = !compact_toolbar;
        bezel_ui::focus::traversal(v_flex())
            .id("vecgra-studio")
            .role(Role::Application)
            .aria_label("Vecgra Studio")
            .key_context("VecgraStudio")
            .track_focus(&self.focus_handle)
            .size_full()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .on_action(cx.listener(|this, _: &FitView, _, cx| this.fit(cx)))
            .on_action(cx.listener(|this, _: &ZoomIn, _, cx| this.zoom(1.22, cx)))
            .on_action(cx.listener(|this, _: &ZoomOut, _, cx| this.zoom(0.82, cx)))
            .on_action(cx.listener(|this, _: &ClearSelection, window, cx| {
                this.escape(window, cx);
            }))
            .on_action(cx.listener(|this, _: &ArrangeAuto, window, cx| {
                this.arrange(LayoutKind::Auto, window, cx);
            }))
            .on_action(cx.listener(|this, _: &ArrangeForce, window, cx| {
                this.arrange(LayoutKind::Force, window, cx);
            }))
            .on_action(cx.listener(|this, _: &ArrangeStructure, window, cx| {
                this.arrange(LayoutKind::Structure, window, cx);
            }))
            .on_action(cx.listener(|this, _: &ArrangeOrbit, window, cx| {
                this.arrange(LayoutKind::Orbit, window, cx);
            }))
            .on_action(cx.listener(|this, _: &ReleaseSelected, _, cx| {
                this.release_selected(cx);
            }))
            .on_action(cx.listener(|this, _: &ActivateSelection, window, cx| {
                this.activate_selection(window, cx);
            }))
            .on_action(cx.listener(|this, _: &FocusSearch, window, cx| {
                this.focus_search(window, cx);
            }))
            .on_action(cx.listener(|this, _: &NextSearchResult, _, cx| {
                this.move_search_selection(1, cx);
            }))
            .on_action(cx.listener(|this, _: &PreviousSearchResult, _, cx| {
                this.move_search_selection(-1, cx);
            }))
            .child(self.render_top_bar(compact_toolbar, cx))
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .child(self.render_left_panel(cx))
                    .child(
                        div()
                            .relative()
                            .flex_1()
                            .h_full()
                            .child(self.render_canvas(show_bezel_controls, cx)),
                    )
                    .when(show_inspector, |this| this.child(self.render_inspector(cx))),
            )
            .child(self.render_status_bar(cx))
    }
}

#[cfg(test)]
#[path = "view_tests.rs"]
mod tests;
