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
use vectorgraph_studio_core::{
    Camera, CameraMotion, DetailLevel, EvidenceNode, EvidencePathReport, EvidencePathStrategy,
    EvidencePathTermination, GraphWorkspace, LayoutKind, LayoutOptions, MAX_CAMERA_ZOOM,
    MIN_CAMERA_ZOOM, PathDirection, PropertyValue, Rect, SceneSelection, SceneSnapshot, SearchMode,
    SearchReport, SnapshotOptions, Vec2, detail_level, evidence_path_database, hit_test_edges,
    hit_test_positions, search_database,
};

use crate::graph_canvas::{
    GraphCanvasPresentation, GraphEmphasis, GraphPathEndpoints, graph_canvas,
};
use crate::theme::{palette, relationship_color};

actions!(
    vectorgraph_studio,
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
    drag: Option<DragState>,
    query_input: Entity<InputState>,
    focus_handle: FocusHandle,
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
        let query_input =
            cx.new(|cx| InputState::new(window, cx).placeholder("Search nodes + relationships…"));
        let subscription = cx.subscribe_in(&query_input, window, Self::on_query_event);
        let focus_handle = cx.focus_handle().tab_stop(true);
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
            drag: None,
            query_input,
            focus_handle,
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
            embedding_model: std::env::var("VG_EMBEDDER")
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

    fn render_brand(&self, database_name: SharedString, cx: &Context<Self>) -> impl IntoElement {
        let colors = palette();
        h_flex()
            .w(px(166.0))
            .flex_shrink_0()
            .gap_2()
            .items_center()
            .child(
                div()
                    .size(px(22.0))
                    .rounded(px(6.0))
                    .bg(colors.cobalt)
                    .text_color(rgb(0xf7fbfd))
                    .text_xs()
                    .font_weight(gpui::FontWeight::BOLD)
                    .flex()
                    .items_center()
                    .justify_center()
                    .child("VG"),
            )
            .child(
                v_flex()
                    .w(px(132.0))
                    .overflow_hidden()
                    .line_height(px(15.0))
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child("Studio"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .truncate()
                            .child(database_name),
                    ),
            )
    }

    fn render_search_modes(&self, cx: &Context<Self>) -> impl IntoElement {
        h_flex()
            .gap_0p5()
            .child(
                Button::new("search-text")
                    .label("Text")
                    .small()
                    .selected(self.search_mode == SearchMode::Text)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.set_search_mode(SearchMode::Text, window, cx);
                    })),
            )
            .child(
                Button::new("search-hybrid")
                    .label("Hybrid")
                    .small()
                    .selected(self.search_mode == SearchMode::Hybrid)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.set_search_mode(SearchMode::Hybrid, window, cx);
                    })),
            )
            .child(
                Button::new("search-semantic")
                    .label("Semantic")
                    .small()
                    .selected(self.search_mode == SearchMode::Semantic)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.set_search_mode(SearchMode::Semantic, window, cx);
                    })),
            )
    }

    fn render_layout_controls(&self, cx: &Context<Self>) -> impl IntoElement {
        h_flex()
            .gap_1()
            .child(
                Button::new("layout-auto")
                    .label("Auto")
                    .small()
                    .selected(self.layout_kind == LayoutKind::Auto)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.arrange(LayoutKind::Auto, window, cx);
                    })),
            )
            .child(
                Button::new("layout-force")
                    .label("Force")
                    .small()
                    .selected(self.layout_kind == LayoutKind::Force)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.arrange(LayoutKind::Force, window, cx);
                    })),
            )
            .child(
                Button::new("layout-structure")
                    .label("Structure")
                    .small()
                    .selected(self.layout_kind == LayoutKind::Structure)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.arrange(LayoutKind::Structure, window, cx);
                    })),
            )
            .child(
                Button::new("layout-orbit")
                    .label("Orbit")
                    .small()
                    .selected(self.layout_kind == LayoutKind::Orbit)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.arrange(LayoutKind::Orbit, window, cx);
                    })),
            )
            .child(
                Button::new("release-node")
                    .label("Release")
                    .small()
                    .disabled(
                        !self
                            .selection
                            .and_then(SceneSelection::node)
                            .is_some_and(|index| {
                                self.workspace
                                    .as_ref()
                                    .is_some_and(|workspace| workspace.borrow().is_pinned(index))
                            }),
                    )
                    .on_click(cx.listener(|this, _, _, cx| this.release_selected(cx))),
            )
    }

    fn render_zoom_controls(&self, cx: &Context<Self>) -> impl IntoElement {
        h_flex()
            .gap_1()
            .child(
                Button::new("zoom-out")
                    .label("−")
                    .small()
                    .on_click(cx.listener(|this, _, _, cx| this.zoom(0.82, cx))),
            )
            .child(
                Button::new("fit-view")
                    .label("Fit")
                    .small()
                    .on_click(cx.listener(|this, _, _, cx| this.fit(cx))),
            )
            .child(
                Button::new("zoom-in")
                    .label("+")
                    .small()
                    .on_click(cx.listener(|this, _, _, cx| this.zoom(1.22, cx))),
            )
    }

    fn render_top_bar(&self, compact: bool, cx: &Context<Self>) -> gpui::AnyElement {
        let database_name: SharedString = match &self.state {
            LoadState::Ready(snapshot) => snapshot.database_name.to_string().into(),
            LoadState::Loading { name } => name.clone(),
            LoadState::Failed(_) => "No database".into(),
        };
        if compact {
            return TitleBar::new()
                .h(px(88.0))
                .child(
                    v_flex()
                        .size_full()
                        .pr_3()
                        .child(
                            h_flex()
                                .debug_selector(|| "compact-toolbar-primary".into())
                                .h(px(48.0))
                                .gap_2()
                                .items_center()
                                .child(self.render_brand(database_name, cx))
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w(px(180.0))
                                        .child(Input::new(&self.query_input).small()),
                                )
                                .child(self.render_zoom_controls(cx)),
                        )
                        .child(
                            h_flex()
                                .debug_selector(|| "compact-toolbar-secondary".into())
                                .h(px(40.0))
                                .pl_2()
                                .gap_2()
                                .items_center()
                                .border_t_1()
                                .border_color(cx.theme().title_bar_border)
                                .child(self.render_search_modes(cx))
                                .child(div().flex_1())
                                .child(self.render_layout_controls(cx)),
                        ),
                )
                .into_any_element();
        }
        TitleBar::new()
            .h(px(52.0))
            .child(
                h_flex()
                    .debug_selector(|| "wide-toolbar".into())
                    .size_full()
                    .pr_3()
                    .gap_2()
                    .items_center()
                    .child(self.render_brand(database_name, cx))
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(180.0))
                            .max_w(px(620.0))
                            .child(Input::new(&self.query_input).small()),
                    )
                    .child(self.render_search_modes(cx))
                    .child(div().flex_1())
                    .child(self.render_layout_controls(cx))
                    .child(self.render_zoom_controls(cx)),
            )
            .into_any_element()
    }

    fn render_left_panel(&self, cx: &Context<Self>) -> impl IntoElement {
        let label_counts = self.node_label_counts.clone();
        let relationship_counts = self.relationship_counts.clone();
        let active_relationship = match self.active_facet.as_ref() {
            Some(FacetLens::Relationship(label)) => Some(label),
            Some(FacetLens::NodeLabel(_)) | None => None,
        };
        let active_node_label = match self.active_facet.as_ref() {
            Some(FacetLens::NodeLabel(label)) => Some(label),
            Some(FacetLens::Relationship(_)) | None => None,
        };
        let relationship_facets = visible_facet_counts(&relationship_counts, active_relationship);
        let node_label_facets = visible_facet_counts(&label_counts, active_node_label);
        let showing_search = !matches!(self.search_state, SearchState::Idle);
        let showing_path = !matches!(self.path_state, PathState::Idle);
        let showing_context = self.context_focus_active && !showing_search && !showing_path;
        let showing_facet = self.active_facet.is_some() && !showing_search && !showing_path;
        v_flex()
            .w(if showing_search || showing_path {
                px(306.0)
            } else {
                px(218.0)
            })
            .h_full()
            .flex_shrink_0()
            .border_r_1()
            .border_color(cx.theme().sidebar_border)
            .bg(cx.theme().sidebar)
            .child(section_label("VIEW"))
            .child(
                div()
                    .id("overview-navigation")
                    .cursor_pointer()
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.show_overview(window, cx);
                    }))
                    .child(nav_row(
                        "Overview",
                        !showing_search && !showing_path && !showing_context && !showing_facet,
                        cx,
                    )),
            )
            .when(!showing_path, |this| {
                this.child(nav_row("Search results", showing_search, cx))
            })
            .when(showing_path, |this| {
                this.child(nav_row("Evidence path", true, cx))
            })
            .when(showing_context, |this| {
                this.child(nav_row("2-hop context", true, cx))
            })
            .when(showing_facet, |this| {
                this.child(nav_row("Facet lens", true, cx))
            })
            .when(showing_search, |this| {
                this.child(self.render_search_results(cx))
            })
            .when(showing_path, |this| {
                this.child(self.render_evidence_path(cx))
            })
            .when(!showing_search && !showing_path, |this| {
                this.child(section_label("RELATIONSHIPS"))
            })
            .when(!showing_search && !showing_path, |this| {
                this.child(
                    v_flex()
                        .id("relationship-facets")
                        .role(Role::List)
                        .aria_label("Relationship type facets")
                        .px_2()
                        .children(relationship_facets.iter().map(|(label, count)| {
                            let active = self.active_facet.as_ref()
                                == Some(&FacetLens::Relationship(label.clone()));
                            let facet = FacetLens::Relationship(label.clone());
                            Button::new(format!("relationship-facet:{label}"))
                                .ghost()
                                .small()
                                .compact()
                                .w_full()
                                .label(label.to_string())
                                .accessibility_id(format!("relationship-facet:{label}"))
                                .toggled(active)
                                .selected(active)
                                .cursor_pointer()
                                .when(active, |this| {
                                    this.border_l_2().border_color(cx.theme().ring)
                                })
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.activate_facet(facet.clone(), window, cx);
                                }))
                                .child(
                                    h_flex()
                                        .flex_1()
                                        .w_full()
                                        .gap_2()
                                        .items_center()
                                        .child(
                                            div()
                                                .size(px(7.0))
                                                .rounded_full()
                                                .bg(relationship_color(label)),
                                        )
                                        .child(div().flex_1())
                                        .when(active, |this| {
                                            this.child(
                                                div()
                                                    .font_family(
                                                        cx.theme().mono_font_family.clone(),
                                                    )
                                                    .text_xs()
                                                    .font_weight(gpui::FontWeight::BOLD)
                                                    .text_color(cx.theme().ring)
                                                    .child("LENS"),
                                            )
                                        })
                                        .child(
                                            div()
                                                .font_family(cx.theme().mono_font_family.clone())
                                                .text_xs()
                                                .text_color(cx.theme().muted_foreground)
                                                .child(format_count(*count)),
                                        ),
                                )
                        })),
                )
            })
            .when(!showing_search && !showing_path, |this| {
                this.child(section_label("NODE LABELS"))
            })
            .when(!showing_search && !showing_path, |this| {
                this.child(
                    v_flex()
                        .id("node-label-facets")
                        .role(Role::List)
                        .aria_label("Node label facets")
                        .px_2()
                        .children(node_label_facets.iter().map(|(label, count)| {
                            let active = self.active_facet.as_ref()
                                == Some(&FacetLens::NodeLabel(label.clone()));
                            let facet = FacetLens::NodeLabel(label.clone());
                            Button::new(format!("node-label-facet:{label}"))
                                .ghost()
                                .small()
                                .compact()
                                .w_full()
                                .label(label.to_string())
                                .accessibility_id(format!("node-label-facet:{label}"))
                                .toggled(active)
                                .selected(active)
                                .cursor_pointer()
                                .when(active, |this| {
                                    this.border_l_2().border_color(cx.theme().ring)
                                })
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.activate_facet(facet.clone(), window, cx);
                                }))
                                .child(
                                    h_flex()
                                        .flex_1()
                                        .w_full()
                                        .items_center()
                                        .child(div().flex_1())
                                        .when(active, |this| {
                                            this.child(
                                                div()
                                                    .mr_2()
                                                    .font_family(
                                                        cx.theme().mono_font_family.clone(),
                                                    )
                                                    .text_xs()
                                                    .font_weight(gpui::FontWeight::BOLD)
                                                    .text_color(cx.theme().ring)
                                                    .child("LENS"),
                                            )
                                        })
                                        .child(
                                            div()
                                                .font_family(cx.theme().mono_font_family.clone())
                                                .text_xs()
                                                .text_color(cx.theme().muted_foreground)
                                                .child(format_count(*count)),
                                        ),
                                )
                        })),
                )
            })
    }

    fn render_evidence_path(&self, cx: &Context<Self>) -> gpui::AnyElement {
        match &self.path_state {
            PathState::Idle => div().into_any_element(),
            PathState::ChoosingEnd(draft) => {
                let origin = self.snapshot().and_then(|snapshot| {
                    snapshot.node_index(draft.start).map(|index| {
                        (
                            snapshot.nodes.labels[index].clone(),
                            scene_node_title(snapshot, index),
                        )
                    })
                });
                let destination = self.path_destination_candidate().and_then(|index| {
                    let snapshot = self.snapshot()?;
                    Some((
                        index,
                        *snapshot.nodes.ids.get(index)?,
                        snapshot.nodes.labels.get(index)?.clone(),
                        scene_node_title(snapshot, index),
                    ))
                });
                let has_destination = destination.is_some();
                v_flex()
                    .flex_1()
                    .min_h_0()
                    .p_4()
                    .gap_2()
                    .child(
                        h_flex()
                            .items_center()
                            .child(
                                div()
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(palette().celadon)
                                    .child("ORIGIN SET"),
                            )
                            .child(div().flex_1())
                            .child(
                                div()
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(if has_destination {
                                        "ENTER RUNS · ESC CANCELS"
                                    } else {
                                        "ESC CANCELS"
                                    }),
                            ),
                    )
                    .when_some(origin, |this, (label, title)| {
                        this.child(path_endpoint_card_data(
                            "FROM",
                            draft.start,
                            &label,
                            &title,
                            palette().celadon,
                            cx,
                        ))
                    })
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                h_flex()
                                    .items_center()
                                    .child(inspector_label("TRAVERSE"))
                                    .child(div().flex_1())
                                    .child(
                                        div()
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .child(path_direction_compact_label(draft.direction)),
                                    ),
                            )
                            .child(
                                h_flex()
                                    .gap_1()
                                    .child(
                                        Button::new("path-direction-both")
                                            .label("↔ Either")
                                            .small()
                                            .selected(draft.direction == PathDirection::Both)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.set_path_direction(PathDirection::Both, cx);
                                            })),
                                    )
                                    .child(
                                        Button::new("path-direction-outgoing")
                                            .label("→ Out")
                                            .small()
                                            .selected(draft.direction == PathDirection::Outgoing)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.set_path_direction(
                                                    PathDirection::Outgoing,
                                                    cx,
                                                );
                                            })),
                                    )
                                    .child(
                                        Button::new("path-direction-incoming")
                                            .label("← In")
                                            .small()
                                            .selected(draft.direction == PathDirection::Incoming)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.set_path_direction(
                                                    PathDirection::Incoming,
                                                    cx,
                                                );
                                            })),
                                    ),
                            ),
                    )
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                h_flex()
                                    .items_center()
                                    .child(inspector_label("HOP LIMIT"))
                                    .child(div().flex_1())
                                    .child(
                                        div()
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .text_xs()
                                            .font_weight(gpui::FontWeight::BOLD)
                                            .text_color(palette().celadon)
                                            .child(format!("≤ {}", draft.max_hops)),
                                    ),
                            )
                            .child(
                                h_flex()
                                    .gap_1()
                                    .child(path_hop_button(1, draft.max_hops, cx))
                                    .child(path_hop_button(2, draft.max_hops, cx))
                                    .child(path_hop_button(4, draft.max_hops, cx))
                                    .child(path_hop_button(6, draft.max_hops, cx)),
                            ),
                    )
                    .when_some(destination, |this, (index, id, label, title)| {
                        this.child(
                            v_flex()
                                .pt_1()
                                .gap_2()
                                .child(path_endpoint_card_data(
                                    "TO",
                                    id,
                                    &label,
                                    &title,
                                    palette().copper,
                                    cx,
                                ))
                                .child(
                                    div()
                                        .debug_selector(|| "evidence-path-run".into())
                                        .w_full()
                                        .child(
                                            Button::new("evidence-path-run")
                                                .accessibility_id("evidence-path-run")
                                                .label("Trace exact path")
                                                .primary()
                                                .w_full()
                                                .on_click(cx.listener(
                                                    move |this, _, window, cx| {
                                                        this.trace_path_to_node(index, window, cx);
                                                    },
                                                )),
                                        ),
                                ),
                        )
                    })
                    .when(!has_destination, |this| {
                        this.child(
                            v_flex()
                                .gap_1()
                                .child(
                                    div()
                                        .text_sm()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .child("Choose a destination"),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .line_height(px(17.0))
                                        .text_color(cx.theme().muted_foreground)
                                        .child(
                                            "Select another node on the canvas. Its exact-path action appears here.",
                                        ),
                                ),
                        )
                    })
                    .into_any_element()
            }
            PathState::Searching { start, end } => v_flex()
                .flex_1()
                .min_h_0()
                .p_4()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child("Tracing exact evidence path"),
                )
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!("node:{start}  →  node:{end}")),
                )
                .child(
                    div()
                        .text_xs()
                        .line_height(px(17.0))
                        .text_color(cx.theme().muted_foreground)
                        .child("Searching the complete database on a background worker…"),
                )
                .into_any_element(),
            PathState::Failed { start, end, error } => v_flex()
                .flex_1()
                .min_h_0()
                .p_4()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child("Could not trace evidence path"),
                )
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!("node:{start}  →  node:{end}")),
                )
                .child(
                    div()
                        .text_xs()
                        .line_height(px(17.0))
                        .text_color(cx.theme().warning)
                        .child(error.clone()),
                )
                .into_any_element(),
            PathState::Ready(report) => {
                let elapsed_ms = report.elapsed.as_secs_f64() * 1_000.0;
                let (outcome, outcome_color) = match report.termination {
                    EvidencePathTermination::Found => ("EXACT", palette().celadon),
                    EvidencePathTermination::NotFoundWithinHops => {
                        ("NO PATH", cx.theme().muted_foreground)
                    }
                    EvidencePathTermination::ExpansionLimit => ("INCOMPLETE", cx.theme().warning),
                };
                let hop_count = report.path.as_ref().map_or(0, |path| path.steps.len());
                let (visible_nodes, visible_edges) = self.snapshot().map_or((0, 0), |snapshot| {
                    report.path.as_ref().map_or_else(
                        || {
                            (
                                usize::from(snapshot.node_index(report.start.id).is_some())
                                    + usize::from(
                                        report.end.id != report.start.id
                                            && snapshot.node_index(report.end.id).is_some(),
                                    ),
                                0,
                            )
                        },
                        |path| {
                            (
                                path.nodes
                                    .iter()
                                    .filter(|node| snapshot.node_index(node.id).is_some())
                                    .count(),
                                path.steps
                                    .iter()
                                    .filter(|step| snapshot.edge_index(step.edge_id).is_some())
                                    .count(),
                            )
                        },
                    )
                });
                let total_nodes = report.path.as_ref().map_or(2, |path| path.nodes.len());
                let total_edges = report.path.as_ref().map_or(0, |path| path.steps.len());
                let partial = visible_nodes < total_nodes || visible_edges < total_edges;
                let summary = match report.termination {
                    EvidencePathTermination::Found if hop_count == 0 => {
                        format!("Same node · {elapsed_ms:.1} ms")
                    }
                    EvidencePathTermination::Found => format!(
                        "{hop_count} hop{} · {elapsed_ms:.1} ms",
                        if hop_count == 1 { "" } else { "s" }
                    ),
                    EvidencePathTermination::NotFoundWithinHops => {
                        format!("None within {} hops · {elapsed_ms:.1} ms", report.max_hops)
                    }
                    EvidencePathTermination::ExpansionLimit => {
                        format!("Work cap reached · {elapsed_ms:.1} ms")
                    }
                };
                v_flex()
                    .flex_1()
                    .min_h_0()
                    .child(
                        v_flex()
                            .px_4()
                            .pt_3()
                            .pb_3()
                            .gap_2()
                            .border_b_1()
                            .border_color(cx.theme().border)
                            .child(
                                h_flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .text_xs()
                                            .font_weight(gpui::FontWeight::BOLD)
                                            .text_color(outcome_color)
                                            .child(outcome),
                                    )
                                    .child(div().flex_1())
                                    .child(
                                        div()
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .child(summary),
                                    ),
                            )
                            .child(path_endpoint_card("FROM", &report.start, palette().celadon, cx))
                            .child(path_endpoint_card("TO", &report.end, palette().copper, cx))
                            .child(path_plan_diagnostics(report, cx))
                            .when_some(report.relationship_label.clone(), |this, label| {
                                this.child(
                                    div()
                                        .font_family(cx.theme().mono_font_family.clone())
                                        .text_xs()
                                        .text_color(relationship_color(&label))
                                        .child(format!("relationship:{label}")),
                                )
                            })
                            .when(partial, |this| {
                                this.child(
                                    div()
                                        .px_2()
                                        .py_1()
                                        .rounded(px(4.0))
                                        .bg(cx.theme().warning.opacity(0.08))
                                        .text_xs()
                                        .line_height(px(16.0))
                                        .text_color(cx.theme().warning)
                                        .child(format!(
                                            "Exact database result; sampled canvas shows {visible_nodes}/{total_nodes} nodes and {visible_edges}/{total_edges} relationships."
                                        )),
                                )
                            }),
                    )
                    .child(match &report.path {
                        Some(path) if path.steps.is_empty() => v_flex()
                            .flex_1()
                            .p_4()
                            .gap_1()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child("Zero-hop identity"),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .line_height(px(17.0))
                                    .text_color(cx.theme().muted_foreground)
                                    .child("The start and end resolve to the same graph node."),
                            )
                            .into_any_element(),
                        Some(path) => v_flex()
                            .id("evidence-path-steps")
                            .role(Role::List)
                            .aria_label("Ordered exact evidence path steps")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .px_2()
                            .py_1()
                            .children(path.steps.iter().enumerate().map(|(index, step)| {
                                let selected = self.snapshot().is_some_and(|snapshot| {
                                    snapshot.edge_index(step.edge_id).map(SceneSelection::Edge)
                                        == self.selection
                                });
                                let orientation = if step.forward {
                                    "stored direction"
                                } else {
                                    "reverse traversal"
                                };
                                let show_relationship_type =
                                    step.title.as_ref() != step.label.replace('_', " ");
                                let accessible_label = format!(
                                    "Evidence step {}, {}, node {} to node {}, relationship {}, {}",
                                    index + 1,
                                    step.title,
                                    step.from,
                                    step.to,
                                    step.label,
                                    orientation
                                );
                                let ring = cx.theme().ring;
                                div()
                                    .id(("evidence-path-step", step.edge_id))
                                    .focusable()
                                    .tab_stop(true)
                                    .role(Role::Button)
                                    .aria_label(accessible_label)
                                    .aria_selected(selected)
                                    .w_full()
                                    .cursor_pointer()
                                    .mb_1()
                                    .px_2()
                                    .py_1()
                                    .rounded(px(5.0))
                                    .border_l_2()
                                    .border_color(relationship_color(&step.label))
                                    .when(selected, |this| {
                                        this.bg(cx.theme().sidebar_accent)
                                            .border_1()
                                            .border_color(cx.theme().ring)
                                    })
                                    .when(!selected, |this| {
                                        this.hover(|style| style.bg(cx.theme().sidebar_accent))
                                    })
                                    .focus(move |style| style.border_1().border_color(ring))
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.activate_evidence_step(index, window, cx);
                                    }))
                                    .on_key_down(cx.listener(
                                        move |this, event: &gpui::KeyDownEvent, window, cx| {
                                            if !event.keystroke.modifiers.modified()
                                                && matches!(
                                                    event.keystroke.key.as_str(),
                                                    "enter" | "space"
                                                )
                                            {
                                                this.activate_evidence_step(index, window, cx);
                                                cx.stop_propagation();
                                            }
                                        },
                                    ))
                                    .child(
                                        h_flex()
                                            .w_full()
                                            .items_start()
                                            .gap_2()
                                            .child(
                                                div()
                                                    .w(px(22.0))
                                                    .flex_shrink_0()
                                                    .font_family(cx.theme().mono_font_family.clone())
                                                    .text_xs()
                                                    .font_weight(gpui::FontWeight::BOLD)
                                                    .text_color(relationship_color(&step.label))
                                                    .child(format!("{:02}", index + 1)),
                                            )
                                            .child(
                                                v_flex()
                                                    .flex_1()
                                                    .min_w_0()
                                                    .gap_0p5()
                                                    .child(
                                                        div()
                                                            .text_sm()
                                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                                            .truncate()
                                                            .child(step.title.to_string()),
                                                    )
                                                    .when(show_relationship_type, |this| {
                                                        this.child(
                                                            div()
                                                                .font_family(
                                                                    cx.theme()
                                                                        .mono_font_family
                                                                        .clone(),
                                                                )
                                                                .text_xs()
                                                                .text_color(relationship_color(
                                                                    &step.label,
                                                                ))
                                                                .truncate()
                                                                .child(step.label.to_string()),
                                                        )
                                                    })
                                                    .child(
                                                        div()
                                                            .font_family(cx.theme().mono_font_family.clone())
                                                            .text_xs()
                                                            .text_color(cx.theme().muted_foreground)
                                                            .child(format!(
                                                                "node:{} → node:{} · {}",
                                                                step.from, step.to, orientation
                                                            )),
                                                    ),
                                            ),
                                    )
                            }))
                            .into_any_element(),
                        None => v_flex()
                            .flex_1()
                            .p_4()
                            .gap_2()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child(match report.termination {
                                        EvidencePathTermination::NotFoundWithinHops => {
                                            "No evidence chain in this hop bound"
                                        }
                                        EvidencePathTermination::ExpansionLimit => {
                                            "Search stopped before it was conclusive"
                                        }
                                        EvidencePathTermination::Found => "No path payload",
                                    }),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .line_height(px(17.0))
                                    .text_color(match report.termination {
                                        EvidencePathTermination::ExpansionLimit => {
                                            cx.theme().warning
                                        }
                                        EvidencePathTermination::Found
                                        | EvidencePathTermination::NotFoundWithinHops => {
                                            cx.theme().muted_foreground
                                        }
                                    })
                                    .child(match report.termination {
                                        EvidencePathTermination::NotFoundWithinHops => format!(
                                            "The complete search found no matching chain up to {} hops. Increase max-hops to widen the proof boundary.",
                                            report.max_hops
                                        ),
                                        EvidencePathTermination::ExpansionLimit => format!(
                                            "The {}-hop search reached its frontier-work budget. This is not proof of absence.",
                                            report.max_hops
                                        ),
                                        EvidencePathTermination::Found => {
                                            "The database returned an inconsistent path result.".into()
                                        }
                                    }),
                            )
                            .into_any_element(),
                    })
                    .into_any_element()
            }
        }
    }

    fn render_search_results(&self, cx: &Context<Self>) -> gpui::AnyElement {
        match &self.search_state {
            SearchState::Idle => div().into_any_element(),
            SearchState::Searching { query, mode } => v_flex()
                .flex_1()
                .min_h_0()
                .p_4()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child(format!("{} search", mode.label())),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!(
                            "Searching all nodes and relationships for “{query}”…"
                        )),
                )
                .into_any_element(),
            SearchState::Failed { query, error } => v_flex()
                .flex_1()
                .min_h_0()
                .p_4()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child(format!("No results for “{query}”")),
                )
                .child(
                    div()
                        .text_xs()
                        .line_height(px(17.0))
                        .text_color(cx.theme().warning)
                        .child(error.clone()),
                )
                .into_any_element(),
            SearchState::Ready(report) => {
                let summary = format!(
                    "{} {} · {:.1} ms",
                    report.hits.len(),
                    report.mode.label().to_lowercase(),
                    report.elapsed.as_secs_f64() * 1_000.0
                );
                v_flex()
                    .flex_1()
                    .min_h_0()
                    .child(
                        v_flex()
                            .px_4()
                            .pt_3()
                            .pb_2()
                            .gap_1()
                            .child(
                                h_flex()
                                    .items_center()
                                    .child(
                                        div()
                                            .flex_1()
                                            .text_sm()
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .truncate()
                                            .child(format!("“{}”", report.query)),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .child(summary),
                                    ),
                            )
                            .when_some(report.embedding_model.clone(), |this, model| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .text_color(cx.theme().muted_foreground)
                                        .truncate()
                                        .child(format!("Vectors · {model}")),
                                )
                            })
                            .when_some(report.warning.clone(), |this, warning| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .line_height(px(16.0))
                                        .text_color(cx.theme().warning)
                                        .child(warning.to_string()),
                                )
                            }),
                    )
                    .child(
                        v_flex()
                            .id("search-results-scroll")
                            .role(Role::List)
                            .aria_label("Ranked graph search results")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .px_2()
                            .pb_3()
                            .when(report.hits.is_empty(), |this| {
                                this.child(
                                    div()
                                        .p_3()
                                        .text_sm()
                                        .text_color(cx.theme().muted_foreground)
                                        .child("No matching graph elements"),
                                )
                            })
                            .children(report.hits.iter().enumerate().map(|(index, hit)| {
                                let selected = index == self.selected_search_result;
                                let result_id = match hit.kind_label() {
                                    "EDGE" => ("search-result-edge", hit.id()),
                                    _ => ("search-result-node", hit.id()),
                                };
                                let kind_color = if hit.kind_label() == "EDGE" {
                                    relationship_color(&hit.label)
                                } else {
                                    palette().celadon
                                };
                                let accessible_label = format!(
                                    "{} {}, {}, relevance {:.0} percent",
                                    hit.kind_label().to_lowercase(),
                                    hit.title,
                                    hit.label,
                                    hit.score * 100.0
                                );
                                v_flex()
                                    .id(result_id)
                                    .role(Role::ListItem)
                                    .aria_label(accessible_label)
                                    .aria_selected(selected)
                                    .mx_0p5()
                                    .mb_1()
                                    .p_2()
                                    .gap_1()
                                    .rounded(px(5.0))
                                    .cursor_pointer()
                                    .when(selected, |this| {
                                        this.bg(cx.theme().sidebar_accent)
                                            .border_1()
                                            .border_color(cx.theme().ring)
                                    })
                                    .when(!selected, |this| {
                                        this.hover(|style| style.bg(cx.theme().sidebar_accent))
                                    })
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.activate_search_result(index, window, cx);
                                    }))
                                    .child(
                                        h_flex()
                                            .gap_2()
                                            .items_center()
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                                    .text_color(kind_color)
                                                    .child(hit.kind_label()),
                                            )
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .truncate()
                                                    .text_xs()
                                                    .text_color(cx.theme().muted_foreground)
                                                    .child(hit.label.to_string()),
                                            )
                                            .child(
                                                div()
                                                    .font_family(
                                                        cx.theme().mono_font_family.clone(),
                                                    )
                                                    .text_xs()
                                                    .text_color(cx.theme().muted_foreground)
                                                    .child(format!("{:.0}%", hit.score * 100.0)),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .truncate()
                                            .child(hit.title.to_string()),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .line_height(px(16.0))
                                            .text_color(cx.theme().muted_foreground)
                                            .truncate()
                                            .child(hit.detail.to_string()),
                                    )
                                    .when(selected, |this| {
                                        this.child(
                                            v_flex()
                                                .mt_1()
                                                .gap_1()
                                                .when_some(hit.lexical_score, |this, score| {
                                                    this.child(search_signal(
                                                        "TEXT",
                                                        score,
                                                        palette().copper,
                                                        cx,
                                                    ))
                                                })
                                                .when_some(hit.semantic_score, |this, score| {
                                                    this.child(search_signal(
                                                        "VECTOR",
                                                        score,
                                                        palette().cobalt,
                                                        cx,
                                                    ))
                                                }),
                                        )
                                    })
                            })),
                    )
                    .into_any_element()
            }
        }
    }

    fn render_inspector(&self, cx: &Context<Self>) -> impl IntoElement {
        let colors = palette();
        let header = section_label("INSPECTOR");
        let content = match (self.snapshot(), self.selection) {
            (Some(snapshot), Some(SceneSelection::Node(index))) => {
                let properties = snapshot.nodes.properties[index].clone();
                let node_id = snapshot.nodes.ids[index];
                let path_origin = match self.path_state {
                    PathState::ChoosingEnd(draft) => Some(draft.start),
                    PathState::Idle
                    | PathState::Searching { .. }
                    | PathState::Ready(_)
                    | PathState::Failed { .. } => None,
                };
                let database_available = self.database_path.is_some();
                let pinned = self
                    .workspace
                    .as_ref()
                    .is_some_and(|workspace| workspace.borrow().is_pinned(index));
                v_flex()
                    .gap_3()
                    .px_3()
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(colors.celadon)
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .child(format!("node:{}", snapshot.nodes.ids[index])),
                            )
                            .child(
                                div()
                                    .text_lg()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child(snapshot.nodes.labels[index].to_string()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(format!(
                                        "{} relationships · {} vectors",
                                        snapshot.nodes.degrees[index],
                                        snapshot.nodes.vector_counts[index]
                                    )),
                            )
                            .when(pinned, |this| {
                                this.child(
                                    h_flex()
                                        .mt_2()
                                        .gap_2()
                                        .items_center()
                                        .child(div().size(px(6.0)).rounded_full().bg(colors.copper))
                                        .child(
                                            div()
                                                .text_xs()
                                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                                .child("Pinned to canvas"),
                                        )
                                        .child(div().flex_1())
                                        .child(
                                            Button::new("inspector-release-node")
                                                .label("Release")
                                                .small()
                                                .on_click(cx.listener(|this, _, _, cx| {
                                                    this.release_selected(cx);
                                                })),
                                        ),
                                )
                            }),
                    )
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .gap_2()
                            .child(inspector_label("PATH"))
                            .when_some(path_origin, |this, start| {
                                this.child(
                                    h_flex()
                                        .items_center()
                                        .gap_2()
                                        .child(
                                            div()
                                                .size(px(6.0))
                                                .rounded_full()
                                                .bg(palette().celadon),
                                        )
                                        .child(
                                            div()
                                                .font_family(
                                                    cx.theme().mono_font_family.clone(),
                                                )
                                                .text_xs()
                                                .text_color(palette().celadon)
                                                .child(format!("ORIGIN · node:{start}")),
                                        ),
                                )
                            })
                            .child(
                                Button::new("inspector-path-start")
                                    .label(if path_origin == Some(node_id) {
                                        "Path origin"
                                    } else if path_origin.is_some() {
                                        "Move path origin here"
                                    } else {
                                        "Set as path origin"
                                    })
                                    .small()
                                    .disabled(!database_available)
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.choose_path_start(index, window, cx);
                                    })),
                            )
                            .when(path_origin == Some(node_id), |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .line_height(px(16.0))
                                        .text_color(cx.theme().muted_foreground)
                                        .child("Select another node on the canvas."),
                                )
                            })
                            .when_some(path_origin.filter(|&start| start != node_id), |this, _| {
                                this.child(
                                    Button::new("inspector-path-end")
                                        .label("Trace exact path")
                                        .small()
                                        .on_click(cx.listener(move |this, _, window, cx| {
                                            this.trace_path_to_node(index, window, cx);
                                        })),
                                )
                            })
                            .when(!database_available, |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .line_height(px(16.0))
                                        .text_color(cx.theme().muted_foreground)
                                        .child("Open a .vg database to trace exact paths."),
                                )
                            }),
                    )
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .gap_2()
                            .child(inspector_label("PROPERTIES"))
                            .children(
                                properties.iter().take(24).map(|property| {
                                    property_row(&property.key, &property.value, cx)
                                }),
                            ),
                    )
                    .into_any_element()
            }
            (Some(snapshot), Some(SceneSelection::Edge(index))) => {
                let source = snapshot.edges.sources[index] as usize;
                let target = snapshot.edges.targets[index] as usize;
                let label = &snapshot.edges.labels[index];
                let properties = snapshot.edges.properties[index].clone();
                v_flex()
                    .gap_3()
                    .px_3()
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(relationship_color(label))
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .child(format!("edge:{}", snapshot.edges.ids[index])),
                            )
                            .child(
                                div()
                                    .text_lg()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child(label.to_string()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(format!(
                                        "{} vector{} · directed relationship",
                                        snapshot.edges.vector_counts[index],
                                        if snapshot.edges.vector_counts[index] == 1 {
                                            ""
                                        } else {
                                            "s"
                                        }
                                    )),
                            ),
                    )
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .gap_2()
                            .child(inspector_label("DIRECTION"))
                            .child(endpoint_row(
                                "FROM",
                                snapshot.nodes.ids[source],
                                &snapshot.nodes.labels[source],
                                cx,
                            ))
                            .child(
                                div()
                                    .pl_2()
                                    .text_color(relationship_color(label))
                                    .text_sm()
                                    .child("↓"),
                            )
                            .child(endpoint_row(
                                "TO",
                                snapshot.nodes.ids[target],
                                &snapshot.nodes.labels[target],
                                cx,
                            )),
                    )
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .gap_2()
                            .child(inspector_label("PROPERTIES"))
                            .when(properties.is_empty(), |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .text_color(cx.theme().muted_foreground)
                                        .child("No relationship properties"),
                                )
                            })
                            .children(
                                properties.iter().take(24).map(|property| {
                                    property_row(&property.key, &property.value, cx)
                                }),
                            ),
                    )
                    .into_any_element()
            }
            _ => v_flex()
                .px_3()
                .gap_2()
                .text_color(cx.theme().muted_foreground)
                .child(div().text_sm().child("Select a node or relationship."))
                .child(
                    div()
                        .text_xs()
                        .line_height(px(18.0))
                        .child(
                            "Click to inspect. Double-click a node—or press Enter after selecting it—to open its two-hop context.",
                        ),
                )
                .into_any_element(),
        };
        v_flex()
            .w(px(296.0))
            .h_full()
            .flex_shrink_0()
            .border_l_1()
            .border_color(cx.theme().sidebar_border)
            .bg(cx.theme().sidebar)
            .child(header)
            .child(
                div()
                    .id("inspector-scroll")
                    .role(Role::Group)
                    .aria_label("Selected graph element properties")
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .pb_4()
                    .overflow_y_scroll()
                    .lock_scroll_axis()
                    .track_scroll(&self.inspector_scroll_handle)
                    .vertical_scrollbar(&self.inspector_scroll_handle)
                    .child(content),
            )
    }

    fn render_canvas(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();
        let dragging_node = match self.drag {
            Some(DragState::Node { index, .. }) => Some(index),
            Some(DragState::Canvas { .. }) | None => None,
        };
        let scene = match &self.state {
            LoadState::Ready(snapshot) => self.workspace.as_ref().map_or_else(
                || {
                    centered_state(
                        "Preparing workspace",
                        "Building interactive scene state.",
                        cx,
                    )
                },
                |workspace| {
                    div()
                        .absolute()
                        .size_full()
                        .child(graph_canvas(
                            snapshot.clone(),
                            workspace.clone(),
                            GraphCanvasPresentation {
                                camera: self.camera,
                                world_bounds: self.world_bounds,
                                selection: self.selection,
                                dragging: dragging_node,
                                emphasis: self.lens.as_ref().map(LensTransition::emphasis),
                                path_endpoints: self.path_endpoints,
                            },
                        ))
                        .into_any_element()
                },
            ),
            LoadState::Loading { name } => centered_state(
                "Opening graph",
                format!("Mapping {name}, building a bounded scene, then laying it out."),
                cx,
            ),
            LoadState::Failed(error) => centered_state("Could not open graph", error.clone(), cx),
        };
        div()
            .id("graph-canvas")
            .role(Role::Group)
            .aria_label("Interactive graph canvas")
            .relative()
            .size_full()
            .overflow_hidden()
            .bg(palette().graphite)
            .track_focus(&self.focus_handle)
            .cursor_grab()
            .when(self.drag.is_some(), |this| this.cursor_grabbing())
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_scroll_wheel(cx.listener(Self::on_scroll))
            .on_pinch(cx.listener(Self::on_pinch))
            .on_prepaint(move |bounds, _window, cx| {
                entity.update(cx, |this, _| {
                    this.canvas_bounds = Some(bounds);
                });
            })
            .child(scene)
            .child(
                div()
                    .absolute()
                    .left_3()
                    .bottom_3()
                    .px_2()
                    .py_1()
                    .rounded(px(4.0))
                    .bg(rgb(0x10181e).opacity(0.9))
                    .border_1()
                    .border_color(cx.theme().border)
                    .font_family(cx.theme().mono_font_family.clone())
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!("{:.0}%", self.camera.zoom * 100.0)),
            )
    }

    fn render_status_bar(&self, cx: &Context<Self>) -> impl IntoElement {
        let (nodes, edges, sampled) = self
            .snapshot()
            .map(|snapshot| {
                (
                    snapshot.nodes.ids.len(),
                    snapshot.edges.ids.len(),
                    snapshot.sampled,
                )
            })
            .unwrap_or_default();
        let (pinned, moving) = self
            .workspace
            .as_ref()
            .map(|workspace| {
                let workspace = workspace.borrow();
                (workspace.pinned_count(), workspace.is_moving())
            })
            .unwrap_or_default();
        h_flex()
            .h(px(28.0))
            .flex_shrink_0()
            .px_3()
            .gap_3()
            .items_center()
            .border_t_1()
            .border_color(cx.theme().status_bar_border)
            .bg(cx.theme().status_bar)
            .font_family(cx.theme().mono_font_family.clone())
            .text_xs()
            .text_color(cx.theme().muted_foreground)
            .child(format!("{} nodes", format_count(nodes)))
            .child(format!("{} edges", format_count(edges)))
            .when(sampled, |this| this.child("sampled"))
            .child(format!(
                "{} layout",
                self.layout_kind.label().to_lowercase()
            ))
            .when(pinned > 0, |this| this.child(format!("{pinned} pinned")))
            .when(moving, |this| this.child("settling"))
            .child(div().flex_1())
            .child(self.status.clone())
            .when_some(self.load_ms, |this, load_ms| {
                this.child(format!("{load_ms:.1} ms load+layout"))
            })
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
        let compact_toolbar = viewport_width < px(1_120.0);
        v_flex()
            .id("vectorgraph-studio")
            .role(Role::Application)
            .aria_label("VectorGraph Studio")
            .key_context("VectorGraphStudio")
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
                            .child(self.render_canvas(cx)),
                    )
                    .when(show_inspector, |this| this.child(self.render_inspector(cx))),
            )
            .child(self.render_status_bar(cx))
    }
}

fn search_signal(
    label: &'static str,
    score: f32,
    color: gpui::Hsla,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    let score = score.clamp(0.0, 1.0);
    h_flex()
        .gap_2()
        .items_center()
        .font_family(cx.theme().mono_font_family.clone())
        .text_xs()
        .child(
            div()
                .w(px(44.0))
                .flex_shrink_0()
                .text_color(color)
                .child(label),
        )
        .child(
            div()
                .h(px(3.0))
                .flex_1()
                .overflow_hidden()
                .rounded_full()
                .bg(cx.theme().border)
                .child(div().h_full().w(relative(score)).rounded_full().bg(color)),
        )
        .child(
            div()
                .w(px(32.0))
                .flex_shrink_0()
                .text_right()
                .text_color(cx.theme().muted_foreground)
                .child(format!("{:.0}", score * 100.0)),
        )
}

fn path_endpoint_card(
    role: &'static str,
    node: &EvidenceNode,
    color: gpui::Hsla,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    path_endpoint_card_data(role, node.id, &node.label, &node.title, color, cx)
}

fn path_endpoint_card_data(
    role: &'static str,
    id: u64,
    label: &str,
    title: &str,
    color: gpui::Hsla,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    h_flex()
        .gap_2()
        .items_center()
        .child(div().w(px(3.0)).h(px(30.0)).rounded_full().bg(color))
        .child(
            v_flex()
                .min_w_0()
                .gap_0p5()
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(color)
                                .child(role),
                        )
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .child(format!("node:{id} · {label}")),
                        ),
                )
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .truncate()
                        .child(title.to_string()),
                ),
        )
}

fn scene_node_title(snapshot: &SceneSnapshot, index: usize) -> Arc<str> {
    const PREFERRED_KEYS: [&str; 6] = ["title", "name", "headline", "path", "tag_name", "login"];
    snapshot.nodes.properties[index]
        .iter()
        .find_map(|property| {
            PREFERRED_KEYS
                .contains(&property.key.as_ref())
                .then_some(&property.value)
        })
        .and_then(|value| match value {
            PropertyValue::String(value) if !value.is_empty() => Some(value.clone()),
            _ => None,
        })
        .unwrap_or_else(|| snapshot.nodes.labels[index].clone())
}

const fn path_direction_label(direction: PathDirection) -> &'static str {
    match direction {
        PathDirection::Outgoing => "outgoing only",
        PathDirection::Incoming => "incoming only",
        PathDirection::Both => "either direction",
    }
}

const fn path_direction_compact_label(direction: PathDirection) -> &'static str {
    match direction {
        PathDirection::Outgoing => "FROM ORIGIN",
        PathDirection::Incoming => "TO ORIGIN",
        PathDirection::Both => "EITHER WAY",
    }
}

fn path_hop_button(max_hops: usize, selected_max_hops: usize, cx: &Context<StudioView>) -> Button {
    Button::new(match max_hops {
        1 => "path-hops-1",
        2 => "path-hops-2",
        4 => "path-hops-4",
        6 => "path-hops-6",
        _ => "path-hops-custom",
    })
    .label(max_hops.to_string())
    .small()
    .selected(max_hops == selected_max_hops)
    .on_click(cx.listener(move |this, _, _, cx| {
        this.set_path_max_hops(max_hops, cx);
    }))
}

fn path_plan_diagnostics(
    report: &EvidencePathReport,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    v_flex()
        .gap_1()
        .child(
            h_flex()
                .items_center()
                .gap_2()
                .child(inspector_label("PLAN"))
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!(
                            "{} · {}",
                            path_strategy_label(report.strategy),
                            path_direction_compact_label(report.direction)
                        )),
                ),
        )
        .child(
            h_flex()
                .items_center()
                .gap_2()
                .child(
                    v_flex()
                        .gap_0p5()
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(palette().celadon)
                                .child("FROM"),
                        )
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .child(format!(
                                    "{} expanded",
                                    format_count(report.start_expanded_nodes)
                                )),
                        ),
                )
                .child(div().h(px(1.0)).flex_1().bg(cx.theme().border))
                .child(
                    v_flex()
                        .gap_0p5()
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_right()
                                .text_color(palette().copper)
                                .child("TO"),
                        )
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .text_right()
                                .text_color(cx.theme().muted_foreground)
                                .child(format!(
                                    "{} expanded",
                                    format_count(report.end_expanded_nodes)
                                )),
                        ),
                ),
        )
        .child(
            h_flex()
                .gap_4()
                .child(path_work_stat("VISITED", report.visited_nodes, cx))
                .child(path_work_stat(
                    "REL READS",
                    report.examined_relationships,
                    cx,
                )),
        )
}

fn path_work_stat(label: &'static str, value: usize, cx: &Context<StudioView>) -> impl IntoElement {
    h_flex()
        .gap_1()
        .font_family(cx.theme().mono_font_family.clone())
        .text_xs()
        .child(div().text_color(cx.theme().muted_foreground).child(label))
        .child(
            div()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(format_count(value)),
        )
}

const fn path_strategy_label(strategy: EvidencePathStrategy) -> &'static str {
    match strategy {
        EvidencePathStrategy::BreadthFirst => "BREADTH-FIRST",
        EvidencePathStrategy::BidirectionalBreadthFirst => "BIDIRECTIONAL",
    }
}

fn section_label(label: &'static str) -> impl IntoElement {
    div()
        .px_4()
        .pt_4()
        .pb_2()
        .text_xs()
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .text_color(rgb(0x70838e))
        .child(label)
}

fn inspector_label(label: &'static str) -> impl IntoElement {
    div()
        .text_xs()
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .text_color(rgb(0x70838e))
        .child(label)
}

fn endpoint_row(
    direction: &'static str,
    id: u64,
    label: &str,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    h_flex()
        .gap_2()
        .items_center()
        .child(
            div()
                .w(px(42.0))
                .flex_shrink_0()
                .text_xs()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(cx.theme().muted_foreground)
                .child(direction),
        )
        .child(
            v_flex()
                .gap_0p5()
                .child(div().text_sm().child(label.to_string()))
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!("node:{id}")),
                ),
        )
}

fn nav_row(label: &'static str, selected: bool, cx: &Context<StudioView>) -> impl IntoElement {
    h_flex()
        .id(label)
        .mx_2()
        .h(px(30.0))
        .px_2()
        .items_center()
        .rounded(px(4.0))
        .text_sm()
        .when(selected, |this| {
            this.bg(cx.theme().sidebar_accent)
                .text_color(cx.theme().sidebar_accent_foreground)
        })
        .when(!selected, |this| {
            this.text_color(cx.theme().muted_foreground)
        })
        .child(label)
}

fn property_row(key: &str, value: &PropertyValue, cx: &Context<StudioView>) -> impl IntoElement {
    v_flex()
        .gap_1()
        .child(
            div()
                .text_xs()
                .text_color(cx.theme().muted_foreground)
                .child(key.to_string()),
        )
        .child(
            div()
                .font_family(cx.theme().mono_font_family.clone())
                .text_xs()
                .line_height(px(17.0))
                .child(format_value(value)),
        )
}

fn format_value(value: &PropertyValue) -> String {
    match value {
        PropertyValue::Null => "null".into(),
        PropertyValue::Bool(value) => value.to_string(),
        PropertyValue::Int(value) => value.to_string(),
        PropertyValue::Float(value) => format!("{value:.5}"),
        PropertyValue::String(value) => value.to_string(),
        PropertyValue::Bytes(value) => format!("<{} bytes>", value.len()),
        PropertyValue::Node(value) => format!("node:{value}"),
        PropertyValue::Edge(value) => format!("edge:{value}"),
    }
}

fn centered_state(
    title: impl Into<SharedString>,
    detail: impl Into<SharedString>,
    cx: &Context<StudioView>,
) -> gpui::AnyElement {
    v_flex()
        .absolute()
        .size_full()
        .items_center()
        .justify_center()
        .gap_2()
        .child(
            div()
                .text_base()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(title.into()),
        )
        .child(
            div()
                .max_w(px(460.0))
                .text_sm()
                .text_color(cx.theme().muted_foreground)
                .child(detail.into()),
        )
        .into_any_element()
}

fn format_count(value: usize) -> String {
    let text = value.to_string();
    let mut output = String::with_capacity(text.len() + text.len() / 3);
    for (index, character) in text.chars().enumerate() {
        if index > 0 && (text.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    output
}

fn visible_facet_counts(
    counts: &[(Arc<str>, usize)],
    active: Option<&Arc<str>>,
) -> Vec<(Arc<str>, usize)> {
    const LIMIT: usize = 10;
    let mut visible: Vec<_> = counts.iter().take(LIMIT).cloned().collect();
    let Some(active) = active else {
        return visible;
    };
    if visible.iter().any(|(label, _)| label == active) {
        return visible;
    }
    let Some(active_count) = counts.iter().find(|(label, _)| label == active).cloned() else {
        return visible;
    };
    if visible.len() == LIMIT {
        visible.pop();
    }
    visible.insert(0, active_count);
    visible
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{KeyBinding, ScrollDelta, TestAppContext, point, size};
    use std::time::Duration;
    use vectorgraph_studio_core::{EvidencePath, EvidenceStep};

    #[test]
    fn lens_transition_retargets_from_its_current_presentation_and_reduces_motion() {
        let mut lens = LensTransition::new(2, 1);
        lens.retarget(vec![1.0, 0.4].into(), vec![0.8].into(), 0.9);
        assert!(lens.step(1.0 / 60.0, false));
        let in_flight = lens.emphasis();
        assert!(in_flight.mix > 0.0 && in_flight.mix < 1.0);

        lens.retarget(vec![0.0, 1.0].into(), vec![0.2].into(), 0.7);
        assert_eq!(lens.from_nodes[0], in_flight.mix);
        assert!(!lens.step(1.0 / 60.0, true));
        assert_eq!(lens.progress, 1.0);
        assert_eq!(lens.emphasis().dim, 0.7);
    }

    #[test]
    fn active_outlier_facet_stays_in_the_bounded_taxonomy() {
        let counts: Vec<(Arc<str>, usize)> = (0..12)
            .map(|index| (format!("L{index}").into(), 100 - index))
            .collect();
        let active: Arc<str> = "L11".into();
        let visible = visible_facet_counts(&counts, Some(&active));
        assert_eq!(visible.len(), 10);
        assert_eq!(visible[0], (active, 89));
        assert!(!visible.iter().any(|(label, _)| label.as_ref() == "L9"));
    }

    #[gpui::test]
    fn exact_evidence_path_maps_to_a_reversible_canvas_lens(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
        });
        let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));

        window
            .update(cx, |view, window, cx| {
                let original_camera = view.camera;
                view.canvas_bounds = Some(Bounds::new(
                    point(px(0.0), px(0.0)),
                    size(px(800.0), px(600.0)),
                ));
                view.world_bounds = Rect {
                    min: Vec2::new(-10_000.0, -10_000.0),
                    max: Vec2::new(10_000.0, 10_000.0),
                };
                let snapshot = view.snapshot().unwrap().clone();
                let edge_index = snapshot
                    .edges
                    .sources
                    .iter()
                    .zip(&snapshot.edges.targets)
                    .position(|(&source, &target)| source == 0 && target == 7)
                    .unwrap();
                let edge_id = snapshot.edges.ids[edge_index];
                let report = Arc::new(EvidencePathReport {
                    start: EvidenceNode {
                        id: 0,
                        label: "Document".into(),
                        title: "Context element 0".into(),
                        vector_count: 0,
                        properties: Arc::from([]),
                    },
                    end: EvidenceNode {
                        id: 7,
                        label: "Source".into(),
                        title: "Context element 7".into(),
                        vector_count: 0,
                        properties: Arc::from([]),
                    },
                    path: Some(EvidencePath {
                        nodes: vec![
                            EvidenceNode {
                                id: 0,
                                label: "Document".into(),
                                title: "Context element 0".into(),
                                vector_count: 0,
                                properties: Arc::from([]),
                            },
                            EvidenceNode {
                                id: 7,
                                label: "Source".into(),
                                title: "Context element 7".into(),
                                vector_count: 0,
                                properties: Arc::from([]),
                            },
                        ]
                        .into(),
                        steps: vec![EvidenceStep {
                            edge_id,
                            from: 0,
                            to: 7,
                            label: "SUPPORTS".into(),
                            title: "SUPPORTS".into(),
                            forward: true,
                            vector_count: 0,
                            properties: Arc::from([]),
                        }]
                        .into(),
                    }),
                    strategy: EvidencePathStrategy::BidirectionalBreadthFirst,
                    termination: EvidencePathTermination::Found,
                    direction: PathDirection::Outgoing,
                    relationship_label: Some("SUPPORTS".into()),
                    max_hops: 6,
                    visited_nodes: 2,
                    start_expanded_nodes: 1,
                    end_expanded_nodes: 0,
                    expanded_nodes: 1,
                    examined_relationships: 3,
                    elapsed: Duration::from_micros(240),
                });

                view.path_state = PathState::Ready(report.clone());
                view.present_evidence_path(&report, window, cx);
                view.settle_presentation_for_capture();
                assert_eq!(view.camera.zoom, 24.0);
                assert_eq!(
                    view.path_endpoints,
                    Some(GraphPathEndpoints {
                        start: 0,
                        end: Some(7),
                    })
                );
                assert_eq!(view.selection, Some(SceneSelection::Node(0)));
                let emphasis = view.lens.as_ref().unwrap().emphasis();
                assert_eq!(emphasis.target_nodes[0], 1.0);
                assert_eq!(emphasis.target_nodes[7], 1.0);
                assert_eq!(emphasis.target_edges[edge_index], 1.0);

                view.activate_evidence_step(0, window, cx);
                view.settle_presentation_for_capture();
                assert_eq!(view.selection, Some(SceneSelection::Edge(edge_index)));

                view.show_overview(window, cx);
                view.settle_presentation_for_capture();
                assert!(matches!(view.path_state, PathState::Idle));
                assert!(view.path_endpoints.is_none());
                assert!(view.lens.is_none());
                assert_eq!(view.camera, original_camera);
            })
            .unwrap();
    }

    #[gpui::test]
    fn path_draft_builds_and_activates_a_direct_destination_query(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
        });
        let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));

        window
            .update(cx, |view, window, cx| {
                view.database_path = Some(PathBuf::from("fixture.vg"));
                view.choose_path_start(0, window, cx);
                assert!(matches!(
                    view.path_state,
                    PathState::ChoosingEnd(PathDraft {
                        start: 0,
                        direction: PathDirection::Both,
                        max_hops: 6,
                    })
                ));
                assert_eq!(
                    view.path_endpoints,
                    Some(GraphPathEndpoints {
                        start: 0,
                        end: None,
                    })
                );
                assert_eq!(view.selection, Some(SceneSelection::Node(0)));
                assert_eq!(view.path_destination_candidate(), None);
                view.activate_selection(window, cx);
                assert!(matches!(view.path_state, PathState::ChoosingEnd(_)));
                assert!(!view.context_focus_active);

                view.execute_command("path-start 0 in 2", window, cx);
                assert!(matches!(
                    view.path_state,
                    PathState::ChoosingEnd(PathDraft {
                        start: 0,
                        direction: PathDirection::Incoming,
                        max_hops: 2,
                    })
                ));
                view.set_path_max_hops(0, cx);

                view.set_selection(Some(SceneSelection::Node(7)));
                assert_eq!(view.path_destination_candidate(), Some(7));
                view.set_path_max_hops(4, cx);
                assert!(view.status.contains("Destination ready"));
                assert!(view.status.contains("Enter traces exact path"));
                view.set_path_max_hops(2, cx);
                let query = view.path_query_to_node(7).unwrap();
                assert_eq!(query.start, 0);
                assert_eq!(query.end, 7);
                assert_eq!(query.direction, PathDirection::Incoming);
                assert_eq!(query.max_hops, 2);
                assert!(query.relationship_label.is_none());

                view.activate_selection(window, cx);
                assert!(matches!(
                    view.path_state,
                    PathState::Searching { start: 0, end: 7 }
                ));
                assert_eq!(
                    view.path_endpoints,
                    Some(GraphPathEndpoints {
                        start: 0,
                        end: Some(7),
                    })
                );
                assert!(!view.context_focus_active);

                view.show_overview(window, cx);
                assert!(matches!(view.path_state, PathState::Idle));
                assert!(view.path_endpoints.is_none());
            })
            .unwrap();
    }

    #[gpui::test]
    fn enter_routes_to_a_ready_path_destination(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
            cx.bind_keys([KeyBinding::new(
                "enter",
                ActivateSelection,
                Some("VectorGraphStudio"),
            )]);
        });
        let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));

        view.update(cx, |view, cx| {
            view.database_path = Some(PathBuf::from("fixture.vg"));
            view.path_state = PathState::ChoosingEnd(PathDraft {
                start: 0,
                direction: PathDirection::Outgoing,
                max_hops: 4,
            });
            view.path_endpoints = Some(GraphPathEndpoints {
                start: 0,
                end: None,
            });
            view.set_selection(Some(SceneSelection::Node(7)));
            cx.notify();
        });
        cx.update(|window, cx| {
            let focus_handle = view.read(cx).focus_handle.clone();
            window.focus(&focus_handle, cx);
            _ = window.draw(cx);
        });

        cx.simulate_keystrokes("enter");

        view.read_with(&*cx, |view, _| {
            assert!(matches!(
                view.path_state,
                PathState::Searching { start: 0, end: 7 }
                    | PathState::Failed {
                        start: 0,
                        end: 7,
                        ..
                    }
            ));
            assert_eq!(
                view.path_endpoints,
                Some(GraphPathEndpoints {
                    start: 0,
                    end: Some(7),
                })
            );
            assert!(!view.context_focus_active);
        });
    }

    #[gpui::test]
    fn evidence_path_destination_action_renders_at_the_minimum_width(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
        });
        let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));

        view.update(cx, |view, cx| {
            view.database_path = Some(PathBuf::from("fixture.vg"));
            view.path_state = PathState::ChoosingEnd(PathDraft {
                start: 0,
                direction: PathDirection::Outgoing,
                max_hops: 4,
            });
            view.path_endpoints = Some(GraphPathEndpoints {
                start: 0,
                end: None,
            });
            view.set_selection(Some(SceneSelection::Node(7)));
            cx.notify();
        });

        cx.simulate_resize(size(px(760.0), px(520.0)));
        cx.run_until_parked();
        cx.update(|window, cx| {
            _ = window.draw(cx);
        });

        let action = cx
            .debug_bounds("evidence-path-run")
            .expect("destination action should render in the evidence rail");
        assert!(action.right() <= px(306.0));
        assert!(action.bottom() <= px(520.0));
        assert!(action.size.height >= px(28.0));
    }

    #[gpui::test]
    fn taxonomy_facets_create_a_reversible_graph_lens(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
        });
        let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));

        window
            .update(cx, |view, window, cx| {
                view.execute_command("facet node Document", window, cx);
                view.settle_presentation_for_capture();
                assert_eq!(
                    view.active_facet,
                    Some(FacetLens::NodeLabel("Document".into()))
                );
                let emphasis = view.lens.as_ref().unwrap().emphasis();
                assert_eq!(
                    emphasis
                        .target_nodes
                        .iter()
                        .filter(|&&score| score == 1.0)
                        .count(),
                    24
                );
                assert!(emphasis.target_edges.iter().any(|&score| score > 0.0));

                view.execute_command("facet relationship SUPPORTS", window, cx);
                view.settle_presentation_for_capture();
                assert_eq!(
                    view.active_facet,
                    Some(FacetLens::Relationship("SUPPORTS".into()))
                );
                let emphasis = view.lens.as_ref().unwrap().emphasis();
                assert!(emphasis.target_edges.contains(&1.0));
                assert!(emphasis.target_nodes.contains(&0.72));

                view.escape(window, cx);
                view.settle_presentation_for_capture();
                assert!(view.active_facet.is_none());
                assert!(view.lens.is_none());

                view.execute_command("facet node Document", window, cx);
                view.execute_command("memory leak", window, cx);
                view.settle_presentation_for_capture();
                assert!(view.active_facet.is_none());
                assert!(view.lens.is_none());
                assert!(matches!(view.search_state, SearchState::Failed { .. }));
            })
            .unwrap();
    }

    #[gpui::test]
    fn commands_select_relationships_and_nodes_without_ambiguous_state(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
        });
        let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));

        window
            .update(cx, |view, window, cx| {
                view.execute_command("edge 0", window, cx);
                assert_eq!(view.selection, Some(SceneSelection::Edge(0)));

                view.execute_command("node 7", window, cx);
                assert_eq!(view.selection, Some(SceneSelection::Node(7)));

                view.execute_command("zoom 2", window, cx);
                assert_eq!(view.camera.zoom, 2.0);

                view.execute_command("edge 999999", window, cx);
                assert_eq!(view.selection, None);

                view.execute_command("memory leak", window, cx);
                assert!(matches!(view.search_state, SearchState::Failed { .. }));
                view.show_overview(window, cx);
                assert!(matches!(view.search_state, SearchState::Idle));
            })
            .unwrap();
    }

    #[gpui::test]
    fn focused_search_context_returns_to_the_saved_presentation(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
        });
        let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));

        window
            .update(cx, |view, window, cx| {
                let original_camera = view.camera;
                let original_positions = view
                    .workspace
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .positions()
                    .to_vec();
                view.saved_overview_camera = Some(original_camera);
                view.selection = Some(SceneSelection::Node(7));
                view.focus_search_result(SceneSelection::Node(7), window, cx);
                view.settle_presentation_for_capture();
                assert_ne!(
                    view.workspace.as_ref().unwrap().borrow().positions(),
                    original_positions
                );
                assert!(view.lens.is_some());

                view.show_overview(window, cx);
                view.settle_presentation_for_capture();
                assert_eq!(view.camera, original_camera);
                assert_eq!(
                    view.workspace.as_ref().unwrap().borrow().positions(),
                    original_positions
                );
                assert!(view.lens.is_none());
            })
            .unwrap();
    }

    #[gpui::test]
    fn double_click_opens_a_reversible_two_hop_context_without_starting_a_drag(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
        });
        let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));

        window
            .update(cx, |view, window, cx| {
                let original_camera = view.camera;
                let original_positions = view
                    .workspace
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .positions()
                    .to_vec();
                let canvas = Bounds::new(point(px(0.0), px(0.0)), size(px(800.0), px(600.0)));
                view.canvas_bounds = Some(canvas);
                let viewport = Vec2::new(800.0, 600.0);
                let screen =
                    view.camera
                        .project(original_positions[7], view.world_bounds, viewport);
                let event = MouseDownEvent {
                    button: MouseButton::Left,
                    position: point(px(screen.x), px(screen.y)),
                    click_count: 2,
                    ..MouseDownEvent::default()
                };

                view.on_mouse_down(&event, window, cx);
                assert_eq!(view.selection, Some(SceneSelection::Node(7)));
                assert!(view.context_focus_active);
                assert!(view.drag.is_none());
                assert!(view.status.contains("two-hop"));

                view.settle_presentation_for_capture();
                assert_ne!(
                    view.workspace.as_ref().unwrap().borrow().positions(),
                    original_positions
                );
                view.escape(window, cx);
                view.settle_presentation_for_capture();
                assert_eq!(view.camera, original_camera);
                assert_eq!(
                    view.workspace.as_ref().unwrap().borrow().positions(),
                    original_positions
                );
                assert!(!view.context_focus_active);
            })
            .unwrap();
    }

    #[gpui::test]
    fn toolbar_reflows_without_clipping_at_the_minimum_window_width(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
        });
        let (_view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));

        cx.simulate_resize(size(px(760.0), px(520.0)));
        cx.run_until_parked();
        cx.update(|window, cx| {
            _ = window.draw(cx);
        });

        let primary = cx
            .debug_bounds("compact-toolbar-primary")
            .expect("compact primary toolbar should render");
        let secondary = cx
            .debug_bounds("compact-toolbar-secondary")
            .expect("compact secondary toolbar should render");
        assert!(primary.right() <= px(760.0));
        assert!(secondary.right() <= px(760.0));
        assert!(primary.size.height >= px(47.0));
        assert!(secondary.size.height >= px(39.0));
        assert!(cx.debug_bounds("wide-toolbar").is_none());

        cx.simulate_resize(size(px(1_340.0), px(820.0)));
        cx.run_until_parked();
        cx.update(|window, cx| {
            _ = window.draw(cx);
        });
        assert!(cx.debug_bounds("wide-toolbar").is_some());
        assert!(cx.debug_bounds("compact-toolbar-primary").is_none());
    }

    #[gpui::test]
    fn inspector_scrolls_long_properties_without_zooming_the_canvas(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_component::init(cx);
            crate::apply_studio_theme(cx);
        });
        let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));

        view.update(cx, |view, cx| {
            let LoadState::Ready(snapshot) = &view.state else {
                panic!("demo snapshot should be ready");
            };
            let mut snapshot = (**snapshot).clone();
            snapshot.nodes.properties[0] = (0..24)
                .map(|index| vectorgraph_studio_core::SceneProperty {
                    key: format!("property_{index:02}").into(),
                    value: PropertyValue::String(
                        format!("A deliberately detailed graph property value {index}").into(),
                    ),
                })
                .collect::<Vec<_>>()
                .into();
            view.state = LoadState::Ready(Arc::new(snapshot));
            view.set_selection(Some(SceneSelection::Node(0)));
            cx.notify();
        });

        cx.simulate_resize(size(px(1_200.0), px(420.0)));
        cx.run_until_parked();
        cx.update(|window, cx| {
            _ = window.draw(cx);
        });

        let (scroll_handle, camera_before) = view.read_with(&*cx, |view, _| {
            (view.inspector_scroll_handle.clone(), view.camera)
        });
        assert!(
            scroll_handle.max_offset().y > px(0.0),
            "long inspector content should exceed its viewport"
        );
        let offset_before = scroll_handle.offset().y;

        cx.simulate_event(ScrollWheelEvent {
            position: scroll_handle.bounds().center(),
            delta: ScrollDelta::Pixels(point(px(0.0), px(-180.0))),
            ..Default::default()
        });
        cx.run_until_parked();

        assert!(scroll_handle.offset().y < offset_before);
        assert_eq!(view.read_with(&*cx, |view, _| view.camera), camera_before);

        view.update(cx, |view, _| {
            view.set_selection(Some(SceneSelection::Node(1)));
        });
        assert_eq!(scroll_handle.offset().y, px(0.0));
    }
}
