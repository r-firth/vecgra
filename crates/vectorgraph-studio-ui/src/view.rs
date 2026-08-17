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
    StatefulInteractiveElement as _, Styled, Subscription, Task, Window, actions, div, px, rgb,
};
use gpui_component::{
    ActiveTheme as _, Disableable as _, ElementExt as _, InteractiveElementExt as _,
    Selectable as _, Sizable as _, TitleBar, button::Button, h_flex, input::Input,
    input::InputEvent, input::InputState, scroll::ScrollableElement as _, v_flex,
};
use vectorgraph_studio_core::{
    Camera, CameraMotion, DetailLevel, GraphWorkspace, LayoutKind, LayoutOptions, MAX_CAMERA_ZOOM,
    MIN_CAMERA_ZOOM, PropertyValue, Rect, SceneSelection, SceneSnapshot, SearchMode, SearchReport,
    SnapshotOptions, Vec2, detail_level, hit_test_edges, hit_test_positions, search_database,
};

use crate::graph_canvas::{GraphEmphasis, graph_canvas};
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
        FocusSelectedContext,
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

    fn focus_selected_context(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self
            .query_input
            .read(cx)
            .focus_handle(cx)
            .is_focused(window)
        {
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
        self.context_focus_active = false;
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
        if self.context_focus_active || !matches!(self.search_state, SearchState::Idle) {
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
            self.status = format!(
                "Selected node {} · drag to arrange",
                snapshot.nodes.ids[index]
            )
            .into();
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

    fn render_top_bar(&self, cx: &Context<Self>) -> impl IntoElement {
        let colors = palette();
        let database_name: SharedString = match &self.state {
            LoadState::Ready(snapshot) => snapshot.database_name.to_string().into(),
            LoadState::Loading { name } => name.clone(),
            LoadState::Failed(_) => "No database".into(),
        };
        TitleBar::new().h(px(52.0)).child(
            h_flex()
                .size_full()
                .pr_3()
                .gap_2()
                .items_center()
                .child(
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
                        ),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w(px(180.0))
                        .max_w(px(620.0))
                        .child(Input::new(&self.query_input).small()),
                )
                .child(
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
                        ),
                )
                .child(div().flex_1())
                .child(
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
                                    !self.selection.and_then(SceneSelection::node).is_some_and(
                                        |index| {
                                            self.workspace.as_ref().is_some_and(|workspace| {
                                                workspace.borrow().is_pinned(index)
                                            })
                                        },
                                    ),
                                )
                                .on_click(cx.listener(|this, _, _, cx| this.release_selected(cx))),
                        ),
                )
                .child(
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
                        ),
                ),
        )
    }

    fn render_left_panel(&self, cx: &Context<Self>) -> impl IntoElement {
        let label_counts = self.node_label_counts.clone();
        let relationship_counts = self.relationship_counts.clone();
        let showing_search = !matches!(self.search_state, SearchState::Idle);
        let showing_context = self.context_focus_active && !showing_search;
        v_flex()
            .w(if showing_search { px(294.0) } else { px(218.0) })
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
                    .child(nav_row("Overview", !showing_search && !showing_context, cx)),
            )
            .child(nav_row("Search results", showing_search, cx))
            .when(showing_context, |this| {
                this.child(nav_row("2-hop context", true, cx))
            })
            .when(showing_search, |this| {
                this.child(self.render_search_results(cx))
            })
            .when(!showing_search, |this| {
                this.child(section_label("RELATIONSHIPS"))
            })
            .when(!showing_search, |this| {
                this.child(v_flex().px_2().children(
                    relationship_counts.iter().take(10).enumerate().map(
                        |(row_index, (label, count))| {
                            h_flex()
                                .id(("relationship-row", row_index))
                                .h(px(28.0))
                                .px_2()
                                .gap_2()
                                .items_center()
                                .rounded(px(4.0))
                                .text_sm()
                                .child(
                                    div()
                                        .size(px(7.0))
                                        .rounded_full()
                                        .bg(relationship_color(label)),
                                )
                                .child(div().flex_1().truncate().child(label.to_string()))
                                .child(
                                    div()
                                        .font_family(cx.theme().mono_font_family.clone())
                                        .text_xs()
                                        .text_color(cx.theme().muted_foreground)
                                        .child(format_count(*count)),
                                )
                        },
                    ),
                ))
            })
            .when(!showing_search, |this| {
                this.child(section_label("NODE LABELS"))
            })
            .when(!showing_search, |this| {
                this.child(
                    v_flex()
                        .px_2()
                        .children(label_counts.iter().take(10).enumerate().map(
                            |(row_index, (label, count))| {
                                h_flex()
                                    .id(("label-row", row_index))
                                    .h(px(28.0))
                                    .px_2()
                                    .items_center()
                                    .justify_between()
                                    .rounded(px(4.0))
                                    .text_sm()
                                    .child(label.to_string())
                                    .child(
                                        div()
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .child(format_count(*count)),
                                    )
                            },
                        )),
                )
            })
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
                                let kind_color = if hit.kind_label() == "EDGE" {
                                    relationship_color(&hit.label)
                                } else {
                                    palette().celadon
                                };
                                v_flex()
                                    .id(("search-result", index))
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
                            self.camera,
                            self.world_bounds,
                            self.selection,
                            dragging_node,
                            self.lens.as_ref().map(LensTransition::emphasis),
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
        let show_inspector = window.viewport_size().width >= px(980.0);
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
            .on_action(cx.listener(|this, _: &FocusSelectedContext, window, cx| {
                this.focus_selected_context(window, cx);
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
            .child(self.render_top_bar(cx))
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

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ScrollDelta, TestAppContext, point, size};

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
