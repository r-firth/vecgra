use std::cell::RefCell;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui::prelude::FluentBuilder as _;
use gpui::{
    AppContext as _, Bounds, Context, Entity, ExternalPaths, FocusHandle, Focusable as _,
    InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, ParentElement as _, PathPromptOptions, PinchEvent, Pixels, Point, Render, Role,
    ScrollHandle, ScrollWheelEvent, SharedString, StatefulInteractiveElement as _, Styled,
    Subscription, Task, Window, actions, div, px, relative, rgb,
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
    Camera, CameraMotion, CommandHistory, DatabaseFingerprint, DetailLevel, EvidenceNode,
    EvidencePathReport, EvidencePathStrategy, EvidencePathTermination, GraphWorkspace, LayoutKind,
    LayoutOptions, MAX_CAMERA_ZOOM, MIN_CAMERA_ZOOM, NodeConnections, PathDirection, PropertyValue,
    RecentDatabases, Rect, ReloadedScene, SceneDelta, SceneSelection, SceneSnapshot, SearchMode,
    SearchReport, SnapshotOptions, Vec2, detail_level, display_path, element_title,
    evidence_path_database, hit_test_edges, hit_test_positions, search_database, split_path_title,
};

mod bezel_controls;
mod facet_menu;
mod help;
mod hidden;
mod node_menu;
mod panels;

use facet_menu::FacetMenu;
use hidden::HiddenScene;
use node_menu::NodeMenu;
use panels::path_direction_label;
#[cfg(test)]
use panels::visible_facet_counts;

use crate::graph_canvas::{
    GraphCanvasPresentation, GraphEmphasis, GraphPathEndpoints, LabelColors, PATH_INTERIOR_SCORE,
    graph_canvas, node_caption_text, scene_svg,
};
use crate::graph_navigator::{
    GraphNavigatorCache, GraphNavigatorPresentation, graph_navigator, navigator_world_position,
};
use crate::theme::palette;

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
        PreviousSearchResult,
        ReloadDatabase,
        ToggleLiveRefresh,
        OpenDatabase,
        SelectionBack,
        SelectionForward,
        ToggleHelp,
        ExportView,
        SearchBoxUp,
        SearchBoxDown,
        CopySelection,
        HideSelection,
        UndoHide,
        ShowHidden
    ]
);

/// How many earlier selections Back can return to.
const SELECTION_HISTORY_LIMIT: usize = 64;
/// The longest path, in hops, Studio asks the engine for.
const MAX_PATH_HOPS: usize = 64;
/// How far below the canvas's top edge framing starts: under the graph
/// navigator (12 px from the top and 104 px tall) with room for a name.
const FRAMING_TOP: f32 = 132.0;

/// How often live refresh checks the database file for new commits. Only file
/// metadata is read on each tick; the database is reopened when it changes.
const LIVE_REFRESH_INTERVAL: Duration = Duration::from_millis(1_000);

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

impl ContextSummary {
    fn counts(&self) -> String {
        format!(
            "{} direct · {} two-hop · {}",
            self.direct,
            self.indirect,
            panels::counted(self.relationships, "relationship", "relationships")
        )
    }
}

/// Stepping through earlier commands from the command box.
struct HistoryBrowse {
    /// The entry in the box.
    index: usize,
    /// What was in the box before the first step back.
    draft: String,
}

/// What a 2-hop context holds, for the heading of its tab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContextSummary {
    node: u64,
    direct: usize,
    indirect: usize,
    relationships: usize,
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

/// A wider search to offer when an exact path search finds nothing within
/// its bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PathRetry {
    EitherDirection,
    AnyRelationship,
    MoreHops(usize),
}

/// The wider searches worth offering after `report`, which are none unless it
/// proved there is no path within its bounds.
fn path_retries(report: &EvidencePathReport) -> Vec<PathRetry> {
    if report.termination != EvidencePathTermination::NotFoundWithinHops {
        return Vec::new();
    }
    let mut retries = Vec::new();
    if report.direction != PathDirection::Both {
        retries.push(PathRetry::EitherDirection);
    }
    if report.relationship_label.is_some() {
        retries.push(PathRetry::AnyRelationship);
    }
    if report.max_hops < MAX_PATH_HOPS {
        retries.push(PathRetry::MoreHops(
            (report.max_hops.max(1) * 2).min(MAX_PATH_HOPS),
        ));
    }
    retries
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectedElement {
    Node(u64),
    Edge(u64),
}

fn live_refresh_default() -> bool {
    std::env::var("VECGRA_STUDIO_LIVE").map_or(true, |value| value != "0")
}

/// A node as the status bar names it: its canvas name, or its label and ID
/// when it has no title.
fn status_node_name(snapshot: &SceneSnapshot, index: usize) -> String {
    match element_title(&snapshot.nodes.properties[index]) {
        Some(title) => node_caption_text(&title).to_string(),
        None => format!(
            "{} {}",
            snapshot.nodes.labels[index], snapshot.nodes.ids[index]
        ),
    }
}

/// A relationship as the status bar names it: its type and both ends.
fn status_edge_name(snapshot: &SceneSnapshot, index: usize) -> String {
    format!(
        "{} · {} → {}",
        snapshot.edges.labels[index],
        status_node_name(snapshot, snapshot.edges.sources[index] as usize),
        status_node_name(snapshot, snapshot.edges.targets[index] as usize)
    )
}

/// Whether command-box text is a command rather than a search. A command
/// word followed by words that do not fit that command, such as
/// "layout refresh", "release notes", or "open dialog for files", is
/// searched instead, so titles that start with a command word stay
/// searchable. Text that names a command's IDs but gets the rest wrong is
/// still a command, so it can say what was expected.
fn reads_as_command(query: &str) -> bool {
    let mut words = query.split_whitespace();
    let Some(first) = words.next() else {
        return false;
    };
    let rest: Vec<&str> = words.collect();
    let is_id = |word: &str| word.parse::<u64>().is_ok();
    match first {
        "fit" | "clear" | "reload" | "refresh" | "help" | "?" | "shortcuts" | "back"
        | "forward" | "release" | "unhide" => rest.is_empty(),
        "hide" => match rest.as_slice() {
            [] => true,
            [id] => is_id(id),
            ["label" | "relationship", label] => !label.is_empty(),
            _ => false,
        },
        "only" => matches!(rest.as_slice(), ["label" | "relationship", _]),
        "zoom" => matches!(rest.as_slice(), [factor] if parse_zoom(factor).is_some()),
        "live" | "watch" => matches!(rest.as_slice(), [] | ["on" | "off"]),
        "layout" => matches!(rest.as_slice(), ["auto" | "force" | "structure" | "orbit"]),
        "node" | "center" | "focus" | "context" | "edge" | "relationship" => {
            matches!(rest.as_slice(), [id] if is_id(id))
        }
        "path" | "path-start" => rest.first().is_some_and(|id| is_id(id)),
        "facet" | "lens" => matches!(
            rest.as_slice(),
            ["node" | "label" | "edge" | "relationship", _, ..]
        ),
        "open" => {
            let argument = query.trim_start()["open".len()..].trim();
            argument.is_empty()
                || argument.starts_with(['"', '\'', '~'])
                || argument.contains(['/', '\\'])
                || argument.ends_with(".vg")
                || resolve_open_argument(argument).is_some_and(|path| path.exists())
        }
        "export" => {
            let argument = query.trim_start()["export".len()..].trim();
            argument.is_empty()
                || argument.starts_with(['"', '\'', '~'])
                || argument.contains(['/', '\\'])
                || argument.ends_with(".svg")
        }
        "text" | "semantic" | "hybrid" | "search" => true,
        _ => false,
    }
}

/// What Enter will do with command-box text, as the box's hint names it:
/// the command word for a command, or "search".
fn enter_hint(query: &str) -> Option<&str> {
    let query = query.trim();
    let word = query.split_whitespace().next()?;
    if !reads_as_command(query) {
        return Some("search");
    }
    Some(match word {
        "?" | "shortcuts" => "help",
        "refresh" => "reload",
        "watch" => "live",
        "lens" => "facet",
        "context" => "focus",
        "relationship" => "edge",
        "hybrid" => "search",
        "text" => "text search",
        "semantic" => "semantic search",
        word => word,
    })
}

/// What a label or type typed in a command matches among those in view.
#[derive(Debug, PartialEq, Eq)]
enum LabelMatch {
    Found(Arc<str>),
    /// Several start with what was typed.
    Ambiguous(Vec<Arc<str>>),
    Missing,
}

/// Finds the label or type a command names among `counts`: exactly, then
/// ignoring case, then as the start of exactly one, so `hide label sym`
/// finds Symbol and `facet relationship read` finds READ.
fn match_label(typed: &str, counts: &[(Arc<str>, usize)]) -> LabelMatch {
    let labels = || counts.iter().map(|(label, _)| label);
    if let Some(label) = labels().find(|label| &***label == typed) {
        return LabelMatch::Found(label.clone());
    }
    let typed = typed.to_lowercase();
    if let Some(label) = labels().find(|label| label.to_lowercase() == typed) {
        return LabelMatch::Found(label.clone());
    }
    let mut starting: Vec<Arc<str>> = labels()
        .filter(|label| label.to_lowercase().starts_with(&typed))
        .cloned()
        .collect();
    match starting.len() {
        0 => LabelMatch::Missing,
        1 => LabelMatch::Found(starting.remove(0)),
        _ => LabelMatch::Ambiguous(starting),
    }
}

/// "A, B or C", for a status bar list of choices.
fn or_list(items: &[Arc<str>]) -> String {
    match items {
        [] => String::new(),
        [only] => only.to_string(),
        [rest @ .., last] => format!(
            "{} or {last}",
            rest.iter()
                .map(|item| &**item)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn parse_zoom(factor: &str) -> Option<f32> {
    factor.parse::<f32>().ok().filter(|zoom| zoom.is_finite())
}

/// Resolves the argument of the `open` command: surrounding quotes are
/// stripped, `~` expands to the home directory, and relative paths resolve
/// against the working directory Studio was started from.
fn resolve_open_argument(argument: &str) -> Option<PathBuf> {
    let argument = argument.trim();
    let argument = argument
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .or_else(|| {
            argument
                .strip_prefix('\'')
                .and_then(|rest| rest.strip_suffix('\''))
        })
        .unwrap_or(argument);
    if argument.is_empty() {
        return None;
    }
    let home = || std::env::var_os("HOME").map(PathBuf::from);
    let path = if argument == "~" {
        home()?
    } else if let Some(rest) = argument.strip_prefix("~/") {
        home()?.join(rest)
    } else {
        PathBuf::from(argument)
    };
    Some(if path.is_absolute() {
        path
    } else {
        std::env::current_dir().map_or(path.clone(), |directory| directory.join(path))
    })
}

fn openable_database_path(path: &std::path::Path) -> Result<(), String> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => Err(format!(
            "{} is a folder, not a .vg database",
            path.display()
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Err(format!("No database at {}", path.display()))
        }
        Err(error) => Err(format!("Cannot open {} · {error}", path.display())),
    }
}

/// Where each node of the scene in view should stay when another scene is
/// anchored on it (pinned nodes where they are presented, others where the
/// layout put them), and the IDs of the pinned ones.
fn anchor_positions(workspace: &GraphWorkspace, snapshot: &SceneSnapshot) -> (Vec<Vec2>, Vec<u64>) {
    let positions = workspace
        .overview_positions()
        .iter()
        .zip(workspace.positions())
        .zip(workspace.pins())
        .map(|((&overview, &presented), &pinned)| if pinned { presented } else { overview })
        .collect();
    let pinned_ids = workspace
        .pins()
        .iter()
        .enumerate()
        .filter(|&(_, &pinned)| pinned)
        .map(|(index, _)| snapshot.nodes.ids[index])
        .collect();
    (positions, pinned_ids)
}

fn reload_status(delta: SceneDelta, relaid: bool, live: bool) -> String {
    let prefix = if live { "Live" } else { "Reloaded" };
    if delta.is_empty() {
        return format!("{prefix} · up to date");
    }
    let mut parts = Vec::new();
    let mut push = |count: usize, sign: &str, noun: &str| {
        if count > 0 {
            let plural = if count == 1 { "" } else { "s" };
            parts.push(format!("{sign}{count} {noun}{plural}"));
        }
    };
    push(delta.added_nodes, "+", "node");
    push(delta.removed_nodes, "−", "node");
    push(delta.added_edges, "+", "relationship");
    push(delta.removed_edges, "−", "relationship");
    let updated = delta.updated_nodes + delta.updated_edges;
    if updated > 0 {
        parts.push(format!("{updated} updated"));
    }
    let suffix = if relaid { " · new layout" } else { "" };
    format!("{prefix} · {}{suffix}", parts.join(" · "))
}

/// The selected node's grouped relationships. Grouping scans every scene
/// edge, so it runs once per selection and scene rather than once per frame;
/// holding the scene's `Arc` makes the identity check immune to reuse of a
/// freed snapshot's address.
/// How long nodes that a reload added stay ringed on the canvas, and how
/// many are ringed at most.
const FRESH_NODE_DURATION: Duration = Duration::from_secs(6);
const FRESH_NODE_LIMIT: usize = 64;

/// Nodes the last reload added, by ID, ringed on the canvas until they fade.
struct FreshNodes {
    ids: Vec<u64>,
    since: Instant,
}

impl FreshNodes {
    /// Ring opacity: full for the first two seconds, then fading out.
    fn opacity(&self) -> f32 {
        let elapsed = self.since.elapsed().as_secs_f32();
        let duration = FRESH_NODE_DURATION.as_secs_f32();
        (1.0 - (elapsed - 2.0).max(0.0) / (duration - 2.0)).clamp(0.0, 1.0)
    }
}

/// The relationship under the pointer, by ID, and where on the canvas the
/// pointer first found it, which is where its card opens.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HoveredEdge {
    id: u64,
    anchor: Vec2,
}

struct ConnectionCache {
    snapshot: Arc<SceneSnapshot>,
    node: usize,
    connections: Rc<NodeConnections>,
}

type ReadyCallback = Box<dyn FnOnce(&mut Window, &mut Context<StudioView>)>;

pub struct StudioView {
    state: LoadState,
    database_path: Option<PathBuf>,
    workspace: Option<Rc<RefCell<GraphWorkspace>>>,
    graph_navigator_cache: Rc<RefCell<GraphNavigatorCache>>,
    node_label_counts: Arc<[(Arc<str>, usize)]>,
    relationship_counts: Arc<[(Arc<str>, usize)]>,
    /// The colour of each node label seen in this database.
    label_colors: Arc<LabelColors>,
    camera: Camera,
    camera_motion: CameraMotion,
    saved_overview_camera: Option<Camera>,
    context_focus_active: bool,
    /// The last context laid out; shown only while `context_focus_active`.
    context_summary: Option<ContextSummary>,
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
    connection_cache: RefCell<Option<ConnectionCache>>,
    /// Connection groups the inspector lists in full for the current
    /// selection, by relationship type and direction.
    expanded_connections: Vec<(Arc<str>, bool)>,
    selection_back: Vec<SelectedElement>,
    selection_forward: Vec<SelectedElement>,
    command_history: CommandHistory,
    history_file: Option<PathBuf>,
    history_browse: Option<HistoryBrowse>,
    node_menu: Option<NodeMenu>,
    facet_menu: Option<FacetMenu>,
    help_visible: bool,
    /// The node under the pointer, by ID so a reload cannot point it at
    /// another node.
    hovered_node: Option<u64>,
    hovered_edge: Option<HoveredEdge>,
    /// The camera the pointer's hover was found under. Once the view moves
    /// under a still pointer, the hover no longer names what is under it.
    hover_camera: Option<Camera>,
    fresh_nodes: Option<FreshNodes>,
    /// Nodes taken out of view, and the scene they were taken from.
    hidden: Option<HiddenScene>,
    recent: RecentDatabases,
    /// Where the recent list is saved; `None` keeps it in memory, as tests do.
    recent_file: Option<PathBuf>,
    /// The open database as the recent list names it.
    recent_current: Option<PathBuf>,
    generation: u64,
    load_task: Option<Task<()>>,
    fingerprint: Option<DatabaseFingerprint>,
    live_refresh: bool,
    live_task: Option<Task<()>>,
    reload_in_progress: bool,
    reload_waiting: bool,
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
        let query_input = cx.new(|cx| {
            InputState::new(window, cx).placeholder("Search nodes + relationships, or type help")
        });
        let subscription = cx.subscribe_in(&query_input, window, Self::on_query_event);
        let focus_handle = cx.focus_handle().tab_stop(true);
        // Studio's shortcuts are bound to its key context, which is only on
        // the dispatch path once something inside the view has focus; without
        // this, Cmd-O and Cmd-K did nothing until the canvas was clicked.
        window.focus(&focus_handle, cx);
        let bezel_search_focus = std::array::from_fn(|_| cx.focus_handle());
        let bezel_zoom_focus = std::array::from_fn(|_| cx.focus_handle());
        let bezel_layout_focus = std::array::from_fn(|_| cx.focus_handle());
        let bezel_release_focus = cx.focus_handle();
        let bezel_overview_focus = cx.focus_handle();
        let (recent_file, history_file) = if cfg!(test) {
            (None, None)
        } else {
            (
                RecentDatabases::default_file(),
                CommandHistory::default_file(),
            )
        };
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
        let mut label_colors = LabelColors::default();
        if let LoadState::Ready(snapshot) = &state {
            label_colors.extend(snapshot);
        }
        let mut view = Self {
            state,
            database_path: path.clone(),
            workspace,
            graph_navigator_cache: Rc::new(RefCell::new(GraphNavigatorCache::default())),
            node_label_counts,
            relationship_counts,
            label_colors: Arc::new(label_colors),
            camera: Camera::default(),
            camera_motion: CameraMotion::new(Camera::default()),
            saved_overview_camera: None,
            context_focus_active: false,
            context_summary: None,
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
            connection_cache: RefCell::new(None),
            expanded_connections: Vec::new(),
            selection_back: Vec::new(),
            selection_forward: Vec::new(),
            command_history: history_file
                .as_deref()
                .map(CommandHistory::load)
                .unwrap_or_default(),
            history_file,
            history_browse: None,
            node_menu: None,
            facet_menu: None,
            help_visible: false,
            hovered_node: None,
            hover_camera: None,
            hovered_edge: None,
            fresh_nodes: None,
            hidden: None,
            recent: recent_file
                .as_deref()
                .map(RecentDatabases::load)
                .unwrap_or_default(),
            recent_file,
            recent_current: None,
            generation: 0,
            load_task: None,
            fingerprint: None,
            live_refresh: path.is_some() && live_refresh_default(),
            live_task: None,
            reload_in_progress: false,
            reload_waiting: false,
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
                .unwrap_or_else(|_| "gemma".into())
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
                "Demo scene · ⌘O opens a .vg database, or drop one here".into()
            },
            on_ready: None,
            on_search_ready: None,
            on_path_ready: None,
            on_layout_ready: None,
            _subscriptions: vec![subscription],
        };
        if let Some(path) = path {
            view.open(path, window, cx);
            if view.live_refresh {
                view.start_live_refresh(window, cx);
            }
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
    /// screenshot runner, whose compositor does not deliver display-link frames.
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
                    // Fingerprint before opening: a commit that lands during
                    // the open then reads as a change on the next live tick.
                    let fingerprint = DatabaseFingerprint::read(&path).ok();
                    let snapshot = SceneSnapshot::open(
                        path,
                        SnapshotOptions::default(),
                        LayoutOptions::default(),
                    );
                    (snapshot, fingerprint, started.elapsed())
                })
                .await;
            this.update_in(cx, |this, window, cx| {
                if this.generation != generation {
                    return;
                }
                let (snapshot, fingerprint, elapsed) = result;
                this.load_ms = Some(elapsed.as_secs_f64() * 1_000.0);
                this.reload_in_progress = false;
                this.reload_waiting = false;
                match snapshot {
                    Ok(snapshot) => {
                        if let Some(path) = this.database_path.clone() {
                            let path = std::fs::canonicalize(&path).unwrap_or(path);
                            this.recent.remember(path.clone());
                            this.recent_current = Some(path);
                            this.save_recent(cx);
                        }
                        this.fingerprint = fingerprint;
                        window.set_window_title(&format!(
                            "{} — Vecgra Studio",
                            snapshot.database_name
                        ));
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
                        this.graph_navigator_cache.borrow_mut().clear();
                        this.node_label_counts = Arc::from(snapshot.label_counts());
                        Arc::make_mut(&mut this.label_colors).extend(&snapshot);
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
                        if let Some(path) = this.database_path.as_deref() {
                            let path = std::fs::canonicalize(path).unwrap_or(path.to_path_buf());
                            this.recent.forget(&path);
                            this.save_recent(cx);
                        }
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

    /// Replaces the current scene with another database. Every presentation
    /// state that indexes the old scene is retired before the load starts, so
    /// nothing from the previous graph can be applied to the new one.
    fn open_database(&mut self, path: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        let name: SharedString = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("database")
            .to_string()
            .into();
        if let Err(problem) = openable_database_path(&path) {
            self.status = problem.into();
            cx.notify();
            return;
        }
        self.retire_transient_presentation();
        self.set_selection(None);
        self.selection_back.clear();
        self.selection_forward.clear();
        if self.database_path.is_none() {
            self.live_refresh = live_refresh_default();
        }
        self.database_path = Some(path.clone());
        self.recent_current = None;
        self.fresh_nodes = None;
        self.hidden = None;
        self.workspace = None;
        self.graph_navigator_cache.borrow_mut().clear();
        self.node_label_counts = Arc::from([]);
        self.label_colors = Arc::default();
        self.relationship_counts = Arc::from([]);
        self.fingerprint = None;
        self.reload_in_progress = false;
        self.reload_waiting = false;
        self.load_ms = None;
        self.status = format!("Opening {name}…").into();
        self.state = LoadState::Loading { name };
        self.open(path, window, cx);
        if self.live_refresh && self.live_task.is_none() {
            self.start_live_refresh(window, cx);
        }
        cx.notify();
    }

    /// Writes the recent list off the application thread. A failed write only
    /// costs the list in the next session, so it is not reported.
    fn save_recent(&self, cx: &mut Context<Self>) {
        let Some(file) = self.recent_file.clone() else {
            return;
        };
        let recent = self.recent.clone();
        cx.background_spawn(async move {
            let _ = recent.save(&file);
        })
        .detach();
    }

    /// Recently opened databases other than the one on screen, newest first.
    fn other_recent_databases(&self) -> impl Iterator<Item = &PathBuf> {
        self.recent
            .paths()
            .iter()
            .filter(|path| Some(*path) != self.recent_current.as_ref())
    }

    fn prompt_open_database(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open".into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            let Ok(Ok(Some(paths))) = paths.await else {
                return;
            };
            let Some(path) = paths.into_iter().next() else {
                return;
            };
            this.update_in(cx, |this, window, cx| this.open_database(path, window, cx))
                .ok();
        })
        .detach();
    }

    /// What the canvas draws now, for the canvas itself and for export.
    fn canvas_presentation(&self, snapshot: &SceneSnapshot) -> GraphCanvasPresentation {
        GraphCanvasPresentation {
            camera: self.camera,
            world_bounds: self.world_bounds,
            selection: self.selection,
            dragging: match self.drag {
                Some(DragState::Node { index, .. }) => Some(index),
                Some(DragState::Canvas { .. }) | None => None,
            },
            emphasis: self.lens.as_ref().map(LensTransition::emphasis),
            path_endpoints: self.path_endpoints,
            hovered_node: self.hovered_node.and_then(|id| snapshot.node_index(id)),
            hovered_edge: self
                .hovered_edge
                .and_then(|hovered| snapshot.edge_index(hovered.id)),
            label_colors: self.label_colors.clone(),
        }
    }

    fn prompt_export_view(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let directory = self
            .database_path
            .as_deref()
            .and_then(std::path::Path::parent)
            .map(std::path::Path::to_path_buf)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_default();
        let name = self
            .database_path
            .as_deref()
            .and_then(std::path::Path::file_stem)
            .map_or_else(|| "graph".into(), |stem| stem.to_string_lossy());
        let path = cx.prompt_for_new_path(&directory, Some(&format!("{name}.svg")));
        cx.spawn_in(window, async move |this, cx| {
            let Ok(Ok(Some(path))) = path.await else {
                return;
            };
            this.update(cx, |this, cx| this.export_view(&path, cx)).ok();
        })
        .detach();
    }

    /// Saves the graph view as drawn, at the canvas's size, as an SVG image.
    fn export_view(&mut self, path: &std::path::Path, cx: &mut Context<Self>) {
        let (Some(snapshot), Some(workspace), Some(bounds)) =
            (self.snapshot(), self.workspace.as_ref(), self.canvas_bounds)
        else {
            self.status = "Open a graph to export its view".into();
            cx.notify();
            return;
        };
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let svg = scene_svg(
            snapshot,
            &workspace.borrow(),
            &self.canvas_presentation(snapshot),
            viewport,
        );
        self.status = match std::fs::write(path, svg) {
            Ok(()) => format!("Exported the view to {}", path.display()).into(),
            Err(error) => format!("Could not export to {}: {error}", path.display()).into(),
        };
        cx.notify();
    }

    fn on_external_paths_dropped(
        &mut self,
        paths: &ExternalPaths,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(path) = paths
            .paths()
            .iter()
            .find(|path| path.extension().is_some_and(|extension| extension == "vg"))
            .or_else(|| paths.paths().first())
            .cloned()
        else {
            return;
        };
        self.open_database(path, window, cx);
    }

    fn inspector_connections(
        &self,
        snapshot: &Arc<SceneSnapshot>,
        node: usize,
    ) -> Rc<NodeConnections> {
        let mut cache = self.connection_cache.borrow_mut();
        if let Some(cached) = cache.as_ref()
            && cached.node == node
            && Arc::ptr_eq(&cached.snapshot, snapshot)
        {
            return cached.connections.clone();
        }
        let connections =
            Rc::new(snapshot.node_connections(node, panels::INSPECTOR_CONNECTIONS_EXPANDED));
        *cache = Some(ConnectionCache {
            snapshot: snapshot.clone(),
            node,
            connections: connections.clone(),
        });
        connections
    }

    /// Walks the graph from the inspector: selects a related node and springs
    /// the camera to it at the current zoom, so the canvas follows along.
    fn select_related_node(
        &mut self,
        index: usize,
        via_edge: Option<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        if index >= snapshot.nodes.ids.len() {
            return;
        }
        self.set_selection(Some(SceneSelection::Node(index)));
        // The row that previewed this node is replaced by the new
        // selection's rows, so it will not report the pointer leaving.
        self.hovered_node = None;
        if let Some(position) = self
            .workspace
            .as_ref()
            .and_then(|workspace| workspace.borrow().position(index))
        {
            self.camera_motion.retarget(Camera {
                center: position,
                zoom: self.camera.zoom,
            });
        }
        self.status = match via_edge {
            Some(edge) => format!(
                "Selected {} · via {}",
                status_node_name(&snapshot, index),
                snapshot.edges.labels[edge]
            ),
            None => format!("Selected {}", status_node_name(&snapshot, index)),
        }
        .into();
        self.begin_presentation_motion(window, cx);
    }

    /// Rereads the open database and swaps in the new scene without moving
    /// anything that survived. A manual reload first retires any search,
    /// path, lens, or context so their per-element state cannot outlive the
    /// scene it was computed for.
    fn reload(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(path) = self.database_path.clone() else {
            self.status = "The demo scene has no database to reload".into();
            cx.notify();
            return;
        };
        let (Some(previous), Some(workspace)) = (self.snapshot().cloned(), self.workspace.clone())
        else {
            return;
        };
        if self.has_transient_presentation() {
            self.retire_transient_presentation();
        }
        let (previous_positions, pinned_ids) = anchor_positions(&workspace.borrow(), &previous);
        let selected = self.selected_element(self.selection);
        let hidden = self.hidden.as_ref().map(|scene| scene.hidden.clone());
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        self.reload_in_progress = true;
        self.reload_waiting = false;
        self.load_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let fingerprint = DatabaseFingerprint::read(&path).ok();
                    // Hidden nodes stay hidden; the full scene is kept so
                    // they can be shown again.
                    let reloaded = match hidden {
                        None => SceneSnapshot::reload(
                            &path,
                            &previous,
                            &previous_positions,
                            SnapshotOptions::default(),
                            LayoutOptions::default(),
                        )
                        .map(|reloaded| (reloaded, None)),
                        Some(hidden) => SceneSnapshot::reload_without(
                            &path,
                            &previous,
                            &previous_positions,
                            &hidden,
                            SnapshotOptions::default(),
                            LayoutOptions::default(),
                        )
                        .map(|(reloaded, full)| (reloaded, Some(full))),
                    };
                    (reloaded, fingerprint)
                })
                .await;
            this.update_in(cx, |this, _window, cx| {
                if this.generation != generation {
                    return;
                }
                this.reload_in_progress = false;
                match result {
                    (Ok((reloaded, full)), fingerprint) => {
                        this.fingerprint = fingerprint;
                        if let (Some(hidden), Some(full)) = (this.hidden.as_mut(), full) {
                            hidden.full = Arc::new(full);
                            hidden.recount();
                        }
                        this.install_reloaded_scene(reloaded, &pinned_ids, selected);
                    }
                    (Err(error), _) => {
                        this.status = format!("Reload failed · {error}").into();
                    }
                }
                cx.notify();
            })
            .ok();
        }));
        self.status = "Reloading database…".into();
        cx.notify();
    }

    fn install_reloaded_scene(
        &mut self,
        reloaded: ReloadedScene,
        pinned_ids: &[u64],
        selected: Option<SelectedElement>,
    ) {
        if self.has_transient_presentation() {
            // A search or context began while the reload was in flight; its
            // state indexes the current scene, so keep it and retry later.
            self.reload_waiting = true;
            self.fingerprint = None;
            self.status = "Reload waiting · return to overview to apply it".into();
            return;
        }
        let ReloadedScene {
            snapshot,
            entrances,
            delta,
            relaid,
        } = reloaded;
        // A relaid scene is new everywhere, so nothing is singled out.
        self.fresh_nodes = (!relaid && delta.added_nodes > 0)
            .then(|| {
                let previous: std::collections::HashSet<u64> = self
                    .snapshot()
                    .map(|previous| previous.nodes.ids.iter().copied().collect())
                    .unwrap_or_default();
                snapshot
                    .nodes
                    .ids
                    .iter()
                    .copied()
                    .filter(|id| !previous.contains(id))
                    .take(FRESH_NODE_LIMIT)
                    .collect::<Vec<_>>()
            })
            .filter(|ids| !ids.is_empty())
            .map(|ids| FreshNodes {
                ids,
                since: Instant::now(),
            });
        let snapshot = Arc::new(snapshot);
        let mut workspace = GraphWorkspace::new(&snapshot);
        for &id in pinned_ids {
            if let Some(index) = snapshot.node_index(id) {
                workspace.set_pinned(index, true);
            }
        }
        workspace.enter_from(&entrances);
        let moving = workspace.is_moving();
        self.workspace = Some(Rc::new(RefCell::new(workspace)));
        self.graph_navigator_cache.borrow_mut().clear();
        self.node_label_counts = Arc::from(snapshot.label_counts());
        Arc::make_mut(&mut self.label_colors).extend(&snapshot);
        self.relationship_counts = Arc::from(snapshot.relationship_counts());
        let selection = selected.and_then(|selected| match selected {
            SelectedElement::Node(id) => snapshot.node_index(id).map(SceneSelection::Node),
            SelectedElement::Edge(id) => snapshot.edge_index(id).map(SceneSelection::Edge),
        });
        self.set_selection(selection);
        if relaid {
            self.world_bounds = snapshot.bounds;
            self.camera = Camera::fit(snapshot.bounds);
            self.camera_motion.cancel_at(self.camera);
            self.layout_kind = LayoutKind::Auto;
        }
        self.last_motion_frame = moving.then(Instant::now);
        self.status = reload_status(delta, relaid, self.live_refresh).into();
        self.state = LoadState::Ready(snapshot);
    }

    /// Search results, paths, lenses, and context layouts all hold per-index
    /// state for the current scene. Live refresh waits while any is active.
    fn has_transient_presentation(&self) -> bool {
        self.context_focus_active
            || self.active_facet.is_some()
            || self.lens.is_some()
            || self.drag.is_some()
            || self.layout_in_progress
            || !matches!(self.search_state, SearchState::Idle)
            || !matches!(self.path_state, PathState::Idle)
    }

    fn retire_transient_presentation(&mut self) {
        self.search_generation = self.search_generation.wrapping_add(1);
        drop(self.search_task.take());
        self.search_state = SearchState::Idle;
        self.path_generation = self.path_generation.wrapping_add(1);
        drop(self.path_task.take());
        self.path_state = PathState::Idle;
        self.path_endpoints = None;
        self.layout_generation = self.layout_generation.wrapping_add(1);
        drop(self.layout_task.take());
        self.layout_in_progress = false;
        self.context_focus_active = false;
        self.active_facet = None;
        self.lens = None;
        self.drag = None;
        if let Some(camera) = self.saved_overview_camera.take() {
            self.camera = camera;
            self.camera_motion.cancel_at(camera);
        }
    }

    fn set_live_refresh(&mut self, enabled: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.database_path.is_none() {
            self.status = "Live refresh needs a .vg database".into();
            cx.notify();
            return;
        }
        self.live_refresh = enabled;
        if enabled {
            self.start_live_refresh(window, cx);
            self.status = "Live refresh on · new commits appear as they land".into();
        } else {
            self.live_task = None;
            self.reload_waiting = false;
            self.status = "Live refresh off · ⌘R reloads on demand".into();
        }
        cx.notify();
    }

    fn start_live_refresh(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.live_task = Some(cx.spawn_in(window, async move |this, cx| {
            loop {
                cx.background_executor().timer(LIVE_REFRESH_INTERVAL).await;
                let Ok(Some(path)) = this.update(cx, |this, _| {
                    this.live_refresh
                        .then(|| this.database_path.clone())
                        .flatten()
                }) else {
                    break;
                };
                let fingerprint = cx
                    .background_spawn(async move { DatabaseFingerprint::read(&path).ok() })
                    .await;
                let keep_polling = this.update_in(cx, |this, window, cx| {
                    this.on_live_fingerprint(fingerprint, window, cx);
                    this.live_refresh
                });
                if !matches!(keep_polling, Ok(true)) {
                    break;
                }
            }
        }));
    }

    fn on_live_fingerprint(
        &mut self,
        fingerprint: Option<DatabaseFingerprint>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // A missing file is usually a compaction swap in progress.
        let Some(fingerprint) = fingerprint else {
            return;
        };
        if self.reload_in_progress || !self.is_ready() {
            return;
        }
        if self.fingerprint == Some(fingerprint) && !self.reload_waiting {
            return;
        }
        if self.has_transient_presentation() {
            if !self.reload_waiting {
                self.reload_waiting = true;
                cx.notify();
            }
            return;
        }
        self.reload(window, cx);
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

    /// The camera that frames `focus` on the canvas with `padding` around
    /// it, at a zoom within `zoom`, and below the graph navigator in the
    /// canvas's top-right corner so that it covers none of what is framed.
    fn framing(&self, focus: Rect, padding: f32, zoom: RangeInclusive<f32>) -> Option<Camera> {
        let bounds = self.canvas_bounds?;
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        Some(Camera::framed_below(
            focus,
            self.world_bounds,
            viewport,
            padding,
            FRAMING_TOP,
            zoom,
        ))
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

    /// Every selection change goes through here. Leaving an element records
    /// it for Back, by ID so the history survives reloads, and a new
    /// direction clears Forward the way a browser does.
    fn set_selection(&mut self, selection: Option<SceneSelection>) {
        if self.selection != selection {
            self.inspector_scroll_handle.set_offset(Default::default());
            self.expanded_connections.clear();
            let previous = self.selected_element(self.selection);
            if previous.is_some() && previous != self.selected_element(selection) {
                self.selection_back.extend(previous);
                if self.selection_back.len() > SELECTION_HISTORY_LIMIT {
                    self.selection_back.remove(0);
                }
                self.selection_forward.clear();
            }
        }
        self.selection = selection;
    }

    fn selected_element(&self, selection: Option<SceneSelection>) -> Option<SelectedElement> {
        let snapshot = self.snapshot()?;
        match selection? {
            SceneSelection::Node(index) => snapshot
                .nodes
                .ids
                .get(index)
                .copied()
                .map(SelectedElement::Node),
            SceneSelection::Edge(index) => snapshot
                .edges
                .ids
                .get(index)
                .copied()
                .map(SelectedElement::Edge),
        }
    }

    /// Steps through selection history. Elements that are no longer in the
    /// scene, for example after a reload removed them, are skipped.
    fn navigate_selection_history(
        &mut self,
        backward: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(snapshot) = self.snapshot().cloned() else {
            return;
        };
        let current = self.selected_element(self.selection);
        loop {
            let next = if backward {
                self.selection_back.pop()
            } else {
                self.selection_forward.pop()
            };
            let Some(element) = next else {
                self.status = if backward {
                    "No earlier selection".into()
                } else {
                    "No later selection".into()
                };
                cx.notify();
                return;
            };
            let resolved = match element {
                SelectedElement::Node(id) => snapshot.node_index(id).map(SceneSelection::Node),
                SelectedElement::Edge(id) => snapshot.edge_index(id).map(SceneSelection::Edge),
            };
            let Some(selection) = resolved else {
                continue;
            };
            if let Some(current) = current {
                if backward {
                    self.selection_forward.push(current);
                } else {
                    self.selection_back.push(current);
                }
            }
            self.inspector_scroll_handle.set_offset(Default::default());
            self.selection = Some(selection);
            self.spring_to_selection(false);
            let name = match selection {
                SceneSelection::Node(index) => status_node_name(&snapshot, index),
                SceneSelection::Edge(index) => status_edge_name(&snapshot, index),
            };
            self.status = format!("{} · {name}", if backward { "Back" } else { "Forward" }).into();
            self.begin_presentation_motion(window, cx);
            return;
        }
    }

    /// Springs the camera, at the current zoom, to the selected node or the
    /// middle of the selected relationship. With `only_if_hidden`, the camera
    /// moves only when part of the selection is outside the visible canvas.
    /// The caller starts the motion.
    fn spring_to_selection(&mut self, only_if_hidden: bool) {
        const MARGIN: f32 = 48.0;
        let (Some(snapshot), Some(workspace), Some(selection)) =
            (self.snapshot(), self.workspace.as_ref(), self.selection)
        else {
            return;
        };
        let focus = match selection {
            SceneSelection::Node(index) => vec![index],
            SceneSelection::Edge(index) => vec![
                snapshot.edges.sources[index] as usize,
                snapshot.edges.targets[index] as usize,
            ],
        };
        let points: Vec<_> = {
            let workspace = workspace.borrow();
            focus
                .iter()
                .filter_map(|&index| workspace.position(index))
                .collect()
        };
        if points.is_empty() {
            return;
        }
        if only_if_hidden && let Some(bounds) = self.canvas_bounds {
            let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
            let visible = points.iter().all(|&point| {
                let local = self.camera.project(point, self.world_bounds, viewport);
                local.x >= MARGIN
                    && local.y >= MARGIN
                    && local.x <= viewport.x - MARGIN
                    && local.y <= viewport.y - MARGIN
            });
            if visible {
                return;
            }
        }
        self.camera_motion.retarget(Camera {
            center: Rect::from_points(&points).center(),
            zoom: self.camera.zoom,
        });
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
            "{} lens · {} · {match_count} matches · ⌫ hides · Esc clears",
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
        // What a bare ID names lights up like the best result.
        let id_matches = self.id_matches(&report.query);
        let workspace = workspace.borrow();
        let matches = id_matches
            .into_iter()
            .map(|selection| (selection, 1.0))
            .chain(report.hits.iter().filter_map(|hit| {
                let score = (hit.score / maximum).clamp(0.18, 1.0);
                Some((hit.scene_selection(&snapshot)?, score))
            }));
        for (selection, score) in matches {
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
        if !focus_points.is_empty()
            && let Some(target) = self.framing(Rect::from_points(&focus_points), 96.0, 0.08..=4.8)
        {
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
        let hidden = self.hidden.as_ref().map(|scene| scene.hidden.clone());

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
        self.status = format!(
            "Tracing exact path · {} → {}…",
            self.status_node_name_by_id(start),
            self.status_node_name_by_id(end)
        )
        .into();
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
                            // Nodes the user hid stay hidden even when the
                            // path runs through them; the rail still names them.
                            .map(|enriched| match &hidden {
                                Some(hidden) => enriched.without_hidden(hidden),
                                None => enriched,
                            })
                            .filter(|enriched| {
                                enriched.nodes.ids.len() != snapshot.nodes.ids.len()
                                    || enriched.edges.ids.len() != snapshot.edges.ids.len()
                            })
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
        Arc::make_mut(&mut self.label_colors).extend(&snapshot);
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
        if draft.max_hops == max_hops || !(1..=MAX_PATH_HOPS).contains(&max_hops) {
            return;
        }
        draft.max_hops = max_hops;
        let draft = *draft;
        self.status = self.path_draft_status(draft);
        cx.notify();
    }

    /// Shift-click: traces an exact path from the selected node, or from a
    /// path origin already chosen, to `index`. Returns false, leaving the
    /// click to select as usual, when there is nothing to trace from.
    fn trace_path_from_selection(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if !matches!(self.path_state, PathState::ChoosingEnd(_)) {
            let Some(start) = self.selection.and_then(SceneSelection::node) else {
                return false;
            };
            if start == index {
                return false;
            }
            self.choose_path_start(start, window, cx);
        }
        if self.path_query_to_node(index).is_none() {
            // No database to trace in; choosing the origin said so.
            return true;
        }
        self.trace_path_to_node(index, window, cx);
        true
    }

    /// Runs the shown path search again with one of its bounds widened.
    fn retry_evidence_path(
        &mut self,
        retry: PathRetry,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let PathState::Ready(report) = &self.path_state else {
            return;
        };
        let mut query = EvidencePathQuery {
            start: report.start.id,
            end: report.end.id,
            direction: report.direction,
            relationship_label: report.relationship_label.as_deref().map(str::to_owned),
            max_hops: report.max_hops,
        };
        match retry {
            PathRetry::EitherDirection => query.direction = PathDirection::Both,
            PathRetry::AnyRelationship => query.relationship_label = None,
            PathRetry::MoreHops(max_hops) => query.max_hops = max_hops,
        }
        self.start_evidence_path(query, window, cx);
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

    /// A node as the status bar names it, or `node:<id>` when it is not in
    /// the current scene.
    fn status_node_name_by_id(&self, id: u64) -> String {
        self.snapshot()
            .and_then(|snapshot| Some(status_node_name(snapshot, snapshot.node_index(id)?)))
            .unwrap_or_else(|| format!("node:{id}"))
    }

    fn path_draft_status(&self, draft: PathDraft) -> SharedString {
        if let Some(end) = self
            .path_destination_candidate()
            .and_then(|index| self.snapshot()?.nodes.ids.get(index).copied())
        {
            format!(
                "Destination ready · {} · {} · up to {} hops · Enter traces exact path",
                self.status_node_name_by_id(end),
                path_direction_label(draft.direction),
                draft.max_hops
            )
            .into()
        } else {
            format!(
                "Path origin · {} · {} · up to {} hops · select a destination",
                self.status_node_name_by_id(draft.start),
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
                node_scores[index] = if endpoint { 1.0 } else { PATH_INTERIOR_SCORE };
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
        if !focus_points.is_empty()
            && let Some(target) = self.framing(Rect::from_points(&focus_points), 116.0, 1.8..=24.0)
        {
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
        if let Some(workspace) = self.workspace.as_ref() {
            let endpoints = {
                let workspace = workspace.borrow();
                snapshot
                    .node_index(from)
                    .zip(snapshot.node_index(to))
                    .and_then(|(from, to)| workspace.position(from).zip(workspace.position(to)))
            };
            if let Some(target) = endpoints.and_then(|(from, to)| {
                self.framing(Rect::from_points(&[from, to]), 164.0, 3.0..=24.0)
            }) {
                self.camera_motion.retarget(target);
            }
        }
        self.status = match self.snapshot().and_then(|snapshot| {
            let index = snapshot.edge_index(edge_id)?;
            Some(status_edge_name(snapshot, index))
        }) {
            Some(name) => format!("Evidence step {} · {name}", step_index + 1),
            None => format!("Evidence step {} · relationship:{edge_id}", step_index + 1),
        }
        .into();
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
        if let Some(target) = focus_bounds.and_then(|focus| self.framing(focus, 112.0, 2.2..=9.0)) {
            self.camera_motion.retarget(target);
        }
        self.begin_presentation_motion(window, cx);
    }

    /// Presents a bounded two-hop context around a node. The expansion is
    /// branch-balanced and relationship-type-diverse in the scene layer, so a
    /// hub or one prolific relationship cannot consume the whole context budget.
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
        if let Some(target) = focus_bounds.and_then(|focus| self.framing(focus, 112.0, 1.4..=9.0)) {
            self.camera_motion.retarget(target);
        }
        let summary = ContextSummary {
            node: snapshot.nodes.ids[index],
            direct: focus.layers.get(1).map_or(0, Vec::len),
            indirect: focus.layers.get(2).map_or(0, Vec::len),
            relationships: focus.edges.len(),
        };
        self.context_summary = Some(summary);
        self.status = format!(
            "Context · {} · {} · Esc restores overview",
            status_node_name(&snapshot, index),
            summary.counts()
        )
        .into();
        self.begin_presentation_motion(window, cx);
    }

    fn activate_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.help_visible {
            self.set_help_visible(false, window, cx);
            return;
        }
        if self
            .query_input
            .read(cx)
            .focus_handle(cx)
            .is_focused(window)
        {
            return;
        }
        // With a control focused, such as a path step, a list row or a
        // button, Enter belongs to that control.
        if !self.focus_handle.is_focused(window) {
            cx.propagate();
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
        if self.node_menu.take().is_some() || self.facet_menu.take().is_some() {
            cx.notify();
        } else if self.help_visible {
            self.set_help_visible(false, window, cx);
        } else if self.context_focus_active
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

    /// Puts the selection on the clipboard as text; see
    /// [`panels::selection_text`].
    fn copy_selection(&mut self, cx: &mut Context<Self>) {
        let (Some(snapshot), Some(selection)) = (self.snapshot(), self.selection) else {
            self.status = "Select a node or relationship to copy it".into();
            cx.notify();
            return;
        };
        let text = panels::selection_text(snapshot, selection);
        self.status = match selection {
            SceneSelection::Node(index) => {
                format!("Copied {} as text", status_node_name(snapshot, index))
            }
            SceneSelection::Edge(index) => {
                format!("Copied {} as text", status_edge_name(snapshot, index))
            }
        }
        .into();
        cx.write_to_clipboard(gpui::ClipboardItem::new_string(text));
        cx.notify();
    }

    /// Adds a submission to the history and writes it off the application
    /// thread. A failed write only costs the history in the next session, so
    /// it is not reported.
    fn remember_command(&mut self, query: &str, cx: &mut Context<Self>) {
        self.history_browse = None;
        if !self.command_history.remember(query) {
            return;
        }
        let Some(file) = self.history_file.clone() else {
            return;
        };
        let history = self.command_history.clone();
        cx.background_spawn(async move {
            let _ = history.save(&file);
        })
        .detach();
    }

    /// Up and Down in the command box. While the box holds the search whose
    /// results are listed, they move through the results, as they do with
    /// the graph focused; otherwise they step through earlier searches and
    /// commands, and stepping past the newest restores what was being typed.
    fn step_search_box(&mut self, earlier: bool, window: &mut Window, cx: &mut Context<Self>) {
        let text = self.query_input.read(cx).value().trim().to_string();
        if self.history_browse.is_none()
            && matches!(
                &self.search_state,
                SearchState::Ready(report) if report.query.as_ref() == text && !report.hits.is_empty()
            )
        {
            self.move_search_selection(if earlier { -1 } else { 1 }, cx);
            return;
        }
        let current = self.history_browse.as_ref().map(|browse| browse.index);
        let count = self.command_history.entries().len();
        let target = match (current, earlier) {
            (None, true) => count.checked_sub(1),
            (None, false) => return,
            (Some(index), true) => Some(index.saturating_sub(1)),
            (Some(index), false) => Some(index + 1).filter(|&next| next < count),
        };
        let value = match (target, self.history_browse.take()) {
            (Some(index), browse) => {
                let draft = browse.map_or_else(
                    || self.query_input.read(cx).value().to_string(),
                    |browse| browse.draft,
                );
                self.history_browse = Some(HistoryBrowse { index, draft });
                self.command_history.entries()[index].clone()
            }
            (None, Some(browse)) => browse.draft,
            (None, None) => return,
        };
        self.query_input
            .update(cx, |input, cx| input.set_value(value, window, cx));
        cx.notify();
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
        let report = report.clone();
        let Some(hit) = report.hits.get(index).cloned() else {
            return;
        };
        let Some(snapshot) = self.snapshot() else {
            return;
        };
        let Some(selection) = hit.scene_selection(snapshot) else {
            // Choosing a hidden node shows it again; anything else hidden
            // stays hidden. Showing it retires the search, whose hits still
            // stand.
            if hit.kind_label() == "NODE" && self.reveal_node(hit.id()) {
                self.search_state = SearchState::Ready(report);
                self.activate_search_result(index, window, cx);
                self.status = format!("Shown again · {}", self.status).into();
                return;
            }
            self.status = if self.hit_is_hidden(&hit) {
                format!("{} is hidden · ⇧⌘H shows what is hidden", hit.title)
            } else {
                format!(
                    "{} matches, but is outside this bounded graph view",
                    hit.title
                )
            }
            .into();
            cx.notify();
            return;
        };
        let name = match selection {
            SceneSelection::Node(index) => {
                format!("{} · {}", status_node_name(snapshot, index), hit.label)
            }
            SceneSelection::Edge(index) => status_edge_name(snapshot, index),
        };
        self.set_selection(Some(selection));
        self.selected_search_result = index;
        self.focus_search_result(selection, window, cx);
        self.status = format!(
            "Focused {name} · {:.0}% match · {}-element context",
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
                        let named = match this.id_matches(&query).as_slice() {
                            [] => String::new(),
                            [SceneSelection::Node(_)] => " · ID names a node".into(),
                            [SceneSelection::Edge(_)] => " · ID names a relationship".into(),
                            _ => " · ID names a node and a relationship".into(),
                        };
                        this.status = if count == 0 {
                            format!(
                                "No {} matches{named} · {elapsed_ms:.1} ms",
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

    /// The node under a window position on the canvas, picked with the
    /// click radius.
    fn node_at(&self, position: Point<Pixels>) -> Option<usize> {
        let bounds = self.canvas_bounds?;
        let workspace = self.workspace.as_ref()?.borrow();
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let local = Vec2::new(
            (position.x - bounds.origin.x).into(),
            (position.y - bounds.origin.y).into(),
        );
        hit_test_positions(
            workspace.positions(),
            self.camera,
            self.world_bounds,
            viewport,
            local,
            15.0,
        )
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
            if event.modifiers.shift && self.trace_path_from_selection(index, window, cx) {
                self.drag = None;
                cx.stop_propagation();
                return;
            }
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
            self.status = match self.path_state {
                PathState::ChoosingEnd(draft) => self.path_draft_status(draft),
                PathState::Idle
                | PathState::Searching { .. }
                | PathState::Ready(_)
                | PathState::Failed { .. } => format!(
                    "Selected {} · drag to arrange",
                    status_node_name(&snapshot, index)
                )
                .into(),
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
            self.update_hover(event.position, cx);
            return;
        }
        self.set_hovered_node(None, cx);
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

    /// Tracks the node under a pointer that is not dragging, within the same
    /// radius a click would select it.
    fn update_hover(&mut self, position: gpui::Point<Pixels>, cx: &mut Context<Self>) {
        enum Hit {
            Node(u64),
            Edge(u64, Vec2),
        }
        let hovered = self
            .canvas_bounds
            .filter(|bounds| bounds.contains(&position))
            .zip(self.snapshot().zip(self.workspace.as_ref()))
            .and_then(|(bounds, (snapshot, workspace))| {
                let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
                let local = Vec2::new(
                    (position.x - bounds.origin.x).into(),
                    (position.y - bounds.origin.y).into(),
                );
                let workspace = workspace.borrow();
                if let Some(index) = hit_test_positions(
                    workspace.positions(),
                    self.camera,
                    self.world_bounds,
                    viewport,
                    local,
                    15.0,
                ) {
                    return Some(Hit::Node(snapshot.nodes.ids[index]));
                }
                // Relationships are only drawn individually below the
                // overview, and are picked with the same radius as a click.
                (detail_level(self.camera, snapshot.nodes.ids.len()) != DetailLevel::Overview)
                    .then(|| {
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
                    .flatten()
                    .map(|index| Hit::Edge(snapshot.edges.ids[index], local))
            });
        match hovered {
            Some(Hit::Node(node)) => self.set_hovered_node(Some(node), cx),
            Some(Hit::Edge(id, anchor)) => {
                if self.hovered_edge.is_none_or(|hovered| hovered.id != id) {
                    self.hovered_node = None;
                    self.hovered_edge = Some(HoveredEdge { id, anchor });
                    cx.notify();
                }
            }
            None => self.set_hovered_node(None, cx),
        }
        self.hover_camera = Some(self.camera);
    }

    /// Points out a node on the canvas while a list row that names it is
    /// hovered, and stops when the pointer leaves that row.
    fn preview_node(&mut self, node: u64, hovered: bool, cx: &mut Context<Self>) {
        if hovered {
            self.set_hovered_node(Some(node), cx);
        } else if self.hovered_node == Some(node) {
            self.set_hovered_node(None, cx);
        }
    }

    /// Points out a relationship on the canvas while a list row that names
    /// it is hovered: its ends are ringed and its card sits at the middle of
    /// its line. Relationships whose middle is off the canvas get no card.
    fn preview_edge(&mut self, edge: u64, hovered: bool, cx: &mut Context<Self>) {
        if !hovered {
            if self.hovered_edge.is_some_and(|hovered| hovered.id == edge) {
                self.set_hovered_node(None, cx);
            }
            return;
        }
        let anchor = self
            .snapshot()
            .zip(self.workspace.as_ref())
            .zip(self.canvas_bounds)
            .and_then(|((snapshot, workspace), bounds)| {
                let index = snapshot.edge_index(edge)?;
                let workspace = workspace.borrow();
                let source = workspace.position(snapshot.edges.sources[index] as usize)?;
                let target = workspace.position(snapshot.edges.targets[index] as usize)?;
                let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
                let anchor =
                    self.camera
                        .project((source + target) * 0.5, self.world_bounds, viewport);
                (anchor.x >= 0.0
                    && anchor.y >= 0.0
                    && anchor.x <= viewport.x
                    && anchor.y <= viewport.y)
                    .then_some(anchor)
            });
        let Some(anchor) = anchor else {
            return;
        };
        self.set_hovered_node(None, cx);
        self.hovered_edge = Some(HoveredEdge { id: edge, anchor });
    }

    /// Sets the hovered node, which also ends any relationship hover.
    fn set_hovered_node(&mut self, node: Option<u64>, cx: &mut Context<Self>) {
        self.hover_camera = None;
        if self.hovered_node != node || self.hovered_edge.is_some() {
            self.hovered_node = node;
            self.hovered_edge = None;
            cx.notify();
        }
    }

    fn on_mouse_up(&mut self, _event: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
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
                        self.status =
                            format!("Selected {}", status_edge_name(snapshot, edge_index)).into();
                    }
                } else if self.selection.is_some() {
                    // A click on empty canvas clears the selection, which Back
                    // can return to; with nothing selected it changes nothing.
                    self.set_selection(None);
                    self.status = "Selection cleared · ⌘[ selects it again".into();
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
        // A menu stays where it opened; the node it names would not.
        self.node_menu = None;
        self.facet_menu = None;
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
        self.update_hover(event.position, cx);
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
        self.update_hover(event.position, cx);
        cx.notify();
    }

    fn on_query_event(
        &mut self,
        input: &Entity<InputState>,
        event: &InputEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(event, InputEvent::Change) {
            // Editing a recalled command makes it the text being written;
            // recalling only sets the text, which raises no event.
            self.history_browse = None;
            // The box's Enter hint depends on its text.
            cx.notify();
            return;
        }
        if !matches!(event, InputEvent::PressEnter { .. }) {
            return;
        }
        let query = input.read(cx).value().trim().to_string();
        self.remember_command(&query, cx);
        self.submit_query(&query, window, cx);
    }

    /// What Enter in the command box does: run a command, search, or go to
    /// the selected result of the search already shown, or with none, to
    /// the element a bare ID names.
    fn submit_query(&mut self, query: &str, window: &mut Window, cx: &mut Context<Self>) {
        if let SearchState::Ready(report) = &self.search_state
            && report.query.as_ref() == query
            && report.mode == self.search_mode
        {
            match self.id_matches(query).first() {
                Some(&selection) if report.hits.is_empty() => {
                    self.go_to_id_match(selection, window, cx);
                }
                _ => self.activate_search_result(self.selected_search_result, window, cx),
            }
            return;
        }
        self.execute_command(query, window, cx);
    }

    /// The node and the relationship a search for a bare number names by
    /// ID, so that "145" offers node 145, not only text that mentions 145.
    pub(super) fn id_matches(&self, query: &str) -> Vec<SceneSelection> {
        let (Some(snapshot), Ok(id)) = (self.snapshot(), query.trim().parse::<u64>()) else {
            return Vec::new();
        };
        snapshot
            .node_index(id)
            .map(SceneSelection::Node)
            .into_iter()
            .chain(snapshot.edge_index(id).map(SceneSelection::Edge))
            .collect()
    }

    /// Selects an ID match as `node <id>` or `edge <id>` would.
    pub(super) fn go_to_id_match(
        &mut self,
        selection: SceneSelection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(snapshot) = self.snapshot() else {
            return;
        };
        let command = match selection {
            SceneSelection::Node(index) => format!("node {}", snapshot.nodes.ids[index]),
            SceneSelection::Edge(index) => format!("edge {}", snapshot.edges.ids[index]),
        };
        self.execute_command(&command, window, cx);
    }

    /// The node label, or with `relationship` the relationship type, that a
    /// command names, forgiving case and an unambiguous start; otherwise
    /// says in the status bar what is in view instead.
    fn command_label(
        &mut self,
        typed: &str,
        relationship: bool,
        cx: &mut Context<Self>,
    ) -> Option<Arc<str>> {
        let (counts, kind) = if relationship {
            (self.relationship_counts.clone(), "relationship type")
        } else {
            (self.node_label_counts.clone(), "node label")
        };
        let status = match match_label(typed, &counts) {
            LabelMatch::Found(label) => return Some(label),
            LabelMatch::Ambiguous(labels) => {
                format!("{typed:?} could be {}", or_list(&labels))
            }
            LabelMatch::Missing if counts.is_empty() => format!("No {kind} is in view"),
            LabelMatch::Missing => {
                // The most common first, as the overview lists them.
                let shown: Vec<&str> = counts.iter().take(4).map(|(label, _)| &**label).collect();
                format!(
                    "No {kind} matches {typed:?} · in view: {}{}",
                    shown.join(", "),
                    if counts.len() > shown.len() {
                        ", …"
                    } else {
                        ""
                    }
                )
            }
        };
        self.status = status.into();
        cx.notify();
        None
    }

    pub fn execute_command(&mut self, query: &str, window: &mut Window, cx: &mut Context<Self>) {
        if !reads_as_command(query) {
            if !query.trim().is_empty() {
                self.start_search(query.trim().to_string(), self.search_mode, window, cx);
            }
            return;
        }
        let mut parts = query.split_whitespace();
        match parts.next() {
            Some("fit") => self.fit(cx),
            Some("zoom") => {
                let Some(zoom) = parts.next().and_then(parse_zoom) else {
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
            Some("reload") | Some("refresh") => self.reload(window, cx),
            Some("help") | Some("?") | Some("shortcuts") => {
                self.query_input
                    .update(cx, |input, cx| input.set_value("", window, cx));
                self.set_help_visible(true, window, cx);
            }
            Some("back") => self.navigate_selection_history(true, window, cx),
            Some("forward") => self.navigate_selection_history(false, window, cx),
            Some("open") => {
                let argument = query.trim_start()["open".len()..].trim();
                if argument.is_empty() {
                    self.prompt_open_database(window, cx);
                    return;
                }
                match resolve_open_argument(argument) {
                    Some(path) => self.open_database(path, window, cx),
                    None => {
                        self.status = "Expected: open <path-to-database.vg>".into();
                        cx.notify();
                    }
                }
            }
            Some("export") => {
                let argument = query.trim_start()["export".len()..].trim();
                if argument.is_empty() {
                    self.prompt_export_view(window, cx);
                    return;
                }
                match resolve_open_argument(argument) {
                    Some(mut path) => {
                        if path.extension().is_none() {
                            path.set_extension("svg");
                        }
                        self.export_view(&path, cx);
                    }
                    None => {
                        self.status = "Expected: export <path.svg>".into();
                        cx.notify();
                    }
                }
            }
            Some("live") | Some("watch") => match parts.next() {
                None => self.set_live_refresh(!self.live_refresh, window, cx),
                Some("on") => self.set_live_refresh(true, window, cx),
                Some("off") => self.set_live_refresh(false, window, cx),
                Some(_) => {
                    self.status = "Expected: live | live on | live off".into();
                    cx.notify();
                }
            },
            Some("release") => self.release_selected(cx),
            Some("hide") => match (parts.next(), parts.next()) {
                (None, _) => self.hide_selected(cx),
                (Some("label"), Some(label)) => {
                    if let Some(label) = self.command_label(label, false, cx) {
                        self.hide_label(&label, cx);
                    }
                }
                (Some("relationship"), Some(label)) => {
                    if let Some(label) = self.command_label(label, true, cx) {
                        self.hide_relationship(&label, cx);
                    }
                }
                (Some(id), _) => {
                    let index = id.parse::<u64>().ok().and_then(|id| {
                        self.snapshot().and_then(|snapshot| snapshot.node_index(id))
                    });
                    match index {
                        Some(index) => self.hide_node(index, cx),
                        None => {
                            self.status = format!("Node {id} is not in this view").into();
                            cx.notify();
                        }
                    }
                }
            },
            Some("only") => {
                let relationship = parts.next() == Some("relationship");
                if let Some(label) = parts
                    .next()
                    .and_then(|label| self.command_label(label, relationship, cx))
                {
                    self.show_only(&label, relationship, cx);
                }
            }
            Some("unhide") => self.show_hidden_nodes(cx),
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
                let index = snapshot.node_index(id);
                self.status = match index {
                    Some(index) => format!("Selected {}", status_node_name(snapshot, index)).into(),
                    None => format!("Node {id} is not in this view").into(),
                };
                self.set_selection(index.map(SceneSelection::Node));
                self.spring_to_selection(true);
                self.begin_presentation_motion(window, cx);
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
                self.status = format!("Centered {}", self.status_node_name_by_id(id)).into();
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
                let index = snapshot.edge_index(id);
                self.status = match index {
                    Some(index) => format!("Selected {}", status_edge_name(snapshot, index)).into(),
                    None => format!("Relationship {id} is not in this view").into(),
                };
                self.set_selection(index.map(SceneSelection::Edge));
                self.spring_to_selection(true);
                self.begin_presentation_motion(window, cx);
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
                        Ok(value) if value <= MAX_PATH_HOPS => value,
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
                let relationship = match kind {
                    Some("node") | Some("label") => false,
                    Some("edge") | Some("relationship") => true,
                    _ => {
                        self.status =
                            "Expected: facet node <label> | facet relationship <label>".into();
                        cx.notify();
                        return;
                    }
                };
                if let Some(label) = self.command_label(&label, relationship, cx) {
                    let facet = if relationship {
                        FacetLens::Relationship(label)
                    } else {
                        FacetLens::NodeLabel(label)
                    };
                    self.activate_facet(facet, window, cx);
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
        if self
            .hover_camera
            .is_some_and(|camera| camera != self.camera)
        {
            self.hovered_node = None;
            self.hovered_edge = None;
            self.hover_camera = None;
        }
        if let Some(fresh) = &self.fresh_nodes {
            if fresh.since.elapsed() >= FRESH_NODE_DURATION {
                self.fresh_nodes = None;
            } else {
                window.request_animation_frame();
            }
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
                this.set_help_visible(false, window, cx);
                this.focus_search(window, cx);
            }))
            .on_action(cx.listener(|this, _: &NextSearchResult, _, cx| {
                this.move_search_selection(1, cx);
            }))
            .on_action(cx.listener(|this, _: &PreviousSearchResult, _, cx| {
                this.move_search_selection(-1, cx);
            }))
            .on_action(cx.listener(|this, _: &CopySelection, _, cx| {
                this.copy_selection(cx);
            }))
            .on_action(cx.listener(|this, _: &HideSelection, _, cx| {
                this.hide_selected(cx);
            }))
            .on_action(cx.listener(|this, _: &UndoHide, _, cx| {
                this.undo_hide(cx);
            }))
            .on_action(cx.listener(|this, _: &ShowHidden, _, cx| {
                this.show_hidden_nodes(cx);
            }))
            .on_action(cx.listener(|this, _: &SearchBoxUp, window, cx| {
                this.step_search_box(true, window, cx);
            }))
            .on_action(cx.listener(|this, _: &SearchBoxDown, window, cx| {
                this.step_search_box(false, window, cx);
            }))
            .on_action(cx.listener(|this, _: &ReloadDatabase, window, cx| {
                this.reload(window, cx);
            }))
            .on_action(cx.listener(|this, _: &ToggleLiveRefresh, window, cx| {
                this.set_live_refresh(!this.live_refresh, window, cx);
            }))
            .on_action(cx.listener(|this, _: &ExportView, window, cx| {
                this.prompt_export_view(window, cx);
            }))
            .on_action(cx.listener(|this, _: &OpenDatabase, window, cx| {
                this.prompt_open_database(window, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectionBack, window, cx| {
                this.navigate_selection_history(true, window, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectionForward, window, cx| {
                this.navigate_selection_history(false, window, cx);
            }))
            .on_action(cx.listener(|this, _: &ToggleHelp, window, cx| {
                this.set_help_visible(!this.help_visible, window, cx);
            }))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, window, cx| {
                this.on_external_paths_dropped(paths, window, cx);
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
            .children(self.render_node_menu(cx))
            .children(self.render_facet_menu(cx))
            .when(self.help_visible, |this| {
                this.child(self.render_help(window, cx))
            })
    }
}

#[cfg(test)]
#[path = "view_tests.rs"]
mod tests;
