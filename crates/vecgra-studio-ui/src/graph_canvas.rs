use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use gpui::{
    App, Bounds, FontWeight, Hsla, IntoElement, PaintQuad, Path, PathBuilder, Pixels, Point,
    SharedString, Styled, TextAlign, TextRun, Window, canvas, fill, point, px, quad, size,
};
use lyon_geom::{
    Line, Point as LyonPoint, Vector as LyonVector, point as lyon_point, vector as lyon_vector,
};
use vecgra_studio_core::{
    Camera, DetailLevel, GraphWorkspace, Rect, SceneSelection, SceneSnapshot, Vec2, detail_level,
    split_path_title,
};

use crate::theme::{
    RELATIONSHIP_COLOR_COUNT, node_color_for_index, palette, relationship_color_for_index,
    relationship_color_index,
};

mod svg;

pub(crate) use svg::scene_svg;

const EDGE_PATH_CHUNK: usize = 6_000;
const COMMUNITY_EDGE_BUDGET: usize = 18_000;
const REPRESENTATIVE_CAPTION_BUDGET: usize = 10;
const LOCAL_CAPTION_BUDGET: usize = 24;
const LOCAL_ARROW_BUDGET: usize = 48;
const NODE_CAPTION_BUDGET: usize = 14;
const NODE_CAPTION_CHARS: usize = 30;

struct PaintNode {
    point: Point<Pixels>,
    radius: f32,
    color: Hsla,
    selected: bool,
    pinned: bool,
    dragging: bool,
    relevance: f32,
    path_start: bool,
    path_end: bool,
}

struct PaintPath {
    path: Path<Pixels>,
    color: Hsla,
}

/// A node's name drawn beside it.
struct PaintNodeCaption {
    text: SharedString,
    origin: Point<Pixels>,
    strong: bool,
}

/// A specialised tessellator for the canvas's disconnected straight strokes.
/// GPUI's general PathBuilder delegates to Lyon; for this common case Lyon
/// emits the same two triangles per segment, with butt caps and no joins.
struct LinePathBuilder {
    path: Option<Path<Pixels>>,
    half_width: f32,
    square_merge_threshold: f32,
}

impl LinePathBuilder {
    fn stroke(width: Pixels) -> Self {
        let width = f32::from(width);
        Self {
            path: None,
            half_width: width * 0.5,
            square_merge_threshold: (0.1_f32 * 0.1 * 0.5).min(width * width * 0.05).max(1e-8),
        }
    }

    fn segment(&mut self, from: Point<Pixels>, to: Point<Pixels>) {
        let lyon_from = lyon_point(f32::from(from.x), f32::from(from.y));
        let lyon_to = lyon_point(f32::from(to.x), f32::from(to.y));
        let edge = lyon_to - lyon_from;
        let square_length = edge.square_length();
        if square_length < self.square_merge_threshold || !square_length.is_finite() {
            return;
        }
        let length = square_length.sqrt();
        // Match Lyon's fixed-width fast path exactly. It derives both side
        // attachments from a normalized tangent; butt caps then intersect the
        // side lines in f64 before converting back to f32.
        let tangent = edge / length;
        let normal = lyon_vector(-tangent.y, tangent.x) * self.half_width;
        let from_positive = lyon_from + normal;
        let from_negative = lyon_from - normal;
        let to_positive = lyon_to + normal;
        let to_negative = lyon_to - normal;
        let start_normal = (lyon_from - lyon_to).normalize();
        let end_normal = (lyon_to - lyon_from).normalize();
        let start_cap_vector = lyon_vector(-start_normal.y, start_normal.x);
        let end_cap_vector = lyon_vector(-end_normal.y, end_normal.x);
        let capped_from_positive = line_intersection_f64(
            lyon_from + start_normal * 0.0,
            start_cap_vector,
            from_positive,
            from_positive - to_positive,
        )
        .unwrap_or(from_positive);
        let capped_from_negative = line_intersection_f64(
            lyon_from + start_normal * 0.0,
            start_cap_vector,
            from_negative,
            from_negative - to_negative,
        )
        .unwrap_or(from_negative);
        let capped_to_positive = line_intersection_f64(
            lyon_to + end_normal * 0.0,
            end_cap_vector,
            to_positive,
            to_positive - from_positive,
        )
        .unwrap_or(to_positive);
        let capped_to_negative = line_intersection_f64(
            lyon_to + end_normal * 0.0,
            end_cap_vector,
            to_negative,
            to_negative - from_negative,
        )
        .unwrap_or(to_negative);
        let capped_from_positive = gpui_point(stroke_vertex_position(
            lyon_from,
            capped_from_positive,
            self.half_width,
        ));
        let capped_from_negative = gpui_point(stroke_vertex_position(
            lyon_from,
            capped_from_negative,
            self.half_width,
        ));
        let capped_to_positive = gpui_point(stroke_vertex_position(
            lyon_to,
            capped_to_positive,
            self.half_width,
        ));
        let capped_to_negative = gpui_point(stroke_vertex_position(
            lyon_to,
            capped_to_negative,
            self.half_width,
        ));
        let path = self
            .path
            .get_or_insert_with(|| Path::new(capped_from_negative));
        path.push_triangle(
            (
                capped_from_negative,
                capped_from_positive,
                capped_to_positive,
            ),
            (point(0.0, 1.0), point(0.0, 1.0), point(0.0, 1.0)),
        );
        path.push_triangle(
            (capped_from_negative, capped_to_positive, capped_to_negative),
            (point(0.0, 1.0), point(0.0, 1.0), point(0.0, 1.0)),
        );
    }

    fn build(self) -> Option<Path<Pixels>> {
        self.path
    }
}

fn gpui_point(value: lyon_geom::Point<f32>) -> Point<Pixels> {
    point(px(value.x), px(value.y))
}

fn stroke_vertex_position(
    position_on_path: LyonPoint<f32>,
    side_position: LyonPoint<f32>,
    half_width: f32,
) -> LyonPoint<f32> {
    let normal = (side_position - position_on_path) / half_width;
    position_on_path + normal * half_width
}

fn line_intersection_f64(
    line_point: LyonPoint<f32>,
    line_vector: LyonVector<f32>,
    other_point: LyonPoint<f32>,
    other_vector: LyonVector<f32>,
) -> Option<LyonPoint<f32>> {
    let line = Line {
        point: line_point,
        vector: line_vector,
    };
    let other = Line {
        point: other_point,
        vector: other_vector,
    };
    line.to_f64()
        .intersection(&other.to_f64())
        .map(|intersection| intersection.to_f32())
}

struct PaintArrow {
    path: Path<Pixels>,
    /// Tip, then the two back corners, for drawing it outside GPUI.
    corners: [Point<Pixels>; 3],
    color: Hsla,
    selected: bool,
}

struct PaintCaption {
    text: SharedString,
    center: Point<Pixels>,
    color: Hsla,
    selected: bool,
}

struct EdgeCandidate {
    from: Vec2,
    to: Vec2,
    length_squared: f32,
}

struct PaintData {
    edge_paths: Vec<PaintPath>,
    emphasis_edge_paths: Vec<PaintPath>,
    incident_edge_paths: Vec<PaintPath>,
    selected_edge_path: Option<PaintPath>,
    arrows: Vec<PaintArrow>,
    captions: Vec<PaintCaption>,
    nodes: Vec<PaintNode>,
    node_captions: Vec<PaintNodeCaption>,
}

struct ScenePresentation<'a> {
    camera: Camera,
    world_bounds: Rect,
    selection: Option<SceneSelection>,
    dragging: Option<usize>,
    emphasis: Option<&'a GraphEmphasis>,
    path_endpoints: Option<GraphPathEndpoints>,
    hovered_node: Option<usize>,
    hovered_edge: Option<usize>,
    label_colors: &'a LabelColors,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GraphPathEndpoints {
    pub start: usize,
    pub end: Option<usize>,
}

/// The colour each node label and relationship type is drawn in. Each
/// takes its palette's colours in the order it first appears, by ID, and
/// keeps them, so one added later, or hidden and shown again, never
/// recolours the rest.
#[derive(Clone, Debug, Default)]
pub(crate) struct LabelColors {
    nodes: HashMap<Arc<str>, usize>,
    relationships: HashMap<Arc<str>, usize>,
}

impl LabelColors {
    /// Gives each label and type in `snapshot` that has no colour yet the
    /// next one.
    pub(crate) fn extend(&mut self, snapshot: &SceneSnapshot) {
        for (seen, labels) in [
            (&mut self.nodes, &snapshot.nodes.labels),
            (&mut self.relationships, &snapshot.edges.labels),
        ] {
            for label in labels {
                if !seen.contains_key(label) {
                    let next = seen.len();
                    seen.insert(label.clone(), next);
                }
            }
        }
    }

    /// A node label's colour.
    pub(crate) fn color(&self, label: &str) -> Hsla {
        match self.nodes.get(label) {
            Some(&index) => node_color_for_index(index),
            None => label_color(label),
        }
    }

    /// Which of the relationship colours a type is drawn in.
    pub(crate) fn relationship_index(&self, label: &str) -> usize {
        self.relationships.get(label).map_or_else(
            || relationship_color_index(label),
            |&index| index % RELATIONSHIP_COLOR_COUNT,
        )
    }

    pub(crate) fn relationship(&self, label: &str) -> Hsla {
        relationship_color_for_index(self.relationship_index(label))
    }
}

pub(crate) struct GraphCanvasPresentation {
    pub camera: Camera,
    pub world_bounds: Rect,
    pub selection: Option<SceneSelection>,
    pub dragging: Option<usize>,
    pub emphasis: Option<GraphEmphasis>,
    pub path_endpoints: Option<GraphPathEndpoints>,
    /// The node or relationship under the pointer, or named by a hovered
    /// list row, whose lines are lit more faintly than the selection's.
    pub hovered_node: Option<usize>,
    pub hovered_edge: Option<usize>,
    pub label_colors: Arc<LabelColors>,
}

struct NodePaintContext<'a> {
    label_colors: &'a LabelColors,
    selected: Option<usize>,
    dragging: Option<usize>,
    emphasis: Option<&'a GraphEmphasis>,
    path_endpoints: Option<GraphPathEndpoints>,
    bounds: Bounds<Pixels>,
    viewport: Vec2,
}

#[derive(Clone)]
pub struct GraphEmphasis {
    pub from_nodes: Arc<[f32]>,
    pub target_nodes: Arc<[f32]>,
    pub from_edges: Arc<[f32]>,
    pub target_edges: Arc<[f32]>,
    pub mix: f32,
    pub dim: f32,
}

impl GraphEmphasis {
    pub(crate) fn node_score(&self, index: usize) -> f32 {
        lerp_score(
            self.from_nodes.get(index).copied().unwrap_or(0.0),
            self.target_nodes.get(index).copied().unwrap_or(0.0),
            self.mix,
        )
    }

    fn edge_score(&self, index: usize) -> f32 {
        lerp_score(
            self.from_edges.get(index).copied().unwrap_or(0.0),
            self.target_edges.get(index).copied().unwrap_or(0.0),
            self.mix,
        )
    }
}

pub fn graph_canvas(
    snapshot: Arc<SceneSnapshot>,
    workspace: Rc<RefCell<GraphWorkspace>>,
    presentation: GraphCanvasPresentation,
) -> impl IntoElement {
    canvas(
        move |bounds, _window, _cx| {
            prepare_scene(
                &snapshot,
                &workspace.borrow(),
                ScenePresentation {
                    camera: presentation.camera,
                    world_bounds: presentation.world_bounds,
                    selection: presentation.selection,
                    dragging: presentation.dragging,
                    emphasis: presentation.emphasis.as_ref(),
                    path_endpoints: presentation.path_endpoints,
                    hovered_node: presentation.hovered_node,
                    hovered_edge: presentation.hovered_edge,
                    label_colors: &presentation.label_colors,
                },
                bounds,
            )
        },
        paint_scene,
    )
    .absolute()
    .size_full()
}

fn prepare_scene(
    snapshot: &SceneSnapshot,
    workspace: &GraphWorkspace,
    presentation: ScenePresentation<'_>,
    bounds: Bounds<Pixels>,
) -> PaintData {
    let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
    let camera = presentation.camera;
    let world_bounds = presentation.world_bounds;
    let emphasis = presentation.emphasis;
    let level = detail_level(camera, snapshot.nodes.ids.len());
    let selected_node = presentation.selection.and_then(SceneSelection::node);
    let selected_edge = presentation.selection.and_then(SceneSelection::edge);
    let projected: Vec<Vec2> = workspace
        .positions()
        .iter()
        .copied()
        .map(|position| camera.project(position, world_bounds, viewport))
        .collect();

    let mut edge_paths = Vec::new();
    let mut representative_edges = BTreeMap::<Arc<str>, EdgeCandidate>::new();
    if level != DetailLevel::Overview && emphasis.is_none_or(|lens| lens.dim < 0.35) {
        let edge_stride = if level == DetailLevel::Communities {
            snapshot
                .edges
                .ids
                .len()
                .div_ceil(COMMUNITY_EDGE_BUDGET)
                .max(1)
        } else {
            1
        };
        for start in (0..snapshot.edges.ids.len()).step_by(EDGE_PATH_CHUNK) {
            let end = (start + EDGE_PATH_CHUNK).min(snapshot.edges.ids.len());
            let width = if level == DetailLevel::Elements {
                0.95
            } else {
                0.74
            };
            let mut builders: [LinePathBuilder; RELATIONSHIP_COLOR_COUNT] =
                std::array::from_fn(|_| LinePathBuilder::stroke(px(width)));
            let mut segment_counts = [0_usize; RELATIONSHIP_COLOR_COUNT];
            for edge_index in (start..end).step_by(edge_stride) {
                let source = snapshot.edges.sources[edge_index] as usize;
                let target = snapshot.edges.targets[edge_index] as usize;
                let from = projected[source];
                let to = projected[target];
                if !segment_may_be_visible(from, to, viewport) {
                    continue;
                }
                if level == DetailLevel::Elements
                    && camera.zoom >= 12.0
                    && !point_is_visible(from, viewport, 48.0)
                    && !point_is_visible(to, viewport, 48.0)
                {
                    // At deep zoom, long edges whose endpoints are both
                    // offscreen are context-free line noise. Endpoint culling
                    // reveals local elements while selected/emphasized paths
                    // remain in their dedicated passes below.
                    continue;
                }
                let color_index = presentation
                    .label_colors
                    .relationship_index(&snapshot.edges.labels[edge_index]);
                builders[color_index]
                    .segment(absolute_point(bounds, from), absolute_point(bounds, to));
                segment_counts[color_index] += 1;

                let length_squared = (to - from).length_squared();
                if length_squared >= 64.0 * 64.0 {
                    let label = snapshot.edges.labels[edge_index].clone();
                    let replace = representative_edges
                        .get(&label)
                        .is_none_or(|candidate| length_squared > candidate.length_squared);
                    if replace {
                        representative_edges.insert(
                            label,
                            EdgeCandidate {
                                from,
                                to,
                                length_squared,
                            },
                        );
                    }
                }
            }
            for (color_index, (builder, segments)) in
                builders.into_iter().zip(segment_counts).enumerate()
            {
                if segments > 0
                    && let Some(path) = builder.build()
                {
                    let lens_dim = emphasis.map_or(0.0, |lens| lens.dim);
                    edge_paths.push(PaintPath {
                        path,
                        color: relationship_color_for_index(color_index)
                            .opacity(0.27 * (1.0 - lens_dim).powf(1.6)),
                    });
                }
            }
        }
    }

    let mut incident_edge_paths = Vec::new();
    let mut emphasis_edge_paths = Vec::new();
    let mut arrows = Vec::new();
    let mut captions = Vec::new();
    if let Some(emphasis) = emphasis {
        let mut captioned_relationships = BTreeSet::new();
        for edge_index in 0..snapshot.edges.ids.len() {
            let score = emphasis.edge_score(edge_index);
            if score <= 0.01 {
                continue;
            }
            let source = snapshot.edges.sources[edge_index] as usize;
            let target = snapshot.edges.targets[edge_index] as usize;
            let from = projected[source];
            let to = projected[target];
            if !segment_may_be_visible(from, to, viewport) {
                continue;
            }
            let label = &snapshot.edges.labels[edge_index];
            let color = presentation
                .label_colors
                .relationship(label)
                .opacity(0.42 + score * 0.58);
            let mut builder = LinePathBuilder::stroke(px(1.2 + score * 1.5));
            builder.segment(absolute_point(bounds, from), absolute_point(bounds, to));
            if let Some(path) = builder.build() {
                emphasis_edge_paths.push(PaintPath { path, color });
            }
            if arrows.len() < LOCAL_ARROW_BUDGET
                && let Some((path, corners)) = arrowhead(bounds, from, to)
            {
                arrows.push(PaintArrow {
                    path,
                    corners,
                    color,
                    selected: score > 0.9,
                });
            }
            if captions.len() < LOCAL_CAPTION_BUDGET
                && score >= 0.54
                && (to - from).length_squared() >= 34.0 * 34.0
                && captioned_relationships.insert(label.clone())
            {
                push_caption(
                    &mut captions,
                    label.clone(),
                    caption_center(bounds, from, to),
                    color,
                    score > 0.9,
                );
            }
        }
    }
    // A context lens already paints its bounded edge set. Repainting every
    // incident edge of the selected hub would punch through the lens and send
    // bright lines to nodes omitted by that bounded view.
    if let Some(node_index) = selected_node
        && emphasis.is_none_or(|lens| lens.dim < 0.35)
    {
        let mut builders: [LinePathBuilder; RELATIONSHIP_COLOR_COUNT] =
            std::array::from_fn(|_| LinePathBuilder::stroke(px(1.65)));
        let mut segment_counts = [0_usize; RELATIONSHIP_COLOR_COUNT];
        for edge_index in 0..snapshot.edges.ids.len() {
            let source = snapshot.edges.sources[edge_index] as usize;
            let target = snapshot.edges.targets[edge_index] as usize;
            if source != node_index && target != node_index {
                continue;
            }
            let from = projected[source];
            let to = projected[target];
            if !segment_may_be_visible(from, to, viewport) {
                continue;
            }
            let label = &snapshot.edges.labels[edge_index];
            let color_index = presentation.label_colors.relationship_index(label);
            builders[color_index].segment(absolute_point(bounds, from), absolute_point(bounds, to));
            segment_counts[color_index] += 1;
            if arrows.len() < LOCAL_ARROW_BUDGET
                && let Some((path, corners)) = arrowhead(bounds, from, to)
            {
                arrows.push(PaintArrow {
                    path,
                    corners,
                    color: relationship_color_for_index(color_index).opacity(0.92),
                    selected: false,
                });
            }
            if captions.len() < LOCAL_CAPTION_BUDGET && (to - from).length_squared() >= 34.0 * 34.0
            {
                push_caption(
                    &mut captions,
                    label.clone(),
                    caption_center(bounds, from, to),
                    relationship_color_for_index(color_index),
                    false,
                );
            }
        }
        for (color_index, (builder, segments)) in
            builders.into_iter().zip(segment_counts).enumerate()
        {
            if segments > 0
                && let Some(path) = builder.build()
            {
                incident_edge_paths.push(PaintPath {
                    path,
                    color: relationship_color_for_index(color_index).opacity(0.78),
                });
            }
        }
    }

    // The node or relationship under the pointer lights its lines, more
    // faintly than the selection, so a neighborhood can be read before
    // clicking. Lenses that dim the scene keep their own bounded edge set.
    if emphasis.is_none_or(|lens| lens.dim < 0.35) {
        if let Some(node_index) = presentation
            .hovered_node
            .filter(|&index| selected_node != Some(index))
        {
            incident_edge_paths.extend(hover_edge_paths(
                snapshot,
                presentation.label_colors,
                &projected,
                |source, target| source == node_index || target == node_index,
                1.4,
                0.7,
                bounds,
            ));
        }
        if let Some(edge_index) = presentation
            .hovered_edge
            .filter(|&index| selected_edge != Some(index))
        {
            let source = snapshot.edges.sources[edge_index] as usize;
            let target = snapshot.edges.targets[edge_index] as usize;
            let mut builder = LinePathBuilder::stroke(px(2.0));
            builder.segment(
                absolute_point(bounds, projected[source]),
                absolute_point(bounds, projected[target]),
            );
            if let Some(path) = builder.build() {
                incident_edge_paths.push(PaintPath {
                    path,
                    color: presentation
                        .label_colors
                        .relationship(&snapshot.edges.labels[edge_index]),
                });
            }
        }
    }

    let selected_edge_path = selected_edge.and_then(|edge_index| {
        let source = snapshot.edges.sources[edge_index] as usize;
        let target = snapshot.edges.targets[edge_index] as usize;
        let from = projected[source];
        let to = projected[target];
        let label = &snapshot.edges.labels[edge_index];
        let color = presentation.label_colors.relationship(label);
        let mut builder = LinePathBuilder::stroke(px(2.5));
        builder.segment(absolute_point(bounds, from), absolute_point(bounds, to));
        if let Some((path, corners)) = arrowhead(bounds, from, to) {
            arrows.push(PaintArrow {
                path,
                corners,
                color,
                selected: true,
            });
        }
        push_caption(
            &mut captions,
            label.clone(),
            caption_center(bounds, from, to),
            color,
            true,
        );
        builder.build().map(|path| PaintPath { path, color })
    });

    if level != DetailLevel::Overview {
        for (label, candidate) in representative_edges
            .into_iter()
            .take(REPRESENTATIVE_CAPTION_BUDGET)
        {
            let color = presentation.label_colors.relationship(&label);
            if push_caption(
                &mut captions,
                label,
                caption_center(bounds, candidate.from, candidate.to),
                color,
                false,
            ) && let Some((path, corners)) = arrowhead(bounds, candidate.from, candidate.to)
            {
                arrows.push(PaintArrow {
                    path,
                    corners,
                    color: color.opacity(0.86),
                    selected: false,
                });
            }
        }
    }

    let nodes = match level {
        DetailLevel::Overview => aggregate_nodes(
            snapshot,
            workspace,
            &projected,
            &NodePaintContext {
                label_colors: presentation.label_colors,
                selected: selected_node,
                dragging: presentation.dragging,
                emphasis,
                path_endpoints: presentation.path_endpoints,
                bounds,
                viewport,
            },
        ),
        DetailLevel::Communities | DetailLevel::Elements => snapshot
            .nodes
            .labels
            .iter()
            .enumerate()
            .filter_map(|(index, label)| {
                let position = projected[index];
                point_is_visible(position, viewport, 24.0).then(|| {
                    let degree = snapshot.nodes.degrees[index] as f32;
                    let relevance = emphasis.map_or(0.0, |lens| lens.node_score(index));
                    let lens_opacity = emphasis.map_or(1.0, |lens| {
                        if relevance > 0.01 {
                            (0.52 + relevance * 0.48).min(1.0)
                        } else {
                            (1.0 - lens.dim).powf(1.35).max(0.012)
                        }
                    });
                    PaintNode {
                        point: absolute_point(bounds, position),
                        radius: if selected_node == Some(index) {
                            7.5
                        } else {
                            (3.1 + degree.ln_1p() * 0.52).min(6.2) * (1.0 + relevance * 0.48)
                        },
                        color: presentation.label_colors.color(label).opacity(lens_opacity),
                        selected: selected_node == Some(index),
                        pinned: workspace.is_pinned(index),
                        dragging: presentation.dragging == Some(index),
                        relevance,
                        path_start: presentation
                            .path_endpoints
                            .is_some_and(|endpoints| endpoints.start == index),
                        path_end: presentation
                            .path_endpoints
                            .and_then(|endpoints| endpoints.end)
                            == Some(index),
                    }
                })
            })
            .collect(),
    };

    let node_captions = if level == DetailLevel::Overview {
        Vec::new()
    } else {
        node_captions(
            snapshot,
            &projected,
            &captions,
            selected_edge,
            &NodePaintContext {
                label_colors: presentation.label_colors,
                selected: selected_node,
                dragging: presentation.dragging,
                emphasis,
                path_endpoints: presentation.path_endpoints,
                bounds,
                viewport,
            },
            level,
        )
    };

    PaintData {
        edge_paths,
        emphasis_edge_paths,
        incident_edge_paths,
        selected_edge_path,
        arrows,
        captions,
        nodes,
        node_captions,
    }
}

/// Chooses which nodes to name on the canvas: the selection, the selected
/// relationship's endpoints, and path endpoints always, then the selection's
/// neighbors by degree, the strongest
/// search matches, and, when zoomed in to elements with nothing else to
/// name, the best-connected visible nodes. A name that would overlap one
/// already placed, or a relationship caption, is skipped, and at most
/// [`NODE_CAPTION_BUDGET`] are drawn.
/// Lines for the relationships `include` accepts, batched by color, drawn
/// `width` wide at `opacity`.
fn hover_edge_paths(
    snapshot: &SceneSnapshot,
    colors: &LabelColors,
    projected: &[Vec2],
    include: impl Fn(usize, usize) -> bool,
    width: f32,
    opacity: f32,
    bounds: Bounds<Pixels>,
) -> Vec<PaintPath> {
    let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
    let mut builders: [LinePathBuilder; RELATIONSHIP_COLOR_COUNT] =
        std::array::from_fn(|_| LinePathBuilder::stroke(px(width)));
    let mut segment_counts = [0_usize; RELATIONSHIP_COLOR_COUNT];
    for edge_index in 0..snapshot.edges.ids.len() {
        let source = snapshot.edges.sources[edge_index] as usize;
        let target = snapshot.edges.targets[edge_index] as usize;
        if !include(source, target) {
            continue;
        }
        let (from, to) = (projected[source], projected[target]);
        if !segment_may_be_visible(from, to, viewport) {
            continue;
        }
        let color_index = colors.relationship_index(&snapshot.edges.labels[edge_index]);
        builders[color_index].segment(absolute_point(bounds, from), absolute_point(bounds, to));
        segment_counts[color_index] += 1;
    }
    builders
        .into_iter()
        .zip(segment_counts)
        .enumerate()
        .filter(|&(_, (_, segments))| segments > 0)
        .filter_map(|(color_index, (builder, _))| {
            builder.build().map(|path| PaintPath {
                path,
                color: relationship_color_for_index(color_index).opacity(opacity),
            })
        })
        .collect()
}

/// The lens score a traced path gives the nodes between its ends.
pub(crate) const PATH_INTERIOR_SCORE: f32 = 0.76;

fn node_captions(
    snapshot: &SceneSnapshot,
    projected: &[Vec2],
    relationship_captions: &[PaintCaption],
    selected_edge: Option<usize>,
    context: &NodePaintContext<'_>,
    level: DetailLevel,
) -> Vec<PaintNodeCaption> {
    let degrees = &snapshot.nodes.degrees;
    let mut candidates: Vec<(usize, bool)> = Vec::new();
    let mut push = |index: usize, strong: bool| {
        if !candidates.iter().any(|&(existing, _)| existing == index) {
            candidates.push((index, strong));
        }
    };
    if let Some(selected) = context.selected {
        push(selected, true);
    }
    if let Some(edge) = selected_edge {
        push(snapshot.edges.sources[edge] as usize, true);
        push(snapshot.edges.targets[edge] as usize, true);
    }
    if let Some(endpoints) = context.path_endpoints {
        push(endpoints.start, true);
        if let Some(end) = endpoints.end {
            push(end, true);
        }
        // The nodes a traced path runs through come before the start's
        // neighbors, which would otherwise fill the budget first.
        if let Some(lens) = context.emphasis {
            for index in 0..projected.len() {
                if lens.node_score(index) >= PATH_INTERIOR_SCORE {
                    push(index, false);
                }
            }
        }
    }
    if let Some(selected) = context.selected {
        let mut neighbors: Vec<usize> = snapshot
            .edges
            .sources
            .iter()
            .zip(&snapshot.edges.targets)
            .filter_map(|(&source, &target)| {
                let (source, target) = (source as usize, target as usize);
                match (source == selected, target == selected) {
                    (true, false) => Some(target),
                    (false, true) => Some(source),
                    _ => None,
                }
            })
            .collect();
        neighbors
            .sort_unstable_by_key(|&neighbor| (std::cmp::Reverse(degrees[neighbor]), neighbor));
        neighbors.dedup();
        for neighbor in neighbors {
            push(neighbor, false);
        }
    }
    if let Some(lens) = context.emphasis {
        let mut matches: Vec<(usize, f32)> = (0..projected.len())
            .map(|index| (index, lens.node_score(index)))
            .filter(|&(_, score)| score > 0.48)
            .collect();
        matches.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        for (index, _) in matches {
            push(index, false);
        }
    }
    if level == DetailLevel::Elements && context.selected.is_none() && context.emphasis.is_none() {
        let mut visible: Vec<usize> = (0..projected.len())
            .filter(|&index| point_is_visible(projected[index], context.viewport, 0.0))
            .collect();
        visible.sort_unstable_by_key(|&index| (std::cmp::Reverse(degrees[index]), index));
        for index in visible.into_iter().take(NODE_CAPTION_BUDGET * 4) {
            push(index, false);
        }
    }

    // Widths here are estimates of the shaped text; they only have to keep
    // names apart.
    let mut placed: Vec<Bounds<Pixels>> = relationship_captions
        .iter()
        .map(|caption| {
            let width = px(caption.text.chars().count() as f32 * 6.0 + 12.0);
            Bounds {
                origin: point(caption.center.x - width / 2.0, caption.center.y - px(9.0)),
                size: size(width, px(18.0)),
            }
        })
        .collect();
    let mut captions = Vec::new();
    for (index, strong) in candidates {
        if captions.len() >= NODE_CAPTION_BUDGET {
            break;
        }
        let position = projected[index];
        if !point_is_visible(position, context.viewport, 0.0) {
            continue;
        }
        // A lens that dims the scene fades what it leaves out nearly to
        // nothing; a name there would label an empty spot.
        if !strong
            && context
                .emphasis
                .is_some_and(|lens| lens.dim >= 0.35 && lens.node_score(index) <= 0.01)
        {
            continue;
        }
        let text = node_caption_text(&snapshot.node_title(index));
        let width = text.chars().count() as f32 * 6.2 + 10.0;
        // Names sit right of their node unless that would run off the canvas.
        let offset_x = if position.x + 11.0 + width > context.viewport.x - 4.0 {
            -11.0 - width
        } else {
            11.0
        };
        let origin = absolute_point(context.bounds, position + Vec2::new(offset_x, -8.0));
        let area = Bounds {
            origin,
            size: size(px(width), px(16.0)),
        };
        if !strong && placed.iter().any(|other| other.intersects(&area)) {
            continue;
        }
        placed.push(area);
        captions.push(PaintNodeCaption {
            text,
            origin,
            strong,
        });
    }
    captions
}

/// A canvas-length name: a path's file name, or the title cut to fit.
pub(crate) fn node_caption_text(title: &str) -> SharedString {
    let name = split_path_title(title).map_or(title, |(name, _)| name);
    if name.chars().count() <= NODE_CAPTION_CHARS {
        return SharedString::new(name);
    }
    let mut short: String = name.chars().take(NODE_CAPTION_CHARS - 1).collect();
    short.push('…');
    short.into()
}

fn aggregate_nodes(
    snapshot: &SceneSnapshot,
    workspace: &GraphWorkspace,
    projected: &[Vec2],
    context: &NodePaintContext<'_>,
) -> Vec<PaintNode> {
    struct Bin {
        sum: Vec2,
        count: usize,
        label_index: usize,
        selected: bool,
        pinned: bool,
        dragging: bool,
        relevance: f32,
        path_start: bool,
        path_end: bool,
    }

    let mut bins = BTreeMap::<(i32, i32), Bin>::new();
    const BIN_SIZE: f32 = 11.0;
    for (index, position) in projected.iter().copied().enumerate() {
        if !point_is_visible(position, context.viewport, 24.0) {
            continue;
        }
        let key = (
            (position.x / BIN_SIZE).floor() as i32,
            (position.y / BIN_SIZE).floor() as i32,
        );
        let bin = bins.entry(key).or_insert(Bin {
            sum: Vec2::ZERO,
            count: 0,
            label_index: index,
            selected: false,
            pinned: false,
            dragging: false,
            relevance: 0.0,
            path_start: false,
            path_end: false,
        });
        bin.sum += position;
        bin.count += 1;
        bin.selected |= context.selected == Some(index);
        bin.pinned |= workspace.is_pinned(index);
        bin.dragging |= context.dragging == Some(index);
        bin.relevance = bin
            .relevance
            .max(context.emphasis.map_or(0.0, |lens| lens.node_score(index)));
        bin.path_start |= context
            .path_endpoints
            .is_some_and(|endpoints| endpoints.start == index);
        bin.path_end |= context.path_endpoints.and_then(|endpoints| endpoints.end) == Some(index);
    }

    bins.into_values()
        .map(|bin| {
            let position = bin.sum / bin.count as f32;
            let density = (bin.count as f32).ln_1p();
            let lens_opacity = context.emphasis.map_or(1.0, |lens| {
                if bin.relevance > 0.01 {
                    (0.52 + bin.relevance * 0.48).min(1.0)
                } else {
                    (1.0 - lens.dim).powf(1.35).max(0.012)
                }
            });
            PaintNode {
                point: absolute_point(context.bounds, position),
                radius: (1.7 + density * 1.15).min(6.8),
                color: context
                    .label_colors
                    .color(&snapshot.nodes.labels[bin.label_index])
                    .opacity((0.24 + density * 0.105).min(0.78) * lens_opacity),
                selected: bin.selected,
                pinned: bin.pinned,
                dragging: bin.dragging,
                relevance: bin.relevance,
                path_start: bin.path_start,
                path_end: bin.path_end,
            }
        })
        .collect()
}

fn paint_scene(bounds: Bounds<Pixels>, scene: PaintData, window: &mut Window, cx: &mut App) {
    let colors = palette();
    for path in scene.edge_paths {
        window.paint_path(path.path, path.color);
    }
    for path in scene.emphasis_edge_paths {
        window.paint_path(path.path, path.color);
    }
    for path in scene.incident_edge_paths {
        window.paint_path(path.path, path.color);
    }
    if let Some(path) = scene.selected_edge_path {
        window.paint_path(path.path, path.color);
    }
    for arrow in scene.arrows {
        window.paint_path(
            arrow.path,
            arrow.color.opacity(if arrow.selected { 1.0 } else { 0.88 }),
        );
    }

    for caption in scene.captions {
        paint_caption(caption, window, cx);
    }

    window.paint_layer(bounds, |window| {
        for node in scene.nodes {
            if node.path_start || node.path_end {
                let endpoint_color = match (node.path_start, node.path_end) {
                    (true, true) => colors.mist,
                    (true, false) => colors.celadon,
                    (false, true) => colors.copper,
                    (false, false) => unreachable!(),
                };
                window.paint_quad(circle(
                    node.point,
                    node.radius + 11.0,
                    endpoint_color.opacity(0.13),
                ));
                window.paint_quad(quad(
                    circle_bounds(node.point, node.radius + 3.2),
                    px(node.radius + 3.2),
                    node.color,
                    px(2.0),
                    endpoint_color,
                    Default::default(),
                ));
                if node.path_start && node.path_end {
                    window.paint_quad(quad(
                        circle_bounds(node.point, node.radius + 0.8),
                        px(node.radius + 0.8),
                        node.color,
                        px(1.2),
                        colors.celadon,
                        Default::default(),
                    ));
                }
                continue;
            }
            if node.relevance > 0.08 && !node.selected {
                window.paint_quad(circle(
                    node.point,
                    node.radius + 3.0 + node.relevance * 5.0,
                    colors.cobalt.opacity(0.05 + node.relevance * 0.13),
                ));
                if node.relevance > 0.48 {
                    window.paint_quad(quad(
                        circle_bounds(node.point, node.radius + 1.6),
                        px(node.radius + 1.6),
                        node.color,
                        px(0.8 + node.relevance * 0.8),
                        colors.cobalt.opacity(0.55 + node.relevance * 0.4),
                        Default::default(),
                    ));
                    continue;
                }
            }
            if node.selected {
                window.paint_quad(circle(
                    node.point,
                    node.radius + if node.dragging { 13.0 } else { 10.0 },
                    colors
                        .cobalt
                        .opacity(if node.dragging { 0.2 } else { 0.11 }),
                ));
                window.paint_quad(quad(
                    circle_bounds(node.point, node.radius + 3.0),
                    px(node.radius + 3.0),
                    colors.celadon,
                    px(1.5),
                    colors.mist,
                    Default::default(),
                ));
                if node.pinned {
                    window.paint_quad(circle(
                        point(
                            node.point.x + px(node.radius + 4.0),
                            node.point.y - px(node.radius + 4.0),
                        ),
                        2.4,
                        colors.copper,
                    ));
                }
            } else if node.pinned {
                window.paint_quad(quad(
                    circle_bounds(node.point, node.radius + 1.5),
                    px(node.radius + 1.5),
                    node.color,
                    px(1.2),
                    colors.copper,
                    Default::default(),
                ));
            } else {
                window.paint_quad(circle(node.point, node.radius, node.color));
            }
        }
    });

    for caption in scene.node_captions {
        paint_node_caption(caption, window, cx);
    }
}

fn paint_node_caption(caption: PaintNodeCaption, window: &mut Window, cx: &mut App) {
    let font_size = px(10.5);
    let line_height = px(14.0);
    let text_run = TextRun {
        len: caption.text.len(),
        font: window
            .text_style()
            .highlight(if caption.strong {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::NORMAL
            })
            .font(),
        color: palette()
            .mist
            .opacity(if caption.strong { 1.0 } else { 0.86 }),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    let line = window
        .text_system()
        .shape_line(caption.text, font_size, &[text_run], None);
    window.paint_quad(fill(
        Bounds {
            origin: caption.origin,
            size: size(line.width() + px(8.0), line_height + px(2.0)),
        },
        palette().graphite.opacity(0.72),
    ));
    let origin = point(caption.origin.x + px(4.0), caption.origin.y + px(1.0));
    let _ = line.paint(origin, line_height, TextAlign::Left, None, window, cx);
}

fn lerp_score(from: f32, to: f32, mix: f32) -> f32 {
    from + (to - from) * mix.clamp(0.0, 1.0)
}

fn paint_caption(caption: PaintCaption, window: &mut Window, cx: &mut App) {
    let font_size = px(10.0);
    let line_height = px(14.0);
    let text_run = TextRun {
        len: caption.text.len(),
        font: window
            .text_style()
            .highlight(if caption.selected {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::NORMAL
            })
            .font(),
        color: palette().mist,
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    let line = window
        .text_system()
        .shape_line(caption.text, font_size, &[text_run], None);
    let text_width = line.width();
    let background = Bounds {
        origin: point(
            caption.center.x - text_width / 2.0 - px(6.0),
            caption.center.y - line_height / 2.0 - px(2.0),
        ),
        size: size(text_width + px(12.0), line_height + px(4.0)),
    };
    window.paint_quad(quad(
        background,
        px(5.0),
        palette()
            .graphite
            .opacity(if caption.selected { 0.98 } else { 0.9 }),
        px(if caption.selected { 1.5 } else { 1.0 }),
        caption
            .color
            .opacity(if caption.selected { 1.0 } else { 0.75 }),
        Default::default(),
    ));
    let origin = point(
        caption.center.x - text_width / 2.0,
        caption.center.y - line_height / 2.0,
    );
    let _ = line.paint(origin, line_height, TextAlign::Left, None, window, cx);
}

fn push_caption(
    captions: &mut Vec<PaintCaption>,
    text: Arc<str>,
    center: Point<Pixels>,
    color: Hsla,
    selected: bool,
) -> bool {
    let available = selected
        || captions.iter().all(|caption| {
            let dx = f32::from(caption.center.x - center.x).abs();
            let dy = f32::from(caption.center.y - center.y).abs();
            dx > 92.0 || dy > 24.0
        });
    if available {
        captions.push(PaintCaption {
            text: SharedString::new(text),
            center,
            color,
            selected,
        });
    }
    available
}

fn caption_center(bounds: Bounds<Pixels>, from: Vec2, to: Vec2) -> Point<Pixels> {
    let delta = to - from;
    let length = delta.length().max(0.001);
    let perpendicular = Vec2::new(-delta.y / length, delta.x / length);
    absolute_point(bounds, (from + to) * 0.5 + perpendicular * 10.0)
}

fn arrowhead(
    bounds: Bounds<Pixels>,
    from: Vec2,
    to: Vec2,
) -> Option<(Path<Pixels>, [Point<Pixels>; 3])> {
    let delta = to - from;
    let length = delta.length();
    if length < 22.0 {
        return None;
    }
    let direction = delta / length;
    let perpendicular = Vec2::new(-direction.y, direction.x);
    let tip = to - direction * 7.0;
    let base = tip - direction * 7.5;
    let corners = [
        absolute_point(bounds, tip),
        absolute_point(bounds, base + perpendicular * 3.8),
        absolute_point(bounds, base - perpendicular * 3.8),
    ];
    let mut builder = PathBuilder::fill();
    builder.move_to(corners[0]);
    builder.line_to(corners[1]);
    builder.line_to(corners[2]);
    builder.close();
    builder.build().ok().map(|path| (path, corners))
}

pub(crate) fn circle(center: Point<Pixels>, radius: f32, color: Hsla) -> PaintQuad {
    fill(circle_bounds(center, radius), color).corner_radii(px(radius))
}

fn circle_bounds(center: Point<Pixels>, radius: f32) -> Bounds<Pixels> {
    Bounds {
        origin: point(center.x - px(radius), center.y - px(radius)),
        size: size(px(radius * 2.0), px(radius * 2.0)),
    }
}

fn absolute_point(bounds: Bounds<Pixels>, point_value: Vec2) -> Point<Pixels> {
    point(
        bounds.origin.x + px(point_value.x),
        bounds.origin.y + px(point_value.y),
    )
}

fn point_is_visible(point_value: Vec2, viewport: Vec2, margin: f32) -> bool {
    point_value.x >= -margin
        && point_value.y >= -margin
        && point_value.x <= viewport.x + margin
        && point_value.y <= viewport.y + margin
}

fn segment_may_be_visible(from: Vec2, to: Vec2, viewport: Vec2) -> bool {
    let min_x = from.x.min(to.x);
    let min_y = from.y.min(to.y);
    let max_x = from.x.max(to.x);
    let max_y = from.y.max(to.y);
    max_x >= -16.0 && max_y >= -16.0 && min_x <= viewport.x + 16.0 && min_y <= viewport.y + 16.0
}

/// A colour for a label no `LabelColors` has seen, from its name alone.
fn label_color(label: &str) -> Hsla {
    let hash = label.bytes().fold(0_u64, |hash, byte| {
        hash.wrapping_mul(109).wrapping_add(byte as u64)
    });
    node_color_for_index(hash as usize)
}

#[cfg(test)]
mod label_color_tests {
    use super::*;
    use crate::theme::NODE_COLOR_COUNT;

    #[test]
    fn every_palette_colour_is_distinct() {
        for (count, color_for) in [
            (NODE_COLOR_COUNT, node_color_for_index as fn(usize) -> Hsla),
            (RELATIONSHIP_COLOR_COUNT, relationship_color_for_index),
        ] {
            let colors: Vec<Hsla> = (0..count).map(color_for).collect();
            for (index, color) in colors.iter().enumerate() {
                assert!(!colors[..index].contains(color), "colour {index} repeats");
            }
            // The palette wraps only after every colour is used.
            assert_eq!(color_for(count), colors[0]);
        }

        // Ten relationship types, as in the agent-memory demo, each get
        // their own colour.
        let mut snapshot = SceneSnapshot::demo();
        let types: Vec<Arc<str>> = (0..10)
            .map(|index| Arc::from(format!("T{index}")))
            .collect();
        snapshot.edges.labels = types.clone();
        let mut colors = LabelColors::default();
        colors.extend(&snapshot);
        let drawn: Vec<Hsla> = types
            .iter()
            .map(|label| colors.relationship(label))
            .collect();
        for (index, color) in drawn.iter().enumerate() {
            assert!(
                !drawn[..index].contains(color),
                "type {index} repeats a colour"
            );
        }
    }

    #[test]
    fn labels_keep_distinct_colours_in_the_order_they_appear() {
        let mut snapshot = SceneSnapshot::demo();
        let labels = ["Repository", "File", "Session", "File", "Commit", "Branch"];
        for (index, label) in labels.iter().enumerate() {
            snapshot.nodes.labels[index] = Arc::from(*label);
        }
        snapshot.nodes.labels.truncate(labels.len());
        let types = ["READ", "CONTAINS", "READ", "EDITED"];
        for (index, label) in types.iter().enumerate() {
            snapshot.edges.labels[index] = Arc::from(*label);
        }
        snapshot.edges.labels.truncate(types.len());
        let mut colors = LabelColors::default();
        colors.extend(&snapshot);
        // Relationship types take theirs the same way, from their own list.
        assert_eq!(
            ["READ", "CONTAINS", "EDITED"].map(|label| colors.relationship_index(label)),
            [0, 1, 2]
        );
        let first: Vec<Hsla> = ["Repository", "File", "Session", "Commit", "Branch"]
            .iter()
            .map(|label| colors.color(label))
            .collect();
        assert_eq!(first[0], node_color_for_index(0));
        assert_eq!(first[4], node_color_for_index(4));
        for (index, color) in first.iter().enumerate() {
            assert!(
                !first[..index].contains(color),
                "label {index} repeats a colour"
            );
        }

        // A label that appears later takes the next colour, and the others
        // keep theirs, even when a scene without them is read.
        snapshot.nodes.labels = vec![Arc::from("ErrorSignature"), Arc::from("File")];
        snapshot.edges.labels = vec![Arc::from("HIT"), Arc::from("READ")];
        colors.extend(&snapshot);
        assert_eq!(colors.relationship_index("HIT"), 3);
        assert_eq!(colors.relationship_index("READ"), 0);
        assert_eq!(colors.color("ErrorSignature"), node_color_for_index(5));
        assert_eq!(colors.color("Repository"), first[0]);
        assert_eq!(colors.color("Branch"), first[4]);
    }
}

#[cfg(test)]
mod line_tests {
    use super::*;

    #[test]
    fn straight_line_builder_matches_lyon_vertices_bit_for_bit() {
        let mut segments = vec![
            (point(px(10.0), px(20.0)), point(px(110.0), px(70.0))),
            (point(px(-33.25), px(400.5)), point(px(912.75), px(-8.0))),
            (point(px(81.0), px(17.0)), point(px(81.0), px(617.0))),
            (
                point(px(516.249_76), px(415.681_4)),
                point(px(0.439_453_13), px(1_959.137)),
            ),
            (point(px(2.0), px(3.0)), point(px(2.001), px(3.001))),
        ];
        let mut state = 0x5645_4347_5241_4c49_u64;
        let mut sample = || {
            state = state
                .wrapping_add(0x9e37_79b9_7f4a_7c15)
                .wrapping_mul(0xbf58_476d_1ce4_e5b9);
            ((state >> 32) as u32 as f32 / u32::MAX as f32) * 4_000.0 - 2_000.0
        };
        for _ in 0..4_000 {
            let from = point(px(sample()), px(sample()));
            let mut to = point(px(sample()), px(sample()));
            if to == from {
                to.x += px(0.25);
            }
            segments.push((from, to));
        }
        for width in [0.74, 0.95, 1.2, 1.65, 2.5] {
            let mut lyon = PathBuilder::stroke(px(width));
            let mut direct = LinePathBuilder::stroke(px(width));
            for &(from, to) in &segments {
                lyon.move_to(from);
                lyon.line_to(to);
                direct.segment(from, to);
            }
            let lyon = lyon.build().unwrap();
            let direct = direct.build().unwrap();
            assert_eq!(direct.vertices.len(), lyon.vertices.len());
            for (vertex_index, (direct, lyon)) in
                direct.vertices.iter().zip(&lyon.vertices).enumerate()
            {
                let (from, to) = segments[vertex_index / 6];
                assert_eq!(
                    f32::from(direct.xy_position.x).to_bits(),
                    f32::from(lyon.xy_position.x).to_bits(),
                    "width {width}, vertex {vertex_index}, segment {from:?}->{to:?}, x: direct {}, lyon {}",
                    f32::from(direct.xy_position.x),
                    f32::from(lyon.xy_position.x),
                );
                assert_eq!(
                    f32::from(direct.xy_position.y).to_bits(),
                    f32::from(lyon.xy_position.y).to_bits(),
                    "width {width}, vertex {vertex_index}, segment {from:?}->{to:?}, y: direct {}, lyon {}",
                    f32::from(direct.xy_position.y),
                    f32::from(lyon.xy_position.y),
                );
            }
        }
    }
}

#[cfg(test)]
mod caption_tests {
    use super::*;

    fn context(selected: Option<usize>, viewport: Vec2) -> NodePaintContext<'static> {
        NodePaintContext {
            label_colors: Box::leak(Box::default()),
            selected,
            dragging: None,
            emphasis: None,
            path_endpoints: None,
            bounds: Bounds {
                origin: point(px(0.0), px(0.0)),
                size: size(px(viewport.x), px(viewport.y)),
            },
            viewport,
        }
    }

    #[test]
    fn the_selection_and_its_neighbors_are_named_without_overlapping() {
        let mut snapshot = SceneSnapshot::demo();
        snapshot.layout(vecgra_studio_core::LayoutOptions::default());
        let viewport = Vec2::new(1_200.0, 900.0);
        let camera = Camera::fit(snapshot.bounds);
        let projected: Vec<Vec2> = snapshot
            .nodes
            .positions
            .iter()
            .map(|&position| camera.project(position, snapshot.bounds, viewport))
            .collect();
        let selected = (0..projected.len())
            .max_by_key(|&index| snapshot.nodes.degrees[index])
            .unwrap();

        let captions = node_captions(
            &snapshot,
            &projected,
            &[],
            None,
            &context(Some(selected), viewport),
            DetailLevel::Communities,
        );
        assert!(!captions.is_empty() && captions.len() <= NODE_CAPTION_BUDGET);
        assert!(captions[0].strong, "the selection is named first");
        assert_eq!(
            captions[0].text,
            node_caption_text(&snapshot.node_title(selected))
        );
        assert!(captions[1..].iter().all(|caption| !caption.strong));
        let areas: Vec<_> = captions
            .iter()
            .map(|caption| Bounds {
                origin: caption.origin,
                size: size(
                    px(caption.text.chars().count() as f32 * 6.2 + 10.0),
                    px(16.0),
                ),
            })
            .collect();
        for (index, area) in areas.iter().enumerate().skip(1) {
            assert!(
                areas[..index].iter().all(|other| !other.intersects(area)),
                "{} overlaps an earlier name",
                captions[index].text
            );
        }

        // A node at the right edge is named on its left.
        let mut edge = projected.clone();
        edge[selected] = Vec2::new(viewport.x - 6.0, viewport.y / 2.0);
        let flipped = node_captions(
            &snapshot,
            &edge,
            &[],
            None,
            &context(Some(selected), viewport),
            DetailLevel::Communities,
        );
        let right = flipped[0].origin.x + px(flipped[0].text.chars().count() as f32 * 6.2 + 10.0);
        assert!(flipped[0].origin.x < px(viewport.x - 6.0) && right <= px(viewport.x));

        let unselected = node_captions(
            &snapshot,
            &projected,
            &[],
            None,
            &context(None, viewport),
            DetailLevel::Communities,
        );
        assert!(
            unselected.is_empty(),
            "nothing is named at community zoom without a reason"
        );

        let relationship = 0;
        let ends = node_captions(
            &snapshot,
            &projected,
            &[],
            Some(relationship),
            &context(None, viewport),
            DetailLevel::Communities,
        );
        let named: Vec<_> = ends.iter().map(|caption| caption.text.clone()).collect();
        assert_eq!(
            named,
            [
                snapshot.edges.sources[relationship],
                snapshot.edges.targets[relationship]
            ]
            .map(|node| node_caption_text(&snapshot.node_title(node as usize))),
            "a selected relationship names both of its ends"
        );
        assert!(ends.iter().all(|caption| caption.strong));
    }

    #[test]
    fn a_traced_path_names_the_nodes_it_runs_through() {
        let mut snapshot = SceneSnapshot::demo();
        snapshot.layout(vecgra_studio_core::LayoutOptions::default());
        let viewport = Vec2::new(1_200.0, 900.0);
        let camera = Camera::fit(snapshot.bounds);
        let mut projected: Vec<Vec2> = snapshot
            .nodes
            .positions
            .iter()
            .map(|&position| camera.project(position, snapshot.bounds, viewport))
            .collect();
        // The path starts at the best-connected node, whose neighbors alone
        // would fill the caption budget.
        let start = (0..projected.len())
            .max_by_key(|&index| snapshot.nodes.degrees[index])
            .unwrap();
        let interior = (0..projected.len())
            .filter(|&index| index != start)
            .min_by_key(|&index| snapshot.nodes.degrees[index])
            .unwrap();
        let end = (0..projected.len())
            .find(|&index| index != start && index != interior)
            .unwrap();
        for other in 0..projected.len() {
            if ![start, interior, end].contains(&other) {
                snapshot.edges.ids.push(snapshot.edges.ids.len() as u64);
                snapshot.edges.sources.push(start as u32);
                snapshot.edges.targets.push(other as u32);
                snapshot.edges.labels.push("SUPPORTS".into());
                snapshot.edges.vector_counts.push(0);
                snapshot.edges.properties.push(Arc::from([]));
                snapshot.nodes.degrees[start] += 1;
                snapshot.nodes.degrees[other] += 1;
            }
        }
        projected[start] = Vec2::new(80.0, 80.0);
        projected[end] = Vec2::new(80.0, 820.0);
        projected[interior] = Vec2::new(600.0, 450.0);
        let mut scores = vec![0.0; projected.len()];
        scores[start] = 1.0;
        scores[end] = 1.0;
        scores[interior] = PATH_INTERIOR_SCORE;
        let scores: Arc<[f32]> = scores.into();
        let edges: Arc<[f32]> = vec![0.0; snapshot.edges.ids.len()].into();
        let lens = GraphEmphasis {
            from_nodes: scores.clone(),
            target_nodes: scores,
            from_edges: edges.clone(),
            target_edges: edges,
            mix: 1.0,
            dim: 0.96,
        };
        let mut context = context(Some(start), viewport);
        context.emphasis = Some(&lens);
        context.path_endpoints = Some(GraphPathEndpoints {
            start,
            end: Some(end),
        });

        let captions = node_captions(
            &snapshot,
            &projected,
            &[],
            None,
            &context,
            DetailLevel::Communities,
        );
        let interior_name = node_caption_text(&snapshot.node_title(interior));
        assert!(
            captions.iter().any(|caption| caption.text == interior_name),
            "the path's interior node is named"
        );
        // The start's other neighbors are dimmed almost out of sight by the
        // path, so they go unnamed.
        let path_names =
            [start, interior, end].map(|node| node_caption_text(&snapshot.node_title(node)));
        assert!(
            captions
                .iter()
                .all(|caption| path_names.contains(&caption.text)),
            "only the path is named, not {:?}",
            captions
                .iter()
                .map(|caption| &caption.text)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn hovering_lights_the_lines_of_what_is_under_the_pointer() {
        let mut snapshot = SceneSnapshot::demo();
        snapshot.layout(vecgra_studio_core::LayoutOptions::default());
        let workspace = GraphWorkspace::new(&snapshot);
        let bounds = Bounds {
            origin: point(px(0.0), px(0.0)),
            size: size(px(1_200.0), px(900.0)),
        };
        let lit = |hovered_node, hovered_edge, selection| {
            prepare_scene(
                &snapshot,
                &workspace,
                ScenePresentation {
                    camera: Camera::fit(snapshot.bounds),
                    world_bounds: snapshot.bounds,
                    selection,
                    dragging: None,
                    emphasis: None,
                    path_endpoints: None,
                    hovered_node,
                    hovered_edge,
                    label_colors: &LabelColors::default(),
                },
                bounds,
            )
            .incident_edge_paths
            .len()
        };
        assert_eq!(lit(None, None, None), 0);
        // Node 5's lines are NEXT and SUPPORTS, one batch per color.
        assert_eq!(lit(Some(5), None, None), 2);
        assert_eq!(lit(None, Some(0), None), 1);
        // The selection's lines are already lit.
        let selection = Some(SceneSelection::Node(5));
        assert_eq!(lit(Some(5), None, selection), lit(None, None, selection));
        let selection = Some(SceneSelection::Edge(0));
        assert_eq!(lit(None, Some(0), selection), lit(None, None, selection));
    }

    #[test]
    fn long_titles_and_paths_are_cut_to_a_canvas_name() {
        assert_eq!(node_caption_text("crates/vecgra/src/lib.rs"), "lib.rs");
        let long = "a".repeat(NODE_CAPTION_CHARS + 5);
        let caption = node_caption_text(&long);
        assert_eq!(caption.chars().count(), NODE_CAPTION_CHARS);
        assert!(caption.ends_with('…'));
    }
}
