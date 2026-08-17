use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use gpui::{
    App, Bounds, FontWeight, Hsla, IntoElement, PaintQuad, Path, PathBuilder, Pixels, Point,
    SharedString, Styled, TextAlign, TextRun, Window, canvas, fill, point, px, quad, rgb, size,
};
use vectorgraph_studio_core::{
    Camera, DetailLevel, GraphWorkspace, Rect, SceneSelection, SceneSnapshot, Vec2, detail_level,
};

use crate::theme::{
    RELATIONSHIP_COLOR_COUNT, palette, relationship_color, relationship_color_for_index,
    relationship_color_index,
};

const EDGE_PATH_CHUNK: usize = 6_000;
const COMMUNITY_EDGE_BUDGET: usize = 18_000;
const REPRESENTATIVE_CAPTION_BUDGET: usize = 10;
const LOCAL_CAPTION_BUDGET: usize = 24;
const LOCAL_ARROW_BUDGET: usize = 48;

struct PaintNode {
    point: Point<Pixels>,
    radius: f32,
    color: Hsla,
    selected: bool,
    pinned: bool,
    dragging: bool,
    relevance: f32,
}

struct PaintPath {
    path: Path<Pixels>,
    color: Hsla,
}

struct PaintArrow {
    path: Path<Pixels>,
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
}

struct ScenePresentation<'a> {
    camera: Camera,
    world_bounds: Rect,
    selection: Option<SceneSelection>,
    dragging: Option<usize>,
    emphasis: Option<&'a GraphEmphasis>,
}

struct NodePaintContext<'a> {
    selected: Option<usize>,
    dragging: Option<usize>,
    emphasis: Option<&'a GraphEmphasis>,
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
    fn node_score(&self, index: usize) -> f32 {
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
    camera: Camera,
    world_bounds: Rect,
    selection: Option<SceneSelection>,
    dragging: Option<usize>,
    emphasis: Option<GraphEmphasis>,
) -> impl IntoElement {
    canvas(
        move |bounds, _window, _cx| {
            prepare_scene(
                &snapshot,
                &workspace.borrow(),
                ScenePresentation {
                    camera,
                    world_bounds,
                    selection,
                    dragging,
                    emphasis: emphasis.as_ref(),
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
            let mut builders: [PathBuilder; RELATIONSHIP_COLOR_COUNT] =
                std::array::from_fn(|_| PathBuilder::stroke(px(width)));
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
                let color_index = relationship_color_index(&snapshot.edges.labels[edge_index]);
                builders[color_index].move_to(absolute_point(bounds, from));
                builders[color_index].line_to(absolute_point(bounds, to));
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
                    && let Ok(path) = builder.build()
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
            let color = relationship_color(label).opacity(0.42 + score * 0.58);
            let mut builder = PathBuilder::stroke(px(1.2 + score * 1.5));
            builder.move_to(absolute_point(bounds, from));
            builder.line_to(absolute_point(bounds, to));
            if let Ok(path) = builder.build() {
                emphasis_edge_paths.push(PaintPath { path, color });
            }
            if arrows.len() < LOCAL_ARROW_BUDGET
                && let Some(path) = arrowhead(bounds, from, to)
            {
                arrows.push(PaintArrow {
                    path,
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
    // bright lines to deliberately hidden nodes.
    if let Some(node_index) = selected_node
        && emphasis.is_none_or(|lens| lens.dim < 0.35)
    {
        let mut builders: [PathBuilder; RELATIONSHIP_COLOR_COUNT] =
            std::array::from_fn(|_| PathBuilder::stroke(px(1.65)));
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
            let color_index = relationship_color_index(label);
            builders[color_index].move_to(absolute_point(bounds, from));
            builders[color_index].line_to(absolute_point(bounds, to));
            segment_counts[color_index] += 1;
            if arrows.len() < LOCAL_ARROW_BUDGET
                && let Some(path) = arrowhead(bounds, from, to)
            {
                arrows.push(PaintArrow {
                    path,
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
                && let Ok(path) = builder.build()
            {
                incident_edge_paths.push(PaintPath {
                    path,
                    color: relationship_color_for_index(color_index).opacity(0.78),
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
        let color = relationship_color(label);
        let mut builder = PathBuilder::stroke(px(2.5));
        builder.move_to(absolute_point(bounds, from));
        builder.line_to(absolute_point(bounds, to));
        if let Some(path) = arrowhead(bounds, from, to) {
            arrows.push(PaintArrow {
                path,
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
        builder.build().ok().map(|path| PaintPath { path, color })
    });

    if level != DetailLevel::Overview {
        for (label, candidate) in representative_edges
            .into_iter()
            .take(REPRESENTATIVE_CAPTION_BUDGET)
        {
            let color = relationship_color(&label);
            if push_caption(
                &mut captions,
                label,
                caption_center(bounds, candidate.from, candidate.to),
                color,
                false,
            ) && let Some(path) = arrowhead(bounds, candidate.from, candidate.to)
            {
                arrows.push(PaintArrow {
                    path,
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
                selected: selected_node,
                dragging: presentation.dragging,
                emphasis,
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
                        color: label_color(label).opacity(lens_opacity),
                        selected: selected_node == Some(index),
                        pinned: workspace.is_pinned(index),
                        dragging: presentation.dragging == Some(index),
                        relevance,
                    }
                })
            })
            .collect(),
    };

    PaintData {
        edge_paths,
        emphasis_edge_paths,
        incident_edge_paths,
        selected_edge_path,
        arrows,
        captions,
        nodes,
    }
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
        });
        bin.sum += position;
        bin.count += 1;
        bin.selected |= context.selected == Some(index);
        bin.pinned |= workspace.is_pinned(index);
        bin.dragging |= context.dragging == Some(index);
        bin.relevance = bin
            .relevance
            .max(context.emphasis.map_or(0.0, |lens| lens.node_score(index)));
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
                color: label_color(&snapshot.nodes.labels[bin.label_index])
                    .opacity((0.24 + density * 0.105).min(0.78) * lens_opacity),
                selected: bin.selected,
                pinned: bin.pinned,
                dragging: bin.dragging,
                relevance: bin.relevance,
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

fn arrowhead(bounds: Bounds<Pixels>, from: Vec2, to: Vec2) -> Option<Path<Pixels>> {
    let delta = to - from;
    let length = delta.length();
    if length < 22.0 {
        return None;
    }
    let direction = delta / length;
    let perpendicular = Vec2::new(-direction.y, direction.x);
    let tip = to - direction * 7.0;
    let base = tip - direction * 7.5;
    let mut builder = PathBuilder::fill();
    builder.move_to(absolute_point(bounds, tip));
    builder.line_to(absolute_point(bounds, base + perpendicular * 3.8));
    builder.line_to(absolute_point(bounds, base - perpendicular * 3.8));
    builder.close();
    builder.build().ok()
}

fn circle(center: Point<Pixels>, radius: f32, color: Hsla) -> PaintQuad {
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

fn label_color(label: &str) -> Hsla {
    let colors = palette();
    let hash = label.bytes().fold(0_u64, |hash, byte| {
        hash.wrapping_mul(109).wrapping_add(byte as u64)
    });
    match hash % 5 {
        0 => colors.cobalt,
        1 => colors.copper,
        2 => colors.celadon,
        3 => rgb(0x9b8cf2).into(),
        _ => rgb(0xd6c56b).into(),
    }
}
