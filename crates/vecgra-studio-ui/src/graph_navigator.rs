use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use gpui::{
    App, Bounds, Hsla, IntoElement, Pixels, Point, Styled as _, Window, canvas, point, px, quad,
    size,
};
use vecgra_studio_core::{Camera, GraphWorkspace, Rect, SceneSelection, SceneSnapshot, Vec2};

use crate::graph_canvas::{GraphEmphasis, circle, label_color};
use crate::theme::palette;

const NAVIGATOR_PADDING: f32 = 9.0;
const BIN_SIZE: f32 = 2.8;

#[derive(Clone, Copy)]
struct NavigatorNode {
    point: Point<Pixels>,
    radius: f32,
    color: Hsla,
    selected: bool,
}

struct NavigatorPaintData {
    nodes: Arc<[NavigatorNode]>,
    viewport: Bounds<Pixels>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct NavigatorCacheKey {
    snapshot: usize,
    positions: usize,
    position_revision: u64,
    selected: Option<usize>,
    bounds: [u32; 4],
    world_bounds: [u32; 4],
}

/// The miniature graph is invariant under main-camera movement; only its
/// viewport rectangle changes. Retaining it removes a full node scan from
/// every pan and zoom frame without changing any painted primitive.
#[derive(Default)]
pub(crate) struct GraphNavigatorCache {
    key: Option<NavigatorCacheKey>,
    nodes: Arc<[NavigatorNode]>,
}

impl GraphNavigatorCache {
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}

struct NavigatorPresentation<'a> {
    camera: Camera,
    world_bounds: Rect,
    main_viewport: Vec2,
    selected: Option<usize>,
    emphasis: Option<&'a GraphEmphasis>,
}

pub(crate) struct GraphNavigatorPresentation {
    pub camera: Camera,
    pub world_bounds: Rect,
    pub main_viewport: Vec2,
    pub selection: Option<SceneSelection>,
    pub emphasis: Option<GraphEmphasis>,
}

/// A bounded, density-aware navigator for deep graph exploration. It bins the
/// complete workspace into screen-space cells, so its paint cost is bounded by
/// the navigator's pixel area rather than by graph cardinality.
pub(crate) fn graph_navigator(
    snapshot: Arc<SceneSnapshot>,
    workspace: Rc<RefCell<GraphWorkspace>>,
    cache: Rc<RefCell<GraphNavigatorCache>>,
    presentation: GraphNavigatorPresentation,
) -> impl IntoElement {
    canvas(
        move |bounds, _window, _cx| {
            prepare_navigator(
                &snapshot,
                &workspace.borrow(),
                &mut cache.borrow_mut(),
                NavigatorPresentation {
                    camera: presentation.camera,
                    world_bounds: presentation.world_bounds,
                    main_viewport: presentation.main_viewport,
                    selected: presentation.selection.and_then(SceneSelection::node),
                    emphasis: presentation.emphasis.as_ref(),
                },
                bounds,
            )
        },
        paint_navigator,
    )
    .absolute()
    .size_full()
}

fn prepare_navigator(
    snapshot: &SceneSnapshot,
    workspace: &GraphWorkspace,
    cache: &mut GraphNavigatorCache,
    presentation: NavigatorPresentation<'_>,
    bounds: Bounds<Pixels>,
) -> NavigatorPaintData {
    struct Bin {
        count: usize,
        label_index: usize,
        selected: bool,
        relevance: f32,
    }

    let world_bounds = presentation.world_bounds;
    let (scale, origin) = navigator_projection(bounds, world_bounds);
    let width = f32::from(bounds.size.width);
    let height = f32::from(bounds.size.height);
    let columns = (width / BIN_SIZE).ceil().max(1.0) as usize;
    let rows = (height / BIN_SIZE).ceil().max(1.0) as usize;
    let cache_key = NavigatorCacheKey {
        snapshot: snapshot as *const SceneSnapshot as usize,
        positions: workspace.positions().as_ptr() as usize,
        position_revision: workspace.position_revision(),
        selected: presentation.selected,
        bounds: [
            f32::from(bounds.origin.x).to_bits(),
            f32::from(bounds.origin.y).to_bits(),
            width.to_bits(),
            height.to_bits(),
        ],
        world_bounds: [
            world_bounds.min.x.to_bits(),
            world_bounds.min.y.to_bits(),
            world_bounds.max.x.to_bits(),
            world_bounds.max.y.to_bits(),
        ],
    };
    let cache_hit = presentation.emphasis.is_none() && cache.key == Some(cache_key);
    let nodes = if cache_hit {
        cache.nodes.clone()
    } else {
        let mut bins = std::iter::repeat_with(|| None)
            .take(columns * rows)
            .collect::<Vec<Option<Bin>>>();
        for (index, position) in workspace.positions().iter().copied().enumerate() {
            let projected = Vec2::new(
                origin.x + (position.x - world_bounds.min.x) * scale,
                origin.y + (position.y - world_bounds.min.y) * scale,
            );
            let column = (projected.x.clamp(0.0, width - f32::EPSILON) / BIN_SIZE) as usize;
            let row = (projected.y.clamp(0.0, height - f32::EPSILON) / BIN_SIZE) as usize;
            let relevance = presentation
                .emphasis
                .map_or(0.0, |lens| lens.node_score(index));
            let bin = bins[row * columns + column].get_or_insert(Bin {
                count: 0,
                label_index: index,
                selected: false,
                relevance: 0.0,
            });
            bin.count += 1;
            bin.selected |= presentation.selected == Some(index);
            if relevance > bin.relevance {
                bin.relevance = relevance;
                bin.label_index = index;
            }
        }
        let nodes: Arc<[NavigatorNode]> = bins
            .into_iter()
            .enumerate()
            .filter_map(|(bin_index, bin)| bin.map(|bin| (bin_index, bin)))
            .map(|(bin_index, bin)| {
                let density = (bin.count as f32).ln_1p();
                let opacity = if presentation.emphasis.is_some() {
                    if bin.relevance > 0.01 {
                        0.46 + bin.relevance * 0.5
                    } else {
                        0.1
                    }
                } else {
                    (0.2 + density * 0.12).min(0.72)
                };
                NavigatorNode {
                    point: point(
                        bounds.origin.x + px(((bin_index % columns) as f32 + 0.5) * BIN_SIZE),
                        bounds.origin.y + px(((bin_index / columns) as f32 + 0.5) * BIN_SIZE),
                    ),
                    radius: if bin.selected {
                        2.8
                    } else {
                        (0.7 + density * 0.42).min(2.0)
                    },
                    color: label_color(&snapshot.nodes.labels[bin.label_index]).opacity(opacity),
                    selected: bin.selected,
                }
            })
            .collect();
        if presentation.emphasis.is_none() {
            cache.key = Some(cache_key);
            cache.nodes = nodes.clone();
        }
        nodes
    };

    let visible_min =
        presentation
            .camera
            .unproject(Vec2::ZERO, world_bounds, presentation.main_viewport);
    let visible_max = presentation.camera.unproject(
        presentation.main_viewport,
        world_bounds,
        presentation.main_viewport,
    );
    let project_world = |position: Vec2| {
        point(
            bounds.origin.x + px(origin.x + (position.x - world_bounds.min.x) * scale),
            bounds.origin.y + px(origin.y + (position.y - world_bounds.min.y) * scale),
        )
    };
    let top_left = project_world(Vec2::new(
        visible_min.x.max(world_bounds.min.x),
        visible_min.y.max(world_bounds.min.y),
    ));
    let bottom_right = project_world(Vec2::new(
        visible_max.x.min(world_bounds.max.x),
        visible_max.y.min(world_bounds.max.y),
    ));
    let viewport = Bounds::from_corners(top_left, bottom_right);
    let viewport = Bounds::centered_at(
        viewport.center(),
        size(
            viewport.size.width.max(px(5.0)),
            viewport.size.height.max(px(5.0)),
        ),
    );

    NavigatorPaintData { nodes, viewport }
}

fn navigator_projection(bounds: Bounds<Pixels>, world_bounds: Rect) -> (f32, Vec2) {
    let width = f32::from(bounds.size.width);
    let height = f32::from(bounds.size.height);
    let available_width = (width - NAVIGATOR_PADDING * 2.0).max(1.0);
    let available_height = (height - NAVIGATOR_PADDING * 2.0).max(1.0);
    let scale = (available_width / world_bounds.width())
        .min(available_height / world_bounds.height())
        .max(0.000_1);
    let graph_width = world_bounds.width() * scale;
    let graph_height = world_bounds.height() * scale;
    (
        scale,
        Vec2::new((width - graph_width) * 0.5, (height - graph_height) * 0.5),
    )
}

pub(crate) fn navigator_world_position(
    bounds: Bounds<Pixels>,
    world_bounds: Rect,
    screen_position: Point<Pixels>,
) -> Vec2 {
    let (scale, origin) = navigator_projection(bounds, world_bounds);
    let position = Vec2::new(
        (f32::from(screen_position.x - bounds.origin.x) - origin.x) / scale + world_bounds.min.x,
        (f32::from(screen_position.y - bounds.origin.y) - origin.y) / scale + world_bounds.min.y,
    );
    Vec2::new(
        position.x.clamp(world_bounds.min.x, world_bounds.max.x),
        position.y.clamp(world_bounds.min.y, world_bounds.max.y),
    )
}

fn paint_navigator(
    bounds: Bounds<Pixels>,
    navigator: NavigatorPaintData,
    window: &mut Window,
    _cx: &mut App,
) {
    window.paint_layer(bounds, |window| {
        for node in navigator.nodes.iter().copied() {
            if node.selected {
                window.paint_quad(circle(
                    node.point,
                    node.radius + 2.0,
                    palette().cobalt.opacity(0.22),
                ));
            }
            window.paint_quad(circle(node.point, node.radius, node.color));
        }
        window.paint_quad(quad(
            navigator.viewport,
            px(2.0),
            palette().cobalt.opacity(0.08),
            px(1.0),
            palette().cobalt.opacity(0.88),
            Default::default(),
        ));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_motion_reuses_identical_navigator_geometry() {
        let snapshot = SceneSnapshot::demo();
        let mut workspace = GraphWorkspace::new(&snapshot);
        let bounds = Bounds {
            origin: point(px(1_200.0), px(12.0)),
            size: size(px(148.0), px(104.0)),
        };
        let viewport = Vec2::new(1_236.0, 918.0);
        let mut cache = GraphNavigatorCache::default();
        let first = prepare_navigator(
            &snapshot,
            &workspace,
            &mut cache,
            NavigatorPresentation {
                camera: Camera::fit(snapshot.bounds),
                world_bounds: snapshot.bounds,
                main_viewport: viewport,
                selected: Some(3),
                emphasis: None,
            },
            bounds,
        );
        let moved = prepare_navigator(
            &snapshot,
            &workspace,
            &mut cache,
            NavigatorPresentation {
                camera: Camera {
                    center: snapshot.bounds.max,
                    zoom: 24.0,
                },
                world_bounds: snapshot.bounds,
                main_viewport: viewport,
                selected: Some(3),
                emphasis: None,
            },
            bounds,
        );
        assert!(Arc::ptr_eq(&first.nodes, &moved.nodes));
        assert_ne!(first.viewport, moved.viewport);

        workspace.drag_to(3, Vec2::new(900.0, -120.0), 48.0);
        let changed = prepare_navigator(
            &snapshot,
            &workspace,
            &mut cache,
            NavigatorPresentation {
                camera: Camera::fit(snapshot.bounds),
                world_bounds: snapshot.bounds,
                main_viewport: viewport,
                selected: Some(3),
                emphasis: None,
            },
            bounds,
        );
        assert!(!Arc::ptr_eq(&first.nodes, &changed.nodes));
    }
}
