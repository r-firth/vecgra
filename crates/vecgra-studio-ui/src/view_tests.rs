use super::*;
use gpui::{KeyBinding, Modifiers, ScrollDelta, TestAppContext, point, size};
use std::time::Duration;
use vecgra_studio_core::{EvidencePath, EvidenceStep};

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
            Some("VecgraStudio"),
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
fn selecting_by_id_brings_hidden_elements_into_view(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    view.update_in(cx, |view, _, _| view.settle_presentation_for_capture());
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    view.update_in(cx, |view, window, cx| {
        let position = |view: &StudioView, index: usize| {
            view.workspace
                .as_ref()
                .unwrap()
                .borrow()
                .position(index)
                .unwrap()
        };
        let node = position(view, 5);

        // Visible already: the camera stays where it is.
        view.camera.center = node + Vec2::new(1.0, 1.0);
        view.camera_motion.cancel_at(view.camera);
        let before = view.camera.center;
        view.execute_command("node 5", window, cx);
        view.settle_presentation_for_capture();
        assert_eq!(view.camera.center, before);

        // Far outside the canvas: the camera springs to it at the same zoom.
        view.camera.center = node + Vec2::new(100_000.0, 0.0);
        view.camera_motion.cancel_at(view.camera);
        let zoom = view.camera.zoom;
        view.execute_command("node 5", window, cx);
        view.settle_presentation_for_capture();
        assert!((view.camera.center - node).length() < 0.5);
        assert_eq!(view.camera.zoom, zoom);

        // A relationship is framed by its midpoint.
        let snapshot = view.snapshot().unwrap().clone();
        let (source, target) = (
            snapshot.edges.sources[0] as usize,
            snapshot.edges.targets[0] as usize,
        );
        let middle = (position(view, source) + position(view, target)) * 0.5;
        view.camera.center = middle + Vec2::new(0.0, 100_000.0);
        view.camera_motion.cancel_at(view.camera);
        view.execute_command(&format!("edge {}", snapshot.edges.ids[0]), window, cx);
        view.settle_presentation_for_capture();
        assert!((view.camera.center - middle).length() < 0.5);
    });
}

#[gpui::test]
fn titles_that_start_with_a_command_word_are_searched(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));
    window
        .update(cx, |view, window, cx| {
            view.execute_command("node 3", window, cx);
            view.execute_command("node 5", window, cx);
            let layout = view.layout_kind;
            view.execute_command("back pressure", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Node(5)));
            assert!(matches!(
                &view.search_state,
                SearchState::Failed { query, .. } if query.as_ref() == "back pressure"
            ));
            view.execute_command("layout refresh", window, cx);
            assert_eq!(view.layout_kind, layout);
            assert!(matches!(
                &view.search_state,
                SearchState::Failed { query, .. } if query.as_ref() == "layout refresh"
            ));
            view.execute_command("back", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Node(3)));
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
            let screen = view
                .camera
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
        bezel_ui::focus::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));

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
    assert!(cx.debug_bounds("bezel-graph-controls").is_none());
    assert!(cx.debug_bounds("bezel-search-modes").is_some());
    assert!(cx.debug_bounds("bezel-layout-modes").is_some());
    assert!(cx.debug_bounds("compact-semantic-depth").is_some());
    assert_eq!(
        cx.debug_bounds("bezel-search-modes").unwrap().center().y,
        cx.debug_bounds("bezel-layout-modes").unwrap().center().y
    );

    let semantic = cx
        .debug_bounds("bezel-search-semantic")
        .expect("Bezel semantic mode should render in the compact toolbar");
    cx.simulate_click(semantic.center(), Modifiers::none());
    assert_eq!(
        cx.read(|cx| view.read(cx).search_mode),
        SearchMode::Semantic
    );
    cx.update(|window, cx| {
        let text_focus = view.read(cx).bezel_search_focus[0].clone();
        window.focus(&text_focus, cx);
        _ = window.draw(cx);
    });
    cx.simulate_keystrokes("space");
    assert_eq!(cx.read(|cx| view.read(cx).search_mode), SearchMode::Text);

    cx.simulate_resize(size(px(1_340.0), px(820.0)));
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("wide-toolbar").is_some());
    assert!(cx.debug_bounds("compact-toolbar-primary").is_none());
    assert!(cx.debug_bounds("bezel-graph-controls").is_some());
    assert!(cx.debug_bounds("semantic-depth-readout").is_some());
    assert!(cx.debug_bounds("compact-semantic-depth").is_none());
    assert!(cx.debug_bounds("bezel-layout-modes").is_none());
    assert!(cx.debug_bounds("bezel-sidebar-tabs").is_some());

    let search = cx
        .debug_bounds("wide-search-field")
        .expect("wide search field should render");
    let canvas = cx
        .debug_bounds("graph-canvas")
        .expect("graph canvas should render");
    assert_eq!(search.left(), canvas.left());
    assert_eq!(
        search.center().y,
        cx.debug_bounds("bezel-search-modes")
            .expect("wide search modes should render")
            .center()
            .y
    );

    cx.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.execute_command("text graph", window, cx);
        });
    });
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let detailed_search = cx
        .debug_bounds("wide-search-field")
        .expect("wide search field should remain visible for search results");
    let detailed_canvas = cx
        .debug_bounds("graph-canvas")
        .expect("graph canvas should remain visible for search results");
    assert_eq!(detailed_search.left(), detailed_canvas.left());

    let zoom_before = cx.read(|cx| view.read(cx).camera.zoom);
    let zoom_in = cx
        .debug_bounds("bezel-zoom-in")
        .expect("Bezel zoom control should render at the wide breakpoint");
    cx.simulate_click(zoom_in.center(), Modifiers::none());
    assert!(cx.read(|cx| view.read(cx).camera.zoom) > zoom_before);
}

#[gpui::test]
fn deep_zoom_navigator_recenters_without_starting_a_canvas_drag(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        bezel_ui::focus::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));

    view.update(cx, |view, cx| {
        view.camera.zoom = 4.0;
        view.camera_motion.cancel_at(view.camera);
        view.set_selection(Some(SceneSelection::Node(0)));
        cx.notify();
    });
    cx.simulate_resize(size(px(1_340.0), px(820.0)));
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });

    let navigator = cx
        .debug_bounds("graph-navigator")
        .expect("deep zoom should show the graph navigator");
    let camera_before = cx.read(|cx| view.read(cx).camera);
    cx.simulate_click(
        point(navigator.right() - px(12.0), navigator.top() + px(12.0)),
        Modifiers::none(),
    );
    view.update(cx, |view, _| view.settle_presentation_for_capture());

    cx.read(|cx| {
        let view = view.read(cx);
        assert_ne!(view.camera.center, camera_before.center);
        assert_eq!(view.camera.zoom, camera_before.zoom);
        assert_eq!(view.selection, Some(SceneSelection::Node(0)));
        assert!(view.drag.is_none());
        assert!(view.status.contains("Navigator"));
    });
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
            .map(|index| vecgra_studio_core::SceneProperty {
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

struct LiveDatabase {
    path: PathBuf,
    database: vecgra::Database,
    files: Vec<u64>,
}

impl LiveDatabase {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("vecgra-studio-ui-{name}-{}.vg", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let database = vecgra::Database::create(
            &path,
            vecgra::DatabaseOptions {
                vector_dimension: 4,
                similarity: vecgra::Similarity::Cosine,
                vector_encoding: vecgra::VectorEncoding::F32,
                sync_on_commit: false,
            },
        )
        .unwrap();
        let mut transaction = database.transaction();
        let files: Vec<_> = (0..8)
            .map(|index| {
                transaction.create_node(
                    "File",
                    [(
                        "path",
                        vecgra::Value::String(format!("src/{index}.rs").into()),
                    )],
                    &[],
                )
            })
            .collect();
        for pair in files.windows(2) {
            transaction.create_edge(
                pair[0],
                pair[1],
                "IMPORTS",
                std::iter::empty::<(&str, vecgra::Value)>(),
                &[],
            );
        }
        transaction.commit().unwrap();
        Self {
            path,
            database,
            files,
        }
    }

    /// Commits one new session that edited the third file.
    fn add_session(&self) -> u64 {
        let mut transaction = self.database.transaction();
        let session = transaction.create_node(
            "Session",
            [("title", vecgra::Value::String("Live session".into()))],
            &[],
        );
        transaction.create_edge(
            session,
            self.files[2],
            "EDITED",
            std::iter::empty::<(&str, vecgra::Value)>(),
            &[],
        );
        transaction.commit().unwrap();
        session
    }
}

impl Drop for LiveDatabase {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn open_live_window(
    cx: &mut TestAppContext,
    path: &std::path::Path,
) -> gpui::WindowHandle<StudioView> {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let path = path.to_path_buf();
    let window = cx.add_window(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.run_until_parked();
    window
        .update(cx, |view, _, _| assert!(view.is_ready(), "{}", view.status))
        .unwrap();
    window
}

#[gpui::test]
fn reload_keeps_the_presentation_and_grows_new_commits_into_place(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("reload");
    let window = open_live_window(cx, &live.path);

    let (before_positions, before_camera, pinned_id, selected_id) = window
        .update(cx, |view, window, cx| {
            view.set_live_refresh(false, window, cx);
            let snapshot = view.snapshot().unwrap().clone();
            let pinned = snapshot.node_index(live.files[4]).unwrap();
            view.workspace
                .as_ref()
                .unwrap()
                .borrow_mut()
                .set_pinned(pinned, true);
            let selected = snapshot.node_index(live.files[6]).unwrap();
            view.set_selection(Some(SceneSelection::Node(selected)));
            view.execute_command("zoom 3", window, cx);
            let positions: Vec<_> = live
                .files
                .iter()
                .map(|&id| {
                    view.workspace
                        .as_ref()
                        .unwrap()
                        .borrow()
                        .position(snapshot.node_index(id).unwrap())
                        .unwrap()
                })
                .collect();
            (positions, view.camera, live.files[4], live.files[6])
        })
        .unwrap();

    let session = live.add_session();
    window
        .update(cx, |view, window, cx| {
            view.execute_command("reload", window, cx)
        })
        .unwrap();
    cx.run_until_parked();

    window
        .update(cx, |view, _, _| {
            let snapshot = view.snapshot().unwrap().clone();
            let workspace = view.workspace.as_ref().unwrap().borrow();
            assert!(snapshot.node_index(session).is_some());
            assert_eq!(view.status.as_ref(), "Reloaded · +1 node · +1 relationship");
            assert_eq!(view.camera, before_camera);
            let after_positions: Vec<_> = live
                .files
                .iter()
                .map(|&id| {
                    workspace
                        .position(snapshot.node_index(id).unwrap())
                        .unwrap()
                })
                .collect();
            assert_eq!(after_positions, before_positions);
            assert!(workspace.is_pinned(snapshot.node_index(pinned_id).unwrap()));
            assert_eq!(
                view.selection,
                snapshot.node_index(selected_id).map(SceneSelection::Node)
            );
            // The new session starts at the file it edited and springs out.
            let anchor = workspace
                .position(snapshot.node_index(live.files[2]).unwrap())
                .unwrap();
            let entering = workspace
                .position(snapshot.node_index(session).unwrap())
                .unwrap();
            assert!(
                (entering - anchor).length() < 4.0,
                "{entering:?} vs {anchor:?}"
            );
            assert!(workspace.is_moving());
            assert!(view.fingerprint.is_some());
        })
        .unwrap();
}

#[gpui::test]
fn nodes_a_reload_adds_are_ringed_until_they_fade(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("fresh");
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let path = live.path.clone();
    let (view, cx) = cx.add_window_view(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    view.update_in(cx, |view, window, cx| {
        assert!(view.is_ready(), "{}", view.status);
        assert!(view.fresh_nodes.is_none(), "opening rings nothing");
        view.set_live_refresh(false, window, cx);
    });

    let session = live.add_session();
    view.update_in(cx, |view, window, cx| {
        view.execute_command("reload", window, cx)
    });
    cx.run_until_parked();
    view.read_with(&*cx, |view, _| {
        let fresh = view
            .fresh_nodes
            .as_ref()
            .expect("the new session is ringed");
        assert_eq!(fresh.ids, [session]);
        assert_eq!(fresh.opacity(), 1.0);
    });
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("graph-fresh-ring").is_some());

    view.update(cx, |view, cx| {
        view.fresh_nodes.as_mut().unwrap().since = Instant::now() - FRESH_NODE_DURATION;
        cx.notify();
    });
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("graph-fresh-ring").is_none());
    view.read_with(&*cx, |view, _| assert!(view.fresh_nodes.is_none()));
}

#[gpui::test]
fn shift_click_traces_a_path_from_the_selection(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("shift-path");
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let path = live.path.clone();
    let (view, cx) = cx.add_window_view(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let end_point = view.update(cx, |view, _| {
        assert!(view.is_ready(), "{}", view.status);
        view.settle_presentation_for_capture();
        let snapshot = view.snapshot().unwrap().clone();
        let start = snapshot.node_index(live.files[0]).unwrap();
        let end = snapshot.node_index(live.files[3]).unwrap();
        view.set_selection(Some(SceneSelection::Node(start)));
        let bounds = view.canvas_bounds.unwrap();
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let local = view.camera.project(
            view.workspace
                .as_ref()
                .unwrap()
                .borrow()
                .position(end)
                .unwrap(),
            view.world_bounds,
            viewport,
        );
        bounds.origin + point(px(local.x), px(local.y))
    });

    cx.simulate_click(end_point, Modifiers::shift());
    cx.run_until_parked();
    view.read_with(&*cx, |view, _| {
        let PathState::Ready(report) = &view.path_state else {
            panic!("a path is traced, not {:?}", view.status);
        };
        let steps = report
            .path
            .as_ref()
            .expect("the files are linked")
            .steps
            .len();
        assert_eq!(
            (report.start.id, report.end.id),
            (live.files[0], live.files[3])
        );
        assert_eq!(steps, 3);
        assert!(view.drag.is_none());
    });
}

#[gpui::test]
fn path_steps_name_their_nodes_and_point_out_their_relationship(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("step-preview");
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let path = live.path.clone();
    let (view, cx) = cx.add_window_view(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    let command = format!("path {} {}", live.files[0], live.files[3]);
    view.update_in(cx, |view, window, cx| {
        view.settle_presentation_for_capture();
        view.execute_command(&command, window, cx);
    });
    cx.run_until_parked();
    view.update_in(cx, |view, _, _| view.settle_presentation_for_capture());
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let step_edge = view.read_with(&*cx, |view, _| {
        let PathState::Ready(report) = &view.path_state else {
            panic!("a path is traced, not {:?}", view.status);
        };
        let path = report.path.as_ref().unwrap();
        assert_eq!(
            panels::evidence_step_names(path, 0).as_deref(),
            Some("0.rs → 1.rs")
        );
        assert_eq!(
            panels::evidence_step_names(path, 2).as_deref(),
            Some("2.rs → 3.rs")
        );
        assert_eq!(panels::evidence_step_names(path, 3), None);
        path.steps[1].edge_id
    });

    let row = cx.debug_bounds("evidence-path-step-1").unwrap();
    cx.simulate_mouse_move(row.center(), None, Modifiers::none());
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.hovered_edge.map(|hovered| hovered.id), Some(step_edge));
        assert_eq!(view.hovered_node, None);
    });
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("graph-hover-edge-card").is_some());
    assert!(cx.debug_bounds("graph-hover-ring").is_some());

    // Leaving the row for somewhere other than the canvas ends the preview.
    cx.simulate_mouse_move(point(px(4.0), px(896.0)), None, Modifiers::none());
    view.read_with(&*cx, |view, _| assert!(view.hovered_edge.is_none()));

    // The path's ends point themselves out too, and choosing one selects it.
    let end = cx.debug_bounds("path-endpoint:to").unwrap();
    cx.simulate_mouse_move(end.center(), None, Modifiers::none());
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.hovered_node, Some(live.files[3]))
    });
    cx.simulate_click(end.center(), Modifiers::none());
    view.read_with(&*cx, |view, _| {
        let snapshot = view.snapshot().unwrap();
        assert_eq!(
            view.selection,
            snapshot.node_index(live.files[3]).map(SceneSelection::Node)
        );
        assert!(matches!(view.path_state, PathState::Ready(_)));
    });
}

#[gpui::test]
fn a_path_search_that_finds_nothing_offers_wider_searches(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("path-retry");
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let path = live.path.clone();
    let (view, cx) = cx.add_window_view(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    // The files import each other in order, so nothing leads back out.
    let command = format!("path {} {} out IMPORTS 4", live.files[3], live.files[0]);
    view.update_in(cx, |view, window, cx| {
        view.settle_presentation_for_capture();
        view.execute_command(&command, window, cx);
    });
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    view.read_with(&*cx, |view, _| {
        let PathState::Ready(report) = &view.path_state else {
            panic!("the search finishes, not {:?}", view.status);
        };
        assert!(report.path.is_none());
        assert_eq!(
            path_retries(report),
            [
                PathRetry::EitherDirection,
                PathRetry::AnyRelationship,
                PathRetry::MoreHops(8)
            ]
        );
    });
    assert!(cx.debug_bounds("path-retry-relationship").is_some());
    assert!(cx.debug_bounds("path-retry-hops").is_some());

    let either = cx.debug_bounds("path-retry-direction").unwrap();
    cx.simulate_click(either.center(), Modifiers::none());
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    view.read_with(&*cx, |view, _| {
        let PathState::Ready(report) = &view.path_state else {
            panic!("the wider search finishes, not {:?}", view.status);
        };
        assert_eq!(report.direction, PathDirection::Both);
        assert_eq!(report.relationship_label.as_deref(), Some("IMPORTS"));
        assert_eq!(report.max_hops, 4);
        assert_eq!(report.path.as_ref().unwrap().steps.len(), 3);
        assert!(path_retries(report).is_empty());
    });
    assert!(cx.debug_bounds("path-retry-direction").is_none());
}

#[gpui::test]
fn enter_activates_the_focused_path_step(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("step-keys");
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::init_bezel(cx);
        crate::apply_studio_theme(cx);
        cx.bind_keys([
            KeyBinding::new("enter", ActivateSelection, Some("VecgraStudio")),
            KeyBinding::new("escape", ClearSelection, Some("VecgraStudio")),
        ]);
    });
    let path = live.path.clone();
    let (view, cx) = cx.add_window_view(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    let command = format!("path {} {}", live.files[0], live.files[3]);
    view.update_in(cx, |view, window, cx| {
        view.settle_presentation_for_capture();
        view.execute_command(&command, window, cx);
    });
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let step_edges: Vec<usize> = view.read_with(&*cx, |view, _| {
        let PathState::Ready(report) = &view.path_state else {
            panic!("a path is traced, not {:?}", view.status);
        };
        let snapshot = view.snapshot().unwrap();
        report
            .path
            .as_ref()
            .unwrap()
            .steps
            .iter()
            .map(|step| snapshot.edge_index(step.edge_id).unwrap())
            .collect()
    });

    let first = cx.debug_bounds("evidence-path-step-0").unwrap();
    cx.simulate_click(first.center(), Modifiers::none());
    assert_eq!(
        view.read_with(&*cx, |view, _| view.selection),
        Some(SceneSelection::Edge(step_edges[0]))
    );
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    cx.simulate_keystrokes("tab");
    cx.simulate_keystrokes("enter");
    assert_eq!(
        view.read_with(&*cx, |view, _| view.selection),
        Some(SceneSelection::Edge(step_edges[1]))
    );
}

#[gpui::test]
fn a_found_path_copies_as_text(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("path-copy");
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let path = live.path.clone();
    let (view, cx) = cx.add_window_view(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    // Traced against the stored direction, so both steps run backwards.
    let command = format!("path {} {}", live.files[2], live.files[0]);
    view.update_in(cx, |view, window, cx| {
        view.settle_presentation_for_capture();
        view.execute_command(&command, window, cx);
    });
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let edges: Vec<u64> = view.read_with(&*cx, |view, _| {
        let PathState::Ready(report) = &view.path_state else {
            panic!("a path is traced, not {:?}", view.status);
        };
        let path = report.path.as_ref().unwrap();
        path.steps.iter().map(|step| step.edge_id).collect()
    });

    let copy = cx.debug_bounds("evidence-path-copy").unwrap();
    cx.simulate_click(copy.center(), Modifiers::none());
    let clipboard = cx.update(|_, cx| cx.read_from_clipboard());
    let [file0, file1, file2] = [live.files[0], live.files[1], live.files[2]];
    assert_eq!(
        clipboard.and_then(|item| item.text()).as_deref(),
        Some(
            format!(
                "Path from src/2.rs to src/0.rs: 2 hops\n\
                 1. src/2.rs (File, node:{file2})\n\
                 \x20  ← IMPORTS (relationship:{})\n\
                 2. src/1.rs (File, node:{file1})\n\
                 \x20  ← IMPORTS (relationship:{})\n\
                 3. src/0.rs (File, node:{file0})\n",
                edges[0], edges[1]
            )
            .as_str()
        )
    );
}

#[gpui::test]
fn live_refresh_applies_new_commits_and_waits_while_a_lens_is_active(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("live");
    let window = open_live_window(cx, &live.path);
    window
        .update(cx, |view, _, _| assert!(view.live_refresh))
        .unwrap();

    let first = live.add_session();
    cx.executor().advance_clock(LIVE_REFRESH_INTERVAL);
    cx.run_until_parked();
    window
        .update(cx, |view, _, _| {
            assert!(view.snapshot().unwrap().node_index(first).is_some());
            assert_eq!(view.status.as_ref(), "Live · +1 node · +1 relationship");
        })
        .unwrap();

    // An active lens holds per-element state, so the new commit waits.
    window
        .update(cx, |view, window, cx| {
            view.execute_command("facet node File", window, cx);
            view.settle_presentation_for_capture();
        })
        .unwrap();
    let second = live.add_session();
    cx.executor().advance_clock(LIVE_REFRESH_INTERVAL);
    cx.run_until_parked();
    window
        .update(cx, |view, window, cx| {
            assert!(view.reload_waiting);
            assert!(view.snapshot().unwrap().node_index(second).is_none());
            assert!(view.active_facet.is_some());
            view.escape(window, cx);
            view.settle_presentation_for_capture();
            assert!(view.lens.is_none());
        })
        .unwrap();
    cx.executor().advance_clock(LIVE_REFRESH_INTERVAL);
    cx.run_until_parked();
    window
        .update(cx, |view, window, cx| {
            assert!(!view.reload_waiting);
            assert!(view.snapshot().unwrap().node_index(second).is_some());

            view.execute_command("live off", window, cx);
            assert!(!view.live_refresh);
        })
        .unwrap();

    let third = live.add_session();
    cx.executor().advance_clock(LIVE_REFRESH_INTERVAL * 3);
    cx.run_until_parked();
    window
        .update(cx, |view, _, _| {
            assert!(view.snapshot().unwrap().node_index(third).is_none());
        })
        .unwrap();
}

#[test]
fn reload_status_names_every_kind_of_change() {
    assert_eq!(
        reload_status(SceneDelta::default(), false, true),
        "Live · up to date"
    );
    assert_eq!(
        reload_status(
            SceneDelta {
                added_nodes: 2,
                removed_nodes: 1,
                updated_nodes: 3,
                added_edges: 1,
                removed_edges: 0,
                updated_edges: 1,
            },
            true,
            false,
        ),
        "Reloaded · +2 nodes · −1 node · +1 relationship · 4 updated · new layout"
    );
}

#[gpui::test]
fn open_replaces_the_demo_scene_and_retires_its_presentation(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("open");
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));
    let missing = live.path.with_extension("missing.vg");

    window
        .update(cx, |view, window, cx| {
            assert!(!view.live_refresh);
            view.execute_command(&format!("open {}", missing.display()), window, cx);
            assert!(view.status.starts_with("No database at "));
            assert!(view.is_ready(), "a bad path keeps the current scene");

            view.execute_command("facet node Document", window, cx);
            view.set_selection(Some(SceneSelection::Node(3)));
            view.execute_command(&format!("open \"{}\"", live.path.display()), window, cx);
            assert!(!view.is_ready());
            assert!(view.active_facet.is_none());
            assert!(view.lens.is_none());
            assert_eq!(view.selection, None);
            assert!(
                view.live_refresh,
                "opening a database turns live refresh on"
            );
        })
        .unwrap();
    cx.run_until_parked();
    window
        .update(cx, |view, _, _| {
            assert!(view.is_ready());
            let snapshot = view.snapshot().unwrap();
            assert_eq!(snapshot.nodes.ids.len(), live.files.len());
            assert_eq!(view.database_path.as_deref(), Some(live.path.as_path()));
            assert!(view.fingerprint.is_some());
        })
        .unwrap();
}

#[gpui::test]
fn dropping_a_database_file_opens_it(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("drop");
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));
    window
        .update(cx, |view, window, cx| {
            let paths = ExternalPaths(
                [PathBuf::from("/tmp/notes.txt"), live.path.clone()]
                    .into_iter()
                    .collect(),
            );
            view.on_external_paths_dropped(&paths, window, cx);
        })
        .unwrap();
    cx.run_until_parked();
    window
        .update(cx, |view, _, _| {
            assert!(view.is_ready());
            assert_eq!(view.database_path.as_deref(), Some(live.path.as_path()));
        })
        .unwrap();
}

#[test]
fn command_words_followed_by_other_words_are_searched() {
    for command in [
        "fit",
        "back",
        "release",
        "zoom 2.5",
        "live",
        "live off",
        "layout orbit",
        "node 12",
        "focus 3",
        "edge 0",
        "path 1 2",
        "path 1 2 sideways",
        "path-start 4 in 2",
        "facet node File",
        "facet relationship READ",
        "open",
        "open ~/graphs/agents.vg",
        "open notes.vg",
        "open \"with space.vg\"",
        "open /tmp/missing",
        "export",
        "export ~/graph.svg",
        "export view.svg",
        "text graph",
        "semantic",
    ] {
        assert!(reads_as_command(command), "{command} is a command");
    }
    for search in [
        "",
        "memory leak",
        "fit test",
        "back pressure",
        "release notes",
        "zoom level",
        "zoom nan",
        "live refresh",
        "layout refresh",
        "node degree",
        "focus mode bug",
        "edge cases",
        "path rail keyboard focus",
        "facet lens accessibility",
        "open dialog for files",
        "export controls",
    ] {
        assert!(!reads_as_command(search), "{search} is a search");
    }
}

#[test]
fn the_command_box_names_what_enter_will_do() {
    assert_eq!(enter_hint(""), None);
    assert_eq!(enter_hint("   "), None);
    assert_eq!(enter_hint("memory leak"), Some("search"));
    assert_eq!(enter_hint("layout refresh"), Some("search"));
    assert_eq!(enter_hint("layout orbit"), Some("layout"));
    assert_eq!(enter_hint("node 12"), Some("node"));
    assert_eq!(enter_hint("relationship 4"), Some("edge"));
    assert_eq!(enter_hint("?"), Some("help"));
    assert_eq!(enter_hint("text graph"), Some("text search"));
    assert_eq!(enter_hint("hybrid graph"), Some("search"));
}

#[gpui::test]
fn the_command_box_hint_follows_its_text(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    let draw = |cx: &mut gpui::VisualTestContext| {
        cx.update(|window, cx| {
            _ = window.draw(cx);
        });
    };
    draw(cx);
    assert!(cx.debug_bounds("search-enter-hint").is_none());
    // Typing would focus the box, which test windows cannot draw on macOS,
    // so the text is set directly.
    view.update_in(cx, |view, window, cx| {
        view.query_input
            .update(cx, |input, cx| input.set_value("layout orbit", window, cx));
        cx.notify();
    });
    draw(cx);
    assert!(cx.debug_bounds("search-enter-hint").is_some());
}

#[test]
fn open_arguments_expand_home_strip_quotes_and_resolve_relative_paths() {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap());
    assert_eq!(
        resolve_open_argument("~/graphs/agents.vg"),
        Some(home.join("graphs/agents.vg"))
    );
    assert_eq!(
        resolve_open_argument("'/tmp/with space.vg'"),
        Some(PathBuf::from("/tmp/with space.vg"))
    );
    assert_eq!(
        resolve_open_argument("\"/tmp/a.vg\""),
        Some(PathBuf::from("/tmp/a.vg"))
    );
    assert_eq!(
        resolve_open_argument("relative.vg"),
        Some(std::env::current_dir().unwrap().join("relative.vg"))
    );
    assert_eq!(resolve_open_argument("  "), None);
    assert!(
        openable_database_path(&std::env::temp_dir())
            .unwrap_err()
            .contains("is a folder")
    );
}

#[gpui::test]
fn inspector_lists_connections_and_walks_the_graph(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    view.update(cx, |view, cx| {
        view.set_selection(Some(SceneSelection::Node(0)));
        cx.notify();
    });
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("inspector-connections").is_some());
    assert!(cx.debug_bounds("inspector-node-title").is_some());

    let (snapshot, connections) = view.read_with(&*cx, |view, _| {
        let snapshot = view.snapshot().unwrap().clone();
        let connections = view.inspector_connections(&snapshot, 0);
        (snapshot, connections)
    });
    // The rendered list and a fresh grouping agree, and a second lookup for
    // the same node and scene reuses the cached grouping.
    assert_eq!(*connections, snapshot.node_connections(0, 6));
    view.read_with(&*cx, |view, _| {
        assert!(Rc::ptr_eq(
            &connections,
            &view.inspector_connections(&snapshot, 0)
        ));
    });

    let next = connections
        .groups
        .iter()
        .find(|group| &*group.label == "NEXT" && group.outgoing)
        .unwrap()
        .connections[0];
    let connection_bounds = cx
        .debug_bounds("inspector-connections")
        .expect("connections render");
    assert!(connection_bounds.size.height > px(0.0));
    view.update_in(cx, |view, window, cx| {
        view.select_related_node(next.neighbor, Some(next.edge), window, cx);
        assert_eq!(view.selection, Some(SceneSelection::Node(next.neighbor)));
        assert!(view.camera_motion.is_moving());
        assert!(view.status.contains("via NEXT"));
        let zoom = view.camera.zoom;
        view.settle_presentation_for_capture();
        let target = view
            .workspace
            .as_ref()
            .unwrap()
            .borrow()
            .position(next.neighbor)
            .unwrap();
        assert!((view.camera.center - target).length() < 0.5);
        assert_eq!(view.camera.zoom, zoom, "walking keeps the zoom");
    });
}

#[gpui::test]
fn inspector_copies_element_ids(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    view.update(cx, |view, cx| {
        view.set_selection(Some(SceneSelection::Node(5)));
        cx.notify();
    });
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let id_bounds = cx.debug_bounds("inspector-node-title").unwrap();
    // The ID sits on the line above the title.
    cx.simulate_click(
        point(id_bounds.origin.x + px(12.0), id_bounds.origin.y - px(10.0)),
        Modifiers::none(),
    );
    cx.run_until_parked();
    let clipboard = cx.update(|_, cx| cx.read_from_clipboard());
    assert_eq!(
        clipboard.and_then(|item| item.text()).as_deref(),
        Some("node:5")
    );
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.status.as_ref(), "Copied node:5");
    });
}

#[gpui::test]
fn inspector_copies_property_values(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    view.update(cx, |view, cx| {
        view.set_selection(Some(SceneSelection::Node(5)));
        cx.notify();
    });
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let property = cx.debug_bounds("inspector-property:title").unwrap();
    cx.simulate_click(property.center(), Modifiers::none());
    cx.run_until_parked();
    let clipboard = cx.update(|_, cx| cx.read_from_clipboard());
    assert_eq!(
        clipboard.and_then(|item| item.text()).as_deref(),
        Some("Context element 5")
    );
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.status.as_ref(), "Copied title");
    });
}

#[gpui::test]
fn long_connection_groups_expand_and_collapse(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("expand");
    let mut transaction = live.database.transaction();
    let session = transaction.create_node(
        "Session",
        [("title", vecgra::Value::String("Wide session".into()))],
        &[],
    );
    for &file in &live.files {
        transaction.create_edge(
            session,
            file,
            "EDITED",
            std::iter::empty::<(&str, vecgra::Value)>(),
            &[],
        );
    }
    transaction.commit().unwrap();
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let path = live.path.clone();
    let (view, cx) = cx.add_window_view(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    view.update(cx, |view, cx| {
        assert!(view.is_ready(), "{}", view.status);
        let index = view
            .snapshot()
            .unwrap()
            .nodes
            .ids
            .iter()
            .position(|&id| id == session)
            .unwrap();
        view.set_selection(Some(SceneSelection::Node(index)));
        cx.notify();
    });
    let draw = |cx: &mut gpui::VisualTestContext| {
        cx.update(|window, cx| {
            _ = window.draw(cx);
        });
    };
    draw(cx);
    assert!(
        cx.debug_bounds("inspector-connections-fewer:EDITED:out")
            .is_none()
    );
    let more = cx
        .debug_bounds("inspector-connections-more:EDITED:out")
        .expect("eight edits list six and offer the rest");
    cx.simulate_click(more.center(), Modifiers::none());
    cx.run_until_parked();
    draw(cx);
    assert!(
        cx.debug_bounds("inspector-connections-more:EDITED:out")
            .is_none()
    );
    let fewer = cx
        .debug_bounds("inspector-connections-fewer:EDITED:out")
        .expect("an expanded group can collapse again");
    cx.simulate_click(fewer.center(), Modifiers::none());
    cx.run_until_parked();
    view.update(cx, |view, cx| {
        assert!(view.expanded_connections.is_empty());
        view.expanded_connections.push(("EDITED".into(), true));
        view.set_selection(None);
        assert!(
            view.expanded_connections.is_empty(),
            "a new selection starts collapsed"
        );
        cx.notify();
    });
}

#[gpui::test]
fn back_and_forward_retrace_selections_by_id(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let window = cx.add_window(|window, cx| StudioView::new(None, window, cx));
    window
        .update(cx, |view, window, cx| {
            view.execute_command("back", window, cx);
            assert_eq!(view.status.as_ref(), "No earlier selection");

            view.execute_command("node 3", window, cx);
            view.execute_command("edge 0", window, cx);
            view.select_related_node(9, None, window, cx);
            view.clear_selection(cx);
            assert_eq!(view.selection, None);

            view.execute_command("back", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Node(9)));
            view.execute_command("back", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Edge(0)));
            assert_eq!(
                view.status.as_ref(),
                "Back · NEXT · Context element 0 → Context element 1"
            );
            view.execute_command("back", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Node(3)));
            view.execute_command("back", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Node(3)));

            view.execute_command("forward", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Edge(0)));
            // Choosing something new discards the forward branch.
            view.execute_command("node 12", window, cx);
            view.execute_command("forward", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Node(12)));
            assert_eq!(view.status.as_ref(), "No later selection");
            view.execute_command("back", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Edge(0)));

            // Elements missing from the scene are skipped rather than
            // resolved to whatever now has that index.
            view.selection_back.push(SelectedElement::Node(999_999));
            view.execute_command("back", window, cx);
            assert_eq!(view.selection, Some(SceneSelection::Node(3)));
        })
        .unwrap();
}

#[gpui::test]
fn inspector_header_buttons_step_through_selection_history(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    view.update_in(cx, |view, window, cx| {
        view.execute_command("node 3", window, cx);
        view.execute_command("node 7", window, cx);
    });
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });

    let back = cx.debug_bounds("inspector-back").expect("back renders");
    cx.simulate_click(back.center(), Modifiers::none());
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.selection, Some(SceneSelection::Node(3)));
        assert!(view.selection_back.is_empty());
    });

    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    // With nothing earlier, the button stays inert rather than clearing.
    let back = cx.debug_bounds("inspector-back").expect("back renders");
    cx.simulate_click(back.center(), Modifiers::none());
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.selection, Some(SceneSelection::Node(3)));
    });

    let forward = cx
        .debug_bounds("inspector-forward")
        .expect("forward renders");
    cx.simulate_click(forward.center(), Modifiers::none());
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.selection, Some(SceneSelection::Node(7)));
    });
}

#[gpui::test]
fn help_lists_commands_and_starts_one_in_the_search_box(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("help-overlay").is_none());

    let toggle = cx
        .debug_bounds("help-toggle")
        .expect("status bar offers help");
    cx.simulate_click(toggle.center(), Modifiers::none());
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(view.read_with(&*cx, |view, _| view.help_visible));
    assert!(cx.debug_bounds("help-overlay").is_some());

    // Escape closes the help before it touches the selection.
    view.update_in(cx, |view, window, cx| {
        view.execute_command("node 3", window, cx);
        view.execute_command("help", window, cx);
        assert!(view.help_visible);
        view.escape(window, cx);
        assert!(!view.help_visible);
        assert_eq!(view.selection, Some(SceneSelection::Node(3)));
        view.set_help_visible(true, window, cx);
    });
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });

    assert!(cx.debug_bounds("help-command:path ").is_some());
    // Choosing a command focuses the search box, which a macOS test window
    // cannot render, so check it and hand focus back within one update
    // rather than clicking the row.
    view.update_in(cx, |view, window, cx| {
        view.begin_command("path ", window, cx);
        assert!(!view.help_visible);
        assert_eq!(view.query_input.read(cx).value().as_ref(), "path ");
        assert!(
            view.query_input
                .read(cx)
                .focus_handle(cx)
                .is_focused(window)
        );
        window.focus(&view.focus_handle, cx);
    });
}

#[gpui::test]
fn enter_dismisses_help_instead_of_activating_the_selection(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
        cx.bind_keys([KeyBinding::new(
            "enter",
            ActivateSelection,
            Some("VecgraStudio"),
        )]);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    view.update_in(cx, |view, window, cx| {
        view.execute_command("node 3", window, cx);
        view.execute_command("help", window, cx);
        assert!(view.help_visible);
    });
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    cx.simulate_keystrokes("enter");
    view.read_with(&*cx, |view, _| {
        assert!(!view.help_visible);
        assert!(!view.context_focus_active);
        assert_eq!(view.selection, Some(SceneSelection::Node(3)));
    });
}

#[gpui::test]
fn hovering_a_node_names_it_beside_the_pointer(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    view.update_in(cx, |view, _, _| view.settle_presentation_for_capture());
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let (point, id, empty) = view.read_with(&*cx, |view, _| {
        let bounds = view.canvas_bounds.unwrap();
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let workspace = view.workspace.as_ref().unwrap().borrow();
        let local =
            view.camera
                .project(workspace.position(4).unwrap(), view.world_bounds, viewport);
        let node = bounds.origin + point(px(local.x), px(local.y));
        // A corner of the canvas far from every node.
        let empty = bounds.origin + point(px(4.0), bounds.size.height - px(4.0));
        (node, view.snapshot().unwrap().nodes.ids[4], empty)
    });

    cx.simulate_mouse_move(point, None, Modifiers::none());
    view.read_with(&*cx, |view, _| assert_eq!(view.hovered_node, Some(id)));
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("graph-hover-card").is_some());

    cx.simulate_mouse_move(empty, None, Modifiers::none());
    view.read_with(&*cx, |view, _| assert_eq!(view.hovered_node, None));
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("graph-hover-card").is_none());
}

#[gpui::test]
fn hovers_end_when_the_view_moves_under_a_still_pointer(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    view.update_in(cx, |view, _, _| view.settle_presentation_for_capture());
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let (node_point, id) = view.read_with(&*cx, |view, _| {
        let bounds = view.canvas_bounds.unwrap();
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let workspace = view.workspace.as_ref().unwrap().borrow();
        let local =
            view.camera
                .project(workspace.position(4).unwrap(), view.world_bounds, viewport);
        (
            bounds.origin + point(px(local.x), px(local.y)),
            view.snapshot().unwrap().nodes.ids[4],
        )
    });
    cx.simulate_mouse_move(node_point, None, Modifiers::none());
    view.read_with(&*cx, |view, _| assert_eq!(view.hovered_node, Some(id)));

    // Wheel zoom keeps the point under the pointer, so the hover stays.
    let camera_before = view.read_with(&*cx, |view, _| view.camera);
    cx.simulate_event(ScrollWheelEvent {
        position: node_point,
        delta: ScrollDelta::Lines(point(0.0, 1.0)),
        ..Default::default()
    });
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    view.read_with(&*cx, |view, _| {
        assert_ne!(view.camera, camera_before);
        assert_eq!(view.hovered_node, Some(id));
    });
    assert!(cx.debug_bounds("graph-hover-card").is_some());

    // A command that moves the view leaves the pointer over something else.
    view.update_in(cx, |view, window, cx| {
        view.execute_command("zoom 3", window, cx)
    });
    view.update_in(cx, |view, _, _| view.settle_presentation_for_capture());
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    view.read_with(&*cx, |view, _| assert_eq!(view.hovered_node, None));
    assert!(cx.debug_bounds("graph-hover-card").is_none());
}

#[gpui::test]
fn hovering_a_relationship_names_its_type_and_ends(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    view.update_in(cx, |view, _, _| view.settle_presentation_for_capture());
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    // The middle of a relationship that is clear of every node.
    let (point, id, empty) = view.read_with(&*cx, |view, _| {
        let bounds = view.canvas_bounds.unwrap();
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let snapshot = view.snapshot().unwrap();
        let workspace = view.workspace.as_ref().unwrap().borrow();
        let positions = workspace.positions();
        let project = |index: u32| {
            view.camera
                .project(positions[index as usize], view.world_bounds, viewport)
        };
        let (edge, middle) = (0..snapshot.edges.ids.len())
            .map(|edge| {
                let middle = (project(snapshot.edges.sources[edge])
                    + project(snapshot.edges.targets[edge]))
                    * 0.5;
                (edge, middle)
            })
            .find(|&(edge, middle)| {
                hit_test_positions(
                    positions,
                    view.camera,
                    view.world_bounds,
                    viewport,
                    middle,
                    15.0,
                )
                .is_none()
                    && hit_test_edges(
                        positions,
                        &snapshot.edges,
                        view.camera,
                        view.world_bounds,
                        viewport,
                        middle,
                        7.0,
                    ) == Some(edge)
            })
            .expect("some relationship is clear of the nodes");
        let empty = bounds.origin + point(px(4.0), bounds.size.height - px(4.0));
        (
            bounds.origin + point(px(middle.x), px(middle.y)),
            snapshot.edges.ids[edge],
            empty,
        )
    });

    cx.simulate_mouse_move(point, None, Modifiers::none());
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.hovered_edge.map(|hovered| hovered.id), Some(id));
        assert_eq!(view.hovered_node, None);
    });
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("graph-hover-edge-card").is_some());
    assert!(cx.debug_bounds("graph-hover-ring").is_some());

    cx.simulate_mouse_move(empty, None, Modifiers::none());
    view.read_with(&*cx, |view, _| assert_eq!(view.hovered_edge, None));
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    assert!(cx.debug_bounds("graph-hover-edge-card").is_none());
}

#[gpui::test]
fn hover_cards_show_properties_other_than_the_title(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("hover-properties");
    let mut transaction = live.database.transaction();
    let session = transaction.create_node(
        "Session",
        [
            ("title", vecgra::Value::String("Tidy the inspector".into())),
            ("agent", vecgra::Value::String("claude".into())),
        ],
        &[],
    );
    transaction.commit().unwrap();
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let path = live.path.clone();
    let (view, cx) = cx.add_window_view(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    let draw = |cx: &mut gpui::VisualTestContext| {
        cx.update(|window, cx| {
            _ = window.draw(cx);
        });
    };
    view.update(cx, |view, cx| {
        assert!(view.is_ready(), "{}", view.status);
        view.settle_presentation_for_capture();
        view.hovered_node = Some(session);
        cx.notify();
    });
    draw(cx);
    assert!(cx.debug_bounds("graph-hover-card").is_some());
    assert!(
        cx.debug_bounds("graph-hover-properties").is_some(),
        "the session's agent is shown"
    );

    // A file's only property is the path that already titles it.
    view.update(cx, |view, cx| {
        view.hovered_node = Some(live.files[0]);
        cx.notify();
    });
    draw(cx);
    assert!(cx.debug_bounds("graph-hover-card").is_some());
    assert!(cx.debug_bounds("graph-hover-properties").is_none());
}

#[gpui::test]
fn hovering_an_inspector_connection_points_out_the_node(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    view.update_in(cx, |view, _, cx| {
        view.settle_presentation_for_capture();
        view.set_selection(Some(SceneSelection::Node(5)));
        cx.notify();
    });
    cx.run_until_parked();
    let draw = |cx: &mut gpui::VisualTestContext| {
        cx.update(|window, cx| {
            _ = window.draw(cx);
        });
    };
    draw(cx);
    let row = cx
        .debug_bounds("inspector-connection:6")
        .expect("node 5 lists its NEXT neighbor");
    let header = cx.debug_bounds("inspector-connections").unwrap();

    cx.simulate_mouse_move(row.center(), None, Modifiers::none());
    view.read_with(&*cx, |view, _| assert_eq!(view.hovered_node, Some(6)));
    draw(cx);
    assert!(cx.debug_bounds("graph-hover-ring").is_some());
    assert!(cx.debug_bounds("graph-hover-card").is_some());

    cx.simulate_mouse_move(
        header.origin + point(px(4.0), px(4.0)),
        None,
        Modifiers::none(),
    );
    view.read_with(&*cx, |view, _| assert_eq!(view.hovered_node, None));
    draw(cx);
    assert!(cx.debug_bounds("graph-hover-ring").is_none());

    cx.simulate_mouse_move(row.center(), None, Modifiers::none());
    cx.simulate_click(row.center(), Modifiers::none());
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.selection, Some(SceneSelection::Node(6)));
        // The new selection's rows replace the hovered one. Whatever row is
        // now under the pointer may preview its own node, but not this one.
        assert_ne!(
            view.hovered_node,
            Some(6),
            "walking to a node ends its preview"
        );
    });
}

#[gpui::test]
fn inspector_rows_can_be_reached_and_activated_from_the_keyboard(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::init_bezel(cx);
        crate::apply_studio_theme(cx);
        cx.bind_keys([KeyBinding::new(
            "enter",
            ActivateSelection,
            Some("VecgraStudio"),
        )]);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    view.update_in(cx, |view, _, cx| {
        view.settle_presentation_for_capture();
        view.set_selection(Some(SceneSelection::Node(5)));
        cx.notify();
    });
    cx.run_until_parked();
    let draw = |cx: &mut gpui::VisualTestContext| {
        cx.update(|window, cx| {
            _ = window.draw(cx);
        });
    };
    draw(cx);
    // Node 5 lists NEXT out to 6, then NEXT in from 4.
    let row = cx.debug_bounds("inspector-connection:6").unwrap();
    let header = cx.debug_bounds("inspector-connections").unwrap();
    // Pressing without releasing on the row focuses it without clicking it.
    cx.simulate_mouse_down(row.center(), MouseButton::Left, Modifiers::none());
    cx.simulate_mouse_up(
        header.origin + point(px(4.0), px(4.0)),
        MouseButton::Left,
        Modifiers::none(),
    );
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.selection, Some(SceneSelection::Node(5)))
    });
    draw(cx);

    cx.simulate_keystrokes("tab");
    draw(cx);
    // A focused control acts on Enter's release, which `simulate_keystrokes`
    // does not send.
    cx.simulate_keystrokes("enter");
    cx.simulate_event(gpui::KeyUpEvent {
        keystroke: gpui::Keystroke::parse("enter").unwrap(),
    });
    view.read_with(&*cx, |view, _| {
        assert_eq!(view.selection, Some(SceneSelection::Node(4)))
    });

    // A pointer click hands focus back to the graph, so Enter afterwards
    // opens the new selection's context rather than clicking a row again.
    draw(cx);
    let row = cx.debug_bounds("inspector-connection:5").unwrap();
    cx.simulate_click(row.center(), Modifiers::none());
    cx.update(|window, cx| {
        assert_eq!(view.read(cx).selection, Some(SceneSelection::Node(5)));
        assert!(view.read(cx).focus_handle.is_focused(window));
    });
}

#[gpui::test]
fn the_view_exports_as_an_svg_file(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    view.update_in(cx, |view, _, _| view.settle_presentation_for_capture());
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let directory = std::env::temp_dir().join(format!("vecgra-export-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    // A path without an extension is saved as SVG.
    let command = format!("export {}", directory.join("view").display());
    view.update_in(cx, |view, window, cx| {
        view.execute_command(&command, window, cx)
    });

    let svg = std::fs::read_to_string(directory.join("view.svg")).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<circle "));
    view.read_with(&*cx, |view, _| {
        assert!(
            view.status.starts_with("Exported the view to"),
            "{}",
            view.status
        )
    });
}

#[gpui::test]
fn opened_databases_are_offered_again_while_nothing_is_selected(cx: &mut TestAppContext) {
    let first = LiveDatabase::new("recent-first");
    let second = LiveDatabase::new("recent-second");
    let recent_file = std::env::temp_dir().join(format!(
        "vecgra-studio-ui-recent-{}/studio-recent",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&recent_file);
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    view.update_in(cx, |view, window, cx| {
        view.recent_file = Some(recent_file.clone());
        view.open_database(first.path.clone(), window, cx);
    });
    cx.run_until_parked();
    view.update_in(cx, |view, window, cx| {
        assert!(view.is_ready());
        assert_eq!(view.other_recent_databases().count(), 0);
        view.open_database(second.path.clone(), window, cx);
    });
    cx.run_until_parked();
    let first_path = std::fs::canonicalize(&first.path).unwrap();
    let second_path = std::fs::canonicalize(&second.path).unwrap();
    view.read_with(&*cx, |view, _| {
        assert_eq!(
            view.recent.paths(),
            [second_path.clone(), first_path.clone()]
        );
        assert_eq!(
            view.other_recent_databases().collect::<Vec<_>>(),
            [&first_path]
        );
    });
    assert_eq!(
        RecentDatabases::load(&recent_file).paths(),
        [second_path.clone(), first_path.clone()],
        "the list is saved for the next session"
    );

    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    let entry = cx
        .debug_bounds("inspector-recent:0")
        .expect("the other database is offered");
    cx.simulate_click(entry.center(), Modifiers::none());
    cx.run_until_parked();
    view.read_with(&*cx, |view, _| {
        assert!(view.is_ready());
        assert_eq!(view.database_path.as_deref(), Some(first_path.as_path()));
        assert_eq!(view.recent.paths()[0], first_path);
    });
    let _ = std::fs::remove_dir_all(recent_file.parent().unwrap());
}

#[gpui::test]
fn shortcuts_work_before_anything_is_clicked(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
        cx.bind_keys([KeyBinding::new("cmd-/", ToggleHelp, Some("VecgraStudio"))]);
    });
    let (view, cx) = cx.add_window_view(|window, cx| StudioView::new(None, window, cx));
    cx.run_until_parked();
    cx.update(|window, cx| {
        _ = window.draw(cx);
    });
    cx.simulate_keystrokes("cmd-/");
    view.read_with(&*cx, |view, _| assert!(view.help_visible));
}

#[gpui::test]
fn the_status_bar_names_what_was_selected(cx: &mut TestAppContext) {
    let live = LiveDatabase::new("status-names");
    let mut transaction = live.database.transaction();
    let marker =
        transaction.create_node("Marker", std::iter::empty::<(&str, vecgra::Value)>(), &[]);
    transaction.commit().unwrap();
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::apply_studio_theme(cx);
    });
    let path = live.path.clone();
    let (view, cx) = cx.add_window_view(move |window, cx| StudioView::new(Some(path), window, cx));
    cx.simulate_resize(size(px(1_400.0), px(900.0)));
    cx.run_until_parked();
    let (first, last) = (live.files[0], live.files[3]);
    view.update_in(cx, |view, window, cx| {
        let snapshot = view.snapshot().unwrap().clone();
        let source = snapshot.node_index(first).unwrap() as u32;
        let edge = (0..snapshot.edges.ids.len())
            .find(|&edge| snapshot.edges.sources[edge] == source)
            .map(|edge| snapshot.edges.ids[edge])
            .unwrap();

        // Files go by their file name, as on the canvas.
        view.execute_command(&format!("node {}", live.files[2]), window, cx);
        assert_eq!(view.status.as_ref(), "Selected 2.rs");
        // A node with nothing to call it by keeps its label and ID.
        view.execute_command(&format!("node {marker}"), window, cx);
        assert_eq!(view.status.as_ref(), format!("Selected Marker {marker}"));
        view.execute_command(&format!("edge {edge}"), window, cx);
        assert_eq!(view.status.as_ref(), "Selected IMPORTS · 0.rs → 1.rs");
        view.execute_command("back", window, cx);
        assert_eq!(view.status.as_ref(), format!("Back · Marker {marker}"));
        view.execute_command(&format!("center {last}"), window, cx);
        assert_eq!(view.status.as_ref(), "Centered 3.rs");
        view.execute_command(&format!("path {first} {last}"), window, cx);
        assert_eq!(view.status.as_ref(), "Tracing exact path · 0.rs → 3.rs…");
    });
}
