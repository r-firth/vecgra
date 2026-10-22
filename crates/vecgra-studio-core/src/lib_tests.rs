use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use vecgra::{DatabaseOptions, Similarity, VectorEncoding};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct TestFile(std::path::PathBuf);

impl TestFile {
    fn new() -> Self {
        let serial = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!("vecgra-studio-{}-{serial}.vg", std::process::id())))
    }
}

impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn snapshot_reads_real_database_and_preserves_symbols() {
    let file = TestFile::new();
    let database = Database::create(
        &file.0,
        DatabaseOptions {
            vector_dimension: 4,
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F32,
            sync_on_commit: false,
        },
    )
    .unwrap();
    let mut transaction = database.transaction();
    let first = transaction.create_node(
        "Document",
        [("title", Value::String("The first".into()))],
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    let second = transaction.create_node(
        "Claim",
        [("confidence", Value::Float(0.92))],
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    transaction.create_edge(
        first,
        second,
        "SUPPORTS",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 0.0, 1.0, 0.0]],
    );
    transaction.commit().unwrap();

    let snapshot = SceneSnapshot::from_database(&database, SnapshotOptions::default()).unwrap();
    assert_eq!(snapshot.nodes.ids, vec![first, second]);
    assert_eq!(&*snapshot.nodes.labels[0], "Document");
    assert_eq!(&*snapshot.edges.labels[0], "SUPPORTS");
    assert_eq!(snapshot.nodes.vector_counts, vec![1, 1]);
    assert_eq!(snapshot.edges.sources, vec![0]);
    assert_eq!(snapshot.edges.targets, vec![1]);
    assert_eq!(
        snapshot.relationship_counts(),
        vec![(Arc::from("SUPPORTS"), 1)]
    );
}

#[test]
fn evidence_paths_preserve_direction_hydration_and_incomplete_outcomes() {
    let file = TestFile::new();
    let database = Database::create(
        &file.0,
        DatabaseOptions {
            vector_dimension: 4,
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F32,
            sync_on_commit: false,
        },
    )
    .unwrap();
    let mut transaction = database.transaction();
    let alpha = transaction.create_node(
        "Document",
        [("title", Value::String("Alpha brief".into()))],
        &[],
    );
    let beta =
        transaction.create_node("Claim", [("name", Value::String("Beta claim".into()))], &[]);
    let gamma = transaction.create_node(
        "Source",
        [("title", Value::String("Gamma source".into()))],
        &[],
    );
    let first = transaction.create_edge(
        alpha,
        beta,
        "SUPPORTS",
        [("title", Value::String("Primary support".into()))],
        &[],
    );
    let second = transaction.create_edge(
        beta,
        gamma,
        "SUPPORTS",
        std::iter::empty::<(&str, Value)>(),
        &[],
    );
    transaction.commit().unwrap();

    let outgoing = evidence_path_database(
        &file.0,
        alpha,
        gamma,
        Direction::Outgoing,
        Some("SUPPORTS"),
        4,
        32,
    )
    .unwrap();
    assert_eq!(outgoing.termination, ShortestPathTermination::Found);
    assert_eq!(
        outgoing.strategy,
        ShortestPathStrategy::BidirectionalBreadthFirst
    );
    assert_eq!(
        outgoing.start_expanded_nodes + outgoing.end_expanded_nodes,
        outgoing.expanded_nodes
    );
    assert_eq!(outgoing.start.title.as_ref(), "Alpha brief");
    let path = outgoing.path.as_ref().unwrap();
    assert_eq!(
        path.nodes.iter().map(|node| node.id).collect::<Vec<_>>(),
        vec![alpha, beta, gamma]
    );
    assert_eq!(
        path.steps
            .iter()
            .map(|step| step.edge_id)
            .collect::<Vec<_>>(),
        vec![first, second]
    );
    assert_eq!(path.steps[0].title.as_ref(), "Primary support");
    assert!(path.steps.iter().all(|step| step.forward));
    assert_eq!(path.nodes[0].vector_count, 0);
    assert_eq!(path.steps[0].properties.len(), 1);

    let mut sampled = SceneSnapshot::from_database(
        &database,
        SnapshotOptions {
            max_nodes: 2,
            max_edges: 1,
        },
    )
    .unwrap();
    sampled.layout(LayoutOptions::default());
    assert!(sampled.node_index(gamma).is_none());
    assert!(sampled.edge_index(second).is_none());
    let overview_positions = sampled.nodes.positions.clone();
    let enriched = sampled
        .including_evidence_path(&outgoing, &overview_positions)
        .unwrap()
        .unwrap();
    assert_eq!(enriched.nodes.ids, vec![alpha, beta, gamma]);
    assert_eq!(enriched.edges.ids, vec![first, second]);
    assert_eq!(
        enriched.nodes.positions[..overview_positions.len()],
        overview_positions
    );
    let gamma_index = enriched.node_index(gamma).unwrap();
    assert_eq!(enriched.nodes.labels[gamma_index].as_ref(), "Source");
    assert_eq!(
        enriched.nodes.properties[gamma_index][0].value,
        Value::String("Gamma source".into())
    );
    let second_index = enriched.edge_index(second).unwrap();
    assert_eq!(
        enriched.nodes.ids[enriched.edges.sources[second_index] as usize],
        beta
    );
    assert_eq!(
        enriched.nodes.ids[enriched.edges.targets[second_index] as usize],
        gamma
    );
    assert!(!enriched.sampled);
    assert!(
        enriched
            .including_evidence_path(&outgoing, &enriched.nodes.positions)
            .unwrap()
            .is_none()
    );

    let incoming =
        evidence_path_database(&file.0, gamma, alpha, Direction::Incoming, None, 4, 32).unwrap();
    assert_eq!(incoming.termination, ShortestPathTermination::Found);
    assert!(
        incoming
            .path
            .unwrap()
            .steps
            .iter()
            .all(|step| !step.forward)
    );

    let limited =
        evidence_path_database(&file.0, alpha, gamma, Direction::Outgoing, None, 4, 0).unwrap();
    assert_eq!(limited.termination, ShortestPathTermination::ExpansionLimit);
    assert!(limited.path.is_none());
    assert_eq!(limited.start_expanded_nodes, 0);
    assert_eq!(limited.end_expanded_nodes, 0);
    assert_eq!(limited.expanded_nodes, 0);

    let error = evidence_path_database(
        &file.0,
        alpha,
        gamma,
        Direction::Outgoing,
        Some("MISSING"),
        4,
        32,
    )
    .unwrap_err();
    assert!(error.contains("relationship label \"MISSING\" does not exist"));
}

fn axis_vector(dimension: usize, axes: &[usize]) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    for &axis in axes {
        vector[axis] = 1.0 / (axes.len() as f32).sqrt();
    }
    vector
}

#[test]
fn hybrid_search_ranks_native_node_and_edge_vectors() {
    let file = TestFile::new();
    let dimension = 64;
    let database = Database::create(
        &file.0,
        DatabaseOptions {
            vector_dimension: dimension,
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F32,
            sync_on_commit: false,
        },
    )
    .unwrap();
    let mut transaction = database.transaction();
    let rust = transaction.create_node(
        "Document",
        [("title", Value::String("Rust memory safety".into()))],
        &[axis_vector(dimension, &[0])],
    );
    let fruit = transaction.create_node(
        "Document",
        [("title", Value::String("Banana bread recipe".into()))],
        &[axis_vector(dimension, &[1])],
    );
    let relationship = transaction.create_edge(
        rust,
        fruit,
        "CONTRASTS_WITH",
        [(
            "body",
            Value::String("ownership prevents use after free".into()),
        )],
        &[axis_vector(dimension, &[0, 2])],
    );
    transaction.commit().unwrap();

    let report = search_database_with(
        &file.0,
        "ownership memory safety",
        SearchMode::Hybrid,
        "test",
        8,
        |dimension, _| Ok(axis_vector(dimension, &[0])),
    )
    .unwrap();
    assert!(!report.hits.is_empty());
    assert!(
        report
            .hits
            .iter()
            .any(|hit| hit.element == ElementRef::Node(rust))
    );
    assert!(
        report
            .hits
            .iter()
            .any(|hit| hit.element == ElementRef::Edge(relationship))
    );
    assert!(report.hits[0].semantic_score.is_some());
    assert!(report.hits.iter().any(|hit| hit.lexical_score.is_some()));
    assert_eq!(report.warning, None);

    // Without an embedding, hybrid search falls back to text and says so
    // first, so the point survives a one-line display.
    let report = search_database_with(
        &file.0,
        "ownership memory safety",
        SearchMode::Hybrid,
        "test",
        8,
        |_, _| Err("model unavailable".into()),
    )
    .unwrap();
    assert!(!report.hits.is_empty());
    assert_eq!(
        report.warning.as_deref(),
        Some("Text matches only: semantic retrieval unavailable (model unavailable)")
    );
}

#[test]
fn layout_is_deterministic_and_finite() {
    let mut first = SceneSnapshot::demo();
    let mut second = SceneSnapshot::demo();
    first.layout(LayoutOptions::default());
    second.layout(LayoutOptions::default());
    assert_eq!(first.nodes.positions, second.nodes.positions);
    assert!(
        first
            .nodes
            .positions
            .iter()
            .all(|point| point.x.is_finite() && point.y.is_finite())
    );
}

#[test]
fn explicit_layouts_are_deterministic_and_finite() {
    let snapshot = SceneSnapshot::demo();
    for kind in [LayoutKind::Force, LayoutKind::Structure, LayoutKind::Orbit] {
        let first = snapshot.arranged_positions(kind, LayoutOptions::default(), None, None);
        let second = snapshot.arranged_positions(kind, LayoutOptions::default(), None, None);
        assert_eq!(first, second, "{kind:?} must be deterministic");
        assert!(
            first
                .iter()
                .all(|point| point.x.is_finite() && point.y.is_finite())
        );
    }
}

#[test]
fn sampled_t_force_is_deterministic_finite_and_not_axis_biased() {
    const NODE_COUNT: usize = 512;
    let mut edges = SceneEdges::default();
    for source in 0..NODE_COUNT {
        for target in [(source + 1) % NODE_COUNT, (source * 37 + 101) % NODE_COUNT] {
            if source == target {
                continue;
            }
            edges.ids.push(edges.ids.len() as EdgeId);
            edges.sources.push(source as u32);
            edges.targets.push(target as u32);
        }
    }
    let seed: Vec<_> = (0..NODE_COUNT)
        .map(|index| {
            let radius = 8.0 * (index as f32 + 1.0).sqrt();
            let angle = index as f32 * 2.399_963_1 + stable_unit(index as u64);
            Vec2::new(angle.cos(), angle.sin()) * radius
        })
        .collect();
    let mut first = seed.clone();
    let mut second = seed;
    let options = LayoutOptions {
        iterations: 32,
        ..LayoutOptions::default()
    };

    run_force_layout(&mut first, &edges, options, None);
    run_force_layout(&mut second, &edges, options, None);

    assert_eq!(first, second);
    assert!(
        first
            .iter()
            .all(|point| point.x.is_finite() && point.y.is_finite())
    );
    assert!(Rect::from_points(&first).width() > options.edge_length * 10.0);

    // The previous large-graph approximation accumulated nodes into a
    // Cartesian cell lattice. Nearest-neighbor bearings catch that visual
    // failure even when graph edges themselves point in arbitrary ways.
    let mut axis_aligned = 0_usize;
    for (index, &position) in first.iter().enumerate() {
        let nearest = first
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, &other)| other - position)
            .min_by(|left, right| left.length_squared().total_cmp(&right.length_squared()))
            .unwrap();
        let axis_ratio = nearest.x.abs().min(nearest.y.abs()) / nearest.length().max(0.000_1);
        axis_aligned += usize::from(axis_ratio < 3_f32.to_radians().sin());
    }
    assert!(
        axis_aligned * 5 < NODE_COUNT,
        "nearest-neighbor axes reveal a lattice: {axis_aligned}/{NODE_COUNT}"
    );
}

#[test]
fn workspace_drag_is_direct_pinned_and_degree_bounded() {
    let snapshot = SceneSnapshot::demo();
    let mut workspace = GraphWorkspace::new(&snapshot);
    let untouched = workspace.position(50).unwrap();
    let destination = Vec2::new(420.0, -180.0);

    assert!(!workspace.begin_drag(0));
    workspace.drag_to(0, destination, LayoutOptions::default().edge_length);

    assert_eq!(workspace.position(0), Some(destination));
    assert!(workspace.is_pinned(0));
    assert_eq!(workspace.position(50), Some(untouched));
    assert!(workspace.is_moving());
}

#[test]
fn workspace_position_revision_tracks_only_presented_coordinate_changes() {
    let snapshot = SceneSnapshot::demo();
    let mut workspace = GraphWorkspace::new(&snapshot);
    let initial_revision = workspace.position_revision();

    workspace.set_pinned(3, true);
    assert_eq!(workspace.position_revision(), initial_revision);

    workspace.drag_to(3, Vec2::new(120.0, -45.0), 48.0);
    let dragged_revision = workspace.position_revision();
    assert_ne!(dragged_revision, initial_revision);

    workspace.restore_pin(3, false);
    assert_eq!(workspace.position_revision(), dragged_revision);
    workspace.step(1.0 / 60.0, true);
    assert_ne!(workspace.position_revision(), dragged_revision);
}

#[test]
fn focus_neighborhood_retargets_and_restores_the_base_layout() {
    let snapshot = SceneSnapshot::demo();
    let selection = SceneSelection::Node(7);
    let focus = snapshot.focus_neighborhood(selection, 18).unwrap();
    assert_eq!(focus.roots, vec![7]);
    assert!(focus.edges.len() <= 18);
    assert!(focus.nodes.contains(&7));

    let mut workspace = GraphWorkspace::new(&snapshot);
    let original = workspace.positions().to_vec();
    let focus_bounds = workspace.retarget_focus(&focus, 46.0).unwrap();
    assert!(focus_bounds.width().is_finite());
    workspace.step(1.0 / 60.0, true);
    assert_ne!(workspace.positions(), original);

    workspace.restore_layout();
    workspace.step(1.0 / 60.0, true);
    assert_eq!(workspace.positions(), original);
}

#[test]
fn layered_focus_reserves_two_hop_context_and_relationship_diversity() {
    let node_count = 9;
    let edges = SceneEdges {
        ids: (0..8).collect(),
        sources: vec![0, 0, 0, 0, 1, 2, 3, 4],
        targets: vec![1, 2, 3, 4, 5, 6, 7, 8],
        labels: vec![
            "COMMON".into(),
            "COMMON".into(),
            "COMMON".into(),
            "RARE".into(),
            "CHILD".into(),
            "CHILD".into(),
            "CHILD".into(),
            "CHILD".into(),
        ],
        vector_counts: vec![0; 8],
        properties: vec![Arc::from([]); 8],
    };
    let snapshot = SceneSnapshot {
        revision: 1,
        database_name: "focus-test".into(),
        source_stats: GraphStats::default(),
        nodes: SceneNodes {
            ids: (0..node_count as u64).collect(),
            labels: vec!["Node".into(); node_count],
            positions: vec![Vec2::ZERO; node_count],
            degrees: vec![4, 2, 2, 2, 2, 1, 1, 1, 1],
            vector_counts: vec![0; node_count],
            properties: vec![Arc::from([]); node_count],
        },
        edges,
        bounds: Rect::default(),
        sampled: false,
    };

    let focus = snapshot
        .focus_neighborhood_layers(SceneSelection::Node(0), 2, 7, 10)
        .unwrap();
    assert_eq!(focus.layers.len(), 3);
    assert!(focus.nodes.len() <= 7);
    assert!(focus.edges.len() <= 10);
    assert!(focus.layers[1].contains(&4), "rare edge type must survive");
    assert!(
        focus.layers[1].iter().any(|node| (1..=3).contains(node)),
        "common edge type must survive"
    );
    assert!(!focus.layers[2].is_empty(), "capacity must reach hop two");
    assert!(
        focus
            .parents
            .iter()
            .all(|(child, parent)| { focus.nodes.contains(child) && focus.nodes.contains(parent) })
    );
}

#[test]
fn workspace_motion_converges_and_reduced_motion_snaps() {
    let snapshot = SceneSnapshot::demo();
    let mut workspace = GraphWorkspace::new(&snapshot);
    let mut targets = snapshot.nodes.positions.clone();
    targets[5] += Vec2::new(100.0, -80.0);
    assert!(workspace.retarget_layout(targets, snapshot.bounds));

    for _ in 0..600 {
        if !workspace.step(1.0 / 120.0, false) {
            break;
        }
    }
    assert!(!workspace.is_moving());
    assert!(
        workspace
            .positions()
            .iter()
            .all(|point| point.x.is_finite())
    );

    let mut targets = snapshot.nodes.positions.clone();
    targets[7] += Vec2::new(-140.0, 95.0);
    assert!(workspace.retarget_layout(targets, snapshot.bounds));
    assert!(!workspace.step(1.0 / 120.0, true));
    assert!(!workspace.is_moving());
}

#[test]
fn workspace_retarget_preserves_velocity_and_survives_long_frames() {
    let snapshot = SceneSnapshot::demo();
    let mut workspace = GraphWorkspace::new(&snapshot);
    let mut first_targets = snapshot.nodes.positions.clone();
    first_targets[9] += Vec2::new(160.0, 20.0);
    assert!(workspace.retarget_layout(first_targets, snapshot.bounds));
    workspace.step(1.0 / 60.0, false);
    let velocity = workspace.velocities[9];
    assert!(velocity.length_squared() > 0.0);

    let mut second_targets = snapshot.nodes.positions.clone();
    second_targets[9] += Vec2::new(-180.0, -30.0);
    assert!(workspace.retarget_layout(second_targets, snapshot.bounds));
    assert_eq!(workspace.velocities[9], velocity);
    workspace.step(8.0, false);
    assert!(
        workspace
            .positions()
            .iter()
            .all(|point| point.x.is_finite())
    );
    assert!(workspace.velocities.iter().all(|point| point.y.is_finite()));
}

#[test]
fn radial_forest_layout_preserves_depth_in_linear_pass() {
    let mut positions = vec![Vec2::ZERO; 4];
    let edges = SceneEdges {
        ids: vec![0, 1, 2],
        sources: vec![0, 0, 1],
        targets: vec![1, 2, 3],
        ..SceneEdges::default()
    };

    assert!(try_layout_radial_forest(&mut positions, &edges, 40.0));
    assert_eq!(positions[0], Vec2::ZERO);
    assert!((positions[1].length() - 80.0).abs() < 0.01);
    assert!((positions[2].length() - 80.0).abs() < 0.01);
    assert!((positions[3].length() - 120.0).abs() < 0.01);
    assert_ne!(positions[1], positions[2]);
}

#[test]
fn clustered_forest_layout_separates_top_level_subtrees() {
    let mut positions = vec![Vec2::ZERO; 13];
    let ids: Vec<NodeId> = (0..13).collect();
    let edges = SceneEdges {
        ids: (0..12).collect(),
        sources: vec![0, 0, 0, 1, 1, 1, 5, 5, 5, 9, 9, 9],
        targets: vec![1, 5, 9, 2, 3, 4, 6, 7, 8, 10, 11, 12],
        ..SceneEdges::default()
    };

    assert!(try_layout_clustered_forest(
        &mut positions,
        &ids,
        &edges,
        40.0
    ));
    assert_eq!(positions[0], Vec2::ZERO);
    for (left, right) in [(1, 5), (5, 9), (9, 1)] {
        assert!((positions[left] - positions[right]).length() > 100.0);
    }
    assert!(
        positions
            .iter()
            .all(|point| point.x.is_finite() && point.y.is_finite())
    );
}

#[test]
fn zoom_anchor_stays_under_pointer() {
    let bounds = Rect {
        min: Vec2::new(-100.0, -50.0),
        max: Vec2::new(100.0, 50.0),
    };
    let viewport = Vec2::new(1000.0, 700.0);
    let pointer = Vec2::new(700.0, 220.0);
    let mut camera = Camera::fit(bounds);
    let before = camera.unproject(pointer, bounds, viewport);
    camera.zoom_about(pointer, 1.8, bounds, viewport);
    let after = camera.unproject(pointer, bounds, viewport);
    assert!((before.x - after.x).abs() < 0.001);
    assert!((before.y - after.y).abs() < 0.001);
}

#[test]
fn deep_zoom_reaches_individual_elements_in_large_scenes() {
    let mut camera = Camera::default();
    camera.zoom_about(
        Vec2::new(400.0, 300.0),
        1_000_000.0,
        Rect {
            min: Vec2::new(-500.0, -500.0),
            max: Vec2::new(500.0, 500.0),
        },
        Vec2::new(800.0, 600.0),
    );
    assert_eq!(camera.zoom, MAX_CAMERA_ZOOM);
    assert_eq!(detail_level(camera, 50_000), DetailLevel::Elements);
    assert_eq!(
        detail_level(
            Camera {
                zoom: 4.0,
                ..camera
            },
            50_000
        ),
        DetailLevel::Communities
    );
}

#[test]
fn framed_camera_and_interruptible_motion_converge() {
    let world = Rect {
        min: Vec2::new(-500.0, -400.0),
        max: Vec2::new(500.0, 400.0),
    };
    let focus = Rect {
        min: Vec2::new(100.0, 80.0),
        max: Vec2::new(220.0, 180.0),
    };
    let viewport = Vec2::new(1_000.0, 700.0);
    let target = Camera::framed(focus, world, viewport, 100.0);
    assert_eq!(target.center, focus.center());
    assert!(target.zoom > 1.0);

    let mut camera = Camera::fit(world);
    let mut motion = CameraMotion::new(camera);
    motion.retarget(target);
    for _ in 0..240 {
        if !motion.step(&mut camera, 1.0 / 120.0, false) {
            break;
        }
    }
    assert!(!motion.is_moving());
    assert_eq!(camera, target);

    let next = Camera {
        center: Vec2::new(-140.0, 40.0),
        zoom: 2.4,
    };
    motion.retarget(next);
    motion.step(&mut camera, 1.0 / 120.0, false);
    let interrupted = camera;
    motion.retarget(target);
    motion.step(&mut camera, 1.0 / 120.0, false);
    assert_ne!(camera, target, "retargeting must not snap");
    assert_ne!(camera, interrupted, "retargeting must keep moving");
    motion.step(&mut camera, 1.0 / 120.0, true);
    assert_eq!(camera, target);
}

#[test]
fn edge_hit_testing_uses_screen_space_and_nearest_segment() {
    let positions = [
        Vec2::new(-100.0, -20.0),
        Vec2::new(100.0, -20.0),
        Vec2::new(-100.0, 20.0),
        Vec2::new(100.0, 20.0),
    ];
    let edges = SceneEdges {
        ids: vec![10, 11],
        sources: vec![0, 2],
        targets: vec![1, 3],
        labels: vec!["ABOVE".into(), "BELOW".into()],
        vector_counts: vec![0, 0],
        properties: vec![Arc::from([]), Arc::from([])],
    };
    let bounds = Rect::from_points(&positions);
    let viewport = Vec2::new(800.0, 400.0);
    let camera = Camera::fit(bounds);
    let on_second = camera.project(Vec2::new(0.0, 20.0), bounds, viewport);

    assert_eq!(
        hit_test_edges(&positions, &edges, camera, bounds, viewport, on_second, 6.0),
        Some(1)
    );
    assert_eq!(
        hit_test_edges(
            &positions,
            &edges,
            camera,
            bounds,
            viewport,
            on_second + Vec2::new(0.0, 9.0),
            6.0,
        ),
        None
    );
}

fn create_test_database(file: &TestFile) -> Database {
    Database::create(
        &file.0,
        DatabaseOptions {
            vector_dimension: 4,
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F32,
            sync_on_commit: false,
        },
    )
    .unwrap()
}

#[test]
fn fingerprint_notices_commits_from_another_handle() {
    let file = TestFile::new();
    let database = create_test_database(&file);
    let before = DatabaseFingerprint::read(&file.0).unwrap();
    assert_eq!(before, DatabaseFingerprint::read(&file.0).unwrap());

    let mut transaction = database.transaction();
    transaction.create_node("Session", std::iter::empty::<(&str, Value)>(), &[]);
    transaction.commit().unwrap();
    assert_ne!(before, DatabaseFingerprint::read(&file.0).unwrap());
}

#[test]
fn reload_keeps_surviving_positions_and_grows_new_nodes_from_their_anchors() {
    let file = TestFile::new();
    let database = create_test_database(&file);
    let mut transaction = database.transaction();
    let chain: Vec<_> = (0..6)
        .map(|index| {
            transaction.create_node(
                "File",
                [("path", Value::String(format!("src/{index}.rs").into()))],
                &[],
            )
        })
        .collect();
    for pair in chain.windows(2) {
        transaction.create_edge(
            pair[0],
            pair[1],
            "IMPORTS",
            std::iter::empty::<(&str, Value)>(),
            &[],
        );
    }
    transaction.commit().unwrap();

    let previous = SceneSnapshot::open(
        &file.0,
        SnapshotOptions::default(),
        LayoutOptions::default(),
    )
    .unwrap();
    // Simulate a user-adjusted presentation that differs from the layout.
    let presented: Vec<_> = previous
        .nodes
        .positions
        .iter()
        .map(|&position| position + Vec2::new(500.0, -250.0))
        .collect();

    let mut transaction = database.transaction();
    let session = transaction.create_node("Session", std::iter::empty::<(&str, Value)>(), &[]);
    let edited = transaction.create_edge(
        session,
        chain[2],
        "EDITED",
        std::iter::empty::<(&str, Value)>(),
        &[],
    );
    let island_a = transaction.create_node("Command", std::iter::empty::<(&str, Value)>(), &[]);
    let island_b =
        transaction.create_node("ErrorSignature", std::iter::empty::<(&str, Value)>(), &[]);
    transaction.create_edge(
        island_a,
        island_b,
        "HIT",
        std::iter::empty::<(&str, Value)>(),
        &[],
    );
    transaction.delete_node(chain[5], true);
    transaction
        .update_node(
            chain[0],
            "File",
            [("path", Value::String("src/renamed.rs".into()))],
            &[],
        )
        .unwrap();
    transaction.commit().unwrap();

    let reloaded = SceneSnapshot::reload(
        &file.0,
        &previous,
        &presented,
        SnapshotOptions::default(),
        LayoutOptions::default(),
    )
    .unwrap();
    assert!(!reloaded.relaid);
    assert_eq!(
        reloaded.delta,
        SceneDelta {
            added_nodes: 3,
            removed_nodes: 1,
            updated_nodes: 1,
            added_edges: 2,
            removed_edges: 1,
            updated_edges: 0,
        }
    );
    let snapshot = &reloaded.snapshot;
    for &id in &chain[..5] {
        let before = presented[previous.node_index(id).unwrap()];
        let after = snapshot.nodes.positions[snapshot.node_index(id).unwrap()];
        assert_eq!(before, after, "surviving node {id} must not move");
        assert_eq!(reloaded.entrances[snapshot.node_index(id).unwrap()], None);
    }
    assert!(snapshot.node_index(chain[5]).is_none());
    assert!(snapshot.edge_index(edited).is_some());

    let anchor = presented[previous.node_index(chain[2]).unwrap()];
    let session_index = snapshot.node_index(session).unwrap();
    assert_eq!(reloaded.entrances[session_index], Some(anchor));
    let offset = (snapshot.nodes.positions[session_index] - anchor).length();
    assert!(
        (LayoutOptions::default().edge_length * 0.9..=LayoutOptions::default().edge_length * 1.1)
            .contains(&offset),
        "new neighbor sits one edge length from its anchor, got {offset}"
    );

    // The unanchored component enters outside the surviving scene, and its
    // second node grows from the first rather than stacking on it.
    let surviving = Rect::from_points(
        &chain[..5]
            .iter()
            .map(|&id| snapshot.nodes.positions[snapshot.node_index(id).unwrap()])
            .collect::<Vec<_>>(),
    );
    let a = snapshot.nodes.positions[snapshot.node_index(island_a).unwrap()];
    let b = snapshot.nodes.positions[snapshot.node_index(island_b).unwrap()];
    let outside = |point: Vec2| {
        point.x < surviving.min.x
            || point.x > surviving.max.x
            || point.y < surviving.min.y
            || point.y > surviving.max.y
    };
    assert!(
        outside(a),
        "island seed {a:?} must be outside {surviving:?}"
    );
    assert!((a - b).length() > 1.0);
    assert!(
        snapshot
            .nodes
            .positions
            .iter()
            .all(|point| point.x.is_finite() && point.y.is_finite())
    );
}

#[test]
fn reload_lays_out_from_scratch_when_most_of_the_scene_is_new() {
    let previous = SceneSnapshot::demo();
    let mut grown = previous.clone();
    let base = grown.nodes.ids.len() as u64;
    for offset in 0..grown.nodes.ids.len() as u64 * 2 {
        grown.nodes.ids.push(base + offset);
        grown.nodes.labels.push("Fresh".into());
        grown.nodes.positions.push(Vec2::ZERO);
        grown.nodes.degrees.push(0);
        grown.nodes.vector_counts.push(0);
        grown.nodes.properties.push(Arc::from([]));
    }
    let reloaded = grown.anchored_on(
        &previous,
        &previous.nodes.positions,
        LayoutOptions::default(),
    );
    assert!(reloaded.relaid);
    assert!(reloaded.entrances.iter().all(Option::is_none));
    assert_eq!(reloaded.delta.added_nodes, previous.nodes.ids.len() * 2);

    // A mismatched presentation is never trusted for anchoring.
    let unchanged =
        previous
            .clone()
            .anchored_on(&previous, &[Vec2::ZERO], LayoutOptions::default());
    assert!(unchanged.relaid);
    assert!(unchanged.delta.is_empty());
}

#[test]
fn workspace_entrances_spring_new_nodes_into_place() {
    let snapshot = SceneSnapshot::demo();
    let mut workspace = GraphWorkspace::new(&snapshot);
    let target = workspace.positions()[3];
    let origin = workspace.positions()[0];
    let mut entrances = vec![None; snapshot.nodes.ids.len()];
    entrances[3] = Some(origin);
    workspace.set_pinned(5, true);
    entrances[5] = Some(origin);
    workspace.enter_from(&entrances);
    assert_eq!(workspace.positions()[3], origin);
    assert_ne!(workspace.positions()[5], origin, "pinned nodes stay put");
    assert!(workspace.is_moving());
    for _ in 0..240 {
        workspace.step(1.0 / 60.0, false);
    }
    assert!((workspace.positions()[3] - target).length() < 0.05);
}

#[test]
fn node_connections_group_by_type_and_direction_with_full_counts() {
    let mut snapshot = SceneSnapshot::demo();
    // Node 0 gets a self-loop on top of the demo's NEXT and SUPPORTS edges.
    snapshot.edges.ids.push(10_000);
    snapshot.edges.sources.push(0);
    snapshot.edges.targets.push(0);
    snapshot.edges.labels.push("MENTIONS".into());
    snapshot.edges.vector_counts.push(0);
    snapshot.edges.properties.push(Arc::from([]));

    let connections = snapshot.node_connections(0, 1);
    let incident = snapshot
        .edges
        .sources
        .iter()
        .zip(&snapshot.edges.targets)
        .filter(|&(&source, &target)| source == 0 || target == 0)
        .count();
    assert_eq!(connections.total, incident);
    assert_eq!(
        connections
            .groups
            .iter()
            .map(|group| group.total)
            .sum::<usize>(),
        incident
    );
    assert!(
        connections
            .groups
            .windows(2)
            .all(|pair| pair[0].total >= pair[1].total)
    );
    assert!(
        connections
            .groups
            .iter()
            .all(|group| group.connections.len() == 1)
    );
    let next_out = connections
        .groups
        .iter()
        .find(|group| &*group.label == "NEXT" && group.outgoing)
        .unwrap();
    let edge = next_out.connections[0].edge;
    assert_eq!(snapshot.edges.sources[edge], 0);
    assert_eq!(next_out.connections[0].neighbor, 1);
    let self_loop = connections
        .groups
        .iter()
        .find(|group| &*group.label == "MENTIONS")
        .unwrap();
    assert!(self_loop.outgoing);
    assert_eq!(self_loop.total, 1);
    assert_eq!(self_loop.connections[0].neighbor, 0);

    assert_eq!(
        snapshot.node_connections(usize::MAX, 4),
        NodeConnections::default()
    );
}

#[test]
fn reload_places_a_node_with_several_anchors_between_them() {
    fn scene(node_count: u64, edges: &[(u32, u32)]) -> SceneSnapshot {
        let mut snapshot = SceneSnapshot::demo();
        snapshot.nodes = SceneNodes {
            ids: (0..node_count).collect(),
            labels: vec!["Node".into(); node_count as usize],
            positions: vec![Vec2::ZERO; node_count as usize],
            degrees: vec![0; node_count as usize],
            vector_counts: vec![0; node_count as usize],
            properties: vec![Arc::from([]); node_count as usize],
        };
        snapshot.edges = SceneEdges {
            ids: (0..edges.len() as u64).collect(),
            sources: edges.iter().map(|&(source, _)| source).collect(),
            targets: edges.iter().map(|&(_, target)| target).collect(),
            labels: vec!["LINKS".into(); edges.len()],
            vector_counts: vec![0; edges.len()],
            properties: vec![Arc::from([]); edges.len()],
        };
        snapshot
    }
    let previous = scene(3, &[]);
    let presented = [
        Vec2::new(-200.0, 0.0),
        Vec2::new(200.0, 0.0),
        Vec2::new(0.0, 300.0),
    ];
    let reloaded =
        scene(4, &[(3, 0), (3, 1)]).anchored_on(&previous, &presented, LayoutOptions::default());
    assert!(!reloaded.relaid);
    let placed = reloaded.snapshot.nodes.positions[3];
    assert!(
        placed.length() <= LayoutOptions::default().edge_length * 0.5 + 0.01,
        "{placed:?} should sit near the anchors' centroid"
    );
    let entrance = reloaded.entrances[3].unwrap();
    assert!(entrance == presented[0] || entrance == presented[1]);
}

#[test]
fn element_titles_prefer_naming_keys_then_first_lines() {
    let property = |key: &str, value: &str| SceneProperty {
        key: key.into(),
        value: Value::String(value.into()),
    };
    // Priority follows the key list, not property order.
    assert_eq!(
        element_title(&[property("path", "src/lib.rs"), property("name", "lib")]).as_deref(),
        Some("lib")
    );
    assert_eq!(
        element_title(&[property("command", "cargo test -p vecgra")]).as_deref(),
        Some("cargo test -p vecgra")
    );
    assert_eq!(
        element_title(&[
            property("sha", "abc123"),
            property("message", "\n  Fix the reload race\n\nLonger body"),
        ])
        .as_deref(),
        Some("Fix the reload race")
    );
    assert_eq!(element_title(&[property("title", "   ")]), None);
    let long = "x".repeat(200);
    let title = element_title(&[property("title", &long)]).unwrap();
    assert_eq!(title.chars().count(), 120);
    assert!(title.ends_with('…'));

    let mut snapshot = SceneSnapshot::demo();
    assert_eq!(&*snapshot.node_title(0), "Context element 0");
    snapshot.nodes.properties[0] = Arc::from([]);
    assert_eq!(snapshot.node_title(0), snapshot.nodes.labels[0].clone());
}

#[test]
fn search_and_paths_name_message_only_nodes_by_their_first_line() {
    let file = TestFile::new();
    let database = create_test_database(&file);
    let mut transaction = database.transaction();
    let source = transaction.create_node(
        "File",
        [("path", Value::String("src/reload.rs".into()))],
        &[],
    );
    let commit = transaction.create_node(
        "Commit",
        [
            ("sha", Value::String("abc123".into())),
            (
                "message",
                Value::String("\n  Fix the reload race\n\nLonger body".into()),
            ),
        ],
        &[],
    );
    transaction.create_edge(
        commit,
        source,
        "CHANGED",
        std::iter::empty::<(&str, Value)>(),
        &[],
    );
    transaction.commit().unwrap();

    let report = search_database_with(
        &file.0,
        "reload race",
        SearchMode::Text,
        "test",
        8,
        |dimension, _| Ok(vec![0.0; dimension]),
    )
    .unwrap();
    let hit = report
        .hits
        .iter()
        .find(|hit| hit.element == ElementRef::Node(commit))
        .unwrap();
    assert_eq!(hit.title.as_ref(), "Fix the reload race");
    assert_eq!(hit.detail.as_ref(), "Fix the reload race Longer body");

    let report =
        evidence_path_database(&file.0, source, commit, Direction::Both, None, 2, 32).unwrap();
    assert_eq!(report.end.title.as_ref(), "Fix the reload race");
    let path = report.path.unwrap();
    assert_eq!(path.nodes[1].title.as_ref(), "Fix the reload race");
}

#[test]
fn relationship_results_without_a_name_are_titled_by_their_ends() {
    let file = TestFile::new();
    let database = create_test_database(&file);
    let mut transaction = database.transaction();
    let session = transaction.create_node(
        "Session",
        [("title", Value::String("Fix flaky layout test".into()))],
        &[],
    );
    let source = transaction.create_node(
        "File",
        [("path", Value::String("src/layout.rs".into()))],
        &[],
    );
    let edited = transaction.create_edge(
        session,
        source,
        "EDITED",
        [("lines", Value::Int(4242))],
        &[],
    );
    let named = transaction.create_edge(
        session,
        source,
        "CITES",
        [("title", Value::String("Layout regression 4242".into()))],
        &[],
    );
    transaction.commit().unwrap();

    let report = search_database_with(
        &file.0,
        "4242",
        SearchMode::Text,
        "test",
        8,
        |dimension, _| Ok(vec![0.0; dimension]),
    )
    .unwrap();
    let title = |id| {
        report
            .hits
            .iter()
            .find(|hit| hit.element == ElementRef::Edge(id))
            .map(|hit| (hit.title.to_string(), hit.detail.to_string()))
    };
    assert_eq!(
        title(edited),
        Some((
            "Fix flaky layout test → layout.rs".to_string(),
            format!("node:{session} → node:{source}")
        ))
    );
    assert_eq!(
        title(named).map(|(title, _)| title).as_deref(),
        Some("Layout regression 4242")
    );
}

#[test]
fn results_that_share_a_name_are_told_apart_by_short_properties() {
    let file = TestFile::new();
    let database = create_test_database(&file);
    let mut transaction = database.transaction();
    let mut session = |agent: &str, model: &str| {
        transaction.create_node(
            "Session",
            [
                ("agent", Value::String(agent.into())),
                ("model", Value::String(model.into())),
                ("title", Value::String("Fix flaky layout test".into())),
            ],
            &[],
        )
    };
    let codex = session("codex", "gpt-5.6");
    let copilot = session("copilot", "gpt-5.6-mini");
    let commit = transaction.create_node(
        "Commit",
        [
            ("message", Value::String("Fix flaky layout test".into())),
            ("sha", Value::String("758dd63cb9".into())),
        ],
        &[],
    );
    // Dates and long text say little at a glance; the ID stays.
    let issue = transaction.create_node(
        "Issue",
        [
            ("title", Value::String("Flaky layout test".into())),
            ("created_at", Value::String("2026-10-19T14:03:00Z".into())),
            ("notes", Value::String("x".repeat(80).into())),
        ],
        &[],
    );
    transaction.commit().unwrap();

    let report = search_database_with(
        &file.0,
        "flaky",
        SearchMode::Text,
        "test",
        8,
        |dimension, _| Ok(vec![0.0; dimension]),
    )
    .unwrap();
    let detail = |id| {
        report
            .hits
            .iter()
            .find(|hit| hit.element == ElementRef::Node(id))
            .map(|hit| hit.detail.to_string())
    };
    assert_eq!(detail(codex).as_deref(), Some("codex · gpt-5.6"));
    assert_eq!(detail(copilot).as_deref(), Some("copilot · gpt-5.6-mini"));
    assert_eq!(detail(commit).as_deref(), Some("758dd63cb9"));
    assert_eq!(detail(issue), Some(format!("node:{issue}")));
}

#[test]
fn path_titles_split_into_name_and_directory() {
    assert_eq!(
        split_path_title("crates/vecgra/src/error.rs"),
        Some(("error.rs", "crates/vecgra/src"))
    );
    assert_eq!(
        split_path_title("crates/vecgra/src/"),
        Some(("src", "crates/vecgra"))
    );
    assert_eq!(
        split_path_title(r"C:\repo\main.rs"),
        Some(("main.rs", r"C:\repo"))
    );
    assert_eq!(split_path_title("/etc"), None);
    assert_eq!(split_path_title("README.md"), None);
    assert_eq!(split_path_title("Fix A/B test flake"), None);
    assert_eq!(split_path_title("https://example.com/a/b"), None);
}

#[test]
fn recent_databases_keep_the_newest_existing_files_once() {
    let directory = std::env::temp_dir().join(format!("vecgra-recent-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let file = directory.join("nested").join("studio-recent");
    assert_eq!(RecentDatabases::load(&file), RecentDatabases::default());

    std::fs::create_dir_all(&directory).unwrap();
    let databases: Vec<_> = (0..RecentDatabases::LIMIT + 2)
        .map(|index| {
            let path = directory.join(format!("graph-{index}.vg"));
            std::fs::write(&path, b"").unwrap();
            path
        })
        .collect();
    let mut recent = RecentDatabases::default();
    for path in &databases {
        recent.remember(path.clone());
    }
    recent.remember(databases[RecentDatabases::LIMIT].clone());
    recent.remember(directory.join("bad\nname.vg"));
    assert_eq!(recent.paths().len(), RecentDatabases::LIMIT);
    assert_eq!(recent.paths()[0], databases[RecentDatabases::LIMIT]);
    assert_eq!(recent.paths()[1], databases[RecentDatabases::LIMIT + 1]);
    assert!(!recent.paths().contains(&databases[0]));

    recent.save(&file).unwrap();
    std::fs::remove_file(&databases[RecentDatabases::LIMIT + 1]).unwrap();
    let loaded = RecentDatabases::load(&file);
    assert_eq!(loaded.paths().len(), RecentDatabases::LIMIT - 1);
    assert_eq!(loaded.paths()[0], databases[RecentDatabases::LIMIT]);
    assert!(
        !loaded
            .paths()
            .contains(&databases[RecentDatabases::LIMIT + 1]),
        "a database deleted since it was saved is dropped"
    );

    let mut forgotten = loaded.clone();
    forgotten.forget(&databases[RecentDatabases::LIMIT]);
    assert_eq!(forgotten.paths().len(), RecentDatabases::LIMIT - 2);
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn snapshots_find_every_relationship_after_a_checkpointed_database_changes() {
    let original = TestFile::new();
    let compacted = TestFile::new();
    let database = Database::create(
        &original.0,
        DatabaseOptions {
            vector_dimension: 4,
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F32,
            sync_on_commit: false,
        },
    )
    .unwrap();
    let mut transaction = database.transaction();
    let nodes: Vec<_> = (0..4)
        .map(|_| transaction.create_node("Item", std::iter::empty::<(&str, Value)>(), &[]))
        .collect();
    let edges: Vec<_> = nodes
        .windows(2)
        .map(|pair| {
            transaction.create_edge(
                pair[0],
                pair[1],
                "NEXT",
                std::iter::empty::<(&str, Value)>(),
                &[],
            )
        })
        .collect();
    transaction.commit().unwrap();
    database
        .compact_to(&compacted.0, VectorEncoding::F32)
        .unwrap();
    drop(database);

    // Changing the first relationship of a compacted file moves it after the
    // rest in the database's own listing.
    let database = Database::open(&compacted.0).unwrap();
    let mut transaction = database.transaction();
    transaction
        .update_edge(
            edges[0],
            nodes[0],
            nodes[1],
            "NEXT",
            [("weight", Value::Int(2))],
            &[],
        )
        .unwrap();
    transaction.commit().unwrap();

    let snapshot = SceneSnapshot::from_database(&database, SnapshotOptions::default()).unwrap();
    assert!(snapshot.edges.ids.is_sorted());
    for &edge in &edges {
        let index = snapshot
            .edge_index(edge)
            .expect("every relationship is found by ID");
        assert_eq!(snapshot.edges.ids[index], edge);
    }
    assert_eq!(snapshot.relationship_counts(), vec![(Arc::from("NEXT"), 3)]);
}

#[test]
fn command_history_keeps_the_newest_entries_across_sessions() {
    let directory =
        std::env::temp_dir().join(format!("vecgra-studio-history-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let file = directory.join("nested").join("studio-history");
    assert_eq!(CommandHistory::load(&file), CommandHistory::default());

    let mut history = CommandHistory::default();
    assert!(history.remember("  path 0 54 out "));
    assert!(!history.remember("path 0 54 out"), "a repeat is kept once");
    assert!(!history.remember("   "));
    assert!(!history.remember("two\nlines"));
    assert!(history.remember("codec"));
    assert!(history.remember("path 0 54 out"));
    assert_eq!(
        history.entries(),
        ["path 0 54 out", "codec", "path 0 54 out"]
    );
    history.save(&file).unwrap();
    assert_eq!(CommandHistory::load(&file), history);

    for index in 0..CommandHistory::LIMIT {
        history.remember(&format!("node {index}"));
    }
    assert_eq!(history.entries().len(), CommandHistory::LIMIT);
    assert_eq!(history.entries()[0], "node 0");
    history.save(&file).unwrap();
    assert_eq!(CommandHistory::load(&file), history);
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn date_properties_read_as_ages() {
    let then = 1_792_620_180_000; // 2026-10-21T22:03:00Z
    assert_eq!(parse_iso_datetime_ms("2026-10-21T22:03:00Z"), Some(then));
    assert_eq!(
        parse_iso_datetime_ms("2026-10-21T23:03:00.25+01:00"),
        Some(then + 250)
    );
    assert_eq!(
        parse_iso_datetime_ms("2024-02-29T12:00:00Z"),
        Some(1_709_208_000_000)
    );
    for text in [
        "2026-02-30T00:00:00Z",
        "2026-10-21",
        "2026-10-21T22:03:00",
        "2026-10-21T22:03:00+0100",
        "a release on 2026-10-21T22:03:00Z",
        "not a date",
    ] {
        assert_eq!(parse_iso_datetime_ms(text), None, "{text}");
    }

    let minute = 60_000;
    let day = 24 * 60 * minute;
    assert_eq!(relative_age(then - 30_000, then), "just now");
    assert_eq!(relative_age(then - 5 * minute, then), "5 minutes ago");
    assert_eq!(relative_age(then - 60 * minute, then), "1 hour ago");
    assert_eq!(relative_age(then - 3 * day, then), "3 days ago");
    assert_eq!(relative_age(then - 15 * day, then), "2 weeks ago");
    assert_eq!(relative_age(then - 400 * day, then), "1 year ago");
    assert_eq!(relative_age(then + 2 * day, then), "in 2 days");

    let now = then + day + 5 * minute;
    assert_eq!(
        property_time_hint(
            "created_at",
            &PropertyValue::String("2026-10-21T22:03:00Z".into()),
            now
        )
        .as_deref(),
        Some("1 day ago")
    );
    assert_eq!(
        property_time_hint("created_at_ms", &PropertyValue::Int(then), now).as_deref(),
        Some("2026-10-21 22:03 UTC · 1 day ago")
    );
    // Numbers are dates only when named as milliseconds and in range.
    assert_eq!(
        property_time_hint("count", &PropertyValue::Int(then), now),
        None
    );
    assert_eq!(
        property_time_hint("size_ms", &PropertyValue::Int(1_500), now),
        None
    );
    assert_eq!(
        property_time_hint(
            "title",
            &PropertyValue::String("Fix flaky test".into()),
            now
        ),
        None
    );
}

#[test]
fn hidden_nodes_leave_the_scene_and_come_back_beside_their_neighbors() {
    let mut full = SceneSnapshot::demo();
    full.layout(LayoutOptions::default());
    // The best-connected node, so hiding it removes many relationships.
    let hub = (0..full.nodes.ids.len())
        .max_by_key(|&index| full.nodes.degrees[index])
        .unwrap();
    let hub_id = full.nodes.ids[hub];
    let touching = (0..full.edges.ids.len())
        .filter(|&edge| {
            full.edges.sources[edge] as usize == hub || full.edges.targets[edge] as usize == hub
        })
        .count();
    let hidden = HiddenNodes {
        ids: [hub_id].into(),
        ..HiddenNodes::default()
    };

    let shown = full.without_nodes(&hidden);
    assert_eq!(shown.nodes.ids.len(), full.nodes.ids.len() - 1);
    assert_eq!(shown.edges.ids.len(), full.edges.ids.len() - touching);
    assert_eq!(shown.node_index(hub_id), None);
    for edge in 0..shown.edges.ids.len() {
        let original = full.edge_index(shown.edges.ids[edge]).unwrap();
        for (now, then) in [
            (shown.edges.sources[edge], full.edges.sources[original]),
            (shown.edges.targets[edge], full.edges.targets[original]),
        ] {
            assert_eq!(shown.nodes.ids[now as usize], full.nodes.ids[then as usize]);
        }
    }
    for (index, &id) in shown.nodes.ids.iter().enumerate() {
        let original = full.node_index(id).unwrap();
        assert_eq!(shown.nodes.positions[index], full.nodes.positions[original]);
    }

    // Showing it again anchors the full scene on the one in view: everything
    // else stays put and the hub enters from a neighbor.
    let restored = full.anchored_on(&shown, &shown.nodes.positions, LayoutOptions::default());
    assert!(!restored.relaid);
    assert_eq!(restored.delta.added_nodes, 1);
    assert_eq!(restored.delta.added_edges, touching);
    assert_eq!(restored.delta.removed_nodes, 0);
    assert!(restored.entrances[hub].is_some());
    for (index, &id) in shown.nodes.ids.iter().enumerate() {
        let back = restored.snapshot.node_index(id).unwrap();
        assert_eq!(
            restored.snapshot.nodes.positions[back],
            shown.nodes.positions[index]
        );
    }
}

#[test]
fn a_hidden_label_hides_every_node_with_it() {
    let mut full = SceneSnapshot::demo();
    full.layout(LayoutOptions::default());
    let label = full.nodes.labels[0].clone();
    let with_label = full
        .nodes
        .labels
        .iter()
        .filter(|other| **other == label)
        .count();
    let hidden = HiddenNodes {
        labels: [label.clone()].into(),
        ..HiddenNodes::default()
    };
    assert!(hidden.hides(full.nodes.ids[0], &label));
    let shown = full.without_nodes(&hidden);
    assert_eq!(shown.nodes.ids.len(), full.nodes.ids.len() - with_label);
    assert!(shown.nodes.labels.iter().all(|other| *other != label));
    assert!(HiddenNodes::default().is_empty() && !hidden.is_empty());
}
