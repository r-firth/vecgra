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
    assert!(error.contains("relationship label “MISSING” does not exist"));
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
        &[vecgra_embedding::feature_vector(
            "ownership makes systems programming memory safe",
            dimension,
        )],
    );
    let fruit = transaction.create_node(
        "Document",
        [("title", Value::String("Banana bread recipe".into()))],
        &[vecgra_embedding::feature_vector(
            "fruit flour baking recipe",
            dimension,
        )],
    );
    let relationship = transaction.create_edge(
        rust,
        fruit,
        "CONTRASTS_WITH",
        [(
            "body",
            Value::String("ownership prevents use after free".into()),
        )],
        &[vecgra_embedding::feature_vector(
            "ownership prevents memory bugs",
            dimension,
        )],
    );
    transaction.commit().unwrap();

    let report = search_database(
        &file.0,
        "ownership memory safety",
        SearchMode::Hybrid,
        "hash",
        8,
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
