use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Bound;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use vectorgraph::{
    BulkLoader, Database, DatabaseOptions, Direction, EdgeFilter, ElementFilter,
    ElementFilterStrategy, ElementRef, ElementSet, GraphRangeSearchOptions, NodeFilter,
    NumericRangeFilter, NumericRangeStrategy, NumericValue, OneHopQuery, SemanticOneHopQuery,
    SemanticPathOptions, Similarity, Value, VectorEncoding, VectorSearchStrategy, VectorTarget,
};

fn temp_database(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "vectorgraph-{name}-{}-{nonce}.vg",
        std::process::id()
    ))
}

fn options() -> DatabaseOptions {
    DatabaseOptions {
        vector_dimension: 4,
        similarity: Similarity::Cosine,
        vector_encoding: VectorEncoding::F16,
        sync_on_commit: true,
    }
}

#[test]
fn graph_and_vector_state_survive_reopen() {
    let path = temp_database("reopen");
    let compacted_path = temp_database("reopen-compacted");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    let project = transaction.create_node(
        "Project",
        [("name", Value::String(Arc::from("VectorGraph")))],
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    let file = transaction.create_node(
        "File",
        [("path", Value::String(Arc::from("src/lib.rs")))],
        &[vec![0.9, 0.1, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]],
    );
    let contains = transaction.create_edge(
        project,
        file,
        "CONTAINS",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.8, 0.2, 0.0, 0.0]],
    );
    transaction.commit().unwrap();

    {
        let read = database.read();
        assert_eq!(read.stats().nodes, 2);
        assert_eq!(read.stats().edges, 1);
        assert_eq!(read.stats().indexed_vectors, 4);
        let neighbors = read
            .neighbors(project, Direction::Outgoing, EdgeFilter::default())
            .unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, contains);
        let hits = read
            .vector_search(&[1.0, 0.0, 0.0, 0.0], VectorTarget::Both, 10, None)
            .unwrap();
        assert_eq!(hits.len(), 3, "multivectors aggregate per element");
        assert_eq!(hits[0].element, ElementRef::Node(project));
        let matches = read.match_one_hop(&OneHopQuery {
            start: NodeFilter {
                label: read.label_id("Project"),
                properties: Vec::new(),
            },
            edge_label: read.label_id("CONTAINS"),
            end: NodeFilter {
                label: read.label_id("File"),
                properties: vec![(
                    read.label_id("path").unwrap(),
                    Value::String(Arc::from("src/lib.rs")),
                )],
            },
            direction: Direction::Outgoing,
            limit: 10,
        });
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start, project);
        assert_eq!(matches[0].edge, contains);
        assert_eq!(matches[0].end, file);
    }

    drop(database);
    let reopened = Database::open(&path).unwrap();
    let read = reopened.read();
    assert_eq!(read.stats().nodes, 2);
    assert_eq!(read.stats().edges, 1);
    let node = read.node(file).unwrap();
    assert_eq!(
        read.property(&node.properties, "path"),
        Some(&Value::String(Arc::from("src/lib.rs")))
    );
    let vector = read.node_vector(file, 0).unwrap().unwrap();
    assert!((vector.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(read.node_vector(file, 2).unwrap().is_none());
    drop(read);
    let before = reopened.read().stats();
    let compacted_stats = reopened
        .compact_to(&compacted_path, VectorEncoding::F32)
        .unwrap();
    assert_eq!(compacted_stats, before);
    drop(reopened);
    let compacted = Database::open(&compacted_path).unwrap();
    assert_eq!(compacted.vector_encoding(), VectorEncoding::F32);
    assert!(compacted.read().node(project).is_some());
    assert!(compacted.read().edge(contains).is_some());
    let compacted_file = compacted.read().node(file).unwrap();
    assert_eq!(
        compacted
            .read()
            .property(&compacted_file.properties, "path"),
        Some(&Value::String(Arc::from("src/lib.rs")))
    );
    let compacted_hits = compacted
        .read()
        .vector_search(&[1.0, 0.0, 0.0, 0.0], VectorTarget::Both, 3, None)
        .unwrap();
    assert_eq!(compacted_hits[0].element, ElementRef::Node(project));
    assert_eq!(
        compacted.read().vector_cache_bytes(),
        0,
        "native F32 checkpoints score directly from their mapping"
    );
    let mut after_snapshot = compacted.transaction();
    let appended = after_snapshot.create_node(
        "AfterSnapshot",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    after_snapshot.commit().unwrap();
    drop(compacted);
    let compacted = Database::open(&compacted_path).unwrap();
    assert!(compacted.read().node(appended).is_some());
    assert_eq!(compacted.read().stats().nodes, 3);
    drop(compacted);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(compacted_path).unwrap();
}

#[test]
fn bulk_loader_writes_an_openable_indexed_checkpoint_directly() {
    let path = temp_database("bulk");
    let mut bulk = BulkLoader::new(&path, options()).unwrap();
    let left = bulk
        .create_node(
            "Topic",
            [("name", Value::String(Arc::from("left")))],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    let right = bulk
        .create_node(
            "Topic",
            [("name", Value::String(Arc::from("right")))],
            &[vec![0.0, 1.0, 0.0, 0.0]],
        )
        .unwrap();
    let edge = bulk
        .create_edge(
            left,
            right,
            "LINKS",
            std::iter::empty::<(&str, Value)>(),
            &[vec![0.8, 0.2, 0.0, 0.0]],
        )
        .unwrap();
    let third = bulk
        .create_node(
            "Topic",
            [("name", Value::String(Arc::from("third")))],
            &[vec![0.0, 0.0, 1.0, 0.0]],
        )
        .unwrap();
    bulk.create_edge(
        right,
        third,
        "LINKS",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 0.8, 0.2, 0.0]],
    )
    .unwrap();
    let stats = bulk.finish().unwrap();
    assert_eq!(stats.nodes, 3);
    assert_eq!(stats.edges, 2);
    assert_eq!(stats.indexed_vectors, 5);

    let database = Database::open(&path).unwrap();
    let read = database.read();
    assert_eq!(read.edge(edge).unwrap().source, left);
    assert_eq!(
        read.vector_search(&[1.0, 0.0, 0.0, 0.0], VectorTarget::Both, 1, None)
            .unwrap()[0]
            .element,
        ElementRef::Node(left)
    );
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn late_interaction_matches_multiple_element_facets() {
    let path = temp_database("late-interaction");
    let mut bulk = BulkLoader::new(&path, options()).unwrap();
    let both = bulk
        .create_node(
            "Artifact",
            std::iter::empty::<(&str, Value)>(),
            &[vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
        )
        .unwrap();
    let first_only = bulk
        .create_node(
            "Artifact",
            std::iter::empty::<(&str, Value)>(),
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    bulk.create_node(
        "Artifact",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 1.0, 0.0, 0.0]],
    )
    .unwrap();
    bulk.finish().unwrap();

    let database = Database::open(&path).unwrap();
    let queries = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    let read = database.read();
    let hits = read
        .late_interaction_search(&queries, None, VectorTarget::Nodes, 3, None)
        .unwrap();
    assert_eq!(hits[0].element, ElementRef::Node(both));
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    assert_eq!(hits[0].matched_vector_indices, vec![0, 1]);
    assert!((hits[1].score - 0.5).abs() < 1e-6);

    let weighted = read
        .late_interaction_search(&queries, Some(&[1.0, 0.0]), VectorTarget::Nodes, 3, None)
        .unwrap();
    assert_eq!(weighted[0].score, 1.0);
    assert!(
        weighted
            .iter()
            .any(|hit| hit.element == ElementRef::Node(first_only))
    );
    assert!(
        read.late_interaction_search(&[], None, VectorTarget::Nodes, 3, None)
            .is_err()
    );
    let mut allowed = ElementSet::new();
    allowed.insert(ElementRef::Node(first_only));
    let filtered = read
        .late_interaction_search_within(&queries, None, &allowed, 3)
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].element, ElementRef::Node(first_only));
    assert!((filtered[0].score - 0.5).abs() < 1e-6);
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn graph_candidate_sets_fuse_traversal_labels_and_vector_search() {
    let path = temp_database("candidate-set");
    let mut bulk = BulkLoader::new(&path, options()).unwrap();
    let root = bulk
        .create_node(
            "Root",
            std::iter::empty::<(&str, Value)>(),
            &[vec![0.0, 0.0, 1.0, 0.0]],
        )
        .unwrap();
    let reachable = bulk
        .create_node(
            "Document",
            [("class", Value::String(Arc::from("eligible")))],
            &[vec![0.0, 1.0, 0.0, 0.0]],
        )
        .unwrap();
    let excluded = bulk
        .create_node(
            "Document",
            [("class", Value::String(Arc::from("excluded")))],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    let descendant = bulk
        .create_node(
            "Chunk",
            std::iter::empty::<(&str, Value)>(),
            &[vec![0.0, 0.0, 0.0, 1.0]],
        )
        .unwrap();
    let edge = bulk
        .create_edge(
            root,
            reachable,
            "LINKS",
            std::iter::empty::<(&str, Value)>(),
            &[],
        )
        .unwrap();
    let second_edge = bulk
        .create_edge(
            reachable,
            descendant,
            "LINKS",
            std::iter::empty::<(&str, Value)>(),
            &[],
        )
        .unwrap();
    let cycle_edge = bulk
        .create_edge(
            descendant,
            root,
            "LINKS",
            std::iter::empty::<(&str, Value)>(),
            &[],
        )
        .unwrap();
    bulk.finish().unwrap();

    let database = Database::open(&path).unwrap();
    let read = database.read();
    let roots = read.elements_with_label(
        read.label_id("Root").unwrap(),
        vectorgraph::VectorTarget::Nodes,
    );
    let documents = read.elements_with_label(
        read.label_id("Document").unwrap(),
        vectorgraph::VectorTarget::Nodes,
    );
    let expanded = read
        .expand_element_set(&roots, Direction::Outgoing, EdgeFilter::default())
        .unwrap();
    assert!(expanded.contains(ElementRef::Node(reachable)));
    assert!(expanded.contains(ElementRef::Edge(edge)));
    assert_eq!(
        read.expand_element_set_hops(&roots, Direction::Outgoing, EdgeFilter::default(), 1)
            .unwrap(),
        expanded
    );
    let two_hops = read
        .expand_element_set_hops(&roots, Direction::Outgoing, EdgeFilter::default(), 2)
        .unwrap();
    assert!(two_hops.contains(ElementRef::Node(reachable)));
    assert!(two_hops.contains(ElementRef::Node(descendant)));
    assert!(two_hops.contains(ElementRef::Edge(edge)));
    assert!(two_hops.contains(ElementRef::Edge(second_edge)));
    assert!(!two_hops.contains(ElementRef::Edge(cycle_edge)));
    let three_hops = read
        .expand_element_set_hops(&roots, Direction::Outgoing, EdgeFilter::default(), 3)
        .unwrap();
    assert!(three_hops.contains(ElementRef::Edge(cycle_edge)));
    assert!(!three_hops.contains(ElementRef::Node(root)));
    assert!(
        read.expand_element_set_hops(&roots, Direction::Outgoing, EdgeFilter::default(), 0)
            .unwrap()
            .is_empty()
    );
    let property_matches = read.elements_matching(
        VectorTarget::Nodes,
        &ElementFilter {
            label: read.label_id("Document"),
            properties: vec![(
                read.label_id("class").unwrap(),
                Value::String(Arc::from("eligible")),
            )],
        },
    );
    let property_plan = read.element_filter_plan(
        VectorTarget::Nodes,
        &ElementFilter {
            label: read.label_id("Document"),
            properties: vec![(
                read.label_id("class").unwrap(),
                Value::String(Arc::from("eligible")),
            )],
        },
    );
    assert_eq!(
        property_plan.strategy,
        ElementFilterStrategy::PropertyPosting
    );
    assert_eq!(property_plan.candidate_upper_bound, 1);
    let tie_plan = read.element_filter_plan(
        VectorTarget::Nodes,
        &ElementFilter {
            label: read.label_id("Root"),
            properties: vec![(
                read.label_id("class").unwrap(),
                Value::String(Arc::from("eligible")),
            )],
        },
    );
    assert_eq!(tie_plan.strategy, ElementFilterStrategy::LabelPosting);
    assert_eq!(tie_plan.candidate_upper_bound, 1);
    assert_eq!(property_matches.node_len(), 1);
    let eligible = expanded
        .intersection(&documents)
        .intersection(&property_matches);
    assert_eq!(eligible.node_len(), 1);
    assert!(!eligible.contains(ElementRef::Node(excluded)));
    assert_eq!(roots.union(&eligible).node_len(), 2);
    assert_eq!(documents.difference(&eligible).node_len(), 1);

    let hits = read
        .vector_search_within(&[1.0, 0.0, 0.0, 0.0], &eligible, 10)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].element, ElementRef::Node(reachable));
    let adaptive = read
        .vector_search_within_adaptive(&[1.0, 0.0, 0.0, 0.0], &eligible, 10)
        .unwrap();
    assert_eq!(adaptive, hits);
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn mapped_property_postings_reconcile_wal_updates_and_new_elements() {
    let path = temp_database("property-posting-wal");
    let mut bulk = BulkLoader::new(&path, options()).unwrap();
    let changed = bulk
        .create_node(
            "Item",
            [("state", Value::String(Arc::from("old")))],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    let unchanged = bulk
        .create_node(
            "Item",
            [("state", Value::String(Arc::from("old")))],
            &[vec![0.0, 1.0, 0.0, 0.0]],
        )
        .unwrap();
    let changed_edge = bulk
        .create_edge(
            changed,
            unchanged,
            "LINK",
            [("state", Value::String(Arc::from("old")))],
            &[vec![0.0, 0.0, 0.0, 1.0]],
        )
        .unwrap();
    bulk.finish().unwrap();

    let database = Database::open(&path).unwrap();
    let state = database.read().label_id("state").unwrap();
    let old_filter = ElementFilter {
        label: None,
        properties: vec![(state, Value::String(Arc::from("old")))],
    };
    let initial = database
        .read()
        .elements_matching(VectorTarget::Nodes, &old_filter);
    assert_eq!(
        initial.node_ids().collect::<Vec<_>>(),
        vec![changed, unchanged]
    );
    let initial_both = database
        .read()
        .elements_matching(VectorTarget::Both, &old_filter);
    assert_eq!(initial_both.node_len(), 2);
    assert_eq!(
        initial_both.edge_ids().collect::<Vec<_>>(),
        vec![changed_edge]
    );

    let mut transaction = database.transaction();
    transaction
        .update_node(
            changed,
            "Item",
            [("state", Value::String(Arc::from("new")))],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    let inserted = transaction.create_node(
        "Item",
        [("state", Value::String(Arc::from("new")))],
        &[vec![0.0, 0.0, 1.0, 0.0]],
    );
    transaction
        .update_edge(
            changed_edge,
            changed,
            unchanged,
            "LINK",
            [("state", Value::String(Arc::from("new")))],
            &[vec![0.0, 0.0, 0.0, 1.0]],
        )
        .unwrap();
    transaction.commit().unwrap();

    let read = database.read();
    let old = read.elements_matching(VectorTarget::Nodes, &old_filter);
    assert_eq!(old.node_ids().collect::<Vec<_>>(), vec![unchanged]);
    let new = read.elements_matching(
        VectorTarget::Nodes,
        &ElementFilter {
            label: None,
            properties: vec![(state, Value::String(Arc::from("new")))],
        },
    );
    assert_eq!(new.node_ids().collect::<Vec<_>>(), vec![changed, inserted]);
    let new_both = read.elements_matching(
        VectorTarget::Both,
        &ElementFilter {
            label: None,
            properties: vec![(state, Value::String(Arc::from("new")))],
        },
    );
    assert_eq!(new_both.edge_ids().collect::<Vec<_>>(), vec![changed_edge]);
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn mapped_numeric_ranges_are_typed_bounded_and_wal_consistent() {
    let path = temp_database("numeric-range-posting");
    let mut bulk = BulkLoader::new(&path, options()).unwrap();
    let low = bulk
        .create_node(
            "Movie",
            [("rating", Value::Float(8.0))],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    let boundary = bulk
        .create_node(
            "Movie",
            [("rating", Value::Float(9.2))],
            &[vec![0.0, 1.0, 0.0, 0.0]],
        )
        .unwrap();
    let high = bulk
        .create_node(
            "Movie",
            [("rating", Value::Float(9.8))],
            &[vec![0.0, 0.0, 1.0, 0.0]],
        )
        .unwrap();
    let integer = bulk
        .create_node(
            "Movie",
            [("rating", Value::Int(10))],
            &[vec![0.0, 0.0, 0.0, 1.0]],
        )
        .unwrap();
    bulk.finish().unwrap();

    let database = Database::open(&path).unwrap();
    let read = database.read();
    let rating = read.label_id("rating").unwrap();
    let filter = NumericRangeFilter {
        label: read.label_id("Movie"),
        key: rating,
        lower: Bound::Included(NumericValue::Float(9.2)),
        upper: Bound::Unbounded,
    };
    assert_eq!(
        read.numeric_range_plan(VectorTarget::Nodes, &filter)
            .unwrap()
            .strategy,
        NumericRangeStrategy::NumericPosting
    );
    let numeric_equality = ElementFilter {
        label: None,
        properties: vec![(rating, Value::Float(9.2))],
    };
    assert_eq!(
        read.element_filter_plan(VectorTarget::Nodes, &numeric_equality)
            .strategy,
        ElementFilterStrategy::PropertyPosting
    );
    assert_eq!(
        read.elements_matching(VectorTarget::Nodes, &numeric_equality)
            .node_ids()
            .collect::<Vec<_>>(),
        vec![boundary]
    );
    assert_eq!(
        read.elements_matching_numeric_range(VectorTarget::Nodes, &filter)
            .unwrap()
            .node_ids()
            .collect::<Vec<_>>(),
        vec![boundary, high]
    );
    let open_interval = NumericRangeFilter {
        label: None,
        key: rating,
        lower: Bound::Excluded(NumericValue::Float(8.0)),
        upper: Bound::Excluded(NumericValue::Float(9.8)),
    };
    assert_eq!(
        read.elements_matching_numeric_range(VectorTarget::Nodes, &open_interval)
            .unwrap()
            .node_ids()
            .collect::<Vec<_>>(),
        vec![boundary]
    );
    let integers = NumericRangeFilter {
        label: None,
        key: rating,
        lower: Bound::Included(NumericValue::Int(9)),
        upper: Bound::Included(NumericValue::Int(10)),
    };
    assert_eq!(
        read.elements_matching_numeric_range(VectorTarget::Nodes, &integers)
            .unwrap()
            .node_ids()
            .collect::<Vec<_>>(),
        vec![integer]
    );
    drop(read);

    let mut transaction = database.transaction();
    transaction
        .update_node(
            low,
            "Movie",
            [("rating", Value::Float(9.6))],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    transaction
        .update_node(
            high,
            "Movie",
            [("rating", Value::Float(7.0))],
            &[vec![0.0, 0.0, 1.0, 0.0]],
        )
        .unwrap();
    let inserted = transaction.create_node(
        "Movie",
        [("rating", Value::Float(9.9))],
        &[vec![0.0, 1.0, 1.0, 0.0]],
    );
    transaction.commit().unwrap();

    let read = database.read();
    assert_eq!(
        read.elements_matching_numeric_range(VectorTarget::Nodes, &filter)
            .unwrap()
            .node_ids()
            .collect::<Vec<_>>(),
        vec![low, boundary, inserted]
    );
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn filtered_vector_execution_fuses_scalar_plans_before_scoring() {
    let path = temp_database("filtered-vector-plan");
    let mut bulk = BulkLoader::new(&path, options()).unwrap();
    let target = bulk
        .create_node(
            "Movie",
            [
                ("state", Value::String(Arc::from("live"))),
                ("rating", Value::Float(9.5)),
            ],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    bulk.create_node(
        "Movie",
        [
            ("state", Value::String(Arc::from("archived"))),
            ("rating", Value::Float(9.8)),
        ],
        &[vec![1.0, 0.0, 0.0, 0.0]],
    )
    .unwrap();
    bulk.create_node(
        "Movie",
        [
            ("state", Value::String(Arc::from("live"))),
            ("rating", Value::Float(8.0)),
        ],
        &[vec![0.0, 1.0, 0.0, 0.0]],
    )
    .unwrap();
    bulk.create_node(
        "Movie",
        [
            ("state", Value::String(Arc::from("live"))),
            ("rating", Value::Int(10)),
        ],
        &[vec![0.0, 0.0, 1.0, 0.0]],
    )
    .unwrap();
    bulk.finish().unwrap();

    let database = Database::open(&path).unwrap();
    let read = database.read();
    let state = read.label_id("state").unwrap();
    let rating = read.label_id("rating").unwrap();
    let equality = ElementFilter {
        label: read.label_id("Movie"),
        properties: vec![(state, Value::String(Arc::from("live")))],
    };
    let ranges = [NumericRangeFilter {
        label: None,
        key: rating,
        lower: Bound::Included(NumericValue::Float(9.0)),
        upper: Bound::Unbounded,
    }];
    let result = read
        .vector_search_filtered_adaptive(
            &[1.0, 0.0, 0.0, 0.0],
            VectorTarget::Nodes,
            10,
            Some(&equality),
            &ranges,
        )
        .unwrap();
    assert_eq!(result.candidate_elements, 1);
    assert!(result.equality_plan.is_some());
    assert_eq!(
        result.numeric_range_plans[0].strategy,
        NumericRangeStrategy::NumericPosting
    );
    assert_eq!(result.vector_plan.strategy, VectorSearchStrategy::Exact);
    assert_eq!(result.hits[0].element, ElementRef::Node(target));
    drop(read);

    let mut transaction = database.transaction();
    transaction
        .update_node(
            target,
            "Movie",
            [
                ("state", Value::String(Arc::from("live"))),
                ("rating", Value::Float(7.0)),
            ],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    let inserted = transaction.create_node(
        "Movie",
        [
            ("state", Value::String(Arc::from("live"))),
            ("rating", Value::Float(9.7)),
        ],
        &[vec![0.9, 0.1, 0.0, 0.0]],
    );
    transaction.commit().unwrap();

    let read = database.read();
    let result = read
        .vector_search_filtered_adaptive(
            &[1.0, 0.0, 0.0, 0.0],
            VectorTarget::Nodes,
            10,
            Some(&equality),
            &ranges,
        )
        .unwrap();
    assert_eq!(result.candidate_elements, 1);
    assert_eq!(result.hits[0].element, ElementRef::Node(inserted));
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn immutable_undirected_visits_each_parallel_self_edge_once() {
    let path = temp_database("self-loop-visitation");
    let mut bulk = BulkLoader::new(&path, options()).unwrap();
    let node = bulk
        .create_node(
            "N",
            std::iter::empty::<(&str, Value)>(),
            std::slice::from_ref(&vec![1.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
    let first = bulk
        .create_edge(node, node, "SELF", std::iter::empty::<(&str, Value)>(), &[])
        .unwrap();
    let second = bulk
        .create_edge(node, node, "SELF", std::iter::empty::<(&str, Value)>(), &[])
        .unwrap();
    bulk.finish().unwrap();
    let database = Database::open(&path).unwrap();
    let mut visited = Vec::new();
    database
        .read()
        .visit_neighbors(
            node,
            Direction::Both,
            EdgeFilter::default(),
            |neighbor, edge| visited.push((neighbor, edge)),
        )
        .unwrap();
    visited.sort_unstable();
    assert_eq!(visited, vec![(node, first), (node, second)]);
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn filtered_ann_handles_physically_interleaved_node_and_edge_owners() {
    let path = temp_database("interleaved-sketch-owners");
    let mut bulk = BulkLoader::new(&path, options()).unwrap();
    let first = bulk
        .create_node(
            "N",
            std::iter::empty::<(&str, Value)>(),
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
    let second = bulk
        .create_node(
            "N",
            std::iter::empty::<(&str, Value)>(),
            &[vec![0.0, 1.0, 0.0, 0.0]],
        )
        .unwrap();
    bulk.create_edge(
        first,
        second,
        "E",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 0.0, 0.0, 1.0]],
    )
    .unwrap();
    // This node follows an edge in physical vector order, so the owner-kind
    // stream is deliberately not globally ordered.
    let third = bulk
        .create_node(
            "N",
            std::iter::empty::<(&str, Value)>(),
            &[vec![0.0, 0.0, 1.0, 0.0]],
        )
        .unwrap();
    let vectorless = bulk
        .create_node("N", std::iter::empty::<(&str, Value)>(), &[])
        .unwrap();
    bulk.finish().unwrap();

    let database = Database::open(&path).unwrap();
    let mut allowed = ElementSet::new();
    allowed.insert(ElementRef::Node(third));
    // Keep the set larger than the explicit budget so the approximate path is
    // exercised, while only `third` has an indexed vector.
    allowed.insert(ElementRef::Node(vectorless));
    let hits = database
        .read()
        .vector_search_within_approximate(&[0.0, 0.0, 1.0, 0.0], &allowed, 1, 1)
        .unwrap();
    assert_eq!(hits[0].element, ElementRef::Node(third));
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn incomplete_tail_is_removed_before_the_next_commit() {
    let path = temp_database("torn-tail");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    let first = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    transaction.commit().unwrap();
    drop(database);

    let clean_len = std::fs::metadata(&path).unwrap().len();
    let mut file = OpenOptions::new().append(true).open(&path).unwrap();
    file.write_all(b"VGRTXN01\x10\x00").unwrap();
    file.sync_all().unwrap();
    drop(file);

    let recovered = Database::open(&path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), clean_len);
    assert!(recovered.read().node(first).is_some());
    let mut transaction = recovered.transaction();
    let second = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    transaction.commit().unwrap();
    drop(recovered);

    let reopened = Database::open(&path).unwrap();
    assert!(reopened.read().node(first).is_some());
    assert!(reopened.read().node(second).is_some());
    drop(reopened);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn read_only_open_ignores_torn_tail_without_repairing_or_writing() {
    let path = temp_database("read-only-torn-tail");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    transaction.create_node(
        "Document",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    transaction.commit().unwrap();
    drop(database);

    let valid_len = std::fs::metadata(&path).unwrap().len();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"torn-tail")
        .unwrap();
    let torn_len = std::fs::metadata(&path).unwrap().len();
    assert!(torn_len > valid_len);

    let database = Database::open_read_only(&path).unwrap();
    assert!(database.is_read_only());
    assert_eq!(database.read().stats().nodes, 1);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), torn_len);
    let mut transaction = database.transaction();
    transaction.create_node(
        "Forbidden",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    assert!(transaction.commit().is_err());
    drop(database);

    let repaired = Database::open(&path).unwrap();
    assert!(!repaired.is_read_only());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), valid_len);
    drop(repaired);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn staged_edge_may_precede_its_nodes() {
    let path = temp_database("operation-order");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    // IDs are known only after staging nodes through the public API, so this
    // checks the important persistence ordering through a node, second node,
    // and edge in one transaction. Internal mutation-order fuzzing covers the
    // stricter permutation case later.
    let left = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    let right = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    transaction.create_edge(left, right, "E", std::iter::empty::<(&str, Value)>(), &[]);
    transaction.commit().unwrap();
    drop(database);
    assert_eq!(Database::open(&path).unwrap().read().stats().edges, 1);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn node_deletion_requires_detach_and_hides_vector_entries() {
    let path = temp_database("delete");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    let left = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    let right = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    transaction.create_edge(
        left,
        right,
        "E",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    transaction.commit().unwrap();

    let mut rejected = database.transaction();
    rejected.delete_node(left, false);
    assert!(rejected.commit().is_err());

    let mut detach = database.transaction();
    detach.delete_node(left, true);
    detach.commit().unwrap();
    let read = database.read();
    assert_eq!(read.stats().nodes, 1);
    assert_eq!(read.stats().edges, 0);
    assert_eq!(read.stats().indexed_vectors, 1);
    let hits = read
        .vector_search(&[1.0, 0.0, 0.0, 0.0], VectorTarget::Both, 10, None)
        .unwrap();
    assert!(hits.iter().all(|hit| hit.element != ElementRef::Node(left)));
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn semantic_expansion_scores_relationship_vectors() {
    let path = temp_database("semantic-path");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    let seed = transaction.create_node(
        "Topic",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    let relevant = transaction.create_node(
        "Document",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.8, 0.2, 0.0, 0.0]],
    );
    let incidental = transaction.create_node(
        "Document",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.8, 0.2, 0.0, 0.0]],
    );
    let supporting = transaction.create_edge(
        seed,
        relevant,
        "SUPPORTS",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    transaction.create_edge(
        seed,
        incidental,
        "MENTIONS",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    transaction.commit().unwrap();

    let options = SemanticPathOptions {
        max_hops: 1,
        limit: 2,
        direction: Direction::Outgoing,
        node_weight: 0.5,
        edge_weight: 0.5,
        path_decay: 0.0,
        ..SemanticPathOptions::default()
    };
    let read = database.read();
    let hits = read
        .semantic_expand(&[1.0, 0.0, 0.0, 0.0], &[seed], &options)
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].node, relevant);
    assert_eq!(hits[0].path, vec![supporting]);
    assert!(hits[0].score > hits[1].score);

    let patterns = read
        .match_semantic_one_hop(
            &[1.0, 0.0, 0.0, 0.0],
            &SemanticOneHopQuery {
                pattern: OneHopQuery {
                    start: NodeFilter {
                        label: read.label_id("Topic"),
                        properties: Vec::new(),
                    },
                    end: NodeFilter {
                        label: read.label_id("Document"),
                        properties: Vec::new(),
                    },
                    direction: Direction::Outgoing,
                    limit: 2,
                    ..OneHopQuery::default()
                },
                seed_count: 4,
                ..SemanticOneHopQuery::default()
            },
        )
        .unwrap();
    assert_eq!(patterns.len(), 2);
    assert_eq!(patterns[0].pattern.end, relevant);
    assert_eq!(patterns[0].pattern.edge, supporting);
    assert!(patterns[0].edge_score > patterns[1].edge_score);
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn updates_replace_vectors_and_edge_adjacency() {
    let path = temp_database("update");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    let left = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    let middle = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    let right = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 0.0, 1.0, 0.0]],
    );
    let edge = transaction.create_edge(
        left,
        middle,
        "OLD",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    transaction.commit().unwrap();

    let mut update = database.transaction();
    update
        .update_node(
            left,
            "N",
            std::iter::empty::<(&str, Value)>(),
            &[vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
        )
        .unwrap();
    update
        .update_edge(
            edge,
            right,
            middle,
            "NEW",
            std::iter::empty::<(&str, Value)>(),
            &[vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
        )
        .unwrap();
    update.commit().unwrap();

    let read = database.read();
    assert!(
        read.neighbors(left, Direction::Outgoing, EdgeFilter::default())
            .unwrap()
            .is_empty()
    );
    let moved = read
        .neighbors(right, Direction::Outgoing, EdgeFilter::default())
        .unwrap();
    assert_eq!(moved.len(), 1);
    assert_eq!(moved[0].id, edge);
    assert_eq!(read.symbol(moved[0].label), Some("NEW"));
    assert_eq!(read.stats().indexed_vectors, 6);
    drop(read);
    drop(database);
    let reopened = Database::open(&path).unwrap();
    assert_eq!(reopened.read().stats().indexed_vectors, 6);
    assert!(
        reopened
            .read()
            .neighbors(left, Direction::Outgoing, EdgeFilter::default())
            .unwrap()
            .is_empty()
    );
    drop(reopened);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn mapped_checkpoint_vectors_are_verified_lazily() {
    let path = temp_database("mapped-checksum-source");
    let compacted_path = temp_database("mapped-checksum");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    transaction.commit().unwrap();
    database
        .compact_to(&compacted_path, VectorEncoding::F16)
        .unwrap();
    drop(database);

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&compacted_path)
        .unwrap();
    file.seek(SeekFrom::End(-8)).unwrap();
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Current(-1)).unwrap();
    file.write_all(&[byte[0] ^ 0x80]).unwrap();
    file.sync_all().unwrap();
    drop(file);

    let reopened = Database::open(&compacted_path).unwrap();
    assert_eq!(reopened.read().stats().nodes, 1);
    assert!(reopened.read().verify_integrity().is_err());
    assert!(
        reopened
            .read()
            .vector_search(&[1.0, 0.0, 0.0, 0.0], VectorTarget::Nodes, 1, None)
            .is_err()
    );
    drop(reopened);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(compacted_path).unwrap();
}

#[test]
fn checkpoint_csr_stays_correct_across_edge_updates_and_deletes() {
    let path = temp_database("checkpoint-csr-source");
    let compacted_path = temp_database("checkpoint-csr");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    let left = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    let middle = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    let right = transaction.create_node(
        "N",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 0.0, 1.0, 0.0]],
    );
    let edge = transaction.create_edge(
        left,
        middle,
        "E",
        [("evidence", Value::String(Arc::from("original")))],
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    transaction.commit().unwrap();
    database
        .compact_to(&compacted_path, VectorEncoding::F16)
        .unwrap();
    drop(database);

    let checkpoint = Database::open(&compacted_path).unwrap();
    assert_eq!(
        checkpoint
            .read()
            .vector_search_plan(VectorTarget::Both, None)
            .strategy,
        VectorSearchStrategy::Exact
    );
    let mapped_edge = checkpoint.read().edge(edge).unwrap();
    assert_eq!(
        checkpoint
            .read()
            .property(&mapped_edge.properties, "evidence"),
        Some(&Value::String(Arc::from("original")))
    );
    let mut update = checkpoint.transaction();
    update
        .update_edge(
            edge,
            right,
            middle,
            "E",
            [("evidence", Value::String(Arc::from("moved")))],
            &[vec![0.0, 0.0, 1.0, 0.0]],
        )
        .unwrap();
    update.commit().unwrap();
    assert!(
        checkpoint
            .read()
            .neighbors(left, Direction::Outgoing, EdgeFilter::default())
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        checkpoint
            .read()
            .neighbors(right, Direction::Outgoing, EdgeFilter::default())
            .unwrap()[0]
            .id,
        edge
    );
    let delta_hits = checkpoint
        .read()
        .vector_search(&[0.0, 0.0, 1.0, 0.0], VectorTarget::Edges, 1, None)
        .unwrap();
    assert_eq!(delta_hits[0].element, ElementRef::Edge(edge));

    let mut delete = checkpoint.transaction();
    delete.delete_edge(edge);
    delete.commit().unwrap();
    assert!(
        checkpoint
            .read()
            .neighbors(right, Direction::Outgoing, EdgeFilter::default())
            .unwrap()
            .is_empty()
    );
    drop(checkpoint);
    let reopened = Database::open(&compacted_path).unwrap();
    assert_eq!(reopened.read().stats().edges, 0);
    assert!(reopened.read().edge(edge).is_none());
    drop(reopened);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(compacted_path).unwrap();
}

#[test]
fn approximate_checkpoint_search_reranks_and_scans_the_wal_delta() {
    let path = temp_database("ann-source");
    let compacted_path = temp_database("ann-checkpoint");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    let mut first = 0;
    for index in 0..40 {
        let angle = index as f32 * 0.157;
        let id = transaction.create_node(
            "Base",
            std::iter::empty::<(&str, Value)>(),
            &[vec![angle.cos(), angle.sin(), 0.1, -0.1]],
        );
        if index == 0 {
            first = id;
        }
    }
    transaction.commit().unwrap();
    database
        .compact_to(&compacted_path, VectorEncoding::F16)
        .unwrap();
    drop(database);

    let checkpoint = Database::open(&compacted_path).unwrap();
    let hits = checkpoint
        .read()
        .vector_search_approximate(&[1.0, 0.0, 0.1, -0.1], VectorTarget::Nodes, 1, None, 8)
        .unwrap();
    assert_eq!(hits[0].element, ElementRef::Node(first));

    let mut append = checkpoint.transaction();
    let fresh = append.create_node(
        "Fresh",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.0, 0.0, 0.0, 1.0]],
    );
    append.commit().unwrap();
    let fresh_hits = checkpoint
        .read()
        .vector_search_approximate(&[0.0, 0.0, 0.0, 1.0], VectorTarget::Nodes, 1, None, 8)
        .unwrap();
    assert_eq!(fresh_hits[0].element, ElementRef::Node(fresh));

    let mut update = checkpoint.transaction();
    update
        .update_node(
            first,
            "Base",
            std::iter::empty::<(&str, Value)>(),
            &[vec![0.0, 0.0, 1.0, 0.0]],
        )
        .unwrap();
    update.commit().unwrap();
    let updated_hits = checkpoint
        .read()
        .vector_search_approximate(
            &[0.0, 0.0, 1.0, 0.0],
            VectorTarget::Nodes,
            1,
            checkpoint.read().label_id("Base"),
            8,
        )
        .unwrap();
    assert_eq!(updated_hits[0].element, ElementRef::Node(first));

    drop(checkpoint);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(compacted_path).unwrap();
}

#[test]
fn graph_range_search_prefilters_reachable_nodes_before_vector_planning() {
    let path = temp_database("graph-range-source");
    let compacted_path = temp_database("graph-range");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    let seed = transaction.create_node(
        "Keep",
        [("scope", Value::String(Arc::from("local")))],
        &[vec![1.0, 0.0, 0.0, 0.0]],
    );
    let middle = transaction.create_node(
        "Drop",
        [("scope", Value::String(Arc::from("local")))],
        &[vec![0.0, 1.0, 0.0, 0.0]],
    );
    let reachable = transaction.create_node(
        "Keep",
        [("scope", Value::String(Arc::from("local")))],
        &[vec![0.0, 0.0, 1.0, 0.0]],
    );
    let wrong_edge = transaction.create_node(
        "Keep",
        [("scope", Value::String(Arc::from("local")))],
        &[vec![0.0, 0.0, 1.0, 0.0]],
    );
    let isolated = transaction.create_node(
        "Keep",
        [("scope", Value::String(Arc::from("local")))],
        &[vec![0.0, 0.0, 1.0, 0.0]],
    );
    transaction.create_edge(
        seed,
        middle,
        "LINK",
        std::iter::empty::<(&str, Value)>(),
        &[],
    );
    transaction.create_edge(
        middle,
        reachable,
        "LINK",
        std::iter::empty::<(&str, Value)>(),
        &[],
    );
    transaction.create_edge(
        seed,
        wrong_edge,
        "OTHER",
        std::iter::empty::<(&str, Value)>(),
        &[],
    );
    transaction.commit().unwrap();
    database
        .compact_to(&compacted_path, VectorEncoding::F16)
        .unwrap();
    drop(database);

    let database = Database::open(&compacted_path).unwrap();
    let read = database.read();
    let mut seeds = ElementSet::new();
    seeds.insert(ElementRef::Node(seed));
    let candidates = read
        .nodes_within_hops(
            &seeds,
            Direction::Outgoing,
            EdgeFilter {
                label: read.label_id("LINK"),
            },
            2,
            false,
            None,
        )
        .unwrap();
    assert_eq!(
        candidates.node_ids().collect::<Vec<_>>(),
        vec![middle, reachable]
    );
    assert_eq!(candidates.edge_len(), 0);

    let options = GraphRangeSearchOptions {
        max_hops: 2,
        limit: 10,
        direction: Direction::Outgoing,
        edge_filter: EdgeFilter {
            label: read.label_id("LINK"),
        },
        include_seeds: false,
        node_filter: Some(ElementFilter {
            label: read.label_id("Keep"),
            properties: vec![(
                read.label_id("scope").unwrap(),
                Value::String(Arc::from("local")),
            )],
        }),
    };
    let result = read
        .vector_search_graph_range_adaptive(&[0.0, 0.0, 1.0, 0.0], &seeds, &options)
        .unwrap();
    assert_eq!(result.candidate_nodes, 1);
    assert_eq!(result.plan.strategy, VectorSearchStrategy::Exact);
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].element, ElementRef::Node(reachable));
    assert!(!result.hits.iter().any(|hit| {
        matches!(hit.element, ElementRef::Node(id) if id == wrong_edge || id == isolated)
    }));

    let mut missing = ElementSet::new();
    missing.insert(ElementRef::Node(999));
    assert!(
        read.nodes_within_hops(
            &missing,
            Direction::Outgoing,
            EdgeFilter::default(),
            0,
            true,
            None,
        )
        .is_err()
    );
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(compacted_path).unwrap();
}

#[test]
fn full_f32_vector_cache_is_explicit_and_inspectable() {
    let path = temp_database("explicit-vector-cache-source");
    let compacted_path = temp_database("explicit-vector-cache");
    let database = Database::create(&path, options()).unwrap();
    let mut transaction = database.transaction();
    for index in 0..32 {
        transaction.create_node(
            "N",
            std::iter::empty::<(&str, Value)>(),
            &[vec![1.0, index as f32 / 32.0, 0.0, 0.0]],
        );
    }
    transaction.commit().unwrap();
    database
        .compact_to(&compacted_path, VectorEncoding::F16)
        .unwrap();
    drop(database);

    let database = Database::open(&compacted_path).unwrap();
    let read = database.read();
    let query = [1.0, 0.0, 0.0, 0.0];
    let compressed = read
        .vector_search(&query, VectorTarget::Nodes, 5, None)
        .unwrap();
    let repeated = read
        .vector_search(&query, VectorTarget::Nodes, 5, None)
        .unwrap();
    assert_eq!(compressed, repeated);
    assert_eq!(read.vector_cache_bytes(), 0);

    assert_eq!(read.warm_vector_cache().unwrap(), 32 * 4 * 4);
    assert_eq!(read.vector_cache_bytes(), 32 * 4 * 4);
    let warmed = read
        .vector_search(&query, VectorTarget::Nodes, 5, None)
        .unwrap();
    assert_eq!(compressed, warmed);
    drop(read);
    drop(database);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(compacted_path).unwrap();
}
