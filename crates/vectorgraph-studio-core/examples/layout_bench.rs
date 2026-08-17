use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::time::Instant;

use vectorgraph_studio_core::{
    GraphWorkspace, LayoutKind, LayoutOptions, MAX_CAMERA_ZOOM, SceneEdges, SceneSnapshot,
    SnapshotOptions, Vec2,
};

#[derive(Clone, Copy, Debug)]
struct LayoutMetrics {
    axis_aligned_pct: f64,
    edge_length_mean: f64,
    edge_length_cv: f64,
}

#[derive(Clone, Copy, Debug)]
struct SeparationMetrics {
    duplicate_positions: usize,
    nearest_p10: f32,
    nearest_p50: f32,
    max_zoom_p10_px: f32,
}

fn separation_metrics(positions: &[Vec2]) -> SeparationMetrics {
    if positions.len() < 2 {
        return SeparationMetrics {
            duplicate_positions: 0,
            nearest_p10: 0.0,
            nearest_p50: 0.0,
            max_zoom_p10_px: 0.0,
        };
    }
    let mut unique = HashSet::with_capacity(positions.len());
    let mut duplicate_positions = 0;
    for position in positions {
        duplicate_positions +=
            usize::from(!unique.insert((position.x.to_bits(), position.y.to_bits())));
    }
    let sample_count = positions.len().min(512);
    let mut nearest = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        let index = sample * positions.len() / sample_count;
        let origin = positions[index];
        let distance = positions
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, &other)| (other - origin).length())
            .filter(|distance| distance.is_finite())
            .min_by(f32::total_cmp)
            .unwrap_or(0.0);
        nearest.push(distance);
    }
    nearest.sort_unstable_by(f32::total_cmp);
    let nearest_p10 = nearest[nearest.len() / 10];
    let nearest_p50 = nearest[nearest.len() / 2];
    let bounds = vectorgraph_studio_core::Rect::from_points(positions);
    let extent = bounds.width().max(bounds.height()).max(0.000_1);
    SeparationMetrics {
        duplicate_positions,
        nearest_p10,
        nearest_p50,
        // Approximate a square 1200 px canvas after Studio's 32 px margins.
        max_zoom_p10_px: nearest_p10 * (1_136.0 / extent) * MAX_CAMERA_ZOOM,
    }
}

fn layout_metrics(edges: &SceneEdges, positions: &[Vec2]) -> LayoutMetrics {
    let mut lengths = Vec::with_capacity(edges.ids.len());
    let mut axis_aligned = 0_usize;
    for (&source, &target) in edges.sources.iter().zip(&edges.targets) {
        let Some((&source, &target)) = positions
            .get(source as usize)
            .zip(positions.get(target as usize))
        else {
            continue;
        };
        let delta = target - source;
        let length = f64::from(delta.length());
        if length <= f64::EPSILON || !length.is_finite() {
            continue;
        }
        // Edges within three degrees of a screen axis expose lattice-shaped
        // approximations without penalizing naturally straight subgraphs.
        if f64::from(delta.x.abs().min(delta.y.abs())) / length < 3_f64.to_radians().sin() {
            axis_aligned += 1;
        }
        lengths.push(length);
    }

    let mean = lengths.iter().sum::<f64>() / lengths.len().max(1) as f64;
    let variance = lengths
        .iter()
        .map(|length| (length - mean).powi(2))
        .sum::<f64>()
        / lengths.len().max(1) as f64;
    LayoutMetrics {
        axis_aligned_pct: axis_aligned as f64 * 100.0 / lengths.len().max(1) as f64,
        edge_length_mean: mean,
        edge_length_cv: variance.sqrt() / mean.max(f64::EPSILON),
    }
}

fn main() {
    let mut arguments = std::env::args_os().skip(1);
    let path = arguments
        .next()
        .map(PathBuf::from)
        .expect("usage: cargo run -p vectorgraph-studio-core --example layout_bench --release -- <graph.vg> [force|structure]");
    let layout_kind = match arguments.next().and_then(|kind| kind.into_string().ok()) {
        None => LayoutKind::Force,
        Some(kind) if kind == "force" => LayoutKind::Force,
        Some(kind) if kind == "structure" => LayoutKind::Structure,
        Some(kind) => panic!("unknown layout kind {kind:?}; expected force or structure"),
    };
    let opened = Instant::now();
    let snapshot = SceneSnapshot::open(&path, SnapshotOptions::default(), LayoutOptions::default())
        .expect("open graph snapshot");
    let open_ms = opened.elapsed().as_secs_f64() * 1_000.0;
    let auto_metrics = layout_metrics(&snapshot.edges, &snapshot.nodes.positions);
    let mut indegree = vec![0_u32; snapshot.nodes.ids.len()];
    let mut children = vec![Vec::new(); snapshot.nodes.ids.len()];
    for (&source, &target) in snapshot.edges.sources.iter().zip(&snapshot.edges.targets) {
        indegree[target as usize] += 1;
        children[source as usize].push(target as usize);
    }
    let roots: Vec<_> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &degree)| (degree == 0).then_some(index))
        .collect();
    let leaves = children
        .iter()
        .filter(|children| children.is_empty())
        .count();
    // This benchmark also runs on general graphs. A path walk without a
    // visited set grows combinatorially on DAGs with shared descendants and
    // never terminates on cycles, so characterize reachability with a bounded
    // BFS instead of pretending every fixture is a forest.
    let mut visited = vec![false; snapshot.nodes.ids.len()];
    let mut queue = VecDeque::new();
    let mut max_depth = 0_usize;
    let mut components = 0_usize;
    for seed in roots.iter().copied().chain(0..snapshot.nodes.ids.len()) {
        if visited[seed] {
            continue;
        }
        components += 1;
        visited[seed] = true;
        queue.push_back((seed, 0_usize));
        while let Some((node, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);
            for &child in &children[node] {
                if !visited[child] {
                    visited[child] = true;
                    queue.push_back((child, depth + 1));
                }
            }
        }
    }
    for &root in roots.iter().take(12) {
        let branches = children[root]
            .iter()
            .take(12)
            .map(|&child| {
                format!(
                    "{}:{}:{}",
                    snapshot.nodes.labels[child],
                    snapshot.nodes.ids[child],
                    children[child].len()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "root={} label={} children={} branches=[{}]",
            snapshot.nodes.ids[root],
            snapshot.nodes.labels[root],
            children[root].len(),
            branches
        );
    }
    if roots.len() > 12 {
        eprintln!("... {} additional roots omitted", roots.len() - 12);
    }

    let mut workspace = GraphWorkspace::new(&snapshot);
    let arranged = Instant::now();
    let targets = snapshot.arranged_positions(
        layout_kind,
        LayoutOptions::default(),
        Some(workspace.positions()),
        Some(workspace.pins()),
    );
    let arrange_ms = arranged.elapsed().as_secs_f64() * 1_000.0;
    let force_metrics = layout_metrics(&snapshot.edges, &targets);
    let separation = separation_metrics(&targets);
    assert!(workspace.retarget_layout(targets, snapshot.bounds));

    let settled = Instant::now();
    let mut frames = 0_u32;
    while workspace.is_moving() && frames < 2_000 {
        workspace.step(1.0 / 120.0, false);
        frames += 1;
    }
    let settle_cpu_ms = settled.elapsed().as_secs_f64() * 1_000.0;

    println!(
        "nodes={} edges={} roots={} components={} leaves={} max_depth={} layout={} open_ms={open_ms:.3} arrange_ms={arrange_ms:.3} auto_axis_pct={:.2} auto_edge_mean={:.2} auto_edge_cv={:.3} arranged_axis_pct={:.2} arranged_edge_mean={:.2} arranged_edge_cv={:.3} duplicate_positions={} nearest_p10={:.4} nearest_p50={:.4} max_zoom_p10_px={:.2} settle_frames={frames} settle_cpu_ms={settle_cpu_ms:.3} cpu_ms_per_frame={:.3}",
        snapshot.nodes.ids.len(),
        snapshot.edges.ids.len(),
        roots.len(),
        components,
        leaves,
        max_depth,
        layout_kind.label().to_lowercase(),
        auto_metrics.axis_aligned_pct,
        auto_metrics.edge_length_mean,
        auto_metrics.edge_length_cv,
        force_metrics.axis_aligned_pct,
        force_metrics.edge_length_mean,
        force_metrics.edge_length_cv,
        separation.duplicate_positions,
        separation.nearest_p10,
        separation.nearest_p50,
        separation.max_zoom_p10_px,
        settle_cpu_ms / f64::from(frames.max(1)),
    );
}
