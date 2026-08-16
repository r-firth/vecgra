use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vectorgraph::{
    BulkLoader, Database, DatabaseOptions, Direction, EdgeFilter, Similarity, Value, VectorEncoding,
};

pub(crate) fn import_graphalytics(
    vertices_path: &Path,
    edges_path: &Path,
    database_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut bulk = BulkLoader::new(
        database_path,
        DatabaseOptions {
            // Graphalytics has no vector column. A non-zero database dimension
            // remains a file invariant, while elements simply own zero vectors.
            vector_dimension: 1,
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F16,
            sync_on_commit: true,
        },
    )?;
    let vertex_label: Arc<str> = Arc::from("Vertex");
    let edge_label: Arc<str> = Arc::from("EDGE");
    let no_vectors: &[Vec<f32>] = &[];
    let mut external_ids = Vec::new();
    let mut dense_ids = true;
    read_data_lines(vertices_path, |line, line_number| {
        let external_id = line
            .split_ascii_whitespace()
            .next()
            .ok_or_else(|| format!("missing vertex id at line {line_number}"))?
            .parse::<u64>()?;
        if external_ids
            .last()
            .is_some_and(|previous| *previous >= external_id)
        {
            return Err(format!(
                "vertex IDs must be strictly increasing; line {line_number} contains {external_id}"
            )
            .into());
        }
        dense_ids &= external_id == external_ids.len() as u64;
        bulk.create_node(
            vertex_label.clone(),
            std::iter::empty::<(&str, Value)>(),
            no_vectors,
        )?;
        external_ids.push(external_id);
        if external_ids.len().is_multiple_of(500_000) {
            eprintln!("stored {} vertices", external_ids.len());
        }
        Ok(())
    })?;

    let mut edge_count = 0usize;
    read_data_lines(edges_path, |line, line_number| {
        let mut fields = line.split_ascii_whitespace();
        let source = fields
            .next()
            .ok_or_else(|| format!("missing edge source at line {line_number}"))?
            .parse::<u64>()?;
        let target = fields
            .next()
            .ok_or_else(|| format!("missing edge target at line {line_number}"))?
            .parse::<u64>()?;
        let resolve = |external_id: u64| -> Result<u64, Box<dyn Error>> {
            if dense_ids {
                if external_id < external_ids.len() as u64 {
                    return Ok(external_id);
                }
            } else if let Ok(index) = external_ids.binary_search(&external_id) {
                return Ok(index as u64);
            }
            Err(format!("edge line {line_number} references unknown vertex {external_id}").into())
        };
        bulk.create_edge(
            resolve(source)?,
            resolve(target)?,
            edge_label.clone(),
            std::iter::empty::<(&str, Value)>(),
            no_vectors,
        )?;
        edge_count += 1;
        if edge_count.is_multiple_of(1_000_000) {
            eprintln!("stored {edge_count} edges");
        }
        Ok(())
    })?;
    let stats = bulk.finish()?;
    println!("database\t{}", database_path.display());
    println!("nodes\t{}", stats.nodes);
    println!("edges\t{}", stats.edges);
    println!("dense_external_ids\t{dense_ids}");
    Ok(())
}

pub(crate) fn benchmark_bfs(
    database_path: &Path,
    source: u64,
    expected_path: Option<&Path>,
    iterations: usize,
) -> Result<(), Box<dyn Error>> {
    if iterations == 0 {
        return Err("BFS iterations must be greater than zero".into());
    }
    let database = Database::open(database_path)?;
    let read = database.read();
    let node_count = read.stats().nodes;
    if source as usize >= node_count {
        return Err(format!("BFS source {source} is outside {node_count} dense nodes").into());
    }

    let first_started = Instant::now();
    let first = bfs(&read, node_count, source)?;
    let first_time = first_started.elapsed();
    if let Some(expected_path) = expected_path {
        validate_bfs(expected_path, &first)?;
    }
    let reached = first
        .iter()
        .filter(|distance| **distance != u32::MAX)
        .count();
    let max_distance = first
        .iter()
        .copied()
        .filter(|distance| *distance != u32::MAX)
        .max()
        .unwrap_or(0);

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let distances = bfs(&read, node_count, source)?;
        samples.push(started.elapsed());
        debug_assert_eq!(distances, first);
    }
    samples.sort_unstable();
    println!("nodes\t{node_count}");
    println!("edges\t{}", read.stats().edges);
    println!("source\t{source}");
    println!("reached\t{reached}");
    println!("max_distance\t{max_distance}");
    println!("reference_validated\t{}", expected_path.is_some());
    println!("first_ms\t{:.3}", millis(first_time));
    println!("iterations\t{iterations}");
    println!("bfs_min_ms\t{:.3}", millis(samples[0]));
    println!("bfs_p50_ms\t{:.3}", millis(percentile(&samples, 0.50)));
    println!("bfs_p95_ms\t{:.3}", millis(percentile(&samples, 0.95)));
    Ok(())
}

pub(crate) fn benchmark_wcc(
    database_path: &Path,
    expected_path: Option<&Path>,
    iterations: usize,
) -> Result<(), Box<dyn Error>> {
    if iterations == 0 {
        return Err("WCC iterations must be greater than zero".into());
    }
    let database = Database::open(database_path)?;
    let read = database.read();
    let node_count = read.stats().nodes;
    let first_started = Instant::now();
    let first = wcc(&read, node_count)?;
    let first_time = first_started.elapsed();
    if let Some(expected_path) = expected_path {
        validate_wcc(expected_path, &first)?;
    }
    let component_count = first
        .iter()
        .enumerate()
        .filter(|(id, component)| **component == *id as u64)
        .count();
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let components = wcc(&read, node_count)?;
        samples.push(started.elapsed());
        debug_assert_eq!(components, first);
    }
    samples.sort_unstable();
    println!("nodes\t{node_count}");
    println!("edges\t{}", read.stats().edges);
    println!("components\t{component_count}");
    println!("reference_validated\t{}", expected_path.is_some());
    println!("first_ms\t{:.3}", millis(first_time));
    println!("iterations\t{iterations}");
    println!("wcc_min_ms\t{:.3}", millis(samples[0]));
    println!("wcc_p50_ms\t{:.3}", millis(percentile(&samples, 0.50)));
    println!("wcc_p95_ms\t{:.3}", millis(percentile(&samples, 0.95)));
    Ok(())
}

pub(crate) fn benchmark_pagerank(
    database_path: &Path,
    expected_path: Option<&Path>,
    benchmark_iterations: usize,
) -> Result<(), Box<dyn Error>> {
    if benchmark_iterations == 0 {
        return Err("PageRank benchmark iterations must be greater than zero".into());
    }
    const DAMPING: f64 = 0.85;
    const ALGORITHM_ITERATIONS: usize = 10;
    let database = Database::open(database_path)?;
    let read = database.read();
    let node_count = read.stats().nodes;
    let first_started = Instant::now();
    let first = pagerank(&read, node_count, DAMPING, ALGORITHM_ITERATIONS)?;
    let first_time = first_started.elapsed();
    let validation = expected_path
        .map(|path| validate_pagerank(path, &first))
        .transpose()?;
    let rank_sum: f64 = first.iter().sum();
    let mut samples = Vec::with_capacity(benchmark_iterations);
    for _ in 0..benchmark_iterations {
        let started = Instant::now();
        let ranks = pagerank(&read, node_count, DAMPING, ALGORITHM_ITERATIONS)?;
        samples.push(started.elapsed());
        debug_assert_eq!(ranks, first);
    }
    samples.sort_unstable();
    println!("nodes\t{node_count}");
    println!("edges\t{}", read.stats().edges);
    println!("damping\t{DAMPING}");
    println!("algorithm_iterations\t{ALGORITHM_ITERATIONS}");
    println!("rank_sum\t{rank_sum:.12}");
    println!("reference_validated\t{}", expected_path.is_some());
    if let Some((maximum_absolute_error, maximum_relative_error)) = validation {
        println!("max_absolute_error\t{maximum_absolute_error:.3e}");
        println!("max_relative_error\t{maximum_relative_error:.3e}");
    }
    println!("first_ms\t{:.3}", millis(first_time));
    println!("benchmark_iterations\t{benchmark_iterations}");
    println!("pagerank_min_ms\t{:.3}", millis(samples[0]));
    println!("pagerank_p50_ms\t{:.3}", millis(percentile(&samples, 0.50)));
    println!("pagerank_p95_ms\t{:.3}", millis(percentile(&samples, 0.95)));
    Ok(())
}

fn bfs(
    read: &vectorgraph::ReadGuard<'_>,
    node_count: usize,
    source: u64,
) -> Result<Vec<u32>, vectorgraph::Error> {
    let mut distances = vec![u32::MAX; node_count];
    let mut queue = VecDeque::new();
    distances[source as usize] = 0;
    queue.push_back(source);
    while let Some(node) = queue.pop_front() {
        let next_distance = distances[node as usize] + 1;
        read.visit_neighbors(
            node,
            Direction::Outgoing,
            EdgeFilter::default(),
            |neighbor, _edge| {
                let slot = neighbor as usize;
                if distances[slot] == u32::MAX {
                    distances[slot] = next_distance;
                    queue.push_back(neighbor);
                }
            },
        )?;
    }
    Ok(distances)
}

fn wcc(
    read: &vectorgraph::ReadGuard<'_>,
    node_count: usize,
) -> Result<Vec<u64>, vectorgraph::Error> {
    let mut components = vec![u64::MAX; node_count];
    let mut queue = VecDeque::new();
    for root in 0..node_count as u64 {
        if components[root as usize] != u64::MAX {
            continue;
        }
        components[root as usize] = root;
        queue.push_back(root);
        while let Some(node) = queue.pop_front() {
            read.visit_neighbors(
                node,
                Direction::Both,
                EdgeFilter::default(),
                |neighbor, _edge| {
                    let slot = neighbor as usize;
                    if components[slot] == u64::MAX {
                        components[slot] = root;
                        queue.push_back(neighbor);
                    }
                },
            )?;
        }
    }
    Ok(components)
}

fn pagerank(
    read: &vectorgraph::ReadGuard<'_>,
    node_count: usize,
    damping: f64,
    iterations: usize,
) -> Result<Vec<f64>, vectorgraph::Error> {
    let mut out_degrees = vec![0usize; node_count];
    for node in 0..node_count as u64 {
        read.visit_neighbors(
            node,
            Direction::Outgoing,
            EdgeFilter::default(),
            |_neighbor, _edge| out_degrees[node as usize] += 1,
        )?;
    }
    let initial = 1.0 / node_count as f64;
    let mut ranks = vec![initial; node_count];
    let mut next = vec![0.0; node_count];
    for _ in 0..iterations {
        let sink_sum: f64 = ranks
            .iter()
            .zip(&out_degrees)
            .filter(|(_rank, degree)| **degree == 0)
            .map(|(rank, _degree)| *rank)
            .sum();
        let base = (1.0 - damping) / node_count as f64 + damping * sink_sum / node_count as f64;
        for (node, rank) in next.iter_mut().enumerate() {
            let mut incoming = 0.0;
            read.visit_neighbors(
                node as u64,
                Direction::Incoming,
                EdgeFilter::default(),
                |source, _edge| {
                    incoming += ranks[source as usize] / out_degrees[source as usize] as f64;
                },
            )?;
            *rank = base + damping * incoming;
        }
        std::mem::swap(&mut ranks, &mut next);
    }
    Ok(ranks)
}

fn validate_bfs(path: &Path, actual: &[u32]) -> Result<(), Box<dyn Error>> {
    let mut rows = 0usize;
    read_data_lines(path, |line, line_number| {
        let mut fields = line.split_ascii_whitespace();
        let id = fields
            .next()
            .ok_or_else(|| format!("missing reference vertex at line {line_number}"))?
            .parse::<usize>()?;
        let expected = fields
            .next()
            .ok_or_else(|| format!("missing reference distance at line {line_number}"))?
            .parse::<u64>()?;
        let actual = *actual
            .get(id)
            .ok_or_else(|| format!("reference vertex {id} exceeds database node slots"))?;
        let matches = (actual == u32::MAX && expected == i64::MAX as u64)
            || (actual != u32::MAX && expected == actual as u64);
        if !matches {
            return Err(format!(
                "BFS reference mismatch at vertex {id}: expected {expected}, found {actual}"
            )
            .into());
        }
        rows += 1;
        Ok(())
    })?;
    if rows != actual.len() {
        return Err(format!(
            "BFS reference has {rows} rows but database has {} nodes",
            actual.len()
        )
        .into());
    }
    Ok(())
}

fn validate_wcc(path: &Path, actual: &[u64]) -> Result<(), Box<dyn Error>> {
    let mut rows = 0usize;
    read_data_lines(path, |line, line_number| {
        let mut fields = line.split_ascii_whitespace();
        let id = fields
            .next()
            .ok_or_else(|| format!("missing WCC vertex at line {line_number}"))?
            .parse::<usize>()?;
        let expected = fields
            .next()
            .ok_or_else(|| format!("missing WCC component at line {line_number}"))?
            .parse::<u64>()?;
        let found = *actual
            .get(id)
            .ok_or_else(|| format!("WCC vertex {id} exceeds database node slots"))?;
        if found != expected {
            return Err(format!(
                "WCC reference mismatch at vertex {id}: expected {expected}, found {found}"
            )
            .into());
        }
        rows += 1;
        Ok(())
    })?;
    if rows != actual.len() {
        return Err(format!(
            "WCC reference has {rows} rows but database has {} nodes",
            actual.len()
        )
        .into());
    }
    Ok(())
}

fn validate_pagerank(path: &Path, actual: &[f64]) -> Result<(f64, f64), Box<dyn Error>> {
    let mut rows = 0usize;
    let mut maximum_absolute_error = 0.0f64;
    let mut maximum_relative_error = 0.0f64;
    read_data_lines(path, |line, line_number| {
        let mut fields = line.split_ascii_whitespace();
        let id = fields
            .next()
            .ok_or_else(|| format!("missing PageRank vertex at line {line_number}"))?
            .parse::<usize>()?;
        let expected = fields
            .next()
            .ok_or_else(|| format!("missing PageRank value at line {line_number}"))?
            .parse::<f64>()?;
        let found = *actual
            .get(id)
            .ok_or_else(|| format!("PageRank vertex {id} exceeds database node slots"))?;
        let absolute_error = (found - expected).abs();
        let relative_error = absolute_error / expected.abs().max(f64::MIN_POSITIVE);
        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
        maximum_relative_error = maximum_relative_error.max(relative_error);
        if absolute_error > 1e-9 && relative_error > 1e-5 {
            return Err(format!(
                "PageRank reference mismatch at vertex {id}: expected {expected}, found {found}"
            )
            .into());
        }
        rows += 1;
        Ok(())
    })?;
    if rows != actual.len() {
        return Err(format!(
            "PageRank reference has {rows} rows but database has {} nodes",
            actual.len()
        )
        .into());
    }
    Ok((maximum_absolute_error, maximum_relative_error))
}

fn read_data_lines(
    path: &Path,
    mut visit: impl FnMut(&str, usize) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let mut input = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut line = String::new();
    let mut line_number = 0usize;
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        line_number += 1;
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            visit(line, line_number)?;
        }
    }
    Ok(())
}

fn percentile(samples: &[Duration], fraction: f64) -> Duration {
    samples[((samples.len() - 1) as f64 * fraction).round() as usize]
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
