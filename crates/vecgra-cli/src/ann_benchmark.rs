use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Bound;
use std::path::Path;
use std::time::{Duration, Instant};
use vecgra::{
    BulkLoader, Database, DatabaseOptions, ElementRef, ElementSet, NumericRangeFilter,
    NumericValue, Similarity, Value, VectorEncoding, VectorHit, VectorTarget,
};

pub(crate) fn import_fbin(input: &Path, database: &Path) -> Result<(), Box<dyn Error>> {
    let (mut input, count, dimension) = open_matrix(input, 4)?;
    let mut bulk = BulkLoader::new(
        database,
        DatabaseOptions {
            vector_dimension: dimension,
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F16,
            sync_on_commit: true,
        },
    )?;
    let mut encoded = vec![0u8; dimension * 4];
    let mut vector = vec![0.0f32; dimension];
    let started = Instant::now();
    for row in 0..count {
        input.read_exact(&mut encoded)?;
        for (value, bytes) in vector.iter_mut().zip(encoded.chunks_exact(4)) {
            *value = f32::from_le_bytes(bytes.try_into().unwrap());
        }
        bulk.create_node(
            "Vector",
            std::iter::empty::<(&str, Value)>(),
            std::slice::from_ref(&vector),
        )?;
        if (row + 1) % 100_000 == 0 {
            eprintln!("stored {}/{} vectors", row + 1, count);
        }
    }
    let stats = bulk.finish()?;
    println!("database\t{}", database.display());
    println!("vectors\t{}", stats.indexed_vectors);
    println!("dimension\t{dimension}");
    println!("elapsed_ms\t{}", started.elapsed().as_millis());
    Ok(())
}

pub(crate) fn benchmark_fbin(
    database_path: &Path,
    queries_path: &Path,
    neighbors_path: &Path,
    query_count: usize,
    candidate_vectors: usize,
    k: usize,
    warm_f32: bool,
) -> Result<(), Box<dyn Error>> {
    if query_count == 0 || candidate_vectors == 0 || k == 0 {
        return Err("query count, candidate vectors, and k must be greater than zero".into());
    }
    let database = Database::open(database_path)?;
    let (mut queries, available_queries, dimension) = open_matrix(queries_path, 4)?;
    let (mut neighbors, neighbor_rows, ground_truth_k) = open_matrix(neighbors_path, 4)?;
    if dimension != database.vector_dimension() {
        return Err(format!(
            "query dimension {dimension} does not match database dimension {}",
            database.vector_dimension()
        )
        .into());
    }
    if neighbor_rows != available_queries || k > ground_truth_k {
        return Err("ground-truth matrix shape does not match queries/k".into());
    }
    let query_count = query_count.min(available_queries);
    let mut query_vectors = Vec::with_capacity(query_count);
    let mut truth = Vec::with_capacity(query_count);
    let mut query_bytes = vec![0u8; dimension * 4];
    let mut truth_bytes = vec![0u8; ground_truth_k * 4];
    for _ in 0..query_count {
        queries.read_exact(&mut query_bytes)?;
        query_vectors.push(
            query_bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
        );
        neighbors.read_exact(&mut truth_bytes)?;
        truth.push(
            truth_bytes
                .chunks_exact(4)
                .take(k)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) as u64)
                .collect::<Vec<_>>(),
        );
    }

    let read = database.read();
    let warmed_bytes = if warm_f32 {
        read.warm_vector_cache()?
    } else {
        0
    };
    // Fault in the persisted sketch and code paths before timed single-query
    // measurements. Exact ground truth below still comes from every vector.
    let _ = read.vector_search_approximate(
        &query_vectors[0],
        VectorTarget::Nodes,
        k,
        None,
        candidate_vectors,
    )?;
    let mut approximate_times = Vec::with_capacity(query_count);
    let mut exact_times = Vec::with_capacity(query_count);
    let mut approximate_vs_official = 0.0;
    let mut approximate_vs_exact = 0.0;
    let mut exact_vs_official = 0.0;
    for (query, official) in query_vectors.iter().zip(&truth) {
        let started = Instant::now();
        let approximate =
            read.vector_search_approximate(query, VectorTarget::Nodes, k, None, candidate_vectors)?;
        approximate_times.push(started.elapsed());
        let started = Instant::now();
        let exact = read.vector_search(query, VectorTarget::Nodes, k, None)?;
        exact_times.push(started.elapsed());
        approximate_vs_official += recall_ids(&approximate, official);
        approximate_vs_exact += recall_hits(&approximate, &exact);
        exact_vs_official += recall_ids(&exact, official);
    }
    approximate_times.sort_unstable();
    exact_times.sort_unstable();
    println!("queries\t{query_count}");
    println!("k\t{k}");
    println!("candidate_vectors\t{candidate_vectors}");
    println!("f32_cache_bytes\t{warmed_bytes}");
    println!("f32_cache_bytes_after\t{}", read.vector_cache_bytes());
    println!(
        "approx_recall_vs_official\t{:.4}",
        approximate_vs_official / query_count as f64
    );
    println!(
        "approx_recall_vs_exact\t{:.4}",
        approximate_vs_exact / query_count as f64
    );
    println!(
        "exact_recall_vs_official\t{:.4}",
        exact_vs_official / query_count as f64
    );
    println!(
        "approximate_p50_ms\t{:.3}",
        millis(percentile(&approximate_times, 0.50))
    );
    println!(
        "approximate_p95_ms\t{:.3}",
        millis(percentile(&approximate_times, 0.95))
    );
    println!(
        "exact_p50_ms\t{:.3}",
        millis(percentile(&exact_times, 0.50))
    );
    println!(
        "exact_p95_ms\t{:.3}",
        millis(percentile(&exact_times, 0.95))
    );
    Ok(())
}

pub(crate) fn benchmark_filtered_fbin(
    database_path: &Path,
    queries_path: &Path,
    query_count: usize,
    stride: usize,
    candidate_elements: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    if query_count == 0 || stride == 0 || candidate_elements == 0 || k == 0 {
        return Err(
            "query count, filter stride, candidate elements, and k must be greater than zero"
                .into(),
        );
    }
    let database = Database::open(database_path)?;
    let (mut queries, available_queries, dimension) = open_matrix(queries_path, 4)?;
    if dimension != database.vector_dimension() {
        return Err(format!(
            "query dimension {dimension} does not match database dimension {}",
            database.vector_dimension()
        )
        .into());
    }
    let query_count = query_count.min(available_queries);
    let mut query_vectors = Vec::with_capacity(query_count);
    let mut query_bytes = vec![0u8; dimension * 4];
    for _ in 0..query_count {
        queries.read_exact(&mut query_bytes)?;
        query_vectors.push(
            query_bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
        );
    }

    let read = database.read();
    let mut allowed = ElementSet::new();
    for id in (0..read.stats().nodes as u64).step_by(stride) {
        allowed.insert(ElementRef::Node(id));
    }
    let plan = read.vector_search_within_plan(&allowed);
    let _ =
        read.vector_search_within_approximate(&query_vectors[0], &allowed, k, candidate_elements)?;
    let mut approximate_times = Vec::with_capacity(query_count);
    let mut exact_times = Vec::with_capacity(query_count);
    let mut recall = 0.0;
    for query in &query_vectors {
        let started = Instant::now();
        let approximate =
            read.vector_search_within_approximate(query, &allowed, k, candidate_elements)?;
        approximate_times.push(started.elapsed());
        let started = Instant::now();
        let exact = read.vector_search_within(query, &allowed, k)?;
        exact_times.push(started.elapsed());
        recall += recall_hits(&approximate, &exact);
    }
    approximate_times.sort_unstable();
    exact_times.sort_unstable();
    println!("queries\t{query_count}");
    println!("k\t{k}");
    println!("filter_stride\t{stride}");
    println!("allowed_elements\t{}", allowed.len());
    println!("candidate_elements\t{candidate_elements}");
    println!("adaptive_strategy\t{:?}", plan.strategy);
    println!("adaptive_candidates\t{}", plan.candidate_vectors);
    println!("approx_recall_vs_exact\t{:.4}", recall / query_count as f64);
    println!(
        "approximate_p50_ms\t{:.3}",
        millis(percentile(&approximate_times, 0.50))
    );
    println!(
        "approximate_p95_ms\t{:.3}",
        millis(percentile(&approximate_times, 0.95))
    );
    println!(
        "exact_p50_ms\t{:.3}",
        millis(percentile(&exact_times, 0.50))
    );
    println!(
        "exact_p95_ms\t{:.3}",
        millis(percentile(&exact_times, 0.95))
    );
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark controls are intentionally explicit command-line dimensions"
)]
pub(crate) fn benchmark_range_fbin(
    database_path: &Path,
    queries_path: &Path,
    neighbors_path: &Path,
    property_name: &str,
    lower_bound: f64,
    query_count: usize,
    candidate_elements: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    if query_count == 0 || candidate_elements == 0 || k == 0 || lower_bound.is_nan() {
        return Err(
            "query count, candidate elements, and k must be positive; the bound may not be NaN"
                .into(),
        );
    }
    let database = Database::open(database_path)?;
    let (mut queries, available_queries, dimension) = open_matrix(queries_path, 4)?;
    let (mut neighbors, neighbor_rows, ground_truth_k) = open_matrix(neighbors_path, 4)?;
    if dimension != database.vector_dimension() {
        return Err(format!(
            "query dimension {dimension} does not match database dimension {}",
            database.vector_dimension()
        )
        .into());
    }
    if neighbor_rows != available_queries || k > ground_truth_k {
        return Err("ground-truth matrix shape does not match queries/k".into());
    }
    let query_count = query_count.min(available_queries);
    let mut query_vectors = Vec::with_capacity(query_count);
    let mut truth = Vec::with_capacity(query_count);
    let mut query_bytes = vec![0u8; dimension * 4];
    let mut truth_bytes = vec![0u8; ground_truth_k * 4];
    for _ in 0..query_count {
        queries.read_exact(&mut query_bytes)?;
        query_vectors.push(
            query_bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
        );
        neighbors.read_exact(&mut truth_bytes)?;
        truth.push(
            truth_bytes
                .chunks_exact(4)
                .take(k)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) as u64)
                .collect::<Vec<_>>(),
        );
    }

    let read = database.read();
    let property = read
        .label_id(property_name)
        .ok_or_else(|| format!("unknown property key {property_name:?}"))?;
    let filter = NumericRangeFilter {
        label: None,
        key: property,
        lower: Bound::Included(NumericValue::Float(lower_bound)),
        upper: Bound::Unbounded,
    };
    let filter_plan = read.numeric_range_plan(VectorTarget::Nodes, &filter)?;
    let filter_started = Instant::now();
    let allowed = read.elements_matching_numeric_range(VectorTarget::Nodes, &filter)?;
    let first_filter_time = filter_started.elapsed();
    if allowed.is_empty() {
        return Err("numeric range selected no nodes".into());
    }
    let adaptive_plan = read.vector_search_within_plan(&allowed);
    let _ =
        read.vector_search_within_approximate(&query_vectors[0], &allowed, k, candidate_elements)?;

    let mut filter_times = Vec::with_capacity(query_count);
    let mut adaptive_total_times = Vec::with_capacity(query_count);
    let mut approximate_times = Vec::with_capacity(query_count);
    let mut exact_times = Vec::with_capacity(query_count);
    let mut adaptive_vs_official = 0.0;
    let mut approximate_vs_official = 0.0;
    let mut exact_vs_official = 0.0;
    let mut approximate_vs_exact = 0.0;
    for (query, official) in query_vectors.iter().zip(&truth) {
        let started = Instant::now();
        let current = read.elements_matching_numeric_range(VectorTarget::Nodes, &filter)?;
        filter_times.push(started.elapsed());
        let started = Instant::now();
        let adaptive = read.vector_search_filtered_adaptive(
            query,
            VectorTarget::Nodes,
            k,
            None,
            std::slice::from_ref(&filter),
        )?;
        debug_assert_eq!(adaptive.candidate_elements, current.len());
        adaptive_total_times.push(started.elapsed());

        let started = Instant::now();
        let approximate =
            read.vector_search_within_approximate(query, &allowed, k, candidate_elements)?;
        approximate_times.push(started.elapsed());
        let started = Instant::now();
        let exact = read.vector_search_within(query, &allowed, k)?;
        exact_times.push(started.elapsed());

        adaptive_vs_official += recall_ids(&adaptive.hits, official);
        approximate_vs_official += recall_ids(&approximate, official);
        exact_vs_official += recall_ids(&exact, official);
        approximate_vs_exact += recall_hits(&approximate, &exact);
    }
    filter_times.sort_unstable();
    adaptive_total_times.sort_unstable();
    approximate_times.sort_unstable();
    exact_times.sort_unstable();

    println!("queries\t{query_count}");
    println!("k\t{k}");
    println!("property\t{property_name}");
    println!("lower_bound_inclusive\t{lower_bound}");
    println!("allowed_elements\t{}", allowed.len());
    println!("filter_strategy\t{:?}", filter_plan.strategy);
    println!(
        "filter_candidate_upper_bound\t{}",
        filter_plan.candidate_upper_bound
    );
    println!("filter_first_ms\t{:.3}", millis(first_filter_time));
    println!(
        "filter_p50_ms\t{:.3}",
        millis(percentile(&filter_times, 0.50))
    );
    println!("adaptive_strategy\t{:?}", adaptive_plan.strategy);
    println!("adaptive_candidates\t{}", adaptive_plan.candidate_vectors);
    println!("explicit_candidate_elements\t{candidate_elements}");
    println!(
        "adaptive_recall_vs_official\t{:.4}",
        adaptive_vs_official / query_count as f64
    );
    println!(
        "approx_recall_vs_official\t{:.4}",
        approximate_vs_official / query_count as f64
    );
    println!(
        "approx_recall_vs_exact\t{:.4}",
        approximate_vs_exact / query_count as f64
    );
    println!(
        "exact_recall_vs_official\t{:.4}",
        exact_vs_official / query_count as f64
    );
    println!(
        "adaptive_total_p50_ms\t{:.3}",
        millis(percentile(&adaptive_total_times, 0.50))
    );
    println!(
        "adaptive_total_p95_ms\t{:.3}",
        millis(percentile(&adaptive_total_times, 0.95))
    );
    println!(
        "approximate_p50_ms\t{:.3}",
        millis(percentile(&approximate_times, 0.50))
    );
    println!(
        "approximate_p95_ms\t{:.3}",
        millis(percentile(&approximate_times, 0.95))
    );
    println!(
        "exact_p50_ms\t{:.3}",
        millis(percentile(&exact_times, 0.50))
    );
    println!(
        "exact_p95_ms\t{:.3}",
        millis(percentile(&exact_times, 0.95))
    );
    Ok(())
}

pub(crate) fn open_matrix(
    path: &Path,
    bytes_per_value: usize,
) -> Result<(BufReader<File>, usize, usize), Box<dyn Error>> {
    let file = File::open(path)?;
    let file_len = usize::try_from(file.metadata()?.len())?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    let rows = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    let columns = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
    let expected = rows
        .checked_mul(columns)
        .and_then(|values| values.checked_mul(bytes_per_value))
        .and_then(|bytes| bytes.checked_add(8))
        .ok_or("matrix file length overflow")?;
    if file_len != expected {
        return Err(format!(
            "{} has {file_len} bytes; header requires {expected}",
            path.display()
        )
        .into());
    }
    Ok((reader, rows, columns))
}

fn recall_ids(hits: &[VectorHit], truth: &[u64]) -> f64 {
    let truth: HashSet<_> = truth.iter().copied().collect();
    hits.iter()
        .filter(|hit| match hit.element {
            ElementRef::Node(id) => truth.contains(&id),
            ElementRef::Edge(_) => false,
        })
        .count() as f64
        / truth.len() as f64
}

fn recall_hits(left: &[VectorHit], right: &[VectorHit]) -> f64 {
    let right: HashSet<_> = right.iter().map(|hit| hit.element).collect();
    left.iter()
        .filter(|hit| right.contains(&hit.element))
        .count() as f64
        / right.len() as f64
}

fn percentile(samples: &[Duration], percentile: f64) -> Duration {
    samples[((samples.len() - 1) as f64 * percentile).ceil() as usize]
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
