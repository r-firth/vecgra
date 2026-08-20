use crate::{ann_benchmark, embedder, exporter, github, graph_benchmark, importer, jsonl, query};
use std::env;
use std::error::Error;
use std::ops::Bound;
use std::path::Path;
use std::time::{Duration, Instant};
use vecgra::{
    Database, Direction, EdgeFilter, ElementFilter, ElementRef, ElementSet,
    GraphRangeSearchOptions, NumericRangeFilter, NumericValue, SemanticPathOptions,
    ShortestPathOptions, Value, VectorEncoding, VectorTarget,
};

pub(crate) fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1).peekable();
    let Some(command) = arguments.next() else {
        print_help();
        return Ok(());
    };
    if matches!(command.as_str(), "help" | "--help" | "-h") {
        print_help();
        return Ok(());
    }
    if matches!(command.as_str(), "--version" | "-V") {
        println!("vecgra {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if arguments
        .peek()
        .is_some_and(|argument| matches!(argument.as_str(), "help" | "--help" | "-h"))
    {
        arguments.next();
        if let Some(extra) = arguments.next() {
            return Err(format!("unexpected argument {extra:?} after help").into());
        }
        print_command_help(&command)?;
        return Ok(());
    }
    let handled = run_import_command(&command, &mut arguments)?
        || run_benchmark_command(&command, &mut arguments)?
        || run_graph_command(&command, &mut arguments)?;
    if handled {
        if let Some(extra) = arguments.next() {
            return Err(format!("unexpected argument {extra:?} for command {command:?}").into());
        }
        return Ok(());
    }
    Err(format!("unknown command {command}").into())
}

fn run_import_command(
    command: &str,
    mut arguments: &mut impl Iterator<Item = String>,
) -> Result<bool, Box<dyn Error>> {
    match command {
        "import-jsonl" => {
            let nodes = required(&mut arguments, "nodes JSONL path")?;
            let edges = required(&mut arguments, "edges JSONL path")?;
            let database = required(&mut arguments, "database path")?;
            let dimension = required(&mut arguments, "vector dimension")?.parse::<usize>()?;
            let encoding = match arguments.next().as_deref().unwrap_or("f16") {
                "f16" => VectorEncoding::F16,
                "f32" => VectorEncoding::F32,
                value => return Err(format!("unknown vector encoding {value:?}").into()),
            };
            let stats = jsonl::import_jsonl(
                Path::new(&nodes),
                Path::new(&edges),
                Path::new(&database),
                dimension,
                encoding,
            )?;
            println!("nodes\t{}", stats.nodes);
            println!("edges\t{}", stats.edges);
            println!("vectors\t{}", stats.indexed_vectors);
        }
        "import-fbin" => {
            let input = required(&mut arguments, "input fbin path")?;
            let database = required(&mut arguments, "database path")?;
            ann_benchmark::import_fbin(Path::new(&input), Path::new(&database))?;
        }
        "import-node-fbin" => {
            let vectors = required(&mut arguments, "input fbin path")?;
            let metadata = required(&mut arguments, "node metadata JSONL path")?;
            let database = required(&mut arguments, "database path")?;
            let encoding = match arguments.next().as_deref().unwrap_or("f16") {
                "f16" => VectorEncoding::F16,
                "f32" => VectorEncoding::F32,
                value => return Err(format!("unknown vector encoding {value:?}").into()),
            };
            let stats = jsonl::import_node_fbin(
                Path::new(&vectors),
                Path::new(&metadata),
                Path::new(&database),
                encoding,
            )?;
            println!("nodes\t{}", stats.nodes);
            println!("vectors\t{}", stats.indexed_vectors);
            println!(
                "dimension\t{}",
                Database::open(&database)?.vector_dimension()
            );
        }
        "import-graphalytics" => {
            let vertices = required(&mut arguments, "vertex file")?;
            let edges = required(&mut arguments, "edge file")?;
            let database = required(&mut arguments, "database path")?;
            graph_benchmark::import_graphalytics(
                Path::new(&vertices),
                Path::new(&edges),
                Path::new(&database),
            )?;
        }
        "import-rust" => {
            let repository = required(&mut arguments, "repository path")?;
            let database = required(&mut arguments, "database path")?;
            let dimension = optional_usize(&mut arguments, "vector dimension")?.unwrap_or(256);
            let embedder_name = arguments
                .next()
                .or_else(|| env::var("VECGRA_EMBEDDER").ok())
                .unwrap_or_else(|| "hash".into());
            let batch_size = optional_usize(&mut arguments, "embedding batch size")?.unwrap_or(128);
            let embedder = embedder::create_embedder(&embedder_name, dimension, batch_size)?;
            importer::import_rust_repository(
                Path::new(&repository),
                Path::new(&database),
                embedder,
            )?;
        }
        "import-github" => {
            let repository = required(&mut arguments, "GitHub owner/repository")?;
            let database = required(&mut arguments, "database path")?;
            let max_issues = optional_usize(&mut arguments, "maximum issues")?.unwrap_or(1_000);
            let max_pull_requests =
                optional_usize(&mut arguments, "maximum pull requests")?.unwrap_or(1_000);
            let max_discussions =
                optional_usize(&mut arguments, "maximum discussions")?.unwrap_or(300);
            let max_releases = optional_usize(&mut arguments, "maximum releases")?.unwrap_or(100);
            let dimension = optional_usize(&mut arguments, "vector dimension")?.unwrap_or(256);
            let embedder_name = arguments
                .next()
                .or_else(|| env::var("VECGRA_EMBEDDER").ok())
                .unwrap_or_else(|| "hash".into());
            let batch_size = optional_usize(&mut arguments, "embedding batch size")?.unwrap_or(128);
            let embedder = embedder::create_embedder(&embedder_name, dimension, batch_size)?;
            github::import_github_repository(
                &repository,
                Path::new(&database),
                github::GithubImportLimits {
                    issues: max_issues,
                    pull_requests: max_pull_requests,
                    discussions: max_discussions,
                    releases: max_releases,
                },
                embedder,
            )?;
        }
        "compact" => {
            let source = required(&mut arguments, "source database")?;
            let destination = required(&mut arguments, "destination database")?;
            let encoding = match arguments.next().as_deref() {
                None | Some("f16") => VectorEncoding::F16,
                Some("f32") => VectorEncoding::F32,
                Some(other) => return Err(format!("unknown vector encoding {other:?}").into()),
            };
            let database = Database::open(source)?;
            let stats = database.compact_to(&destination, encoding)?;
            println!("database\t{destination}");
            println!("nodes\t{}", stats.nodes);
            println!("edges\t{}", stats.edges);
            println!("vectors\t{}", stats.indexed_vectors);
            println!("vector_encoding\t{encoding:?}");
        }
        "export-ladybug" => {
            let database = required(&mut arguments, "database path")?;
            let directory = required(&mut arguments, "output directory")?;
            exporter::export_ladybug_csv(Path::new(&database), Path::new(&directory))?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn run_benchmark_command(
    command: &str,
    mut arguments: &mut impl Iterator<Item = String>,
) -> Result<bool, Box<dyn Error>> {
    match command {
        "bench-bfs" => {
            let database = required(&mut arguments, "database path")?;
            let source = required(&mut arguments, "source node")?.parse::<u64>()?;
            let expected = arguments.next().filter(|path| path != "-");
            let iterations = optional_usize(&mut arguments, "iterations")?.unwrap_or(5);
            graph_benchmark::benchmark_bfs(
                Path::new(&database),
                source,
                expected.as_deref().map(Path::new),
                iterations,
            )?;
        }
        "bench-wcc" => {
            let database = required(&mut arguments, "database path")?;
            let expected = arguments.next().filter(|path| path != "-");
            let iterations = optional_usize(&mut arguments, "iterations")?.unwrap_or(3);
            graph_benchmark::benchmark_wcc(
                Path::new(&database),
                expected.as_deref().map(Path::new),
                iterations,
            )?;
        }
        "bench-pagerank" => {
            let database = required(&mut arguments, "database path")?;
            let expected = arguments.next().filter(|path| path != "-");
            let iterations = optional_usize(&mut arguments, "benchmark iterations")?.unwrap_or(2);
            graph_benchmark::benchmark_pagerank(
                Path::new(&database),
                expected.as_deref().map(Path::new),
                iterations,
            )?;
        }
        "bench-fbin" => {
            let database = required(&mut arguments, "database path")?;
            let queries = required(&mut arguments, "query fbin path")?;
            let neighbors = required(&mut arguments, "ground-truth ibin path")?;
            let query_count = optional_usize(&mut arguments, "query count")?.unwrap_or(100);
            let candidate_vectors =
                optional_usize(&mut arguments, "candidate vectors")?.unwrap_or(12_000);
            let k = optional_usize(&mut arguments, "k")?.unwrap_or(10);
            let warm_f32 = match arguments.next().as_deref() {
                None | Some("compressed") => false,
                Some("hot-f32") => true,
                Some(other) => return Err(format!("unknown vector cache mode {other:?}").into()),
            };
            ann_benchmark::benchmark_fbin(
                Path::new(&database),
                Path::new(&queries),
                Path::new(&neighbors),
                query_count,
                candidate_vectors,
                k,
                warm_f32,
            )?;
        }
        "bench-filtered-fbin" => {
            let database = required(&mut arguments, "database path")?;
            let queries = required(&mut arguments, "query fbin path")?;
            let query_count = optional_usize(&mut arguments, "query count")?.unwrap_or(100);
            let stride = optional_usize(&mut arguments, "filter stride")?.unwrap_or(2);
            let candidate_elements =
                optional_usize(&mut arguments, "candidate elements")?.unwrap_or(20_000);
            let k = optional_usize(&mut arguments, "k")?.unwrap_or(10);
            ann_benchmark::benchmark_filtered_fbin(
                Path::new(&database),
                Path::new(&queries),
                query_count,
                stride,
                candidate_elements,
                k,
            )?;
        }
        "bench-range-fbin" => {
            let database = required(&mut arguments, "database path")?;
            let queries = required(&mut arguments, "query fbin path")?;
            let neighbors = required(&mut arguments, "ground-truth ibin path")?;
            let property = required(&mut arguments, "numeric property key")?;
            let lower_bound =
                required(&mut arguments, "inclusive floating lower bound")?.parse::<f64>()?;
            let query_count = optional_usize(&mut arguments, "query count")?.unwrap_or(100);
            let candidate_elements =
                optional_usize(&mut arguments, "candidate elements")?.unwrap_or(20_000);
            let k = optional_usize(&mut arguments, "k")?.unwrap_or(10);
            ann_benchmark::benchmark_range_fbin(
                Path::new(&database),
                Path::new(&queries),
                Path::new(&neighbors),
                &property,
                lower_bound,
                query_count,
                candidate_elements,
                k,
            )?;
        }
        "bench-property" => {
            let path = required(&mut arguments, "database path")?;
            let target_name = required(&mut arguments, "nodes, edges, or both")?;
            let target = parse_vector_target(Some(&target_name))?;
            let key_name = required(&mut arguments, "property key")?;
            let value = parse_json_scalar(&required(&mut arguments, "JSON scalar value")?)?;
            let iterations = optional_usize(&mut arguments, "iterations")?.unwrap_or(1_000);
            if iterations == 0 {
                return Err("iterations must be greater than zero".into());
            }
            let open_started = Instant::now();
            let database = Database::open(path)?;
            let open_time = open_started.elapsed();
            let read = database.read();
            let key = read
                .label_id(&key_name)
                .ok_or_else(|| format!("unknown property key {key_name:?}"))?;
            let filter = ElementFilter {
                label: None,
                properties: vec![(key, value)],
            };
            let plan = read.element_filter_plan(target, &filter);
            for _ in 0..5 {
                std::hint::black_box(read.elements_matching(target, &filter));
            }
            let mut samples = Vec::with_capacity(iterations);
            let mut result_count = 0u64;
            for _ in 0..iterations {
                let started = Instant::now();
                let result = read.elements_matching(target, &filter);
                samples.push(started.elapsed());
                result_count = result.len();
                std::hint::black_box(result);
            }
            samples.sort_unstable();
            println!("open_ms\t{:.3}", millis(open_time));
            println!("strategy\t{:?}", plan.strategy);
            println!("candidate_upper_bound\t{}", plan.candidate_upper_bound);
            println!("iterations\t{iterations}");
            println!("result_count\t{result_count}");
            println!("query_min_ms\t{:.6}", millis(samples[0]));
            println!("query_p50_ms\t{:.6}", millis(percentile(&samples, 0.50)));
            println!("query_p95_ms\t{:.6}", millis(percentile(&samples, 0.95)));
            println!("query_max_ms\t{:.6}", millis(*samples.last().unwrap()));
        }
        "bench-search" => {
            let path = required(&mut arguments, "database path")?;
            let query = required(&mut arguments, "query text")?;
            let iterations = optional_usize(&mut arguments, "iterations")?.unwrap_or(25);
            if iterations == 0 {
                return Err("iterations must be greater than zero".into());
            }
            let embedder_name = arguments
                .next()
                .or_else(|| env::var("VECGRA_EMBEDDER").ok())
                .unwrap_or_else(|| "hash".into());
            let target = parse_vector_target(arguments.next().as_deref())?;
            let label_name = arguments.next().filter(|name| name != "-");
            let candidate_vectors =
                optional_usize(&mut arguments, "approximate candidate vectors")?;
            if candidate_vectors == Some(0) {
                return Err("approximate candidate vectors must be greater than zero".into());
            }

            let open_started = Instant::now();
            let database = Database::open(path)?;
            let open_time = open_started.elapsed();
            let embedding_started = Instant::now();
            let mut embedder =
                embedder::create_embedder(&embedder_name, database.vector_dimension(), 1)?;
            let vector = embedder.embed_query(&query)?;
            let embedding_time = embedding_started.elapsed();
            let read = database.read();
            let label = label_name
                .as_deref()
                .map(|name| {
                    read.label_id(name)
                        .ok_or_else(|| format!("unknown label {name:?}"))
                })
                .transpose()?;
            let first_search_started = Instant::now();
            let first_hits = if let Some(candidates) = candidate_vectors {
                read.vector_search_approximate(&vector, target, 10, label, candidates)?
            } else {
                read.vector_search(&vector, target, 10, label)?
            };
            let first_search_time = first_search_started.elapsed();
            for _ in 0..2 {
                if let Some(candidates) = candidate_vectors {
                    let _ =
                        read.vector_search_approximate(&vector, target, 10, label, candidates)?;
                } else {
                    let _ = read.vector_search(&vector, target, 10, label)?;
                }
            }
            let mut samples = Vec::with_capacity(iterations);
            let mut result_count = 0;
            for _ in 0..iterations {
                let started = Instant::now();
                let hits = if let Some(candidates) = candidate_vectors {
                    read.vector_search_approximate(&vector, target, 10, label, candidates)?
                } else {
                    read.vector_search(&vector, target, 10, label)?
                };
                samples.push(started.elapsed());
                result_count = hits.len();
            }
            samples.sort_unstable();
            // Evaluate ground truth after timing so an exact
            // full scan cannot warm or promote the vector tier used by ANN.
            let exact_hits = candidate_vectors
                .map(|_| read.vector_search(&vector, target, 10, label))
                .transpose()?;
            println!("open_ms\t{:.3}", millis(open_time));
            println!("query_embedding_ms\t{:.3}", millis(embedding_time));
            println!("first_search_ms\t{:.3}", millis(first_search_time));
            println!("iterations\t{iterations}");
            println!("target\t{target:?}");
            println!("label\t{}", label_name.as_deref().unwrap_or("*"));
            println!(
                "mode\t{}",
                if candidate_vectors.is_some() {
                    "approximate"
                } else {
                    "exact"
                }
            );
            if let Some(candidates) = candidate_vectors {
                println!("candidate_vectors\t{candidates}");
                let exact_hits = exact_hits
                    .as_deref()
                    .ok_or("exact recall baseline was not computed")?;
                println!("recall_at_10\t{:.3}", recall(exact_hits, &first_hits));
            }
            println!("result_count\t{}", result_count.max(first_hits.len()));
            println!("search_min_ms\t{:.3}", millis(samples[0]));
            println!("search_p50_ms\t{:.3}", millis(percentile(&samples, 0.50)));
            println!("search_p95_ms\t{:.3}", millis(percentile(&samples, 0.95)));
            println!("search_max_ms\t{:.3}", millis(*samples.last().unwrap()));
        }
        "bench-ann" => {
            let path = required(&mut arguments, "database path")?;
            let query_count = optional_usize(&mut arguments, "query count")?.unwrap_or(20);
            let candidate_vectors =
                optional_usize(&mut arguments, "approximate candidate vectors")?.unwrap_or(20_000);
            let target = parse_vector_target(arguments.next().as_deref())?;
            if query_count == 0 || candidate_vectors == 0 {
                return Err("query count and candidate vectors must be greater than zero".into());
            }
            let open_started = Instant::now();
            let database = Database::open(path)?;
            let open_time = open_started.elapsed();
            let read = database.read();
            let mut elements = Vec::new();
            if matches!(target, VectorTarget::Nodes | VectorTarget::Both) {
                elements.extend(read.node_ids().into_iter().map(ElementRef::Node));
            }
            if matches!(target, VectorTarget::Edges | VectorTarget::Both) {
                elements.extend(read.edge_ids().into_iter().map(ElementRef::Edge));
            }
            if elements.is_empty() {
                return Err("database has no elements for the requested target".into());
            }
            let stride = (elements.len() / query_count).max(1);
            let mut queries = Vec::with_capacity(query_count);
            let mut cursor = stride / 2;
            while queries.len() < query_count && cursor < elements.len() {
                let primary = element_vector_owned(&read, elements[cursor])?;
                let secondary = element_vector_owned(
                    &read,
                    elements[(cursor + stride / 3 + 1) % elements.len()],
                )?;
                if let Some(mut query) = primary {
                    if let Some(secondary) = secondary {
                        for (left, right) in query.iter_mut().zip(secondary) {
                            *left = *left * 0.85 + right * 0.15;
                        }
                    }
                    queries.push(query);
                }
                cursor = cursor.saturating_add(stride);
            }
            if queries.is_empty() {
                return Err("sampled elements do not have vectors".into());
            }

            let build_started = Instant::now();
            let _ =
                read.vector_search_approximate(&queries[0], target, 10, None, candidate_vectors)?;
            let build_time = build_started.elapsed();
            let mut approximate_times = Vec::with_capacity(queries.len());
            let mut exact_times = Vec::with_capacity(queries.len());
            let mut recalls = Vec::with_capacity(queries.len());
            for query in &queries {
                let started = Instant::now();
                let approximate =
                    read.vector_search_approximate(query, target, 10, None, candidate_vectors)?;
                approximate_times.push(started.elapsed());
                let started = Instant::now();
                let exact = read.vector_search(query, target, 10, None)?;
                exact_times.push(started.elapsed());
                recalls.push(recall(&exact, &approximate));
            }
            approximate_times.sort_unstable();
            exact_times.sort_unstable();
            recalls.sort_by(f64::total_cmp);
            println!("open_ms\t{:.3}", millis(open_time));
            println!("index_build_ms\t{:.3}", millis(build_time));
            println!("queries\t{}", queries.len());
            println!("target\t{target:?}");
            println!("candidate_vectors\t{candidate_vectors}");
            println!("recall_at_10_min\t{:.3}", recalls[0]);
            println!(
                "recall_at_10_mean\t{:.3}",
                recalls.iter().sum::<f64>() / recalls.len() as f64
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
        }
        "bench-pattern" => {
            let path = required(&mut arguments, "database path")?;
            let statement = required(&mut arguments, "query statement")?;
            let iterations = optional_usize(&mut arguments, "iterations")?.unwrap_or(100);
            if iterations == 0 {
                return Err("iterations must be greater than zero".into());
            }
            let open_started = Instant::now();
            let database = Database::open(path)?;
            let open_time = open_started.elapsed();
            let read = database.read();
            for _ in 0..3 {
                let _ = query::execute(&read, &statement)?;
            }
            let mut samples = Vec::with_capacity(iterations);
            let mut result_count = 0;
            for _ in 0..iterations {
                let started = Instant::now();
                result_count = query::execute(&read, &statement)?.len();
                samples.push(started.elapsed());
            }
            samples.sort_unstable();
            println!("open_ms\t{:.3}", millis(open_time));
            println!("iterations\t{iterations}");
            println!("result_count\t{result_count}");
            println!("query_min_ms\t{:.3}", millis(samples[0]));
            println!("query_p50_ms\t{:.3}", millis(percentile(&samples, 0.50)));
            println!("query_p95_ms\t{:.3}", millis(percentile(&samples, 0.95)));
            println!("query_max_ms\t{:.3}", millis(*samples.last().unwrap()));
        }
        "bench-neighbors" => {
            let path = required(&mut arguments, "database path")?;
            let id: u64 = required(&mut arguments, "node id")?.parse()?;
            let iterations = optional_usize(&mut arguments, "iterations")?.unwrap_or(1_000);
            if iterations == 0 {
                return Err("iterations must be greater than zero".into());
            }
            let direction = match arguments.next().as_deref() {
                None | Some("both") => Direction::Both,
                Some("out") => Direction::Outgoing,
                Some("in") => Direction::Incoming,
                Some(other) => return Err(format!("unknown direction {other}").into()),
            };
            let open_started = Instant::now();
            let database = Database::open(path)?;
            let open_time = open_started.elapsed();
            let read = database.read();
            for _ in 0..5 {
                let _ = read.neighbors(id, direction, EdgeFilter::default())?;
            }
            let mut samples = Vec::with_capacity(iterations);
            let mut result_count = 0;
            for _ in 0..iterations {
                let started = Instant::now();
                result_count = read.neighbors(id, direction, EdgeFilter::default())?.len();
                samples.push(started.elapsed());
            }
            samples.sort_unstable();
            println!("open_ms\t{:.3}", millis(open_time));
            println!("iterations\t{iterations}");
            println!("result_count\t{result_count}");
            println!("query_min_ms\t{:.6}", millis(samples[0]));
            println!("query_p50_ms\t{:.6}", millis(percentile(&samples, 0.50)));
            println!("query_p95_ms\t{:.6}", millis(percentile(&samples, 0.95)));
            println!("query_max_ms\t{:.6}", millis(*samples.last().unwrap()));
        }
        "bench-expand" => {
            let path = required(&mut arguments, "database path")?;
            let id: u64 = required(&mut arguments, "seed node id")?.parse()?;
            let hops: usize = required(&mut arguments, "maximum hops")?.parse()?;
            let iterations = optional_usize(&mut arguments, "iterations")?.unwrap_or(5);
            if iterations == 0 {
                return Err("iterations must be greater than zero".into());
            }
            let direction = match arguments.next().as_deref() {
                None | Some("out") => Direction::Outgoing,
                Some("in") => Direction::Incoming,
                Some("both") => Direction::Both,
                Some(other) => return Err(format!("unknown direction {other}").into()),
            };
            let database = Database::open(path)?;
            let read = database.read();
            let mut seeds = ElementSet::new();
            seeds.insert(ElementRef::Node(id));
            let mut samples = Vec::with_capacity(iterations);
            let mut node_only_samples = Vec::with_capacity(iterations);
            let mut node_count = 0u64;
            let mut edge_count = 0u64;
            for _ in 0..iterations {
                let started = Instant::now();
                let result =
                    read.expand_element_set_hops(&seeds, direction, EdgeFilter::default(), hops)?;
                samples.push(started.elapsed());
                node_count = result.node_len();
                edge_count = result.edge_len();
                std::hint::black_box(result);
            }
            for _ in 0..iterations {
                let started = Instant::now();
                let result = read.nodes_within_hops(
                    &seeds,
                    direction,
                    EdgeFilter::default(),
                    hops,
                    false,
                    None,
                )?;
                node_only_samples.push(started.elapsed());
                if result.node_len() != node_count || result.edge_len() != 0 {
                    return Err("node-only expansion disagrees with full expansion".into());
                }
                std::hint::black_box(result);
            }
            samples.sort_unstable();
            node_only_samples.sort_unstable();
            println!("seed\t{id}");
            println!("hops\t{hops}");
            println!("direction\t{direction:?}");
            println!("nodes\t{node_count}");
            println!("edges\t{edge_count}");
            println!("iterations\t{iterations}");
            println!("expand_min_ms\t{:.3}", millis(samples[0]));
            println!("expand_p50_ms\t{:.3}", millis(percentile(&samples, 0.50)));
            println!("expand_p95_ms\t{:.3}", millis(percentile(&samples, 0.95)));
            println!(
                "node_only_p50_ms\t{:.3}",
                millis(percentile(&node_only_samples, 0.50))
            );
            println!(
                "node_only_p95_ms\t{:.3}",
                millis(percentile(&node_only_samples, 0.95))
            );
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn run_graph_command(
    command: &str,
    mut arguments: &mut impl Iterator<Item = String>,
) -> Result<bool, Box<dyn Error>> {
    match command {
        "stats" => {
            let path = required(&mut arguments, "database path")?;
            let database = Database::open(path)?;
            let stats = database.read().stats();
            println!("path\t{}", database.path().display());
            println!("nodes\t{}", stats.nodes);
            println!("edges\t{}", stats.edges);
            println!("symbols\t{}", stats.labels);
            println!("vectors\t{}", stats.indexed_vectors);
            println!("transactions\t{}", stats.transactions);
            println!("dimension\t{}", database.vector_dimension());
            println!("similarity\t{:?}", database.similarity());
            println!("vector_encoding\t{:?}", database.vector_encoding());
        }
        "check" => {
            let path = required(&mut arguments, "database path")?;
            let started = Instant::now();
            let database = Database::open(path)?;
            let report = database.read().verify_integrity()?;
            println!("status\tok");
            println!("nodes\t{}", report.nodes);
            println!("edges\t{}", report.edges);
            println!("vectors\t{}", report.indexed_vectors);
            println!("transactions\t{}", report.transactions);
            println!("vector_bytes_verified\t{}", report.vector_bytes_verified);
            println!(
                "vector_checksum_blocks_verified\t{}",
                report.vector_checksum_blocks_verified
            );
            println!("elapsed_ms\t{:.3}", millis(started.elapsed()));
        }
        "plan-search" => {
            let path = required(&mut arguments, "database path")?;
            let target = parse_vector_target(arguments.next().as_deref())?;
            let label_name = arguments.next();
            let database = Database::open(path)?;
            let read = database.read();
            let label = label_name
                .as_deref()
                .map(|name| {
                    read.label_id(name)
                        .ok_or_else(|| format!("unknown label {name:?}"))
                })
                .transpose()?;
            let plan = read.vector_search_plan(target, label);
            println!("strategy\t{:?}", plan.strategy);
            println!("target\t{target:?}");
            println!("label\t{}", label_name.as_deref().unwrap_or("*"));
            println!("estimated_vectors\t{}", plan.estimated_vectors);
            println!("estimated_floats\t{}", plan.estimated_floats);
            println!("candidate_vectors\t{}", plan.candidate_vectors);
        }
        "numeric-range" => {
            let path = required(&mut arguments, "database path")?;
            let target_name = required(&mut arguments, "nodes, edges, or both")?;
            let target = parse_vector_target(Some(&target_name))?;
            let key_name = required(&mut arguments, "numeric property key")?;
            let numeric_type = required(&mut arguments, "int or float")?;
            let lower = parse_numeric_bound(
                &required(&mut arguments, "inclusive lower bound or -")?,
                &numeric_type,
            )?;
            let upper = parse_numeric_bound(
                &required(&mut arguments, "inclusive upper bound or -")?,
                &numeric_type,
            )?;
            let limit = optional_usize(&mut arguments, "output limit")?.unwrap_or(100);
            let database = Database::open(path)?;
            let read = database.read();
            let key = read
                .label_id(&key_name)
                .ok_or_else(|| format!("unknown property key {key_name:?}"))?;
            let filter = NumericRangeFilter {
                label: None,
                key,
                lower,
                upper,
            };
            let plan = read.numeric_range_plan(target, &filter)?;
            let started = Instant::now();
            let matches = read.elements_matching_numeric_range(target, &filter)?;
            println!("strategy\t{:?}", plan.strategy);
            println!("candidate_upper_bound\t{}", plan.candidate_upper_bound);
            println!("matches\t{}", matches.len());
            println!("elapsed_ms\t{:.6}", millis(started.elapsed()));
            for id in matches.node_ids().take(limit) {
                println!("node\t{id}");
            }
            let remaining = limit.saturating_sub(matches.node_len() as usize);
            for id in matches.edge_ids().take(remaining) {
                println!("edge\t{id}");
            }
        }
        "node" => {
            let path = required(&mut arguments, "database path")?;
            let id: u64 = required(&mut arguments, "node id")?.parse()?;
            let database = Database::open(path)?;
            let read = database.read();
            let node = read.node(id).ok_or("node not found")?;
            println!(
                "node {} :{}",
                node.id,
                read.symbol(node.label).unwrap_or("?")
            );
            for property in node.properties.iter() {
                println!(
                    "  {} = {}",
                    read.symbol(property.key).unwrap_or("?"),
                    display_value(&read, &property.value)
                );
            }
            println!("  vectors = {}", node.vector_count);
        }
        "neighbors" => {
            let path = required(&mut arguments, "database path")?;
            let id: u64 = required(&mut arguments, "node id")?.parse()?;
            let direction = match arguments.next().as_deref() {
                None | Some("out") => Direction::Outgoing,
                Some("in") => Direction::Incoming,
                Some("both") => Direction::Both,
                Some(other) => return Err(format!("unknown direction {other}").into()),
            };
            let database = Database::open(path)?;
            let read = database.read();
            for edge in read.neighbors(id, direction, EdgeFilter::default())? {
                println!(
                    "{}\t{} -[:{}]-> {}",
                    edge.id,
                    edge.source,
                    read.symbol(edge.label).unwrap_or("?"),
                    edge.target
                );
            }
        }
        "shortest-path" => {
            let path = required(&mut arguments, "database path")?;
            let start: u64 = required(&mut arguments, "start node id")?.parse()?;
            let end: u64 = required(&mut arguments, "end node id")?.parse()?;
            let max_hops = optional_usize(&mut arguments, "maximum hops")?.unwrap_or(6);
            let direction = match arguments.next().as_deref() {
                None | Some("both") => Direction::Both,
                Some("out") => Direction::Outgoing,
                Some("in") => Direction::Incoming,
                Some(other) => return Err(format!("unknown direction {other}").into()),
            };
            let edge_label_name = arguments.next().filter(|name| name != "-");
            let max_expansions =
                optional_usize(&mut arguments, "maximum expansions")?.unwrap_or(100_000);
            let database = Database::open(path)?;
            let read = database.read();
            let edge_label = edge_label_name
                .as_deref()
                .map(|name| {
                    read.label_id(name)
                        .ok_or_else(|| format!("unknown relationship label {name:?}"))
                })
                .transpose()?;
            let started = Instant::now();
            let result = read.shortest_path(
                start,
                end,
                &ShortestPathOptions {
                    max_hops,
                    max_expansions,
                    direction,
                    edge_filter: EdgeFilter { label: edge_label },
                },
            )?;
            println!("strategy\t{:?}", result.strategy);
            println!("termination\t{:?}", result.termination);
            println!("visited_nodes\t{}", result.visited_nodes);
            println!("start_expanded_nodes\t{}", result.start_expanded_nodes);
            println!("end_expanded_nodes\t{}", result.end_expanded_nodes);
            println!("expanded_nodes\t{}", result.expanded_nodes);
            println!("examined_relationships\t{}", result.examined_relationships);
            println!("elapsed_ms\t{:.6}", millis(started.elapsed()));
            if let Some(path) = result.path {
                println!("hops\t{}", path.edges.len());
                for (index, (&edge_id, endpoints)) in
                    path.edges.iter().zip(path.nodes.windows(2)).enumerate()
                {
                    let edge = read.edge(edge_id).ok_or("path relationship disappeared")?;
                    println!(
                        "step\t{}\t{}\t{}\t{}\t{}",
                        index + 1,
                        endpoints[0],
                        edge_id,
                        read.symbol(edge.label).unwrap_or("?"),
                        endpoints[1]
                    );
                }
                if path.edges.is_empty() {
                    println!("node\t{start}");
                }
            }
        }
        "search-text" => {
            let path = required(&mut arguments, "database path")?;
            let query = required(&mut arguments, "query text")?;
            let limit = optional_usize(&mut arguments, "limit")?.unwrap_or(10);
            let database = Database::open(path)?;
            let embedder_name = arguments
                .next()
                .or_else(|| env::var("VECGRA_EMBEDDER").ok())
                .unwrap_or_else(|| "hash".into());
            let mut embedder =
                embedder::create_embedder(&embedder_name, database.vector_dimension(), 1)?;
            let vector = embedder.embed_query(&query)?;
            let read = database.read();
            for hit in read.vector_search_adaptive(&vector, VectorTarget::Both, limit, None)? {
                match hit.element {
                    ElementRef::Node(id) => {
                        let node = read.node(id).ok_or("search result node disappeared")?;
                        println!(
                            "{:.6}\tnode\t{}\t{}\t{}",
                            hit.score,
                            id,
                            read.symbol(node.label).unwrap_or("?"),
                            concise_element(&read, &node.properties)
                        );
                    }
                    ElementRef::Edge(id) => {
                        let edge = read
                            .edge(id)
                            .ok_or("search result relationship disappeared")?;
                        println!(
                            "{:.6}\tedge\t{}\t{}\t{} -> {}",
                            hit.score,
                            id,
                            read.symbol(edge.label).unwrap_or("?"),
                            edge.source,
                            edge.target
                        );
                    }
                }
            }
        }
        "range-text" => {
            let path = required(&mut arguments, "database path")?;
            let seed: u64 = required(&mut arguments, "seed node id")?.parse()?;
            let query = required(&mut arguments, "query text")?;
            let max_hops = optional_usize(&mut arguments, "maximum hops")?.unwrap_or(2);
            let limit = optional_usize(&mut arguments, "limit")?.unwrap_or(10);
            let embedder_name = arguments
                .next()
                .or_else(|| env::var("VECGRA_EMBEDDER").ok())
                .unwrap_or_else(|| "hash".into());
            let direction = match arguments.next().as_deref() {
                None | Some("both") => Direction::Both,
                Some("out") => Direction::Outgoing,
                Some("in") => Direction::Incoming,
                Some(other) => return Err(format!("unknown direction {other}").into()),
            };
            let edge_label_name = arguments.next().filter(|name| name != "-");
            let node_label_name = arguments.next().filter(|name| name != "-");
            let database = Database::open(path)?;
            let mut embedder =
                embedder::create_embedder(&embedder_name, database.vector_dimension(), 1)?;
            let vector = embedder.embed_query(&query)?;
            let read = database.read();
            let edge_label = edge_label_name
                .as_deref()
                .map(|name| {
                    read.label_id(name)
                        .ok_or_else(|| format!("unknown edge label {name:?}"))
                })
                .transpose()?;
            let node_filter = node_label_name
                .as_deref()
                .map(|name| {
                    read.label_id(name)
                        .map(|label| ElementFilter {
                            label: Some(label),
                            properties: Vec::new(),
                        })
                        .ok_or_else(|| format!("unknown node label {name:?}"))
                })
                .transpose()?;
            let mut seeds = ElementSet::new();
            seeds.insert(ElementRef::Node(seed));
            let result = read.vector_search_graph_range_adaptive(
                &vector,
                &seeds,
                &GraphRangeSearchOptions {
                    max_hops,
                    limit,
                    direction,
                    edge_filter: EdgeFilter { label: edge_label },
                    include_seeds: true,
                    node_filter,
                },
            )?;
            println!("strategy\t{:?}", result.plan.strategy);
            println!("candidate_nodes\t{}", result.candidate_nodes);
            println!("candidate_vectors\t{}", result.plan.candidate_vectors);
            for hit in result.hits {
                let ElementRef::Node(id) = hit.element else {
                    unreachable!("graph-range search produces only node candidates")
                };
                let node = read.node(id).ok_or("matched node disappeared")?;
                println!(
                    "{:.6}\tnode\t{}\t{}\t{}",
                    hit.score,
                    id,
                    read.symbol(node.label).unwrap_or("?"),
                    concise_element(&read, &node.properties)
                );
            }
        }
        "search-facets" => {
            let path = required(&mut arguments, "database path")?;
            let facet_text = required(&mut arguments, "facet text separated by ||")?;
            let facets: Vec<_> = facet_text
                .split("||")
                .map(str::trim)
                .filter(|facet| !facet.is_empty())
                .collect();
            if facets.is_empty() {
                return Err("at least one non-empty query facet is required".into());
            }
            let limit = optional_usize(&mut arguments, "limit")?.unwrap_or(10);
            let embedder_name = arguments
                .next()
                .or_else(|| env::var("VECGRA_EMBEDDER").ok())
                .unwrap_or_else(|| "hash".into());
            let target = parse_vector_target(arguments.next().as_deref())?;
            let label_name = arguments.next().filter(|name| name != "-");
            let candidate_elements = optional_usize(&mut arguments, "candidate elements")?;
            let database = Database::open(path)?;
            let mut embedder =
                embedder::create_embedder(&embedder_name, database.vector_dimension(), 1)?;
            let queries = facets
                .into_iter()
                .map(|facet| embedder.embed_query(facet))
                .collect::<Result<Vec<_>, _>>()?;
            let read = database.read();
            let label = label_name
                .as_deref()
                .map(|name| {
                    read.label_id(name)
                        .ok_or_else(|| format!("unknown label {name:?}"))
                })
                .transpose()?;
            let hits = if let Some(candidate_elements) = candidate_elements {
                read.late_interaction_search_approximate(
                    &queries,
                    None,
                    target,
                    limit,
                    label,
                    candidate_elements,
                )?
            } else {
                read.late_interaction_search_adaptive(&queries, None, target, limit, label)?
            };
            for hit in hits {
                let matched = hit
                    .matched_vector_indices
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                match hit.element {
                    ElementRef::Node(id) => {
                        let node = read.node(id).ok_or("search result node disappeared")?;
                        println!(
                            "{:.6}\tnode\t{}\t{}\t{}\tfacets={}",
                            hit.score,
                            id,
                            read.symbol(node.label).unwrap_or("?"),
                            concise_element(&read, &node.properties),
                            matched
                        );
                    }
                    ElementRef::Edge(id) => {
                        let edge = read
                            .edge(id)
                            .ok_or("search result relationship disappeared")?;
                        println!(
                            "{:.6}\tedge\t{}\t{}\t{} -> {}\tfacets={}",
                            hit.score,
                            id,
                            read.symbol(edge.label).unwrap_or("?"),
                            edge.source,
                            edge.target,
                            matched
                        );
                    }
                }
            }
        }
        "semantic-text" => {
            let path = required(&mut arguments, "database path")?;
            let query = required(&mut arguments, "query text")?;
            let limit = optional_usize(&mut arguments, "limit")?.unwrap_or(20);
            let max_hops = optional_usize(&mut arguments, "max hops")?.unwrap_or(2);
            let embedder_name = arguments
                .next()
                .or_else(|| env::var("VECGRA_EMBEDDER").ok())
                .unwrap_or_else(|| "hash".into());
            let database = Database::open(path)?;
            let mut embedder =
                embedder::create_embedder(&embedder_name, database.vector_dimension(), 1)?;
            let vector = embedder.embed_query(&query)?;
            let read = database.read();
            let hits = read.semantic_paths(
                &vector,
                &SemanticPathOptions {
                    limit,
                    max_hops,
                    ..SemanticPathOptions::default()
                },
            )?;
            for hit in hits {
                let Some(node) = read.node(hit.node) else {
                    continue;
                };
                let path = hit
                    .path
                    .iter()
                    .filter_map(|id| read.edge(*id))
                    .map(|edge| read.symbol(edge.label).unwrap_or("?").to_owned())
                    .collect::<Vec<_>>()
                    .join(">");
                println!(
                    "{:.6}\tseed={}\tnode={}\thops={}\tpath={}\t{}\t{}",
                    hit.score,
                    hit.seed,
                    hit.node,
                    hit.path.len(),
                    path,
                    read.symbol(node.label).unwrap_or("?"),
                    concise_element(&read, &node.properties)
                );
            }
        }
        "query" => {
            let path = required(&mut arguments, "database path")?;
            let statement = required(&mut arguments, "query statement")?;
            let database = Database::open(path)?;
            let read = database.read();
            for matched in query::execute(&read, &statement)? {
                let start = read
                    .node(matched.start)
                    .ok_or("matched start disappeared")?;
                let edge = read.edge(matched.edge).ok_or("matched edge disappeared")?;
                let end = read.node(matched.end).ok_or("matched end disappeared")?;
                println!(
                    "{}:{} {} -[{}:{}]-> {}:{} {}",
                    matched.start,
                    read.symbol(start.label).unwrap_or("?"),
                    concise_element(&read, &start.properties),
                    matched.edge,
                    read.symbol(edge.label).unwrap_or("?"),
                    matched.end,
                    read.symbol(end.label).unwrap_or("?"),
                    concise_element(&read, &end.properties)
                );
            }
        }
        "query-text" => {
            let path = required(&mut arguments, "database path")?;
            let statement = required(&mut arguments, "query statement")?;
            let text = required(&mut arguments, "semantic query text")?;
            let database = Database::open(path)?;
            let embedder_name = arguments
                .next()
                .or_else(|| env::var("VECGRA_EMBEDDER").ok())
                .unwrap_or_else(|| "hash".into());
            let mut embedder =
                embedder::create_embedder(&embedder_name, database.vector_dimension(), 1)?;
            let vector = embedder.embed_query(&text)?;
            let read = database.read();
            for matched in query::execute_semantic(&read, &statement, &vector)? {
                let start = read
                    .node(matched.pattern.start)
                    .ok_or("matched start disappeared")?;
                let edge = read
                    .edge(matched.pattern.edge)
                    .ok_or("matched edge disappeared")?;
                let end = read
                    .node(matched.pattern.end)
                    .ok_or("matched end disappeared")?;
                println!(
                    "{:.6}\tstart={:.6}\tedge={}\tend={}\t{}:{} {} -[{}:{}]-> {}:{} {}",
                    matched.score,
                    matched.start_score,
                    optional_score(matched.edge_score),
                    optional_score(matched.end_score),
                    matched.pattern.start,
                    read.symbol(start.label).unwrap_or("?"),
                    concise_element(&read, &start.properties),
                    matched.pattern.edge,
                    read.symbol(edge.label).unwrap_or("?"),
                    matched.pattern.end,
                    read.symbol(end.label).unwrap_or("?"),
                    concise_element(&read, &end.properties)
                );
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn required(
    arguments: &mut impl Iterator<Item = String>,
    description: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| format!("missing {description}").into())
}

fn optional_usize(
    arguments: &mut impl Iterator<Item = String>,
    description: &str,
) -> Result<Option<usize>, Box<dyn Error>> {
    arguments
        .next()
        .map(|value| {
            value
                .parse()
                .map_err(|error| format!("invalid {description} {value:?}: {error}").into())
        })
        .transpose()
}

fn concise_element(read: &vecgra::ReadGuard<'_>, properties: &[vecgra::Property]) -> String {
    for key in ["name", "title", "path", "text", "kind"] {
        if let Some(value) = read.property(properties, key) {
            return display_value(read, value);
        }
    }
    String::new()
}

fn display_value(read: &vecgra::ReadGuard<'_>, value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(value) => value.to_string(),
        Value::Int(value) => value.to_string(),
        Value::Float(value) => value.to_string(),
        Value::String(value) => value.to_string(),
        Value::Bytes(value) => format!("<{} bytes>", value.len()),
        Value::Node(id) => read
            .node(*id)
            .map(|node| format!("node:{id}:{}", read.symbol(node.label).unwrap_or("?")))
            .unwrap_or_else(|| format!("node:{id}")),
        Value::Edge(id) => read
            .edge(*id)
            .map(|edge| format!("edge:{id}:{}", read.symbol(edge.label).unwrap_or("?")))
            .unwrap_or_else(|| format!("edge:{id}")),
    }
}

fn element_vector_owned(
    read: &vecgra::ReadGuard<'_>,
    element: ElementRef,
) -> Result<Option<Vec<f32>>, Box<dyn Error>> {
    match element {
        ElementRef::Node(id) => Ok(read.node_vector_owned(id, 0)?),
        ElementRef::Edge(id) => Ok(read.edge_vector_owned(id, 0)?),
    }
}

fn parse_vector_target(value: Option<&str>) -> Result<VectorTarget, Box<dyn Error>> {
    match value {
        None | Some("both") => Ok(VectorTarget::Both),
        Some("nodes") => Ok(VectorTarget::Nodes),
        Some("edges") => Ok(VectorTarget::Edges),
        Some(other) => Err(format!("unknown vector target {other:?}").into()),
    }
}

fn percentile(samples: &[Duration], percentile: f64) -> Duration {
    let index = ((samples.len() - 1) as f64 * percentile).ceil() as usize;
    samples[index]
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn optional_score(score: Option<f32>) -> String {
    score.map_or_else(|| "-".to_owned(), |score| format!("{score:.6}"))
}

fn parse_json_scalar(encoded: &str) -> Result<Value, Box<dyn Error>> {
    Ok(match serde_json::from_str(encoded)? {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(value) => Value::Bool(value),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Value::Int(value)
            } else if let Some(value) = value.as_f64() {
                Value::Float(value)
            } else {
                return Err("numeric property value is outside the supported range".into());
            }
        }
        serde_json::Value::String(value) => Value::String(value.into()),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            return Err("property benchmark value must be a JSON scalar".into());
        }
    })
}

fn parse_numeric_bound(
    encoded: &str,
    numeric_type: &str,
) -> Result<Bound<NumericValue>, Box<dyn Error>> {
    if encoded == "-" {
        return Ok(Bound::Unbounded);
    }
    Ok(Bound::Included(match numeric_type {
        "int" => NumericValue::Int(encoded.parse()?),
        "float" => NumericValue::Float(encoded.parse()?),
        other => return Err(format!("unknown numeric type {other:?}").into()),
    }))
}

fn recall(exact: &[vecgra::VectorHit], approximate: &[vecgra::VectorHit]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let approximate: std::collections::HashSet<_> =
        approximate.iter().map(|hit| hit.element).collect();
    exact
        .iter()
        .filter(|hit| approximate.contains(&hit.element))
        .count() as f64
        / exact.len() as f64
}

const COMMAND_USAGES: &[(&str, &str)] = &[
    (
        "import-jsonl",
        "import-jsonl <nodes.jsonl> <edges.jsonl> <database> <dimension> [f16|f32]",
    ),
    ("import-fbin", "import-fbin <train.fbin> <database>"),
    (
        "import-node-fbin",
        "import-node-fbin <train.fbin> <metadata.jsonl> <database> [f16|f32]",
    ),
    (
        "bench-fbin",
        "bench-fbin <database> <test.fbin> <neighbors.ibin> [queries] [candidate-vectors] [k] [compressed|hot-f32]",
    ),
    (
        "bench-filtered-fbin",
        "bench-filtered-fbin <database> <test.fbin> [queries] [stride] [candidate-elements] [k]",
    ),
    (
        "bench-range-fbin",
        "bench-range-fbin <database> <test.fbin> <neighbors.ibin> <property> <inclusive-lower-bound> [queries] [candidate-elements] [k]",
    ),
    (
        "import-graphalytics",
        "import-graphalytics <vertices> <edges> <database>",
    ),
    (
        "bench-bfs",
        "bench-bfs <database> <source> [reference-output|-] [iterations]",
    ),
    (
        "bench-wcc",
        "bench-wcc <database> [reference-output|-] [iterations]",
    ),
    (
        "bench-pagerank",
        "bench-pagerank <database> [reference-output|-] [benchmark-iterations]",
    ),
    (
        "import-rust",
        "import-rust <repository> <database> [dimension] [hash|qwen] [batch-size]",
    ),
    (
        "import-github",
        "import-github <owner/repository> <database> [issues] [pulls] [discussions] [releases] [dimension] [hash|qwen] [batch-size]",
    ),
    ("stats", "stats <database>"),
    ("check", "check <database>"),
    (
        "plan-search",
        "plan-search <database> [nodes|edges|both] [label]",
    ),
    (
        "bench-property",
        "bench-property <database> <nodes|edges|both> <key> <json-scalar> [iterations]",
    ),
    (
        "numeric-range",
        "numeric-range <database> <nodes|edges|both> <key> <int|float> <inclusive-lower|-> <inclusive-upper|-> [limit]",
    ),
    ("node", "node <database> <node-id>"),
    ("neighbors", "neighbors <database> <node-id> [out|in|both]"),
    (
        "shortest-path",
        "shortest-path <database> <start-node-id> <end-node-id> [max-hops] [out|in|both] [edge-label|-] [max-expansions]",
    ),
    (
        "search-text",
        "search-text <database> <query> [limit] [hash|qwen]",
    ),
    (
        "range-text",
        "range-text <database> <seed-node-id> <query> [hops] [limit] [hash|qwen] [out|in|both] [edge-label|-] [node-label|-]",
    ),
    (
        "search-facets",
        "search-facets <database> '<facet-1> || <facet-2>' [limit] [hash|qwen] [nodes|edges|both] [label|-] [candidate-elements]",
    ),
    (
        "semantic-text",
        "semantic-text <database> <query> [limit] [max-hops] [hash|qwen]",
    ),
    (
        "bench-search",
        "bench-search <database> <query> [iterations] [hash|qwen] [nodes|edges|both] [label|-] [candidate-vectors]",
    ),
    (
        "bench-ann",
        "bench-ann <database> [queries] [candidate-vectors] [nodes|edges|both]",
    ),
    (
        "query",
        "query <database> 'MATCH (a:A)-[e:E]->(b:B) RETURN a,e,b LIMIT 10'",
    ),
    (
        "query-text",
        "query-text <database> '<MATCH query>' <semantic-text> [hash|qwen]",
    ),
    (
        "bench-pattern",
        "bench-pattern <database> <query> [iterations]",
    ),
    (
        "bench-neighbors",
        "bench-neighbors <database> <node-id> [iterations] [out|in|both]",
    ),
    (
        "bench-expand",
        "bench-expand <database> <seed-node-id> <hops> [iterations] [out|in|both]",
    ),
    (
        "compact",
        "compact <source-database> <destination-database> [f16|f32]",
    ),
    (
        "export-ladybug",
        "export-ladybug <database> <output-directory>",
    ),
];

fn print_help() {
    println!("Vecgra CLI\n\nUsage:\n  vecgra --version");
    for (_, usage) in COMMAND_USAGES {
        println!("  vecgra {usage}");
    }
    println!(
        "\nRun `vecgra <command> --help` for command help.\n\
         OPENROUTER_API_KEY is required for qwen. VECGRA_EMBEDDER sets the default embedder."
    );
}

fn print_command_help(command: &str) -> Result<(), Box<dyn Error>> {
    let Some((_, usage)) = COMMAND_USAGES.iter().find(|(name, _)| *name == command) else {
        return Err(format!("unknown command {command}").into());
    };
    println!("Usage:\n  vecgra {usage}");
    match command {
        "import-github" => println!(
            "\nDefaults: 1000 issues, 1000 pull requests, 300 discussions, \
             100 releases, 256 dimensions, hash embeddings, batch size 128.\n\
             Authentication: GITHUB_TOKEN, GH_TOKEN, or `gh auth login`."
        ),
        "import-rust" => println!("\nDefaults: 256 dimensions, hash embeddings, batch size 128."),
        "semantic-text" => println!("\nDefaults: 20 results, 2 hops, hash embeddings."),
        _ => {}
    }
    Ok(())
}
