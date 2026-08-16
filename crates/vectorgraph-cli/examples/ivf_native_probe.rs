use gemm::Parallelism;
use std::error::Error;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let train_path = args.next().ok_or("missing train fbin")?;
    let query_path = args.next().ok_or("missing query fbin")?;
    let truth_path = args.next().ok_or("missing truth ibin")?;
    let clusters = args.next().map_or(Ok(4096), |v| v.parse())?;
    let sample_rows = args.next().map_or(Ok(200_000), |v| v.parse())?;
    let iterations = args.next().map_or(Ok(8), |v| v.parse())?;
    let query_count = args.next().map_or(Ok(200), |v| v.parse())?;

    let train_bytes = fs::read(&train_path)?;
    let (train_rows, dimensions) = matrix_shape(&train_bytes, 4)?;
    let query_bytes = fs::read(&query_path)?;
    let (available_queries, query_dimensions) = matrix_shape(&query_bytes, 4)?;
    let truth_bytes = fs::read(&truth_path)?;
    let (truth_rows, truth_width) = matrix_shape(&truth_bytes, 4)?;
    if dimensions != query_dimensions || available_queries != truth_rows {
        return Err("matrix shapes do not agree".into());
    }
    let sample_rows = sample_rows.min(train_rows);
    let query_count = query_count.min(available_queries);
    if clusters == 0 || clusters > sample_rows || query_count == 0 {
        return Err("invalid clusters/sample/query count".into());
    }

    let mut sample = vec![0.0f32; sample_rows * dimensions];
    for row in 0..sample_rows {
        let source = row * train_rows / sample_rows;
        decode_row(
            &train_bytes,
            source,
            dimensions,
            &mut sample[row * dimensions..(row + 1) * dimensions],
        );
    }
    normalize_rows(&mut sample, dimensions);
    let mut centroids = vec![0.0f32; clusters * dimensions];
    for cluster in 0..clusters {
        // A full-period modular walk is deterministic and avoids selecting a
        // semantically ordered run of source rows as the initial directory.
        let sample_row = cluster.wrapping_mul(104_729) % sample_rows;
        centroids[cluster * dimensions..(cluster + 1) * dimensions]
            .copy_from_slice(&sample[sample_row * dimensions..(sample_row + 1) * dimensions]);
    }

    let training_started = Instant::now();
    let mut assignments = vec![0u32; sample_rows];
    for iteration in 0..iterations {
        assign_rows(&sample, dimensions, &centroids, clusters, &mut assignments);
        let mut sums = vec![0.0f32; centroids.len()];
        let mut counts = vec![0u32; clusters];
        for (row, &cluster) in sample.chunks_exact(dimensions).zip(&assignments) {
            let cluster = cluster as usize;
            counts[cluster] += 1;
            for (sum, value) in sums[cluster * dimensions..(cluster + 1) * dimensions]
                .iter_mut()
                .zip(row)
            {
                *sum += *value;
            }
        }
        let mut empty = 0;
        for cluster in 0..clusters {
            if counts[cluster] == 0 {
                empty += 1;
                continue;
            }
            let centroid = &mut sums[cluster * dimensions..(cluster + 1) * dimensions];
            normalize(centroid);
        }
        for cluster in 0..clusters {
            if counts[cluster] != 0 {
                centroids[cluster * dimensions..(cluster + 1) * dimensions]
                    .copy_from_slice(&sums[cluster * dimensions..(cluster + 1) * dimensions]);
            }
        }
        eprintln!("iteration={} empty_clusters={empty}", iteration + 1);
    }
    let training_time = training_started.elapsed();

    let assignment_started = Instant::now();
    let block_rows = 8192;
    let mut row_to_cluster = vec![0u32; train_rows];
    let mut list_sizes = vec![0u32; clusters];
    let mut decoded = Vec::new();
    for start in (0..train_rows).step_by(block_rows) {
        let rows = block_rows.min(train_rows - start);
        decoded.clear();
        decoded.resize(rows * dimensions, 0.0);
        for local in 0..rows {
            decode_row(
                &train_bytes,
                start + local,
                dimensions,
                &mut decoded[local * dimensions..(local + 1) * dimensions],
            );
        }
        normalize_rows(&mut decoded, dimensions);
        assign_rows(
            &decoded,
            dimensions,
            &centroids,
            clusters,
            &mut row_to_cluster[start..start + rows],
        );
        for &cluster in &row_to_cluster[start..start + rows] {
            list_sizes[cluster as usize] += 1;
        }
    }
    let assignment_time = assignment_started.elapsed();

    let mut queries = vec![0.0f32; query_count * dimensions];
    for row in 0..query_count {
        decode_row(
            &query_bytes,
            row,
            dimensions,
            &mut queries[row * dimensions..(row + 1) * dimensions],
        );
    }
    normalize_rows(&mut queries, dimensions);
    let mut query_scores = vec![0.0f32; query_count * clusters];
    multiply_scores(
        &queries,
        query_count,
        dimensions,
        &centroids,
        clusters,
        &mut query_scores,
    );
    let truth = decode_u32_matrix(&truth_bytes, truth_rows, truth_width);
    let mut sorted_sizes = list_sizes.clone();
    sorted_sizes.sort_unstable();
    println!(
        "rows={train_rows} dimensions={dimensions} clusters={clusters} sample_rows={sample_rows} iterations={iterations} training_s={:.3} assignment_s={:.3} empty_lists={} list_p50={} list_p95={} list_max={}",
        training_time.as_secs_f64(),
        assignment_time.as_secs_f64(),
        list_sizes.iter().filter(|&&size| size == 0).count(),
        percentile(&sorted_sizes, 0.50),
        percentile(&sorted_sizes, 0.95),
        sorted_sizes.last().copied().unwrap_or(0),
    );
    for probes in [64usize, 128, 256, 512, 768, 1024] {
        if probes > clusters {
            continue;
        }
        let mut covered = 0usize;
        let mut inspected = Vec::with_capacity(query_count);
        for query in 0..query_count {
            let scores = &query_scores[query * clusters..(query + 1) * clusters];
            let mut selected = (0..clusters).collect::<Vec<_>>();
            selected.select_nth_unstable_by(probes, |&left, &right| {
                scores[right].total_cmp(&scores[left])
            });
            selected.truncate(probes);
            let mut selected_bitmap = vec![false; clusters];
            let mut rows = 0u64;
            for cluster in selected {
                selected_bitmap[cluster] = true;
                rows += list_sizes[cluster] as u64;
            }
            inspected.push(rows);
            covered += truth[query * truth_width..query * truth_width + 10]
                .iter()
                .filter(|&&row| selected_bitmap[row_to_cluster[row as usize] as usize])
                .count();
        }
        inspected.sort_unstable();
        println!(
            "probes={probes} coverage={:.6} rows_p50={} rows_p95={}",
            covered as f64 / (query_count * 10) as f64,
            percentile(&inspected, 0.50),
            percentile(&inspected, 0.95),
        );
    }
    Ok(())
}

fn matrix_shape(bytes: &[u8], value_width: usize) -> Result<(usize, usize), Box<dyn Error>> {
    if bytes.len() < 8 {
        return Err("matrix header is truncated".into());
    }
    let rows = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
    let columns = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let expected = 8usize
        .checked_add(
            rows.checked_mul(columns)
                .ok_or("matrix shape overflow")?
                .checked_mul(value_width)
                .ok_or("matrix byte length overflow")?,
        )
        .ok_or("matrix byte length overflow")?;
    if bytes.len() != expected {
        return Err("matrix byte length does not match its header".into());
    }
    Ok((rows, columns))
}

fn decode_row(bytes: &[u8], row: usize, dimensions: usize, output: &mut [f32]) {
    let start = 8 + row * dimensions * 4;
    for (value, encoded) in output
        .iter_mut()
        .zip(bytes[start..start + dimensions * 4].chunks_exact(4))
    {
        *value = f32::from_le_bytes(encoded.try_into().unwrap());
    }
}

fn decode_u32_matrix(bytes: &[u8], rows: usize, columns: usize) -> Vec<u32> {
    bytes[8..]
        .chunks_exact(4)
        .take(rows * columns)
        .map(|value| u32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}

fn normalize_rows(rows: &mut [f32], dimensions: usize) {
    for row in rows.chunks_exact_mut(dimensions) {
        normalize(row);
    }
}

fn normalize(row: &mut [f32]) {
    let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in row {
            *value /= norm;
        }
    }
}

fn assign_rows(
    rows: &[f32],
    dimensions: usize,
    centroids: &[f32],
    clusters: usize,
    assignments: &mut [u32],
) {
    let row_count = rows.len() / dimensions;
    let block_rows = 8192;
    let mut scores = Vec::new();
    for start in (0..row_count).step_by(block_rows) {
        let count = block_rows.min(row_count - start);
        scores.clear();
        scores.resize(count * clusters, 0.0);
        multiply_scores(
            &rows[start * dimensions..(start + count) * dimensions],
            count,
            dimensions,
            centroids,
            clusters,
            &mut scores,
        );
        for (assignment, scores) in assignments[start..start + count]
            .iter_mut()
            .zip(scores.chunks_exact(clusters))
        {
            *assignment = scores
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .unwrap()
                .0 as u32;
        }
    }
}

fn multiply_scores(
    rows: &[f32],
    row_count: usize,
    dimensions: usize,
    centroids: &[f32],
    clusters: usize,
    scores: &mut [f32],
) {
    debug_assert_eq!(rows.len(), row_count * dimensions);
    debug_assert_eq!(centroids.len(), clusters * dimensions);
    debug_assert_eq!(scores.len(), row_count * clusters);
    // SAFETY: all matrices are complete contiguous row-major buffers. The
    // strides describe A[row,dim], B[cluster,dim] as B^T[dim,cluster], and
    // C[row,cluster]; read_dst=false means no uninitialized output is read.
    unsafe {
        gemm::gemm(
            row_count,
            clusters,
            dimensions,
            scores.as_mut_ptr(),
            1,
            clusters as isize,
            false,
            rows.as_ptr(),
            1,
            dimensions as isize,
            centroids.as_ptr(),
            dimensions as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }
}

fn percentile<T: Copy>(values: &[T], percentile: f64) -> T {
    values[((values.len() - 1) as f64 * percentile).round() as usize]
}
