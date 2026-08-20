#!/usr/bin/env python3
"""Measure an IVF partition tier before committing it to Vecgra's format.

This is deliberately an engineering probe, not a product benchmark.  It uses
Faiss as a convenient oracle for the clustering/search shape, reports recall
against supplied neighbors, and reports how many rows each nprobe setting
would make a native contiguous-partition executor inspect.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import faiss
import numpy as np


def open_matrix(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        rows, dimensions = struct.unpack("<II", stream.read(8))
    expected = 8 + rows * dimensions * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {path.stat().st_size}")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dimensions))


def open_neighbors(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        rows, width = struct.unpack("<II", stream.read(8))
    expected = 8 + rows * width * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {path.stat().st_size}")
    return np.memmap(path, dtype="<u4", mode="r", offset=8, shape=(rows, width))


def recall_at(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    total = 0
    for actual, expected in zip(found[:, :k], truth[:, :k], strict=True):
        total += len(set(map(int, actual)) & set(map(int, expected)))
    return total / (len(found) * k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("queries", type=Path)
    parser.add_argument("neighbors", type=Path)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--train-rows", type=int, default=100_000)
    parser.add_argument("--queries-count", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nprobes", default="8,16,32,64,128")
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)
    train = open_matrix(args.train)
    queries = open_matrix(args.queries)
    truth = open_neighbors(args.neighbors)
    count = min(args.queries_count, len(queries), len(truth))
    dimensions = train.shape[1]

    # Evenly spaced rows are deterministic and avoid materializing a random
    # million-row permutation.  Copy only the training sample and queries so
    # normalization never mutates the source mmap.
    sample_ordinals = np.linspace(
        0, len(train) - 1, min(args.train_rows, len(train)), dtype=np.int64
    )
    sample = np.asarray(train[sample_ordinals], dtype=np.float32).copy()
    query_matrix = np.asarray(queries[:count], dtype=np.float32).copy()
    faiss.normalize_L2(sample)
    faiss.normalize_L2(query_matrix)

    quantizer = faiss.IndexFlatIP(dimensions)
    index = faiss.IndexIVFFlat(quantizer, dimensions, args.nlist, faiss.METRIC_INNER_PRODUCT)
    index.cp.niter = 15
    started = time.perf_counter()
    index.train(sample)
    train_seconds = time.perf_counter() - started

    # Add in bounded blocks: this matches the intended second pass over a
    # vector spool and avoids owning a normalized copy of the full dataset.
    started = time.perf_counter()
    block_rows = 25_000
    for start in range(0, len(train), block_rows):
        block = np.asarray(train[start : start + block_rows], dtype=np.float32).copy()
        faiss.normalize_L2(block)
        index.add(block)
    add_seconds = time.perf_counter() - started

    list_sizes = np.array(
        [index.invlists.list_size(i) for i in range(args.nlist)], dtype=np.int64
    )
    print(
        f"rows={len(train)} dimensions={dimensions} nlist={args.nlist} "
        f"train_s={train_seconds:.3f} add_s={add_seconds:.3f} "
        f"empty_lists={(list_sizes == 0).sum()} "
        f"list_p50={np.percentile(list_sizes, 50):.0f} "
        f"list_p95={np.percentile(list_sizes, 95):.0f} "
        f"list_max={list_sizes.max()}"
    )

    # Search the centroid directory separately so inspected-row counts use the
    # actual selected partition sizes rather than nprobe * mean-size.
    centroids = faiss.downcast_index(index.quantizer)
    for nprobe in map(int, args.nprobes.split(",")):
        index.nprobe = nprobe
        _, selected = centroids.search(query_matrix, nprobe)
        inspected = list_sizes[selected].sum(axis=1)
        started = time.perf_counter()
        _, found = index.search(query_matrix, args.k)
        elapsed = time.perf_counter() - started
        print(
            f"nprobe={nprobe} recall={recall_at(found, truth[:count], args.k):.6f} "
            f"query_pseudo_mean_ms={elapsed * 1000 / count:.3f} "
            f"rows_mean={inspected.mean():.0f} "
            f"rows_p50={np.percentile(inspected, 50):.0f} "
            f"rows_p95={np.percentile(inspected, 95):.0f}"
        )


if __name__ == "__main__":
    main()
