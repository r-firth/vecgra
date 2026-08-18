#!/usr/bin/env python3
"""Probe binary IVF partitions over Vecgra's persisted 512-bit sketches.

The important metric is partition coverage: if exact vector scoring visited all
rows in the selected partitions, what fraction of supplied true neighbors
could it possibly recover?  This separates directory quality from the later
full-signature/exact-vector ranking policy.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import faiss
import numpy as np


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def mapped_sketches(path: Path) -> np.memmap:
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    if bytes(raw[:8]) != b"VGRPHDB\0":
        raise ValueError(f"{path}: not a Vecgra database")
    metadata_length = struct.unpack_from("<Q", raw, 24)[0]
    metadata = raw[64 : 64 + metadata_length]
    if bytes(metadata[:8]) not in (b"VGSNAP04", b"VGSNAP05"):
        raise ValueError(f"{path}: checkpoint has no mapped sketch columns")
    section_offset, section_length = struct.unpack_from("<QQ", metadata, 64 + 8 * 16)
    section = metadata[section_offset : section_offset + section_length]
    if bytes(section[:8]) != b"VGSIG002":
        raise ValueError(f"{path}: checkpoint does not use owner-column sketches")
    rows = struct.unpack_from("<Q", section, 8)[0]
    words = struct.unpack_from("<I", section, 16)[0]
    owner_offset = 24
    owner_kind_offset = owner_offset + rows * 8
    label_offset = align(owner_kind_offset + rows, 4)
    signature_offset = align(label_offset + rows * 4, 8)
    expected = signature_offset + rows * words * 8
    if expected != section_length:
        raise ValueError(f"{path}: invalid sketch section length")
    absolute = 64 + section_offset + signature_offset
    return np.memmap(path, dtype=np.uint8, mode="r", offset=absolute, shape=(rows, words * 8))


def open_neighbors(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        rows, width = struct.unpack("<II", stream.read(8))
    return np.memmap(path, dtype="<u4", mode="r", offset=8, shape=(rows, width))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("query_database", type=Path)
    parser.add_argument("neighbors", type=Path)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--train-rows", type=int, default=200_000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nprobes", default="64,128,256,512,768,1024")
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)
    sketches = mapped_sketches(args.database)
    queries = mapped_sketches(args.query_database)
    truth = open_neighbors(args.neighbors)
    query_count = min(args.queries, len(queries), len(truth))
    bits = sketches.shape[1] * 8
    ordinals = np.linspace(
        0, len(sketches) - 1, min(args.train_rows, len(sketches)), dtype=np.int64
    )
    sample = np.ascontiguousarray(sketches[ordinals])
    query_codes = np.ascontiguousarray(queries[:query_count])

    quantizer = faiss.IndexBinaryFlat(bits)
    index = faiss.IndexBinaryIVF(quantizer, bits, args.nlist)
    index.cp.niter = 12
    started = time.perf_counter()
    index.train(sample)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    index.add(np.ascontiguousarray(sketches))
    add_seconds = time.perf_counter() - started

    list_sizes = np.array(
        [index.invlists.list_size(i) for i in range(args.nlist)], dtype=np.int64
    )
    row_to_list = np.empty(len(sketches), dtype=np.int32)
    for list_id, size in enumerate(list_sizes):
        if size:
            ids = faiss.rev_swig_ptr(index.invlists.get_ids(list_id), int(size))
            row_to_list[ids] = list_id

    print(
        f"rows={len(sketches)} bits={bits} nlist={args.nlist} "
        f"train_s={train_seconds:.3f} add_s={add_seconds:.3f} "
        f"empty_lists={(list_sizes == 0).sum()} "
        f"list_p50={np.percentile(list_sizes, 50):.0f} "
        f"list_p95={np.percentile(list_sizes, 95):.0f} "
        f"list_max={list_sizes.max()}"
    )

    for nprobe in map(int, args.nprobes.split(",")):
        started = time.perf_counter()
        _, selected = quantizer.search(query_codes, nprobe)
        directory_ms = (time.perf_counter() - started) * 1000 / query_count
        inspected = list_sizes[selected].sum(axis=1)
        covered = 0
        for query_index, selected_lists in enumerate(selected):
            selected_set = set(map(int, selected_lists))
            covered += sum(
                int(row_to_list[int(row)]) in selected_set
                for row in truth[query_index, : args.k]
            )
        print(
            f"nprobe={nprobe} coverage={covered / (query_count * args.k):.6f} "
            f"directory_mean_ms={directory_ms:.3f} "
            f"rows_mean={inspected.mean():.0f} "
            f"rows_p50={np.percentile(inspected, 50):.0f} "
            f"rows_p95={np.percentile(inspected, 95):.0f}"
        )


if __name__ == "__main__":
    main()
