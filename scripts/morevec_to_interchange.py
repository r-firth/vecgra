#!/usr/bin/env python3
"""Convert the public MoReVec movie workload to Vecgra interchange files.

The output deliberately uses ordinary fbin/ibin plus JSONL rather than a
benchmark-specific database loader.  `vecgra import-node-fbin` can therefore ingest
the same files used by other ANN systems while retaining typed movie metadata.
"""

import argparse
import json
import re
import struct
from pathlib import Path

import h5py
import numpy as np


def text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def write_matrix(path: Path, values: object, dtype: str, chunk_rows: int = 4096) -> None:
    rows, columns = values.shape
    with path.open("wb") as output:
        output.write(struct.pack("<II", rows, columns))
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            np.asarray(values[start:end], dtype=dtype, order="C").tofile(output)


def scalar_number(value: float) -> int | float | None:
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return int(numeric) if numeric.is_integer() else numeric


def float_number(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def convert_train(source_path: Path, output_dir: Path) -> dict[str, int]:
    with h5py.File(source_path, "r") as source:
        vectors = source["train_mvector"]
        count, dimension = vectors.shape
        write_matrix(output_dir / "train.fbin", vectors, "<f4")

        mids = source["train_mid"]
        titles = source["train_title"]
        genres = source["train_genre"]
        ratings = source["train_avgrating"]
        votes = source["train_num_votes"]
        years = source["train_year"]
        id_by_mid: dict[str, int] = {}
        with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as output:
            for row in range(count):
                mid = text(mids[row])
                if mid in id_by_mid:
                    raise ValueError(f"duplicate movie id {mid!r}")
                id_by_mid[mid] = row
                record = {
                    "label": "Movie",
                    "properties": {
                        "mid": mid,
                        "title": text(titles[row]),
                        "genre": text(genres[row]),
                        "avg_rating": float_number(ratings[row]),
                        "num_votes": scalar_number(votes[row]),
                        "year": scalar_number(years[row]),
                    },
                }
                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                output.write("\n")
    return id_by_mid


def parse_filter(encoded: str) -> dict[str, object]:
    if encoded == "No_filter":
        return {"expression": encoded}
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|>|<|=)\s*(-?[0-9]+(?:\.[0-9]+)?)", encoded)
    if match is None:
        raise ValueError(f"unsupported MoReVec filter {encoded!r}")
    key, operator, value = match.groups()
    return {
        "expression": encoded,
        "property": key,
        "operator": operator,
        "value": float(value),
    }


def convert_queries(
    paths: list[Path],
    filters_path: Path,
    output_dir: Path,
    id_by_mid: dict[str, int],
) -> None:
    with h5py.File(filters_path, "r") as source:
        filters = [text(value) for value in source["filters"][:]]
        selectivities = [float(value) for value in source["selectivities"][:]]

    manifest = []
    for path in paths:
        match = re.search(r"([0-9]+)$", path.stem)
        if match is None:
            raise ValueError(f"cannot infer filter id from query file {path}")
        filter_id = int(match.group(1))
        if filter_id >= len(filters):
            raise ValueError(f"query filter id {filter_id} exceeds filter manifest")
        query_name = f"query-{filter_id}.fbin"
        truth_name = f"truth-{filter_id}.ibin"
        with h5py.File(path, "r") as source:
            write_matrix(output_dir / query_name, source["test"], "<f4")
            mids = source["mids"]
            truth = np.empty(mids.shape, dtype="<u4")
            for row in range(mids.shape[0]):
                for column in range(mids.shape[1]):
                    mid = text(mids[row, column])
                    try:
                        truth[row, column] = id_by_mid[mid]
                    except KeyError as error:
                        raise ValueError(
                            f"official neighbor {mid!r} in {path} is absent from training data"
                        ) from error
            write_matrix(output_dir / truth_name, truth, "<u4")

        entry = parse_filter(filters[filter_id])
        entry.update(
            {
                "id": filter_id,
                "selectivity": selectivities[filter_id],
                "queries": query_name,
                "truth": truth_name,
            }
        )
        manifest.append(entry)

    manifest.sort(key=lambda entry: entry["id"])
    with (output_dir / "filters.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_hdf5", type=Path)
    parser.add_argument("filters_hdf5", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--queries",
        type=Path,
        nargs="*",
        help="query HDF5 files; defaults to query*.hdf5 beside the training file",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    queries = args.queries
    if queries is None:
        queries = sorted(args.train_hdf5.parent.glob("query*.hdf5"))
    if not queries:
        raise ValueError("no query HDF5 files found")
    id_by_mid = convert_train(args.train_hdf5, args.output_directory)
    convert_queries(
        queries,
        args.filters_hdf5,
        args.output_directory,
        id_by_mid,
    )


if __name__ == "__main__":
    main()
