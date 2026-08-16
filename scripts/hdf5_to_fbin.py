#!/usr/bin/env python3
"""Convert an ANN-Benchmarks HDF5 file to portable fbin/ibin matrices."""

import argparse
import struct

import h5py
import numpy as np


def write_matrix(path: str, values: np.ndarray, dtype: str) -> None:
    values = np.asarray(values, dtype=dtype, order="C")
    if values.ndim != 2:
        raise ValueError(f"{path}: expected a rank-2 matrix")
    with open(path, "wb") as output:
        output.write(struct.pack("<II", values.shape[0], values.shape[1]))
        values.tofile(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_prefix")
    args = parser.parse_args()
    with h5py.File(args.input, "r") as source:
        write_matrix(f"{args.output_prefix}.train.fbin", source["train"], "<f4")
        write_matrix(f"{args.output_prefix}.test.fbin", source["test"], "<f4")
        write_matrix(f"{args.output_prefix}.neighbors.ibin", source["neighbors"], "<u4")


if __name__ == "__main__":
    main()
