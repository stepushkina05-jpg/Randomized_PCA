#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_graph(path: Path):
    with h5py.File(path, "r") as h5:
        data = h5["data"][:]
        indices = h5["indices"][:]
        indptr = h5["indptr"][:]
        n = len(h5["cell_ids"])
        cell_ids = h5["cell_ids"][:].astype(str)

        distances = csr_matrix((data, indices, indptr), shape=(n, n))

        conn = h5["connectivities"]
        connectivities = csr_matrix((conn["data"][:], conn["indices"][:], conn["indptr"][:]), shape=(n, n))

    return distances, connectivities, cell_ids


def compare_graphs(graph_a: Path, graph_b: Path):
    A, WA, ids_a = load_graph(graph_a)
    B, WB, ids_b = load_graph(graph_b)

    if not np.array_equal(ids_a, ids_b):
        raise ValueError("Cell IDs or cell order differ between graphs.")

    neighbor_retention = []

    for i in range(A.shape[0]):
        neighbors_a = set(A.indices[A.indptr[i]:A.indptr[i + 1]])
        neighbors_b = set(B.indices[B.indptr[i]:B.indptr[i + 1]])
        neighbor_retention.append(len(neighbors_a & neighbors_b) / len(neighbors_a))

    neighbor_retention = np.asarray(neighbor_retention)

    diff = WA - WB
    connectivity_error = np.sqrt(diff.power(2).sum()) / np.sqrt(WA.power(2).sum())

    return neighbor_retention, connectivity_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_knn_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.selected_knn_manifest, sep="\t")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for (method, k), group in manifest.groupby(["method", "k"]):
        exact_rows = group[group["selection"] == "exact"]

        if len(exact_rows) != 1:
            raise ValueError(f"Expected exactly one exact graph for {method}, k={k}")

        exact_graph = Path(exact_rows.iloc[0]["neighbors_file"])

        for selection in ["best", "worst"]:
            random_rows = group[group["selection"] == selection]

            if len(random_rows) != 1:
                raise ValueError(f"Expected exactly one {selection} graph for {method}, k={k}")

            row = random_rows.iloc[0]
            random_graph = Path(row["neighbors_file"])
            retention, connectivity_error = compare_graphs(exact_graph, random_graph)

            results.append({
                "method": method,
                "k": int(k),
                "selection": selection,
                "pca_seed": int(row["pca_seed"]),
                "mean_neighbor_retention": retention.mean(),
                "median_neighbor_retention": np.median(retention),
                "q10_neighbor_retention": np.quantile(retention, 0.10),
                "connectivity_error": connectivity_error,
            })

    results = pd.DataFrame(results)
    output_file = output_dir / f"{args.name}_knn_overlap.tsv"

    results.to_csv(output_file, sep="\t", index=False)

    print()
    print("kNN overlap results")
    print("-------------------")
    print(results.to_string(index=False))
    print()
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()