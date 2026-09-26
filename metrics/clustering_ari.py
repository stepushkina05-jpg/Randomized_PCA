#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def get_dataset(path: Path):
    parts = path.parts
    if "DATA" not in parts:
        raise ValueError(f"Could not identify dataset from {path}")
    return parts[parts.index("DATA") + 1]

def load_clusters(path: Path):
    df = pd.read_csv(path, sep="\t")

    if "cell_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} must contain cell_id and cluster columns.")

    return df[["cell_id", "cluster"]]


def compare_clusterings(path_a: Path, path_b: Path, column_b="cluster", require_same_cells=True):
    A = load_clusters(path_a).rename(columns={"cluster": "cluster_a"})
    B = pd.read_csv(path_b, sep="\t")

    if "cell_id" not in B.columns or column_b not in B.columns:
        raise ValueError(f"{path_b} must contain cell_id and {column_b} columns.")

    B = B[["cell_id", column_b]].dropna(subset=[column_b]).rename(columns={column_b: "cluster_b"})
    merged = A.merge(B, on="cell_id", how="inner")

    if require_same_cells:
        if len(merged) != len(A) or len(merged) != len(B):
            raise ValueError("Cell IDs differ between clustering and reference clustering.")
    else:
        if len(merged) != len(B):
            raise ValueError("Some ground-truth cell IDs are missing from the clustering.")

    ari = adjusted_rand_score(merged["cluster_a"], merged["cluster_b"])
    n_clusters_test = merged["cluster_a"].nunique()
    n_clusters_reference = merged["cluster_b"].nunique()

    return ari, n_clusters_test, n_clusters_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_clustering_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--labels_tsv", nargs="+", default=None)
    parser.add_argument("--label_column", default="label")
    args = parser.parse_args()

    manifest = pd.read_csv(args.selected_clustering_manifest, sep="\t")

    required = {
        "dataset",
        "method",
        "k",
        "selection",
        "pca_seed",
        "clusters_file",
    }

    if not required.issubset(manifest.columns):
        raise ValueError(
            f"selected_clustering_manifest must contain columns {sorted(required)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Ground-truth mode
    if args.labels_tsv:
        labels_files = {}

        for path in args.labels_tsv:
            path = Path(path)
            dataset = get_dataset(path)

            if dataset in labels_files:
                raise ValueError(f"Multiple ground-truth label files found for {dataset}")

            labels_files[dataset] = path

        for _, row in manifest.iterrows():
            dataset = row["dataset"]

            if dataset not in labels_files:
                raise ValueError(f"No ground-truth labels found for {dataset}")

            labels_file = labels_files[dataset]

            ari, n_clusters_test, n_clusters_reference = compare_clusterings(
                Path(row["clusters_file"]),
                labels_file,
                args.label_column,
                require_same_cells=False,
            )

            results.append({
                "dataset": dataset,
                "method": row["method"],
                "k": int(row["k"]),
                "selection": row["selection"],
                "pca_seed": row["pca_seed"],
                "reference": "ground_truth",
                "n_clusters_test": n_clusters_test,
                "n_clusters_reference": n_clusters_reference,
                "ari": ari,
            })

    # Exact-PCA clustering reference mode
    else:
        for (dataset, method, k), group in manifest.groupby(
            ["dataset", "method", "k"]
        ):
            exact_rows = group[group["selection"] == "exact"]

            if len(exact_rows) != 1:
                raise ValueError(
                    f"Expected exactly one exact clustering for "
                    f"{dataset}, {method}, k={k}"
                )

            exact_file = Path(exact_rows.iloc[0]["clusters_file"])

            for selection in ["best", "worst"]:
                rows = group[group["selection"] == selection]

                if len(rows) != 1:
                    raise ValueError(
                        f"Expected exactly one {selection} clustering for "
                        f"{dataset}, {method}, k={k}"
                    )

                row = rows.iloc[0]

                ari, n_clusters_test, n_clusters_reference = compare_clusterings(
                    Path(row["clusters_file"]),
                    exact_file,
                    require_same_cells=True,
                )

                results.append({
                    "dataset": dataset,
                    "method": method,
                    "k": int(k),
                    "selection": selection,
                    "pca_seed": row["pca_seed"],
                    "reference": "exact",
                    "n_clusters_test": n_clusters_test,
                    "n_clusters_reference": n_clusters_reference,
                    "ari": ari,
                })

    results = pd.DataFrame(results)

    if not results.empty:
        results = results.sort_values(
            ["dataset", "method", "k", "selection"]
        )

    output_file = output_dir / f"{args.name}_clustering_ari.tsv"
    results.to_csv(output_file, sep="\t", index=False)

    print()
    print("Clustering ARI results")
    print("----------------------")
    print(results.to_string(index=False))
    print()
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()