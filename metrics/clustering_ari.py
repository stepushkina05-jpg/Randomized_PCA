#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def load_clusters(path: Path):
    df = pd.read_csv(path, sep="\t")
    if "cell_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} must contain cell_id and cluster columns.")
    return df[["cell_id", "cluster"]]


def compare_clusterings(path_a: Path, path_b: Path, column_b="cluster"):
    A = load_clusters(path_a).rename(columns={"cluster": "cluster_a"})
    B = pd.read_csv(path_b, sep="\t")

    if "cell_id" not in B.columns or column_b not in B.columns:
        raise ValueError(f"{path_b} must contain cell_id and {column_b} columns.")

    B = B[["cell_id", column_b]].rename(columns={column_b: "cluster_b"})
    merged = A.merge(B, on="cell_id", how="inner")

    if len(merged) != len(A) or len(merged) != len(B):
        raise ValueError("Cell IDs differ between clustering and reference labels.")
    ari = adjusted_rand_score(merged["cluster_a"], merged["cluster_b"])
    n_clusters_test = merged["cluster_a"].nunique()
    n_clusters_reference = merged["cluster_b"].nunique()

    return  ari, n_clusters_test, n_clusters_reference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_clustering_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--labels_tsv", default=None)
    parser.add_argument("--label_column", default="label")
    args = parser.parse_args()

    manifest = pd.read_csv(args.selected_clustering_manifest, sep="\t")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if args.labels_tsv:
        labels_file = Path(args.labels_tsv)

        for _, row in manifest.iterrows():
            ari, n_clusters_test, n_clusters_reference = compare_clusterings(Path(row["clusters_file"]), labels_file, args.label_column)

            results.append({
                "method": row["method"],
                "k": int(row["k"]),
                "selection": row["selection"],
                "pca_seed": row["pca_seed"],
                "reference": "ground_truth",
                "n_clusters_test": n_clusters_test,
                "n_clusters_reference": n_clusters_reference,
                "ari": ari,
            })

    else:
        for (method, k), group in manifest.groupby(["method", "k"]):
            exact_rows = group[group["selection"] == "exact"]

            if len(exact_rows) != 1:
                raise ValueError(f"Expected exactly one exact clustering for {method}, k={k}")

            exact_file = Path(exact_rows.iloc[0]["clusters_file"])

            for selection in ["best", "worst"]:
                rows = group[group["selection"] == selection]

                if len(rows) != 1:
                    raise ValueError(f"Expected exactly one {selection} clustering for {method}, k={k}")

                row = rows.iloc[0]
                ari, n_clusters_test, n_clusters_reference = compare_clusterings(Path(row["clusters_file"]), exact_file)

                results.append({
                    "method": method,
                    "k": int(k),
                    "selection": selection,
                    "pca_seed": row["pca_seed"],
                    "reference": "exact",
                    "ari": ari,
                    "n_clusters_test": n_clusters_test,
                    "n_clusters_reference": n_clusters_reference,
                })

    results = pd.DataFrame(results)
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