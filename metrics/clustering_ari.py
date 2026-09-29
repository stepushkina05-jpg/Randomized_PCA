#!/usr/bin/env python3
"""Evaluate Leiden clustering accuracy using the adjusted Rand index.

The script reads clustering results generated at one Leiden resolution and
compares every exact-, best-seed-, and worst-seed PCA clustering with the
corresponding ground-truth labels supplied in the clustering manifest.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score


RESULT_COLUMNS = [
    "dataset",
    "method",
    "k",
    "resolution",
    "selection",
    "pca_seed",
    "reference",
    "ari",
    "n_clusters_test",
    "n_clusters_reference",
]


def load_clusters(path: Path, cluster_column="cluster"):
    df = pd.read_csv(path, sep="\t")

    required = {"cell_id", cluster_column}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}.")

    df = df[["cell_id", cluster_column]].dropna(subset=[cluster_column])

    if df["cell_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate cell IDs.")

    return df


def compare_clusterings(
    path_a: Path,
    path_b: Path,
    column_b="cluster",
    require_same_cells=True,
):
    clustering = load_clusters(path_a).rename(
        columns={"cluster": "cluster_a"}
    )
    reference = load_clusters(path_b, column_b).rename(
        columns={column_b: "cluster_b"}
    )

    merged = clustering.merge(
        reference,
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )

    if require_same_cells:
        if len(merged) != len(clustering) or len(merged) != len(reference):
            raise ValueError(
                f"Cell IDs differ between {path_a} and {path_b}."
            )
    elif len(merged) != len(reference):
        raise ValueError(
            f"Some ground-truth cell IDs from {path_b} are missing in {path_a}."
        )

    ari = adjusted_rand_score(
        merged["cluster_a"],
        merged["cluster_b"],
    )

    return (
        ari,
        merged["cluster_a"].nunique(),
        merged["cluster_b"].nunique(),
    )


def result_metadata(row):
    pca_seed = "" if pd.isna(row["pca_seed"]) else int(row["pca_seed"])

    return {
        "dataset": row["dataset"],
        "method": row["method"],
        "k": int(row["k"]),
        "resolution": float(row["resolution"]),
        "selection": row["selection"],
        "pca_seed": pca_seed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate clustering ARI against ground-truth labels."
    )
    parser.add_argument("--selected_clustering_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--label_column", default="label")
    args = parser.parse_args()

    manifest = pd.read_csv(
        args.selected_clustering_manifest,
        sep="\t",
    )

    required = {
        "dataset",
        "method",
        "k",
        "resolution",
        "selection",
        "pca_seed",
        "clusters_file",
        "labels_file",
    }

    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"selected_clustering_manifest is missing columns {missing}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for _, row in manifest.iterrows():
        clusters_file = Path(row["clusters_file"])
        labels_file = Path(row["labels_file"])

        if not clusters_file.exists():
            raise FileNotFoundError(
                f"Missing clustering file: {clusters_file}"
            )

        if not labels_file.exists():
            raise FileNotFoundError(
                f"Missing ground-truth labels: {labels_file}"
            )

        ari, n_test, n_reference = compare_clusterings(
            clusters_file,
            labels_file,
            column_b=args.label_column,
            require_same_cells=False,
        )

        results.append({
            **result_metadata(row),
            "reference": "ground_truth",
            "ari": ari,
            "n_clusters_test": n_test,
            "n_clusters_reference": n_reference,
        })

    results = pd.DataFrame(results, columns=RESULT_COLUMNS)

    if not results.empty:
        results = results.sort_values([
            "dataset",
            "method",
            "k",
            "resolution",
            "selection",
        ])

    output_file = output_dir / f"{args.name}_clustering_ari.tsv"
    results.to_csv(output_file, sep="\t", index=False)

    print()
    print("Clustering ARI against ground truth")
    print("-----------------------------------")
    print(results.to_string(index=False))
    print()
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()