#!/usr/bin/env python3
"""Evaluate Leiden clustering stability using the adjusted Rand index.

The script reads clustering results generated at one Leiden resolution. It
either compares each clustering with ground-truth labels or compares the
best- and worst-seed PCA clusterings with the corresponding exact-PCA
clustering at the same dataset, method, k, and Leiden resolution.
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


def get_dataset(path: Path):
    parts = path.parts

    if "DATA" not in parts:
        raise ValueError(f"Could not identify dataset from {path}")

    return parts[parts.index("DATA") + 1]


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
    clustering = load_clusters(path_a).rename(columns={"cluster": "cluster_a"})
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
        description=(
            "Calculate clustering ARI against ground-truth labels or the "
            "corresponding exact-PCA clustering."
        )
    )
    parser.add_argument("--selected_clustering_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--labels_tsv", nargs="+", default=None)
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
    }

    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"selected_clustering_manifest is missing columns {missing}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Compare every clustering with ground-truth labels.
    if args.labels_tsv:
        labels_files = {}

        for path in map(Path, args.labels_tsv):
            dataset = get_dataset(path)

            if dataset in labels_files:
                raise ValueError(
                    f"Multiple ground-truth label files found for {dataset}"
                )

            labels_files[dataset] = path

        for _, row in manifest.iterrows():
            dataset = row["dataset"]

            if dataset not in labels_files:
                raise ValueError(
                    f"No ground-truth labels found for {dataset}"
                )

            ari, n_test, n_reference = compare_clusterings(
                Path(row["clusters_file"]),
                labels_files[dataset],
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

    # Compare best- and worst-seed clusterings with exact-PCA clustering.
    else:
        group_columns = [
            "dataset",
            "method",
            "k",
            "resolution",
        ]

        for group_key, group in manifest.groupby(
            group_columns,
            dropna=False,
        ):
            config = dict(zip(group_columns, group_key))
            exact_rows = group[group["selection"] == "exact"]

            if len(exact_rows) != 1:
                raise ValueError(
                    "Expected exactly one exact clustering for "
                    f"configuration {config}, found {len(exact_rows)}."
                )

            exact_file = Path(exact_rows.iloc[0]["clusters_file"])

            for selection in ["best", "worst"]:
                selected_rows = group[group["selection"] == selection]

                if len(selected_rows) != 1:
                    raise ValueError(
                        f"Expected exactly one {selection} clustering for "
                        f"configuration {config}, found {len(selected_rows)}."
                    )

                row = selected_rows.iloc[0]

                ari, n_test, n_reference = compare_clusterings(
                    Path(row["clusters_file"]),
                    exact_file,
                    require_same_cells=True,
                )

                results.append({
                    **result_metadata(row),
                    "reference": "exact",
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
    print("Clustering ARI results")
    print("----------------------")
    print(results.to_string(index=False))
    print()
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()