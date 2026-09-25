##!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from cluster import build_adata, cluster_leiden


def main():
    parser = argparse.ArgumentParser(
        description="Run Leiden clustering on selected exact/best/worst kNN graphs"
    )
    parser.add_argument("--selected_knn_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--flavor", choices=["igraph", "leidenalg"], required=True)
    parser.add_argument(
        "--partition_type",
        choices=["RBConfiguration", "CPM", "Modularity"],
        default="RBConfiguration",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.selected_knn_manifest, sep="\t")

    required = {
        "dataset",
        "method",
        "k",
        "selection",
        "pca_seed",
        "neighbors_file",
    }

    if not required.issubset(manifest.columns):
        raise ValueError(
            f"selected_knn_manifest must contain columns {sorted(required)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clustering_manifest = []

    for _, row in manifest.iterrows():
        dataset = row["dataset"]
        method = row["method"]
        k = int(row["k"])
        selection = row["selection"]
        neighbors_file = Path(row["neighbors_file"])

        if selection == "exact":
            pca_seed = ""
            stem = f"{dataset}_{method}_k{k}_exact"
        else:
            pca_seed = int(row["pca_seed"])
            stem = f"{dataset}_{method}_k{k}_{selection}_seed{pca_seed}"

        adata, cell_ids = build_adata(neighbors_file)

        labels = cluster_leiden(
            adata,
            args.flavor,
            args.partition_type,
            args.resolution,
            args.random_seed,
        )

        clusters_file = output_dir / f"{stem}_clusters.tsv"

        pd.DataFrame({
            "cell_id": cell_ids,
            "cluster": labels,
        }).to_csv(
            clusters_file,
            sep="\t",
            index=False,
        )

        clustering_manifest.append({
            "dataset": dataset,
            "method": method,
            "k": k,
            "selection": selection,
            "pca_seed": pca_seed,
            "neighbors_file": str(neighbors_file),
            "clusters_file": str(clusters_file),
        })

    clustering_manifest = pd.DataFrame(clustering_manifest)
    clustering_manifest = clustering_manifest.sort_values(
        ["dataset", "method", "k", "selection"]
    )

    manifest_file = (
        output_dir / f"{args.name}_selected_clustering_manifest.tsv"
    )

    clustering_manifest.to_csv(
        manifest_file,
        sep="\t",
        index=False,
    )

    print()
    print("Selected clustering manifest")
    print("----------------------------")
    print(clustering_manifest.to_string(index=False))
    print()
    print(f"Wrote: {manifest_file}")


if __name__ == "__main__":
    main()