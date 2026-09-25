#!/usr/bin/env python3

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix


def load_neighbors(path: Path):
    with h5py.File(path, "r") as h5:
        n = len(h5["cell_ids"])
        cell_ids = h5["cell_ids"][:].astype(str)
        distances = csr_matrix((h5["data"][:], h5["indices"][:], h5["indptr"][:]), shape=(n, n))
        conn = h5["connectivities"]
        connectivities = csr_matrix((conn["data"][:], conn["indices"][:], conn["indptr"][:]), shape=(n, n))
    return distances, connectivities, cell_ids


def cluster_graph(path: Path, resolution: float, random_seed: int):
    distances, connectivities, cell_ids = load_neighbors(path)
    adata = ad.AnnData(X=np.zeros((len(cell_ids), 1)))
    adata.obs_names = cell_ids
    adata.obsp["distances"] = distances
    adata.obsp["connectivities"] = connectivities
    adata.uns["neighbors"] = {"distances_key": "distances", "connectivities_key": "connectivities"}
    sc.tl.leiden(adata, resolution=resolution, random_state=random_seed, flavor="igraph", n_iterations=2, directed=False)
    return cell_ids, adata.obs["leiden"].astype(str).to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_knn_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.selected_knn_manifest, sep="\t")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clustering_manifest = []

    for _, row in manifest.iterrows():
        method = row["method"]
        k = int(row["k"])
        selection = row["selection"]
        neighbors_file = Path(row["neighbors_file"])

        if selection == "exact":
            pca_seed = ""
            stem = f"{method}_k{k}_exact"
        else:
            pca_seed = int(row["pca_seed"])
            stem = f"{method}_k{k}_{selection}_seed{pca_seed}"

        cell_ids, labels = cluster_graph(neighbors_file, args.resolution, args.random_seed)
        clusters_file = output_dir / f"{stem}_clusters.tsv"
        pd.DataFrame({"cell_id": cell_ids, "cluster": labels}).to_csv(clusters_file, sep="\t", index=False)

        clustering_manifest.append({"method": method, "k": k, "selection": selection, "pca_seed": pca_seed, "neighbors_file": str(neighbors_file), "clusters_file": str(clusters_file)})

    clustering_manifest = pd.DataFrame(clustering_manifest)
    manifest_file = output_dir / f"{args.name}_selected_clustering_manifest.tsv"
    clustering_manifest.to_csv(manifest_file, sep="\t", index=False)

    print()
    print("Selected clustering manifest")
    print("----------------------------")
    print(clustering_manifest.to_string(index=False))
    print()
    print(f"Wrote: {manifest_file}")


if __name__ == "__main__":
    main()