#!/usr/bin/env python3
"""kNN graph module (scanpy-backed) for omnibenchmark.

Output HDF5: {output_dir}/{name}_neighbors.h5

The flat CSR distance graph is stored at the file root:
    /cell_ids
    /data
    /indices
    /indptr

Scanpy connectivities are stored under:
    /connectivities/{data,indices,indptr}

In selected mode, exact and randomized PCA runs are matched separately for
each dataset and PCA implementation.
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc


def parse_args():
    p = argparse.ArgumentParser(description="kNN graph module (scanpy-backed)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", required=True)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pcas_tsv", dest="pcas_tsv", type=Path, help="PCA embedding TSV")
    g.add_argument("--corrected_tsv", dest="pcas_tsv", type=Path, help="Batch-corrected embedding TSV")
    g.add_argument("--selected_seeds", type=Path, help="Selected seeds TSV from principal-angle analysis")

    p.add_argument("--pca_manifest", type=Path, help="PCA manifest produced by principal-angle collector")
    p.add_argument("--n_neighbors", type=int, required=True, help="Number of nearest neighbors")
    p.add_argument("--n_components", type=int, help="Number of leading PCs; required only in direct mode")
    p.add_argument("--flavor", type=str, required=True, choices=["umap", "gauss"], help="Method to compute connectivities")
    p.add_argument("--random_seed", type=int, required=True, help="Fixed random seed for kNN construction")

    return p.parse_args()


def _write_csr(grp, m):
    m = m.tocsr()
    grp.create_dataset("data", data=m.data)
    grp.create_dataset("indices", data=m.indices)
    grp.create_dataset("indptr", data=m.indptr)


def write_neighbors_graph(adata, out_dir, name):
    out = Path(out_dir) / f"{name}_neighbors.h5"

    with h5py.File(out, "w") as h5:
        h5.create_dataset("cell_ids", data=np.array(adata.obs_names.to_list(), dtype="S"))
        _write_csr(h5, adata.obsp["distances"])
        _write_csr(h5.create_group("connectivities"), adata.obsp["connectivities"])

    print(f"  wrote: {out}")
    return out


def run_knn(pcas_tsv, n_components, n_neighbors, flavor, random_seed):
    df = pl.read_csv(pcas_tsv, separator="\t", skip_rows=1, has_header=False)

    if n_components > df.width - 1:
        raise ValueError(f"Requested {n_components} PCs, but PCA file contains only {df.width - 1}.")

    embedding = df[:, 1:1 + n_components].to_numpy().astype(np.float64)

    adata = ad.AnnData(X=np.zeros((embedding.shape[0], 1)))
    adata.obs_names = df[:, 0].to_list()
    adata.obsm["X_pca"] = embedding

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        method=flavor,
        use_rep="X_pca",
        random_state=random_seed,
    )

    return adata


def run_direct(args):
    if args.n_components is None:
        raise ValueError("--n_components is required when using --pcas_tsv.")

    adata = run_knn(
        args.pcas_tsv,
        args.n_components,
        args.n_neighbors,
        args.flavor,
        args.random_seed,
    )

    write_neighbors_graph(
        adata,
        args.output_dir,
        args.name,
    )


def run_selected(args):
    if args.pca_manifest is None:
        raise ValueError("--pca_manifest is required when using --selected_seeds.")

    selected = pd.read_csv(args.selected_seeds, sep="\t")

    required_selected = {"dataset", "method", "k", "selection", "seed"}
    if not required_selected.issubset(selected.columns):
        raise ValueError(f"selected_seeds must contain columns {sorted(required_selected)}")

    selected = selected[selected["selection"].isin(["best", "worst"])].copy()

    pca_manifest = pd.read_csv(args.pca_manifest, sep="\t")

    required_manifest = {"dataset", "method", "pca_type", "seed", "pca_file"}
    if not required_manifest.issubset(pca_manifest.columns):
        raise ValueError(f"pca_manifest must contain columns {sorted(required_manifest)}")

    exact_scores = {}
    random_scores = {}

    for _, row in pca_manifest.iterrows():
        dataset = row["dataset"]
        method = row["method"]
        pca_type = row["pca_type"]
        path = Path(row["pca_file"])

        if pca_type == "exact":
            key = (dataset, method)

            if key in exact_scores:
                raise ValueError(f"Multiple exact PCA score files found for {key}")

            exact_scores[key] = path

        elif pca_type == "random":
            key = (dataset, method, int(row["seed"]))

            if key in random_scores:
                raise ValueError(f"Multiple randomized PCA score files found for {key}")

            random_scores[key] = path

        else:
            raise ValueError(f"Unknown pca_type: {pca_type}")

    manifest = []

    # Exact PCA: one graph for every selected k within each dataset/method pair.
    for (dataset, method), subset in selected.groupby(["dataset", "method"]):
        exact_key = (dataset, method)

        if exact_key not in exact_scores:
            raise ValueError(f"No exact PCA score file found for {exact_key}")

        ks = sorted(subset["k"].astype(int).unique())

        for k in ks:
            name = f"{dataset}_{method}_k{k}_exact"

            adata = run_knn(
                exact_scores[exact_key],
                k,
                args.n_neighbors,
                args.flavor,
                args.random_seed,
            )

            out = write_neighbors_graph(
                adata,
                args.output_dir,
                name,
            )

            manifest.append({
                "dataset": dataset,
                "method": method,
                "k": k,
                "selection": "exact",
                "pca_seed": "",
                "pca_file": str(exact_scores[exact_key]),
                "neighbors_file": str(out),
            })

    # Randomized PCA: best + worst for every selected k.
    for _, row in selected.iterrows():
        dataset = row["dataset"]
        method = row["method"]
        k = int(row["k"])
        selection = row["selection"]
        seed = int(row["seed"])

        key = (dataset, method, seed)

        if key not in random_scores:
            raise ValueError(
                f"No randomized PCA score file found for "
                f"dataset={dataset}, method={method}, seed={seed}"
            )

        name = f"{dataset}_{method}_k{k}_{selection}_seed{seed}"

        adata = run_knn(
            random_scores[key],
            k,
            args.n_neighbors,
            args.flavor,
            args.random_seed,
        )

        out = write_neighbors_graph(
            adata,
            args.output_dir,
            name,
        )

        manifest.append({
            "dataset": dataset,
            "method": method,
            "k": k,
            "selection": selection,
            "pca_seed": seed,
            "pca_file": str(random_scores[key]),
            "neighbors_file": str(out),
        })

    manifest = pd.DataFrame(manifest)
    manifest = manifest.sort_values(["dataset", "method", "k", "selection"])

    manifest_file = Path(args.output_dir) / f"{args.name}_selected_knn_manifest.tsv"
    manifest.to_csv(manifest_file, sep="\t", index=False)

    print("\nSelected kNN graphs")
    print("-------------------")
    print(manifest.to_string(index=False))
    print(f"\nWrote: {manifest_file}")


def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Full command: {' '.join(sys.argv)}")

    if args.selected_seeds is not None:
        print("  mode: selected")
        print(f"  selected_seeds: {args.selected_seeds}")
        print(f"  pca_manifest: {args.pca_manifest}")

        run_selected(args)

    else:
        print("  mode: direct")
        print(f"  pcas_tsv: {args.pcas_tsv}")
        print(f"  n_components: {args.n_components}")

        run_direct(args)


if __name__ == "__main__":
    main()