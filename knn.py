#!/usr/bin/env python3
"""kNN graph module (scanpy-backed) for omnibenchmark.

Output HDF5: {output_dir}/{name}_neighbors.h5
  Flat CSR of the kNN distance graph at the file root, matching the R metrics
  reader (graph.R::read_csr_h5), which reads these datasets from the root:
    /cell_ids   string array (n_cells,)
    /data       CSR data
    /indices    CSR column indices (0-based)
    /indptr     CSR row pointers
  The connectivities scanpy builds alongside the distances are stored under a
  nested group so the clustering stage can read them directly instead of
  reconstructing them (both graphs share /cell_ids):
    /connectivities/{data,indices,indptr}  CSR (n_cells, n_cells)
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import polars as pl
import scanpy as sc
import pandas as pd



def parse_args():
    p = argparse.ArgumentParser(description="kNN graph module (scanpy-backed)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", required=True)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pcas_tsv", dest="pcas_tsv", type=Path,
                   help="PCA embedding TSV")
    g.add_argument("--corrected_tsv", dest="pcas_tsv", type=Path,
                   help="Batch-corrected embedding TSV")
    g.add_argument("--selected_seeds", type=Path,
                   help="Selected seeds TSV from principal-angle analysis")

    p.add_argument("--pca_manifest", type=Path,
                   help="PCA manifest produced by principal-angle collector")
    p.add_argument("--n_neighbors", type=int, required=True,
                   help="Number of nearest neighbors")
    p.add_argument("--n_components", type=int,
                   help="Number of leading PCs; required only in direct mode")
    p.add_argument("--flavor", type=str, required=True,
                   choices=["umap", "gauss"], help="Method to compute connectivities")
    p.add_argument("--random_seed", type=int, required=True,
                   help="Fixed random seed for kNN construction")

    return p.parse_args()

def _write_csr(grp, m):
    m = m.tocsr()
    grp.create_dataset("data",    data=m.data)
    grp.create_dataset("indices", data=m.indices)
    grp.create_dataset("indptr",  data=m.indptr)


def write_neighbors_graph(adata, out_dir, name):
    out = Path(out_dir) / f"{name}_neighbors.h5"
    with h5py.File(out, "w") as h5:
        # dtype="S": h5py can't write numpy unicode ('<U') arrays; bytes give portable fixed-length HDF5 strings.
        h5.create_dataset("cell_ids", data=np.array(adata.obs_names.to_list(), dtype="S"))
        _write_csr(h5, adata.obsp["distances"])  # distances flat at the root (R metrics reader)
        _write_csr(h5.create_group("connectivities"), adata.obsp["connectivities"])
    print(f"  wrote: {out}")
    return out



def run_knn(pcas_tsv, n_components, n_neighbors, flavor, random_seed):
    # TSV has N header cols and N+1 data cols (first data col = row IDs, unnamed).
    df = pl.read_csv(pcas_tsv, separator="\t", skip_rows=1, has_header=False)
    if n_components > df.width - 1:
        raise ValueError(f"Requested {n_components} PCs, but PCA file contains only {df.width - 1}.")
    embedding = df[:, 1:1 + n_components].to_numpy().astype(np.float64)

    adata = ad.AnnData(X=np.zeros((embedding.shape[0], 1)))
    adata.obs_names = df[:, 0].to_list()
    adata.obsm["X_pca"] = embedding

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, method=flavor,
                    use_rep="X_pca", random_state=random_seed)

    return adata

def run_direct(args):
    if args.n_components is None:
        raise ValueError(
            "--n_components is required when using --pcas_tsv."
        )

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
    required = {"method", "k", "selection", "seed"}

    if not required.issubset(selected.columns):
        raise ValueError(f"selected_seeds must contain columns {sorted(required)}")

    selected = selected[selected["selection"].isin(["best", "worst"])].copy()

    pca_manifest = pd.read_csv(args.pca_manifest, sep="\t")

    exact_scores = {}
    random_scores = {}

    for _, row in pca_manifest.iterrows():
        method = row["method"]
        pca_type = row["pca_type"]
        path = Path(row["pca_file"])

        if pca_type == "exact":
            exact_scores[method] = path
        elif pca_type == "random":
            random_scores[(method, int(row["seed"]))] = path
        else:
            raise ValueError(f"Unknown pca_type: {pca_type}")

    manifest = []

    # exact PCA: one graph for every selected k
    for method in selected["method"].unique():
        if method not in exact_scores:
            raise ValueError(f"No exact PCA score file found for {method}")

        ks = sorted(selected.loc[selected["method"] == method, "k"].astype(int).unique())

        for k in ks:
            name = f"{method}_k{k}_exact"
            adata = run_knn(exact_scores[method], k, args.n_neighbors, args.flavor, args.random_seed)
            out = write_neighbors_graph(adata, args.output_dir, name)

            manifest.append({
                "method": method,
                "k": k,
                "selection": "exact",
                "pca_seed": "",
                "pca_file": str(exact_scores[method]),
                "neighbors_file": str(out),
            })

    # randomized PCA: best + worst for every selected k
    for _, row in selected.iterrows():
        method = row["method"]
        k = int(row["k"])
        selection = row["selection"]
        seed = int(row["seed"])

        key = (method, seed)

        if key not in random_scores:
            raise ValueError(
                f"No randomized PCA score file found for method={method}, seed={seed}"
            )

        name = f"{method}_k{k}_{selection}_seed{seed}"
        adata = run_knn(random_scores[key], k, args.n_neighbors, args.flavor, args.random_seed)
        out = write_neighbors_graph(adata, args.output_dir, name)

        manifest.append({
            "method": method,
            "k": k,
            "selection": selection,
            "pca_seed": seed,
            "pca_file": str(random_scores[key]),
            "neighbors_file": str(out),
        })

    manifest = pd.DataFrame(manifest)
    manifest_file = Path(args.output_dir) / f"{args.name}_selected_knn_manifest.tsv"
    manifest.to_csv(manifest_file, sep="\t", index=False)

    print("\nSelected kNN graphs")
    print("-------------------")
    print(manifest.to_string(index=False))
    print(f"\nWrote: {manifest_file}")


def main():
    args = parse_args()

    Path(args.output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

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
