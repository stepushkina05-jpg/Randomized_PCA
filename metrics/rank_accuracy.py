#!/usr/bin/env python3
"""Rank-k PCA reconstruction accuracy, one invocation per PCA run.

Metric = ||X_centered - Z_exact V_exact.T||_2 /
         ||X_centered - Z_approx V_approx.T||_2.

Input HDF5 must have /matrix/{data,indices,indptr,shape,genes,barcodes};
its sparse CSC matrix is genes x cells. PCA TSVs are ID-indexed, with
cells x components for scores and genes x components for loadings.

Upstream stages must explicitly provide the input matrix,
exact reference scores/loadings, and approximate scores/loadings

Output: csv file summary with approximation accuracy score for each randomized PCA run.
"""

import argparse
from pathlib import Path
import warnings
import re
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, svds


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_root", type=Path, required=True)
    p.add_argument("--exact_root", type=Path, required=True)
    p.add_argument("--random_root", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--n_components", type=int, default=50)
    p.add_argument("--reference_map", nargs="*", default=[], metavar="METHOD=REFERENCE",
                   help="Optional exact reference mapping, e.g. scrapper=scanpy")
    return p.parse_args()


def decode(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def load_input(path):
    with h5py.File(path, "r") as h5:
        group = h5["matrix"]
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=tuple(map(int, group["shape"][:])),
        ).T.tocsr().astype(np.float64)
        cell_ids = decode(group["barcodes"][:])
        gene_ids = decode(group["genes"][:])
    if matrix.shape != (len(cell_ids), len(gene_ids)):
        raise ValueError(f"Matrix shape {matrix.shape} does not match cell/gene IDs")
    if len(set(cell_ids)) != len(cell_ids) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("Duplicate cell or gene IDs in input matrix")
    gene_means = np.asarray(matrix.mean(axis=0)).ravel()
    return matrix, gene_means, cell_ids, gene_ids


def load_tsv(path, ordered_ids, rank):
    table = pd.read_csv(path, sep="\t", index_col=0)
    table.index = table.index.astype(str)
    if not table.index.is_unique or len(table.index) != len(ordered_ids):
        raise ValueError(f"Duplicate or missing identifiers in {path}")
    if set(table.index) != set(ordered_ids):
        raise ValueError(f"Identifiers in {path} do not match the input matrix")
    if table.shape[1] < rank:
        raise ValueError(f"{path}: expected >= {rank} PCs, found {table.shape[1]}")
    result = table.loc[ordered_ids].iloc[:, :rank].to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite values in {path}")
    return result


def spectral_residual(matrix, gene_means, scores, loadings):
    n_cells, n_genes = matrix.shape
    ones = np.ones(n_cells, dtype=np.float64)

    def matvec(v):
        # (X - 1 mean.T - scores @ loadings.T) @ v
        return matrix @ v - ones * (gene_means @ v) - scores @ (loadings.T @ v)

    def rmatvec(u):
        # transpose of the identical centered residual operator
        return matrix.T @ u - gene_means * np.sum(u) - loadings @ (scores.T @ u)

    def matmat(v):
        return matrix @ v - np.outer(ones, gene_means @ v) - scores @ (loadings.T @ v)

    def rmatmat(u):
        return matrix.T @ u - np.outer(gene_means, np.sum(u, axis=0)) - loadings @ (scores.T @ u)

    operator = LinearOperator((n_cells, n_genes), matvec=matvec, rmatvec=rmatvec,
                              matmat=matmat, rmatmat=rmatmat, dtype=np.float64)
    # Repeatable iterative estimate; no dense residual materialization.
    return float(svds(operator, k=1, which="LM", return_singular_vectors=False,
                      tol=1e-6, random_state=0)[0])


def main():
    args = arguments()
    k = args.n_components
    if k < 1:
        raise ValueError("n_components must be positive")
    reference_map = dict(pair.split("=", 1) for pair in args.reference_map)
    matrix_cache, exact_cache, rows = {}, {}, []

    for method_dir in sorted(p for p in args.random_root.iterdir() if p.is_dir()):
        method = method_dir.name
        reference_method = reference_map.get(method, method)
        for dataset_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            dataset = dataset_dir.name
            if dataset not in matrix_cache:
                input_path = args.input_root / f"{dataset}_normalized_selected.h5"
                matrix_cache[dataset] = load_input(input_path)
            X, means, cells, genes = matrix_cache[dataset]
            if k >= min(X.shape):
                raise ValueError(f"{dataset}: k={k} must be smaller than min(matrix.shape)={min(X.shape)}")

            ref_key = (reference_method, dataset)
            if ref_key not in exact_cache:
                ref_dir = args.exact_root / reference_method / dataset
                Z_ref = load_tsv(ref_dir / f"{dataset}_exact_pcas.tsv", cells, k)
                V_ref = load_tsv(ref_dir / f"{dataset}_exact_loadings.tsv", genes, k)
                exact_cache[ref_key] = spectral_residual(X, means, Z_ref, V_ref)
            exact_error = exact_cache[ref_key]

            for param_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                match = re.fullmatch(r"n(\d+)_over(\d+)", param_dir.name)
                if not match:
                    continue
                n_iter, n_over = map(int, match.groups())
                pattern = rf"{re.escape(dataset)}_seed(\d+)_n{n_iter}_over{n_over}_pcas\.tsv"
                for pca_file in sorted(param_dir.glob("*_pcas.tsv")):
                    run = re.fullmatch(pattern, pca_file.name)
                    if run is None:
                        raise ValueError(f"Unexpected PCA filename: {pca_file}")
                    seed = int(run.group(1))
                    loading_file = pca_file.with_name(pca_file.name.replace("_pcas.tsv", "_loadings.tsv"))
                    Z = load_tsv(pca_file, cells, k)
                    V = load_tsv(loading_file, genes, k)
                    random_error = spectral_residual(X, means, Z, V)
                    if random_error <= 0:
                        raise ValueError(f"Non-positive randomized residual: {pca_file}")
                    accuracy = exact_error / random_error
                    if accuracy > 1.001:
                        warnings.warn(f"Accuracy > 1 for {pca_file}: verify reference and PCA conventions")
                    rows.append(dict(method=method, dataset=dataset, n_iter=n_iter,
                                     n_oversamples=n_over, seed=seed, exact_error=exact_error,
                                     randomized_error=random_error, rank_accuracy=accuracy,
                                     row_label=f"{method} | n{n_iter}_over{n_over}"))
                    print(f"{method} | {dataset} | n{n_iter}_over{n_over} | seed {seed}: {accuracy:.6f}")

    if not rows:
        raise ValueError(f"No randomized PCA runs found beneath {args.random_root}")
    cols = ["method", "dataset", "n_iter", "n_oversamples", "seed", "exact_error",
            "randomized_error", "rank_accuracy", "row_label"]
    results = pd.DataFrame(rows)[cols].sort_values(cols[:5]).reset_index(drop=True)
    if results.duplicated(cols[:5]).any():
        raise ValueError("Duplicate method/dataset/parameter/seed runs found")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(results)} runs to {args.output_csv}")

if __name__ == "__main__":
    main()