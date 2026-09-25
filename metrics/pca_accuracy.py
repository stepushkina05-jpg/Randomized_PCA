#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_matrix(path: Path):
    with h5py.File(path, "r") as h5:
        g = h5["matrix"]
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(g["shape"][:])
        gene_ids = g["genes"][:].astype(str)
        cell_ids = g["barcodes"][:].astype(str)

    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr().astype(float)
    X = X.toarray()
    X -= X.mean(axis=0, keepdims=True)

    return X, cell_ids, gene_ids


def load_pca_table(path: Path):
    df = pd.read_csv(path, sep="\t")
    ids = df.iloc[:, 0].astype(str)
    values = df.iloc[:, 1:].astype(float)
    values.index = ids
    return values


def get_pca_module(path: Path):
    parts = list(path.parts)

    if "PCA" not in parts:
        raise ValueError(f"Could not identify PCA module from {path}")

    return parts[parts.index("PCA") + 1]


def get_method_and_type(module: str):
    if module.endswith("_exact"):
        return module[:-6], "exact"

    if module.endswith("_random"):
        return module[:-7], "random"

    raise ValueError(f"Unknown PCA module: {module}")


def get_parameters(path: Path):
    parameter_file = path.parent / "parameters.json"

    if not parameter_file.exists():
        return {}

    with open(parameter_file) as f:
        params = json.load(f)

    if "parameters" in params and isinstance(params["parameters"], dict):
        params = params["parameters"]

    return params


def align_table(df: pd.DataFrame, ids, name):
    ids = pd.Index(ids)

    missing = ids.difference(df.index)
    extra = df.index.difference(ids)

    if len(missing) > 0 or len(extra) > 0:
        raise ValueError(f"{name} IDs do not match input matrix.")

    return df.loc[ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_files", "--pca_loadings", dest="pca_files", nargs="+", required=True)
    parser.add_argument("--matrix_h5", "--normalized_selected_h5", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_paths = [Path(path) for path in args.pca_files]
    score_paths = [path for path in pca_paths if str(path).endswith("_pcas.tsv")]
    loading_paths = [path for path in pca_paths if str(path).endswith("_loadings.tsv")]

    loading_by_dir = {path.parent: path for path in loading_paths}

    runs = []

    for score_path in score_paths:
        if score_path.parent not in loading_by_dir:
            raise ValueError(f"No loading file found for {score_path}")

        module = get_pca_module(score_path)
        method, pca_type = get_method_and_type(module)
        params = get_parameters(score_path)

        runs.append({
            "method": method,
            "pca_type": pca_type,
            "score_file": score_path,
            "loading_file": loading_by_dir[score_path.parent],
            "params": params,
        })

    exact_runs = {}

    for run in runs:
        if run["pca_type"] != "exact":
            continue

        method = run["method"]

        if method in exact_runs:
            raise ValueError(f"Multiple exact PCA runs found for {method}")

        exact_runs[method] = run

    X, cell_ids, gene_ids = load_matrix(Path(args.matrix_h5))

    accuracy_rows = []
    correlation_rows = []

    for run in runs:
        if run["pca_type"] != "random":
            continue

        method = run["method"]

        if method not in exact_runs:
            raise ValueError(f"No exact PCA run found for {method}")

        exact = exact_runs[method]

        exact_scores = align_table(load_pca_table(exact["score_file"]), cell_ids, "Exact score")
        random_scores = align_table(load_pca_table(run["score_file"]), cell_ids, "Random score")
        exact_loadings = align_table(load_pca_table(exact["loading_file"]), gene_ids, "Exact loading")
        random_loadings = align_table(load_pca_table(run["loading_file"]), gene_ids, "Random loading")

        n_components = min(exact_scores.shape[1], random_scores.shape[1], exact_loadings.shape[1], random_loadings.shape[1])

        exact_scores_np = exact_scores.iloc[:, :n_components].to_numpy()
        random_scores_np = random_scores.iloc[:, :n_components].to_numpy()
        exact_loadings_np = exact_loadings.iloc[:, :n_components].to_numpy()
        random_loadings_np = random_loadings.iloc[:, :n_components].to_numpy()

        pc_correlations = []

        for j in range(n_components):
            corr = np.corrcoef(exact_scores_np[:, j], random_scores_np[:, j])[0, 1]
            pc_correlations.append(abs(corr))

        pc_correlations = np.asarray(pc_correlations)

        X_exact = exact_scores_np @ exact_loadings_np.T
        X_random = random_scores_np @ random_loadings_np.T

        exact_error = np.linalg.norm(X - X_exact, ord=2)
        random_error = np.linalg.norm(X - X_random, ord=2)
        approximation_quality = exact_error / random_error

        params = run["params"]
        seed = params.get("random_seed", "")
        n_iter = params.get("n_iter", "")
        n_oversamples = params.get("n_oversamples", "")
        dataset = run["score_file"].name.removesuffix("_pcas.tsv")

        accuracy_rows.append({
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "n_iter": n_iter,
            "n_oversamples": n_oversamples,
            "n_components": n_components,
            "approximation_quality": approximation_quality,
            "exact_residual_norm": exact_error,
            "random_residual_norm": random_error,
            "mean_abs_pc_corr": pc_correlations.mean(),
        })

        correlation_row = {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "n_iter": n_iter,
            "n_oversamples": n_oversamples,
        }

        for j, corr in enumerate(pc_correlations, start=1):
            correlation_row[f"PC{j}_corr"] = corr

        correlation_rows.append(correlation_row)

    accuracy = pd.DataFrame(accuracy_rows)
    correlations = pd.DataFrame(correlation_rows)

    accuracy_file = output_dir / "approximation_accuracy.tsv"
    correlation_file = output_dir / "pc_correlations.tsv"

    accuracy.to_csv(accuracy_file, sep="\t", index=False)
    correlations.to_csv(correlation_file, sep="\t", index=False)

    print()
    print("PCA approximation accuracy")
    print("--------------------------")
    print(accuracy.to_string(index=False))

    print()
    print(f"Wrote: {accuracy_file}")
    print(f"Wrote: {correlation_file}")


if __name__ == "__main__":
    main()