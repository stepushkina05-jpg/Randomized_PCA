#!/usr/bin/env python3
"""Collect absolute PC-score correlations with exact PCA across all randomized runs.

Expected folders:
  exact_root/<reference_method>/<dataset>/<dataset>_exact_pcas.tsv
  random_root/<method>/<dataset>/n<iter>_over<over>/<dataset>_seed<seed>_n<iter>_over<over>_pcas.tsv

One output CSV contains one row per method/dataset/parameter/seed combination.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exact_root", type=Path, required=True)
    p.add_argument("--random_root", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--n_components", type=int, default=50)
    p.add_argument("--reference_map", nargs="*", default=[], metavar="METHOD=REFERENCE")
    return p.parse_args()


def load_scores(path, k, cell_ids=None):
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    if not df.index.is_unique or df.shape[1] < k:
        raise ValueError(f"Duplicate cell IDs or fewer than {k} PCs in {path}")
    if cell_ids is not None:
        if len(df) != len(cell_ids) or set(df.index) != set(cell_ids):
            raise ValueError(f"Cell IDs do not match exact PCA: {path}")
        df = df.loc[cell_ids]
    Z = df.iloc[:, :k].to_numpy(dtype=float)
    if not np.isfinite(Z).all():
        raise ValueError(f"Non-finite PCA scores: {path}")
    return Z, df.index


def absolute_correlations(exact, random, path):
    x = exact - exact.mean(axis=0)
    y = random - random.mean(axis=0)
    denominator = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
    if np.any(denominator == 0):
        raise ValueError(f"Cannot correlate a constant PC score vector: {path}")
    return np.clip(np.abs(np.sum(x * y, axis=0) / denominator), 0, 1)


def main():
    args = arguments()
    k = args.n_components
    if k < 1:
        raise ValueError("n_components must be positive")
    reference_map = dict(pair.split("=", 1) for pair in args.reference_map)
    exact_cache, rows = {}, []

    for method_dir in sorted(p for p in args.random_root.iterdir() if p.is_dir()):
        method = method_dir.name
        reference_method = reference_map.get(method, method)
        for dataset_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            dataset = dataset_dir.name
            key = (reference_method, dataset)
            if key not in exact_cache:
                exact_file = args.exact_root / reference_method / dataset / f"{dataset}_exact_pcas.tsv"
                exact_cache[key] = load_scores(exact_file, k)
            exact, cell_ids = exact_cache[key]

            for param_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                match = re.fullmatch(r"n(\d+)_over(\d+)", param_dir.name)
                if not match:
                    continue
                n_iter, n_over = map(int, match.groups())
                pattern = rf"{re.escape(dataset)}_seed(\d+)_n{n_iter}_over{n_over}_pcas\.tsv"
                for score_file in sorted(param_dir.glob("*_pcas.tsv")):
                    match = re.fullmatch(pattern, score_file.name)
                    if match is None:
                        raise ValueError(f"Unexpected PCA score filename: {score_file}")
                    seed = int(match.group(1))
                    random, _ = load_scores(score_file, k, cell_ids)
                    correlations = absolute_correlations(exact, random, score_file)
                    row = dict(dataset=dataset, method=method, seed=seed,
                               n_iter=n_iter, n_oversamples=n_over)
                    row.update({f"PC{j}_corr": float(c) for j, c in enumerate(correlations, 1)})
                    rows.append(row)
                    print(f"{method} | {dataset} | n{n_iter}_over{n_over} | seed {seed}: mean |r|={correlations.mean():.4f}", flush=True)

    if not rows:
        raise ValueError(f"No randomized PCA runs found in {args.random_root}")
    result = pd.DataFrame(rows).sort_values(["dataset", "method", "n_iter", "n_oversamples", "seed"])
    if result.duplicated(["dataset", "method", "n_iter", "n_oversamples", "seed"]).any():
        raise ValueError("Duplicate method/dataset/parameter/seed runs")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(result)} runs to {args.output_csv}")


if __name__ == "__main__":
    main()
