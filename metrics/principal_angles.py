#!/usr/bin/env python3
"""
Principal-angle subspace error collector for OmniBenchmark.

This module compares randomized PCA solutions against the corresponding exact
PCA solution at truncation dimensions selected by the eigengap analysis.

The collector receives:
- all PCA loading files produced by the PCA stage;
- the selected k values produced from exact PCA eigengaps.

For each PCA implementation (e.g. Scanpy or Scrapper), the exact loading
matrix is paired with all randomized-seed loading matrices from the same
implementation.

For each selected k, the first k loading vectors define the exact and
randomized PCA subspaces. Their disagreement is measured using principal
angles:

    subspace_error = sqrt(mean(sin(theta_i)^2))

where theta_i are the principal angles between the two k-dimensional
subspaces.

For every implementation and k, the randomized seeds with the smallest,
median-nearest, and largest subspace errors are selected.

Outputs
-------
subspace_errors.tsv
    Subspace error and maximum principal angle for every randomized seed and k.

selected_seeds.tsv
    Best, median and worst randomized seed for every implementation and k.

pca_manifest.tsv
    Exact and randomized PCA score files together with implementation,
    PCA type and random seed, for downstream selected kNN construction.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles


def load_loadings(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.astype(np.float64)


def get_pca_module(path):
    parts = Path(path).parts
    if "PCA" not in parts:
        raise ValueError(f"Could not identify PCA module from path: {path}")
    i = parts.index("PCA")
    return parts[i + 1]

def get_dataset(path: Path):
    parts = path.parts
    if "DATA" not in parts:
        raise ValueError(f"Could not identify dataset from {path}")
    return parts[parts.index("DATA") + 1]

def get_method_and_type(module):
    if module.endswith("_exact"):
        return module.removesuffix("_exact"), "exact"
    if module.endswith("_random"):
        return module.removesuffix("_random"), "random"

    raise ValueError(f"PCA module '{module}' must end in '_exact' or '_random'.")


def get_random_seed(loadings_file):
    parameters_file = Path(loadings_file).parent / "parameters.json"
    if not parameters_file.exists():
        raise FileNotFoundError(f"Could not find parameters.json for {loadings_file}")
    with open(parameters_file) as f:
        parameters = json.load(f)

    return int(parameters["random_seed"])


def compute_subspace_error(exact_loadings, randomized_loadings, k):
    """
    Compare the first k-dimensional PCA subspaces.

    Returns
    -------
    subspace_error :
        sqrt(mean(sin(theta_i)^2))

        0 = identical subspaces
        1 = maximally different

    max_angle_deg :
        Largest principal angle in degrees.
    """

    U_exact = exact_loadings.iloc[:, :k].to_numpy()
    U_random = randomized_loadings.iloc[:, :k].to_numpy()

    angles = subspace_angles(U_exact, U_random)

    subspace_error = np.sqrt(np.mean(np.sin(angles) ** 2))
    max_angle_deg = np.degrees(np.max(angles))

    return subspace_error, max_angle_deg


def main():
    parser = argparse.ArgumentParser(description="Collect PCA loadings and compute principal-angle subspace errors.")

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pca_loadings", nargs="+", required=True)
    parser.add_argument("--selected_k", nargs="+", required=True)

    args, _ = parser.parse_known_args()
    pca_loading_paths = [Path(path) for path in args.pca_loadings if str(path).endswith("_loadings.tsv")]
    pca_score_paths = [Path(path) for path in args.pca_loadings if str(path).endswith("_pcas.tsv")]
    selected_k_paths = [Path(path) for path in args.selected_k if str(path).endswith("_smallest_eigengaps.tsv")]
    pca_manifest = []

    for path in pca_score_paths:
        dataset = get_dataset(path)
        module = get_pca_module(path)
        method, pca_type = get_method_and_type(module)

        if pca_type == "exact":
            seed = ""
        else:
            seed = get_random_seed(path)

        pca_manifest.append({
            "dataset": dataset,
            "method": method,
            "pca_type": pca_type,
            "seed": seed,
            "pca_file": str(path),
        })

    pca_manifest = pd.DataFrame(pca_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

#file lookup
#loadings 
    exact_files = {}
    random_files = {}

    for path in pca_loading_paths:
        module = get_pca_module(path)
        method, pca_type = get_method_and_type(module)
        dataset = get_dataset(path)
        key = (dataset, method)

        if pca_type == "exact":
            if key in exact_files:
                raise ValueError(f"Multiple exact PCA files found for {key}")
            exact_files[key] = Path(path)
        else:
            random_files.setdefault(key, []).append(Path(path))

#selected values from eigenvalue gap metric 
    selected_k_files = {}

    for path in selected_k_paths:
        module = get_pca_module(path)
        method, pca_type = get_method_and_type(module)
        dataset = get_dataset(path)
        key = (dataset, method)

        if pca_type != "exact":
            raise ValueError(f"selected_k must originate from exact PCA, got {module}")
        if key in selected_k_files:
            raise ValueError(f"Multiple selected-k files found for {key}")

        selected_k_files[key] = Path(path)

#compute subspace errors using angles from above
    results = []

    for (dataset, method), exact_file in exact_files.items():
        key = (dataset, method)

        if key not in random_files:
            raise ValueError(f"No randomized PCA files found for {key}")
        if key not in selected_k_files:
            raise ValueError(f"No selected-k file found for {key}")

        exact = load_loadings(exact_file)
        selected = pd.read_csv(selected_k_files[key], sep="\t")
        selected_k = selected["k"].astype(int).tolist()

        print(f"\n{dataset} / {method}")
        print(f"Exact: {exact_file}")
        print(f"Selected k: {selected_k}")
        print(f"Randomized runs: {len(random_files[key])}")

        for random_file in random_files[key]:
            seed = get_random_seed(random_file)
            randomized = load_loadings(random_file)

            if set(exact.index) != set(randomized.index):
                raise ValueError(f"Genes differ between exact and randomized PCA for {dataset}, {method}, seed {seed}")

            randomized = randomized.loc[exact.index]

            for k in selected_k:
                error, max_angle = compute_subspace_error(exact, randomized, k)

                results.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "k": k,
                    "subspace_error": error,
                    "max_angle_deg": max_angle
                })

    results = pd.DataFrame(results).sort_values(["dataset", "method", "k", "seed"])
#selecting best/median/worst seeds 
    selected_rows = []
    for (method, k), subset in results.groupby(["dataset", "method", "k"]):
        best = subset.loc[subset["subspace_error"].idxmin()]
        worst = subset.loc[subset["subspace_error"].idxmax()]
        median_value = subset["subspace_error"].median()
        median = subset.loc[(subset["subspace_error"] - median_value).abs().idxmin()]

        for label, row in [("best", best), ("median", median), ("worst", worst)]:
            selected_rows.append({
                "dataset": dataset,
                "method": method,
                "k": int(k),
                "selection": label,
                "seed": int(row["seed"]),
                "subspace_error": row["subspace_error"],
                "max_angle_deg": row["max_angle_deg"]})

    selected_seeds = pd.DataFrame(selected_rows)
    errors_file = output_dir / "subspace_errors.tsv"
    selected_file = output_dir / "selected_seeds.tsv"
    manifest_file = output_dir / "pca_manifest.tsv"

    results.to_csv(errors_file, sep="\t", index=False)
    selected_seeds.to_csv(selected_file, sep="\t", index=False)
    pca_manifest.to_csv(manifest_file, sep="\t", index=False)

    print("\nSelected seeds")
    print("--------------")
    print(selected_seeds.to_string(index=False))
    print(f"\nWrote: {errors_file}")
    print(f"Wrote: {selected_file}")
    print(f"Wrote: {manifest_file}")


if __name__ == "__main__":
    main()