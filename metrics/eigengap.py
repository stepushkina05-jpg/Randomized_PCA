#!/usr/bin/env python3
"""
Exact-PCA eigengap metric for OmniBenchmark.

This module analyzes the spectrum of an exact PCA solution and identifies
truncation dimensions at which the PCA subspace is weakly separated.

Input
-----
pca_scores:
    TSV file containing exact PCA scores. The first column contains cell IDs
    and the remaining columns contain principal-component scores.

Method
------
For each principal component, the corresponding eigenvalue is estimated as
the sample variance of the exact PCA scores.

The relative spectral gap after component k is

    g_k = (lambda_k - lambda_{k+1}) / lambda_k

Small relative eigengaps indicate weak separation between adjacent principal
components and are therefore candidate dimensions at which randomized PCA may
show increased subspace sensitivity.

Outputs
-------
{dataset}_eigengaps.tsv
    Relative eigengaps for all k = 1, ..., n_components - 1.

{dataset}_smallest_eigengaps.tsv
    The five k values with the smallest relative eigengaps.

{dataset}_eigengaps.png
    Plot of the full relative eigengap spectrum with the five selected
    truncation dimensions highlighted.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_relative_spectral_gaps(pca_file):
    scores = pd.read_csv(pca_file, sep="\t", index_col=0)
    scores = scores.apply(pd.to_numeric, errors="raise")
    eigenvalues = scores.var(axis=0, ddof=1).to_numpy(dtype=np.float64)
    # relative gap after PC k
    relative_gaps = (eigenvalues[:-1] - eigenvalues[1:]) / eigenvalues[:-1]
    return pd.DataFrame({
        "k": np.arange(1, len(eigenvalues)),
        "eigenvalue_k": eigenvalues[:-1],
        "eigenvalue_next": eigenvalues[1:],
        "relative_gap": relative_gaps,})
def plot_spectral_gaps(gaps, smallest, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(gaps["k"], gaps["relative_gap"], marker="o", markersize=3, label="Relative spectral gap")
    plt.scatter(smallest["k"], smallest["relative_gap"], s=80, label="5 smallest gaps")
    for _, row in smallest.iterrows():
        plt.annotate(
            f"k={int(row['k'])}",
            (row["k"], row["relative_gap"]),
            xytext=(5, 5),
            textcoords="offset points")
    plt.xlabel("Number of retained PCs (k)")
    plt.ylabel("Relative spectral gap")
    plt.title("Relative spectral gaps of exact PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
def main():
    parser = argparse.ArgumentParser(description="Compute relative eigengaps from exact PCA scores.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pca_scores", required=True)
    parser.add_argument("--n_smallest", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gaps = compute_relative_spectral_gaps(args.pca_scores)
    smallest = gaps.nsmallest(args.n_smallest, "relative_gap").sort_values("k")

    gaps_file = output_dir / f"{args.name}_eigengaps.tsv"
    smallest_file = output_dir / f"{args.name}_smallest_eigengaps.tsv"
    figure_file = output_dir / f"{args.name}_eigengaps.png"

    gaps.to_csv(gaps_file, sep="\t", index=False)
    smallest.to_csv(smallest_file, sep="\t", index=False)
    plot_spectral_gaps(gaps, smallest, figure_file)
    print("\nSelected smallest eigengaps:")
    print(smallest.to_string(index=False))
    print(f"\nWrote: {gaps_file}")
    print(f"Wrote: {smallest_file}")
    print(f"Wrote: {figure_file}")

if __name__ == "__main__":
    main()