#!/usr/bin/env python3
"""Compute PCA eigenvalues from exact (unscaled) PC scores and plot their spectrum.

For centered PCA scores Z = U S, lambda_j = Var(Z_j) = S_j^2/(n_cells-1).
The script searches a user-supplied directory; no dataset or method names are fixed.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exact_dir", type=Path, required=True, help="Directory containing exact PCA score TSVs (searched recursively)")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--n_components", type=int, default=None, help="Use only the first N PCs; by default use all")
    args = p.parse_args()

    files = sorted(args.exact_dir.rglob("*_pcas.tsv"))
    if not files:
        raise FileNotFoundError(f"No *_pcas.tsv files found below {args.exact_dir}")
    if args.n_components is not None and args.n_components < 1:
        raise ValueError("--n_components must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 19, "axes.titlesize": 23, "axes.labelsize": 21,
                         "xtick.labelsize": 18, "ytick.labelsize": 18, "pdf.fonttype": 42})
    rows = []

    for file in files:
        scores = pd.read_csv(file, sep="\t", index_col=0)
        if args.n_components and scores.shape[1] < args.n_components:
            raise ValueError(f"{file}: expected {args.n_components} PCs, found {scores.shape[1]}")
        z = scores.iloc[:, :args.n_components].to_numpy(dtype=float)
        if z.shape[0] < 2 or not np.isfinite(z).all():
            raise ValueError(f"{file}: need at least 2 cells and finite numeric PC scores")
        eigenvalues = np.var(z, axis=0, ddof=1)
        if eigenvalues[0] <= 0:
            raise ValueError(f"{file}: variance of PC1 must be positive")
        ratios = eigenvalues / eigenvalues[0]
        if np.any(ratios <= 0):
            raise ValueError(f"{file}: cannot plot nonpositive eigenvalues on a log axis")

        dataset = file.stem.removesuffix("_exact_pcas").removesuffix("_pcas")
        source = str(file.relative_to(args.exact_dir))
        folders = list(file.relative_to(args.exact_dir).parts[:-1])
        if not folders or folders[-1] != dataset:
            folders.append(dataset)
        tag = "__".join(folders)
        for k, (eig, ratio) in enumerate(zip(eigenvalues, ratios), 1):
            rows.append({"dataset": dataset, "source": source, "pc": k,
                         "eigenvalue": eig, "relative_eigenvalue": ratio})

        fig, ax = plt.subplots(figsize=(12, 7.5))
        ax.plot(np.arange(1, len(ratios) + 1), ratios, marker="o", lw=2.8, markersize=6)
        ax.set_yscale("log")
        ax.set_xlabel(r"Principal component ($k$)", labelpad=10)
        ax.set_ylabel(r"Relative eigenvalue $\lambda_k/\lambda_1$", labelpad=12)
        ax.set_title(f"{dataset} PCA eigenvalue spectrum", pad=17)
        ax.set_xlim(0, len(ratios) + 1)
        ax.tick_params(axis="both", which="major", length=6, width=1.1)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = args.output_dir / f"{tag}_spectrum.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Wrote: {path}", flush=True)
        plt.close(fig)

    csv = args.output_dir / "pca_eigenvalue_spectra.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"Wrote: {csv} ({len(files)} input files)", flush=True)


if __name__ == "__main__":
    main()
