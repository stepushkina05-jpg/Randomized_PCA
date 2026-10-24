#!/usr/bin/env python3
"""Plot saved kNN retention and Leiden ARI results; no analyses are recomputed. Specify the folder your knn overlap results and clustering ari scores are stored in """

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# Figure styling belongs in the script, not the command-line parser.
plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 22, "axes.labelsize": 23,
    "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 25,
    "figure.titlesize": 27, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
STYLES = {"exact": ("#52525B", "s"), "best": ("#0072B2", "o"), "worst": ("#D55E00", "^")}
OVERLAP_COLS = ["mean_neighbor_retention", "mean_overlap", "mean_knn_overlap", "mean_neighbor_overlap",
                "knn_overlap", "overlap_fraction", "mean_shared_fraction"]


def save(fig, folder, name):
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ari(data, methods, cutoffs, folder, name):
    subset = data[data["k"].isin(cutoffs)]
    fig, axes = plt.subplots(len(cutoffs), len(methods), squeeze=False, sharey=True, figsize=(7 * len(methods), 4.5 * len(cutoffs) + 1.0))
    ref_counts = sorted(subset["n_clusters_reference"].dropna().unique())
    selections = [s for s in ("exact", "best", "worst") if s in set(subset["selection"])]
    selections += sorted(set(subset["selection"]) - set(selections))
    palette = plt.get_cmap("tab10")
    colors = {s: STYLES.get(s, (palette(i % 10), "o")) for i, s in enumerate(selections)}
    ymin, ymax = subset["ari"].min(), subset["ari"].max()
    pad = max(0.03, 0.07 * (ymax - ymin))
    xvals = pd.concat([subset["n_clusters_test"], subset["n_clusters_reference"]]).dropna()

    for i, k in enumerate(cutoffs):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            df = subset[(subset["k"] == k) & (subset["method"] == method)]
            for n in ref_counts:
                ax.axvline(n, ls="--", lw=1, color="0.55", alpha=0.6)
            for selection in selections:
                values = df[df["selection"] == selection].sort_values("resolution")
                if not values.empty:
                    color, marker = colors[selection]
                    ax.plot(values["n_clusters_test"], values["ari"], color=color,
                            marker=marker, lw=2.1, markersize=8)
            ax.set_xlim(xvals.min() - 2, xvals.max() + 2)
            ax.set_ylim(max(-1, ymin - pad), min(1, ymax + pad))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.grid(axis="y", alpha=0.25)
            if i == 0: ax.set_title(method)
            if j == 0: ax.set_ylabel(f"{k} PCs\nARI")
            if i < len(cutoffs) - 1: ax.tick_params(labelbottom=False)

    handles = [Line2D([], [], color=colors[s][0], marker=colors[s][1], lw=2,
                      markersize=9, label={"exact": "Exact PCA", "best": "Best seed", "worst": "Worst seed"}.get(s, s))
               for s in selections]
    fig.suptitle("Clustering agreement versus detected cluster count", y=0.99, weight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=min(4, len(handles)), frameon=False)
    fig.supxlabel("Number of Leiden clusters", y=0.055)
    counts = ", ".join(str(int(n)) for n in ref_counts)
    fig.text(0.5, 0.015, f"Dashed line: {counts} reference annotation groups  |  Points joined by Leiden resolution",
             ha="center", fontsize=14)
    fig.subplots_adjust(left=0.11 if len(methods) == 1 else 0.08,
                        right=0.98, top=0.82, bottom=0.12, wspace=0.22, hspace=0.22)
    save(fig, folder, name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--main_cutoffs", type=int, nargs="+")
    args = p.parse_args()

    overlap_files = list(args.results_dir.rglob("*_knn_overlap.tsv"))
    ari_files = list(args.results_dir.rglob("*_clustering_ari.tsv"))

    if not overlap_files or not ari_files:
        raise FileNotFoundError(f"Missing kNN-overlap or clustering-ARI TSVs in {args.results_dir}")
    overlap = pd.read_csv(max(overlap_files, key=lambda f: f.stat().st_mtime), sep="\t")
    ari = pd.concat([pd.read_csv(f, sep="\t") for f in ari_files], ignore_index=True)
    overlap_col = next((c for c in OVERLAP_COLS if c in overlap), None)
    if overlap_col is None:
        raise ValueError(f"Missing neighbour-retention metric. Columns: {list(overlap)}")
    if "reference" not in ari: ari["reference"] = "reference"
    for df in (overlap, ari):
        df["selection"] = df["selection"].astype(str).str.lower()
    ari = ari.drop_duplicates(["dataset", "reference", "method", "k", "resolution", "selection", "pca_seed"], keep="last")

    for (dataset, reference), data in ari.groupby(["dataset", "reference"], sort=True):
        folder = args.output_dir / (re.sub(r"[^\w.-]+", "_", str(dataset)) + "__" + re.sub(r"[^\w.-]+", "_", str(reference)))
        folder.mkdir(parents=True, exist_ok=True)
        methods = sorted(data["method"].unique())
        cutoffs = sorted(data["k"].unique())
        short = [k for k in (args.main_cutoffs or [cutoffs[0], cutoffs[-1]]) if k in cutoffs]
        if not short: short = [cutoffs[0], cutoffs[-1]]
        short = sorted(set(short))
        data.to_csv(folder / "cluster_counts_ari.tsv", sep="\t", index=False)

        ov = overlap[(overlap["dataset"] == dataset) & (overlap["method"].isin(methods))]
        fig, axes = plt.subplots(1, len(methods), squeeze=False, sharey=True, figsize=(7 * len(methods), 6.5))
        for j, method in enumerate(methods):
            ax = axes[0, j]
            df = ov[ov["method"] == method]
            ks = sorted(df["k"].unique())
            for selection in sorted(df["selection"].unique()):
                if selection == "exact": continue
                values = df[df["selection"] == selection].sort_values("k")
                color, marker = STYLES.get(selection, ("C0", "o"))
                ax.plot([ks.index(k) for k in values["k"]], values[overlap_col],
                        color=color, marker=marker, lw=2.5, markersize=9, label=selection.capitalize())
            ax.set_xticks(range(len(ks)), ks)
            ax.set_title(method)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.25)
            if j == 0: ax.set_ylabel("Mean neighbour retention")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle("Neighbour retention relative to exact PCA", y=0.99, weight="bold")
        if handles: fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=len(handles), frameon=False)
        fig.supxlabel("Retained principal components (k)", y=0.055)
        fig.subplots_adjust(left=0.12 if len(methods) == 1 else 0.08, right=0.98, top=0.69, bottom=0.17, wspace=0.22)
        save(fig, folder, "neighbour_retention")

        plot_ari(data, methods, short, folder, "ari_vs_cluster_count_main")
        plot_ari(data, methods, cutoffs, folder, "ari_vs_cluster_count_all_cutoffs")
        print(f"Wrote figures for {dataset} / {reference}: {folder}")


if __name__ == "__main__":
    main()
