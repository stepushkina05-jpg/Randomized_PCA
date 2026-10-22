#!/usr/bin/env python3
"""Figure 3: rank-50 accuracy by PCA approximation settings.

Requires: pandas, numpy, matplotlib.
Input: rank50_approximation_accuracy.csv (one row per seed and setting).
Outputs: figure3_rank50_median_iqr.pdf and .png.

The lines show median accuracy; vertical whiskers are the 25th–75th
percentile range over the 50 seeds at each setting.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Edit this path if the CSV is stored elsewhere.
INPUT = Path("test_results/newrank50_accuracy/rank50_approximation_accuracy.csv")
OUT = Path(__file__).resolve().parent

methods = ["scanpy", "rapids", "scrapper"]
datasets = ["31_narrow_signal", "35_broad_signal", "38_strong_signal"]
method_names = {"scanpy": "Scanpy", "rapids": "RAPIDS-singlecell", "scrapper": "Scrapper"}
dataset_names = {"31_narrow_signal": "Narrow signal", "35_broad_signal": "Broad signal", "38_strong_signal": "Strong signal"}
iterations, extras = [2, 3, 4, 5], [10, 20, 30, 40]

# Large, print-friendly typography; figure sized to occupy a full landscape page.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 19, "axes.titleweight": "bold",
    "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 12,
    "legend.fontsize": 15, "figure.titlesize": 23,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

df = pd.read_csv(INPUT)
required = {"method", "dataset", "n_iter", "n_oversamples", "seed", "rank_accuracy"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing columns: {sorted(missing)}")

# Prevent silent averaging over missing or duplicated seed runs.
df = df[df.method.isin(methods) & df.dataset.isin(datasets)].copy()
counts = df.groupby(["dataset", "method", "n_iter", "n_oversamples"])["seed"].nunique()
if len(counts) != 3 * 3 * 4 * 4 or (counts != 50).any():
    raise ValueError("Expected 50 distinct seeds for each of 144 dataset/method/settings groups")

summary = (df.groupby(["dataset", "method", "n_iter", "n_oversamples"])
           .rank_accuracy.agg(median="median", q25=lambda x: x.quantile(.25),
                              q75=lambda x: x.quantile(.75)).reset_index())

# Give each solver a consistent y-range across datasets. Ranges differ by
# solver because Scrapper's accuracy varies much more in the low-budget runs.
y_limits = {}
for method in methods:
    sm = summary[summary.method == method]
    low = np.floor((sm.q25.min() - .02) / .02) * .02
    y_limits[method] = (max(0., low), 1.007)

fig, axes = plt.subplots(3, 3, figsize=(18, 13.4), sharex=True)
markers = ["o", "s", "^", "D"]
# Small horizontal offsets separate overlapping percentile intervals at same x.
x_offsets = [-0.9, -0.3, 0.3, 0.9]

for row, dataset in enumerate(datasets):
    for col, method in enumerate(methods):
        ax = axes[row, col]
        for idx, n_iter in enumerate(iterations):
            v = (summary[(summary.dataset == dataset) & (summary.method == method)
                         & (summary.n_iter == n_iter)].sort_values("n_oversamples"))
            x = v.n_oversamples.to_numpy(dtype=float) + x_offsets[idx]
            y = v["median"].to_numpy(dtype=float)
            lo = np.maximum(y - v.q25.to_numpy(dtype=float), 0)
            hi = np.maximum(v.q75.to_numpy(dtype=float) - y, 0)
            ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt=markers[idx] + "-",
                        linewidth=2.2, markersize=7.5, elinewidth=1.5,
                        capsize=3.5, capthick=1.4, label=f"{n_iter}", zorder=3)
        ax.axhline(1., ls="--", lw=1, color="0.48", alpha=.65, zorder=0)
        ax.set_ylim(*y_limits[method])
        ax.set_xlim(7.5, 42.5)
        ax.set_xticks(extras)
        ax.grid(axis="y", color="0.88", lw=.8, zorder=0)
        ax.tick_params(axis="both", pad=5)
        if row == 0:
            ax.set_title(method_names[method], pad=15)
        if col == 0:
            ax.set_ylabel(dataset_names[dataset] + "\nRank-50 accuracy", labelpad=12)
        else:
            ax.set_ylabel("Rank-50 accuracy", labelpad=8)
        if row == 2:
            ax.set_xlabel("extra.work" if method == "scrapper" else "n_oversamples", labelpad=9)

# One shared legend makes the 9 panels much easier to scan.
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title="Iteration setting", loc="upper center", ncol=4,
           bbox_to_anchor=(.52, .938), frameon=False, handlelength=2.4,
           columnspacing=2.0, title_fontsize=15)
fig.suptitle("Rank-50 approximation accuracy across 50 PCA seeds", y=.991, fontweight="bold")
fig.text(.5, .025, "Points: median over 50 seeds  |  Whiskers: interquartile range (25th–75th percentile)  |  Y-axis scale differs by method",
         ha="center", fontsize=13)
fig.subplots_adjust(left=.095, right=.985, bottom=.10, top=.812, wspace=.26, hspace=.34)

for ext in ("pdf", "png"):
    file = OUT / f"figure3_rank50_median_iqr.{ext}"
    fig.savefig(file, dpi=300, facecolor="white", bbox_inches="tight")
    print("Saved:", file)
plt.close(fig)
