#!/usr/bin/env python3
"""Plot PCA accuracy distributions from a collected CSV, without fixed experimental settings."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def parse_args():
    p = argparse.ArgumentParser(description="Plot PCA accuracy across random seeds.")
    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--name", default="pca_accuracy")
    p.add_argument("--metric_col", default="rank_accuracy")
    p.add_argument("--dataset_col", default="dataset")
    p.add_argument("--method_col", default="method")
    p.add_argument("--seed_col", default="seed")
    p.add_argument("--param_cols", nargs="+", help="Optional: specify parameter columns; inferred otherwise.")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    dataset_col, method_col, seed_col, metric_col = (
        args.dataset_col, args.method_col, args.seed_col, args.metric_col
    )
    required = {dataset_col, method_col, seed_col, metric_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ignore standard output fields from the accuracy metric, not experimental settings.
    non_parameters = required | {
        "exact_error", "randomized_error", "row_label", "n_components"
    }
    param_cols = args.param_cols if args.param_cols is not None else [
        c for c in df.columns if c not in non_parameters and not c.startswith("Unnamed:")
    ]
    if missing := set(param_cols) - set(df.columns):
        raise ValueError(f"Unknown parameter columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input CSV has no rows.")
    if df[list(required) + param_cols].isna().any().any():
        raise ValueError("Missing dataset, method, seed, accuracy, or parameter values.")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="raise")

    datasets = sorted(df[dataset_col].unique(), key=str)
    methods = sorted(df[method_col].unique(), key=str)
    n_rows, n_cols = len(datasets), len(methods)

    # Scale the panel width to the number of settings actually present.
    if param_cols:
        max_settings = max(
            len(sub[param_cols].drop_duplicates())
            for _, sub in df.groupby([dataset_col, method_col])
        )
    else:
        max_settings = 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(5.4, 0.39 * max_settings) * n_cols, 4.7 * n_rows),
        squeeze=False, sharey=False
    )


    # A common y-axis range within each method makes comparisons across datasets fair.
    limits = {}
    for method in methods:
        values = df.loc[df[method_col] == method, metric_col]
        low, high = float(values.min()), float(values.max())
        pad = 0.05 * (high - low) if high > low else max(0.01, abs(low) * 0.02)
        limits[method] = (low - pad, high + pad)

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            sub = df[(df[dataset_col] == dataset) & (df[method_col] == method)]
            if sub.empty:
                ax.set_visible(False)
                continue

            if param_cols:
                groups = list(sub.groupby(param_cols, sort=True, dropna=False))
            else:
                groups = [((), sub)]
            box_data, means, labels, first_params = [], [], [], []
            for key, group in groups:
                key = key if isinstance(key, tuple) else (key,)
                vals = group[metric_col].dropna().to_numpy(dtype=float)
                if not len(vals):
                    continue
                box_data.append(vals)
                means.append(np.mean(vals))
                labels.append(" / ".join(map(str, key)) if param_cols else "All runs")
                first_params.append(key[0] if param_cols else None)

            if not box_data:
                ax.set_visible(False)
                continue
            bp = ax.boxplot(
                box_data, patch_artist=True, widths=0.65, showfliers=True,
                medianprops=dict(linewidth=1.4, color="black"),
                whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0),
                boxprops=dict(linewidth=1.0),
                flierprops=dict(marker="o", markersize=2.8, alpha=0.5),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#d9d9d9")
                patch.set_edgecolor("black")

            positions = np.arange(1, len(box_data) + 1)
            ax.plot(positions, means, "o", markersize=3.2, color="black")
            for k in range(1, len(first_params)):
                if first_params[k] != first_params[k - 1]:
                    ax.axvline(k + 0.5, color="0.75", linestyle="--", linewidth=0.8, zorder=0)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=90 if param_cols else 0, fontsize=17)

            ax.tick_params(axis="y", labelsize=17)
            ax.set_ylim(*limits[method])
            ax.grid(axis="y", alpha=0.25)

            if i == 0:
                ax.set_title(str(method), fontsize=22, fontweight="bold")

            if j == 0:
                ax.set_ylabel(
                    f"{dataset}\n\n{metric_col.replace('_', ' ').title()}",
                    fontsize=22, labelpad=12
                )

            if i == n_rows - 1:
                ax.set_xlabel(
                    " / ".join(param_cols) if param_cols else "Parameter setting",
                    fontsize=20, labelpad=10
                )

    fig.suptitle("PCA approximation accuracy across random seeds", fontsize=17, fontweight="bold")
    fig.legend(
        handles=[Line2D([0], [0], marker="o", linestyle="None", color="black", markersize=5, label="Mean")],
        loc="upper right", bbox_to_anchor=(0.98, 0.99), frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / f"{args.name}_boxplots.png"
    pdf = args.output_dir / f"{args.name}_boxplots.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Parameter columns: {param_cols or '(none)'}")
    print(f"Datasets: {n_rows}, methods: {n_cols}, rows: {len(df)}")
    print(f"Wrote: {png}\nWrote: {pdf}")


if __name__ == "__main__":
    main()
