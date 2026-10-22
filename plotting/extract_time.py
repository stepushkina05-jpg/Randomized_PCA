#!/usr/bin/env python3
"""Collect Scanpy and Scrapper PCA runtimes and plot seed-level distributions."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 23, "axes.labelsize": 21,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 26, "figure.titlesize": 26,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_scanpy(root):
    rows = []
    for file in sorted(root.rglob("obkit-events.jsonl")):
        match = re.fullmatch(r"n(\d+)_over(\d+)", file.parent.name)
        if not match:
            print(f"Skipping unexpected folder: {file.parent}")
            continue
        n_iter, n_over = map(int, match.groups())
        dataset, start, durations = file.parent.parent.name, None, []
        with file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event") != "pca":
                    continue
                if event.get("phase") == "start":
                    start = pd.Timestamp(event["ts"])
                elif event.get("phase") == "end":
                    if start is None:
                        raise ValueError(f"Unmatched PCA end in {file}")
                    durations.append((pd.Timestamp(event["ts"]) - start).total_seconds())
                    start = None
        if start is not None:
            raise ValueError(f"Unmatched PCA start in {file}")
        # Scanpy logs are assumed to contain runs in seed order (1, 2, ...).
        rows.extend(dict(method="Scanpy", dataset=dataset, seed=seed, n_iter=n_iter,
                         n_over=n_over, runtime_s=seconds, source=str(file))
                    for seed, seconds in enumerate(durations, 1))
        print(f"Scanpy | {dataset} | n={n_iter}, over={n_over}: {len(durations)} runs")
    if not rows:
        raise ValueError(f"No Scanpy PCA logs found under {root}")
    return pd.DataFrame(rows)


def load_scrapper(root):
    rows = []
    for file in sorted(root.rglob("*_irlba_sweep_metrics.tsv")):
        df = pd.read_csv(file, sep="\t")
        required = {"name", "seed", "n_iter", "n_over", "runtime_s"}
        if missing := required - set(df.columns):
            raise ValueError(f"{file} is missing columns: {sorted(missing)}")
        for rec in df.dropna(subset=["runtime_s"]).itertuples(index=False):
            rows.append(dict(method="Scrapper", dataset=str(rec.name), seed=int(rec.seed),
                             n_iter=int(rec.n_iter), n_over=int(rec.n_over),
                             runtime_s=float(rec.runtime_s), source=str(file)))
        print(f"Scrapper | {file.name}: {len(df.dropna(subset=['runtime_s']))} runs")
    if not rows:
        raise ValueError(f"No Scrapper runtimes found under {root}")
    return pd.DataFrame(rows)


def plot_runtimes(data, output, title):
    datasets = sorted(data["dataset"].unique())
    n_overs = sorted(data["n_over"].unique())
    n_iters = sorted(data["n_iter"].unique())
    methods = list(data["method"].unique())
    colors = {method: plt.get_cmap("tab10")(i) for i, method in enumerate(methods)}
    offsets = dict(zip(methods, np.linspace(-0.12, 0.12, len(methods)) if len(methods) > 1 else [0.0]))

    fig, axes = plt.subplots(len(datasets), len(n_overs), figsize=(6.2 * len(n_overs), 5.2 * len(datasets)),
                             squeeze=False, sharex=True, sharey=False)
    for i, dataset in enumerate(datasets):
        for j, n_over in enumerate(n_overs):
            ax = axes[i, j]
            panel = data[(data["dataset"] == dataset) & (data["n_over"] == n_over)]
            for method in methods:
                xs, ys = [], []
                subset = panel[panel["method"] == method]
                for n_iter in n_iters:
                    values = subset.loc[subset["n_iter"] == n_iter, "runtime_s"].dropna().to_numpy()
                    if not len(values):
                        continue
                    jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.array([0.0])
                    xs.extend(n_iter + offsets[method] + jitter)
                    ys.extend(np.sort(values))
                    ax.scatter(n_iter + offsets[method], np.median(values), s=100, marker="D",
                               color=colors[method], edgecolors="black", linewidths=0.9, zorder=4)
                if xs:
                    ax.scatter(xs, ys, s=24, alpha=0.38, color=colors[method], edgecolors="none")
            ax.set_xticks(n_iters)
            ax.tick_params(axis="both", labelsize=20)
            ax.grid(axis="y", alpha=0.22)
            if i == len(datasets) - 1:
                ax.set_xlabel("n_iter", fontsize=25, labelpad=8, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{dataset}\nRuntime (s)", fontsize=22, labelpad=10)
            if i == 0:
                ax.set_title(f"n_over = {n_over}", fontsize=25, fontweight="bold")

    handles = [Line2D([0], [0], marker="o", linestyle="None", color=colors[m],
                      markersize=11, label=m) for m in methods]
    handles.append(Line2D([0], [0], marker="D", linestyle="None", color="black",
                          markerfacecolor="white", markersize=10, label="Median (diamond)"))
    fig.suptitle(title or "PCA runtime across iterations and oversampling", fontsize=26,
                 fontweight="bold", y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=len(handles), fontsize=25, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2, h_pad=1.8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scanpy_root", type=Path, required=True)
    p.add_argument("--scrapper_root", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--output_plot", type=Path, required=True)
    p.add_argument("--summary_csv", type=Path)
    p.add_argument("--title")
    args = p.parse_args()

    data = pd.concat([load_scanpy(args.scanpy_root), load_scrapper(args.scrapper_root)], ignore_index=True)
    data = data.sort_values(["dataset", "method", "n_over", "n_iter", "seed"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_csv, index=False)
    print(f"Saved {len(data)} seed-level runtimes: {args.output_csv}")

    summary = data.groupby(["method", "dataset", "n_iter", "n_over"])["runtime_s"].agg(
        count="count", mean="mean", median="median", std="std", min="min", max="max").reset_index()
    print("\nRuntime summary:\n", summary.to_string(index=False))
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)
        print(f"Saved summary: {args.summary_csv}")
    plot_runtimes(data, args.output_plot, args.title)


if __name__ == "__main__":
    main()
