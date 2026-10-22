#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# SCANPY
# --------------------------------------------------

def parse_scanpy_folder(folder):
    """
    Parse folder name such as:
        n2_over10
    """
    match = re.fullmatch(r"n(\d+)_over(\d+)", folder.name)

    if match is None:
        return None

    return int(match.group(1)), int(match.group(2))


def extract_scanpy_runtimes(jsonl_file):
    """
    Read one obkit-events.jsonl.

    The first PCA start/end pair is treated as seed 1,
    the second as seed 2, etc.
    """

    runtimes = []
    pca_start = None

    with open(jsonl_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record.get("event") != "pca":
                continue

            phase = record.get("phase")

            if phase == "start":
                pca_start = pd.Timestamp(record["ts"])

            elif phase == "end":
                if pca_start is None:
                    raise ValueError(
                        f"PCA end without preceding start in {jsonl_file}"
                    )

                pca_end = pd.Timestamp(record["ts"])

                runtime_s = (
                    pca_end - pca_start
                ).total_seconds()

                runtimes.append(runtime_s)

                pca_start = None

    return runtimes


def load_scanpy(scanpy_root):
    scanpy_root = Path(scanpy_root)

    rows = []

    for jsonl_file in scanpy_root.rglob("obkit-events.jsonl"):

        parameter_folder = jsonl_file.parent
        parsed = parse_scanpy_folder(parameter_folder)

        if parsed is None:
            print(
                f"Skipping {jsonl_file}: "
                f"cannot parse parameter folder "
                f"'{parameter_folder.name}'"
            )
            continue

        n_iter, n_over = parsed

        dataset = parameter_folder.parent.name

        runtimes = extract_scanpy_runtimes(jsonl_file)

        if len(runtimes) != 50:
            print(
                f"WARNING: {dataset} "
                f"n_iter={n_iter}, n_over={n_over}: "
                f"found {len(runtimes)} PCA runs, expected 50"
            )

        for seed, runtime_s in enumerate(runtimes, start=1):

            rows.append({
                "method": "Scanpy",
                "dataset": dataset,
                "seed": seed,
                "n_iter": n_iter,
                "n_over": n_over,
                "runtime_s": runtime_s,
                "source": str(jsonl_file)
            })

    if not rows:
        raise RuntimeError(
            f"No Scanpy runtime data found under {scanpy_root}"
        )

    return pd.DataFrame(rows)


# --------------------------------------------------
# SCRAPPER
# --------------------------------------------------

def load_scrapper(scrapper_root):
    scrapper_root = Path(scrapper_root)

    rows = []

    files = sorted(
        scrapper_root.rglob("*_irlba_sweep_metrics.tsv")
    )

    if not files:
        raise RuntimeError(
            f"No *_irlba_sweep_metrics.tsv files found "
            f"under {scrapper_root}"
        )

    for file in files:

        df = pd.read_csv(file, sep="\t")

        required = {
            "name",
            "seed",
            "n_iter",
            "n_over",
            "runtime_s"
        }

        missing = required - set(df.columns)

        if missing:
            print(
                f"Skipping {file}: "
                f"missing columns {sorted(missing)}"
            )
            continue

        if "status" in df.columns:
            print(f"\n{file.name} status counts:")
            print(df["status"].value_counts(dropna=False))

        for _, row in df.iterrows():

            if pd.isna(row["runtime_s"]):
                continue

            rows.append({
                "method": "Scrapper",
                "dataset": row["name"],
                "seed": int(row["seed"]),
                "n_iter": int(row["n_iter"]),
                "n_over": int(row["n_over"]),
                "runtime_s": float(row["runtime_s"]),
                "source": str(file)
            })

    if not rows:
        raise RuntimeError(
            f"No valid Scrapper runtime rows found "
            f"under {scrapper_root}"
        )

    return pd.DataFrame(rows)


# --------------------------------------------------
# CHECKS
# --------------------------------------------------

def print_checks(data):

    print("\n==============================")
    print("RUN COUNTS")
    print("==============================")

    print(
        data.groupby(
            ["method", "dataset"]
        ).size()
    )

    print("\n==============================")
    print("RUNS PER PARAMETER SETTING")
    print("==============================")

    counts = (
        data
        .groupby(
            ["method", "dataset", "n_iter", "n_over"]
        )
        .size()
        .rename("runs")
        .reset_index()
    )

    print(counts.to_string(index=False))

    bad = counts[counts["runs"] != 50]

    if len(bad) > 0:
        print("\nWARNING: settings without exactly 50 runs:")
        print(bad.to_string(index=False))

    print("\n==============================")
    print("RUNTIME SUMMARY")
    print("==============================")

    summary = (
        data
        .groupby(
            ["method", "dataset", "n_iter", "n_over"]
        )["runtime_s"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max"
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    return summary


# --------------------------------------------------
# PLOT
# --------------------------------------------------

def plot_runtime_points(data, output, title=None):

    datasets = sorted(data["dataset"].unique())
    n_overs = sorted(data["n_over"].unique())
    n_iters = sorted(data["n_iter"].unique())
    methods = list(data["method"].unique())

    fig, axes = plt.subplots(
        len(datasets),
        len(n_overs),
        figsize=(
            4.0 * len(n_overs),
            3.3 * len(datasets)
        ),
        squeeze=False,
        sharex=True,
        sharey=False
    )

    # Horizontal offset so Scanpy and Scrapper
    # do not completely overlap
    if len(methods) == 1:
        method_offsets = {methods[0]: 0.0}
    else:
        offsets = np.linspace(
            -0.12,
            0.12,
            len(methods)
        )

        method_offsets = dict(
            zip(methods, offsets)
        )

    for row_idx, dataset in enumerate(datasets):

        for col_idx, n_over in enumerate(n_overs):

            ax = axes[row_idx, col_idx]

            panel = data[
                (data["dataset"] == dataset) &
                (data["n_over"] == n_over)
            ]

            for method in methods:

                method_data = panel[
                    panel["method"] == method
                ]

                xs = []
                ys = []

                for n_iter in n_iters:

                    values = method_data.loc[
                        method_data["n_iter"] == n_iter,
                        "runtime_s"
                    ].dropna().to_numpy()

                    if len(values) == 0:
                        continue

                    # Small deterministic jitter within each
                    # parameter setting so all 50 seeds are visible
                    if len(values) == 1:
                        jitter = np.array([0.0])
                    else:
                        jitter = np.linspace(
                            -0.055,
                            0.055,
                            len(values)
                        )

                    order = np.argsort(values)

                    x = (
                        n_iter
                        + method_offsets[method]
                        + jitter
                    )

                    xs.extend(x)
                    ys.extend(values[order])

                    # Median marker
                    ax.scatter(
                        n_iter + method_offsets[method],
                        np.median(values),
                        s=55,
                        marker="D",
                        edgecolors="black",
                        linewidths=0.6,
                        label=None
                    )

                if xs:
                    ax.scatter(
                        xs,
                        ys,
                        s=13,
                        alpha=0.32,
                        label=method
                    )

            ax.set_xticks(n_iters)

            if row_idx == len(datasets) - 1:
                ax.set_xlabel("n_iter")

            if col_idx == 0:
                ax.set_ylabel(
                    f"{dataset}\nRuntime (s)"
                )

            if row_idx == 0:
                ax.set_title(
                    f"n_over = {n_over}"
                )

            ax.grid(
                axis="y",
                alpha=0.2
            )

    # One legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    if unique:
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper right",
            frameon=False
        )

    if title is None:
        title = (
            "PCA runtime across power iterations "
            "and oversampling settings"
        )

    fig.suptitle(
        title,
        fontsize=14,
        y=0.995
    )

    fig.tight_layout(
        rect=[0, 0, 0.96, 0.97]
    )

    output = Path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nSaved plot to:\n{output}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract Scanpy and Scrapper PCA runtimes "
            "and plot all seed-level timing points."
        )
    )

    parser.add_argument(
        "--scanpy_root",
        required=True,
        help=(
            "Root directory containing Scanpy dataset folders."
        )
    )

    parser.add_argument(
        "--scrapper_root",
        required=True,
        help=(
            "Root directory containing Scrapper dataset folders."
        )
    )

    parser.add_argument(
        "--output_csv",
        required=True,
        help="Combined runtime CSV."
    )

    parser.add_argument(
        "--output_plot",
        required=True,
        help="Output plot (.png, .pdf, ...)."
    )

    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Optional runtime summary CSV."
    )

    parser.add_argument(
        "--title",
        default=None
    )

    args = parser.parse_args()

    scanpy = load_scanpy(
        args.scanpy_root
    )

    scrapper = load_scrapper(
        args.scrapper_root
    )

    data = pd.concat(
        [scanpy, scrapper],
        ignore_index=True
    )

    data = data.sort_values(
        [
            "dataset",
            "method",
            "n_over",
            "n_iter",
            "seed"
        ]
    )

    output_csv = Path(args.output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        output_csv,
        index=False
    )

    print(
        f"\nSaved {len(data)} individual runtimes to:\n"
        f"{output_csv}"
    )

    summary = print_checks(data)

    if args.summary_csv:

        summary_path = Path(
            args.summary_csv
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        summary.to_csv(
            summary_path,
            index=False
        )

        print(
            f"\nSaved summary to:\n"
            f"{summary_path}"
        )

    plot_runtime_points(
        data=data,
        output=args.output_plot,
        title=args.title
    )


if __name__ == "__main__":
    main()