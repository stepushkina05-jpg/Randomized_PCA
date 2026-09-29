#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import pandas as pd


def find_labels_file(out_root: Path, dataset: str) -> Path:
    candidates = list(
        (out_root / "DATA" / dataset).glob(
            ".*/*.clusters_truth.tsv"
        )
    )

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one truth-label file for {dataset}, "
            f"found {len(candidates)}: {candidates}"
        )

    return candidates[0].resolve()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy principal-angle outputs into the second-pass KNN "
            "benchmark and attach ground-truth label files."
        )
    )

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--selected_seeds_source", required=True)
    parser.add_argument("--pca_manifest_source", required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example:
    # output_dir = out/KNN_INPUT/local_knn_inputs/.default
    # Therefore parents[2] is the main out/ directory.
    out_root = output_dir.parents[2]

    selected_source = out_root / args.selected_seeds_source
    manifest_source = out_root / args.pca_manifest_source

    if not selected_source.exists():
        raise FileNotFoundError(
            f"Missing selected_seeds source file: {selected_source}"
        )

    if not manifest_source.exists():
        raise FileNotFoundError(
            f"Missing pca_manifest source file: {manifest_source}"
        )

    selected_target = output_dir / "selected_seeds.tsv"
    manifest_target = output_dir / "pca_manifest.tsv"

    shutil.copy2(selected_source, selected_target)

    # Add the appropriate ground-truth label file to every PCA row.
    pca_manifest = pd.read_csv(manifest_source, sep="\t")

    if "dataset" not in pca_manifest.columns:
        raise ValueError(
            f"{manifest_source} must contain a dataset column"
        )

    labels_files = {
        dataset: str(find_labels_file(out_root, dataset))
        for dataset in pca_manifest["dataset"].drop_duplicates()
    }

    pca_manifest["labels_file"] = pca_manifest["dataset"].map(
        labels_files
    )

    pca_manifest.to_csv(
        manifest_target,
        sep="\t",
        index=False,
    )

    print(f"Copied: {selected_source} -> {selected_target}")
    print(f"Enriched: {manifest_source} -> {manifest_target}")
    print()
    print("Ground-truth label files")

    for dataset, labels_file in labels_files.items():
        print(f"  {dataset}: {labels_file}")


if __name__ == "__main__":
    main()