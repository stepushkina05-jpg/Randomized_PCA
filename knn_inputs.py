#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Copy principal-angle outputs into the second-pass KNN benchmark.")

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--selected_seeds_source", required=True)
    parser.add_argument("--pca_manifest_source", required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example:
    # output_dir =
    # out/KNN_INPUT/local_knn_inputs/.default
    #
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
    shutil.copy2(manifest_source, manifest_target)

    print(f"Copied: {selected_source} -> {selected_target}")
    print(f"Copied: {manifest_source} -> {manifest_target}")


if __name__ == "__main__":
    main()