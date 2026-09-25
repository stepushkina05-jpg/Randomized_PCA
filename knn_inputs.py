#!/usr/bin/env python3

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    required_files = [
        output_dir / "selected_seeds.tsv",
        output_dir / "pca_manifest.tsv",
    ]

    missing = [
        path for path in required_files
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing KNN input files: "
            + ", ".join(str(path) for path in missing)
        )

    print("KNN input files already exist:")
    for path in required_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()