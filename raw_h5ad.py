#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output_dir", type=Path, required=True)
p.add_argument("--name", required=True)
p.add_argument("--dataset_file", required=True)
args = p.parse_args()

source = Path(__file__).parent / "datasets" / args.dataset_file
args.output_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, args.output_dir / f"{args.name}.h5ad")