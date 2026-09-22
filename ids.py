#!/usr/bin/env python3

import argparse
import gzip
from pathlib import Path
import scanpy as sc

p = argparse.ArgumentParser(description="Keep all cells and features")
p.add_argument("--output_dir", type=Path, required=True)
p.add_argument("--name", type=str, required=True)
p.add_argument("--rawdata_h5ad", type=Path, required=True)
args = p.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
adata = sc.read_h5ad(args.rawdata_h5ad)

with gzip.open(args.output_dir / f"{args.name}_cellids.txt.gz", "wt") as f:
    f.write("\n".join(map(str, adata.obs_names)) + "\n")

with gzip.open(args.output_dir / f"{args.name}_featureids.txt.gz", "wt") as f:
    f.write("\n".join(map(str, adata.var_names)) + "\n")

print("cells:", adata.n_obs)
print("features:", adata.n_vars)