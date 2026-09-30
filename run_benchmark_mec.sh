#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MAIN_PLAN="benchmark_human_mec.yaml"
KNN_PLAN="benchmark_knn_human_mec.yaml"
RUN_PLAN="$MAIN_PLAN"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA GPU detected: Scanpy, Scrapper and RAPIDS will run."
else
    echo "No NVIDIA GPU detected: RAPIDS will be removed from this run."

    CPU_TEMP="$(mktemp "$ROOT/.benchmark_mec_cpu.XXXXXX")"
    CPU_PLAN="${CPU_TEMP}.yaml"
    mv "$CPU_TEMP" "$CPU_PLAN"

    RUN_PLAN="$CPU_PLAN"
    trap 'rm -f "$CPU_PLAN"' EXIT

    python - "$MAIN_PLAN" "$CPU_PLAN" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])

with source.open() as handle:
    plan = yaml.safe_load(handle)

for stage in plan.get("stages", []):
    modules = stage.get("modules", [])
    stage["modules"] = [
        module
        for module in modules
        if "gpu" not in module.get("requires_capabilities", [])
    ]

environments = plan.get("software_environments", {})

if isinstance(environments, dict):
    environments.pop("rapids_env", None)
else:
    plan["software_environments"] = [
        environment
        for environment in environments
        if environment.get("id") != "rapids_env"
    ]

with destination.open("w") as handle:
    yaml.safe_dump(plan, handle, sort_keys=False)
PY
fi

echo "Validating first-pass benchmark: $RUN_PLAN"
ob validate plan "$RUN_PLAN"

echo "Validating second-pass benchmark: $KNN_PLAN"
ob validate plan "$KNN_PLAN"

echo "Running PCA, eigengap and seed-selection benchmark"
ob run "$RUN_PLAN" -- \
    --cores 1 \
    --scheduler greedy \
    --rerun-incomplete \
    --latency-wait 60

test -s out/principal_angles/selected_seeds.tsv
test -s out/principal_angles/pca_manifest.tsv

echo "Checking generated PCA manifest"
python - <<'PY'
import pandas as pd

manifest = pd.read_csv("out/principal_angles/pca_manifest.tsv", sep="\t")
selected = pd.read_csv("out/principal_angles/selected_seeds.tsv", sep="\t")

print("\nPCA manifest:")
print(manifest.groupby(["method", "pca_type"]).size())

print("\nSelected seeds:")
print(selected.groupby(["method", "selection"]).size())
PY

echo "Running kNN, overlap, clustering and ARI benchmark"
ob run "$KNN_PLAN" -- \
    --cores 1 \
    --scheduler greedy \
    --rerun-incomplete \
    --latency-wait 60

echo "Human MEC benchmark completed successfully."