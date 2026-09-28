#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MAIN_PLAN="benchmark.yaml"
RUN_PLAN="$MAIN_PLAN"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA GPU detected: Scanpy, Scrapper and RAPIDS will run."
else
    echo "No NVIDIA GPU detected: RAPIDS will be removed from this run."

    CPU_TEMP="$(mktemp "$ROOT/.benchmark_cpu.XXXXXX")"
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

ob validate plan "$RUN_PLAN"
ob validate plan benchmark_knn.yaml

ob run "$RUN_PLAN" --cores 1 -- --scheduler greedy
ob run benchmark_knn.yaml --cores 1 -- --scheduler greedy