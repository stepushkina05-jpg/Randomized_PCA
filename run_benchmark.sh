#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ob run benchmark.yaml --cores 1 --dirty --scheduler greedy

ob run benchmark_knn.yaml --cores 1 --dirty --scheduler greedy