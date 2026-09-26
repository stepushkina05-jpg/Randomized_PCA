# Randomized PCA stability in single-cell RNA-seq

This [OmniBenchmark](https://omnibenchmark.org) workflow studies how the random seed used by randomized PCA affects single-cell RNA-seq analysis. It compares randomized PCA with exact PCA on the same processed data, then follows the differences into k-nearest-neighbor (kNN) graphs and Leiden clusters. It measures **computational variability from PCA randomization** while holding the input data and preprocessing fixed.

## Benchmark design

The first pass (`benchmark.yaml`) processes three datasets from [randomized-pca-data](https://github.com/stepushkina05-jpg/randomized-pca-data), selects 2,000 highly variable genes, and runs exact and randomized PCA with Scanpy and Scrapper. It retains 50 PCs. The current configuration uses randomized seeds 1–10 for Scanpy and one randomized seed for Scrapper.

| Analysis | What it measures |
| --- | --- |
| PCA accuracy | Reconstruction residuals relative to exact PCA and absolute correlations between corresponding PC scores. |
| Relative eigengaps | Separation between adjacent exact-PCA eigenvalues, estimated from PC-score variances. The five smallest gaps select dimensions for downstream comparisons. |
| Principal-angle error | Disagreement between the exact and randomized PCA loading subspaces at each selected dimension. |

For each dataset, implementation, and selected number of PCs, the principal-angle analysis identifies the best, median, and worst randomized seeds by subspace error.

The second pass (`benchmark_knn.yaml`) uses the **best and worst** seeds and the exact-PCA reference to construct kNN graphs with 15 neighbors, measure neighbor retention and connectivity differences, run Leiden clustering, and calculate adjusted Rand indices (ARI) against the exact-PCA clustering. The kNN and clustering seeds are held fixed to focus the comparison on PCA seed variation.

**Current scope:** Scrapper has only one randomized PCA run, so its selected best and worst seeds coincide. Its current results cannot measure variation across Scrapper seeds. Downstream comparisons cover selected eigengap dimensions and seeds rather than every possible combination.

## Code provenance

The preprocessing and PCA modules build on the [omni-scrna Scanpy](https://github.com/omni-scrna/scanpy) and [omni-scrna Scrapper](https://github.com/omni-scrna/scrapper) implementations. Diana Stepushkina adapted them for this benchmark in [scanpy_random](https://github.com/stepushkina05-jpg/scanpy_random) and [scrapper_random](https://github.com/stepushkina05-jpg/scrapper_random). The `knn.py` implementation in this repository is adapted from the omni-scrna Scanpy kNN module.

The scripts under `metrics/`, the second-pass input bridge (`knn_inputs.py`), and the benchmark configurations and runner were developed for this project. Please retain the upstream attribution when reusing adapted modules.

## Run the workflow

The project was developed with OmniBenchmark v0.6.0. From the repository root:

```bash
./run_benchmark.sh
```

The runner executes `benchmark.yaml` and then `benchmark_knn.yaml`. The second pass reads `principal_angles/selected_seeds.tsv` and `principal_angles/pca_manifest.tsv` produced by the first pass in `out/`.

**Local setup required:** The YAML files currently contain absolute paths to the author's checkout, and the referenced `envs/` files are not tracked in this repository. Replace the paths and add the Conda environment files before running on another machine. The second-pass clustering stage also needs to point to a repository revision containing `selected_clustering.py`; that script is no longer in this repository. Generated outputs under `out/` are ignored by Git.

## Repository contents

| Path | Purpose |
| --- | --- |
| `benchmark.yaml` | First pass: data, preprocessing, PCA, eigengaps, and PCA comparisons. |
| `benchmark_knn.yaml` | Second pass: selected kNN graphs, overlap, Leiden clustering, and ARI. |
| `metrics/` | Benchmark-specific analysis scripts. |
| `knn.py`, `knn_inputs.py` | Graph construction and transfer of selected first-pass outputs. |
| `run_benchmark.sh` | Runs both passes in sequence. |

## Citation and license

For this benchmark, use [`CITATION.cff`](CITATION.cff). When reusing adapted processing modules, also credit the relevant omni-scrna projects linked above. The citation metadata lists MIT; a standalone `LICENSE` file has not yet been added to this repository.
