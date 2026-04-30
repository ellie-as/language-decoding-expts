# LibriBrain GTR Decode

End-to-end baseline for decoding recent speech/text content from LibriBrain MEG. The first milestone is intentionally simple: 10-second transcript windows are embedded with `sentence-transformers/gtr-t5-base`, MEG windows around the same period are flattened, PCA-reduced, and fit with multi-output ridge regression. Evaluation is retrieval of the correct text-window embedding from distractors.

This scaffold is designed for local development on mock/debug data and full execution on a cluster. The repository does not need the full LibriBrain dataset locally.

## Installation

Conda:

```bash
cd libibrain_gtr_decode
conda env create -f environment.yml
conda activate libribrain-gtr
```

Pip:

```bash
cd libibrain_gtr_decode
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The cluster environment should use CUDA-enabled PyTorch. GTR embedding computation uses GPU when `torch.cuda.is_available()`; PCA and ridge are CPU-based.

## Local Debug Run

The debug config uses deterministic mock transcripts, mock MEG, mock embeddings, 64 PCA components, and a small example count:

```bash
python scripts/run_ridge_pipeline.py \
  --config configs/debug_local.yaml \
  --debug
```

This validates the command-line path, caching, text windows, embedding arrays, MEG memmap creation, train-only standardization, PCA, ridge fitting, retrieval metrics, plots, and controls. Debug performance is not meaningful.

## Cluster Run

Edit `configs/ridge_cluster.yaml` so `project.output_dir`, `project.cache_dir`, `data.pnpl_root`, and `data.libribrain_root` point to cluster locations. Then submit:

```bash
sbatch slurm/run_ridge_pipeline_a100.sbatch
```

The Slurm script keeps the partition, repo path, and conda environment easy to edit. You can also override:

```bash
REPO_DIR=/path/to/repo/libibrain_gtr_decode CONDA_ENV=libribrain-gtr \
  sbatch slurm/run_ridge_pipeline_a100.sbatch
```

## Expected Data Layout

The loaders first provide a robust debug/mock path. For real data, `src/data.py` currently discovers BIDS-like files under `data.libribrain_root` using names such as:

- `sub-*`
- `ses-*`
- `run-*`
- transcript/event files matching `*events*.tsv`, `*transcript*.tsv`, or `*words*.tsv`
- MEG files matching `*.fif`, `*.fif.gz`, `*.npy`, or `*.npz`

Run this first on any real subset:

```bash
python scripts/inspect_libribrain.py --config configs/ridge_cluster.yaml
```

It writes `outputs/inspection/libribrain_summary.json` and reports subjects, sessions, runs, sampling rate, channel count, event rows, example MEG shape, and whether word-level timing appears available. If PNPL exposes data through a different API than these generic files, adapt `src/data.py` in one place.

To ask PNPL to download/load just one run for a real-data smoke test:

```bash
python scripts/download_tiny_libribrain.py \
  --config configs/debug_local.yaml \
  --data-path ./data/LibriBrain \
  --dataset speech \
  --run-index 0 \
  --tmin 0.0 \
  --tmax 0.2 \
  --max-items 1
```

This uses `pnpl.datasets.LibriBrainSpeech` with `include_run_keys=[constants.RUN_KEYS[0]]`, prints the first item shape/info, and saves `outputs/debug/inspection/tiny_libribrain_summary.json`. The script intentionally keeps `data_path` as an argument so local scratch and cluster scratch can differ.

For a fuller smoke test on one real LibriBrain run, use:

```bash
python scripts/smoketest_tiny_libribrain.py \
  --config configs/debug_local.yaml \
  --data-path ./data/LibriBrain \
  --output-dir outputs/tiny_libribrain_smoketest \
  --run-index 0 \
  --max-duration-sec 180 \
  --max-examples 96
```

This asks PNPL to download/load one run, copies only the first few minutes into a small local H5, builds 10-second transcript windows from PNPL `events.tsv` word rows, uses mock embeddings by default for speed, builds MEG memmap features, trains PCA + ridge, and writes `outputs/tiny_libribrain_smoketest/inspection/tiny_libribrain_smoketest_summary.json`. Add `--real-gtr` if you also want to smoke-test `sentence-transformers/gtr-t5-base`.

To download all PNPL run files before a full run:

```bash
python scripts/download_libribrain_runs.py \
  --config configs/ridge_cluster.yaml \
  --data-path /ceph/behrens/ellie/language-decoding-expts/libribrain_gtr_decode/data/LibriBrain \
  --partition all
```

For a staged download, add `--max-runs 5` first. After the download finishes, run:

```bash
python scripts/run_ridge_pipeline.py \
  --config configs/ridge_cluster.yaml \
  --force
```

The full raw LibriBrain cache is around tens of GB, and derived MEG windows can also become large. Keep `project.output_dir`, `project.cache_dir`, and `data.libribrain_root` on a filesystem with enough space.

For a faster full-dataset ridge baseline, use random projection instead of PCA:

```bash
python scripts/run_ridge_pipeline.py \
  --config configs/ridge_cluster_random_projection.yaml
```

Random projection avoids the expensive IncrementalPCA fit and reduces flattened MEG windows to `projection_components` dimensions before ridge regression. If you already built `outputs/ridge_cluster/meg_features`, you can reuse it by copying the `text_windows`, `embeddings`, `splits`, and `meg_features` subdirectories into `outputs/ridge_cluster_random_projection/`, then running `scripts/train_ridge_baseline.py` with the random-projection config.

## Pipeline Steps

Every step is command-line runnable and cacheable:

```bash
python scripts/build_text_windows.py --config configs/debug_local.yaml
python scripts/compute_gtr_embeddings.py --config configs/debug_local.yaml
python scripts/create_splits.py --config configs/debug_local.yaml
python scripts/build_meg_features.py --config configs/debug_local.yaml
python scripts/train_ridge_baseline.py --config configs/debug_local.yaml
python scripts/evaluate_retrieval.py --config configs/debug_local.yaml
```

The one-command pipeline skips completed cached steps when the config hash matches. Use `--force` to recompute.

## Outputs

Key outputs include:

- `text_windows/text_windows_10s_1hz.parquet`
- `text_windows/examples.txt`
- `embeddings/gtr_base_10s_1hz.npy`
- `embeddings/gtr_base_10s_1hz_metadata.parquet`
- `splits/split_by_run_seed0.parquet`
- `meg_features/meg_10s_plus1s_50hz.dat`
- `meg_features/meg_10s_plus1s_metadata.parquet`
- `models/ridge_gtr_base_10s.joblib`
- `models/pca_gtr_base_10s.joblib`
- `models/scaler_gtr_base_10s.joblib`
- `predictions/ridge_val_predictions.npy`
- `predictions/ridge_test_predictions.npy`
- `results/ridge_metrics.json`
- `results/retrieval_test_by_distractor_type.csv`
- `plots/*.png`

Generated outputs, caches, arrays, models, logs, and raw data are ignored by git.

## Leakage-Safe Splitting

The intended split is by `subject/session/run`, not by individual 1 Hz examples. Overlapping 10-second text windows are kept, but neighbouring windows must not be randomly split across train and test. With a single debug mock run, the code falls back to a time split only to exercise the full pipeline locally; do not use that fallback for scientific results.

## Retrieval Metrics

The primary task ranks candidate true GTR embeddings by cosine similarity to the predicted MEG embedding. Metrics include top-1, top-5, top-10, MRR, median rank, cosine to the true target, MSE, R2, and relaxed retrieval within temporal tolerances.

Candidate sets are:

- full-test retrieval
- random global distractors
- same-run distractors
- nearby temporal distractors

Do not interpret global random retrieval alone. It can be inflated by story, book, run, or subject identity. Meaningful evidence requires same-run retrieval, nearby temporal distractors, and controls.

## Controls

The first version runs:

- time-shifted labels within each run
- shuffled labels across examples

A useful first result is aligned labels outperforming both controls, with retrieval above chance for random global distractors and ideally for same-run or nearby-window distractors. If performance is only above chance for global random distractors, treat it as possible broad story/book identity decoding rather than precise 10-second semantic decoding.

## Next Steps

After the ridge baseline and evaluation are trustworthy, add neural models such as a spatiotemporal ConvNet, CLIP-style contrastive loss, multi-task speech/phoneme/word objectives, and longer semantic-state targets. Do not add those before the baseline pipeline, controls, and retrieval diagnostics are working.
