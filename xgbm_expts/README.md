# LightGBM GTR-PCA Decoding Experiments

Directly decode compact text-window embeddings from the Huth language voxels.

The first experiment trains:

```text
Huth selected 10k voxels -> PCA(GTR 5-TR text-window embedding)
```

This is not a Huth-style encoding model; it is a direct semantic decoder. The
goal is to test whether the Huth language voxels contain enough signal to
recover low-dimensional, longer-window GTR semantic state.

## Quick Start

From the repo root on the cluster:

```bash
python -u xgbm_expts/train_gtr_pca_xgbm.py \
  --subjects S1 S2 S3 \
  --backend lightgbm \
  --pca-dim 10 \
  --brain-offsets 0 1 2 3 4 \
  --n-estimators 600 \
  --max-depth 3 \
  --learning-rate 0.03 \
  --n-jobs 8
```

For local use with the Ceph share mounted:

```bash
python -u xgbm_expts/train_gtr_pca_xgbm.py \
  --subjects S1 \
  --local-compute-mode \
  --backend lightgbm \
  --pca-dim 10 \
  --brain-offsets 0
```

If `lightgbm` is unavailable, use:

```bash
--backend sklearn_hist
```

## Outputs

Results are written under `xgbm_expts/results/<tag>/`:

- `metrics.json`: held-out story metrics
- `predictions.npz`: predicted and true PCA-space validation targets
- `model.pkl`: trained regressors plus PCA/z-score transforms, if `--save-models`

`xgbm_expts/results/summary.csv` contains one row per subject.

Useful metrics:

- `mean_dim_r`: mean Pearson correlation over the 10 PCA dimensions
- `mean_cosine`: cosine similarity in z-scored PCA target space
- `retrieval_top1` / `retrieval_top10`: whether the predicted vector retrieves
  the matching held-out text window among all held-out windows
- `pca_explained_variance`: fraction of GTR embedding variance retained by the
  PCA target
