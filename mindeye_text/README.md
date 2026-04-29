# MindEye-style multi-subject text decoder

Decodes the lagged 5-TR text-embedding window from a **single TR** of fMRI by
sharing as much of the model as possible across S1, S2, S3.

```
voxels_S1 ──┐
            ├─► Linear_S1(n_vox_S1, 4096) ──┐
voxels_S2 ──┤   Linear_S2(n_vox_S2, 4096) ──┼─► 4-block residual MLP (4096) ─► LN ─► Linear(4096, 768) ─► gtr-base embedding
voxels_S3 ──┘   Linear_S3(n_vox_S3, 4096) ──┘
                      (subject-specific)              (shared backbone + head)
```

This mirrors the MindEyeV2 idea: every subject keeps a private input
projection, but all of the heavy parameters (the 4 residual blocks and the
projection head) are trained jointly on `≈ N_subjects ×` more data than a
single-subject run.

The target is the same 5-TR-window GTR text embedding used by
`27-04-expts/train_5tr_chunk_nn.py` (chunk index `i` covers text TRs
`[i, i+5)`); the input is one brain TR at index `i + lag + brain_offset`. With
the defaults (`--lag-trs 3 --chunk-trs 5 --brain-offset 0`) that's brain TR
`i+3`, i.e. exactly one TR three TRs after the start of the text window.

## Files

- `model.py` - `MindEyeText` (subject Linear + 4 residual blocks + head) and
  the `mse / cosine / mse_cosine / mse_clip` losses.
- `data.py` - per-subject voxel z-scoring, the single-TR chunk dataset, and
  the shared/per-subject target z-score utilities. Reuses the on-disk text
  embedding cache produced by `27-04-expts/train_5tr_chunk_nn.py`.
- `train_mindeye_text.py` - the training entry point. Pools batches across
  every selected subject in each step so the InfoNCE term sees cross-subject
  negatives.
- `eval_mindeye_text.py` - load a saved `model.pt`, recompute per-subject and
  pooled retrieval / cosine / dim-r metrics on the held-out test stories.
- `run_train.sh` - example launch script for an A100 cluster job.

## Cluster / A100 usage

The script defaults are tuned for an A100 with bf16 autocast on. From the repo
root on a node where `<repo>/data_train` already points at the dataset:

```bash
./mindeye_text/run_train.sh
```

That runs:

```bash
python -u mindeye_text/train_mindeye_text.py \
  --subjects S1 S2 S3 \
  --roi full_frontal \
  --feature-model gtr-base \
  --latent-dim 4096 --n-blocks 4 --dropout 0.15 \
  --loss mse_clip --clip-weight 0.5 --clip-temp 0.05 \
  --lr 3e-4 --weight-decay 1e-2 \
  --batch-size 256 --max-epochs 200 --patience 30 \
  --lag-trs 3 --chunk-trs 5 --brain-offset 0 \
  --torch-device cuda \
  --output-dir mindeye_text/results
```

`--batch-size` is **per subject** - total effective batch size is
`N_subjects × batch_size` (e.g. 768 with three subjects). `--no-amp` disables
bf16, `--torch-device cpu/mps` is for local debug only.

## Outputs

A run at `mindeye_text/results/<tag>/` (tag auto-derived from hyperparameters
unless `--tag` is set) contains:

- `metrics.csv` - one row per subject plus an `ALL` row with retrieval top-1,
  MRR, mean rank, mean cosine similarity, and per-dim Pearson `r`.
- `history.csv` - per-epoch train/val loss including per-subject val loss.
- `metadata.json` - all CLI args, train/test story splits, voxel counts,
  embedding cache paths, total training time.
- `model.pt` - checkpoint with the model weights *and* every per-subject
  artifact needed to re-run prediction (voxel indices, voxel z-score stats,
  target z-score stats, ROI name, `chunk_trs`, `lag_trs`, `brain_offset`,
  etc.).
- `predictions.npz` - `pooled_pred / pooled_true / pooled_subject` plus
  `pred_emb__<subj> / true_emb__<subj>` for downstream analyses
  (e.g. running `27-04-expts/invert_last20words_predictions.py` style
  inversion against these predictions).

## Running held-out evaluation

```bash
python mindeye_text/eval_mindeye_text.py \
  --checkpoint mindeye_text/results/<tag>/model.pt \
  --torch-device cuda
```

By default this reuses the train-time stories list, ROI, embedding cache, etc.
Override anything with the matching `--*` flag (the same names as in
`train_mindeye_text.py`). The script writes `eval_metrics.csv` and
`eval_predictions.npz` next to the checkpoint.

## Loss options

- `mse` - plain MSE on z-scored embedding targets.
- `cosine` - mean of `1 - cos sim` on L2-normalized vectors.
- `mse_cosine` - convex mix, `--cosine-weight` controls the cosine term.
- `mse_clip` (default) - MSE plus a symmetric InfoNCE term across the full
  cross-subject batch. `--clip-weight` is the InfoNCE weight, `--clip-temp`
  the temperature. This is the variant closest to MindEye's training and
  benefits most from pooling all subjects in one step.

## ROI / voxel selection

`--roi` is applied to **every** selected subject (default `full_frontal`,
~20-29k voxels). For a whole-brain run pass `--roi all`. Voxel counts are
allowed to differ between subjects - the subject-specific `Linear` absorbs
the difference.

## Caching

The 5-TR text embeddings live under `27-04-expts/cache/<subject>/` and are
shared with the existing `27-04-expts/train_5tr_chunk_nn.py` runs. If the
cache for a subject/feature-model/lag/chunk doesn't exist yet, the first run
on that combination will build it from the `train_stimulus/*.TextGrid` files;
that needs the full `data_train` directory to be reachable, not just
`train_response`.
